# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare resident calibration with the pre-workspace, host-blended solver."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_shoe.calibration import CalibrationWorkspace
from projects.digital_shoe.runtime import FoundationParams, cycle_force, cycle_overstress, relax_surround


def _bed():
    """Build a nonuniform bed with a partial indenter and an asymmetric cycle."""
    rows, columns = 4, 5
    count = rows * columns
    neighbors = np.full((count, 4), -1, dtype=np.int32)
    for row in range(rows):
        for column in range(columns):
            i = row * columns + column
            for side, (dr, dc) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
                nr, nc = row + dr, column + dc
                if 0 <= nr < rows and 0 <= nc < columns:
                    neighbors[i, side] = nr * columns + nc
    slack = np.linspace(0.018, 0.031, count, dtype=np.float32)
    driven = np.zeros(count, dtype=np.int32)
    driven[[6, 7, 12, 13]] = 1
    cycle = np.array([0.0, 0.07, 0.19, 0.36, 0.46, 0.43, 0.31, 0.16, 0.04, 0.0, 0.0], np.float32)
    imposed = cycle[:, None] * slack[driven != 0][None, :] * np.array([0.9, 1.0, 0.8, 0.95], np.float32)
    return {
        "driven_compression": np.ascontiguousarray(imposed),
        "driven": driven,
        "neighbors": neighbors,
        "slack_m": slack,
        "dt_s": np.linspace(0.006, 0.014, len(cycle), dtype=np.float32),
        "area_m2": 0.008**2,
        "spacing_m": 0.008,
        "attachment_n_m": 17.0,
        "max_strain": 0.88,
        "coupling_scale": 0.73,
    }


def _params(alternate=False):
    """Supply two active Ogden-Hill terms with distinct material candidates."""
    params = FoundationParams()
    params.g_eq = 41000.0 if alternate else 52000.0
    params.alpha = 2.4 if alternate else 1.7
    params.g_eq2 = 19000.0 if alternate else 13000.0
    params.alpha2 = -3.1 if alternate else -2.2
    poisson = 0.07 if alternate else 0.03
    params.beta = poisson / (1.0 - 2.0 * poisson)
    params.one_minus_two_poisson = 1.0 - 2.0 * poisson
    params.inv_h2 = 1.0 / 0.008**2
    params.stretch_floor = 1.0e-3
    return params


def _options(**overrides):
    """Set an explicit finite solve schedule without hiding convergence defaults."""
    options = {
        "fraction": (1.0 - 0.62) / 0.62,
        "tau_s": 0.08,
        "blend": 0.62,
        "passes": 4,
        "tolerance_m": 0.0,
        "sweeps": 53,
        "solve_tolerance_m": 0.0,
        "check_every": 25,
        "over_relaxation": 1.05,
    }
    options.update(overrides)
    return options


def _legacy_solve(bed, params, device, *, initial=None, **options):
    """Retain the original allocation-heavy runtime and NumPy blending path.

    In particular, keep the existing interval-ratio stopping estimate. It is
    not a certified error bound, and this regression does not change its math.
    """
    frames, count = len(bed["dt_s"]), len(bed["slack_m"])
    host_overstress = np.zeros((frames, count), np.float32)
    slack = wp.array(bed["slack_m"], dtype=wp.float32, device=device)
    dt = wp.array(bed["dt_s"], dtype=wp.float32, device=device)
    carried = None
    previous = None
    warm = initial
    changes = []
    sweeps_per_pass = []
    solver_stats = {}
    for _ in range(options["passes"]):
        compression = relax_surround(
            bed["driven_compression"],
            bed["driven"],
            bed["neighbors"],
            bed["slack_m"],
            params,
            area_m2=bed["area_m2"],
            spacing_m=bed["spacing_m"],
            attachment_n_m=bed["attachment_n_m"],
            max_strain=bed["max_strain"],
            coupling_scale=bed["coupling_scale"],
            over_relaxation=options["over_relaxation"],
            sweeps=options["sweeps"],
            overstress=carried,
            initial=warm,
            tolerance_m=options["solve_tolerance_m"],
            check_every=options["check_every"],
            stats=solver_stats,
            device=device,
        )
        warm = compression
        sweeps_per_pass.append(int(solver_stats["sweeps"]))
        refreshed = wp.zeros((frames, count), dtype=wp.float32, device=device)
        wp.launch(
            cycle_overstress,
            dim=count,
            inputs=[compression, slack, dt, params, options["fraction"], options["tau_s"], refreshed],
            device=device,
        )
        host_overstress += options["blend"] * (refreshed.numpy() - host_overstress)
        carried = wp.array(host_overstress, dtype=wp.float32, device=device)
        relaxed = compression.numpy().copy()
        if previous is not None:
            changes.append(float(np.max(np.abs(relaxed - previous))))
        previous = relaxed
        if changes and changes[-1] < options["tolerance_m"]:
            break
    force = wp.zeros(frames, dtype=wp.float32, device=device)
    wp.launch(
        cycle_force,
        dim=(frames, count),
        inputs=[compression, refreshed, slack, params, bed["area_m2"], force],
        device=device,
    )
    return (
        force.numpy().copy(),
        compression,
        {
            "pass_change_m": changes,
            "max_compression_m": float(np.max(previous)),
            "sweeps_per_pass": sweeps_per_pass,
            "solver_remaining_m": solver_stats["remaining_m"],
            "solver_update_m": solver_stats["update_m"],
        },
    )


class TestDigitalShoeCalibration(unittest.TestCase):
    """Preserve legacy outputs and mutable-state boundaries on CPU and CUDA."""

    @classmethod
    def setUpClass(cls):
        """Include the CPU and each available CUDA device explicitly."""
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def _modes(self):
        """Exercise eager and requested graph operation on every device."""
        return [(device, graph) for device in self.devices for graph in (False, True)]

    def _assert_parity(self, workspace, actual, expected, compression, stats):
        """Compare force, whole-bed compression, and the legacy stopping status."""
        self.assertIsInstance(actual, wp.array)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(workspace.compression.numpy(), compression.numpy(), rtol=2.0e-5, atol=2.0e-8)
        self.assertEqual(workspace.stats["sweeps_per_pass"], stats["sweeps_per_pass"])
        for key in ("pass_change_m", "max_compression_m", "solver_remaining_m", "solver_update_m"):
            np.testing.assert_allclose(
                workspace.stats[key], stats[key], rtol=3.0e-4, atol=2.0e-8, equal_nan=True, err_msg=key
            )

    def test_nonuniform_two_term_cycle_matches_legacy(self):
        """Match nonuniform, coupled two-term cycles with odd chunks and remainders."""
        bed = _bed()
        schedules = (
            _options(),
            _options(passes=3, sweeps=31, check_every=7, solve_tolerance_m=1.0e-12),
            _options(passes=5, tolerance_m=1.0),
            _options(passes=2, sweeps=3, check_every=0, solve_tolerance_m=1.0e-12),
        )
        for device, graph in self._modes():
            workspace = CalibrationWorkspace(**bed, device=device, use_graph=graph)
            for options in schedules:
                with self.subTest(device=str(device), graph=graph, sweeps=options["sweeps"], passes=options["passes"]):
                    params = _params()
                    expected, compression, stats = _legacy_solve(bed, params, device, **options)
                    actual = workspace.solve(params, **options)
                    self._assert_parity(workspace, actual, expected, compression, stats)
                    if options["tolerance_m"] == 1.0:
                        self.assertEqual(len(workspace.stats["sweeps_per_pass"]), 2)
                    self.assertGreater(float(np.max(expected)), 0.0)
                    self.assertGreater(float(np.max(compression.numpy()[:, bed["driven"] == 0])), 0.0)

    def test_paired_cold_and_warm_status_matches_legacy(self):
        """Keep the original cold and warm stopping decisions rather than repairing them."""
        bed = _bed()
        options = _options(passes=9, sweeps=225, check_every=25, solve_tolerance_m=2.0e-7, tolerance_m=2.0e-7)
        for device, graph in self._modes():
            with self.subTest(device=str(device), graph=graph):
                workspace = CalibrationWorkspace(**bed, device=device, use_graph=graph)
                legacy_warm = None
                resident_warm = None
                for warm in (False, True):
                    with self.subTest(warm=warm):
                        params = _params()
                        expected, compression, stats = _legacy_solve(
                            bed, params, device, initial=legacy_warm, **options
                        )
                        actual = workspace.solve(params, initial=resident_warm, **options)
                        self._assert_parity(workspace, actual, expected, compression, stats)
                        self.assertTrue(any(value < options["sweeps"] for value in stats["sweeps_per_pass"]))
                        legacy_warm = compression
                        resident_warm = workspace.compression

    def test_zero_sweeps_preserves_warm_passive_field_and_stamps_driven(self):
        """Preserve supplied passive compression without mutating external warm storage."""
        bed = _bed()
        shape = (len(bed["dt_s"]), len(bed["slack_m"]))
        seed = np.linspace(0.0002, 0.0013, np.prod(shape), dtype=np.float32).reshape(shape)
        seed[:, bed["driven"] != 0] = 0.014
        options = _options(passes=3, sweeps=0, solve_tolerance_m=1.0e-7)
        for device, graph in self._modes():
            with self.subTest(device=str(device), graph=graph):
                workspace = CalibrationWorkspace(**bed, device=device, use_graph=graph)
                initial = wp.array(seed.copy(), dtype=wp.float32, device=device)
                params = _params()
                expected, compression, stats = _legacy_solve(bed, params, device, initial=initial, **options)
                actual = workspace.solve(params, initial=initial, **options)
                self._assert_parity(workspace, actual, expected, compression, stats)
                np.testing.assert_array_equal(initial.numpy(), seed)
                current = workspace.compression.numpy().copy()
                np.testing.assert_array_equal(current[:, bed["driven"] == 0], seed[:, bed["driven"] == 0])
                np.testing.assert_array_equal(current[:, bed["driven"] != 0], bed["driven_compression"])
                alias_expected, alias_compression, alias_stats = _legacy_solve(
                    bed, params, device, initial=workspace.compression, **options
                )
                alias_actual = workspace.solve(params, initial=workspace.compression, **options)
                self._assert_parity(workspace, alias_actual, alias_expected, alias_compression, alias_stats)
                cold_expected, cold_compression, cold_stats = _legacy_solve(bed, params, device, **options)
                cold_actual = workspace.solve(params, **options)
                self._assert_parity(workspace, cold_actual, cold_expected, cold_compression, cold_stats)
                np.testing.assert_array_equal(workspace.compression.numpy()[:, bed["driven"] == 0], 0.0)

    def test_material_changes_reset_overstress_and_keep_stable_outputs(self):
        """Read changed struct values in captured work and reset Maxwell state per solve."""
        bed = _bed()
        for device, graph in self._modes():
            with self.subTest(device=str(device), graph=graph):
                workspace = CalibrationWorkspace(**bed, device=device, use_graph=graph)
                params = _params()
                compression_ptr = workspace.compression.ptr
                force_ptr = None
                first = None
                for alternate in (False, True, False):
                    replacement = _params(alternate)
                    for key in ("g_eq", "alpha", "g_eq2", "alpha2", "beta", "one_minus_two_poisson"):
                        setattr(params, key, getattr(replacement, key))
                    options = _options(
                        fraction=1.5 if alternate else (1.0 - 0.62) / 0.62,
                        tau_s=0.19 if alternate else 0.08,
                        blend=0.4 if alternate else 0.62,
                        over_relaxation=0.85 if alternate else 1.05,
                    )
                    expected, compression, stats = _legacy_solve(bed, params, device, **options)
                    actual = workspace.solve(params, **options)
                    self._assert_parity(workspace, actual, expected, compression, stats)
                    self.assertEqual(workspace.compression.ptr, compression_ptr)
                    if force_ptr is None:
                        force_ptr = actual.ptr
                        first = actual.numpy().copy()
                    else:
                        self.assertEqual(actual.ptr, force_ptr)
                        if alternate:
                            self.assertGreater(float(np.max(np.abs(actual.numpy() - first))), 0.1)
                        else:
                            np.testing.assert_array_equal(actual.numpy(), first)

    def test_graph_capture_does_not_advance_state(self):
        """Leave constructor state unchanged and actually replay CUDA graph chunks."""
        bed = _bed()
        options = _options(passes=2, sweeps=53)
        for device in self.devices:
            with self.subTest(device=str(device)):
                eager = CalibrationWorkspace(**bed, device=device, use_graph=False)
                captured = CalibrationWorkspace(**bed, device=device, use_graph=True)
                np.testing.assert_array_equal(captured.compression.numpy(), eager.compression.numpy())
                np.testing.assert_array_equal(captured.compression.numpy(), np.zeros(captured.compression.shape))
                self.assertEqual(captured.stats, eager.stats)
                expected, compression, stats = _legacy_solve(bed, _params(), device, **options)
                result = captured.solve(_params(), **options)
                self._assert_parity(captured, result, expected, compression, stats)
                eager_result = eager.solve(_params(), **options)
                np.testing.assert_array_equal(result.numpy(), eager_result.numpy())
                np.testing.assert_array_equal(captured.compression.numpy(), eager.compression.numpy())
                if device.is_cuda:
                    self.assertIsNotNone(captured.graph, captured.graph_fallback_reason)
                    self.assertGreater(captured.stats["graph_chunks"], 0)
                else:
                    self.assertIsNone(captured.graph)
                self.assertGreater(captured.stats["eager_sweeps"], 0)

    def test_repeated_solve_has_no_field_download_or_large_allocation(self):
        """Allow scalar diagnostics but reject field-sized downloads and allocations."""
        bed = _bed()
        options = _options(passes=3, sweeps=53, solve_tolerance_m=1.0e-10)
        numpy_original = wp.array.numpy
        init_original = wp.array.__init__
        for device, graph in self._modes():
            with self.subTest(device=str(device), graph=graph):
                workspace = CalibrationWorkspace(**bed, device=device, use_graph=graph)
                force_ptr, compression_ptr = None, workspace.compression.ptr
                field_size = workspace.compression.size
                downloads = []
                allocations = []

                def checked_numpy(array, *args, downloads=downloads, **kwargs):
                    downloads.append(array.size)
                    self.assertLessEqual(array.size, 4, "solve downloaded a field rather than scalar diagnostics")
                    return numpy_original(array, *args, **kwargs)

                def checked_init(array, *args, allocations=allocations, field_size=field_size, **kwargs):
                    init_original(array, *args, **kwargs)
                    if getattr(array, "ptr", None):
                        allocations.append(array.size)
                        self.assertLess(array.size, field_size, "solve constructed a new frame-by-column array")

                with patch.object(wp.array, "numpy", checked_numpy), patch.object(wp.array, "__init__", checked_init):
                    for alternate in (False, True, False):
                        initial = None if force_ptr is None else workspace.compression
                        result = workspace.solve(_params(alternate), initial=initial, **options)
                        if force_ptr is None:
                            force_ptr = result.ptr
                        self.assertEqual(result.ptr, force_ptr)
                        self.assertEqual(workspace.compression.ptr, compression_ptr)
                self.assertTrue(downloads, "exercise scalar convergence readback, not only fixed schedules")
                self.assertTrue(np.all(np.isfinite(result.numpy())))

    def test_static_inputs_and_distinct_workspaces_do_not_alias(self):
        """Own static input snapshots and isolate independent candidate workspaces."""
        bed = _bed()
        original = copy.deepcopy(bed)
        options = _options(passes=3)
        for device, graph in self._modes():
            with self.subTest(device=str(device), graph=graph):
                source = copy.deepcopy(original)
                first = CalibrationWorkspace(**source, device=device, use_graph=graph)
                second = CalibrationWorkspace(**source, device=device, use_graph=graph)
                for value in source.values():
                    if isinstance(value, np.ndarray):
                        value.fill(0)
                expected, compression, stats = _legacy_solve(original, _params(), device, **options)
                first_force = first.solve(_params(), **options)
                self._assert_parity(first, first_force, expected, compression, stats)
                first_values = first_force.numpy().copy()
                first_compression = first.compression.numpy().copy()
                first_stats = copy.deepcopy(first.stats)
                second_force = second.solve(_params(True), **options)
                self.assertNotEqual(first_force.ptr, second_force.ptr)
                self.assertNotEqual(first.compression.ptr, second.compression.ptr)
                np.testing.assert_array_equal(first_force.numpy(), first_values)
                np.testing.assert_array_equal(first.compression.numpy(), first_compression)
                for key in ("sweeps_per_pass", "pass_change_m", "max_compression_m", "solver_remaining_m"):
                    np.testing.assert_array_equal(first.stats[key], first_stats[key])
                expected, compression, stats = _legacy_solve(original, _params(True), device, **options)
                self._assert_parity(second, second_force, expected, compression, stats)


if __name__ == "__main__":
    unittest.main()
