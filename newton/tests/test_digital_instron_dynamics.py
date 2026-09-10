# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dynamic elastic-foundation midsole example."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer
from projects.digital_instron_v2 import core, dynamics, workflow
from projects.digital_instron_v2.example import SURROUND_SWEEPS, Example
from projects.digital_instron_v2.geometry import build_column_grid, load_mesh
from projects.digital_shoe import runtime

MANIFEST = "DigitalInstron/manifest_v2.json"


def _available_devices() -> list:
    """Return every Warp device the probe kernels should be checked on."""
    devices = [wp.get_device("cpu")]
    if wp.is_cuda_available():
        devices.append(wp.get_device("cuda:0"))
    return devices


@wp.kernel
def _hyperfoam_probe(
    strain: wp.array[wp.float32],
    params: runtime.FoundationParams,
    pressure_out: wp.array[wp.float32],
):
    """Evaluate the runtime Hyperfoam law once per sample so a host test can read it."""
    i = wp.tid()
    pressure_out[i] = runtime._hyperfoam_pressure(strain[i], params)


@wp.kernel
def _pasternak_flux_probe(
    compression: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    params: runtime.FoundationParams,
    flux_out: wp.array[wp.float32],
):
    """Recompute the pairwise shear flux of every column exactly as ``foundation_apply`` does."""
    i = wp.tid()
    ci = compression[i]
    flux = float(0.0)
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            flux += runtime._pasternak_coupling(rest_len[i], rest_len[j], params) * (compression[j] - ci)
    flux_out[i] = flux


def _probe_params(material: core.Material) -> runtime.FoundationParams:
    """Build device-side constants for the probe kernels from a fitted material."""
    params = runtime.FoundationParams()
    poisson = core.EFFECTIVE_POISSON_RATIO
    # Both Ogden-Hill terms, through the one host helper that fills them, so the
    # probe cannot silently collapse a two-term material onto its first term.
    runtime.set_hyperfoam_series(params, material)
    params.beta = poisson / (1.0 - 2.0 * poisson)
    params.one_minus_two_poisson = 1.0 - 2.0 * poisson
    params.stretch_floor = 1.0e-3
    return params


def _run_mode(mode: str, num_frames: int) -> Example:
    """Instantiate and step the midsole example headlessly for ``num_frames``."""

    class _Args:
        pass

    args = _Args()
    args.mode = mode
    args.manifest = MANIFEST
    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    example = Example(viewer, args)
    for _ in range(num_frames):
        example.step()
    return example


class TestFoundationGeometry(unittest.TestCase):
    def test_material_loader_falls_back_to_calibrated_prediction_baseline(self):
        """Use the fitted prediction baseline when no ignored cache artifact exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(json.dumps({"cache_dir": "missing", "fit": {"unused": 1.0}}))
            material = dynamics.load_fitted_material(manifest)

        self.assertEqual(material, core.CALIBRATED_MATERIAL)

    def test_column_bed(self):
        """Sample the whole midsole with valid clearances, thicknesses, neighbours, and a driven subset.

        The bed spans every column of the manifest grid, so the only negative
        neighbour code left is the natural outer boundary; an interior gap code
        would mean the bed had dropped material the shear layer needs.
        """
        geo = dynamics.build_foundation_geometry(MANIFEST)
        grid = build_column_grid(load_mesh(geo.midsole_mesh_path, 0.001), geo.spacing_m)
        column_count = len(geo.slack_m)
        self.assertEqual(column_count, len(grid.slack_m))
        self.assertTrue(np.all(geo.slack_m > 0.0))
        self.assertTrue(np.all(geo.gap0_m >= 0.0))
        self.assertTrue(np.all(geo.z_free_m > geo.z_bottom_m))
        self.assertEqual(geo.neighbors.shape, (column_count, 4))
        self.assertTrue(np.all(geo.neighbors < column_count))
        self.assertTrue(np.all(geo.neighbors >= -1))
        # The indenter drives a strict, nonempty subset; the rest is the passive surround.
        self.assertEqual(geo.driven.shape, (column_count,))
        self.assertGreater(int(geo.driven.sum()), 0)
        self.assertLess(int(geo.driven.sum()), column_count)
        # Columns the indenter misses are anchored on their own foam top.
        self.assertTrue(np.all(geo.gap0_m[~geo.driven] == 0.0))

    def test_pasternak_neighbours_match_calibration_laplacian(self):
        """Reconstruct the compression Laplacian from the neighbour table and match the calibration operator."""
        geo = dynamics.build_foundation_geometry(MANIFEST)
        grid = build_column_grid(load_mesh(geo.midsole_mesh_path, 0.001), geo.spacing_m)
        rng = np.random.default_rng(0)
        compression = rng.random(len(geo.slack_m))

        laplacian = np.empty_like(compression)
        for i in range(len(compression)):
            total = -4.0 * compression[i]
            for side in range(4):
                j = geo.neighbors[i, side]
                if j >= 0:
                    total += compression[j]
                elif j == -1:
                    total += compression[i]
            laplacian[i] = total / geo.spacing_m**2

        reference = workflow.compression_laplacian(compression[None, :], geo.uv_m, grid.uv_m, geo.spacing_m)[0]
        np.testing.assert_allclose(laplacian, reference, atol=1.0e-10)

    def test_pasternak_flux_cancels_over_the_free_edged_bed(self):
        """Sum the material-pinned shear flux over the whole bed and require machine zero.

        The layer may move load between columns but must never create it. Writing
        the coupling as a symmetric per-face term makes every pair cancel exactly,
        which is the guarantee that lets the reported Instron force be the summed
        unilateral ground reaction with nothing else booked into it.
        """
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = core.CALIBRATED_MATERIAL
        params = _probe_params(material)
        rng = np.random.default_rng(0)
        compression = np.ascontiguousarray(rng.random(len(geo.slack_m)) * 0.02, np.float32)

        for device in _available_devices():
            with self.subTest(device=str(device)):
                flux = wp.zeros(len(geo.slack_m), dtype=wp.float32, device=device)
                wp.launch(
                    _pasternak_flux_probe,
                    dim=len(geo.slack_m),
                    inputs=[
                        wp.array(compression, dtype=wp.float32, device=device),
                        wp.array(np.ascontiguousarray(geo.slack_m, np.float32), dtype=wp.float32, device=device),
                        wp.array(np.ascontiguousarray(geo.neighbors, np.int32), dtype=wp.int32, device=device),
                        params,
                    ],
                    outputs=[flux],
                    device=device,
                )
                values = flux.numpy().astype(np.float64)
                self.assertGreater(float(np.abs(values).sum()), 1.0)  # the layer really is transporting load
                self.assertLess(abs(float(values.sum())), 1.0e-6 * float(np.abs(values).sum()))

    def test_surround_solver_reaches_the_swept_answer(self):
        """Land on the fully swept bed with residual stopping and a warm start.

        Almost all of a fit is spent in this relaxation, so it stops on an
        extrapolated remaining travel instead of a fixed sweep count, and reuses
        the previous solve. Both are only legitimate if they reach the same fixed
        point, which is what this pins; it also pins that the cheap path really
        is cheap, because a stopping rule that never fires is not a speed-up.
        """
        geo = dynamics.build_foundation_geometry(MANIFEST)
        params = _probe_params(core.CALIBRATED_MATERIAL)
        driven = np.asarray(geo.driven, dtype=bool)
        imposed = np.tile(np.linspace(0.0, 0.012, 5)[:, None], (1, int(driven.sum())))
        kwargs = {
            "area_m2": geo.area_m2,
            "spacing_m": geo.spacing_m,
            "attachment_n_m": 0.0,
            "max_strain": 0.9,
            "sweeps": 3000,
        }
        swept_stats: dict[str, float] = {}
        swept = runtime.relax_surround(
            imposed, driven, geo.neighbors, geo.slack_m, params, stats=swept_stats, **kwargs
        ).numpy()

        cold_stats: dict[str, float] = {}
        cold = runtime.relax_surround(
            imposed,
            driven,
            geo.neighbors,
            geo.slack_m,
            params,
            tolerance_m=core.SURROUND_SOLVE_TOLERANCE_M,
            check_every=core.SURROUND_CHECK_EVERY,
            stats=cold_stats,
            **kwargs,
        )
        warm_stats: dict[str, float] = {}
        warm = runtime.relax_surround(
            imposed,
            driven,
            geo.neighbors,
            geo.slack_m,
            params,
            initial=cold,
            tolerance_m=core.SURROUND_SOLVE_TOLERANCE_M,
            check_every=core.SURROUND_CHECK_EVERY,
            stats=warm_stats,
            **kwargs,
        ).numpy()

        self.assertEqual(swept_stats["sweeps"], 3000.0)  # no tolerance means the cap runs in full
        self.assertLess(cold_stats["sweeps"], 3000.0)
        self.assertLess(float(np.max(np.abs(cold.numpy() - swept))), 1.0e-6)
        self.assertLess(float(np.max(np.abs(warm - swept))), 1.0e-6)

    def test_zero_poisson_hyperfoam_matches_the_host_law_on_every_device(self):
        """Evaluate the Hyperfoam law at zero Poisson ratio on CPU and GPU without a degenerate power.

        At ``nu = 0`` the volumetric exponent is one and ``beta`` is zero, so the
        volumetric factor is ``pow(x, 0)``. The stretch floor keeps ``x`` strictly
        positive, so that is exactly one and no ``pow(0, 0)`` or division by zero
        can appear, including past full densification.
        """
        material = core.CALIBRATED_MATERIAL
        params = _probe_params(material)
        strain = np.ascontiguousarray(np.linspace(0.0, 1.2, 25), np.float32)
        expected = core._hyperfoam_pressure(strain.astype(np.float64), material)

        self.assertEqual(params.beta, 0.0)
        self.assertEqual(params.one_minus_two_poisson, 1.0)
        for device in _available_devices():
            with self.subTest(device=str(device)):
                pressure = wp.zeros(len(strain), dtype=wp.float32, device=device)
                wp.launch(
                    _hyperfoam_probe,
                    dim=len(strain),
                    inputs=[wp.array(strain, dtype=wp.float32, device=device), params],
                    outputs=[pressure],
                    device=device,
                )
                values = pressure.numpy().astype(np.float64)
                self.assertTrue(np.all(np.isfinite(values)))
                np.testing.assert_allclose(values, expected, rtol=1.0e-5)


class TestMidsoleFoundation(unittest.TestCase):
    def test_kernel_reproduces_calibrated_model(self):
        """Sweep the Warp foundation through the Instron cycle and match core.predict to float precision.

        The live per-substep Hyperfoam-Maxwell-Pasternak force integration must reproduce
        the periodic-fixed-point forward model of the same column bed.

        Identification and simulation are one model now: both span the whole midsole,
        drive the same indenter footprint, and settle the untouched foam with the shared
        :func:`projects.digital_shoe.runtime._surround_balance`. The trial therefore keeps
        its ``Trial.surround`` instead of being stripped, so this compares the live
        per-substep integration against the periodic fixed point of the identical bed.

        Previously the trial had to be stripped because the live foundation held every
        cell outside the fixture footprint at zero compression, which was a modelling
        gap, not a kernel difference.
        """
        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        midsole = load_mesh(base / config["midsole_mesh"], 0.001)
        grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
        trials, _, _ = workflow.prepare_trials(base, config, grid, midsole)
        trial = next(t for t in trials if t.name == "fullfoot_185ms")
        self.assertIsNotNone(trial.surround)

        material = dynamics.load_fitted_material(MANIFEST)
        predicted = core.predict(trial, material)

        geo = dynamics.build_foundation_geometry(MANIFEST)
        device = wp.get_preferred_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.surface_m])
        foundation = dynamics.MidsoleFoundation(
            anchor,
            geo.z_free_m,
            geo.slack_m,
            np.full(len(geo.slack_m), geo.area_m2),
            geo.neighbors,
            geo.spacing_m,
            material,
            body,
            model.body_com,
            dynamics.FoundationConfig(stretch_floor=1.0e-3),
            device,
            dynamics.SurroundConfig(
                driven=geo.driven,
                attachment_n_m=trial.surround.attachment_n_m,
                max_strain=trial.surround.max_strain,
                sweeps=SURROUND_SWEEPS,
            ),
        )

        displacement = np.asarray(trial.displacement_m)
        dt = np.asarray(trial.dt_s)
        collected = np.zeros_like(displacement)
        for cycle in range(6):
            for k in range(len(displacement)):
                state.body_q.assign(np.array([[0.0, 0.0, -displacement[k], 0.0, 0.0, 0.0, 1.0]], np.float32))
                state.body_qd.zero_()
                state.clear_forces()
                foundation.apply(state, float(dt[k]))
                if cycle == 5:
                    collected[k] = foundation.diagnostics()["normal_force_n"]

        peak = float(predicted.max())
        self.assertLess(np.sqrt(np.mean((collected - predicted) ** 2)) / peak, 5.0e-3)
        self.assertLess(abs(collected.max() - peak) / peak, 2.0e-2)

    def test_bristle_friction_sticks_below_cone_and_slips_above(self):
        """Verify anchored stick-slip friction: a planted patch resists a static offset, saturating at the cone.

        A purely viscous friction law produces zero tangential force at zero slip velocity,
        so the static-offset probes below would all read zero and fail without the anchored
        bristle model.

        The cone is per column and unilateral: it is built on the pressed (nonnegative)
        Pasternak base pressure, while the reported normal force is the net wrench, which
        also carries the small pull of the columns the shear layer lifts. The saturated
        total therefore equals ``mu`` times the summed pressed load, not ``mu`` times the
        net normal force, and the two differ by that lifted part.
        """
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        device = wp.get_preferred_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.surface_m])
        kt, mu, depth = 2.0e4, 1.0, 0.006
        foundation = dynamics.MidsoleFoundation(
            anchor,
            geo.z_free_m,
            geo.slack_m,
            np.full(len(geo.slack_m), geo.area_m2),
            geo.neighbors,
            geo.spacing_m,
            material,
            body,
            model.body_com,
            dynamics.FoundationConfig(stretch_floor=1.0e-3, friction_stiffness=kt, friction=0.0, mu=mu),
            device,
            dynamics.SurroundConfig(driven=geo.driven, sweeps=SURROUND_SWEEPS),
        )

        def probe(dx: float) -> tuple[float, float, float]:
            # Hold the bed a fixed depth into the ground and offset it by ``dx`` tangentially with
            # zero velocity, returning the tangential reaction opposing the offset, the pressed
            # load the per-column cones are built on, and the net normal wrench on the carrier.
            state.body_q.assign(np.array([[dx, 0.0, -depth, 0.0, 0.0, 0.0, 1.0]], np.float32))
            state.body_qd.zero_()
            state.clear_forces()
            foundation.apply(state, 1.0e-3)
            diagnostics = foundation.diagnostics()
            return (
                -float(state.body_f.numpy()[0][0]),
                diagnostics["pressed_force_n"],
                diagnostics["normal_force_n"],
            )

        seat_force, pressed_force, normal_force = probe(0.0)  # fresh contact seats the bristles unstretched
        cone = mu * pressed_force
        # The traction cones are built on the pressed load, so that is what has to be
        # substantial for this probe. The reference moved from the net normal wrench
        # (previously required above 100 N) because the bed now spans the whole midsole.
        self.assertGreater(pressed_force, 200.0)
        # With the outer bond removed and the shear flux cancelling pairwise, the net
        # wrench is exactly the summed unilateral ground reaction, so it can no longer
        # come out negative the way the hidden bond used to make it.
        self.assertGreater(normal_force, 0.0)
        # The shear layer still lifts the columns around every dimple of the last, and a
        # lifted column transmits a small pull, so the pressed load that carries traction
        # exceeds the net normal wrench.
        self.assertGreater(pressed_force, normal_force)
        self.assertLess(abs(seat_force), 1.0e-3 * cone)

        near_force, *_ = probe(2.0e-6)  # 2 um static offset -> static shear with zero slip velocity
        stuck = int(foundation.tangent_stuck.numpy().sum())
        far_force, *_ = probe(4.0e-6)  # 4 um static offset -> larger elastic build-up, still stuck
        # Every gripping bristle is a linear spring, so kt * dx per stuck column is the
        # ceiling of the static force and doubling the offset doubles it. The ceiling is no
        # longer reached exactly: with the outer bond removed, the perimeter columns the
        # shear layer only lightly loads have a tiny cone of their own, so a few percent of
        # the bristles saturate at any offset. The probe therefore uses micrometre offsets,
        # where saturation costs about a tenth of the total and the response is linear to
        # within one percent, and it bounds rather than equates the elastic sum.
        self.assertLess(near_force, kt * 2.0e-6 * stuck)
        self.assertGreater(near_force, 0.8 * kt * 2.0e-6 * stuck)
        self.assertAlmostEqual(far_force, 2.0 * near_force, delta=0.05 * near_force)
        self.assertLess(far_force, cone)

        foundation.reset()  # release the bristles, then re-seat before driving well past the cone
        probe(0.0)
        slip_force, slip_pressed, _ = probe(0.02)  # 20 mm offset saturates every column at mu * pressed
        self.assertAlmostEqual(slip_force, mu * slip_pressed, delta=0.03 * mu * slip_pressed)


class TestMidsoleExample(unittest.TestCase):
    def test_instron_hysteresis(self):
        """Run the digital Instron mode and audit its dissipative hysteresis loop."""
        example = _run_mode("instron", 190)
        example.test_final()

    def test_settle_supports_mass_with_friction(self):
        """Run the massive-midsole mode and audit stable weight support and lateral grip."""
        example = _run_mode("settle", 150)
        example.test_final()

    def test_stride_ground_reaction(self):
        """Run the synthetic-stride mode and audit its ground-reaction force and center-of-pressure roll."""
        example = _run_mode("stride", 40)
        example.test_final()

    def test_attached_dynamic_stride(self):
        """Run the attached, foot-mounted dynamic shoe and audit its stance/flight ground reaction.

        The dynamic shoe carries mass and inertia and stays coupled to the foot the whole
        stride, so the run must remain finite (no flight blow-up), develop a real stance
        ground reaction, and unload to near zero in flight without the shoe separating.
        """
        example = _run_mode("attached", 95)
        example.test_final()


if __name__ == "__main__":
    unittest.main()
