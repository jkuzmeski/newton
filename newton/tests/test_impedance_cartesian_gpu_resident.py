# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the device-resident Cartesian equilibrium optimizer.

The tests use a small quadratic Warp objective.  It is deliberately synthetic:
its only purpose is to make host transfers, invalid rollouts, and winner
selection observable without running a physical simulation.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.unittest_utils import get_test_devices
from projects.impedance_instron.cartesian.gpu import resident
from projects.impedance_instron.cartesian.trajectory import Spline

PROFILE = {
    "equilibrium_lower": [-0.18706925802385221, 0.43664648117189697, -2.240879416600583, -2.0605111429635445],
    "equilibrium_upper": [1.90736507709526, 1.5050702188249552, 1.4235366453481377, 1.7643870219635445],
    "equilibrium_rate_limit": [10.0, 10.0, 20.0, 25.0],
    "equilibrium_acceleration_limit": [200.0, 200.0, 400.0, 600.0],
}

# The values are inside PROFILE and leave enough room for poll/trial steps.
INITIAL_COEFFICIENTS = np.array(
    [
        [0.45459739588202797, 0.9340312757475928, -0.3055765178073986, 0.004710197607608539],
        [0.3937610999183263, 1.0286930233898068, 0.08976600833185336, -0.07940390562580132],
        [0.6626990815913381, 0.8187369239817625, -0.7182846715211476, 0.28482009988251367],
        [1.0514718377945014, 0.8897660782037417, 0.41244079908766146, -0.055474844019835025],
        [1.300622790946952, 1.040936077392083, -0.01786034618299534, -0.7283481743749493],
        [1.4152195376949004, 1.0731203456945806, -0.6668291924918452, -0.7623658402281136],
    ],
    dtype=np.float64,
)


def _initial_coefficients(control_count: int) -> np.ndarray:
    """Interpolate the canonical valid fixture to a requested control count."""
    if control_count == 6:
        return INITIAL_COEFFICIENTS.copy()
    new_u = np.linspace(0.0, 1.0, control_count)[:, None]
    return INITIAL_COEFFICIENTS[0] + new_u * (INITIAL_COEFFICIENTS[-1] - INITIAL_COEFFICIENTS[0])


@wp.kernel
def _quadratic_kernel(
    coefficients: wp.array3d[wp.float64],
    target: wp.array2d[wp.float64],
    loss: wp.array[wp.float64],
    residual: wp.array2d[wp.float64],
    rmse: wp.array2d[wp.float64],
    costs: wp.array2d[wp.float64],
    integrated: wp.array[int],
    failure: wp.array[int],
    steps: int,
    launch_number: int,
    fail_first: int,
    fail_all: int,
    worsen_later: int,
    control_count: int,
    parameter_count: int,
):
    """Evaluate a quadratic residual without a device-to-host transfer."""
    world = wp.tid()
    value = wp.float64(0.0)
    for k in range(parameter_count):
        control = k // 4
        channel = k % 4
        diff = coefficients[world, control, channel] - target[world, k]
        residual[k, world] = diff
        value += diff * diff
    if worsen_later != 0 and launch_number > 1:
        residual[0, world] = wp.sqrt(residual[0, world] * residual[0, world] + wp.float64(100.0))
        value += wp.float64(100.0)
    loss[world] = value
    # Populate every field consumed by _evaluate_usable_kernel.
    for i in range(6):
        rmse[world, i] = wp.sqrt(value + wp.float64(1.0e-30))
    for i in range(3):
        costs[world, i] = value
    integrated[world] = steps
    failure[world] = 0
    if fail_all == 1 or (fail_first == 1 and launch_number == 1):
        failure[world] = 9


@wp.kernel
def _copy_snapshot_kernel(
    winner: wp.array[int],
    coefficients: wp.array3d[wp.float64],
    loss: wp.array[wp.float64],
    residual: wp.array2d[wp.float64],
    integrated: wp.array[int],
    failure: wp.array[int],
    snapshot_coefficients: wp.array3d[wp.float64],
    snapshot_loss: wp.array[wp.float64],
    snapshot_residual: wp.array2d[wp.float64],
    snapshot_integrated: wp.array[int],
    snapshot_failure: wp.array[int],
    control_count: int,
    parameter_count: int,
):
    """Gather one winning world into a device-resident snapshot."""
    k = wp.tid()
    source = winner[0]
    if source < 0:
        return
    if k < parameter_count:
        control = k // 4
        channel = k % 4
        snapshot_coefficients[0, control, channel] = coefficients[source, control, channel]
        snapshot_residual[k, 0] = residual[k, source]
    if k == 0:
        snapshot_loss[0] = loss[source]
        snapshot_integrated[0] = integrated[source]
        snapshot_failure[0] = failure[source]


@wp.kernel
def _check_bounds_kernel(
    coeffs: wp.array3d[wp.float64],
    output: wp.array[int],
    duration: wp.float64,
    lower: wp.vec4d,
    upper: wp.vec4d,
    rates: wp.vec4d,
    accelerations: wp.vec4d,
    first_scale: wp.array[wp.float64],
    second_scale: wp.array[wp.float64],
    control_count: int,
):
    """Evaluate bounds for one candidate per thread."""
    output[wp.tid()] = resident._check_bounds(
        coeffs, wp.tid(), duration, lower, upper, rates, accelerations, first_scale, second_scale, control_count
    )


class _Snapshot:
    """Device snapshot with host reads restricted to score/trace at the end."""

    def __init__(self, engine: _QuadraticEngine):
        self.engine = engine
        dev = engine.device
        self.coefficients = wp.zeros((1, engine.control_count, 4), dtype=wp.float64, device=dev)
        self.loss = wp.zeros(1, dtype=wp.float64, device=dev)
        self.residual = wp.zeros((engine.parameter_count, 1), dtype=wp.float64, device=dev)
        self.integrated = wp.zeros(1, dtype=wp.int32, device=dev)
        self.failure = wp.zeros(1, dtype=wp.int32, device=dev)

    def score(self) -> dict:
        """Read the saved score once the search has ended."""
        self.engine.snapshot_host_reads += 1
        return {
            "loss": self.loss.numpy(),
            "rmse": np.zeros((1, 6), dtype=np.float64),
            "costs": np.zeros((1, 3), dtype=np.float64),
            "residual": self.residual.numpy(),
            "integrated_steps": self.integrated.numpy(),
            "failure_code": self.failure.numpy(),
        }

    def trace(self) -> tuple[dict, dict]:
        """Read a minimal trace from the saved state."""
        self.engine.snapshot_host_reads += 1
        return (
            {"time_s": np.linspace(0.0, self.engine.duration, self.engine.steps + 1)},
            {"metrics": {}, "summary": {}},
        )


class _QuadraticEngine:
    """Synthetic resident engine whose objective is a device quadratic bowl."""

    def __init__(
        self,
        world_count: int,
        device,
        *,
        control_count: int = 12,
        fail_first: bool = False,
        fail_all: bool = False,
        worsen_later: bool = False,
    ):
        self.world_count = world_count
        self.control_count = control_count
        self.parameter_count = 4 * control_count
        self.duration = 0.36
        self.steps = 8
        self.chunk_steps = 8
        self.device = device
        self.settings = SimpleNamespace(control_count=control_count, parameter_scale=np.array([0.05, 0.05, 0.1, 0.1]))
        self.profile = dict(PROFILE)
        self.coefficients = wp.zeros((world_count, control_count, 4), dtype=wp.float64, device=device)
        self.states = wp.zeros(1, dtype=wp.float64, device=device)
        self.forces = wp.zeros(1, dtype=wp.float64, device=device)
        self.integrated = wp.zeros(world_count, dtype=wp.int32, device=device)
        self.failure = wp.zeros(world_count, dtype=wp.int32, device=device)
        self.residual_dim = self.parameter_count
        self.objective = SimpleNamespace(
            residual_dim=self.parameter_count,
            description={"kind": "synthetic_quadratic"},
            loss=wp.zeros(world_count, dtype=wp.float64, device=device),
            rmse=wp.zeros((world_count, 6), dtype=wp.float64, device=device),
            costs=wp.zeros((world_count, 3), dtype=wp.float64, device=device),
            residual=wp.zeros((self.parameter_count, world_count), dtype=wp.float64, device=device),
        )
        # Target is deliberately different from the initial spline.
        initial = _initial_coefficients(control_count)
        target = initial.reshape(1, self.parameter_count) + np.tile([0.01, -0.008, 0.006, -0.004], control_count)
        self.target = wp.array(np.repeat(target, world_count, axis=0), dtype=wp.float64, device=device)
        self.capture_calls = 0
        self.capture_resident_calls = 0
        self.evaluate_device_calls = 0
        self.evaluate_calls = 0
        self.trace_calls = 0
        self.snapshot_host_reads = 0
        self.create_snapshot_calls = 0
        self.snapshot_device_calls = 0
        self.fail_first = fail_first
        self.fail_all = fail_all
        self.worsen_later = worsen_later
        self._captured = False

    def capture(self, coefficients):
        """Record legacy graph capture, which is allowed only during setup."""
        self.capture_calls += 1
        self._captured = True

    def capture_resident(self):
        """Record resident graph capture without reading device arrays."""
        self.capture_resident_calls += 1
        self._captured = True

    def evaluate_device(self):
        """Run the complete objective on device; never call .numpy or .assign."""
        self.evaluate_device_calls += 1
        wp.launch(
            _quadratic_kernel,
            dim=self.world_count,
            inputs=[
                self.coefficients,
                self.target,
                self.objective.loss,
                self.objective.residual,
                self.objective.rmse,
                self.objective.costs,
                self.integrated,
                self.failure,
                self.steps,
                self.evaluate_device_calls,
                int(self.fail_first),
                int(self.fail_all),
                int(self.worsen_later),
                self.control_count,
                self.parameter_count,
            ],
            device=self.device,
        )

    def evaluate(self, coefficients):
        """Reject the old host-evaluating API during resident search."""
        self.evaluate_calls += 1
        raise AssertionError("resident search must not call Engine.evaluate")

    def create_snapshot(self):
        """Allocate the one-world snapshot before search begins."""
        self.create_snapshot_calls += 1
        self.latest_snapshot = _Snapshot(self)
        return self.latest_snapshot

    def snapshot_device(self, winner, snapshot):
        """Copy the selected world to snapshot arrays on device."""
        self.snapshot_device_calls += 1
        wp.launch(
            _copy_snapshot_kernel,
            dim=self.parameter_count,
            inputs=[
                winner,
                self.coefficients,
                self.objective.loss,
                self.objective.residual,
                self.integrated,
                self.failure,
                snapshot.coefficients,
                snapshot.loss,
                snapshot.residual,
                snapshot.integrated,
                snapshot.failure,
                self.control_count,
                self.parameter_count,
            ],
            device=self.device,
        )

    def trace(self, world: int = 0):
        """Reject legacy trace extraction; snapshots provide the trace."""
        self.trace_calls += 1
        raise AssertionError("resident search must use the saved snapshot")


class TestImpedanceCartesianGpuResident(unittest.TestCase):
    """Exercise resident kernels and search using the synthetic objective."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp and select the available CPU and CUDA test devices."""
        wp.init()
        cls.devices = [device for device in get_test_devices() if device.is_cpu or device.is_cuda]
        if not cls.devices:
            cls.devices = [wp.get_device("cpu")]

    def _initial(self, control_count: int = 12) -> Spline:
        """Build a valid synthetic starting spline."""
        return Spline(0.36, _initial_coefficients(control_count))

    def _engine(self, device, control_count: int = 12, **kwargs) -> _QuadraticEngine:
        """Build an engine with the resident optimizer's exact world layout."""
        worlds = 128
        return _QuadraticEngine(worlds, device, control_count=control_count, **kwargs)

    def test_bounds_kernel_matches_spline_at_boundary_on_all_devices(self):
        """Match CPU bounds for twelve controls, including a near-boundary violation."""
        limits = tuple(
            PROFILE[key]
            for key in (
                "equilibrium_lower",
                "equilibrium_upper",
                "equilibrium_rate_limit",
                "equilibrium_acceleration_limit",
            )
        )
        for control_count in (12,):
            spline = self._initial(control_count)
            candidates = np.stack([spline.coefficients, spline.coefficients.copy()])
            candidates[1, 0, 0] = PROFILE["equilibrium_lower"][0] - 1.0e-12
            expected = [Spline(spline.duration_s, candidate).bounds(*limits) for candidate in candidates]
            first_scale, second_scale = resident.canonical_knot_scales()
            for device in self.devices:
                with self.subTest(control_count=control_count, device=device):
                    coeffs = wp.array(candidates, dtype=wp.float64, device=device)
                    valid = wp.zeros(2, dtype=wp.int32, device=device)
                    wp.launch(
                        _check_bounds_kernel,
                        dim=2,
                        inputs=[
                            coeffs,
                            valid,
                            spline.duration_s,
                            wp.vec4d(*limits[0]),
                            wp.vec4d(*limits[1]),
                            wp.vec4d(*limits[2]),
                            wp.vec4d(*limits[3]),
                            wp.array(first_scale, dtype=wp.float64, device=device),
                            wp.array(second_scale, dtype=wp.float64, device=device),
                            control_count,
                        ],
                        device=device,
                    )
                    self.assertEqual(expected, [bool(value) for value in valid.numpy()])

    def test_usable_kernel_rejects_failed_partial_and_nonfinite_rollouts(self):
        """Reject every failed, partial, negative, or nonfinite candidate."""
        device = self.devices[0]
        count = 7
        real = wp.ones(count, dtype=wp.int32, device=device)
        failure = wp.zeros(count, dtype=wp.int32, device=device)
        integrated = wp.full(count, 8, dtype=wp.int32, device=device)
        loss = wp.ones(count, dtype=wp.float64, device=device)
        rmse = wp.ones((count, 6), dtype=wp.float64, device=device)
        costs = wp.ones((count, 3), dtype=wp.float64, device=device)
        residual = wp.zeros((24, count), dtype=wp.float64, device=device)
        # Construct bad values on host before one device upload; no temporary .numpy mutation.
        failure_h = np.zeros(count, dtype=np.int32)
        failure_h[1] = 2
        integrated_h = np.full(count, 8, dtype=np.int32)
        integrated_h[2] = 7
        loss_h = np.ones(count)
        loss_h[3] = np.nan
        loss_h[4] = -1.0
        rmse_h = np.ones((count, 6))
        rmse_h[5, 2] = np.inf
        residual_h = np.zeros((24, count))
        residual_h[6, 6] = np.nan
        failure = wp.array(failure_h, dtype=wp.int32, device=device)
        integrated = wp.array(integrated_h, dtype=wp.int32, device=device)
        loss = wp.array(loss_h, dtype=wp.float64, device=device)
        rmse = wp.array(rmse_h, dtype=wp.float64, device=device)
        residual = wp.array(residual_h, dtype=wp.float64, device=device)
        usable = wp.zeros(count, dtype=wp.int32, device=device)
        completed = wp.zeros(count, dtype=wp.int32, device=device)
        active = wp.ones(1, dtype=wp.int32, device=device)
        wp.launch(
            resident._evaluate_usable_kernel,
            dim=count,
            inputs=[real, failure, integrated, loss, rmse, costs, residual, 8, 24, usable, completed, active],
            device=device,
        )
        self.assertEqual(completed.numpy().tolist(), [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(usable.numpy().tolist(), [1, 0, 0, 0, 0, 0, 0])

    def test_resident_search_uses_snapshot_and_never_legacy_io(self):
        """Run CPU and CUDA searches without legacy evaluate, trace, or host reads."""
        for device in self.devices:
            with self.subTest(device=device):
                engine = self._engine(device)
                result = resident.fit_resident(engine, self._initial(), max_iterations=2, max_wall_s=10.0, seed=3)
                _, _, _, summary = result
                self.assertEqual(engine.world_count, 128)
                self.assertEqual(summary["optimizer"]["batch_size_policy"], "fixed")
                self.assertEqual(summary["optimizer"]["independent_controllers"], 1)
                self.assertEqual(summary["optimizer"]["worlds"], 128)
                self.assertEqual(summary["counts"]["physics_worlds"], 128 * summary["counts"]["batches"])
                self.assertGreater(engine.evaluate_device_calls, 0)
                self.assertEqual(engine.create_snapshot_calls, 1)
                self.assertGreaterEqual(engine.snapshot_device_calls, engine.evaluate_device_calls)
                self.assertEqual(summary["counts"]["proposed_candidates"], summary["counts"]["physics_worlds"])
                self.assertEqual(engine.evaluate_calls, 0)
                self.assertEqual(engine.trace_calls, 0)
                self.assertGreaterEqual(engine.snapshot_host_reads, 1)
                self.assertTrue(np.isfinite(summary["loss"]))

    def test_failed_initial_poll_cannot_win_or_corrupt_snapshot(self):
        """Keep a failed initial rollout out of selection and preserve safe diagnostics."""
        engine = self._engine(self.devices[0], fail_first=True)
        _, _, _, summary = resident.fit_resident(engine, self._initial(), max_iterations=1, max_wall_s=10.0, seed=5)
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["best"])
        self.assertIsNone(summary["loss"])
        self.assertEqual(summary["termination"], "initial_rollout_incomplete")
        self.assertEqual(engine.evaluate_calls, 0)

    def test_global_best_survives_a_later_worse_batch_without_replay(self):
        """Report the saved best candidate even when subsequent proposals are worse."""
        engine = self._engine(self.devices[0], worsen_later=True)
        result, _, _, summary = resident.fit_resident(
            engine, self._initial(), max_iterations=3, max_wall_s=10.0, seed=12
        )
        self.assertLess(summary["loss"], 100.0)
        self.assertGreaterEqual(float(engine.objective.loss.numpy().min()), 100.0)
        np.testing.assert_array_equal(result.coefficients, engine.latest_snapshot.coefficients.numpy()[0])
        self.assertEqual(summary["loss"], float(engine.latest_snapshot.loss.numpy()[0]))
        self.assertEqual(engine.evaluate_calls, 0)
        self.assertEqual(engine.trace_calls, 0)
        self.assertEqual(summary["counts"]["batches"], 6)
        self.assertIn("no host full-world replay", summary["optimizer"].get("final_replay", ""))

    def test_minimum_step_and_tiny_wall_still_execute_initial_poll(self):
        """Do not claim convergence at minimum step or skip the mandatory initial poll."""
        engine = self._engine(self.devices[0])
        _, _, _, summary = resident.fit_resident(
            engine, self._initial(), max_iterations=2, max_wall_s=1.0e-12, seed=2, minimum_step_fraction=0.05
        )
        self.assertGreaterEqual(engine.evaluate_device_calls, 1)
        self.assertGreaterEqual(summary["counts"]["batches"], 1)
        self.assertNotEqual(summary["termination"], "converged")

    def test_counts_distinguish_padding_failed_and_completed_execution(self):
        """Account for every device execution and do not count padding as physics."""
        engine = self._engine(self.devices[0], fail_all=True)
        _, _, _, summary = resident.fit_resident(engine, self._initial(), max_iterations=1, max_wall_s=30.0, seed=8)
        counts = summary["counts"]
        self.assertEqual(engine.world_count, 128)
        self.assertEqual(counts["batches"], 2)
        self.assertEqual(counts["physics_worlds"], 2 * engine.world_count)
        self.assertEqual(counts["proposed_candidates"], engine.world_count)
        self.assertEqual(counts["completed_physics_worlds"], 0)
        failed_worlds = counts["physics_worlds"] - counts["completed_physics_worlds"]
        self.assertEqual(failed_worlds, 2 * engine.world_count)
        self.assertEqual(counts["physics_worlds"], engine.evaluate_device_calls * engine.world_count)

    def test_runtime_transfer_guard_catches_an_injected_read(self):
        """Reject any fitting-array transfer inside the real loop, including nested engine calls."""
        lines, first_line = inspect.getsourcelines(resident.fit_resident)
        tree = ast.parse("".join(lines))
        loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "_iteration"
        )
        lower, upper = first_line + loop.lineno - 1, first_line + loop.end_lineno - 1

        def in_search():
            frame = inspect.currentframe()
            while frame:
                if frame.f_code is resident.fit_resident.__code__:
                    return lower <= frame.f_lineno <= upper
                frame = frame.f_back
            return False

        def guard(original):
            def checked(*args, **kwargs):
                if in_search():
                    raise AssertionError("host fitting-array transfer inside search")
                return original(*args, **kwargs)

            return checked

        original_copy = wp.copy

        def checked_copy(dest, src, *args, **kwargs):
            if in_search() and dest.device.is_cpu != src.device.is_cpu:
                raise AssertionError("host fitting-array transfer inside search")
            return original_copy(dest, src, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(wp.array, "numpy", guard(wp.array.numpy)))
            stack.enter_context(patch.object(wp.array, "assign", guard(wp.array.assign)))
            stack.enter_context(patch.object(wp, "copy", checked_copy))
            for device in self.devices:
                engine = _QuadraticEngine(128, device, control_count=12)
                resident.fit_resident(engine, self._initial(), max_iterations=2, max_wall_s=20, plateau_patience=None)
                engine = _QuadraticEngine(128, device, control_count=12)
                engine.evaluate_device = engine.coefficients.numpy
                with self.assertRaisesRegex(AssertionError, "host fitting-array transfer"):
                    resident.fit_resident(
                        engine, self._initial(), max_iterations=1, max_wall_s=20, plateau_patience=None
                    )

    def test_first_failure_keeps_original_controller_and_diagnostic(self):
        """Keep the original failed diagnostic across later complete but disabled rollouts."""
        engine = self._engine(self.devices[0], fail_first=True)
        initial = self._initial()
        result, _, _, summary = resident.fit_resident(engine, initial, max_iterations=3, max_wall_s=20)
        np.testing.assert_array_equal(result.coefficients, initial.coefficients)
        self.assertIsNone(summary["best"])
        self.assertIsNone(summary["loss"])
        self.assertEqual(summary["counts"]["physics_worlds"], 6 * 128)
        self.assertEqual(summary["counts"]["completed_physics_worlds"], 5 * 128)
        self.assertEqual(summary["counts"]["real_candidates"], 128)
        self.assertEqual(summary["counts"]["padding_worlds"], 5 * 128)

    def test_generic_search_preserves_residency_counters_and_failure_safety(self):
        """Preserve fixed-layout residency counters and failure safety."""
        device = self.devices[0]
        engine = self._engine(device, control_count=12)
        result, _, _, summary = resident.fit_resident(
            engine, self._initial(12), max_iterations=1, max_wall_s=30.0, seed=19
        )
        self.assertEqual(result.coefficients.shape, (12, 4))
        self.assertEqual(summary["optimizer"]["worlds_per_island"], 128)
        self.assertEqual(summary["optimizer"]["control_count"], 12)
        self.assertEqual(summary["counts"]["physics_worlds"], engine.evaluate_device_calls * engine.world_count)
        self.assertEqual(engine.evaluate_calls, 0)
        self.assertEqual(engine.trace_calls, 0)
        failed = self._engine(device, control_count=12, fail_first=True)
        result, _, _, failure_summary = resident.fit_resident(
            failed, self._initial(12), max_iterations=1, max_wall_s=30.0, seed=23
        )
        np.testing.assert_array_equal(result.coefficients, self._initial(12).coefficients)
        self.assertEqual(failure_summary["termination"], "initial_rollout_incomplete")
        self.assertIsNone(failure_summary["loss"])
        self.assertEqual(failed.evaluate_calls, 0)

    def test_wall_budget_does_not_admit_a_batch_that_will_overrun(self):
        """Stop before the next measured full batch would exceed the wall budget."""
        engine = self._engine(self.devices[0])
        clock = [0.0]
        original = engine.evaluate_device

        def timed_evaluate():
            """Charge each complete batch three seconds of synthetic wall time."""
            original()
            clock[0] += 3.0

        def now():
            """Return the synthetic wall clock."""
            return clock[0]

        engine.evaluate_device = timed_evaluate
        with patch.object(resident, "perf_counter", now):
            _, _, _, summary = resident.fit_resident(engine, self._initial(), max_iterations=10, max_wall_s=8.0)
        self.assertEqual(engine.evaluate_device_calls, 2)
        self.assertEqual(summary["wall_s"], 6.0)
        self.assertEqual(summary["termination"], "wall_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
