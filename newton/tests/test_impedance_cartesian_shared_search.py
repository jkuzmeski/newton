# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that one resident controller receives every world's search result."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton.tests.test_impedance_cartesian_gpu_resident import _initial_coefficients, _QuadraticEngine
from newton.tests.unittest_utils import get_test_devices
from projects.impedance_instron.cartesian.gpu import resident
from projects.impedance_instron.cartesian.trajectory import Spline


class _ObservedEngine(_QuadraticEngine):
    """Observe proposals and place the quadratic minimum in the last world."""

    def __init__(self, device, control_count, *, reject_tail=False):
        super().__init__(128, device, control_count=control_count)
        # Keep every local proposal bounded so slot coverage is unambiguous.
        self.profile["equilibrium_rate_limit"] = [1.0e6] * 4
        self.profile["equilibrium_acceleration_limit"] = [1.0e6] * 4
        self.batches = []
        self.reject_tail = reject_tail

    def evaluate_device(self):
        """Record test-only host observations before evaluating a common objective."""
        coefficients = self.coefficients.numpy().copy()
        self.batches.append(coefficients)
        if len(self.batches) == 1:
            target = np.repeat(coefficients[-1:].reshape(1, self.parameter_count), self.world_count, axis=0)
            self.target.assign(target)
        super().evaluate_device()
        if self.reject_tail:
            failure = np.zeros(self.world_count, dtype=np.int32)
            failure[-1] = 9
            self.failure.assign(failure)


class TestImpedanceCartesianSharedSearch(unittest.TestCase):
    """Exercise the shared layout with 128 slots on CPU and CUDA."""

    @classmethod
    def setUpClass(cls):
        """Initialize the available CPU and CUDA devices."""
        wp.init()
        cls.devices = [device for device in get_test_devices() if device.is_cpu or device.is_cuda]

    def test_all_128_worlds_advance_one_controller(self):
        """Use a last-slot winner as the common incumbent in later batches."""
        for device in self.devices:
            for controls in (12,):
                with self.subTest(device=device, controls=controls):
                    engine = _ObservedEngine(device, controls)
                    initial = Spline(engine.duration, _initial_coefficients(controls))
                    result, _, _, summary = resident.fit_resident(
                        engine, initial, max_iterations=2, max_wall_s=60.0, seed=31
                    )
                    optimizer = summary["optimizer"]
                    self.assertEqual(optimizer["search_mode"], "shared_controller")
                    self.assertEqual(optimizer["independent_controllers"], 1)
                    self.assertEqual(optimizer["islands"], 1)
                    self.assertEqual(optimizer["worlds_per_island"], 128)
                    self.assertEqual(len(optimizer["final_step_fractions"]), 1)
                    self.assertEqual(summary["iterations_completed"], 2)
                    self.assertEqual(summary["island_iterations_completed"], 2)
                    self.assertEqual(summary["counts"]["physics_worlds"], 4 * 128)
                    self.assertEqual(summary["counts"]["completed_real_candidates"], 4 * 128)
                    poll = engine.batches[0]
                    np.testing.assert_array_equal(poll[0], initial.coefficients)
                    self.assertEqual(np.unique(poll.reshape(128, -1), axis=0).shape[0], 128)
                    # These worlds were separate island baselines in the old layout.
                    for slot in (48, 64, 127):
                        self.assertFalse(np.array_equal(poll[slot], initial.coefficients))
                    winner = poll[-1]
                    np.testing.assert_array_equal(engine.batches[1][0], winner)
                    np.testing.assert_array_equal(engine.batches[2][0], winner)
                    np.testing.assert_array_equal(result.coefficients, winner)
                    self.assertEqual(summary["loss"], 0.0)
                    # The next poll is centered on the shared winner, not an old island start.
                    delta = engine.settings.parameter_scale[0] * optimizer["initial_step_fraction"]
                    expected = winner.copy()
                    expected[0, 0] += delta
                    np.testing.assert_array_equal(engine.batches[2][1], expected)

    def test_failed_last_world_cannot_update_shared_controller(self):
        """Exclude a failed last-slot zero-loss candidate from global selection."""
        for device in self.devices:
            with self.subTest(device=device):
                engine = _ObservedEngine(device, 12, reject_tail=True)
                initial = Spline(engine.duration, _initial_coefficients(12))
                result, _, _, summary = resident.fit_resident(
                    engine, initial, max_iterations=1, max_wall_s=60.0, seed=31
                )
                self.assertTrue(summary["complete"])
                self.assertGreater(summary["loss"], 0.0)
                self.assertFalse(np.array_equal(result.coefficients, engine.batches[0][-1]))
                self.assertEqual(summary["counts"]["completed_real_candidates"], 2 * 127)

    def test_fixed_layout_rejects_wrong_world_count(self):
        """Reject engines that do not provide the fixed 128-world batch."""
        device = self.devices[0]
        engine = _QuadraticEngine(64, device, control_count=12)
        initial = Spline(engine.duration, _initial_coefficients(12))
        with self.assertRaisesRegex(ValueError, "exactly 128 worlds"):
            resident.fit_resident(engine, initial)

    def test_fixed_layout_rejects_wrong_control_count(self):
        """Reject controllers that do not contain exactly twelve points."""
        device = self.devices[0]
        engine = _QuadraticEngine(128, device, control_count=6)
        initial = Spline(engine.duration, _initial_coefficients(6))
        with self.assertRaisesRegex(ValueError, "exactly 12 control points"):
            resident.fit_resident(engine, initial)


if __name__ == "__main__":
    unittest.main()
