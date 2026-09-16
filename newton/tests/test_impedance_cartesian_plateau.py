# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for loss-plateau early stopping in the device-resident Cartesian optimizer.

The tests use synthetic quadratic objectives to verify that:
1. Flat objectives terminate after exactly plateau_patience completed iterations.
2. The GPU kernel resets the stale counter on significant relative improvement and
   treats cumulative sub-threshold improvements as stale.
3. Invalid plateau parameters (boolean, zero, negative, non-finite) are rejected.
4. When plateau stopping is disabled, iteration budget is exhausted as before.
5. Failed initial baseline rollouts do not trigger plateau stopping falsely.
6. The residency transfer whitelist allows only the single 4-byte stop flag per iteration.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.test_impedance_cartesian_gpu_resident import (
    _initial_coefficients,
    _QuadraticEngine,
)
from newton.tests.unittest_utils import get_test_devices
from projects.impedance_instron.cartesian.gpu import resident
from projects.impedance_instron.cartesian.trajectory import Spline


class TestImpedanceCartesianPlateau(unittest.TestCase):
    """Exercise loss-plateau early stopping on CPU and CUDA devices."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp and discover test devices."""
        wp.init()
        cls.devices = get_test_devices()

    def _initial(self, control_count: int = 12) -> Spline:
        """Build a valid synthetic starting spline."""
        return Spline(0.36, _initial_coefficients(control_count))

    def _engine(self, device, control_count: int = 12, **kwargs) -> _QuadraticEngine:
        """Build a synthetic quadratic engine matching resident world layout."""
        worlds = 128
        return _QuadraticEngine(worlds, device, control_count=control_count, **kwargs)

    def _flat_engine(self, device, control_count: int = 12) -> _QuadraticEngine:
        """Build an engine with target matching the initial spline, creating a flat optimal trajectory."""
        engine = self._engine(device, control_count=control_count, worsen_later=True)
        initial = self._initial(control_count)
        target = initial.coefficients.reshape(1, engine.parameter_count)
        engine.target = wp.array(np.repeat(target, engine.world_count, axis=0), dtype=wp.float64, device=device)
        return engine

    def test_flat_objective_stops_after_exactly_patience_iterations(self):
        """Stop search after exactly plateau_patience iterations when no loss improvement occurs."""
        for device in self.devices:
            with self.subTest(device=str(device)):
                # Flat engine has initial loss 0.0; no candidate can improve.
                engine = self._flat_engine(device)
                patience = 3
                max_iterations = 10
                _, _, _, summary = resident.fit_resident(
                    engine,
                    self._initial(),
                    max_iterations=max_iterations,
                    max_wall_s=60.0,
                    plateau_patience=patience,
                    plateau_rtol=1e-4,
                )
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["termination"], "loss_plateau")
                self.assertEqual(summary["status"], "loss_plateau")
                self.assertFalse(summary["converged"])
                self.assertEqual(summary["iterations_completed"], patience)
                self.assertEqual(summary["counts"]["batches"], 2 * patience)
                self.assertEqual(summary["optimizer"]["plateau_patience"], patience)
                self.assertEqual(summary["optimizer"]["plateau_rtol"], 1e-4)
                self.assertIn("stop after plateau_patience", summary["optimizer"]["plateau_criterion"])

    def test_disabled_plateau_exhausts_iteration_budget(self):
        """Run to iteration budget exhaustion when plateau_patience is None."""
        for device in self.devices:
            with self.subTest(device=str(device)):
                engine = self._flat_engine(device)
                max_iterations = 4
                _, _, _, summary = resident.fit_resident(
                    engine,
                    self._initial(),
                    max_iterations=max_iterations,
                    max_wall_s=60.0,
                    plateau_patience=None,
                )
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["termination"], "iteration_budget_exhausted")
                self.assertEqual(summary["status"], "budget_exhausted")
                self.assertEqual(summary["iterations_completed"], max_iterations)
                self.assertEqual(summary["counts"]["batches"], 2 * max_iterations)
                self.assertIsNone(summary["optimizer"]["plateau_patience"])
                self.assertEqual(summary["optimizer"]["plateau_criterion"], "disabled")

    def test_improving_resets_stale_and_cumulative_tiny_improvements_stall(self):
        """Verify unit kernel resets stale count on significant drop and accumulates tiny improvements."""
        for device in self.devices:
            with self.subTest(device=str(device)):
                active = wp.array([1], dtype=wp.int32, device=device)
                best_val = wp.array([100.0], dtype=wp.float64, device=device)
                anchor = wp.array([100.0], dtype=wp.float64, device=device)
                stale = wp.array([0], dtype=wp.int32, device=device)
                stop_flag = wp.array([0], dtype=wp.int32, device=device)
                patience = 3
                rtol = 0.01  # threshold = 0.01 * 100.0 = 1.0

                # Case 1: First iteration with tiny drop (< rtol * anchor).
                # Drop from 100.0 to 99.5 (gain 0.5 < 1.0).
                # Stale count should increment to 1, anchor stays 100.0, stop_flag stays 0.
                best_val = wp.array([99.5], dtype=wp.float64, device=device)
                wp.launch(
                    resident._update_plateau_kernel,
                    dim=1,
                    inputs=[active, best_val, anchor, stale, stop_flag, patience, wp.float64(rtol)],
                    device=device,
                )
                self.assertEqual(int(stale.numpy()[0]), 1)
                self.assertEqual(int(stop_flag.numpy()[0]), 0)
                self.assertAlmostEqual(float(anchor.numpy()[0]), 100.0)

                # Case 2: Another tiny drop from 99.5 to 99.1 (gain 0.4 from last, total gain 0.9 < 1.0 from anchor).
                # Cumulative improvement is still < 1.0 from anchor 100.0.
                # Stale count increments to 2, anchor stays 100.0, stop_flag stays 0.
                best_val = wp.array([99.1], dtype=wp.float64, device=device)
                wp.launch(
                    resident._update_plateau_kernel,
                    dim=1,
                    inputs=[active, best_val, anchor, stale, stop_flag, patience, wp.float64(rtol)],
                    device=device,
                )
                self.assertEqual(int(stale.numpy()[0]), 2)
                self.assertEqual(int(stop_flag.numpy()[0]), 0)
                self.assertAlmostEqual(float(anchor.numpy()[0]), 100.0)

                # Case 3: Significant drop from 99.1 to 98.0 (gain from anchor 100.0 is 2.0 >= 1.0).
                # Stale count should reset to 0, anchor updates to 98.0.
                best_val = wp.array([98.0], dtype=wp.float64, device=device)
                wp.launch(
                    resident._update_plateau_kernel,
                    dim=1,
                    inputs=[active, best_val, anchor, stale, stop_flag, patience, wp.float64(rtol)],
                    device=device,
                )
                self.assertEqual(int(stale.numpy()[0]), 0)
                self.assertEqual(int(stop_flag.numpy()[0]), 0)
                self.assertAlmostEqual(float(anchor.numpy()[0]), 98.0)

                # Case 4: Reaching patience after consecutive stale iterations triggers stop_flag.
                # Threshold for anchor 98.0 is 0.98. Best stays 98.0.
                for expected_stale in (1, 2, 3):
                    wp.launch(
                        resident._update_plateau_kernel,
                        dim=1,
                        inputs=[active, best_val, anchor, stale, stop_flag, patience, wp.float64(rtol)],
                        device=device,
                    )
                    self.assertEqual(int(stale.numpy()[0]), expected_stale)
                self.assertEqual(int(stop_flag.numpy()[0]), 1)

    def test_invalid_plateau_arguments_raise_value_error(self):
        """Reject invalid plateau_patience and plateau_rtol inputs with clear errors."""
        device = self.devices[0]
        engine = self._engine(device)
        initial = self._initial()

        # Invalid patience inputs: bool, zero, negative, float, non-int
        for bad_patience in (True, False, 0, -1, -5, 1.5, "3", [2]):
            with self.subTest(bad_patience=bad_patience):
                with self.assertRaises(ValueError):
                    resident.fit_resident(engine, initial, plateau_patience=bad_patience)

        # Invalid rtol inputs: bool, zero, negative, non-finite, string
        for bad_rtol in (True, False, 0.0, -1e-4, float("inf"), float("nan"), "1e-4"):
            with self.subTest(bad_rtol=bad_rtol):
                with self.assertRaises(ValueError):
                    resident.fit_resident(engine, initial, plateau_patience=2, plateau_rtol=bad_rtol)

    def test_failed_initial_baseline_does_not_plateau_falsely(self):
        """Preserve initial_rollout_incomplete status when baseline fails, never reporting loss_plateau."""
        for device in self.devices:
            with self.subTest(device=str(device)):
                failed_engine = self._engine(device, fail_first=True)
                _, _, _, summary = resident.fit_resident(
                    failed_engine,
                    self._initial(),
                    max_iterations=5,
                    max_wall_s=30.0,
                    plateau_patience=2,
                )
                self.assertFalse(summary["complete"])
                self.assertEqual(summary["termination"], "initial_rollout_incomplete")
                self.assertEqual(summary["status"], "incomplete")
                self.assertIsNone(summary["loss"])
                self.assertIsNone(summary["best"])

    def test_residency_whitelist_permits_only_stop_flag_transfer(self):
        """Verify that enabled plateau mode transfers only the single 4-byte stop-flag integer."""
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

        copied_transfers = []
        original_copy = wp.copy

        def checked_copy(dest, src, *args, **kwargs):
            if in_search() and dest.device.is_cpu != src.device.is_cpu:
                # Whitelist: only 1D int32 array of size 1 (the 4-byte stop flag)
                is_stop_flag = dest.device.is_cpu and dest.dtype == wp.int32 and dest.size == 1 and dest.ndim == 1
                if not is_stop_flag:
                    raise AssertionError(
                        f"Disallowed host transfer inside search loop: "
                        f"src={src.device}({src.dtype}, shape={src.shape}) -> "
                        f"dest={dest.device}({dest.dtype}, shape={dest.shape})"
                    )
                copied_transfers.append((src.shape, src.dtype))
            return original_copy(dest, src, *args, **kwargs)

        def guarded_numpy(arr, *args, **kwargs):
            if in_search():
                # Only the CPU stop-flag buffer is permitted to call .numpy() to read the integer
                if not (arr.device.is_cpu and arr.dtype == wp.int32 and arr.size == 1 and arr.ndim == 1):
                    raise AssertionError(
                        f"Disallowed .numpy() call inside search loop: "
                        f"device={arr.device}, dtype={arr.dtype}, shape={arr.shape}"
                    )
            return arr_numpy_orig(arr, *args, **kwargs)

        arr_numpy_orig = wp.array.numpy

        with ExitStack() as stack:
            stack.enter_context(patch.object(wp, "copy", checked_copy))
            stack.enter_context(patch.object(wp.array, "numpy", guarded_numpy))
            for device in self.devices:
                copied_transfers.clear()
                engine = self._flat_engine(device)
                patience = 2
                resident.fit_resident(
                    engine,
                    self._initial(),
                    max_iterations=10,
                    max_wall_s=30.0,
                    plateau_patience=patience,
                )
                # Exactly `patience` copies of shape (1,) int32 occurred across device->host boundary
                if not device.is_cpu:
                    self.assertEqual(len(copied_transfers), patience)
                    for shape, dtype in copied_transfers:
                        self.assertEqual(shape, (1,))
                        self.assertEqual(dtype, wp.int32)
