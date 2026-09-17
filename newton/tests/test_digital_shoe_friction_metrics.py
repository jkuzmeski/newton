# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for digital shoe braking, propulsive, and horizontal friction metrics."""

import os
import unittest
from pathlib import Path

import numpy as np

from projects.digital_shoe.friction_metrics import (
    _find_contiguous_intervals,
    _segment_piecewise_linear_impulses,
    _validate_intervals,
    compute_braking_propulsive_impulses,
    compute_force_peaks,
    score_friction_trace,
)


class TestDigitalShoeFrictionMetrics(unittest.TestCase):
    """Verify ground reaction force metrics, zero-crossing splits, masks, and clock handling."""

    def test_stance_mask_is_independent_of_signed_observation(self):
        """Use declared contact events without clipping signed force observations."""
        time = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        force = np.column_stack(([-10, -20, -30, 20, 10], [-10, -5, 10, 100, 0]))
        reference = {"grf_time_s": time, "grf_target_n": force}
        trace = {"time_s": time, "grf_n": force.copy()}
        before = force.copy()
        scored = score_friction_trace(reference, trace, 1, stance_normal_n=np.array([0, 100, 100, 100, 0]))
        self.assertTrue(scored["complete"])
        self.assertEqual(scored["metadata"]["stance_source"], "explicit_normal_signal")
        self.assertGreater(scored["reference_metrics"]["braking_impulse_ns"], 0)
        np.testing.assert_array_equal(force, before)
        with self.assertRaisesRegex(ValueError, "stance_normal_n"):
            score_friction_trace(reference, trace, 1, stance_normal_n=np.ones(3))

    def test_piecewise_linear_zero_crossing_split(self):
        """Verify exact analytic impulse calculation on piecewise-linear zero crossing."""
        # Linear transition from -2 to +2 between t=0 and t=2.
        # Zero crossing is at t=1.
        # Negative triangle: base [0, 1] (dt=1), height -2 -> negative area = 0.5 * 1 * 2 = 1.0 (braking)
        # Positive triangle: base [1, 2] (dt=1), height +2 -> positive area = 0.5 * 1 * 2 = 1.0 (propulsion)
        # Naive trapezoid clipping at grid points would give 0.5 * (0 + 2) * 2 = 2.0 (100% error!)
        t = np.array([0.0, 2.0])
        f = np.array([-2.0, 2.0])
        pos, neg, net = _segment_piecewise_linear_impulses(t, f)
        self.assertAlmostEqual(pos, 1.0, places=9)
        self.assertAlmostEqual(neg, 1.0, places=9)
        self.assertAlmostEqual(net, 0.0, places=9)

    def test_synthetic_analytic_triangles(self):
        """Verify analytic triangle profile with known braking and propulsive phases."""
        # Construct synthetic gait-like Fx:
        # t in [0, 1]:
        # 0.0 to 0.5: braking triangle reaching -100 N at t=0.25 (area = 0.5 * 0.5 * 100 = 25 N*s)
        # 0.5 to 1.0: propulsive triangle reaching +200 N at t=0.75 (area = 0.5 * 0.5 * 200 = 50 N*s)
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        f = np.array([0.0, -100.0, 0.0, 200.0, 0.0])

        braking_imp, prop_imp, net_imp = compute_braking_propulsive_impulses(t, f)
        self.assertAlmostEqual(braking_imp, 25.0, places=9)
        self.assertAlmostEqual(prop_imp, 50.0, places=9)
        self.assertAlmostEqual(net_imp, 25.0, places=9)

        peaks = compute_force_peaks(t, f)
        self.assertAlmostEqual(peaks["braking_peak_magnitude_n"], 100.0, places=9)
        self.assertAlmostEqual(peaks["braking_peak_signed_n"], -100.0, places=9)
        self.assertAlmostEqual(peaks["braking_peak_time_s"], 0.25, places=9)
        self.assertAlmostEqual(peaks["propulsive_peak_magnitude_n"], 200.0, places=9)
        self.assertAlmostEqual(peaks["propulsive_peak_signed_n"], 200.0, places=9)
        self.assertAlmostEqual(peaks["propulsive_peak_time_s"], 0.75, places=9)

    def test_forward_sign_reversal(self):
        """Verify that reversing forward_sign inverts braking and propulsive roles."""
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        f_raw = np.array([0.0, -100.0, 0.0, 200.0, 0.0])

        # forward_sign = +1: -100 is braking, 200 is propulsion
        b1, p1, n1 = compute_braking_propulsive_impulses(t, 1.0 * f_raw)
        # forward_sign = -1: -(-100)=+100 is propulsion, -(200)=-200 is braking
        b2, p2, n2 = compute_braking_propulsive_impulses(t, -1.0 * f_raw)

        self.assertAlmostEqual(b1, p2, places=9)
        self.assertAlmostEqual(p1, b2, places=9)
        self.assertAlmostEqual(n1, -n2, places=9)

    def test_absent_phase_and_missing_phase_error(self):
        """Verify that missing target phase is None, but missing predicted phase evaluates to 1.0."""
        # Pure positive force trace (no braking phase)
        t = np.array([0.0, 0.5, 1.0])
        f_pos = np.array([10.0, 50.0, 20.0])

        peaks = compute_force_peaks(t, f_pos)
        self.assertIsNone(peaks["braking_peak_magnitude_n"])
        self.assertIsNone(peaks["braking_peak_signed_n"])
        self.assertIsNone(peaks["braking_peak_time_s"])
        self.assertAlmostEqual(peaks["propulsive_peak_magnitude_n"], 50.0, places=9)

        # Synthetic score comparison where reference has both phases but trace has only positive
        ref_t = np.linspace(0.0, 1.0, 101)
        ref_fx = np.sin(2 * np.pi * ref_t) * 100.0  # has braking and propulsive phases
        ref_fz = np.full_like(ref_t, 100.0)

        tr_t = np.linspace(0.0, 1.0, 101)
        tr_fx = np.abs(np.sin(2 * np.pi * tr_t) * 100.0)  # purely positive: braking absent!
        tr_fz = np.full_like(tr_t, 100.0)

        ref_dict = {
            "grf_time_s": ref_t,
            "grf_target_n": np.column_stack((ref_fx, ref_fz)),
        }
        tr_dict = {
            "time_s": tr_t,
            "grf_n": np.column_stack((tr_fx, tr_fz)),
        }

        res = score_friction_trace(ref_dict, tr_dict, forward_sign=1, normal_threshold_n=50.0)
        self.assertTrue(res["complete"])
        self.assertTrue(res["rollout_complete"])
        self.assertIsNone(res["trace_metrics"]["braking_peak_magnitude_n"])

        # Flags: target has braking, so missing_predicted_braking_phase must be True
        self.assertTrue(res["comparison_metrics"]["missing_predicted_braking_phase"])
        self.assertFalse(res["comparison_metrics"]["missing_predicted_propulsive_phase"])

        # Relative error when target exists but prediction is absent must be 1.0 (100% missing phase error)
        self.assertEqual(res["comparison_metrics"]["braking_peak_relative_error"], 1.0)
        # Timing difference is None since no predicted peak time exists
        self.assertIsNone(res["comparison_metrics"]["braking_peak_timing_diff_s"])

    def test_invalid_and_non_monotonic_clocks(self):
        """Verify that non-monotonic, non-finite, or backwards clocks raise errors or mark incomplete."""
        # Decreasing time
        t_bad = np.array([0.0, 0.5, 0.4])
        f = np.array([0.0, 10.0, 20.0])
        with self.assertRaises(ValueError):
            compute_braking_propulsive_impulses(t_bad, f)

        # Nonfinite time
        t_nan = np.array([0.0, np.nan, 1.0])
        with self.assertRaises(ValueError):
            compute_force_peaks(t_nan, f)

        # In score_friction_trace, non-monotonic trace should mark incomplete
        ref_dict = {
            "grf_time_s": np.array([0.0, 0.5, 1.0]),
            "grf_target_n": np.zeros((3, 2)),
        }
        tr_bad_dict = {
            "time_s": np.array([0.0, 0.5, 0.4]),
            "grf_n": np.zeros((3, 2)),
        }
        res = score_friction_trace(ref_dict, tr_bad_dict, forward_sign=1)
        self.assertFalse(res["complete"])
        self.assertIn("Invalid trace time_s", res["failure_reason"])

    def test_partial_trace_coverage_marked_incomplete(self):
        """Verify that a truncated or incomplete rollout is marked incomplete and not falsely scored."""
        # Reference extends from 0.0 to 1.0 s
        ref_t = np.linspace(0.0, 1.0, 101)
        ref_dict = {
            "grf_time_s": ref_t,
            "grf_target_n": np.column_stack((np.zeros(101), np.full(101, 100.0))),
        }
        # Trace truncated early at 0.5 s
        tr_t = np.linspace(0.0, 0.5, 51)
        tr_dict = {
            "time_s": tr_t,
            "grf_n": np.zeros((51, 2)),
        }
        res = score_friction_trace(ref_dict, tr_dict, forward_sign=1)
        self.assertFalse(res["complete"])
        self.assertIn("Incomplete rollout", res["failure_reason"])

    def test_invalid_and_overlapping_intervals_rejected(self):
        """Verify that invalid, overlapping, or out-of-order intervals raise ValueError."""
        # Out of bounds
        with self.assertRaises(ValueError):
            _validate_intervals([(0, 10)], n_samples=5)

        # Start > end
        with self.assertRaises(ValueError):
            _validate_intervals([(3, 2)], n_samples=10)

        # Overlapping
        with self.assertRaises(ValueError):
            _validate_intervals([(0, 4), (3, 7)], n_samples=10)

        # Out of order
        with self.assertRaises(ValueError):
            _validate_intervals([(5, 7), (1, 3)], n_samples=10)

        # Overlapping intervals in compute_braking_propulsive_impulses
        t = np.arange(10, dtype=float)
        f = np.ones(10)
        with self.assertRaises(ValueError):
            compute_braking_propulsive_impulses(t, f, intervals=[(0, 4), (3, 7)])

    def test_normal_mask_preserves_contiguous_intervals(self):
        """Verify normal mask groups intervals separately without joining noncontiguous flight phases."""
        mask = np.array([False, True, True, False, False, True, True, True, False])
        intervals = _find_contiguous_intervals(mask)
        self.assertEqual(intervals, [(1, 2), (5, 7)])

        # Construct signal with two stance phases separated by flight (Fz=0)
        # Stance 1: t in [1, 2], Fx = -10 (braking = 10 N*s)
        # Flight: t in (2, 5), Fx = 100 (should be excluded!)
        # Stance 2: t in [5, 7], Fx = 20 (propulsion = 40 N*s)
        t = np.arange(9, dtype=float)
        fx = np.array([0.0, -10.0, -10.0, 100.0, 100.0, 20.0, 20.0, 20.0, 0.0])

        b_imp, p_imp, n_imp = compute_braking_propulsive_impulses(t, fx, intervals)
        self.assertAlmostEqual(b_imp, 10.0, places=9)
        self.assertAlmostEqual(p_imp, 40.0, places=9)
        self.assertAlmostEqual(n_imp, 30.0, places=9)

    def test_full_vs_stance_horizontal_rmse_separated(self):
        """Verify that full horizontal RMSE and stance horizontal RMSE are distinctly evaluated."""
        # 100 samples from 0 to 1s.
        # Stance defined where Fz >= 50 N (say middle half: 0.25 to 0.75s).
        t = np.linspace(0.0, 1.0, 101)
        # Reference: Fx = 0 everywhere, Fz = 100 N in [0.25, 0.75], 0 elsewhere
        ref_fx = np.zeros(101)
        ref_fz = np.where((t >= 0.25) & (t <= 0.75), 100.0, 0.0)

        # Trace: Fx error is 10 N in stance, but 100 N in flight!
        tr_fx = np.where((t >= 0.25) & (t <= 0.75), 10.0, 100.0)
        tr_fz = ref_fz.copy()

        ref_dict = {
            "grf_time_s": t,
            "grf_target_n": np.column_stack((ref_fx, ref_fz)),
        }
        tr_dict = {
            "time_s": t,
            "grf_n": np.column_stack((tr_fx, tr_fz)),
        }

        res = score_friction_trace(ref_dict, tr_dict, forward_sign=1, normal_threshold_n=50.0)
        self.assertTrue(res["complete"])
        comp = res["comparison_metrics"]

        # In stance, error is 10 N
        self.assertAlmostEqual(comp["stance_horizontal_force_rmse_n"], 10.0, places=5)
        # Full RMSE includes the 100 N flight errors, so full RMSE must be much higher
        self.assertGreater(comp["full_horizontal_force_rmse_n"], comp["stance_horizontal_force_rmse_n"])
        self.assertAlmostEqual(comp["stance_horizontal_force_max_abs_error_n"], 10.0, places=5)
        self.assertAlmostEqual(comp["full_horizontal_force_max_abs_error_n"], 100.0, places=5)

    def test_short_missing_tail_requires_completion_evidence(self):
        """Reject even one missing native sample without explicit completed-step evidence."""
        time = np.linspace(0.0, 0.01, 21)
        reference = {"grf_time_s": time, "grf_target_n": np.column_stack((np.ones(21), np.full(21, 100.0)))}
        trace = {"time_s": time[:-1], "grf_n": reference["grf_target_n"][:-1]}
        self.assertFalse(score_friction_trace(reference, trace, 1)["complete"])
        summary = {"complete": True, "integrated_steps": 20, "actual_dt_s": 0.0005}
        result = score_friction_trace(reference, trace, 1, summary=summary)
        self.assertTrue(result["complete"])
        self.assertFalse(result["support"]["force_support_complete"])
        self.assertEqual(result["support"]["uncovered_native_sample_count"], 1)
        summary["complete"] = False
        self.assertFalse(score_friction_trace(reference, trace, 1, summary=summary)["complete"])
        summary = {"failure": {"reason": "stopped"}}
        self.assertFalse(score_friction_trace(reference, trace, 1, summary=summary)["complete"])

    def test_summary_cannot_certify_a_truncated_trace(self):
        """Reject stale completion metadata and nonfinite stance thresholds."""
        time = np.linspace(0.0, 1.0, 101)
        force = np.column_stack((np.ones(101), np.full(101, 100.0)))
        reference = {"grf_time_s": time, "grf_target_n": force}
        trace = {"time_s": time[:-2], "grf_n": force[:-2]}
        summary = {"complete": True, "integrated_steps": 99, "actual_dt_s": 0.01}
        self.assertFalse(score_friction_trace(reference, trace, 1, summary=summary)["complete"])
        with self.assertRaises(ValueError):
            score_friction_trace(reference, {"time_s": time, "grf_n": force}, 1, normal_threshold_n=float("nan"))

    def test_baseline12_trace_regression_match(self):
        """Verify scoring against the sealed baseline12 trace and Cartesian reference if available."""
        # Allow opt-in path override via environment variable or default relative fixture location
        baseline_env = os.environ.get("NEWTON_BASELINE12_DIR")
        if baseline_env:
            baseline_dir = Path(baseline_env)
        else:
            default_candidate = Path("outputs/impedance_instron/baseline12")
            if default_candidate.exists():
                baseline_dir = default_candidate
            else:
                self.skipTest("baseline12 directory not found; set NEWTON_BASELINE12_DIR to enable")
                return

        ref_path = baseline_dir / "reference.npz"
        trace_path = baseline_dir / "trace.npz"
        sum_path = baseline_dir / "summary.json"

        if not (ref_path.exists() and trace_path.exists() and sum_path.exists()):
            self.skipTest("Baseline files not available in specified directory")

        res = score_friction_trace(
            reference=ref_path,
            trace=trace_path,
            forward_sign=1,
            normal_threshold_n=50.0,
            summary=sum_path,
        )
        self.assertTrue(res["complete"])
        self.assertTrue(res["rollout_complete"])
        self.assertIsNone(res["failure_reason"])

        # Check support accounting: 720 covered, 1 excluded endpoint (no extrapolation)
        self.assertEqual(res["support"]["covered_native_sample_count"], 720)
        self.assertEqual(res["support"]["uncovered_native_sample_count"], 1)

        # Verify reference braking and propulsive impulses are in expected physical gait ranges
        # For subject ~82 kg at 3.0 m/s: braking ~14.7 N*s, propulsion ~16.7 N*s
        self.assertAlmostEqual(res["reference_metrics"]["braking_impulse_ns"], 14.69441577, places=5)
        self.assertAlmostEqual(res["reference_metrics"]["propulsive_impulse_ns"], 16.69166024, places=5)

        # Verify full vs stance horizontal force RMSE separation
        self.assertAlmostEqual(res["comparison_metrics"]["full_horizontal_force_rmse_n"], 97.7635, delta=0.1)
        self.assertAlmostEqual(res["comparison_metrics"]["stance_horizontal_force_rmse_n"], 107.6259, delta=0.1)


if __name__ == "__main__":
    unittest.main()
