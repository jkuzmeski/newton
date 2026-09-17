# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for digital shoe friction force observation and comparison."""

import json
import os
import unittest
from pathlib import Path

import numpy as np

from projects.digital_shoe.friction_observation import (
    approximate_hann_filter_highrate,
    observe_friction_comparison,
    resample_to_reference_clock,
)


class TestDigitalShoeFrictionObservation(unittest.TestCase):
    """Verify friction observation filtering, resampling, signed Fz, and chatter guards."""

    def setUp(self):
        """Prepare synthetic clocks, forces, and declared filter metadata."""
        # 2000 Hz reference clock over 0.36 s
        self.ref_rate = 2000.0
        self.ref_n = 721
        self.ref_time = np.linspace(0.0, 0.36, self.ref_n)
        # 16000 Hz simulation clock over 0.36 s (preintegration: ends at 0.36 - dt)
        self.sim_rate = 16000.0
        self.sim_n = 5760
        self.sim_time = np.linspace(0.0, 0.3599375, self.sim_n)
        self.filter_spec = {
            "family": "Butterworth",
            "order": 4,
            "cutoff_hz": 20.0,
            "passes": "forward/backward",
            "hann_duration_s": 0.010,
        }

    def test_missing_hann_policy_is_rejected(self):
        """Reject unknown preprocessing instead of silently assuming a Hann kernel."""
        spec = dict(self.filter_spec)
        spec.pop("hann_duration_s")
        reference = {"grf_time_s": self.ref_time, "unfiltered_grf_target_n": np.zeros((self.ref_n, 2))}
        with self.assertRaisesRegex(ValueError, "Hann preprocessing is undeclared"):
            observe_friction_comparison(reference, self.sim_time, np.zeros((self.sim_n, 2)), filter_spec=spec)

    def test_constant_dc_preservation(self):
        """Verify that constant DC signals are preserved in the interior window away from padding."""
        c_fx = 150.0
        c_fz = 600.0
        target = np.column_stack((np.full(self.ref_n, c_fx), np.full(self.ref_n, c_fz)))
        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": target,
        }
        pred_forces = np.column_stack((np.full(self.sim_n, c_fx), np.full(self.sim_n, c_fz)))

        obs = observe_friction_comparison(ref, self.sim_time, pred_forces, filter_spec=self.filter_spec)
        self.assertTrue(obs["complete"])

        # Interior samples (away from filter boundaries) must preserve DC levels
        padlen = obs["metadata"]["butterworth_20hz"]["padlen_samples"]
        interior = slice(padlen + 20, len(obs["clock"]) - padlen - 20)

        np.testing.assert_allclose(obs["observed_target"][interior, 0], c_fx, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(obs["observed_target"][interior, 1], c_fz, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(obs["observed_prediction"][interior, 0], c_fx, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(obs["observed_prediction"][interior, 1], c_fz, rtol=1e-2, atol=1e-2)

    def test_equal_true_source_filtered_matches_prediction(self):
        """Verify identical source sampled at native vs high-rate matches through observation pipeline."""
        # Known synthetic smooth trajectory with multiple frequency components below cutoff
        t_n = self.ref_time
        t_s = self.sim_time
        freq1, freq2 = 3.0, 7.0
        raw_fx_n = 50.0 * np.sin(2.0 * np.pi * freq1 * t_n) + 20.0 * np.cos(2.0 * np.pi * freq2 * t_n)
        raw_fz_n = 400.0 * np.sin(np.pi * 2.0 * t_n) + 100.0
        raw_target_n = np.column_stack((raw_fx_n, raw_fz_n))

        # Primary source preparation: unfiltered_grf_target_n is pre-filtered with 21-sample Hann
        w21 = np.hanning(21)
        w21 /= w21.sum()
        from scipy.ndimage import convolve1d

        target_pre20hz = convolve1d(raw_target_n, w21, axis=0, mode="constant", cval=0.0)

        # Simulation produces high-rate true source
        fx_s = 50.0 * np.sin(2.0 * np.pi * freq1 * t_s) + 20.0 * np.cos(2.0 * np.pi * freq2 * t_s)
        fz_s = 400.0 * np.sin(np.pi * 2.0 * t_s) + 100.0
        pred_forces = np.column_stack((fx_s, fz_s))

        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": target_pre20hz,
        }

        obs = observe_friction_comparison(ref, t_s, pred_forces, filter_spec=self.filter_spec)
        self.assertTrue(obs["complete"])

        # Interior comparison away from 300-sample Butterworth boundary padding
        interior = slice(305, len(obs["clock"]) - 305)
        target_int = obs["observed_target"][interior]
        pred_int = obs["observed_prediction"][interior]

        max_abs_diff = np.max(np.abs(target_int - pred_int))
        self.assertLess(max_abs_diff, 0.1)

    def test_forward_sign_reversal_and_nonmutation(self):
        """Verify forward_sign reversal flips forward force sign and never mutates inputs."""
        c_fx = 100.0
        c_fz = 500.0
        target = np.column_stack((np.full(self.ref_n, c_fx), np.full(self.ref_n, c_fz)))
        target_copy = target.copy()
        pred_forces = np.column_stack((np.full(self.sim_n, c_fx), np.full(self.sim_n, c_fz)))
        pred_copy = pred_forces.copy()

        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": target,
        }

        obs_pos = observe_friction_comparison(
            ref, self.sim_time, pred_forces, forward_sign=1, filter_spec=self.filter_spec
        )
        obs_neg = observe_friction_comparison(
            ref, self.sim_time, pred_forces, forward_sign=-1, filter_spec=self.filter_spec
        )

        # Inputs must remain untouched
        np.testing.assert_array_equal(target, target_copy)
        np.testing.assert_array_equal(pred_forces, pred_copy)

        # Fx must be opposite sign
        interior = slice(50, -50)
        np.testing.assert_allclose(obs_pos["observed_target"][interior, 0], -obs_neg["observed_target"][interior, 0])
        np.testing.assert_allclose(
            obs_pos["observed_prediction"][interior, 0], -obs_neg["observed_prediction"][interior, 0]
        )
        np.testing.assert_allclose(obs_pos["pre20hz_target"][interior, 0], -obs_neg["pre20hz_target"][interior, 0])

        # Fz must remain untouched and identical
        np.testing.assert_allclose(obs_pos["observed_target"][:, 1], obs_neg["observed_target"][:, 1])
        np.testing.assert_allclose(obs_pos["observed_prediction"][:, 1], obs_neg["observed_prediction"][:, 1])
        np.testing.assert_allclose(obs_pos["pre20hz_normal"], obs_neg["pre20hz_normal"])

    def test_signed_negatives_retained(self):
        """Verify that negative vertical force Fz is preserved without component-wise clamping."""
        target = np.column_stack((np.zeros(self.ref_n), -30.0 * np.sin(np.linspace(0, np.pi, self.ref_n))))
        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": target,
        }
        pred_forces = np.column_stack((np.zeros(self.sim_n), -30.0 * np.sin(np.linspace(0, np.pi, self.sim_n))))

        obs = observe_friction_comparison(ref, self.sim_time, pred_forces, filter_spec=self.filter_spec)
        self.assertTrue(obs["complete"])
        min_target_fz = np.min(obs["observed_target"][:, 1])
        min_pred_fz = np.min(obs["observed_prediction"][:, 1])
        self.assertLess(min_target_fz, -10.0)
        self.assertLess(min_pred_fz, -10.0)
        self.assertTrue(obs["metadata"]["diagnostics"]["target_negative_fz_retained"])
        self.assertFalse(obs["metadata"]["butterworth_20hz"]["clamp_vertical_zero"])

    def test_partial_trace_does_not_pass(self):
        """Verify that partial/truncated trace is marked incomplete even if common segment exists."""
        short_sim_time = self.sim_time[self.sim_time <= 0.20]
        short_sim_forces = np.zeros((len(short_sim_time), 2))

        target = np.zeros((self.ref_n, 2))
        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": target,
        }

        obs = observe_friction_comparison(ref, short_sim_time, short_sim_forces, filter_spec=self.filter_spec)
        self.assertFalse(obs["complete"])
        self.assertIn("fails full preintegration support", obs["failure_reason"])

    def test_unsupported_endpoint_excluded_in_resample_helper(self):
        """Verify resample_to_reference_clock strictly excludes endpoints outside source without extrapolation."""
        src_t = np.array([0.0, 0.1, 0.2])
        src_f = np.ones((3, 2))
        ref_t = np.linspace(-0.1, 0.3, 5)

        _resampled, mask, meta = resample_to_reference_clock(src_t, src_f, ref_t)
        self.assertEqual(meta["excluded_reference_samples"], 2)
        self.assertEqual(meta["covered_reference_samples"], 3)
        self.assertFalse(meta["extrapolation_permitted"])
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_highrate_chatter_guard_not_hidden(self):
        """Verify that raw force chatter remains visible and serializable to JSON."""
        chatter_freq = 2000.0  # 2 kHz chatter
        chatter = 50.0 * np.sin(2.0 * np.pi * chatter_freq * self.sim_time)
        pred_forces = np.column_stack((chatter, np.full(self.sim_n, 500.0)))

        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": np.zeros((self.ref_n, 2)),
        }

        obs = observe_friction_comparison(ref, self.sim_time, pred_forces, filter_spec=self.filter_spec)
        self.assertTrue(obs["complete"])
        diag = obs["raw_prediction_chatter"]
        self.assertGreater(diag["max_abs_force_rate_n_s"], 500000.0)
        self.assertGreater(diag["components"]["fx_max_rate_n_s"], 500000.0)
        # Ensure chatter diagnostics can be serialized to JSON
        json_str = json.dumps(diag)
        self.assertIsInstance(json_str, str)
        # Raw prediction in output preserves the raw un-smoothed chatter
        np.testing.assert_array_equal(obs["raw_prediction"], pred_forces)

    def test_known_synthetic_force_smoothing(self):
        """Verify Hann pre-smoothing approximates physical window duration."""
        target_dur = 0.010  # 10 ms
        _smoothed, meta = approximate_hann_filter_highrate(
            self.sim_time, np.ones((self.sim_n, 2)), target_duration_s=target_dur
        )
        self.assertEqual(meta["method"], "symmetric Hann")
        self.assertEqual(meta["window_samples"], 161)
        self.assertAlmostEqual(meta["actual_duration_s"], target_dur, places=6)
        self.assertTrue(meta["kernel_normalized"])

    def test_batch_leading_dimension_support(self):
        """Verify support for arbitrary leading batch dimensions [B, T, 2] and [B1, B2, T, 2]."""
        batch_shape = (3, 4, self.sim_n, 2)
        batch_forces = np.random.default_rng(23).normal(size=batch_shape)

        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": np.zeros((self.ref_n, 2)),
        }

        obs = observe_friction_comparison(ref, self.sim_time, batch_forces, filter_spec=self.filter_spec)
        self.assertTrue(obs["complete"])
        expected_shape = (3, 4, len(obs["clock"]), 2)
        self.assertEqual(obs["observed_prediction"].shape, expected_shape)
        self.assertEqual(obs["observed_target"].shape, (len(obs["clock"]), 2))

    def test_no_fallback_to_grf_target_n(self):
        """Verify strict error if unfiltered_grf_target_n is absent, prohibiting fallback."""
        ref = {
            "grf_time_s": self.ref_time,
            "grf_target_n": np.zeros((self.ref_n, 2)),
        }
        with self.assertRaisesRegex(ValueError, "unfiltered_grf_target_n"):
            observe_friction_comparison(ref, self.sim_time, np.zeros((self.sim_n, 2)), filter_spec=self.filter_spec)

    def test_strict_filter_metadata_validation(self):
        """Verify that incorrect filter family, order, cutoff, or passes are rejected."""
        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": np.zeros((self.ref_n, 2)),
        }
        # Wrong family
        with self.assertRaisesRegex(ValueError, "filter family must be 'Butterworth'"):
            observe_friction_comparison(
                ref,
                self.sim_time,
                np.zeros((self.sim_n, 2)),
                filter_spec={"family": "Chebyshev", "order": 4, "cutoff_hz": 20.0, "passes": "forward/backward"},
            )
        # Wrong order
        with self.assertRaisesRegex(ValueError, "filter order must be 4"):
            observe_friction_comparison(
                ref,
                self.sim_time,
                np.zeros((self.sim_n, 2)),
                filter_spec={"family": "Butterworth", "order": 2, "cutoff_hz": 20.0, "passes": "forward/backward"},
            )
        # Wrong cutoff
        with self.assertRaisesRegex(ValueError, "filter cutoff_hz must be 20.0"):
            observe_friction_comparison(
                ref,
                self.sim_time,
                np.zeros((self.sim_n, 2)),
                filter_spec={"family": "Butterworth", "order": 4, "cutoff_hz": 15.0, "passes": "forward/backward"},
            )
        # Wrong passes
        with self.assertRaisesRegex(ValueError, "filter passes must be 'forward/backward'"):
            observe_friction_comparison(
                ref,
                self.sim_time,
                np.zeros((self.sim_n, 2)),
                filter_spec={"family": "Butterworth", "order": 4, "cutoff_hz": 20.0, "passes": "forward"},
            )

    def test_strict_shape_and_finite_validation(self):
        """Verify strict error on scalar, 1D, or wrong channel force arrays."""
        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": np.zeros((self.ref_n, 2)),
        }
        # 1-D force array (wrong channels)
        with self.assertRaisesRegex(ValueError, "pred_forces must have shape"):
            observe_friction_comparison(ref, self.sim_time, np.zeros(self.sim_n), filter_spec=self.filter_spec)

        # 3 channels instead of 2
        with self.assertRaisesRegex(ValueError, "pred_forces must have shape"):
            observe_friction_comparison(ref, self.sim_time, np.zeros((self.sim_n, 3)), filter_spec=self.filter_spec)

        # Target wrong shape
        with self.assertRaisesRegex(ValueError, "pre-20 Hz source must have shape"):
            observe_friction_comparison(
                {"grf_time_s": self.ref_time, "unfiltered_grf_target_n": np.zeros((self.ref_n, 3))},
                self.sim_time,
                np.zeros((self.sim_n, 2)),
                filter_spec=self.filter_spec,
            )

        # Non-finite tolerance
        with self.assertRaisesRegex(ValueError, "tolerance must be finite"):
            observe_friction_comparison(
                ref, self.sim_time, np.zeros((self.sim_n, 2)), filter_spec=self.filter_spec, tolerance=-1.0
            )

        # Non-finite normal_threshold
        with self.assertRaisesRegex(ValueError, "normal_threshold_n must be finite and positive"):
            observe_friction_comparison(
                ref, self.sim_time, np.zeros((self.sim_n, 2)), filter_spec=self.filter_spec, normal_threshold_n=-5.0
            )

    def test_nonuniform_and_gap_clock_rejected(self):
        """Verify that clocks with interior gaps or non-uniform timesteps are rejected."""
        bad_time = self.sim_time.copy()
        bad_time[100:] += 0.01
        ref = {
            "grf_time_s": self.ref_time,
            "unfiltered_grf_target_n": np.zeros((self.ref_n, 2)),
        }
        with self.assertRaisesRegex(ValueError, "non-uniform spacing or interior gaps"):
            observe_friction_comparison(ref, bad_time, np.zeros((self.sim_n, 2)), filter_spec=self.filter_spec)

    def test_baseline12_trace_observation(self):
        """Verify observation on real baseline12 reference and simulation trace."""
        base_dir = Path(os.environ.get("NEWTON_BASELINE12_DIR", "outputs/impedance_instron/baseline12"))
        if not (base_dir / "reference.npz").exists() or not (base_dir / "trace.npz").exists():
            self.skipTest("baseline12 reference or trace not found")

        with np.load(base_dir / "reference.npz") as ref_data:
            ref = {k: ref_data[k].copy() for k in ref_data.files}
        with np.load(base_dir / "trace.npz") as tr_data:
            tr_time = tr_data["time_s"].copy()
            tr_grf = tr_data["grf_n"].copy()

        obs = observe_friction_comparison(ref, tr_time, tr_grf)
        self.assertTrue(obs["complete"])
        self.assertEqual(len(obs["clock"]), 720)
        self.assertTrue(obs["metadata"]["diagnostics"]["target_negative_fz_retained"])
        self.assertIn("max_abs_force_rate_n_s", obs["raw_prediction_chatter"])


if __name__ == "__main__":
    unittest.main()
