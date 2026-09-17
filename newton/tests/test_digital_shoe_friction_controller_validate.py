# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for independent CPU validation of refit controller equilibrium under fixed Maxwell friction."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.digital_shoe.friction_controller_validate import (
    compute_actuator_diagnostics,
    compute_engineering_guards,
    get_artifact_maxwell_candidate,
    load_coefficients,
    validate_controller,
    verify_spline_profile_bounds,
)
from projects.digital_shoe.friction_dynamic import (
    configure_candidate_friction,
    load_baseline_bundle,
    parse_candidate,
)
from projects.impedance_instron.cartesian.trajectory import Spline

BASELINE_DIR = Path(
    os.environ.get(
        "NEWTON_BASELINE12_DIR",
        "outputs/impedance_instron/baseline12",
    )
)


class TestDigitalShoeFrictionControllerValidate(unittest.TestCase):
    """Test suite for controller validation runner, contract bounds, overwrite protection, and invariance."""

    def test_full_rollout_completion_uses_original_duration(self) -> None:
        """Keep preintegration force support separate from full simulation duration."""
        from projects.digital_shoe.friction_metrics import score_friction_trace  # noqa: PLC0415

        time = np.arange(4) * 0.01
        force = np.column_stack(([-1.0, -1.0, 1.0, 1.0], np.full(4, 100.0)))
        trace = {"time_s": time, "grf_n": force}
        common = {"grf_time_s": time, "grf_target_n": force}
        # After the original full-clock completion check, raw metrics use the
        # force-support clock without reinterpreting the original run summary.
        self.assertTrue(score_friction_trace(common, trace, 1)["complete"])
        summary = {
            "status": "completed",
            "failure": None,
            "integrated_steps": 4,
            "integrated_duration_s": 0.04,
            "actual_dt_s": 0.01,
        }
        wrong = score_friction_trace(common, trace, 1, summary=summary)
        self.assertFalse(wrong["complete"])
        self.assertIn("duration", wrong["failure_reason"])

    def test_output_directory_overwrite_rejected_before_simulation(self) -> None:
        """Reject existing output directory before starting any simulation."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            existing_dir = Path(tmp_dir) / "already_exists"
            existing_dir.mkdir()

            with patch("projects.digital_shoe.friction_controller_validate.simulate") as mock_sim:
                with self.assertRaises(FileExistsError):
                    validate_controller(
                        baseline_dir=BASELINE_DIR,
                        coefficients_path=BASELINE_DIR / "equilibrium.npz",
                        output_dir=existing_dir,
                    )
                mock_sim.assert_not_called()

    def test_coefficients_file_contract_and_validation(self) -> None:
        """Enforce coefficients file shape (12, 4), duration scalar, and finite numerical values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            p_missing_field = Path(tmp_dir) / "missing_field.npz"
            np.savez(p_missing_field, coefficients=np.zeros((12, 4)))
            with self.assertRaises(KeyError):
                load_coefficients(p_missing_field)

            p_bad_shape = Path(tmp_dir) / "bad_shape.npz"
            np.savez(p_bad_shape, coefficients=np.zeros((10, 4)), duration_s=0.36)
            with self.assertRaises(ValueError):
                load_coefficients(p_bad_shape)

            p_nonfinite = Path(tmp_dir) / "nonfinite.npz"
            coeffs = np.zeros((12, 4))
            coeffs[0, 0] = np.nan
            np.savez(p_nonfinite, coefficients=coeffs, duration_s=0.36)
            with self.assertRaises(ValueError):
                load_coefficients(p_nonfinite)

            p_duration_mismatch = Path(tmp_dir) / "duration_mismatch.npz"
            np.savez(p_duration_mismatch, coefficients=np.zeros((12, 4)), duration_s=0.5)
            with self.assertRaises(ValueError):
                load_coefficients(p_duration_mismatch, expected_duration_s=0.36)

            # Valid file
            p_valid = Path(tmp_dir) / "valid.npz"
            np.savez(p_valid, coefficients=np.ones((12, 4)), duration_s=0.36)
            coeffs_loaded, dur, file_sha = load_coefficients(p_valid, expected_duration_s=0.36)
            self.assertEqual(coeffs_loaded.shape, (12, 4))
            self.assertEqual(dur, 0.36)
            self.assertEqual(len(file_sha), 64)

    def test_spline_profile_bounds_verification_and_rejection(self) -> None:
        """Verify profile bounds checking and ensure out-of-bounds splines never simulate."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")

        with open(BASELINE_DIR / "profile.json") as f:
            profile = json.load(f)

        # Baseline spline should satisfy its own bounds
        with np.load(BASELINE_DIR / "equilibrium.npz") as arch:
            orig_coeffs = arch["coefficients"].copy()
            duration_s = float(arch["duration_s"])
            spline_base = Spline(duration_s, orig_coeffs)

        bounds_check = verify_spline_profile_bounds(spline_base, profile)
        self.assertTrue(bounds_check["within_bounds"])
        self.assertFalse(any(bounds_check["position_lower_violation"]))
        self.assertFalse(any(bounds_check["position_upper_violation"]))
        self.assertFalse(any(bounds_check["rate_violation"]))
        self.assertFalse(any(bounds_check["acceleration_violation"]))

        # Out-of-bounds spline must be caught
        bad_coeffs = orig_coeffs.copy()
        bad_coeffs[0, 0] = float(profile["equilibrium_lower"][0]) - 1.0
        spline_bad = Spline(duration_s, bad_coeffs)
        bad_check = verify_spline_profile_bounds(spline_bad, profile)
        self.assertFalse(bad_check["within_bounds"])
        self.assertTrue(bad_check["position_lower_violation"][0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_npz = Path(tmp_dir) / "bad_coeffs.npz"
            np.savez(bad_npz, coefficients=bad_coeffs, duration_s=duration_s)

            # With raise_on_failure=True: raises ValueError
            with patch("projects.digital_shoe.friction_controller_validate.simulate") as mock_sim:
                with self.assertRaises(ValueError):
                    validate_controller(
                        baseline_dir=BASELINE_DIR,
                        coefficients_path=bad_npz,
                        output_dir=Path(tmp_dir) / "out_raise",
                        raise_on_failure=True,
                    )
                mock_sim.assert_not_called()

            # With raise_on_failure=False: returns bounds_violated and NEVER simulates
            with patch("projects.digital_shoe.friction_controller_validate.simulate") as mock_sim:
                rep = validate_controller(
                    baseline_dir=BASELINE_DIR,
                    coefficients_path=bad_npz,
                    output_dir=Path(tmp_dir) / "out_noraise",
                    raise_on_failure=False,
                )
                mock_sim.assert_not_called()
                self.assertEqual(rep["status"], "bounds_violated")
                self.assertFalse(rep["complete"])

    def test_frozen_friction_and_gains_invariance(self) -> None:
        """Verify candidate Maxwell friction configuration and gains are frozen and invariant."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")

        _ref, prof, _spline, shoe, _base_cfg, _hashes, _sum_raw, _actual_dt = load_baseline_bundle(
            BASELINE_DIR, device="cpu"
        )

        fric_candidate = get_artifact_maxwell_candidate(BASELINE_DIR)
        fric_params, _ = parse_candidate(fric_candidate)
        self.assertEqual(fric_params.method, 7)
        self.assertEqual(fric_params.mu, 0.8)
        self.assertEqual(fric_params.kt_scale, 0.1)
        self.assertEqual(fric_params.kv_scale, 1.0)
        self.assertEqual(fric_params.viscous_ratio, 0.0)
        self.assertEqual(fric_params.release_dwell_s, 0.0005)
        self.assertEqual(fric_params.yield_width, 0.0)
        self.assertAlmostEqual(fric_params.shear_relaxation_time_s, 0.005150109522, places=8)

        # Non-Maxwell or altered friction must be rejected
        with self.assertRaises(ValueError):
            validate_controller(
                baseline_dir=BASELINE_DIR,
                friction_candidate={"method": 0, "mu": 0.8, "kt_scale": 1.0, "kv_scale": 1.0},
            )

        # Configure friction on shoe foundation and verify normal properties remain unchanged
        orig_block = shoe.foundation.world_blocks[0]
        normal_attrs = (
            "g_eq",
            "alpha",
            "g_eq2",
            "alpha2",
            "beta",
            "one_minus_two_poisson",
            "tau_s",
            "overstress",
            "inv_h2",
            "stretch_floor",
            "normal_damping",
        )
        orig_normal_vals = {attr: getattr(orig_block, attr) for attr in normal_attrs}

        configure_candidate_friction(shoe, fric_params)
        updated_block = shoe.foundation.world_params.numpy()[0]

        for attr in normal_attrs:
            np.testing.assert_allclose(
                float(updated_block[attr]),
                float(orig_normal_vals[attr]),
                rtol=1e-5,
                err_msg=f"Normal contact parameter {attr} was altered by friction configuration!",
            )

        # Profile gains must match baseline profile
        with open(BASELINE_DIR / "profile.json") as f:
            prof_raw = json.load(f)
        self.assertEqual(prof["hip_stiffness_n_m"], prof_raw["hip_stiffness_n_m"])
        self.assertEqual(prof["hip_damping_ns_m"], prof_raw["hip_damping_ns_m"])
        self.assertEqual(prof["joint_stiffness_nm_rad"], prof_raw["joint_stiffness_nm_rad"])
        self.assertEqual(prof["joint_damping_nms_rad"], prof_raw["joint_damping_nms_rad"])

    def test_simulation_config_invariance(self) -> None:
        """Verify simulation numerical screens and dt match baseline bundle without altered limits."""
        if not BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {BASELINE_DIR}")

        _ref, _prof, _spline, _shoe, base_cfg, _hashes, sum_raw, actual_dt = load_baseline_bundle(
            BASELINE_DIR, device="cpu"
        )
        sim_cfg_raw = sum_raw["simulation_config"]

        self.assertEqual(base_cfg.gravity_m_s2, float(sim_cfg_raw["gravity_m_s2"]))
        self.assertEqual(base_cfg.compression_limit, float(sim_cfg_raw["compression_limit"]))
        self.assertEqual(base_cfg.maximum_force_n, float(sim_cfg_raw["maximum_force_n"]))
        self.assertEqual(base_cfg.minimum_hip_height_m, float(sim_cfg_raw["minimum_hip_height_m"]))
        self.assertEqual(base_cfg.maximum_speed, float(sim_cfg_raw["maximum_speed"]))
        self.assertEqual(base_cfg.joint_limits_diagnostic, bool(sim_cfg_raw["joint_limits_diagnostic"]))
        self.assertEqual(base_cfg.dt_s, actual_dt)

    def test_actuator_4channel_and_engineering_guards(self) -> None:
        """Verify component-wise 4-channel peak and rate engineering guard checks."""
        trace_base = {
            "hip_force_n": np.array([[100.0, 200.0], [120.0, 210.0], [90.0, 195.0]]),
            "joint_torque_nm": np.array([[10.0, -5.0], [15.0, -6.0], [8.0, -4.0]]),
            "time_s": np.array([0.0, 0.001, 0.002]),
        }
        dt_s = 0.001
        diag_base = compute_actuator_diagnostics(trace_base, dt_s)
        self.assertEqual(len(diag_base["effort_4ch_max_abs"]), 4)
        self.assertEqual(len(diag_base["effort_rate_4ch_max_abs"]), 4)

        # Refit within 1.5x peak and 2.0x rate
        trace_refit_pass = {
            "hip_force_n": np.array([[110.0, 220.0], [130.0, 230.0], [100.0, 210.0]]),
            "joint_torque_nm": np.array([[11.0, -5.5], [16.0, -6.5], [9.0, -4.5]]),
            "time_s": np.array([0.0, 0.001, 0.002]),
        }
        diag_refit_pass = compute_actuator_diagnostics(trace_refit_pass, dt_s)
        guards_pass = compute_engineering_guards(diag_base, diag_refit_pass)
        self.assertTrue(guards_pass["overall_guard_passed"])
        self.assertTrue(guards_pass["all_peaks_passed"])
        self.assertTrue(guards_pass["all_rates_passed"])

        # Refit exceeding 1.5x peak on knee torque
        trace_refit_fail = {
            "hip_force_n": trace_base["hip_force_n"],
            "joint_torque_nm": np.array([[10.0, -5.0], [30.0, -6.0], [8.0, -4.0]]),
            "time_s": trace_base["time_s"],
        }
        diag_refit_fail = compute_actuator_diagnostics(trace_refit_fail, dt_s)
        guards_fail = compute_engineering_guards(diag_base, diag_refit_fail)
        self.assertFalse(guards_fail["overall_guard_passed"])
        self.assertFalse(guards_fail["all_peaks_passed"])


if __name__ == "__main__":
    unittest.main()
