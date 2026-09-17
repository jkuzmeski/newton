# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for frozen-controller free-leg digital shoe dynamic qualification."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.digital_shoe.friction_dynamic import (
    DEFAULT_BASELINE_DIR,
    CandidateParameters,
    _finite_report,
    configure_candidate_friction,
    load_baseline_bundle,
    parse_candidate,
    run_dynamic_qualification,
)
from projects.digital_shoe.friction_solver import FrictionSolver


class TestDigitalShoeFrictionDynamic(unittest.TestCase):
    """Test suite for free-leg friction dynamic qualification runner and checks."""

    def test_failed_diagnostics_use_strict_json(self) -> None:
        """Preserve failure status while replacing unavailable numeric diagnostics with null."""
        report = {"complete": False, "metrics": [float("nan"), np.float32("inf"), -float("inf"), 2.0]}
        safe = _finite_report(report)
        self.assertFalse(safe["complete"])
        self.assertEqual(safe["metrics"], [None, None, None, 2.0])
        json.dumps(safe, allow_nan=False)
        self.assertTrue(np.isnan(report["metrics"][0]))

    def test_candidate_parameter_validation(self) -> None:
        """Verify candidate parameter validation, rejection of diagnostic modes, booleans, and nonzero yield."""
        # Valid legacy and deflection configurations
        p_legacy = CandidateParameters(
            method=0, mu=0.5, kt_scale=1.2, kv_scale=0.8, viscous_ratio=0.3, release_dwell_s=0.001
        )
        self.assertEqual(p_legacy.method, 0)

        p_deflection = CandidateParameters(
            method=1, mu=0.4, kt_scale=0.9, kv_scale=1.1, viscous_ratio=0.1, release_dwell_s=0.002, yield_width=0.0
        )
        self.assertEqual(p_deflection.method, 1)

        # Reject boolean method or yield_width
        with self.assertRaises(TypeError):
            CandidateParameters(
                method=True, mu=0.5, kt_scale=1.0, kv_scale=1.0, viscous_ratio=0.2, release_dwell_s=0.001
            )
        with self.assertRaises(TypeError):
            parse_candidate({"method": True, "mu": 0.5, "kt_scale": 1.0, "kv_scale": 1.0})
        with self.assertRaises(TypeError):
            CandidateParameters(
                method=0, mu=0.5, kt_scale=1.0, kv_scale=1.0, viscous_ratio=0.2, release_dwell_s=0.001, yield_width=True
            )

        # Reject fractional method
        with self.assertRaises(ValueError):
            parse_candidate({"method": 0.5, "mu": 0.5, "kt_scale": 1.0, "kv_scale": 1.0})

        # Reject diagnostic method 2 (regularized) and method 3 (anchor_nominal)
        with self.assertRaises(ValueError) as ctx:
            CandidateParameters(method=2, mu=0.5, kt_scale=1.0, kv_scale=1.0, viscous_ratio=0.2, release_dwell_s=0.001)
        self.assertIn("regularized", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            CandidateParameters(method=3, mu=0.5, kt_scale=1.0, kv_scale=1.0, viscous_ratio=0.2, release_dwell_s=0.001)
        self.assertIn("anchor_nominal", str(ctx.exception))

        # Reject nonzero yield_width
        with self.assertRaises(ValueError) as ctx:
            CandidateParameters(
                method=1,
                mu=0.5,
                kt_scale=1.0,
                kv_scale=1.0,
                viscous_ratio=0.2,
                release_dwell_s=0.001,
                yield_width=0.05,
            )
        self.assertIn("yield_width", str(ctx.exception))

        # Reject non-positive kt_scale
        with self.assertRaises(ValueError):
            CandidateParameters(method=0, mu=0.5, kt_scale=0.0, kv_scale=1.0, viscous_ratio=0.2, release_dwell_s=0.001)

        # Test string parsing via parse_candidate
        p_parsed, _ = parse_candidate({"method": "legacy", "mu": 0.6, "kt_scale": 1.0, "kv_scale": 1.0})
        self.assertEqual(p_parsed.method, 0)
        self.assertAlmostEqual(p_parsed.mu, 0.6)

        p_parsed_def, _ = parse_candidate({"method": "deflection", "mu": 0.3, "kt_scale": 0.5, "kv_scale": 0.5})
        self.assertEqual(p_parsed_def.method, 1)

        with self.assertRaises(ValueError):
            parse_candidate({"method": "regularized"})

    def test_config_mapping_only_friction_changes(self) -> None:
        """Verify candidate friction configuration changes only friction fields and preserves normal laws."""
        if not DEFAULT_BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {DEFAULT_BASELINE_DIR}")

        _ref, _prof, _spline, shoe, _cfg, _hashes, _sum_raw, _actual_dt = load_baseline_bundle(
            DEFAULT_BASELINE_DIR, device="cpu"
        )

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
        orig_rest_lens = shoe.foundation.rest_len.numpy().copy()
        orig_areas = shoe.foundation.area.numpy().copy()
        orig_kt = shoe.foundation.friction_kt.numpy().copy()
        orig_kv = shoe.foundation.friction_kv.numpy().copy()

        # 1. Apply legacy candidate with 2x kt, 0.5x kv, mu=0.45
        cand_legacy = CandidateParameters(
            method=0,
            mu=0.45,
            kt_scale=2.0,
            kv_scale=0.5,
            viscous_ratio=0.15,
            release_dwell_s=0.002,
        )
        configure_candidate_friction(shoe, cand_legacy)

        self.assertIsNone(shoe.foundation.friction_solver)
        np.testing.assert_allclose(shoe.foundation.friction_kt.numpy(), orig_kt * 2.0, rtol=1e-5)
        np.testing.assert_allclose(shoe.foundation.friction_kv.numpy(), orig_kv * 0.5, rtol=1e-5)
        updated_block = shoe.foundation.world_params.numpy()[0]
        self.assertAlmostEqual(float(updated_block["mu"]), 0.45, places=5)
        self.assertAlmostEqual(float(updated_block["friction_viscous_ratio"]), 0.15, places=5)
        self.assertAlmostEqual(float(updated_block["friction_release_dwell_s"]), 0.002, places=5)

        for attr in normal_attrs:
            np.testing.assert_allclose(
                float(updated_block[attr]),
                float(orig_normal_vals[attr]),
                rtol=1e-5,
                err_msg=f"Normal property {attr} was modified!",
            )
        np.testing.assert_array_equal(shoe.foundation.rest_len.numpy(), orig_rest_lens)
        np.testing.assert_array_equal(shoe.foundation.area.numpy(), orig_areas)

        # 2. Apply deflection candidate
        cand_defl = CandidateParameters(
            method=1,
            mu=0.35,
            kt_scale=0.8,
            kv_scale=1.2,
            viscous_ratio=0.4,
            release_dwell_s=0.003,
            yield_width=0.0,
        )
        configure_candidate_friction(shoe, cand_defl)

        self.assertIsNotNone(shoe.foundation.friction_solver)
        self.assertEqual(shoe.foundation.friction_solver.solver.mode, FrictionSolver.MODES["deflection"])
        np.testing.assert_allclose(shoe.foundation.friction_kt.numpy(), orig_kt * 0.8, rtol=1e-5)
        np.testing.assert_allclose(shoe.foundation.friction_kv.numpy(), orig_kv * 1.2, rtol=1e-5)

        updated_block2 = shoe.foundation.world_params.numpy()[0]
        for attr in normal_attrs:
            np.testing.assert_allclose(
                float(updated_block2[attr]),
                float(orig_normal_vals[attr]),
                rtol=1e-5,
                err_msg=f"Normal property {attr} modified during deflection config!",
            )

    def test_output_directory_validation_before_simulation(self) -> None:
        """Verify that existing output directories are rejected before any simulations execute."""
        if not DEFAULT_BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {DEFAULT_BASELINE_DIR}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            existing_dir = Path(tmp_dir) / "already_exists"
            existing_dir.mkdir()

            with patch("projects.digital_shoe.friction_dynamic.simulate") as mock_sim:
                with self.assertRaises(FileExistsError):
                    run_dynamic_qualification(
                        baseline_dir=DEFAULT_BASELINE_DIR,
                        candidate_source=None,
                        output_dir=existing_dir,
                    )
                # Ensure simulate was NEVER called
                mock_sim.assert_not_called()

    def test_input_nonmutation(self) -> None:
        """Verify baseline data inputs and candidate definitions are not mutated by qualification."""
        if not DEFAULT_BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {DEFAULT_BASELINE_DIR}")

        ref, prof, spline, _shoe, _cfg, _hashes, _sum_raw, _actual_dt = load_baseline_bundle(
            DEFAULT_BASELINE_DIR, device="cpu"
        )

        ref_keys = list(ref.keys())
        ref_time_copy = ref["time_s"].copy()
        ref_grf_copy = ref["grf_target_n"].copy()
        prof_copy = copy.deepcopy(prof)
        spline_coeffs_copy = spline.coefficients.copy()
        spline_dur = spline.duration_s

        cand_dict = {
            "method": 0,
            "mu": 0.55,
            "kt_scale": 1.1,
            "kv_scale": 0.9,
            "viscous_ratio": 0.25,
            "release_dwell_s": 0.001,
            "yield_width": 0.0,
        }
        cand_dict_copy = copy.deepcopy(cand_dict)

        report = run_dynamic_qualification(
            baseline_dir=DEFAULT_BASELINE_DIR,
            candidate_source=cand_dict,
            output_dir=None,
            device="cpu",
            dt_scale=16.0,
            raise_on_failure=True,
        )

        self.assertTrue(report["complete"])
        self.assertEqual(list(ref.keys()), ref_keys)
        np.testing.assert_array_equal(ref["time_s"], ref_time_copy)
        np.testing.assert_array_equal(ref["grf_target_n"], ref_grf_copy)
        self.assertEqual(prof, prof_copy)
        np.testing.assert_array_equal(spline.coefficients, spline_coeffs_copy)
        self.assertEqual(spline.duration_s, spline_dur)
        self.assertEqual(cand_dict, cand_dict_copy)

    def test_mocked_simulate_verifies_uses_original_controller(self) -> None:
        """Verify dynamic qualification invokes projects.impedance_instron.cartesian.run.simulate directly."""
        with patch("projects.digital_shoe.friction_dynamic.simulate") as mock_sim:
            dummy_time = np.linspace(0.0, 0.36, 100)
            dummy_trace = {
                "time_s": dummy_time,
                "state": np.zeros((100, 5)),
                "velocity": np.zeros((100, 5)),
                "grf_n": np.zeros((100, 2)),
                "joints_m": np.zeros((100, 4, 2)),
                "equilibrium": np.zeros((100, 4)),
                "hip_force_n": np.zeros((100, 2)),
                "joint_torque_nm": np.zeros((100, 2)),
                "ankle_contact_moment_nm": np.zeros(100),
                "compression_fraction": np.zeros(100),
                "driven_compression_fraction": np.zeros(100),
                "passive_compression_fraction": np.zeros(100),
                "passive_cap_column_count": np.zeros(100, dtype=np.int32),
            }
            dummy_summary = {
                "status": "completed",
                "failure": None,
                "integrated_steps": 100,
                "integrated_duration_s": 0.36,
                "terminal_state": [0.0] * 5,
                "actual_dt_s": 0.36 / 100,
            }
            mock_sim.return_value = (dummy_trace, dummy_summary)

            if not DEFAULT_BASELINE_DIR.exists():
                self.skipTest(f"Baseline directory not found: {DEFAULT_BASELINE_DIR}")

            run_dynamic_qualification(
                baseline_dir=DEFAULT_BASELINE_DIR,
                candidate_source={
                    "method": 0,
                    "mu": 0.6,
                    "kt_scale": 1.0,
                    "kv_scale": 1.0,
                    "viscous_ratio": 0.2,
                    "release_dwell_s": 0.0005,
                },
                output_dir=None,
                device="cpu",
                dt_scale=1.0,
                raise_on_failure=False,
            )

            self.assertEqual(mock_sim.call_count, 2)
            call_args_baseline = mock_sim.call_args_list[0]
            self.assertEqual(len(call_args_baseline[0]), 4)
            self.assertIn("config", call_args_baseline[1])

    def test_dynamic_qualification_short_smoke(self) -> None:
        """Verify dynamic qualification end-to-end smoke run generates traces, scores, and summary report."""
        if not DEFAULT_BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {DEFAULT_BASELINE_DIR}")

        cand_dict = {
            "method": "deflection",
            "mu": 0.45,
            "kt_scale": 0.8,
            "kv_scale": 1.2,
            "viscous_ratio": 0.3,
            "release_dwell_s": 0.002,
            "yield_width": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "smoke_qual"
            report = run_dynamic_qualification(
                baseline_dir=DEFAULT_BASELINE_DIR,
                candidate_source=cand_dict,
                output_dir=out_dir,
                device="cpu",
                dt_scale=8.0,
                raise_on_failure=True,
            )

            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["complete"])
            self.assertIsNone(report["failure_reason"])

            self.assertTrue((out_dir / "trace_baseline.npz").exists())
            self.assertTrue((out_dir / "trace_candidate.npz").exists())
            self.assertTrue((out_dir / "summary.json").exists())

            self.assertIn("full_horizontal_force_rmse_n", report["baseline"]["friction_metrics"]["comparison_metrics"])
            self.assertIn(
                "full_horizontal_force_rmse_n", report["candidate_run"]["friction_metrics"]["comparison_metrics"]
            )
            self.assertIn("hip_rmse_m", report["baseline"]["six_channel_metrics"])
            self.assertIn("joint_rmse_rad", report["baseline"]["six_channel_metrics"])
            self.assertIn("force_rmse_n", report["baseline"]["six_channel_metrics"])
            self.assertIn("six_channel_metrics_diff", report["comparison"])

    def test_full_baseline_actual_run_optional_env(self) -> None:
        """Run full 5760-step dynamic qualification when NEWTON_RUN_FULL_DYNAMIC_FRICTION is enabled."""
        if not os.environ.get("NEWTON_RUN_FULL_DYNAMIC_FRICTION"):
            self.skipTest(
                "Optional full dynamic qualification disabled (enable with NEWTON_RUN_FULL_DYNAMIC_FRICTION=1)"
            )

        if not DEFAULT_BASELINE_DIR.exists():
            self.skipTest(f"Baseline directory not found: {DEFAULT_BASELINE_DIR}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "full_qual"
            report = run_dynamic_qualification(
                baseline_dir=DEFAULT_BASELINE_DIR,
                candidate_source=None,
                output_dir=out_dir,
                device="cpu",
                dt_scale=1.0,
                raise_on_failure=True,
            )
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["baseline"]["integrated_steps"], 5760)
            self.assertEqual(report["candidate_run"]["integrated_steps"], 5760)
