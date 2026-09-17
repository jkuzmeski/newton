# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU-batched free-leg friction dynamic qualification."""

import os
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe.friction_dynamic_gpu import (
    FrictionDynamicGPUWorkspace,
)
from projects.impedance_instron.cartesian import data
from projects.impedance_instron.cartesian.gpu.engine import Engine
from projects.impedance_instron.cartesian.profile import load as load_profile

BASELINE_DIR = Path(
    os.environ.get(
        "NEWTON_BASELINE12_DIR",
        "outputs/impedance_instron/baseline12",
    )
)


class TestDigitalShoeFrictionDynamicGPU(unittest.TestCase):
    """Test suite for GPU-batched free-leg friction qualification and parity."""

    @classmethod
    def setUpClass(cls):
        """Verify CUDA availability and baseline directory existence."""
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for GPU friction dynamic tests")
        if not BASELINE_DIR.is_dir():
            raise unittest.SkipTest(f"Baseline directory not found: {BASELINE_DIR}")

    def test_observation_postprocessing_does_not_change_physics(self) -> None:
        """Keep raw force and mechanics identical when enabling matched observation scores."""
        workspace = FrictionDynamicGPUWorkspace(BASELINE_DIR, world_count=2)
        baseline = [0, 0.8, 1, 1, 0.2, 0.0005, 0, 0.8, 0.1]
        failing = [4, 0.8, 1, 1, 0.2, 0.0005, 0, 0.2, 0.03]
        raw = workspace.evaluate([baseline, failing], curves=True)
        workspace.matched_observation = True
        observed = workspace.evaluate([baseline, failing], curves=True)
        np.testing.assert_array_equal(raw["curves"], observed["curves"])
        np.testing.assert_array_equal(raw["engine_rmse"], observed["engine_rmse"])
        for key in raw["fit_scores"]:
            np.testing.assert_array_equal(raw["fit_scores"][key], observed["fit_scores"][key])
        self.assertTrue(np.isfinite(observed["observation_scores"]["loss"][0]))
        self.assertTrue(np.isinf(observed["observation_scores"]["loss"][1]))
        self.assertTrue(np.isnan(observed["observation_force_rmse_n"][1]).all())
        self.assertNotEqual(observed["observation_scores"]["loss"][0], raw["fit_scores"]["loss"][0])
        self.assertFalse(observed["observation_metadata"]["butterworth_20hz"]["clamp_vertical_zero"])

    def test_maxwell_reset_and_raw_smoothness(self) -> None:
        """Reset internal shear stress and reduce raw force steps without output filtering."""
        workspace = FrictionDynamicGPUWorkspace(BASELINE_DIR, world_count=2)
        direct = [1, 0.8, 1, 1, 0.2, 0.0005, 0, 0.8, 0.1, 1e12, 0.001, 0.005150109522]
        maxwell = [7, 0.8, 0.1, 1, 0, 0.0005, 0, 0.8, 0.1, 1e12, 0.001, 0.005150109522]
        first = workspace.evaluate([direct, maxwell], curves=True)
        self.assertTrue(first["completed"].all())
        self.assertLess(first["raw_force_diagnostics"][1, 2], 0.4 * first["raw_force_diagnostics"][0, 2])
        self.assertEqual(float(first["fit_scores"]["positive_energy_residual_j"][1]), 0.0)
        second = workspace.evaluate([maxwell, direct], curves=True)
        np.testing.assert_array_equal(first["curves"][0], second["curves"][1])
        np.testing.assert_array_equal(first["curves"][1], second["curves"][0])

    def test_baseline_parity_with_original_engine(self) -> None:
        """Verify exact mathematical parity between FrictionParameterAdapter and original Engine."""
        # 4 worlds smoke test
        workspace = FrictionDynamicGPUWorkspace(
            baseline_dir=BASELINE_DIR,
            world_count=4,
            device="cuda:0",
        )

        baseline_cand = [0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]
        params = np.tile(baseline_cand, (4, 1))

        # Evaluate through workspace
        workspace_results = workspace.evaluate(params, curves=True)
        self.assertLess(float(workspace.target_time.numpy()[-1]), workspace.duration_s)
        self.assertTrue(np.all(workspace.fraction.numpy() <= 1.0 + 1e-7))
        self.assertEqual(workspace.sample_count, len(workspace.reference["grf_time_s"]) - 1)

        # Directly evaluate original Engine without adapter
        ref = data.load(BASELINE_DIR / "reference.npz")
        prof = load_profile(BASELINE_DIR / "profile.json")
        mount_m = workspace.summary["shoe"]["mount_m"]
        static_pitch_rad = float(workspace.summary["shoe"]["static_pitch_rad"])
        orig_engine = Engine(
            ref,
            prof,
            BASELINE_DIR / "digital_shoe.json",
            mount_m,
            static_pitch_rad,
            config=workspace.sim_config,
            settings=workspace.fit_config,
            world_count=4,
            device="cuda:0",
        )
        orig_scores = orig_engine.evaluate(workspace.frozen_coefficients)

        # Check exact bit-for-bit parity
        np.testing.assert_allclose(
            workspace_results["engine_loss"],
            orig_scores["loss"],
            rtol=1e-7,
            atol=1e-7,
            err_msg="Engine loss differs between adapter and original Engine",
        )
        np.testing.assert_allclose(
            workspace_results["engine_rmse"],
            orig_scores["rmse"],
            rtol=1e-7,
            atol=1e-7,
            err_msg="Engine RMSE differs between adapter and original Engine",
        )
        np.testing.assert_array_equal(
            workspace_results["integrated_steps"],
            orig_scores["integrated_steps"],
            err_msg="Integrated steps differ between adapter and original Engine",
        )
        np.testing.assert_array_equal(
            workspace_results["failure_code"],
            orig_scores["failure_code"],
            err_msg="Failure codes differ between adapter and original Engine",
        )

    def test_world_isolation_and_permutation_invariance(self) -> None:
        """Verify candidate isolation and order permutation invariance across GPU worlds."""
        cand_base = [0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]
        cand_defl = [1.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]

        workspace = FrictionDynamicGPUWorkspace(
            baseline_dir=BASELINE_DIR,
            world_count=2,
            device="cuda:0",
        )

        # Forward permutation: [base, defl]
        perm_a = np.array([cand_base, cand_defl], dtype=np.float32)
        res_a = workspace.evaluate(perm_a, curves=True)

        # Reversed permutation: [defl, base]
        perm_b = np.array([cand_defl, cand_base], dtype=np.float32)
        res_b = workspace.evaluate(perm_b, curves=True)

        # Candidate 0 in A should match candidate 1 in B exactly
        self.assertAlmostEqual(res_a["engine_loss"][0], res_b["engine_loss"][1], places=7)
        np.testing.assert_allclose(res_a["curves"][0], res_b["curves"][1], rtol=1e-6, atol=1e-6)

        # Candidate 1 in A should match candidate 0 in B exactly
        self.assertAlmostEqual(res_a["engine_loss"][1], res_b["engine_loss"][0], places=7)
        np.testing.assert_allclose(res_a["curves"][1], res_b["curves"][0], rtol=1e-6, atol=1e-6)

    def test_deflection_reset_between_rollouts(self) -> None:
        """Verify that state and deflection history are cleanly reset on repeated rollouts."""
        workspace = FrictionDynamicGPUWorkspace(
            baseline_dir=BASELINE_DIR,
            world_count=2,
            device="cuda:0",
        )

        cand = [1.0, 0.497158, 0.416209, 5.20228, 0.004689, 0.00186345, 0.0]
        params = np.tile(cand, (2, 1))

        # First rollout
        res_first = workspace.evaluate(params, curves=True)

        # Second rollout with same parameters
        res_second = workspace.evaluate(params, curves=True)

        np.testing.assert_allclose(
            res_first["engine_loss"],
            res_second["engine_loss"],
            rtol=1e-7,
            atol=1e-7,
            err_msg="Loss differs on consecutive rollouts; deflection or history leaked",
        )
        np.testing.assert_allclose(
            res_first["curves"],
            res_second["curves"],
            rtol=1e-6,
            atol=1e-6,
            err_msg="Curves differ on consecutive rollouts; deflection or history leaked",
        )

    def test_rejection_of_unsupported_methods_and_yield_width(self) -> None:
        """Verify rejection of diagnostic methods 2/3 and non-zero yield_width."""
        workspace = FrictionDynamicGPUWorkspace(
            baseline_dir=BASELINE_DIR,
            world_count=2,
            device="cuda:0",
        )

        # Method 2 (regularized) should be rejected
        with self.assertRaises(ValueError):
            workspace.evaluate([[2.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0], [0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]])

        # Method 3 (anchor_nominal) should be rejected
        with self.assertRaises(ValueError):
            workspace.evaluate([[3.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0], [0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]])

        # Non-zero yield_width should be rejected
        with self.assertRaises(ValueError):
            workspace.evaluate([[1.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.05], [0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]])

    def test_failed_candidate_cannot_reuse_previous_force_scores(self) -> None:
        """Invalidate full-stance scores when a candidate stops before the end."""
        workspace = FrictionDynamicGPUWorkspace(BASELINE_DIR, world_count=2)
        baseline = [0, 0.8, 1, 1, 0.2, 0.0005, 0, 0.8, 0.1]
        failing = [4, 0.8, 1, 1, 0.2, 0.0005, 0, 0.2, 0.03]
        workspace.evaluate([baseline, baseline])
        result = workspace.evaluate([baseline, failing], curves=True)
        self.assertTrue(result["completed"][0])
        self.assertFalse(result["completed"][1])
        self.assertTrue(np.isinf(result["friction_scores"]["loss"][1]))
        self.assertTrue(np.isnan(result["friction_scores"]["braking_impulse_ns"][1]))
        self.assertTrue(np.isinf(result["fit_scores"]["loss"][1]))
        recorded = int(result["recorded_steps"][1])
        self.assertLess(recorded, workspace.steps)
        self.assertTrue(np.isnan(result["curves"][1, recorded:]).all())
        self.assertTrue(np.isfinite(result["curves"][0]).all())

    def test_candidate_normal_force_unmodified(self) -> None:
        """Verify normal ground force Z is untouched by friction parameter changes given identical pose."""
        workspace = FrictionDynamicGPUWorkspace(
            baseline_dir=BASELINE_DIR,
            world_count=2,
            device="cuda:0",
        )

        # Set up two candidates with different friction methods and coefficients
        cand_base = [0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]
        cand_mod = [7.0, 0.5, 0.5, 2.0, 0.0, 0.001, 0.0, 0.5, 0.1, 1e12, 0.001, 0.005]
        cand_base = [*cand_base, 0.8, 0.1, 1e12, 0.001, 0.005]
        params = np.array([cand_base, cand_mod], dtype=np.float32)
        workspace.adapter.set_parameters(params)

        # Prescribe a valid, loaded pose directly; the uninitialized controller
        # coefficient buffer must not decide this isolated contact test.
        f = workspace.engine.foundation
        anchors = f.anchor_local.numpy()
        driven = f.driven.numpy().astype(bool)
        height = -float(np.min(anchors[driven, 2])) - 0.005
        poses = np.tile(np.array([0, 0, height, 0, 0, 0, 1], np.float32), (2, 1))
        velocities = np.tile(np.array([0.2, 0, 0, 0, 0, 0], np.float32), (2, 1))
        f.reset()
        workspace.engine.carriers.body_q.assign(poses)
        workspace.engine.carriers.body_qd.assign(velocities)
        f.apply(workspace.engine.carriers, workspace.engine.dt, clear_body_force=True)
        gf_world0 = f.ground_force.numpy().reshape(2, f.column_count, 3)[0, :, 2]
        gf_world1 = f.ground_force.numpy().reshape(2, f.column_count, 3)[1, :, 2]

        self.assertGreater(float(np.max(gf_world0)), 0.0)
        # In identical kinematics, normal reaction force Z must match bit-for-bit across candidate worlds
        np.testing.assert_array_equal(
            gf_world0,
            gf_world1,
            err_msg="Normal ground force Z altered by friction parameters under identical kinematics",
        )
