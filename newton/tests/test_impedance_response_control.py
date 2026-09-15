# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify experimental response without changing the default policy mechanics."""

import copy
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import warp as wp

from newton.tests.test_impedance_simple_rig import _artifact, _reference, _rollout
from projects.impedance_instron.simple.response_control import ResponseConfig, RigResponse
from projects.impedance_instron.simple.rig import Rig, RigConfig


class TestResponseControl(unittest.TestCase):
    """Exercise frozen movement intent, independent gains, and external loads."""

    def setUp(self):
        """Create a small shoe and a reference with explicit nominal movement."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.artifact = Path(directory.name) / "shoe.json"
        self.artifact.write_text(json.dumps(_artifact()))
        self.config = RigConfig(substeps=16, use_graph=False)
        reference = _reference(self.artifact)
        provenance = copy.deepcopy(reference.provenance)
        provenance["inverse_dynamics"] = {
            "leg_measured_length_m": [1.0] * 4,
            "leg_measured_rate_m_s": [0.0] * 4,
            "leg_damping_n_s_m": 0.5 * math.sqrt(12000 * 8),
            "ankle_damping_n_m_s_rad": math.sqrt(4000 * 0.025),
        }
        self.reference = replace(
            reference,
            pitch_rad=np.zeros(4),
            pitch_rate_rad_s=np.zeros(4),
            inverse_leg_force_n=np.full(4, 100.0),
            inverse_ankle_torque_n_m=np.full(4, 5.0),
            provenance=provenance,
        )

    def rig(self, response=None, reference=None, config=None, *, worlds=1, device="cpu"):
        """Construct one response without refitting or modifying source artifacts."""
        return RigResponse(
            reference or self.reference,
            self.artifact,
            response_config=response,
            config=config or self.config,
            num_worlds=worlds,
            device=device,
        )

    def test_config_validation_and_zero_action_contract(self):
        """Reject malformed settings and nonzero policy controls without advancing."""
        response = ResponseConfig()
        self.assertEqual(ResponseConfig.from_dict(response.to_dict()), response)
        with self.assertRaises(FrozenInstanceError):
            response.controller_mode = "intent"
        for kwargs in (
            {"controller_mode": "learned"},
            {"leg_stiffness_n_m": 0},
            {"ankle_stiffness_n_m_rad": -1},
            {"leg_damping_n_s_m": -1},
            {"ground_height_m": float("nan")},
            {"push_force_z_n": 1e100, "push_duration_s": 0.01},
            {"leg_stiffness_n_m": None},
            {"push_start_s": -1},
            {"push_duration_s": -1},
            {"push_force_x_n": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResponseConfig(**kwargs)
        with self.assertRaises(TypeError):
            ResponseConfig.from_dict({"action_dim": 3})
        with self.assertRaisesRegex(ValueError, "contained"):
            self.rig(ResponseConfig(push_start_s=0.02, push_duration_s=0.01, push_force_x_n=1))
        rig = self.rig()
        for action in ([[1e-30, 0]], [[0, -0.1]], [0, 0], [[float("nan"), 0]]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                rig.step(action)
            self.assertEqual(rig.index, 0)
        rig.step([[0, 0]])
        self.assertEqual(rig.frame, 1)

    def test_noop_response_matches_default_zero_action_rig(self):
        """Match original mechanics and observations with default response settings."""
        reference = _reference(self.artifact)
        original = Rig(reference, self.artifact, config=self.config, device="cpu")
        response = self.rig(reference=reference)
        np.testing.assert_array_equal(original.reset(), response.reset())
        for _ in range(original.episode_frames):
            old_obs, old_reward, old_done, _ = original.step([[0, 0]])
            new_obs, new_reward, new_done, _ = response.step([[0, 0]])
            np.testing.assert_allclose(old_obs, new_obs, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(old_reward, new_reward, rtol=2e-6, atol=2e-6)
            self.assertEqual(old_done, new_done)
        for name, value in original.trace().items():
            np.testing.assert_allclose(value, response.trace()[name], rtol=5e-6, atol=2e-4, err_msg=name)
        self.assertEqual(original.config, response.config)
        self.assertEqual(original.input_fingerprints, response.input_fingerprints)
        self.assertNotEqual(original.action_contract, response.action_contract)

    def test_nominal_load_invariance_and_independent_damping(self):
        """Keep nominal motion loads independent of selected stiffness and damping."""
        first_damping = None
        for k, ka, b, ba in ((3000, 1000, 0, 0), (12000, 4000, 40, 3), (48000, 16000, 300, 20)):
            response = ResponseConfig(
                controller_mode="intent",
                leg_stiffness_n_m=k,
                ankle_stiffness_n_m_rad=ka,
                leg_damping_n_s_m=b,
                ankle_damping_n_m_s_rad=ba,
            )
            rig = self.rig(response)
            rig.step([[0, 0]])
            trace = rig.trace()
            self.assertAlmostEqual(trace["leg_raw_force_n"][0], 100, delta=0.002)
            self.assertAlmostEqual(trace["ankle_raw_torque_n_m"][0], 5, delta=0.002)
            np.testing.assert_allclose(trace["leg_stiffness_n_m"], k, rtol=1e-6)
            np.testing.assert_allclose(trace["ankle_stiffness_n_m_rad"], ka, rtol=1e-6)
            np.testing.assert_array_equal(trace["leg_damping_n_s_m"], b)
            np.testing.assert_array_equal(trace["ankle_damping_n_m_s_rad"], ba)
            np.testing.assert_array_equal(trace["leg_stiffness_rate_n_m_s"], 0)
            nominal = replace(response, leg_damping_n_s_m=None, ankle_damping_n_m_s_rad=None).resolved(self.reference)
            damping = (nominal.leg_damping_n_s_m, nominal.ankle_damping_n_m_s_rad)
            if first_damping is None:
                first_damping = damping
            self.assertEqual(damping, first_damping)
        with self.assertRaisesRegex(ValueError, "frozen reference"):
            self.rig(config=replace(self.config, leg_damping_ratio=0.7))

    def test_perturbation_force_slopes_from_initial_fixed_gains(self):
        """Apply signed stiffness and damping feedback immediately without a ramp."""
        response = ResponseConfig(
            controller_mode="intent",
            leg_stiffness_n_m=6000,
            ankle_stiffness_n_m_rad=2000,
            leg_damping_n_s_m=60,
            ankle_damping_n_m_s_rad=4,
        )
        for delta_length, delta_rate, delta_angle, delta_omega in (
            (0.002, 0, 0, 0),
            (-0.002, 0, 0, 0),
            (0, 0.03, 0, 0),
            (0, 0, 0.002, 0),
            (0, 0, -0.002, 0),
            (0, 0, 0, 0.1),
        ):
            rig = self.rig(response)
            q, qd = rig.state_0.body_q.numpy(), rig.state_0.body_qd.numpy()
            q[1, 2] += delta_length
            qd[1, 2] += delta_rate
            q[0, 3:7] = [0, math.sin(delta_angle / 2), 0, math.cos(delta_angle / 2)]
            qd[0, 4] += delta_omega
            rig.state_0.body_q.assign(q)
            rig.state_0.body_qd.assign(qd)
            rig.step([[0, 0]])
            trace = rig.trace()
            self.assertAlmostEqual(
                trace["leg_raw_force_n"][0], 100 - 6000 * delta_length - 60 * delta_rate, delta=0.002
            )
            self.assertAlmostEqual(
                trace["ankle_raw_torque_n_m"][0], 5 - 2000 * delta_angle - 4 * delta_omega, delta=0.001
            )
            self.assertEqual(trace["leg_force_limited"][0], 0)
            self.assertEqual(trace["ankle_torque_limited"][0], 0)

    def test_reference_and_checkpoint_identity_remain_separate(self):
        """Keep response settings out of the unchanged physical config fingerprint."""
        before = self.reference.to_dict()
        identity = self.reference.identity
        baseline = self.rig()
        changed = self.rig(ResponseConfig(controller_mode="intent", leg_stiffness_n_m=3000))
        self.assertEqual(baseline.input_fingerprints, changed.input_fingerprints)
        self.assertNotEqual(baseline.response_fingerprints, changed.response_fingerprints)
        self.assertEqual(changed.metadata["response_fingerprints"], changed.response_fingerprints)
        _rollout(changed)
        changed.reset()
        self.assertEqual(self.reference.to_dict(), before)
        self.assertEqual(self.reference.identity, identity)
        legacy = _reference(self.artifact)
        self.rig(reference=legacy)
        with self.assertRaisesRegex(ValueError, "leg_measured_length_m"):
            self.rig(ResponseConfig(controller_mode="intent"), reference=legacy)
        for length in (0, -1):
            provenance = copy.deepcopy(self.reference.provenance)
            provenance["inverse_dynamics"]["leg_measured_length_m"][0] = length
            with self.assertRaisesRegex(ValueError, "must be positive"):
                self.rig(
                    ResponseConfig(controller_mode="intent"), reference=replace(self.reference, provenance=provenance)
                )

    def test_intent_ledger_with_moving_targets_and_both_clamps(self):
        """Close signed power ledgers with nominal assistance and moving targets."""
        for sign in (-1, 1):
            t = self.reference.time_s
            provenance = copy.deepcopy(self.reference.provenance)
            provenance["inverse_dynamics"].update(
                {
                    "leg_measured_length_m": (1 + sign * (0.02 + 0.2 * t + t * t)).tolist(),
                    "leg_measured_rate_m_s": (sign * (0.2 + 2 * t)).tolist(),
                }
            )
            reference = replace(
                self.reference,
                provenance=provenance,
                pitch_rad=sign * (0.03 + 0.4 * t + t * t),
                pitch_rate_rad_s=sign * (0.4 + 2 * t),
                inverse_leg_force_n=sign * (100 + 50 * t),
                inverse_ankle_torque_n_m=sign * (5 + t),
            )
            rig = self.rig(
                ResponseConfig(controller_mode="intent"),
                reference=reference,
                config=replace(self.config, force_limit_bw=0.01, ankle_torque_limit_n_m=0.1),
            )
            _rollout(rig)
            trace = rig.trace()
            self.assertTrue(np.any(trace["leg_force_limited"]))
            self.assertTrue(np.any(trace["ankle_torque_limited"]))
            np.testing.assert_allclose(
                trace["movement_leg_length_m"],
                1 + sign * (0.02 + 0.2 * trace["time_s"] + trace["time_s"] ** 2),
                atol=3e-7,
            )
            np.testing.assert_allclose(trace["movement_leg_rate_m_s"], sign * (0.2 + 2 * trace["time_s"]), atol=5e-5)
            for actuator in ("leg", "ankle"):
                lhs = trace[f"{actuator}_body_power_w"] + trace[f"{actuator}_spring_energy_rate_w"]
                rhs = trace[f"{actuator}_source_power_w"] + trace[f"{actuator}_damping_power_w"]
                np.testing.assert_allclose(lhs, rhs, rtol=2e-5, atol=2e-3)
                split = sum(
                    trace[f"{actuator}_{part}_power_w"]
                    for part in ("nominal_body", "target_motion", "stiffness_source", "limit")
                )
                np.testing.assert_allclose(split, trace[f"{actuator}_source_power_w"], rtol=2e-5, atol=2e-4)
            np.testing.assert_allclose(
                trace["leg_spring_energy_j"],
                0.5 * trace["leg_stiffness_n_m"] * (trace["leg_length_m"] - trace["movement_leg_length_m"]) ** 2,
                rtol=2e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                trace["leg_nominal_body_power_w"],
                trace["leg_nominal_force_n"] * trace["leg_rate_m_s"],
                rtol=2e-6,
                atol=1e-5,
            )

    def test_push_analytic_impulse_and_upper_body_momentum(self):
        """Deliver the exact supported pulse impulse and separate its body power."""
        response = ResponseConfig(
            controller_mode="intent",
            push_start_s=0.01013,
            push_duration_s=0.01021,
            push_force_x_n=20,
            push_force_z_n=-10,
        )
        force = response.push_interval_averages(duration_s=self.reference.duration_s, sample_count=48)
        expected_impulse = np.array([20, -10]) * response.push_duration_s / 2
        np.testing.assert_allclose(force.sum(axis=0) * self.reference.duration_s / 48, expected_impulse, rtol=1e-14)
        pushed, baseline = self.rig(response), self.rig(replace(response, push_force_x_n=0, push_force_z_n=0))
        _rollout(pushed)
        _rollout(baseline)
        trace, original = pushed.trace(), baseline.trace()
        np.testing.assert_allclose(
            np.array([trace["push_force_x_n"].sum(), trace["push_force_z_n"].sum()]) * pushed.sim_dt,
            expected_impulse,
            rtol=1e-7,
        )
        before = trace["time_s"] + pushed.sim_dt < response.push_start_s
        for name in ("pelvis_x_m", "pelvis_z_m", "pitch_rad", "leg_force_n"):
            np.testing.assert_array_equal(trace[name][before], original[name][before])
        qd = pushed.state_0.body_qd.numpy()
        initial = pushed._initial_qd
        momentum_change = (2 * (qd[0, :3] - initial[0, :3]) + 8 * (qd[1, :3] - initial[1, :3]))[[0, 2]]
        expected = expected_impulse + np.array([0, -pushed.mass * pushed.gravity * pushed.duration])
        np.testing.assert_allclose(momentum_change, expected, rtol=2e-5, atol=3e-5)
        self.assertFalse(np.any(trace["shoe_fz_n"]))
        np.testing.assert_allclose(
            trace["push_power_w"],
            trace["push_force_x_n"] * trace["pelvis_vx_m_s"] + trace["push_force_z_n"] * trace["pelvis_vz_m_s"],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertAlmostEqual(trace["push_work_j"][-1], trace["push_power_w"].sum() * pushed.sim_dt)

    def test_ground_contact_clearance_observations_and_reset(self):
        """Use the same unchanged static plane for contact and both clearance paths."""
        for all_driven in (False, True):
            if all_driven:
                artifact = _artifact()
                fixture = artifact["instron_fixtures"]["fullfoot_last"]
                fixture.update(
                    {
                        "carrier_anchor_m": artifact["column_bed"]["anchor_bottom_m"],
                        "foam_free_top_m": [0.02] * 3,
                        "foam_bottom_m": [0] * 3,
                        "rest_length_m": [0.02] * 3,
                        "area_m2": [0.0004] * 3,
                        "neighbors": artifact["column_bed"]["neighbors"],
                    }
                )
                self.artifact.write_text(json.dumps(artifact))
            reference = replace(_reference(self.artifact, foot_height=0.11), pitch_rad=np.zeros(4))
            ground = 0.01
            raised = self.rig(ResponseConfig(ground_height_m=ground), reference=reference)
            flat = self.rig(reference=reference)
            np.testing.assert_array_equal(raised.state_0.body_q.numpy(), flat.state_0.body_q.numpy())
            np.testing.assert_array_equal(raised.foundation.z_free.numpy(), np.float32(ground))
            self.assertEqual(raised.foundation.ground_height_m, ground)
            self.assertEqual(raised.contact_config.ground_height_m, ground)
            initial = raised.reset()
            difference = (initial - flat.reset()) * raised.observation_scale
            expected = np.zeros_like(difference)
            expected[:, [0, 1, 14]] = -ground
            np.testing.assert_allclose(difference, expected, atol=1e-7)
            _rollout(raised)
            _rollout(flat)
            trace = raised.trace()
            self.assertGreater(trace["shoe_fz_n"][0], flat.trace()["shoe_fz_n"][0])
            local = raised.shoe.visual_mesh("fullfoot_last").vertices_m - raised.ankle_mount
            expected_clearance = (
                trace["foot_z_m"]
                - ground
                + np.min(
                    -np.sin(trace["pitch_rad"][:, None]) * local[:, 0]
                    + np.cos(trace["pitch_rad"][:, None]) * local[:, 2],
                    axis=1,
                )
            )
            np.testing.assert_allclose(trace["last_clearance_m"], expected_clearance, atol=3e-8)
            np.testing.assert_array_equal(raised.reset(), initial)
            np.testing.assert_array_equal(raised.foundation.z_free.numpy(), np.float32(ground))
            _rollout(raised)
            for name, value in trace.items():
                np.testing.assert_array_equal(value, raised.trace()[name], err_msg=name)
            intersecting = self.rig(ResponseConfig(ground_height_m=0.04), reference=reference)
            self.assertTrue(intersecting._flags_device.numpy()[0] & 2)
            intersecting.step([[0, 0]])
            self.assertLess(intersecting.trace()["last_clearance_m"][0], self.config.minimum_last_clearance_m)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_graph_frame_index_reset_and_batch(self):
        """Replay pushes and fixed intent at correct indices with odd/even frames."""
        response = ResponseConfig(
            controller_mode="intent",
            ground_height_m=0.001,
            push_start_s=0.011,
            push_duration_s=0.010,
            push_force_x_n=20,
        )
        for substeps in (15, 16):
            config = replace(self.config, substeps=substeps)
            eager = self.rig(response, config=config, worlds=2, device="cuda:0")
            graph = self.rig(response, config=replace(config, use_graph=True), worlds=2, device="cuda:0")
            _rollout(eager)
            _rollout(graph)
            self.assertEqual(graph.graph_status, "captured")
            for world in range(2):
                for name, value in eager.trace(world).items():
                    np.testing.assert_allclose(value, graph.trace(world)[name], rtol=1e-6, atol=1e-6, err_msg=name)
            first = graph.trace()
            graph.reset()
            _rollout(graph)
            for name, value in first.items():
                np.testing.assert_allclose(value, graph.trace()[name], rtol=1e-6, atol=1e-6, err_msg=name)
            expected = response.push_force_x_n * response.push_duration_s / 2
            self.assertAlmostEqual(graph.trace()["push_force_x_n"].sum() * graph.sim_dt, expected, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
