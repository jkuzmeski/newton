# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the standalone two-stiffness rig without archived experiment imports."""

import ast
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe import load_artifact
from projects.impedance_instron.orientation import orient_shoe
from projects.impedance_instron.simple.reference import Reference, geometry_identity, tracking_error
from projects.impedance_instron.simple.rig import Rig, RigConfig


def _artifact():
    """Build a small independent shoe with a passive outer column."""
    anchors = [[-0.04, 0, 0], [0, 0, 0], [0.04, 0, 0]]
    return {
        "schema_version": "digital_shoe_1",
        "shoe": {"id": "simple_test_shoe"},
        "coordinate_system": {"handedness": "right", "up_axis": "+Z", "length_unit": "m", "force_unit": "N"},
        "constitutive_model": {
            "type": "effective_hyperfoam_maxwell_pasternak_foundation",
            "parameters": {
                "instantaneous_shear_modulus_pa": 19000.0,
                "hyperfoam_exponent": 5.1,
                "equilibrium_fraction": 0.11,
                "pasternak_n_per_m": 900.0,
                "effective_poisson_ratio": 0.3,
                "maxwell_relaxation_time_s": 0.08,
            },
        },
        "column_bed": {
            "anchor_bottom_m": anchors,
            "rest_length_m": [0.02] * 3,
            "area_m2": [0.0004] * 3,
            "neighbors": [[-1, 1, -1, -1], [0, 2, -1, -1], [1, -1, -1, -1]],
            "spacing_m": 0.04,
        },
        "visual_meshes": {
            "fullfoot_last": {
                "vertices_m": [[-0.05, -0.02, 0.03], [0.05, -0.02, 0.03], [0.05, 0.02, 0.03], [-0.05, 0.02, 0.03]],
                "triangles": [[0, 1, 2], [0, 2, 3]],
            }
        },
        "instron_fixtures": {
            "fullfoot_last": {
                "carrier_anchor_m": anchors[:2],
                "foam_free_top_m": [0.02, 0.02],
                "foam_bottom_m": [0, 0],
                "rest_length_m": [0.02, 0.02],
                "area_m2": [0.0004, 0.0004],
                "neighbors": [[-1, 1, -1, -1], [0, -1, -1, -1]],
                "spacing_m": 0.04,
            }
        },
        "validation": {},
        "provenance": {},
    }


def _reference(artifact_path, *, duration=0.025, foot_height=0.4, length=1.01, equilibrium_angle=0.18):
    """Freeze a synthetic reference independently of the controller implementation."""
    shoe, _ = orient_shoe(load_artifact(artifact_path), target_side="left", source_side="right")
    t = np.linspace(0, duration, 4)
    zero = np.zeros(4)
    config = {
        "foot_mass_kg": 2.0,
        "pitch_inertia_kg_m2": 0.025,
        "ankle_mount_local_m": [-0.075, 0, 0.105],
        "nominal_leg_stiffness_n_m": 12000.0,
        "nominal_ankle_stiffness_n_m_rad": 4000.0,
        "leg_damping_ratio": 0.25,
        "ankle_damping_ratio": 0.5,
        "source_side": "right",
        "target_side": "left",
    }
    return Reference(
        time_s=t,
        pelvis_z_m=np.full(4, foot_height + 1),
        pelvis_vz_m_s=zero,
        pitch_rad=np.full(4, 0.15),
        pitch_rate_rad_s=zero,
        leg_length_m=np.full(4, length),
        leg_rate_m_s=zero,
        ankle_equilibrium_rad=np.full(4, equilibrium_angle),
        ankle_equilibrium_rate_rad_s=zero,
        inverse_leg_force_n=zero,
        inverse_ankle_torque_n_m=zero,
        reference_fz_n=np.full(4, 1234.0),
        reference_fx_n=np.full(4, 345.0),
        foot_position_m=np.array([0, 0, foot_height]),
        upper_position_m=np.array([0, 0, foot_height + 1]),
        foot_velocity_m_s=np.array([0.2, 0, 0]),
        upper_velocity_m_s=np.array([0.2, 0, 0]),
        mass_kg=10,
        gravity_m_s2=9.80665,
        contact_start_s=0.002,
        contact_duration_s=duration - 0.004,
        pelvis_scale_m=0.01,
        pitch_scale_rad=0.05,
        provenance={"config": config, "geometry_identity": geometry_identity(shoe)},
    )


def _rollout(rig, action=None):
    """Collect an entire episode with a constant bounded action."""
    if action is None:
        action = np.zeros((rig.num_worlds, 2))
    rewards = np.zeros(rig.num_worlds)
    for _ in range(rig.episode_frames):
        _, reward, done, info = rig.step(action)
        rewards += reward
    return rewards, done, info


class TestSimpleRig(unittest.TestCase):
    """Exercise mechanics, public contracts and physical accounting on the CPU."""

    def setUp(self):
        """Create a portable tiny shoe and frozen reference for each check."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.artifact = Path(self.directory.name) / "shoe.json"
        self.artifact.write_text(json.dumps(_artifact()))
        self.reference = _reference(self.artifact)
        self.config = RigConfig(substeps=16, use_graph=False)

    def rig(self, reference=None, config=None, worlds=1, device="cpu"):
        """Construct a test rig without source preparation or external input files."""
        return Rig(reference or self.reference, self.artifact, config or self.config, num_worlds=worlds, device=device)

    def test_config_and_actions_fail_closed(self):
        """Reject invalid settings and every legacy or malformed action shape."""
        restored = RigConfig.from_dict(json.loads(json.dumps(self.config.to_dict())))
        self.assertEqual(restored, self.config)
        for kwargs in (
            {"substeps": 0},
            {"frame_rate_hz": float("nan")},
            {"log_stiffness_slew_s": -1},
            {"leg_stiffness_max_n_m": 1},
            {"ankle_mount_m": [0, 0, float("inf")]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RigConfig(**kwargs)
        with self.assertRaises(TypeError):
            RigConfig.from_dict({"action_mode": "equilibrium"})
        rig = self.rig()
        self.assertEqual(rig.action_dim, 2)
        for action in ([0, 0], np.zeros((1, 3)), [[0, float("nan")]], [[1.01, 0]]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                rig.step(action)
        with self.assertRaisesRegex(ValueError, "frozen reference"):
            self.rig(config=replace(self.config, foot_mass_kg=3))

    def test_fixed_clock_and_pure_full_episode_reward(self):
        """Keep flight error and use upper-body height on the one unshifted clock."""
        rig = self.rig()
        rewards, done, info = _rollout(rig)
        trace = rig.trace()
        self.assertTrue(done)
        self.assertFalse(info["safety_ok"][0])
        self.assertIn("no_detected_contact", info["safety_reasons"][0])
        self.assertGreater(-rewards[0], 0)
        np.testing.assert_array_equal(trace["ref_time_s"], trace["time_s"])
        np.testing.assert_array_equal(trace["ref_clock_rate"], np.ones(rig.sample_count))
        ref = self.reference.sample(trace["time_s"])
        error = tracking_error(
            trace["pelvis_z_m"],
            trace["pitch_rad"],
            ref["pelvis_z_m"],
            ref["pitch_rad"],
            self.reference.pelvis_scale_m,
            self.reference.pitch_scale_rad,
        )
        np.testing.assert_allclose(trace["tracking_error"], error, rtol=2e-5, atol=2e-6)
        self.assertAlmostEqual(float(info["tracking_loss"][0]), float(error.sum() * rig.sim_dt), delta=2e-7)
        np.testing.assert_array_equal(info["tracking_loss"], -rewards)
        self.assertTrue(np.all(trace["shoe_fz_n"] == 0))
        # Measured evaluation GRF is deliberately huge, but cannot load either body.
        qd = rig.state_0.body_qd.numpy()
        total_vertical_momentum = 2 * qd[0, 2] + 8 * qd[1, 2]
        self.assertAlmostEqual(total_vertical_momentum, -10 * 9.80665 * self.reference.duration_s, delta=2e-5)
        with self.assertRaises(RuntimeError):
            rig.step([[0, 0]])

    def test_signed_actuator_ledger_including_both_clamps(self):
        """Close both actuator ledgers for signed forces and explicit limit work."""
        for sign in (-1, 1):
            reference = _reference(self.artifact, length=1 + sign * 0.05, equilibrium_angle=0.15 + sign * 0.2)
            reference = replace(
                reference,
                leg_length_m=reference.leg_length_m + sign * 0.2 * reference.time_s,
                leg_rate_m_s=np.full(4, sign * 0.2),
                ankle_equilibrium_rad=reference.ankle_equilibrium_rad + sign * 2 * reference.time_s,
                ankle_equilibrium_rate_rad_s=np.full(4, sign * 2.0),
            )
            rig = self.rig(
                reference=reference, config=replace(self.config, force_limit_bw=0.01, ankle_torque_limit_n_m=0.1)
            )
            _, _, info = _rollout(rig, np.array([[0.8, -0.6]]))
            trace = rig.trace()
            self.assertIn("leg_force_limit", info["safety_reasons"][0])
            self.assertIn("ankle_torque_limit", info["safety_reasons"][0])
            self.assertEqual(np.sign(trace["leg_force_n"][0]), sign)
            self.assertEqual(np.sign(trace["ankle_torque_n_m"][0]), sign)
            for name in ("leg", "ankle"):
                lhs = trace[f"{name}_body_power_w"] + trace[f"{name}_spring_energy_rate_w"]
                rhs = trace[f"{name}_source_power_w"] + trace[f"{name}_damping_power_w"]
                np.testing.assert_allclose(lhs, rhs, rtol=2e-5, atol=1e-3)
                self.assertTrue(np.all(trace[f"{name}_damping_power_w"] <= 0))
                self.assertAlmostEqual(
                    trace[f"{name}_source_work_j"][-1], trace[f"{name}_source_power_w"].sum() * rig.sim_dt, delta=1e-8
                )
            np.testing.assert_allclose(
                trace["leg_damping_n_s_m"],
                2 * self.config.leg_damping_ratio * np.sqrt(trace["leg_stiffness_n_m"] * 8),
                rtol=2e-6,
            )
            np.testing.assert_allclose(
                trace["ankle_damping_n_m_s_rad"],
                2 * self.config.ankle_damping_ratio * np.sqrt(trace["ankle_stiffness_n_m_rad"] * 0.025),
                rtol=2e-6,
            )

    def test_absolute_log_stiffness_ramps_and_analytic_derivatives(self):
        """Bound instantaneous log-rate and start every frame without rate jumps."""
        rig = self.rig()
        _rollout(rig, np.array([[1, -1]]))
        trace = rig.trace()
        for channel, lo, hi in (("leg", 3000, 48000), ("ankle", 1000, 16000)):
            key = "leg_stiffness_n_m" if channel == "leg" else "ankle_stiffness_n_m_rad"
            rate_key = "leg_stiffness_rate_n_m_s" if channel == "leg" else "ankle_stiffness_rate_n_m_rad_s"
            k, kdot = trace[key], trace[rate_key]
            self.assertTrue(np.all((k >= lo - 1e-3) & (k <= hi + 1e-3)))
            self.assertLessEqual(float(np.max(np.abs(kdot / k))), self.config.log_stiffness_slew_s + 1e-4)
            np.testing.assert_allclose(kdot[:: self.config.substeps], 0, atol=1e-7)
            delta = self.config.log_stiffness_slew_s * rig.frame_dt / 1.875
            direction = 1 if channel == "leg" else -1
            expected = math.sqrt(lo * hi) * math.exp(direction * delta)
            self.assertAlmostEqual(k[self.config.substeps], expected, delta=0.03)
        np.testing.assert_allclose(trace["leg_equilibrium_rate_m_s"], 0, atol=3e-5)

    def test_constant_absolute_action_does_not_accumulate(self):
        """Reach one absolute target and hold it instead of adding action each frame."""
        rig = self.rig(config=replace(self.config, log_stiffness_slew_s=10000))
        _rollout(rig, np.array([[0.5, -0.5]]))
        trace = rig.trace()
        expected_leg = np.exp(np.log(3000) + 0.75 * np.log(16))
        expected_ankle = np.exp(np.log(1000) + 0.25 * np.log(16))
        np.testing.assert_allclose(trace["leg_stiffness_n_m"][self.config.substeps :], expected_leg, rtol=2e-6)
        np.testing.assert_allclose(trace["ankle_stiffness_n_m_rad"][self.config.substeps :], expected_ankle, rtol=2e-6)
        np.testing.assert_array_equal(trace["leg_stiffness_rate_n_m_s"][self.config.substeps :], 0)

    def test_unclamped_moving_equilibrium_laws(self):
        """Use analytic moving-equilibrium rates in signed forces and both energy ledgers."""
        t = self.reference.time_s
        reference = replace(
            self.reference,
            leg_length_m=1 + 0.2 * t + t * t,
            leg_rate_m_s=0.2 + 2 * t,
            ankle_equilibrium_rad=0.15 + 0.5 * t + 2 * t * t,
            ankle_equilibrium_rate_rad_s=0.5 + 4 * t,
        )
        rig = self.rig(reference=reference)
        _rollout(rig, np.array([[0.6, -0.6]]))
        trace = rig.trace()
        samples = reference.sample(trace["ref_time_s"])
        np.testing.assert_allclose(trace["leg_equilibrium_m"], samples["leg_length_m"], atol=2e-7)
        np.testing.assert_allclose(trace["leg_equilibrium_rate_m_s"], samples["leg_rate_m_s"], atol=3e-5)
        np.testing.assert_allclose(
            trace["ankle_equilibrium_rate_rad_s"], samples["ankle_equilibrium_rate_rad_s"], atol=1e-5
        )
        for actuator, position, target, rate, target_rate, k_key, kdot_key, b_key, force_key in (
            (
                "leg",
                "leg_length_m",
                "leg_equilibrium_m",
                "leg_rate_m_s",
                "leg_equilibrium_rate_m_s",
                "leg_stiffness_n_m",
                "leg_stiffness_rate_n_m_s",
                "leg_damping_n_s_m",
                "leg_force_n",
            ),
            (
                "ankle",
                "pitch_rad",
                "ankle_equilibrium_rad",
                "pitch_rate_rad_s",
                "ankle_equilibrium_rate_rad_s",
                "ankle_stiffness_n_m_rad",
                "ankle_stiffness_rate_n_m_rad_s",
                "ankle_damping_n_m_s_rad",
                "ankle_torque_n_m",
            ),
        ):
            error = trace[position] - trace[target]
            slip = trace[rate] - trace[target_rate]
            k, b, kdot = trace[k_key], trace[b_key], trace[kdot_key]
            np.testing.assert_allclose(trace[force_key], -k * error - b * slip, rtol=2e-5, atol=2e-3)
            stored_rate = 0.5 * kdot * error**2 + k * error * slip
            expected_source = trace[force_key] * trace[rate] + stored_rate + b * slip**2
            np.testing.assert_allclose(trace[f"{actuator}_source_power_w"], expected_source, rtol=3e-5, atol=2e-3)

    def test_clearance_uses_achieved_pitch_not_target(self):
        """Compute last clearance from achieved pitch and the untouched local mesh."""
        rig = self.rig()
        _rollout(rig)
        trace = rig.trace()
        local = rig.shoe.visual_mesh("fullfoot_last").vertices_m - rig.ankle_mount
        angles = trace["pitch_rad"]
        expected = trace["foot_z_m"] + np.min(
            -np.sin(angles[:, None]) * local[None, :, 0] + np.cos(angles[:, None]) * local[None, :, 2], axis=1
        )
        np.testing.assert_allclose(trace["last_clearance_m"], expected, atol=5e-8)
        self.assertGreater(np.max(np.abs(trace["pitch_rad"] - trace["reference_pitch_rad"])), 0.005)
        self.assertEqual(rig.foundation.free_column_count, 1)
        self.assertEqual(rig.surround_config.attachment_n_m, 0)
        self.assertTrue(rig.surround_config.carrier_bond)
        self.assertEqual(rig.contact_config.normal_damping, 0)

    def test_reset_restores_all_foam_and_stiffness_states(self):
        """Repeat a contact-loaded rollout after resetting foam and action history."""
        reference = _reference(self.artifact, foot_height=0.11)
        rig = self.rig(reference=reference)
        initial = rig.reset()
        _rollout(rig, np.array([[0.3, -0.3]]))
        first = rig.trace()
        self.assertGreater(first["shoe_fz_n"].max(), 0)
        np.testing.assert_array_equal(first["time_s"], first["ref_time_s"])
        np.testing.assert_array_equal(rig.reset(), initial)
        for name in (
            "q_state",
            "peq_prev",
            "tangent_anchor",
            "tangent_stuck",
            "surround_compression",
            "surround_rate",
            "compression",
            "column_force",
            "resultant_force",
        ):
            self.assertFalse(np.any(getattr(rig.foundation, name).numpy()))
        _rollout(rig, np.array([[0.3, -0.3]]))
        second = rig.trace()
        for name in first:
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)

    def test_material_only_override_preserves_geometry_and_initial_state(self):
        """Allow only material substitution without re-registration or new schedules."""
        first = self.rig()
        data = _artifact()
        data["constitutive_model"]["parameters"]["instantaneous_shear_modulus_pa"] *= 2
        self.artifact.write_text(json.dumps(data))
        second = self.rig()
        self.assertEqual(first.input_fingerprints["geometry_identity"], second.input_fingerprints["geometry_identity"])
        self.assertNotEqual(
            first.input_fingerprints["material_identity"], second.input_fingerprints["material_identity"]
        )
        np.testing.assert_array_equal(first.state_0.body_q.numpy(), second.state_0.body_q.numpy())
        np.testing.assert_array_equal(first.state_0.body_qd.numpy(), second.state_0.body_qd.numpy())
        self.assertIs(first.reference, second.reference)
        data["column_bed"]["rest_length_m"][0] += 0.001
        self.artifact.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.rig()

    def test_batch_matches_independent_worlds(self):
        """Keep each world's foam and body forces independent in a native batch."""
        actions = np.array([[0.2, -0.3], [-0.7, 0.8]])
        batched = self.rig(worlds=2)
        _rollout(batched, actions)
        for world in range(2):
            single = self.rig()
            _rollout(single, actions[world : world + 1])
            for name, value in single.trace().items():
                np.testing.assert_array_equal(value, batched.trace(world)[name], err_msg=name)

    def test_nonfinite_physics_cannot_be_a_valid_low_loss(self):
        """Disqualify numerical failures while keeping observations policy-callable."""
        rig = self.rig()
        q = rig.state_0.body_q.numpy()
        q[1, 2] = np.nan
        rig.state_0.body_q.assign(q)
        observation, reward, _, info = rig.step([[0, 0]])
        self.assertTrue(np.isfinite(observation).all())
        self.assertFalse(info["safety_ok"][0])
        self.assertTrue(np.isinf(info["tracking_loss"][0]))
        self.assertTrue(np.isneginf(reward[0]))
        self.assertIn("nonfinite_state_or_force", info["safety_reasons"][0])

    def test_no_archived_experiment_dependency(self):
        """Keep the active rig independent of every archived controller and mode."""
        source = Path("projects/impedance_instron/simple/rig.py").read_text()
        tree = ast.parse(source)
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        for name in ("env", "example", "control", "train", "objective", "optimize"):
            self.assertFalse(any(module == name or module.endswith(f"impedance_instron.{name}") for module in imports))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_cuda_graph_matches_eager_and_reset_for_odd_substeps(self):
        """Replay graph-bound odd ping-pong frames without stale states or indices."""
        config = replace(self.config, substeps=15)
        eager = self.rig(config=config, worlds=2, device="cuda:0")
        graph = self.rig(config=replace(config, use_graph=True), worlds=2, device="cuda:0")
        action = np.array([[0.2, -0.3], [-0.7, 0.8]])
        _rollout(eager, action)
        _rollout(graph, action)
        self.assertEqual(graph.graph_status, "captured")
        for world in range(2):
            for name, value in eager.trace(world).items():
                np.testing.assert_array_equal(value, graph.trace(world)[name], err_msg=name)
        before = graph.trace()
        graph.reset()
        _rollout(graph, action)
        for name, value in before.items():
            np.testing.assert_array_equal(value, graph.trace()[name], err_msg=name)


if __name__ == "__main__":
    unittest.main()
