# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise two-output learning and frozen checkpoint rules with a tiny rig."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.simple import policy


@dataclass
class ToyConfig:
    """Expose nondefault mechanics to detect incomplete checkpoint restoration."""

    frame_dt: float = 0.1
    frames: int = 3
    height_error_m: float = 0.2
    pitch_error_rad: float = 0.1
    force_n: float = 10.0
    damping_ratio: float = 0.73
    nominal_leg_stiffness: float = 14567.0
    valid: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class ToyReference:
    """Embed optical targets and all four inverse-dynamics equilibrium arrays."""

    def __init__(self, data=None):
        self.data = (
            copy.deepcopy(data)
            if data is not None
            else {
                "time_s": [0.0, 0.1, 0.2, 0.3],
                "pelvis_z_m": [1.0, 1.01, 1.02, 1.03],
                "pitch_rad": [0.0, 0.01, 0.02, 0.03],
                "leg_length_m": [0.91, 0.92, 0.93, 0.94],
                "leg_rate_m_s": [0.1, 0.1, 0.1, 0.1],
                "ankle_equilibrium_rad": [0.5, 0.6, 0.7, 0.8],
                "ankle_equilibrium_rate_rad_s": [1.0, 1.0, 1.0, 1.0],
                "inverse_leg_force_n": [123.0, 124.0, 125.0, 126.0],
                "inverse_ankle_torque_n_m": [12.0, 13.0, 14.0, 15.0],
                "pelvis_scale_m": 0.1,
                "pitch_scale_rad": 0.2,
                "provenance": {"optical_source_sha256": "fixture", "processing": {"smooth_s": 0.04}},
            }
        )

    @property
    def identity(self):
        return policy._json_hash(self.data)

    def to_dict(self):
        return copy.deepcopy(self.data)

    @classmethod
    def from_dict(cls, data):
        return cls(data)


class ToyRig:
    """Provide measured motion, irrelevant force/work, and two-action feedback."""

    action_dim = 2
    source_hash = "toy_runtime_v1"
    observation_dim = 4
    observation_names = ("pelvis", "pitch", "force", "time")
    observation_scale = np.array([0.1, 0.2, 700.0, 0.3])

    def __init__(self, reference, artifact_path, config=None, num_worlds=1, device=None):
        self.reference = reference
        self.config = config or ToyConfig()
        self.artifact_path = Path(artifact_path)
        self.num_worlds = num_worlds
        self.device = device
        self.episode_frames = self.config.frames
        self.frame_dt = self.config.frame_dt
        artifact = policy.artifact_identity(artifact_path)
        self.input_fingerprints = {
            "artifact_sha256": artifact["sha256"],
            "geometry_identity": artifact["geometry_sha256"],
            "material_identity": artifact["material_sha256"],
            "reference_identity": reference.identity,
            "config_identity": policy._json_hash(self.config.to_dict()),
            "rig_source_sha256": self.source_hash,
        }
        self.reset()

    def reset(self):
        self.frame = 0
        self.total = np.zeros(self.num_worlds)
        self.actions = []
        self.errors = []
        return np.zeros((self.num_worlds, self.observation_dim), dtype=np.float32)

    def step(self, action):
        if np.asarray(action).shape != (self.num_worlds, 2):
            raise ValueError("two actions required")
        self.actions.append(np.asarray(action).copy())
        self.frame += 1
        height_error = self.config.height_error_m + 0.01 * action[:, 0]
        pitch_error = self.config.pitch_error_rad + 0.01 * action[:, 1]
        scale_z = self.reference.data["pelvis_scale_m"]
        scale_pitch = self.reference.data["pitch_scale_rad"]
        error = 0.5 * ((height_error / scale_z) ** 2 + (pitch_error / scale_pitch) ** 2)
        self.errors.append(error)
        reward = -error * self.frame_dt
        self.total += reward
        obs = np.tile(
            [height_error[0] / scale_z, pitch_error[0] / scale_pitch, self.config.force_n / 700, self.frame / 3],
            (self.num_worlds, 1),
        ).astype(np.float32)
        return (
            obs,
            reward,
            self.frame == self.episode_frames,
            {
                "tracking_loss": -self.total,
                "safety_ok": np.full(self.num_worlds, self.config.valid),
                "safety_reasons": [
                    [] if self.config.valid else ["toy physical failure"] for _ in range(self.num_worlds)
                ],
                "work_j": np.full(self.num_worlds, self.config.force_n * 1000.0),
                "shoe_fz_n": np.full(self.num_worlds, self.config.force_n),
            },
        )

    def trace(self, world=0):
        return {
            "time_s": np.arange(1, self.frame + 1) * self.frame_dt,
            "tracking_error": np.asarray(self.errors)[:, world],
            "actions": np.asarray(self.actions)[:, world],
            "shoe_fz_n": np.full(self.frame, self.config.force_n),
            "pelvis_z_m": np.full(self.frame, 1.0 + self.config.height_error_m),
        }


def _artifact():
    return {
        "schema_version": "digital_shoe_1",
        "shoe": {"id": "toy"},
        "coordinate_system": {"up_axis": "+Z", "length_unit": "m"},
        "column_bed": {"anchor_bottom_m": [[0, 0, 0]], "rest_length_m": [0.03]},
        "visual_meshes": {"test": {"vertices_m": [[0, 0, 0]], "triangles": []}},
        "constitutive_model": {"type": "test", "parameters": {"modulus": 100.0}},
        "provenance": {},
        "validation": {},
    }


class TestSimplePolicy(unittest.TestCase):
    """Check the motion objective without requiring Torch or native physics."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.artifact_path = self.directory / "shoe.json"
        self.artifact_path.write_text(json.dumps(_artifact()))
        self.reference = ToyReference()
        self.addCleanup(patch.stopall)
        patch.object(policy, "_rig_types", return_value=(ToyRig, ToyConfig)).start()
        patch.object(policy, "_reference_type", return_value=ToyReference).start()

    def test_reward_is_only_measured_motion(self):
        """Change only measured motion to change loss, never force or work."""
        with patch.object(policy, "_torch", side_effect=AssertionError("baseline needs no Torch")):
            base = policy.evaluate_policy(ToyRig(self.reference, self.artifact_path))
            forces = policy.evaluate_policy(ToyRig(self.reference, self.artifact_path, ToyConfig(force_n=1.0e9)))
            heights = policy.evaluate_policy(ToyRig(self.reference, self.artifact_path, ToyConfig(height_error_m=0.4)))
            pitches = policy.evaluate_policy(ToyRig(self.reference, self.artifact_path, ToyConfig(pitch_error_rad=0.4)))
        self.assertEqual(base["tracking_loss_mean"], forces["tracking_loss_mean"])
        self.assertGreater(heights["tracking_loss_mean"], base["tracking_loss_mean"])
        self.assertGreater(pitches["tracking_loss_mean"], base["tracking_loss_mean"])
        self.assertAlmostEqual(base["tracking_loss_mean"], 0.5 * (2.0**2 + 0.5**2) * 0.3, places=6)
        self.assertAlmostEqual(base["tracking_loss_mean"], -base["cumulative_reward"][0])

    def test_numpy_callable_needs_no_torch(self):
        """Evaluate a NumPy baseline without importing the optional learner."""
        rig = ToyRig(self.reference, self.artifact_path, num_worlds=2)
        with patch.object(policy, "_torch", side_effect=AssertionError("no Torch")):
            report = policy.evaluate_policy(rig, lambda obs: np.zeros((len(obs), 2)))
        self.assertEqual(report["valid_world_count"], 2)
        self.assertEqual(rig.trace()["actions"].shape, (3, 2))

    def test_physical_invalid_not_best(self):
        """Reject physically invalid candidates even with perfect motion loss."""
        rig = ToyRig(
            self.reference, self.artifact_path, ToyConfig(height_error_m=0.0, pitch_error_rad=0.0, valid=False)
        )
        invalid = policy.evaluate_policy(rig)
        self.assertEqual(invalid["tracking_loss_mean"], 0.0)
        self.assertFalse(policy.is_better_evaluation(invalid, 100.0))
        self.assertEqual(invalid["status"], "physically_invalid")
        self.assertIn("not assessed", invalid["target_achievement"])

    def test_nonfinite_is_not_zero_loss_success(self):
        """Keep nonfinite episodes invalid and report no fabricated finite loss."""
        rig = ToyRig(self.reference, self.artifact_path)
        rig.config.height_error_m = float("nan")
        report = policy.evaluate_policy(rig)
        self.assertIsNone(report["tracking_loss_mean"])
        self.assertFalse(report["eligible_for_best"])
        self.assertIn("nonfinite rollout", report["safety_reasons"][0])
        json.dumps(report, allow_nan=False)

    def test_incomplete_episode_is_not_best(self):
        """Exclude shortened episodes instead of rewarding a shorter time integral."""
        rig = ToyRig(self.reference, self.artifact_path)
        step = rig.step

        def stop_early(action):
            obs, reward, _, info = step(action)
            return obs, reward, True, info

        rig.step = stop_early
        summary = policy.evaluate_policy(rig)
        self.assertFalse(summary["eligible_for_best"])
        self.assertFalse(summary["numerical_ok"][0])
        self.assertIn("incomplete episode", summary["safety_reasons"][0])

    def test_mismatched_reward_report_rejected(self):
        """Reject rig summaries that rank a loss different from cumulative reward."""
        rig = ToyRig(self.reference, self.artifact_path)
        info = {"tracking_loss": np.array([2.0]), "safety_ok": np.array([True])}
        with self.assertRaisesRegex(ValueError, "differs from negative cumulative"):
            policy._summarize(rig, np.array([-1.0]), np.array([True]), info)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch dependency is not installed")
class TestSimplePolicyCheckpoint(TestSimplePolicy):
    """Exercise tiny PPO updates and complete frozen replay restoration."""

    def test_exactly_two_output_channels(self):
        """Require two bounded actor outputs and two Gaussian noise channels."""
        torch = policy._torch()
        model = policy.create_policy(4, hidden_size=8)
        observations = torch.zeros((7, 4))
        self.assertEqual(model(observations).shape, (7, 2))
        self.assertEqual(model.log_std.shape, (2,))
        self.assertEqual(model.distribution(observations).mean.shape, (7, 2))
        self.assertTrue(torch.all(model(observations).abs() <= 1.0))
        self.assertEqual(policy.PPO_SETTINGS["gamma"], 1.0)
        self.assertEqual(policy.PPO_SETTINGS["entropy_coefficient"], 0.0)

    def _train(self, name="run", **kwargs):
        return policy.train(self.reference, self.artifact_path, self.directory / name, device="cpu", **kwargs)

    def test_checkpoint_restores_all_settings_and_schedules(self):
        """Restore every config field, inverse schedule, scale, and policy weight."""
        config = ToyConfig(frame_dt=0.03, frames=2, damping_ratio=0.82, nominal_leg_stiffness=22222.0)
        trained = self._train(iterations=1, num_worlds=2, config=config)
        rig, report = policy.evaluate(Path(trained["best_checkpoint"]), num_worlds=3, device="cpu")
        self.assertEqual(asdict(rig.config), asdict(config))
        self.assertEqual(rig.reference.to_dict(), self.reference.to_dict())
        self.assertEqual(report["reference_identity"], self.reference.identity)
        self.assertAlmostEqual(report["tracking_loss_mean"], trained["best_tracking_loss"], places=6)
        saved = policy._torch().load(trained["best_checkpoint"], weights_only=True)
        self.assertEqual(saved["observation"]["scale"], ToyRig.observation_scale.tolist())
        self.assertTrue(saved["observation"]["already_scaled"])
        self.assertEqual(saved["artifact"]["path"], str(self.artifact_path))
        self.assertEqual(saved["rig_config"], asdict(config))
        self.assertEqual(saved["reference"]["ankle_equilibrium_rad"], self.reference.data["ankle_equilibrium_rad"])
        self.assertEqual(saved["policy_state"]["actor.4.bias"].shape, (2,))
        history = json.loads((self.directory / "run/history.json").read_text())
        self.assertEqual(len(history), 2)
        trace = np.load(history[0]["traces"][0], allow_pickle=False)
        self.assertEqual(trace["actions"].shape, (2, 2))
        self.assertTrue((self.directory / "run/history.csv").exists())

    def test_default_training_preserves_reference_construction(self):
        """Let the rig resolve nondefault frozen construction values from reference."""

        class ReferenceConfigRig(ToyRig):
            def __init__(self, reference, artifact_path, config=None, num_worlds=1, device=None):
                super().__init__(reference, artifact_path, config or ToyConfig(damping_ratio=0.44), num_worlds, device)

        with patch.object(policy, "_rig_types", return_value=(ReferenceConfigRig, ToyConfig)):
            trained = self._train(iterations=0, num_worlds=1)
            rig, _ = policy.restore(trained["best_checkpoint"])
        self.assertEqual(rig.config.damping_ratio, 0.44)

    def test_live_restore_does_not_advance_and_freezes_weights(self):
        """Restore a live actor for viewer stepping without consuming an episode."""
        trained = self._train(iterations=0, num_worlds=1)
        rig, model = policy.restore(trained["best_checkpoint"], num_worlds=2)
        self.assertEqual(rig.frame, 0)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        obs = rig.reset()
        action = model.act(obs, deterministic=True)
        self.assertEqual(action.shape, (2, 2))
        np.testing.assert_array_equal(action, model.act(obs, deterministic=True))
        rig.step(action)
        self.assertEqual(rig.frame, 1)

    def test_nonfinite_worlds_cannot_update(self):
        """Skip PPO for nonfinite worlds without replacing their loss by zero."""
        original = ToyRig.step

        def nonfinite(rig, action):
            obs, reward, done, info = original(rig, action)
            reward[:] = np.nan
            info["tracking_loss"][:] = np.nan
            return obs, reward, done, info

        with patch.object(ToyRig, "step", nonfinite):
            trained = self._train(iterations=1, num_worlds=2)
        self.assertIsNone(trained["best_checkpoint"])
        updates = json.loads((self.directory / "run/updates.json").read_text())
        self.assertFalse(updates[0]["update_performed"])
        self.assertIsNone(updates[0]["rollout"]["tracking_loss_mean"])

    def test_save_best_not_last_or_invalid(self):
        """Keep an earlier valid best when later or invalid evaluations look worse."""
        original = policy.evaluate_policy
        scores = iter([(4.0, True), (1.0, True), (0.0, False), (3.0, True)])

        def scored(rig, model):
            result = original(rig, model)
            score, valid = next(scores)
            result.update(
                tracking_loss_mean=score,
                tracking_loss=[score],
                cumulative_reward=[-score],
                eligible_for_best=valid,
                safety_ok=[valid],
            )
            return result

        with patch.object(policy, "evaluate_policy", side_effect=scored):
            result = self._train(iterations=3, num_worlds=2, eval_interval=1)
        self.assertEqual(result["best_iteration"], 1)
        self.assertEqual(result["best_tracking_loss"], 1.0)
        torch = policy._torch()
        best = torch.load(result["best_checkpoint"], weights_only=True)
        last = torch.load(result["last_checkpoint"], weights_only=True)
        self.assertEqual(best["iteration"], 1)
        self.assertEqual(last["iteration"], 3)
        self.assertFalse(torch.equal(best["policy_state"]["actor.4.weight"], last["policy_state"]["actor.4.weight"]))

    def test_no_valid_policy_does_not_make_best(self):
        """Save a clearly separate last checkpoint when all physical runs fail."""
        result = self._train(iterations=1, num_worlds=2, config=ToyConfig(valid=False))
        self.assertIsNone(result["best_checkpoint"])
        self.assertFalse((self.directory / "run/best.pt").exists())
        self.assertEqual(result["status"], "no_physically_valid_policy")
        updates = json.loads((self.directory / "run/updates.json").read_text())
        self.assertTrue(updates[0]["update_performed"])
        self.assertEqual(updates[0]["training_finite_worlds"], 2)
        self.assertEqual(updates[0]["physically_valid_worlds"], 0)

    def test_material_override_only_and_default_hash(self):
        """Permit explicit material swaps but reject changed default or geometry."""
        result = self._train(iterations=0, num_worlds=1)
        original = _artifact()
        changed = copy.deepcopy(original)
        changed["constitutive_model"]["parameters"]["modulus"] = 200.0
        self.artifact_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "Original artifact changed"):
            policy.evaluate(result["best_checkpoint"])
        rig, report = policy.evaluate(result["best_checkpoint"], artifact_path=self.artifact_path)
        self.assertTrue(report["material_override"])
        self.assertEqual(rig.reference.to_dict(), self.reference.to_dict())
        changed["column_bed"]["rest_length_m"] = [0.04]
        self.artifact_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "cannot change shoe geometry"):
            policy.evaluate(result["best_checkpoint"], artifact_path=self.artifact_path)

    def test_material_override_cannot_bypass_runtime_fingerprint(self):
        """Reject changed runtime code even when a material override is explicit."""
        trained = self._train(iterations=0, num_worlds=1)
        with patch.object(ToyRig, "source_hash", "toy_runtime_changed"):
            with self.assertRaisesRegex(ValueError, "runtime source differ"):
                policy.evaluate(trained["best_checkpoint"], artifact_path=self.artifact_path)

    def test_explicit_physics_update_marks_old_scores_inapplicable(self):
        """Re-evaluate unchanged weights only after explicit source-update consent."""
        trained = self._train(iterations=0, num_worlds=1)
        checkpoint = Path(trained["best_checkpoint"])
        before = checkpoint.read_bytes()
        with patch.object(ToyRig, "source_hash", "toy_runtime_changed"):
            with self.assertWarnsRegex(UserWarning, "changed physics source"):
                rig, actor = policy.restore(checkpoint, allow_physics_update=True)
        self.assertEqual(rig.frame, 0)
        self.assertTrue(actor.checkpoint_metadata["physics_updated"])
        self.assertFalse(actor.checkpoint_metadata["checkpoint_scores_applicable"])
        self.assertEqual(
            actor.checkpoint_metadata["physics_source_changes"]["rig_source_sha256"],
            {"checkpoint": "toy_runtime_v1", "current": "toy_runtime_changed"},
        )
        self.assertEqual(before, checkpoint.read_bytes())
        self.assertTrue(all(not p.requires_grad for p in actor.parameters()))

    def test_physics_update_cannot_bypass_geometry_or_settings(self):
        """Keep frozen geometry and settings checks strict during source migration."""
        trained = self._train(iterations=0, num_worlds=1)
        changed = _artifact()
        changed["column_bed"]["rest_length_m"] = [0.04]
        self.artifact_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "cannot change shoe geometry"):
            policy.restore(trained["best_checkpoint"], artifact_path=self.artifact_path, allow_physics_update=True)
        self.artifact_path.write_text(json.dumps(_artifact()))
        original = ToyRig.__init__

        def changed_config(rig, *args, **kwargs):
            original(rig, *args, **kwargs)
            rig.input_fingerprints["config_identity"] = "changed settings"

        with patch.object(ToyRig, "__init__", changed_config):
            with self.assertRaisesRegex(ValueError, "inputs or runtime source differ"):
                policy.restore(trained["best_checkpoint"], allow_physics_update=True)

    def test_evaluation_records_physics_source_migration(self):
        """Include source-change provenance in the newly measured policy report."""
        trained = self._train(iterations=0, num_worlds=1)
        with patch.object(ToyRig, "source_hash", "toy_runtime_changed"):
            with self.assertWarnsRegex(UserWarning, "changed physics source"):
                _, report = policy.evaluate(trained["best_checkpoint"], allow_physics_update=True)
        self.assertTrue(report["physics_updated"])
        self.assertFalse(report["checkpoint_scores_applicable"])
        self.assertIn("tracking_loss_mean", report)

    def test_reject_old_six_action_checkpoint(self):
        """Reject legacy checkpoint formats and disguised six-action metadata."""
        torch = policy._torch()
        old_path = self.directory / "legacy.pt"
        torch.save({"format": 1, "action_dim": 6, "policy_state": {}}, old_path)
        with self.assertRaisesRegex(ValueError, "old six-action"):
            policy.evaluate(old_path)
        torch.save({"format": policy.CHECKPOINT_FORMAT, "action_dim": 6}, old_path)
        with self.assertRaisesRegex(ValueError, "two-stiffness"):
            policy.evaluate(old_path)

    def test_reject_reference_config_and_scaling_tamper(self):
        """Fail closed on corrupted schedules, omitted rig fields, and changed scaling."""
        trained = self._train(iterations=0, num_worlds=1)
        torch = policy._torch()
        original = torch.load(trained["best_checkpoint"], weights_only=True)
        path = self.directory / "tampered.pt"
        bad = copy.deepcopy(original)
        bad["reference"]["ankle_equilibrium_rad"][0] += 1.0
        torch.save(bad, path)
        with self.assertRaisesRegex(ValueError, "reference content hash"):
            policy.evaluate(path)
        bad = copy.deepcopy(original)
        del bad["rig_config"]["damping_ratio"]
        bad["rig_config_sha256"] = policy._json_hash(bad["rig_config"])
        torch.save(bad, path)
        with self.assertRaisesRegex(ValueError, "ALL current rig settings"):
            policy.evaluate(path)
        with patch.object(ToyRig, "observation_scale", np.ones(4)):
            with self.assertRaisesRegex(ValueError, "observation contract"):
                policy.evaluate(trained["best_checkpoint"])


if __name__ == "__main__":
    unittest.main()
