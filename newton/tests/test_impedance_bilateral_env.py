# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify the CUDA bilateral environment against the NumPy/CPU-shoe reference."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton.tests.test_impedance_hogan import _tiny_shoe
from projects.impedance_instron.bilateral import data as bilateral_data
from projects.impedance_instron.cartesian.shoe import Shoe

_PROFILE = {
    "masses_kg": [0.5, 0.2, 0.1],
    "com_local_m": [[0.17, 0.0], [0.17, 0.0], [0.05, 0.0]],
    "inertias_kg_m2": [0.01, 0.004, 0.0005],
}
_SUBJECT_KG = 2.0
_LENGTHS = np.array([[0.4, 0.4], [0.4, 0.4]])


def _pose(z: float) -> np.ndarray:
    """Right leg nearly vertical with a level foot, left leg swung forward."""
    q = np.array([0.0, z, math.pi / 2, 0.0, -0.2, 0.2, 0.3, -1.0, 0.7])
    return q


def _write_dataset(directory: Path) -> Path:
    shoe = _tiny_shoe(directory)
    raw = json.loads(Path(shoe.artifact_path).read_text())
    raw["column_bed"]["area_m2"] = [0.004, 0.004]
    raw["instron_fixtures"]["fullfoot_last"]["area_m2"] = [0.004, 0.004]
    root = directory / "dataset"
    (root / "assets").mkdir(parents=True)
    for name in ("digital_shoe_left.json", "digital_shoe_right.json"):
        (root / "assets" / name).write_text(json.dumps(raw))
    (root / "assets" / "profile.json").write_text(json.dumps(_PROFILE))
    frames = 121
    time = np.linspace(0.0, 0.6, frames)
    q = np.tile(_pose(0.9), (frames, 1))
    q[:, 0] = 0.5 * time
    zeros = np.zeros((frames, 2))
    reference = {
        "time_s": time,
        "q": q,
        "q_measured": q.copy(),
        "hip_target_m": q[:, :2].copy(),
        "side_hip_m": np.stack([q[:, :2]] * 2),
        "ankle_target_m": np.stack([zeros, zeros]),
        "pelvis_target_rad": q[:, 2].copy(),
        "grf_time_s": time.copy(),
        "grf_target_n": zeros.copy(),
        "grf_side_n": np.stack([zeros, zeros]),
        "contact_side": np.zeros((2, frames), dtype=np.int8),
        "cop_target_m": np.zeros(frames),
        "lengths_m": _LENGTHS,
        "endpoint_local_m": np.array([[0.15, -0.05], [0.15, -0.05]]),
        "static_pitch_rad": np.zeros(2),
        "shoe_static_pitch_rad": np.asarray(0.0),
        "subject_mass_kg": np.asarray(_SUBJECT_KG),
        "belt_speed_m_s": np.asarray(0.5),
        "metadata_json": np.asarray(json.dumps({"schema": bilateral_data.REFERENCE_SCHEMA})),
    }
    (root / "train").mkdir()
    np.savez_compressed(root / "train" / "s0.npz", **reference)
    manifest = {
        "schema": bilateral_data.DATASET_SCHEMA,
        "head_arms_trunk": {"com_local_m": [0.2, 0.0], "radius_of_gyration_m": 0.1, "mass_kg": None},
        "shoe": {"mount_m": [0.0, 0.0, 0.1], "pitch_rad": 0.0},
        "shared_assets": {
            name: {"file": f"assets/{name}"}
            for name in ("profile.json", "digital_shoe_left.json", "digital_shoe_right.json")
        },
        "members": [
            {
                "id": "s0",
                "split": "train",
                "reference": "train/s0.npz",
                "speed_m_s": 0.5,
                "start_s": 0.0,
                "contacts": [],
                "flight_velocity_m_s": None,
                "flight_velocity_frame": 2,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for bilateral environment tests")
class TestBilateralEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from projects.impedance_instron.bilateral.env import BilateralEnv, EnvConfig, ReferenceLibrary  # noqa: PLC0415

        cls.torch = torch
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = _write_dataset(Path(cls.directory.name))
        cls.config = EnvConfig(substeps=8, compression_limit=0.95)
        library = ReferenceLibrary(cls.root, ("train",), cls.config.control_dt_s, "cuda:0")
        assets = cls.root / "assets"
        cls.env = BilateralEnv(
            library,
            assets / "digital_shoe_left.json",
            assets / "digital_shoe_right.json",
            [0.0, 0.0, 0.1],
            0.0,
            2,
            cls.config,
        )
        cls.action = torch.tensor(
            np.concatenate((np.full(6, 0.3), np.full(6, -1.5), np.full(6, -2.0))),
            dtype=torch.float32,
            device=cls.env.torch_device,
        )[None].repeat(2, 1)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _start(self, mask):
        torch = self.torch
        env = self.env
        right_ankle_z = env.chain.ankle(_pose(0.0), "right")[0][1]
        q = torch.tensor(_pose(0.1005 - right_ankle_z), dtype=torch.float64, device=env.torch_device)
        v = torch.zeros(9, dtype=torch.float64, device=env.torch_device)
        v[0], v[1] = 0.3, -0.4
        count = int(mask.sum())
        env.segment[mask] = 0
        env.index[mask] = 3
        env.origin[mask] = 0.0
        env.t.speed[mask] = 0.5
        env._write_state(
            mask,
            q.expand(count, 9),
            v.expand(count, 9),
            torch.full((count,), 0.4, dtype=torch.float64, device=env.torch_device),
        )
        return q.cpu().numpy(), v.cpu().numpy()

    def _run(self, steps):
        states = []
        for _ in range(steps):
            self.env.set_impedance(self.action)
            self.env.integrate()
            states.append(self.env.t.q.cpu().numpy().copy())
        return np.array(states)

    def test_cuda_world_matches_cpu_reference_through_contact(self):
        """Both feet, both shoes, actuation, sensors, and integration agree with the CPU reference."""
        from projects.impedance_instron.bilateral.reference import ReferenceState, step  # noqa: PLC0415

        env = self.env
        mask = self.torch.ones(2, dtype=self.torch.bool, device=env.torch_device)
        q0, v0 = self._start(mask)
        gpu_states = self._run(15)
        self.assertTrue(np.all(env.t.status.cpu().numpy() == 0))
        shoes = {
            side: Shoe(self.root / "assets" / f"digital_shoe_{side}.json", [0, 0, 0.1], 0.0)
            for side in ("left", "right")
        }
        state = ReferenceState(q0.copy(), v0.copy(), 0.4)
        impedance = env.lower.cpu().numpy() + (env.upper - env.lower).cpu().numpy() / (1 + np.exp(-0.3))
        impedance = (
            impedance,
            env.stiffness_max.cpu().numpy() / (1 + np.exp(1.5)),
            env.damping_max.cpu().numpy() / (1 + np.exp(2.0)),
        )
        peak = 0.0
        for k in range(15 * self.config.substeps):
            wrenches = step(env.chain, shoes, state, impedance, self.config, env.bounds, 0.5)
            peak = max(peak, wrenches["right"][1])
            if (k + 1) % self.config.substeps == 0:
                np.testing.assert_allclose(gpu_states[k // self.config.substeps, 0], state.q, atol=2e-5)
        self.assertGreater(peak, 1.0)
        np.testing.assert_allclose(gpu_states[:, 1], gpu_states[:, 0], atol=0.0)
        np.testing.assert_allclose(env.t.phase.cpu().numpy()[0], state.phase_rad, atol=1e-6)
        np.testing.assert_allclose(env.t.sense.cpu().numpy()[0], state.load_bw, atol=1e-5)

    def test_world_reset_clears_shoe_history(self):
        """Resetting one loaded world reproduces a fresh rollout while the other world continues."""
        torch = self.torch
        both = torch.ones(2, dtype=torch.bool, device=self.env.torch_device)
        self._start(both)
        first = self._run(10)
        only = torch.tensor([True, False], device=self.env.torch_device)
        self._start(only)
        second = self._run(10)
        np.testing.assert_array_equal(second[:, 0], first[:, 0])
        self.assertFalse(np.allclose(second[:, 1], first[:, 1]))

    def test_reward_kernels_are_one_on_the_reference(self):
        """A reference-initialized world scores unit pose and force kernels; one bad joint costs less per term."""
        from dataclasses import replace  # noqa: PLC0415

        env = self.env
        env.reset()
        env.t.grf_sum.zero_()
        env.t.substeps.fill_(1)
        for kernels in ("per_term", "pooled"):
            env.config = replace(self.config, reward_kernels=kernels)
            _, terms = env.reward()
            self.assertAlmostEqual(terms["pose"], 1.0, places=6)
            self.assertAlmostEqual(terms["force"], 1.0, places=6)
        env.t.q[:, 5] += 0.5
        env.config = replace(self.config, reward_kernels="per_term")
        per_term = env.reward()[1]["pose"]
        env.config = replace(self.config, reward_kernels="pooled")
        pooled = env.reward()[1]["pose"]
        env.config = self.config
        # Seven of nine per-term kernels are untouched (pelvis, five joints); the left foot pitch is unchanged too.
        self.assertGreater(per_term, 7.0 / 9.0)
        self.assertLess(pooled, per_term)

    def test_body_points_match_cpu_chain_and_shoe_outline(self):
        """Torch knee/ankle/heel/toe points and velocities equal the CPU chain and shoe outline."""
        torch = self.torch
        env = self.env
        q = torch.tensor(_pose(0.9), dtype=torch.float64, device=env.torch_device)[None]
        v = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64, device=env.torch_device)[None]
        lengths = env.library.segment_lengths[:1]
        points = env.body_points(q, lengths)[0].cpu().numpy()
        chain, row = env.chain, _pose(0.9)
        shoe = Shoe(self.root / "assets" / "digital_shoe_right.json", [0, 0, 0.1], 0.0)
        for s, (side, body) in enumerate((("right", 3), ("left", 6))):
            hip = row[:2]
            np.testing.assert_allclose(points[4 * s + 1] + hip, chain.ankle(row, side)[0], atol=1e-12)
            outline = shoe.outline(chain.ankle(row, side)[0], chain.angle(row, body))
            np.testing.assert_allclose(
                points[4 * s + 2] + hip, outline[np.argmin(shoe.outline_local[:, 0])], atol=1e-12
            )
            np.testing.assert_allclose(
                points[4 * s + 3] + hip, outline[np.argmax(shoe.outline_local[:, 0])], atol=1e-12
            )
            jacobian = chain.ankle(row, side)[1]
            ankle_velocity = env.body_point_velocities(q, v, lengths)[0, 4 * s + 1].cpu().numpy()
            np.testing.assert_allclose(
                ankle_velocity, (jacobian @ v[0].cpu().numpy()) - v[0, :2].cpu().numpy(), atol=1e-6
            )

    def test_new_reward_terms_on_the_reference(self):
        """On-reference worlds score unit body kernels, no contact mismatch, and expose previous actions."""
        from dataclasses import replace  # noqa: PLC0415

        from projects.impedance_instron.bilateral.env import observation_names  # noqa: PLC0415

        env = self.env
        config = replace(
            self.config,
            body_position_weight=0.3,
            body_velocity_weight=0.1,
            contact_match_penalty=0.1,
            action_smoothness_penalty=0.02,
            previous_action_obs=True,
        )
        env.config = config
        try:
            env.reset()
            k = env.library.starts[env.segment] + env.index
            env.t.v.copy_(env.library.qd[k])
            env.t.grf_sum.zero_()
            env.t.substeps.fill_(1)
            env.set_impedance(self.action)
            env.set_impedance(self.action)
            _, terms = env.reward()
            self.assertAlmostEqual(terms["body_position"], 1.0, places=9)
            self.assertAlmostEqual(terms["body_velocity"], 1.0, places=9)
            self.assertEqual(terms["contact_mismatch"], 0.0)
            self.assertEqual(terms["action_change"], 0.0)
            observation = env.observe()
            self.assertEqual(observation.shape[1], len(observation_names(config)))
            np.testing.assert_allclose(
                observation[:, -18:].cpu().numpy(), self.torch.sigmoid(self.action).cpu().numpy(), atol=1e-6
            )
        finally:
            env.config = self.config


if __name__ == "__main__":
    unittest.main()
