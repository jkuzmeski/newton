# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test synthetic native marker IK."""

import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

import newton
from projects.gait_c3d.native_motion_fit import (
    joint_limit_violation,
    marker_attachments_from_model,
    marker_positions_from_joint_q,
    solve_marker_sequence,
)


class TestNativeMotionFit(unittest.TestCase):
    """Test public-API marker fitting on the calibrated native subject."""

    @classmethod
    def setUpClass(cls):
        """Build the free-root calibrated S001 test model."""
        cls.previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        base = Path(__file__).parents[2] / "projects" / "gait_c3d" / "assets" / "s001_calibrated"
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(base / "model" / "subject.xml"), floating=True, parse_sites=True)
        cls.model = builder.finalize(device="cpu")
        cls.attachments = marker_attachments_from_model(cls.model)
        cls.seed = cls.model.joint_q.numpy().copy()

    @classmethod
    def tearDownClass(cls):
        """Restore the global coordinate-target option."""
        newton.use_coord_layout_targets = cls.previous_target_layout

    def _target(self, phase: float) -> np.ndarray:
        """Create an in-limit free-root target configuration."""
        target = self.seed.copy()
        target[:3] += np.asarray((0.02 * np.sin(phase), 0.015 * np.cos(phase), 0.01 * np.sin(phase)), dtype=np.float32)
        angle = 0.05 * np.sin(phase)
        target[3:7] = np.asarray((0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)), dtype=np.float32)
        target[7:] = np.asarray(
            [0.05 * np.sin(phase + index * 0.2) for index in range(self.model.joint_coord_count - 7)],
            dtype=np.float32,
        )
        target[13] = 0.15 + 0.04 * np.sin(phase)
        target[18] = 0.15 + 0.04 * np.cos(phase)
        return target

    def test_uses_one_site_per_tracking_cluster(self):
        """Use one native attachment for each thigh and shank cluster."""
        self.assertEqual(len(self.attachments), 27)
        self.assertEqual(
            {attachment.name for attachment in self.attachments}
            & {"L.Thigh.Centroid", "R.Thigh.Centroid", "L.Shank.Centroid", "R.Shank.Centroid"},
            {"L.Thigh.Centroid", "R.Thigh.Centroid", "L.Shank.Centroid", "R.Shank.Centroid"},
        )
        self.assertFalse(
            any(attachment.name.endswith((".Upper", ".Front", ".Rear")) for attachment in self.attachments)
        )

    def test_recovers_clean_target_with_public_ik(self):
        """Recover clean synthetic markers to submillimeter residuals."""
        target_q = self._target(0.6)
        target = marker_positions_from_joint_q(self.model, self.attachments, target_q)
        frames = solve_marker_sequence(self.model, self.attachments, target[None], self.seed, iterations=80)
        self.assertLess(frames[0].marker_rms, 1.0e-4)
        self.assertLess(frames[0].marker_max, 3.0e-4)
        self.assertLess(joint_limit_violation(self.model, frames[0].joint_q), 1.0e-5)
        self.assertAlmostEqual(float(np.linalg.norm(frames[0].joint_q[3:7])), 1.0, places=5)

    def test_noise_and_occlusion_improve_over_neutral_seed(self):
        """Keep noisy and occluded solves finite and better than the neutral seed."""
        target_q = self._target(1.2)
        target = marker_positions_from_joint_q(self.model, self.attachments, target_q)
        rng = np.random.default_rng(17)
        noisy = target + rng.normal(0.0, 0.001, size=target.shape)
        visible = np.arange(len(self.attachments))[np.arange(len(self.attachments)) % 4 != 0]
        attachments = tuple(self.attachments[index] for index in visible)
        target_visible = noisy[visible]
        neutral = marker_positions_from_joint_q(self.model, attachments, self.seed)
        neutral_rms = float(np.sqrt(np.mean((neutral - target_visible) ** 2)))
        frames = solve_marker_sequence(self.model, attachments, target_visible[None], self.seed, iterations=80)
        self.assertTrue(np.all(np.isfinite(frames[0].predicted_markers)))
        self.assertLess(frames[0].marker_rms, neutral_rms * 0.5)
        self.assertLess(joint_limit_violation(self.model, frames[0].joint_q), 1.0e-5)

    def test_batched_sequence_stays_on_device_and_converges(self):
        """Solve independent frames in one GPU-oriented batch with low residuals."""
        target_coordinates = np.asarray(
            [self._target(phase) for phase in (0.2, 0.35, 0.5, 0.65, 0.8)], dtype=np.float32
        )
        targets = np.asarray(
            [marker_positions_from_joint_q(self.model, self.attachments, target) for target in target_coordinates]
        )
        frames = solve_marker_sequence(
            self.model,
            self.attachments,
            targets,
            self.seed,
            iterations=60,
            batch_size=4,
        )
        self.assertEqual(len(frames), len(targets))
        self.assertTrue(all(frame.marker_rms < 1.0e-4 for frame in frames))
        self.assertTrue(all(np.all(np.isfinite(frame.predicted_markers)) for frame in frames))

    def test_warm_start_keeps_sequence_continuous(self):
        """Warm-start adjacent synthetic frames without large coordinate jumps."""
        target_coordinates = np.asarray([self._target(phase) for phase in (0.2, 0.35, 0.5, 0.65)], dtype=np.float32)
        targets = np.asarray(
            [marker_positions_from_joint_q(self.model, self.attachments, target) for target in target_coordinates]
        )
        frames = solve_marker_sequence(self.model, self.attachments, targets, self.seed, iterations=60)
        jumps = [np.linalg.norm(current.joint_q[7:] - previous.joint_q[7:]) for previous, current in pairwise(frames)]
        self.assertLess(max(jumps), 0.25)
        self.assertTrue(all(frame.marker_rms < 1.0e-4 for frame in frames))


if __name__ == "__main__":
    unittest.main()
