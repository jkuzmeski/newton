# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for saved Hogan report motion animation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.impedance_instron.hogan.motion_gif import write_motion_gif


class _Chain:
    masses_kg = np.ones(4)

    @staticmethod
    def kinematics(q):
        hip = np.array([q[0], q[1]])
        knee = hip + np.array([0.0, -0.35])
        ankle = knee + np.array([0.12, -0.35])
        return hip, knee, ankle, ankle

    @staticmethod
    def angle(q, _index):
        return float(q[5])


def _trial(trial_id="stance"):
    vertices = np.array([[-0.12, -0.03], [0.12, -0.03], [0.12, 0.03], [-0.12, 0.03]])
    shoe = SimpleNamespace(
        last_vertices_local_m=np.column_stack((vertices[:, 0], np.zeros(4), vertices[:, 1])),
        anchor_local_m=np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        attachment_local_m=np.array([[-0.1, 0.0, -0.03], [0.1, 0.0, -0.03]]),
        static_pitch_rad=0.0,
    )
    times = np.array([0.0, 0.05, 0.1])
    poses = np.zeros((3, 6))
    poses[:, 0] = [0.0, 0.02, 0.04]
    poses[:, 1] = [0.9, 0.85, 0.8]
    poses[:, 5] = [0.0, 0.15, 0.3]
    return SimpleNamespace(
        id=trial_id,
        chain=_Chain(),
        shoe=shoe,
        provenance={"rest_of_body": {"com_local_m": [0.0, 0.05]}, "reference": "unused"},
        time_s=times,
        q=poses,
        duration_s=0.1,
        force_time_s=times,
        grf_n=np.array([[0.0, 0.0], [20.0, 100.0], [0.0, 0.0]]),
        force_observation=None,
        task=SimpleNamespace(speed_m_s=0.0),
    )


def _saved_trace(trial, shift=0.0):
    state = trial.q.copy()
    state[:, 0] += shift
    return {
        "time_s": trial.time_s.copy(),
        "state": state,
        "reference_state": trial.q.copy(),
        "grf_n": trial.grf_n.copy(),
        "ankle_contact_moment_nm": np.zeros(3),
        "equilibrium_rad": np.zeros((3, 3)),
        "stiffness_nm_rad": np.zeros((3, 3)),
        "damping_nms_rad": np.zeros((3, 3)),
        "load": np.zeros((3, 6)),
    }


class TestHoganMotionGif(unittest.TestCase):
    """Check saved fit reports render motion without rerunning the model."""

    def test_write_motion_gif_saves_multiple_distinct_frames(self):
        """Render an animated GIF from supplied saved poses."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is provided by the examples extra")
        trial = _trial()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_motion_gif(trial, _saved_trace(trial), Path(temp_dir) / "motion.gif", title="Best motion")
            with Image.open(path) as gif:
                self.assertGreater(gif.n_frames, 1)
                gif.seek(0)
                first = gif.convert("RGB").copy()
                gif.seek(gif.n_frames - 1)
                last = gif.convert("RGB")
                self.assertNotEqual(Image.eval(first, lambda value: value).tobytes(), last.tobytes())

    def test_gif_accepts_independent_force_sampling(self):
        """Render reference forces sampled at a different rate from motion."""
        trial = _trial()
        trace = _saved_trace(trial)
        trial.force_time_s = np.linspace(0, 0.1, 7)
        trial.grf_n = np.column_stack((np.zeros(7), np.arange(7) * 10))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_motion_gif(trial, trace, Path(temp_dir) / "motion.gif", title="Motion")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
