# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for saved Hogan report motion animation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.hogan import fit_report
from projects.impedance_instron.hogan.motion_gif import write_motion_gif
from projects.impedance_instron.hogan.presentation import format_report


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

    def test_shared_layout_embeds_motion_and_preserves_results(self):
        """Embed portable motion GIFs and keep results through repeat formatting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run = Path(temp_dir)
            write_motion_gif(_trial(), _saved_trace(_trial()), run / "motion.gif", title="Motion")
            (run / "report.html").write_text(
                '<h1>Saved fit</h1><section id="motion"><h2>Motion</h2>'
                '<figure class="motion-gif"><img src="motion.gif"></figure></section>'
                '<section id="results"><h2>Results</h2><p>Loss 11.5</p></section>'
            )
            path = format_report(run)
            first = path.read_text()
            self.assertIn('data-generative-layout="shared"', first)
            self.assertIn('src="data:image/gif;base64,', first)
            self.assertIn('data-initial-reading="motion"', first)
            self.assertIn('data-kind="biomechanics"', first)
            self.assertIn("Loss 11.5", first)
            format_report(run)
            self.assertEqual(first, path.read_text())

    def test_rebuild_motion_uses_saved_traces_without_prediction(self):
        """Rebuild a previous report's motion section without calling predict_many."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run = Path(temp_dir)
            traces = []
            rows = []
            for index, trial_id in enumerate(("stance", "other")):
                current = _trial(trial_id)
                path = run / f"trace_{index}.npz"
                np.savez(path, **_saved_trace(current, shift=0.02 * index))
                traces.append({"id": trial_id, "trace": path.name})
                rows.append(
                    {
                        "id": trial_id,
                        "loss": float(index + 1),
                        "tracking_rmse": [0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
                        "grf_rmse_n": [1.0, 2.0],
                        "contact_duration_error_s": 0.0,
                        "peak_grf_n": [0.0, 100.0],
                        "peak_fz_error_n": 0.0,
                        "contact_duration_s": 0.1,
                        "status": "completed",
                    }
                )
            summary = {
                "command": {
                    "dataset": "unused",
                    "mount": [0.0, 0.0, 0.0],
                    "pitch": 0.0,
                    "speed": 0.0,
                    "height_offset": 0.0,
                    "friction_model": "default",
                    "limit_per_split": None,
                },
                "trials": traces,
                "splits": {"eval": {"learned": {"trials": rows}}},
            }
            (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run / "report.html").write_text(
                '<!doctype html><section id="motion"><h2>2. Motion</h2>'
                '<h3>Best motion</h3><p class="note">Solid blue: fitted model.</p><svg class="scene"></svg>'
                '<div class="grid"><figure>Original seed comparison 1</figure></div>'
                '<h3>Typical motion</h3><p class="note">Solid blue: fitted model.</p><svg class="scene"></svg>'
                '<div class="grid"><figure>Original seed comparison 2</figure></div></section>',
                encoding="utf-8",
            )
            with (
                patch.object(fit_report, "load_trials", return_value=[_trial("stance"), _trial("other")]),
                patch.object(fit_report, "force_observation", return_value=None),
                patch.object(
                    fit_report, "predict_many", side_effect=AssertionError("must use saved traces")
                ) as predict,
            ):
                fit_report.rebuild_motion_from_saved_traces(run)
            self.assertFalse(predict.called)
            page = (run / "report.html").read_text(encoding="utf-8")
            self.assertIn('class="motion-gif"', page)
            self.assertIn('src="motion_gifs/stance.gif"', page)
            self.assertIn("measured", page)
            self.assertIn("simulated", page)
            self.assertIn("Original seed comparison 1", page)
            self.assertIn("Original seed comparison 2", page)
            self.assertNotIn('class="scene"', page)


if __name__ == "__main__":
    unittest.main()
