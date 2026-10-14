# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the distance accounting, fall causes, and stance counting of the 3D endurance run."""

import unittest

import numpy as np

from projects.impedance_instron.bilateral import endurance3d as end


def _records(rows: int, worlds: int) -> np.ndarray:
    """A zeroed record buffer with a plausible start pose in every world."""
    records = np.zeros((rows, worlds, len(end.RECORDS)), dtype=np.float32)
    records[:, :, end.RECORDS.index("z_m")] = 0.8
    records[:, :, end.RECORDS.index("alive")] = 1.0
    return records


def _traces(records: np.ndarray, **overrides) -> dict:
    """A minimal :func:`~.endurance3d.run` output around one record buffer."""
    worlds = records.shape[1]
    traces = {
        "records": records,
        "record_dt_s": 0.01,
        "causes": [0] * worlds,
        "fall_steps": [-1] * worlds,
        "finish_steps": [-1] * worlds,
        "starts": [{"world": w, "trial": "T", "segment": "s0", "speed_m_s": 3.65} for w in range(worlds)],
        "timing": {"control_dt_s": 0.005, "wall_s": 1.0, "control_steps": records.shape[0] * 2},
    }
    traces.update(overrides)
    return traces


class TestHeading(unittest.TestCase):
    def test_yaw_matches_the_environment_heading(self):
        """The recorded yaw turns world velocity into the same heading-frame speed the actor sees."""
        import torch

        from projects.impedance_instron.bilateral.env3d import _heading, _quat_to_matrix  # noqa: PLC0415

        rng = np.random.default_rng(0)
        quat = rng.normal(size=(64, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        velocity = rng.normal(size=(64, 3))
        velocity_t = torch.tensor(velocity, dtype=torch.float32)
        quat_t = torch.tensor(quat, dtype=torch.float32)
        yaw, _ = end.heading(quat_t)
        heading = _heading(_quat_to_matrix(quat_t))
        local = torch.einsum("nji,nj->ni", heading, velocity_t)
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        torch.testing.assert_close(cosine, heading[:, 0, 0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sine, heading[:, 1, 0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            torch.stack((local[:, 0], local[:, 1]), -1),
            torch.stack(
                (
                    velocity_t[:, 0] * cosine + velocity_t[:, 1] * sine,
                    -velocity_t[:, 0] * sine + velocity_t[:, 1] * cosine,
                ),
                -1,
            ),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_tilt_matches_the_observed_gravity(self):
        """The recorded tilt is the angle the environment's gravity observation implies."""
        import torch

        from projects.impedance_instron.bilateral.env3d import _quat_to_matrix  # noqa: PLC0415

        rng = np.random.default_rng(1)
        quat = rng.normal(size=(64, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        quat_t = torch.tensor(quat, dtype=torch.float32)
        rotation = _quat_to_matrix(quat_t)
        gravity = torch.einsum("nji,j->ni", rotation, torch.tensor([0.0, 0.0, -1.0]))
        _, tilt = end.heading(quat_t)
        torch.testing.assert_close(tilt, torch.arccos((-gravity[:, 2]).clamp(-1.0, 1.0)), atol=1e-6, rtol=1e-5)

    def test_observation_names_hold_the_pacer_and_heading(self):
        """The actor sees the pacer offset and its heading, which the run records."""
        from projects.impedance_instron.bilateral.env3d import OBSERVATION_NAMES  # noqa: PLC0415

        self.assertEqual(OBSERVATION_NAMES[-4:], ("pacer_x_m", "pacer_y_m", "heading_sin", "heading_cos"))


class TestStances(unittest.TestCase):
    def test_peaks_ignore_chatter_and_count_loaded_time(self):
        """Two 0.2 s bumps of 900 and 1200 N give two peaks and 41 loaded frames of 300."""
        force = np.zeros(300)
        force[20:40] = 900.0
        force[80:100] = 1200.0
        force[250] = 4000.0  # one frame of chatter, shorter than the 0.05 s minimum stance
        peaks, loaded = end._stances(force, end.EnduranceConfig(), 0.01)
        np.testing.assert_allclose(peaks, [900.0, 1200.0])
        self.assertAlmostEqual(loaded, 41 / 300, places=6)

    def test_an_all_zero_window_has_no_stances(self):
        """A swing-only window reports no peaks and no loaded frames."""
        peaks, loaded = end._stances(np.zeros(20), end.EnduranceConfig(), 0.01)
        self.assertEqual(peaks.size, 0)
        self.assertEqual(loaded, 0.0)


class TestSummarize(unittest.TestCase):
    def setUp(self):
        self.config = end.EnduranceConfig(distance_m=10.0, check_every_steps=10, record_every_steps=2)

    def test_fallen_world_reports_its_cause_and_distance(self):
        """A world whose height rule fires keeps the distance reached at that frame."""
        records = _records(11, 1)
        column = {name: i for i, name in enumerate(end.RECORDS)}
        records[:, 0, column["x_m"]] = np.arange(11) * 0.0365
        records[6:, 0, column["alive"]] = 0.0
        row = end.summarize(_traces(records, causes=[100], fall_steps=[110]), self.config)[0]
        self.assertTrue(row["fell"])
        self.assertEqual(row["cause"], end.CAUSES[100])
        self.assertEqual(row["fall_time_s"], 0.55)
        self.assertAlmostEqual(row["distance_x_m"], 6 * 0.0365, places=2)
        self.assertEqual(row["time_s"], 0.06)

    def test_finished_world_stops_at_the_distance(self):
        """A world that reaches the distance reports the crossing frame, not the run end."""
        records = _records(11, 2)
        column = {name: i for i, name in enumerate(end.RECORDS)}
        records[:, 0, column["x_m"]] = np.arange(11) * 2.0
        records[:, 1, column["x_m"]] = np.arange(11) * 0.5
        records[4, 0, column["fz_r_n"]] = records[5, 0, column["fz_r_n"]] = 800.0
        row, incomplete = end.summarize(_traces(records, finish_steps=[10, -1]), self.config)
        self.assertTrue(row["reached_distance"])
        self.assertEqual(row["finish_time_s"], 0.05)
        self.assertAlmostEqual(row["distance_x_m"], 10.0, places=6)
        self.assertEqual(row["stances"], 0)  # the two loaded frames are shorter than a stance
        self.assertFalse(incomplete["reached_distance"])
        self.assertAlmostEqual(incomplete["distance_x_m"], 5.0, places=6)

    def test_a_long_run_pools_peaks_cadence_and_drift(self):
        """Four 0.2 s stances in one second give 240 steps/min and the median of their peaks."""
        rows, dt = 101, 0.01
        records = _records(rows, 1)
        column = {name: i for i, name in enumerate(end.RECORDS)}
        time = np.arange(rows) * dt
        for side, phase in (("r", 0.0), ("l", 0.25)):
            loaded = ((time - phase) % 0.5 < 0.2) & (time >= 0.1)
            records[loaded, 0, column[f"fz_{side}_n"]] = 900.0 + 200.0 * (side == "l")
        records[:, 0, column["y_m"]] = 0.01 * np.sin(np.arange(rows))
        records[:, 0, column["yaw_rad"]] = 0.002 * np.arange(rows)
        row = end.summarize(_traces(records, record_dt_s=dt), self.config)[0]
        self.assertEqual(row["stances"], 4)
        self.assertAlmostEqual(row["cadence_steps_per_min"], 240.0, places=6)
        self.assertEqual(row["grf_peak"]["peak_n_median"], 1000.0)
        self.assertAlmostEqual(row["heading_drift_rad"], 0.2, places=6)
        self.assertAlmostEqual(row["lateral_deviation_m"], 0.01, places=3)


if __name__ == "__main__":
    unittest.main()
