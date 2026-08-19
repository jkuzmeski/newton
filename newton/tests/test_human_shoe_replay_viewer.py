# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact prescribed human-shoe replay visualization."""

import unittest

import numpy as np
import warp as wp

import newton.viewer
from projects.human_shoe.replay_viewer import Example


class TestHumanShoeReplayViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build one coarse exact replay viewer shared by the tests."""
        cls.viewer = newton.viewer.ViewerNull(num_frames=1)
        args = Example.create_parser().parse_args(["--replay-dt", "0.005", "--playback-fps", "30", "--hide-muscles"])
        cls.example = Example(cls.viewer, args)

    def test_scene_uses_exact_pose_and_recorded_column_history(self):
        """Synchronize exact body poses, pressure colors, columns, GRF, and COP."""
        example = self.example
        example.test_final()
        self.assertEqual(example.model.shape_label[example.midsole_shape], "digital_instron_midsole")
        self.assertEqual(example.replay.column_compression_m.shape[1], len(example.prepared_sole.column_bottom_local))
        self.assertEqual(example.replay.column_force_n.shape[2], 3)
        self.assertGreater(example.force_scale_n, 1.0)
        self.assertTrue(np.any(example._display_valid))
        self.assertTrue(np.all(np.isfinite(example._color_history.numpy())))
        np.testing.assert_array_equal(example.display_time, example.replay.time_s[example.replay_index])
        self.assertLessEqual(
            float(np.max(example.target_time_error_s)),
            0.5 * float(np.max(example.replay.dt_s)) + 1.0e-12,
        )

    def test_render_peak_force_frame_headlessly(self):
        """Render a peak-load frame with a correctly scaled GRF arrow."""
        example = self.example
        example.frame = int(np.argmax(example._display_force[:, 2]))
        example.sim_time = float(example.display_time[example.frame])
        wp.copy(example.state.body_q, example.body_q_frames[example.frame])
        example.render()

        start = example._grf_start.numpy()[example.frame]
        end = example._grf_end.numpy()[example.frame]
        np.testing.assert_allclose(
            end - start,
            example.grf_scale * example._display_force[example.frame],
            rtol=1.0e-5,
            atol=2.0e-8,
        )
        bottom = example._bottom_world.numpy()
        self.assertGreaterEqual(float(np.min(bottom[:, 2])), example.ground_height_m - 1.0e-7)
        self.assertTrue(np.any(np.isclose(bottom[:, 2], example.ground_height_m, atol=1.0e-7)))
        self.assertEqual(self.viewer.frame_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
