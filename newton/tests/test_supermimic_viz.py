# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from projects.supermimic.body import BODY_NAMES, BodyParams, SegmentParams
from projects.supermimic.reference import forward_kinematics
from projects.supermimic.viz import Replay, inertia_box


class TestSupermimicViz(unittest.TestCase):
    def test_inertia_box_has_segment_inertia(self):
        """A uniform box with the returned half extents has the segment's principal inertia."""
        segment = SegmentParams(6.65, (0.0, 0.0, -0.18), (0.122, 0.11, 0.025))
        a, b, c = 2.0 * inertia_box(segment)
        inertia = segment.mass / 12.0 * np.array((b * b + c * c, a * a + c * c, a * a + b * b))
        np.testing.assert_allclose(inertia, segment.inertia, rtol=1e-12)

    def test_replay_poses_match_reference_kinematics(self):
        """Replayed body poses equal the reference forward kinematics of the same coordinates."""
        body = BodyParams()
        replay = Replay(newton.viewer.ViewerNull(), body)
        rng = np.random.default_rng(0)
        for _ in range(3):
            quat = rng.normal(size=4)
            joint_q = np.concatenate((rng.normal(size=3), quat / np.linalg.norm(quat), rng.uniform(-1.0, 1.0, 10)))
            replay.show(0.0, joint_q)
            body_q = replay.state.body_q.numpy()
            frames = forward_kinematics(joint_q, body)
            for index, name in enumerate(BODY_NAMES):
                np.testing.assert_allclose(body_q[index, :3], frames[name][1], atol=1e-5)

    def test_columns_shorten_at_the_ground(self):
        """Columns below the ground start at the ground and report the depth as shortening."""
        replay = Replay(newton.viewer.ViewerNull(), BodyParams())
        anchors = np.array(((0.1, 0.0, -0.1), (-0.05, 0.0, -0.06)))
        replay.columns = ([anchors, anchors], [np.full(2, 0.03), np.full(2, 0.03)])
        joint_q = np.zeros(17)
        joint_q[2], joint_q[6] = 0.94, 1.0  # ankles at 0.94 - 0.42 - 0.43 = 0.09 m
        replay.show(0.0, joint_q)
        bottoms, tops, shortening = replay.column_segments(replay.state.body_q.numpy())
        np.testing.assert_allclose(shortening, (0.01, 0.0, 0.01, 0.0), atol=1e-6)
        np.testing.assert_allclose(bottoms[:, 2], (0.0, 0.03, 0.0, 0.03), atol=1e-6)
        np.testing.assert_allclose(tops[:, 2] - bottoms[:, 2] + shortening, 0.03, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
