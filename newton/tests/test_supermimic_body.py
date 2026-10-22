# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from projects.supermimic.body import BODY_NAMES, BodyParams, add_body


def _body_q(q: np.ndarray) -> np.ndarray:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    add_body(builder, BodyParams())
    model = builder.finalize(device="cpu")
    state = model.state()
    joint_q = np.zeros(17, np.float32)
    joint_q[6] = 1.0
    joint_q[7:] = q
    state.joint_q.assign(joint_q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state.body_q.numpy()


def _origin(body_q: np.ndarray, name: str) -> np.ndarray:
    return body_q[BODY_NAMES.index(name), :3]


class TestSupermimicBody(unittest.TestCase):
    def test_mass_and_layout(self):
        """Build seven bodies, 17 coordinates and the placeholder 66.5 kg runner."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        add_body(builder, BodyParams())
        model = builder.finalize(device="cpu")
        self.assertEqual(list(model.body_label), list(BODY_NAMES))
        self.assertEqual(model.joint_coord_count, 17)
        self.assertAlmostEqual(float(model.body_mass.numpy().sum()), 66.5, places=4)
        self.assertAlmostEqual(BodyParams().mass, 66.5)

    def test_positive_angles_follow_the_coordinate_contract(self):
        """Move each segment in its anatomical direction for positive angles on both sides."""
        rest = _body_q(np.zeros(10))
        for side, offset, medial in (("r", 0, 1.0), ("l", 5, -1.0)):
            q = np.zeros(10)
            q[offset] = 0.5
            self.assertGreater(_origin(_body_q(q), f"shank_{side}")[0], 0.1)  # hip flexion: knee forward
            q = np.zeros(10)
            q[offset + 1] = 0.3
            moved = _origin(_body_q(q), f"shank_{side}")[1] - _origin(rest, f"shank_{side}")[1]
            self.assertGreater(medial * moved, 0.05)  # adduction: knee toward the midline
            q = np.zeros(10)
            q[offset + 3] = 0.5
            self.assertLess(_origin(_body_q(q), f"foot_{side}")[0], -0.1)  # knee flexion: ankle back
            q = np.zeros(10)
            q[offset + 4] = 0.5
            foot = _body_q(q)[BODY_NAMES.index(f"foot_{side}")]
            toe = wp.quat_rotate(wp.quat(*foot[3:]), wp.vec3(1.0, 0.0, 0.0))
            self.assertGreater(toe[2], 0.4)  # dorsiflexion: toe up
            # Internal rotation with hip and knee at 90 deg swings the foot outward.
            q = np.zeros(10)
            q[[offset, offset + 2, offset + 3]] = np.pi / 2
            moved = _origin(_body_q(q), f"foot_{side}")[1] - _origin(rest, f"foot_{side}")[1]
            self.assertAlmostEqual(-medial * moved, 0.43, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
