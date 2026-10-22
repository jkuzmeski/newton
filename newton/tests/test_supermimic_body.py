# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from projects.supermimic.body import BODY_NAMES, JOINT_NAMES, BodyParams, add_body

CUDA = wp.is_cuda_available()


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


def _kinetic_energy(model: newton.Model, state: newton.State) -> float:
    pose, twist = state.body_q.numpy(), state.body_qd.numpy()
    energy = 0.0
    for p, v, m, inertia in zip(pose, twist, model.body_mass.numpy(), model.body_inertia.numpy(), strict=True):
        omega = np.array(wp.quat_rotate_inv(wp.quat(*p[3:]), wp.vec3(*v[3:])))
        energy += 0.5 * m * float(v[:3] @ v[:3]) + 0.5 * float(omega @ inertia @ omega)
    return energy


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

    def test_joint_limits_bound_both_legs(self):
        """Give every actuated joint the range of motion of the parameters on both sides."""
        params = BodyParams()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        add_body(builder, params)
        model = builder.finalize(device="cpu")
        limits = np.array(params.joint_limits * 2)
        np.testing.assert_allclose(model.joint_limit_lower.numpy()[6:], limits[:, 0], atol=1e-6)
        np.testing.assert_allclose(model.joint_limit_upper.numpy()[6:], limits[:, 1], atol=1e-6)
        self.assertLess(params.joint_limits[3][0], 0.0)

    @unittest.skipUnless(CUDA, "needs CUDA")
    def test_joint_limit_absorbs_energy(self):
        """Take energy out of an ankle that a stiff torque drives into its limit and then releases."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        add_body(builder, BodyParams())
        model = builder.finalize(device="cuda:0")
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        ankle = 6 + JOINT_NAMES.index("ankle_dorsiflexion_r")
        state_0, state_1, control = model.state(), model.state(), model.control()
        q = state_0.joint_q.numpy()
        q[7:] = np.clip(q[7:], lower[6:] + 0.05, upper[6:] - 0.05)
        q[ankle + 1] = 0.5 * (lower[ankle] + upper[ankle])
        state_0.joint_q.assign(q)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver = newton.solvers.SolverMuJoCo(model, integrator="euler", jacobian="dense", disable_contacts=True)
        dt, start, work = 1.25e-4, _kinetic_energy(model, state_0), 0.0
        torque = np.zeros(model.joint_dof_count, np.float32)
        for n in range(2400):
            # For 0.15 s a 300 N m/rad spring pulls the ankle 0.3 rad past its upper limit, then lets go.
            angle, speed = state_0.joint_q.numpy()[ankle + 1], state_0.joint_qd.numpy()[ankle]
            torque[ankle] = 300.0 * (upper[ankle] + 0.3 - angle) - 2.0 * speed if n < 1200 else 0.0
            control.joint_f.assign(torque)
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            work += torque[ankle] * 0.5 * (speed + state_0.joint_qd.numpy()[ankle]) * dt
        self.assertLess(state_0.joint_q.numpy()[ankle + 1], upper[ankle])
        self.assertLess(_kinetic_energy(model, state_0) - start - work, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
