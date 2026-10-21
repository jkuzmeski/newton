# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the ``"mujoco"`` leg-contact profile: MuJoCo's own collision between the two legs.

The legs carry the penalty kernel's capsules as MuJoCo shapes that collide right leg against left
only. Two properties matter: legs pushed into each other stay out of each other, and while the
legs are apart the runner (joints and shoe ground force) moves as it does with contacts off.
"""

import os
import unittest

import numpy as np

from projects.impedance_instron.bilateral import env3d
from projects.impedance_instron.bilateral.endurance3d import reset_once

DATASET = "outputs/bilateral/dataset_3d_static_sidefix"
HIP_ADDUCTION_DOFS = (7, 12)
"""Hip adduction of the right and left leg among a world's 16 generalized velocities."""


class MujocoLegContactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():  # pragma: no cover
            raise unittest.SkipTest("the 3D runner needs a CUDA device")
        if not os.path.isdir(DATASET):  # pragma: no cover
            raise unittest.SkipTest(f"{DATASET} is not staged")
        cls.library = env3d.ReferenceLibrary3D(DATASET, ("train",), 1.0e-2, "cuda:0")

    def _environment(self, leg_contact: str, worlds: int = 4) -> env3d.BilateralEnv3D:
        config = env3d.Env3DConfig(dt_s=2.5e-4, substeps=40, solver="mujoco", leg_contact=leg_contact)
        env = env3d.BilateralEnv3D(self.library, worlds, config, seed=3)
        reset_once(env)
        return env

    def _squeeze(self, env: env3d.BilateralEnv3D, substeps: int = 1200) -> float:
        """Drive both hips into adduction at 150 N m with no ground, and return the deepest crossing [m]."""
        torque = np.zeros((env.worlds, 16), dtype=np.float32)
        torque[:, HIP_ADDUCTION_DOFS] = 150.0
        env.control.joint_f.assign(torque.ravel())
        state, scratch = env.states
        env.leg_contact.depth.zero_()
        for _ in range(substeps):
            state.body_f.zero_()
            env.leg_contact.apply(env.data.status, state.body_q, state.body_qd, env.model.body_com, state.body_f)
            env.solver.step(state, scratch, env.control, None, env.config.dt_s)
            state, scratch = scratch, state
        return float(env.leg_contact.depth.numpy().max())

    def test_the_capsules_collide_right_leg_against_left_only(self):
        env = self._environment("mujoco", worlds=1)
        mj = env.solver.mj_model
        self.assertEqual(mj.ngeom, 6)
        for g in range(6):
            for h in range(g + 1, 6):
                type_g, affinity_g = int(mj.geom_contype[g]), int(mj.geom_conaffinity[g])
                type_h, affinity_h = int(mj.geom_contype[h]), int(mj.geom_conaffinity[h])
                collides = bool(type_g & affinity_h or type_h & affinity_g)
                self.assertEqual(collides, (g < 3) != (h < 3), (g, h))
        off = self._environment("off", worlds=1)
        np.testing.assert_array_equal(env.model.body_mass.numpy(), off.model.body_mass.numpy())
        np.testing.assert_array_equal(env.model.body_inertia.numpy(), off.model.body_inertia.numpy())

    def test_legs_driven_into_each_other_do_not_cross(self):
        """150 N m of adduction on both hips crosses the capsules by 100 mm without contact."""
        self.assertGreater(self._squeeze(self._environment("probe")), 0.05)
        self.assertLess(self._squeeze(self._environment("mujoco")), 0.006)

    def test_legs_apart_move_as_with_contacts_off(self):
        """No contact, no difference: joints and the shoe ground wrench match the contacts-off runner."""
        import torch

        off, on = self._environment("off", worlds=8), self._environment("mujoco", worlds=8)
        self.assertTrue(torch.equal(off.joint_q, on.joint_q))
        action = torch.zeros(off.worlds, off.action_size, device=off.torch_device)
        loaded = 0.0
        for _ in range(15):
            off.advance(action)
            on.advance(action)
            self.assertEqual(float(on.leg_contact.depth.numpy().max()), 0.0, "the legs touched")
            self.assertEqual(int(on.solver.mjw_data.nacon.numpy()[0]), 0)
            self.assertLess(float((off.joint_q - on.joint_q).abs().max()), 1.0e-5)
            force_off, force_on = off._interval_force(), on._interval_force()
            self.assertLess(float((force_off - force_on).abs().max()), 0.05)
            loaded = max(loaded, float(force_off.abs().max()))
        self.assertGreater(loaded, 500.0, "the comparison must include stance")

    def test_mujoco_contact_needs_the_mujoco_solver(self):
        config = env3d.Env3DConfig(solver="featherstone", leg_contact="mujoco")
        with self.assertRaises(ValueError):
            env3d.BilateralEnv3D(self.library, 1, config)


if __name__ == "__main__":
    unittest.main()
