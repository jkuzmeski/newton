# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe.column_bed import TOP_BONDED, TOP_FREE
from projects.supermimic.body import BODY_NAMES, BodyParams
from projects.supermimic.sim import FOOT_BODIES, SIDES, RunnerSim, RunnerSimConfig, load_shoes

DATASET = Path("Z:/newton_grid/data/dataset_3d_column_bed_v3")
CUDA = wp.is_cuda_available()


def _standing(shoes, shoe_of_world):
    """Upright pose with each shoe resting flat, and a stiff action holding it."""
    body = BodyParams()
    q = np.zeros((len(shoe_of_world), 17), np.float32)
    q[:, 6] = 1.0
    for w, s in enumerate(shoe_of_world):
        anchors = shoes[s].anchors_in_foot("right", body.sole_pitch)
        for pitch in np.linspace(0.0, 0.2, 2001):
            c, sn = np.cos(pitch), np.sin(pitch)
            z = anchors[:, 0] * -sn + anchors[:, 2] * c
            touching = anchors[np.abs(z - z.min()) < 1e-3, 0]
            if touching.min() < -0.03 and touching.max() > 0.03:
                break
        q[w, 2] = -(body.knee_offset[2] + body.ankle_offset[2]) - z.min() + 1e-3
        q[w, [11, 16]] = -pitch
    action = np.zeros((len(shoe_of_world), 10, 3), np.float32)
    action[:, :, 0] = q[:, 7:]
    action[:, :, 1] = 1e4
    action[:, :, 2] = 50.0
    return q, action


def _run(shoes, shoe_of_world, steps):
    sim = RunnerSim(BodyParams(), shoes, shoe_of_world, config=RunnerSimConfig(belt_speed_m_s=0.0))
    q, action = _standing(shoes, shoe_of_world)
    n = len(shoe_of_world)
    sim.reset(wp.ones(n, dtype=bool), wp.array(q, dtype=float), wp.zeros((n, 16), dtype=float))
    action = wp.array(action, dtype=float)
    for _ in range(steps):
        sim.step(action)
    return sim


@unittest.skipUnless(CUDA and DATASET.exists(), "needs CUDA and the column-bed dataset")
class TestSupermimicSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        masses = (0.173, 0.204)
        cls.shoes = [
            replace(shoe, left=replace(shoe.left, mass_kg=m), right=replace(shoe.right, mass_kg=m))
            for shoe, m in zip(load_shoes(DATASET), masses, strict=True)
        ]

    def test_shoe_mass_moves_to_feet(self):
        """Add each world's shoes to its feet, conserving mass, first and second moments about the ankle."""
        body = BodyParams()
        sim = RunnerSim(body, self.shoes, [0, 1, 0])
        model = sim.model
        bodies = len(BODY_NAMES)
        mass = model.body_mass.numpy().reshape(3, bodies)
        com = model.body_com.numpy().reshape(3, bodies, 3)
        inertia = model.body_inertia.numpy().reshape(3, bodies, 3, 3)

        def second_moment(m, r):
            r = np.atleast_2d(r)
            return np.einsum(
                "n,nij->ij", np.atleast_1d(m), (r * r).sum(1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :]
            )

        for w, s in enumerate([0, 1, 0]):
            shoe = self.shoes[s]
            np.testing.assert_allclose(mass[w].sum(), body.mass + shoe.left.mass_kg + shoe.right.mass_kg, rtol=1e-6)
            for side, foot in zip(SIDES, FOOT_BODIES, strict=True):
                points, positions = shoe.mass_in_foot(side, body.sole_pitch)
                f = body.foot
                c = np.asarray(f.com) * (1.0, -1.0, 1.0) if side == "left" else np.asarray(f.com)
                expected_com = (f.mass * c + points @ positions) / (f.mass + points.sum())
                np.testing.assert_allclose(com[w, foot], expected_com, atol=1e-6)
                about_ankle = np.diag(f.inertia) + second_moment(f.mass, c) + second_moment(points, positions)
                np.testing.assert_allclose(
                    inertia[w, foot] + second_moment(mass[w, foot], com[w, foot]), about_ankle, atol=1e-7
                )
        np.testing.assert_allclose(sim.solver.mjw_model.body_mass.numpy()[:, 1:], mass, rtol=1e-6)
        np.testing.assert_array_equal(inertia[0], inertia[2])
        self.assertGreater(np.abs(inertia[0, FOOT_BODIES[0]] - inertia[1, FOOT_BODIES[0]]).max(), 1e-5)

    def test_mixed_shoes_match_single_shoe_runs(self):
        """Give each world of a mixed batch the result of a batch wearing only its shoe."""
        mixed = _run(self.shoes, [0, 1, 0], 30)
        for s in (0, 1):
            single = _run(self.shoes, [s], 30)
            for w in np.flatnonzero(np.array([0, 1, 0]) == s):
                np.testing.assert_allclose(mixed.joint_q.numpy()[w], single.joint_q.numpy()[0], atol=1e-5)
                np.testing.assert_allclose(mixed.grf.numpy()[w], single.grf.numpy()[0], atol=1e-2)

    def test_step_outputs_are_consistent(self):
        """Average the fine reaction to the step reaction and split power into its work."""
        sim = _run(self.shoes, [0], 50)
        np.testing.assert_allclose(sim.grf_fine.numpy().mean(axis=1), sim.grf.numpy(), rtol=1e-4, atol=1e-3)
        work = sim.joint_work_pos.numpy() + sim.joint_work_neg.numpy()
        np.testing.assert_allclose(work, sim.joint_power.numpy() * sim.config.control_dt_s, atol=1e-6)
        self.assertTrue(np.all(sim.joint_work_neg.numpy() <= 0.0))
        self.assertGreater(sim.grf.numpy()[0, :, 2].sum(), 300.0)
        cop = sim.cop.numpy()[0]
        np.testing.assert_allclose(cop[:, 1], [-0.085, 0.085], atol=0.02)

    def test_feet_hold_only_the_tops_under_the_last(self):
        """Bond the tops under the bench last to the foot and leave the rim tops free."""
        sim = RunnerSim(BodyParams(), self.shoes, [0, 1])
        for bed in sim.beds:
            shoe = self.shoes[bed.worlds[0]]
            bonded = (shoe.right if SIDES[bed.side] == "right" else shoe.left).column_bed.top_bonded
            self.assertTrue(bonded.any() and not bonded.all())
            np.testing.assert_array_equal(bed.bed.bed.connection.numpy(), np.where(bonded, TOP_BONDED, TOP_FREE))

    def test_masked_reset_clears_only_masked_worlds(self):
        """Unload the masked world's shoes and leave the other world running."""
        sim = _run(self.shoes, [0, 0], 20)
        q = sim.joint_q.numpy()
        before = sim.beds[0].bed.bed.q_state.numpy().reshape(2, -1)
        sim.reset(wp.array([True, False], dtype=bool), wp.array(q, dtype=float), wp.zeros((2, 16), dtype=float))
        after = sim.beds[0].bed.bed.q_state.numpy().reshape(2, -1)
        self.assertTrue(np.all(after[0] == 0.0))
        self.assertTrue(np.any(before[0] != 0.0))
        np.testing.assert_array_equal(after[1], before[1])
        np.testing.assert_array_equal(sim._fresh.numpy()[:, 0], [1, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
