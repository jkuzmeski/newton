# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.supermimic.body import BodyParams
from projects.supermimic.landing import LandingRig, LandingSchedule, reference_schedule, touchdown_starts
from projects.supermimic.sim import RunnerSim, RunnerSimConfig, load_shoes

DATASET = Path("Z:/newton_grid/data/dataset_3d_column_bed_v3d3s")
CUDA = wp.is_cuda_available()


def _schedule(q_eq: np.ndarray, stiffness: float, damping: float) -> LandingSchedule:
    n = len(q_eq)
    gains = np.full((n, 10), stiffness), np.full((n, 10), damping)
    return LandingSchedule(
        equilibrium=q_eq[:, None, :],
        feed_forward=np.zeros((n, 1, 10)),
        stiffness=gains[0],
        damping=gains[1],
        touchdown_stiffness=gains[0],
        touchdown_damping=gains[1],
        touchdown_hold_s=0.0,
    )


class TestSupermimicLandingStarts(unittest.TestCase):
    def test_touchdown_starts_keep_the_last_flight_frame(self):
        """Start each touchdown at its last flight frame and name the landing foot."""
        contact = np.zeros((1, 12, 2), bool)
        contact[0, 6:, 1] = True
        rows = touchdown_starts(contact, np.zeros(3, int), np.array([3, 4, 5]))
        np.testing.assert_array_equal(rows, [[0, 5, 1]])

    def test_reference_schedule_damps_the_velocity_error(self):
        """Track the reference at 1 ms with a feed-forward that carries the moment plus B qd_ref."""
        t = np.arange(20) * 0.005
        joint_q, joint_qd = np.zeros((2, 20, 17)), np.zeros((2, 20, 16))
        joint_q[1, :, 7:] = 0.1 + 2.0 * t[:, None]
        joint_qd[1, :, 6:] = 2.0
        moment = np.zeros((2, 20, 10))
        moment[1] = 30.0
        q0, qd0, schedule = reference_schedule(
            joint_q,
            joint_qd,
            moment,
            0.005,
            np.array([[1, 4, 0]]),
            0.01,
            np.full((1, 10), 200.0),
            np.full((1, 10), 5.0),
        )
        np.testing.assert_allclose(q0, joint_q[1, 4][None])
        np.testing.assert_allclose(qd0, joint_qd[1, 4][None])
        np.testing.assert_allclose(schedule.equilibrium[0, :, 0], 0.1 + 2.0 * (0.02 + np.arange(11) * 1e-3))
        np.testing.assert_allclose(schedule.feed_forward, np.full((1, 11, 10), 30.0 + 5.0 * 2.0))
        np.testing.assert_array_equal(schedule.touchdown_damping, schedule.damping)


@unittest.skipUnless(CUDA and DATASET.exists(), "needs CUDA and the column-bed dataset")
class TestSupermimicLandingRig(unittest.TestCase):
    def test_knee_limit_torque_balances_the_drive(self):
        """Report the limit torque that holds a knee driven past its extension limit, in the air."""
        body = BodyParams()
        shoes = load_shoes(DATASET)
        sim = RunnerSim(body, shoes, [0], config=RunnerSimConfig(belt_speed_m_s=0.0))
        q = np.zeros((1, 17))
        q[0, 2], q[0, 6] = 5.0, 1.0
        q_eq = np.zeros((1, 10))
        q_eq[0, 3] = body.joint_limits[3][0] - 0.5
        log = LandingRig(sim, 0.3).run(q, np.zeros((1, 16)), _schedule(q_eq, 200.0, 5.0))
        knee, last = 3, slice(-800, None)
        self.assertLess(log["q"][0, last, knee].max(), body.joint_limits[3][0] + 0.02)
        drive = np.mean(log["k_torque"][0, last, knee] + log["b_torque"][0, last, knee])
        self.assertLess(drive, -80.0)
        # The limit is a lightly damped spring, so it balances the drive on average.
        np.testing.assert_allclose(log["limit_torque"][0, last, knee].mean(), -drive, rtol=0.05)
        self.assertLess(np.abs(log["limit_torque"][0, :, [0, 1, 2, 4]]).max(), 1e-3)
        np.testing.assert_array_equal(log["touchdown_s"], -1.0)
        self.assertEqual(log["loaded_columns"].max(), 0)

    def test_rig_gives_the_beds_the_foot_mobility(self):
        """Give each bed its foot's current effective mobility while the rig steps."""
        sim = RunnerSim(BodyParams(), load_shoes(DATASET), [0], config=RunnerSimConfig(belt_speed_m_s=0.0))
        q = np.zeros((1, 17))
        q[0, 2], q[0, 6] = 5.0, 1.0
        q[0, 10] = -0.8
        LandingRig(sim, 2 * sim.config.substep_dt_s).run(q, np.zeros((1, 16)), _schedule(q[:, 7:], 200.0, 5.0))
        used = [bed.mobility.numpy() for bed in sim.beds]
        sim.update_foot_mobility(sim.state_0)
        for bed, mobility in zip(sim.beds, used, strict=True):
            self.assertGreater(np.abs(mobility).max(), 0.0)
            # One substep separates the rig's last update from this state.
            np.testing.assert_allclose(mobility, bed.mobility.numpy(), rtol=2e-2, atol=1e-5)

    def test_sim_steps_from_its_reset_after_an_odd_landing(self):
        """Let RunnerSim.step advance the reset state after a landing of an odd number of substeps."""
        sim = RunnerSim(BodyParams(), load_shoes(DATASET), [0], config=RunnerSimConfig(belt_speed_m_s=0.0))
        q = np.zeros((1, 17))
        q[0, 2], q[0, 6] = 5.0, 1.0
        rig = LandingRig(sim, 3 * sim.config.substep_dt_s)
        rig.run(q, np.zeros((1, 16)), _schedule(np.zeros((1, 10)), 200.0, 5.0))
        q[0, 2] = 3.0
        mask = wp.ones(1, dtype=bool, device=sim.device)
        sim.reset(mask, wp.array(q, dtype=float, device=sim.device), wp.zeros((1, 16), dtype=float, device=sim.device))
        action = np.zeros((1, 10, 3), np.float32)
        sim.step(wp.array(action, dtype=float, device=sim.device))
        pelvis_z = sim.state_0.joint_q.numpy().reshape(-1)[2]
        self.assertLess(pelvis_z, 3.0)
        self.assertGreater(pelvis_z, 2.9)
