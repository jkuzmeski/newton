# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.supermimic.body import BodyParams
from projects.supermimic.landing import LandingRig, LandingSchedule, impact_metrics, touchdown_starts
from projects.supermimic.sim import RunnerSim, RunnerSimConfig, load_shoes

DATASET = Path("Z:/newton_grid/data/dataset_3d_column_bed_v2")
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


class TestSupermimicLandingMetrics(unittest.TestCase):
    def test_impact_metrics_of_a_ramp(self):
        """Measure the rise and slope of a linear ramp to a peak that a later ripple does not exceed."""
        dt = 1e-3
        force = np.concatenate((np.zeros(5), np.linspace(0.0, 1000.0, 21), [900.0, 950.0, 800.0], np.full(20, 700.0)))
        m = impact_metrics(force, dt, body_weight=500.0)
        self.assertAlmostEqual(m["impact_bw"], 2.0)
        self.assertAlmostEqual(m["rise_s"], 0.012)
        self.assertAlmostEqual(m["loading_rate_bw_s"], 100.0)

    def test_touchdown_starts_keep_the_last_flight_frame(self):
        """Start each touchdown at its last flight frame and name the landing foot."""
        contact = np.zeros((1, 12, 2), bool)
        contact[0, 6:, 1] = True
        rows = touchdown_starts(contact, np.zeros(3, int), np.array([3, 4, 5]))
        np.testing.assert_array_equal(rows, [[0, 5, 1]])


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
