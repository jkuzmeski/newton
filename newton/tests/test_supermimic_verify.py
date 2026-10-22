# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.supermimic.body import BodyParams
from projects.supermimic.sim import RunnerSim, RunnerSimConfig, load_shoes

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    from projects.supermimic.verify import SubstepWork, first_flight_starts, mechanical_energy, stance_metrics

# The grid mounts locally as Z:.
DATASET = next(
    (
        p
        for p in (
            Path("Z:/newton_grid/data/dataset_3d_column_bed_v3"),
            Path.home() / "newton_grid/data/dataset_3d_column_bed_v3",
        )
        if p.exists()
    ),
    None,
)


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestSupermimicVerify(unittest.TestCase):
    def test_stance_metrics_of_a_half_sine_stance(self):
        """Measure contact time, peak and the braking and propulsive impulse of one right-foot stance."""
        dt, weight = 1e-3, 700.0
        grf = np.zeros((400, 2, 3))
        t = np.arange(200) * dt
        grf[100:300, 0, 2] = 2.0 * weight * np.sin(np.pi * t / 0.2)
        grf[100:200, 0, 0] = -100.0
        grf[200:300, 0, 0] = 50.0

        result = stance_metrics(grf, dt, weight)

        self.assertEqual(result["left"], [])
        (stance,) = result["right"]
        self.assertAlmostEqual(stance["contact_time_s"], 0.2 - 2 * dt, delta=0.01)
        self.assertAlmostEqual(stance["peak_bw"], 2.0, delta=1e-3)
        self.assertAlmostEqual(stance["braking_impulse_bw_s"], 100.0 * 0.1 / weight, delta=1e-3)
        self.assertAlmostEqual(stance["propulsive_impulse_bw_s"], 50.0 * 0.1 / weight, delta=1e-3)

    def test_first_flight_starts_keep_the_first_phase_of_each_window(self):
        """Keep the flight frames up to the first gap in each window."""
        starts = np.array([[3, 10], [3, 11], [3, 12], [3, 60], [3, 61], [5, 40], [5, 41]])

        np.testing.assert_array_equal(first_flight_starts(starts), [[3, 10], [3, 11], [3, 12], [5, 40], [5, 41]])


if __name__ == "__main__":
    unittest.main(verbosity=2)


@unittest.skipUnless(
    _HAS_TORCH and wp.is_cuda_available() and DATASET, "requires torch, CUDA and the column-bed dataset"
)
class TestSupermimicVerifyWork(unittest.TestCase):
    def test_substep_work_balances_a_flight_into_a_joint_limit(self):
        """Balance a flight's mechanical energy with the joint and joint-limit work while a knee hits its limit."""
        sim = RunnerSim(BodyParams(), load_shoes(DATASET)[:1], [0], config=RunnerSimConfig(belt_speed_m_s=0.0))
        work = SubstepWork(sim)
        lower = sim.model.joint_limit_lower.numpy()[6:]
        q = np.zeros((1, 17), np.float32)
        q[0, 2] = 3.0
        q[0, 6] = 1.0
        q[0, 10] = lower[3] + 0.05
        qd = np.zeros((1, 16), np.float32)
        qd[0, 9] = -10.0
        action = np.zeros((1, 10, 3), np.float32)
        action[0, :, 0] = q[0, 7:]
        action[0, :, 1] = 10.0
        action[0, :, 2] = 1.0
        sim.reset(wp.ones(1, dtype=bool), wp.array(q, dtype=float), wp.array(qd, dtype=float))
        action = wp.array(action, dtype=float)

        energy = [mechanical_energy(sim).item()]
        joint_work, constraint_work = [], []
        for _ in range(10):
            sim.step(action)
            energy.append(mechanical_energy(sim).item())
            joint_work.append(work.joint_work.numpy()[0])
            constraint_work.append(work.constraint_work.numpy()[0])

        self.assertEqual(np.abs(sim.grf.numpy()).max(), 0.0)
        self.assertGreater(np.abs(constraint_work).sum(), 1.0)
        self.assertLess(np.abs(work.carrier_work.numpy()).max(), 1e-6)
        np.testing.assert_allclose(np.diff(energy), np.add(joint_work, constraint_work), atol=0.1)
