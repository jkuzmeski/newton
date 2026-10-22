# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

import numpy as np

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    from projects.supermimic.verify import first_flight_starts, stance_metrics


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
