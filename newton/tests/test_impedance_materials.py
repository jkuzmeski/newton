# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify mechanical work accounting for matched material experiments."""

import unittest

import numpy as np

from projects.impedance_instron.hogan.materials import work_metrics


class TestMaterialWork(unittest.TestCase):
    def test_signed_work_uses_native_interval_durations(self):
        """Integrate each joint's signed power over unequal native intervals."""
        trace = {
            "time_s": np.array([0.0, 0.1, 0.3]),
            "load": np.array([[999, 999, 999, 2, -3, 4], [999, 999, 999, -2, 3, 4]]),
            "velocity": np.array([[999, 999, 999, 1, 2, -1], [999, 999, 999, 3, 2, 1], [999] * 6]),
        }
        metrics = work_metrics(trace)
        np.testing.assert_allclose(metrics["joint_positive_work_j"], [0.2, 1.2, 0.8])
        np.testing.assert_allclose(metrics["joint_negative_work_j"], [-1.2, -0.6, -0.4])
        np.testing.assert_allclose(metrics["joint_net_work_j"], [-1.0, 0.6, 0.4])
        self.assertAlmostEqual(metrics["total_positive_work_j"], 2.2)
        self.assertAlmostEqual(metrics["total_absorbed_work_j"], 2.2)
        np.testing.assert_allclose(metrics["joint_peak_positive_power_w"], [2, 6, 4])
        np.testing.assert_allclose(metrics["joint_peak_absorbed_power_w"], [6, 6, 4])


if __name__ == "__main__":
    unittest.main()
