# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check observation-score selection independently of expensive rollouts."""

import unittest

import numpy as np

from projects.digital_shoe.friction_dynamic_search import _rank
from projects.digital_shoe.friction_sweep import SCORE_NAMES


class TestFrictionObservationRanking(unittest.TestCase):
    """Keep filtered fit and raw mechanics distinct in candidate ranking."""

    def test_matched_score_does_not_relabel_filter_attenuation_as_chatter(self):
        """Use observation loss but measure hidden peaks against raw native samples."""
        raw = {name: np.zeros(1) for name in SCORE_NAMES}
        raw.update(loss=np.array([10.0]), braking_peak_n=np.array([100.0]), propulsive_peak_n=np.array([100.0]))
        observed = {name: value.copy() for name, value in raw.items()}
        observed.update(loss=np.array([1.0]), braking_peak_n=np.array([50.0]), propulsive_peak_n=np.array([50.0]))
        result = {
            "fit_scores": raw,
            "friction_scores": raw,
            "observation_scores": observed,
            "engine_rmse": np.zeros((1, 6)),
            "observation_force_rmse_n": np.zeros((1, 2)),
            "raw_force_diagnostics": np.array([[100.0, 100.0, 1.0]]),
            "completed": np.array([True]),
        }
        self.assertAlmostEqual(_rank(result, np.ones(6), 100.0, "observation_scores")[0], 1.0)
        self.assertAlmostEqual(_rank(result, np.ones(6), 100.0)[0], 10.0)
        result["raw_force_diagnostics"][0, 0] = 200.0
        self.assertGreater(_rank(result, np.ones(6), 100.0, "observation_scores")[0], 1.0)
        result["completed"][0] = False
        self.assertTrue(np.isinf(_rank(result, np.ones(6), 100.0, "observation_scores")[0]))


if __name__ == "__main__":
    unittest.main()
