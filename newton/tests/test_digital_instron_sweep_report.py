# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the foam-sweep report builder."""

import unittest

import numpy as np

from projects.digital_instron_v2.core import Material, metrics
from projects.digital_instron_v2.sweep_report import FIELDS, flagged


class TestDigitalInstronSweepReport(unittest.TestCase):
    def test_flags_unreproduced_scores(self):
        """A stored score the fresh replay reproduces passes; a stored blow-up is flagged."""
        time = np.linspace(0.0, 0.5, 51)
        force = 1000.0 * np.sin(np.pi * time / 0.5)
        trace = {"time_s": time, "displacement_m": 0.02 * force / 1000.0, "force_n": force}
        good = {**metrics(force, 0.98 * force, trace["displacement_m"]), "contact": {"max_strain": 0.6}}
        blown = {**metrics(force, 50.0 * force, trace["displacement_m"]), "contact": {"max_strain": 3.0}}
        fit = {"conditions": {"a": {"validate": {"fitted": good}}, "b": {"validate": {"fitted": blown}}}}
        replay = {"keys": [("a", "validate"), ("b", "validate")], "traces": [trace, trace], "force": [0.98 * force] * 2}
        self.assertEqual(list(flagged(fit, replay)), [("b", "validate")])

    def test_reports_every_material_parameter(self):
        """Every fitted parameter has a report row, so a parameter at its bound can be named."""
        self.assertEqual(set(FIELDS), set(Material.__dataclass_fields__))


if __name__ == "__main__":
    unittest.main(verbosity=2)
