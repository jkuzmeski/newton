# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that Hogan profiles contain the physical inputs used by the chain."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_impedance_hogan import _reference
from projects.impedance_instron.cartesian import profile
from projects.impedance_instron.hogan.mechanics import chain_from_profile


class TestRunnerProfile(unittest.TestCase):
    def setUp(self):
        """Build a complete profile from inertial inputs."""
        self.profile = {
            "schema": profile.SCHEMA,
            "masses_kg": [8.0, 4.0, 1.0],
            "com_local_m": [[0.2, 0.0], [0.2, 0.0], [0.05, -0.01]],
            "inertias_kg_m2": [0.15, 0.05, 0.008],
            "provenance": {"inertial": "Synthetic test geometry"},
        }

    def test_load_physical_inputs(self):
        """Load only the inertial fields used to build the runner chain."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(self.profile))
            loaded = profile.load(path)
        self.assertEqual(loaded, self.profile)
        chain = chain_from_profile(_reference(), loaded)
        np.testing.assert_array_equal(chain.masses_kg[1:], self.profile["masses_kg"])
        np.testing.assert_array_equal(chain.inertias_kg_m2[1:], self.profile["inertias_kg_m2"])

    def test_reject_invalid_physical_inputs(self):
        """Reject missing, malformed, nonfinite, and nonpositive inertias."""
        for name, shape in profile._SHAPES.items():
            missing = self.profile.copy()
            del missing[name]
            with self.subTest(name=name, value="missing"), self.assertRaises(ValueError):
                profile.validate(missing)
            values = (np.full(shape, np.nan).tolist(), ["1"], [[1]], [True] * shape[0])
            if name != "com_local_m":
                values += ([0.0] * 3, [-1.0] * 3)
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    profile.validate(self.profile | {name: value})

    def test_reject_unrecorded_or_unknown_physical_fields(self):
        """Require inertial provenance and reject unsupported profile fields."""
        for extra in (
            {"schema": "unrecognized"},
            {"provenance": {}},
            {"provenance": {"inertial": " "}},
            {"provenance": {"inertial": "test", "mass": np.nan}},
            {"shoe_mass_kg": 1.0},
            {"hip_stiffness_n_m": 10.0},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                profile.validate(self.profile | extra)


if __name__ == "__main__":
    unittest.main()
