# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Digital Instron material model."""

import unittest

import numpy as np

from projects.digital_instron_v2.core import (
    EFFECTIVE_POISSON_RATIO,
    MAXWELL_RELAXATION_TIME_S,
    Material,
    Trial,
    fit_material,
    predict,
)


class TestDigitalInstronCore(unittest.TestCase):
    def test_lock_unidentified_model_assumptions(self):
        """Keep fixed assumptions explicit until multi-rate data identify them.

        The effective Poisson ratio is no longer a guess: identical confined and
        unconfined compression of racing-shoe midsole foam put it at zero
        (McCulloch, Delp and Kuhl, arXiv:2602.12694), which also makes the
        Ogden-Hill exponent ``beta`` vanish. The relaxation time is still an
        assumption, because that paper used a single loading rate.
        """

        self.assertEqual(EFFECTIVE_POISSON_RATIO, 0.0)
        self.assertEqual(MAXWELL_RELAXATION_TIME_S, 0.08)

    def test_pin_pasternak_coupling_to_the_material(self):
        """Derive each column's Pasternak coefficient as the shear modulus times its thickness.

        The layer coefficient is no longer fitted, so this checks the rule
        ``k_i = mu_eq * t_i`` and that it lands in the 0.6-9.6 kN/m band the
        measured shear moduli of racing-shoe foam imply over a 5-44 mm midsole.
        """

        material = Material(140_000.0, 1.4, 0.72)
        thickness = np.array([0.005, 0.044])

        coupling = material.coupling_n_per_m(thickness)

        self.assertAlmostEqual(material.equilibrium_shear_modulus_pa, 140_000.0 * 0.72)
        np.testing.assert_allclose(coupling, material.equilibrium_shear_modulus_pa * thickness)
        self.assertGreater(coupling[0], 400.0)
        self.assertLess(coupling[1], 9.6e3)

    def test_material_carries_no_fixture_specific_parameter(self):
        """Keep the fitted vector one shared material with no per-fixture freedom.

        The Pasternak coefficient is pinned to the material and nothing replaced
        it, so a rearfoot punch and a full-foot last are described by the same
        four numbers and any disagreement between them stays in the residual.
        """

        self.assertEqual(
            tuple(Material.__dataclass_fields__),
            (
                "instantaneous_shear_modulus_pa",
                "hyperfoam_exponent",
                "equilibrium_fraction",
                "maxwell_relaxation_time_s",
            ),
        )
        self.assertFalse([name for name in Trial.__dataclass_fields__ if "fixture" in name or "compliance" in name])

    def test_reject_invalid_material(self):
        """Reject nonphysical material parameters."""

        with self.assertRaises(ValueError):
            Material(1.0, 1.0, -0.1)

    def test_predict_periodic_maxwell_hysteresis(self):
        """Produce a repeatable force loop from a periodic Maxwell branch."""

        lengths = np.array([[1.0], [0.9], [0.8], [0.9], [1.0]])
        trial = Trial(
            "test",
            np.array([1.0]),
            1.0,
            lengths,
            np.full(5, 0.01),
            np.zeros(5),
            np.array([0.0, 0.1, 0.2, 0.1, 0.0]),
        )
        force = predict(trial, Material(100_000.0, 2.0, 0.5))

        self.assertEqual(force[0], 0.0)
        self.assertGreater(force[1], 0.0)
        self.assertGreater(force[2], force[1])
        self.assertGreater(force[1], force[3])
        self.assertEqual(force[4], 0.0)

    def test_hyperfoam_stiffens_with_compression(self):
        """Increase tangent stiffness smoothly under large compression."""

        trial = Trial(
            "test",
            np.array([1.0]),
            1.0,
            np.array([[0.3], [0.2]]),
            np.array([0.01, 0.01]),
            np.zeros(2),
            np.zeros(2),
        )
        force = predict(trial, Material(100_000.0, 3.0, 1.0))

        self.assertGreater(force[1], force[0])

    def test_record_fit_history(self):
        """Record loss and every material parameter during fitting."""

        material = Material(100_000.0, 3.0, 0.5)
        lengths = np.array([[1.0], [0.9], [0.8], [1.0]])
        trial = Trial(
            "test",
            np.array([1.0]),
            1.0,
            lengths,
            np.full(4, 0.01),
            np.zeros(4),
            np.array([0.0, 0.1, 0.2, 0.0]),
        )
        trial = Trial(
            trial.name,
            trial.slack_m,
            trial.area_m2,
            trial.lengths_m,
            trial.dt_s,
            predict(trial, material),
            trial.displacement_m,
        )
        history = []

        fit_material([trial], material, 2, history)

        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["loss"], 0.0)
        self.assertIn("loss_test", history[0])
        self.assertTrue(all(name in history[0] for name in Material.__dataclass_fields__))


if __name__ == "__main__":
    unittest.main()
