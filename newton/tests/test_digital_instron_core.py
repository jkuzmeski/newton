# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Digital Instron material model."""

import unittest

import numpy as np

from projects.digital_instron_v2.core import (
    EFFECTIVE_POISSON_RATIO,
    MAXWELL_RELAXATION_TIME_S,
    Material,
    _hyperfoam_pressure,
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

    def test_material_carries_no_fixture_specific_parameter(self):
        """Keep the fitted vector one shared material with no per-fixture freedom.

        The Pasternak coefficient is pinned to the material and nothing replaced
        it, so a rearfoot punch and a full-foot last are described by the same
        numbers -- two Ogden-Hill terms plus the Maxwell branch, the shear-layer
        fraction and the Jenkin slider -- and any disagreement between them
        stays in the residual. The second term, the shear layer and the slider
        widen the shared law, not the per-fixture freedom.
        """

        self.assertEqual(
            tuple(Material.__dataclass_fields__),
            (
                "instantaneous_shear_modulus_pa",
                "hyperfoam_exponent",
                "equilibrium_fraction",
                "maxwell_relaxation_time_s",
                "instantaneous_shear_modulus_2_pa",
                "hyperfoam_exponent_2",
                "shear_layer_fraction",
                "slider_modulus_pa",
                "slider_yield_pa",
            ),
        )

    def test_reject_invalid_material(self):
        """Reject nonphysical material parameters."""

        with self.assertRaises(ValueError):
            Material(1.0, 1.0, -0.1)
        with self.assertRaises(ValueError):
            Material(1.0, 1.0, 0.5, 0.08, -1.0)  # negative second-term modulus
        with self.assertRaises(ValueError):
            Material(1.0, 1.0, 0.5, 0.08, slider_modulus_pa=-1.0)  # negative slider modulus
        with self.assertRaises(ValueError):
            Material(1.0, 1.0, 0.5, 0.08, slider_yield_pa=-1.0)  # negative slider yield stress

    def test_disabled_second_term_reproduces_the_single_term_law(self):
        """Return the first-order pressure exactly when the second modulus is zero.

        Backward compatibility of every artifact and every earlier fit rests on
        this: a single-term material is the two-term material with ``mu_2 = 0``,
        not an approximation of it.
        """

        strain = np.linspace(0.0, 0.95, 40)
        single = Material(120_000.0, 1.4, 0.72)
        disabled = Material(120_000.0, 1.4, 0.72, 0.08, 0.0, 5.0)

        np.testing.assert_array_equal(_hyperfoam_pressure(strain, disabled), _hyperfoam_pressure(strain, single))

    def test_two_term_hyperfoam_is_the_sum_of_its_terms(self):
        """Add the two Ogden-Hill terms, and make the series modulus their sum.

        Each term contributes ``2 mu_n`` to the small-strain compressive tangent
        whatever its exponent, so the equilibrium modulus that pins the Pasternak
        coupling has to be the sum and not the first term alone.
        """

        strain = np.linspace(0.0, 0.9, 40)
        first = Material(120_000.0, 1.4, 0.72)
        second = Material(30_000.0, 7.5, 0.72)
        both = Material(120_000.0, 1.4, 0.72, 0.08, 30_000.0, 7.5)

        np.testing.assert_allclose(
            _hyperfoam_pressure(strain, both),
            _hyperfoam_pressure(strain, first) + _hyperfoam_pressure(strain, second),
            rtol=1.0e-12,
        )
        self.assertAlmostEqual(both.equilibrium_shear_modulus_pa, (120_000.0 + 30_000.0) * 0.72)
        # Small-strain compressive tangent of the series is 2 (mu_1 + mu_2).
        step = 1.0e-6
        tangent = _hyperfoam_pressure(np.array([step]), both)[0] / step
        self.assertAlmostEqual(tangent / (2.0 * both.equilibrium_shear_modulus_pa), 1.0, places=4)

    def test_negative_second_exponent_densifies(self):
        """Accept a negative second exponent and keep the pressure finite and rising.

        An Ogden-Hill series admits either sign. A positive exponent gives the
        soft ``1 / lambda`` plateau and a negative one gives densification, and
        the published two-term fits of this foam family use one of each, so the
        fit must be free to choose. Zero is a removable singularity and is
        evaluated at its limit.
        """

        strain = np.linspace(0.0, 0.9, 40)
        densifying = Material(120_000.0, 1.4, 0.72, 0.08, 20_000.0, -1.5)
        vanishing = Material(120_000.0, 1.4, 0.72, 0.08, 20_000.0, 0.0)

        for material in (densifying, vanishing):
            pressure = _hyperfoam_pressure(strain, material)
            self.assertTrue(np.all(np.isfinite(pressure)))
            self.assertTrue(np.all(np.diff(pressure) > 0.0))
        # The limit branch agrees with the quotient just outside its cutoff.
        near = Material(120_000.0, 1.4, 0.72, 0.08, 20_000.0, 2.0e-3)
        np.testing.assert_allclose(
            _hyperfoam_pressure(strain, vanishing), _hyperfoam_pressure(strain, near), rtol=2.0e-3
        )

    def test_hyperfoam_stiffens_with_compression(self):
        """Increase tangent stiffness smoothly under large compression."""

        pressure = _hyperfoam_pressure(np.array([0.7, 0.8]), Material(100_000.0, 3.0, 1.0))

        self.assertGreater(pressure[1], pressure[0])


if __name__ == "__main__":
    unittest.main()
