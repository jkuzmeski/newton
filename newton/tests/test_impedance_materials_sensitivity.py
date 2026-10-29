# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-at-a-time shoe material sensitivity variants."""

import unittest

import numpy as np
from projects.impedance_instron.hogan.materials_sensitivity import (
    VARIANTS,
    combine,
    instron_physics,
    scale_material,
    variant_artifact,
)

from projects.digital_shoe.artifact import SCHEMA_VERSION
from projects.digital_shoe.material import hyperfoam_pressure_from_stretch_numpy
from projects.digital_shoe.runtime import ShoeMaterial

FITTED = {
    "instantaneous_shear_modulus_pa": 365000.0,
    "hyperfoam_exponent": 13.9,
    "equilibrium_fraction": 0.95,
    "pasternak_n_per_m": 788.0,
    "effective_poisson_ratio": 0.0,
    "maxwell_relaxation_time_s": 0.019,
    "instantaneous_shear_modulus_2_pa": 18700.0,
    "hyperfoam_exponent_2": -1.23,
    "equilibrium_shear_modulus_override_pa": 0.0,
    "shear_layer_fraction": 0.072,
    "slider_modulus_pa": 202000.0,
    "slider_yield_pa": 3200.0,
}


def equilibrium_moduli(p):
    """Return both equilibrium Ogden-Hill moduli."""
    return (
        np.array([p["instantaneous_shear_modulus_pa"], p["instantaneous_shear_modulus_2_pa"]])
        * p["equilibrium_fraction"]
    )


class MaterialSensitivityTests(unittest.TestCase):
    """Each knob changes the property it names and leaves the others alone."""

    def test_stiffness_scales_the_whole_foam_law(self):
        """Scale equilibrium and instantaneous moduli together, so mu_eq follows."""
        p = scale_material(FITTED, VARIANTS["stiffness_x2"])
        np.testing.assert_allclose(equilibrium_moduli(p), 2.0 * equilibrium_moduli(FITTED))
        self.assertEqual(p["equilibrium_fraction"], FITTED["equilibrium_fraction"])
        self.assertEqual(p["slider_yield_pa"], FITTED["slider_yield_pa"])

    def test_maxwell_share_keeps_the_equilibrium_curve(self):
        """Turn off or double the Maxwell share of the instantaneous modulus at fixed equilibrium."""
        off = scale_material(FITTED, VARIANTS["maxwell_off"])
        self.assertEqual(off["equilibrium_fraction"], 1.0)
        np.testing.assert_allclose(equilibrium_moduli(off), equilibrium_moduli(FITTED))
        doubled = scale_material(FITTED, VARIANTS["maxwell_share_x2"])
        self.assertAlmostEqual(1.0 - doubled["equilibrium_fraction"], 2.0 * (1.0 - FITTED["equilibrium_fraction"]))
        np.testing.assert_allclose(equilibrium_moduli(doubled), equilibrium_moduli(FITTED))
        self.assertEqual(doubled["maxwell_relaxation_time_s"], FITTED["maxwell_relaxation_time_s"])

    def test_no_loss_turns_off_slider_and_maxwell(self):
        """Remove both loss branches; the slider of the elastic control must not survive."""
        p = scale_material(FITTED, VARIANTS["no_loss"])
        self.assertEqual(p["slider_yield_pa"], 0.0)
        self.assertEqual(p["equilibrium_fraction"], 1.0)
        np.testing.assert_allclose(equilibrium_moduli(p), equilibrium_moduli(FITTED))

    def test_exponent_keeps_small_strain_stiffness(self):
        """Scale both Ogden-Hill exponents without changing the small-strain tangent."""

        def tangent(p):
            g1, g2 = equilibrium_moduli(p)
            args = (p["hyperfoam_exponent"], g2, p["hyperfoam_exponent_2"], 0.0, 1.0)
            return (
                hyperfoam_pressure_from_stretch_numpy(1.0 - 1.0e-6, g1, *args)
                - hyperfoam_pressure_from_stretch_numpy(1.0, g1, *args)
            ) / 1.0e-6

        for variant in ("exponent_x0.7", "exponent_x1.4"):
            p = scale_material(FITTED, VARIANTS[variant])
            self.assertAlmostEqual(
                p["hyperfoam_exponent_2"] / FITTED["hyperfoam_exponent_2"], VARIANTS[variant]["exponent"]
            )
            self.assertAlmostEqual(tangent(p) / tangent(FITTED), 1.0, places=4)

    def test_combination_rejects_one_knob_twice(self):
        """Merge distinct knobs and refuse two settings of one knob."""
        self.assertEqual(combine(["stiffness_x0.7", "shear_layer_x2"]), {"stiffness": 0.7, "shear_layer": 2.0})
        with self.assertRaises(ValueError):
            combine(["slider_yield_x0", "no_loss"])

    def test_variant_artifact_changes_the_coupling_the_runtime_reads(self):
        """The runtime couples columns by shear_layer_fraction * mu_eq; the reported Pasternak value follows."""
        raw = {
            "schema_version": SCHEMA_VERSION,
            "shoe": {"id": "fr3", "model_scope": "test"},
            "coordinate_system": {"length_unit": "m", "up_axis": "+Z"},
            "constitutive_model": {
                "type": "effective_hyperfoam_maxwell_pasternak_foundation",
                "parameters": FITTED,
                "derived_quantities": {},
            },
            "column_bed": {
                "anchor_bottom_m": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
                "rest_length_m": [0.02, 0.04],
                "area_m2": [1e-4, 1e-4],
                "neighbors": [[-1, -1, -1, -1], [-1, -1, -1, -1]],
                "spacing_m": 0.01,
                "top_bonded": [True, True],
            },
            "identification": {"calibration": "invalid after edit"},
            "validation": {},
            "provenance": {},
        }
        result = variant_artifact(raw, "shear_layer_x2", VARIANTS["shear_layer_x2"])
        self.assertEqual(result["column_bed"], raw["column_bed"])
        self.assertNotIn("identification", result)
        before = ShoeMaterial(**FITTED)
        after = ShoeMaterial(**result["constitutive_model"]["parameters"])
        self.assertAlmostEqual(after.equilibrium_shear_modulus_pa, before.equilibrium_shear_modulus_pa)
        self.assertAlmostEqual(after.shear_layer_fraction, 2.0 * before.shear_layer_fraction)
        self.assertAlmostEqual(
            after.pasternak_n_per_m, after.shear_layer_fraction * after.equilibrium_shear_modulus_pa * 0.03
        )

    def test_instron_physics_of_a_linear_loop(self):
        """Read peak force, secant stiffness and loss from a loop that unloads at half the loading force."""
        depth = np.concatenate((np.linspace(0.0, 0.01, 101), np.linspace(0.01, 0.0, 101)[1:]))
        force = np.concatenate((1.0e5 * depth[:101], 0.5e5 * depth[101:]))
        force[100] = 1000.0
        result = instron_physics(depth, force)
        self.assertAlmostEqual(result["peak_force_n"], 1000.0)
        self.assertAlmostEqual(result["secant_stiffness_n_per_mm"], 100.0)
        self.assertAlmostEqual(result["loading_work_j"], 5.0)
        self.assertAlmostEqual(result["energy_loss_fraction"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
