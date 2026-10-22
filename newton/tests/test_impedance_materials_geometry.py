# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for materials-by-geometry experiment artifacts."""

import unittest

import numpy as np

from projects.digital_shoe.artifact import validate_artifact
from projects.impedance_instron.hogan.materials_geometry import geometry_variant


class MaterialsGeometryTests(unittest.TestCase):
    """Verify geometry variants preserve column registration and artifact validity."""

    def setUp(self):
        """Create a tiny valid shoe artifact with both fixture mappings."""
        xy = [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]]
        bottom = [[x, y, 0.0] for x, y in xy]
        rest = [0.02, 0.03, 0.04]
        area = [1e-4, 1e-4, 1e-4]
        neighbors = [[-1, -1, -1, -1] for _ in xy]

        def fixture(indices):
            """Create a fixture mapping for selected column sites."""
            return {
                "carrier_anchor_m": [[xy[i][0], xy[i][1], 0.05] for i in indices],
                "foam_free_top_m": [rest[i] for i in indices],
                "foam_bottom_m": [0.0 for _ in indices],
                "rest_length_m": [rest[i] for i in indices],
                "area_m2": [area[i] for i in indices],
                "neighbors": [neighbors[i] for i in indices],
                "spacing_m": 0.01,
                "bed_bottom_above_platen_m": [0.0 for _ in xy],
            }

        self.source = {
            "schema_version": "digital_shoe_1",
            "shoe": {"id": "synthetic", "model_scope": "test"},
            "coordinate_system": {"length_unit": "m", "up_axis": "+Z"},
            "constitutive_model": {
                "type": "effective_hyperfoam_maxwell_pasternak_foundation",
                "parameters": {
                    "instantaneous_shear_modulus_pa": 10000.0,
                    "hyperfoam_exponent": 1.0,
                    "equilibrium_fraction": 1.0,
                    "pasternak_n_per_m": 200.0,
                    "effective_poisson_ratio": 0.0,
                    "maxwell_relaxation_time_s": 0.1,
                    "instantaneous_shear_modulus_2_pa": 0.0,
                    "hyperfoam_exponent_2": 1.0,
                },
                "derived_quantities": {"equilibrium_shear_modulus_pa": 10000.0},
            },
            "column_bed": {
                "anchor_bottom_m": bottom,
                "rest_length_m": rest,
                "area_m2": area,
                "neighbors": neighbors,
                "spacing_m": 0.01,
                "top_bonded": [True] * len(rest),
            },
            "instron_fixtures": {"fullfoot_last": fixture(range(3)), "rearfoot_punch": fixture(range(2))},
            "visual_meshes": {
                "fullfoot_last": {
                    "vertices_m": [[0.0, 0.0, 0.06], [0.01, 0.0, 0.06], [0.0, 0.01, 0.06]],
                    "triangles": [[0, 1, 2]],
                },
                "midsole": {
                    "vertices_m": [[0.0, 0.0, 0.02], [0.01, 0.0, 0.02], [0.0, 0.01, 0.02]],
                    "triangles": [[0, 1, 2]],
                },
            },
            "validation": {},
            "provenance": {},
            "identification": {"calibration": "source geometry; invalid after edit"},
        }

    def test_long_variant_preserves_xy_topology_and_original_top(self):
        """Extend every spring by one quarter while keeping its upper surface fixed."""
        result = geometry_variant(self.source, "long125")
        original = self.source["column_bed"]
        changed = result["column_bed"]
        np.testing.assert_array_equal(
            np.asarray(changed["anchor_bottom_m"])[:, :2], np.asarray(original["anchor_bottom_m"])[:, :2]
        )
        np.testing.assert_allclose(changed["rest_length_m"], np.asarray(original["rest_length_m"]) * 1.25)
        np.testing.assert_allclose(
            np.asarray(changed["anchor_bottom_m"])[:, 2] + changed["rest_length_m"],
            np.asarray(original["anchor_bottom_m"])[:, 2] + original["rest_length_m"],
        )
        np.testing.assert_array_equal(changed["neighbors"], original["neighbors"])
        for key, old_fixture in self.source["instron_fixtures"].items():
            new_fixture = result["instron_fixtures"][key]
            np.testing.assert_array_equal(new_fixture["carrier_anchor_m"], old_fixture["carrier_anchor_m"])
            np.testing.assert_array_equal(new_fixture["foam_free_top_m"], old_fixture["foam_free_top_m"])
            self.assertNotEqual(new_fixture["foam_bottom_m"], old_fixture["foam_bottom_m"])
        self.assertNotIn("identification", result)
        validate_artifact(result)

    def test_block_variant_uses_flat_forty_mm_bed(self):
        """Give every original XY spring site a flat 40 mm column."""
        result = geometry_variant(self.source, "block40")
        original = self.source["column_bed"]
        changed = result["column_bed"]
        np.testing.assert_array_equal(
            np.asarray(changed["anchor_bottom_m"])[:, :2], np.asarray(original["anchor_bottom_m"])[:, :2]
        )
        np.testing.assert_array_equal(
            np.asarray(changed["anchor_bottom_m"])[:, 2], np.zeros(len(original["rest_length_m"]))
        )
        np.testing.assert_array_equal(changed["rest_length_m"], np.full(len(original["rest_length_m"]), 0.04))
        fixture = result["instron_fixtures"]["fullfoot_last"]
        np.testing.assert_array_equal(fixture["foam_free_top_m"], np.full(len(fixture["rest_length_m"]), 0.04))
        np.testing.assert_array_equal(
            fixture["carrier_anchor_m"], self.source["instron_fixtures"]["fullfoot_last"]["carrier_anchor_m"]
        )
        self.assertNotIn("rearfoot_punch", result["instron_fixtures"])
        self.assertNotIn("identification", result)
        validate_artifact(result)

    def test_reference_variant_retains_original_artifact_data(self):
        """Keep reference material conditions on the unchanged source geometry."""
        result = geometry_variant(self.source, "reference")
        self.assertEqual(result["column_bed"], self.source["column_bed"])
        self.assertEqual(result["instron_fixtures"], self.source["instron_fixtures"])


if __name__ == "__main__":
    unittest.main()
