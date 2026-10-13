"""Verify fixed-geometry material sweep parameterization."""

import json
import tempfile
import unittest
from pathlib import Path

from .materials_sweep import build_candidate


class TestMaterialsSweep(unittest.TestCase):
    """Check that sweep axes remain independent on a copied shoe."""

    def test_build_candidate_preserves_geometry_and_scales_material_axes(self):
        """Preserve shoe geometry while scaling compression, shear, and relaxation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            target = root / "candidate.json"
            source.write_text(
                json.dumps(
                    {
                        "shoe": {"id": "fr3", "geometry": {"columns": 4}},
                        "column_bed": {"rest_length_m": [0.02, 0.03]},
                        "constitutive_model": {
                            "parameters": {
                                "instantaneous_shear_modulus_pa": 100.0,
                                "instantaneous_shear_modulus_2_pa": 50.0,
                                "equilibrium_fraction": 0.5,
                                "maxwell_relaxation_time_s": 0.01,
                            },
                            "derived_quantities": {"equilibrium_shear_modulus_pa": 75.0},
                        },
                    }
                )
            )
            build_candidate(source, target, compression=2.0, shear=0.5, fraction=0.25, tau=0.02)
            before, after = json.loads(source.read_text()), json.loads(target.read_text())
            self.assertEqual(before["column_bed"], after["column_bed"])
            self.assertEqual(before["shoe"]["geometry"], after["shoe"]["geometry"])
            params = after["constitutive_model"]["parameters"]
            self.assertAlmostEqual(
                (params["instantaneous_shear_modulus_pa"] + params["instantaneous_shear_modulus_2_pa"])
                * params["equilibrium_fraction"],
                150.0,
            )
            self.assertAlmostEqual(params["equilibrium_shear_modulus_override_pa"], 37.5)
            self.assertEqual(params["maxwell_relaxation_time_s"], 0.02)


if __name__ == "__main__":
    unittest.main()
