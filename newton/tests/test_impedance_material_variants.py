# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test fixed material-only snapshots with a tiny independent shoe fixture."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe.artifact import load_artifact
from projects.digital_shoe.contact import pasternak_coupling, pasternak_coupling_numpy
from projects.digital_shoe.material import (
    hyperfoam_pressure,
    hyperfoam_pressure_numpy,
    maxwell_coefficients,
    maxwell_coefficients_numpy,
    maxwell_step,
    maxwell_step_numpy,
)
from projects.digital_shoe.runtime import FoundationParams, set_material_block
from projects.impedance_instron.simple.material_variants import build_material_variants, material_artifact_identity


def _tiny_shoe() -> dict:
    """Build two unequal columns with independent physical and claim metadata."""
    bed = {
        "anchor_bottom_m": [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.002]],
        "rest_length_m": [0.02, 0.03],
        "area_m2": [0.0001, 0.0002],
        "neighbors": [[1, -1, -1, -1], [0, -1, -1, -1]],
        "spacing_m": 0.02,
    }
    fixture = {
        "carrier_anchor_m": [[-0.01, 0.0, 0.08], [0.01, 0.0, 0.08]],
        "foam_free_top_m": [0.02, 0.03],
        "foam_bottom_m": [0.0, 0.0],
        "rest_length_m": [0.02, 0.03],
        "area_m2": bed["area_m2"],
        "neighbors": bed["neighbors"],
        "spacing_m": 0.02,
        "mount_contact": {"bond_n_per_m": 30.0},
    }
    return {
        "schema_version": "digital_shoe_1",
        "shoe": {"id": "two-column-test", "model_scope": "fixture only", "mass_kg": 0.2, "side": "left"},
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "length_unit": "m",
            "force_unit": "N",
            "origin": "test",
        },
        "constitutive_model": {
            "type": "effective_hyperfoam_maxwell_pasternak_foundation",
            "unknown_physical_option": {"contact_scale": 1.0},
            "parameters": {
                "instantaneous_shear_modulus_pa": 100000.0,
                "hyperfoam_exponent": 2.4,
                "instantaneous_shear_modulus_2_pa": 20000.0,
                "hyperfoam_exponent_2": -1.2,
                "equilibrium_fraction": 0.4,
                "pasternak_n_per_m": 1200.0,
                "effective_poisson_ratio": 0.1,
                "maxwell_relaxation_time_s": 0.08,
            },
            "derived_quantities": {
                "hyperfoam_term_count": 2,
                "pasternak_rule": "k_i = equilibrium_shear_modulus_pa * rest_length_m[i]",
                "pasternak_n_per_m_is_fitted": False,
                "equilibrium_shear_modulus_pa": 48000.0,
                "equilibrium_shear_modulus_term_1_pa": 40000.0,
                "equilibrium_shear_modulus_term_2_pa": 8000.0,
                "small_strain_compressive_modulus_pa": 105600.0,
                "pasternak_n_per_m_min": 960.0,
                "pasternak_n_per_m_max": 1440.0,
                "pasternak_n_per_m_mean": 1200.0,
                "pasternak_n_per_m_by_column": [960.0, 1440.0],
                "pasternak_n_per_m_by_fixture": {"tiny": [960.0, 1440.0]},
            },
        },
        "column_bed": bed,
        "visual_meshes": {"sole": {"vertices_m": [[0, 0, 0], [0.02, 0, 0], [0, 0.03, 0]], "triangles": [[0, 1, 2]]}},
        "instron_fixtures": {"tiny": fixture},
        "mount": {"translation_m": [0.0, 0.0, 0.05], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "unknown_future_physics": {"orientation": "toe_forward", "constraint": [1, 0, 1]},
        "identification": {
            "backend": "test",
            "metrics": {"force_rmse_relative": 0.001},
            "gates": {"passed": True},
            "passed_all_declared_gates": True,
            "fixture_specific_parameters": [],
            "unknown_fixture_setting": {"stiffness": 7.0},
        },
        "validation": {
            "scope": "baseline only",
            "curves": [{"metrics": {"passed": True}}],
            "claim_boundary": "a fixture, not a validated shoe",
            "unknown_physical_setting": 2.0,
        },
        "provenance": {"generator": "test", "source_files": [], "unknown_mount_setting": [1, 2, 3]},
    }


def _history(material) -> tuple[np.ndarray, np.ndarray, float]:
    """Run shared native host expressions over compression followed by a hold."""
    strain = np.array([0.0, 0.08, 0.25, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0])
    fraction = material.equilibrium_fraction
    poisson = material.effective_poisson_ratio
    pressure = hyperfoam_pressure_numpy(
        strain,
        material.instantaneous_shear_modulus_pa * fraction,
        material.hyperfoam_exponent,
        material.instantaneous_shear_modulus_2_pa * fraction,
        material.hyperfoam_exponent_2,
        poisson / (1 - 2 * poisson),
        1 - 2 * poisson,
        0.05,
    )
    decay, ramp = maxwell_coefficients_numpy(0.01, material.maxwell_relaxation_time_s)
    q, previous = 0.0, 0.0
    overstress = []
    for peq in pressure:
        q = maxwell_step_numpy(q, peq, previous, (1.0 - fraction) / fraction, decay, ramp)
        overstress.append(q)
        previous = peq
    coupling = pasternak_coupling_numpy(0.02, 0.03, material.equilibrium_shear_modulus_pa)
    return pressure, np.asarray(overstress), coupling


@wp.kernel
def _native_history(params: FoundationParams, out: wp.array2d[float]):
    """Evaluate the actual shared Warp law with a fresh zero-history state."""
    q = float(0.0)
    previous = float(0.0)
    decay, ramp = maxwell_coefficients(0.01, params.tau_s)
    for index in range(8):
        strain = wp.min(float(index) * 0.1, 0.5)
        peq = hyperfoam_pressure(
            strain,
            params.g_eq,
            params.alpha,
            params.g_eq2,
            params.alpha2,
            params.beta,
            params.one_minus_two_poisson,
            0.05,
        )
        q = maxwell_step(q, peq, previous, params.overstress, decay, ramp)
        previous = peq
        out[index, 0] = peq
        out[index, 1] = q
        out[index, 2] = pasternak_coupling(0.02, 0.03, params.g_eq + params.g_eq2)


class TestImpedanceMaterialVariants(unittest.TestCase):
    """Verify offline isolation, provenance, physical identity, and shared laws."""

    def setUp(self):
        """Create an independent source artifact for every test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.json"
        self.data = _tiny_shoe()
        self.source.write_text(json.dumps(self.data, indent=2) + "\n")
        self.original = self.source.read_bytes()
        self.output = self.root / "variants"

    def _build(self, **kwargs):
        return build_material_variants(self.source, self.output, **kwargs)

    def test_baseline_first_and_preserve_source_and_every_physical_field(self):
        """Copy the baseline exactly and preserve unknown physical fields."""
        records = self._build()
        self.assertEqual(len(records), 5)
        self.assertEqual(
            [record["type"] for record in records], ["baseline", "modulus", "modulus", "relaxation", "relaxation"]
        )
        self.assertTrue(records[0]["baseline"])
        self.assertEqual(Path(records[0]["path"]).read_bytes(), self.original)
        self.assertEqual(self.source.read_bytes(), self.original)
        baseline = material_artifact_identity(self.source)
        for record in records:
            with self.subTest(variant=record["id"]):
                path = Path(record["path"])
                snapshot = json.loads(path.read_text())
                self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(record["source_sha256"], hashlib.sha256(self.original).hexdigest())
                self.assertEqual(record["nonmaterial_identity"], baseline["geometry_sha256"])
                self.assertEqual(record["material_identity"], material_artifact_identity(path)["material_sha256"])
                self.assertEqual(material_artifact_identity(path)["geometry"], baseline["geometry"])
                for key in ("mount", "unknown_future_physics", "column_bed", "instron_fixtures", "visual_meshes"):
                    self.assertEqual(snapshot[key], self.data[key])
                self.assertEqual(snapshot["shoe"]["mass_kg"], self.data["shoe"]["mass_kg"])
        json.dumps(records, allow_nan=False)

    def test_scale_both_terms_and_all_derived_coupling_reports(self):
        """Scale both Ogden-Hill terms and the native material-pinned shear layer."""
        records = self._build()
        baseline = records[0]
        for record, factor in zip(records[1:3], (0.75, 1.25), strict=True):
            parameters = record["parameters"]
            for key, value in baseline["parameters"].items():
                amplitude = key in {
                    "instantaneous_shear_modulus_pa",
                    "instantaneous_shear_modulus_2_pa",
                    "pasternak_n_per_m",
                }
                self.assertAlmostEqual(parameters[key], value * factor if amplitude else value)
            for key in (
                "equilibrium_shear_modulus_pa",
                "equilibrium_shear_modulus_term_1_pa",
                "equilibrium_shear_modulus_term_2_pa",
                "small_strain_compressive_modulus_pa",
                "pasternak_n_per_m_min",
                "pasternak_n_per_m_max",
                "pasternak_n_per_m_mean",
            ):
                self.assertAlmostEqual(record["derived_quantities"][key], baseline["derived_quantities"][key] * factor)
                self.assertAlmostEqual(
                    record["artifact_derived_quantities"][key], baseline["derived_quantities"][key] * factor
                )
            expected = np.asarray([960.0, 1440.0]) * factor
            np.testing.assert_allclose(record["derived_quantities"]["pasternak_n_per_m_by_column"], expected)
            np.testing.assert_allclose(
                record["artifact_derived_quantities"]["pasternak_n_per_m_by_fixture"]["tiny"], expected
            )
            self.assertEqual(record["constitutive_type"], baseline["constitutive_type"])

    def test_relaxation_changes_only_tau_in_material(self):
        """Leave every other parameter and derived report untouched for tau cases."""
        records = self._build()
        baseline_model = json.loads(Path(records[0]["path"]).read_text())["constitutive_model"]
        for record, factor in zip(records[3:], (0.5, 2.0), strict=True):
            actual = json.loads(Path(record["path"]).read_text())["constitutive_model"]
            expected = copy.deepcopy(baseline_model)
            expected["parameters"]["maxwell_relaxation_time_s"] *= factor
            self.assertEqual(actual, expected)
            self.assertEqual(record["derived_quantities"], records[0]["derived_quantities"])

    def test_shared_numpy_law_scales_history_and_tau_changes_hold_only(self):
        """Prove scaling and relaxation effects using the runtime's shared law source."""
        records = self._build()
        histories = [_history(load_artifact(record["path"]).material) for record in records]
        pressure, overstress, coupling = histories[0]
        for actual, factor in zip(histories[1:3], (0.75, 1.25), strict=True):
            np.testing.assert_allclose(actual[0], pressure * factor, rtol=1.0e-13)
            np.testing.assert_allclose(actual[1], overstress * factor, rtol=1.0e-13, atol=1.0e-10)
            self.assertAlmostEqual(actual[2], coupling * factor)
        for actual, factor in zip(histories[3:], (0.5, 2.0), strict=True):
            np.testing.assert_array_equal(actual[0], pressure)
            self.assertEqual(actual[2], coupling)
            self.assertFalse(np.allclose(actual[1], overstress))
            self.assertAlmostEqual(actual[1][5] / actual[1][4], np.exp(-0.01 / (0.08 * factor)))
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_compressive_tangent_matches_shared_law_with_nonzero_poisson(self):
        """Verify the derived tangent against native pressure at small compression."""
        records = self._build()
        material = load_artifact(records[0]["path"]).material
        poisson = material.effective_poisson_ratio
        strain = 1.0e-7
        pressure = hyperfoam_pressure_numpy(
            strain,
            material.instantaneous_shear_modulus_pa * material.equilibrium_fraction,
            material.hyperfoam_exponent,
            material.instantaneous_shear_modulus_2_pa * material.equilibrium_fraction,
            material.hyperfoam_exponent_2,
            poisson / (1 - 2 * poisson),
            1 - 2 * poisson,
            0.05,
        )
        self.assertAlmostEqual(
            pressure / strain / records[0]["derived_quantities"]["small_strain_compressive_modulus_pa"], 1.0, places=6
        )

    def test_native_warp_law_has_the_same_effect_with_fresh_history(self):
        """Check true CPU Warp material constants and history without a mock rig."""
        records = self._build()
        histories = []
        for record in records:
            params = FoundationParams()
            set_material_block(params, load_artifact(record["path"]).material)
            out = wp.zeros((8, 3), dtype=float, device="cpu")
            wp.launch(_native_history, 1, inputs=[params, out], device="cpu")
            histories.append(out.numpy())
        for actual, factor in zip(histories[1:3], (0.75, 1.25), strict=True):
            np.testing.assert_allclose(actual, histories[0] * factor, rtol=5.0e-6, atol=0.02)
        for actual, factor in zip(histories[3:], (0.5, 2.0), strict=True):
            np.testing.assert_array_equal(actual[:, [0, 2]], histories[0][:, [0, 2]])
            self.assertFalse(np.allclose(actual[:, 1], histories[0][:, 1]))
            self.assertAlmostEqual(actual[7, 1] / actual[6, 1], np.exp(-0.01 / (0.08 * factor)), places=6)

    def test_synthetic_cases_do_not_inherit_baseline_qualification(self):
        """Keep baseline fit claims only in the original and baseline snapshot."""
        records = self._build()
        for record in records[1:]:
            data = json.loads(Path(record["path"]).read_text())
            self.assertEqual(record["qualification"], "not_validated")
            for container in ("validation", "identification"):
                self.assertEqual(data[container]["status"], "not_validated")
                self.assertEqual(data[container]["baseline_reference"]["sha256"], records[0]["sha256"])
                for claim in ("curves", "metrics", "gates", "passed_all_declared_gates"):
                    self.assertNotIn(claim, data[container])
            self.assertEqual(data["identification"]["unknown_fixture_setting"], {"stiffness": 7.0})
            self.assertEqual(data["validation"]["unknown_physical_setting"], 2.0)
            self.assertEqual(data["provenance"]["unknown_mount_setting"], [1, 2, 3])
            self.assertIn("not a newly calibrated or validated shoe", record["warning"])
        self.assertTrue(json.loads(Path(records[0]["path"]).read_text())["identification"]["passed_all_declared_gates"])

    def test_imported_material_can_change_known_metadata_not_geometry(self):
        """Accept a user material and retain its source bytes without endorsing it."""
        imported = copy.deepcopy(self.data)
        imported["constitutive_model"]["parameters"]["maxwell_relaxation_time_s"] = 0.3
        imported["shoe"]["id"] = "other-export"
        imported["identification"]["metrics"] = {"arbitrary_source_metric": 4.0}
        imported["validation"]["curves"] = []
        path = self.root / "imported.json"
        path.write_text(json.dumps(imported))
        records = self._build(modulus_multipliers=(), relaxation_multipliers=(), material_paths=(path,))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["type"], "imported")
        self.assertEqual(Path(records[1]["path"]).read_bytes(), path.read_bytes())
        self.assertEqual(records[1]["qualification"], "source_claims_not_requalified")
        self.assertEqual(records[0]["nonmaterial_identity"], records[1]["nonmaterial_identity"])
        self.assertNotEqual(records[0]["material_identity"], records[1]["material_identity"])

    def test_imports_reject_every_nonmaterial_change_including_unknown_fields(self):
        """Fail closed on physical changes even inside metadata-like containers."""
        mutations = (
            ("shoe", "mass_kg", 0.3),
            ("shoe", "side", "right"),
            ("coordinate_system", "handedness", "left"),
            ("mount", "translation_m", [1, 0, 0]),
            ("unknown_future_physics", "orientation", "heel_forward"),
            ("unknown_future_physics", "constraint", [True, 0, 1]),
            ("identification", "unknown_fixture_setting", {"stiffness": 8.0}),
            ("validation", "unknown_physical_setting", 9.0),
            ("provenance", "unknown_mount_setting", [9, 9, 9]),
            ("constitutive_model", "unknown_physical_option", {"contact_scale": 0.2}),
            ("column_bed", "anchor_bottom_m", [[-0.01, 0, 0.001], [0.01, 0, 0.002]]),
        )
        for container, key, value in mutations:
            with self.subTest(container=container, key=key):
                data = copy.deepcopy(self.data)
                data[container][key] = value
                path = self.root / "mismatch.json"
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "identical geometry and all nonmaterial"):
                    self._build(material_paths=(path,))
                self.assertFalse(self.output.exists())
        data = copy.deepcopy(self.data)
        data["instron_fixtures"]["tiny"]["mount_contact"]["bond_n_per_m"] += 1.0
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "identical geometry and all nonmaterial"):
            self._build(material_paths=(path,))

    def test_reject_invalid_factors_and_never_overwrite_output(self):
        """Reject invalid factors and preserve existing output files on failure."""
        for value in (0.0, -1.0, float("nan"), float("inf"), 1.0e40, 1.0e-60, True, "1.2"):
            for key in ("modulus_multipliers", "relaxation_multipliers"):
                with self.subTest(value=value, key=key), self.assertRaises(ValueError):
                    self._build(**{key: (value,)})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._build(modulus_multipliers=(0.75, 0.75))
        self.output.mkdir()
        marker = self.output / "owned.txt"
        marker.write_text("do not replace")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self._build()
        self.assertEqual(marker.read_text(), "do not replace")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_reject_invalid_and_float32_overflowing_material(self):
        """Reject nonfinite, malformed, and float32-unsafe material constants."""
        changes = (
            ("instantaneous_shear_modulus_pa", float("nan")),
            ("instantaneous_shear_modulus_pa", 1.0e40),
            ("instantaneous_shear_modulus_pa", -100.0),
            ("instantaneous_shear_modulus_2_pa", -1.0),
            ("hyperfoam_exponent", 0.0),
            ("hyperfoam_exponent_2", float("inf")),
            ("equilibrium_fraction", 0.0),
            ("equilibrium_fraction", 1.1),
            ("equilibrium_fraction", 1.0e-44),
            ("effective_poisson_ratio", 0.5),
            ("maxwell_relaxation_time_s", 0.0),
            ("maxwell_relaxation_time_s", 1.0e-80),
            ("pasternak_n_per_m", -1.0),
            ("instantaneous_shear_modulus_pa", True),
            ("instantaneous_shear_modulus_pa", "100000"),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data["constitutive_model"]["parameters"][key] = value
                self.source.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self._build()
                self.assertFalse(self.output.exists())
        self.source.write_bytes(self.original)
        with self.assertRaises(ValueError):
            self._build(modulus_multipliers=(1.0e35,))
        self.assertEqual(list(self.output.iterdir()) if self.output.exists() else [], [])

    def test_reject_malformed_geometry_and_reporting_without_partial_snapshots(self):
        """Reject geometry coercion, unknown contracts, and stale derived values."""
        malformed = []
        for container, key, value in (
            ("column_bed", "spacing_m", float("inf")),
            ("column_bed", "spacing_m", 1.0e-30),
            ("column_bed", "area_m2", [0.0, 0.0001]),
            ("column_bed", "rest_length_m", [0.02, float("nan")]),
            ("column_bed", "rest_length_m", [[0.02], [0.03]]),
            ("column_bed", "neighbors", [[1.5, -1, -1, -1], [0, -1, -1, -1]]),
            ("column_bed", "neighbors", [[1, -1, -1, -1], [0, -1, -1, 2**40]]),
            ("column_bed", "neighbors", [[0, -1, -1, -1], [0, -1, -1, -1]]),
            ("constitutive_model", "type", "other_law"),
            ("constitutive_model", "parameters", []),
            ("constitutive_model", "derived_quantities", []),
        ):
            data = copy.deepcopy(self.data)
            data[container][key] = value
            malformed.append(data)
        data = copy.deepcopy(self.data)
        data["constitutive_model"]["derived_quantities"]["equilibrium_shear_modulus_pa"] = 123.0
        malformed.append(data)
        data = copy.deepcopy(self.data)
        data["constitutive_model"]["parameters"]["pasternak_n_per_m"] = 123.0
        malformed.append(data)
        malformed.extend([[], {"schema_version": "unknown"}])
        for index, data in enumerate(malformed):
            with self.subTest(index=index):
                self.source.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self._build()
                self.assertFalse(self.output.exists())

    def test_reject_duplicate_json_fields(self):
        """Reject ambiguous duplicate JSON keys rather than silently choosing one."""
        self.source.write_text(self.original.decode().replace('"schema_version":', '"shoe": {}, "schema_version":'))
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            self._build()

    def test_legacy_single_term_and_report_only_scalar_remain_usable(self):
        """Keep old single-term artifacts valid without inventing a second term."""
        model = self.data["constitutive_model"]
        del model["derived_quantities"]
        del model["parameters"]["instantaneous_shear_modulus_2_pa"]
        del model["parameters"]["hyperfoam_exponent_2"]
        model["parameters"]["pasternak_n_per_m"] = 999.0
        self.source.write_text(json.dumps(self.data))
        records = self._build()
        self.assertEqual(records[1]["parameters"]["instantaneous_shear_modulus_2_pa"], 0.0)
        self.assertEqual(records[1]["parameters"]["pasternak_n_per_m"], 999.0 * 0.75)
        self.assertEqual(records[1]["derived_quantities"]["equilibrium_shear_modulus_pa"], 30000.0)
        self.assertEqual(records[1]["derived_quantities"]["pasternak_n_per_m_by_column"], [600.0, 900.0])

    def test_allow_roundoff_in_optional_per_column_derived_reports(self):
        """Accept numerical report roundoff without silently accepting stale material."""
        derived = self.data["constitutive_model"]["derived_quantities"]
        derived["pasternak_n_per_m_by_column"][0] += 1.0e-10
        derived["pasternak_n_per_m_by_fixture"]["tiny"][1] -= 1.0e-10
        self.source.write_text(json.dumps(self.data))
        records = self._build()
        self.assertEqual(len(records), 5)

    def test_empty_factor_lists_write_only_baseline_into_empty_directory(self):
        """Accept the baseline-only smoke case without synthetic perturbations."""
        self.output.mkdir()
        records = self._build(modulus_multipliers=(), relaxation_multipliers=())
        self.assertEqual([record["id"] for record in records], ["baseline"])
        self.assertEqual(Path(records[0]["path"]).read_bytes(), self.original)

    def test_deterministic_and_independent_of_output_location(self):
        """Write byte-stable synthetic artifacts whose provenance uses relative baseline paths."""
        first = self._build()
        second = build_material_variants(self.source, self.root / "other")
        for a, b in zip(first, second, strict=True):
            self.assertEqual(a["sha256"], b["sha256"])
            self.assertEqual(a["material_identity"], b["material_identity"])
            self.assertEqual(Path(a["path"]).read_bytes(), Path(b["path"]).read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
