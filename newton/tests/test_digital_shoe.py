# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for portable Digital Shoe artifacts and reports."""

import copy
import json
import os
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from projects.digital_instron_v2.export_digital_shoe import build_artifact
from projects.digital_instron_v2.phase1 import evaluate
from projects.digital_shoe.acquisition import validate_acquisition_manifest
from projects.digital_shoe.artifact import load_artifact, validate_artifact
from projects.digital_shoe.report import render_html

MANIFEST = Path("DigitalInstron/manifest_v2.json")


def _tiny_artifact() -> dict:
    return {
        "schema_version": "digital_shoe_1",
        "shoe": {"id": "tiny<&shoe", "model_scope": "synthetic test"},
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "length_unit": "m",
            "force_unit": "N",
            "origin": "test",
        },
        "constitutive_model": {
            "type": "effective_hyperfoam_maxwell_pasternak_foundation",
            "parameters": {
                "instantaneous_shear_modulus_pa": 19000.0,
                "hyperfoam_exponent": 5.1,
                "equilibrium_fraction": 0.11,
                "pasternak_n_per_m": 900.0,
                "effective_poisson_ratio": 0.3,
                "maxwell_relaxation_time_s": 0.08,
            },
        },
        "column_bed": {
            "anchor_bottom_m": [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]],
            "rest_length_m": [0.02, 0.02],
            "area_m2": [0.0001, 0.0001],
            "neighbors": [[1, -1, -1, -1], [0, -1, -1, -1]],
            "spacing_m": 0.01,
        },
        "identification": {"passed_all_declared_gates": False},
        "validation": {
            "claim_boundary": "synthetic only",
            "curves": [
                {
                    "name": "held_out<&",
                    "fixture": "test",
                    "time_s": [0.0, 0.5, 1.0],
                    "displacement_m": [0.0, 0.01, 0.0],
                    "measured_force_n": [0.0, 100.0, 0.0],
                    "predicted_force_n": [0.0, 90.0, 0.0],
                    "metrics": {
                        "peak_force_error": 0.1,
                        "force_rmse_relative": 0.05,
                        "hysteresis_error": 0.2,
                        "measured_peak_force_n": 100.0,
                        "passed": False,
                    },
                }
            ],
        },
        "provenance": {"source_files": [{"name": "synthetic.csv", "role": "test", "sha256": "0" * 64}]},
    }


class _Parser(HTMLParser):
    pass


class TestDigitalShoeArtifact(unittest.TestCase):
    def test_loads_independently_of_current_directory(self):
        """Load a complete artifact without resolving paths from the current directory."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digital_shoe.json"
            path.write_text(json.dumps(_tiny_artifact()))
            previous = Path.cwd()
            try:
                os.chdir("/")
                shoe = load_artifact(path)
            finally:
                os.chdir(previous)
        self.assertEqual(shoe.shoe_id, "tiny<&shoe")
        self.assertEqual(len(shoe.column_bed.rest_length_m), 2)

    def test_rejects_unknown_schema_and_self_neighbor(self):
        """Reject unknown schema versions and invalid column topology."""
        unknown = _tiny_artifact()
        unknown["schema_version"] = "digital_shoe_99"
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            validate_artifact(unknown)
        self_neighbor = _tiny_artifact()
        self_neighbor["column_bed"]["neighbors"][0][0] = 0
        with self.assertRaisesRegex(ValueError, "self-neighbor"):
            validate_artifact(self_neighbor)

    def test_renders_deterministic_visible_failure_report(self):
        """Render byte-stable escaped HTML that keeps failed gates visible."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digital_shoe.json"
            path.write_text(json.dumps(_tiny_artifact()))
            shoe = load_artifact(path)
            first = render_html(shoe)
            second = render_html(shoe)
        self.assertEqual(first, second)
        self.assertIn("RESEARCH BASELINE", first)
        self.assertIn("SOME DECLARED GATES FAILED", first)
        self.assertIn("tiny&lt;&amp;shoe", first)
        self.assertNotIn("tiny<&shoe", first)
        parser = _Parser()
        parser.feed(first)
        parser.close()

    def test_embeds_experiment_gifs_without_external_paths(self):
        """Embed all experiment loops as deterministic data URIs in the report."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "digital_shoe.json"
            artifact_path.write_text(json.dumps(_tiny_artifact()))
            for mode in ("instron", "drop", "rocker"):
                (root / f"{mode}.gif").write_bytes(b"GIF89a" + mode.encode())
            shoe = load_artifact(artifact_path)
            first = render_html(shoe, media_dir=root)
            second = render_html(shoe, media_dir=root)
        self.assertEqual(first, second)
        self.assertEqual(first.count("data:image/gif;base64,"), 3)
        self.assertNotIn(directory, first)
        self.assertIn("Mechanical experiment loops", first)

    def test_validates_physical_holdout_manifest(self):
        """Accept the planned acquisition matrix and reject leakage across splits."""
        example_path = Path("projects/digital_shoe/acquisition_manifest.example.json")
        data = json.loads(example_path.read_text())
        validate_acquisition_manifest(data)
        leaked = copy.deepcopy(data)
        leaked["splits"]["validate_acquisition_ids"].append(leaked["splits"]["train_acquisition_ids"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_acquisition_manifest(leaked)

    def test_exports_whole_bed_without_absolute_paths(self):
        """Export all shoe columns, fixture mappings, held-out curves, and portable provenance."""
        report = evaluate(MANIFEST, evaluations=10, write_report=False)
        artifact = build_artifact(MANIFEST, report, shoe_id="test_shoe")
        validate_artifact(artifact)
        self.assertEqual(artifact["column_bed"]["column_count"], 910)
        self.assertEqual(artifact["instron_fixtures"]["fullfoot_last"]["column_count"], 611)
        self.assertGreater(artifact["visual_meshes"]["midsole"]["vertex_count"], 7000)
        self.assertGreater(artifact["visual_meshes"]["fullfoot_last"]["vertex_count"], 7000)
        self.assertFalse(artifact["identification"]["passed_all_declared_gates"])
        encoded = json.dumps(artifact, sort_keys=True)
        self.assertNotIn(str(Path.cwd()), encoded)
        roles = {item["role"] for item in artifact["provenance"]["source_files"]}
        self.assertIn("rearfoot_140ms_raw_measurement", roles)
        self.assertIn("fullfoot_185ms_raw_measurement", roles)


if __name__ == "__main__":
    unittest.main()
