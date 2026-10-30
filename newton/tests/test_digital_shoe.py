# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for portable Digital Shoe artifacts and reports."""

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import warp as wp

from projects.digital_instron_v2 import export_digital_shoe
from projects.digital_instron_v2.core import CALIBRATED_MATERIAL
from projects.digital_instron_v2.registration import load_last, register_last
from projects.digital_instron_v2.sweep_fit import DEFAULT_LAST, load_grid
from projects.digital_shoe.artifact import DEFAULT_SHOE_MASS_KG, load_artifact, validate_artifact
from projects.digital_shoe.report import render_html
from projects.digital_shoe.showcase import Example


def _tiny_artifact() -> dict:
    return {
        "schema_version": "digital_shoe_2",
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
            "top_bonded": [True, True],
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

    def test_reads_shoe_mass_or_defaults(self):
        """Read the shoe mass from the artifact and fall back to the documented default."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digital_shoe.json"
            data = _tiny_artifact()
            path.write_text(json.dumps(data))
            self.assertEqual(load_artifact(path).mass_kg, DEFAULT_SHOE_MASS_KG)
            data["shoe"]["mass_kg"] = 0.173
            path.write_text(json.dumps(data))
            self.assertEqual(load_artifact(path).mass_kg, 0.173)
            data["shoe"]["mass_kg"] = 0.0
            with self.assertRaisesRegex(ValueError, "mass_kg"):
                validate_artifact(data)

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
        short_bond = _tiny_artifact()
        short_bond["column_bed"]["top_bonded"] = [True]
        with self.assertRaisesRegex(ValueError, "top_bonded has shape"):
            validate_artifact(short_bond)

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
        self.assertIn("3. Examples", first)
        self.assertIn(".experiment-grid { display: block; }", first)
        self.assertIn("An 80 kg body-weight load", first)
        self.assertIn("Blue: 0 mm", first)
        self.assertIn("Red: 20+ mm", first)
        self.assertIn("two Ogden", first)
        self.assertLess(first.index("1. Methods"), first.index("2. Results"))
        self.assertLess(first.index("2. Results"), first.index("3. Examples"))

    def test_keeps_status_visible_and_navigation_complete_without_media(self):
        """Keep warnings above the methods and link to every report section."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digital_shoe.json"
            path.write_text(json.dumps(_tiny_artifact()))
            report = render_html(load_artifact(path))
        self.assertLess(report.index("SOME DECLARED GATES FAILED"), report.index('id="methods"'))
        self.assertLess(report.index("synthetic only"), report.index('id="methods"'))
        for section in ("methods", "results", "examples", "reproduce"):
            self.assertIn(f'href="#{section}"', report)
            self.assertEqual(report.count(f'id="{section}"'), 1)
        self.assertIn("No recordings are embedded yet", report)
        self.assertNotIn("@import", report)
        self.assertIn('<details class="derivation"><summary>Model equations and assumptions</summary>', report)
        self.assertIn("Rebuild this report only", report)
        self.assertIn("--media-dir", report)

    def test_distinguishes_response_curves_without_color(self):
        """Give response plots dashed predictions and accessible axis labels."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digital_shoe.json"
            path.write_text(json.dumps(_tiny_artifact()))
            report = render_html(load_artifact(path))
        self.assertIn('stroke-dasharray="7 4"', report)
        self.assertIn('aria-label="Measured and predicted force versus Compression [mm]"', report)
        self.assertIn('aria-label="Measured and predicted force versus Time [s]"', report)
        self.assertIn('role="region" aria-label="Held-out validation metrics"', report)
        self.assertIn("repeat(2, minmax(0, 1fr))", report)
        self.assertIn("@media (max-width: 760px)", report)

    def test_export_names_the_missing_substeps_of_an_old_fit(self):
        """Refuse a fit written before ``substeps`` was recorded, with a message that names it."""
        shoe = Path("DigitalInstron/puma-fast-r-nitro-elite-3")
        with tempfile.TemporaryDirectory() as directory:
            fit_path = Path(directory) / "material_fit.json"
            fit_path.write_text(json.dumps({"shoe": shoe.name, "side": "left"}))
            with self.assertRaisesRegex(ValueError, "substeps"):
                export_digital_shoe.build_artifact(shoe, fit_path, Path(directory))

    @unittest.skipUnless(wp.is_cuda_available(), "the validation replay needs CUDA")
    def test_exports_fitted_shoe_without_absolute_paths(self):
        """Export the shoe grid, fitted material, registered last and held-out replay as a portable artifact."""
        shoe = Path("DigitalInstron/puma-fast-r-nitro-elite-3")
        grid = load_grid(shoe / "left" / "column_grid.npz")
        registration = register_last(grid, load_last(DEFAULT_LAST), modulus_pa=5.0e5, preload_n=20.0)
        preload = {"modulus_pa": 5.0e5, "preload_n": 20.0}
        fit = {
            "shoe": shoe.name,
            "side": "left",
            "sweep": "synthetic",
            "material": asdict(CALIBRATED_MATERIAL),
            "registration": {"pose": registration.pose, "qc": {"preload": preload}},
            "conditions": {},
            "substeps": 1,
            "start_offset_m": 0.002,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "material_fit.json").write_text(json.dumps(fit))
            condition = {"stance_ms": 200, "target_n": 800, "role": "condition", "validate_path": "validate.csv"}
            (root / "conditions.json").write_text(json.dumps({"conditions": [condition]}))
            time_s = np.arange(501) * 1.0e-3
            displacement = 0.004 * np.sin(np.pi * time_s / time_s[-1])
            np.savetxt(
                root / "validate.csv",
                np.column_stack((time_s, displacement, 2.0e5 * displacement)),
                delimiter=",",
                header="time_s,displacement_m,force_n",
                comments="",
            )
            with patch.object(export_digital_shoe, "register_last", return_value=registration):
                artifact = export_digital_shoe.build_artifact(
                    shoe, root / "material_fit.json", root, mass_kg=0.173, mass_source="test scale"
                )
        validate_artifact(artifact)
        self.assertEqual(artifact["shoe"]["mass_kg"], 0.173)
        bed = np.asarray(artifact["column_bed"]["anchor_bottom_m"])
        fixture = artifact["instron_fixtures"]["fullfoot_last"]
        self.assertEqual(len(bed), len(grid.slack_m))
        self.assertEqual(fixture["column_count"], int(np.isfinite(registration.surface_m).sum()))
        self.assertEqual(bed[:, 2].min(), 0.0)
        # The bench replays the shoe in the pose the fit stood it on the platen.
        np.testing.assert_array_equal(fixture["bed_bottom_above_platen_m"], registration.bottom_above_platen_m)
        # The foot is bonded to the columns the bench last presses; the rim outside it hangs on shear.
        np.testing.assert_array_equal(artifact["column_bed"]["top_bonded"], np.isfinite(registration.surface_m))
        np.testing.assert_array_equal(artifact["column_bed"]["area_m2"], grid.area_m2)
        # The baked last bottoms out on the first-contact surface the fixture records.
        last = np.asarray(artifact["visual_meshes"]["fullfoot_last"]["vertices_m"])
        anchors = np.asarray(fixture["carrier_anchor_m"])
        self.assertAlmostEqual(last[:, 2].min(), anchors[:, 2].min(), delta=1.0e-3)
        (curve,) = artifact["validation"]["curves"]
        self.assertEqual(curve["name"], "200ms_800N")
        self.assertGreater(max(curve["predicted_force_n"]), 0.0)
        # The curve records the depth the replay drove: stroke plus preload indentation plus start offset.
        np.testing.assert_allclose(
            curve["indenter_depth_m"],
            np.maximum(displacement, 0.0) + registration.qc["preload"]["indentation_m"] + 0.002,
        )
        encoded = json.dumps(artifact, sort_keys=True)
        self.assertNotIn(str(Path.cwd()), encoded)
        self.assertNotIn(directory, encoded)

        right = export_digital_shoe.mirror_artifact(artifact, "0" * 64)
        self.assertEqual(right["shoe"]["id"], "puma_fast_r_nitro_elite_3_right")
        np.testing.assert_array_equal(np.asarray(right["column_bed"]["anchor_bottom_m"]), bed * [1.0, -1.0, 1.0])
        np.testing.assert_array_equal(
            np.asarray(right["column_bed"]["neighbors"]),
            np.asarray(artifact["column_bed"]["neighbors"])[:, [0, 1, 3, 2]],
        )
        self.assertEqual(
            right["instron_fixtures"]["fullfoot_last"]["bed_bottom_above_platen_m"],
            fixture["bed_bottom_above_platen_m"],
        )
        left = export_digital_shoe.mirror_artifact(right, "0" * 64)
        self.assertEqual(left["column_bed"], artifact["column_bed"])
        self.assertEqual(left["instron_fixtures"], artifact["instron_fixtures"])


class TestDigitalShoeRendering(unittest.TestCase):
    def test_instron_renders_connected_nodes_without_midsole_surface(self):
        """Show both spring endpoints and connectors without an opaque midsole mesh."""
        example = MagicMock(mode="instron", sim_time=0.0, history=[], column_count=2, device="cpu")
        with patch("projects.digital_shoe.showcase.wp.launch"):
            Example.render(example)
        example.viewer.log_mesh.assert_not_called()
        example.viewer.log_lines.assert_called_once_with(
            "digital_shoe/columns", example._bottoms, example._tops, example._colors, width=0.003
        )
        self.assertEqual(example.viewer.log_points.call_count, 2)
        example.viewer.log_points.assert_any_call(
            "digital_shoe/spring_bases", example._bottoms, radii=0.0025, colors=example._colors
        )
        example.viewer.log_points.assert_any_call(
            "digital_shoe/spring_tops", example._tops, radii=0.0025, colors=example._colors
        )
        example.viewer.log_state.assert_called_once_with(example.state_0)


if __name__ == "__main__":
    unittest.main()
