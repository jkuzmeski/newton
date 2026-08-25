# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for importing accepted official RRA motion into contact analysis."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from projects.gait_c3d.adapters import rra_adjusted_contact_input as adjusted


class TestRRAAdjustedContactInput(unittest.TestCase):
    """Exercise storage units, acceptance checks, publication, and provenance."""

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _write_model(path: Path, *, include_motion_types: bool) -> None:
        document = ET.Element("OpenSimDocument", {"Version": "20302"})
        model = ET.SubElement(document, "Model", {"name": "synthetic"})
        coordinate_set = ET.SubElement(model, "CoordinateSet")
        objects = ET.SubElement(coordinate_set, "objects")
        for name, motion_type in (("angle", "rotational"), ("slide", "translational")):
            coordinate = ET.SubElement(objects, "Coordinate", {"name": name})
            if include_motion_types:
                ET.SubElement(coordinate, "motion_type").text = motion_type
            ET.SubElement(coordinate, "range").text = "-10 10"
            ET.SubElement(coordinate, "locked").text = "false"
        ET.indent(document, space="  ")
        path.write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True) + b"\n")

    @staticmethod
    def _write_storage(path: Path, values: list[tuple[float, float, float]]) -> None:
        rows = "\n".join("\t".join(str(value) for value in row) for row in values)
        path.write_text(
            f"synthetic\nversion=1\nnRows=3\nnColumns=3\ninDegrees=yes\nendheader\ntime\tangle\tslide\n{rows}\n",
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        data_dir = root / "latest"
        rra_dir = root / "rra"
        results = rra_dir / "results"
        inputs = rra_dir / "inputs"
        data_dir.mkdir()
        results.mkdir(parents=True)
        inputs.mkdir()
        original_model = data_dir / "S001_scaled.osim"
        adjusted_model = results / "trial_adjusted.osim"
        self._write_model(original_model, include_motion_types=True)
        self._write_model(adjusted_model, include_motion_types=False)
        source_times = np.asarray([0.0, 1.0, 2.0])
        grf = np.zeros((3, 2, 3))
        grf[:, :, 1] = np.asarray([[100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
        cop = np.full((3, 2, 3), np.nan)
        cop[0, 0] = [0.0, 0.0, 0.0]
        cop[2, 0] = [2.0, 0.0, 2.0]
        cop[0, 1] = [1.0, 0.0, 1.0]
        cop[2, 1] = [3.0, 0.0, 3.0]
        target_markers = np.asarray([[[0.0, 0.0, 0.0]], [[np.nan, np.nan, np.nan]], [[2.0, 4.0, 6.0]]])
        np.savez_compressed(
            data_dir / "analysis.npz",
            schema_version=np.asarray(adjusted._ANALYSIS_SCHEMA),
            times=source_times,
            grf=grf,
            cop=cop,
            free_torque=np.zeros((3, 2, 3)),
            target_markers=target_markers,
            marker_names=np.asarray(["marker"]),
            foot_names=np.asarray(["left", "right"]),
            id_names=np.asarray(["angle", "slide"]),
            motion_types=np.asarray(["rotational", "translational"]),
            id_generalized_forces=np.full((3, 2), 123.0),
        )
        self._write_json(
            data_dir / "manifest.json",
            {"schema_version": adjusted._ANALYSIS_SCHEMA, "status": "source", "runtime": {}},
        )

        q_path = results / "trial_Kinematics_q.sto"
        u_path = results / "trial_Kinematics_u.sto"
        dudt_path = results / "trial_Kinematics_dudt.sto"
        self._write_storage(q_path, [(0.25, 180.0, 2.0), (1.0, 90.0, 3.0), (1.75, 0.0, 4.0)])
        self._write_storage(u_path, [(0.25, 90.0, 5.0), (1.0, 45.0, 6.0), (1.75, 0.0, 7.0)])
        self._write_storage(dudt_path, [(0.25, 360.0, 8.0), (1.0, 180.0, 9.0), (1.75, 0.0, 10.0)])
        setup = inputs / "Setup_RRA.xml"
        setup.write_text("<OpenSimDocument/>\n", encoding="utf-8")
        prepare = {
            "schema_version": adjusted._RRA_SCHEMA,
            "scope": adjusted._RRA_SCOPE,
            "tool_name": "trial",
            "source_inputs": {str(original_model): {"sha256": self._sha256(original_model)}},
            "generated_inputs": {str(setup): {"sha256": self._sha256(setup)}},
        }
        prepare_path = rra_dir / "prepare_manifest.json"
        self._write_json(prepare_path, prepare)
        runtime_path = rra_dir / "run_runtime.json"
        summary_path = rra_dir / "summary.json"
        stable_hashes = adjusted._artifact_hashes(rra_dir, excluded={runtime_path, summary_path})
        runtime = {
            "schema_version": adjusted._RRA_SCHEMA,
            "scope": adjusted._RRA_SCOPE,
            "run_id": "0123456789abcdef0123456789abcdef",
            "success": True,
            "opensim_version": "OpenSim synthetic 4.6",
            "prepare_manifest_sha256": self._sha256(prepare_path),
            "artifact_linkage": {"run_id": "0123456789abcdef0123456789abcdef", "root": str(rra_dir)},
            "artifacts": stable_hashes,
            "deferred_artifacts_finalized_after_process_exit": [],
        }
        self._write_json(runtime_path, runtime)
        gates = {
            "runtime_success": True,
            "no_bad_residual_component": True,
            "no_bad_perr_coordinate": True,
            "no_silent_mass_application": True,
            "normalized_resultants_passed": True,
            "production_candidate": True,
            "okay_components_require_explicit_review": [],
        }
        summary = {
            "schema_version": adjusted._RRA_SCHEMA,
            "scope": adjusted._RRA_SCOPE,
            "gates": gates,
            "runtime": runtime,
            "runtime_linkage": {
                "run_id": runtime["run_id"],
                "runtime_sha256": self._sha256(runtime_path),
            },
            "residual_components": {"FX": {"grades": {"overall": "GOOD"}}},
            "perr": {"angle": {"grades": {"overall": "GOOD"}, "included_in_no_bad_perr_gate": True}},
            "artifacts": adjusted._artifact_hashes(rra_dir, excluded={summary_path}),
        }
        self._write_json(summary_path, summary)
        return rra_dir, data_dir, root / "published"

    def test_storage_and_model_motion_types_convert_only_rotation(self):
        """Convert q/u/udot degrees while leaving translations unchanged."""
        with tempfile.TemporaryDirectory() as temporary:
            rra_dir, data_dir, _ = self._fixture(Path(temporary))
            products = adjusted._paths_for_rra_products(
                rra_dir, json.loads((rra_dir / "prepare_manifest.json").read_text())
            )
            result = adjusted.load_adjusted_kinematics(
                products["model"],
                products["q"],
                products["u"],
                products["dudt"],
                motion_type_model_path=data_dir / "S001_scaled.osim",
            )
        np.testing.assert_allclose(result.coordinates[0], [np.pi, 2.0])
        np.testing.assert_allclose(result.speeds[0], [np.pi / 2.0, 5.0])
        np.testing.assert_allclose(result.accelerations[0], [2.0 * np.pi, 8.0])
        self.assertEqual(result.coordinate_names, ("angle", "slide"))

    def test_publish_interpolates_optional_data_and_records_provenance(self):
        """Publish schema-3 shapes, finite loaded COP, and official run linkage."""
        with tempfile.TemporaryDirectory() as temporary:
            rra_dir, data_dir, output_dir = self._fixture(Path(temporary))
            adjusted.publish_rra_adjusted_contact_input(rra_dir, data_dir, output_dir)
            with np.load(output_dir / "analysis.npz", allow_pickle=False) as archive:
                self.assertNotIn("id_generalized_forces", archive.files)
                self.assertEqual(archive["id_coordinates"].shape, (3, 2))
                self.assertEqual(archive["grf"].shape, (3, 2, 3))
                self.assertEqual(archive["target_markers"].shape, (3, 1, 3))
                self.assertEqual(list(archive["foot_names"]), ["left", "right"])
                np.testing.assert_allclose(archive["target_markers"][1, 0], [1.0, 2.0, 3.0])
                self.assertTrue(np.all(np.isfinite(archive["cop"][archive["contact"]])))
                np.testing.assert_array_equal(archive["contact"], archive["grf"][:, :, 1] >= 50.0)
            manifest = json.loads((output_dir / "manifest.json").read_text())
            qc = json.loads((output_dir / "qc_summary.json").read_text())
            self.assertEqual(manifest["schema_version"], adjusted._ANALYSIS_SCHEMA)
            self.assertEqual(manifest["rra_reference"]["run_id"], "0123456789abcdef0123456789abcdef")
            self.assertEqual(manifest["rra_reference"]["opensim_version"], "OpenSim synthetic 4.6")
            self.assertFalse(manifest["information_set"]["original_id_generalized_forces_carried_forward"])
            self.assertTrue(qc["rra_acceptance"]["gates"]["no_bad_residual_component"])
            self.assertEqual(
                (output_dir / "S001_scaled.osim").read_bytes(),
                next((rra_dir / "results").glob("*adjusted.osim")).read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                adjusted.publish_rra_adjusted_contact_input(rra_dir, data_dir, output_dir)

    def test_rejects_unaccepted_gate(self):
        """Do not import a current run that failed an official acceptance gate."""
        with tempfile.TemporaryDirectory() as temporary:
            rra_dir, data_dir, output_dir = self._fixture(Path(temporary))
            summary_path = rra_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["gates"]["no_bad_perr_coordinate"] = False
            self._write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "not an accepted production candidate"):
                adjusted.publish_rra_adjusted_contact_input(rra_dir, data_dir, output_dir)
        self.assertFalse(output_dir.exists())

    def test_rejects_stale_current_artifact_hash(self):
        """Reject result mutation after the accepted summary and runtime were written."""
        with tempfile.TemporaryDirectory() as temporary:
            rra_dir, data_dir, output_dir = self._fixture(Path(temporary))
            q_path = rra_dir / "results" / "trial_Kinematics_q.sto"
            q_path.write_text(q_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted summary hashes"):
                adjusted.publish_rra_adjusted_contact_input(rra_dir, data_dir, output_dir)

    def test_rejects_coordinate_order_mismatch(self):
        """The official Storage order must equal the adjusted-model coordinate order."""
        with tempfile.TemporaryDirectory() as temporary:
            rra_dir, data_dir, _ = self._fixture(Path(temporary))
            path = rra_dir / "results" / "trial_Kinematics_q.sto"
            path.write_text(path.read_text().replace("time\tangle\tslide", "time\tslide\tangle"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coordinate order"):
                adjusted.load_adjusted_kinematics(
                    rra_dir / "results" / "trial_adjusted.osim",
                    path,
                    rra_dir / "results" / "trial_Kinematics_u.sto",
                    rra_dir / "results" / "trial_Kinematics_dudt.sto",
                    motion_type_model_path=data_dir / "S001_scaled.osim",
                )


if __name__ == "__main__":
    unittest.main()
