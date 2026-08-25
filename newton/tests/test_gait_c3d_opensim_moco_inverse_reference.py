# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the official OpenSim MocoInverse reference adapter."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from projects.gait_c3d.oracles import opensim_moco_inverse_reference as moco


class _FakeSolution:
    def __init__(self, success: bool) -> None:
        self._success = success
        self.unsealed = False
        self.written: list[str] = []

    def success(self) -> bool:
        return self._success

    def unseal(self) -> None:
        self.unsealed = True

    def write(self, path: str) -> None:
        if not self._success and not self.unsealed:
            raise RuntimeError("sealed")
        self.written.append(path)


class TestOpenSimMocoInverseReference(unittest.TestCase):
    @staticmethod
    def _write_model(path: Path) -> None:
        document = ET.Element("OpenSimDocument")
        model = ET.SubElement(document, "Model", {"name": "tiny"})
        joints = ET.SubElement(ET.SubElement(model, "JointSet"), "objects")

        ground = ET.SubElement(joints, "CustomJoint", {"name": "ground_pelvis"})
        coordinates = ET.SubElement(ground, "coordinates")
        ET.SubElement(coordinates, "Coordinate", {"name": "pelvis_tilt"})
        ET.SubElement(coordinates, "Coordinate", {"name": "pelvis_tx"})
        transform = ET.SubElement(ground, "SpatialTransform")
        rotation = ET.SubElement(transform, "TransformAxis", {"name": "rotation1"})
        ET.SubElement(rotation, "coordinates").text = "pelvis_tilt"
        translation = ET.SubElement(transform, "TransformAxis", {"name": "translation1"})
        ET.SubElement(translation, "coordinates").text = "pelvis_tx"

        for side in ("r", "l"):
            mtp = ET.SubElement(joints, "PinJoint", {"name": f"mtp_{side}"})
            coordinates = ET.SubElement(mtp, "coordinates")
            ET.SubElement(coordinates, "Coordinate", {"name": f"mtp_angle_{side}"})
        ET.indent(document, space="  ")
        path.write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True) + b"\n")

    @staticmethod
    def _write_kinematics(path: Path) -> None:
        path.write_text(
            "RRA Kinematics_q\n"
            "version=1\n"
            "nRows=2\n"
            "nColumns=5\n"
            "inDegrees=yes\n"
            "endheader\n"
            "time\tpelvis_tilt\tpelvis_tx\tmtp_angle_r\tmtp_angle_l\n"
            "0.1\t180\t1.25\t10\t-10\n"
            "0.2\t90\t1.5\t20\t-20\n",
            encoding="utf-8",
        )

    def _accepted_rra(self, root: Path) -> Path:
        results = root / "results"
        inputs = root / "inputs"
        results.mkdir(parents=True)
        inputs.mkdir()
        tool_name = "reference"
        model = results / f"{tool_name}_adjusted.osim"
        q_path = results / f"{tool_name}_Kinematics_q.sto"
        u_path = results / f"{tool_name}_Kinematics_u.sto"
        dudt_path = results / f"{tool_name}_Kinematics_dudt.sto"
        self._write_model(model)
        for path in (q_path, u_path, dudt_path):
            self._write_kinematics(path)

        # Keep measured ExternalLoads sources outside the RRA artifact root so
        # their prepare hashes, rather than the artifact maps, seal mutations.
        data_file = root.parent / "trial_grf.mot"
        data_file.write_text("GRF data\n", encoding="utf-8")
        external = root.parent / "trial_grf_context.xml"
        external.write_text(
            "<?xml version='1.0'?><OpenSimDocument><ExternalLoads name='loads'>"
            "<datafile>trial_grf.mot</datafile></ExternalLoads></OpenSimDocument>\n",
            encoding="utf-8",
        )
        setup = inputs / "Setup_RRA.xml"
        setup.write_text("<OpenSimDocument/>\n", encoding="utf-8")
        prepare = {
            "schema_version": moco._RRA_SCHEMA,
            "scope": moco._rra_contact._RRA_SCOPE,
            "tool_name": tool_name,
            "source_inputs": {
                str(external): {"sha256": moco._sha256(external)},
                str(data_file): {"sha256": moco._sha256(data_file)},
            },
            "generated_inputs": {str(setup): {"sha256": moco._sha256(setup)}},
        }
        prepare_path = root / "prepare_manifest.json"
        moco._write_json(prepare_path, prepare)

        # Exercise the verifier's deferred-artifact state as well as its stable
        # runtime artifact map.
        deferred_path = root / "child_process.json"
        deferred_path.write_text('{"returncode": 0}\n', encoding="utf-8")
        runtime_path = root / "run_runtime.json"
        summary_path = root / "summary.json"
        run_id = "0123456789abcdef0123456789abcdef"
        runtime = {
            "schema_version": moco._RRA_SCHEMA,
            "scope": moco._rra_contact._RRA_SCOPE,
            "run_id": run_id,
            "success": True,
            "opensim_version": "OpenSim synthetic 4.6",
            "prepare_manifest_sha256": moco._sha256(prepare_path),
            "artifact_linkage": {"run_id": run_id, "root": str(root)},
            "artifacts": moco._artifact_hashes(
                root,
                excluded={runtime_path, summary_path, deferred_path},
            ),
            "deferred_artifacts_finalized_after_process_exit": [deferred_path.name],
        }
        moco._write_json(runtime_path, runtime)
        gates = {
            "runtime_success": True,
            "no_bad_residual_component": True,
            "no_bad_perr_coordinate": True,
            "no_silent_mass_application": True,
            "normalized_resultants_passed": True,
            "production_candidate": True,
        }
        summary = {
            "schema_version": moco._RRA_SCHEMA,
            "scope": moco._rra_contact._RRA_SCOPE,
            "gates": gates,
            "runtime": runtime,
            "runtime_linkage": {"run_id": run_id, "runtime_sha256": moco._sha256(runtime_path)},
            "artifacts": moco._artifact_hashes(root, excluded={summary_path}),
        }
        moco._write_json(summary_path, summary)
        return root

    def test_coordinate_paths_and_degree_conversion_omit_welded_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.osim"
            source = root / "Kinematics_q.sto"
            destination = root / "coordinates.sto"
            self._write_model(model)
            self._write_kinematics(source)
            infos = moco.coordinate_info_from_model(model)
            self.assertEqual(infos["pelvis_tilt"].state_path, "/jointset/ground_pelvis/pelvis_tilt/value")
            self.assertTrue(infos["pelvis_tilt"].rotational)
            self.assertFalse(infos["pelvis_tx"].rotational)

            moco.convert_rra_kinematics(source, model, destination)
            metadata, labels, rows = moco.parse_storage(destination)
            self.assertEqual(metadata["indegrees"], "no")
            self.assertEqual(
                labels,
                [
                    "time",
                    "/jointset/ground_pelvis/pelvis_tilt/value",
                    "/jointset/ground_pelvis/pelvis_tx/value",
                ],
            )
            self.assertAlmostEqual(rows[0][1], math.pi)
            self.assertEqual(rows[0][2], 1.25)
            self.assertNotIn("mtp", destination.read_text(encoding="utf-8"))

    def test_coupled_knee_translation_does_not_override_rotational_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.osim"
            self._write_model(model)
            tree = ET.parse(model)
            joints = tree.getroot().find(".//JointSet/objects")
            knee = ET.SubElement(joints, "CustomJoint", {"name": "knee_r"})
            coordinates = ET.SubElement(knee, "coordinates")
            ET.SubElement(coordinates, "Coordinate", {"name": "knee_angle_r"})
            transform = ET.SubElement(knee, "SpatialTransform")
            rotation = ET.SubElement(transform, "TransformAxis", {"name": "rotation1"})
            ET.SubElement(rotation, "coordinates").text = "knee_angle_r"
            translation = ET.SubElement(transform, "TransformAxis", {"name": "translation1"})
            ET.SubElement(translation, "coordinates").text = "knee_angle_r"
            tree.write(model, encoding="utf-8", xml_declaration=True)
            self.assertTrue(moco.coordinate_info_from_model(model)["knee_angle_r"].rotational)

    def test_conversion_rejects_absolute_or_incomplete_legacy_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.osim"
            source = root / "bad.sto"
            self._write_model(model)
            source.write_text(
                "bad\ninDegrees=yes\nendheader\ntime /jointset/x/y/value\n0 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy short"):
                moco.convert_rra_kinematics(source, model, root / "out.sto")

    def test_prepare_requires_accepted_rra_and_records_exact_pinned_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra_root = self._accepted_rra(root / "rra")
            guess = root / "guess.sto"
            guess.write_text("guess\n", encoding="utf-8")
            prepared = moco.prepare_reference(
                rra_root / "results",
                root / "moco",
                mesh_interval=0.01,
                max_iterations=7,
                guess_file=guess,
            )
            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            config = manifest["configuration"]
            self.assertEqual(config["mesh_interval_s"], 0.01)
            self.assertEqual(config["max_iterations"], 7)
            self.assertEqual(config["guess_file"], str(guess))
            self.assertEqual(
                [entry["operation"] for entry in config["model_processor_operations"]],
                [
                    "ModOpAddExternalLoads",
                    "ModOpReplaceJointsWithWelds",
                    "ModOpAddResiduals",
                    "ModOpIgnoreTendonCompliance",
                    "ModOpReplaceMusclesWithDeGrooteFregly2016",
                    "ModOpIgnorePassiveFiberForcesDGF",
                    "ModOpScaleActiveFiberForceCurveWidthDGF",
                    "ModOpAddReserves",
                ],
            )
            self.assertIn("not_claimed", config["prescribed_motion_scope"])
            self.assertEqual(manifest["pinned_upstream"]["commit"], moco._PINNED_COMMIT)

            summary = json.loads((rra_root / "summary.json").read_text(encoding="utf-8"))
            summary["gates"]["production_candidate"] = False
            (rra_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted production candidate"):
                moco.prepare_reference(rra_root, root / "rejected")

    def test_prepare_rejects_each_rra_acceptance_gate(self) -> None:
        required_gates = (
            "runtime_success",
            "no_bad_residual_component",
            "no_bad_perr_coordinate",
            "no_silent_mass_application",
            "normalized_resultants_passed",
            "production_candidate",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra_root = self._accepted_rra(root / "rra")
            summary_path = rra_root / "summary.json"
            pristine = summary_path.read_bytes()
            for gate in required_gates:
                with self.subTest(gate=gate):
                    summary = json.loads(pristine)
                    summary["gates"][gate] = False
                    moco._write_json(summary_path, summary)
                    with self.assertRaisesRegex(ValueError, "accepted production candidate"):
                        moco.prepare_reference(rra_root, root / f"rejected_{gate}")
                    summary_path.write_bytes(pristine)

    def test_prepare_rejects_mutated_rra_provenance_and_inputs(self) -> None:
        cases = (
            "summary",
            "runtime",
            "prepare",
            "summary_artifact_map",
            "runtime_artifact_map",
            "result_artifact",
            "deferred_artifact",
            "generated_input",
            "external_xml",
            "external_datafile",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rra_root = self._accepted_rra(root / "rra")
                summary_path = rra_root / "summary.json"
                runtime_path = rra_root / "run_runtime.json"
                prepare_path = rra_root / "prepare_manifest.json"
                if case == "summary":
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["scope"] = "mutated"
                    moco._write_json(summary_path, summary)
                elif case == "runtime":
                    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                    runtime["opensim_version"] = "mutated"
                    moco._write_json(runtime_path, runtime)
                elif case == "prepare":
                    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
                    prepare["mutation"] = True
                    moco._write_json(prepare_path, prepare)
                elif case == "summary_artifact_map":
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["artifacts"].pop("child_process.json")
                    moco._write_json(summary_path, summary)
                elif case == "runtime_artifact_map":
                    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                    runtime["artifacts"].pop("inputs/Setup_RRA.xml")
                    moco._write_json(runtime_path, runtime)
                    # Keep the summary equal to the current runtime and refresh
                    # only its direct runtime hash. The stable runtime artifact
                    # map must still reject this selectively resealed metadata.
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["runtime"] = runtime
                    summary["runtime_linkage"]["runtime_sha256"] = moco._sha256(runtime_path)
                    summary["artifacts"]["run_runtime.json"] = moco._sha256(runtime_path)
                    moco._write_json(summary_path, summary)
                elif case == "result_artifact":
                    path = rra_root / "results" / "reference_Kinematics_q.sto"
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                elif case == "deferred_artifact":
                    (rra_root / "child_process.json").unlink()
                elif case == "generated_input":
                    path = rra_root / "inputs" / "Setup_RRA.xml"
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                elif case == "external_xml":
                    path = root / "trial_grf_context.xml"
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                else:
                    path = root / "trial_grf.mot"
                    path.write_text(path.read_text(encoding="utf-8") + "mutated\n", encoding="utf-8")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    moco.prepare_reference(rra_root, root / "moco")
                self.assertFalse((root / "moco").exists())

    def test_model_processor_operation_order_and_arguments(self) -> None:
        operations: list[tuple[str, tuple[object, ...]]] = []

        class Vector(list[str]):
            pass

        class Processor:
            def __init__(self, path: str) -> None:
                operations.append(("ModelProcessor", (path,)))

            def append(self, operation: tuple[str, tuple[object, ...]]) -> None:
                operations.append(operation)

        fake = mock.Mock()
        fake.ModelProcessor.side_effect = Processor
        fake.StdVectorString.side_effect = Vector
        for name in (
            "ModOpAddExternalLoads",
            "ModOpReplaceJointsWithWelds",
            "ModOpAddResiduals",
            "ModOpIgnoreTendonCompliance",
            "ModOpReplaceMusclesWithDeGrooteFregly2016",
            "ModOpIgnorePassiveFiberForcesDGF",
            "ModOpScaleActiveFiberForceCurveWidthDGF",
            "ModOpAddReserves",
        ):
            setattr(fake, name, mock.Mock(side_effect=lambda *args, _name=name: (_name, args)))
        moco.build_model_processor(fake, {"model_path": "model.osim", "external_loads_path": "loads.xml"})
        self.assertEqual(
            [name for name, _ in operations],
            [
                "ModelProcessor",
                "ModOpAddExternalLoads",
                "ModOpReplaceJointsWithWelds",
                "ModOpAddResiduals",
                "ModOpIgnoreTendonCompliance",
                "ModOpReplaceMusclesWithDeGrooteFregly2016",
                "ModOpIgnorePassiveFiberForcesDGF",
                "ModOpScaleActiveFiberForceCurveWidthDGF",
                "ModOpAddReserves",
            ],
        )
        self.assertEqual(operations[2][1][0], ["mtp_r", "mtp_l"])
        self.assertEqual(operations[3][1], (250.0, 50.0, 1.0))
        self.assertEqual(operations[-1][1], (1.0,))

    def test_sealed_failure_becomes_failed_guess_without_writing_solution(self) -> None:
        solution = _FakeSolution(False)
        path, success = moco.write_solution_or_failed_guess(
            solution,
            Path("solution.sto"),
            Path("failed_guess.sto"),
        )
        self.assertFalse(success)
        self.assertTrue(solution.unsealed)
        self.assertEqual(path, Path("failed_guess.sto"))
        self.assertEqual(solution.written, ["failed_guess.sto"])

    def test_success_is_not_unsealed(self) -> None:
        solution = _FakeSolution(True)
        path, success = moco.write_solution_or_failed_guess(solution, Path("solution.sto"), Path("failed.sto"))
        self.assertTrue(success)
        self.assertFalse(solution.unsealed)
        self.assertEqual(path, Path("solution.sto"))

    def test_optional_official_bindings_have_required_api(self) -> None:
        try:
            opensim = moco._import_official_opensim()
        except RuntimeError:
            self.skipTest("official OpenSim Python bindings are absent")
        required = (
            "MocoInverse",
            "ModelProcessor",
            "ModOpAddExternalLoads",
            "ModOpReplaceJointsWithWelds",
            "MocoCasADiSolver",
        )
        self.assertTrue(all(hasattr(opensim, name) for name in required))


if __name__ == "__main__":
    unittest.main()
