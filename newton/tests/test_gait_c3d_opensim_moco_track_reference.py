# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the official torque-driven MocoTrack contact adapter."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from projects.gait_c3d import opensim_moco_contact_reference as contact
from projects.gait_c3d import opensim_moco_track_reference as moco

_HAS_OFFICIAL = importlib.util.find_spec("opensim") is not None
_REAL_RRA = Path("/home/jo31399/newton-data/gait/processed/trial_101/opensim_rra_official_reference_fy4")
_REAL_CONTACT = Path("/home/jo31399/newton-data/gait/processed/trial_101/opensim_moco_contact_reference")
_REAL_EXTERNAL = Path("/home/jo31399/newton-data/gait/processed/trial_101/latest/trial_grf_context.xml")


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


class TestOpenSimMocoTrackReference(unittest.TestCase):
    @staticmethod
    def _write_model(path: Path, *, mtp_locked: bool = False) -> None:
        document = ET.Element("OpenSimDocument")
        model = ET.SubElement(document, "Model", {"name": "tiny"})
        joints = ET.SubElement(ET.SubElement(model, "JointSet"), "objects")
        joint = ET.SubElement(joints, "CustomJoint", {"name": "ground_pelvis"})
        coordinates = ET.SubElement(joint, "coordinates")
        names = ("pelvis_tilt", "pelvis_tx", "pelvis_ty", "mtp_angle_l", "mtp_angle_r")
        for name in names:
            coordinate = ET.SubElement(coordinates, "Coordinate", {"name": name})
            if name.startswith("mtp_angle_"):
                ET.SubElement(coordinate, "locked").text = str(mtp_locked).lower()
        transform = ET.SubElement(joint, "SpatialTransform")
        for index, name in enumerate(names):
            kind = "rotation" if name in {"pelvis_tilt", "mtp_angle_l", "mtp_angle_r"} else "translation"
            axis = ET.SubElement(transform, "TransformAxis", {"name": f"{kind}{index}"})
            ET.SubElement(axis, "coordinates").text = name
        ET.indent(document, space="  ")
        path.write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True) + b"\n")

    @staticmethod
    def _write_kinematics(path: Path) -> None:
        path.write_text(
            "RRA Kinematics_q\n"
            "version=1\n"
            "nRows=2\n"
            "nColumns=6\n"
            "inDegrees=yes\n"
            "endheader\n"
            "time\tpelvis_tilt\tpelvis_tx\tpelvis_ty\tmtp_angle_l\tmtp_angle_r\n"
            "0.1\t180\t1.25\t0.9\t0\t0\n"
            "0.2\t90\t1.5\t0.91\t0\t0\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_external_loads(root: Path, *, basename: str = "trial") -> tuple[Path, Path]:
        data = root / f"{basename}.mot"
        data.write_text(
            "measured\nendheader\ntime\tground_force_l_vy\tground_force_r_vy\n0.1\t100\t0\n0.2\t0\t100\n",
            encoding="utf-8",
        )
        document = ET.Element("OpenSimDocument", {"Version": "40000"})
        loads = ET.SubElement(document, "ExternalLoads", {"name": "corrected"})
        objects = ET.SubElement(loads, "objects")
        for side, suffix in (("left", "l"), ("right", "r")):
            force = ET.SubElement(objects, "ExternalForce", {"name": side})
            ET.SubElement(force, "applied_to_body").text = f"calcn_{suffix}"
            ET.SubElement(force, "force_expressed_in_body").text = "ground"
            ET.SubElement(force, "point_expressed_in_body").text = "ground"
            ET.SubElement(force, "force_identifier").text = f"ground_force_{suffix}_v"
            ET.SubElement(force, "point_identifier").text = f"ground_force_{suffix}_p"
            ET.SubElement(force, "torque_identifier").text = f"ground_torque_{suffix}_"
        ET.SubElement(loads, "datafile").text = data.name
        path = root / f"{basename}.xml"
        ET.indent(document, space="  ")
        path.write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True) + b"\n")
        return path, data

    def _accepted_rra(self, root: Path, external: Path, data: Path, *, mtp_locked: bool = False) -> Path:
        results = root / "results"
        inputs = root / "inputs"
        results.mkdir(parents=True)
        inputs.mkdir()
        tool_name = "reference"
        model = results / f"{tool_name}_adjusted.osim"
        q_path = results / f"{tool_name}_Kinematics_q.sto"
        u_path = results / f"{tool_name}_Kinematics_u.sto"
        dudt_path = results / f"{tool_name}_Kinematics_dudt.sto"
        self._write_model(model, mtp_locked=mtp_locked)
        for path in (q_path, u_path, dudt_path):
            self._write_kinematics(path)
        setup = inputs / "Setup_RRA.xml"
        setup.write_text("<OpenSimDocument/>\n", encoding="utf-8")
        prepare = {
            "schema_version": moco._inverse._RRA_SCHEMA,
            "scope": moco._inverse._rra_contact._RRA_SCOPE,
            "tool_name": tool_name,
            "source_inputs": {
                str(external): {"sha256": moco._sha256(external)},
                str(data): {"sha256": moco._sha256(data)},
            },
            "generated_inputs": {str(setup): {"sha256": moco._sha256(setup)}},
        }
        prepare_path = root / "prepare_manifest.json"
        moco._write_json(prepare_path, prepare)
        deferred = root / "child_process.json"
        deferred.write_text('{"returncode": 0}\n', encoding="utf-8")
        runtime_path = root / "run_runtime.json"
        summary_path = root / "summary.json"
        run_id = "0123456789abcdef0123456789abcdef"
        runtime = {
            "schema_version": moco._inverse._RRA_SCHEMA,
            "scope": moco._inverse._rra_contact._RRA_SCOPE,
            "run_id": run_id,
            "success": True,
            "opensim_version": "OpenSim synthetic 4.6",
            "prepare_manifest_sha256": moco._sha256(prepare_path),
            "artifact_linkage": {"run_id": run_id, "root": str(root)},
            "artifacts": moco._artifact_hashes(root, excluded={runtime_path, summary_path, deferred}),
            "deferred_artifacts_finalized_after_process_exit": [deferred.name],
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
            "schema_version": moco._inverse._RRA_SCHEMA,
            "scope": moco._inverse._rra_contact._RRA_SCOPE,
            "gates": gates,
            "runtime": runtime,
            "runtime_linkage": {"run_id": run_id, "runtime_sha256": moco._sha256(runtime_path)},
            "artifacts": moco._artifact_hashes(root, excluded={summary_path}),
        }
        moco._write_json(summary_path, summary)
        return root

    def _inputs(self, root: Path, *, mtp_locked: bool = False) -> tuple[Path, Path, Path]:
        external, data = self._write_external_loads(root)
        rra = self._accepted_rra(root / "rra", external, data, mtp_locked=mtp_locked)
        contact_root = root / "contact"
        contact.write_reference_files(contact_root, external_loads_path=external)
        return rra, contact_root, external

    def test_absolute_rra_values_and_spline_derived_speed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.osim"
            source = root / "q.sto"
            output = root / "states.sto"
            self._write_model(model)
            self._write_kinematics(source)
            moco.convert_rra_states_reference(source, model, output)
            metadata, labels, rows = moco._inverse.parse_storage(output)
            self.assertEqual(metadata["indegrees"], "no")
            self.assertTrue(all(label.endswith("/value") for label in labels[1:]))
            self.assertFalse(any("/speed" in label for label in labels))
            self.assertAlmostEqual(rows[0][1], math.pi)
            self.assertEqual(rows[0][2], 1.25)

    def test_prepare_records_exact_torque_processor_goals_groups_and_periodicity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root)
            guess = root / "guess.sto"
            guess.write_text("guess\n", encoding="utf-8")
            prepared = moco.prepare_reference(
                rra / "results",
                contact_root,
                external,
                root / "track",
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
                [operation["operation"] for operation in config["model_processor_operations"]],
                ["ModOpRemoveMuscles", "ModOpAddReserves"],
            )
            self.assertNotIn("ModOpAddExternalLoads", json.dumps(config))
            self.assertEqual(
                [goal["type"] for goal in config["goals"]],
                ["MocoStateTrackingGoal", "MocoControlGoal", "MocoContactTrackingGoal", "MocoPeriodicityGoal"],
            )
            groups = config["goals"][2]["groups"]
            self.assertEqual([group["side"] for group in groups], ["left", "right"])
            self.assertEqual(groups[0]["alternative_frame_paths"], ["/bodyset/toes_l"])
            self.assertEqual(len(groups[0]["contact_force_paths"]), 6)
            periodicity = config["periodicity"]
            self.assertNotIn("/jointset/ground_pelvis/pelvis_tx/value", periodicity["value_state_pairs"])
            self.assertIn("/jointset/ground_pelvis/pelvis_tx/speed", periodicity["speed_state_pairs"])
            self.assertIn("TabOpAppendCoordinateValueDerivativesAsSpeeds", config["reference_table_operations"])
            for toe in config["toe_policy"]["coordinates"].values():
                self.assertFalse(toe["locked"])
                self.assertEqual(toe["mode"], "unlocked_official_example")
                self.assertEqual(toe["passive_force"]["expression"], "-25.0*q-2.0*qdot")
            self.assertIn("fixed toes to their calcaneus", config["toe_policy"]["locked_policy"])
            self.assertIn("not_predictive", manifest["scope"])

    def test_prepare_rejects_rra_gate_and_mutated_contact_or_external_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root)
            summary_path = rra / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["gates"]["production_candidate"] = False
            moco._write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "accepted production candidate"):
                moco.prepare_reference(rra, contact_root, external, root / "rejected_gate")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root)
            geometry = contact_root / "S001_ContactGeometrySet.xml"
            geometry.write_bytes(geometry.read_bytes() + b"<!-- mutation -->\n")
            with self.assertRaisesRegex(ValueError, "generated hash mismatch"):
                moco.prepare_reference(rra, contact_root, external, root / "rejected_contact")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root)
            other_external, _ = self._write_external_loads(root, basename="other")
            with self.assertRaisesRegex(ValueError, "not hash-linked"):
                moco.prepare_reference(rra, contact_root, other_external, root / "rejected_loads")

    def test_model_processor_has_contact_torque_actuators_reserves_and_no_external_load_operator(self) -> None:
        model = mock.MagicMock()
        geometry = mock.MagicMock()
        geometry.getSize.return_value = 1
        force_set = mock.MagicMock()
        force_set.getSize.return_value = 1
        processor = mock.MagicMock()
        actuators: list[mock.MagicMock] = []
        passive_forces: list[mock.MagicMock] = []
        model.getCoordinateSet.return_value.get.side_effect = lambda _name: SimpleNamespace(
            getLocked=lambda _state: False
        )

        def expression_force() -> mock.MagicMock:
            force = mock.MagicMock()
            passive_forces.append(force)
            return force

        def coordinate_actuator(name: str) -> mock.MagicMock:
            actuator = mock.MagicMock()
            actuator.coordinate_name = name
            actuators.append(actuator)
            return actuator

        opensim = SimpleNamespace(
            Model=mock.Mock(return_value=model),
            ContactGeometrySet=mock.Mock(return_value=geometry),
            ForceSet=mock.Mock(return_value=force_set),
            ExpressionBasedCoordinateForce=expression_force,
            CoordinateActuator=coordinate_actuator,
            ModelProcessor=mock.Mock(return_value=processor),
            ModOpRemoveMuscles=mock.Mock(return_value="remove-muscles"),
            ModOpAddReserves=mock.Mock(return_value="add-reserves"),
        )
        config = {
            "model_path": "model.osim",
            "contact_geometry_path": "geometry.xml",
            "contact_force_path": "forces.xml",
            "toe_policy": {
                "coordinates": {
                    "mtp_angle_l": {"locked": False},
                    "mtp_angle_r": {"locked": False},
                }
            },
        }
        with mock.patch.object(moco._contact, "assert_model_has_no_external_loads") as assertion:
            returned_model, returned_processor = moco.build_model_processor(opensim, config)
        self.assertIs(returned_model, model)
        self.assertIs(returned_processor, processor)
        self.assertEqual([actuator.coordinate_name for actuator in actuators], ["mtp_angle_l", "mtp_angle_r"])
        self.assertEqual(len(passive_forces), 2)
        for side, force in zip(("l", "r"), passive_forces, strict=True):
            force.setName.assert_called_once_with(f"PassiveToeDamping_{side}")
            force.set_coordinate.assert_called_once_with(f"mtp_angle_{side}")
            force.set_expression.assert_called_once_with("-25.0*q-2.0*qdot")
        for actuator in actuators:
            actuator.setOptimalForce.assert_called_once_with(10.0)
            actuator.setMinControl.assert_called_once_with(-1.0)
            actuator.setMaxControl.assert_called_once_with(1.0)
        self.assertEqual(processor.append.call_args_list, [mock.call("remove-muscles"), mock.call("add-reserves")])
        opensim.ModOpAddReserves.assert_called_once_with(500.0, 1.0, True, True)
        self.assertEqual(assertion.call_count, 2)

    def test_locked_mtp_policy_omits_toe_dynamics_bounds_and_periodicity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root, mtp_locked=True)
            prepared = moco.prepare_reference(rra, contact_root, external, root / "track")
            config = json.loads(prepared.config_path.read_text(encoding="utf-8"))
            for toe in config["toe_policy"]["coordinates"].values():
                self.assertTrue(toe["locked"])
                self.assertEqual(toe["mode"], "locked_fixed_to_calcaneus")
                self.assertIsNone(toe["passive_force"])
                self.assertIsNone(toe["weak_actuator"])
                self.assertIsNone(toe["state_bounds"])
                self.assertNotIn(toe["value_state_path"], config["periodicity"]["value_state_pairs"])
                self.assertNotIn(toe["speed_state_path"], config["periodicity"]["speed_state_pairs"])

        model = mock.MagicMock()
        empty_set = mock.MagicMock()
        empty_set.getSize.return_value = 0
        processor = mock.MagicMock()
        opensim = SimpleNamespace(
            Model=mock.Mock(return_value=model),
            ContactGeometrySet=mock.Mock(return_value=empty_set),
            ForceSet=mock.Mock(return_value=empty_set),
            ExpressionBasedCoordinateForce=mock.Mock(),
            CoordinateActuator=mock.Mock(),
            ModelProcessor=mock.Mock(return_value=processor),
            ModOpRemoveMuscles=mock.Mock(return_value="remove-muscles"),
            ModOpAddReserves=mock.Mock(return_value="add-reserves"),
        )
        locked_config = {
            "model_path": "model.osim",
            "contact_geometry_path": "geometry.xml",
            "contact_force_path": "forces.xml",
            "toe_policy": {
                "coordinates": {
                    "mtp_angle_l": {"locked": True},
                    "mtp_angle_r": {"locked": True},
                }
            },
        }
        with (
            mock.patch.object(moco._contact, "assert_model_has_no_external_loads"),
            mock.patch.object(
                moco, "_opensim_mtp_locked_states", return_value={"mtp_angle_l": True, "mtp_angle_r": True}
            ),
        ):
            moco.build_model_processor(opensim, locked_config)
        opensim.ExpressionBasedCoordinateForce.assert_not_called()
        opensim.CoordinateActuator.assert_not_called()

    def test_goal_builder_uses_reference_only_toe_alternative_groups(self) -> None:
        class Vector(list):
            def append(self, value: str) -> None:
                super().append(value)

        class Goal:
            def __init__(self, name: str, weight: float) -> None:
                self.name = name
                self.weight = weight
                self.external = ""
                self.groups: list[tuple[list[str], str, list[str]]] = []

            def setExternalLoadsFile(self, path: str) -> None:
                self.external = path

            def addContactGroup(self, group: tuple[list[str], str, list[str]]) -> None:
                self.groups.append(group)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external, _ = self._write_external_loads(root)
            opensim = SimpleNamespace(
                MocoContactTrackingGoal=Goal,
                StdVectorString=Vector,
                MocoContactTrackingGoalGroup=lambda forces, name, alternatives: (forces, name, alternatives),
            )
            goal = moco.configure_contact_tracking_goal(opensim, {"external_loads_reference_path": str(external)})
            self.assertEqual(goal.external, str(external.resolve()))
            self.assertEqual([group[1] for group in goal.groups], ["left", "right"])
            self.assertEqual(goal.groups[0][2], ["/bodyset/toes_l"])
            self.assertEqual(len(goal.groups[1][0]), 6)

    def test_run_rejects_manifest_and_hashed_config_divergence_before_opensim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root)
            prepared = moco.prepare_reference(rra, contact_root, external, root / "track")
            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            manifest["configuration"]["max_iterations"] += 1
            moco._write_json(prepared.manifest_path, manifest)
            with mock.patch.object(moco, "_import_official_opensim") as importer:
                with self.assertRaisesRegex(ValueError, "configuration diverges"):
                    moco._run_reference_in_process(prepared.output_dir)
            importer.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root)
            prepared = moco.prepare_reference(rra, contact_root, external, root / "track")
            config = json.loads(prepared.config_path.read_text(encoding="utf-8"))
            config["max_iterations"] += 1
            moco._write_json(prepared.config_path, config)
            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            manifest["generated_inputs"][str(prepared.config_path)]["sha256"] = moco._sha256(prepared.config_path)
            moco._write_json(prepared.manifest_path, manifest)
            with mock.patch.object(moco, "_import_official_opensim") as importer:
                with self.assertRaisesRegex(ValueError, "configuration diverges"):
                    moco._run_reference_in_process(prepared.output_dir)
            importer.assert_not_called()

    def test_sealed_failure_is_unsealed_and_preserved_only_as_failed_guess(self) -> None:
        solution = _FakeSolution(False)
        path, success = moco.write_solution_or_failed_guess(solution, Path("solution.sto"), Path("failed_guess.sto"))
        self.assertFalse(success)
        self.assertTrue(solution.unsealed)
        self.assertEqual(path, Path("failed_guess.sto"))
        self.assertEqual(solution.written, ["failed_guess.sto"])
        successful = _FakeSolution(True)
        path, success = moco.write_solution_or_failed_guess(successful, Path("solution.sto"), Path("failed_guess.sto"))
        self.assertTrue(success)
        self.assertEqual(path, Path("solution.sto"))
        self.assertFalse(successful.unsealed)

    def test_summary_rechecks_hash_linkage_and_never_claims_prediction_or_fd1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rra, contact_root, external = self._inputs(root)
            prepared = moco.prepare_reference(rra, contact_root, external, root / "track", max_iterations=0)
            result_path = prepared.output_dir / "results" / "moco_track_failed_guess.sto"
            result_path.write_text(
                "failed\nversion=1\nnRows=2\nnColumns=2\ninDegrees=no\nendheader\ntime\tx\n0.1\t0\n0.2\t1\n",
                encoding="utf-8",
            )
            runtime_path = prepared.output_dir / "run_runtime.json"
            runtime = {
                "schema_version": moco._SCHEMA,
                "scope": moco._SCOPE,
                "run_id": "run",
                "success": False,
                "sealed_failure_captured_as_guess": True,
                "error": "failed",
                "wall_time_s": 1.0,
                "opensim_version": "synthetic",
                "python_version": "synthetic",
                "platform": "synthetic",
                "prepare_manifest_sha256": moco._sha256(prepared.manifest_path),
                "result_path": str(result_path.relative_to(prepared.output_dir)),
                "model_path": None,
                "artifacts": moco._artifact_hashes(prepared.output_dir, excluded={runtime_path}),
            }
            moco._write_json(runtime_path, runtime)
            summary_path = moco.summarize_reference(prepared.output_dir)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertFalse(summary["success"])
            self.assertTrue(summary["sealed_failure_captured_as_guess"])
            self.assertEqual(summary["result"]["kind"], "failed_guess")
            self.assertFalse(summary["claims"]["newton_predictive_forward_dynamics"])
            self.assertFalse(summary["claims"]["fd1"])
            external.write_bytes(external.read_bytes() + b"<!-- mutation -->\n")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                moco.summarize_reference(prepared.output_dir)

    @unittest.skipUnless(
        _HAS_OFFICIAL and _REAL_RRA.is_dir() and _REAL_CONTACT.is_dir() and _REAL_EXTERNAL.is_file(),
        "official OpenSim or accepted S001 references unavailable",
    )
    def test_optional_official_short_model_and_table_initialization(self) -> None:
        # No solve. max_iterations=0 records the intended short-init setting,
        # but this test stops after processing the official model and table.
        with tempfile.TemporaryDirectory() as temporary:
            prepared = moco.prepare_reference(
                _REAL_RRA,
                _REAL_CONTACT,
                _REAL_EXTERNAL,
                Path(temporary) / "track",
                max_iterations=0,
            )
            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            opensim = __import__("opensim")
            model, processor = moco.build_model_processor(opensim, manifest["configuration"])
            table = moco.build_reference_table_processor(opensim, manifest["configuration"])
            processed = processor.process()
            processed.initSystem()
            contact.assert_model_has_no_external_loads(model)
            contact.assert_model_has_no_external_loads(processed)
            updated = table.process(model)
            labels = list(updated.getColumnLabels())
            self.assertTrue(any(label.endswith("/speed") for label in labels))
            self.assertEqual(processed.getContactGeometrySet().getSize(), 13)

            # Initialize MocoTrack itself, but do not call solve(). This checks
            # the official state/control goal construction and reference table.
            config = manifest["configuration"]
            track = opensim.MocoTrack()
            track.setName("torque_driven_tracking_short_init")
            track.setModel(processor)
            track.setStatesReference(table)
            track.set_states_global_tracking_weight(0.05)
            track.set_control_effort_weight(0.1)
            track.set_allow_unused_references(True)
            track.set_track_reference_position_derivatives(True)
            track.set_initial_time(config["initial_time_s"])
            track.set_final_time(config["final_time_s"])
            track.set_mesh_interval(config["mesh_interval_s"])
            study = track.initialize()
            self.assertIsNotNone(study.updProblem())


if __name__ == "__main__":
    unittest.main()
