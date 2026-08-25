# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the official OpenSim RRA reference adapter."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from projects.gait_c3d.oracles import opensim_rra_reference as rra


class TestOpenSimRRAReference(unittest.TestCase):
    """Verify deterministic XML generation and output parsing without OpenSim."""

    @staticmethod
    def _write_model(path: Path) -> None:
        document = ET.Element("OpenSimDocument", {"Version": "20302"})
        model = ET.SubElement(document, "Model", {"name": "synthetic_gait2354"})
        bodies = ET.SubElement(ET.SubElement(model, "BodySet"), "objects")
        pelvis = ET.SubElement(bodies, "Body", {"name": "pelvis"})
        ET.SubElement(pelvis, "mass").text = "10"
        ET.SubElement(pelvis, "mass_center").text = "-0.0644674 0 0"
        torso = ET.SubElement(bodies, "Body", {"name": "torso"})
        ET.SubElement(torso, "mass").text = "30"
        ET.SubElement(torso, "mass_center").text = "0 0.3 0"
        coordinates = ET.SubElement(model, "CoordinateSet")
        objects = ET.SubElement(coordinates, "objects")
        translations = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
        for name in rra._TASK_WEIGHTS:
            coordinate = ET.SubElement(objects, "Coordinate", {"name": name})
            ET.SubElement(coordinate, "motion_type").text = "translational" if name in translations else "rotational"
            ET.SubElement(coordinate, "range").text = "0 0" if name.startswith("mtp_angle") else "-2 2"
            ET.SubElement(coordinate, "clamped").text = "true" if name.startswith("mtp_angle") else "false"
            ET.SubElement(coordinate, "locked").text = "false"
        ET.indent(document, space="  ")
        path.write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True) + b"\n")

    @staticmethod
    def _write_storage(path: Path, labels: list[str], rows: list[list[float]]) -> None:
        path.write_text(
            "synthetic\nversion=1\nendheader\n"
            + "\t".join(labels)
            + "\n"
            + "\n".join("\t".join(str(value) for value in row) for row in rows)
            + "\n",
            encoding="utf-8",
        )

    def _write_inputs(self, data_dir: Path) -> None:
        data_dir.mkdir()
        self._write_model(data_dir / "S001_scaled.osim")
        self._write_storage(
            data_dir / "trial_ik_dynamics_context.mot",
            ["time", *rra._TASK_WEIGHTS],
            [[0.0, *([0.0] * 23)], [1.0, *([0.0] * 23)]],
        )
        (data_dir / "trial_grf_context.xml").write_text(
            "<OpenSimDocument><ExternalLoads><datafile>trial_grf_context.mot</datafile>"
            "</ExternalLoads></OpenSimDocument>\n",
            encoding="utf-8",
        )
        self._write_storage(
            data_dir / "trial_grf_context.mot",
            ["time", "ground_force_l_vx"],
            [[0.0, 0.0], [1.0, 0.0]],
        )
        (data_dir / "qc_summary.json").write_text(
            json.dumps(
                {
                    "stride": {"start_time_s": 0.2, "stop_time_s": 0.8},
                    "pelvis_residuals": {"normalization": {"body_weight_N": 100.0, "marker_height_m": 2.0}},
                }
            ),
            encoding="utf-8",
        )

    def test_model_parser_treats_clamped_zero_range_as_locked(self):
        """Detect S001's fixed MTP coordinates even when locked is false."""
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.osim"
            self._write_model(model)
            coordinates, pelvis_com = rra.parse_model_spec(model)
        self.assertEqual([item.name for item in coordinates if item.locked], ["mtp_angle_r", "mtp_angle_l"])
        self.assertEqual(pelvis_com, (-0.0644674, 0.0, 0.0))

    def test_actuators_retain_locked_force_slots_at_scaled_pelvis_com(self):
        """Retain MTP actuators, six residuals, and official optimal forces."""
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.osim"
            self._write_model(model)
            coordinates, pelvis_com = rra.parse_model_spec(model)
            root = rra.build_actuator_xml(coordinates, pelvis_com)
            fy4_root = rra.build_actuator_xml(coordinates, pelvis_com, fy_optimal_force=4.0)
        coordinates_by_name = {element.get("name"): element for element in root.iter("CoordinateActuator")}
        self.assertEqual(len(coordinates_by_name), 17)
        self.assertEqual(coordinates_by_name["mtp_angle_r"].findtext("optimal_force"), "100.00000000")
        self.assertEqual(coordinates_by_name["mtp_angle_l"].findtext("optimal_force"), "100.00000000")
        points = list(root.iter("PointActuator"))
        torques = list(root.iter("TorqueActuator"))
        self.assertEqual([item.get("name") for item in points + torques], ["FX", "FY", "FZ", "MX", "MY", "MZ"])
        self.assertEqual({item.findtext("point") for item in points}, {"-0.06446740 0.00000000 0.00000000"})
        self.assertEqual(
            {item.get("name"): float(item.findtext("optimal_force", "nan")) for item in points + torques},
            {"FX": 4.0, "FY": 8.0, "FZ": 4.0, "MX": 2.0, "MY": 2.0, "MZ": 2.0},
        )
        fy4 = next(item for item in fy4_root.iter("PointActuator") if item.get("name") == "FY")
        self.assertEqual(float(fy4.findtext("optimal_force", "nan")), 4.0)

    def test_tasks_default_to_exact_upstream_and_offer_locked_omission_experiment(self):
        """Keep all 23 tasks by default; make the infeasible omission explicit."""
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.osim"
            self._write_model(model)
            coordinates, _ = rra.parse_model_spec(model)
            root = rra.build_task_xml(coordinates)
            omitted_root = rra.build_task_xml(coordinates, omit_locked_tasks=True)
        tasks = {item.get("name"): item for item in root.iter("CMC_Joint")}
        omitted_tasks = {item.get("name"): item for item in omitted_root.iter("CMC_Joint")}
        self.assertEqual(len(tasks), 23)
        self.assertIn("mtp_angle_r", tasks)
        self.assertIn("mtp_angle_l", tasks)
        self.assertEqual(len(omitted_tasks), 21)
        self.assertNotIn("mtp_angle_r", omitted_tasks)
        self.assertNotIn("mtp_angle_l", omitted_tasks)
        self.assertEqual(float(tasks["pelvis_tilt"].findtext("weight", "nan")), 1000.0)
        self.assertEqual(float(tasks["lumbar_rotation"].findtext("weight", "nan")), 10.0)
        for task in tasks.values():
            self.assertEqual(task.findtext("kp"), "100.00000000 1.00000000 1.00000000")
            self.assertEqual(task.findtext("kv"), "20.00000000 1.00000000 1.00000000")
            self.assertEqual(task.findtext("ka"), "1.00000000 1.00000000 1.00000000")

    def test_prepare_is_deterministic_clamps_time_and_does_not_import_opensim(self):
        """Prepare pure XML with pinned provenance and official settings."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / "data"
            output = base / "output"
            self._write_inputs(data)
            prepared = rra.prepare_reference(data, output, initial_time=-1.0, final_time=0.75)
            first = {path.name: path.read_bytes() for path in (output / "inputs").iterdir()}
            rra.prepare_reference(data, output, initial_time=-1.0, final_time=0.75)
            second = {path.name: path.read_bytes() for path in (output / "inputs").iterdir()}
            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            setup = ET.parse(prepared.setup_path).getroot().find("RRATool")
        self.assertEqual(first, second)
        self.assertEqual(manifest["scope"], "official_opensim_rra_reference_not_newton_native_prediction")
        self.assertEqual(manifest["time_range_s"]["effective"], [0.0, 0.75])
        self.assertEqual(manifest["pinned_upstream"]["commit"], rra._PINNED_COMMIT)
        self.assertFalse(manifest["method"]["mass_recommendation_automatically_applied"])
        self.assertEqual(manifest["method"]["locked_task_policy"], "exact_upstream_included")
        self.assertEqual(manifest["method"]["fy_optimal_force_N"], 8.0)
        self.assertTrue(manifest["method"]["fy_optimal_force_is_upstream_default"])
        self.assertEqual(set(manifest["model"]["locked_coordinates"]), {"mtp_angle_l", "mtp_angle_r"})
        grf_data = data / "trial_grf_context.mot"
        qc_path = data / "qc_summary.json"
        self.assertEqual(manifest["external_loads_datafile"]["path"], str(grf_data.resolve()))
        self.assertIn(str(grf_data.resolve()), manifest["source_inputs"])
        self.assertEqual(manifest["normalization_source"]["path"], str(qc_path.resolve()))
        self.assertIsNotNone(setup)
        assert setup is not None
        self.assertEqual(setup.findtext("lowpass_cutoff_frequency"), "6.00000000")
        self.assertEqual(setup.findtext("optimization_convergence_tolerance"), "0.00001000")
        self.assertEqual(setup.findtext("maximum_integrator_step_size"), "0.00100000")
        self.assertEqual(setup.findtext("adjusted_com_body"), "torso")

        command = [
            sys.executable,
            "-c",
            "import sys; import projects.gait_c3d.oracles.opensim_rra_reference; print('opensim' in sys.modules)",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout.strip(), "False")

    def test_run_rejects_any_mutated_prepared_input(self):
        """Hash all source and generated dependencies before importing OpenSim."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for name in ("actuator", "task", "setup", "model", "motion", "external_xml", "external_mot", "qc"):
                with self.subTest(name=name):
                    data = base / f"data_{name}"
                    output = base / f"output_{name}"
                    self._write_inputs(data)
                    rra.prepare_reference(data, output)
                    targets = {
                        "actuator": output / "inputs" / "gait2354_RRA_Actuators_S001.xml",
                        "task": output / "inputs" / "gait2354_RRA_Tasks_S001.xml",
                        "setup": output / "inputs" / "Setup_RRA.xml",
                        "model": data / "S001_scaled.osim",
                        "motion": data / "trial_ik_dynamics_context.mot",
                        "external_xml": data / "trial_grf_context.xml",
                        "external_mot": data / "trial_grf_context.mot",
                        "qc": data / "qc_summary.json",
                    }
                    targets[name].write_bytes(targets[name].read_bytes() + b"\nmutated")
                    with mock.patch.object(rra, "_import_official_opensim") as import_opensim:
                        with self.assertRaisesRegex(ValueError, "hash changed"):
                            rra.run_reference(output)
                    import_opensim.assert_not_called()

    def test_failed_rerun_replaces_stale_success_and_clears_results(self):
        """Never retain a prior successful runtime or its artifacts after failure."""

        class FakeLogger:
            @staticmethod
            def addFileSink(_path):
                return None

            @staticmethod
            def removeFileSink():
                return None

        class Tool:
            succeed = True

            def __init__(self, setup):
                self.output = Path(setup).parents[1]

            def run(self):
                (self.output / "results" / "attempt.txt").write_text(
                    "success" if self.succeed else "failed", encoding="utf-8"
                )
                return self.succeed

        class OpenSim:
            Logger = FakeLogger
            RRATool = Tool

            @staticmethod
            def GetVersionAndDate():
                return "synthetic OpenSim"

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / "data"
            output = base / "output"
            self._write_inputs(data)
            rra.prepare_reference(data, output)
            with mock.patch.object(rra, "_import_official_opensim", return_value=OpenSim):
                first = json.loads(rra.run_reference(output).read_text(encoding="utf-8"))
                self.assertTrue(first["success"])
                (output / "results" / "stale.txt").write_text("old", encoding="utf-8")
                Tool.succeed = False
                with self.assertRaisesRegex(RuntimeError, "returned false"):
                    rra.run_reference(output)
            second = json.loads((output / "run_runtime.json").read_text(encoding="utf-8"))
            self.assertFalse(second["success"])
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertFalse((output / "results" / "stale.txt").exists())
            with self.assertRaisesRegex(ValueError, "did not succeed"):
                rra.summarize_reference(output)

    def test_locked_perr_is_required_except_for_omission_experiment(self):
        """Default exact-upstream evidence must include locked-task pErr."""
        coordinates = [
            rra.CoordinateSpec("moving", "rotational", False),
            rra.CoordinateSpec("fixed", "rotational", True),
        ]
        with self.assertRaisesRegex(ValueError, "missing required task fixed"):
            rra.summarize_perr(["time", "moving"], [[0.0, 0.0]], coordinates)
        result = rra.summarize_perr(
            ["time", "moving"],
            [[0.0, 0.0]],
            coordinates,
            locked_task_policy="omit_experiment",
        )
        self.assertEqual(set(result), {"moving"})

    def test_summarize_requires_current_success_and_verified_artifacts(self):
        """Link summaries to one successful run and reject later mutations."""

        class FakeLogger:
            @staticmethod
            def addFileSink(_path):
                return None

            @staticmethod
            def removeFileSink():
                return None

        test_case = self

        class Tool:
            def __init__(self, setup):
                self.setup = Path(setup)

            def run(self):
                output = self.setup.parents[1]
                manifest = json.loads((output / "prepare_manifest.json").read_text(encoding="utf-8"))
                tool_name = manifest["tool_name"]
                results = output / "results"
                test_case._write_storage(
                    results / f"{tool_name}_Actuation_force.sto",
                    ["time", "FX", "FY", "FZ", "MX", "MY", "MZ"],
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                )
                test_case._write_storage(
                    results / f"{tool_name}_pErr.sto",
                    ["time", *rra._TASK_WEIGHTS],
                    [[0.0, *([0.0] * len(rra._TASK_WEIGHTS))]],
                )
                (results / f"{tool_name}_avgResiduals.txt").write_text(
                    "\n".join(f"{name} average = 0" for name in ("FX", "FY", "FZ", "MX", "MY", "MZ")),
                    encoding="utf-8",
                )
                model = next(Path(path) for path in manifest["source_inputs"] if Path(path).name == "S001_scaled.osim")
                (results / f"{tool_name}_adjusted.osim").write_bytes(model.read_bytes())
                (output / "rratool.log").write_text("synthetic successful RRA\n", encoding="utf-8")
                return True

        class OpenSim:
            Logger = FakeLogger
            RRATool = Tool

            @staticmethod
            def GetVersionAndDate():
                return "synthetic OpenSim"

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / "data"
            output = base / "output"
            self._write_inputs(data)
            rra.prepare_reference(data, output)
            with mock.patch.object(rra, "_import_official_opensim", return_value=OpenSim):
                runtime_path = rra.run_reference(output)
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            summary = json.loads(rra.summarize_reference(output).read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime_linkage"]["run_id"], runtime["run_id"])
            self.assertTrue(summary["gates"]["runtime_success"])
            self.assertTrue(summary["gates"]["production_candidate"])

            force_path = next((output / "results").glob("*_Actuation_force.sto"))
            original_force = force_path.read_bytes()
            force_path.write_bytes(original_force + b"\n")
            with self.assertRaisesRegex(ValueError, "artifacts do not match"):
                rra.summarize_reference(output)
            force_path.write_bytes(original_force)
            (data / "qc_summary.json").write_bytes((data / "qc_summary.json").read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "source input hash changed"):
                rra.summarize_reference(output)

    def test_storage_residual_and_perr_parsers(self):
        """Parse tables and apply official component and pErr thresholds."""
        labels = ["time", "FX", "FY", "FZ", "MX", "MY", "MZ"]
        rows = [[0.0, 4.0, 9.0, 26.0, 40.0, 60.0, 80.0], [1.0, -4.0, -9.0, -26.0, -40.0, -60.0, -80.0]]
        components = rra.summarize_residual_table(labels, rows)
        self.assertEqual(components["FX"]["grades"]["overall"], "GOOD")
        self.assertEqual(components["FY"]["grades"]["overall"], "OKAY")
        self.assertEqual(components["FZ"]["grades"]["overall"], "BAD")
        self.assertEqual(components["MX"]["grades"]["overall"], "OKAY")
        self.assertEqual(components["MZ"]["grades"]["overall"], "BAD")

        coordinates = [
            rra.CoordinateSpec("slide", "translational", False),
            rra.CoordinateSpec("turn", "rotational", False),
            rra.CoordinateSpec("fixed", "rotational", True),
        ]
        perr = rra.summarize_perr(
            ["time", "slide", "turn", "fixed"],
            [
                [0.0, 0.01, math.radians(1.0), math.radians(20.0)],
                [1.0, -0.01, math.radians(-1.0), math.radians(-20.0)],
            ],
            coordinates,
        )
        self.assertEqual(perr["slide"]["grades"]["overall"], "GOOD")
        self.assertAlmostEqual(perr["turn"]["rms"], 1.0)
        self.assertEqual(perr["fixed"]["grades"]["overall"], "BAD")
        self.assertTrue(perr["fixed"]["locked_coordinate_diagnostic"])
        self.assertFalse(perr["fixed"]["included_in_no_bad_perr_gate"])

    def test_average_and_log_parser_records_unapplied_mass_recommendation(self):
        """Extract official recommendations without representing them as applied."""
        averages = rra.parse_average_residuals(
            "Average Residuals:\nFX average = -1\nFY average = 2\nFZ average = 3\n"
            "MX average = 4\nMY average = 5\nMZ average = 6\n"
        )
        self.assertEqual(averages["FY"], 2.0)
        parsed = rra.parse_rra_log(
            """* Body adjusted: torso
* Mass Center (COM) adjustment: dx =-0.0806385, dz =0.00737031
* New COM location: ~[0.051968,0.305818,-0.00737031]
* Recommended mass adjustments:
*  Total mass change: -0.139662
*  pelvis: orig mass = 12.8372, new mass = 12.8153
*  torso: orig mass = 37.3187, new mass = 37.2551
"""
        )
        self.assertEqual(parsed["adjusted_body"], "torso")
        self.assertEqual(parsed["com_adjustment_m"], {"dx": -0.0806385, "dz": 0.00737031})
        self.assertAlmostEqual(parsed["recommended_total_mass_change_kg"], -0.139662)
        self.assertFalse(parsed["mass_recommendation_automatically_applied"])

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_OFFICIAL_OPENSIM_RRA") == "1"
        and Path("/home/jo31399/newton-data/gait/processed/trial_101/latest/S001_scaled.osim").is_file(),
        "set NEWTON_TEST_OFFICIAL_OPENSIM_RRA=1 to run the optional official RRATool integration",
    )
    def test_optional_official_rra_short_run(self):
        """Run a short official reference only when explicitly requested."""
        data = Path("/home/jo31399/newton-data/gait/processed/trial_101/latest")
        with tempfile.TemporaryDirectory(prefix="newton_official_rra_") as temporary:
            output = Path(temporary) / "reference"
            rra.prepare_reference(
                data, output, initial_time=20.60, final_time=20.62, tool_name="optional_real_rra", fy_optimal_force=4.0
            )
            runtime = json.loads(rra.run_reference(output).read_text(encoding="utf-8"))
            self.assertTrue(runtime["success"])
            summary = json.loads(rra.summarize_reference(output).read_text(encoding="utf-8"))
            self.assertEqual(summary["scope"], "official_opensim_rra_reference_not_newton_native_prediction")
            self.assertFalse(summary["anthropometry"]["silent_mass_application_detected"])


if __name__ == "__main__":
    unittest.main()
