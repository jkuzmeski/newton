# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test fixed-gain suite coverage, pairing, frozen inputs and replay guards."""

import copy
import hashlib
import io
import json
import runpy
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from newton.tests.test_impedance_response import _Array, _reference, _ResponseConfig, _trace
from projects.impedance_instron.simple import sensitivity

_MODULE = "projects.impedance_instron.simple.sensitivity"
_REAL_SOURCES = sensitivity._sensitivity_sources


@dataclass(frozen=True)
class _Config:
    frame_rate_hz: float = 100.0
    substeps: int = 1
    contact_threshold_n: float = 20.0
    force_limit_bw: float = 5.0
    ankle_torque_limit_n_m: float = 400.0

    def to_dict(self):
        return asdict(self)


class _Rig:
    instances: ClassVar[list] = []
    hook = None
    invalid_original = False
    fail_push = False
    fail_terminal = False
    short_terminal = False

    def __init__(self, reference, artifact_path, *, response_config, config, num_worlds, device):
        self.__class__.instances.append(self)
        self.reference = reference
        self.artifact_path = Path(artifact_path)
        self.material = json.loads(self.artifact_path.read_text())
        self.response_config = response_config
        self.config, self.num_worlds, self.device = config, num_worlds, device or "cpu"
        self.metadata = {"native_mock": True}
        self.input_fingerprints = {"artifact_sha256": hashlib.sha256(self.artifact_path.read_bytes()).hexdigest()}
        self.graph_status = "eager"
        self.sim_dt, self.sim_time = 0.01, 0.12
        self.data = _trace()
        for name in (
            "foot_z_m",
            "pelvis_vx_m_s",
            "pelvis_vz_m_s",
            "foot_vx_m_s",
            "foot_vz_m_s",
            "leg_rate_m_s",
            "pitch_rate_rad_s",
        ):
            self.data[name] = np.zeros(12)
        self.data["foot_z_m"][:] = 0.1
        perturbation = (response_config.push_force_x_n + response_config.push_force_z_n) * 1e-5
        self.data["pelvis_z_m"] += self.material["value"] * 0.001 + perturbation
        self.data["push_force_x_n"][:] = response_config.push_force_x_n / 2
        self.data["push_force_z_n"][:] = response_config.push_force_z_n / 2
        for name in sensitivity._GAIN_NAMES:
            self.data[name][:] = getattr(response_config, name)
        self.data["arbitrary_numeric_channel"] = np.arange(12)
        self.invalid = self.invalid_original and self.material["value"] == 1
        if self.invalid:
            self.data["safety_flags"][0] = 1
        q = [[0, 0, 0.1, 0, 0, 0, 1], [0, 0, self.data["pelvis_z_m"][-1] + 0.002, 0, 0, 0, 1]]
        qd = np.zeros((2, 6))
        qd[1, 2] = 0.001
        self.state_0 = SimpleNamespace(body_q=_Array(q), body_qd=_Array(qd))
        if self.fail_terminal:
            self.state_0.body_q = None
        if self.short_terminal and perturbation:
            self.sim_time -= 0.01
        if self.hook is not None:
            self.hook()

    def trace(self, world):
        return copy.deepcopy(self.data)


def _evaluate(rig):
    if rig.fail_push and (rig.response_config.push_force_x_n or rig.response_config.push_force_z_n):
        raise FloatingPointError("Synthetic integration failure")
    return {"tracking_loss_mean": 0.0, "safety_ok": [not rig.invalid], "numerical_ok": [True]}


def _materials(artifact_path, output_dir, *, modulus_multipliers, relaxation_multipliers, material_paths):
    sources = [("baseline", Path(artifact_path).read_bytes(), "baseline")]
    for kind, values in (("modulus", modulus_multipliers), ("relaxation", relaxation_multipliers)):
        for index, value in enumerate(values):
            sources.append((f"{kind}_{index}", json.dumps({"value": value}).encode(), kind))
    for index, path in enumerate(material_paths):
        sources.append((f"imported_{index}", Path(path).read_bytes(), "imported"))
    output_dir.mkdir()
    result = []
    for name, value, kind in sources:
        path = output_dir / f"{name}.json"
        path.write_bytes(value)
        result.append(
            {
                "id": name,
                "path": str(path),
                "sha256": hashlib.sha256(value).hexdigest(),
                "baseline": kind == "baseline",
                "type": kind,
                "parameters": json.loads(value),
                "material_identity": hashlib.sha256(value).hexdigest(),
                "geometry_identity": "same",
            }
        )
    return result


def _report(destination, record):
    path = destination / "report.html"
    path.write_text("<!doctype html><title>Mock report</title>")
    return path


class TestSensitivitySuite(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.artifact = self.directory / "artifact.json"
        self.artifact.write_text('{"value": 1}')
        provenance = copy.deepcopy(_reference().provenance)
        provenance["config"] = {"nominal_leg_stiffness_n_m": 9000.0, "nominal_ankle_stiffness_n_m_rad": 2500.0}
        self.reference = replace(_reference(), provenance=provenance)
        self.source_reference = self.directory / "source_reference.json"
        self.reference.save(self.source_reference)
        _Rig.instances = []
        _Rig.hook = None
        _Rig.invalid_original = _Rig.fail_push = _Rig.fail_terminal = _Rig.short_terminal = False
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch(f"{_MODULE}._response_types", return_value=(_Rig, _ResponseConfig)))
        self.stack.enter_context(patch(f"{_MODULE}._resolve_rig_config", return_value=_Config()))
        self.stack.enter_context(
            patch(f"{_MODULE}._rig_config_from_dict", side_effect=lambda values: _Config(**values))
        )
        self.stack.enter_context(patch(f"{_MODULE}.evaluate_policy", side_effect=_evaluate))
        self.sources = self.stack.enter_context(
            patch(f"{_MODULE}._sensitivity_sources", return_value={"native.py": "old"})
        )
        self.builder = self.stack.enter_context(
            patch("projects.impedance_instron.simple.material_variants.build_material_variants", side_effect=_materials)
        )
        self.stack.enter_context(
            patch("projects.impedance_instron.simple.sensitivity_report.write_sensitivity_report", side_effect=_report)
        )
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def run_suite(self, name="output", **options):
        return sensitivity.run_sensitivity_suite(self.source_reference, self.artifact, self.directory / name, **options)

    def small_suite(self, name="output", **options):
        return self.run_suite(name, gain_multipliers=(), modulus_multipliers=(), relaxation_multipliers=(), **options)

    def record(self, report):
        return json.loads(report.with_name("summary.json").read_text())

    def test_sparse_coverage_and_frozen_nominal_oat(self):
        """Run all 97 sparse cases with four independent gain axes and exact pairing."""
        record = self.record(self.run_suite())
        self.assertEqual(len(record["cases"]), 97)
        self.assertEqual(len(record["controllers"]), 9)
        self.assertEqual(len(record["materials"]), 5)
        nominal = record["controllers"][0]["gains"]
        self.assertEqual(nominal["leg_stiffness_n_m"], 9000.0)
        self.assertEqual(nominal["ankle_stiffness_n_m_rad"], 2500.0)
        self.assertEqual(nominal["leg_damping_n_s_m"], 483.7355)
        for controller in record["controllers"][1:]:
            changed = [name for name in nominal if nominal[name] != controller["gains"][name]]
            self.assertEqual(changed, [controller["varied_gain"]])
        cases = {case["case_id"]: case for case in record["cases"]}
        quiet = [case for case in cases.values() if case["direction"] is None]
        pushes = [case for case in cases.values() if case["direction"] is not None]
        self.assertEqual((len(quiet), len(pushes)), (45, 52))
        self.assertEqual(sum(case["material_id"] == "baseline" for case in pushes), 36)
        self.assertEqual(sum(case["material_id"] != "baseline" for case in pushes), 16)
        for case in cases.values():
            baseline = cases[case["baseline_case_id"]]
            self.assertEqual(case["controller_id"], baseline["controller_id"])
            self.assertEqual(case["controller_mode"], baseline["controller_mode"])
            self.assertIsNone(baseline["direction"])
            self.assertTrue(case["pair_valid"])
            if case["direction"] is not None:
                self.assertEqual(case["material_id"], baseline["material_id"])
                settings = case["response_config"]
                self.assertEqual(np.hypot(settings["push_force_x_n"], settings["push_force_z_n"]), 150.0)
                self.assertEqual(case["recovery"]["status"], "insufficient_window")
            else:
                self.assertEqual(baseline["material_id"], "baseline")
                expected = (
                    "unperturbed_baseline"
                    if case["comparison_type"] == "unperturbed_baseline"
                    else "persistent_material_change"
                )
                self.assertEqual(case["recovery"]["status"], expected)
            with np.load(self.directory / "output" / case["trace_file"], allow_pickle=False) as trace:
                self.assertEqual(trace["arbitrary_numeric_channel"].shape, (12,))
                self.assertEqual(trace["terminal_body_qd"].shape, (2, 6))
                self.assertNotEqual(float(trace["terminal_pelvis_z_m"]), trace["pelvis_z_m"][-1])
        self.assertEqual(record["runtime"]["environment_archive"], False)
        self.assertTrue(all(rig.num_worlds == 1 for rig in _Rig.instances))

    def test_full_factorial_and_optional_modes(self):
        """Cover 225 cases per mode without combining gain axes."""
        controllers = sensitivity._controllers(self.reference, (0.5, 2.0))
        materials = [{"id": str(index)} for index in range(5)]
        for modes, full, expected in (
            (("intent",), True, 225),
            (("intent", "equilibrium"), False, 194),
            (("intent", "equilibrium"), True, 450),
        ):
            plan = sensitivity._plan(controllers, materials, modes, full)
            self.assertEqual(len(plan), expected)
            self.assertEqual(len({case["case_id"] for case in plan}), expected)

    def test_multiplier_deduplication_and_zero_damping(self):
        """Remove duplicate and neutral multipliers without inventing nonzero damping."""
        self.assertEqual(sensitivity._multipliers((1, 0.5, 0.5, 2, 1), "test"), (0.5, 2.0))
        zero = self.reference.to_dict()
        provenance = copy.deepcopy(self.reference.provenance)
        provenance["inverse_dynamics"]["leg_damping_n_s_m"] = 0.0
        reference = replace(self.reference, provenance=provenance)
        controllers = sensitivity._controllers(reference, (0.5, 2.0))
        self.assertEqual(len(controllers), 7)
        self.assertTrue(all(item["gains"]["leg_damping_n_s_m"] == 0 for item in controllers))
        self.assertEqual(zero, self.reference.to_dict())
        for values in ((0,), (-1,), (float("nan"),), (float("inf"),)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                sensitivity._multipliers(values, "test")

    def test_snapshot_only_and_replay_exact_variants(self):
        """Ignore later source mutation and replay imported snapshots without regeneration."""
        imported = self.directory / "imported.json"
        imported.write_text('{"value": 4}')
        original_identity = self.reference.identity

        def mutate_sources():
            self.artifact.write_text("tampered")
            self.source_reference.write_text("tampered")
            imported.write_text("tampered")
            self.reference.provenance["config"]["nominal_leg_stiffness_n_m"] = 999999

        _Rig.hook = staticmethod(mutate_sources)
        record = self.record(self.small_suite(material_paths=(imported,)))
        self.assertEqual(record["reference_identity"], original_identity)
        self.assertEqual({rig.material["value"] for rig in _Rig.instances}, {1, 4})
        self.assertTrue(
            all(rig.artifact_path.is_relative_to(self.directory / "output/materials") for rig in _Rig.instances)
        )
        self.assertTrue(all(rig.reference.identity == original_identity for rig in _Rig.instances))
        _Rig.hook = None
        self.builder.side_effect = AssertionError("Replay must never regenerate variants")
        replay = sensitivity.replay_sensitivity(self.directory / "output", self.directory / "replayed")
        replay_record = self.record(replay)
        self.assertEqual(record["materials"], replay_record["materials"])
        self.assertEqual(record["snapshot_manifest"], replay_record["snapshot_manifest"])
        self.assertFalse(replay_record["replay"]["old_results_inherited"])

    def test_reference_object_is_detached_before_instantiation(self):
        """Freeze caller-owned nested provenance before any rig can observe mutation."""
        identity = self.reference.identity

        def mutate_reference():
            self.reference.provenance["config"]["nominal_leg_stiffness_n_m"] = 1e9

        _Rig.hook = staticmethod(mutate_reference)
        report = sensitivity.run_sensitivity_suite(
            self.reference,
            self.artifact,
            self.directory / "object",
            gain_multipliers=(),
            modulus_multipliers=(),
            relaxation_multipliers=(),
        )
        record = self.record(report)
        self.assertEqual(record["reference_identity"], identity)
        self.assertIsNone(record["reference_source_path"])
        self.assertTrue(all(rig.reference.identity == identity for rig in _Rig.instances))
        self.assertEqual(len({id(rig.reference) for rig in _Rig.instances}), 5)

    def test_generated_replay_uses_fresh_scores(self):
        """Execute the generated replay entrypoint without inheriting edited old scores."""
        report = self.small_suite()
        record = self.record(report)
        for case in record["cases"]:
            case["metrics"]["tracking_loss"] = 9999
            case["status"] = "invented_old_score"
        report.with_name("summary.json").write_text(json.dumps(record))
        target = self.directory / "script_replay"
        script = report.with_name("replay.py")
        with patch("sys.argv", [str(script), str(target), "--device", "cpu"]):
            runpy.run_path(str(script), run_name="__main__")
        fresh = self.record(target / "report.html")
        self.assertTrue(all(case["status"] == "valid" for case in fresh["cases"]))
        self.assertTrue(all(case["metrics"]["tracking_loss"] == 0 for case in fresh["cases"]))
        self.assertEqual(fresh["replay"]["requested_device_override"], "cpu")
        self.assertEqual(fresh["commands"]["invocation_argv"][-2:], ["--device", "cpu"])

    def test_pair_validity_not_inherited_from_material_comparison(self):
        """Allow valid variant pushes even when their quiet material comparison is invalid."""
        _Rig.invalid_original = True
        record = self.record(
            self.run_suite(gain_multipliers=(), modulus_multipliers=(0.75,), relaxation_multipliers=())
        )
        original = [case for case in record["cases"] if case["material_id"] == "baseline"]
        variant = [case for case in record["cases"] if case["material_id"] != "baseline"]
        self.assertTrue(all(not case["pair_valid"] for case in original))
        self.assertFalse(variant[0]["pair_valid"])
        self.assertEqual(variant[0]["status"], "valid")
        self.assertTrue(all(case["pair_valid"] for case in variant[1:]))
        self.assertTrue(all(case["recovery"]["status"] == "invalid_pair" for case in original))

    def test_execution_errors_and_true_terminal_required(self):
        """Retain broken evaluations and reject missing or unequal terminal clocks."""
        for flag in ("fail_push", "fail_terminal", "short_terminal"):
            with self.subTest(flag=flag):
                setattr(_Rig, flag, True)
                record = self.record(self.small_suite(flag))
                pushes = [case for case in record["cases"] if case["direction"] is not None]
                self.assertTrue(all(not case["pair_valid"] for case in pushes))
                self.assertTrue(all(case["recovery"]["status"] == "invalid_pair" for case in pushes))
                self.assertTrue(all((self.directory / flag / case["trace_file"]).is_file() for case in pushes))
                setattr(_Rig, flag, False)

    def test_replay_configuration_input_and_source_guards(self):
        """Reject setting edits and input edits even when source updates are allowed."""
        report = self.small_suite()
        directory = report.parent
        summary = directory / "summary.json"
        original = summary.read_bytes()
        for name in ("suite_config", "rig_config", "controllers", "recovery_config"):
            record = json.loads(original)
            if name == "controllers":
                record[name][0]["gains"]["leg_stiffness_n_m"] *= 2
            else:
                record[name][next(iter(record[name]))] = "tampered"
            summary.write_text(json.dumps(record))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "seal"):
                sensitivity.replay_sensitivity(directory, self.directory / "rejected", allow_physics_update=True)
        summary.write_bytes(original)
        material = directory / "materials/baseline.json"
        payload = material.read_bytes()
        material.write_text('{"value": 999}')
        with self.assertRaisesRegex(ValueError, "input changed"):
            sensitivity.replay_sensitivity(directory, self.directory / "rejected", allow_physics_update=True)
        material.write_bytes(payload)
        self.sources.return_value = {"native.py": "new"}
        with self.assertRaisesRegex(ValueError, "sources changed"):
            sensitivity.replay_sensitivity(directory, self.directory / "rejected")
        with self.assertWarnsRegex(UserWarning, "NEW EXPERIMENT"):
            changed_report = sensitivity.replay_sensitivity(
                directory, self.directory / "updated", allow_physics_update=True
            )
        changed = self.record(changed_report)
        self.assertEqual(changed["source_fingerprints"], {"native.py": "new"})
        self.assertTrue(changed["replay"]["changed_sources"])
        self.assertFalse(changed["replay"]["old_results_inherited"])
        self.assertEqual(len(changed["cases"]), 5)

    def test_original_reference_rejects_duplicate_json_keys(self):
        """Reject ambiguous source JSON before freezing a new reference snapshot."""
        payload = self.source_reference.read_text()
        self.source_reference.write_text(payload.replace("{", '{"schema_version":"impedance_simple_reference_1",', 1))
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.small_suite()
        self.assertFalse((self.directory / "output").exists())
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
            sensitivity._decode_json('{"value": NaN}')

    def test_no_overwrites_or_missing_nominal_fallback(self):
        """Refuse occupied outputs and references without nominal construction gains."""
        self.small_suite()
        with self.assertRaises(FileExistsError):
            self.small_suite()
        with self.assertRaises(FileExistsError):
            sensitivity.replay_sensitivity(self.directory / "output", self.directory / "output")
        occupied = self.directory / "occupied"
        occupied.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.small_suite("occupied")
        with self.assertRaisesRegex(ValueError, "nominal"):
            sensitivity._controllers(_reference(), (0.5, 2.0))

    def test_guard_fingerprints_cover_new_and_existing_sources(self):
        """Fingerprint all sensitivity helpers and the existing response source set."""
        with patch(f"{_MODULE}._source_fingerprints", return_value={"native_existing.py": "digest"}):
            current = _REAL_SOURCES()
        self.assertEqual(current["native_existing.py"], "digest")
        self.assertIn("newton/_src/solvers/solver.py", current)
        self.assertIn("newton/_src/sim/model.py", current)
        self.assertIn("newton/_src/sim/state.py", current)
        for name in ("sensitivity.py", "material_variants.py", "recovery.py", "sensitivity_report.py", "report.py"):
            self.assertIn(f"projects/impedance_instron/simple/{name}", current)


if __name__ == "__main__":
    unittest.main()
