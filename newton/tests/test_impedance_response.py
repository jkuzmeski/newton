# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test paired response reporting without retained data, Warp runs, or Torch."""

import copy
import json
import runpy
import tempfile
import unittest
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import numpy as np

from projects.impedance_instron.simple.reference import Reference
from projects.impedance_instron.simple.response import (
    _check_replay_sources,
    _panel,
    run_response_suite,
    summarize_response,
)

_MODULE = "projects.impedance_instron.simple.response"


def _trace():
    time = np.arange(12, dtype=float) * 0.01
    zeros = np.zeros_like(time)
    result = {
        name: zeros.copy()
        for name in (
            "pitch_rad",
            "leg_length_m",
            "foot_x_m",
            "pelvis_x_m",
            "pelvis_z_m",
            "tracking_error",
            "leg_source_power_w",
            "ankle_source_power_w",
            "leg_damping_power_w",
            "ankle_damping_power_w",
            "leg_limit_power_w",
            "ankle_limit_power_w",
            "leg_force_n",
            "leg_raw_force_n",
            "ankle_torque_n_m",
            "ankle_raw_torque_n_m",
            "leg_force_limited",
            "ankle_torque_limited",
            "push_force_x_n",
            "push_force_z_n",
            "push_power_w",
            "safety_flags",
            "shoe_fx_n",
            "leg_stiffness_n_m",
            "ankle_stiffness_n_m_rad",
            "leg_damping_n_s_m",
            "ankle_damping_n_m_s_rad",
            "last_clearance_m",
            "compression_m",
            "leg_nominal_force_n",
            "leg_feedback_force_n",
            "ankle_nominal_torque_n_m",
            "ankle_feedback_torque_n_m",
            "leg_nominal_body_power_w",
            "ankle_nominal_body_power_w",
            "leg_target_motion_power_w",
            "ankle_target_motion_power_w",
            "leg_stiffness_source_power_w",
            "ankle_stiffness_source_power_w",
            "leg_body_power_w",
            "ankle_body_power_w",
            "leg_spring_energy_j",
            "ankle_spring_energy_j",
        )
    }
    result.update(time_s=time, shoe_fz_n=np.full_like(time, 100.0), detected_touchdown_s=np.full_like(time, 0.01))
    result["pelvis_z_m"][:] = 1.0
    result["leg_length_m"][:] = 0.9
    return result


def _reference():
    time = np.array([0.0, 0.12])
    return Reference(
        time_s=time,
        pelvis_z_m=np.ones(2),
        pelvis_vz_m_s=np.zeros(2),
        pitch_rad=np.zeros(2),
        pitch_rate_rad_s=np.zeros(2),
        leg_length_m=np.ones(2),
        leg_rate_m_s=np.zeros(2),
        ankle_equilibrium_rad=np.zeros(2),
        ankle_equilibrium_rate_rad_s=np.zeros(2),
        inverse_leg_force_n=np.full(2, 800.0),
        inverse_ankle_torque_n_m=np.zeros(2),
        reference_fz_n=np.full(2, 800.0),
        reference_fx_n=np.zeros(2),
        foot_position_m=np.array([0, 0, 0.1]),
        upper_position_m=np.array([0, 0, 1.0]),
        foot_velocity_m_s=np.zeros(3),
        upper_velocity_m_s=np.zeros(3),
        mass_kg=80,
        gravity_m_s2=9.80665,
        contact_start_s=0.01,
        contact_duration_s=0.1,
        pelvis_scale_m=0.003,
        pitch_scale_rad=0.02,
        provenance={"inverse_dynamics": {"leg_damping_n_s_m": 483.7355, "ankle_damping_n_m_s_rad": 10.0}},
    )


@dataclass(frozen=True)
class _ResponseConfig:
    controller_mode: str = "equilibrium"
    leg_stiffness_n_m: float = 12000.0
    ankle_stiffness_n_m_rad: float = 4000.0
    leg_damping_n_s_m: float | None = None
    ankle_damping_n_m_s_rad: float | None = None
    ground_height_m: float = 0.0
    push_start_s: float = 0.0
    push_duration_s: float = 0.04
    push_force_x_n: float = 0.0
    push_force_z_n: float = 0.0

    def resolved(self, reference):
        values = {
            name: getattr(self, name)
            if getattr(self, name) is not None
            else reference.provenance["inverse_dynamics"][name]
            for name in ("leg_damping_n_s_m", "ankle_damping_n_m_s_rad")
        }
        return replace(self, **values)


@dataclass(frozen=True)
class _RigConfig:
    substeps: int = 1
    contact_threshold_n: float = 20.0
    force_limit_bw: float = 5.0
    ankle_torque_limit_n_m: float = 400.0

    def to_dict(self):
        return asdict(self)


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value.copy()


class _Rig:
    instances: ClassVar[list] = []
    fail_push = False
    nonfinite_baseline = False

    def __init__(self, reference, artifact_path, *, response_config, config=None, num_worlds=1, device=None):
        self.__class__.instances.append(self)
        self.reference = reference
        self.artifact_path = Path(artifact_path)
        self.response_config = response_config
        self.config = config or _RigConfig()
        self.num_worlds, self.device = num_worlds, device or "cpu"
        self.metadata = {
            "response_config": asdict(response_config),
            "response_fingerprints": {"controller": "test_hash"},
        }
        self.input_fingerprints = {"foundation_source_sha256": "test_foundation"}
        self.graph_status = "eager"
        self.sim_dt, self.sim_time = 0.01, 0.12
        self.data = _trace()
        scale = response_config.leg_stiffness_n_m / 12000
        perturb = response_config.ground_height_m + response_config.push_force_x_n / 10000
        self.data["pelvis_z_m"] += scale / 100 + perturb * np.linspace(0, 1, 12)
        self.data["foot_x_m"] += perturb
        self.data["pelvis_x_m"] += 2 * perturb
        self.data["pitch_rad"] += perturb
        self.data["leg_length_m"] += perturb
        for name in ("leg_stiffness_n_m", "ankle_stiffness_n_m_rad", "leg_damping_n_s_m", "ankle_damping_n_m_s_rad"):
            self.data[name][:] = getattr(response_config, name)
        self.data["push_force_x_n"][:] = response_config.push_force_x_n / 2
        self.data["push_force_z_n"][:] = response_config.push_force_z_n / 2
        self.data["push_power_w"][:] = response_config.push_force_x_n
        self.data["arbitrary_full_resolution_channel"] = np.arange(12)
        if self.nonfinite_baseline and perturb == 0:
            self.data["leg_source_power_w"][3] = np.nan
        q = np.array([[perturb, 0, 0.1, 0, 0, 0, 1], [2 * perturb, 0, 1 + scale / 100 + 2 * perturb, 0, 0, 0, 1]])
        self.state_0 = SimpleNamespace(body_q=_Array(q), body_qd=_Array(np.zeros((2, 6))))

    def trace(self, world):
        return copy.deepcopy(self.data)


def _evaluate(rig):
    if rig.fail_push and rig.response_config.push_force_x_n:
        raise FloatingPointError("synthetic divergent push")
    return {
        "tracking_loss": [0.0],
        "tracking_loss_mean": 0.0,
        "safety_ok": [True],
        "numerical_ok": [True],
        "safety_reasons": [[]],
        "status": "physically_valid",
    }


class TestResponseMetrics(unittest.TestCase):
    def test_pair_same_clock_and_actual_terminal_state(self):
        """Compare matching clocks and keep terminal state distinct from last trace."""
        baseline = _trace()
        trace = copy.deepcopy(baseline)
        trace["pelvis_z_m"] += np.linspace(0, 0.02, 12)
        trace["foot_x_m"] += 0.03
        trace["pelvis_x_m"] -= 0.04
        result = summarize_response(
            trace,
            baseline,
            0.01,
            terminal={"pelvis_z_m": 1.025, "time_s": 0.12},
            baseline_terminal={"pelvis_z_m": 1.0, "time_s": 0.12},
        )
        z = result["deviations"]["pelvis_z_m"]
        self.assertAlmostEqual(z["peak_abs_deviation"], 0.02)
        self.assertAlmostEqual(z["last_trace_deviation"], 0.02)
        self.assertAlmostEqual(z["final_deviation"], 0.025)
        self.assertAlmostEqual(result["deviations"]["foot_x_m"]["peak_abs_deviation"], 0.03)
        self.assertAlmostEqual(result["deviations"]["pelvis_x_m"]["peak_abs_deviation"], 0.04)
        self.assertIsNone(result["deviations"]["pitch_rad"]["final_deviation"])
        trace["time_s"] += 1e-6
        result = summarize_response(trace, baseline, 0.01)
        self.assertFalse(result["paired_clock_match"])
        self.assertIsNone(result["deviations"]["pelvis_z_m"]["peak_abs_deviation"])

    def test_partial_rollout_endpoints_are_not_compared(self):
        """Reject endpoint deltas at unequal or absent terminal physical times."""
        baseline = _trace()
        trace = {name: values[:6] for name, values in baseline.items()}
        baseline_terminal = {"pelvis_z_m": 1.0, "time_s": 0.12}
        for terminal in ({"pelvis_z_m": 1.1, "time_s": 0.06}, {"pelvis_z_m": 1.1}):
            result = summarize_response(trace, baseline, 0.01, terminal=terminal, baseline_terminal=baseline_terminal)
            self.assertFalse(result["paired_clock_match"])
            self.assertFalse(result["terminal_clock_match"])
            self.assertIsNone(result["deviations"]["pelvis_z_m"]["final_deviation"])

    def test_wrapped_pitch_and_rectangular_signed_work(self):
        """Wrap pitch differences and retain positive, negative, and signed work."""
        baseline = _trace()
        trace = copy.deepcopy(baseline)
        baseline["pitch_rad"][:] = np.pi - 0.01
        trace["pitch_rad"][:] = -np.pi + 0.02
        trace["leg_source_power_w"][:] = [-10, 20] * 6
        trace["ankle_source_power_w"][:] = 5
        trace["push_force_x_n"][:] = 25
        trace["push_power_w"][:] = 10
        trace["leg_force_limited"][0] = 1
        trace["leg_body_power_w"][:] = [15, -5] * 6
        trace["leg_spring_energy_j"][:] = np.linspace(2, 4, 12)
        trace["leg_nominal_body_power_w"][:] = 30
        trace["leg_target_motion_power_w"][:] = -15
        trace["leg_nominal_force_n"][:] = 90
        trace["last_clearance_m"][:] = 0.002
        trace["leg_force_n"][:] = -100
        trace["leg_raw_force_n"][:] = -120
        trace["detected_touchdown_s"][:] = 0.008
        trace["shoe_fz_n"][:6] = 0
        result = summarize_response(trace, baseline, 0.01, contact_start_s=0.01)
        self.assertAlmostEqual(result["deviations"]["pitch_rad"]["peak_abs_deviation"], 0.03)
        self.assertAlmostEqual(result["actuators"]["leg"]["signed_source_work_j"], 0.6)
        self.assertAlmostEqual(result["actuators"]["leg"]["positive_source_work_j"], 1.2)
        self.assertAlmostEqual(result["actuators"]["leg"]["negative_source_work_j"], -0.6)
        self.assertAlmostEqual(result["actuators"]["leg"]["limited_sample_fraction"], 1 / 12)
        self.assertEqual(result["actuators"]["leg"]["peak_abs_raw_force_n"], 120)
        self.assertAlmostEqual(result["actuators"]["leg"]["signed_body_work_j"], 0.6)
        self.assertAlmostEqual(result["actuators"]["leg"]["positive_body_work_j"], 0.9)
        self.assertAlmostEqual(result["actuators"]["leg"]["negative_body_work_j"], -0.3)
        self.assertEqual(result["actuators"]["leg"]["spring_energy_first_sample_j"], 2)
        self.assertEqual(result["actuators"]["leg"]["spring_energy_last_sample_j"], 4)
        self.assertEqual(result["actuators"]["leg"]["spring_energy_sampled_change_j"], 2)
        self.assertAlmostEqual(result["actuators"]["leg"]["nominal_body_signed_work_j"], 3.6)
        self.assertAlmostEqual(result["actuators"]["leg"]["target_motion_signed_work_j"], -1.8)
        self.assertEqual(result["actuators"]["leg"]["target_motion_positive_work_j"], 0)
        self.assertEqual(result["actuators"]["leg"]["peak_abs_nominal_force_n"], 90)
        self.assertAlmostEqual(result["minimum_last_clearance_m"], 0.002)
        self.assertAlmostEqual(result["shoe_vertical_impulse_n_s"], 6)
        self.assertEqual(result["push"]["impulse_x_n_s"], 3)
        self.assertAlmostEqual(result["push"]["signed_work_j"], 1.2)
        self.assertEqual(result["push"]["force_weighted_contact_fraction"], 0.5)
        self.assertAlmostEqual(result["contact"]["touchdown_delta_vs_baseline_s"], -0.002)

    def test_nonfinite_values_stay_invalid_and_json_safe(self):
        """Keep failed metrics null instead of dropping nonfinite samples."""
        trace, baseline = _trace(), _trace()
        trace["pelvis_z_m"][3] = np.nan
        trace["leg_source_power_w"][5] = np.inf
        trace["tracking_error"][-1] = np.inf
        trace["detected_touchdown_s"][:] = -1
        result = summarize_response(trace, baseline, 0.01, evaluation=_evaluate(SimpleNamespace(fail_push=False)))
        self.assertEqual(result["physical_validity"], "invalid")
        self.assertIsNone(result["tracking_loss"])
        self.assertIsNone(result["actuators"]["leg"]["signed_source_work_j"])
        self.assertIsNone(result["deviations"]["pelvis_z_m"]["peak_abs_deviation"])
        self.assertIsNone(result["contact"]["touchdown_delta_vs_baseline_s"])
        self.assertEqual(result["nonfinite_trace_counts"]["pelvis_z_m"], 1)
        self.assertFalse(result["contact"]["detected"])
        json.dumps(result, allow_nan=False)

    def test_plot_preserves_every_sample_and_breaks_invalid_gap(self):
        """Render all raw samples without smoothing, decimation, or gap bridging."""
        time = np.arange(2001)
        values = np.zeros(2001)
        values[100] = 1
        values[102] = np.nan
        panel = _panel(time, [("raw", values)], "Test", "m")
        curve = panel.split('fill="none"')[0].rsplit('<path d="', 1)[1]
        self.assertEqual(curve.count("M"), 2)
        self.assertEqual(curve.count("L"), 1998)
        self.assertNotIn("nan", panel.lower())


class TestResponseSuite(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.artifact = self.directory / "shoe.json"
        self.artifact.write_text('{"synthetic":true}\n')
        self.reference = _reference()
        _Rig.instances, _Rig.fail_push, _Rig.nonfinite_baseline = [], False, False
        types_patch = patch(_MODULE + "._response_types", return_value=(_Rig, _ResponseConfig))
        evaluate_patch = patch(_MODULE + ".evaluate_policy", side_effect=_evaluate)
        types_patch.start()
        self.evaluation = evaluate_patch.start()
        self.addCleanup(types_patch.stop)
        self.addCleanup(evaluate_patch.stop)

    def test_default_suite_fixed_damping_and_matching_baselines(self):
        """Run all 24 default cases with fixed damping and unchanged inputs."""
        frozen = self.reference.identity
        report = run_response_suite(
            self.reference, self.artifact, self.directory / "report", command=["response", "test"]
        )
        self.assertEqual(len(_Rig.instances), 24)
        self.assertEqual(self.evaluation.call_count, 24)
        self.assertEqual({rig.num_worlds for rig in _Rig.instances}, {1})
        self.assertEqual({rig.response_config.leg_damping_n_s_m for rig in _Rig.instances}, {483.7355})
        self.assertEqual({rig.response_config.ankle_damping_n_m_s_rad for rig in _Rig.instances}, {10.0})
        self.assertEqual({rig.response_config.leg_stiffness_n_m for rig in _Rig.instances}, {6000, 12000, 24000})
        self.assertEqual({rig.response_config.ground_height_m for rig in _Rig.instances}, {-0.005, 0, 0.005})
        self.assertEqual(self.reference.identity, frozen)
        record = json.loads(report.with_name("summary.json").read_text())
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["valid_case_count"], 24)
        self.assertEqual(record["commands"]["invocation_argv"], ["response", "test"])
        for start in range(0, 24, 4):
            baseline = record["cases"][start]
            for case in record["cases"][start : start + 4]:
                self.assertEqual(case["baseline_case_id"], baseline["case_id"])
                self.assertTrue(case["pair_valid"])
                self.assertEqual(case["input_fingerprints"]["foundation_source_sha256"], "test_foundation")
            push = record["cases"][start + 1]
            self.assertEqual(push["requested_push_impulse_n_s"], [3.0, 0.0])
            self.assertAlmostEqual(push["metrics"]["deviations"]["pelvis_z_m"]["peak_abs_deviation"], 0.015)
            self.assertAlmostEqual(push["metrics"]["deviations"]["pelvis_z_m"]["final_deviation"], 0.03)
        page = report.read_text()
        self.assertNotIn("https://", page)
        self.assertNotIn("<script src", page)
        self.assertIn("Upper-body horizontal position", page)
        self.assertIn("Foot horizontal position", page)
        self.assertIn("larger stiffness is not automatically better", page)
        self.assertIn("Intent mode applies frozen reduced-rig inverse-dynamics loads at runtime", page)
        self.assertIn("nominal and feedback commands", page)
        self.assertIn("nominal body and target-motion power", page)
        self.assertTrue(report.with_name("replay.py").is_file())
        self.assertEqual(report.with_name("artifact.json").read_bytes(), self.artifact.read_bytes())
        self.assertEqual(Reference.load(report.with_name("reference.json")).identity, frozen)
        with np.load(report.parent / record["cases"][0]["trace_file"], allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace["arbitrary_full_resolution_channel"], np.arange(12))
            self.assertEqual(trace["terminal_body_q"].shape, (2, 7))
            self.assertAlmostEqual(float(trace["terminal_time_s"]), 0.12)

    def test_never_overwrite_nonempty_output(self):
        """Reject a nonempty destination before any physics call or file change."""
        output = self.directory / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("untouched")
        with self.assertRaises(FileExistsError):
            run_response_suite(self.reference, self.artifact, output)
        self.assertEqual(marker.read_text(), "untouched")
        self.assertFalse(_Rig.instances)
        self.assertEqual(list(output.iterdir()), [marker])

    def test_failed_push_and_nonfinite_baseline_remain_visible(self):
        """Keep failed cases, partial traces, and invalid paired baselines."""
        _Rig.fail_push = True
        _Rig.nonfinite_baseline = True
        report = run_response_suite(
            self.reference, self.artifact, self.directory / "failed", modes=("intent",), stiffness_multipliers=(1,)
        )
        record = json.loads(report.with_name("summary.json").read_text())
        self.assertEqual(len(record["cases"]), 4)
        baseline, push, raised, lowered = record["cases"]
        self.assertEqual(baseline["status"], "invalid")
        self.assertEqual(push["status"], "execution_error")
        self.assertIn("synthetic divergent push", push["error"]["message"])
        self.assertEqual(raised["status"], "valid")
        self.assertFalse(raised["pair_valid"])
        self.assertFalse(lowered["pair_valid"])
        self.assertEqual(raised["baseline_status"], "invalid")
        self.assertIsNone(baseline["metrics"]["actuators"]["leg"]["signed_source_work_j"])
        with np.load(report.parent / baseline["trace_file"], allow_pickle=False) as trace:
            self.assertTrue(np.isnan(trace["leg_source_power_w"][3]))
        self.assertTrue((report.parent / push["trace_file"]).is_file())
        self.assertIn("execution_error", report.read_text())
        self.assertIn("valid; INVALID PAIR", report.read_text())
        self.assertIn("not a qualified response comparison", report.read_text())

    def test_freeze_artifact_snapshot_for_every_case(self):
        """Keep paired inputs fixed even when the original artifact changes."""
        original = self.artifact.read_bytes()

        def mutate_source(rig):
            self.artifact.write_text('{"modified":true}')
            self.assertEqual(rig.artifact_path.read_bytes(), original)
            return _evaluate(rig)

        self.evaluation.side_effect = mutate_source
        report = run_response_suite(
            self.reference, self.artifact, self.directory / "snapshot", modes=("intent",), stiffness_multipliers=(1,)
        )
        self.assertEqual({rig.artifact_path for rig in _Rig.instances}, {report.with_name("artifact.json")})
        self.assertEqual(report.with_name("artifact.json").read_bytes(), original)
        self.assertNotEqual(self.artifact.read_bytes(), original)

    def test_replay_script_rejects_changed_input_snapshot(self):
        """Stop generated replay before physics when an input snapshot changes."""
        report = run_response_suite(
            self.reference,
            self.artifact,
            self.directory / "replay_guard",
            modes=("intent",),
            stiffness_multipliers=(1,),
        )
        script = report.with_name("replay.py")
        replay_source = script.read_text()
        self.assertIn('"old_results_inherited": False', replay_source)
        self.assertIn('"source_matches": not bool(changed)', replay_source)
        self.assertIn('"requested_device_override": args.device', replay_source)
        self.assertNotIn('"old_results_apply"', replay_source)
        for name in ("artifact.json", "reference.json"):
            path = report.with_name(name)
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            try:
                with (
                    patch("sys.argv", [str(script), str(self.directory / "new_replay")]),
                    patch(_MODULE + ".run_response_suite") as rerun,
                    self.assertRaisesRegex(ValueError, "Saved replay input changed"),
                ):
                    runpy.run_path(str(script), run_name="__main__")
                rerun.assert_not_called()
            finally:
                path.write_bytes(original)
        self.assertFalse((self.directory / "new_replay").exists())

    def test_replay_rejects_source_changes_by_default(self):
        """Require explicit source-update permission and return exact mismatches."""
        with patch(_MODULE + "._source_fingerprints", return_value={"rig.py": "new"}):
            with self.assertRaisesRegex(ValueError, "old results do not apply"):
                _check_replay_sources({"rig.py": "old"})
            changed = _check_replay_sources({"rig.py": "old"}, allow_physics_update=True)
            self.assertEqual(changed, {"rig.py": {"saved": "old", "current": "new"}})
            self.assertEqual(_check_replay_sources({"rig.py": "new"}), {})

    def test_all_constructor_failures_still_produce_report(self):
        """Retain every construction failure and a replayable null rig config."""
        failed = Mock(side_effect=ValueError("synthetic construction failure"))
        with patch(_MODULE + "._response_types", return_value=(failed, _ResponseConfig)):
            report = run_response_suite(
                self.reference,
                self.artifact,
                self.directory / "construction",
                modes=("intent",),
                stiffness_multipliers=(1,),
            )
        record = json.loads(report.with_name("summary.json").read_text())
        self.assertEqual(len(record["cases"]), 4)
        self.assertEqual({case["status"] for case in record["cases"]}, {"execution_error"})
        self.assertEqual(record["invalid_or_failed_case_count"], 4)
        self.assertIsNone(record["rig_config"])
        self.assertIn('if record["rig_config"] is not None else None', report.with_name("replay.py").read_text())
        self.assertTrue(report.is_file())

    def test_validate_protocol_before_starting(self):
        """Reject invalid sweep settings and pushes outside nominal contact."""
        for options in (
            {"stiffness_multipliers": (1, 1)},
            {"stiffness_multipliers": (np.nan,)},
            {"modes": ("unknown",)},
            {"ground_offset_m": -0.005},
            {"push_start_s": 0.0},
            {"push_force_x_n": 0.0},
            {"push_duration_s": 0.2},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_response_suite(self.reference, self.artifact, self.directory / "bad", **options)
        self.assertFalse(_Rig.instances)
        self.assertFalse((self.directory / "bad").exists())


if __name__ == "__main__":
    unittest.main()
