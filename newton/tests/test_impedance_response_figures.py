# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test figure-first response rendering with temporary saved traces only."""

import copy
import json
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.simple.figures import BASELINE_COLOR, GAIN_COLORS, line_figure
from projects.impedance_instron.simple.response import _response_overview, _response_terminal, _write_html


class _TableVisibility(HTMLParser):
    def __init__(self):
        super().__init__()
        self.collapsed = []
        self.tables = []

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.collapsed.append("open" not in dict(attrs))
        elif tag == "table":
            self.tables.append(any(self.collapsed))

    def handle_endtag(self, tag):
        if tag == "details":
            self.collapsed.pop()


class TestImpedanceResponseFigures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "cases").mkdir()
        self.record = {"cases": [], "suite_config": {"modes": ["equilibrium", "intent"]}}
        for mode in ("equilibrium", "intent"):
            for gain in (0.5, 1.0, 2.0):
                self._add_pair(mode, gain)

    def _add_pair(self, mode, gain):
        baseline_id = f"{mode}_k{gain:g}_unperturbed"
        for disturbed in (False, True):
            case_id = f"{mode}_k{gain:g}_{'push' if disturbed else 'unperturbed'}"
            time = np.arange(4, dtype=float) * 0.01
            displacement = np.array([0.0, 0.001, 0.002, 0.003]) / gain if disturbed else np.zeros(4)
            velocity = np.array([0.0, 0.002, 0.003, 0.004]) / gain if disturbed else np.zeros(4)
            trace = {
                "time_s": time,
                "pelvis_z_m": 1.0 + 0.02 * gain + time * 0.01,
                "pitch_rad": 0.001 * gain + time * 0.1,
                "reference_pelvis_z_m": np.ones(4),
                "reference_pitch_rad": np.zeros(4),
                "pelvis_x_m": 0.02 * gain + time * 0.1 + displacement,
                "pelvis_vx_m_s": 0.1 + 0.03 * gain + velocity,
                "foot_x_m": time * 0.1,
                "leg_length_m": np.ones(4),
            }
            terminal_angle = 0.001 * gain + 0.004
            q = np.zeros((2, 7))
            q[:, 6] = 1.0
            q[0, 4] = np.sin(terminal_angle / 2)
            q[0, 6] = np.cos(terminal_angle / 2)
            q[1, 2] = 1.0 + 0.02 * gain + 0.0004
            q[1, 0] = 0.02 * gain + 0.004 + (0.007 / gain if disturbed else 0.0)
            qd = np.zeros((2, 6))
            qd[1, 0] = 0.1 + 0.03 * gain + (0.009 / gain if disturbed else 0.0)
            trace.update(terminal_body_q=q, terminal_body_qd=qd, terminal_time_s=np.asarray(0.04))
            filename = f"cases/{case_id}.npz"
            np.savez_compressed(self.directory / filename, **trace)
            self.record["cases"].append(
                {
                    "case_id": case_id,
                    "controller_mode": mode,
                    "stiffness_multiplier": gain,
                    "perturbation": "push" if disturbed else "unperturbed",
                    "trace_file": filename,
                    "baseline_trace_file": f"cases/{baseline_id}.npz",
                    "baseline_case_id": baseline_id,
                    "status": "valid",
                    "pair_valid": True,
                    "metrics": {"tracking_loss": gain + 0.123},
                    "response_config": {
                        "controller_mode": mode,
                        "leg_stiffness_n_m": 12000 * gain,
                        "ankle_stiffness_n_m_rad": 4000 * gain,
                        "leg_damping_n_s_m": 10.0,
                        "ankle_damping_n_m_s_rad": 2.0,
                        "ground_height_m": 0.0,
                        "push_force_x_n": 100.0 if disturbed else 0.0,
                        "push_force_z_n": 0.0,
                        "push_start_s": 0.01,
                        "push_duration_s": 0.02,
                    },
                    "input_fingerprints": {"reference_identity": "frozen", "material_identity": "fixed"},
                }
            )

    def _capture(self, record=None):
        calls = []

        def render(curves, **kwargs):
            calls.append((curves, kwargs))
            return line_figure(curves, **kwargs)

        with patch("projects.impedance_instron.simple.figures.line_figure", side_effect=render):
            page = _response_overview(self.directory, self.record if record is None else record)
        return page, calls

    def _replace_trace(self, case, **changes):
        path = self.directory / case["trace_file"]
        with np.load(path, allow_pickle=False) as saved:
            trace = dict(saved)
        trace.update(changes)
        np.savez_compressed(path, **trace)

    def test_figures_precede_collapsed_tables_and_preserve_saved_inputs(self):
        """Put figures before collapsed tables without mutating saved evidence."""
        before = copy.deepcopy(self.record)
        summary = self.directory / "summary.json"
        summary.write_text(json.dumps(self.record))
        files = {path: path.read_bytes() for path in self.directory.rglob("*.npz")}
        summary_bytes = summary.read_bytes()
        page = _write_html(self.directory, self.record).read_text()
        self.assertLess(page.index("<figure"), page.index("<table"))
        visibility = _TableVisibility()
        visibility.feed(page)
        self.assertTrue(visibility.tables)
        self.assertTrue(all(visibility.tables))
        self.assertIn("Raw case plots and metadata", page)
        self.assertIn('href="cases/response_000.html"', page)
        self.assertNotIn("Leg applied and unclamped force", page)
        detail = (self.directory / "cases/response_000.html").read_text()
        self.assertIn("Leg applied and unclamped force", detail)
        self.assertIn("Time [s]", detail)
        self.assertEqual(self.record, before)
        self.assertEqual(summary.read_bytes(), summary_bytes)
        for path, contents in files.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertNotIn("source_type", self.record)
        exports = sorted((self.directory / "figures").glob("response_*.svg"))
        self.assertEqual(len(exports), 10)
        self.assertIn('href="figures/response_00.svg" download', page)
        svg = exports[0].read_text()
        self.assertIn("Height [mm]", svg)
        self.assertIn("Time [ms]", svg)
        self.assertIn("equilibrium", svg)
        self.assertIn("intent", svg)

    def test_compare_both_nominal_controllers_and_separate_stiffness_groups(self):
        """Overlay both nominal laws and separate each mode's actual gain sweep."""
        page, calls = self._capture()
        nominal = [(curves, options) for curves, options in calls if options["title"].startswith("Nominal")]
        self.assertEqual(len(nominal), 2)
        self.assertEqual({options["y_label"] for _, options in nominal}, {"Height [mm]", "Foot pitch [mrad]"})
        for curves, _ in nominal:
            self.assertEqual(len(curves), 2)
            self.assertTrue(all("K x 1" in curve.label for curve in curves))
            self.assertEqual([curve.color for curve in curves], [BASELINE_COLOR, GAIN_COLORS[0]])
        tracking = [
            (curves, options)
            for curves, options in calls
            if "unperturbed" in options["title"] and not options["title"].startswith("Nominal")
        ]
        self.assertEqual(len(tracking), 4)
        for curves, _ in tracking:
            self.assertEqual(len(curves), 3)
            modes = {"equilibrium" if "equilibrium_" in curve.label else "intent" for curve in curves}
            self.assertEqual(len(modes), 1)
            self.assertEqual([curve.color for curve in curves], [GAIN_COLORS[0], BASELINE_COLOR, GAIN_COLORS[1]])
            self.assertTrue(all(curve.terminal_time is None for curve in curves))
        self.assertIn("nearly the same motion", page)
        self.assertIn("nominal force term only in the old equilibrium", page)
        self.assertIn("not a stiffness ranking", page)

    def test_paired_displacement_velocity_use_matching_baseline_and_true_terminal(self):
        """Subtract each gain's own baseline and append real q and qd terminals."""
        _, calls = self._capture()
        paired = [(curves, options) for curves, options in calls if "forward" in options["title"]]
        self.assertEqual(len(paired), 4)
        for curves, options in paired:
            velocity = "velocity" in options["title"]
            self.assertIn("[mm/s]" if velocity else "[mm]", options["y_label"])
            for gain, curve in zip((0.5, 1.0, 2.0), curves, strict=True):
                np.testing.assert_allclose(curve.time, [0, 10, 20, 30])
                np.testing.assert_allclose(curve.value, np.array([0, 2, 3, 4] if velocity else [0, 1, 2, 3]) / gain)
                self.assertEqual(curve.terminal_time, 40.0)
                self.assertAlmostEqual(curve.terminal_value, (9.0 if velocity else 7.0) / gain)
                self.assertNotAlmostEqual(curve.terminal_value, curve.value[-1])
                self.assertTrue(curve.qualified)
            self.assertEqual(options["spans"][0]["start"], 10.0)
            self.assertEqual(options["spans"][0]["end"], 30.0)
            self.assertIn("not reference tracking", options["caption"])

    def test_keep_failed_raw_samples_gaps_and_unqualified_terminal(self):
        """Retain failed curves and every finite point without bridging a gap."""
        case = self.record["cases"][1]
        case.update(status="execution_error", pair_valid=False, error={"message": "synthetic failure"})
        self._replace_trace(case, pelvis_x_m=np.array([0.01, 0.012, np.nan, 0.018]))
        page, calls = self._capture()
        selected = next(
            curve
            for curves, options in calls
            if "displacement" in options["title"]
            for curve in curves
            if case["case_id"] in curve.label
        )
        self.assertEqual(len(selected.value), 4)
        self.assertTrue(np.isnan(selected.value[2]))
        self.assertFalse(selected.qualified)
        self.assertIsNotNone(selected.terminal_value)
        self.assertIn('class="terminal-marker unqualified"', page)
        path = next(path for path in re.findall(r'<path class="raw-trace"[^>]+>', page) if case["case_id"] in path)
        self.assertIn('data-sample-count="3"', path)
        commands = re.search(r' d="([^"]+)"', path).group(1)
        self.assertEqual(commands.count("M"), 2)
        self.assertEqual(commands.count("L"), 1)
        report = _write_html(self.directory, self.record).read_text()
        self.assertLess(report.index("synthetic failure"), report.index("<figure"))
        self.assertIn("not a qualified response comparison", report)

    def test_show_saved_physical_failure_reasons_and_general_push_label(self):
        """Expose saved safety failures and avoid assuming every push is forward."""
        case = self.record["cases"][1]
        case.update(status="invalid", pair_valid=False)
        case["metrics"]["evaluation"] = {"safety_reasons": [["rigid_last_ground_intersection"]]}
        case["response_config"].update(push_force_x_n=0.0, push_force_z_n=100.0)
        page = _write_html(self.directory, self.record).read_text()
        self.assertIn("Applied push: displacement and velocity response", page)
        self.assertIn("Both leg and ankle stiffness change together", page)
        self.assertLess(page.index("Saved safety reasons: rigid_last_ground_intersection"), page.index("<figure"))
        self.assertIn('href="cases/response_001.html">Raw case details</a>', page)

    def test_missing_cases_traces_and_terminals_have_explicit_fallbacks(self):
        """Report missing data without substituting a nominal curve or terminal."""
        self.record["cases"] = self.record["cases"][:2]
        baseline, push = self.record["cases"]
        self._replace_trace(push, terminal_body_q=np.array([]), terminal_body_qd=np.array([]))
        page, calls = self._capture()
        self.assertIn("no unperturbed K x 1 cases", page)
        self.assertNotIn("K x 2", page)
        self.assertIn("pre-integration only", page)
        for curves, options in calls:
            if "forward" in options["title"]:
                self.assertIsNone(curves[0].terminal_time)
        (self.directory / baseline["trace_file"]).unlink()
        page = _write_html(self.directory, self.record).read_text()
        self.assertIn("trace unavailable", page)
        self.assertIn("saved reduced-rig simulation traces", page)
        self.assertIn("not a full-state recovery test", page)
        self.assertIn("does not establish natural frequency", page)
        self.assertNotIn("passivity verified", page)
        empty = _write_html(self.directory, {"cases": []}).read_text()
        self.assertIn("no push cases", empty)

    def test_arbitrary_gains_and_mismatched_metadata_do_not_mix(self):
        """Label actual multipliers and keep different damping or identities apart."""
        self.record["cases"] = []
        self._add_pair("intent", 0.75)
        self._add_pair("intent", 1.5)
        page, calls = self._capture()
        self.assertIn("K x 0.75", page)
        self.assertIn("K x 1.5", page)
        self.assertNotIn("K x 0.5", page)
        self.record["cases"][2]["response_config"]["leg_damping_n_s_m"] = 22.0
        self.record["cases"][3]["input_fingerprints"]["material_identity"] = "different"
        page, calls = self._capture()
        tracking = [curves for curves, options in calls if "unperturbed" in options["title"]]
        self.assertEqual(len(tracking), 4)
        self.assertTrue(all(len(curves) == 1 for curves in tracking))
        self.assertIn("baseline metadata unavailable or mismatched", page)
        invalid_pair = next(
            curves[0] for curves, options in calls if "forward" in options["title"] and "1.5" in curves[0].label
        )
        self.assertEqual(invalid_pair.time.size, 0)
        self.assertFalse(invalid_pair.qualified)

    def test_clock_mismatch_and_terminal_decoding_remain_conservative(self):
        """Reject mismatched paired clocks and decode terminal pitch from q only."""
        push = self.record["cases"][1]
        self._replace_trace(push, time_s=np.array([0, 0.01, 0.02, 0.03001]))
        page, _ = self._capture()
        self.assertIn("paired clock or channel mismatch; no interpolation", page)
        with np.load(self.directory / push["trace_file"], allow_pickle=False) as data:
            trace = dict(data)
        time, angle = _response_terminal(trace, "pitch_rad")
        self.assertEqual(time, 0.04)
        self.assertAlmostEqual(angle, 0.0045)
        self.assertIsNone(_response_terminal({"time_s": [0.0], "pitch_rad": [0.2]}, "pitch_rad"))


if __name__ == "__main__":
    unittest.main()
