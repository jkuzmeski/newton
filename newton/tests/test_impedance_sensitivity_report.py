# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check offline sensitivity reports without running the physics engine."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.impedance_instron.simple.sensitivity_report import write_sensitivity_report


class TestSensitivityReport(unittest.TestCase):
    """Keep physical validity, recovery and permanent material effects separate."""

    def setUp(self):
        """Create numeric case files and a minimal independent manifest."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        (self.output / "cases").mkdir()
        channels = (
            "pelvis_x_m",
            "pelvis_z_m",
            "pitch_rad",
            "foot_x_m",
            "foot_z_m",
            "leg_length_m",
            "pelvis_vx_m_s",
            "pelvis_vz_m_s",
            "pitch_rate_rad_s",
            "foot_vx_m_s",
            "foot_vz_m_s",
            "leg_rate_m_s",
        )
        trace = {name: np.linspace(0, 0.002, 5) for name in channels}
        trace["time_s"] = np.arange(5) * 0.001
        trace["leg_nominal_force_n"] = np.full(5, 100.0)
        trace["leg_feedback_force_n"] = np.full(5, 5.0)
        trace["leg_force_n"] = np.full(5, 105.0)
        np.savez_compressed(self.output / "cases/case.npz", **trace)
        np.savez_compressed(
            self.output / "cases/base.npz",
            **{name: np.zeros_like(value) if name != "time_s" else value for name, value in trace.items()},
        )
        deviations = {
            name: {"peak_abs_including_terminal_deviation": 0.003, "final_deviation": 0.003, "tolerance": 0.001}
            for name in channels
        }
        self.case = {
            "case_id": "push_forward",
            "controller_id": "nominal",
            "material_id": "baseline",
            "comparison_type": "push_recovery",
            "direction": "forward",
            "status": "valid",
            "pair_valid": True,
            "trace_file": "cases/case.npz",
            "baseline_trace_file": "cases/base.npz",
            "recovery": {
                "status": "not_returned_within_window",
                "deviations": deviations,
                "window": {"available_post_pulse_s": 0.2},
            },
        }
        self.record = {
            "status": "complete",
            "suite_config": {"full_factorial": False},
            "cases": [self.case],
            "controllers": [{"id": "nominal"}],
            "materials": [{"id": "baseline"}],
        }

    def test_small_index_and_raw_case_page(self):
        """Link raw plots from a compact index and include true terminal rates."""
        path = write_sensitivity_report(self.output, self.record)
        index = path.read_text()
        page = (self.output / "pages/case_0000.html").read_text()
        self.assertIn("<svg", index)
        self.assertLess(index.index("<svg"), index.index("<table"))
        self.assertIn("How to read this report", index)
        self.assertIn("1. What did the shoe material change?", index)
        self.assertIn("2. What did the controller change?", index)
        self.assertIn("3. Why is safety not the same as recovery?", index)
        self.assertIn('href="pages/case_0000.html"', index)
        self.assertIn("Final upper VX [mm/s]", index)
        self.assertIn("Upper-body forward velocity deviation", page)
        self.assertIn("True final deviation", page)
        self.assertIn("Declared engineering band", page)
        self.assertIn("Leg nominal, feedback and delivered force", page)
        self.assertNotIn("<script", page)
        self.assertNotIn("https://", page)
        self.assertIn("Non-nominal controller", index)

    def test_invalid_pair_stays_visible(self):
        """Warn when a valid case lacks a qualified matching baseline."""
        self.case["pair_valid"] = False
        path = write_sensitivity_report(self.output, self.record)
        self.assertIn("INVALID PAIR", path.read_text())
        self.assertIn("not a qualified response", (self.output / "pages/case_0000.html").read_text())

    def test_material_change_is_not_transient_recovery(self):
        """Label permanent material differences and synthetic qualification limits."""
        self.case["comparison_type"] = "material_sensitivity"
        self.case["recovery"]["status"] = "persistent_material_change"
        page = write_sensitivity_report(self.output, self.record).read_text()
        self.assertIn("persistent_material_change", page)
        self.assertIn("not newly calibrated or validated shoes", page)
        self.assertIn("OWN material and controller", page)

    def test_escape_labels_and_reject_external_trace(self):
        """Escape imported labels and prevent report reads outside the suite."""
        self.case["case_id"] = "<script>bad()</script>"
        index = write_sensitivity_report(self.output, self.record).read_text()
        self.assertIn("&lt;script&gt;", index)
        self.assertNotIn("<script>", index)
        self.case["trace_file"] = "../elsewhere.npz"
        with self.assertRaisesRegex(ValueError, "inside"):
            write_sensitivity_report(self.output, self.record)

    def test_failure_outside_overview_stays_visible(self):
        """List failures from every saved mode before the main figures."""
        failed = copy.deepcopy(self.case)
        failed.update(case_id="other_mode_failure", controller_mode="equilibrium", status="invalid", pair_valid=False)
        failed["metrics"] = {"evaluation": {"safety_reasons": [["rigid_last_ground_intersection"]]}}
        self.case["controller_mode"] = "intent"
        self.record["cases"].append(failed)
        page = write_sensitivity_report(self.output, self.record).read_text()
        self.assertLess(page.index("other_mode_failure"), page.index("1. What did the shoe material change?"))
        self.assertIn('href="pages/case_0001.html"', page)
        self.assertIn("rigid_last_ground_intersection", page)
        self.assertIn("NOT successful recovery", page)

    def test_execution_error_without_trace_remains_reportable(self):
        """Retain failed constructions without fabricating traces or metrics."""
        self.case.update(
            status="execution_error", pair_valid=False, trace_file=None, baseline_trace_file=None, recovery={}
        )
        self.assertIn("execution_error", write_sensitivity_report(self.output, self.record).read_text())


if __name__ == "__main__":
    unittest.main()
