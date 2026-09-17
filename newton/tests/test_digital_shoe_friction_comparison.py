# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check saved-run reporting without restricted experimental data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.digital_shoe.friction_comparison import build_comparison
from projects.digital_shoe.friction_report import write_friction_report


class TestFrictionComparison(unittest.TestCase):
    """Require trace support and matching input provenance in offline reports."""

    def setUp(self):
        """Create independent source and candidate clocks for a synthetic stance."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.baseline.mkdir()
        self.candidate.mkdir()
        self.t = np.linspace(0, 1, 9)
        force = np.column_stack((100 * np.sin(2 * np.pi * self.t), np.full(9, 100.0)))
        self.reference = {"grf_time_s": self.t, "grf_target_n": force}
        np.savez(self.baseline / "trace.npz", time_s=self.t, grf_n=force)
        (self.baseline / "summary.json").write_text("{}")
        t = np.linspace(0, 1, 17)
        force = np.column_stack((100 * np.sin(2 * np.pi * t), np.full(17, 100.0)))
        np.savez(self.candidate / "trace.npz", time_s=t, grf_n=force)
        (self.candidate / "run.json").write_text("{}")
        self.hashes = {"reference.npz": "reference-digest", "equilibrium.npz": "frozen-controller"}
        (self.candidate / "report.json").write_text(json.dumps({"input_hashes": self.hashes}))
        self.verify = self.enterContext(
            patch("projects.digital_shoe.friction_comparison.verify_baseline_inputs", return_value=self.hashes)
        )
        self.enterContext(patch("projects.digital_shoe.friction_comparison.load", return_value=self.reference))

    def test_independent_clocks_and_safe_labels(self):
        """Render each force on its own clock and escape labels in tables and SVG."""
        label = "candidate<script>"
        output = self.root / "output"
        report = build_comparison(self.baseline, {label: self.candidate}, output, score_policy="legacy")
        self.assertTrue(report["scores"][label]["complete"])
        rendered = (output / "report.html").read_text()
        self.assertIn("candidate&lt;script&gt;", rendered)
        self.assertNotIn("candidate<script>", rendered)
        self.assertIn("separate audit", rendered)
        json.loads((output / "report.json").read_text(), parse_constant=self.fail)

    def test_renderer_rejects_nonfinite_curves(self):
        """Reject invalid SVG coordinates instead of concealing missing force support."""
        force = self.reference["grf_target_n"].copy()
        force[-1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite forces"):
            write_friction_report(
                {"forward_sign": 1}, self.t, {"invalid": force}, self.reference, self.root / "invalid.html"
            )
        self.assertFalse((self.root / "invalid.html").exists())

    def test_raw_zoom_uses_stored_samples(self):
        """Render a transition zoom without filtering or inventing force samples."""
        force = self.reference["grf_target_n"].copy()
        report = {
            "forward_sign": 1,
            "scores": {"raw": {}},
            "steps_evaluated": len(self.t),
            "total_source_steps": len(self.t),
            "is_partial_smoke": False,
            "zoom_interval_s": [0.25, 0.75],
        }
        output = self.root / "zoom.html"
        write_friction_report(report, self.t, {"raw": force}, self.reference, output)
        rendered = output.read_text()
        self.assertIn("Contact-transition detail (stored samples, no smoothing)", rendered)
        np.testing.assert_array_equal(force, self.reference["grf_target_n"])

    def test_input_mismatch_and_output_overwrite(self):
        """Reject mixed baseline runs and existing outputs before writing reports."""
        (self.candidate / "report.json").write_text(json.dumps({"input_hashes": {}}))
        output = self.root / "output"
        with self.assertRaises(ValueError):
            build_comparison(self.baseline, {"candidate": self.candidate}, output, score_policy="legacy")
        self.assertFalse(output.exists())
        output.mkdir()
        with self.assertRaises(FileExistsError):
            build_comparison(self.baseline, {}, output, score_policy="legacy")

    def test_incomplete_run_is_not_plotted_as_complete(self):
        """Retain incomplete status without displaying an unsupported full-stance curve."""
        np.savez(self.candidate / "trace.npz", time_s=self.t[:4], grf_n=self.reference["grf_target_n"][:4])
        output = self.root / "output"
        report = build_comparison(self.baseline, {"incomplete": self.candidate}, output, score_policy="legacy")
        self.assertFalse(report["scores"]["incomplete"]["complete"])
        self.assertNotIn("<th>incomplete</th>", (output / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
