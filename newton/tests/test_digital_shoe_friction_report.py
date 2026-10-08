# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check standalone friction reporting without restricted experimental data."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.digital_shoe.friction_report import write_friction_report


class TestFrictionReport(unittest.TestCase):
    """Require valid curve support and safe labels in offline reports."""

    def setUp(self):
        """Create independent source and candidate clocks for a synthetic stance."""
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.t = np.linspace(0, 1, 9)
        force = np.column_stack((100 * np.sin(2 * np.pi * self.t), np.full(9, 100.0)))
        self.reference = {"grf_time_s": self.t, "grf_target_n": force}

    def test_independent_clocks_and_safe_labels(self):
        """Render each force on its own clock and escape labels in tables and SVG."""
        label = "candidate<script>"
        clock = np.linspace(0, 1, 17)
        force = np.column_stack((100 * np.sin(2 * np.pi * clock), np.full(17, 100.0)))
        report = {
            "forward_sign": 1,
            "scores": {label: {}},
            "steps_evaluated": len(clock),
            "total_source_steps": len(clock),
            "is_partial_smoke": False,
        }
        output = self.root / "report.html"
        write_friction_report(report, self.t, {label: force}, self.reference, output, force_times={label: clock})
        rendered = output.read_text()
        self.assertIn("candidate&lt;script&gt;", rendered)
        self.assertNotIn("candidate<script>", rendered)

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


if __name__ == "__main__":
    unittest.main()
