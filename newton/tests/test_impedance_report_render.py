# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check redraw-only CLI integrity without rerunning any physics."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.__main__ import create_parser, main
from projects.impedance_instron.simple.render_report import render_saved_report


class TestReportRender(unittest.TestCase):
    """Keep saved numeric results separate from new presentation assets."""

    def setUp(self):
        """Build an independent minimal saved response record with file hashes."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        (self.output / "cases").mkdir()
        for name in ("reference.json", "artifact.json"):
            (self.output / name).write_text("{}\n")
        np.savez_compressed(self.output / "cases/quiet.npz", time_s=np.arange(3) * 0.01)

        def sha(name):
            return hashlib.sha256((self.output / name).read_bytes()).hexdigest()

        self.record = {
            "schema_version": "impedance_paired_response_1",
            "status": "complete",
            "cases": [{"case_id": "quiet", "trace_file": "cases/quiet.npz", "trace_sha256": sha("cases/quiet.npz")}],
            "reference_snapshot_sha256": sha("reference.json"),
            "artifact_sha256": sha("artifact.json"),
        }
        self.summary_path = self.output / "summary.json"
        self.summary_path.write_text(json.dumps(self.record))

    def test_redraw_preserves_summary_and_numeric_files(self):
        """Write only presentation files while retaining saved hashes and scores."""
        before = {
            str(path.relative_to(self.output)): path.read_bytes() for path in self.output.rglob("*") if path.is_file()
        }
        with patch(
            "projects.impedance_instron.simple.response._write_html", return_value=self.output / "report.html"
        ) as render:
            result = render_saved_report(self.output)
        self.assertEqual(result, self.output / "report.html")
        render.assert_called_once()
        for name, value in before.items():
            self.assertEqual((self.output / name).read_bytes(), value)
        metadata = json.loads((self.output / "render_metadata.json").read_text())
        self.assertFalse(metadata["physics_rerun"])
        self.assertFalse(metadata["saved_metrics_modified"])
        self.assertFalse(metadata["saved_source_fingerprints_modified"])

    def test_tampered_trace_rejected_before_writing_figures(self):
        """Reject changed saved data before invoking the renderer."""
        (self.output / "cases/quiet.npz").write_bytes(b"not the saved result")
        with patch("projects.impedance_instron.simple.response._write_html") as render:
            with self.assertRaisesRegex(ValueError, "Saved data changed"):
                render_saved_report(self.output)
        render.assert_not_called()
        self.assertFalse((self.output / "render_metadata.json").exists())

    def test_reject_unknown_or_duplicate_json(self):
        """Reject unknown experiment formats and duplicate keys."""
        self.summary_path.write_text('{"schema_version":"unknown"}')
        with self.assertRaisesRegex(ValueError, "Expected"):
            render_saved_report(self.output)
        self.summary_path.write_text('{"schema_version":"a","schema_version":"b"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            render_saved_report(self.output)

    def test_no_physics_update_flag_needed_for_redraw(self):
        """Dispatch report presentation independently of physics evaluation."""
        args = create_parser().parse_args(["report", str(self.output), "--overview-only"])
        self.assertTrue(args.overview_only)
        self.assertFalse(hasattr(args, "allow_physics_update"))
        self.assertFalse(hasattr(args, "device"))
        with (
            patch(
                "projects.impedance_instron.simple.render_report.render_saved_report",
                return_value=self.output / "report.html",
            ) as render,
            patch("builtins.print"),
        ):
            main(["report", str(self.output), "--overview-only"])
        render.assert_called_once_with(self.output, overview_only=True)


if __name__ == "__main__":
    unittest.main()
