# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the fixed two-term report provenance and offline rendering contract."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from projects.digital_shoe.mechanics_report import _provenance, render


class TestFootwearMechanicsReport(unittest.TestCase):
    """Check reporting without requiring restricted footwear assets."""

    def setUp(self):
        """Create a synthetic two-term provenance fixture and six local figures."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifact = self.root / "shoe.json"
        self.artifact.write_text(
            json.dumps(
                {
                    "constitutive_model": {
                        "parameters": {
                            "instantaneous_shear_modulus_pa": 100.0,
                            "instantaneous_shear_modulus_2_pa": 20.0,
                            "effective_poisson_ratio": 0.0,
                        }
                    }
                }
            )
        )
        self.physics = self.root / "physics.py"
        self.physics.write_text("# Synthetic source identity fixture.\n")
        self.manifest = self.root / "sources.json"
        self._manifest()
        self.source = self.root / "REPORT.md"
        images = "\n\n".join(f"![{name}](figures/{name}.svg)" for name in render.FIGURE_NAMES)
        self.source.write_text("# Report\n\n## Methods\n\n" + images)
        self.output = self.root / "result" / "report.html"
        (self.output.parent / "figures").mkdir(parents=True)
        for name in render.FIGURE_NAMES:
            (self.output.parent / "figures" / f"{name}.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"><title>Test figure</title></svg>'
            )
        self.metadata = self.output.parent / "figures" / "metadata.json"
        self.metadata.write_text(
            json.dumps(
                {
                    "artifact_sha256": hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
                    "source_sha256": {"physics.py": hashlib.sha256(self.physics.read_bytes()).hexdigest()},
                    "figure_sha256": {
                        f"{name}.svg": hashlib.sha256(
                            (self.output.parent / "figures" / f"{name}.svg").read_bytes()
                        ).hexdigest()
                        for name in render.FIGURE_NAMES
                    },
                }
            )
        )

    def _manifest(self):
        """Pin the current synthetic artifact and physics source bytes."""
        self.manifest.write_text(
            json.dumps(
                {
                    "artifact_sha256": hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
                    "files": {"physics.py": hashlib.sha256(self.physics.read_bytes()).hexdigest()},
                }
            )
        )

    def test_verified_artifact(self):
        """Accept unchanged source and both positive material terms."""
        value = _provenance.load_verified_artifact(self.artifact, manifest_path=self.manifest, root=self.root)
        self.assertEqual(value["constitutive_model"]["parameters"]["instantaneous_shear_modulus_2_pa"], 20.0)

    def test_changed_artifact(self):
        """Reject altered artifact bytes before report output is written."""
        self.artifact.write_text(self.artifact.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Artifact changed"):
            _provenance.load_verified_artifact(self.artifact, manifest_path=self.manifest, root=self.root)

    def test_changed_source(self):
        """Reject source drift without silently updating the frozen audit."""
        self.physics.write_text("# Changed physics.\n")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            _provenance.load_verified_artifact(self.artifact, manifest_path=self.manifest, root=self.root)

    def test_require_two_terms(self):
        """Reject a disabled second term even with a matching artifact hash."""
        value = json.loads(self.artifact.read_text())
        value["constitutive_model"]["parameters"]["instantaneous_shear_modulus_2_pa"] = 0
        self.artifact.write_text(json.dumps(value))
        self._manifest()
        with self.assertRaisesRegex(ValueError, "two-term"):
            _provenance.load_verified_artifact(self.artifact, manifest_path=self.manifest, root=self.root)

    def test_require_zero_poisson(self):
        """Reject a material outside the report's declared Poisson assumption."""
        value = json.loads(self.artifact.read_text())
        value["constitutive_model"]["parameters"]["effective_poisson_ratio"] = 0.3
        self.artifact.write_text(json.dumps(value))
        self._manifest()
        with self.assertRaisesRegex(ValueError, "zero-Poisson"):
            _provenance.load_verified_artifact(self.artifact, manifest_path=self.manifest, root=self.root)

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Optional Markdown renderer is unavailable")
    def test_offline_report(self):
        """Embed all six local SVGs and generate offline section navigation."""
        render.render_report(
            self.artifact, self.output, source_path=self.source, manifest_path=self.manifest, root=self.root
        )
        text = self.output.read_text()
        self.assertEqual(text.count('src="data:image/svg+xml;base64,'), 6)
        self.assertIn('href="#section-1"', text)
        self.assertNotIn("<script", text)
        self.assertNotIn('src="https:', text)

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Optional Markdown renderer is unavailable")
    def test_missing_figure(self):
        """Fail without emitting a misleading partially illustrated report."""
        (self.output.parent / "figures" / "contact_bristle.svg").unlink()
        with self.assertRaises(FileNotFoundError):
            render.render_report(
                self.artifact, self.output, source_path=self.source, manifest_path=self.manifest, root=self.root
            )
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Optional Markdown renderer is unavailable")
    def test_changed_figure(self):
        """Reject a modified figure rather than mixing unaudited plot content."""
        (self.output.parent / "figures" / "contact_bristle.svg").write_text("<svg/>")
        with self.assertRaisesRegex(ValueError, "Figure changed"):
            render.render_report(
                self.artifact, self.output, source_path=self.source, manifest_path=self.manifest, root=self.root
            )
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Optional Markdown renderer is unavailable")
    def test_foreign_figure_metadata(self):
        """Reject figures made from another artifact despite valid image files."""
        value = json.loads(self.metadata.read_text())
        value["artifact_sha256"] = "0" * 64
        self.metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Figure artifact"):
            render.render_report(
                self.artifact, self.output, source_path=self.source, manifest_path=self.manifest, root=self.root
            )
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Optional Markdown renderer is unavailable")
    def test_changed_figure_source(self):
        """Reject a figure record that refers to different generator bytes."""
        value = json.loads(self.metadata.read_text())
        value["source_sha256"]["physics.py"] = "0" * 64
        self.metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Figure source changed"):
            render.render_report(
                self.artifact, self.output, source_path=self.source, manifest_path=self.manifest, root=self.root
            )
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Optional Markdown renderer is unavailable")
    def test_preserve_input_files(self):
        """Reject an output path that would overwrite the audited artifact."""
        before = self.artifact.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            render.render_report(
                self.artifact, self.artifact, source_path=self.source, manifest_path=self.manifest, root=self.root
            )
        self.assertEqual(self.artifact.read_bytes(), before)

    @unittest.skipUnless(importlib.util.find_spec("markdown_it"), "Optional Markdown renderer is unavailable")
    def test_unresolved_draft(self):
        """Reject unfinished narrative placeholders before publishing HTML."""
        self.source.write_text(self.source.read_text() + "\n{{TODO}}\n")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            render.render_report(
                self.artifact, self.output, source_path=self.source, manifest_path=self.manifest, root=self.root
            )
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
