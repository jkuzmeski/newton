# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the reusable two-term footwear report figure generator."""

from __future__ import annotations

import builtins
import contextlib
import importlib
import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from projects.digital_shoe.mechanics_report import figures


class TestFootwearMechanicsFigures(unittest.TestCase):
    """Check optional plotting, provenance gates and reusable output paths."""

    def test_help_without_matplotlib(self):
        """Show CLI help without importing optional plotting libraries."""
        original_import = builtins.__import__

        def without_matplotlib(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError("matplotlib deliberately unavailable")
            return original_import(name, *args, **kwargs)

        output = io.StringIO()
        with (
            mock.patch("builtins.__import__", side_effect=without_matplotlib),
            mock.patch.object(sys, "argv", ["figures", "--help"]),
            contextlib.redirect_stdout(output),
        ):
            importlib.reload(figures)
            with self.assertRaises(SystemExit) as raised:
                figures.main()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--output", output.getvalue())
        self.assertIn("--artifact", output.getvalue())

    def test_reject_wrong_hash_before_writing(self):
        """Reject unapproved artifact bytes before creating output directories."""
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact.json"
            artifact.write_text("{}\n")
            output = Path(temporary) / "nested" / "figures"
            with self.assertRaises(ValueError):
                figures.generate_figures(artifact, output)
            self.assertFalse(output.exists())

    def test_generate_external_paths(self):
        """Generate six qualified figure pairs from an external artifact copy."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Figure generation requires optional matplotlib")
        artifact_source = figures.ROOT / "outputs/impedance_instron/baseline12/digital_shoe.json"
        if not artifact_source.is_file():
            self.skipTest("The approved saved footwear artifact is not available")
        with tempfile.TemporaryDirectory(prefix="footwear figures ") as temporary:
            artifact = Path(temporary) / "artifact copy.json"
            artifact.write_bytes(artifact_source.read_bytes())
            output = Path(temporary) / "nested" / "figures"
            metadata = figures.generate_figures(artifact, output)
            saved = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata, saved)
            stems = {
                "bed_geometry",
                "material_equilibrium",
                "material_rate_dependence",
                "validation_rearfoot",
                "validation_fullfoot",
                "contact_bristle",
            }
            self.assertEqual(set(metadata["figures"]), stems)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {f"{stem}.{suffix}" for stem in stems for suffix in ("png", "svg")} | {"metadata.json"},
            )
            self.assertEqual(
                metadata["figure_sha256"],
                {
                    f"{stem}.{suffix}": figures.sha256(output / f"{stem}.{suffix}")
                    for stem in stems
                    for suffix in ("png", "svg")
                },
            )
            self.assertEqual(metadata["artifact_path"], str(artifact.resolve()))
            self.assertEqual(metadata["hyperfoam_term_count"], 2)
            self.assertEqual(metadata["material_parameters"]["effective_poisson_ratio"], 0.0)
            self.assertGreater(metadata["material_parameters"]["instantaneous_shear_modulus_2_pa"], 0.0)
            self.assertIn("projects/digital_shoe/mechanics_report/figures.py", metadata["source_sha256"])
            self.assertIn("projects/digital_shoe/mechanics_report/_provenance.py", metadata["source_sha256"])
            self.assertEqual(
                shlex.split(metadata["command"]),
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "-m",
                    "projects.digital_shoe.mechanics_report.figures",
                    "--artifact",
                    str(artifact.resolve()),
                    "--output",
                    str(output.resolve()),
                ],
            )
            settings = metadata["figures"]["contact_bristle"]["settings_assumed_not_identified"]
            for key, expected in {
                "kt_n_per_m": 10000.0,
                "kv_n_s_per_m": 10.0,
                "mu": 0.8,
                "viscous_ratio": 0.2,
                "release_dwell_s": 0.0005,
            }.items():
                self.assertEqual(settings[key], expected)
            for stem in ("validation_rearfoot", "validation_fullfoot"):
                record = metadata["figures"][stem]
                self.assertFalse(record["prediction_rerun"])
                self.assertTrue(record["metric_recomputation"]["agrees_with_stored_within_tolerance"])
                self.assertEqual(record["samples"], 501)


if __name__ == "__main__":
    unittest.main()
