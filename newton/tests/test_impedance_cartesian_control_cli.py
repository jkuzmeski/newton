# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the fixed twelve-point Cartesian GPU command and its evidence guard."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from projects.impedance_instron.cartesian.gpu.__main__ import _validation, build_parser


class TestCartesianGpuCli(unittest.TestCase):
    """Protect the one supported controller and world layout."""

    def test_parser_uses_fixed_search_defaults(self):
        """Expose the selected 12-point, 128-world search defaults only."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "baseline",
                "--output",
                "fit",
                "--single-validation",
                "single.json",
                "--batch-validation",
                "batch.json",
            ]
        )
        self.assertEqual(args.iterations, 200)
        self.assertEqual(args.plateau_patience, 20)
        self.assertEqual(args.plateau_rtol, 1.0e-4)
        self.assertFalse(hasattr(args, "controls"))
        self.assertFalse(hasattr(args, "worlds"))
        self.assertFalse(hasattr(args, "islands"))

    def test_parser_rejects_retired_layout_options(self):
        """Reject control-count, world-count, and island compatibility flags."""
        parser = build_parser()
        base = [
            "baseline",
            "--output",
            "fit",
            "--single-validation",
            "single.json",
            "--batch-validation",
            "batch.json",
        ]
        for option in ("--controls", "--worlds", "--islands"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parser.parse_args([*base, option, "2"])

    def test_validation_rejects_non_twelve_control_request(self):
        """Fail before reading evidence when a caller requests another control count."""
        with self.assertRaisesRegex(ValueError, "exactly 12 controls"):
            _validation(Path("missing.json"), Path("missing"), mixed=False, expected_controls=9)

    def test_validation_fails_closed_on_changed_sources(self):
        """Reject otherwise passing evidence after any runtime source identity changes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name in ("reference.npz", "profile.json", "equilibrium.npz", "trace.npz", "summary.json"):
                data = name.encode()
                (root / name).write_bytes(data)
                artifacts[name] = hashlib.sha256(data).hexdigest()
            report = {
                "same_timestep_parity_passed": True,
                "execution_identity": {"runtime": "test"},
                "control_count": 12,
                "sources": {"physics.py": "old"},
                "frozen_artifacts_sha256": artifacts,
            }
            evidence = root / "single.json"
            evidence.write_text(json.dumps(report))
            with (
                patch(
                    "projects.impedance_instron.cartesian.gpu.__main__.execution_identity",
                    return_value={"runtime": "test"},
                ),
                patch(
                    "projects.impedance_instron.cartesian.gpu.__main__.source_snapshot",
                    return_value={"physics.py": "new"},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "runtime changed"):
                    _validation(evidence, root, mixed=False)


if __name__ == "__main__":
    unittest.main()
