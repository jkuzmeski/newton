# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep frozen policy viewers usable after their input directory is relocated."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from projects.impedance_instron.simple.example import Example, create_parser


class TestViewerRelocation(unittest.TestCase):
    """Pass explicit material locations through the existing strict restore API."""

    def test_parse_explicit_checkpoint_material_location(self):
        """Accept a material path without changing default checkpoint behavior."""
        default = create_parser().parse_args(["--viewer", "null"])
        self.assertIsNone(default.material)
        moved = create_parser().parse_args(
            ["--viewer", "null", "--checkpoint", "best.pt", "--material", "moved/shoe.json"]
        )
        self.assertEqual(moved.material, Path("moved/shoe.json"))
        self.assertFalse(moved.allow_physics_update)

    def test_forward_material_to_strict_restore(self):
        """Forward the override and source consent without rewriting checkpoint data."""
        args = SimpleNamespace(
            checkpoint=Path("best.pt"), material=Path("moved/shoe.json"), device="cpu", allow_physics_update=True
        )
        with patch(
            "projects.impedance_instron.simple.example.restore", side_effect=RuntimeError("stop after restore")
        ) as restore:
            with self.assertRaisesRegex(RuntimeError, "stop after restore"):
                Example(None, args)
        restore.assert_called_once_with(
            args.checkpoint, artifact_path=args.material, device="cpu", allow_physics_update=True
        )

    def test_material_override_requires_checkpoint(self):
        """Reject checkpoint-only material overrides on a baseline run."""
        args = SimpleNamespace(
            checkpoint=None,
            material=Path("moved/shoe.json"),
            reference=Path("reference.json"),
            artifact=Path("shoe.json"),
            device="cpu",
        )
        with patch(
            "projects.impedance_instron.simple.example.Reference.load",
            side_effect=AssertionError("must not read reference"),
        ) as load:
            with self.assertRaisesRegex(ValueError, "requires --checkpoint"):
                Example(None, args)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
