# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check additive CLI controls for the material and impedance sensitivity suite."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from projects.impedance_instron.__main__ import create_parser, main


class TestSensitivityCLI(unittest.TestCase):
    """Keep the new experiment separate from existing RL and response defaults."""

    def test_default_protocol_and_empty_smoke_sweeps(self):
        """Select independent sweeps and permit explicit nominal-only runs."""
        args = create_parser().parse_args(["sensitivity"])
        self.assertEqual(args.modes, ["intent"])
        self.assertEqual(args.gain_multipliers, [0.5, 2.0])
        self.assertEqual(args.modulus_multipliers, [0.75, 1.25])
        self.assertEqual(args.relaxation_multipliers, [0.5, 2.0])
        self.assertEqual(args.push_phase, 0.25)
        self.assertFalse(args.full_factorial)
        smoke = create_parser().parse_args(
            ["sensitivity", "--gain-multipliers", "--modulus-multipliers", "--relaxation-multipliers"]
        )
        self.assertEqual(smoke.gain_multipliers, [])
        self.assertEqual(smoke.modulus_multipliers, [])
        self.assertEqual(smoke.relaxation_multipliers, [])

    def test_material_paths_and_no_old_contract_change(self):
        """Accept explicit imported materials without changing old workflow arguments."""
        args = create_parser().parse_args(["sensitivity", "--material", "soft.json", "--material", "firm.json"])
        self.assertEqual(args.material, [Path("soft.json"), Path("firm.json")])
        for argv in (["train"], ["response"], ["evaluate", "best.pt"]):
            old = create_parser().parse_args(argv)
            self.assertFalse(hasattr(old, "gain_multipliers"))
            self.assertFalse(hasattr(old, "modulus_multipliers"))
            self.assertFalse(hasattr(old, "minimum_recovery_window_s"))

    def test_dispatch_recovery_settings_and_frozen_paths(self):
        """Pass declared recovery bounds and material inputs through CLI dispatch."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            argv = [
                "sensitivity",
                "--reference",
                "reference.json",
                "--artifact",
                "shoe.json",
                "--output",
                str(output),
                "--device",
                "cpu",
                "--material",
                "variant.json",
                "--full-factorial",
                "--recovery-dwell-s",
                "0.06",
            ]
            with (
                patch(
                    "projects.impedance_instron.simple.sensitivity.run_sensitivity_suite",
                    return_value=output / "report.html",
                ) as run,
                patch("builtins.print") as write,
            ):
                main(argv)
            args, kwargs = run.call_args
            self.assertEqual(args, (Path("reference.json"), Path("shoe.json"), output))
            self.assertEqual(kwargs["material_paths"], [Path("variant.json")])
            self.assertTrue(kwargs["full_factorial"])
            self.assertEqual(kwargs["recovery_config"].dwell_s, 0.06)
            self.assertEqual(kwargs["recovery_config"].velocity_tolerance_m_s, 0.01)
            self.assertEqual(kwargs["command"], ["projects.impedance_instron", *argv])
            write.assert_called_once_with(f"Report: {output / 'report.html'}")


if __name__ == "__main__":
    unittest.main()
