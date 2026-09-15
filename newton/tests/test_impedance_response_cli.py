# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep response experiments separate from frozen policy evaluation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from projects.impedance_instron.__main__ import create_parser, main


class TestResponseCLI(unittest.TestCase):
    """Check opt-in controls and dispatch without advancing physics."""

    def test_response_defaults(self):
        """Select paired disturbances without adding controls to RL training."""
        args = create_parser().parse_args(["response"])
        self.assertEqual(args.stiffness_multipliers, [0.5, 1.0, 2.0])
        self.assertEqual(args.push_force_x_n, 150.0)
        self.assertEqual(args.push_duration_s, 0.04)
        self.assertEqual(args.ground_offset_m, 0.005)
        self.assertEqual(args.output, Path("outputs/impedance_instron/simple/response"))
        for command in (["train"], ["evaluate", "best.pt"]):
            old = create_parser().parse_args(command)
            self.assertFalse(hasattr(old, "stiffness_multipliers"))
            self.assertFalse(hasattr(old, "push_force_x_n"))
            self.assertFalse(hasattr(old, "ground_offset_m"))

    def test_response_dispatch(self):
        """Pass explicit experiment settings through the runnable CLI."""
        reference = object()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "response"
            with (
                patch("projects.impedance_instron.simple.reference.Reference.load", return_value=reference) as load,
                patch(
                    "projects.impedance_instron.simple.response.run_response_suite", return_value=output / "report.html"
                ) as run,
                patch("builtins.print") as write,
            ):
                arguments = [
                    "response",
                    "--reference",
                    "frozen.json",
                    "--artifact",
                    "shoe.json",
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                    "--stiffness-multipliers",
                    "1",
                    "--push-force-x-n",
                    "-100",
                    "--push-duration-s",
                    "0.03",
                    "--ground-offset-m",
                    "0.003",
                ]
                main(arguments)
            load.assert_called_once_with(Path("frozen.json"))
            run.assert_called_once_with(
                reference,
                Path("shoe.json"),
                output,
                device="cpu",
                stiffness_multipliers=[1.0],
                push_force_x_n=-100.0,
                push_duration_s=0.03,
                ground_offset_m=0.003,
                command=["projects.impedance_instron", *arguments],
            )
            write.assert_called_once_with(f"Report: {output / 'report.html'}")


if __name__ == "__main__":
    unittest.main()
