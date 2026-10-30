# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the command surface for the Hogan runner pipeline."""

import contextlib
import io
import sys
import unittest
from types import ModuleType

from newton.tests.unittest_utils import patch_sys_module
from projects.impedance_instron import __main__ as pipeline


class TestHoganCommands(unittest.TestCase):
    def test_forward_commands_without_changing_options(self):
        """Forward each command to its parser without altering its options."""
        for command, (name, _) in pipeline._COMMANDS.items():
            with self.subTest(command=command):
                module = ModuleType(name)
                seen = []
                module.main = lambda seen=seen: seen.append(sys.argv[1:])
                arguments = [command, "--output", "new-run"]
                original = sys.argv
                with patch_sys_module(name, module):
                    pipeline.main(arguments)
                prefix = [command] if command in pipeline._IDENTIFY_COMMANDS else []
                self.assertEqual(seen, [[*prefix, "--output", "new-run"]])
                self.assertEqual(arguments, [command, "--output", "new-run"])
                self.assertIs(sys.argv, original)

    def test_help_lists_active_pipeline_commands(self):
        """List the preparation, fitting, evaluation, and generation commands."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            pipeline.main([])
        for command in pipeline._COMMANDS:
            self.assertIn(command, stream.getvalue())

    def test_reject_unknown_commands(self):
        """Reject commands that are not in the active router."""
        for command in ("learn", "control", "quick-fit", "--output"):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    pipeline.main([command])


if __name__ == "__main__":
    unittest.main(verbosity=2)
