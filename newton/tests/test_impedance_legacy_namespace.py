# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve old impedance modules while separating the retired implementation."""

from __future__ import annotations

import base64
import importlib
import json
import pickle
import re
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = "projects.impedance_instron"
_MODULES = (
    "cmaes",
    "control",
    "objective",
    "example",
    "optimize",
    "env",
    "train",
    "report",
    "explain",
    "dashboard",
    "summary",
)
_COMMANDS = ("example", "env", "optimize", "train", "dashboard", "explain", "summary")

# Capture these protocol-4 globals and state on e5e3336a, before moving the modules.
_HISTORICAL_PICKLES = {
    "tolerances_instance": "gASViQAAAAAAAACMJHByb2plY3RzLmltcGVkYW5jZV9pbnN0cm9uLm9iamVjdGl2ZZSMClRvbGVyYW5jZXOUk5QpgZR9lCiMCmR1cmF0aW9uX3OURz95mZmZmZmajBBpbXB1bHNlX2ZyYWN0aW9ulEc/n752yLQ5WIwMbW9tZW50dW1fbV9zlEc/pocrAgxJunViLg==",
    "leg_profile_type": "gASVPQAAAAAAAACMInByb2plY3RzLmltcGVkYW5jZV9pbnN0cm9uLmNvbnRyb2yUjBJMZWdDb21tYW5kLlByb2ZpbGWUk5Qu",
    "ankle_profile_type": "gASVPwAAAAAAAACMInByb2plY3RzLmltcGVkYW5jZV9pbnN0cm9uLmNvbnRyb2yUjBRBbmtsZUNvbW1hbmQuUHJvZmlsZZSTlC4=",
}


class TestLegacyNamespace(unittest.TestCase):
    """Exercise imports and commands through both supported module paths."""

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run an isolated interpreter in the current project's environment."""
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_alias_identity_in_both_import_orders(self):
        """Resolve old imports, new imports and package attributes to one object."""
        script = textwrap.dedent(
            """
            import importlib
            import sys

            name, first = sys.argv[1:]
            parent = importlib.import_module("projects.impedance_instron")
            old_name = f"projects.impedance_instron.{name}"
            new_name = f"projects.impedance_instron.legacy.{name}"
            order = (old_name, new_name) if first == "old" else (new_name, old_name)
            one, two = (importlib.import_module(path) for path in order)
            assert one is two
            assert sys.modules[old_name] is sys.modules[new_name] is one
            assert getattr(parent, name) is one
            imported = __import__("projects.impedance_instron", fromlist=[name])
            assert getattr(imported, name) is one
            assert one.__name__ == new_name
            from pathlib import Path
            assert Path(one.__file__).parts[-2:] == ("legacy", f"{name}.py")
            """
        )
        for name in _MODULES:
            for first in ("old", "new"):
                with self.subTest(module=name, first=first):
                    result = self._run("-c", script, name, first)
                    self.assertNotIn("FutureWarning", result.stderr)

    def test_private_symbols_are_shared(self):
        """Keep explicit private helper imports and Warp objects on the old path."""
        helpers = {
            "example": ("_apply_leg", "_curve", "_prescribe_axes"),
            "env": ("_apply_leg_and_record", "CONTACT_FORCE_FRACTION"),
            "report": ("_AUDIT_FIELDS", "_REQUIRED", "_audit"),
            "train": ("_require_torch", "_assert_synchronized"),
        }
        for name, attributes in helpers.items():
            old = importlib.import_module(f"{_PACKAGE}.{name}")
            new = importlib.import_module(f"{_PACKAGE}.legacy.{name}")
            for attribute in attributes:
                with self.subTest(module=name, attribute=attribute):
                    imported = __import__(f"{_PACKAGE}.{name}", fromlist=[attribute])
                    self.assertIs(getattr(imported, attribute), getattr(new, attribute))
                    self.assertIs(getattr(old, attribute), getattr(new, attribute))

    def test_old_path_patches_implementation_globals(self):
        """Apply patches through the historical namespace to real function globals."""
        old = importlib.import_module(f"{_PACKAGE}.cmaes")
        new = importlib.import_module(f"{_PACKAGE}.legacy.cmaes")
        with patch(f"{_PACKAGE}.cmaes.CMAES", side_effect=RuntimeError("old-path patch")) as patched:
            with self.assertRaisesRegex(RuntimeError, "old-path patch"):
                new.minimize(lambda x: 0.0, [1.0], 0.1)
            patched.assert_called_once()
        self.assertIs(old.CMAES, new.CMAES)

    def test_historical_pickle_globals_and_state(self):
        """Restore actual pre-relocation pickle payloads through compatibility imports."""
        control = importlib.import_module(f"{_PACKAGE}.legacy.control")
        objective = importlib.import_module(f"{_PACKAGE}.legacy.objective")
        restored = {name: pickle.loads(base64.b64decode(data)) for name, data in _HISTORICAL_PICKLES.items()}
        self.assertIs(restored["leg_profile_type"], control.LegCommand.Profile)
        self.assertIs(restored["ankle_profile_type"], control.AnkleCommand.Profile)
        self.assertIs(type(restored["tolerances_instance"]), objective.Tolerances)
        self.assertEqual(restored["tolerances_instance"], objective.Tolerances())
        self.assertEqual(pickle.loads(pickle.dumps(restored["tolerances_instance"])), objective.Tolerances())
        self.assertIs(pickle.loads(pickle.dumps(control.LegCommand.Profile)), control.LegCommand.Profile)

    def test_cli_help_parity_in_fresh_processes(self):
        """Keep historical command options and deprecation timing on both paths."""
        for name in _COMMANDS:
            with self.subTest(module=name):
                old = self._run("-m", f"{_PACKAGE}.{name}", "--help")
                new = self._run("-m", f"{_PACKAGE}.legacy.{name}", "--help")
                self.assertIn("usage:", old.stdout)
                self.assertEqual(old.stdout, new.stdout)
                old_warnings = re.findall(r"FutureWarning: (.*)", old.stderr)
                new_warnings = re.findall(r"FutureWarning: (.*)", new.stderr)
                self.assertEqual(old_warnings, new_warnings)
                self.assertEqual(len(old_warnings), int(name in ("example", "optimize", "train")))
                self.assertNotIn("RuntimeWarning", old.stderr + new.stderr)

    def test_active_import_closure_excludes_legacy(self):
        """Keep the active workflow and preparation helpers outside the retired stack."""
        script = textwrap.dedent(
            """
            import importlib
            import importlib.abc
            import json
            import sys

            forbidden = {f"projects.impedance_instron.{name}" for name in json.loads(sys.argv[1])}

            class RejectLegacy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in forbidden or fullname.startswith("projects.impedance_instron.legacy"):
                        raise AssertionError(f"active workflow imported {fullname}")
                    return None

            sys.meta_path.insert(0, RejectLegacy())
            cli = importlib.import_module("projects.impedance_instron.__main__")
            cli.create_parser().parse_args(["prepare"])
            assert "warp" not in sys.modules
            assert "torch" not in sys.modules
            for name in ("simple.reference", "simple.rig", "simple.policy", "simple.report",
                         "simple.example", "profile", "orientation", "trajectory", "variability"):
                importlib.import_module(f"projects.impedance_instron.{name}")
            assert not forbidden.intersection(sys.modules)
            assert not any(name.startswith("projects.impedance_instron.legacy") for name in sys.modules)
            """
        )
        self._run("-c", script, json.dumps(_MODULES))


if __name__ == "__main__":
    unittest.main()
