# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test lazy project APIs and additive Shoe/Instron command routing."""

import contextlib
import importlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
_EXPORTS = {
    "CalibrationWorkspace": "calibration",
    "ColumnBed": "artifact",
    "DigitalShoe": "artifact",
    "FoundationConfig": "runtime",
    "InstronFixture": "artifact",
    "MidsoleFoundation": "runtime",
    "ShoeMaterial": "runtime",
    "SurroundConfig": "runtime",
    "VisualMesh": "artifact",
    "hyperfoam_pressure_numpy": "material",
    "load_artifact": "artifact",
    "maxwell_coefficients_numpy": "material",
    "maxwell_step_numpy": "material",
    "physics_source_identity": "provenance",
}
_COMMANDS = {
    "projects.digital_shoe": {
        "view": "showcase",
        "report": "report",
        "record": "record_gifs",
        "check-acquisition": "acquisition",
    },
    "projects.digital_instron_v2": {
        "fit": "workflow",
        "validate": "phase1",
        "replay": "phase2",
        "export": "export_digital_shoe",
        "view": "example",
        "profile": "profile_calibration",
    },
}


def _fresh_python(source: str, *args: str) -> subprocess.CompletedProcess:
    """Run an isolated probe through the current project interpreter."""
    return subprocess.run(
        [sys.executable, "-c", source, *args], cwd=_ROOT, capture_output=True, text=True, timeout=60, check=False
    )


class TestShoeProjectImports(unittest.TestCase):
    """Keep package discovery independent of the mechanics implementation."""

    def test_lightweight_imports_do_not_load_mechanics(self):
        """Import discovery and metadata modules without NumPy, Warp, or Newton."""
        source = (
            "import importlib,json,sys; importlib.import_module(sys.argv[1]); "
            "print(json.dumps([n for n in sys.modules if n.split('.')[0] "
            "in ('numpy','warp','newton','scipy','torch')]))"
        )
        for module in (
            "projects.digital_shoe",
            "projects.digital_shoe.acquisition",
            "projects.digital_shoe.provenance",
            "projects.digital_shoe.__main__",
            "projects.digital_instron_v2",
            "projects.digital_instron_v2.__main__",
        ):
            with self.subTest(module=module):
                result = _fresh_python(source, module)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), [])

    def test_package_exports_preserve_object_identity(self):
        """Resolve every historical export to its unchanged defining module object."""
        shoe = importlib.import_module("projects.digital_shoe")
        self.assertEqual(shoe.__all__, list(_EXPORTS))
        for name, module in _EXPORTS.items():
            with self.subTest(name=name):
                original = getattr(importlib.import_module(f"projects.digital_shoe.{module}"), name)
                self.assertIs(getattr(shoe, name), original)
                self.assertIs(getattr(shoe, name), original)
        self.assertEqual(shoe.ShoeMaterial.__module__, "projects.digital_shoe.runtime")
        self.assertEqual(shoe.DigitalShoe.__module__, "projects.digital_shoe.artifact")
        with self.assertRaises(AttributeError):
            _ = shoe.not_a_shoe_export

    def test_instron_compatibility_exports_remain_shared(self):
        """Keep historical foundation imports bound to the same runtime objects."""
        dynamics = importlib.import_module("projects.digital_instron_v2.dynamics")
        runtime = importlib.import_module("projects.digital_shoe.runtime")
        for name in (
            "FoundationConfig",
            "FoundationParams",
            "MidsoleFoundation",
            "SurroundConfig",
            "foundation_apply",
            "foundation_pressure",
            "foundation_reset",
            "surround_relax",
            "surround_write_free_top",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(dynamics, name), getattr(runtime, name))

    def test_lazy_names_are_discoverable_without_loading_them(self):
        """List public exports before importing their defining modules."""
        result = _fresh_python(
            "import json,sys; import projects.digital_shoe as shoe; "
            "print(json.dumps({'missing': sorted(set(shoe.__all__) - set(dir(shoe))), "
            "'runtime': 'projects.digital_shoe.runtime' in sys.modules}))"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"missing": [], "runtime": False})


class TestShoeProjectRouting(unittest.TestCase):
    """Preserve selected-tool arguments and keep discovery free of side effects."""

    def test_top_level_help_and_no_arguments_are_lightweight(self):
        """Show package help without importing simulation or fitting modules."""
        source = """
import contextlib, io, json, runpy, sys
package = sys.argv[1]
sys.argv = [package, *sys.argv[2:]]
out = io.StringIO()
code = 0
with contextlib.redirect_stdout(out):
    try:
        runpy.run_module(package, run_name='__main__')
    except SystemExit as error:
        code = error.code
print(json.dumps({'code': code, 'help': out.getvalue(), 'loaded': [n for n in sys.modules
    if n.split('.')[0] in ('numpy', 'warp', 'newton', 'scipy', 'torch')]}))
"""
        for package, commands in _COMMANDS.items():
            for args in ([], ["--help"], ["-h"]):
                with self.subTest(package=package, args=args):
                    result = _fresh_python(source, package, *args)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads(result.stdout)
                    self.assertEqual(data["code"], 0)
                    self.assertEqual(data["loaded"], [])
                    for command in commands:
                        self.assertIn(command, data["help"])
                    if package.endswith("digital_instron_v2"):
                        self.assertIn("averaged", data["help"])
                        self.assertIn("held-out", data["help"])

    def test_routes_each_command_and_preserves_remaining_arguments(self):
        """Forward every selected command without parsing or changing its arguments."""
        for package, commands in _COMMANDS.items():
            router = importlib.import_module(package + ".__main__")
            for command, target in commands.items():
                with self.subTest(package=package, command=command):
                    original = sys.argv
                    remaining = ["--unknown-tool-option=value", "path with spaces", "--", "-h", "-4"]
                    arguments = [command, *remaining]
                    observed = []
                    tool = SimpleNamespace(main=lambda observed=observed: observed.append(list(sys.argv)))
                    with patch("projects._cli.import_module", return_value=tool) as load:
                        router.main(arguments)
                    load.assert_called_once_with(f"{package}.{target}")
                    self.assertEqual(observed, [[f"{package} {command}", *remaining]])
                    self.assertIs(sys.argv, original)
                    self.assertEqual(arguments, [command, *remaining])

    def test_default_argv_is_forwarded_and_restored(self):
        """Use process arguments when no explicit argument list is supplied."""
        router = importlib.import_module("projects.digital_shoe.__main__")
        arguments = ["caller", "report", "shoe.json", "--output", "report.html"]
        observed = []
        with (
            patch.object(sys, "argv", arguments),
            patch("projects._cli.import_module", return_value=SimpleNamespace(main=lambda: observed.append(sys.argv))),
        ):
            router.main()
            self.assertIs(sys.argv, arguments)
        self.assertEqual(observed, [["projects.digital_shoe report", *arguments[2:]]])

    def test_restores_argv_after_tool_and_import_failures(self):
        """Restore the original argument object after exceptions and help exits."""
        router = importlib.import_module("projects.digital_shoe.__main__")
        for failure in (SystemExit(0), RuntimeError("tool failed")):
            with self.subTest(failure=type(failure).__name__):
                original = sys.argv
                with (
                    patch("projects._cli.import_module") as load,
                    self.assertRaises(type(failure)),
                ):
                    load.return_value.main.side_effect = failure
                    router.main(["view", "--help"])
                self.assertIs(sys.argv, original)
        original = sys.argv
        with patch("projects._cli.import_module", side_effect=ImportError("missing")), self.assertRaises(ImportError):
            router.main(["view"])
        self.assertIs(sys.argv, original)

    def test_unknown_commands_do_not_import_tools(self):
        """Reject unknown commands rather than silently running a default task."""
        for package in _COMMANDS:
            router = importlib.import_module(package + ".__main__")
            with self.subTest(package=package):
                original = sys.argv
                with (
                    patch("projects._cli.import_module") as load,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    router.main(["unknown-command", "--help"])
                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()
                self.assertIs(sys.argv, original)

    def test_selected_help_keeps_existing_parsers_without_running_work(self):
        """Delegate help to all ten existing parsers without executing their work."""
        blockers = {
            "showcase": ["Example", "newton.examples.run"],
            "report": ["write_report"],
            "record_gifs": ["record_experiment_gifs"],
            "acquisition": ["validate_acquisition_manifest"],
            "workflow": ["run"],
            "phase1": ["evaluate", "compare_backends"],
            "phase2": ["evaluate"],
            "export_digital_shoe": ["identify_and_export"],
            "example": ["Example", "newton.examples.run"],
            "profile_calibration": ["_load_baseline"],
        }
        for package, commands in _COMMANDS.items():
            router = importlib.import_module(package + ".__main__")
            for command, target in commands.items():
                with self.subTest(package=package, command=command):
                    tool = importlib.import_module(f"{package}.{target}")
                    with contextlib.ExitStack() as stack:
                        work = [stack.enter_context(patch(f"{tool.__name__}.{name}")) for name in blockers[target]]
                        help_texts = []
                        for invoke in (
                            lambda tool=tool: tool.main(),
                            lambda command=command, router=router: router.main([command, "--help"]),
                        ):
                            output = io.StringIO()
                            with (
                                patch.object(sys, "argv", [tool.__name__, "--help"]),
                                contextlib.redirect_stdout(output),
                                self.assertRaises(SystemExit) as raised,
                            ):
                                invoke()
                            self.assertEqual(raised.exception.code, 0)
                            self.assertIn("usage:", output.getvalue())
                            help_texts.append(
                                " ".join(output.getvalue().replace(tool.__name__, f"{package} {command}").split())
                            )
                        self.assertEqual(help_texts[0], help_texts[1])
                        for task in work:
                            task.assert_not_called()

    def test_showcase_main_keeps_recording_after_example_run(self):
        """Save recorded media only after the same example instance finishes."""
        module = importlib.import_module("projects.digital_shoe.showcase")
        events = []
        instance = SimpleNamespace(save_gif=lambda: events.append("save"))
        viewer, args = object(), object()
        with (
            patch.object(module.newton.examples, "init", return_value=(viewer, args)),
            patch.object(module, "Example", return_value=instance) as example,
            patch.object(
                module.newton.examples, "run", side_effect=lambda scene, options: events.append((scene, options))
            ),
        ):
            module.main()
        example.assert_called_once_with(viewer, args)
        self.assertEqual(events, [(instance, args), "save"])


if __name__ == "__main__":
    unittest.main()
