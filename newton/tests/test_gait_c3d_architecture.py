# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Enforce the C3D OpenSim-adapter/Newton-runtime architecture boundary."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGaitC3DArchitecture(unittest.TestCase):
    """Reject compatibility or oracle dependencies in production runtime modules."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        config_path = cls.root / "projects/gait_c3d/ARCHITECTURE_BOUNDARIES.json"
        if not config_path.is_file():
            raise unittest.SkipTest("project-only architecture policy is not included in installed wheels")
        cls.config = json.loads(config_path.read_text())

    @staticmethod
    def _dotted(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = TestGaitC3DArchitecture._dotted(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    @classmethod
    def _module_analysis(cls, path: Path) -> tuple[set[str], set[str], set[str]]:
        tree = ast.parse(path.read_text())
        imports: set[str] = set()
        dynamic_imports: set[str] = set()
        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
                imports.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Call):
                dotted = cls._dotted(node.func)
                if dotted:
                    calls.add(dotted)
                if dotted in {"importlib.import_module", "__import__"} and node.args:
                    try:
                        dynamic_imports.add(ast.literal_eval(node.args[0]))
                    except (ValueError, TypeError):
                        pass
        return imports, dynamic_imports, calls

    def test_every_audited_module_declares_its_role(self):
        """Make adapter, oracle, compatibility, and native roles explicit in source."""
        for relative, role in self.config["modules"].items():
            path = self.root / relative
            self.assertTrue(path.is_file(), relative)
            tree = ast.parse(path.read_text())
            declarations = {
                target.id: ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name) and target.id == "ARCHITECTURE_ROLE"
            }
            self.assertEqual(declarations.get("ARCHITECTURE_ROLE"), role, relative)

    def test_relocated_modules_do_not_use_the_old_repository_depth(self):
        """Make moved modules resolve repository paths from their new package depth."""
        for relative in self.config["relocated_modules"].values():
            source = (self.root / relative).read_text()
            self.assertNotIn("parents[2]", source, relative)

    def test_relocated_modules_keep_deprecated_redirects(self):
        """Preserve old import paths for one deprecation period."""
        for old, replacement in self.config["relocated_modules"].items():
            tree = ast.parse((self.root / old).read_text())
            declarations = {
                target.id: ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in {"ARCHITECTURE_ROLE", "_REPLACEMENT"}
            }
            self.assertEqual(declarations.get("ARCHITECTURE_ROLE"), "deprecated_redirect", old)
            self.assertEqual(declarations.get("_REPLACEMENT"), replacement.removesuffix(".py").replace("/", "."), old)

        code = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import projects.gait_c3d.rra_adjusted_contact_input as old
from projects.gait_c3d.adapters import rra_adjusted_contact_input as replacement
assert old.main is replacement.main
assert any(item.category is DeprecationWarning for item in caught)
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_production_entrypoint_is_native_and_transitively_isolated(self):
        """Forbid any native -> adapter/reference dependency path."""
        modules = self.config["modules"]
        entrypoints = set(self.config["production_entrypoints"])
        self.assertTrue(entrypoints)
        for entrypoint in entrypoints:
            self.assertEqual(modules.get(entrypoint), "native_runtime")
        module_by_import = {relative.removesuffix(".py").replace("/", "."): relative for relative in modules}
        graph: dict[str, set[str]] = {}
        for relative in modules:
            imports, _, _ = self._module_analysis(self.root / relative)
            graph[relative] = {module_by_import[name] for name in imports if name in module_by_import}
        visited: set[str] = set()
        pending = list(entrypoints)
        while pending:
            relative = pending.pop()
            if relative in visited:
                continue
            visited.add(relative)
            pending.extend(graph[relative])
        forbidden_roles = set(self.config["reference_only_roles"]) | {"source_adapter"}
        leaked = {relative: modules[relative] for relative in visited if modules[relative] in forbidden_roles}
        self.assertFalse(leaked, leaked)

    def test_native_runtime_has_required_core_calls_and_no_opensim(self):
        """Inspect AST imports and calls rather than accepting strings or comments."""
        requirements = self.config["native_runtime_requirements"]
        for relative in self.config["production_entrypoints"]:
            imports, dynamic_imports, calls = self._module_analysis(self.root / relative)
            lowered = {name.lower() for name in imports | dynamic_imports}
            for fragment in requirements["forbidden_import_fragments"]:
                self.assertFalse(any(fragment in name for name in lowered), f"{relative}: {fragment}")
            source = (self.root / relative).read_text()
            for fragment in requirements["forbidden_symbol_fragments"]:
                self.assertNotIn(fragment, source, relative)
            for symbol in requirements["required_symbols"]:
                self.assertIn(symbol, calls, f"{relative}: missing call {symbol}")

    def test_reference_clis_require_explicit_acknowledgement(self):
        """Prevent compatibility mechanics from being selected as production by accident."""
        for relative, role in self.config["modules"].items():
            if role != "compatibility_reference":
                continue
            source = (self.root / relative).read_text()
            self.assertIn("--reference-only", source, relative)

    def test_runtime_import_succeeds_with_opensim_and_references_blocked(self):
        """Import the production module in a fresh process that blocks boundary modules."""
        blocked = [
            "opensim",
            "newton.opensim",
            *[
                relative.removesuffix(".py").replace("/", ".")
                for relative, role in self.config["modules"].items()
                if role in set(self.config["reference_only_roles"]) | {"source_adapter"}
            ],
        ]
        code = f"""
import importlib.abc, sys
blocked = {blocked!r}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError('blocked architecture boundary: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
import projects.gait_c3d.newton_contact_calibration
assert not any(
    name == 'opensim' or name.startswith('opensim.') or name == 'newton.opensim' or name.startswith('newton.opensim.')
    for name in sys.modules
)
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compatibility_library_is_not_mislabeled_native(self):
        """Keep branch changes under newton._src.opensim at the optional boundary."""
        for relative in self.config["compatibility_library_files"]:
            self.assertTrue((self.root / relative).is_file())
            self.assertNotIn(relative, self.config["production_entrypoints"])

    def test_all_project_modules_are_classified(self):
        """Classify every current project module, including untracked modules."""
        project_root = self.root / "projects/gait_c3d"
        current = {
            path.relative_to(self.root).as_posix()
            for path in project_root.rglob("*.py")
            if "__pycache__" not in path.parts
        }
        self.assertTrue(current)
        missing = current - self.config["modules"].keys()
        stale = self.config["modules"].keys() - current
        self.assertFalse(missing, sorted(missing))
        self.assertFalse(stale, sorted(stale))


if __name__ == "__main__":
    unittest.main()
