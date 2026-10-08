# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Protect shared presentation/scenario helpers and project import boundaries."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_instron_v2 import example, geometry, scenario_common, scenarios_diff
from projects.digital_shoe import rendering, showcase

ROOT = Path(__file__).resolve().parents[2]


def _import_names(path: Path):
    """Resolve direct and relative module imports, including from-import aliases."""
    package = ".".join(path.relative_to(ROOT).parent.parts)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            yield module
            yield from (f"{module}.{alias.name}" for alias in node.names if alias.name != "*")


@wp.kernel
def _pd_probe(force: wp.array[wp.vec3], moment: wp.array[wp.vec3], limit: float, sign: float):
    target = wp.transform(wp.vec3(1.1, 1.8, 3.2), sign * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.2))
    f, m = scenario_common.attachment_pd_wrench(
        wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity()),
        wp.spatial_vector(wp.vec3(0.1, 0.2, 0.3), wp.vec3(0.0, 0.2, 0.0)),
        target,
        wp.spatial_vector(wp.vec3(0.2, 0.0, 0.1), wp.vec3(0.0, 0.3, 0.0)),
        100.0,
        4.0,
        40.0,
        3.0,
        limit,
    )
    force[0] = f
    moment[0] = m


class TestSharedProjectHelpers(unittest.TestCase):
    def test_camera_and_quaternion_imports_are_shared_objects(self):
        """Use one helper object instead of independent camera or quaternion copies."""
        self.assertIs(example._look_at, showcase._look_at)
        self.assertIs(example._look_at, rendering.camera_look_at)
        self.assertIs(example._quat_mul, scenarios_diff._quat_mul)
        self.assertIs(example._quat_mul, scenario_common.quat_multiply)
        self.assertIs(example._quat_inv, scenarios_diff._quat_inv)
        position, pitch, yaw = rendering.camera_look_at([1, 0, 0], [0, 0, 0])
        np.testing.assert_array_equal(np.asarray(position), [1.0, 0.0, 0.0])
        self.assertEqual(pitch, 0.0)
        self.assertEqual(abs(yaw), 180.0)
        np.testing.assert_array_equal(example._quat_mul([0, 0, 0, 1], [1, 2, 3, 4]), np.array([1, 2, 3, 4], np.float32))

    def test_shared_surround_preserves_explicit_local_settings(self):
        """Keep scenario-level settings overridable without copying a factory body."""
        driven = np.array([1, 0], np.int32)
        with patch.object(scenarios_diff, "SURROUND_SWEEPS", 9):
            config = scenarios_diff.default_surround(driven, carrier_bond=False)
        self.assertEqual(config.sweeps, 9)
        np.testing.assert_array_equal(config.driven, driven)
        self.assertFalse(config.carrier_bond)
        self.assertEqual(config.attachment_n_m, 0.0)

    def test_shared_pd_preserves_clamp_and_quaternion_sign(self):
        """Retain original PD force, moment, clamp and equivalent-quaternion behavior."""
        raw = np.array([10.4, -20.8, 19.2])
        expected_moment = np.array([0.0, 80.0 * np.sin(0.1) + 0.3, 0.0])
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            for limit in (20.0, 1000.0):
                for sign in (-1.0, 1.0):
                    with self.subTest(device=str(device), limit=limit, sign=sign):
                        force = wp.zeros(1, dtype=wp.vec3, device=device)
                        moment = wp.zeros_like(force)
                        wp.launch(_pd_probe, dim=1, inputs=[force, moment, limit, sign], device=device)
                        expected = raw * min(1.0, limit / np.linalg.norm(raw))
                        np.testing.assert_allclose(force.numpy()[0], expected, atol=1e-5, rtol=1e-6)
                        np.testing.assert_allclose(moment.numpy()[0], expected_moment, atol=1e-5, rtol=1e-6)

    def test_geometry_rotation_and_ray_setup_are_shared(self):
        """Retain known rotations and ray origins through the common geometry helpers."""
        rotation = geometry._rotation_xyz([0.0, 0.0, 90.0])
        np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)
        mesh = SimpleNamespace(
            vertices=np.array([[1.0, 0.0, 0.0]]), bounds=np.array([[-1.0, -2.0, 0.3], [1.0, 2.0, 0.9]])
        )
        geometry.transform_mesh(mesh, [0.0, 0.0, 90.0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(mesh.vertices, [[0.1, 1.2, 0.3]], atol=1e-15)
        origins, directions = geometry._positive_rays(mesh, np.array([[0.2, 0.4], [-0.1, 0.8]]), 2)
        np.testing.assert_allclose(origins, [[0.2, 0.4, 0.29], [-0.1, 0.8, 0.29]], atol=1e-15)
        np.testing.assert_array_equal(directions, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    def test_unused_normal_readout_is_gone_but_legacy_shear_is_explicit(self):
        """Remove the dead recomputation without deleting an explicit compatibility helper."""
        self.assertFalse(hasattr(scenarios_diff, "_ground_reaction_force"))
        self.assertTrue(hasattr(scenarios_diff, "_shear_reaction_force"))


class TestProjectDependencyBoundaries(unittest.TestCase):
    def test_portable_shoe_has_no_controller_or_fitter_dependency(self):
        """Keep shared shoe code independent of its fitting and controller consumers."""
        runtime_modules = (
            "__init__",
            "artifact",
            "runtime",
            "material",
            "contact",
            "rendering",
            "provenance",
            "friction_law",
            "friction_deflection",
            "friction_stribeck",
            "friction_pressure",
            "friction_slip_history",
            "friction_maxwell",
            "friction_parameter_adapter",
            "friction_solver",
            "friction_adapter",
        )
        for module in runtime_modules:
            path = ROOT / "projects" / "digital_shoe" / f"{module}.py"
            for name in _import_names(path):
                self.assertFalse(
                    name.startswith(("projects.impedance_instron", "projects.digital_instron_v2")), (path, name)
                )

    def test_active_impedance_sources_do_not_import_retired_trees(self):
        """Keep retained Hogan and data-preparation sources outside retired experiment trees."""
        base = ROOT / "projects/impedance_instron"
        files = sorted(base.rglob("*.py"))
        self.assertIn(base / "__main__.py", files)
        self.assertIn(base / "hogan" / "runner.py", files)
        retired = {"legacy", "simple", "paper", "joint_space"}
        retired_modules = {"projects.impedance_instron.pipeline"}
        retired_modules.update(
            f"projects.impedance_instron.cartesian.{name}"
            for name in (
                "run",
                "fit",
                "trajectory",
                "spline",
                "mechanics",
                "phase",
                "diagnostics",
                "report",
                "rendering",
                "springs",
                "prepare_subject",
            )
        )
        retired_modules.update(
            f"projects.impedance_instron.hogan.{name}"
            for name in (
                "adaptation",
                "control",
                "plan",
                "registration",
                "rollout",
                "batch",
                "learn",
                "learn_report",
                "quick_fit",
                "recovery",
                "gpu_objective",
                "identify.FitConfig",
                "identify.fit",
            )
        )
        gpu_package = "projects.impedance_instron.cartesian.gpu"
        for path in files:
            for module in _import_names(path):
                location = (path.relative_to(base), module)
                self.assertTrue(retired.isdisjoint(module.split(".")), location)
                self.assertFalse(
                    any(module == old or module.startswith(old + ".") for old in retired_modules), location
                )
                if module.startswith(gpu_package + "."):
                    self.assertTrue(
                        module == gpu_package + ".foundation" or module.startswith(gpu_package + ".foundation."),
                        location,
                    )

    def test_dependency_scan_resolves_relative_from_import_aliases(self):
        """Expose module aliases that would otherwise hide retired relative imports."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "projects" / "impedance_instron" / "hogan" / "module.py"
            path.parent.mkdir(parents=True)
            path.write_text(
                "from . import plan\n"
                "from ..cartesian import mechanics\n"
                "from ..cartesian.gpu import foundation\n"
                "from projects import digital_instron_v2\n",
                encoding="utf-8",
            )
            with patch("newton.tests.test_shoe_project_helpers.ROOT", root):
                modules = set(_import_names(path))
        self.assertIn("projects.impedance_instron.hogan.plan", modules)
        self.assertIn("projects.impedance_instron.cartesian.mechanics", modules)
        self.assertIn("projects.impedance_instron.cartesian.gpu.foundation", modules)
        self.assertIn("projects.digital_instron_v2", modules)


class TestDuplicateReviewTool(unittest.TestCase):
    def test_repository_has_no_unreviewed_exact_clones(self):
        """Require an explicit review for any new cross-project function clone."""
        spec = importlib.util.spec_from_file_location(
            "shoe_duplicates", ROOT / "scripts/check_shoe_project_duplicates.py"
        )
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        review = json.loads((ROOT / "projects/shoe_duplicate_review.json").read_text())
        result = checker.scan(ROOT, near_limit=0)
        self.assertEqual(checker.unreviewed_groups(result, review), [])

    def test_detects_renamed_functions_without_merging_different_constants(self):
        """Find structural clones while retaining semantic constants and requiring review."""
        spec = importlib.util.spec_from_file_location(
            "shoe_duplicates", ROOT / "scripts/check_shoe_project_duplicates.py"
        )
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for package, text in (
                ("digital_shoe", "def a(x):\n    return x + 1\n"),
                ("digital_instron_v2", "def b(y):\n    return y + 1\n"),
                ("impedance_instron", "def c(z):\n    return z + 2\n"),
            ):
                path = root / "projects" / package / "module.py"
                path.parent.mkdir(parents=True)
                path.write_text(text)
            result = checker.scan(root, min_nodes=5, near_limit=0)
        groups = result["exact_or_renamed_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual({f["function"] for f in groups[0]["functions"]}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
