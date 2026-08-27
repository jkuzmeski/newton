# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the approximate Newton-native gait articulation."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

import newton
from newton.solvers import SolverFeatherstone
from projects.gait_c3d import native_model


class TestSimpleGaitModel(unittest.TestCase):
    """Test the structure and short-horizon dynamics of the simple model."""

    @classmethod
    def setUpClass(cls):
        cls.previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        cls.build = native_model.build_simple_gait_model()
        cls.model = cls.build.builder.finalize(device="cpu")

    @classmethod
    def tearDownClass(cls):
        newton.use_coord_layout_targets = cls.previous_target_layout

    def test_runtime_import_is_opensim_independent(self):
        """Import the native model while blocking OpenSim namespaces."""
        path = Path(native_model.__file__)
        tree = ast.parse(path.read_text())
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports.update(
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        self.assertFalse(any("opensim" in name.lower() for name in imports))

        code = """
import importlib.abc
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == 'opensim'
            or fullname.startswith('opensim.')
            or fullname == 'newton.opensim'
            or fullname.startswith('newton._src.opensim')
        ):
            raise ImportError('blocked: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
import projects.gait_c3d.native_model
assert 'newton.opensim' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=path.parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_builds_declared_simple_topology(self):
        """Build two D6 hips, two hinge knees, and two hinge ankles."""
        self.assertEqual(self.model.body_count, 8)
        self.assertEqual(self.model.joint_count, 8)
        self.assertEqual(self.model.joint_coord_count, 17)
        self.assertEqual(self.model.joint_dof_count, 16)
        self.assertEqual(len(self.build.body_shape_indices), 6)
        self.assertEqual(len(self.build.contact_shape_indices), 8)
        self.assertEqual(self.model.shape_count, 15)
        labels = self.model.joint_label
        types = self.model.joint_type.numpy()
        for side in ("left", "right"):
            hip = labels.index(f"hip_{side}")
            knee = labels.index(f"knee_{side}")
            ankle = labels.index(f"ankle_{side}")
            self.assertEqual(types[hip], newton.JointType.D6)
            self.assertEqual(types[knee], newton.JointType.REVOLUTE)
            self.assertEqual(types[ankle], newton.JointType.REVOLUTE)
            qd_starts = self.model.joint_qd_start.numpy()
            self.assertEqual(qd_starts[knee + 1] - qd_starts[knee], 1)
            self.assertEqual(qd_starts[ankle + 1] - qd_starts[ankle], 1)
        self.assertAlmostEqual(float(np.sum(self.model.body_mass.numpy())), 81.4, places=4)

    def test_uses_boxes_capsules_and_foot_spheres(self):
        """Use primitive fallback bodies and sphere-only foot geometry."""
        shape_types = self.model.shape_type.numpy()
        shape_flags = self.model.shape_flags.numpy()
        for label in ("pelvis", "torso"):
            shape = self.build.body_shape_indices[label]
            self.assertEqual(shape_types[shape], newton.GeoType.BOX)
            self.assertFalse(shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
        for side in ("left", "right"):
            for segment in ("femur", "tibia"):
                shape = self.build.body_shape_indices[f"{segment}_{side}"]
                self.assertEqual(shape_types[shape], newton.GeoType.CAPSULE)
                self.assertFalse(shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
        for shape in self.build.contact_shape_indices:
            self.assertEqual(shape_types[shape], newton.GeoType.SPHERE)
            self.assertTrue(shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)

    def test_scales_geometry_and_mass_for_subject(self):
        """Scale all segment lengths and masses from height and body mass."""
        config = native_model.SimpleGaitConfig.for_subject(body_mass=100.0, body_height=2.0, hip_width=0.25)
        length_scale = 2.0 / 1.695898298375747
        self.assertAlmostEqual(config.thigh_length, 0.45 * length_scale)
        self.assertAlmostEqual(config.shank_length, 0.40 * length_scale)
        self.assertAlmostEqual(config.contact_radius, 0.04 * length_scale)
        self.assertAlmostEqual(config.hip_half_width, 0.125)
        scaled_build = native_model.build_simple_gait_model(config)
        scaled_model = scaled_build.builder.finalize(device="cpu")
        self.assertAlmostEqual(float(np.sum(scaled_model.body_mass.numpy())), 100.0, places=4)
        shape_scale = scaled_model.shape_scale.numpy()
        pelvis_shape = scaled_build.body_shape_indices["pelvis"]
        np.testing.assert_allclose(
            shape_scale[pelvis_shape],
            0.5 * np.asarray(config.pelvis_dimensions),
            atol=1.0e-6,
        )

    def test_initializes_finite_bilateral_pose(self):
        """Initialize finite mirrored legs with feet tangent to the ground."""
        state = native_model.initialize_simple_gait_state(self.model, self.build)
        body_q = state.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(state.joint_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state.joint_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(state.body_qd.numpy())))
        np.testing.assert_allclose(np.linalg.norm(body_q[:, 3:], axis=1), 1.0, atol=1.0e-6)
        left = body_q[self.build.body_indices["foot_left"], :3]
        right = body_q[self.build.body_indices["foot_right"], :3]
        np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=1.0e-6)
        self.assertAlmostEqual(left[1], -right[1], places=6)
        self.assertAlmostEqual(left[2], 0.08, places=6)

    def test_actuation_policy_excludes_root(self):
        """Apply internal test torques while keeping free-pelvis entries zero."""
        self.assertEqual(self.build.root_dof_slice, slice(0, 6))
        self.assertTrue(all(index >= self.build.root_dof_slice.stop for index in self.build.actuated_dof_indices))
        joint_force = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        joint_force[list(self.build.actuated_dof_indices)] = 1.0
        control = self.model.control()
        control.joint_f.assign(joint_force)
        result = control.joint_f.numpy()
        np.testing.assert_array_equal(result[self.build.root_dof_slice], np.zeros(6, dtype=result.dtype))
        np.testing.assert_array_equal(result[list(self.build.actuated_dof_indices)], np.ones(10, dtype=result.dtype))

    def test_featherstone_contact_rollout_stays_finite(self):
        """Advance a short unactuated contact rollout without nonfinite state."""
        state = native_model.initialize_simple_gait_state(self.model, self.build)
        state_out = self.model.state()
        control = self.model.control()
        pipeline = newton.CollisionPipeline(self.model)
        contacts = pipeline.contacts()
        solver = SolverFeatherstone(self.model, angular_damping=0.01)
        initial_pelvis_z = float(state.body_q.numpy()[self.build.body_indices["pelvis"], 2])
        maximum_contact_count = 0
        for _ in range(20):
            state.clear_forces()
            pipeline.collide(state, contacts)
            maximum_contact_count = max(maximum_contact_count, int(contacts.rigid_contact_count.numpy()[0]))
            solver.step(state, state_out, control, contacts, 0.001)
            state, state_out = state_out, state
        body_q = state.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(state.joint_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state.joint_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(state.body_qd.numpy())))
        self.assertGreater(maximum_contact_count, 0)
        self.assertLess(body_q[self.build.body_indices["pelvis"], 2], initial_pelvis_z)
        self.assertGreater(body_q[self.build.body_indices["pelvis"], 2], 0.9)


if __name__ == "__main__":
    unittest.main()
