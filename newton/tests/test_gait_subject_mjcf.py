# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the saved MJCF form of a scaled gait subject."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.subject_mjcf import write_subject_mjcf


class TestGaitSubjectMJCF(unittest.TestCase):
    """Test MJCF interoperability and one-call ModelBuilder loading."""

    @classmethod
    def setUpClass(cls):
        cls.previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True

    @classmethod
    def tearDownClass(cls):
        newton.use_coord_layout_targets = cls.previous_target_layout

    def test_loads_scaled_subject_in_one_builder_call(self):
        """Load all bodies, simple joints, contacts, and controls with add_mjcf."""
        config = SimpleGaitConfig.for_subject(body_mass=74.0, body_height=1.82, hip_width=0.20)
        with tempfile.TemporaryDirectory() as directory:
            path = write_subject_mjcf(config, Path(directory) / "subject.xml", model_name="subject_001")
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(path), enable_self_collisions=True)
            model = builder.finalize(device="cpu")
        self.assertEqual(model.body_count, 8)
        self.assertEqual(model.joint_count, 8)
        self.assertEqual(model.joint_coord_count, 17)
        self.assertEqual(model.joint_dof_count, 16)
        self.assertEqual(model.shape_count, 30)
        self.assertAlmostEqual(float(np.sum(model.body_mass.numpy())), 74.0, places=4)
        modes = model.joint_target_mode.numpy()
        np.testing.assert_array_equal(modes[:6], np.zeros(6, dtype=modes.dtype))
        np.testing.assert_array_equal(
            modes[6:],
            np.full(10, newton.JointTargetMode.POSITION_VELOCITY, dtype=modes.dtype),
        )
        np.testing.assert_allclose(model.joint_target_ke.numpy()[6:], 100.0)
        np.testing.assert_allclose(model.joint_target_kd.numpy()[6:], 20.0)

    def test_imports_self_collision_proxies_and_neighbor_filters(self):
        """Import invisible segment proxies and preserve adjacent-link filters."""
        config = SimpleGaitConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = write_subject_mjcf(config, Path(directory) / "subject.xml")
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(path), enable_self_collisions=True)
            model = builder.finalize(device="cpu")
        labels = model.shape_label
        shape_by_name = {
            name: next(index for index, label in enumerate(labels) if label.endswith(f"/{name}"))
            for name in (
                "geometry_femur_left",
                "geometry_tibia_left",
                "collision_pelvis",
                "collision_torso",
                "collision_femur_left",
                "collision_femur_right",
                "collision_tibia_left",
                "collision_tibia_right",
            )
        }
        flags = model.shape_flags.numpy()
        shape_scale = model.shape_scale.numpy()
        for name in (
            "collision_pelvis",
            "collision_torso",
            "collision_femur_left",
            "collision_femur_right",
            "collision_tibia_left",
            "collision_tibia_right",
        ):
            shape = shape_by_name[name]
            self.assertTrue(flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
            self.assertFalse(flags[shape] & newton.ShapeFlags.VISIBLE)
        for name in ("geometry_femur_left", "geometry_tibia_left"):
            shape = shape_by_name[name]
            self.assertTrue(flags[shape] & newton.ShapeFlags.VISIBLE)
            self.assertFalse(flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
        expected_thigh_half_length = 0.5 * (config.thigh_length - 2.0 * config.self_collision_joint_clearance)
        for name in ("geometry_femur_left", "collision_femur_left"):
            self.assertAlmostEqual(
                shape_scale[shape_by_name[name], 1],
                expected_thigh_half_length,
                places=6,
            )
        filters = set(model.shape_collision_filter_pairs)
        self.assertIn(
            tuple(sorted((shape_by_name["collision_pelvis"], shape_by_name["collision_femur_left"]))),
            filters,
        )
        self.assertIn(
            tuple(sorted((shape_by_name["collision_femur_left"], shape_by_name["collision_tibia_left"]))),
            filters,
        )
        self.assertNotIn(
            tuple(sorted((shape_by_name["collision_femur_left"], shape_by_name["collision_femur_right"]))),
            filters,
        )

    def test_mujoco_loads_export_and_neutral_keyframe(self):
        """Load the same saved subject with MuJoCo and apply its neutral keyframe."""
        try:
            import mujoco
        except ImportError as error:
            self.skipTest(str(error))
        config = SimpleGaitConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = write_subject_mjcf(config, Path(directory) / "subject.xml")
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
        self.assertEqual(model.nq, 17)
        self.assertEqual(model.nv, 16)
        self.assertEqual(model.nu, 20)
        self.assertTrue(np.all(np.isfinite(data.qpos)))
        self.assertAlmostEqual(float(data.qpos[2]), config.pelvis_height)


if __name__ == "__main__":
    unittest.main()
