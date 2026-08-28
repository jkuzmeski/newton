# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the saved MJCF form of a scaled gait subject."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import newton
from newton.examples.opensim.example_opensim_subject import _resolve_subject_artifact, create_parser
from projects.gait_c3d.c3d_adapter import C3DMarkerTrajectory
from projects.gait_c3d.calibrated_subject import write_calibrated_subject_mjcf
from projects.gait_c3d.marker_layout import scale_subject_marker_layout_from_base
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.segment_calibration import build_static_segment_calibration, load_static_segment_calibration
from projects.gait_c3d.subject_mjcf import scale_subject_mjcf_from_base, write_subject_mjcf


class TestGaitSubjectMJCF(unittest.TestCase):
    """Test MJCF interoperability and one-call ModelBuilder loading."""

    @classmethod
    def setUpClass(cls):
        cls.previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True

    @classmethod
    def tearDownClass(cls):
        newton.use_coord_layout_targets = cls.previous_target_layout

    def test_scales_complete_s001_subject_from_base_geometry(self):
        """Scale S001 meshes, marker sites, frames, and inertias as one MJCF."""
        base = Path(__file__).parents[2] / "projects" / "gait_c3d" / "assets" / "s001_base"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model" / "subject.xml"
            scaled = scale_subject_mjcf_from_base(
                base,
                output,
                body_height=1.8,
                body_mass=90.0,
                model_name="scaled_s001",
            )
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(output), floating=True, parse_sites=True, enable_self_collisions=True)
            model = builder.finalize(device="cpu")
            shape_flags = model.shape_flags.numpy()
            visual_meshes = [
                index
                for index, label in enumerate(model.shape_label)
                if label.rsplit("/", 1)[-1].startswith("visual_")
                and model.shape_type.numpy()[index] == newton.GeoType.MESH
            ]
            self.assertEqual(len(visual_meshes), 19)
            self.assertTrue(all(shape_flags[index] & newton.ShapeFlags.VISIBLE for index in visual_meshes))
            self.assertTrue(all(not shape_flags[index] & newton.ShapeFlags.COLLIDE_SHAPES for index in visual_meshes))
            obj = Path(directory) / "model" / "Geometry" / "visual_00_pelvis_sacrum_ec82986d.obj"
            base_obj = base / "model" / "Geometry" / obj.name
            base_vertex = np.asarray([float(value) for value in base_obj.read_text().splitlines()[0].split()[1:]])
            scaled_vertex = np.asarray([float(value) for value in obj.read_text().splitlines()[0].split()[1:]])
            root = ET.parse(output).getroot()
            pelvis = next(body for body in root.iter("body") if body.get("name") == "pelvis")
            pelvis_position = np.asarray([float(value) for value in pelvis.get("pos").split()])
        self.assertAlmostEqual(scaled.length_scale, 1.8 / 1.695898298375747)
        self.assertAlmostEqual(float(np.sum(model.body_mass.numpy())), 90.0, places=4)
        self.assertAlmostEqual(pelvis_position[2], scaled.config.pelvis_height, places=7)
        self.assertAlmostEqual(scaled.config.contact_radius, scaled.length_scale * 0.0245631567, places=7)
        self.assertEqual(sum(1 for flags in model.shape_flags.numpy() if flags & newton.ShapeFlags.SITE), 35)
        np.testing.assert_allclose(scaled_vertex, scaled.length_scale * base_vertex, atol=1.0e-8)

    def test_scales_bilateral_geometry_from_static_segment_dimensions(self):
        """Scale each leg segment independently from changed static landmarks."""
        base = Path(__file__).parents[2] / "projects" / "gait_c3d" / "assets" / "s001_base"
        base_calibration = load_static_segment_calibration(base / "model" / "segment_calibration.json")
        marker_positions = {name: value.copy() for name, value in base_calibration.marker_positions.items()}
        for name in ("LKNE", "LMKNE"):
            marker_positions[name][2] -= 0.04
        for name in ("RANK", "RMANK"):
            marker_positions[name][1] += 0.01 if name == "RMANK" else -0.01
        names = tuple(marker_positions)
        trajectory = C3DMarkerTrajectory(
            times=np.asarray((0.5,), dtype=np.float64),
            positions=np.asarray([[marker_positions[name] for name in names]], dtype=np.float32),
            valid=np.ones((1, len(names)), dtype=bool),
            marker_names=names,
            rate=100.0,
            first_frame=0,
            lab_to_newton=np.eye(3),
            source_file="personalized_static.c3d",
            source_sha256="1" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            target_calibration = build_static_segment_calibration(
                trajectory,
                Path(directory) / "calibration.json",
                marker_radius=0.006,
            )
            output = Path(directory) / "model" / "subject.xml"
            write_calibrated_subject_mjcf(
                base,
                target_calibration,
                output,
                body_mass=90.0,
                base_calibration=base_calibration,
            )
            manifest = json.loads((output.parent / "manifest.json").read_text(encoding="utf-8"))
        scales = manifest["segment_scales"]
        self.assertGreater(scales["thigh_left"][2], 1.05)
        self.assertLess(scales["shank_left"][2], 0.95)
        self.assertGreater(scales["shank_right"][0], 1.10)

    def test_builds_s001_calibrated_model_with_flat_foot_contacts(self):
        """Build calibrated S001 frames and keep every foot contact on z=0."""
        base = Path(__file__).parents[2] / "projects" / "gait_c3d" / "assets" / "s001_base"
        calibration = load_static_segment_calibration(base / "model" / "segment_calibration.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model" / "subject.xml"
            write_calibrated_subject_mjcf(base, calibration, output, body_mass=81.9312118)
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(output), floating=False, parse_sites=True, enable_self_collisions=True)
            model = builder.finalize(device="cpu")
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            body_q = state.body_q.numpy()
            shape_body = model.shape_body.numpy()
            shape_transform = model.shape_transform.numpy()
            body_names = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(model.body_label)}
            site_by_name = {
                label.rsplit("/", 1)[-1]: index
                for index, label in enumerate(model.shape_label)
                if label.rsplit("/", 1)[-1].startswith("marker_")
            }
            ground_offset = np.asarray(
                json.loads((output.parent / "manifest.json").read_text(encoding="utf-8"))["ground"]["global_offset_m"],
                dtype=np.float64,
            )

        def rotate(point_quaternion, point):
            """Rotate one point by an xyzw quaternion."""
            x, y, z, w = point_quaternion[3], point_quaternion[4], point_quaternion[5], point_quaternion[6]
            q = np.asarray((x, y, z), dtype=np.float64)
            return point + 2.0 * np.cross(q, w * point + np.cross(q, point))

        foot_bodies = {body_names["foot_left"], body_names["foot_right"]}
        contact_indices = [
            index for index, label in enumerate(model.shape_label) if label.rsplit("/", 1)[-1].startswith("contact_")
        ]
        self.assertEqual(len(contact_indices), 8)
        self.assertEqual(model.joint_coord_count, 13)
        self.assertTrue(any(label.rsplit("/", 1)[-1].startswith("torso_flexion") for label in model.joint_label))
        expected_torso_origin = calibration.marker_positions["VSAC"] + ground_offset
        np.testing.assert_allclose(body_q[body_names["torso"], :3], expected_torso_origin, atol=2.0e-6)
        for site_name, source_name in (
            ("marker_L.ASIS", "LASI"),
            ("marker_R.ASIS", "RASI"),
            ("marker_V.Sacral", "VSAC"),
            ("marker_Top.Head", "TOPHEAD"),
        ):
            site = site_by_name[site_name]
            body = shape_body[site]
            world = body_q[body, :3] + rotate(body_q[body], shape_transform[site, :3])
            np.testing.assert_allclose(world, calibration.marker_positions[source_name] + ground_offset, atol=2.0e-6)
        self.assertGreater(np.linalg.norm(body_q[body_names["pelvis"], 3:6]), 0.01)
        for body in foot_bodies:
            up = rotate(body_q[body], np.asarray((0.0, 0.0, 1.0)))
            np.testing.assert_allclose(up, (0.0, 0.0, 1.0), atol=1.0e-6)
        for index in contact_indices:
            body = shape_body[index]
            center = body_q[body, :3] + rotate(body_q[body], shape_transform[index, :3])
            radius = float(model.shape_scale.numpy()[index, 0])
            self.assertAlmostEqual(center[2] - radius, 0.0, delta=1.0e-5)

    def test_applies_explicit_base_hip_width_to_xml_and_marker_frames(self):
        """Keep explicit hip width consistent across MJCF and marker layout."""
        base = Path(__file__).parents[2] / "projects" / "gait_c3d" / "assets" / "s001_base"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model" / "subject.xml"
            scaled = scale_subject_mjcf_from_base(
                base,
                output,
                body_height=1.8,
                body_mass=90.0,
                hip_width=0.30,
            )
            layout_path = Path(directory) / "model" / "marker_layout.json"
            layout = scale_subject_marker_layout_from_base(
                base / "model" / "marker_layout.json",
                layout_path,
                length_scale=scaled.length_scale,
                hip_width=0.30,
            )
            root = ET.parse(output).getroot()
            femur_positions = {
                body.get("name"): np.asarray([float(value) for value in body.get("pos").split()])
                for body in root.iter("body")
                if body.get("name") in {"femur_left", "femur_right"}
            }
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(output), floating=True, parse_sites=True)
            model = builder.finalize(device="cpu")
            shape_body = model.shape_body.numpy()
            shape_transform = model.shape_transform.numpy()
            sites = {
                model.shape_label[index].rsplit("/", 1)[-1]: index
                for index, flags in enumerate(model.shape_flags.numpy())
                if flags & newton.ShapeFlags.SITE
            }
        self.assertAlmostEqual(femur_positions["femur_left"][1], 0.15, places=7)
        self.assertAlmostEqual(femur_positions["femur_right"][1], -0.15, places=7)
        self.assertAlmostEqual(layout.target_body_transforms["femur_left"][1, 3], 0.15, places=7)
        self.assertAlmostEqual(layout.target_body_transforms["femur_right"][1, 3], -0.15, places=7)
        body_by_name = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(model.body_label)}
        for marker in layout.markers:
            site = sites[marker.site_name]
            self.assertEqual(shape_body[site], body_by_name[marker.body])
            np.testing.assert_allclose(shape_transform[site, :3], marker.position, atol=1.0e-7)

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

    def test_resolves_only_declared_in_bundle_artifacts(self):
        """Resolve declared artifacts and reject path escape or missing files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model" / "marker_layout.json"
            artifact.parent.mkdir()
            artifact.write_text("{}", encoding="utf-8")
            manifest = {"artifacts": {"marker_layout": "model/marker_layout.json"}}
            self.assertEqual(_resolve_subject_artifact(root, manifest, "marker_layout"), artifact)
            self.assertIsNone(_resolve_subject_artifact(root, manifest, "absent"))
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                _resolve_subject_artifact(root, {"artifacts": {"marker_layout": "../outside"}}, "marker_layout")
            with self.assertRaisesRegex(FileNotFoundError, "is missing"):
                _resolve_subject_artifact(
                    root,
                    {"artifacts": {"marker_layout": "model/missing.json"}},
                    "marker_layout",
                )

    def test_cli_exposes_concise_subject_options(self):
        """Expose concise subject options while accepting legacy names."""
        parser = create_parser()
        help_text = parser.format_help()
        for option in (
            "--subject",
            "--mass",
            "--height",
            "--template",
            "--geometry",
            "--substeps",
            "--show-collision",
            "--show-markers",
            "--marker-demo",
            "--base-subject",
            "--static-cal",
            "--show-calibration",
        ):
            self.assertIn(option, help_text)
        for option in (
            "--body-mass",
            "--body-height",
            "--template-osim",
            "--geometry-dir",
            "--subject-dir",
            "--subject-substeps",
        ):
            self.assertNotIn(option, help_text)
        args = parser.parse_args(
            [
                "--body-mass",
                "90",
                "--body-height",
                "1.8",
                "--subject-substeps",
                "12",
                "--show-self-collision",
            ]
        )
        self.assertEqual(args.body_mass, 90.0)
        self.assertEqual(args.body_height, 1.8)
        self.assertEqual(args.subject_substeps, 12)
        self.assertTrue(args.show_self_collision)

    def test_scales_default_inertia_proxies_with_subject(self):
        """Scale default inertia-derived proxies with subject dimensions."""
        config = SimpleGaitConfig.for_subject(body_mass=100.0, body_height=2.0, hip_width=0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = write_subject_mjcf(config, Path(directory) / "subject.xml")
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(path), enable_self_collisions=True)
            model = builder.finalize(device="cpu")
        labels = model.shape_label
        shape_by_name = {
            name: next(index for index, label in enumerate(labels) if label.endswith(f"/{name}"))
            for name in ("collision_pelvis", "collision_femur_left", "collision_tibia_left")
        }
        scales = model.shape_scale.numpy()
        np.testing.assert_allclose(
            scales[shape_by_name["collision_pelvis"]],
            0.5 * np.asarray(config.pelvis_dimensions),
            atol=1.0e-6,
        )
        self.assertAlmostEqual(scales[shape_by_name["collision_femur_left"], 0], config.thigh_radius, places=6)
        self.assertAlmostEqual(
            scales[shape_by_name["collision_femur_left"], 1],
            0.5 * (config.thigh_length - 2.0 * config.self_collision_joint_clearance),
            places=6,
        )
        self.assertAlmostEqual(scales[shape_by_name["collision_tibia_left"], 0], config.shank_radius, places=6)
        self.assertAlmostEqual(
            scales[shape_by_name["collision_tibia_left"], 1],
            0.5 * (config.shank_length - 2.0 * config.self_collision_joint_clearance),
            places=6,
        )
        self.assertAlmostEqual(float(np.sum(model.body_mass.numpy())), 100.0, places=4)

    def test_imports_self_collision_proxies_and_neighbor_filters(self):
        """Import invisible segment proxies and preserve adjacent-link filters."""
        config = SimpleGaitConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = write_subject_mjcf(config, Path(directory) / "subject.xml")
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(path), enable_self_collisions=True)
            model = builder.finalize(device="cpu")
            visible_builder = newton.ModelBuilder()
            visible_builder.add_mjcf(
                str(path),
                enable_self_collisions=True,
                force_show_colliders=True,
            )
            visible_model = visible_builder.finalize(device="cpu")
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
        visible_flags = visible_model.shape_flags.numpy()
        for label, flag in zip(visible_model.shape_label, visible_flags, strict=True):
            if label.rsplit("/", 1)[-1].startswith("collision_"):
                self.assertTrue(flag & newton.ShapeFlags.VISIBLE)
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
