# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test anatomical foot contact spheres and standing ground registration."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
from projects.gait_c3d.foot_contact import (
    FOOT_SPHERE_LAYOUT,
    RADIUS_BOUNDS,
    SolePlane,
    fit_sole_plane,
    foot_axes_from_bounds,
    foot_sphere_radius,
    place_foot_spheres,
    standing_foot_poses,
)
from projects.gait_c3d.native_model import SimpleGaitConfig
from projects.gait_c3d.subject_mjcf import write_subject_mjcf
from projects.gait_c3d.vtp_adapter import simple_gait_body_transforms


def _pose(pitch_deg: float, roll_deg: float, height: float) -> np.ndarray:
    """Build a foot body transform tilted in pitch and roll at a given height."""
    pitch, roll = math.radians(pitch_deg), math.radians(roll_deg)
    rotation_y = np.asarray(
        ((math.cos(pitch), 0.0, math.sin(pitch)), (0.0, 1.0, 0.0), (-math.sin(pitch), 0.0, math.cos(pitch)))
    )
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, math.cos(roll), -math.sin(roll)), (0.0, math.sin(roll), math.cos(roll)))
    )
    pose = np.eye(4)
    pose[:3, :3] = rotation_y @ rotation_x
    pose[:3, 3] = (0.0, 0.0, height)
    return pose


class TestGaitSolePlane(unittest.TestCase):
    """Test the standing sole plane fitted from a static capture."""

    def test_recovers_a_known_standing_tilt_and_height(self):
        """Recover the tilt and the height of a tilted standing foot."""
        sagittal = fit_sole_plane(np.stack([_pose(-5.0, 0.0, 0.14)] * 8))
        self.assertAlmostEqual(sagittal.tilt_degrees()[0], -5.0, places=9)
        self.assertAlmostEqual(sagittal.offset, -0.14, places=9)
        self.assertAlmostEqual(sagittal.residual, 0.0, places=9)
        self.assertEqual(sagittal.samples, 8)
        frontal = fit_sole_plane(np.stack([_pose(0.0, 3.0, 0.14)] * 8))
        self.assertAlmostEqual(frontal.tilt_degrees()[1], 3.0, places=9)

    def test_measures_height_above_and_projects_onto_the_plane(self):
        """Report foot-frame heights above the plane and project points onto it."""
        plane = fit_sole_plane(np.stack([_pose(0.0, 0.0, 0.12)] * 4))
        np.testing.assert_allclose(plane.height(np.zeros(3)), 0.12)
        np.testing.assert_allclose(plane.project(np.zeros(3)), (0.0, 0.0, -0.12), atol=1.0e-12)

    def test_levels_frontal_tilt_and_keeps_sagittal_tilt(self):
        """Remove the roll of a standing plane without disturbing its pitch."""
        rolled = fit_sole_plane(np.stack([_pose(0.0, 6.0, 0.14)] * 4))
        leveled = rolled.level_roll()
        self.assertAlmostEqual(leveled.tilt_degrees()[1], 0.0, places=9)
        self.assertAlmostEqual(leveled.height(np.zeros(3)), 0.14 * math.cos(math.radians(6.0)), places=9)
        pitched = fit_sole_plane(np.stack([_pose(-5.0, 6.0, 0.14)] * 4)).level_roll()
        self.assertAlmostEqual(pitched.tilt_degrees()[1], 0.0, places=9)
        self.assertAlmostEqual(pitched.tilt_degrees()[0], -5.0, delta=0.05)

    def test_recovers_foot_poses_from_measured_markers(self):
        """Fit the standing foot pose from model sites and measured markers."""
        sites = np.asarray(((-0.2, 0.0, 0.0), (0.0, 0.03, -0.02), (0.02, -0.03, -0.02), (-0.1, 0.04, 0.01)))
        pose = _pose(-4.0, 2.0, 0.13)
        measured = np.stack([sites @ pose[:3, :3].T + pose[:3, 3]] * 5)
        poses = standing_foot_poses(sites, measured)
        np.testing.assert_allclose(poses[0], pose, atol=1.0e-9)

    def test_skips_samples_without_three_visible_markers(self):
        """Drop standing samples that cannot define a rigid foot pose."""
        sites = np.asarray(((-0.2, 0.0, 0.0), (0.0, 0.03, -0.02), (0.02, -0.03, -0.02)))
        pose = _pose(0.0, 0.0, 0.13)
        measured = np.stack([sites @ pose[:3, :3].T + pose[:3, 3]] * 3)
        valid = np.ones((3, 3), dtype=bool)
        valid[1, 0] = False
        self.assertEqual(len(standing_foot_poses(sites, measured, valid)), 2)

    def test_rejects_a_normal_without_a_vertical_component(self):
        """Refuse to level a plane that is parallel to the vertical axis."""
        with self.assertRaisesRegex(ValueError, "without a vertical component"):
            SolePlane(np.asarray((0.0, 1.0, 0.0)), 0.1, 4, 0.0).level_roll()


class TestGaitFootContactSpheres(unittest.TestCase):
    """Test the anatomical sphere layout placed on a fitted sole plane."""

    @staticmethod
    def _spheres(side: str = "left", pitch: float = -5.0, length: float = 0.279, radius=None):
        """Place one foot's spheres on a tilted standing plane."""
        plane = fit_sole_plane(np.stack([_pose(pitch, 0.0, 0.12)] * 4))
        origin, forward, lateral, _ = foot_axes_from_bounds(
            side, (-0.5 * length, -0.05, -0.12), (0.5 * length, 0.05, 0.02)
        )
        return plane, place_foot_spheres(
            side,
            plane,
            origin=origin,
            forward=forward,
            lateral=lateral,
            foot_length=length,
            toe_body=f"foot_{side}",
            foot_body=f"foot_{side}",
            radius=radius,
        )

    def test_places_every_sphere_one_radius_above_the_sole_plane(self):
        """Rest all six spheres exactly on the fitted standing plane."""
        plane, spheres = self._spheres()
        self.assertEqual(len(spheres), len(FOOT_SPHERE_LAYOUT))
        heights = plane.height(np.asarray([sphere.center for sphere in spheres]))
        np.testing.assert_allclose(heights, spheres[0].radius, atol=1.0e-12)
        self.assertEqual([sphere.name for sphere in spheres], [name for name, _, _, _ in FOOT_SPHERE_LAYOUT])

    def test_keeps_spheres_from_overlapping(self):
        """Choose a radius no larger than half the closest landmark spacing."""
        _, spheres = self._spheres()
        centers = np.asarray([sphere.center for sphere in spheres])
        spacing = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        closest = float(np.min(spacing[np.triu_indices(len(spheres), 1)]))
        self.assertGreater(closest, 2.0 * spheres[0].radius)

    def test_mirrors_the_lateral_landmarks_between_feet(self):
        """Point the lateral landmarks of each foot away from the other foot."""
        _, left = self._spheres("left")
        _, right = self._spheres("right")
        for one, other in zip(left, right, strict=True):
            self.assertAlmostEqual(one.center[0], other.center[0], places=9)
            self.assertAlmostEqual(one.center[1], -other.center[1], places=9)

    def test_scales_the_radius_with_foot_length(self):
        """Follow the published radius rule inside the accepted bounds."""
        self.assertAlmostEqual(foot_sphere_radius(0.2), 0.026, places=9)
        self.assertAlmostEqual(foot_sphere_radius(0.1), RADIUS_BOUNDS[0], places=9)
        self.assertAlmostEqual(foot_sphere_radius(0.4), RADIUS_BOUNDS[1], places=9)
        self.assertAlmostEqual(foot_sphere_radius(0.3, spacing=0.04), 0.019, places=9)

    def test_rejects_a_degenerate_foot_box(self):
        """Refuse foot bounds that do not describe a real box."""
        with self.assertRaisesRegex(ValueError, "nondegenerate box"):
            foot_axes_from_bounds("left", (0.0, 0.0, 0.0), (0.0, 0.1, 0.1))


class TestGaitFootBodyRegistration(unittest.TestCase):
    """Test that the written MJCF agrees with the neutral body registration."""

    def test_foot_body_matches_the_neutral_body_transform(self):
        """Place the foot body at its declared neutral height.

        The foot height must not follow the contact sphere radius. When it did,
        the whole model floated by the difference between the configured
        contact radius and the mesh-derived sphere radius.
        """
        config = SimpleGaitConfig()
        centers = dict.fromkeys(
            ("left", "right"),
            tuple((0.05 * index - 0.1, 0.0, -0.09) for index in range(len(FOOT_SPHERE_LAYOUT))),
        )
        transforms = simple_gait_body_transforms(config)
        with tempfile.TemporaryDirectory() as directory:
            path = write_subject_mjcf(
                config,
                Path(directory) / "subject.xml",
                contact_centers=centers,
                contact_radius=0.5 * config.contact_radius,
            )
            newton.use_coord_layout_targets = True
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(path), floating=True, parse_sites=True)
            model = builder.finalize(device="cpu")
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            body_q = state.body_q.numpy()
        for side in ("left", "right"):
            index = model.body_label.index(next(label for label in model.body_label if label.endswith(f"foot_{side}")))
            self.assertAlmostEqual(float(body_q[index][2]), float(transforms[f"foot_{side}"][2, 3]), places=6)
        spheres = [index for index, label in enumerate(model.shape_label) if "/contact_" in label]
        self.assertEqual(len(spheres), 2 * len(FOOT_SPHERE_LAYOUT))


if __name__ == "__main__":
    unittest.main()
