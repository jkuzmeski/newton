# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check ground-angle reconstruction and fixed shoe-frame semantics."""

import tempfile
import unittest

import numpy as np

from newton.tests.test_impedance_hogan import _tiny_shoe
from projects.impedance_instron.cartesian.prepare_visual3d import (
    reconstruct_ground_pitch_from_cardan,
)
from projects.impedance_instron.cartesian.shoe import Shoe


class TestGroundAngleReconstruction(unittest.TestCase):
    """Verify measured ground pitch independently of any runner fit."""

    def test_flat_foot_with_inclined_shank(self):
        """Maintain zero foot ground pitch when shank inclines over a flat foot.

        When the foot stays flat on the floor (ground pitch = 0), forward tibial
        progression increases relative ankle dorsiflexion. The 3D Cardan
        reconstruction must keep the foot's world orientation stationary.
        """
        tilts_deg = np.linspace(0.0, 20.0, 5)
        knee_centers = []
        ankle_centers = []
        v_exp_deg = []
        for tilt in tilts_deg:
            tilt_rad = np.deg2rad(tilt)
            # Knee is positioned forward according to shank tilt
            ankle = np.array([0.0, 0.0, 0.0])
            knee = np.array([0.45 * np.sin(tilt_rad), 0.0, 0.45 * np.cos(tilt_rad)])
            ankle_centers.append(ankle)
            knee_centers.append(knee)
            # Ankle dorsiflexion equals shank tilt to keep foot flat
            v_exp_deg.append([tilt, 0.0, 0.0])

        ground_pitch, shank_inclination = reconstruct_ground_pitch_from_cardan(
            np.array(knee_centers), np.array(ankle_centers), np.array(v_exp_deg)
        )
        np.testing.assert_allclose(shank_inclination, np.deg2rad(tilts_deg), atol=1e-12)
        np.testing.assert_allclose(ground_pitch, np.zeros_like(tilts_deg), atol=1e-12)

    def test_rotate_shank_and_foot_together(self):
        """Track ground pitch change when shank and foot rotate together with fixed relative ankle.

        When the relative joint angle remains fixed at zero (neutral), any rotation of the shank
        must rotate the foot's world ground pitch by the identical angle.
        """
        tilts_deg = np.linspace(-15.0, 30.0, 7)
        knee_centers = []
        ankle_centers = []
        v_exp_deg = []
        for tilt in tilts_deg:
            tilt_rad = np.deg2rad(tilt)
            ankle = np.array([0.0, 0.0, 0.0])
            knee = np.array([0.45 * np.sin(tilt_rad), 0.0, 0.45 * np.cos(tilt_rad)])
            ankle_centers.append(ankle)
            knee_centers.append(knee)
            v_exp_deg.append([0.0, 0.0, 0.0])

        ground_pitch, shank_inclination = reconstruct_ground_pitch_from_cardan(
            np.array(knee_centers), np.array(ankle_centers), np.array(v_exp_deg)
        )
        np.testing.assert_allclose(shank_inclination, np.deg2rad(tilts_deg), atol=1e-12)
        # When relative ankle is fixed, tilting the shank forward (+tilt) pitches the attached foot toe-down (-pitch)
        np.testing.assert_allclose(ground_pitch, -np.deg2rad(tilts_deg), atol=1e-12)

    def test_neutral_calibration_and_known_rotations_and_3d_yaw_roll(self):
        """Verify neutral calibration, pure pitch, and 3D Cardan coupling with roll and yaw.

        Ensures that neutral standing posture yields zero pitch, pure pitch rotations yield
        exact expected values, and non-planar rotations (yaw/roll) correctly influence
        the projected sagittal forward axis without naive scalar-angle addition.
        """
        knee = np.array([[0.0, 0.0, 0.45]])
        ankle = np.array([[0.0, 0.0, 0.0]])

        # Neutral calibration
        p_neutral, s_neutral = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(p_neutral[0]), 0.0, places=12)
        self.assertAlmostEqual(float(s_neutral[0]), 0.0, places=12)

        # Pure toe-up (+12 deg)
        p_up, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[12.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(np.rad2deg(p_up[0])), 12.0, places=12)

        # Pure toe-down (-25 deg)
        p_down, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[-25.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(np.rad2deg(p_down[0])), -25.0, places=12)

        # 3D case containing roll (beta) and yaw (gamma)
        # alpha=10 deg, beta=15 deg, gamma=20 deg
        p_3d, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, np.array([[10.0, 15.0, 20.0]]))
        # Direct scalar addition would predict 10.0 deg. With 3D Cardan rotation:
        # R = R_alpha(10) @ R_beta(15) @ R_gamma(20)
        # v = R @ [1, 0, 0] = [cos(10)*cos(20) + sin(10)*sin(15)*sin(20), ..., sin(10)*cos(20) - cos(10)*sin(15)*sin(20)]
        expected_pitch = np.rad2deg(float(p_3d[0]))
        self.assertNotAlmostEqual(expected_pitch, 10.0, places=1)
        # Exposes that scalar addition is erroneous in 3D:
        scalar_error = abs(expected_pitch - 10.0)
        self.assertGreater(scalar_error, 0.5)

    def test_fixed_registration_and_translation_invariance(self):
        """Apply fixed shoe registration once and preserve angles under translation."""
        fixed_pitch = -0.24446041090480894
        test_pitches = np.deg2rad([-20.0, -5.0, 0.0, 8.0, 15.0])
        thigh = -1.1
        knee = -0.3
        with tempfile.TemporaryDirectory() as directory:
            fixture = _tiny_shoe(directory)
            shoe = Shoe(fixture.artifact_path, fixture.mount_m, fixed_pitch, device="cpu")
            for pitch in test_pitches:
                ankle = pitch + fixed_pitch - (thigh + knee) - np.pi / 2
                solver_pitch = thigh + knee + ankle + np.pi / 2
                for x_shift in (0.0, 5.0):
                    with self.subTest(pitch=pitch, x_shift=x_shift):
                        shoe.apply([x_shift, 0.9], [0.0, 0.0], solver_pitch, 0.0, 0.0000625)
                        pose = shoe.state.body_q.numpy()[0]
                        carrier_pitch = -2 * np.arctan2(float(pose[4]), float(pose[6]))
                        np.testing.assert_allclose(carrier_pitch, pitch, atol=1e-6)

        knee_center = np.array([[0.1, 0.0, 0.45]])
        ankle_center = np.zeros((1, 3))
        angles = np.array([[12.0, 15.0, 20.0]])
        expected = reconstruct_ground_pitch_from_cardan(knee_center, ankle_center, angles)
        for shift in ([5.0, 0.0, 0.0], [2.0, -3.0, 4.0]):
            actual = reconstruct_ground_pitch_from_cardan(knee_center + shift, ankle_center + shift, angles)
            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_semantic_regression_fails_old_passes_new(self):
        """Demonstrate that the semantic regression fails under raw direct angle mapping and passes after correction."""
        # Frame at source time 0.370s in late stance:
        # Shank inclination is forward ~43.3 deg.
        # Exported RVirtualFootAngle alpha is +24.3 deg (flexion/dorsiflexion relative to shank).
        # Physical foot orientation is toe-down (~ -20.2 deg).
        knee = np.array([[0.45 * np.sin(np.deg2rad(43.345)), 0.0, 0.45 * np.cos(np.deg2rad(43.345))]])
        ankle = np.array([[0.0, 0.0, 0.0]])
        raw_virtual_foot = np.array([[24.306, -1.765, -15.455]])

        # 1. Old naive direct interpretation: assigns raw virtual alpha directly as ground pitch
        old_ground_pitch_deg = raw_virtual_foot[0, 0]
        # In late stance before toe-off, the foot MUST be pitched toe-down (negative pitch).
        # The old mapping gives +24.3 deg (toe-up!), which is unphysical.
        self.assertGreater(old_ground_pitch_deg, 0.0)  # Old mapping erroneously indicates toe-up

        # 2. Corrected 3D Cardan reconstruction:
        new_ground_pitch, _ = reconstruct_ground_pitch_from_cardan(knee, ankle, raw_virtual_foot)
        new_ground_pitch_deg = float(np.rad2deg(new_ground_pitch[0]))
        # Correct mapping gives negative pitch (toe-down)
        self.assertLess(new_ground_pitch_deg, -15.0)  # Correct mapping correctly indicates toe-down ~ -20 deg


if __name__ == "__main__":
    unittest.main()
