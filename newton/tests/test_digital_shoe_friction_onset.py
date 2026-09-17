# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify onset kinematics, velocity decomposition, and friction replay diagnostic."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.digital_shoe.friction_onset import (
    compute_reference_derived_kinematics,
    decompose_tangential_velocity,
    run_friction_onset_diagnostic,
    sha256_bytes,
)
from projects.impedance_instron.cartesian.mechanics import Body


class TestDigitalShoeFrictionOnset(unittest.TestCase):
    """Test velocity decomposition, Hermite spline kinematics, and frozen-normal replay."""

    def setUp(self):
        """Build synthetic leg body model and test states."""
        self.lengths = np.array([0.5, 0.4], dtype=np.float64)
        self.endpoint = np.array([0.15, -0.04], dtype=np.float64)
        self.masses = np.array([10.0, 4.0, 1.5], dtype=np.float64)
        self.com = np.array([[0.2, 0.0], [0.15, 0.0], [0.05, 0.0]], dtype=np.float64)
        self.inertias = np.array([0.15, 0.08, 0.01], dtype=np.float64)
        self.body = Body(self.lengths, self.endpoint, self.masses, self.com, self.inertias)

    def test_pure_hip_translation_mapping(self):
        """Verify pure hip translation produces equal ankle velocity and zero rotation."""
        q = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v = np.array([2.5, -0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        dec = decompose_tangential_velocity(self.body, q, v, ground_height_m=0.0)

        self.assertAlmostEqual(dec["hip_vel_m_s"][0], 2.5, places=9)
        self.assertAlmostEqual(dec["hip_vel_m_s"][1], -0.5, places=9)
        self.assertAlmostEqual(dec["ankle_vel_m_s"][0], 2.5, places=9)
        self.assertAlmostEqual(dec["leg_rel_vel_x_m_s"], 0.0, places=9)
        self.assertAlmostEqual(dec["omega_foot_rad_s"], 0.0, places=9)
        self.assertAlmostEqual(dec["foot_rot_contrib_x_m_s"], 0.0, places=9)
        self.assertAlmostEqual(dec["tangential_plane_vel_x_m_s"], 2.5, places=9)

    def test_pure_foot_rotation_lever_arm(self):
        """Verify foot rotation lever arm contributes omega * ankle_z to plane velocity."""
        q = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        omega = 2.0
        v = np.array([0.0, 0.0, 0.0, 0.0, omega], dtype=np.float64)
        ground_z = 0.0
        dec = decompose_tangential_velocity(self.body, q, v, ground_height_m=ground_z)

        self.assertAlmostEqual(dec["hip_vel_m_s"][0], 0.0, places=9)
        self.assertAlmostEqual(dec["ankle_vel_m_s"][0], 0.0, places=9)
        self.assertAlmostEqual(dec["omega_foot_rad_s"], omega, places=9)

        expected_lever = dec["ankle_pos_m"][1] - ground_z
        self.assertAlmostEqual(dec["lever_arm_z_m"], expected_lever, places=9)
        self.assertAlmostEqual(dec["foot_rot_contrib_x_m_s"], omega * expected_lever, places=9)
        self.assertAlmostEqual(dec["tangential_plane_vel_x_m_s"], omega * expected_lever, places=9)

    def test_hermite_spline_exact_derivatives_and_bounds(self):
        """Verify CubicHermiteSpline preserves exact knot values and rejects extrapolation."""
        t_knots = np.linspace(0.0, 0.1, 11)
        q_knots = np.zeros((11, 5), dtype=np.float64)
        v_knots = np.zeros((11, 5), dtype=np.float64)
        for i in range(11):
            q_knots[i, 0] = 0.5 + 0.1 * i
            q_knots[i, 1] = 1.0 - 0.02 * i
            v_knots[i, 0] = 1.0
            v_knots[i, 1] = -0.2

        ref_mock = {"time_s": t_knots, "state": q_knots, "velocity": v_knots}
        res = compute_reference_derived_kinematics(ref_mock, t_knots, self.body)

        # Exact knot preservation for both position and velocity (dq/dt = v)
        np.testing.assert_allclose(res["q"], q_knots, atol=1e-12)
        np.testing.assert_allclose(res["v"], v_knots, atol=1e-12)

        # Extrapolation rejection
        with self.assertRaises(ValueError):
            compute_reference_derived_kinematics(ref_mock, np.array([-0.01, 0.05]), self.body)
        with self.assertRaises(ValueError):
            compute_reference_derived_kinematics(ref_mock, np.array([0.05, 0.15]), self.body)

    def test_run_diagnostic_replay_and_nonmutation(self):
        """Verify diagnostic runs end-to-end, enforces strict requirements, and preserves inputs."""
        hist_path = Path(os.environ.get("NEWTON_FRICTION_HISTORY", "outputs/friction_identification/history_exact.npz"))
        base_dir = Path(os.environ.get("NEWTON_BASELINE12_DIR", "outputs/impedance_instron/baseline12"))

        if not hist_path.exists() or not base_dir.exists():
            self.skipTest("Sealed baseline or history_exact not found")

        with np.load(hist_path, allow_pickle=False) as data:
            normal_before = np.array(data["normal_n"], copy=True)
            normal_before_hash = sha256_bytes(normal_before.tobytes())

        with tempfile.TemporaryDirectory() as tmp_parent:
            tmp_out = Path(tmp_parent) / "new_onset_dir"

            # Rejection of existing directory
            tmp_out.mkdir()
            with self.assertRaises(FileExistsError):
                run_friction_onset_diagnostic(hist_path, base_dir, tmp_out)
            tmp_out.rmdir()

            # Rejection of missing candidate path
            with self.assertRaises(FileNotFoundError):
                run_friction_onset_diagnostic(hist_path, base_dir, tmp_out, candidate_path="nonexistent.json")

            # Successful run
            summary = run_friction_onset_diagnostic(
                history_path=hist_path,
                baseline_dir=base_dir,
                output_dir=tmp_out,
                candidate_path=None,
            )

            # Normal non-mutation assertion
            with np.load(hist_path, allow_pickle=False) as data:
                normal_after = np.array(data["normal_n"], copy=True)
                normal_after_hash = sha256_bytes(normal_after.tobytes())
            self.assertEqual(normal_before_hash, normal_after_hash)
            self.assertTrue(np.array_equal(normal_before, normal_after))

            # Initial state parity
            parity = summary["initial_state_parity"]
            self.assertTrue(parity["is_q0_exact_match"])
            self.assertTrue(parity["is_v0_exact_match"])
            self.assertEqual(parity["error_q0"], [0.0, 0.0, 0.0, 0.0, 0.0])
            self.assertEqual(parity["error_v0"], [0.0, 0.0, 0.0, 0.0, 0.0])

            # Output file verification
            json_file = tmp_out / "onset_diagnostic.json"
            npz_file = tmp_out / "curves_forces.npz"
            self.assertTrue(json_file.exists())
            self.assertTrue(npz_file.exists())

            # Structure of observer inputs: column 1 must be normal_force_z
            with np.load(npz_file) as npz:
                self.assertIn("time_s", npz)
                self.assertIn("actual_forces", npz)
                self.assertIn("ref_velocity_forces", npz)
                self.assertIn("normal_force_z", npz)
                self.assertEqual(npz["actual_forces"].ndim, 2)
                self.assertEqual(npz["actual_forces"].shape[1], 2)
                np.testing.assert_allclose(npz["actual_forces"][:, 1], npz["normal_force_z"], atol=1e-6)
                np.testing.assert_allclose(npz["ref_velocity_forces"][:, 1], npz["normal_force_z"], atol=1e-6)

            # Decompositions at required intervals
            decomps = summary["decompositions"]
            for key in ["0.0ms", "20.0ms", "30.0ms", "40.0ms"]:
                self.assertIn(key, decomps)
                diffs = decomps[key]["differences"]
                self.assertIn("delta_tangential_plane_vel_x_m_s", diffs)
                self.assertIn("delta_hip_vel_x_m_s", diffs)
                self.assertIn("delta_leg_rot_vel_x_m_s", diffs)
                self.assertIn("delta_foot_rot_contrib_x_m_s", diffs)


if __name__ == "__main__":
    unittest.main()
