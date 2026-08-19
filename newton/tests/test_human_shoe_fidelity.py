# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact OpenSim pose-fidelity diagnostics."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton import opensim
from projects.human_shoe.fidelity import compare_imported_state
from projects.human_shoe.landing import Example


class TestHumanShoePoseFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build one imported Gait2354 example shared by the fidelity tests."""
        cls.example = Example(newton.viewer.ViewerNull(num_frames=1), Example.create_parser().parse_args([]))

    def test_current_d6_import_is_characterized_as_approximate(self):
        """Expose the Gait2354 CustomJoint error instead of accepting it silently."""
        example = self.example
        report = example.pose_fidelity
        errors = {error.body_name: error for error in report.body_errors}

        self.assertFalse(report.within())
        self.assertGreater(report.max_position_m, 0.4)
        self.assertGreater(report.max_orientation_deg, 15.0)
        self.assertGreater(errors["calcn_r"].position_m, 0.4)
        self.assertGreater(errors["calcn_r"].orientation_deg, 15.0)
        self.assertAlmostEqual(errors["pelvis"].position_m, 0.0, places=6)
        self.assertAlmostEqual(errors["pelvis"].orientation_deg, 0.0, places=4)

    def test_full_motion_speed_mapping_exposes_velocity_error(self):
        """Characterize velocity drift when every mapped gait speed drives the approximate D6 model."""
        example = self.example
        times, coordinates = opensim.read_motion(example.osim_model, example.motion_path)
        speeds = np.gradient(coordinates, times, axis=0, edge_order=1)
        frame = example.experiment.initial_motion_frame
        joint_q = example.state_0.joint_q.numpy()
        joint_qd = np.zeros_like(example.state_0.joint_qd.numpy())
        coordinate_names = [coordinate.name for joint in example.osim_model.joints for coordinate in joint.coordinates]
        for column, name in enumerate(coordinate_names):
            target = example.import_result.coordinate_dof.get(name)
            if target is not None:
                joint_qd[target] = speeds[frame, column]
        state = example.model.state()
        state.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=example.device))
        state.joint_qd.assign(wp.array(joint_qd, dtype=wp.float32, device=example.device))
        newton.eval_fk(example.model, state.joint_q, state.joint_qd, state)
        source_coordinates = coordinates[frame].copy()
        source_coordinates[coordinate_names.index("pelvis_ty")] = joint_q[
            example.import_result.coordinate_dof["pelvis_ty"]
        ]
        report = compare_imported_state(
            example.model,
            state,
            example.osim_model,
            source_coordinates,
            speeds[frame],
            device=example.device,
        )

        self.assertGreater(report.max_linear_velocity_m_s, 4.0)
        self.assertGreater(report.max_angular_velocity_rad_s, 6.0)

    def test_exact_body_state_satisfies_fidelity_limits(self):
        """Accept exact Z-up OpenSim poses and velocities after root alignment."""
        example = self.example
        coordinates = example._initial_osim_coordinates
        speeds = example._initial_osim_speeds
        visualizer = opensim.MotionVisualizer(example.osim_model, coordinates[None, :], device=example.device)
        exact_pose = visualizer.body_transforms(example.model.body_label)[0]

        fk = opensim.ForwardKinematics(example.osim_model, device=example.device)
        velocity = fk.body_velocities_batch(coordinates[None, :], speeds[None, :])
        basis = opensim.OsimFrameConverter().matrix
        angular = velocity["angular_velocity"][0] @ basis.T
        origin_linear = velocity["linear_velocity"][0] @ basis.T
        exact_index = {name: index for index, name in enumerate(fk.body_names)}
        pose_np = exact_pose.numpy()
        com = example.model.body_com.numpy()
        twist = np.zeros((example.model.body_count, 6), dtype=np.float32)
        for body, name in enumerate(example.model.body_label):
            source = exact_index[name]
            rotation = wp.quat_to_matrix(wp.quat(*pose_np[body, 3:]))
            com_offset = np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ com[body]
            twist[body, :3] = origin_linear[source] + np.cross(angular[source], com_offset)
            twist[body, 3:] = angular[source]

        state = example.model.state()
        state.body_q.assign(exact_pose)
        state.body_qd.assign(wp.array(twist, dtype=wp.spatial_vector, device=example.device))
        report = compare_imported_state(
            example.model,
            state,
            example.osim_model,
            coordinates,
            speeds,
            device=example.device,
        )

        self.assertTrue(
            report.within(
                position_m=1.0e-5, orientation_deg=0.01, linear_velocity_m_s=1.0e-4, angular_velocity_rad_s=1.0e-4
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
