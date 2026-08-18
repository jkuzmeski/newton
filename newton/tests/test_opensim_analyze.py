# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenSim ``Analyze``-style report tools (BodyKinematics,
MuscleAnalysis, JointReaction) of the Newton OpenSim port."""

import unittest

import numpy as np

import newton.examples
from newton._src.opensim.analyze import (
    BodyKinematics,
    JointReaction,
    MuscleAnalysis,
    _differentiated_coordinates,
    euler_xyz_from_matrix,
)
from newton._src.opensim.kinematics import ForwardKinematics, euler_xyz_to_matrix
from newton._src.opensim.mocap import Storage
from newton._src.opensim.parser import parse_osim
from newton._src.opensim.visualize import read_motion

# A minimal self-contained OpenSim 4.x model: a single pendulum body hanging from
# ground via a PinJoint, with one muscle spanning ground -> rod. Used for exact
# numeric assertions on MuscleAnalysis and JointReaction.
_PENDULUM_MUSCLE = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <Model name="pendulum">
        <gravity>0 -9.80665 0</gravity>
        <BodySet name="bodyset">
            <objects>
                <Body name="rod">
                    <mass>2.0</mass>
                    <mass_center>0 -0.5 0</mass_center>
                    <inertia>0.1 0.01 0.1 0 0 0</inertia>
                </Body>
            </objects>
        </BodySet>
        <JointSet name="jointset">
            <objects>
                <PinJoint name="pin">
                    <socket_parent_frame>ground_offset</socket_parent_frame>
                    <socket_child_frame>rod_offset</socket_child_frame>
                    <coordinates>
                        <objects>
                            <Coordinate name="pin_angle">
                                <default_value>0.0</default_value>
                                <range>-3.14 3.14</range>
                                <clamped>true</clamped>
                            </Coordinate>
                        </objects>
                    </coordinates>
                    <frames>
                        <PhysicalOffsetFrame name="ground_offset">
                            <socket_parent>/ground</socket_parent>
                            <translation>0 1.0 0</translation>
                            <orientation>0 0 0</orientation>
                        </PhysicalOffsetFrame>
                        <PhysicalOffsetFrame name="rod_offset">
                            <socket_parent>/bodyset/rod</socket_parent>
                            <translation>0 0 0</translation>
                            <orientation>0 0 0</orientation>
                        </PhysicalOffsetFrame>
                    </frames>
                </PinJoint>
            </objects>
        </JointSet>
        <ForceSet name="forceset">
            <objects>
                <DeGrooteFregly2016Muscle name="m0">
                    <min_control>0.01</min_control>
                    <max_control>1</max_control>
                    <GeometryPath>
                        <PathPointSet>
                            <objects>
                                <PathPoint name="m0-P1">
                                    <location>0 1.0 0</location>
                                    <socket_parent_frame>/ground</socket_parent_frame>
                                </PathPoint>
                                <PathPoint name="m0-P2">
                                    <location>0 -0.2 0</location>
                                    <socket_parent_frame>/bodyset/rod</socket_parent_frame>
                                </PathPoint>
                            </objects>
                        </PathPointSet>
                    </GeometryPath>
                    <max_isometric_force>500</max_isometric_force>
                    <optimal_fiber_length>0.12</optimal_fiber_length>
                    <tendon_slack_length>0.15</tendon_slack_length>
                    <pennation_angle_at_optimal>0.1</pennation_angle_at_optimal>
                    <max_contraction_velocity>10</max_contraction_velocity>
                </DeGrooteFregly2016Muscle>
            </objects>
        </ForceSet>
    </Model>
</OpenSimDocument>
"""


def _static_motion(coordinate: str, value_deg: float, n: int = 30, dt: float = 0.01) -> Storage:
    """Return a constant-coordinate motion :class:`Storage` for static-hold tests."""
    times = np.arange(n) * dt
    data = np.full((n, 1), value_deg)
    return Storage(times=times, labels=[coordinate], data=data, in_degrees=True, name="static")


class TestBodyKinematics(unittest.TestCase):
    """Verify BodyKinematics reproduces the forward-kinematics body poses."""

    @classmethod
    def setUpClass(cls):
        cls.model = parse_osim(newton.examples.get_asset("gait2354_subject01.osim"))
        cls.mot = newton.examples.get_asset("gait2354_subject01_walk.mot")
        cls.bk = BodyKinematics(cls.model)
        cls.pos = cls.bk.solve(cls.mot)

    def test_pelvis_height_physical_and_finite(self):
        """Report a physically plausible, finite pelvis height over the walk."""
        pelvis_y = self.pos.column("pelvis_Y")
        self.assertTrue(np.all(np.isfinite(self.pos.data)))
        self.assertGreater(pelvis_y.min(), 0.7)
        self.assertLess(pelvis_y.max(), 1.3)

    def test_origin_matches_forward_kinematics(self):
        """Body origin columns match body_transforms_batch and COM matches center_of_mass_batch."""
        _, coords = read_motion(self.model, self.mot)
        fk = ForwardKinematics(self.model)
        transforms = fk.body_transforms_batch(coords)
        com = fk.center_of_mass_batch(coords)
        for name in ("pelvis", "femur_r", "tibia_l", "torso"):
            b = fk.body_names.index(name)
            for a, axis in enumerate("XYZ"):
                np.testing.assert_allclose(self.pos.column(f"{name}_{axis}"), transforms[:, b, a, 3], atol=1e-9)
        for a, axis in enumerate("XYZ"):
            np.testing.assert_allclose(self.pos.column(f"center_of_mass_{axis}"), com[:, a], atol=1e-9)

    def test_orientation_columns_reconstruct_rotation(self):
        """Reported XYZ Euler orientation columns rebuild each body's rotation matrix."""
        _, coords = read_motion(self.model, self.mot)
        fk = ForwardKinematics(self.model)
        transforms = fk.body_transforms_batch(coords)
        b = fk.body_names.index("femur_r")
        ang = np.deg2rad(
            np.stack(
                [self.pos.column("femur_r_Ox"), self.pos.column("femur_r_Oy"), self.pos.column("femur_r_Oz")], axis=1
            )
        )
        for f in (0, 50, 120):
            np.testing.assert_allclose(euler_xyz_to_matrix(*ang[f]), transforms[f, b, :3, :3], atol=1e-9)

    def test_euler_roundtrip(self):
        """euler_xyz_from_matrix inverts euler_xyz_to_matrix for generic angles."""
        for angs in [(0.1, -0.4, 0.9), (1.2, 0.05, -0.7), (-0.3, 0.0, 0.2)]:
            rec = euler_xyz_from_matrix(euler_xyz_to_matrix(*angs))
            np.testing.assert_allclose(rec, angs, atol=1e-9)

    def test_velocities_and_accelerations_finite(self):
        """Velocity and acceleration reports are finite with the full column layout."""
        vel = self.bk.solve_velocities(self.mot)
        acc = self.bk.solve_accelerations(self.mot)
        self.assertEqual(len(vel.labels), len(self.pos.labels))
        self.assertEqual(len(acc.labels), len(self.pos.labels))
        self.assertTrue(np.all(np.isfinite(vel.data)))
        self.assertTrue(np.all(np.isfinite(acc.data)))

    def test_compound_reports_keep_kinematics_on_device(self):
        """Keep shared body poses and rates on device until each report is copied."""
        bk = BodyKinematics(self.model)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("BodyKinematics called a host-returning kinematics wrapper")

        bk.fk.body_transforms_batch = reject_host_wrapper
        bk.fk.center_of_mass_batch = reject_host_wrapper
        bk.fk.body_velocities_batch = reject_host_wrapper
        bk.fk.body_accelerations_batch = reject_host_wrapper
        self.assertTrue(np.all(np.isfinite(bk.solve(self.mot).data)))
        self.assertTrue(np.all(np.isfinite(bk.solve_velocities(self.mot).data)))
        self.assertTrue(np.all(np.isfinite(bk.solve_accelerations(self.mot).data)))


class TestMuscleAnalysis(unittest.TestCase):
    """Verify MuscleAnalysis quantities match the muscle path/force primitives."""

    @classmethod
    def setUpClass(cls):
        cls.gait_model = parse_osim(newton.examples.get_asset("gait2354_subject01.osim"))
        cls.gait_mot = newton.examples.get_asset("gait2354_subject01_walk.mot")
        cls.pend_model = parse_osim(_PENDULUM_MUSCLE)

    def test_length_equals_muscle_paths(self):
        """Reported Length equals MusclePaths.lengths at the analysis coordinates."""
        ma = MuscleAnalysis(self.pend_model)
        motion = _static_motion("pin_angle", 20.0)
        res = ma.solve(motion, activations=0.5, cutoff=-1.0, include_speeds=False)
        _, q, _, _ = _differentiated_coordinates(
            self.pend_model, ma.coordinate_names, ma.motion_types, motion, -1.0, None
        )
        np.testing.assert_allclose(res["Length"].data, ma.paths.lengths(q), atol=1e-9)
        self.assertEqual(res["Length"].labels, ma.muscle_names)

    def test_tendon_force_nonnegative_and_finite(self):
        """Tendon force is nonnegative and every muscle quantity is finite (gait2354)."""
        ma = MuscleAnalysis(self.gait_model)
        res = ma.solve(self.gait_mot, activations=0.1)
        self.assertTrue((res["TendonForce"].data >= -1e-6).all())
        for storage in res.values():
            self.assertTrue(np.all(np.isfinite(storage.data)))
            self.assertEqual(storage.labels, ma.muscle_names)

    def test_fiber_force_decomposition(self):
        """FiberForce equals ActiveFiberForce + PassiveFiberForce (gait2354)."""
        ma = MuscleAnalysis(self.gait_model)
        res = ma.solve(self.gait_mot, activations=0.3)
        np.testing.assert_allclose(
            res["FiberForce"].data,
            res["ActiveFiberForce"].data + res["PassiveFiberForce"].data,
            atol=1e-4,
        )

    def test_moment_arm_tables_present_per_coordinate(self):
        """A MomentArm_<coordinate> table exists for every coordinate, columns are muscles."""
        ma = MuscleAnalysis(self.gait_model)
        res = ma.solve(self.gait_mot, activations=0.1)
        for coord in ma.coordinate_names:
            key = f"MomentArm_{coord}"
            self.assertIn(key, res)
            self.assertEqual(res[key].labels, ma.muscle_names)


class TestJointReaction(unittest.TestCase):
    """Verify JointReaction reproduces the analytic pin reaction and runs on gait2354."""

    @classmethod
    def setUpClass(cls):
        cls.pend_model = parse_osim(_PENDULUM_MUSCLE)
        cls.gait_model = parse_osim(newton.examples.get_asset("gait2354_subject01.osim"))
        cls.gait_mot = newton.examples.get_asset("gait2354_subject01_walk.mot")

    def test_pendulum_static_reaction_is_mg(self):
        """A statically held pendulum's pin reaction force magnitude equals m*g."""
        jr = JointReaction(self.pend_model)
        motion = _static_motion("pin_angle", 0.0)
        sto = jr.solve(motion, cutoff=-1.0)
        mid = sto.data.shape[0] // 2
        force = sto.data[mid, :3]
        moment = sto.data[mid, 3:6]
        mg = 2.0 * 9.80665
        self.assertAlmostEqual(np.linalg.norm(force), mg, places=3)
        np.testing.assert_allclose(force, [0.0, mg, 0.0], atol=1e-3)
        np.testing.assert_allclose(moment, [0.0, 0.0, 0.0], atol=1e-3)

    def test_pendulum_reaction_magnitude_frame_invariant(self):
        """The reaction force magnitude is the same expressed in ground/child/parent."""
        jr = JointReaction(self.pend_model)
        motion = _static_motion("pin_angle", 25.0)
        mid = motion.data.shape[0] // 2
        mags = []
        for frame in ("ground", "child", "parent"):
            sto = jr.solve(motion, cutoff=-1.0, express_in=frame)
            mags.append(np.linalg.norm(sto.data[mid, :3]))
        for m in mags:
            self.assertAlmostEqual(m, 2.0 * 9.80665, places=3)

    def test_gait2354_smoke_finite_and_labels(self):
        """gait2354 joint reactions are finite with correct per-joint column labels."""
        jr = JointReaction(self.gait_model)
        sto = jr.solve(self.gait_mot, activations=0.1)
        self.assertEqual(len(sto.labels), 6 * len(jr.joint_names))
        self.assertTrue(np.all(np.isfinite(sto.data)))
        stem = "ground_pelvis_on_pelvis_in_ground"
        for suffix in ("_fx", "_fy", "_fz", "_mx", "_my", "_mz"):
            self.assertIn(stem + suffix, sto.labels)

    def test_muscle_load_pipeline_stays_on_device(self):
        """Keep muscle path-point loads and joint reduction on device until the report is copied."""
        jr = JointReaction(self.gait_model)
        activations = np.full(len(self.gait_model.muscles), 0.1)
        expected = jr.solve(self.gait_mot, activations=activations)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("JointReaction called a host-returning muscle-force wrapper")

        jr.muscles.forces = reject_host_wrapper
        actual = jr.solve(self.gait_mot, activations=activations)
        np.testing.assert_allclose(actual.data, expected.data, atol=1.0e-9)

    def test_gait2354_runs_without_activations(self):
        """Joint reactions are finite when muscle activations are omitted (muscles zero)."""
        jr = JointReaction(self.gait_model)
        sto = jr.solve(self.gait_mot)
        self.assertTrue(np.all(np.isfinite(sto.data)))


if __name__ == "__main__":
    unittest.main()
