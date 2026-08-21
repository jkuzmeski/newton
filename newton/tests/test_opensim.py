# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Newton-native OpenSim (.osim) port."""

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
import newton.opensim as osim
from newton._src.geometry.collision_primitive import collide_plane_sphere, collide_sphere_sphere
from newton._src.opensim import gcvspl
from newton._src.opensim import muscle as M
from newton._src.opensim.collocation import (
    DirectCollocationSolver,
    OptimalControlProblem,
    _point_covector_hessian,
    _point_jacobian,
    create_torque_driven_dynamics,
    solve_optimal_control,
)
from newton._src.opensim.contact import (
    OpenSimContact,
    _mat44d,
    _vec3d,
    elastic_foundation_kernel,
    hunt_crossley_kernel,
    smooth_sphere_halfspace_kernel,
)
from newton._src.opensim.dynamics import (
    ForwardDynamics,
    InverseDynamics,
    lowpass_iir,
    pad_signal,
    solve_forward_dynamics,
    solve_inverse_dynamics,
)
from newton._src.opensim.functions import SimmSpline, build_function
from newton._src.opensim.ik import InverseKinematics
from newton._src.opensim.kinematics import ForwardKinematics, euler_xyz_to_matrix, make_transform
from newton._src.opensim.mocap import MarkerData, read_storage, read_trc, write_storage, write_trc
from newton._src.opensim.visualize import _read_vtp, _resolve_geometry_file
from newton.viewer import ViewerNull

# A minimal self-contained OpenSim 4.x model: a single pendulum body hanging
# from ground via a PinJoint, with one muscle spanning ground -> rod.
MINIMAL_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
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
                                <default_value>0.3</default_value>
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

_WRAP_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <Model name="pendulum">
        <gravity>0 -9.80665 0</gravity>
        <BodySet name="bodyset">
            <objects>
                <Body name="rod">
                    <mass>2.0</mass>
                    <mass_center>0 -0.5 0</mass_center>
                    <inertia>0.1 0.01 0.1 0 0 0</inertia>
                    <WrapObjectSet name="wrapobjectset">
                        <objects>
                            <WrapSphere name="knee_wrap">
                                <active>true</active>
                                <translation>0 -0.3 0</translation>
                                <xyz_body_rotation>0 0 0</xyz_body_rotation>
                                <quadrant>all</quadrant>
                                <radius>0.05</radius>
                            </WrapSphere>
                        </objects>
                    </WrapObjectSet>
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
                                <default_value>0.3</default_value>
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
                        <PathWrapSet name="pathwrapset">
                            <objects>
                                <PathWrap name="pw">
                                    <wrap_object>knee_wrap</wrap_object>
                                    <method>hybrid</method>
                                    <range>-1 -1</range>
                                </PathWrap>
                            </objects>
                        </PathWrapSet>
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


class TestOsimParser(unittest.TestCase):
    def test_parse_minimal_model(self):
        """Parse a minimal 4.x model and verify bodies, joints, and gravity."""
        m = osim.parse_osim(MINIMAL_OSIM)
        self.assertEqual(m.name, "pendulum")
        self.assertEqual(m.version, 40000)
        self.assertEqual(len(m.bodies), 1)
        self.assertEqual(m.bodies[0].name, "rod")
        self.assertAlmostEqual(m.bodies[0].mass, 2.0)
        self.assertEqual(len(m.joints), 1)
        self.assertEqual(m.joints[0].type, "PinJoint")
        self.assertEqual(m.joints[0].parent_body, "ground")
        self.assertEqual(m.joints[0].child_body, "rod")
        np.testing.assert_allclose(m.gravity, (0.0, -9.80665, 0.0))

    def test_parse_joint_frames(self):
        """Verify PhysicalOffsetFrame sockets resolve to parent/child offsets."""
        m = osim.parse_osim(MINIMAL_OSIM)
        j = m.joints[0]
        np.testing.assert_allclose(j.parent_transform.translation, (0.0, 1.0, 0.0))
        np.testing.assert_allclose(j.child_transform.translation, (0.0, 0.0, 0.0))
        self.assertEqual(len(j.coordinates), 1)
        self.assertEqual(j.coordinates[0].name, "pin_angle")
        self.assertEqual(j.coordinates[0].range, (-3.14, 3.14))

    def test_parse_muscle(self):
        """Verify muscle parameters and path points are parsed."""
        m = osim.parse_osim(MINIMAL_OSIM)
        self.assertEqual(len(m.muscles), 1)
        mu = m.muscles[0]
        self.assertEqual(mu.type, "DeGrooteFregly2016Muscle")
        self.assertEqual(len(mu.path_points), 2)
        self.assertEqual(mu.path_points[0].body, "ground")
        self.assertEqual(mu.path_points[1].body, "rod")
        self.assertAlmostEqual(mu.params["max_isometric_force"], 500.0)
        self.assertAlmostEqual(mu.params["optimal_fiber_length"], 0.12)


class TestOsimImport(unittest.TestCase):
    def test_frame_converter_maps_opensim_y_up_to_newton_z_up(self):
        """Preserve handedness while mapping OpenSim +Y onto Newton +Z."""
        converter = osim.OsimFrameConverter()
        basis = converter.matrix
        vectors = np.eye(3)
        mapped = converter.transform_vectors(vectors)

        np.testing.assert_allclose(mapped[0], [1.0, 0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(mapped[1], [0.0, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(mapped[2], [0.0, -1.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(converter.inverse_vectors(mapped), vectors, atol=1.0e-6)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=6)

    def test_import_into_builder(self):
        """Import the minimal model and finalize a valid Newton Model."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        result = osim.add_osim(builder, MINIMAL_OSIM)
        self.assertIn("rod", result.body_index)
        self.assertEqual(result.body_index["ground"], -1)
        self.assertEqual(len(result.muscles), 1)

        model = builder.finalize()
        self.assertEqual(model.body_count, 1)
        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_coord_count, 1)  # one PinJoint DOF

    def test_import_converts_declared_root_to_default_z_up(self):
        """Rotate a declared OpenSim root and gravity into Newton's default Z-up world."""
        builder = newton.ModelBuilder()
        result = osim.add_osim(builder, MINIMAL_OSIM)

        np.testing.assert_allclose(np.asarray(builder.joint_X_p[0])[:3], [0.0, 0.0, 1.0], atol=1.0e-6)
        mapped_up = wp.transform_vector(result.world_xform, wp.vec3(0.0, 1.0, 0.0))
        mapped_lateral = wp.transform_vector(result.world_xform, wp.vec3(0.0, 0.0, 1.0))
        np.testing.assert_allclose(mapped_up, [0.0, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(mapped_lateral, [0.0, -1.0, 0.0], atol=1.0e-6)

        model = builder.finalize(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        np.testing.assert_allclose(state.body_q.numpy()[0, :3], [0.0, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(model.gravity.numpy()[0], [0.0, 0.0, -9.80665], atol=1.0e-5)

    def test_import_composes_user_placement_after_up_axis_conversion(self):
        """Apply user root placement after the fixed OpenSim-to-Newton basis rotation."""
        builder = newton.ModelBuilder()
        placement = wp.transform(wp.vec3(2.0, 3.0, 4.0), wp.quat_identity())
        osim.add_osim(builder, MINIMAL_OSIM, xform=placement)

        np.testing.assert_allclose(np.asarray(builder.joint_X_p[0])[:3], [2.0, 3.0, 5.0], atol=1.0e-6)

    def test_import_converts_opensim_halfspace_to_newton_plane_normal(self):
        """Map an OpenSim ground halfspace's outward normal onto Newton +Z."""
        builder = newton.ModelBuilder()
        placement = wp.transform(wp.vec3(2.0, 3.0, 4.0), wp.quat_identity())
        osim.add_osim(builder, _CONTACT_MODEL_OSIM, xform=placement, parse_muscles=False)
        ground_shape = next(i for i, body in enumerate(builder.shape_body) if body == -1)
        shape_xform = builder.shape_transform[ground_shape]
        np.testing.assert_allclose(np.asarray(shape_xform)[:3], [0.0, 0.0, 0.0], atol=1.0e-6)
        normal = wp.quat_rotate(shape_xform.q, wp.vec3(0.0, 0.0, 1.0))

        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1.0e-6)

    def test_muscle_metadata(self):
        """Verify muscle metadata maps path points to Newton body indices."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        result = osim.add_osim(builder, MINIMAL_OSIM)
        mm = result.muscles[0]
        self.assertEqual(mm.name, "m0")
        self.assertAlmostEqual(mm.fmax, 500.0)
        self.assertAlmostEqual(mm.l_opt, 0.12)
        self.assertAlmostEqual(mm.lt_slack, 0.15)
        self.assertEqual(mm.body_indices[0], -1)  # ground
        self.assertEqual(mm.body_indices[1], result.body_index["rod"])


class TestMuscleCurves(unittest.TestCase):
    def test_curves_physiology(self):
        """De Groote-Fregly curves peak at optimal length and unit velocity."""

        @wp.kernel
        def _eval(fal: wp.array[float], fv: wp.array[float], fpe: wp.array[float]):
            fal[0] = M.dgf_active_force_length(1.0)
            fal[1] = M.dgf_active_force_length(0.5)
            fv[0] = M.dgf_force_velocity(0.0)
            fpe[0] = M.dgf_passive_force_length(1.0)
            fpe[1] = M.dgf_passive_force_length(0.5)

        fal = wp.zeros(2, dtype=float)
        fv = wp.zeros(1, dtype=float)
        fpe = wp.zeros(2, dtype=float)
        wp.launch(_eval, 1, inputs=[fal, fv, fpe])
        fal_n, fv_n, fpe_n = fal.numpy(), fv.numpy(), fpe.numpy()
        # Active force-length ~1 at optimal, smaller away from optimal.
        self.assertAlmostEqual(float(fal_n[0]), 1.0, delta=0.02)
        self.assertLess(float(fal_n[1]), float(fal_n[0]))
        # Force-velocity ~1 at zero velocity.
        self.assertAlmostEqual(float(fv_n[0]), 1.0, delta=0.02)
        # Passive force ~0 at/below optimal length.
        self.assertAlmostEqual(float(fpe_n[0]), 0.0, delta=1e-3)
        self.assertAlmostEqual(float(fpe_n[1]), 0.0, delta=1e-3)

    def test_rigid_tendon_isometric(self):
        """Rigid-tendon force at optimal length and full activation ~ fmax."""

        @wp.kernel
        def _eval(out: wp.array[float]):
            lmt = 0.12 + 0.15  # l_opt + lt_slack
            out[0] = M.muscle_force_rigid_tendon(1.0, lmt, 0.0, 500.0, 0.12, 0.15, 10.0, 1.0)

        out = wp.zeros(1, dtype=float)
        wp.launch(_eval, 1, inputs=[out])
        self.assertAlmostEqual(float(out.numpy()[0]), 500.0, delta=10.0)


# A minimal self-contained legacy (Version < 30000) model: a planar leg with two
# CustomJoints. The knee couples a fore-aft translation to its flexion angle via
# a SimmSpline, mirroring how gait models represent tibiofemoral translation.
LEGACY_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="20302">
    <Model name="leg">
        <gravity> 0 -9.80665 0</gravity>
        <BodySet>
            <objects>
                <Body name="ground">
                    <mass>0</mass>
                    <mass_center> 0 0 0</mass_center>
                    <inertia_xx>0</inertia_xx><inertia_yy>0</inertia_yy><inertia_zz>0</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint/>
                </Body>
                <Body name="thigh">
                    <mass>5.0</mass>
                    <mass_center> 0 -0.2 0</mass_center>
                    <inertia_xx>0.1</inertia_xx><inertia_yy>0.02</inertia_yy><inertia_zz>0.1</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint>
                        <CustomJoint name="hip">
                            <SpatialTransform>
                                <TransformAxis name="rotation1"><coordinates>hip_angle</coordinates><axis>0 0 1</axis>
                                    <function><LinearFunction><coefficients> 1 0</coefficients></LinearFunction></function></TransformAxis>
                                <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="rotation3"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation1"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                            </SpatialTransform>
                            <parent_body>ground</parent_body>
                            <location_in_parent> 0 1.0 0</location_in_parent>
                            <orientation_in_parent> 0 0 0</orientation_in_parent>
                            <location> 0 0 0</location>
                            <orientation> 0 0 0</orientation>
                            <CoordinateSet>
                                <objects>
                                    <Coordinate name="hip_angle"><motion_type>rotational</motion_type>
                                        <default_value>0</default_value><range>-2 2</range></Coordinate>
                                </objects>
                            </CoordinateSet>
                        </CustomJoint>
                    </Joint>
                </Body>
                <Body name="shank">
                    <mass>3.0</mass>
                    <mass_center> 0 -0.2 0</mass_center>
                    <inertia_xx>0.05</inertia_xx><inertia_yy>0.01</inertia_yy><inertia_zz>0.05</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint>
                        <CustomJoint name="knee">
                            <SpatialTransform>
                                <TransformAxis name="rotation1"><coordinates>knee_angle</coordinates><axis>0 0 1</axis>
                                    <function><LinearFunction><coefficients> 1 0</coefficients></LinearFunction></function></TransformAxis>
                                <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="rotation3"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation1"><coordinates>knee_angle</coordinates><axis>1 0 0</axis>
                                    <function><SimmSpline><x> -2.0 -1.0 0.0 1.0</x><y> 0.02 0.01 0.0 -0.015</y></SimmSpline></function></TransformAxis>
                                <TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                            </SpatialTransform>
                            <parent_body>thigh</parent_body>
                            <location_in_parent> 0 -0.4 0</location_in_parent>
                            <orientation_in_parent> 0 0 0</orientation_in_parent>
                            <location> 0 0 0</location>
                            <orientation> 0 0 0</orientation>
                            <CoordinateSet>
                                <objects>
                                    <Coordinate name="knee_angle"><motion_type>rotational</motion_type>
                                        <default_value>0</default_value><range>-2 1</range></Coordinate>
                                </objects>
                            </CoordinateSet>
                        </CustomJoint>
                    </Joint>
                </Body>
            </objects>
        </BodySet>
        <MarkerSet>
            <objects>
                <Marker name="thigh_top"><body>thigh</body><location> 0.06 -0.05 0.02</location></Marker>
                <Marker name="thigh_bot"><body>thigh</body><location> -0.05 -0.35 -0.03</location></Marker>
                <Marker name="shank_top"><body>shank</body><location> 0.05 -0.05 0.02</location></Marker>
                <Marker name="shank_mid"><body>shank</body><location> -0.04 -0.25 -0.02</location></Marker>
                <Marker name="ankle"><body>shank</body><location> 0 -0.4 0.03</location></Marker>
            </objects>
        </MarkerSet>
    </Model>
</OpenSimDocument>
"""


class TestSimmSpline(unittest.TestCase):
    def test_interpolates_knots(self):
        """SimmSpline passes exactly through its (x, y) knots."""
        x = [-2.0944, -1.0, 0.1974, 1.5, 2.0944]
        y = [-0.0032, 0.0041, -0.0052, 0.001, -0.006]
        s = SimmSpline(x, y)
        for xi, yi in zip(x, y, strict=True):
            self.assertAlmostEqual(s.value(xi), yi, places=12)

    def test_extrapolates_linearly(self):
        """SimmSpline extrapolates linearly outside its knot range."""
        s = SimmSpline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        # Beyond the last knot the spline is a straight line (constant slope).
        v3, v4, v5 = s.value(3.0), s.value(4.0), s.value(5.0)
        self.assertAlmostEqual(v4 - v3, v5 - v4, places=9)
        # Extrapolation is continuous with the terminal knot value.
        self.assertAlmostEqual(s.value(2.0), 0.0, places=12)

    def test_build_function_types(self):
        """build_function constructs each supported coordinate function."""
        lin = build_function("LinearFunction", {"coefficients": [2.0, 1.0]})
        self.assertAlmostEqual(lin(3.0), 7.0)
        const = build_function("Constant", {"value": 0.5})
        self.assertAlmostEqual(const(123.0), 0.5)
        mult = build_function(
            "MultiplierFunction", {"scale": 2.0, "inner": {"type": "LinearFunction", "coefficients": [1.0, 0.0]}}
        )
        self.assertAlmostEqual(mult(3.0), 6.0)


class TestLegacyParserAndKinematics(unittest.TestCase):
    def test_parse_legacy_custom_joint(self):
        """Parse a legacy (<30000) model with inline CustomJoints and a SimmSpline."""
        m = osim.parse_osim(LEGACY_OSIM)
        self.assertEqual(m.version, 20302)
        self.assertEqual([b.name for b in m.bodies], ["thigh", "shank"])
        knee = next(j for j in m.joints if j.name == "knee")
        self.assertEqual(knee.type, "CustomJoint")
        self.assertEqual(knee.parent_body, "thigh")
        self.assertEqual(knee.child_body, "shank")
        # translation1 is a SimmSpline coupled to knee_angle.
        tx = knee.spatial_transform[3]
        self.assertEqual(tx.function_type, "SimmSpline")
        self.assertEqual(tx.coordinates, ["knee_angle"])
        self.assertEqual(len(m.markers), 5)

    def test_modern_joint_motion_types_are_inferred_without_motion_type_tags(self):
        """Keep FreeJoint translations in metres and coupled knee coordinates angular."""
        model = osim.parse_osim(
            """<OpenSimDocument Version="40600"><Model name="motion_types">
            <JointSet><objects>
              <FreeJoint name="free"><coordinates>
                <Coordinate name="r1"/><Coordinate name="r2"/><Coordinate name="r3"/>
                <Coordinate name="tx"/><Coordinate name="ty"/><Coordinate name="tz"/>
              </coordinates></FreeJoint>
              <CustomJoint name="custom"><coordinates>
                <Coordinate name="knee"/><Coordinate name="slide"/>
              </coordinates><SpatialTransform>
                <TransformAxis name="rotation1"><coordinates>knee</coordinates><axis>1 0 0</axis></TransformAxis>
                <TransformAxis name="rotation2"><coordinates/><axis>0 1 0</axis></TransformAxis>
                <TransformAxis name="rotation3"><coordinates/><axis>0 0 1</axis></TransformAxis>
                <TransformAxis name="translation1"><coordinates>knee</coordinates><axis>1 0 0</axis></TransformAxis>
                <TransformAxis name="translation2"><coordinates>slide</coordinates><axis>0 1 0</axis></TransformAxis>
                <TransformAxis name="translation3"><coordinates/><axis>0 0 1</axis></TransformAxis>
              </SpatialTransform></CustomJoint>
            </objects></JointSet></Model></OpenSimDocument>"""
        )
        free = next(joint for joint in model.joints if joint.name == "free")
        self.assertEqual(
            [coordinate.motion_type for coordinate in free.coordinates],
            ["rotational", "rotational", "rotational", "translational", "translational", "translational"],
        )
        custom = next(joint for joint in model.joints if joint.name == "custom")
        self.assertEqual([coordinate.motion_type for coordinate in custom.coordinates], ["rotational", "translational"])

    def test_modern_direct_named_transform_function(self):
        """Parse OpenSim 4.x functions written directly with name="function"."""
        legacy = "<function><LinearFunction><coefficients> 1 0</coefficients></LinearFunction></function>"
        modern = '<LinearFunction name="function"><coefficients> 1 0</coefficients></LinearFunction>'
        modern_model = _PENDULUM_OSIM.replace(legacy, modern).replace('Version="20302"', 'Version="40600"')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "modern_direct_function.osim")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(modern_model)
            model = osim.parse_osim(path)

        axis = model.joints[0].spatial_transform[0]
        self.assertEqual(axis.function_type, "LinearFunction")
        self.assertEqual(axis.function, {"type": "LinearFunction", "coefficients": [1.0, 0.0]})
        fk = ForwardKinematics(model)
        zero = fk.body_transforms({"theta": 0.0})["link"]
        rotated = fk.body_transforms({"theta": 0.5})["link"]
        self.assertGreater(float(np.max(np.abs(rotated - zero))), 0.1)

    def test_modern_direct_named_spline_function(self):
        """Parse a direct OpenSim 4.x SimmSpline transform function."""
        legacy = "<function><SimmSpline><x> -2.0 -1.0 0.0 1.0</x><y> 0.02 0.01 0.0 -0.015</y></SimmSpline></function>"
        modern = '<SimmSpline name="function"><x> -2.0 -1.0 0.0 1.0</x><y> 0.02 0.01 0.0 -0.015</y></SimmSpline>'
        modern_model = LEGACY_OSIM.replace(legacy, modern).replace('Version="20302"', 'Version="40600"')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "modern_direct_spline.osim")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(modern_model)
            model = osim.parse_osim(path)

        knee = next(joint for joint in model.joints if joint.name == "knee")
        self.assertEqual(knee.spatial_transform[3].function_type, "SimmSpline")
        fk = ForwardKinematics(model)
        transformed = fk.body_transforms({"hip_angle": 0.0, "knee_angle": -1.0})
        np.testing.assert_allclose(transformed["shank"][:3, 3], (0.01, 0.6, 0.0), atol=1e-9)

    def test_forward_kinematics_coupling(self):
        """CustomJoint forward kinematics applies the SimmSpline coupled translation."""
        m = osim.parse_osim(LEGACY_OSIM)
        fk = ForwardKinematics(m)
        self.assertEqual(fk.coordinate_names, ["hip_angle", "knee_angle"])
        # At q = 0 the shank origin sits at the knee (thigh y=1-0.4=0.6), with the
        # SimmSpline translation being 0 at knee_angle = 0.
        x = fk.body_transforms({"hip_angle": 0.0, "knee_angle": 0.0})
        np.testing.assert_allclose(x["shank"][:3, 3], (0.0, 0.6, 0.0), atol=1e-9)
        # At knee_angle = -1 the SimmSpline adds +0.01 m along the shank x-axis.
        x2 = fk.body_transforms({"hip_angle": 0.0, "knee_angle": -1.0})
        np.testing.assert_allclose(x2["shank"][:3, 3], (0.01, 0.6, 0.0), atol=1e-9)

    def test_marker_positions_batch_matches_single(self):
        """Batched Warp marker positions match single-pose evaluation exactly."""
        m = osim.parse_osim(LEGACY_OSIM)
        fk = ForwardKinematics(m)
        rng = np.random.default_rng(3)
        coords = rng.uniform(-0.8, 0.5, size=(6, len(fk.coordinate_names)))
        batch = fk.marker_positions_batch(coords)
        self.assertEqual(batch.shape, (6, len(fk.marker_names), 3))
        for s in range(coords.shape[0]):
            single = fk.marker_positions(coords[s])
            for i, name in enumerate(fk.marker_names):
                np.testing.assert_allclose(batch[s, i], single[name], atol=1e-12)


class TestCenterOfMass(unittest.TestCase):
    """Whole-body center of mass from a Warp kernel over forward-kinematics poses."""

    @staticmethod
    def _model():
        """A double pendulum with distinct per-body masses and mass centers."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        for i, b in enumerate(model.bodies):
            b.mass = 1.0 + i
            b.mass_center = (0.05 * (i + 1), -0.1 * (i + 1), 0.02 * i)
        return model

    def test_center_of_mass_matches_host_reduction(self):
        """The COM kernel reproduces the mass-weighted reduction of the FK poses.

        The reference transforms each body's mass center to ground with the
        already-validated ``body_transforms_batch`` and forms the mass-weighted
        average independently of the kernel.
        """
        model = self._model()
        fk = ForwardKinematics(model)
        rng = np.random.default_rng(1)
        coords = rng.uniform(-0.6, 0.6, size=(5, fk.ncoord))
        com = fk.center_of_mass_batch(coords)

        x = fk.body_transforms_batch(coords)
        masses = np.array([0.0] + [b.mass for b in model.bodies])
        centers = np.array([(0.0, 0.0, 0.0)] + [list(b.mass_center) for b in model.bodies])
        ref = np.zeros((coords.shape[0], 3))
        for s in range(coords.shape[0]):
            acc = np.zeros(3)
            for b in range(len(masses)):
                acc += masses[b] * (x[s, b] @ np.array([*centers[b], 1.0]))[:3]
            ref[s] = acc / masses.sum()
        np.testing.assert_allclose(com, ref, atol=1e-9)
        np.testing.assert_allclose(fk.center_of_mass(coords[0]), com[0], atol=1e-12)

    def test_single_body_com_is_its_transformed_mass_center(self):
        """With one massive body the COM is exactly that body's ground mass center."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.bodies = model.bodies[:1]
        model.bodies[0].mass = 3.0
        model.bodies[0].mass_center = (0.1, 0.2, 0.3)
        model.joints = [j for j in model.joints if j.parent_body == "ground" and j.child_body == model.bodies[0].name][
            :1
        ]
        fk = ForwardKinematics(model)
        q0 = np.zeros((1, fk.ncoord))
        x0 = fk.body_transforms_batch(q0)[0]
        bidx = fk.body_names.index(model.bodies[0].name)
        expect = (x0[bidx] @ np.array([0.1, 0.2, 0.3, 1.0]))[:3]
        np.testing.assert_allclose(fk.center_of_mass_batch(q0)[0], expect, atol=1e-9)


class TestBodyVelocities(unittest.TestCase):
    """Body angular and linear velocities from a Warp twist-extraction kernel."""

    def test_matches_host_extraction(self):
        """The kernel reproduces the host central-difference twist of the FK poses."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        rng = np.random.default_rng(3)
        coords = rng.uniform(-0.7, 0.7, size=(4, fk.ncoord))
        speeds = rng.uniform(-1.5, 1.5, size=(4, fk.ncoord))
        res = fk.body_velocities_batch(coords, speeds)

        h = 1e-6
        xp = fk.body_transforms_batch(coords + h * speeds)
        xm = fk.body_transforms_batch(coords - h * speeds)
        x0 = fk.body_transforms_batch(coords)
        ang = np.zeros_like(res["angular_velocity"])
        lin = np.zeros_like(res["linear_velocity"])
        for s in range(coords.shape[0]):
            for b in range(fk.nbody):
                rdot = (xp[s, b, :3, :3] - xm[s, b, :3, :3]) / (2 * h)
                w = rdot @ x0[s, b, :3, :3].T
                ang[s, b] = [(w[2, 1] - w[1, 2]) / 2, (w[0, 2] - w[2, 0]) / 2, (w[1, 0] - w[0, 1]) / 2]
                lin[s, b] = (xp[s, b, :3, 3] - xm[s, b, :3, 3]) / (2 * h)
        np.testing.assert_allclose(res["angular_velocity"], ang, atol=1e-10)
        np.testing.assert_allclose(res["linear_velocity"], lin, atol=1e-10)

    def test_pendulum_closed_form(self):
        """A pure base-joint rate gives rigid rotation about z with v = omega x r.

        With only the first coordinate rate nonzero, both pendulum links share the
        base angular velocity, and each body origin moves at ``omega x position``.
        """
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        q = np.zeros((1, fk.ncoord))
        qd = np.zeros((1, fk.ncoord))
        qd[0, 0] = 0.75
        res = fk.body_velocities_batch(q, qd)
        ang, lin = res["angular_velocity"][0], res["linear_velocity"][0]
        i1, i2 = fk.body_names.index("link1"), fk.body_names.index("link2")
        np.testing.assert_allclose(ang[i1], [0, 0, 0.75], atol=1e-6)
        np.testing.assert_allclose(ang[i2], [0, 0, 0.75], atol=1e-6)
        p2 = fk.body_transforms_batch(q)[0, i2, :3, 3]
        np.testing.assert_allclose(lin[i2], np.cross([0, 0, 0.75], p2), atol=1e-5)
        # Single-pose dict API and the zero-speed case.
        d = fk.body_velocities(q[0], qd[0])
        np.testing.assert_allclose(d["link1"]["angular"], [0, 0, 0.75], atol=1e-6)
        zero = fk.body_velocities_batch(q, np.zeros((1, fk.ncoord)))
        self.assertEqual(float(np.abs(zero["angular_velocity"]).max()), 0.0)
        self.assertEqual(float(np.abs(zero["linear_velocity"]).max()), 0.0)


class TestBodyJacobian(unittest.TestCase):
    """Per-body spatial Jacobian from a Warp central-difference kernel."""

    def test_jacobian_times_speed_matches_velocities(self):
        """``J @ qd`` reproduces the body spatial velocities for random speeds."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        rng = np.random.default_rng(0)
        coords = rng.uniform(-1.0, 1.0, size=(4, fk.ncoord))
        speeds = rng.uniform(-2.0, 2.0, size=(4, fk.ncoord))
        jac = fk.body_jacobian_batch(coords)
        self.assertEqual(jac.shape, (4, fk.nbody, 6, fk.ncoord))
        spatial = (jac @ speeds[:, None, :, None])[..., 0]
        vel = fk.body_velocities_batch(coords, speeds)
        np.testing.assert_allclose(spatial[..., :3], vel["angular_velocity"], atol=1e-6)
        np.testing.assert_allclose(spatial[..., 3:], vel["linear_velocity"], atol=1e-6)

    def test_pendulum_base_column_closed_form(self):
        """The base-coordinate column is the z-hinge screw ``[z, z x p]`` for each moved body."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        q = np.array([0.3, -0.4])
        jac = fk.body_jacobian(q)
        transforms = fk.body_transforms_batch(q[None, :])[0]
        z = np.array([0.0, 0.0, 1.0])
        for b, name in enumerate(fk.body_names):
            if name == "ground":
                np.testing.assert_allclose(jac[name], 0.0, atol=1e-9)
                continue
            col0 = jac[name][:, 0]
            origin = transforms[b][:3, 3]
            np.testing.assert_allclose(col0[:3], z, atol=1e-6)
            np.testing.assert_allclose(col0[3:], np.cross(z, origin), atol=1e-6)


class TestBodyLoadGeneralizedForces(unittest.TestCase):
    """External body-load to generalized-force mapping via a Warp Jacobian projection."""

    def test_point_force_matches_virtual_work_gradient(self):
        """A point force's generalized forces equal the gradient of ``F . p`` (virtual work).

        This checks the transposed-Jacobian projection against a fully independent
        central difference of the load-point's ground position, with no reuse of
        the body Jacobian kernel.
        """
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        rng = np.random.default_rng(1)
        coords = rng.uniform(-1.0, 1.0, size=(3, fk.ncoord))
        point = np.array([0.05, -0.1, 0.2])
        force = np.array([3.0, -2.0, 1.5])
        b = fk.body_names.index("link2")
        tau = fk.generalized_forces_from_body_load(coords, "link2", point=point, force=force)

        def p_point(q):
            x = fk.body_transforms_batch(q[None, :])[0, b]
            return x[:3, :3] @ point + x[:3, 3]

        h = 1e-6
        for row in range(coords.shape[0]):
            grad = np.array(
                [
                    force
                    @ (
                        p_point(coords[row] + h * np.eye(fk.ncoord)[i])
                        - p_point(coords[row] - h * np.eye(fk.ncoord)[i])
                    )
                    / (2.0 * h)
                    for i in range(fk.ncoord)
                ]
            )
            np.testing.assert_allclose(tau[row], grad, atol=1e-6)

    def test_pure_axis_torque_on_base_link(self):
        """A pure z-torque on link1 loads only the base coordinate by its magnitude."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        coords = np.array([[0.3, -0.4], [0.1, 0.9]])
        tau = fk.generalized_forces_from_body_load(coords, "link1", torque=np.array([0.0, 0.0, 4.0]))
        np.testing.assert_allclose(tau[:, 0], 4.0, atol=1e-6)
        np.testing.assert_allclose(tau[:, 1], 0.0, atol=1e-6)

    def test_projection_keeps_kinematics_on_device(self):
        """Keep body transforms and Jacobians on device until the projected load is copied."""
        fk = osim.ForwardKinematics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        coords = np.array([[0.3, -0.4], [0.1, 0.9]])
        kwargs = {
            "body": "link2",
            "point": np.array([0.05, -0.1, 0.2]),
            "force": np.array([3.0, -2.0, 1.5]),
            "torque": np.array([0.1, 0.2, -0.3]),
        }
        expected = fk.generalized_forces_from_body_load(coords, **kwargs)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("body-load projection called a host-returning kinematics wrapper")

        fk.body_transforms_batch = reject_host_wrapper
        fk.body_jacobian_batch = reject_host_wrapper
        np.testing.assert_allclose(fk.generalized_forces_from_body_load(coords, **kwargs), expected)


class TestWholeBodyMomentum(unittest.TestCase):
    """Whole-body linear and angular momentum from a Warp assembly kernel."""

    @staticmethod
    def _model():
        """A double pendulum with distinct masses, mass centers, and inertias."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        for i, b in enumerate(model.bodies):
            b.mass = 1.0 + i
            b.mass_center = (0.05 * (i + 1), -0.08 * (i + 1), 0.03 * i)
            b.inertia = (0.02 + 0.01 * i, 0.03 + 0.01 * i, 0.04 + 0.01 * i, 0.001 * i, 0.0, 0.002 * i)
        return model

    def test_matches_host_assembly(self):
        """The kernel reproduces an independent host assembly of the same primitives.

        The reference builds momentum from the already-validated body velocities,
        body poses, whole-body COM, and per-body inertias.
        """
        fk = ForwardKinematics(self._model())
        rng = np.random.default_rng(7)
        coords = rng.uniform(-0.6, 0.6, size=(4, fk.ncoord))
        speeds = rng.uniform(-1.2, 1.2, size=(4, fk.ncoord))
        res = fk.whole_body_momentum_batch(coords, speeds)

        vel = fk.body_velocities_batch(coords, speeds)
        x0 = fk.body_transforms_batch(coords)
        com = fk.center_of_mass_batch(coords)
        masses = [0.0] + [b.mass for b in fk.model.bodies]
        centers = [(0.0, 0.0, 0.0)] + [b.mass_center for b in fk.model.bodies]
        inertias = [(0.0,) * 6] + [b.inertia for b in fk.model.bodies]

        def imat(v):
            ixx, iyy, izz, ixy, ixz, iyz = v
            return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

        p_ref = np.zeros_like(res["linear_momentum"])
        h_ref = np.zeros_like(res["angular_momentum"])
        for s in range(coords.shape[0]):
            for b in range(fk.nbody):
                r = x0[s, b, :3, :3]
                off = r @ np.array(centers[b])
                r_b = x0[s, b, :3, 3] + off
                w = vel["angular_velocity"][s, b]
                v_b = vel["linear_velocity"][s, b] + np.cross(w, off)
                p_ref[s] += masses[b] * v_b
                h_ref[s] += (r @ imat(inertias[b]) @ r.T) @ w + masses[b] * np.cross(r_b - com[s], v_b)
        np.testing.assert_allclose(res["linear_momentum"], p_ref, atol=1e-9)
        np.testing.assert_allclose(res["angular_momentum"], h_ref, atol=1e-9)

    def test_momentum_keeps_body_velocities_on_device(self):
        """Keep body velocities and base poses on device until momentum is copied."""
        fk = ForwardKinematics(self._model())
        coords = np.array([[0.3, -0.4], [0.1, 0.9]])
        speeds = np.array([[0.5, -0.2], [-0.3, 0.4]])
        expected = fk.whole_body_momentum_batch(coords, speeds)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("momentum assembly called a host-returning velocity wrapper")

        fk.body_velocities_batch = reject_host_wrapper
        actual = fk.whole_body_momentum_batch(coords, speeds)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key])

    def test_single_body_on_axis_closed_form(self):
        """One body spinning about z with its COM on the axis gives P=0, H=Izz*omega*z."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.bodies = model.bodies[:1]
        model.bodies[0].mass = 4.0
        model.bodies[0].mass_center = (0.0, 0.0, 0.25)
        model.bodies[0].inertia = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)
        model.joints = [j for j in model.joints if j.parent_body == "ground" and j.child_body == model.bodies[0].name][
            :1
        ]
        fk = ForwardKinematics(model)
        q = np.zeros((1, fk.ncoord))
        qd = np.zeros((1, fk.ncoord))
        qd[0, 0] = 0.9
        res = fk.whole_body_momentum_batch(q, qd)
        np.testing.assert_allclose(res["linear_momentum"][0], 0.0, atol=1e-6)
        np.testing.assert_allclose(res["angular_momentum"][0], [0.0, 0.0, 0.3 * 0.9], atol=1e-5)
        d = fk.whole_body_momentum(q[0], qd[0])
        np.testing.assert_allclose(d["angular"], [0.0, 0.0, 0.27], atol=1e-5)


class TestBodyAccelerations(unittest.TestCase):
    """Body angular and linear accelerations from a Warp second-difference kernel."""

    def test_matches_host_extraction(self):
        """The kernel reproduces the host second central difference of the FK poses."""
        fk = ForwardKinematics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        rng = np.random.default_rng(11)
        coords = rng.uniform(-0.6, 0.6, size=(4, fk.ncoord))
        speeds = rng.uniform(-1.0, 1.0, size=(4, fk.ncoord))
        accels = rng.uniform(-2.0, 2.0, size=(4, fk.ncoord))
        dt = 1e-4
        res = fk.body_accelerations_batch(coords, speeds, accels, dt=dt)

        drift = 0.5 * dt * dt * accels
        xp = fk.body_transforms_batch(coords + dt * speeds + drift)
        xm = fk.body_transforms_batch(coords - dt * speeds + drift)
        x0 = fk.body_transforms_batch(coords)
        ang = np.zeros_like(res["angular_acceleration"])
        lin = np.zeros_like(res["linear_acceleration"])
        inv = 1.0 / (dt * dt)
        for s in range(coords.shape[0]):
            for b in range(fk.nbody):
                rdd = (xp[s, b, :3, :3] + xm[s, b, :3, :3] - 2 * x0[s, b, :3, :3]) * inv
                a = rdd @ x0[s, b, :3, :3].T
                ang[s, b] = [(a[2, 1] - a[1, 2]) / 2, (a[0, 2] - a[2, 0]) / 2, (a[1, 0] - a[0, 1]) / 2]
                lin[s, b] = (xp[s, b, :3, 3] + xm[s, b, :3, 3] - 2 * x0[s, b, :3, 3]) * inv
        np.testing.assert_allclose(res["angular_acceleration"], ang, atol=1e-6)
        np.testing.assert_allclose(res["linear_acceleration"], lin, atol=1e-6)

    def test_pendulum_closed_form(self):
        """A base-joint rate and acceleration give alpha about z and a = alpha x r + omega x (omega x r)."""
        fk = ForwardKinematics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        q = np.zeros((1, fk.ncoord))
        qd = np.zeros((1, fk.ncoord))
        qdd = np.zeros((1, fk.ncoord))
        w0, a0 = 0.8, 1.7
        qd[0, 0], qdd[0, 0] = w0, a0
        res = fk.body_accelerations_batch(q, qd, qdd)
        aa, la = res["angular_acceleration"][0], res["linear_acceleration"][0]
        x0 = fk.body_transforms_batch(q)[0]
        for name in ("link1", "link2"):
            bi = fk.body_names.index(name)
            p = x0[bi, :3, 3]
            expect_l = np.cross([0, 0, a0], p) + np.cross([0, 0, w0], np.cross([0, 0, w0], p))
            np.testing.assert_allclose(aa[bi], [0, 0, a0], atol=1e-3)
            np.testing.assert_allclose(la[bi], expect_l, atol=1e-3)
        d = fk.body_accelerations(q[0], qd[0], qdd[0])
        np.testing.assert_allclose(d["link1"]["angular"], [0, 0, a0], atol=1e-3)
        zero = fk.body_accelerations_batch(q, np.zeros((1, fk.ncoord)), np.zeros((1, fk.ncoord)))
        self.assertLess(float(np.abs(zero["angular_acceleration"]).max()), 1e-6)
        self.assertLess(float(np.abs(zero["linear_acceleration"]).max()), 1e-6)


class TestCoordinateActuators(unittest.TestCase):
    """Generalized forces from non-muscle coordinate actuators via a Warp scatter kernel."""

    @staticmethod
    def _coords(model):
        """Global coordinate names in model order."""
        return [c.name for j in model.joints for c in j.coordinates]

    def test_scatter_clamp_and_collision(self):
        """Actuators scatter optimal_force*clamp(control) and sum when sharing a coordinate."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        c = self._coords(model)
        model.actuators = [
            osim.OsimActuator(name="a0", type="CoordinateActuator", coordinate=c[0], optimal_force=10.0),
            osim.OsimActuator(
                name="a1", type="CoordinateActuator", coordinate=c[0], optimal_force=3.0, min_control=-1.0
            ),
            osim.OsimActuator(
                name="a2",
                type="CoordinateActuator",
                coordinate=c[1],
                optimal_force=7.0,
                min_control=-1.0,
                max_control=2.0,
            ),
        ]
        act = osim.CoordinateActuators(model)
        self.assertEqual(act.actuator_names, ["a0", "a1", "a2"])
        # a1 clamps -5 -> -1; a2 clamps 3 -> 2; coord0 = 10*2 + 3*(-1) = 17; coord1 = 7*2 = 14.
        tau = act.generalized_forces(np.array([[2.0, -5.0, 3.0]]))
        np.testing.assert_allclose(tau, [[17.0, 14.0]])

    def test_unbounded_default_and_broadcast(self):
        """Unbounded controls pass through, and a 1-D control vector broadcasts to one batch."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        c = self._coords(model)
        model.actuators = [osim.OsimActuator(name="a", type="CoordinateActuator", coordinate=c[1], optimal_force=5.0)]
        act = osim.CoordinateActuators(model)
        np.testing.assert_allclose(act.generalized_forces(np.array([[100.0]])), [[0.0, 500.0]])
        np.testing.assert_allclose(act.generalized_forces(np.array([4.0])), [[0.0, 20.0]])

    def test_no_actuators(self):
        """A model without coordinate actuators yields zero generalized forces."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.actuators = []
        act = osim.CoordinateActuators(model)
        self.assertEqual(act.num_actuators, 0)
        np.testing.assert_allclose(act.generalized_forces(np.zeros((3, 0))), np.zeros((3, act.ncoord)))


class TestSpatialActuators(unittest.TestCase):
    """Point/torque actuator generalized forces via a fused Warp Jacobian-projection kernel."""

    def test_matches_per_load_reconstruction(self):
        """Fused point+torque actuator forces equal summed per-actuator body-load projections."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        model.actuators = [
            osim.OsimActuator(
                name="pa",
                type="PointActuator",
                body="link2",
                point=(0.05, -0.1, 0.2),
                direction=(1.0, 0.0, 0.0),
                force_is_global=True,
                optimal_force=3.0,
            ),
            osim.OsimActuator(
                name="ta",
                type="TorqueActuator",
                body="link1",
                body_b="ground",
                direction=(0.0, 0.0, 1.0),
                force_is_global=True,
                optimal_force=4.0,
            ),
        ]
        sa = osim.SpatialActuators(model)
        self.assertEqual(sa.actuator_names, ["pa", "ta"])
        rng = np.random.default_rng(2)
        coords = rng.uniform(-1.0, 1.0, size=(3, fk.ncoord))
        u = rng.uniform(-1.0, 1.0, size=(3, 2))
        tau = sa.generalized_forces(coords, u)
        ref = np.zeros_like(tau)
        for row in range(3):
            c = coords[row : row + 1]
            ref[row] += fk.generalized_forces_from_body_load(
                c, "link2", point=(0.05, -0.1, 0.2), force=3.0 * u[row, 0] * np.array([1.0, 0.0, 0.0])
            )[0]
            ref[row] += fk.generalized_forces_from_body_load(
                c, "link1", torque=4.0 * u[row, 1] * np.array([0.0, 0.0, 1.0])
            )[0]
        np.testing.assert_allclose(tau, ref, atol=1e-9)

    def test_torque_actuator_closed_form(self):
        """A unit-control z torque actuator (optimal_force 4) loads only the base coordinate."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.actuators = [
            osim.OsimActuator(
                name="ta",
                type="TorqueActuator",
                body="link1",
                body_b="ground",
                direction=(0.0, 0.0, 1.0),
                force_is_global=True,
                optimal_force=4.0,
            )
        ]
        sa = osim.SpatialActuators(model)
        coords = np.array([[0.3, -0.4], [0.1, 0.9]])
        tau = sa.generalized_forces(coords, np.array([1.0]))
        np.testing.assert_allclose(tau[:, 0], 4.0, atol=1e-6)
        np.testing.assert_allclose(tau[:, 1], 0.0, atol=1e-6)

    def test_body_frame_force_direction(self):
        """A body-frame PointActuator direction is rotated to ground before projection."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        model.actuators = [
            osim.OsimActuator(
                name="p",
                type="PointActuator",
                body="link2",
                direction=(0.0, 1.0, 0.0),
                force_is_global=False,
                optimal_force=2.0,
            )
        ]
        sa = osim.SpatialActuators(model)
        coords = np.array([[0.2, 0.5], [-0.3, 0.1]])
        tau = sa.generalized_forces(coords, np.array([[1.0], [1.0]]))
        transforms = fk.body_transforms_batch(coords)
        b = fk.body_names.index("link2")
        ref = np.array(
            [
                fk.generalized_forces_from_body_load(
                    coords[r : r + 1], "link2", force=2.0 * (transforms[r, b, :3, :3] @ np.array([0.0, 1.0, 0.0]))
                )[0]
                for r in range(2)
            ]
        )
        np.testing.assert_allclose(tau, ref, atol=1e-9)


class TestBodyActuators(unittest.TestCase):
    """Body (spatial-force) actuator generalized forces via a fused Warp kernel."""

    def test_matches_per_load_reconstruction(self):
        """Fused body-actuator forces equal summed per-actuator spatial body-load projections."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        model.actuators = [
            osim.OsimActuator(
                name="ba",
                type="BodyActuator",
                body="link2",
                point=(0.03, 0.02, -0.05),
                force_is_global=True,
                optimal_force=2.0,
            ),
            osim.OsimActuator(name="bb", type="BodyActuator", body="link1", force_is_global=True, optimal_force=1.5),
        ]
        ba = osim.BodyActuators(model)
        self.assertEqual(ba.actuator_names, ["ba", "bb"])
        rng = np.random.default_rng(3)
        coords = rng.uniform(-1.0, 1.0, size=(3, fk.ncoord))
        controls = rng.uniform(-1.0, 1.0, size=(3, 2, 6))
        tau = ba.generalized_forces(coords, controls)
        specs = [("link2", (0.03, 0.02, -0.05), 2.0), ("link1", (0.0, 0.0, 0.0), 1.5)]
        ref = np.zeros_like(tau)
        for row in range(3):
            for a, (body, point, scale) in enumerate(specs):
                ref[row] += fk.generalized_forces_from_body_load(
                    coords[row : row + 1],
                    body,
                    point=point,
                    force=scale * controls[row, a, 3:],
                    torque=scale * controls[row, a, :3],
                )[0]
        np.testing.assert_allclose(tau, ref, atol=1e-9)

    def test_body_frame_force_direction(self):
        """A body-frame spatial force is rotated to ground before projection."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = osim.ForwardKinematics(model)
        model.actuators = [
            osim.OsimActuator(name="b", type="BodyActuator", body="link2", force_is_global=False, optimal_force=1.0)
        ]
        ba = osim.BodyActuators(model)
        coords = np.array([[0.2, 0.5], [-0.3, 0.1], [0.4, -0.6]])
        controls = np.zeros((3, 1, 6))
        controls[:, 0, 4] = 1.0  # unit force along the body y-axis
        tau = ba.generalized_forces(coords, controls)
        transforms = fk.body_transforms_batch(coords)
        b = fk.body_names.index("link2")
        ref = np.array(
            [
                fk.generalized_forces_from_body_load(
                    coords[r : r + 1], "link2", force=transforms[r, b, :3, :3] @ np.array([0.0, 1.0, 0.0])
                )[0]
                for r in range(3)
            ]
        )
        np.testing.assert_allclose(tau, ref, atol=1e-9)


class TestMocapIO(unittest.TestCase):
    def test_trc_round_trip(self):
        """Marker data survives a .trc write/read round trip in meters."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(4, 3, 3))
        md = MarkerData(times=np.arange(4) * 0.01, marker_names=["a", "b", "c"], data=data, rate=100.0, units="m")
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/m.trc"
            write_trc(path, md, units="mm")
            back = read_trc(path)
        self.assertEqual(back.marker_names, ["a", "b", "c"])
        np.testing.assert_allclose(back.data, data, atol=1e-6)

    def test_storage_round_trip(self):
        """Coordinate storage survives a .mot write/read round trip."""
        times = np.linspace(0.0, 1.0, 5)
        data = np.column_stack([np.sin(times), np.cos(times)])
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/c.mot"
            write_storage(path, times, ["q0", "q1"], data, name="test")
            s = read_storage(path)
        self.assertEqual(s.labels, ["q0", "q1"])
        self.assertTrue(s.in_degrees)
        np.testing.assert_allclose(s.times, times, atol=1e-7)
        np.testing.assert_allclose(s.data, data, atol=1e-7)


class TestInverseKinematics(unittest.TestCase):
    def test_recovers_synthetic_trajectory(self):
        """Marker IK recovers the coordinate trajectory that generated the markers.

        Synthetic markers are produced by forward kinematics from a known
        hip/knee trajectory (including the coupled SimmSpline knee translation).
        Marker-based inverse kinematics must recover those coordinates exactly.
        """
        m = osim.parse_osim(LEGACY_OSIM)
        fk = ForwardKinematics(m)
        n = 10
        hip = np.linspace(0.1, 0.6, n)
        knee = np.linspace(-0.2, -1.2, n)
        names = [mk.name for mk in m.markers]
        data = np.zeros((n, len(names), 3))
        for fi in range(n):
            positions = fk.marker_positions({"hip_angle": hip[fi], "knee_angle": knee[fi]})
            data[fi] = np.array([positions[nm] for nm in names])
        markers = MarkerData(times=np.arange(n) * 0.1, marker_names=names, data=data, rate=10.0, units="m")

        result = InverseKinematics(m).solve(markers)
        self.assertLess(result.marker_rms.max(), 1e-6)
        np.testing.assert_allclose(result.values[:, 0], hip, atol=1e-5)
        np.testing.assert_allclose(result.values[:, 1], knee, atol=1e-5)

    def test_batched_matches_sequential(self):
        """The default batched solve reproduces the sequential per-frame solve.

        Both paths minimize the same weighted least-squares marker fit; the
        device-resident batched loop must converge to the same coordinate
        trajectory the warm-started sequential fallback does, to well within the
        finite-difference Jacobian's precision.
        """
        m = osim.parse_osim(LEGACY_OSIM)
        fk = ForwardKinematics(m)
        n = 12
        hip = np.linspace(-0.3, 0.7, n)
        knee = np.linspace(-0.1, -1.3, n)
        names = [mk.name for mk in m.markers]
        data = np.zeros((n, len(names), 3))
        for fi in range(n):
            positions = fk.marker_positions({"hip_angle": hip[fi], "knee_angle": knee[fi]})
            data[fi] = np.array([positions[nm] for nm in names])
        markers = MarkerData(times=np.arange(n) * 0.1, marker_names=names, data=data, rate=10.0, units="m")

        batched = InverseKinematics(m, batched=True).solve(markers)
        sequential = InverseKinematics(m, batched=False).solve(markers)
        self.assertTrue(InverseKinematics(m).batched)
        np.testing.assert_allclose(batched.values, sequential.values, atol=1e-7)
        np.testing.assert_allclose(batched.marker_rms, sequential.marker_rms, atol=1e-9)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_cuda_graph_matches_cpu_batch(self):
        """Match the CPU batch while replaying LM iteration chunks in a CUDA graph."""
        model = osim.parse_osim(LEGACY_OSIM)
        fk = ForwardKinematics(model)
        n = 12
        expected_values = np.stack([np.linspace(-0.3, 0.7, n), np.linspace(-0.1, -1.3, n)], axis=1)
        names = [marker.name for marker in model.markers]
        data = np.empty((n, len(names), 3))
        for frame, values in enumerate(expected_values):
            positions = fk.marker_positions(dict(zip(fk.coordinate_names, values, strict=True)))
            data[frame] = np.array([positions[name] for name in names])
        markers = MarkerData(times=np.arange(n) * 0.1, marker_names=names, data=data, rate=10.0, units="m")

        expected = InverseKinematics(model, device="cpu").solve(markers)
        actual = InverseKinematics(model, device="cuda:0").solve(markers)
        np.testing.assert_allclose(actual.values, expected.values, atol=1.0e-10)
        np.testing.assert_allclose(actual.marker_rms, expected.marker_rms, atol=1.0e-10)
        np.testing.assert_allclose(actual.marker_max, expected.marker_max, atol=1.0e-10)


# Opt-in 1-for-1 regression against OpenSim's own gait2354 synthetic-marker IK
# reference. Point NEWTON_OPENSIM_GAIT2354 at a directory containing
# subject01_simbody.osim, subject01_synthetic_marker_data.trc, and
# std_subject01_walk1_ik.mot (from opensim-core/Applications/IK/test).
_GAIT2354_DIR = os.environ.get("NEWTON_OPENSIM_GAIT2354", "")


@unittest.skipUnless(
    _GAIT2354_DIR and os.path.isdir(_GAIT2354_DIR),
    "set NEWTON_OPENSIM_GAIT2354 to the opensim-core IK test data directory",
)
class TestGait2354InverseKinematics(unittest.TestCase):
    def test_reproduce_opensim_reference(self):
        """Reproduce OpenSim's gait2354 IK reference within its own 0.2 deg bar.

        OpenSim's testIK regression recovers std_subject01_walk1_ik.mot from
        synthetic markers and requires each coordinate within 0.2 deg (RMS
        < 0.1 deg). This test runs the Newton-native IK on the same inputs and
        asserts the same agreement.
        """
        model = osim.parse_osim(os.path.join(_GAIT2354_DIR, "subject01_simbody.osim"))
        markers = read_trc(os.path.join(_GAIT2354_DIR, "subject01_synthetic_marker_data.trc"))
        reference = read_storage(os.path.join(_GAIT2354_DIR, "std_subject01_walk1_ik.mot"))

        result = InverseKinematics(model).solve(markers)
        values_deg = result.values_in_degrees()
        for i, name in enumerate(result.coordinate_names):
            if result.motion_types[i] != "rotational":
                continue
            error = values_deg[:, i] - reference.column(name)
            self.assertLess(np.max(np.abs(error)), 0.2, f"{name} exceeds 0.2 deg")
            self.assertLess(np.sqrt(np.mean(error**2)), 0.1, f"{name} RMS exceeds 0.1 deg")


_PIN_AXES = """                            <SpatialTransform>
                                <TransformAxis name="rotation1"><coordinates>{coord}</coordinates><axis>0 0 1</axis>
                                    <function><LinearFunction><coefficients> 1 0</coefficients></LinearFunction></function></TransformAxis>
                                <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="rotation3"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation1"><coordinates></coordinates><axis>1 0 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                                <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis>
                                    <function><Constant><value>0</value></Constant></function></TransformAxis>
                            </SpatialTransform>"""


# A single-pendulum model (rod pinned about +Z at the origin, mass center at
# x = 0.5 m) with a closed-form inverse dynamics: tau = I_pivot * qdd
# + m * g * L * cos(theta), I_pivot = I_zz + m * L**2.
_PENDULUM_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="20302">
    <Model name="pendulum">
        <gravity> 0 -9.80665 0</gravity>
        <BodySet>
            <objects>
                <Body name="ground"><mass>0</mass><mass_center> 0 0 0</mass_center>
                    <inertia_xx>0</inertia_xx><inertia_yy>0</inertia_yy><inertia_zz>0</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz><Joint/></Body>
                <Body name="link">
                    <mass>2.0</mass>
                    <mass_center> 0.5 0 0</mass_center>
                    <inertia_xx>0.01</inertia_xx><inertia_yy>0.05</inertia_yy><inertia_zz>0.05</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint>
                        <CustomJoint name="pin">
{axes}
                            <parent_body>ground</parent_body>
                            <location_in_parent> 0 0 0</location_in_parent>
                            <orientation_in_parent> 0 0 0</orientation_in_parent>
                            <location> 0 0 0</location>
                            <orientation> 0 0 0</orientation>
                            <CoordinateSet><objects>
                                <Coordinate name="theta"><motion_type>rotational</motion_type>
                                    <default_value>0</default_value><range>-3.15 3.15</range></Coordinate>
                            </objects></CoordinateSet>
                        </CustomJoint>
                    </Joint>
                </Body>
            </objects>
        </BodySet>
    </Model>
</OpenSimDocument>""".format(axes=_PIN_AXES.format(coord="theta"))


def _link_body(name, mass, comx, joint, parent, locx):
    return """                <Body name="{name}">
                    <mass>{mass}</mass>
                    <mass_center> {comx} 0 0</mass_center>
                    <inertia_xx>0.01</inertia_xx><inertia_yy>0.03</inertia_yy><inertia_zz>0.03</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                    <Joint>
                        <CustomJoint name="{joint}">
{axes}
                            <parent_body>{parent}</parent_body>
                            <location_in_parent> {locx} 0 0</location_in_parent>
                            <orientation_in_parent> 0 0 0</orientation_in_parent>
                            <location> 0 0 0</location>
                            <orientation> 0 0 0</orientation>
                            <CoordinateSet><objects>
                                <Coordinate name="{joint}_q"><motion_type>rotational</motion_type>
                                    <default_value>0</default_value><range>-3.15 3.15</range></Coordinate>
                            </objects></CoordinateSet>
                        </CustomJoint>
                    </Joint>
                </Body>""".format(
        name=name,
        mass=mass,
        comx=comx,
        joint=joint,
        parent=parent,
        locx=locx,
        axes=_PIN_AXES.format(coord=joint + "_q"),
    )


# A planar two-link chain to exercise the Jacobian-transpose reduction across a
# kinematic chain (link 2's weight loads joint 1).
_DOUBLE_PENDULUM_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="20302">
    <Model name="double_pendulum">
        <gravity> 0 -9.80665 0</gravity>
        <BodySet><objects>
                <Body name="ground"><mass>0</mass><mass_center> 0 0 0</mass_center>
                    <inertia_xx>0</inertia_xx><inertia_yy>0</inertia_yy><inertia_zz>0</inertia_zz>
                    <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz><Joint/></Body>
{link1}
{link2}
        </objects></BodySet>
    </Model>
</OpenSimDocument>""".format(
    link1=_link_body("link1", 3.0, 0.4, "j1", "ground", 0.0),
    link2=_link_body("link2", 1.5, 0.3, "j2", "link1", 0.9),
)


# OpenSim's own Applications/Forward/test pendulum: a 10 kg point mass 0.5 m from
# a pin about +Z. Released from theta0 = -pi/20 at rest it is a near-ideal simple
# harmonic oscillator, theta(t) = -amp*cos(k*t), k = sqrt(g / 0.5); this is exactly
# what OpenSim's testForward.cpp asserts (to 1e-2).
_FD_PENDULUM_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="20302">
    <Model name="pendulum">
        <gravity> 0 -9.80665 0</gravity>
        <BodySet><objects>
            <Body name="ground"><mass>0</mass><mass_center> 0 0 0</mass_center>
                <inertia_xx>0</inertia_xx><inertia_yy>0</inertia_yy><inertia_zz>0</inertia_zz>
                <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz><Joint/></Body>
            <Body name="pendulum">
                <mass>10.0</mass>
                <mass_center> 0 0 0</mass_center>
                <inertia_xx>0</inertia_xx><inertia_yy>0</inertia_yy><inertia_zz>0</inertia_zz>
                <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
                <Joint>
                    <PinJoint name="GroundJoint">
                        <parent_body>ground</parent_body>
                        <location_in_parent> 0 0 0</location_in_parent>
                        <orientation_in_parent> 0 0 0</orientation_in_parent>
                        <location> 0 0.5 0</location>
                        <orientation> 0 0 0</orientation>
                        <CoordinateSet><objects>
                            <Coordinate name="Pendulum_r1"><motion_type>rotational</motion_type>
                                <default_value>-0.1570796326795</default_value>
                                <range>-3.14159265 3.14159265</range></Coordinate>
                        </objects></CoordinateSet>
                    </PinJoint>
                </Joint>
            </Body>
        </objects></BodySet>
    </Model>
</OpenSimDocument>"""


class TestGcvspline(unittest.TestCase):
    def test_interpolates_and_differentiates(self):
        """Interpolate knots exactly and approximate derivatives of a smooth signal."""
        x = np.linspace(0.0, 1.0, 21)
        y = np.sin(3.0 * x)
        coeffs = gcvspl.fit_gcvspline(x, y)
        knot_err = max(abs(gcvspl.eval_gcvspline(x, coeffs, xi, 0) - np.sin(3.0 * xi)) for xi in x)
        self.assertLess(knot_err, 1.0e-10)
        d1 = gcvspl.eval_gcvspline(x, coeffs, 0.5, 1)
        d2 = gcvspl.eval_gcvspline(x, coeffs, 0.5, 2)
        self.assertLess(abs(d1 - 3.0 * np.cos(1.5)), 1.0e-3)
        self.assertLess(abs(d2 + 9.0 * np.sin(1.5)), 1.0e-2)

    def test_device_batch_matches_host_evaluator(self):
        """The Warp batch evaluator reproduces the host de Boor evaluation exactly.

        ``differentiate_coordinates`` evaluates fitted splines on-device; the
        kernel must bit-match the host ``eval_gcvspline`` across knot counts,
        columns, and derivative orders (0, 1, 2).
        """
        rng = np.random.default_rng(0)
        max_err = 0.0
        for _ in range(4):
            x = np.unique(np.sort(rng.uniform(0.0, 10.0, size=rng.integers(9, 22))))
            cols = [np.sin(0.7 * x) + 0.2 * rng.standard_normal(x.size), np.cos(0.4 * x), 0.01 * x**2]
            coeffs = np.stack([gcvspl.fit_gcvspline(x, y, half_order=3) for y in cols], axis=0)
            out_times = np.sort(rng.uniform(x[0], x[-1], size=13))
            host = np.array(
                [
                    [[gcvspl.eval_gcvspline(x, coeffs[c], t, o) for o in range(3)] for c in range(len(cols))]
                    for t in out_times
                ]
            )
            device = gcvspl.eval_gcvspline_batch(x, coeffs, out_times)
            max_err = max(max_err, float(np.abs(device - host).max()))
        self.assertEqual(max_err, 0.0)


class TestSignalFilter(unittest.TestCase):
    def test_pad_reflects_through_endpoints(self):
        """Pad by reflecting and negating through each endpoint, preserving the ends."""
        sig = np.array([1.0, 2.0, 4.0, 7.0])
        padded = pad_signal(sig, 2)
        self.assertEqual(len(padded), 8)
        np.testing.assert_allclose(padded[2:6], sig)
        # reflection through the first point x0=1: p[1] = 2*x0 - sig[1] = 0
        self.assertAlmostEqual(padded[1], 2.0 * 1.0 - 2.0)

    def test_lowpass_preserves_dc_and_is_zero_lag(self):
        """Pass a constant unchanged and a slow sine without phase lag."""
        dt = 0.01
        const = np.full(200, 3.0)
        np.testing.assert_allclose(lowpass_iir(const, dt, 6.0), 3.0, atol=1e-9)
        t = np.arange(400) * dt
        sine = np.sin(2.0 * np.pi * 0.5 * t)  # 0.5 Hz, well below 6 Hz cutoff
        filtered = lowpass_iir(sine, dt, 6.0)
        self.assertLess(np.max(np.abs(filtered[50:-50] - sine[50:-50])), 0.02)


class TestInverseDynamics(unittest.TestCase):
    def test_pendulum_matches_analytic(self):
        """Match the closed-form single-pendulum joint torque in Warp kernels."""
        model = osim.parse_osim(_PENDULUM_OSIM)
        idyn = InverseDynamics(model)
        m, ell, g = 2.0, 0.5, 9.80665
        i_pivot = 0.05 + m * ell * ell
        theta = np.array([0.0, 0.3, -0.7, 1.2, 2.0, -1.5])
        thetadd = np.array([0.0, 1.0, -2.0, 3.5, -0.5, 4.0])
        thetad = np.array([0.0, 0.5, -1.0, 2.0, -3.0, 1.5])
        tau = idyn.solve(theta[:, None], thetad[:, None], thetadd[:, None])[:, 0]
        analytic = i_pivot * thetadd + m * g * ell * np.cos(theta)
        np.testing.assert_allclose(tau, analytic, atol=1e-4)

    def test_double_pendulum_static_gravity(self):
        """Recover the static gravity load across a two-link chain (Jacobian transpose)."""
        idyn = InverseDynamics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        g, m1, lc1, l1, m2, lc2 = 9.80665, 3.0, 0.4, 0.9, 1.5, 0.3
        q = np.array([[0.0, 0.0], [0.4, -0.6], [-1.0, 1.3], [2.0, 0.5], [-0.8, -1.2]])
        zeros = np.zeros_like(q)
        tau = idyn.solve(q, zeros, zeros)
        q1, q2 = q[:, 0], q[:, 1]
        g1 = m1 * g * lc1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + lc2 * np.cos(q1 + q2))
        g2 = m2 * g * lc2 * np.cos(q1 + q2)
        np.testing.assert_allclose(tau[:, 0], g1, atol=1e-6)
        np.testing.assert_allclose(tau[:, 1], g2, atol=1e-6)

    def test_from_motion_recovers_analytic_and_labels(self):
        """Filter, spline-differentiate, and solve a sampled motion end to end."""
        model = osim.parse_osim(_PENDULUM_OSIM)
        m, ell, g = 2.0, 0.5, 9.80665
        i_pivot = 0.05 + m * ell * ell
        amp, freq = 0.6, 2.0
        t = np.linspace(0.0, 2.0, 201)
        theta = amp * np.sin(freq * t)
        coords = osim.Storage(times=t, labels=["theta"], data=np.rad2deg(theta)[:, None], in_degrees=True)
        out = t[10:-10]
        result = solve_inverse_dynamics(model, coords, cutoff=0.0, time_range=(out[0], out[-1]))
        self.assertEqual(result.column_labels, ["theta_moment"])
        th = amp * np.sin(freq * result.times)
        thdd = -amp * freq * freq * np.sin(freq * result.times)
        analytic = i_pivot * thdd + m * g * ell * np.cos(th)
        np.testing.assert_allclose(result.generalized_forces[:, 0], analytic, atol=1e-3)
        self.assertIsNotNone(result.coordinates)
        self.assertIsNotNone(result.speeds)
        self.assertIsNotNone(result.accelerations)
        np.testing.assert_allclose(result.coordinates[:, 0], th, atol=1.0e-9)
        np.testing.assert_allclose(result.speeds[:, 0], amp * freq * np.cos(freq * result.times), atol=1.0e-5)
        np.testing.assert_allclose(result.accelerations[:, 0], thdd, atol=2.0e-4)

    def test_result_storage_round_trip(self):
        """Write and re-read an inverse-dynamics result as an OpenSim storage."""
        idyn = InverseDynamics(osim.parse_osim(_PENDULUM_OSIM))
        q = np.array([[0.3], [0.4]])
        tau = idyn.solve(q, np.zeros_like(q), np.zeros_like(q))
        result = osim.IDResult(
            times=np.array([0.0, 0.1]),
            coordinate_names=idyn.coordinate_names,
            generalized_forces=tau,
            motion_types=idyn.motion_types,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "id.sto")
            result.write_sto(path)
            back = read_storage(path)
        self.assertFalse(back.in_degrees)
        np.testing.assert_allclose(back.column("theta_moment"), tau[:, 0], rtol=1e-6)


class TestForwardDynamics(unittest.TestCase):
    def test_pendulum_matches_simple_harmonic_oscillator(self):
        """Integrate OpenSim's forward-test pendulum and match its SHO reference.

        Reproduces OpenSim's ``testForward`` pendulum check: a 10 kg point mass
        0.5 m from a pin, released from ``-pi/20`` at rest, tracks
        ``theta(t) = -amp*cos(k*t)`` with ``k = sqrt(g/0.5)`` to within 1e-2 over
        one second.
        """
        model = osim.parse_osim(_FD_PENDULUM_OSIM)
        result = solve_forward_dynamics(
            model,
            initial_coordinates=np.array([-np.pi / 20.0]),
            initial_speeds=np.array([0.0]),
            duration=1.0,
            dt=1.0e-3,
        )
        self.assertEqual(result.times[0], 0.0)
        self.assertAlmostEqual(result.times[-1], 1.0)
        self.assertEqual(result.column_labels, ["Pendulum_r1", "Pendulum_r1_u"])
        amp = np.pi / 20.0
        k = np.sqrt(9.80665 / 0.5)
        theta = -amp * np.cos(k * result.times)
        omega = amp * k * np.sin(k * result.times)
        self.assertLess(np.max(np.abs(result.coordinates[:, 0] - theta)), 1.0e-2)
        self.assertLess(np.max(np.abs(result.speeds[:, 0] - omega)), 1.0e-2)

    def test_zero_width_clamped_coordinate_remains_fixed(self):
        """Constrain a zero-width clamped coordinate during acceleration and rollout."""
        xml = _FD_PENDULUM_OSIM.replace(
            "<range>-3.14159265 3.14159265</range>",
            "<range>0 0</range><clamped>true</clamped>",
        )
        forward = ForwardDynamics(osim.parse_osim(xml))

        acceleration = forward.accelerations(
            np.array([[0.0]]),
            np.array([[3.0]]),
            np.array([[100.0]]),
        )
        result = forward.simulate(
            initial_coordinates=np.array([0.2]),
            initial_speeds=np.array([3.0]),
            duration=0.01,
            dt=0.001,
            controls=lambda _t, _q, _qd: np.array([100.0]),
        )

        np.testing.assert_array_equal(acceleration, 0.0)
        np.testing.assert_array_equal(result.coordinates, 0.0)
        np.testing.assert_array_equal(result.speeds, 0.0)

    def test_locked_coordinate_eliminates_coupled_mass_row_and_column(self):
        """Hold a locked coordinate while solving the coupled free-coordinate equation."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        coordinates = {coordinate.name: coordinate for joint in model.joints for coordinate in joint.coordinates}
        coordinates["j1_q"].locked = True
        forward = ForwardDynamics(model)
        q = np.array([[0.2, -0.3]])
        qd = np.zeros_like(q)
        tau = np.array([[100.0, 2.0]])

        acceleration = forward.accelerations(q, qd, tau)
        acceleration_without_reaction = forward.accelerations(q, qd, np.array([[-100.0, 2.0]]))
        mass = forward.mass_matrix(q)[0]
        bias = forward.idyn.solve(q, qd, np.zeros_like(q))[0]
        expected_free = (tau[0, 1] - bias[1]) / mass[1, 1]

        self.assertEqual(acceleration[0, 0], 0.0)
        self.assertAlmostEqual(acceleration[0, 1], expected_free, places=9)
        np.testing.assert_allclose(acceleration_without_reaction, acceleration, atol=1.0e-12)

    def test_clamped_coordinate_without_range_remains_free(self):
        """Accept a clamped coordinate whose optional range is absent."""
        model = osim.parse_osim(_FD_PENDULUM_OSIM)
        coordinate = model.joints[0].coordinates[0]
        coordinate.clamped = True
        coordinate.range = None

        forward = ForwardDynamics(model)
        acceleration = forward.accelerations(
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[100.0]]),
        )

        self.assertNotEqual(acceleration[0, 0], 0.0)

    def test_locked_coordinate_remains_fixed_in_batched_cuda_rollout(self):
        """Preserve locked values and zero their speeds in captured batched simulation."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA is unavailable")
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        coordinates = {coordinate.name: coordinate for joint in model.joints for coordinate in joint.coordinates}
        coordinates["j1_q"].locked = True
        forward = ForwardDynamics(model, device="cuda:0")
        initial_q = np.array([[0.2, -0.3], [0.5, 0.1]])
        initial_qd = np.array([[3.0, 0.0], [-2.0, 0.0]])

        result = forward.simulate_batch(
            initial_q,
            initial_qd,
            duration=0.005,
            dt=0.001,
            tau_applied=np.array([100.0, 0.0]),
            use_graph=True,
        )

        expected_locked = np.broadcast_to(initial_q[None, :, 0], result.coordinates[:, :, 0].shape)
        np.testing.assert_allclose(result.coordinates[:, :, 0], expected_locked, atol=1.0e-12)
        np.testing.assert_array_equal(result.speeds[:, :, 0], 0.0)

    def test_mass_matrix_is_symmetric_positive_definite(self):
        """Return a symmetric, positive-definite joint-space mass matrix."""
        fdyn = ForwardDynamics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        mass = fdyn.mass_matrix(np.array([0.3, -0.5]))
        self.assertEqual(mass.shape, (2, 2))
        np.testing.assert_allclose(mass, mass.T, atol=1e-9)
        self.assertGreater(np.linalg.eigvalsh(mass).min(), 0.0)

    def test_accelerations_invert_inverse_dynamics(self):
        """Recover the accelerations that inverse dynamics turned into torques.

        Forward and inverse dynamics are exact inverses: for a random state and
        acceleration, the inverse-dynamics torque fed back through forward
        dynamics returns the original acceleration.
        """
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        idyn = InverseDynamics(model)
        fdyn = ForwardDynamics(model)
        rng = np.random.default_rng(0)
        q = rng.uniform(-1.0, 1.0, (4, 2))
        qd = rng.uniform(-2.0, 2.0, (4, 2))
        qdd = rng.uniform(-3.0, 3.0, (4, 2))
        tau = idyn.solve(q, qd, qdd)
        recovered = fdyn.accelerations(q, qd, tau)
        np.testing.assert_allclose(recovered, qdd, atol=1e-5)

    def test_external_load_accelerations_invert_inverse_dynamics(self):
        """Recover coordinate accelerations while expanding compact external wrenches on device."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        inverse = InverseDynamics(model)
        forward = ForwardDynamics(model)
        q = np.array([[0.2, -0.3], [-0.1, 0.4]])
        qd = np.array([[0.5, -0.2], [-0.3, 0.1]])
        qdd = np.array([[0.7, -0.6], [0.2, 0.8]])
        wrench = np.array(
            [
                [[3.0, -2.0, 1.0, 0.2, 0.1, 0.0, 0.5, -0.4, 0.3]],
                [[-1.0, 4.0, 0.5, -0.1, 0.3, 0.0, -0.2, 0.1, 0.6]],
            ]
        )
        bodies = ["link2"]
        tau = inverse.solve(q, qd, qdd, external_bodies=bodies, external_wrenches=wrench)
        actual = forward.accelerations(q, qd, tau, external_bodies=bodies, external_wrenches=wrench)
        np.testing.assert_allclose(actual, qdd, atol=1.0e-7)

    def test_double_pendulum_falls_under_gravity(self):
        """Release a double pendulum from rest and conserve energy while it swings."""
        fdyn = ForwardDynamics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        result = fdyn.simulate(
            initial_coordinates=np.array([0.2, -0.3]),
            initial_speeds=np.array([0.0, 0.0]),
            duration=0.5,
            dt=5.0e-4,
        )
        # It must actually move away from the initial configuration.
        self.assertGreater(np.max(np.abs(result.coordinates - result.coordinates[0])), 0.05)
        # Passive conservative system: total mechanical energy is nearly constant.
        g, m1, lc1, l1, m2, lc2 = 9.80665, 3.0, 0.4, 0.9, 1.5, 0.3
        q1, q2 = result.coordinates[:, 0], result.coordinates[:, 1]
        # Potential energy (heights of the two mass centers about the pin).
        pe = m1 * g * (lc1 * np.sin(q1)) + m2 * g * (l1 * np.sin(q1) + lc2 * np.sin(q1 + q2))
        # Kinetic energy from the Warp mass matrix, KE = 0.5 * u^T M(q) u.
        mass = fdyn.mass_matrix(result.coordinates)
        ke = 0.5 * np.einsum("ti,tij,tj->t", result.speeds, mass, result.speeds)
        energy = ke + pe
        self.assertLess(np.max(np.abs(energy - energy[0])), 1.0e-2)

    def test_batch_matches_per_trajectory(self):
        """Batched multi-trajectory forward dynamics matches per-trajectory integration.

        Several double-pendulum trajectories integrated together in one
        device-resident RK4 (and symplectic-Euler) loop must reproduce, sample
        for sample, the trajectories obtained by integrating each initial
        condition separately.
        """
        fdyn = ForwardDynamics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        nc = fdyn.ncoord
        rng = np.random.default_rng(0)
        q0 = rng.uniform(-0.4, 0.4, (4, nc))
        v0 = rng.uniform(-0.5, 0.5, (4, nc))
        dur, dt = 0.3, 5.0e-4
        for integrator in ("rk4", "semi_implicit"):
            batch = fdyn.simulate_batch(q0, v0, dur, dt, integrator=integrator)
            self.assertEqual(batch.coordinates.shape, (int(round(dur / dt)) + 1, 4, nc))
            for b in range(4):
                single = fdyn.simulate(q0[b], v0[b], dur, dt, integrator=integrator, use_graph=False)
                np.testing.assert_allclose(batch.coordinates[:, b, :], single.coordinates, atol=1e-6)
                np.testing.assert_allclose(batch.speeds[:, b, :], single.speeds, atol=1e-6)
                np.testing.assert_allclose(batch.trajectory(b).coordinates, single.coordinates, atol=1e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_cuda_graph_batch_matches_uncaptured(self):
        """Replay the CUDA-graph forward stepper with the same result as direct launches."""
        fdyn = ForwardDynamics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM), device="cuda:0")
        q0 = np.array([[0.2, -0.3], [-0.1, 0.4]])
        v0 = np.array([[0.1, 0.0], [0.0, -0.1]])
        captured = fdyn.simulate_batch(q0, v0, 0.01, 1.0e-3, use_graph=True)
        direct = fdyn.simulate_batch(q0, v0, 0.01, 1.0e-3, use_graph=False)
        np.testing.assert_allclose(captured.coordinates, direct.coordinates, atol=1.0e-12)
        np.testing.assert_allclose(captured.speeds, direct.speeds, atol=1.0e-12)

    def test_batch_final_only_and_subsampled(self):
        """`record_every` controls sampling: 0 keeps the final state, N subsamples the trajectory."""
        fdyn = ForwardDynamics(osim.parse_osim(_DOUBLE_PENDULUM_OSIM))
        q0 = np.array([[0.2, -0.3], [-0.1, 0.4]])
        v0 = np.zeros((2, 2))
        dur, dt = 0.2, 5.0e-4
        full = fdyn.simulate_batch(q0, v0, dur, dt, record_every=1)
        final = fdyn.simulate_batch(q0, v0, dur, dt, record_every=0)
        self.assertEqual(final.coordinates.shape, (1, 2, 2))
        np.testing.assert_allclose(final.coordinates[0], full.coordinates[-1], atol=1e-9)
        sub = fdyn.simulate_batch(q0, v0, dur, dt, record_every=5)
        np.testing.assert_allclose(sub.coordinates, full.coordinates[::5], atol=1e-9)
        np.testing.assert_allclose(sub.times, full.times[::5], atol=1e-12)


_ID_DIR = os.environ.get("NEWTON_OPENSIM_ID", "")


@unittest.skipUnless(
    _ID_DIR and os.path.isdir(_ID_DIR),
    "set NEWTON_OPENSIM_ID to the opensim-core Applications/ID/test data directory",
)
class TestOpenSimInverseDynamics(unittest.TestCase):
    def test_arm26_matches_reference(self):
        """Reproduce OpenSim's arm26 ID reference within its 1e-2 N*m bar.

        arm26 has no external loads; OpenSim's testID requires each joint moment
        within 0.01 N*m of std_arm26_InverseDynamics.sto. This runs the
        Newton-native ID on the same inputs and asserts the same agreement.
        """
        result = solve_inverse_dynamics(
            os.path.join(_ID_DIR, "arm26.osim"),
            os.path.join(_ID_DIR, "arm26_InverseKinematics.mot"),
        )
        reference = read_storage(os.path.join(_ID_DIR, "std_arm26_InverseDynamics.sto"))
        ref_cols = {lab: i for i, lab in enumerate(reference.labels)}
        for i, label in enumerate(result.column_labels):
            if label in ref_cols:
                error = np.abs(result.generalized_forces[:, i] - reference.data[:, ref_cols[label]])
                self.assertLess(np.max(error), 1.0e-2, f"{label} exceeds 0.01 N*m")

    def test_subject01_gait_matches_reference(self):
        """Reproduce OpenSim's subject01 gait ID reference within its 2 N*m bar.

        The gait2354 subject01 case adds experimental ground reactions; OpenSim's
        testID uses a 2 N*m tolerance for this configuration. The reference is
        evaluated on its own output times (the setup and reference grids differ
        by a few padded edge frames).
        """
        loads = osim.read_external_loads(
            os.path.join(_ID_DIR, "subject01_walk1_grf.xml"),
            os.path.join(_ID_DIR, "subject01_walk1_grf.mot"),
        )
        reference = read_storage(os.path.join(_ID_DIR, "std_subject01_InverseDynamics.sto"))
        idyn = InverseDynamics(osim.parse_osim(os.path.join(_ID_DIR, "subject01.osim")))
        result = idyn.solve_from_motion(
            os.path.join(_ID_DIR, "subject01_walk1_ik.mot"),
            external_loads=loads,
            output_times=reference.times,
        )
        ref_cols = {lab: i for i, lab in enumerate(reference.labels)}
        for i, label in enumerate(result.column_labels):
            if label in ref_cols:
                error = np.abs(result.generalized_forces[:, i] - reference.data[:, ref_cols[label]])
                self.assertLess(np.max(error), 2.0, f"{label} exceeds 2 N*m")


class TestMusclePath(unittest.TestCase):
    """Muscle-tendon path length and moment arms (``MusclePaths``)."""

    def test_minimal_muscle_length(self):
        """Parse a 2-point GeometryPath and match the analytic path length.

        ``MINIMAL_OSIM`` places the first muscle point on the pin axis, so the
        muscle-tendon length is the constant pivot-to-insertion distance (0.2 m).
        """
        model = osim.parse_osim(MINIMAL_OSIM)
        mp = osim.MusclePaths(model)
        fk = ForwardKinematics(model)
        angles = np.linspace(-1.0, 1.0, 9)[:, None]
        lengths = mp.lengths(angles)
        np.testing.assert_allclose(lengths[:, 0], 0.2, atol=1e-12)
        transforms = fk.body_transforms_batch(angles)
        bidx = {n: i for i, n in enumerate(fk.body_names)}
        for k in range(len(angles)):
            rod = transforms[k, bidx["rod"]]
            p2 = rod[:3, :3] @ np.array([0.0, -0.2, 0.0]) + rod[:3, 3]
            self.assertAlmostEqual(lengths[k, 0], float(np.linalg.norm(np.array([0.0, 1.0, 0.0]) - p2)), places=12)

    def _ground_pos(self, transforms, bidx, body, loc, k):
        t = transforms[k, bidx[body]]
        return t[:3, :3] @ np.asarray(loc, float) + t[:3, 3]

    def test_fixed_points_length_and_moment_arm(self):
        """Match kernel path length to geometry and moment arm to ``-dL/dq``.

        Two straight-line muscles are attached to the double pendulum; the Warp
        path length must equal the Euclidean via-point distance from the forward
        kinematics, and the moment arm must equal a central finite difference of
        the length (OpenSim's moment-arm definition).
        """
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        b0, b1 = fk.body_names[1], fk.body_names[2]
        nc = fk.ncoord
        paths = [("ground", (0.1, 0.3, 0.0), b0, (0.2, 0.05, 0.0)), (b0, (0.4, 0.1, 0.0), b1, (0.15, -0.05, 0.0))]
        model.muscles = [
            osim.OsimMuscle(
                name="m_ab",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="a", body=paths[0][0], location=paths[0][1]),
                    osim.OsimPathPoint(name="b", body=paths[0][2], location=paths[0][3]),
                ],
            ),
            osim.OsimMuscle(
                name="m_cd",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="c", body=paths[1][0], location=paths[1][1]),
                    osim.OsimPathPoint(name="d", body=paths[1][2], location=paths[1][3]),
                ],
            ),
        ]
        mp = osim.MusclePaths(model)
        rng = np.random.default_rng(0)
        q = rng.uniform(-1.0, 1.0, size=(12, nc))
        lengths = mp.lengths(q)
        transforms = fk.body_transforms_batch(q)
        bidx = {n: i for i, n in enumerate(fk.body_names)}
        ref = np.array(
            [
                [
                    float(
                        np.linalg.norm(
                            self._ground_pos(transforms, bidx, pb, pl, k)
                            - self._ground_pos(transforms, bidx, qb, ql, k)
                        )
                    )
                    for (pb, pl, qb, ql) in paths
                ]
                for k in range(len(q))
            ]
        )
        np.testing.assert_allclose(lengths, ref, atol=1e-12)

        moment_arms = mp.moment_arms(q, eps=1e-5)
        step = 1e-6
        for k in range(len(q)):
            single = fk.body_transforms_batch
            for c in range(nc):
                qp = q[k].copy()
                qp[c] += step
                qm = q[k].copy()
                qm[c] -= step
                xp = single(qp[None, :])[0]
                xm = single(qm[None, :])[0]
                lp = np.array(
                    [
                        float(
                            np.linalg.norm(
                                (xp[bidx[pb]][:3, :3] @ np.asarray(pl) + xp[bidx[pb]][:3, 3])
                                - (xp[bidx[qb]][:3, :3] @ np.asarray(ql) + xp[bidx[qb]][:3, 3])
                            )
                        )
                        for (pb, pl, qb, ql) in paths
                    ]
                )
                lm = np.array(
                    [
                        float(
                            np.linalg.norm(
                                (xm[bidx[pb]][:3, :3] @ np.asarray(pl) + xm[bidx[pb]][:3, 3])
                                - (xm[bidx[qb]][:3, :3] @ np.asarray(ql) + xm[bidx[qb]][:3, 3])
                            )
                        )
                        for (pb, pl, qb, ql) in paths
                    ]
                )
                np.testing.assert_allclose(moment_arms[k, :, c], -(lp - lm) / (2 * step), atol=1e-7)

    def test_conditional_path_point(self):
        """A ConditionalPathPoint joins the path only within its coordinate range."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        c0 = fk.coordinate_names[0]
        b0 = fk.body_names[1]
        via = osim.OsimPathPoint(
            name="v",
            body=b0,
            location=(0.2, 0.1, 0.0),
            type="ConditionalPathPoint",
            conditional_coordinate=c0,
            conditional_range=(-0.2, 0.2),
        )
        pts = [
            osim.OsimPathPoint(name="a", body="ground", location=(0.1, 0.3, 0.0)),
            via,
            osim.OsimPathPoint(name="b", body=b0, location=(0.3, -0.1, 0.0)),
        ]
        model.muscles = [osim.OsimMuscle(name="mc", type="Thelen2003Muscle", path_points=pts)]
        mp = osim.MusclePaths(model)
        bidx = {n: i for i, n in enumerate(fk.body_names)}
        for qval, active in ((0.0, True), (1.0, False)):
            q = np.zeros((1, fk.ncoord))
            q[0, 0] = qval
            t = fk.body_transforms_batch(q)
            a = np.array([0.1, 0.3, 0.0])
            vv = self._ground_pos(t, bidx, b0, (0.2, 0.1, 0.0), 0)
            bb = self._ground_pos(t, bidx, b0, (0.3, -0.1, 0.0), 0)
            expected = (
                float(np.linalg.norm(a - vv) + np.linalg.norm(vv - bb)) if active else float(np.linalg.norm(a - bb))
            )
            self.assertAlmostEqual(mp.lengths(q)[0, 0], expected, places=12)

    def test_moving_path_point(self):
        """A MovingPathPoint location tracks a coordinate through its function."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        c0 = fk.coordinate_names[0]
        b0 = fk.body_names[1]
        moving = {"x": (c0, "LinearFunction", {"type": "LinearFunction", "coefficients": [0.5, 0.1]})}
        pts = [
            osim.OsimPathPoint(name="a", body="ground", location=(0.1, 0.3, 0.0)),
            osim.OsimPathPoint(name="m", body=b0, location=(0.1, 0.05, 0.0), type="MovingPathPoint", moving=moving),
        ]
        model.muscles = [osim.OsimMuscle(name="mm", type="Thelen2003Muscle", path_points=pts)]
        mp = osim.MusclePaths(model)
        q = np.zeros((5, fk.ncoord))
        q[:, 0] = np.linspace(-0.5, 0.5, 5)
        lengths = mp.lengths(q)
        transforms = fk.body_transforms_batch(q)
        bidx = {n: i for i, n in enumerate(fk.body_names)}
        for k in range(len(q)):
            loc = np.array([0.5 * q[k, 0] + 0.1, 0.05, 0.0])
            mg = transforms[k, bidx[b0]][:3, :3] @ loc + transforms[k, bidx[b0]][:3, 3]
            self.assertAlmostEqual(lengths[k, 0], float(np.linalg.norm(np.array([0.1, 0.3, 0.0]) - mg)), places=12)


class TestMuscleWrap(unittest.TestCase):
    """Muscle-path wrapping over a ``WrapSphere`` (:func:`muscle_path.wrap_sphere_extra`)."""

    @staticmethod
    def _ref_extra(p1, p2, c, radius):
        """Closed-form tangent-arc-tangent detour length over a sphere."""
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        c = np.asarray(c, float)
        d1v, d2v = p1 - c, p2 - c
        d1, d2 = np.linalg.norm(d1v), np.linalg.norm(d2v)
        if d1 <= radius or d2 <= radius:
            return 0.0
        l1 = np.sqrt(d1 * d1 - radius * radius)
        l2 = np.sqrt(d2 * d2 - radius * radius)
        cphi = np.clip(np.dot(d1v, d2v) / (d1 * d2), -1.0, 1.0)
        beta = np.arccos(cphi) - np.arccos(radius / d1) - np.arccos(radius / d2)
        if beta <= 0.0:
            return 0.0
        return (l1 + l2 + radius * beta) - np.linalg.norm(p2 - p1)

    def _wrap_model(
        self,
        p1,
        p2,
        center,
        radius,
        active=True,
        wtype="WrapSphere",
        rotation=(0.0, 0.0, 0.0),
        dimensions=(0.0, 0.0, 0.0),
        inner_radius=0.0,
        outer_radius=0.0,
    ):
        """Build a 2-point ground muscle wrapping a single ground-fixed wrap surface."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.muscles = [
            osim.OsimMuscle(
                name="mw",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=p1),
                    osim.OsimPathPoint(name="b", body="ground", location=p2),
                ],
                wraps=[osim.OsimWrap(wrap_object="ws")],
            )
        ]
        model.wrap_objects = [
            osim.OsimWrapObject(
                name="ws",
                type=wtype,
                body="ground",
                translation=center,
                rotation=rotation,
                radius=radius,
                length=1.0,
                dimensions=dimensions,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                active=active,
            )
        ]
        return model

    @staticmethod
    def _cyl_geodesic(p1, p2, o, axis, radius):
        """Independent cylinder-wrap length by developing the surface (brute force)."""
        p1, p2, o = np.asarray(p1, float), np.asarray(p2, float), np.asarray(o, float)
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        q1 = np.array([(p1 - o) @ u, (p1 - o) @ v])
        q2 = np.array([(p2 - o) @ u, (p2 - o) @ v])
        z1, z2 = (p1 - o) @ a, (p2 - o) @ a
        d1, d2 = np.linalg.norm(q1), np.linalg.norm(q2)
        if d1 <= radius or d2 <= radius:
            return float(np.linalg.norm(p2 - p1))
        best = None
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                a1 = np.arctan2(q1[1], q1[0]) + s1 * np.arccos(radius / d1)
                a2 = np.arctan2(q2[1], q2[0]) + s2 * np.arccos(radius / d2)
                t1 = radius * np.array([np.cos(a1), np.sin(a1)])
                t2 = radius * np.array([np.cos(a2), np.sin(a2)])
                if abs((q1 - t1) @ t1) > 1e-9 or abs((q2 - t2) @ t2) > 1e-9:
                    continue
                da = (a2 - a1 + np.pi) % (2.0 * np.pi) - np.pi
                s = np.linalg.norm(q1 - t1) + radius * abs(da) + np.linalg.norm(q2 - t2)
                if best is None or s < best[0]:
                    best = (s, a1, da, t1, np.linalg.norm(q1 - t1), radius * abs(da))
        s_total, a1, da, t1, l1, larc = best
        pts = []
        for t in np.linspace(0.0, 1.0, 4000):
            p2d = q1 + (t1 - q1) * t
            pts.append((p2d, l1 * t))
        for t in np.linspace(0.0, 1.0, 4000):
            ang = a1 + da * t
            pts.append((radius * np.array([np.cos(ang), np.sin(ang)]), l1 + larc * t))
        t2 = radius * np.array([np.cos(a1 + da), np.sin(a1 + da)])
        for t in np.linspace(0.0, 1.0, 4000):
            pts.append((t2 + (q2 - t2) * t, l1 + larc + np.linalg.norm(q2 - t2) * t))
        dz = z2 - z1
        poly = np.array([o + p2d[0] * u + p2d[1] * v + (z1 + dz * (s / s_total)) * a for p2d, s in pts])
        return float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))

    def test_sphere_wrap_matches_geodesic(self):
        """A penetrated WrapSphere adds the analytic tangent-arc-tangent length."""
        p1, p2, c, r = (-0.2, 0.05, 0.0), (0.2, 0.05, 0.0), (0.0, 0.0, 0.0), 0.1
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, r))
        length = mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]
        expected = float(np.linalg.norm(np.array(p2) - np.array(p1))) + self._ref_extra(p1, p2, c, r)
        self.assertGreater(self._ref_extra(p1, p2, c, r), 0.0)
        self.assertAlmostEqual(length, expected, places=9)

    def test_sphere_wrap_3d_matches_geodesic(self):
        """Off-plane points wrap along the great circle in the points-centre plane."""
        p1, p2, c, r = (-0.2, 0.05, 0.1), (0.2, 0.03, -0.08), (0.01, 0.0, 0.0), 0.09
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, r))
        length = mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]
        expected = float(np.linalg.norm(np.array(p2) - np.array(p1))) + self._ref_extra(p1, p2, c, r)
        self.assertGreater(self._ref_extra(p1, p2, c, r), 0.0)
        self.assertAlmostEqual(length, expected, places=9)

    def test_non_penetrating_sphere_leaves_length_unchanged(self):
        """A wrap surface the straight path misses does not change the path length."""
        p1, p2, c, r = (-0.2, 0.5, 0.0), (0.2, 0.5, 0.0), (0.0, 0.0, 0.0), 0.1
        straight = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, r))
        length = mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]
        self.assertAlmostEqual(length, straight, places=12)

    def test_inactive_sphere_is_ignored(self):
        """An inactive WrapSphere never wraps the path."""
        p1, p2, c, r = (-0.2, 0.05, 0.0), (0.2, 0.05, 0.0), (0.0, 0.0, 0.0), 0.1
        straight = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, r, active=False))
        length = mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]
        self.assertAlmostEqual(length, straight, places=12)

    def test_cylinder_planar_matches_sphere(self):
        """A z-axis cylinder wraps an in-plane segment exactly like a sphere."""
        p1, p2, c, r = (-0.2, 0.05, 0.0), (0.2, 0.05, 0.0), (0.0, 0.0, 0.0), 0.1
        angles = np.zeros((1, osim.MusclePaths(self._wrap_model(p1, p2, c, r)).fk.ncoord))
        sphere = osim.MusclePaths(self._wrap_model(p1, p2, c, r, wtype="WrapSphere")).lengths(angles)[0, 0]
        cylinder = osim.MusclePaths(self._wrap_model(p1, p2, c, r, wtype="WrapCylinder")).lengths(angles)[0, 0]
        self.assertGreater(cylinder, float(np.linalg.norm(np.array(p2) - np.array(p1))))
        self.assertAlmostEqual(cylinder, sphere, places=9)

    def test_cylinder_axial_matches_geodesic(self):
        """An axially-offset cylinder wrap matches the developed-surface geodesic."""
        p1, p2, c, r = (-0.2, 0.05, 0.1), (0.2, 0.05, -0.05), (0.0, 0.0, 0.0), 0.1
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, r, wtype="WrapCylinder"))
        length = mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]
        self.assertAlmostEqual(length, self._cyl_geodesic(p1, p2, c, (0.0, 0.0, 1.0), r), places=5)

    def test_cylinder_tilted_axis_matches_geodesic(self):
        """A cylinder rotated so its axis is not global-z still wraps correctly."""
        rot = (np.pi / 2.0, 0.0, 0.0)  # body-fixed XYZ -> axis becomes -y
        axis = (0.0, -1.0, 0.0)
        p1, p2, c, r = (-0.2, 0.0, 0.05), (0.2, 0.0, 0.05), (0.0, 0.0, 0.0), 0.1
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, r, wtype="WrapCylinder", rotation=rot))
        length = mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]
        self.assertAlmostEqual(length, self._cyl_geodesic(p1, p2, c, axis, r), places=5)

    def test_non_penetrating_cylinder_leaves_length_unchanged(self):
        """A cylinder the straight path misses does not change the path length."""
        p1, p2, c, r = (-0.2, 0.5, 0.0), (0.2, 0.5, 0.0), (0.0, 0.0, 0.0), 0.1
        straight = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, r, wtype="WrapCylinder"))
        self.assertAlmostEqual(mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0], straight, places=12)

    def test_isotropic_ellipsoid_matches_sphere(self):
        """An ellipsoid with equal semi-axes wraps exactly like a sphere of that radius."""
        p1, p2, c, r = (-0.2, 0.05, 0.0), (0.2, 0.05, 0.0), (0.0, 0.0, 0.0), 0.1
        angles = np.zeros((1, osim.MusclePaths(self._wrap_model(p1, p2, c, r)).fk.ncoord))
        sphere = osim.MusclePaths(self._wrap_model(p1, p2, c, r, wtype="WrapSphere")).lengths(angles)[0, 0]
        ellipsoid = osim.MusclePaths(
            self._wrap_model(p1, p2, c, 0.0, wtype="WrapEllipsoid", dimensions=(r, r, r))
        ).lengths(angles)[0, 0]
        self.assertGreater(sphere, float(np.linalg.norm(np.array(p2) - np.array(p1))))
        # The mapped-back arc is a 12-segment chord sum, so allow a sub-millimetre tolerance.
        self.assertAlmostEqual(ellipsoid, sphere, places=4)

    def test_anisotropic_ellipsoid_wraps_more_than_smaller_sphere(self):
        """A taller ellipsoid detours more than the sphere of its smallest semi-axis.

        Stretching the semi-axis normal to the segment plane pushes the surface
        farther into the straight path, so the wrapped length must exceed the
        detour of a sphere whose radius equals the in-plane semi-axis.
        """
        p1, p2, c, r = (-0.2, 0.05, 0.0), (0.2, 0.05, 0.0), (0.0, 0.0, 0.0), 0.1
        angles = np.zeros((1, osim.MusclePaths(self._wrap_model(p1, p2, c, r)).fk.ncoord))
        sphere = osim.MusclePaths(self._wrap_model(p1, p2, c, r, wtype="WrapSphere")).lengths(angles)[0, 0]
        tall = osim.MusclePaths(
            self._wrap_model(p1, p2, c, 0.0, wtype="WrapEllipsoid", dimensions=(r, 0.18, r))
        ).lengths(angles)[0, 0]
        self.assertGreater(tall, sphere)

    def test_non_penetrating_ellipsoid_leaves_length_unchanged(self):
        """An ellipsoid the straight path misses does not change the path length."""
        p1, p2, c = (-0.2, 0.5, 0.0), (0.2, 0.5, 0.0), (0.0, 0.0, 0.0)
        straight = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        mp = osim.MusclePaths(self._wrap_model(p1, p2, c, 0.0, wtype="WrapEllipsoid", dimensions=(0.1, 0.1, 0.1)))
        self.assertAlmostEqual(mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0], straight, places=12)

    def test_torus_tube_wraps_like_sphere_on_ring(self):
        """A torus tube penetrated head-on wraps like a sphere of the tube radius.

        With the segment crossing the ring plane radially outside the hole, the
        nearest ring point supplies a sphere of the tube radius, so the detour
        matches the closed-form sphere geodesic about that ring point.
        """
        ring, tube = 0.2, 0.05
        inner, outer = ring - tube, ring + tube  # inner=0.15, outer=0.25
        # Segment along +x at the ring height (y = ring), penetrating the tube.
        p1, p2, c = (-0.2, ring, 0.03), (0.2, ring, 0.03), (0.0, 0.0, 0.0)
        mp = osim.MusclePaths(
            self._wrap_model(p1, p2, c, 0.0, wtype="WrapTorus", inner_radius=inner, outer_radius=outer)
        )
        length = mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]
        ring_center = (0.0, ring, 0.0)
        expected = float(np.linalg.norm(np.array(p2) - np.array(p1))) + self._ref_extra(p1, p2, ring_center, tube)
        self.assertGreater(self._ref_extra(p1, p2, ring_center, tube), 0.0)
        self.assertAlmostEqual(length, expected, places=9)

    def test_non_penetrating_torus_leaves_length_unchanged(self):
        """A torus tube the straight path misses does not change the path length."""
        ring, tube = 0.2, 0.05
        inner, outer = ring - tube, ring + tube
        # Segment far above the ring plane, clearing the tube entirely.
        p1, p2, c = (-0.2, 0.6, 0.0), (0.2, 0.6, 0.0), (0.0, 0.0, 0.0)
        straight = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        mp = osim.MusclePaths(
            self._wrap_model(p1, p2, c, 0.0, wtype="WrapTorus", inner_radius=inner, outer_radius=outer)
        )
        self.assertAlmostEqual(mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0], straight, places=12)

    def test_path_wrap_range_limits_segments(self):
        """A PathWrap ``range`` restricts wrapping to its path-point span.

        A 3-point muscle penetrates a sphere on its first segment only. The
        wrap applies with the default whole-path range or a range covering the
        first segment, but a range covering only the (non-penetrating) second
        segment leaves the straight-line length unchanged.
        """
        p0, p1, p2, c, r = (-0.2, 0.05, 0.0), (0.2, 0.05, 0.0), (0.4, 0.05, 0.0), (0.0, 0.0, 0.0), 0.1

        def length(rng):
            model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
            model.muscles = [
                osim.OsimMuscle(
                    name="mw",
                    type="Thelen2003Muscle",
                    path_points=[
                        osim.OsimPathPoint(name="p0", body="ground", location=p0),
                        osim.OsimPathPoint(name="p1", body="ground", location=p1),
                        osim.OsimPathPoint(name="p2", body="ground", location=p2),
                    ],
                    wraps=[osim.OsimWrap(wrap_object="ws", range=rng)],
                )
            ]
            model.wrap_objects = [
                osim.OsimWrapObject(name="ws", type="WrapSphere", body="ground", translation=c, radius=r)
            ]
            mp = osim.MusclePaths(model)
            return mp.lengths(np.zeros((1, mp.fk.ncoord)))[0, 0]

        straight = float(np.linalg.norm(np.array(p1) - np.array(p0)) + np.linalg.norm(np.array(p2) - np.array(p1)))
        extra = self._ref_extra(p0, p1, c, r)
        self.assertGreater(extra, 0.0)
        self.assertAlmostEqual(length((-1, -1)), straight + extra, places=9)
        self.assertAlmostEqual(length((1, 2)), straight + extra, places=9)
        self.assertAlmostEqual(length((2, 3)), straight, places=12)

    def test_parse_wrap_object_set(self):
        """``parse_osim`` reads a body's WrapObjectSet into ``OsimWrapObject``s."""
        model = osim.parse_osim(_WRAP_OSIM)
        self.assertEqual(len(model.wrap_objects), 1)
        w = model.wrap_objects[0]
        self.assertEqual(w.name, "knee_wrap")
        self.assertEqual(w.type, "WrapSphere")
        self.assertEqual(w.body, "rod")
        self.assertAlmostEqual(w.radius, 0.05, places=12)
        self.assertTrue(w.active)
        self.assertEqual(model.muscles[0].wraps[0].wrap_object, "knee_wrap")

    def test_parse_ellipsoid_and_torus_dimensions(self):
        """``parse_osim`` reads WrapEllipsoid ``dimensions`` and WrapTorus radii."""
        xml = _WRAP_OSIM.replace(
            '<WrapSphere name="knee_wrap">\n'
            "                                <active>true</active>\n"
            "                                <translation>0 -0.3 0</translation>\n"
            "                                <xyz_body_rotation>0 0 0</xyz_body_rotation>\n"
            "                                <quadrant>all</quadrant>\n"
            "                                <radius>0.05</radius>\n"
            "                            </WrapSphere>",
            '<WrapEllipsoid name="knee_wrap">\n'
            "                                <active>true</active>\n"
            "                                <translation>0 -0.3 0</translation>\n"
            "                                <xyz_body_rotation>0 0 0</xyz_body_rotation>\n"
            "                                <dimensions>0.04 0.06 0.05</dimensions>\n"
            "                            </WrapEllipsoid>\n"
            '                            <WrapTorus name="tor_wrap">\n'
            "                                <active>true</active>\n"
            "                                <inner_radius>0.15</inner_radius>\n"
            "                                <outer_radius>0.25</outer_radius>\n"
            "                            </WrapTorus>",
        )
        model = osim.parse_osim(xml)
        wobj = {w.name: w for w in model.wrap_objects}
        ell = wobj["knee_wrap"]
        self.assertEqual(ell.type, "WrapEllipsoid")
        self.assertEqual(tuple(round(v, 12) for v in ell.dimensions), (0.04, 0.06, 0.05))
        tor = wobj["tor_wrap"]
        self.assertEqual(tor.type, "WrapTorus")
        self.assertAlmostEqual(tor.inner_radius, 0.15, places=12)
        self.assertAlmostEqual(tor.outer_radius, 0.25, places=12)


class TestMuscleForces(unittest.TestCase):
    """Rigid-tendon muscle forces and muscle-generated generalized forces."""

    def _two_muscle_model(self):
        """Attach two straight-line De Groote-Fregly muscles to the double pendulum."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        b0, b1 = fk.body_names[1], fk.body_names[2]
        params = {
            "max_isometric_force": 500.0,
            "optimal_fiber_length": 0.1,
            "tendon_slack_length": 0.15,
            "pennation_angle_at_optimal": 0.1,
            "max_contraction_velocity": 10.0,
        }
        model.muscles = [
            osim.OsimMuscle(
                name="m_ab",
                type="DeGrooteFregly2016Muscle",
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=(0.1, 0.3, 0.0)),
                    osim.OsimPathPoint(name="b", body=b0, location=(0.2, 0.05, 0.0)),
                ],
                params=dict(params),
            ),
            osim.OsimMuscle(
                name="m_cd",
                type="DeGrooteFregly2016Muscle",
                path_points=[
                    osim.OsimPathPoint(name="c", body=b0, location=(0.4, 0.1, 0.0)),
                    osim.OsimPathPoint(name="d", body=b1, location=(0.15, -0.05, 0.0)),
                ],
                params=dict(params),
            ),
        ]
        return model

    def test_muscle_tendon_velocity(self):
        """Match muscle-tendon velocity to the finite-difference time derivative of length.

        The lengthening velocity ``V_MT = -sum_i r_i * qdot_i`` must equal
        ``d/dt L_MT`` evaluated by central differences along ``q(t) = q0 + t*qdot``.
        """
        model = self._two_muscle_model()
        mp = osim.MusclePaths(model)
        coords = np.array([[0.3, -0.4], [-0.2, 0.5]])
        speeds = np.array([[1.5, -0.7], [-1.1, 0.9]])
        vmt = mp.velocities(coords, speeds)
        dt = 1e-6
        vfd = (mp.lengths(coords + dt * speeds) - mp.lengths(coords - dt * speeds)) / (2 * dt)
        np.testing.assert_allclose(vmt, vfd, atol=1e-8)

    def test_generalized_force_moment_arm_projection(self):
        """Project muscle forces onto coordinates as ``tau_i = sum_m r_{m,i} F_m``.

        The generalized forces must equal the moment-arm projection of the
        rigid-tendon muscle forces, and the forces themselves must be finite and
        non-negative.
        """
        model = self._two_muscle_model()
        mf = osim.MuscleForces(model)
        coords = np.array([[0.3, -0.4]])
        speeds = np.array([[0.5, -0.2]])
        acts = np.array([[0.8, 0.4]])
        forces = mf.forces(acts, coords, speeds)
        r = mf.paths.moment_arms(coords)
        tau = mf.generalized_forces(acts, coords, speeds)
        np.testing.assert_allclose(tau, np.einsum("bmc,bm->bc", r, forces), atol=1e-9)
        self.assertTrue(np.all(np.isfinite(forces)) and np.all(forces >= 0.0))

    def test_force_pipeline_keeps_geometry_on_device(self):
        """Keep path geometry on device until the final muscle result is copied."""
        mf = osim.MuscleForces(self._two_muscle_model())
        coords = np.array([[0.3, -0.4], [-0.2, 0.5]])
        speeds = np.array([[0.5, -0.2], [-0.3, 0.4]])
        activations = np.array([0.8, 0.4])
        expected_forces = mf.forces(activations, coords, speeds)
        expected_tau = mf.generalized_forces(activations, coords, speeds)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("muscle force pipeline called a host-returning path wrapper")

        mf.paths.lengths = reject_host_wrapper
        mf.paths.velocities = reject_host_wrapper
        np.testing.assert_allclose(mf.forces(activations, coords, speeds), expected_forces)
        np.testing.assert_allclose(mf.generalized_forces(activations, coords, speeds), expected_tau)

    def test_affine_coefficients_share_device_geometry(self):
        """Share one device geometry pass across moment arms and affine force coefficients."""
        mf = osim.MuscleForces(self._two_muscle_model())
        coords = np.array([[0.3, -0.4], [-0.2, 0.5]])
        speeds = np.array([[0.5, -0.2], [-0.3, 0.4]])
        expected_r = mf.paths.moment_arms(coords)
        f0 = mf.forces(np.zeros(2), coords, speeds)
        f1 = mf.forces(np.ones(2), coords, speeds)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("affine coefficient pipeline called a host-returning geometry wrapper")

        mf.paths.moment_arms = reject_host_wrapper
        mf.forces = reject_host_wrapper
        moment_arms, active, passive = mf._affine_coefficients(coords, speeds)
        np.testing.assert_allclose(moment_arms, expected_r)
        np.testing.assert_allclose(passive, f0)
        np.testing.assert_allclose(active, f1 - f0)

    def test_analysis_quantities_share_device_geometry(self):
        """Share one device geometry pass across muscle-analysis kinematics and forces."""
        mf = osim.MuscleForces(self._two_muscle_model())
        coords = np.array([[0.3, -0.4], [-0.2, 0.5]])
        speeds = np.array([[0.5, -0.2], [-0.3, 0.4]])
        activations = np.array([[0.8, 0.4], [0.2, 0.9]])
        expected_length = mf.paths.lengths(coords)
        expected_r = mf.paths.moment_arms(coords)
        expected_kinematics = mf.fiber_kinematics(coords, speeds)
        expected_forces = mf.fiber_forces(activations, coords, speeds)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("muscle analysis called a host-returning geometry wrapper")

        mf.paths.lengths = reject_host_wrapper
        mf.paths.moment_arms = reject_host_wrapper
        mf.fiber_kinematics = reject_host_wrapper
        mf.fiber_forces = reject_host_wrapper
        length, moment_arms, quantities = mf._analysis_quantities(activations, coords, speeds)
        np.testing.assert_allclose(length, expected_length)
        np.testing.assert_allclose(moment_arms, expected_r)
        for key, expected in expected_kinematics.items():
            np.testing.assert_allclose(quantities[key], expected)
        for key, expected in expected_forces.items():
            np.testing.assert_allclose(quantities[key], expected)

    def test_rigid_tendon_force_matches_degroote_fregly(self):
        """Reproduce the De Groote-Fregly (2016) rigid-tendon force definition.

        The Warp muscle-force kernel must match an independent NumPy evaluation of
        the published active/passive force-length, force-velocity and pennation
        formulas, and force must increase monotonically with activation.
        """
        model = self._two_muscle_model()
        mf = osim.MuscleForces(model)
        coords = np.array([[0.25, 0.35], [-0.15, -0.45]])
        speeds = np.array([[0.4, -0.3], [-0.6, 0.2]])
        acts = np.array([[0.7, 0.5], [0.9, 0.2]])
        forces = mf.forces(acts, coords, speeds)
        lmt = mf.paths.lengths(coords)
        vmt = mf.paths.velocities(coords, speeds)

        bcoef = [
            (0.814483478343008, 1.055033428970575, 0.162384573599574, 0.063303448465465),
            (0.433004984392647, 0.716775413397760, -0.029947116970696, 0.200356847296188),
            (0.100, 1.000, 0.354, 0.000),
        ]
        d1, d2, d3, d4, kpe, e0 = -0.318, -8.149, -0.374, 0.886, 4.0, 0.6

        def fref(a, lm, vm, p):
            lopt, ltsl = p["optimal_fiber_length"], p["tendon_slack_length"]
            fmax, vmax = p["max_isometric_force"], p["max_contraction_velocity"]
            cospo = np.cos(p["pennation_angle_at_optimal"])
            h = lopt * np.sqrt(max(1.0 - cospo * cospo, 0.0))
            along = lm - ltsl
            lfib = np.sqrt(max(along * along + h * h, 1e-18))
            lnorm = lfib / lopt
            cospen = max((lm - ltsl) / max(lfib, 1e-9), 0.0)
            vn = (vm * cospen) / max(lopt * vmax, 1e-9)
            fal = sum(b1 * np.exp(-0.5 * (lnorm - b2) ** 2 / (b3 + b4 * lnorm) ** 2) for b1, b2, b3, b4 in bcoef)
            fv = d1 * np.arcsinh(d2 * vn + d3) + d4
            fpe = max((np.exp(kpe * (lnorm - 1.0) / e0) - 1.0) / (np.exp(kpe) - 1.0), 0.0)
            return max(fmax * (a * fal * fv + fpe) * cospen, 0.0)

        ref = np.array(
            [
                [fref(acts[b, j], lmt[b, j], vmt[b, j], m.params) for j, m in enumerate(model.muscles)]
                for b in range(coords.shape[0])
            ]
        )
        np.testing.assert_allclose(forces, ref, atol=1e-2, rtol=1e-4)

        pose = coords[:1]
        lo = mf.forces(np.array([[0.1, 0.1]]), pose)
        hi = mf.forces(np.array([[0.9, 0.9]]), pose)
        self.assertTrue(np.all(hi >= lo - 1e-6))


_SLIDER_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="cart">
    <gravity>0 -9.80665 0</gravity>
    <BodySet name="bodyset"><objects>
      <Body name="cart"><mass>1.0</mass><mass_center>0 0 0</mass_center>
        <inertia>0 0 0 0 0 0</inertia></Body>
    </objects></BodySet>
    <JointSet name="jointset"><objects>
      <SliderJoint name="slider">
        <socket_parent_frame>ground_offset</socket_parent_frame>
        <socket_child_frame>cart_offset</socket_child_frame>
        <coordinates><objects>
          <Coordinate name="tx"><default_value>0</default_value>
            <range>-10 10</range><clamped>false</clamped></Coordinate>
        </objects></coordinates>
        <frames>
          <PhysicalOffsetFrame name="ground_offset"><socket_parent>/ground</socket_parent>
            <translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="cart_offset"><socket_parent>/bodyset/cart</socket_parent>
            <translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
        </frames>
      </SliderJoint>
    </objects></JointSet>
  </Model>
</OpenSimDocument>
"""


class TestMuscleFiberForces(unittest.TestCase):
    """Rigid-tendon muscle force breakdown (active/passive/fiber/tendon) from a Warp kernel."""

    @staticmethod
    def _model():
        """Two ground muscles (one un-pennated, one at 15 deg) near optimal fiber length."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        # lmt = 0.255, lt_slack = 0.15 -> along = 0.105 ~= 1.05 * l_opt (mild stretch).
        model.muscles = [
            osim.OsimMuscle(
                name=f"m{i}",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name=f"a{i}", body="ground", location=(0.0, 0.0, 0.0)),
                    osim.OsimPathPoint(name=f"b{i}", body="ground", location=(0.255, 0.0, 0.0)),
                ],
                params={
                    "max_isometric_force": 500.0,
                    "optimal_fiber_length": 0.10,
                    "tendon_slack_length": 0.15,
                    "pennation_angle_at_optimal": penn,
                },
            )
            for i, penn in enumerate((0.0, np.deg2rad(15.0)))
        ]
        return osim.MuscleForces(model)

    def test_tendon_force_matches_total(self):
        """The tendon-force component equals the validated total muscle force."""
        mf = self._model()
        q = np.zeros((1, mf.paths.fk.ncoord))
        a = np.array([[0.5, 0.5]])
        fb = mf.fiber_forces(a, q)
        np.testing.assert_allclose(fb["tendon_force"], mf.forces(a, q), atol=1e-3)
        np.testing.assert_allclose(fb["fiber_force"], fb["active_fiber_force"] + fb["passive_fiber_force"], atol=1e-3)

    def test_active_scales_with_activation_passive_does_not(self):
        """Active fiber force is linear in activation; passive is activation-independent."""
        mf = self._model()
        q = np.zeros((1, mf.paths.fk.ncoord))
        f0 = mf.fiber_forces(np.array([[0.0, 0.0]]), q)
        fh = mf.fiber_forces(np.array([[0.5, 0.5]]), q)
        f1 = mf.fiber_forces(np.array([[1.0, 1.0]]), q)
        np.testing.assert_allclose(f0["active_fiber_force"], 0.0, atol=1e-4)
        np.testing.assert_allclose(fh["active_fiber_force"], 0.5 * f1["active_fiber_force"], atol=1e-3)
        np.testing.assert_allclose(f0["passive_fiber_force"], f1["passive_fiber_force"], atol=1e-6)
        self.assertGreater(float(f1["passive_fiber_force"].min()), 0.0)

    def test_pennation_reduces_tendon_relative_to_fiber(self):
        """An un-pennated fiber puts its full force on the tendon; pennation reduces it."""
        mf = self._model()
        q = np.zeros((1, mf.paths.fk.ncoord))
        fb = mf.fiber_forces(np.array([[1.0, 1.0]]), q)
        # m0 (penn=0): tendon == fiber; m1 (penn>0): tendon < fiber.
        self.assertAlmostEqual(fb["tendon_force"][0, 0], fb["fiber_force"][0, 0], places=3)
        self.assertLess(fb["tendon_force"][0, 1], fb["fiber_force"][0, 1])


class TestMuscleFiberKinematics(unittest.TestCase):
    """Rigid-tendon fiber kinematics (length, pennation, velocity) from a Warp kernel."""

    @staticmethod
    def _ground_muscle(name, penn, l_opt, lt_slack, vmax, x1):
        """A two-point ground muscle with a fixed (pose-independent) path length."""
        return osim.OsimMuscle(
            name=name,
            type="Thelen2003Muscle",
            path_points=[
                osim.OsimPathPoint(name=name + "a", body="ground", location=(0.0, 0.0, 0.0)),
                osim.OsimPathPoint(name=name + "b", body="ground", location=(x1, 0.0, 0.0)),
            ],
            params={
                "optimal_fiber_length": l_opt,
                "tendon_slack_length": lt_slack,
                "pennation_angle_at_optimal": penn,
                "max_contraction_velocity": vmax,
            },
        )

    def test_fiber_length_and_pennation_match_closed_form(self):
        """Kernel fiber length and pennation match the constant-width geometry.

        For a rigid tendon the along-tendon fiber projection is ``lmt - lt_slack``
        and the fiber width ``h = l_opt sin(penn_opt)`` is constant, so the fiber
        length is ``sqrt((lmt - lt_slack)^2 + h^2)``.
        """
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        params = [(0.0, 0.10, 0.15, 10.0, 0.30), (np.deg2rad(20.0), 0.12, 0.20, 8.0, 0.40)]
        model.muscles = [self._ground_muscle(f"m{i}", *p) for i, p in enumerate(params)]
        mf = osim.MuscleForces(model)

        lmt = np.array([0.30, 0.40])
        l_opt = np.array([0.10, 0.12])
        lt_slack = np.array([0.15, 0.20])
        penn_opt = np.array([0.0, np.deg2rad(20.0)])
        h = l_opt * np.sin(penn_opt)
        along = lmt - lt_slack
        lm_ref = np.sqrt(along**2 + h**2)
        penn_ref = np.arccos(np.clip(along / lm_ref, 0.0, 1.0))

        fk = mf.fiber_kinematics(np.zeros((1, mf.paths.fk.ncoord)))
        np.testing.assert_allclose(fk["fiber_length"][0], lm_ref, atol=1e-6)
        np.testing.assert_allclose(fk["normalized_fiber_length"][0], lm_ref / l_opt, atol=1e-6)
        np.testing.assert_allclose(fk["pennation_angle"][0], penn_ref, atol=1e-6)
        np.testing.assert_allclose(fk["normalized_fiber_velocity"][0], 0.0, atol=1e-9)

    def test_fiber_velocity_projects_muscle_tendon_velocity(self):
        """Normalized fiber velocity is the pennation-projected, normalized MT velocity."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        rod = model.bodies[1].name if len(model.bodies) > 1 else "ground"
        model.muscles = [
            osim.OsimMuscle(
                name="mp",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="o", body="ground", location=(0.1, 0.0, 0.0)),
                    osim.OsimPathPoint(name="i", body=rod, location=(0.0, -0.1, 0.0)),
                ],
                params={"optimal_fiber_length": 0.1, "tendon_slack_length": 0.1, "max_contraction_velocity": 10.0},
            )
        ]
        mf = osim.MuscleForces(model)
        nc = mf.paths.fk.ncoord
        q = np.zeros((1, nc))
        qd = np.zeros((1, nc))
        qd[0, 0] = 0.5
        lmt = mf.paths.lengths(q)[0, 0]
        vmt = mf.paths.velocities(q, qd)[0, 0]
        cos_penn = max((lmt - 0.1) / np.sqrt((lmt - 0.1) ** 2), 0.0)
        v_norm_ref = vmt * cos_penn / (0.1 * 10.0)
        fk = mf.fiber_kinematics(q, qd)
        self.assertAlmostEqual(fk["normalized_fiber_velocity"][0, 0], v_norm_ref, places=5)


_PATH_SPRING_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
    <Model name="ps">
        <gravity>0 -9.80665 0</gravity>
        <ground name="ground"/>
        <ForceSet><objects>
            <PathSpring name="ps0">
                <resting_length>0.42</resting_length>
                <stiffness>1234.0</stiffness>
                <dissipation>0.7</dissipation>
                <GeometryPath>
                    <PathPointSet><objects>
                        <PathPoint name="p1"><location>0 0.1 0</location>
                            <socket_parent_frame>/ground</socket_parent_frame></PathPoint>
                        <PathPoint name="p2"><location>0.2 0 0</location>
                            <socket_parent_frame>/ground</socket_parent_frame></PathPoint>
                    </objects></PathPointSet>
                </GeometryPath>
            </PathSpring>
        </objects></ForceSet>
    </Model>
</OpenSimDocument>"""


_P2P_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
    <Model name="p2p">
        <gravity>0 -9.80665 0</gravity>
        <ground name="ground"/>
        <ForceSet><objects>
            <PointToPointSpring name="p2p0">
                <socket_body1>/ground</socket_body1>
                <socket_body2>/bodyset/link2</socket_body2>
                <point1>0.1 0.2 0.3</point1>
                <point2>0.4 0.5 0.6</point2>
                <stiffness>321.0</stiffness>
                <rest_length>0.12</rest_length>
            </PointToPointSpring>
        </objects></ForceSet>
    </Model>
</OpenSimDocument>"""


_BUSHING_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
    <Model name="bush">
        <ground name="ground"/>
        <ForceSet><objects>
            <BushingForce name="bush0">
                <socket_frame1>frame1</socket_frame1>
                <socket_frame2>frame2</socket_frame2>
                <frames>
                    <PhysicalOffsetFrame name="frame1">
                        <socket_parent>/ground</socket_parent>
                        <translation>0.01 0.02 0.03</translation>
                        <orientation>0.1 0.2 0.3</orientation>
                    </PhysicalOffsetFrame>
                    <PhysicalOffsetFrame name="frame2">
                        <socket_parent>/bodyset/link2</socket_parent>
                        <translation>0.04 0.05 0.06</translation>
                        <orientation>-0.1 0.0 0.2</orientation>
                    </PhysicalOffsetFrame>
                </frames>
                <rotational_stiffness>10 20 30</rotational_stiffness>
                <translational_stiffness>100 200 300</translational_stiffness>
                <rotational_damping>1 2 3</rotational_damping>
                <translational_damping>4 5 6</translational_damping>
            </BushingForce>
        </objects></ForceSet>
    </Model>
</OpenSimDocument>"""


class TestBushingForce(unittest.TestCase):
    """Elastic 6-DOF frame-bushing potential energy and generalized forces (Warp kernel)."""

    _KR = (30.0, 15.0, 22.0)
    _KT = (200.0, 120.0, 80.0)
    _TF1 = ((0.05, 0.1, -0.02), (0.1, -0.2, 0.05))
    _TF2 = ((-0.03, 0.04, 0.06), (-0.15, 0.08, 0.2))

    def _bushing(self):
        """Double pendulum with a link1-to-link2 offset-frame bushing."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.bushing_forces = [
            osim.OsimBushingForce(
                name="bush",
                body1="link1",
                body2="link2",
                frame1_transform=osim.OsimTransform(translation=self._TF1[0], orientation=self._TF1[1]),
                frame2_transform=osim.OsimTransform(translation=self._TF2[0], orientation=self._TF2[1]),
                rotational_stiffness=self._KR,
                translational_stiffness=self._KT,
            )
        ]
        return osim.BushingForces(model)

    def _energy(self, bf, q):
        """Independent quadratic deflection potential from validated forward kinematics."""
        t1 = make_transform(euler_xyz_to_matrix(*self._TF1[1]), self._TF1[0])
        t2 = make_transform(euler_xyz_to_matrix(*self._TF2[1]), self._TF2[0])
        b1 = bf.fk.body_names.index("link1")
        b2 = bf.fk.body_names.index("link2")
        x = bf.fk.body_transforms_batch(q)
        out = np.zeros(q.shape[0])
        for k in range(q.shape[0]):
            xf1 = x[k, b1] @ t1
            xf2 = x[k, b2] @ t2
            rrel = xf1[:3, :3].T @ xf2[:3, :3]
            d = xf1[:3, :3].T @ (xf2[:3, 3] - xf1[:3, 3])
            th = np.array(
                [
                    np.arctan2(-rrel[1, 2], rrel[2, 2]),
                    np.arcsin(np.clip(rrel[0, 2], -1.0, 1.0)),
                    np.arctan2(-rrel[0, 1], rrel[0, 0]),
                ]
            )
            out[k] = 0.5 * (np.dot(self._KR, th * th) + np.dot(self._KT, d * d))
        return out

    def test_potential_energy_matches_independent_potential(self):
        """Per-bushing potential energy equals the independent quadratic deflection potential."""
        bf = self._bushing()
        q = np.array([[0.3, -0.4], [-0.2, 0.5], [0.6, 0.1]])
        np.testing.assert_allclose(bf.potential_energy(q)[:, 0], self._energy(bf, q), atol=1e-10)

    def test_generalized_forces_match_potential_gradient(self):
        """Generalized forces equal -dU/dq of the deflection potential."""
        bf = self._bushing()
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        tau = bf.generalized_forces(q)
        eps = 1e-6
        ref = np.zeros_like(tau)
        for i in range(q.shape[1]):
            qp = q.copy()
            qp[:, i] += eps
            qm = q.copy()
            qm[:, i] -= eps
            ref[:, i] = -(self._energy(bf, qp) - self._energy(bf, qm)) / (2 * eps)
        np.testing.assert_allclose(tau, ref, atol=1e-6)

    def test_coincident_frames_have_zero_load(self):
        """A bushing whose frames coincide has zero energy and zero generalized force."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        zero = osim.OsimTransform()
        model.bushing_forces = [
            osim.OsimBushingForce(
                name="z",
                body1="link2",
                body2="link2",
                frame1_transform=zero,
                frame2_transform=zero,
                rotational_stiffness=self._KR,
                translational_stiffness=self._KT,
            )
        ]
        bf = osim.BushingForces(model)
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        self.assertLess(np.abs(bf.potential_energy(q)).max(), 1e-12)
        self.assertLess(np.abs(bf.generalized_forces(q)).max(), 1e-9)

    def test_parser_reads_bushing_force(self):
        """parse_osim reads BushingForce frames, bodies, and stiffness/damping vectors."""
        model = osim.parse_osim(_BUSHING_OSIM)
        self.assertEqual(len(model.bushing_forces), 1)
        bush = model.bushing_forces[0]
        self.assertEqual(bush.name, "bush0")
        self.assertEqual((bush.body1, bush.body2), ("ground", "link2"))
        self.assertEqual(bush.frame1_transform.translation, (0.01, 0.02, 0.03))
        self.assertEqual(bush.frame2_transform.orientation, (-0.1, 0.0, 0.2))
        self.assertEqual(bush.rotational_stiffness, (10.0, 20.0, 30.0))
        self.assertEqual(bush.translational_stiffness, (100.0, 200.0, 300.0))
        self.assertEqual(bush.rotational_damping, (1.0, 2.0, 3.0))
        self.assertEqual(bush.translational_damping, (4.0, 5.0, 6.0))


class TestSpringGeneralizedForce(unittest.TestCase):
    """Passive single-coordinate spring-damper generalized forces (Warp kernel)."""

    @staticmethod
    def _coords():
        """Coordinate names of the double pendulum in model order."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        return [c.name for j in model.joints for c in j.coordinates]

    def test_force_law_with_damping_and_accumulation(self):
        """Force is -k*(q-rest) - v*qd, and springs on one coordinate accumulate."""
        cn = self._coords()
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.spring_generalized_forces = [
            osim.OsimSpringGeneralizedForce(name="a", coordinate=cn[0], stiffness=50.0, rest_length=0.1, viscosity=2.0),
            osim.OsimSpringGeneralizedForce(
                name="b", coordinate=cn[0], stiffness=20.0, rest_length=-0.2, viscosity=0.0
            ),
            osim.OsimSpringGeneralizedForce(name="c", coordinate=cn[1], stiffness=80.0, rest_length=0.3, viscosity=1.5),
        ]
        sg = osim.SpringGeneralizedForces(model)
        q = np.array([[0.4, -0.3], [-0.1, 0.6]])
        qd = np.array([[1.0, -0.5], [-0.8, 0.2]])
        tau = sg.generalized_forces(q, qd)
        expect = np.zeros_like(tau)
        expect[:, 0] = (-50.0 * (q[:, 0] - 0.1) - 2.0 * qd[:, 0]) + (-20.0 * (q[:, 0] + 0.2))
        expect[:, 1] = -80.0 * (q[:, 1] - 0.3) - 1.5 * qd[:, 1]
        np.testing.assert_allclose(tau, expect, atol=1e-9)

    def test_no_speeds_drops_damping(self):
        """Omitting speeds evaluates the pure spring term."""
        cn = self._coords()
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.spring_generalized_forces = [
            osim.OsimSpringGeneralizedForce(name="a", coordinate=cn[1], stiffness=80.0, rest_length=0.3, viscosity=5.0),
        ]
        sg = osim.SpringGeneralizedForces(model)
        q = np.array([[0.4, -0.3]])
        tau = sg.generalized_forces(q)
        self.assertAlmostEqual(tau[0, 1], -80.0 * (-0.3 - 0.3))
        self.assertAlmostEqual(tau[0, 0], 0.0)

    def test_parser_reads_spring_generalized_force(self):
        """parse_osim reads SpringGeneralizedForce coordinate, stiffness, rest, viscosity."""
        xml = (
            '<?xml version="1.0"?><OpenSimDocument Version="40000"><Model name="m">'
            '<ground name="ground"/><ForceSet><objects>'
            '<SpringGeneralizedForce name="sg"><coordinate>/jointset/j/q1</coordinate>'
            "<stiffness>77.0</stiffness><rest_length>0.25</rest_length><viscosity>3.0</viscosity>"
            "</SpringGeneralizedForce></objects></ForceSet></Model></OpenSimDocument>"
        )
        model = osim.parse_osim(xml)
        self.assertEqual(len(model.spring_generalized_forces), 1)
        sg = model.spring_generalized_forces[0]
        self.assertEqual(sg.name, "sg")
        self.assertEqual(sg.coordinate, "q1")
        self.assertAlmostEqual(sg.stiffness, 77.0)
        self.assertAlmostEqual(sg.rest_length, 0.25)
        self.assertAlmostEqual(sg.viscosity, 3.0)


class TestPointToPointSpring(unittest.TestCase):
    """Two-point spring tensions and generalized forces (Warp kernel)."""

    @staticmethod
    def _model(stiffness=300.0, rest_length=0.4):
        """Double pendulum with a ground-to-link2 point-to-point spring."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.point_to_point_springs = [
            osim.OsimPointToPointSpring(
                name="p2p",
                body1="ground",
                body2="link2",
                point1=(0.2, 0.3, 0.1),
                point2=(0.15, -0.05, 0.0),
                stiffness=stiffness,
                rest_length=rest_length,
            )
        ]
        return model

    def _distances(self, p2p, q):
        """Ground-frame attachment-point separations from validated forward kinematics."""
        x = p2p.fk.body_transforms_batch(q)
        b1 = p2p.fk.body_names.index("ground")
        b2 = p2p.fk.body_names.index("link2")
        g1 = (x[:, b1] @ np.array([0.2, 0.3, 0.1, 1.0]))[:, :3]
        g2 = (x[:, b2] @ np.array([0.15, -0.05, 0.0, 1.0]))[:, :3]
        return np.linalg.norm(g2 - g1, axis=1)

    def test_tension_is_linear_in_stretch(self):
        """Tension is stiffness*(distance - rest_length) with the validated FK distance."""
        k, rest = 300.0, 0.4
        p2p = osim.PointToPointSprings(self._model(stiffness=k, rest_length=rest))
        q = np.array([[0.3, -0.4], [-0.2, 0.5], [0.6, 0.6]])
        expect = k * (self._distances(p2p, q) - rest)
        np.testing.assert_allclose(p2p.forces(q)[:, 0], expect, atol=1e-4)

    def test_generalized_forces_match_virtual_work(self):
        """Generalized forces equal the virtual-work gradient -f * d(distance)/dq_i."""
        k, rest = 300.0, 0.4
        p2p = osim.PointToPointSprings(self._model(stiffness=k, rest_length=rest))
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        f = p2p.forces(q)[:, 0]
        tau = p2p.generalized_forces(q)
        eps = 1e-6
        ref = np.zeros_like(tau)
        for i in range(q.shape[1]):
            qp = q.copy()
            qp[:, i] += eps
            qm = q.copy()
            qm[:, i] -= eps
            dddq = (self._distances(p2p, qp) - self._distances(p2p, qm)) / (2 * eps)
            ref[:, i] = -f * dddq
        np.testing.assert_allclose(tau, ref, atol=1e-3)

    def test_parser_reads_point_to_point_spring(self):
        """parse_osim reads PointToPointSpring bodies, points, stiffness, and rest length."""
        model = osim.parse_osim(_P2P_OSIM)
        self.assertEqual(len(model.point_to_point_springs), 1)
        spring = model.point_to_point_springs[0]
        self.assertEqual(spring.name, "p2p0")
        self.assertEqual((spring.body1, spring.body2), ("ground", "link2"))
        self.assertEqual(spring.point1, (0.1, 0.2, 0.3))
        self.assertEqual(spring.point2, (0.4, 0.5, 0.6))
        self.assertAlmostEqual(spring.stiffness, 321.0)
        self.assertAlmostEqual(spring.rest_length, 0.12)


class TestPathSpring(unittest.TestCase):
    """Path-spring tensions and generalized forces (Warp kernel)."""

    @staticmethod
    def _model(resting_length=0.25, stiffness=500.0, dissipation=0.5):
        """Double pendulum with a ground-to-link2 path spring."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.path_springs = [
            osim.OsimPathSpring(
                name="ps",
                resting_length=resting_length,
                stiffness=stiffness,
                dissipation=dissipation,
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=(0.30, 0.10, 0.0)),
                    osim.OsimPathPoint(name="b", body="link2", location=(0.20, 0.0, 0.0)),
                ],
                wraps=[],
            )
        ]
        return model

    def test_force_law_and_clamp(self):
        """Static tension is the clamped linear-elastic law k*max(L-L0, 0)."""
        k, l0 = 500.0, 0.25
        ps = osim.PathSpringForces(self._model(resting_length=l0, stiffness=k))
        q = np.array([[0.3, -0.4], [-0.2, 0.5], [0.0, 0.0]])
        expect = k * np.maximum(ps.paths.lengths(q) - l0, 0.0)
        np.testing.assert_allclose(ps.forces(q), expect, atol=1e-3)
        # A resting length above every path length leaves the spring slack.
        slack = osim.PathSpringForces(self._model(resting_length=10.0, stiffness=k))
        self.assertEqual(np.abs(slack.forces(q)).max(), 0.0)

    def test_dissipation_couples_to_lengthening_rate(self):
        """Dynamic tension scales the elastic force by (1 + dissipation * Ldot)."""
        k, l0, d = 500.0, 0.25, 0.5
        ps = osim.PathSpringForces(self._model(resting_length=l0, stiffness=k, dissipation=d))
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        qd = np.array([[1.0, -0.5], [-0.8, 0.3]])
        ldot = ps.paths.velocities(q, qd)
        expect = np.maximum(k * np.maximum(ps.paths.lengths(q) - l0, 0.0) * (1.0 + d * ldot), 0.0)
        np.testing.assert_allclose(ps.forces(q, qd), expect, atol=1e-3)

    def test_generalized_forces_match_virtual_work(self):
        """Generalized forces equal the virtual-work gradient -sum_s F_s dL_s/dq_i."""
        ps = osim.PathSpringForces(self._model())
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        qd = np.array([[1.0, -0.5], [-0.8, 0.3]])
        f = ps.forces(q, qd)
        tau = ps.generalized_forces(q, qd)
        eps = 1e-6
        ref = np.zeros_like(tau)
        for i in range(q.shape[1]):
            qp = q.copy()
            qp[:, i] += eps
            qm = q.copy()
            qm[:, i] -= eps
            dldq = (ps.paths.lengths(qp) - ps.paths.lengths(qm)) / (2 * eps)
            ref[:, i] = -(f * dldq).sum(axis=1)
        np.testing.assert_allclose(tau, ref, atol=1e-4)

    def test_force_pipeline_keeps_path_geometry_on_device(self):
        """Keep path-spring lengths and velocities on device until final results are copied."""
        ps = osim.PathSpringForces(self._model())
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        qd = np.array([[1.0, -0.5], [-0.8, 0.3]])
        expected_force = ps.forces(q, qd)
        expected_tau = ps.generalized_forces(q, qd)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("path-spring pipeline called a host-returning path wrapper")

        ps.paths.lengths = reject_host_wrapper
        ps.paths.velocities = reject_host_wrapper
        np.testing.assert_allclose(ps.forces(q, qd), expected_force)
        np.testing.assert_allclose(ps.generalized_forces(q, qd), expected_tau)

    def test_parser_reads_path_spring(self):
        """parse_osim reads PathSpring stiffness, resting length, dissipation, and path."""
        model = osim.parse_osim(_PATH_SPRING_OSIM)
        self.assertEqual(len(model.path_springs), 1)
        spring = model.path_springs[0]
        self.assertEqual(spring.name, "ps0")
        self.assertAlmostEqual(spring.resting_length, 0.42)
        self.assertAlmostEqual(spring.stiffness, 1234.0)
        self.assertAlmostEqual(spring.dissipation, 0.7)
        self.assertEqual([p.name for p in spring.path_points], ["p1", "p2"])


_LIGAMENT_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
    <Model name="lig">
        <gravity>0 -9.80665 0</gravity>
        <ground name="ground"/>
        <ForceSet><objects>
            <Ligament name="lig0">
                <resting_length>0.33</resting_length>
                <pcsa_force>987.0</pcsa_force>
                <GeometryPath>
                    <PathPointSet><objects>
                        <PathPoint name="p1"><location>0 0.1 0</location>
                            <socket_parent_frame>/ground</socket_parent_frame></PathPoint>
                        <PathPoint name="p2"><location>0.2 0 0</location>
                            <socket_parent_frame>/ground</socket_parent_frame></PathPoint>
                    </objects></PathPointSet>
                </GeometryPath>
                <force_length_curve>
                    <PiecewiseLinearFunction>
                        <x>1.0 1.5 2.0</x>
                        <y>0.0 0.5 1.0</y>
                    </PiecewiseLinearFunction>
                </force_length_curve>
            </Ligament>
        </objects></ForceSet>
    </Model>
</OpenSimDocument>"""


class TestLigament(unittest.TestCase):
    """Ligament tensions and generalized forces (Warp kernel)."""

    _CX = (1.0, 2.0, 3.0)
    _CY = (0.0, 0.3, 1.0)

    @classmethod
    def _model(cls, resting_length=0.30, pcsa=800.0, curve=None):
        """Double pendulum with a ground-to-link2 ligament."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.ligaments = [
            osim.OsimLigament(
                name="lig",
                resting_length=resting_length,
                pcsa_force=pcsa,
                force_length_curve=curve or {"type": "PiecewiseLinearFunction", "x": cls._CX, "y": cls._CY},
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=(0.30, 0.10, 0.0)),
                    osim.OsimPathPoint(name="b", body="link2", location=(0.20, 0.0, 0.0)),
                ],
                wraps=[],
            )
        ]
        return model

    def test_force_law_scales_normalized_curve(self):
        """Tension is pcsa_force times the curve sampled at length / resting_length."""
        L0, pcsa = 0.30, 800.0
        lg = osim.LigamentForces(self._model(resting_length=L0, pcsa=pcsa))
        q = np.array([[0.3, -0.4], [-0.2, 0.5], [0.6, 0.6]])
        norm = lg.paths.lengths(q) / L0
        expect = pcsa * np.interp(norm.ravel(), self._CX, self._CY).reshape(norm.shape)
        np.testing.assert_allclose(lg.forces(q), expect, atol=1e-3)
        # The sampled lengths exercise the sloped interpolation region, not just a flat tail.
        self.assertGreater(np.ptp(lg.forces(q)), 1.0)

    def test_below_resting_length_is_slack(self):
        """A resting length above every path length yields zero tension."""
        lg = osim.LigamentForces(self._model(resting_length=10.0))
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        self.assertEqual(np.abs(lg.forces(q)).max(), 0.0)

    def test_generalized_forces_match_virtual_work(self):
        """Generalized forces equal the virtual-work gradient -sum_g F_g dL_g/dq_i."""
        lg = osim.LigamentForces(self._model())
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        f = lg.forces(q)
        tau = lg.generalized_forces(q)
        eps = 1e-6
        ref = np.zeros_like(tau)
        for i in range(q.shape[1]):
            qp = q.copy()
            qp[:, i] += eps
            qm = q.copy()
            qm[:, i] -= eps
            dldq = (lg.paths.lengths(qp) - lg.paths.lengths(qm)) / (2 * eps)
            ref[:, i] = -(f * dldq).sum(axis=1)
        np.testing.assert_allclose(tau, ref, atol=1e-3)

    def test_force_pipeline_keeps_path_geometry_on_device(self):
        """Keep ligament lengths on device until final results are copied."""
        lg = osim.LigamentForces(self._model())
        q = np.array([[0.3, -0.4], [-0.2, 0.5]])
        expected_force = lg.forces(q)
        expected_tau = lg.generalized_forces(q)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("ligament pipeline called a host-returning path wrapper")

        lg.paths.lengths = reject_host_wrapper
        np.testing.assert_allclose(lg.forces(q), expected_force)
        np.testing.assert_allclose(lg.generalized_forces(q), expected_tau)

    def test_simmspline_curve_matches_host_spline(self):
        """A SimmSpline force-length curve matches the host SimmSpline evaluation."""
        L0, pcsa = 0.30, 800.0
        sx, sy = [1.0, 2.0, 2.5, 3.0], [0.0, 0.3, 0.6, 1.2]
        lg = osim.LigamentForces(
            self._model(resting_length=L0, pcsa=pcsa, curve={"type": "SimmSpline", "x": sx, "y": sy})
        )
        q = np.array([[0.3, -0.4], [-0.2, 0.5], [0.6, 0.6]])
        norm = lg.paths.lengths(q) / L0
        sp = SimmSpline(np.array(sx), np.array(sy))
        expect = pcsa * np.array([sp.value(v) for v in norm.ravel()]).reshape(norm.shape)
        np.testing.assert_allclose(lg.forces(q), expect, atol=1e-2)

    def test_parser_reads_ligament(self):
        """parse_osim reads Ligament resting length, pcsa force, curve, and path."""
        model = osim.parse_osim(_LIGAMENT_OSIM)
        self.assertEqual(len(model.ligaments), 1)
        lig = model.ligaments[0]
        self.assertEqual(lig.name, "lig0")
        self.assertAlmostEqual(lig.resting_length, 0.33)
        self.assertAlmostEqual(lig.pcsa_force, 987.0)
        self.assertEqual(lig.force_length_curve["type"], "PiecewiseLinearFunction")
        self.assertEqual([p.name for p in lig.path_points], ["p1", "p2"])


class TestMuscleElasticTendon(unittest.TestCase):
    """Isometric elastic-tendon equilibrium force from a Warp bisection kernel."""

    # Exact De Groote-Fregly (2016) constants mirrored from ``muscle.py`` so the
    # independent NumPy solver posts the same force balance the kernel solves.
    _B = (
        0.814483478343008,
        1.055033428970575,
        0.162384573599574,
        0.063303448465465,
        0.433004984392647,
        0.716775413397760,
        -0.029947116970696,
        0.200356847296188,
        0.100,
        1.000,
        0.354,
        0.000,
    )

    @classmethod
    def _fal(cls, l):
        """De Groote-Fregly active force-length multiplier (NumPy mirror)."""
        b = cls._B
        g1 = b[0] * np.exp(-0.5 * (l - b[1]) ** 2 / (b[2] + b[3] * l) ** 2)
        g2 = b[4] * np.exp(-0.5 * (l - b[5]) ** 2 / (b[6] + b[7] * l) ** 2)
        g3 = b[8] * np.exp(-0.5 * (l - b[9]) ** 2 / (b[10] + b[11] * l) ** 2)
        return g1 + g2 + g3

    @staticmethod
    def _fpe(l):
        """De Groote-Fregly passive force-length multiplier (NumPy mirror)."""
        return max((np.exp(4.0 * (l - 1.0) / 0.6) - 1.0) / (np.exp(4.0) - 1.0), 0.0)

    @staticmethod
    def _ft(ltn, kt):
        """De Groote-Fregly normalized tendon force (NumPy mirror)."""
        return 0.200 * np.exp(kt * (ltn - 0.995)) - 0.250

    @classmethod
    def _solve(cls, a, lmt, l_opt, lt_slack, cos_penn_opt, kt):
        """Independent NumPy bisection for the equilibrium tendon force [normalized]."""
        sinp = np.sqrt(max(1.0 - cos_penn_opt**2, 0.0))
        lo, hi = sinp + 1e-6, np.sqrt((lmt / l_opt) ** 2 + sinp**2) - 1e-6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            along = l_opt * np.sqrt(max(mid**2 - sinp**2, 0.0))
            cosp = along / (mid * l_opt)
            fiber_t = (a * cls._fal(mid) + cls._fpe(mid)) * cosp
            if cls._ft((lmt - along) / lt_slack, kt) - fiber_t > 0.0:
                lo = mid
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        along = l_opt * np.sqrt(max(mid**2 - sinp**2, 0.0))
        return max(cls._ft((lmt - along) / lt_slack, kt), 0.0)

    @staticmethod
    def _model(lmt, l_opt, lt_slack, penn, e_t, fmax=500.0):
        """A single ground muscle whose two path points span ``lmt``."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.muscles = [
            osim.OsimMuscle(
                name="m",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=(0.0, 0.0, 0.0)),
                    osim.OsimPathPoint(name="b", body="ground", location=(lmt, 0.0, 0.0)),
                ],
                params={
                    "max_isometric_force": fmax,
                    "optimal_fiber_length": l_opt,
                    "tendon_slack_length": lt_slack,
                    "pennation_angle_at_optimal": penn,
                    "tendon_strain_at_one_norm_force": e_t,
                },
            )
        ]
        return osim.MuscleForces(model), fmax

    def test_matches_independent_solver(self):
        """The kernel equilibrium force matches an independent NumPy bisection solver."""
        l_opt, lt_slack, penn, e_t = 0.10, 0.20, np.deg2rad(12.0), 0.049
        kt = np.log(1.25 / 0.2) / (1.0 + e_t - 0.995)
        cpo = np.cos(penn)
        for lmt, a in [(0.302, 0.6), (0.315, 0.9), (0.308, 0.2), (0.33, 1.0)]:
            mf, fmax = self._model(lmt, l_opt, lt_slack, penn, e_t)
            q = np.zeros((1, mf.paths.fk.ncoord))
            got = mf.forces_elastic_tendon(np.array([[a]]), q)[0, 0]
            expect = fmax * self._solve(a, lmt, l_opt, lt_slack, cpo, kt)
            self.assertAlmostEqual(got, expect, delta=0.05)

    def test_round_trip_recovers_constructed_equilibrium(self):
        """A muscle built at a known equilibrium reports that equilibrium's tendon force."""
        l_opt, lt_slack, penn, e_t, a = 0.10, 0.20, np.deg2rad(12.0), 0.049, 0.6
        kt = np.log(1.25 / 0.2) / (1.0 + e_t - 0.995)
        sinp = np.sin(penn)
        for lstar in (1.02, 1.10, 0.97):
            along = l_opt * np.sqrt(lstar**2 - sinp**2)
            cosp = along / (lstar * l_opt)
            fmt = (a * self._fal(lstar) + self._fpe(lstar)) * cosp
            ltn = 0.995 + np.log((fmt + 0.25) / 0.2) / kt
            lmt = along + ltn * lt_slack
            mf, fmax = self._model(lmt, l_opt, lt_slack, penn, e_t)
            q = np.zeros((1, mf.paths.fk.ncoord))
            got = mf.forces_elastic_tendon(np.array([[a]]), q)[0, 0]
            self.assertAlmostEqual(got, fmax * fmt, delta=0.05)

    def test_elastic_differs_from_rigid(self):
        """Tendon compliance shifts the fiber operating point, changing the force."""
        mf, _ = self._model(0.315, 0.10, 0.20, np.deg2rad(12.0), 0.049)
        q = np.zeros((1, mf.paths.fk.ncoord))
        a = np.array([[0.8]])
        self.assertGreater(abs(mf.forces_elastic_tendon(a, q)[0, 0] - mf.forces(a, q)[0, 0]), 1.0)


class TestMuscleElasticTendonDynamics(unittest.TestCase):
    """Compliant-tendon fiber velocity from the series force balance (Warp kernel)."""

    _B = TestMuscleElasticTendon._B

    @classmethod
    def _fal(cls, l):
        """De Groote-Fregly active force-length multiplier (NumPy mirror)."""
        return TestMuscleElasticTendon._fal(l)

    @staticmethod
    def _fpe(l):
        """De Groote-Fregly passive force-length multiplier (NumPy mirror)."""
        return TestMuscleElasticTendon._fpe(l)

    @staticmethod
    def _fv(v):
        """De Groote-Fregly force-velocity multiplier (NumPy mirror)."""
        d1, d2, d3, d4 = -0.318, -8.149, -0.374, 0.886
        arg = d2 * v + d3
        return d1 * np.log(arg + np.sqrt(arg * arg + 1.0)) + d4

    @staticmethod
    def _model(lmt, l_opt, lt_slack, penn, e_t, vmax, fmax=600.0):
        """A single ground muscle whose two path points span ``lmt``."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.muscles = [
            osim.OsimMuscle(
                name="m",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=(0.0, 0.0, 0.0)),
                    osim.OsimPathPoint(name="b", body="ground", location=(lmt, 0.0, 0.0)),
                ],
                params={
                    "max_isometric_force": fmax,
                    "optimal_fiber_length": l_opt,
                    "tendon_slack_length": lt_slack,
                    "pennation_angle_at_optimal": penn,
                    "tendon_strain_at_one_norm_force": e_t,
                    "max_contraction_velocity": vmax,
                },
            )
        ]
        return osim.MuscleForces(model), fmax

    def test_round_trip_recovers_fiber_velocity_and_force(self):
        """A muscle built at a known fiber length and velocity reports both back."""
        l_opt, lt_slack, penn, e_t, vmax = 0.10, 0.20, np.deg2rad(10.0), 0.049, 8.0
        kt = np.log(1.25 / 0.2) / (1.0 + e_t - 0.995)
        sinp = np.sin(penn)
        for a, lstar, vstar in [(0.7, 1.05, -0.3), (0.9, 0.95, 0.2), (0.4, 1.10, -0.6), (1.0, 1.0, 0.0)]:
            lm = lstar * l_opt
            along = l_opt * np.sqrt(lstar**2 - sinp**2)
            cos_cur = along / lm
            fm_norm = a * self._fal(lstar) * self._fv(vstar) + self._fpe(lstar)
            ft = fm_norm * cos_cur
            ltn = 0.995 + np.log((ft + 0.25) / 0.2) / kt
            lmt = ltn * lt_slack + along
            mf, fmax = self._model(lmt, l_opt, lt_slack, penn, e_t, vmax)
            q = np.zeros((1, mf.paths.fk.ncoord))
            out = mf.elastic_tendon_fiber_velocity(np.array([[a]]), q, np.array([[lm]]))
            self.assertAlmostEqual(out["fiber_velocity"][0, 0], vstar * vmax * l_opt, delta=3e-3)
            self.assertAlmostEqual(out["tendon_force"][0, 0], fmax * ft, delta=0.05)

    def test_equilibrium_state_has_zero_fiber_velocity(self):
        """At the isometric-equilibrium fiber length the fiber velocity vanishes.

        The tendon force there also matches the independent equilibrium solve of
        :meth:`~newton.opensim.MuscleForces.forces_elastic_tendon`.
        """
        l_opt, lt_slack, penn, e_t, vmax, a = 0.10, 0.20, np.deg2rad(10.0), 0.049, 8.0, 0.8
        kt = np.log(1.25 / 0.2) / (1.0 + e_t - 0.995)
        sinp, lstar = np.sin(penn), 1.03
        lm = lstar * l_opt
        along = l_opt * np.sqrt(lstar**2 - sinp**2)
        cos_cur = along / lm
        ft = (a * self._fal(lstar) + self._fpe(lstar)) * cos_cur  # v* = 0 -> fv = 1
        lmt = (0.995 + np.log((ft + 0.25) / 0.2) / kt) * lt_slack + along
        mf, _fmax = self._model(lmt, l_opt, lt_slack, penn, e_t, vmax)
        q = np.zeros((1, mf.paths.fk.ncoord))
        out = mf.elastic_tendon_fiber_velocity(np.array([[a]]), q, np.array([[lm]]))
        self.assertAlmostEqual(out["fiber_velocity"][0, 0], 0.0, delta=3e-3)
        self.assertAlmostEqual(out["tendon_force"][0, 0], mf.forces_elastic_tendon(np.array([[a]]), q)[0, 0], delta=0.1)


class TestMuscleActivation(unittest.TestCase):
    """First-order muscle activation dynamics integrated on-device."""

    @staticmethod
    def _model(taus):
        """A ground model whose muscles carry the given (tau_act, tau_deact) pairs."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        model.muscles = [
            osim.OsimMuscle(
                name=f"m{i}",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name=f"a{i}", body="ground", location=(0.0, 0.0, 0.0)),
                    osim.OsimPathPoint(name=f"b{i}", body="ground", location=(0.1, 0.0, 0.0)),
                ],
                params={"activation_time_constant": ta, "deactivation_time_constant": td},
            )
            for i, (ta, td) in enumerate(taus)
        ]
        return osim.MuscleForces(model)

    @staticmethod
    def _reference(a0, u, ta, td, dt, fine=20000):
        """Dependency-free dense forward-Euler integration of the same ODE."""
        a = float(a0)
        h = dt / fine
        for _ in range(fine):
            diff = u - a
            tau = ta * (0.5 + 1.5 * a) if diff > 0.0 else td / (0.5 + 1.5 * a)
            a += h * diff / tau
        return a

    def test_activation_matches_reference_ode(self):
        """Kernel RK4 activation trajectory matches a dense independent integration."""
        taus = [(0.010, 0.040), (0.020, 0.060)]
        mf = self._model(taus)
        a0 = np.array([0.05, 0.80])
        u = np.array([0.90, 0.10])  # one ramps up, one ramps down
        dt, nsteps = 0.005, 40
        a_k = a0.reshape(1, 2).copy()
        a_ref = a0.copy()
        for _ in range(nsteps):
            a_k = mf.integrate_activation(a_k, u.reshape(1, 2), dt, substeps=16)
            a_ref = np.array([self._reference(a_ref[j], u[j], *taus[j], dt) for j in range(2)])
        self.assertLess(float(np.abs(a_k.ravel() - a_ref).max()), 5.0e-5)

    def test_activation_reaches_excitation(self):
        """Under a constant excitation the activation converges to that excitation."""
        mf = self._model([(0.010, 0.040), (0.020, 0.060)])
        u = np.array([0.90, 0.10])
        a = mf.integrate_activation(np.array([0.2, 0.9]), u, dt=2.0, substeps=400)
        np.testing.assert_allclose(a.ravel(), u, atol=5.0e-6)

    def test_activation_faster_than_deactivation(self):
        """A shorter activation time constant makes activation outrun deactivation."""
        mf = self._model([(0.010, 0.040)])
        up = mf.integrate_activation(np.array([0.1]), np.array([0.6]), dt=0.02, substeps=32).ravel()[0]
        down = mf.integrate_activation(np.array([0.6]), np.array([0.1]), dt=0.02, substeps=32).ravel()[0]
        self.assertGreater(up - 0.1, 0.6 - down)

    def test_activation_batch_broadcast(self):
        """A 1-D excitation broadcasts across a batch of initial activations."""
        mf = self._model([(0.010, 0.040), (0.020, 0.060)])
        out = mf.integrate_activation(
            np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]), np.array([0.5, 0.5]), dt=0.01, substeps=8
        )
        self.assertEqual(out.shape, (3, 2))
        # Rows starting below 0.5 rise, the row above 0.5 falls, toward 0.5.
        self.assertTrue(np.all(out[0] > 0.1) and np.all(out[0] < 0.5))
        self.assertTrue(np.all(out[2] < 0.9) and np.all(out[2] > 0.5))


class TestDirectCollocation(unittest.TestCase):
    """Direct-collocation trajectory optimization against analytic benchmarks.

    Reproduces OpenSim Moco's analytic optimal-control tests
    (``testMocoAnalytic.cpp``) and the sliding-mass tutorial, and drives the
    Warp forward-dynamics engine through the collocation solver.
    """

    def test_point_jacobian_batches_all_perturbations(self):
        """Evaluate every state and control perturbation in one batched dynamics call."""
        calls = 0

        def dynamics(t, x, u):
            nonlocal calls
            calls += 1
            return np.stack([2.0 * x[:, 0] - x[:, 1] + 3.0 * u[:, 0], x[:, 0] + 4.0 * u[:, 0]], axis=1)

        t = np.array([0.0, 0.5, 1.0])
        x = np.array([[0.1, 0.2], [0.3, -0.4], [-0.2, 0.5]])
        u = np.array([[0.4], [-0.1], [0.2]])
        fx, fu = _point_jacobian(dynamics, t, x, u, 2, 1)
        self.assertEqual(calls, 1)
        np.testing.assert_allclose(fx, np.broadcast_to([[2.0, -1.0], [1.0, 0.0]], fx.shape))
        np.testing.assert_allclose(fu, np.broadcast_to([[3.0], [4.0]], fu.shape))

    def test_covector_hessian_batches_all_perturbations(self):
        """Evaluate every Hessian perturbation in one batched dynamics call."""
        calls = 0

        def dynamics(t, x, u):
            nonlocal calls
            calls += 1
            return np.stack([x[:, 0] ** 2 + x[:, 1] * u[:, 0], 2.0 * x[:, 0] * u[:, 0] + u[:, 0] ** 2], axis=1)

        hessian = _point_covector_hessian(
            dynamics,
            0.4,
            np.array([0.2, -0.3]),
            np.array([0.5]),
            np.array([0.5, -0.3]),
            2,
            1,
        )
        self.assertEqual(calls, 1)
        np.testing.assert_allclose(hessian, [[1.0, 0.0, -0.6], [0.0, 0.0, 0.5], [-0.6, 0.5, -0.6]], atol=1.0e-8)

    def test_double_integrator_minimum_effort(self):
        """Recover the analytic minimum-effort double-integrator trajectory."""
        prob = OptimalControlProblem(
            2,
            1,
            dynamics=lambda t, x, u: np.stack([x[:, 1], u[:, 0]], axis=1),
            initial_state=[0.0, 0.0],
            final_state=[1.0, 0.0],
            integral_cost=lambda t, x, u: 0.5 * u[:, 0] ** 2,
            time_initial=0.0,
            time_final=1.0,
        )
        sol = solve_optimal_control(prob, 40)
        self.assertTrue(sol.converged)
        t = sol.time
        x_analytic = np.stack([3 * t**2 - 2 * t**3, 6 * t - 6 * t**2], axis=1)
        u_analytic = 6 - 12 * t
        self.assertLess(np.max(np.abs(sol.states - x_analytic)), 1e-6)
        self.assertLess(np.max(np.abs(sol.controls[:, 0] - u_analytic)), 1e-5)
        self.assertLess(sol.constraint_violation, 1e-9)

    def test_kirk_second_order_minimum_effort(self):
        """Match Kirk's analytic second-order linear minimum-effort solution.

        This is the ``testMocoAnalytic`` second-order linear minimum-effort
        benchmark; OpenSim asserts a 1e-5 match to the closed-form solution.
        """
        e2, em2 = np.exp(2.0), np.exp(-2.0)
        amat = np.array(
            [[-2 - 0.5 * em2 + 0.5 * e2, 1 - 0.5 * em2 - 0.5 * e2], [-1 + 0.5 * em2 + 0.5 * e2, 0.5 * em2 - 0.5 * e2]]
        )
        c2, c3 = np.linalg.solve(amat, np.array([5.0, 2.0]))
        prob = OptimalControlProblem(
            2,
            1,
            dynamics=lambda t, x, u: np.stack([x[:, 1], x[:, 1] + u[:, 0]], axis=1),
            initial_state=[0.0, 0.0],
            final_state=[5.0, 2.0],
            integral_cost=lambda t, x, u: 0.5 * u[:, 0] ** 2,
            time_initial=0.0,
            time_final=2.0,
        )
        sol = solve_optimal_control(prob, 50)
        t = sol.time
        x0 = c2 * (-t - 0.5 * np.exp(-t) + 0.5 * np.exp(t)) + c3 * (1 - 0.5 * np.exp(-t) - 0.5 * np.exp(t))
        x1 = c2 * (-1 + 0.5 * np.exp(-t) + 0.5 * np.exp(t)) + c3 * (0.5 * np.exp(-t) - 0.5 * np.exp(t))
        self.assertLess(np.max(np.abs(sol.states - np.stack([x0, x1], axis=1))), 1e-5)

    def test_linear_tangent_steering(self):
        """Match the analytic linear-tangent-steering optimal control.

        A point mass under constant thrust ``a`` steers to a target height while
        maximizing final horizontal speed; the optimal thrust angle is
        ``atan(tan(theta0) - c t)`` (Bryson & Ho / ``testMocoAnalytic``).
        """
        thrust, tf, hgt = 5.0, 1.0, 1.0

        def residual(ang):
            secx = 1.0 / np.cos(ang)
            tanx = np.tan(ang)
            return (
                1.0 / np.sin(ang)
                - np.log((secx + tanx) / (secx - tanx)) / (2.0 * tanx * tanx)
                - 4.0 * hgt / (thrust * tf * tf)
            )

        lo, hi = 0.01, 0.99 * 0.5 * np.pi
        flo = residual(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = residual(mid)
            if abs(fm) < 1e-13 or (hi - lo) < 1e-13:
                break
            if (flo < 0) == (fm < 0):
                lo, flo = mid, fm
            else:
                hi = mid
        theta0 = 0.5 * (lo + hi)
        cc = 2.0 * np.tan(theta0) / tf

        prob = OptimalControlProblem(
            4,
            1,
            dynamics=lambda t, x, u: np.stack(
                [x[:, 2], x[:, 3], thrust * np.cos(u[:, 0]), thrust * np.sin(u[:, 0])], axis=1
            ),
            initial_state=[0.0, 0.0, 0.0, 0.0],
            final_state=[None, hgt, None, 0.0],
            endpoint_cost=lambda xN: -xN[2],
            time_initial=0.0,
            time_final=tf,
        )
        sol = solve_optimal_control(prob, 100, control_guess=lambda t: 1.0 - 2.0 * t / tf, max_iterations=60)
        self.assertTrue(sol.converged)
        angle_analytic = np.arctan(np.tan(theta0) - cc * sol.time)
        self.assertLess(np.max(np.abs(sol.controls[:, 0] - angle_analytic)), 1e-3)

    def test_slider_model_forward_dynamics(self):
        """Drive the Warp forward-dynamics engine through the collocation solver.

        A ``SliderJoint`` unit point mass is torque-driven so its dynamics reduce
        to the double integrator; the collocation solution -- with accelerations
        from :class:`ForwardDynamics` -- must recover the analytic cubic.
        """
        model = osim.parse_osim(_SLIDER_OSIM)
        dyn = create_torque_driven_dynamics(model, gains=np.array([1.0]), device="cpu")
        prob = OptimalControlProblem(
            2,
            1,
            dynamics=dyn,
            initial_state=[0.0, 0.0],
            final_state=[1.0, 0.0],
            integral_cost=lambda t, x, u: 0.5 * u[:, 0] ** 2,
            time_initial=0.0,
            time_final=1.0,
        )
        solver = DirectCollocationSolver(num_mesh_intervals=30, exact_hessian=False, max_iterations=25)
        sol = solver.solve(prob)
        t = sol.time
        x_analytic = np.stack([3 * t**2 - 2 * t**3, 6 * t - 6 * t**2], axis=1)
        self.assertLess(np.max(np.abs(sol.states - x_analytic)), 1e-6)
        self.assertLess(sol.constraint_violation, 1e-8)

    def test_minimum_time_sliding_mass(self):
        """Reproduce OpenSim Moco's minimum-time sliding-mass benchmark.

        Matches ``testMocoInterface.cpp``: a 10 kg point mass with a coordinate
        actuator bounded to [-10, 10] moves from position 0 to 1 (zero end
        speeds) in the shortest time. The bounded acceleration is 1 m/s^2, so the
        bang-bang optimum is a full-thrust / full-brake switch with final time 2 s.
        """
        prob = OptimalControlProblem(
            2,
            1,
            dynamics=lambda t, x, u: np.stack([x[:, 1], u[:, 0] / 10.0], axis=1),
            initial_state=[0.0, 0.0],
            final_state=[1.0, 0.0],
            state_bounds=[(0.0, 1.0), (-100.0, 100.0)],
            control_bounds=(-10.0, 10.0),
            final_time_bounds=(0.01, 10.0),
            minimize_final_time=True,
            time_final=3.0,
        )
        solver = DirectCollocationSolver(num_mesh_intervals=30, exact_hessian=False)
        sol = solver.solve(prob)
        self.assertAlmostEqual(sol.time[-1], 2.0, places=3)
        self.assertLess(sol.constraint_violation, 1e-6)
        self.assertLess(np.max(np.abs(sol.states[-1] - np.array([1.0, 0.0]))), 1e-5)
        u = sol.controls[:, 0]
        self.assertLess(np.max(np.abs(u)), 10.0 + 1e-6)
        self.assertGreater(u[: len(u) // 2 - 1].mean(), 9.9)
        self.assertLess(u[len(u) // 2 + 1 :].mean(), -9.9)

    def test_control_bounded_double_integrator(self):
        """Enforce control box bounds on a fixed-horizon double integrator.

        With ``|u| <= 4`` the double integrator can only just reach unit distance
        in unit time from rest to rest, so the unique feasible (hence optimal)
        control is the bang-bang ``+4 / -4`` switch at the midpoint; a wide bound
        instead recovers the smooth analytic minimum-effort cubic.
        """
        dyn = lambda t, x, u: np.stack([x[:, 1], u[:, 0]], axis=1)  # noqa: E731
        effort = lambda t, x, u: 0.5 * u[:, 0] ** 2  # noqa: E731
        prob = OptimalControlProblem(
            2,
            1,
            dynamics=dyn,
            initial_state=[0.0, 0.0],
            final_state=[1.0, 0.0],
            integral_cost=effort,
            control_bounds=(-4.0, 4.0),
            time_final=1.0,
        )
        sol = DirectCollocationSolver(num_mesh_intervals=40, exact_hessian=False).solve(prob)
        u = sol.controls[:, 0]
        self.assertLess(np.max(np.abs(u)), 4.0 + 1e-6)
        self.assertGreater(u[:18].mean(), 3.9)
        self.assertLess(u[22:].mean(), -3.9)
        self.assertLess(np.max(np.abs(sol.states[-1] - np.array([1.0, 0.0]))), 1e-5)

        wide = OptimalControlProblem(
            2,
            1,
            dynamics=dyn,
            initial_state=[0.0, 0.0],
            final_state=[1.0, 0.0],
            integral_cost=effort,
            control_bounds=(-50.0, 50.0),
            time_final=1.0,
        )
        sol_wide = DirectCollocationSolver(num_mesh_intervals=40, exact_hessian=False).solve(wide)
        t = sol_wide.time
        self.assertLess(np.max(np.abs(sol_wide.states[:, 0] - (3 * t**2 - 2 * t**3))), 1e-6)

    def test_path_constraint_tracking(self):
        """Force a state to track a prescribed reference with an equality path constraint.

        A single first-order system with dynamics ``adot = e - a`` is driven so an
        equality path constraint ``a - aref(t) = 0`` pins the state to a sinusoidal
        reference; the state must match the reference exactly and the control must
        equal ``aref + aref'`` (the mechanism behind an inverse muscle problem for a
        fully-determined coordinate).
        """
        w = 2.0 * np.pi

        def aref(t):
            return 0.5 + 0.3 * np.sin(w * t)

        def arefdot(t):
            return 0.3 * w * np.cos(w * t)

        def dyn(t, x, u):
            return u - x

        def effort(t, x, u):
            return u[:, 0] ** 2

        def path(t, x, u):
            return (x[:, 0] - aref(t))[:, None]

        prob = OptimalControlProblem(
            1,
            1,
            dynamics=dyn,
            initial_state=[None],
            final_state=[None],
            integral_cost=effort,
            control_bounds=(-5.0, 5.0),
            state_bounds=(0.0, 1.0),
            path_constraints=path,
        )
        sol = DirectCollocationSolver(num_mesh_intervals=40, tolerance=1e-10).solve(
            prob, control_guess=lambda t: np.array([0.5])
        )
        self.assertTrue(sol.converged)
        self.assertLess(sol.constraint_violation, 1e-9)
        t = sol.time
        self.assertLess(np.max(np.abs(sol.states[:, 0] - aref(t))), 1e-9)
        self.assertLess(np.max(np.abs(sol.controls[:, 0] - (aref(t) + arefdot(t)))), 3e-3)

    def test_path_constraint_muscle_sharing(self):
        """Resolve a redundant muscle-sharing problem to the least-norm solution.

        Two activation states share a prescribed net moment through an equality
        path constraint ``r1 a1 + r2 a2 = M``; minimizing the integral of squared
        excitation must recover the analytic least-norm excitations
        ``e_i = r_i M / (r1^2 + r2^2)``, the miniature of an inverse muscle problem
        with more muscles than degrees of freedom.
        """
        r1, r2, m_net = 1.0, 2.0, 1.0
        e1s = r1 * m_net / (r1**2 + r2**2)
        e2s = r2 * m_net / (r1**2 + r2**2)

        def dyn(t, x, u):
            return u - x

        def effort(t, x, u):
            return u[:, 0] ** 2 + u[:, 1] ** 2

        def path(t, x, u):
            return (r1 * x[:, 0] + r2 * x[:, 1] - m_net)[:, None]

        prob = OptimalControlProblem(
            2,
            2,
            dynamics=dyn,
            initial_state=[None, None],
            final_state=[None, None],
            integral_cost=effort,
            control_bounds=(0.0, 1.0),
            state_bounds=(0.0, 1.0),
            path_constraints=path,
        )
        sol = DirectCollocationSolver(num_mesh_intervals=20, tolerance=1e-9).solve(
            prob, control_guess=lambda t: np.array([0.3, 0.3])
        )
        self.assertTrue(sol.converged)
        self.assertLess(sol.constraint_violation, 1e-9)
        self.assertLess(np.max(np.abs(sol.controls[:, 0] - e1s)), 1e-4)
        self.assertLess(np.max(np.abs(sol.controls[:, 1] - e2s)), 1e-4)
        self.assertAlmostEqual(sol.objective, e1s**2 + e2s**2, places=5)


def _quat_to_matrix(q):
    """Rotation matrix from a Warp ``transformf`` quaternion ``[x, y, z, w]``."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class TestMotionVisualizer(unittest.TestCase):
    """Verify Warp-native OpenSim motion visualization."""

    def test_geometry_fallback_prefers_matching_side_family(self):
        """Prefer r_name/l_name geometry so bilateral meshes use one file family."""
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                name: os.path.join(directory, name)
                for name in ("r_tibia.vtp", "tibia_r.vtp", "r_femur.vtp", "femur_r.vtp")
            }
            for path in paths.values():
                open(path, "w").close()

            tibia = _resolve_geometry_file("tibia.vtp", "tibia_r", directory)
            femur = _resolve_geometry_file("femur.vtp", "femur_r", directory)

            self.assertEqual(tibia, paths["r_tibia.vtp"])
            self.assertEqual(femur, paths["femur_r.vtp"])

    def _model_with_muscles(self):
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        b0, b1 = fk.body_names[1], fk.body_names[2]
        model.muscles = [
            osim.OsimMuscle(
                name="m_ab",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=(0.1, 0.3, 0.0)),
                    osim.OsimPathPoint(name="b", body=b0, location=(0.2, 0.05, 0.0)),
                ],
            ),
            osim.OsimMuscle(
                name="m_cd",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="c", body=b0, location=(0.4, 0.1, 0.0)),
                    osim.OsimPathPoint(name="d", body=b1, location=(0.15, -0.05, 0.0)),
                ],
            ),
        ]
        return model, fk, (b0, b1)

    def test_body_transforms_match_forward_kinematics(self):
        """The per-frame body transforms reproduce the OpenSim-exact FK poses."""
        model, fk, (b0, b1) = self._model_with_muscles()
        rng = np.random.default_rng(0)
        q = rng.uniform(-1.0, 1.0, size=(7, fk.ncoord))
        viz = osim.MotionVisualizer(model, q)
        names = [b0, b1]
        xforms = viz.body_transforms(names).numpy()
        ref = fk.body_transforms_batch(q)
        basis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        bidx = {n: i for i, n in enumerate(fk.body_names)}
        probe = np.array([0.13, -0.24, 0.31])
        for f in range(len(q)):
            for j, nm in enumerate(names):
                p = xforms[f, j, :3]
                rot = _quat_to_matrix(xforms[f, j, 3:7])
                m = ref[f, bidx[nm]]
                np.testing.assert_allclose(p, basis @ m[:3, 3], atol=1e-6)
                np.testing.assert_allclose(
                    rot @ probe + p,
                    basis @ (m[:3, :3] @ probe + m[:3, 3]),
                    atol=1e-6,
                )

    def test_body_transforms_can_preserve_native_opensim_y_up(self):
        """Allow explicit Y-up output for OpenSim-native validation workflows."""
        model, fk, (b0, _b1) = self._model_with_muscles()
        q = np.zeros((1, fk.ncoord))
        viz = osim.MotionVisualizer(model, q, up_axis=newton.Axis.Y)
        xform = viz.body_transforms([b0]).numpy()[0, 0]
        native = fk.body_transforms_batch(q)[0, fk.body_names.index(b0)]

        np.testing.assert_allclose(xform[:3], native[:3, 3], atol=1.0e-6)
        np.testing.assert_allclose(_quat_to_matrix(xform[3:7]), native[:3, :3], atol=1.0e-6)

    def test_muscle_segments_match_path_geometry(self):
        """Muscle line segments span the ground-space path points frame for frame."""
        model, fk, (b0, b1) = self._model_with_muscles()
        rng = np.random.default_rng(1)
        q = rng.uniform(-1.0, 1.0, size=(5, fk.ncoord))
        viz = osim.MotionVisualizer(model, q)
        self.assertEqual(viz.num_segments, 2)  # two 2-point muscles => one segment each
        starts = viz.muscle_starts.numpy()
        ends = viz.muscle_ends.numpy()
        transforms = fk.body_transforms_batch(q)
        bidx = {n: i for i, n in enumerate(fk.body_names)}

        basis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

        def world(body, loc, f):
            t = transforms[f, bidx[body]]
            return basis @ (t[:3, :3] @ np.asarray(loc) + t[:3, 3])

        seg_defs = [
            (("ground", (0.1, 0.3, 0.0)), (b0, (0.2, 0.05, 0.0))),
            ((b0, (0.4, 0.1, 0.0)), (b1, (0.15, -0.05, 0.0))),
        ]
        for f in range(len(q)):
            for s, ((pb, pl), (qb, ql)) in enumerate(seg_defs):
                np.testing.assert_allclose(starts[f, s], world(pb, pl, f), atol=1e-5)
                np.testing.assert_allclose(ends[f, s], world(qb, ql, f), atol=1e-5)

    def test_muscle_colors_track_normalized_length(self):
        """Muscle segment colors interpolate the colormap by normalized length.

        Both coordinates are exercised so each muscle-tendon length varies over
        the trajectory; the segment color must be the length-normalized blend of
        the colormap endpoints.
        """
        model, fk, _ = self._model_with_muscles()
        rng = np.random.default_rng(3)
        q = rng.uniform(-1.0, 1.0, size=(8, fk.ncoord))
        lo = (0.1, 0.2, 0.3)
        hi = (0.9, 0.7, 0.5)
        viz = osim.MotionVisualizer(model, q, muscle_color=(lo, hi))
        colors = viz.muscle_colors.numpy()
        lengths = osim.MusclePaths(model).lengths(q)  # matches the internal paths
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        # each 2-point muscle contributes exactly one segment, so segment == muscle
        for s in range(viz.num_segments):
            span = lengths[:, s].max() - lengths[:, s].min()
            self.assertGreater(span, 1e-6)
            t = (lengths[:, s] - lengths[:, s].min()) / span
            expected = lo[None, :] * (1 - t[:, None]) + hi[None, :] * t[:, None]
            np.testing.assert_allclose(colors[:, s, :], expected, atol=1e-6)

    def test_color_muscles_by_scalar_field(self):
        """``color_muscles_by`` maps a per-muscle scalar through the blue->red heatmap.

        A zero-valued muscle renders at the low (blue) colormap stop and a
        fully-active one at the high (red) stop, and passing ``times`` resamples a
        coarse field onto the visualizer frames.
        """
        model, fk, _ = self._model_with_muscles()
        q = np.zeros((5, fk.ncoord))
        viz = osim.MotionVisualizer(model, q)
        nmus = len(viz.muscle_names)
        # Two sample times: all-quiet then all-active; frame 0 -> blue, last -> red.
        values = np.zeros((2, nmus))
        values[1, :] = 1.0
        viz.color_muscles_by(values, times=np.array([viz.time[0], viz.time[-1]]))
        colors = viz.muscle_colors.numpy()
        blue = np.array([0.30, 0.32, 0.65])
        red = np.array([0.95, 0.10, 0.12])
        for s in range(viz.num_segments):
            np.testing.assert_allclose(colors[0, s, :], blue, atol=1e-6)
            np.testing.assert_allclose(colors[-1, s, :], red, atol=1e-6)
        # A single active muscle only recolors its own segments.
        values = np.zeros((viz.num_frames, nmus))
        values[:, 0] = 1.0
        viz.color_muscles_by(values, vmin=0.0, vmax=1.0)
        colors = viz.muscle_colors.numpy()
        for s in range(viz.num_segments):
            expected = red if viz._seg_muscle[s] == 0 else blue
            np.testing.assert_allclose(colors[0, s, :], expected, atol=1e-6)

    def test_color_muscles_by_rejects_bad_shape(self):
        """``color_muscles_by`` rejects a value array whose column count is wrong."""
        model, fk, _ = self._model_with_muscles()
        viz = osim.MotionVisualizer(model, np.zeros((3, fk.ncoord)))
        with self.assertRaises(ValueError):
            viz.color_muscles_by(np.zeros((3, len(viz.muscle_names) + 1)))

    def test_read_motion_converts_degrees(self):
        """``read_motion`` maps storage columns to model order and degrees to radians."""
        model = osim.parse_osim(MINIMAL_OSIM)
        names = [c.name for j in model.joints for c in j.coordinates]
        header = "motion\nnRows=2\nnColumns=2\ninDegrees=yes\nendheader\n"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.mot")
            with open(path, "w") as fh:
                fh.write(header)
                fh.write("time\t" + "\t".join(names) + "\n")
                fh.write("0.0\t90.0\n0.1\t180.0\n")
            time, coords = osim.read_motion(model, path)
        np.testing.assert_allclose(time, [0.0, 0.1])
        np.testing.assert_allclose(coords[:, 0], [np.pi / 2, np.pi])

    _QUAD_VTP = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        '    <Piece NumberOfPoints="4" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="1">\n'
        "      <Points>\n"
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
        "          0 0 0  1 0 0  1 1 0  0 1 0\n"
        "        </DataArray>\n"
        "      </Points>\n"
        "      <Polys>\n"
        '        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>\n'
        '        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>\n'
        "      </Polys>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )

    def _geometry_xml(self, body: str, file: str, scale=(2.0, 2.0, 2.0)) -> str:
        """A minimal OpenSim document attaching one display mesh to ``body``."""
        sx, sy, sz = scale
        return (
            "<OpenSimDocument><Model><BodySet><objects>"
            f'<Body name="{body}"><VisibleObject><GeometrySet><objects><DisplayGeometry>'
            f"<geometry_file>{file}</geometry_file>"
            "<transform>0 0 0 0 0 0</transform>"
            f"<scale_factors>{sx} {sy} {sz}</scale_factors>"
            "</DisplayGeometry></objects></GeometrySet></VisibleObject></Body>"
            "</objects></BodySet></Model></OpenSimDocument>"
        )

    def test_read_vtp_triangulates_polys(self):
        """``_read_vtp`` reads ASCII PolyData points and fan-triangulates polygons."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "quad.vtp")
            with open(path, "w") as fh:
                fh.write(self._QUAD_VTP)
            points, tris = _read_vtp(path)
        np.testing.assert_allclose(points, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        # one quad -> two triangles fanned from vertex 0
        self.assertEqual(tris.shape, (2, 3))
        np.testing.assert_array_equal(np.sort(tris, axis=1), [[0, 1, 2], [0, 2, 3]])

    def test_display_geometry_parses_file_and_scale(self):
        """``read_display_geometry`` maps a body to its geometry file, transform, and scale."""
        geom = osim.read_display_geometry(self._geometry_xml("pelvis", "pelvis.vtp", scale=(1.1, 1.2, 1.3)))
        self.assertIn("pelvis", geom)
        file, transform, scale = geom["pelvis"][0]
        self.assertEqual(file, "pelvis.vtp")
        np.testing.assert_allclose(transform, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(scale, [1.1, 1.2, 1.3])

    def test_load_meshes_skins_by_body_transform(self):
        """Loaded display meshes are scaled, then rigidly skinned by the body's FK pose."""
        model, fk, (b0, _b1) = self._model_with_muscles()
        rng = np.random.default_rng(4)
        q = rng.uniform(-1.0, 1.0, size=(3, fk.ncoord))
        viz = osim.MotionVisualizer(model, q)

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "quad.vtp"), "w") as fh:
                fh.write(self._QUAD_VTP)
            n = viz.load_meshes(self._geometry_xml(b0, "quad.vtp", scale=(2.0, 2.0, 2.0)), d)
            self.assertEqual(n, 1)
            self.assertEqual(viz.num_meshes, 1)

            mesh = viz._meshes[0]
            quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
            np.testing.assert_allclose(mesh["local"].numpy(), quad * 2.0, atol=1e-6)

            viewer = ViewerNull(num_frames=viz.num_frames)
            poses = viz._poses.numpy()
            body = viz.body_names.index(b0)
            for f in range(viz.num_frames):
                viz.render_meshes(viewer, f)  # skins into mesh["world"] and logs
                world = mesh["world"].numpy()
                x = poses[f, body]
                expected = (quad * 2.0) @ x[:3, :3].T + x[:3, 3]
                np.testing.assert_allclose(world, expected, atol=1e-5)

    def test_render_smoke_null_viewer(self):
        """A full playback + render pass runs against a headless viewer."""
        model, fk, _ = self._model_with_muscles()
        q = np.zeros((4, fk.ncoord))
        q[:, 0] = np.linspace(-0.5, 0.5, 4)
        viz = osim.MotionVisualizer(model, q)

        builder = newton.ModelBuilder()
        osim.add_osim(builder, model, parse_muscles=False, parse_contacts=False)
        nmodel = builder.finalize()
        state = nmodel.state()
        frames = viz.body_transforms(nmodel.body_label)

        viewer = ViewerNull(num_frames=viz.num_frames)
        viewer.set_model(nmodel)
        for f in range(viz.num_frames):
            wp.copy(state.body_q, frames[f])
            viewer.begin_frame(float(f))
            viewer.log_state(state)
            viz.render_skeleton(viewer, f)
            viz.render_muscles(viewer, f)
            viewer.end_frame()
        self.assertTrue(np.all(np.isfinite(frames.numpy())))


# --------------------------------------------------------------------------- #
# Compliant contact forces (SmoothSphereHalfSpaceForce / HuntCrossleyForce /
# ElasticFoundationForce). The Warp kernels are validated against exact NumPy
# ports of the Simbody force laws they reproduce.
# --------------------------------------------------------------------------- #
_CONTACT_MODEL_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="20302">
  <Model name="dropper">
    <gravity>0 -9.80665 0</gravity>
    <BodySet><objects>
      <Body name="ground"><mass>0</mass><mass_center>0 0 0</mass_center>
        <inertia_xx>0</inertia_xx><inertia_yy>0</inertia_yy><inertia_zz>0</inertia_zz>
        <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz><Joint/></Body>
      <Body name="foot">
        <mass>2.0</mass><mass_center>0 0 0</mass_center>
        <inertia_xx>0.01</inertia_xx><inertia_yy>0.01</inertia_yy><inertia_zz>0.01</inertia_zz>
        <inertia_xy>0</inertia_xy><inertia_xz>0</inertia_xz><inertia_yz>0</inertia_yz>
        <Joint>
          <CustomJoint name="slide">
            <SpatialTransform>
              <TransformAxis name="rotation1"><coordinates></coordinates><axis>0 0 1</axis>
                <function><Constant><value>0</value></Constant></function></TransformAxis>
              <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis>
                <function><Constant><value>0</value></Constant></function></TransformAxis>
              <TransformAxis name="rotation3"><coordinates></coordinates><axis>1 0 0</axis>
                <function><Constant><value>0</value></Constant></function></TransformAxis>
              <TransformAxis name="translation1"><coordinates>tx</coordinates><axis>1 0 0</axis>
                <function><LinearFunction><coefficients>1 0</coefficients></LinearFunction></function></TransformAxis>
              <TransformAxis name="translation2"><coordinates>ty</coordinates><axis>0 1 0</axis>
                <function><LinearFunction><coefficients>1 0</coefficients></LinearFunction></function></TransformAxis>
              <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis>
                <function><Constant><value>0</value></Constant></function></TransformAxis>
            </SpatialTransform>
            <parent_body>ground</parent_body>
            <location_in_parent>0 0 0</location_in_parent><orientation_in_parent>0 0 0</orientation_in_parent>
            <location>0 0 0</location><orientation>0 0 0</orientation>
            <CoordinateSet><objects>
              <Coordinate name="tx"><motion_type>translational</motion_type>
                <default_value>0</default_value><range>-5 5</range></Coordinate>
              <Coordinate name="ty"><motion_type>translational</motion_type>
                <default_value>0.1</default_value><range>-5 5</range></Coordinate>
            </objects></CoordinateSet>
          </CustomJoint>
        </Joint>
      </Body>
    </objects></BodySet>
    <ContactGeometrySet><objects>
      <ContactHalfSpace name="floor"><socket_frame>/ground</socket_frame>
        <location>0 0 0</location><orientation>0 0 -1.5707963267948966</orientation></ContactHalfSpace>
      <ContactSphere name="ball"><socket_frame>/bodyset/foot</socket_frame>
        <location>0 0 0</location><radius>0.05</radius></ContactSphere>
    </objects></ContactGeometrySet>
    <ForceSet><objects>
      <SmoothSphereHalfSpaceForce name="ball_smooth">
        <socket_sphere>/contactgeometryset/ball</socket_sphere>
        <socket_half_space>/contactgeometryset/floor</socket_half_space>
        <stiffness>1000000</stiffness><dissipation>2.0</dissipation>
        <static_friction>0.8</static_friction><dynamic_friction>0.75</dynamic_friction>
        <viscous_friction>0.5</viscous_friction><transition_velocity>0.2</transition_velocity>
      </SmoothSphereHalfSpaceForce>
    </objects></ForceSet>
  </Model>
</OpenSimDocument>"""

_HUNT_CROSSLEY_OSIM = """<?xml version="1.0"?>
<OpenSimDocument Version="40000"><Model name="hc">
  <ForceSet><objects>
    <HuntCrossleyForce name="hc">
      <contact_parameters><objects>
        <HuntCrossleyForce::ContactParameters>
          <geometry>ball floor</geometry>
          <stiffness>1000000</stiffness><dissipation>1.0</dissipation>
          <static_friction>0.8</static_friction><dynamic_friction>0.7</dynamic_friction>
          <viscous_friction>0.0</viscous_friction>
        </HuntCrossleyForce::ContactParameters>
      </objects></contact_parameters>
      <transition_velocity>0.1</transition_velocity>
    </HuntCrossleyForce>
  </objects></ForceSet>
</Model></OpenSimDocument>"""


def _smooth_ref(center, vs, hs_origin, vh, normal_into, R, stiffness, diss, us, ud, uv, vt, cf, bd, bv):
    """Exact NumPy port of SimTK::SmoothSphereHalfSpaceForceImpl::calcForce (force on sphere)."""
    n = np.asarray(normal_into, float)
    indentation = R + np.dot(np.asarray(center) - np.asarray(hs_origin), n)
    v = np.asarray(vs) - np.asarray(vh)
    vn = np.dot(v, n)
    vt_vec = v - vn * n
    k = 0.5 * stiffness ** (2.0 / 3.0)
    fh_pos = (4.0 / 3.0) * k * np.sqrt(R * k) * (np.sqrt(indentation * indentation + cf)) ** 1.5
    fh = fh_pos * (0.5 + 0.5 * np.tanh(bd * indentation))
    fhc = fh * (1.0 + 1.5 * diss * vn)
    if diss != 0.0:
        fhc = fhc * (0.5 + 0.5 * np.tanh(bv * (vn + 2.0 / (3.0 * diss))))
    force = fhc * n
    vslip = np.sqrt(np.dot(vt_vec, vt_vec) + cf)
    vrel = vslip / vt
    ff = fhc * (min(vrel, 1.0) * (ud + 2 * (us - ud) / (1 + vrel * vrel)) + uv * vslip)
    force = force + ff * vt_vec / vslip
    return -force


def _hc_ref(loc, normal, depth, eff_r, vA, vB, p1, p2, tv):
    """Exact NumPy port of SimTK::HuntCrossleyForceImpl::calcForce (force on surface1)."""
    n = np.asarray(normal, float)
    s1 = p2["stiffness"] / (p1["stiffness"] + p2["stiffness"])
    s2 = 1.0 - s1
    k = p1["stiffness"] * s1
    c = p1["dissipation"] * s1 + p2["dissipation"] * s2
    fH = (4.0 / 3.0) * k * depth * np.sqrt(eff_r * k * depth)
    v = np.asarray(vA) - np.asarray(vB)
    vn = np.dot(v, n)
    vt_vec = v - vn * n
    f = fH * (1.0 + 1.5 * c * vn)
    if f <= 0.0:
        return np.zeros(3)
    force = f * n
    vslip = np.linalg.norm(vt_vec)
    if vslip != 0.0:
        us = (
            2 * p1["static_friction"] * p2["static_friction"] / (p1["static_friction"] + p2["static_friction"])
            if (p1["static_friction"] or p2["static_friction"])
            else 0.0
        )
        ud = (
            2 * p1["dynamic_friction"] * p2["dynamic_friction"] / (p1["dynamic_friction"] + p2["dynamic_friction"])
            if (p1["dynamic_friction"] or p2["dynamic_friction"])
            else 0.0
        )
        uv = (
            2 * p1["viscous_friction"] * p2["viscous_friction"] / (p1["viscous_friction"] + p2["viscous_friction"])
            if (p1["viscous_friction"] or p2["viscous_friction"])
            else 0.0
        )
        vrel = vslip / tv
        ff = f * (min(vrel, 1.0) * (ud + 2 * (us - ud) / (1 + vrel * vrel)) + uv * vslip)
        force = force + ff * vt_vec / vslip
    return -force


def _ef_ref(sp, nearest, area, vmesh, vother, k, c, us, ud, uv, tv):
    """Exact NumPy port of SimTK::ElasticFoundationForceImpl::processContact (force on mesh)."""
    disp = np.asarray(nearest) - np.asarray(sp)
    dist = np.linalg.norm(disp)
    if dist == 0.0:
        return np.zeros(3)
    fdir = disp / dist
    v = np.asarray(vother) - np.asarray(vmesh)
    vn = np.dot(v, fdir)
    vt_vec = v - vn * fdir
    f = k * area * dist * (1.0 + c * vn)
    if f <= 0.0:
        return np.zeros(3)
    force = f * fdir
    vslip = np.linalg.norm(vt_vec)
    if vslip != 0.0:
        vrel = vslip / tv
        ff = f * (min(vrel, 1.0) * (ud + 2 * (us - ud) / (1 + vrel * vrel)) + uv * vslip)
        force = force + ff * vt_vec / vslip
    return force


def _stencil_poses(base_pos, vel, h, device):
    """Build a [3, nbody] float64 pose stencil (position + linear velocity, no rotation)."""

    nb = len(base_pos)
    poses = np.zeros((3, nb, 4, 4))
    for b in range(nb):
        for s, sgn in ((0, 0.0), (1, 1.0), (2, -1.0)):
            poses[s, b] = np.eye(4)
            poses[s, b, :3, 3] = np.asarray(base_pos[b]) + sgn * h * np.asarray(vel[b])
    return wp.array(poses.reshape(3, nb, 16), dtype=_mat44d, device=device)


class TestContactParser(unittest.TestCase):
    def test_parse_all_contact_force_types(self):
        """Parse the smooth, Hunt-Crossley and elastic-foundation force schemas.

        Confirms the ``::``-scoped nested ContactParameters tag is tolerated and
        that per-surface material properties and the geometry list are captured.
        """
        smooth = osim.parse_osim(_CONTACT_MODEL_OSIM)
        cf = smooth.contact_forces[0]
        self.assertEqual(cf.type, "SmoothSphereHalfSpaceForce")
        self.assertEqual(cf.sphere, "ball")
        self.assertEqual(cf.half_space, "floor")
        self.assertAlmostEqual(cf.params["stiffness"], 1e6)
        self.assertAlmostEqual(cf.params["dynamic_friction"], 0.75)

        hc = osim.parse_osim(_HUNT_CROSSLEY_OSIM)
        f = hc.contact_forces[0]
        self.assertEqual(f.type, "HuntCrossleyForce")
        self.assertEqual(sorted(f.geometries), ["ball", "floor"])
        self.assertAlmostEqual(f.params["transition_velocity"], 0.1)
        self.assertAlmostEqual(f.surface_params["ball"]["stiffness"], 1e6)
        self.assertAlmostEqual(f.surface_params["floor"]["dynamic_friction"], 0.7)


class TestContactForceLaws(unittest.TestCase):
    """Validate each Warp contact kernel against the exact Simbody force law."""

    def setUp(self):

        self.device = wp.get_device("cpu")
        self.vec3d = _vec3d
        self.h = 1e-6
        self.inv2h = wp.float64(1.0 / (2.0 * self.h))

    def _arr(self, x, dt):
        return wp.array(np.asarray(x), dtype=dt, device=self.device)

    def test_smooth_matches_simbody(self):
        """Match SmoothSphereHalfSpaceForce (normal + dissipation + friction)."""

        f64, vec3d = wp.float64, self.vec3d
        R, ind = 0.05, 0.006
        center = np.array([0.0, R - ind, 0.0])
        vs, vh = np.array([0.3, -0.4, 0.0]), np.zeros(3)
        normal_into = np.array([0.0, -1.0, 0.0])
        poses = _stencil_poses([np.zeros(3), center], [vh, vs], self.h, self.device)
        bf = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        bt = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        ef = wp.zeros((1, 1), dtype=vec3d, device=self.device)
        wp.launch(
            smooth_sphere_halfspace_kernel,
            dim=(1, 1),
            inputs=[
                poses,
                self._arr([1], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([R], f64),
                self._arr([0], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([normal_into], vec3d),
                self._arr([1e6], f64),
                self._arr([2.0], f64),
                self._arr([0.8], f64),
                self._arr([0.75], f64),
                self._arr([0.5], f64),
                self._arr([0.2], f64),
                self._arr([1e-5], f64),
                self._arr([300.0], f64),
                self._arr([50.0], f64),
                self.inv2h,
                bf,
                bt,
                ef,
            ],
            device=self.device,
        )
        ref = _smooth_ref(center, vs, np.zeros(3), vh, normal_into, R, 1e6, 2.0, 0.8, 0.75, 0.5, 0.2, 1e-5, 300.0, 50.0)
        np.testing.assert_allclose(ef.numpy()[0, 0], ref, atol=1e-6, rtol=1e-9)

    def test_smooth_force_negligible_without_penetration(self):
        """Produce ~zero force when the sphere clears the half-space."""

        f64, vec3d = wp.float64, self.vec3d
        R = 0.05
        center = np.array([0.0, R + 0.1, 0.0])  # 10 cm clearance -> tanh smoothing ~ 0
        poses = _stencil_poses([np.zeros(3), center], [np.zeros(3), np.zeros(3)], self.h, self.device)
        bf = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        bt = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        ef = wp.zeros((1, 1), dtype=vec3d, device=self.device)
        wp.launch(
            smooth_sphere_halfspace_kernel,
            dim=(1, 1),
            inputs=[
                poses,
                self._arr([1], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([R], f64),
                self._arr([0], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([[0.0, -1.0, 0.0]], vec3d),
                self._arr([1e6], f64),
                self._arr([2.0], f64),
                self._arr([0.8], f64),
                self._arr([0.75], f64),
                self._arr([0.5], f64),
                self._arr([0.2], f64),
                self._arr([1e-5], f64),
                self._arr([300.0], f64),
                self._arr([50.0], f64),
                self.inv2h,
                bf,
                bt,
                ef,
            ],
            device=self.device,
        )
        self.assertLess(np.linalg.norm(ef.numpy()[0, 0]), 1e-6)

    def test_hunt_crossley_sphere_halfspace_matches_simbody(self):
        """Match classic HuntCrossleyForce for a sphere on a half-space."""

        f64, vec3d = wp.float64, self.vec3d
        R, depth = 0.04, 0.005
        center = np.array([0.0, R - depth, 0.0])
        vs = np.array([0.2, -0.3, 0.0])
        normal_into = np.array([0.0, -1.0, 0.0])
        poses = _stencil_poses([np.zeros(3), center], [np.zeros(3), vs], self.h, self.device)
        bf = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        bt = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        hf = wp.zeros((1, 1), dtype=vec3d, device=self.device)
        p = {
            "stiffness": 1e6,
            "dissipation": 1.0,
            "static_friction": 0.8,
            "dynamic_friction": 0.7,
            "viscous_friction": 0.0,
        }
        wp.launch(
            hunt_crossley_kernel,
            dim=(1, 1),
            inputs=[
                poses,
                self._arr([0], wp.int32),
                self._arr([1], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([R], f64),
                self._arr([0], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([0.0], f64),
                self._arr([normal_into], vec3d),
                self._arr([p["stiffness"]], f64),
                self._arr([p["dissipation"]], f64),
                self._arr([p["static_friction"]], f64),
                self._arr([p["dynamic_friction"]], f64),
                self._arr([p["viscous_friction"]], f64),
                self._arr([p["stiffness"]], f64),
                self._arr([p["dissipation"]], f64),
                self._arr([p["static_friction"]], f64),
                self._arr([p["dynamic_friction"]], f64),
                self._arr([p["viscous_friction"]], f64),
                self._arr([0.1], f64),
                self.inv2h,
                bf,
                bt,
                hf,
            ],
            device=self.device,
        )
        loc = center + (R - 0.5 * depth) * normal_into
        ref = _hc_ref(loc, normal_into, depth, R, vs, np.zeros(3), p, p, 0.1)
        np.testing.assert_allclose(hf.numpy()[0, 0], ref, atol=1e-4, rtol=1e-9)

    def test_hunt_crossley_sphere_sphere_matches_simbody(self):
        """Match classic HuntCrossleyForce for two spheres with unequal materials."""

        f64, vec3d = wp.float64, self.vec3d
        R1, R2, depth = 0.02, 0.05, 0.004
        c1 = np.array([0.0, R1 + R2 - depth, 0.0])
        vA = np.array([0.05, -0.2, 0.0])
        poses = _stencil_poses([np.zeros(3), c1], [np.zeros(3), vA], self.h, self.device)
        bf = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        bt = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        hf = wp.zeros((1, 1), dtype=vec3d, device=self.device)
        pa = {
            "stiffness": 1e6,
            "dissipation": 1.5,
            "static_friction": 0.6,
            "dynamic_friction": 0.5,
            "viscous_friction": 0.0,
        }
        pb = {
            "stiffness": 2e6,
            "dissipation": 1.0,
            "static_friction": 0.9,
            "dynamic_friction": 0.8,
            "viscous_friction": 0.0,
        }
        wp.launch(
            hunt_crossley_kernel,
            dim=(1, 1),
            inputs=[
                poses,
                self._arr([1], wp.int32),
                self._arr([1], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([R1], f64),
                self._arr([0], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([R2], f64),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([pa["stiffness"]], f64),
                self._arr([pa["dissipation"]], f64),
                self._arr([pa["static_friction"]], f64),
                self._arr([pa["dynamic_friction"]], f64),
                self._arr([pa["viscous_friction"]], f64),
                self._arr([pb["stiffness"]], f64),
                self._arr([pb["dissipation"]], f64),
                self._arr([pb["static_friction"]], f64),
                self._arr([pb["dynamic_friction"]], f64),
                self._arr([pb["viscous_friction"]], f64),
                self._arr([0.15], f64),
                self.inv2h,
                bf,
                bt,
                hf,
            ],
            device=self.device,
        )
        n = (np.zeros(3) - c1) / np.linalg.norm(c1)
        eff = R1 * R2 / (R1 + R2)
        loc = c1 + (R1 - 0.5 * depth) * n
        ref = _hc_ref(loc, n, depth, eff, vA, np.zeros(3), pa, pb, 0.15)
        np.testing.assert_allclose(hf.numpy()[0, 0], ref, atol=1e-4, rtol=1e-9)

    def test_hunt_crossley_hard_cutoff_on_separation(self):
        """Zero the Hunt-Crossley force when the dissipation term drives it non-positive."""

        f64, vec3d = wp.float64, self.vec3d
        R, depth = 0.04, 0.001
        center = np.array([0.0, R - depth, 0.0])
        vs = np.array([0.0, 60.0, 0.0])  # separating fast -> f = fH(1+1.5 c vn) < 0
        normal_into = np.array([0.0, -1.0, 0.0])
        poses = _stencil_poses([np.zeros(3), center], [np.zeros(3), vs], self.h, self.device)
        bf = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        bt = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        hf = wp.zeros((1, 1), dtype=vec3d, device=self.device)
        p = {
            "stiffness": 1e6,
            "dissipation": 1.0,
            "static_friction": 0.0,
            "dynamic_friction": 0.0,
            "viscous_friction": 0.0,
        }
        wp.launch(
            hunt_crossley_kernel,
            dim=(1, 1),
            inputs=[
                poses,
                self._arr([0], wp.int32),
                self._arr([1], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([R], f64),
                self._arr([0], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([0.0], f64),
                self._arr([normal_into], vec3d),
                self._arr([p["stiffness"]], f64),
                self._arr([p["dissipation"]], f64),
                self._arr([0.0], f64),
                self._arr([0.0], f64),
                self._arr([0.0], f64),
                self._arr([p["stiffness"]], f64),
                self._arr([p["dissipation"]], f64),
                self._arr([0.0], f64),
                self._arr([0.0], f64),
                self._arr([0.0], f64),
                self._arr([0.1], f64),
                self.inv2h,
                bf,
                bt,
                hf,
            ],
            device=self.device,
        )
        np.testing.assert_allclose(hf.numpy()[0, 0], np.zeros(3), atol=0.0)

    def test_elastic_foundation_matches_simbody(self):
        """Match ElasticFoundationForce springs against a half-space and a sphere."""

        f64, vec3d = wp.float64, self.vec3d
        # Half-space: outward normal +y (toward mesh), spring 5 mm below the floor.
        sp = np.array([0.1, -0.005, 0.0])
        vmesh = np.array([0.4, -0.1, 0.0])
        poses = _stencil_poses([np.zeros(3), np.zeros(3)], [np.zeros(3), vmesh], self.h, self.device)
        bf = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        bt = wp.zeros((1, 2), dtype=vec3d, device=self.device)
        ff = wp.zeros((1, 1), dtype=vec3d, device=self.device)
        wp.launch(
            elastic_foundation_kernel,
            dim=(1, 1),
            inputs=[
                poses,
                self._arr([1], wp.int32),
                self._arr([sp], vec3d),
                self._arr([1e-4], f64),
                self._arr([0], wp.int32),
                self._arr([0], wp.int32),
                self._arr([[0.0, 0, 0]], vec3d),
                self._arr([[0.0, 1.0, 0.0]], vec3d),
                self._arr([0.0], f64),
                self._arr([2e7], f64),
                self._arr([1.0], f64),
                self._arr([0.8], f64),
                self._arr([0.8], f64),
                self._arr([0.0], f64),
                self._arr([0.2], f64),
                self._arr([1.0], f64),
                self.inv2h,
                bf,
                bt,
                ff,
            ],
            device=self.device,
        )
        nearest = np.array([0.1, 0.0, 0.0])
        ref = _ef_ref(sp, nearest, 1e-4, vmesh, np.zeros(3), 2e7, 1.0, 0.8, 0.8, 0.0, 0.2)
        np.testing.assert_allclose(ff.numpy()[0, 0], ref, atol=1e-7, rtol=1e-9)

    def test_geometry_matches_newton_primitives(self):
        """Cross-check the float64 contact geometry against newton.geometry primitives.

        The opensim contact indentation/normal must agree with
        :func:`~newton._src.geometry.collision_primitive.collide_plane_sphere` and
        :func:`collide_sphere_sphere` (which are float32).
        """

        @wp.kernel
        def geo(out: wp.array[wp.float32], nrm: wp.array[wp.vec3]):
            d1, _ = collide_plane_sphere(
                wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.045, 0.0), wp.float32(0.05)
            )
            out[0] = d1
            d2, _, n = collide_sphere_sphere(
                wp.vec3(0.0, 0.066, 0.0), wp.float32(0.02), wp.vec3(0.0, 0.0, 0.0), wp.float32(0.05)
            )
            out[1] = d2
            nrm[0] = n

        out = wp.zeros(2, dtype=wp.float32, device=self.device)
        nrm = wp.zeros(1, dtype=wp.vec3, device=self.device)
        wp.launch(geo, dim=1, inputs=[out, nrm], device=self.device)
        o = out.numpy()
        # opensim indentation R + dot(center-origin, normal_into); newton penetration = -dist.
        self.assertAlmostEqual(-o[0], 0.005, places=5)  # sphere/half-space penetration
        self.assertAlmostEqual(-o[1], 0.004, places=5)  # sphere/sphere penetration
        np.testing.assert_allclose(nrm.numpy()[0], [0.0, -1.0, 0.0], atol=1e-6)


class TestContactSimulation(unittest.TestCase):
    """Validate the full forward-kinematics -> wrench -> dynamics contact path."""

    def test_smooth_contact_at_rest_matches_hertz(self):
        """Reproduce the smooth Hertz force through the forward kinematics."""

        model = osim.parse_osim(_CONTACT_MODEL_OSIM)
        contact = OpenSimContact(model)
        R, ind = 0.05, 0.005
        f = contact.forces(np.array([0.0, R - ind]), frame="opensim")  # (tx, ty)
        ref = _smooth_ref(
            np.array([0.0, R - ind, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.array([0.0, -1.0, 0.0]),
            R,
            1e6,
            2.0,
            0.8,
            0.75,
            0.5,
            0.2,
            1e-5,
            300.0,
            50.0,
        )
        np.testing.assert_allclose(f[0], ref, atol=1e-6)

    def test_public_contact_force_defaults_to_newton_z_up(self):
        """Rotate native OpenSim Y-up contact vectors into Newton's Z-up world."""
        contact = OpenSimContact(osim.parse_osim(_CONTACT_MODEL_OSIM))
        q = np.array([0.0, 0.045])
        native = contact.forces(q, frame="opensim")
        converted = contact.forces(q)

        expected = native[..., [0, 2, 1]].copy()
        expected[..., 1] *= -1.0
        np.testing.assert_allclose(converted, expected, atol=1.0e-8)
        self.assertGreater(float(converted[0, 2]), 0.0)

    def test_generalized_force_projection(self):
        """Project the contact wrench onto the vertical slider coordinate."""

        model = osim.parse_osim(_CONTACT_MODEL_OSIM)
        contact = OpenSimContact(model)
        q = np.array([0.0, 0.045])
        tau = contact.generalized_forces(q)
        f = contact.forces(q, frame="opensim")[0]
        self.assertAlmostEqual(tau[0], f[0], places=6)  # tx <- Fx
        self.assertAlmostEqual(tau[1], f[1], places=6)  # ty <- Fy

    def test_generalized_force_pipeline_stays_on_device(self):
        """Keep contact wrenches and Jacobians on device until generalized forces are copied."""
        contact = OpenSimContact(osim.parse_osim(_CONTACT_MODEL_OSIM))
        q = np.array([[0.0, 0.045], [0.01, 0.044]])
        qd = np.array([[0.1, -0.2], [-0.1, 0.3]])
        expected = contact.generalized_forces(q, qd)

        def reject_host_wrapper(*_args, **_kwargs):
            self.fail("contact projection called a host-returning intermediate wrapper")

        contact._run = reject_host_wrapper
        contact.fk.body_transforms_batch = reject_host_wrapper
        np.testing.assert_allclose(contact.generalized_forces(q, qd), expected)

    def test_contact_driven_forward_dynamics_reaches_equilibrium(self):
        """Settle a dropped sphere so the contact force balances its weight."""

        model = osim.parse_osim(_CONTACT_MODEL_OSIM)
        contact = OpenSimContact(model)
        fd = ForwardDynamics(model)
        mass, g = 2.0, 9.80665
        q0 = np.array([0.0, 0.2])
        v0 = np.array([0.0, 0.0])
        result = fd.simulate(
            q0,
            v0,
            duration=4.0,
            dt=1e-4,
            contact_forces=contact,
            integrator="semi_implicit",
            use_graph=False,
        )
        q = result.coordinates[-1]
        v = result.speeds[-1]
        fc = contact.forces(q, v, frame="opensim")[0]
        self.assertLess(abs(v[1]), 1e-3)
        np.testing.assert_allclose(fc[1], mass * g, rtol=2e-3)

    def test_forward_simulation_evaluates_contact_at_rk_stages(self):
        """Recompute body contact from each current RK state without measured loads."""
        model = osim.parse_osim(_CONTACT_MODEL_OSIM)
        forward = ForwardDynamics(model)
        calls = []

        class Contact:
            def body_wrenches(self, q, qd, *, h, frame):
                calls.append((q.copy(), qd.copy(), h, frame))
                return ["ball"], np.zeros((1, 1, 9))

        def zero_acceleration(q, qd, tau, *, external_bodies, external_wrenches, h, eps):
            self.assertEqual(external_bodies, ["ball"])
            self.assertEqual(external_wrenches.shape, (1, 1, 9))
            return np.zeros_like(q)

        original = forward.accelerations
        forward.accelerations = zero_acceleration
        try:
            result = forward.simulate(
                np.array([0.0, 0.2]),
                np.array([0.1, -0.2]),
                duration=0.002,
                dt=0.001,
                contact_forces=Contact(),
                integrator="rk4",
                use_graph=False,
            )
        finally:
            forward.accelerations = original

        self.assertEqual(len(calls), 8)
        self.assertTrue(all(call[2:] == (1.0e-6, "opensim") for call in calls))
        self.assertGreater(float(np.max(np.abs(calls[1][0] - calls[0][0]))), 0.0)
        self.assertEqual(result.coordinates.shape, (3, 2))
        with self.assertRaisesRegex(ValueError, "contact_h"):
            forward.simulate(
                np.zeros(2), np.zeros(2), 0.001, 0.001, contact_forces=Contact(), contact_h=0.0, use_graph=False
            )

    def test_elastic_foundation_mesh_on_floor(self):
        """Drive an ElasticFoundationForce with a supplied triangle mesh vs the floor.

        A single downward-facing triangle penetrating the floor must push the mesh
        body up with the summed spring force (k * area * penetration).
        """

        ef_osim = _CONTACT_MODEL_OSIM.replace(
            '<ContactSphere name="ball"><socket_frame>/bodyset/foot</socket_frame>\n'
            "        <location>0 0 0</location><radius>0.05</radius></ContactSphere>",
            '<ContactMesh name="pad"><socket_frame>/bodyset/foot</socket_frame>\n'
            "        <location>0 0 0</location><orientation>0 0 0</orientation><filename></filename></ContactMesh>",
        ).replace(
            """<SmoothSphereHalfSpaceForce name="ball_smooth">
        <socket_sphere>/contactgeometryset/ball</socket_sphere>
        <socket_half_space>/contactgeometryset/floor</socket_half_space>
        <stiffness>1000000</stiffness><dissipation>2.0</dissipation>
        <static_friction>0.8</static_friction><dynamic_friction>0.75</dynamic_friction>
        <viscous_friction>0.5</viscous_friction><transition_velocity>0.2</transition_velocity>
      </SmoothSphereHalfSpaceForce>""",
            """<ElasticFoundationForce name="pad_ef">
        <contact_parameters><objects>
          <ElasticFoundationForce::ContactParameters>
            <geometry>pad floor</geometry>
            <stiffness>20000000</stiffness><dissipation>0.0</dissipation>
            <static_friction>0.0</static_friction><dynamic_friction>0.0</dynamic_friction>
            <viscous_friction>0.0</viscous_friction>
          </ElasticFoundationForce::ContactParameters>
        </objects></contact_parameters>
        <transition_velocity>0.2</transition_velocity>
      </ElasticFoundationForce>""",
        )
        model = osim.parse_osim(ef_osim)
        # A unit-ish triangle in the foot's x-z plane at local y=0 (area 0.5).
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        faces = np.array([[0, 1, 2]])
        contact = OpenSimContact(model, meshes={"pad": (verts, faces)})
        self.assertEqual(contact.n_ef, 1)
        pen = 0.01  # push the foot 1 cm below the floor
        f = contact.forces(np.array([0.0, -pen]), frame="opensim")[0]
        area = 0.5
        # spring centroid penetration = pen; force = k * area * pen (up).
        np.testing.assert_allclose(f[1], 2e7 * area * pen, rtol=1e-6)
        self.assertGreater(f[1], 0.0)


if __name__ == "__main__":
    unittest.main()
