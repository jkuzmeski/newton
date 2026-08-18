# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the OpenSim ``.osim`` writer (round-trip against the parser)."""

import dataclasses
import math
import os
import tempfile
import unittest

import newton.examples
import newton.opensim as osim
from newton._src.opensim import types as T


def _diff(a, b, path=""):
    """Return a list of ``(path, a, b)`` mismatches, comparing floats with tolerance."""
    out = []
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        for f in dataclasses.fields(a):
            out += _diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            out += _diff(a.get(k), b.get(k), f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path + ".len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            out += _diff(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        try:
            if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12):
                out.append((path, a, b))
        except (TypeError, ValueError):
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out


def _roundtrip(model):
    """Serialize ``model`` and re-parse it, returning the reparsed model."""
    reparsed = osim.parse_osim(osim.osim_to_xml(model))
    model.frames = {}
    reparsed.frames = {}
    return reparsed


class TestOsimWriter(unittest.TestCase):
    """Verify the writer produces documents that round-trip through the parser."""

    def test_gait2354_round_trips_exactly(self):
        """Serialize the full gait2354 model and re-parse it with no field changes."""
        model = osim.parse_osim(newton.examples.get_asset("gait2354_subject01.osim"))
        self.assertEqual(_diff(model, _roundtrip(model), "gait2354"), [])

    def test_gait2354_has_expected_content(self):
        """The reparsed gait2354 model keeps its bodies, joints, muscles, and markers."""
        model = osim.parse_osim(newton.examples.get_asset("gait2354_subject01.osim"))
        reparsed = _roundtrip(model)
        self.assertEqual(len(reparsed.bodies), 12)
        self.assertEqual(len(reparsed.joints), 12)
        self.assertEqual(len(reparsed.muscles), 54)
        self.assertEqual(len(reparsed.markers), 39)

    def test_custom_joint_spatial_transform_round_trips(self):
        """A CustomJoint's coordinate functions and offset frames survive serialization."""
        model = T.OsimModel(name="cj", version=40000)
        model.bodies = [T.OsimBody(name="seg", mass=1.0, mass_center=(0.0, -0.1, 0.0), inertia=(0.1,) * 6)]
        model.joints = [
            T.OsimJoint(
                name="hip",
                type="CustomJoint",
                parent_body="ground",
                child_body="seg",
                parent_transform=T.OsimTransform(translation=(0.0, 1.0, 0.0), orientation=(0.0, 0.0, 0.1)),
                child_transform=T.OsimTransform(translation=(0.0, 0.05, 0.0)),
                coordinates=[
                    T.OsimCoordinate(name="hip_flex", default_value=0.1, range=(-1.0, 1.5), clamped=True),
                ],
                spatial_transform=[
                    T.OsimTransformAxis(
                        axis=(0.0, 0.0, 1.0),
                        coordinates=["hip_flex"],
                        function_type="LinearFunction",
                        function={"type": "LinearFunction", "coefficients": [1.0, 0.0]},
                        is_identity=False,
                    ),
                ],
            )
        ]
        self.assertEqual(_diff(model, _roundtrip(model), "cj"), [])

    def test_wrap_objects_round_trip(self):
        """WrapEllipsoid and WrapTorus surfaces on bodies and ground round-trip."""
        model = T.OsimModel(name="w", version=40000)
        model.bodies = [T.OsimBody(name="seg", mass=1.0)]
        model.joints = [T.OsimJoint(name="j", type="PinJoint", parent_body="ground", child_body="seg")]
        model.wrap_objects = [
            T.OsimWrapObject(
                name="ell",
                type="WrapEllipsoid",
                body="seg",
                translation=(0.01, 0.02, 0.03),
                rotation=(0.1, 0.2, 0.3),
                dimensions=(0.04, 0.05, 0.06),
            ),
            T.OsimWrapObject(
                name="tor",
                type="WrapTorus",
                body="ground",
                translation=(0.0, 0.5, 0.0),
                inner_radius=0.02,
                outer_radius=0.08,
                quadrant="x",
            ),
        ]
        self.assertEqual(_diff(model, _roundtrip(model), "w"), [])

    def test_muscle_geometry_path_round_trips(self):
        """A muscle's path points, wrap set, and Thelen parameters round-trip."""
        model = T.OsimModel(name="m", version=40000)
        model.bodies = [T.OsimBody(name="seg", mass=1.0)]
        model.joints = [T.OsimJoint(name="j", type="PinJoint", parent_body="ground", child_body="seg")]
        model.wrap_objects = [T.OsimWrapObject(name="ell", type="WrapEllipsoid", body="seg", dimensions=(0.04,) * 3)]
        model.muscles = [
            T.OsimMuscle(
                name="mus",
                type="Thelen2003Muscle",
                path_points=[
                    T.OsimPathPoint(name="p0", body="ground", location=(0.0, 0.5, 0.0)),
                    T.OsimPathPoint(name="p1", body="seg", location=(0.03, -0.1, 0.0)),
                ],
                wraps=[T.OsimWrap(wrap_object="ell", method="hybrid", range=(-1, -1))],
                params={
                    "max_isometric_force": 200.0,
                    "optimal_fiber_length": 0.12,
                    "tendon_slack_length": 0.05,
                    "ignore_tendon_compliance": 1.0,
                },
                min_control=0.01,
                max_control=1.0,
            )
        ]
        self.assertEqual(_diff(model, _roundtrip(model), "m"), [])

    def test_forces_and_contact_round_trip(self):
        """Actuators, contact geometry/forces, springs, ligaments, and bushings round-trip."""
        model = T.OsimModel(name="f", version=40000)
        model.bodies = [T.OsimBody(name="seg", mass=1.0)]
        model.joints = [
            T.OsimJoint(
                name="j",
                type="PinJoint",
                parent_body="ground",
                child_body="seg",
                coordinates=[T.OsimCoordinate(name="q")],
            )
        ]
        model.actuators = [
            T.OsimActuator(
                name="res",
                type="CoordinateActuator",
                coordinate="q",
                optimal_force=50.0,
                min_control=-1.0,
                max_control=1.0,
            ),
            T.OsimActuator(
                name="pt",
                type="PointActuator",
                optimal_force=10.0,
                body="seg",
                point=(0.0, -0.1, 0.0),
                direction=(1.0, 0.0, 0.0),
            ),
        ]
        model.contact_geometry = [
            T.OsimContactGeometry(
                name="floor", type="ContactHalfSpace", body="ground", orientation=(0.0, 0.0, -1.5707963)
            ),
            T.OsimContactGeometry(
                name="heel", type="ContactSphere", body="seg", location=(0.0, -0.3, 0.0), radius=0.03
            ),
        ]
        model.contact_forces = [
            T.OsimContactForce(
                name="c",
                type="SmoothSphereHalfSpaceForce",
                sphere="heel",
                half_space="floor",
                geometries=["heel", "floor"],
                params={
                    "stiffness": 1e6,
                    "dissipation": 2.0,
                    "static_friction": 0.8,
                    "dynamic_friction": 0.8,
                    "viscous_friction": 0.5,
                    "transition_velocity": 0.1,
                },
            ),
        ]
        model.point_to_point_springs = [
            T.OsimPointToPointSpring(
                name="p2p",
                body1="ground",
                body2="seg",
                point1=(0.0, 0.5, 0.0),
                point2=(0.0, -0.1, 0.0),
                stiffness=500.0,
                rest_length=0.4,
            ),
        ]
        model.spring_generalized_forces = [
            T.OsimSpringGeneralizedForce(name="sgf", coordinate="q", stiffness=20.0, viscosity=1.0),
        ]
        model.bushing_forces = [
            T.OsimBushingForce(
                name="bush",
                body1="ground",
                body2="seg",
                frame1_transform=T.OsimTransform(translation=(0.1, 0.2, 0.3)),
                frame2_transform=T.OsimTransform(orientation=(0.04, 0.05, 0.06)),
                rotational_stiffness=(1.0, 2.0, 3.0),
                translational_stiffness=(10.0, 20.0, 30.0),
                rotational_damping=(0.1, 0.2, 0.3),
                translational_damping=(0.4, 0.5, 0.6),
            ),
        ]
        model.path_springs = [
            T.OsimPathSpring(
                name="ps",
                path_points=[
                    T.OsimPathPoint(name="s0", body="ground", location=(0.0, 0.5, 0.0)),
                    T.OsimPathPoint(name="s1", body="seg", location=(0.0, -0.1, 0.0)),
                ],
                resting_length=0.3,
                stiffness=1000.0,
                dissipation=0.1,
            ),
        ]
        model.ligaments = [
            T.OsimLigament(
                name="lig",
                path_points=[
                    T.OsimPathPoint(name="l0", body="ground", location=(0.0, 0.5, 0.0)),
                    T.OsimPathPoint(name="l1", body="seg", location=(0.0, -0.1, 0.0)),
                ],
                resting_length=0.35,
                pcsa_force=500.0,
                force_length_curve={"type": "SimmSpline", "x": [0.0, 1.0, 2.0], "y": [0.0, 0.5, 1.0]},
            ),
        ]
        model.markers = [T.OsimMarker(name="MK", body="seg", location=(0.0, -0.3, 0.0))]
        self.assertEqual(_diff(model, _roundtrip(model), "f"), [])

    def test_write_osim_file_round_trips(self):
        """write_osim writes a file that re-parses to an equivalent model."""
        model = osim.parse_osim(newton.examples.get_asset("gait2354_subject01.osim"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.osim")
            osim.write_osim(model, path)
            self.assertTrue(os.path.exists(path))
            reparsed = osim.parse_osim(path)
        model.frames = {}
        reparsed.frames = {}
        self.assertEqual(_diff(model, reparsed, "file"), [])


if __name__ == "__main__":
    unittest.main()
