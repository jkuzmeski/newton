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

"""Tests for muscle-driven forward simulation and prescribed controllers."""

import unittest

import numpy as np
import warp as wp

import newton.opensim as osim
from newton._src.opensim.controllers import ControlSet, PrescribedController
from newton._src.opensim.dynamics import InverseDynamics
from newton._src.opensim.muscle_force import MuscleForces
from newton._src.opensim.muscle_forward import MuscleDrivenForward, simulate_muscle_driven
from newton._src.opensim.static_optimization import solve_frame_activations

_PIN_MUSCLE_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="pin_muscle">
    <gravity>0 -9.80665 0</gravity>
    <BodySet name="bodyset"><objects>
      <Body name="seg">
        <mass>0.5</mass>
        <mass_center>0 -0.15 0</mass_center>
        <inertia>0.01 0.01 0.01 0 0 0</inertia>
      </Body>
    </objects></BodySet>
    <JointSet name="jointset"><objects>
      <PinJoint name="pin">
        <socket_parent_frame>g_off</socket_parent_frame>
        <socket_child_frame>c_off</socket_child_frame>
        <coordinates><objects>
          <Coordinate name="pin_angle"><default_value>0.0</default_value><range>-2 2</range></Coordinate>
        </objects></coordinates>
        <frames>
          <PhysicalOffsetFrame name="g_off"><socket_parent>/ground</socket_parent><translation>0 0.5 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="c_off"><socket_parent>/bodyset/seg</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>
        </frames>
      </PinJoint>
    </objects></JointSet>
    <ForceSet name="forceset"><objects></objects></ForceSet>
  </Model>
</OpenSimDocument>"""

_MUSCLE_PARAMS = {
    "max_isometric_force": 200.0,
    "optimal_fiber_length": 0.15,
    "tendon_slack_length": 0.05,
    "pennation_angle_at_optimal": 0.0,
}


def _pin_model():
    """Build the pin model with two flexor muscles."""
    model = osim.parse_osim(_PIN_MUSCLE_OSIM)
    model.muscles = [
        osim.OsimMuscle(
            name="flex",
            type="Thelen2003Muscle",
            path_points=[
                osim.OsimPathPoint(name="a", body="ground", location=(0.08, 0.5, 0.0)),
                osim.OsimPathPoint(name="b", body="seg", location=(0.03, -0.1, 0.0)),
            ],
            params=_MUSCLE_PARAMS.copy(),
        ),
        osim.OsimMuscle(
            name="flex2",
            type="Thelen2003Muscle",
            path_points=[
                osim.OsimPathPoint(name="c", body="ground", location=(0.12, 0.5, 0.0)),
                osim.OsimPathPoint(name="d", body="seg", location=(0.05, -0.1, 0.0)),
            ],
            params=_MUSCLE_PARAMS.copy(),
        ),
    ]
    return model


def _holding_activation(model, angle):
    """Return the least-effort activation that balances gravity at ``angle``."""
    q = np.array([[angle]])
    z = np.zeros_like(q)
    idv = InverseDynamics(model)
    mf = MuscleForces(model)
    tau = idv.solve(q, z, z)[0]
    arm = mf.paths.moment_arms(q)[0].T
    active = (mf.forces(np.ones((1, 2)), q, z) - mf.forces(np.zeros((1, 2)), q, z))[0]
    passive = mf.forces(np.zeros((1, 2)), q, z)[0]
    a_ref, _, _ = solve_frame_activations(arm, active, passive, tau, use_reserves=False)
    return a_ref


class TestControlSet(unittest.TestCase):
    """Validate ControlSet interpolation and PrescribedController alignment."""

    def test_control_set_interpolates_and_round_trips(self):
        """A control set interpolates linearly and round-trips through a Storage."""
        cs = ControlSet(labels=["m1", "m2"], times=[0.0, 1.0], data=[[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(cs.sample(0.5), [0.5, 0.5])
        np.testing.assert_allclose(cs.value("m1", 0.25), [0.25])
        back = ControlSet.from_storage(cs.to_storage())
        np.testing.assert_allclose(back.data, cs.data)

    def test_prescribed_controller_aligns_and_defaults(self):
        """The controller emits controls in actuator order, defaulting missing names."""
        cs = ControlSet(labels=["m2"], times=[0.0, 1.0], data=[[0.2], [0.4]])
        ctrl = PrescribedController(["m1", "m2"], cs, default=0.0)
        np.testing.assert_allclose(ctrl(0.5), [0.0, 0.3])

    def test_prescribed_controller_from_dict(self):
        """A dict of constants and callables produces the aligned control vector."""
        ctrl = PrescribedController(["m1", "m2"], {"m1": 0.5, "m2": lambda t: 0.1 * t})
        np.testing.assert_allclose(ctrl(2.0), [0.5, 0.2])


class TestMuscleDrivenForward(unittest.TestCase):
    """Validate the muscle-driven closed-loop forward simulation."""

    def test_holds_equilibrium(self):
        """Constant holding excitation keeps the segment at its start angle."""
        model = _pin_model()
        ang = 0.2
        a_ref = _holding_activation(model, ang)
        res = MuscleDrivenForward(model).simulate(
            np.array([ang]),
            np.array([0.0]),
            a_ref,
            duration=0.5,
            dt=0.001,
            initial_activations=a_ref,
        )
        self.assertLess(np.max(np.abs(res.coordinates[:, 0] - ang)), 0.02)

    def test_passive_segment_falls(self):
        """With no excitation the segment swings toward the hanging equilibrium."""
        model = _pin_model()
        ang = 0.2
        res = MuscleDrivenForward(model).simulate(
            np.array([ang]),
            np.array([0.0]),
            np.zeros(2),
            duration=0.5,
            dt=0.001,
            initial_activations=np.zeros(2),
        )
        self.assertLess(res.coordinates[:, 0].min(), ang - 0.05)

    def test_activation_dynamics_ramp(self):
        """Activations ramp from zero toward a step excitation over the simulation."""
        model = _pin_model()
        ang = 0.2
        a_ref = _holding_activation(model, ang)
        res = simulate_muscle_driven(
            model,
            a_ref,
            np.array([ang]),
            np.array([0.0]),
            duration=0.3,
            dt=0.001,
            initial_activations=np.zeros(2),
        )
        # activation starts at zero and rises toward the excitation
        self.assertTrue(np.all(res.activations[0] < 1e-9))
        self.assertGreater(res.activations[-1, 0], 0.5 * a_ref[0])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_cuda_graph_matches_host_orchestration(self):
        """Match host orchestration while keeping constant-excitation steps in a CUDA graph."""
        args = (np.array([0.2]), np.array([0.0]), np.array([0.1, 0.05]))
        kwargs = {"duration": 0.01, "dt": 0.001, "initial_activations": np.zeros(2)}
        for integrator in ("semi_implicit", "rk4"):
            sim = MuscleDrivenForward(_pin_model(), device="cuda:0")
            expected = sim.simulate(*args, integrator=integrator, use_graph=False, **kwargs)

            def reject_host_wrapper(*_args, **_kwargs):
                self.fail("CUDA graph path called a host-returning simulation wrapper")

            sim.muscles.forces = reject_host_wrapper
            sim.muscles.generalized_forces = reject_host_wrapper
            sim.muscles.integrate_activation = reject_host_wrapper
            sim.fd.accelerations = reject_host_wrapper
            actual = sim.simulate(*args, integrator=integrator, use_graph=True, **kwargs)
            for field in ("coordinates", "speeds", "activations", "excitations", "muscle_forces"):
                np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1.0e-10)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_cuda_graph_presamples_controller(self):
        """Presample a time-varying controller once and replay it inside the CUDA graph."""
        ctrl = PrescribedController(
            ["flex", "flex2"],
            {"flex": lambda t: 0.1 + 2.0 * t, "flex2": lambda t: 0.05 + t},
        )
        args = (np.array([0.2]), np.array([0.0]), ctrl)
        kwargs = {"duration": 0.01, "dt": 0.001, "initial_activations": np.zeros(2)}
        sim = MuscleDrivenForward(_pin_model(), device="cuda:0")
        expected = sim.simulate(*args, use_graph=False, **kwargs)
        actual = sim.simulate(*args, use_graph=True, **kwargs)
        for field in ("coordinates", "speeds", "activations", "excitations", "muscle_forces"):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1.0e-10)

    def test_controller_excitation_source(self):
        """A PrescribedController drives the forward simulation."""
        model = _pin_model()
        ang = 0.2
        a_ref = _holding_activation(model, ang)
        ctrl = PrescribedController(["flex", "flex2"], {"flex": a_ref[0], "flex2": a_ref[1]})
        res = MuscleDrivenForward(model).simulate(
            np.array([ang]),
            np.array([0.0]),
            ctrl,
            duration=0.4,
            dt=0.001,
            initial_activations=a_ref,
        )
        self.assertLess(np.max(np.abs(res.coordinates[:, 0] - ang)), 0.03)


_CONTACT_DROP_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
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


def _contact_drop_model():
    """Parse the 2-DOF foot ball dropped onto a smooth half-space floor."""
    return osim.parse_osim(_CONTACT_DROP_OSIM)


class TestContactLoop(unittest.TestCase):
    """Verify OpenSim contact forces close the loop in the forward integrator."""

    def _ty(self, res):
        return res.coordinates[:, res.coordinate_names.index("ty")]

    def test_contact_loop_settles_on_floor(self):
        """A ball dropped with contact settles on the floor instead of falling through.

        Feeds an :class:`~newton.opensim.OpenSimContact` into
        :func:`~newton.opensim.simulate_muscle_driven` so the smooth
        sphere/half-space force reacts against gravity every substep.
        """
        model = _contact_drop_model()
        res = simulate_muscle_driven(
            model,
            np.zeros(0),
            initial_coordinates=np.array([0.0, 0.10]),
            initial_speeds=np.zeros(2),
            duration=0.4,
            dt=5.0e-4,
            integrator="rk4",
            contact=True,
        )
        ty = self._ty(res)
        self.assertTrue(np.all(np.isfinite(res.coordinates)))
        self.assertGreater(ty.min(), -0.01)  # never falls through the floor
        self.assertLess(abs(ty[-1] - 0.05), 3.0e-3)  # settles near the sphere radius

    def test_without_contact_ball_falls_through(self):
        """Without contact the same drop falls freely through the floor plane."""
        model = _contact_drop_model()
        res = simulate_muscle_driven(
            model,
            np.zeros(0),
            initial_coordinates=np.array([0.0, 0.10]),
            initial_speeds=np.zeros(2),
            duration=0.4,
            dt=5.0e-4,
            integrator="rk4",
        )
        self.assertLess(self._ty(res)[-1], -0.5)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_contact_bypasses_contact_free_cuda_graph(self):
        """Keep contact in the host-orchestrated path until it is part of the captured step."""
        model = _contact_drop_model()
        sim = MuscleDrivenForward(model, device="cuda:0")
        contact = osim.OpenSimContact(model, device="cuda:0")

        def reject_contact_free_graph(*_args, **_kwargs):
            self.fail("contact simulation entered the contact-free CUDA graph")

        sim._simulate_cuda_graph = reject_contact_free_graph
        result = sim.simulate(
            initial_coordinates=np.array([0.0, 0.10]),
            initial_speeds=np.zeros(2),
            excitations=np.zeros(0),
            duration=0.002,
            dt=0.001,
            contact=contact,
            use_graph=True,
        )
        self.assertTrue(np.all(np.isfinite(result.coordinates)))

    def test_contact_instance_matches_auto_build(self):
        """Passing an OpenSimContact instance matches ``contact=True`` auto-build."""
        model = _contact_drop_model()
        kwargs = {
            "initial_coordinates": np.array([0.0, 0.10]),
            "initial_speeds": np.zeros(2),
            "duration": 0.25,
            "dt": 5.0e-4,
            "integrator": "rk4",
        }
        auto = simulate_muscle_driven(model, np.zeros(0), contact=True, **kwargs)
        inst = simulate_muscle_driven(model, np.zeros(0), contact=osim.OpenSimContact(model), **kwargs)
        self.assertTrue(np.allclose(self._ty(auto), self._ty(inst)))


if __name__ == "__main__":
    unittest.main()
