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

"""Tests for the Newton-native Moco tools (``MocoInverse`` and ``MocoTrack``)."""

import os
import tempfile
import unittest

import numpy as np

import newton.opensim as osim
from newton._src.opensim.dynamics import InverseDynamics
from newton._src.opensim.mocap import Storage, read_storage
from newton._src.opensim.moco import MocoInverse, MocoTrack
from newton._src.opensim.muscle_force import MuscleForces
from newton._src.opensim.static_optimization import solve_frame_activations

# A light single-pin segment whose gravity moment two flexor muscles can balance.
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


def _pin_model(num_muscles=2):
    """Build the pin model with one or two flexor muscles."""
    model = osim.parse_osim(_PIN_MUSCLE_OSIM)
    muscles = [
        osim.OsimMuscle(
            name="flex",
            type="Thelen2003Muscle",
            path_points=[
                osim.OsimPathPoint(name="a", body="ground", location=(0.08, 0.5, 0.0)),
                osim.OsimPathPoint(name="b", body="seg", location=(0.03, -0.1, 0.0)),
            ],
            params=_MUSCLE_PARAMS.copy(),
        )
    ]
    if num_muscles == 2:
        muscles.append(
            osim.OsimMuscle(
                name="flex2",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="c", body="ground", location=(0.12, 0.5, 0.0)),
                    osim.OsimPathPoint(name="d", body="seg", location=(0.05, -0.1, 0.0)),
                ],
                params=_MUSCLE_PARAMS.copy(),
            )
        )
    model.muscles = muscles
    return model


def _held_motion(angle_rad, t_end=0.8, n=30):
    """A constant-angle prescribed coordinate trajectory (degrees)."""
    times = np.linspace(0.0, t_end, n)
    data = np.column_stack([np.full_like(times, np.rad2deg(angle_rad))])
    return Storage(times=times, labels=["pin_angle"], data=data, in_degrees=True)


class TestMocoInverse(unittest.TestCase):
    """Validate MocoInverse redundancy resolution and excitation recovery."""

    def test_single_muscle_recovers_exact_activation(self):
        """A fully determined single muscle returns the algebraic required activation."""
        model = _pin_model(1)
        ang = 0.2
        q = np.array([[ang]])
        z = np.zeros_like(q)
        idv = InverseDynamics(model)
        mf = MuscleForces(model)
        tau = idv.solve(q, z, z)[0]
        arm = mf.paths.moment_arms(q)[0].T
        active = (mf.forces(np.ones((1, 1)), q, z) - mf.forces(np.zeros((1, 1)), q, z))[0]
        passive = mf.forces(np.zeros((1, 1)), q, z)[0]
        a_exact = (tau[0] - arm[0, 0] * passive[0]) / (arm[0, 0] * active[0])

        sol = MocoInverse(model, use_reserves=False).solve(_held_motion(ang), cutoff=0.0, num_nodes=25)
        mid = len(sol.times) // 2
        self.assertLess(sol.constraint_violation, 1e-4)
        self.assertAlmostEqual(sol.activations[mid, 0], a_exact, places=3)
        # Held pose: excitation equals activation (steady-state activation dynamics).
        self.assertAlmostEqual(sol.excitations[mid, 0], a_exact, places=3)

    def test_redundant_muscles_match_least_norm(self):
        """Two redundant muscles at a held pose reproduce the least-effort activations."""
        model = _pin_model(2)
        ang = 0.2
        q = np.array([[ang]])
        z = np.zeros_like(q)
        idv = InverseDynamics(model)
        mf = MuscleForces(model)
        tau = idv.solve(q, z, z)[0]
        arm = mf.paths.moment_arms(q)[0].T
        active = (mf.forces(np.ones((1, 2)), q, z) - mf.forces(np.zeros((1, 2)), q, z))[0]
        passive = mf.forces(np.zeros((1, 2)), q, z)[0]
        a_ref, _, _ = solve_frame_activations(arm, active, passive, tau, use_reserves=False)

        sol = MocoInverse(model, use_reserves=False).solve(_held_motion(ang), cutoff=0.0, num_nodes=25)
        mid = len(sol.times) // 2
        np.testing.assert_allclose(sol.activations[mid], a_ref, atol=2e-3)
        # Held pose: excitations equal activations.
        np.testing.assert_allclose(sol.excitations[mid], sol.activations[mid], atol=2e-3)

    def test_reproduces_inverse_dynamics_moments(self):
        """The recovered muscle forces reproduce the ID net joint moment at every node."""
        model = _pin_model(2)
        ang = 0.2
        sol = MocoInverse(model, use_reserves=False).solve(_held_motion(ang), cutoff=0.0, num_nodes=25)
        idv = InverseDynamics(model)
        mf = MuscleForces(model)
        q = np.full((len(sol.times), 1), ang)
        z = np.zeros_like(q)
        tau = idv.solve(q, z, z)
        arms = mf.paths.moment_arms(q)
        moment = np.einsum("nmc,nm->nc", arms, sol.muscle_forces)
        np.testing.assert_allclose(moment[:, 0], tau[:, 0], atol=1e-3)

    def test_excitation_inversion_matches_forward_dynamics(self):
        """Recovered excitations reproduce the activation trajectory under forward dynamics."""
        model = _pin_model(1)
        times = np.linspace(0.0, 1.0, 60)
        angle = 0.15 + 0.1 * np.sin(2.0 * np.pi * 0.8 * times)
        motion = Storage(times=times, labels=["pin_angle"], data=np.column_stack([np.rad2deg(angle)]), in_degrees=True)
        sol = MocoInverse(model, use_reserves=False).solve(motion, cutoff=6.0, num_nodes=60, activation_dynamics=True)
        mf = MuscleForces(model)
        act = sol.activations[:, 0]
        exc = sol.excitations[:, 0]
        forward = np.zeros_like(act)
        forward[0] = act[0]
        for k in range(len(sol.times) - 1):
            dt = sol.times[k + 1] - sol.times[k]
            forward[k + 1] = mf.integrate_activation(np.array([forward[k]]), np.array([exc[k]]), dt, substeps=16)[0, 0]
        rms = np.sqrt(np.mean((forward - act) ** 2))
        self.assertLess(rms, 1.5e-2)

    def test_ignore_activation_dynamics_sets_excitation_to_activation(self):
        """With activation dynamics disabled the excitations equal the activations."""
        model = _pin_model(1)
        times = np.linspace(0.0, 1.0, 40)
        angle = 0.15 + 0.05 * np.sin(2.0 * np.pi * 0.5 * times)
        motion = Storage(times=times, labels=["pin_angle"], data=np.column_stack([np.rad2deg(angle)]), in_degrees=True)
        sol = MocoInverse(model, use_reserves=False).solve(motion, cutoff=6.0, num_nodes=40, activation_dynamics=False)
        np.testing.assert_allclose(sol.excitations, sol.activations, atol=1e-12)

    def test_write_sto_round_trip(self):
        """Excitations written to ``.sto`` round-trip through the reader."""
        model = _pin_model(2)
        sol = MocoInverse(model, use_reserves=False).solve(_held_motion(0.2), cutoff=0.0, num_nodes=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "controls.sto")
            sol.write_sto(path)
            back = read_storage(path)
        self.assertEqual(list(back.labels), list(sol.muscle_names))
        np.testing.assert_allclose(back.data, sol.excitations, atol=1e-6)


class TestMocoTrack(unittest.TestCase):
    """Validate MocoTrack coordinate tracking on a torque-driven model."""

    def test_tracks_smooth_reference(self):
        """A torque-driven pin joint tracks a smooth reference to sub-mrad RMS."""
        model = _pin_model(1)
        times = np.linspace(0.0, 1.0, 60)
        ref_angle = 0.1 * np.sin(2.0 * np.pi * 0.5 * times)
        ref = Storage(times=times, labels=["pin_angle"], data=np.column_stack([ref_angle]), in_degrees=False)
        sol = MocoTrack(model).solve(
            ref,
            tracking_weight=100.0,
            control_effort_weight=1e-4,
            control_bounds=(-50.0, 50.0),
            num_mesh_intervals=25,
            tolerance=1e-7,
        )
        self.assertTrue(sol.converged)
        self.assertLess(sol.constraint_violation, 1e-4)
        self.assertLess(sol.tracking_rms, 5e-3)


if __name__ == "__main__":
    unittest.main()
