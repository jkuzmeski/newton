# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import unittest

import numpy as np

import newton
from projects.supermimic.analysis import (
    GRAVITY,
    impact_metrics,
    inverse_dynamics,
    quasi_stiffness,
    stance_intervals,
    stride_metrics,
)
from projects.supermimic.body import BodyParams, add_body
from projects.supermimic.reference import F01, SHOE_DATASET, load_shoe, load_subject
from projects.supermimic.sim import FOOT_BODIES, SIDES, shod_foot


def _standing(frames: int = 5) -> np.ndarray:
    joint_q = np.zeros((frames, 17))
    joint_q[:, 2] = 1.0
    joint_q[:, 6] = 1.0
    return joint_q


class TestSupermimicAnalysis(unittest.TestCase):
    def test_static_foot_load_moments(self):
        """Balance a vertical force 0.1 m in front of the right ankle and a free moment at every joint."""
        params = BodyParams()
        joint_q = _standing()
        ankle = (*params.hip_offset[:2], 1.0 + params.knee_offset[2] + params.ankle_offset[2])
        grf, cop, free = np.zeros((3, 5, 2, 3))
        grf[:, 0] = (0.0, 0.0, 600.0)
        cop[:, 0] = np.add(ankle, (0.1, 0.0, -0.08))
        free[:, 0] = (0.0, 0.0, 5.0)
        moments = inverse_dynamics(joint_q, 0.01, params, grf, cop, free)[2]
        # Moments about +y that hold each foot: the ground force in front of the right ankle and the foot weights.
        weight = params.foot.com[0] * params.foot.mass * GRAVITY
        np.testing.assert_allclose(
            moments[:5], np.array((-1.0, 0.0, 0.0, 1.0, -1.0)) * (60.0 - weight) - (0, 0, 5, 0, 0), atol=1e-9
        )
        np.testing.assert_allclose(moments[5:], np.array((-1.0, 0.0, 0.0, 1.0, -1.0)) * -weight, atol=1e-9)

    def test_recovers_simulated_joint_torques(self):
        """Recover the joint torques driving a free-flying runner from its simulated coordinates."""
        self._recover_simulated_joint_torques(None)

    @unittest.skipUnless(SHOE_DATASET.exists(), "needs the column-bed dataset")
    def test_recovers_simulated_joint_torques_with_shoes(self):
        """Recover the joint torques of a free-flying runner whose feet carry the shoe mass."""
        shoe = load_shoe(load_subject(F01).shoe)
        shoe = dataclasses.replace(
            shoe, left=dataclasses.replace(shoe.left, mass_kg=1.0), right=dataclasses.replace(shoe.right, mass_kg=1.0)
        )
        self._recover_simulated_joint_torques(shoe)

    def _recover_simulated_joint_torques(self, shoe):
        # Free joints: the swinging segments must not reach the range-of-motion limits.
        params = dataclasses.replace(BodyParams(), joint_limits=((-1.0e6, 1.0e6),) * 5)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        add_body(builder, params)
        model = builder.finalize(device="cpu")
        if shoe is not None:
            mass, com, inertia = (a.numpy() for a in (model.body_mass, model.body_com, model.body_inertia))
            for side, foot in zip(SIDES, FOOT_BODIES, strict=True):
                mass[foot], com[foot], inertia[foot] = shod_foot(params, shoe, side)
            model.body_mass.assign(mass)
            model.body_inv_mass.assign(1.0 / mass)
            model.body_com.assign(com)
            model.body_inertia.assign(inertia)
            model.body_inv_inertia.assign(np.linalg.inv(inertia))
        solver = newton.solvers.SolverFeatherstone(model)
        state, next_state, control = model.state(), model.state(), model.control()
        joint_q = np.zeros(17, np.float32)
        joint_q[2], joint_q[6] = 1.0, 1.0
        joint_q[7:] = (0.3, 0.05, 0.1, 0.5, 0.1, -0.2, -0.05, 0.0, 0.3, -0.1)
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        dt, steps, stride = 5.0e-5, 4000, 40
        amplitude = np.array((20.0, 10.0, 5.0, 8.0, 2.0, -15.0, 8.0, -4.0, 6.0, -1.5))
        torque, coordinates = [], []
        for step in range(steps):
            tau = amplitude * np.sin(2.0 * np.pi * 3.0 * step * dt + np.arange(10))
            control.joint_f.assign(np.concatenate((np.zeros(6), tau)).astype(np.float32))
            torque.append(tau)
            coordinates.append(state.joint_q.numpy().astype(float))
            solver.step(state, next_state, control, None, dt)
            state, next_state = next_state, state
        # Sample coarser than the step so float32 coordinates still give accelerations by differences.
        zero = np.zeros((steps // stride, 2, 3))
        moments = inverse_dynamics(np.array(coordinates[::stride]), stride * dt, params, zero, zero, zero, shoe)
        error = (moments - np.array(torque[::stride]))[2:-2]
        np.testing.assert_array_less(np.abs(error).max(0), 0.02 * np.abs(amplitude))

    def test_quasi_stiffness_of_a_linear_spring(self):
        """Return the spring constant in both phases and no loading phase for a monotonic joint."""
        t = np.linspace(0.0, 1.0, 201)
        flexion = 0.2 + 0.4 * np.sin(np.pi * t)
        stances = np.array(((20, 181),))
        np.testing.assert_allclose(quasi_stiffness(flexion, -300.0 * (flexion - 0.2), stances), [[300.0, 300.0]])
        extension = 0.5 - 0.6 * t
        stiffness = quasi_stiffness(extension, 100.0 * (extension - 0.5) * np.sin(np.pi * t), stances)
        self.assertTrue(np.isnan(stiffness[0, 0]))

    def test_impact_metrics_of_a_ramp(self):
        """Measure the rise and slope of a linear ramp to a peak that a later ripple does not exceed."""
        dt = 1e-3
        force = np.concatenate((np.zeros(5), np.linspace(0.0, 1000.0, 21), [900.0, 950.0, 800.0], np.full(20, 700.0)))
        m = impact_metrics(force, dt, body_weight=500.0)
        self.assertAlmostEqual(m["impact_bw"], 2.0)
        self.assertAlmostEqual(m["rise_s"], 0.012)
        self.assertAlmostEqual(m["loading_rate_bw_s"], 100.0)

    def test_impact_metrics_reject_a_force_already_in_contact(self):
        """Refuse a record that starts in contact instead of wrapping around to its peak."""
        force = np.concatenate((np.linspace(400.0, 1500.0, 12), np.full(10, 1200.0)))
        with self.assertRaises(ValueError):
            impact_metrics(force, 1e-3, body_weight=600.0)

    def test_stride_metrics_of_alternating_half_sines(self):
        """Measure contact, flight, cadence, peak and the 20-80 % loading rate of half-sine stances."""
        dt, contact, flight, peak = 1.0e-4, 0.25, 0.1, 1500.0
        t = np.arange(0.0, 1.9, dt)
        phase = np.mod(t, 2.0 * (contact + flight))
        grf_z = np.zeros((len(t), 2))
        for foot in range(2):
            local = phase - foot * (contact + flight)
            grf_z[:, foot] = np.where((local >= 0.0) & (local < contact), peak * np.sin(np.pi * local / contact), 0.0)
        self.assertEqual(len(stance_intervals(grf_z[:, 0])), 3)
        metrics = stride_metrics(grf_z, dt, 600.0)
        rise = (np.arcsin(0.8) - np.arcsin(0.2)) * contact / np.pi
        self.assertEqual(metrics["stances"], 5)
        self.assertAlmostEqual(metrics["cadence_steps_per_min"], 60.0 / (contact + flight), delta=0.1)
        self.assertAlmostEqual(metrics["peak_grf_bw"], 2.5, places=4)
        self.assertAlmostEqual(metrics["loading_rate_bw_s"], 0.6 * 2.5 / rise, delta=0.05)
        self.assertAlmostEqual(
            metrics["contact_time_s"], contact - 2.0 * np.arcsin(50.0 / peak) * contact / np.pi, delta=2 * dt
        )
        self.assertAlmostEqual(
            metrics["flight_time_s"], flight + 2.0 * np.arcsin(50.0 / peak) * contact / np.pi, delta=2 * dt
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
