# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the bilateral planar runner mechanics against the single-leg chain and Lagrangian identities."""

import math
import unittest

import numpy as np

from projects.impedance_instron.bilateral.mechanics import BilateralChain, mirror_coordinates
from projects.impedance_instron.hogan.mechanics import Chain


def _chain(left_scale: float = 1.0) -> BilateralChain:
    leg_masses = np.array([7.0, 3.2, 1.1])
    leg_com = np.array([[0.18, 0.01], [0.17, -0.005], [0.05, 0.02]])
    leg_inertia = np.array([0.12, 0.05, 0.01])
    return BilateralChain(
        [[0.42, 0.41], [0.43, 0.40]],
        [[0.15, -0.05], [0.16, -0.04]],
        np.concatenate(([40.0], leg_masses, left_scale * leg_masses)),
        np.vstack(([0.33, 0.02], leg_com, leg_com)),
        np.concatenate(([2.7], leg_inertia, left_scale * leg_inertia)),
    )


_Q = np.array([0.1, 0.95, 1.45, 0.3, -0.6, 0.15, -0.35, -1.2, 0.05])
_V = np.array([3.5, -0.4, 0.8, -2.0, 4.0, -3.0, 2.5, -5.0, 1.5])


class TestBilateralMechanics(unittest.TestCase):
    def test_right_leg_block_matches_single_chain_with_massless_left_leg(self):
        """A negligible left leg reduces the right-leg block to the single-leg chain."""
        bilateral = _chain(left_scale=1e-12)
        single = Chain(
            bilateral.lengths_m[0],
            bilateral.endpoint_local_m[0],
            bilateral.masses_kg[:4],
            bilateral.com_local_m[:4],
            bilateral.inertias_kg_m2[:4],
        )
        mass, bias = bilateral.dynamics(_Q, _V)
        expected_mass, expected_bias = single.dynamics(_Q[:6], _V[:6])
        np.testing.assert_allclose(mass[:6, :6], expected_mass, atol=1e-9)
        np.testing.assert_allclose(bias[:6], expected_bias, atol=1e-9)
        position, jacobian, centripetal = bilateral.point(_Q, 3, np.array([0.1, -0.02]), _V)
        expected = single.point(_Q[:6], 3, np.array([0.1, -0.02]), _V[:6])
        np.testing.assert_allclose(position, expected[0], atol=1e-12)
        np.testing.assert_allclose(jacobian[:, :6], expected[1], atol=1e-12)
        np.testing.assert_allclose(jacobian[:, 6:], 0.0)
        np.testing.assert_allclose(centripetal, expected[2], atol=1e-12)

    def test_bias_matches_lagrangian_coriolis_and_gravity(self):
        """The velocity bias equals dM/dt v - grad(T) by finite differences, plus gravity."""
        chain = _chain()
        eps = 1e-6
        mass, bias = chain.dynamics(_Q, _V, gravity=9.81)
        _, bias_no_gravity = chain.dynamics(_Q, _V, gravity=0.0)
        mass_dot = (chain.dynamics(_Q + eps * _V, _V)[0] - chain.dynamics(_Q - eps * _V, _V)[0]) / (2 * eps)
        gradient = np.zeros(9)
        for i in range(9):
            step = np.zeros(9)
            step[i] = eps
            plus = _V @ chain.dynamics(_Q + step, _V)[0] @ _V
            minus = _V @ chain.dynamics(_Q - step, _V)[0] @ _V
            gradient[i] = 0.25 * (plus - minus) / eps
        np.testing.assert_allclose(bias_no_gravity, mass_dot @ _V - gradient, rtol=1e-6, atol=1e-5)
        gravity = np.zeros(9)
        for i in range(9):
            step = np.zeros(9)
            step[i] = eps
            gravity[i] = (chain.energy(_Q + step, np.zeros(9)) - chain.energy(_Q - step, np.zeros(9))) / (2 * eps)
        np.testing.assert_allclose(bias - bias_no_gravity, gravity, rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh(mass) > 0))

    def test_free_flight_conserves_energy_and_com_momentum(self):
        """Unforced RK4 integration conserves energy; the COM follows a parabola."""
        chain = _chain()
        q, v = _Q.copy(), _V.copy()
        start_energy = chain.energy(q, v)
        com_velocity = chain.com_jacobian(q) @ v

        def acceleration(q, v):
            mass, bias = chain.dynamics(q, v)
            return np.linalg.solve(mass, -bias)

        dt, steps = 1e-4, 1000
        for _ in range(steps):
            k1 = (v, acceleration(q, v))
            k2 = (v + 0.5 * dt * k1[1], acceleration(q + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1]))
            k3 = (v + 0.5 * dt * k2[1], acceleration(q + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1]))
            k4 = (v + dt * k3[1], acceleration(q + dt * k3[0], v + dt * k3[1]))
            q = q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.assertAlmostEqual(chain.energy(q, v), start_energy, delta=1e-6 * abs(start_energy))
        np.testing.assert_allclose(
            chain.com_jacobian(q) @ v, com_velocity - np.array([0.0, 9.81 * dt * steps]), atol=1e-8
        )

    def test_reach_places_each_ankle_and_keeps_foot_angle(self):
        """Per-leg inverse kinematics reaches its target without moving the other leg."""
        chain = _chain()
        q = np.tile(_Q, (2, 1))
        q[1, 2] += 0.05
        for side, body in (("right", 3), ("left", 6)):
            targets = np.array([chain.ankle(row, side)[0] + np.array([0.02, 0.03]) for row in q])
            solved = chain.reach(q, targets, side)
            for row, original, target in zip(solved, q, targets, strict=True):
                np.testing.assert_allclose(chain.ankle(row, side)[0], target, atol=1e-10)
                self.assertAlmostEqual(chain.angle(row, body), chain.angle(original, body), places=10)
            other = slice(6, 9) if side == "right" else slice(3, 6)
            np.testing.assert_array_equal(solved[:, other], q[:, other])

    def test_regularized_leg_fit_matches_reachable_and_holds_unreachable(self):
        """Regularized IK matches reachable ankles and keeps the knee bent past full reach."""
        chain = _chain()
        q = np.tile(_Q, (2, 1))
        target = chain.ankle(_Q, "right")[0] + np.array([0.01, 0.02])
        hip = q[0, :2]
        q[1, 4] = -0.25
        too_far = hip + 1.005 * chain.lengths_m[0].sum() * (target - hip) / np.linalg.norm(target - hip)
        solved = chain.fit_leg(q, np.vstack((target, too_far)), "right")
        np.testing.assert_allclose(chain.ankle(solved[0], "right")[0], target, atol=1e-3)
        self.assertAlmostEqual(chain.angle(solved[0], 3), chain.angle(_Q, 3), places=10)
        self.assertLess(solved[1, 4], -0.05)
        np.testing.assert_array_equal(solved[:, 6:], q[:, 6:])

    def test_mirror_swaps_leg_blocks(self):
        """Mirroring swaps legs and is an involution."""
        mirrored = mirror_coordinates(_Q)
        np.testing.assert_array_equal(mirrored[3:6], _Q[6:9])
        np.testing.assert_array_equal(mirror_coordinates(mirrored), _Q)
        symmetric = BilateralChain(
            [[0.42, 0.41]] * 2,
            [[0.15, -0.05]] * 2,
            [40.0, 7.0, 3.2, 1.1, 7.0, 3.2, 1.1],
            [[0.33, 0.02]] + [[0.18, 0.01], [0.17, -0.005], [0.05, 0.02]] * 2,
            [2.7, 0.12, 0.05, 0.01, 0.12, 0.05, 0.01],
        )
        mass, bias = symmetric.dynamics(_Q, _V)
        mirrored_mass, mirrored_bias = symmetric.dynamics(mirror_coordinates(_Q), mirror_coordinates(_V))
        order = [0, 1, 2, 6, 7, 8, 3, 4, 5]
        np.testing.assert_allclose(mirrored_mass, mass[np.ix_(order, order)], atol=1e-12)
        np.testing.assert_allclose(mirrored_bias, bias[order], atol=1e-12)
        self.assertAlmostEqual(symmetric.angle(_Q, 0), _Q[2] + 0.0 * math.pi)


if __name__ == "__main__":
    unittest.main()
