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

"""Tests for OpenSim-style Static Optimization (per-frame muscle-force distribution)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton.opensim as osim
from newton._src.opensim.kinematics import ForwardKinematics
from newton._src.opensim.mocap import Storage
from newton._src.opensim.static_optimization import (
    StaticOptimization,
    solve_frame_activations,
    solve_static_optimization,
)
from newton.tests.test_opensim import _DOUBLE_PENDULUM_OSIM


class TestStaticOptimizationCore(unittest.TestCase):
    """The per-frame quadratic program that distributes a net moment among muscles."""

    def test_recovers_analytic_least_norm(self):
        """Two redundant muscles split a moment as the analytic least-norm solution.

        With passive force zero and ``use_reserves=False`` the frame QP minimizes
        ``a1^2 + a2^2`` subject to ``c1 a1 + c2 a2 = M`` (``c_i = r_i A_i``), whose
        minimizer is ``a_i = c_i M / (c1^2 + c2^2)``.
        """
        r1, r2, A1, A2, m = 0.03, 0.05, 400.0, 300.0, 8.0
        R = np.array([[r1, r2]])
        A = np.array([A1, A2])
        P = np.zeros(2)
        c1, c2 = r1 * A1, r2 * A2
        a1s, a2s = c1 * m / (c1**2 + c2**2), c2 * m / (c1**2 + c2**2)
        a, _u, resid = solve_frame_activations(R, A, P, np.array([m]), use_reserves=False)
        self.assertAlmostEqual(a[0], a1s, places=6)
        self.assertAlmostEqual(a[1], a2s, places=6)
        self.assertLess(abs(resid[0]), 1e-8)

    def test_passive_force_offsets_the_active_moment(self):
        """A nonzero passive force shifts the moment the active fibers must supply."""
        r1, r2, A1, A2, m = 0.03, 0.05, 400.0, 300.0, 8.0
        R = np.array([[r1, r2]])
        A = np.array([A1, A2])
        P = np.array([50.0, 20.0])
        c1, c2 = r1 * A1, r2 * A2
        d = m - (r1 * P[0] + r2 * P[1])
        a1s, a2s = c1 * d / (c1**2 + c2**2), c2 * d / (c1**2 + c2**2)
        a, _, resid = solve_frame_activations(R, A, P, np.array([m]), use_reserves=False)
        self.assertAlmostEqual(a[0], a1s, places=6)
        self.assertAlmostEqual(a[1], a2s, places=6)
        self.assertLess(abs(resid[0]), 1e-8)

    def test_reserve_actuators_close_an_infeasible_balance(self):
        """When the muscles saturate, reserve actuators supply the residual moment."""
        R = np.array([[0.03, 0.05]])
        A = np.array([400.0, 300.0])
        P = np.zeros(2)
        a, u, resid = solve_frame_activations(R, A, P, np.array([1000.0]), use_reserves=True, reserve_optimal_force=1.0)
        self.assertLessEqual(a.max(), 1.0 + 1e-9)
        self.assertGreaterEqual(a.min(), -1e-9)
        self.assertGreater(abs(u[0]), 1.0)  # reserve carries the leftover moment
        self.assertLess(abs(resid[0]), 1e-5)


class TestStaticOptimization(unittest.TestCase):
    """Whole-model static optimization on a muscle-driven double pendulum."""

    def _model(self):
        """Double pendulum with two redundant muscles crossing the first joint."""
        model = osim.parse_osim(_DOUBLE_PENDULUM_OSIM)
        fk = ForwardKinematics(model)
        b0 = fk.body_names[1]
        params = {
            "max_isometric_force": 2000.0,
            "optimal_fiber_length": 0.5,
            "tendon_slack_length": 0.05,
            "pennation_angle_at_optimal": 0.0,
        }
        model.muscles = [
            osim.OsimMuscle(
                name="m1",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="a", body="ground", location=(0.05, 0.05, 0.0)),
                    osim.OsimPathPoint(name="b", body=b0, location=(0.03, -0.1, 0.0)),
                ],
                params=dict(params),
            ),
            osim.OsimMuscle(
                name="m2",
                type="Thelen2003Muscle",
                path_points=[
                    osim.OsimPathPoint(name="c", body="ground", location=(0.08, 0.05, 0.0)),
                    osim.OsimPathPoint(name="d", body=b0, location=(0.05, -0.1, 0.0)),
                ],
                params=dict(params),
            ),
        ]
        return model

    def test_moment_balance_and_bounds(self):
        """Every frame satisfies the moment balance with activations in [0, 1]."""
        so = StaticOptimization(self._model(), use_reserves=True, reserve_optimal_force=1.0)
        q = np.array([[0.2, 0.0], [0.4, 0.0], [0.1, 0.05]])
        z = np.zeros_like(q)
        res = so.solve(q, z, z)
        self.assertEqual(res.activations.shape, (3, 2))
        self.assertLess(np.max(np.abs(res.moment_residuals)), 1e-4)
        self.assertGreaterEqual(res.activations.min(), -1e-9)
        self.assertLessEqual(res.activations.max(), 1.0 + 1e-9)
        # Recovered muscle force equals a * A + P by construction.
        self.assertTrue(np.all(np.isfinite(res.muscle_forces)))

    def test_sto_round_trip(self):
        """Activations written to ``.sto`` read back unchanged."""
        so = StaticOptimization(self._model())
        q = np.array([[0.2, 0.0], [0.4, 0.0]])
        z = np.zeros_like(q)
        res = so.solve(q, z, z)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "so.sto"
            res.write_sto(path)
            back = osim.read_storage(path)
        self.assertEqual(back.labels, [f"{n}_activation" for n in res.muscle_names])
        np.testing.assert_allclose(back.data, res.activations, atol=1e-6)

    def test_solve_from_motion_pipeline(self):
        """The motion-driven pipeline resolves activations at the requested frames."""
        model = self._model()
        times = np.linspace(0.0, 0.4, 9)
        angle = 0.2 + 0.1 * np.sin(2.0 * np.pi * times)
        data = np.column_stack([np.rad2deg(angle), np.zeros_like(times)])
        motion = Storage(times=times, labels=["j1_q", "j2_q"], data=data, in_degrees=True)
        res = solve_static_optimization(model, motion, cutoff=6.0)
        self.assertEqual(res.activations.shape[0], len(times))
        self.assertEqual(res.activations.shape[1], 2)
        self.assertGreaterEqual(res.activations.min(), -1e-9)
        self.assertLessEqual(res.activations.max(), 1.0 + 1e-9)
        self.assertLess(np.max(np.abs(res.moment_residuals)), 1e-3)


if __name__ == "__main__":
    unittest.main()
