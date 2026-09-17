# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check explicit leg-solve adjoints and local controller backpropagation."""

import unittest

import numpy as np
import warp as wp

from projects.impedance_instron.cartesian.gpu.gradient_audit import audit_step
from projects.impedance_instron.cartesian.gpu.mechanics import Mat5, Params, Vec5, solve
from projects.impedance_instron.cartesian.trajectory import basis

wp.set_module_options({"enable_backward": True, "fuse_fp": False})


@wp.kernel
def _solve_loss(m: wp.array[Mat5], rhs: wp.array[Vec5], seed: wp.array[Vec5], loss: wp.array[wp.float64]):
    """Seed the solve with a nontrivial scalar cotangent."""
    loss[0] = wp.dot(solve(m[0], rhs[0]), seed[0])


class TestCartesianAdjoint(unittest.TestCase):
    """Verify the shared mass solve and a diagnostic fixed-wrench step."""

    @classmethod
    def setUpClass(cls):
        """Initialize available native test devices."""
        wp.init()
        cls.devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            cls.devices.append(wp.get_device("cuda:0"))

    def test_lower_triangle_solve_adjoint(self):
        """Differentiate the lower-triangle SPD solve, not an unrelated full-matrix inverse."""
        rng = np.random.default_rng(7)
        for device in self.devices:
            for _ in range(3):
                a = rng.normal(size=(5, 5))
                symmetric = a @ a.T + np.eye(5)
                m = symmetric.copy()
                # The forward solve ignores these entries; their adjoints must be zero.
                m[np.triu_indices(5, 1)] = rng.normal(size=10)
                rhs = rng.normal(size=5)
                seed = rng.normal(size=5)
                result = np.linalg.solve(symmetric, rhs)
                dual = np.linalg.solve(symmetric, seed)
                outer = -np.outer(dual, result)
                expected_m = np.tril(outer + outer.T, -1) + np.diag(np.diag(outer))
                m_wp = wp.array(m[None], dtype=Mat5, device=device, requires_grad=True)
                rhs_wp = wp.array(rhs[None], dtype=Vec5, device=device, requires_grad=True)
                seed_wp = wp.array(seed[None], dtype=Vec5, device=device)
                loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    wp.launch(_solve_loss, dim=1, inputs=[m_wp, rhs_wp, seed_wp, loss], device=device)
                tape.backward(loss)
                np.testing.assert_allclose(loss.numpy()[0], result @ seed, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(m_wp.grad.numpy()[0], expected_m, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(rhs_wp.grad.numpy()[0], dual, rtol=1e-11, atol=1e-12)

    def test_fixed_wrench_step_gradients(self):
        """Match central differences through spline, state, velocity, and wrench inputs."""
        params = Params()
        params.lengths = wp.vec2d(0.4, 0.43)
        params.masses = wp.vec3d(7.0, 3.0, 0.7)
        params.com0 = wp.vec2d(0.21, 0.02)
        params.com1 = wp.vec2d(0.16, -0.01)
        params.com2 = wp.vec2d(0.08, 0.02)
        params.inertias = wp.vec3d(0.09, 0.05, 0.003)
        values = {
            "q": np.array([[0.4, 0.85, -1.4, 0.5, -0.4]]),
            "v": np.array([[0.2, -0.3, 0.05, 0.1, -0.15]]),
            "coeff": np.tile([0.42, 0.9, 0.6, -0.25], (12, 1)) + np.arange(12)[:, None] * 0.001,
            "wrench": np.array([[5.0, 120.0, 2.0]]),
        }
        for device in self.devices:
            with self.subTest(device=device):
                result = audit_step(
                    params,
                    6.25e-5,
                    9.81,
                    [1000.0, 1000.0, 80.0, 60.0],
                    [20.0, 20.0, 2.0, 1.0],
                    basis(np.array([0.18]), 0.36, 12)[0],
                    values,
                    device=device,
                )
                self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
