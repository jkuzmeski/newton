# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check tangential coupling, isolated histories, gradients and graph replay."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.friction_solver import FrictionParams, FrictionSolver


@wp.kernel
def _loss(force: wp.array[wp.vec2], loss: wp.array[float]):
    loss[0] = force[0][0] + 0.3 * force[1][1]


class TestFrictionSolver(unittest.TestCase):
    """Test the friction-only solver against independent mechanical references."""

    def _inputs(self, device, *, worlds=1, grad=False):
        """Prepare two off-center contacts with shared stiffness and finite mobility."""
        points = np.array([[-0.08, -0.03, -0.04], [0.06, 0.04, -0.04]], np.float32)
        matrices = np.diag([0.5, 0.5, 0.5, 5.0, 4.0, 8.0]).astype(np.float32)
        params = []
        for _ in range(worlds):
            p = FrictionParams()
            p.mu = 100.0
            p.viscous_ratio = 0.2
            p.release_dwell = 0.001
            p.smoothing_speed = 0.01
            params.append(p)
        arrays = [
            (np.tile(points, (worlds, 1)), wp.vec3),
            (np.full(2 * worlds, 100.0), float),
            (np.zeros((worlds, 3)), wp.vec3),
            (np.tile([0.4, -0.2, 0, 0.2, 0.0, 0.3], (worlds, 1)), wp.spatial_vector),
            (np.tile(matrices, (worlds, 1, 1)), wp.spatial_matrix),
            ([80000.0, 100000.0], float),
            ([0.0, 0.0], float),
        ]
        values = [wp.array(a, dtype=d, device=device, requires_grad=grad) for a, d in arrays]
        values.append(wp.array(params, dtype=FrictionParams, device=device, requires_grad=grad))
        values.extend(
            [
                wp.array(np.tile(points[:, :2], (worlds, 1)), dtype=wp.vec2, device=device, requires_grad=grad),
                wp.ones(worlds * 2, dtype=int, device=device),
                wp.zeros(worlds * 2, dtype=float, device=device, requires_grad=grad),
            ]
        )
        return values

    def test_sticking_matches_coupled_backward_euler(self):
        """Match an independent six-dimensional linear sticking solve on CPU and CUDA."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=str(device)):
                inputs = self._inputs(device)
                dt = 0.01
                points = inputs[0].numpy()
                v0 = inputs[3].numpy()[0]
                mobility = inputs[4].numpy()[0]
                kt = inputs[5].numpy()
                rows = []
                stiffness = np.zeros((6, 6))
                for r, k in zip(points, kt, strict=True):
                    b = np.array([[1, 0, 0, 0, r[2], -r[1]], [0, 1, 0, -r[2], 0, r[0]]])
                    rows.append(b)
                    stiffness += k * b.T @ b
                expected = np.linalg.solve(np.eye(6) + dt**2 * mobility @ stiffness, v0)
                solver = FrictionSolver(2, iterations=4, device=device)
                result = solver.solve(*inputs, dt)
                np.testing.assert_allclose(result.velocity.numpy()[0], expected, rtol=2e-5, atol=3e-7)
                expected_force = np.array([-k * dt * b @ expected for k, b in zip(kt, rows, strict=True)])
                np.testing.assert_allclose(result.force.numpy(), expected_force, rtol=3e-5, atol=2e-4)
                self.assertLess(float(result.linear_residual.numpy()[0]), 2e-6)
                self.assertLess(float(result.angular_residual.numpy()[0]), 2e-6)

    def test_large_step_does_not_reverse_linear_sticking_velocity(self):
        """Avoid the explicit stiff-bristle overshoot in a controlled translation problem."""
        device = wp.get_device("cpu")
        values = self._inputs(device)
        values[3].assign(np.array([[0.5, 0, 0, 0, 0, 0]], np.float32))
        values[4].assign(np.array([np.diag([0.5, 0, 0, 0, 0, 0])], np.float32))
        dt = 0.01
        legacy = FrictionSolver(2, mode="bristle", device=device).solve(*values, dt)
        implicit = FrictionSolver(2, iterations=4, device=device).solve(*values, dt)
        explicit_v = 0.5 + dt * 0.5 * legacy.wrench.numpy()[0, 0]
        self.assertLess(explicit_v, 0.0)
        self.assertGreater(float(implicit.velocity.numpy()[0, 0]), 0.0)
        self.assertLess(float(implicit.velocity.numpy()[0, 0]), 0.5)

    def test_worlds_and_frozen_inputs(self):
        """Keep normal data read-only and match isolated solves in a two-world batch."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            values = self._inputs(device, worlds=2)
            vel = values[3].numpy()
            vel[1] *= -0.7
            values[3].assign(vel)
            before = [a.numpy().copy() for a in values if a.dtype != FrictionParams]
            result = FrictionSolver(2, 2, iterations=4, device=device).solve(*values, 0.003)
            for world in range(2):
                isolated = self._inputs(device)
                isolated[3].assign(vel[world : world + 1])
                expected = FrictionSolver(2, iterations=4, device=device).solve(*isolated, 0.003)
                np.testing.assert_allclose(
                    result.force.numpy()[2 * world : 2 * world + 2], expected.force.numpy(), atol=1e-5
                )
            for actual, expected in zip((a for a in values if a.dtype != FrictionParams), before, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected)

    def test_stiffness_gradient(self):
        """Match the unrolled coupled-solve gradient to central finite differences."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            values = self._inputs(device, grad=True)
            solver = FrictionSolver(2, iterations=3, device=device, requires_grad=True)
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            with wp.Tape() as tape:
                result = solver.solve(*values, 0.002)
                wp.launch(_loss, dim=1, inputs=[result.force, loss], device=device)
            tape.backward(loss)
            analytic = values[5].grad.numpy().copy()
            base = values[5].numpy().copy()
            numeric = []
            for index in range(2):
                h = base[index] * 0.003
                changed = base.copy()
                changed[index] += h
                values[5].assign(changed)
                high = solver.solve(*values, 0.002).force.numpy().copy()
                changed[index] -= 2 * h
                values[5].assign(changed)
                low = solver.solve(*values, 0.002).force.numpy().copy()
                numeric.append(((high[0, 0] + 0.3 * high[1, 1]) - (low[0, 0] + 0.3 * low[1, 1])) / (2 * h))
            np.testing.assert_allclose(analytic, numeric, rtol=0.015, atol=1e-6)

    def test_sliding_mu_gradient_and_history_slots(self):
        """Differentiate two sliding steps while retaining each step's incoming history."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            values = self._inputs(device, grad=True)
            values[3].assign(np.array([[2.0, 0.3, 0, 0, 0, 0]], np.float32))
            params = FrictionParams()
            params.mu = 0.6
            params.viscous_ratio = 0.2
            params.release_dwell = 0.001
            params.smoothing_speed = 0.01
            values[7].assign([params])
            solver = FrictionSolver(2, iterations=5, max_steps=2, device=device, requires_grad=True)
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

            def forward(solver=solver, values=values, loss=loss, device=device):
                first = solver.solve(*values, 0.002, step=0)
                second_values = list(values)
                second_values[3] = first.velocity
                second_values[8:11] = [first.anchor, first.stuck, first.dwell]
                second = solver.solve(*second_values, 0.002, step=1)
                wp.launch(_loss, dim=1, inputs=[second.force, loss], device=device)
                return first, second

            with wp.Tape() as tape:
                first, second = forward()
            tape.backward(loss)
            analytic = float(values[7].grad.numpy()["mu"][0])
            first_force = first.force.numpy().copy()
            self.assertIsNot(first.force, second.force)
            self.assertGreater(np.linalg.norm(first_force), 0.0)
            h = 0.003
            params.mu = 0.6 + h
            values[7].assign([params])
            forward()
            high = float(loss.numpy()[0])
            params.mu = 0.6 - h
            values[7].assign([params])
            forward()
            low = float(loss.numpy()[0])
            numeric = (high - low) / (2 * h)
            self.assertGreater(abs(numeric), 1.0)
            self.assertAlmostEqual(analytic / numeric, 1.0, delta=0.02)

    def test_graph_replay(self):
        """Match eager and captured coupled friction with stable caller-owned inputs."""
        for device in wp.get_cuda_devices():
            values = self._inputs(device)
            solver = FrictionSolver(2, iterations=3, device=device)
            expected = solver.solve(*values, 0.002).force.numpy().copy()
            with wp.ScopedCapture(device=device) as capture:
                result = solver.solve(*values, 0.002)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(result.force.numpy(), expected)

    def test_configuration_validation(self):
        """Reject invalid counts, models and step slots before a kernel launch."""
        with self.assertRaises(ValueError):
            FrictionSolver(2, mode="unknown")
        with self.assertRaises(ValueError):
            FrictionSolver(0)
        solver = FrictionSolver(2, device="cpu")
        inputs = self._inputs(wp.get_device("cpu"))
        with self.assertRaises(ValueError):
            solver.solve(*inputs, -0.01)
        with self.assertRaises(ValueError):
            solver.solve(*inputs, 0.01, step=2)


if __name__ == "__main__":
    unittest.main()
