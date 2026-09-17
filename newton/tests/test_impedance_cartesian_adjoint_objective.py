# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the exact measured loss and its tape-safe sampling derivatives."""

import unittest

import numpy as np
import warp as wp

from projects.impedance_instron.cartesian.fit import FitConfig
from projects.impedance_instron.cartesian.gpu.adjoint_objective import ObjectiveAdjoint
from projects.impedance_instron.cartesian.gpu.mechanics import Vec5
from projects.impedance_instron.cartesian.gpu.objective import MeasuredObjective


class TestAdjointMeasuredObjective(unittest.TestCase):
    """Preserve native sampling, block normalization, and gradients through history views."""

    def test_loss_and_gradient_through_history_copies(self):
        """Match the measured score and finite differences including terminal and interpolated samples."""
        wp.init()
        devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))
        rng = np.random.default_rng(19)
        reference = {
            "time_s": np.array([0.0, 0.15, 0.35, 0.5]),
            "hip_target_m": rng.normal(size=(4, 2)),
            "joint_target_rad": rng.normal(size=(4, 2)),
            "grf_time_s": np.array([0.0, 0.17, 0.33, 0.4, 0.5]),
            "grf_target_n": rng.normal(size=(5, 2)) * 100,
        }
        q_values = rng.normal(size=(6, 2, 5))
        f_values = rng.normal(size=(5, 2, 2)) * 100
        for device in devices:
            with self.subTest(device=device):
                source = MeasuredObjective(reference, FitConfig(), np.linspace(0.0, 0.5, 6), 2, device)
                adjoint = ObjectiveAdjoint(source)
                q_rows = [wp.array(row[None], dtype=Vec5, device=device, requires_grad=True) for row in q_values]
                f_rows = [wp.array(row[None], dtype=wp.vec2d, device=device, requires_grad=True) for row in f_values]
                q = wp.zeros((6, 2), dtype=Vec5, device=device, requires_grad=True)
                f = wp.zeros((5, 2), dtype=wp.vec2d, device=device, requires_grad=True)
                loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
                integrated = wp.full(2, 5, dtype=int, device=device)
                failure = wp.zeros(2, dtype=int, device=device)

                def forward(q_rows=q_rows, q=q, f_rows=f_rows, f=f, adjoint=adjoint, loss=loss):
                    for t, row in enumerate(q_rows):
                        wp.copy(q[t : t + 1], row)
                    for t, row in enumerate(f_rows):
                        wp.copy(f[t : t + 1], row)
                    adjoint.launch(q, f, loss)

                with wp.Tape() as tape:
                    forward()
                source.launch(q, f, integrated, failure)
                np.testing.assert_array_equal(adjoint.residual.numpy(), source.residual.numpy())
                np.testing.assert_array_equal(adjoint.costs.numpy(), source.costs.numpy())
                np.testing.assert_allclose(loss.numpy()[0], source.loss.numpy().sum(), rtol=1e-15, atol=1e-12)
                tape.backward(loss)
                q_gradient = np.concatenate([row.grad.numpy() for row in q_rows])
                f_gradient = np.concatenate([row.grad.numpy() for row in f_rows])
                np.testing.assert_array_equal(q_gradient[:, :, 2], 0)
                self.assertGreater(np.linalg.norm(q_gradient[-1]), 0.0)
                self.assertEqual(source.description["native_force_samples_outside_simulated_support"], 1)
                for rows, values, gradient in ((q_rows, q_values, q_gradient), (f_rows, f_values, f_gradient)):
                    direction = rng.normal(size=values.shape)
                    direction /= np.linalg.norm(direction)
                    predicted = float(np.sum(gradient * direction))
                    eps = 1e-4
                    results = []
                    for sign in (1.0, -1.0):
                        for row, value in zip(rows, values + sign * eps * direction, strict=True):
                            row.assign(value[None])
                        loss.zero_()
                        forward()
                        results.append(float(loss.numpy()[0]))
                    actual = (results[0] - results[1]) / (2 * eps)
                    self.assertAlmostEqual(predicted, actual, delta=1e-6 + 1e-6 * abs(predicted))
                    for row, value in zip(rows, values, strict=True):
                        row.assign(value[None])


if __name__ == "__main__":
    unittest.main()
