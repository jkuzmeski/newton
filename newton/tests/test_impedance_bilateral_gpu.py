# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the Warp bilateral mechanics and the bilateral policy interfaces."""

import unittest

import numpy as np
import warp as wp

from projects.impedance_instron.bilateral import gpu
from projects.impedance_instron.bilateral.mechanics import BilateralChain


@wp.kernel
def _evaluate(
    q: wp.array[gpu.Vec9],
    v: wp.array[gpu.Vec9],
    params: wp.array[gpu.ChainParams],
    mass: wp.array[gpu.Mat9],
    bias: wp.array[gpu.Vec9],
    ankles: wp.array2d[wp.vec2d],
    angles: wp.array2d[wp.float64],
    jacobians: wp.array2d[gpu.Vec9],
    solution: wp.array[gpu.Vec9],
):
    i = wp.tid()
    m, b = gpu.dynamics9(q[i], v[i], params[0], wp.float64(9.81))
    mass[i] = m
    bias[i] = b
    for side in range(2):
        position, jx, jz, angle = gpu.ankle9(q[i], params[0], side)
        ankles[i, side] = position
        angles[i, side] = angle
        jacobians[i, 2 * side] = jx
        jacobians[i, 2 * side + 1] = jz
    x, _ok = gpu.cholesky_solve9(m, b)
    solution[i] = x


def _chain() -> BilateralChain:
    return BilateralChain(
        [[0.42, 0.41], [0.43, 0.40]],
        [[0.15, -0.05], [0.16, -0.04]],
        [40.0, 7.0, 3.2, 1.1, 7.2, 3.1, 1.2],
        [[0.33, 0.02], [0.18, 0.01], [0.17, -0.005], [0.05, 0.02], [0.17, 0.0], [0.16, 0.01], [0.04, -0.03]],
        [2.7, 0.12, 0.05, 0.01, 0.13, 0.04, 0.012],
    )


class TestBilateralWarpMechanics(unittest.TestCase):
    def test_warp_dynamics_match_numpy_reference(self):
        """The float64 Warp mass matrix, bias, ankles, and solve match the CPU chain."""
        chain = _chain()
        rng = np.random.default_rng(3)
        count = 8
        q = rng.normal(0.0, 0.6, (count, 9))
        q[:, 1] += 0.9
        q[:, 2] += np.pi / 2
        v = rng.normal(0.0, 3.0, (count, 9))
        device = "cpu"
        outputs = {
            "mass": wp.zeros(count, dtype=gpu.Mat9, device=device),
            "bias": wp.zeros(count, dtype=gpu.Vec9, device=device),
            "ankles": wp.zeros((count, 2), dtype=wp.vec2d, device=device),
            "angles": wp.zeros((count, 2), dtype=wp.float64, device=device),
            "jacobians": wp.zeros((count, 4), dtype=gpu.Vec9, device=device),
            "solution": wp.zeros(count, dtype=gpu.Vec9, device=device),
        }
        wp.launch(
            _evaluate,
            dim=count,
            inputs=[
                wp.array(q, dtype=gpu.Vec9, device=device),
                wp.array(v, dtype=gpu.Vec9, device=device),
                wp.array([gpu.chain_params(chain)], dtype=gpu.ChainParams, device=device),
                *outputs.values(),
            ],
            device=device,
        )
        result = {name: array.numpy() for name, array in outputs.items()}
        for i in range(count):
            mass, bias = chain.dynamics(q[i], v[i])
            np.testing.assert_allclose(result["mass"][i], mass, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(result["bias"][i], bias, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(result["solution"][i], np.linalg.solve(mass, bias), rtol=1e-9, atol=1e-9)
            for s, (side, body) in enumerate((("right", 3), ("left", 6))):
                position, jacobian = chain.ankle(q[i], side)
                np.testing.assert_allclose(result["ankles"][i, s], position, atol=1e-12)
                np.testing.assert_allclose(result["jacobians"][i, 2 * s], jacobian[0], atol=1e-12)
                np.testing.assert_allclose(result["jacobians"][i, 2 * s + 1], jacobian[1], atol=1e-12)
                self.assertAlmostEqual(result["angles"][i, s], chain.angle(q[i], body), places=12)


class TestBilateralInterfaces(unittest.TestCase):
    def test_seed_action_reproduces_engineering_impedance(self):
        """The policy's initial output is the unfitted Hogan seed impedance for both legs."""
        from projects.impedance_instron.bilateral.policy import logits_to_impedance, seed_action  # noqa: PLC0415

        impedance = logits_to_impedance(seed_action())
        np.testing.assert_allclose(impedance["equilibrium_rad"], [0.2, -0.5, 0.0] * 2, atol=1e-12)
        np.testing.assert_allclose(impedance["stiffness_nm_rad"], [250, 300, 200] * 2, atol=1e-9)
        np.testing.assert_allclose(impedance["damping_nms_rad"], [12, 10, 6] * 2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
