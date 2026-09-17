# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for digital shoe friction laws and tangent Jacobians."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.contact import bristle_step
from projects.digital_shoe.friction_law import bristle_force_tangent, regularized_force_tangent


@wp.kernel
def _eval_bristle_step(
    pos: wp.vec2,
    vel: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    anchor: wp.vec2,
    stuck: int,
    dwell: float,
    out_force: wp.array[wp.vec2],
    out_anchor: wp.array[wp.vec2],
    out_stuck: wp.array[int],
    out_dwell: wp.array[float],
):
    f, a, s, d = bristle_step(pos, vel, dt, normal, kt, kv, mu, viscous_ratio, release_dwell, anchor, stuck, dwell)
    out_force[0] = f
    out_anchor[0] = a
    out_stuck[0] = s
    out_dwell[0] = d


@wp.kernel
def _eval_bristle_force_tangent(
    pos: wp.vec2,
    vel: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    anchor: wp.vec2,
    stuck: int,
    dwell: float,
    out_force: wp.array[wp.vec2],
    out_jacobian: wp.array[wp.mat22],
    out_anchor: wp.array[wp.vec2],
    out_stuck: wp.array[int],
    out_dwell: wp.array[float],
):
    f, J, a, s, d = bristle_force_tangent(
        pos, vel, dt, normal, kt, kv, mu, viscous_ratio, release_dwell, anchor, stuck, dwell
    )
    out_force[0] = f
    out_jacobian[0] = J
    out_anchor[0] = a
    out_stuck[0] = s
    out_dwell[0] = d


@wp.kernel
def _eval_regularized_force_tangent(
    vel: wp.vec2,
    dt: float,
    normal: float,
    mu: float,
    smoothing_speed: float,
    out_force: wp.array[wp.vec2],
    out_jacobian: wp.array[wp.mat22],
):
    f, J = regularized_force_tangent(vel, dt, normal, mu, smoothing_speed)
    out_force[0] = f
    out_jacobian[0] = J


@wp.kernel
def _eval_bristle_loss_for_ad(
    pos: wp.vec2,
    vel: wp.vec2,
    dt: float,
    normal: float,
    kt: wp.array[float],
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    anchor: wp.vec2,
    stuck: int,
    dwell: float,
    out_loss: wp.array[float],
):
    f, _J, _a, _s, _d = bristle_force_tangent(
        pos, vel, dt, normal, kt[0], kv, mu, viscous_ratio, release_dwell, anchor, stuck, dwell
    )
    out_loss[0] = f[0] * 1.5 + f[1] * 0.7


class TestDigitalShoeFrictionLaw(unittest.TestCase):
    def test_bristle_canonical_parity_cpu_and_cuda(self):
        """Verify bristle_force_tangent matches bristle_step exactly across devices."""
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")

        test_cases = [
            # Sticking elastic trial
            {
                "pos": (0.0, 0.0),
                "vel": (0.01, 0.02),
                "dt": 0.01,
                "normal": 50.0,
                "kt": 1000.0,
                "kv": 10.0,
                "mu": 0.8,
                "viscous_ratio": 0.5,
                "release_dwell": 0.05,
                "anchor": (0.0, 0.0),
                "stuck": 1,
                "dwell": 0.0,
            },
            # Sliding elastic + cone-limited viscous
            {
                "pos": (0.1, -0.05),
                "vel": (2.0, -1.0),
                "dt": 0.01,
                "normal": 20.0,
                "kt": 2000.0,
                "kv": 50.0,
                "mu": 0.5,
                "viscous_ratio": 0.3,
                "release_dwell": 0.05,
                "anchor": (0.0, 0.0),
                "stuck": 1,
                "dwell": 0.0,
            },
            # Unloaded dwell transition
            {
                "pos": (0.0, 0.0),
                "vel": (0.0, 0.0),
                "dt": 0.02,
                "normal": 0.0,
                "kt": 100.0,
                "kv": 0.0,
                "mu": 0.5,
                "viscous_ratio": 0.5,
                "release_dwell": 0.05,
                "anchor": (0.1, 0.2),
                "stuck": 1,
                "dwell": 0.04,
            },
            # Inactive bristle (stuck == 0) activating
            {
                "pos": (0.5, 0.5),
                "vel": (0.1, -0.2),
                "dt": 0.01,
                "normal": 10.0,
                "kt": 500.0,
                "kv": 20.0,
                "mu": 0.6,
                "viscous_ratio": 0.5,
                "release_dwell": 0.05,
                "anchor": (0.0, 0.0),
                "stuck": 0,
                "dwell": 0.0,
            },
        ]

        for device in devices:
            out_b_force = wp.zeros(1, dtype=wp.vec2, device=device)
            out_b_anchor = wp.zeros(1, dtype=wp.vec2, device=device)
            out_b_stuck = wp.zeros(1, dtype=int, device=device)
            out_b_dwell = wp.zeros(1, dtype=float, device=device)

            out_t_force = wp.zeros(1, dtype=wp.vec2, device=device)
            out_t_jacobian = wp.zeros(1, dtype=wp.mat22, device=device)
            out_t_anchor = wp.zeros(1, dtype=wp.vec2, device=device)
            out_t_stuck = wp.zeros(1, dtype=int, device=device)
            out_t_dwell = wp.zeros(1, dtype=float, device=device)

            for tc in test_cases:
                inputs_list = [
                    wp.vec2(*tc["pos"]),
                    wp.vec2(*tc["vel"]),
                    tc["dt"],
                    tc["normal"],
                    tc["kt"],
                    tc["kv"],
                    tc["mu"],
                    tc["viscous_ratio"],
                    tc["release_dwell"],
                    wp.vec2(*tc["anchor"]),
                    tc["stuck"],
                    tc["dwell"],
                ]
                wp.launch(
                    _eval_bristle_step,
                    dim=1,
                    inputs=inputs_list,
                    outputs=[out_b_force, out_b_anchor, out_b_stuck, out_b_dwell],
                    device=device,
                )
                wp.launch(
                    _eval_bristle_force_tangent,
                    dim=1,
                    inputs=inputs_list,
                    outputs=[out_t_force, out_t_jacobian, out_t_anchor, out_t_stuck, out_t_dwell],
                    device=device,
                )

                np.testing.assert_allclose(
                    out_t_force.numpy(),
                    out_b_force.numpy(),
                    atol=1e-5,
                    err_msg=f"Force mismatch on device {device}",
                )
                np.testing.assert_allclose(
                    out_t_anchor.numpy(),
                    out_b_anchor.numpy(),
                    atol=1e-5,
                    err_msg=f"Anchor mismatch on device {device}",
                )
                np.testing.assert_array_equal(
                    out_t_stuck.numpy(),
                    out_b_stuck.numpy(),
                    err_msg=f"Stuck flag mismatch on device {device}",
                )
                np.testing.assert_allclose(
                    out_t_dwell.numpy(),
                    out_b_dwell.numpy(),
                    atol=1e-5,
                    err_msg=f"Dwell time mismatch on device {device}",
                )

    def test_bristle_derivative_finite_diff(self):
        """Verify analytical tangent Jacobian matches numerical velocity differences across branches."""
        device = "cpu"
        h = 1e-4

        cases = [
            # 1. Pure elastic stick
            {
                "pos": (0.0, 0.0),
                "vel": (0.05, -0.03),
                "dt": 0.01,
                "normal": 50.0,
                "kt": 100.0,
                "kv": 0.0,
                "mu": 1.0,
                "viscous_ratio": 0.5,
                "release_dwell": 0.1,
                "anchor": (0.0, 0.0),
                "stuck": 1,
                "dwell": 0.0,
            },
            # 2. Pure elastic sliding (radial return)
            {
                "pos": (0.0, 0.0),
                "vel": (2.0, 3.0),
                "dt": 0.01,
                "normal": 10.0,
                "kt": 500.0,
                "kv": 0.0,
                "mu": 0.5,
                "viscous_ratio": 0.5,
                "release_dwell": 0.1,
                "anchor": (0.0, 0.0),
                "stuck": 1,
                "dwell": 0.0,
            },
            # 3. Stick elastic + uncapped viscous (eta == 1)
            {
                "pos": (0.0, 0.0),
                "vel": (0.02, 0.01),
                "dt": 0.01,
                "normal": 50.0,
                "kt": 100.0,
                "kv": 5.0,
                "mu": 1.0,
                "viscous_ratio": 0.5,
                "release_dwell": 0.1,
                "anchor": (0.0, 0.0),
                "stuck": 1,
                "dwell": 0.0,
            },
            # 4. Stick elastic + cone-limited viscous (eta in (0, 1))
            {
                "pos": (0.0, 0.0),
                "vel": (0.1, 0.2),
                "dt": 0.01,
                "normal": 10.0,
                "kt": 100.0,
                "kv": 50.0,
                "mu": 1.0,
                "viscous_ratio": 0.5,
                "release_dwell": 0.1,
                "anchor": (-0.08, 0.0),
                "stuck": 1,
                "dwell": 0.0,
            },
            # 5. Sliding elastic + viscous (eta == 0)
            {
                "pos": (0.0, 0.0),
                "vel": (5.0, 5.0),
                "dt": 0.01,
                "normal": 10.0,
                "kt": 500.0,
                "kv": 20.0,
                "mu": 0.5,
                "viscous_ratio": 0.5,
                "release_dwell": 0.1,
                "anchor": (0.0, 0.0),
                "stuck": 1,
                "dwell": 0.0,
            },
        ]

        out_force = wp.zeros(1, dtype=wp.vec2, device=device)
        out_jacobian = wp.zeros(1, dtype=wp.mat22, device=device)
        out_anchor = wp.zeros(1, dtype=wp.vec2, device=device)
        out_stuck = wp.zeros(1, dtype=int, device=device)
        out_dwell = wp.zeros(1, dtype=float, device=device)

        def eval_force(tc, vel_vec):
            inputs_list = [
                wp.vec2(*tc["pos"]),
                wp.vec2(float(vel_vec[0]), float(vel_vec[1])),
                tc["dt"],
                tc["normal"],
                tc["kt"],
                tc["kv"],
                tc["mu"],
                tc["viscous_ratio"],
                tc["release_dwell"],
                wp.vec2(*tc["anchor"]),
                tc["stuck"],
                tc["dwell"],
            ]
            wp.launch(
                _eval_bristle_force_tangent,
                dim=1,
                inputs=inputs_list,
                outputs=[out_force, out_jacobian, out_anchor, out_stuck, out_dwell],
                device=device,
            )
            return out_force.numpy()[0].copy(), out_jacobian.numpy()[0].copy()

        for tc in cases:
            base_v = np.array(tc["vel"], dtype=float)
            _f0, J_ana = eval_force(tc, base_v)

            J_fd = np.zeros((2, 2))
            for i in range(2):
                dv = np.zeros(2)
                dv[i] = h
                f_plus, _ = eval_force(tc, base_v + dv)
                f_minus, _ = eval_force(tc, base_v - dv)
                J_fd[:, i] = (f_plus - f_minus) / (2.0 * h)

            np.testing.assert_allclose(J_ana, J_fd, rtol=2e-2, atol=1e-2)

    def test_bristle_autodiff_kt_vs_finite_diff(self):
        """Verify automatic differentiation of returned bristle force wrt kt matches finite difference."""
        device = "cpu"
        h = 1e-3

        cases = [
            # 1. Pure elastic stick
            {
                "pos": (0.0, 0.0),
                "vel": (0.05, -0.03),
                "normal": 50.0,
                "kt": 100.0,
                "kv": 0.0,
                "mu": 1.0,
                "visc": 0.5,
                "anchor": (0.0, 0.0),
            },
            # 2. Pure elastic sliding
            {
                "pos": (0.0, 0.0),
                "vel": (2.0, 3.0),
                "normal": 10.0,
                "kt": 500.0,
                "kv": 0.0,
                "mu": 0.5,
                "visc": 0.5,
                "anchor": (0.0, 0.0),
            },
            # 3. Stick elastic + uncapped viscous
            {
                "pos": (0.0, 0.0),
                "vel": (0.02, 0.01),
                "normal": 50.0,
                "kt": 100.0,
                "kv": 5.0,
                "mu": 1.0,
                "visc": 0.5,
                "anchor": (0.0, 0.0),
            },
            # 4. Stick elastic + cone-limited viscous
            {
                "pos": (0.0, 0.0),
                "vel": (0.1, 0.2),
                "normal": 10.0,
                "kt": 100.0,
                "kv": 50.0,
                "mu": 1.0,
                "visc": 0.5,
                "anchor": (-0.08, 0.0),
            },
        ]

        for tc in cases:
            kt_val = tc["kt"]
            kt_arr = wp.array([kt_val], dtype=float, device=device, requires_grad=True)
            out_loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

            inputs_common = [
                wp.vec2(*tc["pos"]),
                wp.vec2(*tc["vel"]),
                0.01,
                tc["normal"],
                kt_arr,
                tc["kv"],
                tc["mu"],
                tc["visc"],
                0.1,
                wp.vec2(*tc["anchor"]),
                1,
                0.0,
            ]

            tape = wp.Tape()
            with tape:
                wp.launch(
                    _eval_bristle_loss_for_ad,
                    dim=1,
                    inputs=inputs_common,
                    outputs=[out_loss],
                    device=device,
                )
            tape.backward(loss=out_loss)
            grad_ad = kt_arr.grad.numpy()[0]
            tape.zero()

            kt_p = wp.array([kt_val + h], dtype=float, device=device)
            kt_m = wp.array([kt_val - h], dtype=float, device=device)
            out_p = wp.zeros(1, dtype=float, device=device)
            out_m = wp.zeros(1, dtype=float, device=device)

            inputs_p = list(inputs_common)
            inputs_p[4] = kt_p
            inputs_m = list(inputs_common)
            inputs_m[4] = kt_m

            wp.launch(_eval_bristle_loss_for_ad, dim=1, inputs=inputs_p, outputs=[out_p], device=device)
            wp.launch(_eval_bristle_loss_for_ad, dim=1, inputs=inputs_m, outputs=[out_m], device=device)

            grad_fd = (out_p.numpy()[0] - out_m.numpy()[0]) / (2.0 * h)
            np.testing.assert_allclose(grad_ad, grad_fd, rtol=2e-2, atol=1e-4)

    def test_cone_constraint_normalcy(self):
        """Verify that when the cone limit is strictly active in interior eta, f dot df = 0."""
        device = "cpu"
        tc = {
            "pos": (0.0, 0.0),
            "vel": (0.1, 0.2),
            "dt": 0.01,
            "normal": 10.0,
            "kt": 100.0,
            "kv": 50.0,
            "mu": 1.0,
            "viscous_ratio": 0.5,
            "release_dwell": 0.1,
            "anchor": (-0.08, 0.0),
            "stuck": 1,
            "dwell": 0.0,
        }
        out_force = wp.zeros(1, dtype=wp.vec2, device=device)
        out_jacobian = wp.zeros(1, dtype=wp.mat22, device=device)
        out_anchor = wp.zeros(1, dtype=wp.vec2, device=device)
        out_stuck = wp.zeros(1, dtype=int, device=device)
        out_dwell = wp.zeros(1, dtype=float, device=device)

        inputs_list = [
            wp.vec2(*tc["pos"]),
            wp.vec2(*tc["vel"]),
            tc["dt"],
            tc["normal"],
            tc["kt"],
            tc["kv"],
            tc["mu"],
            tc["viscous_ratio"],
            tc["release_dwell"],
            wp.vec2(*tc["anchor"]),
            tc["stuck"],
            tc["dwell"],
        ]
        wp.launch(
            _eval_bristle_force_tangent,
            dim=1,
            inputs=inputs_list,
            outputs=[out_force, out_jacobian, out_anchor, out_stuck, out_dwell],
            device=device,
        )
        f = out_force.numpy()[0]
        J = out_jacobian.numpy()[0]
        f_dot_J = f @ J
        np.testing.assert_allclose(f_dot_J, np.zeros(2), atol=1e-5)

    def test_regularized_force_tangent_finite_diff(self):
        """Verify regularized_force_tangent analytical Jacobian matches finite differences."""
        device = "cpu"
        h = 1e-4
        dt = 0.01
        normal = 20.0
        mu = 0.8
        smoothing_speed = 0.1

        out_f = wp.zeros(1, dtype=wp.vec2, device=device)
        out_J = wp.zeros(1, dtype=wp.mat22, device=device)

        def eval_reg(vel_vec):
            wp.launch(
                _eval_regularized_force_tangent,
                dim=1,
                inputs=[wp.vec2(float(vel_vec[0]), float(vel_vec[1])), dt, normal, mu, smoothing_speed],
                outputs=[out_f, out_J],
                device=device,
            )
            return out_f.numpy()[0].copy(), out_J.numpy()[0].copy()

        # Test sliding, regularized ramp, and zero velocity
        velocities = [
            np.array([0.02, 0.03]),  # inside ramp (speed 0.036 < 0.1)
            np.array([0.5, -0.4]),  # sliding (speed 0.64 > 0.1)
            np.array([0.0, 0.0]),  # zero velocity
        ]

        for vel in velocities:
            _f0, J_ana = eval_reg(vel)
            J_fd = np.zeros((2, 2))
            for i in range(2):
                dv = np.zeros(2)
                dv[i] = h
                f_p, _ = eval_reg(vel + dv)
                f_m, _ = eval_reg(vel - dv)
                J_fd[:, i] = (f_p - f_m) / (2.0 * h)

            np.testing.assert_allclose(J_ana, J_fd, rtol=2e-2, atol=1.0)

    def test_edges_zero_load_mu_and_kt(self):
        """Verify edge cases for zero/negative normal force, mu, and stiffness."""
        device = "cpu"
        out_force = wp.zeros(1, dtype=wp.vec2, device=device)
        out_jacobian = wp.zeros(1, dtype=wp.mat22, device=device)
        out_anchor = wp.zeros(1, dtype=wp.vec2, device=device)
        out_stuck = wp.zeros(1, dtype=int, device=device)
        out_dwell = wp.zeros(1, dtype=float, device=device)

        # normal = 0
        wp.launch(
            _eval_bristle_force_tangent,
            dim=1,
            inputs=[
                wp.vec2(0.0, 0.0),
                wp.vec2(1.0, 1.0),
                0.01,
                0.0,
                100.0,
                10.0,
                1.0,
                0.5,
                0.1,
                wp.vec2(0.0, 0.0),
                1,
                0.0,
            ],
            outputs=[out_force, out_jacobian, out_anchor, out_stuck, out_dwell],
            device=device,
        )
        np.testing.assert_array_equal(out_force.numpy()[0], np.zeros(2))
        np.testing.assert_array_equal(out_jacobian.numpy()[0], np.zeros((2, 2)))

        # kt <= 0
        wp.launch(
            _eval_bristle_force_tangent,
            dim=1,
            inputs=[
                wp.vec2(0.0, 0.0),
                wp.vec2(1.0, 1.0),
                0.01,
                10.0,
                0.0,
                10.0,
                1.0,
                0.5,
                0.1,
                wp.vec2(0.0, 0.0),
                1,
                0.0,
            ],
            outputs=[out_force, out_jacobian, out_anchor, out_stuck, out_dwell],
            device=device,
        )
        np.testing.assert_array_equal(out_force.numpy()[0], np.zeros(2))
        np.testing.assert_array_equal(out_jacobian.numpy()[0], np.zeros((2, 2)))

        # regularized friction with normal = 0
        wp.launch(
            _eval_regularized_force_tangent,
            dim=1,
            inputs=[wp.vec2(1.0, 1.0), 0.01, 0.0, 1.0, 0.1],
            outputs=[out_force, out_jacobian],
            device=device,
        )
        np.testing.assert_array_equal(out_force.numpy()[0], np.zeros(2))
        np.testing.assert_array_equal(out_jacobian.numpy()[0], np.zeros((2, 2)))

        # regularized friction with mu = 0
        wp.launch(
            _eval_regularized_force_tangent,
            dim=1,
            inputs=[wp.vec2(1.0, 1.0), 0.01, 10.0, 0.0, 0.1],
            outputs=[out_force, out_jacobian],
            device=device,
        )
        np.testing.assert_array_equal(out_force.numpy()[0], np.zeros(2))
        np.testing.assert_array_equal(out_jacobian.numpy()[0], np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
