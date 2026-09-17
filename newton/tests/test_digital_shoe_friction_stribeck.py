# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for friction-only consistent deflection law with Stribeck cap."""

import tempfile
import unittest

import numpy as np
import warp as wp

# Ensure an isolated kernel cache directory before Warp initialization/compilation
if wp.config.kernel_cache_dir is None:
    _isolated_cache_dir = tempfile.mkdtemp(prefix="warp_stribeck_cache_")
    wp.config.kernel_cache_dir = _isolated_cache_dir

from projects.digital_shoe.friction_deflection import bristle_deflection_step
from projects.digital_shoe.friction_stribeck import (
    bristle_stribeck_step,
    stribeck_coefficient,
)


@wp.kernel
def _eval_stribeck_step(
    vel: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu_s: float,
    mu_d: float,
    vs: float,
    viscous_ratio: float,
    release_dwell: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
    out_force: wp.array[wp.vec2],
    out_jacobian: wp.array[wp.mat22],
    out_deflection: wp.array[wp.vec2],
    out_stuck: wp.array[int],
    out_dwell: wp.array[float],
):
    f, J, z, s, d = bristle_stribeck_step(
        vel,
        dt,
        normal,
        kt,
        kv,
        mu_s,
        mu_d,
        vs,
        viscous_ratio,
        release_dwell,
        deflection,
        stuck,
        dwell,
    )
    out_force[0] = f
    out_jacobian[0] = J
    out_deflection[0] = z
    out_stuck[0] = s
    out_dwell[0] = d


@wp.kernel
def _eval_deflection_step(
    vel: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    yield_width: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
    out_force: wp.array[wp.vec2],
    out_jacobian: wp.array[wp.mat22],
    out_deflection: wp.array[wp.vec2],
    out_stuck: wp.array[int],
    out_dwell: wp.array[float],
):
    f, J, z, s, d = bristle_deflection_step(
        vel,
        dt,
        normal,
        kt,
        kv,
        mu,
        viscous_ratio,
        release_dwell,
        yield_width,
        deflection,
        stuck,
        dwell,
    )
    out_force[0] = f
    out_jacobian[0] = J
    out_deflection[0] = z
    out_stuck[0] = s
    out_dwell[0] = d


@wp.kernel
def _eval_stribeck_coefficient(
    vel: wp.vec2,
    mu_s: float,
    mu_d: float,
    vs: float,
    out_mu: wp.array[float],
):
    out_mu[0] = stribeck_coefficient(vel, mu_s, mu_d, vs)


@wp.kernel
def _eval_stribeck_tape(
    vel: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu_s_arr: wp.array[float],
    mu_d_arr: wp.array[float],
    vs_arr: wp.array[float],
    viscous_ratio: float,
    release_dwell: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
    loss: wp.array[float],
):
    f, _J, _z, _s, _d = bristle_stribeck_step(
        vel,
        dt,
        normal,
        kt,
        kv,
        mu_s_arr[0],
        mu_d_arr[0],
        vs_arr[0],
        viscous_ratio,
        release_dwell,
        deflection,
        stuck,
        dwell,
    )
    loss[0] = f[0] * 1.5 - f[1] * 0.7


class TestDigitalShoeFrictionStribeck(unittest.TestCase):
    """Test Stribeck bristle friction law, Jacobians, conservation, and Tape gradients."""

    def setUp(self):
        self.devices = ["cpu"]
        if wp.is_cuda_available():
            self.devices.append("cuda:0")

    def test_constant_mu_parity_with_deflection_step(self):
        """Verify exact parity between Stribeck law with mu_static == mu_dynamic and bristle_deflection_step."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f_stribeck = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J_stribeck = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z_stribeck = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s_stribeck = wp.zeros(1, dtype=int, device=device)
                out_d_stribeck = wp.zeros(1, dtype=float, device=device)

                out_f_deflection = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J_deflection = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z_deflection = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s_deflection = wp.zeros(1, dtype=int, device=device)
                out_d_deflection = wp.zeros(1, dtype=float, device=device)

                dt = 0.01
                normal = 80.0
                kt = 40000.0
                mu_val = 0.75
                vr = 0.25
                rd = 0.005

                test_cases = [
                    (wp.vec2(0.02, 0.01), 0.0, "sticking kv=0"),
                    (wp.vec2(0.02, 0.01), 40.0, "sticking kv=40"),
                    (wp.vec2(0.5, -0.3), 0.0, "sliding kv=0"),
                    (wp.vec2(0.5, -0.3), 50.0, "sliding kv=50"),
                    (wp.vec2(1.5, -1.0), 100.0, "fast sliding kv=100"),
                ]

                z_init = wp.vec2(0.0004, -0.0002)
                for vel, kv, desc in test_cases:
                    wp.launch(
                        _eval_stribeck_step,
                        dim=1,
                        inputs=[
                            vel,
                            dt,
                            normal,
                            kt,
                            kv,
                            mu_val,
                            mu_val,
                            0.2,
                            vr,
                            rd,
                            z_init,
                            1,
                            0.0,
                            out_f_stribeck,
                            out_J_stribeck,
                            out_z_stribeck,
                            out_s_stribeck,
                            out_d_stribeck,
                        ],
                        device=device,
                    )
                    wp.launch(
                        _eval_deflection_step,
                        dim=1,
                        inputs=[
                            vel,
                            dt,
                            normal,
                            kt,
                            kv,
                            mu_val,
                            vr,
                            rd,
                            0.0,
                            z_init,
                            1,
                            0.0,
                            out_f_deflection,
                            out_J_deflection,
                            out_z_deflection,
                            out_s_deflection,
                            out_d_deflection,
                        ],
                        device=device,
                    )

                    np.testing.assert_allclose(
                        out_f_stribeck.numpy()[0],
                        out_f_deflection.numpy()[0],
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Force mismatch for {desc}",
                    )
                    np.testing.assert_allclose(
                        out_J_stribeck.numpy()[0],
                        out_J_deflection.numpy()[0],
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Jacobian mismatch for {desc}",
                    )
                    np.testing.assert_allclose(
                        out_z_stribeck.numpy()[0],
                        out_z_deflection.numpy()[0],
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Deflection mismatch for {desc}",
                    )
                    self.assertEqual(int(out_s_stribeck.numpy()[0]), int(out_s_deflection.numpy()[0]))
                    self.assertAlmostEqual(float(out_d_stribeck.numpy()[0]), float(out_d_deflection.numpy()[0]))

    def test_dynamic_limit_and_static_limit_bounds(self):
        """Verify Stribeck coefficient and Coulomb cap at zero velocity and asymptotic high velocity."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_mu = wp.zeros(1, dtype=float, device=device)

                mu_s = 0.85
                mu_d = 0.45
                vs = 0.25

                # 1. Zero velocity limit: mu(0) == mu_static
                wp.launch(
                    _eval_stribeck_coefficient, dim=1, inputs=[wp.vec2(0.0, 0.0), mu_s, mu_d, vs, out_mu], device=device
                )
                self.assertAlmostEqual(float(out_mu.numpy()[0]), mu_s, places=6)

                # 2. Near-zero velocity: mu(v) ~ mu_static
                wp.launch(
                    _eval_stribeck_coefficient,
                    dim=1,
                    inputs=[wp.vec2(1e-5, 0.0), mu_s, mu_d, vs, out_mu],
                    device=device,
                )
                self.assertAlmostEqual(float(out_mu.numpy()[0]), mu_s, places=5)

                # 3. Transition speed: mu(vs) == mu_dynamic + (mu_static - mu_dynamic) * exp(-1)
                expected_vs = mu_d + (mu_s - mu_d) * np.exp(-1.0)
                wp.launch(
                    _eval_stribeck_coefficient, dim=1, inputs=[wp.vec2(vs, 0.0), mu_s, mu_d, vs, out_mu], device=device
                )
                self.assertAlmostEqual(float(out_mu.numpy()[0]), expected_vs, places=5)

                # 4. Asymptotic high speed: mu(v) -> mu_dynamic
                wp.launch(
                    _eval_stribeck_coefficient,
                    dim=1,
                    inputs=[wp.vec2(10.0 * vs, 0.0), mu_s, mu_d, vs, out_mu],
                    device=device,
                )
                self.assertAlmostEqual(float(out_mu.numpy()[0]), mu_d, places=6)

    def test_slip_direction_and_coulomb_cap_bound(self):
        """Verify force opposes relative motion and strictly respects the speed-dependent Coulomb cap."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.01
                normal = 120.0
                kt = 50000.0
                mu_s = 0.8
                mu_d = 0.4
                vs = 0.3
                vr = 0.2
                rd = 0.005

                slip_velocities = [
                    [0.1, 0.0],
                    [-0.5, 0.2],
                    [0.8, -0.6],
                    [2.5, 1.8],
                ]

                for vel_val in slip_velocities:
                    wp.launch(
                        _eval_stribeck_step,
                        dim=1,
                        inputs=[
                            wp.vec2(*vel_val),
                            dt,
                            normal,
                            kt,
                            40.0,
                            mu_s,
                            mu_d,
                            vs,
                            vr,
                            rd,
                            wp.vec2(0.001, -0.001),
                            1,
                            0.0,
                            out_f,
                            out_J,
                            out_z,
                            out_s,
                            out_d,
                        ],
                        device=device,
                    )
                    f = out_f.numpy()[0]
                    f_mag = np.linalg.norm(f)

                    # Compute expected Coulomb cap C(v)
                    v_sq = np.dot(vel_val, vel_val)
                    mu_v = mu_d + (mu_s - mu_d) * np.exp(-v_sq / (vs * vs))
                    c_v = normal * mu_v

                    # Total friction force magnitude must never exceed the cone cap
                    self.assertLessEqual(f_mag, c_v + 1e-5)

                    # For sliding contacts, force opposes velocity
                    f_dot_v = np.dot(f, vel_val)
                    self.assertLess(f_dot_v, 0.0)

    def test_zero_slip_force_invariance_and_idempotence(self):
        """Verify exact zero-slip force invariance and idempotence under fixed normal and params."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                normal = 20.0
                kt = 2000.0
                mu_s = 0.9
                mu_d = 0.5
                vs = 0.2
                z_initial = 0.008  # Initial force 16 N (cap is 18 N at v=0)

                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                # Test across different timesteps and step counts
                for dt, n_steps in [(0.02, 5), (0.005, 20), (0.001, 100)]:
                    cur_z = wp.vec2(z_initial, 0.0)
                    cur_stuck = 1
                    cur_dwell = 0.0
                    for _ in range(n_steps):
                        wp.launch(
                            _eval_stribeck_step,
                            dim=1,
                            inputs=[
                                wp.vec2(0.0, 0.0),
                                dt,
                                normal,
                                kt,
                                0.0,
                                mu_s,
                                mu_d,
                                vs,
                                0.0,
                                0.01,
                                cur_z,
                                cur_stuck,
                                cur_dwell,
                                out_f,
                                out_J,
                                out_z,
                                out_s,
                                out_d,
                            ],
                            device=device,
                        )
                        cur_z = out_z.numpy()[0]
                        cur_stuck = int(out_s.numpy()[0])
                        cur_dwell = float(out_d.numpy()[0])

                    # Strictly invariant at -16 N with no per-step creep
                    np.testing.assert_allclose(out_f.numpy()[0], [-16.0, 0.0], atol=1e-5)
                    np.testing.assert_allclose(out_z.numpy()[0], [z_initial, 0.0], atol=1e-7)

    def test_unforced_tangential_energy_dissipation_with_variable_normal(self):
        """Verify energy dissipation inequality E_new - E_old + F . v * dt <= tol across variable normal."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.001
                kt = 40000.0
                mu_s = 0.85
                mu_d = 0.55
                vs = 0.25
                z_cur_np = np.array([0.0004, -0.0002], dtype=np.float32)

                normals = [30.0, 80.0, 150.0]
                vels = [
                    [0.02, 0.01],  # sticking
                    [0.2, 0.15],  # transition
                    [0.8, -0.5],  # sliding
                ]

                for normal in normals:
                    for vel_val in vels:
                        wp.launch(
                            _eval_stribeck_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*vel_val),
                                dt,
                                normal,
                                kt,
                                25.0,
                                mu_s,
                                mu_d,
                                vs,
                                0.2,
                                0.01,
                                wp.vec2(*z_cur_np),
                                1,
                                0.0,
                                out_f,
                                out_J,
                                out_z,
                                out_s,
                                out_d,
                            ],
                            device=device,
                        )
                        f = out_f.numpy()[0]
                        z_new = out_z.numpy()[0]
                        z_old = z_cur_np

                        e_old = 0.5 * kt * np.dot(z_old, z_old)
                        e_new = 0.5 * kt * np.dot(z_new, z_new)
                        friction_work = np.dot(f, vel_val) * dt

                        excess = (e_new - e_old) + friction_work
                        midpoint_bound = 0.5 * kt * np.dot(np.array(vel_val) * dt, np.array(vel_val) * dt) + 1e-6
                        self.assertLessEqual(excess, midpoint_bound)

    def test_velocity_finite_difference_derivatives(self):
        """Match analytical tangent Jacobian to central finite differences across Stribeck velocity regimes."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.01
                normal = 80.0
                kt = 40000.0
                mu_s = 0.9
                mu_d = 0.6
                vs = 0.3
                vr = 0.25
                rd = 0.005

                test_configs = [
                    ([0.02, 0.01], 0.0, "sticking kv=0"),
                    ([0.02, 0.01], 50.0, "sticking kv=50"),
                    ([0.5, -0.3], 0.0, "sliding kv=0"),
                    ([0.5, -0.3], 50.0, "sliding kv=50 interior"),
                    ([0.2, 0.15], 0.0, "near transition kv=0"),
                    ([0.2, 0.15], 30.0, "near transition kv=30"),
                    ([1.5, -1.0], 100.0, "fast sliding kv=100"),
                ]

                h = 1e-4
                for vel_raw, kv, desc in test_configs:
                    z_init = wp.vec2(0.0004, -0.0002)
                    wp.launch(
                        _eval_stribeck_step,
                        dim=1,
                        inputs=[
                            wp.vec2(*vel_raw),
                            dt,
                            normal,
                            kt,
                            kv,
                            mu_s,
                            mu_d,
                            vs,
                            vr,
                            rd,
                            z_init,
                            1,
                            0.0,
                            out_f,
                            out_J,
                            out_z,
                            out_s,
                            out_d,
                        ],
                        device=device,
                    )
                    exact_J = out_J.numpy()[0]

                    fd_J = np.zeros((2, 2))
                    for comp in range(2):
                        v_p = list(vel_raw)
                        v_p[comp] += h
                        wp.launch(
                            _eval_stribeck_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*v_p),
                                dt,
                                normal,
                                kt,
                                kv,
                                mu_s,
                                mu_d,
                                vs,
                                vr,
                                rd,
                                z_init,
                                1,
                                0.0,
                                out_f,
                                out_J,
                                out_z,
                                out_s,
                                out_d,
                            ],
                            device=device,
                        )
                        f_p = out_f.numpy()[0].copy()

                        v_m = list(vel_raw)
                        v_m[comp] -= h
                        wp.launch(
                            _eval_stribeck_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*v_m),
                                dt,
                                normal,
                                kt,
                                kv,
                                mu_s,
                                mu_d,
                                vs,
                                vr,
                                rd,
                                z_init,
                                1,
                                0.0,
                                out_f,
                                out_J,
                                out_z,
                                out_s,
                                out_d,
                            ],
                            device=device,
                        )
                        f_m = out_f.numpy()[0].copy()
                        fd_J[:, comp] = (f_p - f_m) / (2.0 * h)

                    np.testing.assert_allclose(
                        exact_J,
                        fd_J,
                        rtol=0.01,
                        atol=0.1,
                        err_msg=f"Jacobian mismatch for {desc} on {device_name}",
                    )

    def test_parameter_differentiation_tape_and_fd(self):
        """Verify Warp Tape autodiff gradients wrt mu_static, mu_dynamic, and transition_speed against FD."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                dt = 0.01
                normal = 100.0
                kt = 40000.0
                kv = 0.0
                vr = 0.2
                rd = 0.005

                vel = wp.vec2(0.3, 0.2)
                z_init = wp.vec2(0.0003, -0.0001)

                mu_s_val = 0.8
                mu_d_val = 0.5
                vs_val = 0.4

                mu_s_arr = wp.array([mu_s_val], dtype=float, device=device, requires_grad=True)
                mu_d_arr = wp.array([mu_d_val], dtype=float, device=device, requires_grad=True)
                vs_arr = wp.array([vs_val], dtype=float, device=device, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

                with wp.Tape() as tape:
                    wp.launch(
                        _eval_stribeck_tape,
                        dim=1,
                        inputs=[
                            vel,
                            dt,
                            normal,
                            kt,
                            kv,
                            mu_s_arr,
                            mu_d_arr,
                            vs_arr,
                            vr,
                            rd,
                            z_init,
                            1,
                            0.0,
                            loss,
                        ],
                        device=device,
                    )
                tape.backward(loss)

                grad_s = float(mu_s_arr.grad.numpy()[0])
                grad_d = float(mu_d_arr.grad.numpy()[0])
                grad_vs = float(vs_arr.grad.numpy()[0])

                # Finite differences
                h = 1e-4
                loss_eval = wp.zeros(1, dtype=float, device=device)

                # mu_s FD
                mu_s_p = wp.array([mu_s_val + h], dtype=float, device=device)
                wp.launch(
                    _eval_stribeck_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt, kv, mu_s_p, mu_d_arr, vs_arr, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lp = float(loss_eval.numpy()[0])
                mu_s_m = wp.array([mu_s_val - h], dtype=float, device=device)
                wp.launch(
                    _eval_stribeck_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt, kv, mu_s_m, mu_d_arr, vs_arr, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lm = float(loss_eval.numpy()[0])
                fd_s = (lp - lm) / (2.0 * h)

                # mu_d FD
                mu_d_p = wp.array([mu_d_val + h], dtype=float, device=device)
                wp.launch(
                    _eval_stribeck_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt, kv, mu_s_arr, mu_d_p, vs_arr, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lp = float(loss_eval.numpy()[0])
                mu_d_m = wp.array([mu_d_val - h], dtype=float, device=device)
                wp.launch(
                    _eval_stribeck_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt, kv, mu_s_arr, mu_d_m, vs_arr, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lm = float(loss_eval.numpy()[0])
                fd_d = (lp - lm) / (2.0 * h)

                # vs FD
                vs_p = wp.array([vs_val + h], dtype=float, device=device)
                wp.launch(
                    _eval_stribeck_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt, kv, mu_s_arr, mu_d_arr, vs_p, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lp = float(loss_eval.numpy()[0])
                vs_m = wp.array([vs_val - h], dtype=float, device=device)
                wp.launch(
                    _eval_stribeck_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt, kv, mu_s_arr, mu_d_arr, vs_m, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lm = float(loss_eval.numpy()[0])
                fd_vs = (lp - lm) / (2.0 * h)

                np.testing.assert_allclose(grad_s, fd_s, rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(grad_d, fd_d, rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(grad_vs, fd_vs, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
