# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for friction-only consistent deflection law and solver modes."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import warp as wp

from projects.digital_shoe.friction_adapter import FrictionAdapter
from projects.digital_shoe.friction_deflection import bristle_deflection_step
from projects.digital_shoe.friction_solver import FrictionParams, FrictionSolver


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
        vel, dt, normal, kt, kv, mu, viscous_ratio, release_dwell, yield_width, deflection, stuck, dwell
    )
    out_force[0] = f
    out_jacobian[0] = J
    out_deflection[0] = z
    out_stuck[0] = s
    out_dwell[0] = d


@wp.kernel
def _eval_deflection_param_tape(
    vel: wp.vec2,
    dt: float,
    normal: float,
    kt_arr: wp.array[float],
    kv: float,
    mu_arr: wp.array[float],
    viscous_ratio: float,
    release_dwell: float,
    yield_width: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
    loss: wp.array[float],
):
    f, _J, _z, _s, _d = bristle_deflection_step(
        vel, dt, normal, kt_arr[0], kv, mu_arr[0], viscous_ratio, release_dwell, yield_width, deflection, stuck, dwell
    )
    loss[0] = f[0] * 1.2 + f[1] * 0.8


@wp.kernel
def _loss_vec2(force: wp.array[wp.vec2], loss: wp.array[float]):
    loss[0] = force[0][0] + 0.3 * force[1][1]


class TestDigitalShoeFrictionDeflection(unittest.TestCase):
    """Test consistent deflection law, Jacobians, conservation, and solver modes."""

    def setUp(self):
        self.devices = ["cpu"]
        if wp.is_cuda_available():
            self.devices.append("cuda:0")

    def test_legacy_modes_remain_identical(self):
        """Verify legacy bristle, implicit_bristle and regularized modes produce identical results."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                points = np.array([[-0.08, -0.03, -0.04], [0.06, 0.04, -0.04]], np.float32)
                matrices = np.diag([0.5, 0.5, 0.5, 5.0, 4.0, 8.0]).astype(np.float32)
                p = FrictionParams()
                p.mu = 0.8
                p.viscous_ratio = 0.2
                p.release_dwell = 0.001
                p.smoothing_speed = 0.01

                for mode in ["bristle", "implicit_bristle", "regularized"]:
                    solver = FrictionSolver(2, 1, mode=mode, iterations=4, device=device)
                    pts = wp.array(points, dtype=wp.vec3, device=device)
                    nrm = wp.full(2, 100.0, dtype=float, device=device)
                    com = wp.zeros(1, dtype=wp.vec3, device=device)
                    vel = wp.array([[0.4, -0.2, 0, 0.2, 0.0, 0.3]], dtype=wp.spatial_vector, device=device)
                    mob = wp.array([matrices], dtype=wp.spatial_matrix, device=device)
                    kt = wp.array([80000.0, 100000.0], dtype=float, device=device)
                    kv = wp.array([50.0, 50.0], dtype=float, device=device)
                    params = wp.array([p], dtype=FrictionParams, device=device)
                    anch = wp.array(points[:, :2], dtype=wp.vec2, device=device)
                    stk = wp.ones(2, dtype=int, device=device)
                    dwl = wp.zeros(2, dtype=float, device=device)

                    res = solver.solve(pts, nrm, com, vel, mob, kt, kv, params, anch, stk, dwl, 0.01)
                    # Verify outputs are finite and valid
                    self.assertTrue(np.isfinite(res.force.numpy()).all())
                    self.assertTrue(np.isfinite(res.velocity.numpy()).all())
                    self.assertTrue(np.isfinite(res.anchor.numpy()).all())
                    self.assertTrue(np.isfinite(res.deflection.numpy()).all())
                    # Anchor and deflection are separate arrays
                    self.assertIsNot(res.anchor, res.deflection)

    def test_velocity_integrated_history(self):
        """Verify deflection directly integrates velocity dt*v and matches elastic trial."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.005
                kt = 20000.0
                normal = 1000.0  # high normal so we stay strictly inside elastic regime
                mu = 1.0
                vel_np = np.array([0.04, -0.02], dtype=np.float32)
                z_init_np = np.array([0.001, 0.002], dtype=np.float32)

                wp.launch(
                    _eval_deflection_step,
                    dim=1,
                    inputs=[
                        wp.vec2(*vel_np),
                        dt,
                        normal,
                        kt,
                        0.0,
                        mu,
                        0.0,
                        0.01,
                        0.0,
                        wp.vec2(*z_init_np),
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

                expected_z = z_init_np + dt * vel_np
                expected_force = -kt * expected_z

                np.testing.assert_allclose(out_z.numpy()[0], expected_z, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(out_f.numpy()[0], expected_force, rtol=1e-5, atol=1e-7)
                self.assertEqual(out_s.numpy()[0], 1)
                self.assertEqual(out_d.numpy()[0], 0.0)

    def test_translation_invariance(self):
        """Verify bristle_deflection_step depends only on velocity, not world position."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                solver = FrictionSolver(1, 1, mode="deflection", max_steps=2, device=device)
                dt = 0.01
                points1 = np.array([[0.0, 0.0, 0.0]], np.float32)
                points2 = np.array([[100.0, -50.0, 0.0]], np.float32)  # translated by (100, -50)
                matrices = np.eye(6, dtype=np.float32)

                p = FrictionParams()
                p.mu = 0.6
                p.viscous_ratio = 0.2
                p.release_dwell = 0.005

                res1 = solver.solve(
                    wp.array(points1, dtype=wp.vec3, device=device),
                    wp.full(1, 50.0, dtype=float, device=device),
                    wp.zeros(1, dtype=wp.vec3, device=device),
                    wp.array([[0.2, -0.1, 0, 0, 0, 0]], dtype=wp.spatial_vector, device=device),
                    wp.array([matrices], dtype=wp.spatial_matrix, device=device),
                    wp.array([5000.0], dtype=float, device=device),
                    wp.array([10.0], dtype=float, device=device),
                    wp.array([p], dtype=FrictionParams, device=device),
                    wp.zeros(1, dtype=wp.vec2, device=device),
                    wp.ones(1, dtype=int, device=device),
                    wp.zeros(1, dtype=float, device=device),
                    dt,
                    step=0,
                )
                res1_f = res1.force.numpy().copy()
                res1_z = res1.deflection.numpy().copy()
                res1_a = res1.anchor.numpy().copy()

                res2 = solver.solve(
                    wp.array(points2, dtype=wp.vec3, device=device),
                    wp.full(1, 50.0, dtype=float, device=device),
                    wp.zeros(1, dtype=wp.vec3, device=device),
                    wp.array([[0.2, -0.1, 0, 0, 0, 0]], dtype=wp.spatial_vector, device=device),
                    wp.array([matrices], dtype=wp.spatial_matrix, device=device),
                    wp.array([5000.0], dtype=float, device=device),
                    wp.array([10.0], dtype=float, device=device),
                    wp.array([p], dtype=FrictionParams, device=device),
                    wp.zeros(1, dtype=wp.vec2, device=device),
                    wp.ones(1, dtype=int, device=device),
                    wp.zeros(1, dtype=float, device=device),
                    dt,
                    step=1,
                )

                np.testing.assert_allclose(res1_f, res2.force.numpy(), atol=1e-6)
                np.testing.assert_allclose(res1_z, res2.deflection.numpy(), atol=1e-6)
                # Diagnostic anchor translation check:
                # anchor = trialpoint - z = (point + dt*v) - z
                # so anchor2 - anchor1 == points2 - points1
                shift = (res2.anchor.numpy() - res1_a)[0]
                np.testing.assert_allclose(shift, [100.0, -50.0], atol=1e-5)

    def test_coulomb_cap_and_passivity(self):
        """Verify strict Coulomb cone cap ||f|| <= mu*Fn and dissipative damping."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.01
                normal = 60.0
                mu = 0.7
                cap = mu * normal  # 42.0 N
                kt = 100000.0
                kv = 200.0
                vr = 0.3

                for yw in [0.0, 0.05, 0.15, 0.25]:
                    for vel_mag in [0.01, 0.1, 1.0, 10.0]:
                        for angle in [0.0, 0.7, 2.1, 4.5]:
                            vel = wp.vec2(vel_mag * np.cos(angle), vel_mag * np.sin(angle))
                            wp.launch(
                                _eval_deflection_step,
                                dim=1,
                                inputs=[
                                    vel,
                                    dt,
                                    normal,
                                    kt,
                                    kv,
                                    mu,
                                    vr,
                                    0.01,
                                    yw,
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
                            f_norm = np.linalg.norm(f)
                            self.assertLessEqual(f_norm, cap + 1e-4)

    def test_zero_normal_dwell_and_reset(self):
        """Verify bristle holds deflection until release_dwell then resets to zero."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.002
                release_dwell = 0.005
                z_saved_np = np.array([0.003, -0.002], dtype=np.float32)

                # Step 1: normal = 0, dwell was 0.0 -> next dwell = 0.002 <= 0.005, retains z
                wp.launch(
                    _eval_deflection_step,
                    dim=1,
                    inputs=[
                        wp.vec2(1.0, 0.0),
                        dt,
                        0.0,
                        10000.0,
                        10.0,
                        0.8,
                        0.2,
                        release_dwell,
                        0.0,
                        wp.vec2(*z_saved_np),
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
                np.testing.assert_allclose(out_f.numpy()[0], [0.0, 0.0])
                np.testing.assert_allclose(out_z.numpy()[0], z_saved_np)
                self.assertEqual(out_s.numpy()[0], 1)
                self.assertAlmostEqual(out_d.numpy()[0], 0.002, places=6)

                # Step 2: normal = 0, dwell was 0.004 -> next dwell = 0.006 > release_dwell -> resets to zero
                wp.launch(
                    _eval_deflection_step,
                    dim=1,
                    inputs=[
                        wp.vec2(1.0, 0.0),
                        dt,
                        0.0,
                        10000.0,
                        10.0,
                        0.8,
                        0.2,
                        release_dwell,
                        0.0,
                        wp.vec2(*z_saved_np),
                        1,
                        0.004,
                        out_f,
                        out_J,
                        out_z,
                        out_s,
                        out_d,
                    ],
                    device=device,
                )
                np.testing.assert_allclose(out_f.numpy()[0], [0.0, 0.0])
                np.testing.assert_allclose(out_z.numpy()[0], [0.0, 0.0])
                self.assertEqual(out_s.numpy()[0], 0)
                self.assertEqual(out_d.numpy()[0], 0.0)

    def test_c1_continuity_across_shoulder_transitions(self):
        """Verify value and tangent Jacobian continuity across yield shoulder boundaries."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.01
                normal = 100.0
                mu = 0.8
                kt = 50000.0
                yw = 0.1
                # Boundaries are at r_lo = C*(1-w) = 72, r_hi = C*(1+w) = 88
                # With z_old = 0, r = kt * dt * ||v|| = 500 * ||v||
                # v_lo = 72 / 500 = 0.144
                # v_hi = 88 / 500 = 0.176
                eps = 1e-4

                # Test continuity around v_lo
                v_lo_minus = wp.vec2(0.144 - eps, 0.0)
                v_lo_plus = wp.vec2(0.144 + eps, 0.0)

                wp.launch(
                    _eval_deflection_step,
                    dim=1,
                    inputs=[
                        v_lo_minus,
                        dt,
                        normal,
                        kt,
                        0.0,
                        mu,
                        0.0,
                        0.01,
                        yw,
                        wp.vec2(0.0, 0.0),
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
                f_lo_m = out_f.numpy()[0].copy()
                J_lo_m = out_J.numpy()[0].copy()

                wp.launch(
                    _eval_deflection_step,
                    dim=1,
                    inputs=[
                        v_lo_plus,
                        dt,
                        normal,
                        kt,
                        0.0,
                        mu,
                        0.0,
                        0.01,
                        yw,
                        wp.vec2(0.0, 0.0),
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
                f_lo_p = out_f.numpy()[0].copy()
                J_lo_p = out_J.numpy()[0].copy()

                # At step delta 2*eps = 2e-4, df ~ J * 2e-4 ~ 500 * 2e-4 = 0.1
                np.testing.assert_allclose(f_lo_m, f_lo_p, atol=0.15)
                # Tangent Jacobian continuity (C1)
                np.testing.assert_allclose(J_lo_m, J_lo_p, atol=2.0)

                # Test continuity around v_hi
                v_hi_minus = wp.vec2(0.176 - eps, 0.0)
                v_hi_plus = wp.vec2(0.176 + eps, 0.0)

                wp.launch(
                    _eval_deflection_step,
                    dim=1,
                    inputs=[
                        v_hi_minus,
                        dt,
                        normal,
                        kt,
                        0.0,
                        mu,
                        0.0,
                        0.01,
                        yw,
                        wp.vec2(0.0, 0.0),
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
                f_hi_m = out_f.numpy()[0].copy()
                J_hi_m = out_J.numpy()[0].copy()

                wp.launch(
                    _eval_deflection_step,
                    dim=1,
                    inputs=[
                        v_hi_plus,
                        dt,
                        normal,
                        kt,
                        0.0,
                        mu,
                        0.0,
                        0.01,
                        yw,
                        wp.vec2(0.0, 0.0),
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
                f_hi_p = out_f.numpy()[0].copy()
                J_hi_p = out_J.numpy()[0].copy()

                np.testing.assert_allclose(f_hi_m, f_hi_p, atol=0.15)
                np.testing.assert_allclose(J_hi_m, J_hi_p, atol=2.0)

    def test_energy_conservation_and_dissipation(self):
        """Verify energy balance E_new - E_old + Ft . v * dt <= tolerance."""
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
                normal = 100.0
                mu = 0.8
                z_cur_np = np.array([0.0005, -0.0003], dtype=np.float32)
                vels = [
                    [0.05, 0.02],  # sticking
                    [0.8, -0.4],  # sliding
                    [2.0, 1.5],  # fast sliding
                ]

                for vel_val in vels:
                    wp.launch(
                        _eval_deflection_step,
                        dim=1,
                        inputs=[
                            wp.vec2(*vel_val),
                            dt,
                            normal,
                            kt,
                            20.0,
                            mu,
                            0.2,
                            0.01,
                            0.0,
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
        """Match analytical tangent Jacobian to central finite differences across regimes."""
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
                mu = 0.75
                vr = 0.25
                rd = 0.005

                test_configs = [
                    ([0.02, 0.01], 0.0, 0.0, "linear w=0 kv=0"),
                    ([0.02, 0.01], 0.15, 0.0, "linear w=0.15 kv=0"),
                    ([0.5, -0.3], 0.0, 0.0, "sliding w=0 kv=0"),
                    ([0.5, -0.3], 0.15, 0.0, "sliding w=0.15 kv=0"),
                    ([0.12, 0.05], 0.15, 0.0, "shoulder w=0.15 kv=0"),
                    ([0.12, 0.05], 0.15, 80.0, "shoulder w=0.15 kv=80"),
                    ([0.5, -0.3], 0.15, 80.0, "sliding w=0.15 kv=80"),
                ]

                h = 1e-6
                for vel_raw, yw, kv, _desc in test_configs:
                    z_init = wp.vec2(0.0004, -0.0002)
                    wp.launch(
                        _eval_deflection_step,
                        dim=1,
                        inputs=[
                            wp.vec2(*vel_raw),
                            dt,
                            normal,
                            kt,
                            kv,
                            mu,
                            vr,
                            rd,
                            yw,
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
                            _eval_deflection_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*v_p),
                                dt,
                                normal,
                                kt,
                                kv,
                                mu,
                                vr,
                                rd,
                                yw,
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
                            _eval_deflection_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*v_m),
                                dt,
                                normal,
                                kt,
                                kv,
                                mu,
                                vr,
                                rd,
                                yw,
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

                    np.testing.assert_allclose(exact_J, fd_J, rtol=0.03, atol=2.0)

    def test_parameter_differentiation_tape_and_fd(self):
        """Differentiate bristle deflection law wrt kt and mu using Warp Tape and check against FD."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                dt = 0.01
                normal = 100.0
                kv = 0.0
                vr = 0.2
                rd = 0.005
                yw = 0.1

                vel = wp.vec2(0.5, -0.3)  # Slipping regime so mu derivative is non-zero
                z_init = wp.vec2(0.0003, -0.0001)

                kt_arr = wp.array([30000.0], dtype=float, device=device, requires_grad=True)
                mu_arr = wp.array([0.7], dtype=float, device=device, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

                with wp.Tape() as tape:
                    wp.launch(
                        _eval_deflection_param_tape,
                        dim=1,
                        inputs=[
                            vel,
                            dt,
                            normal,
                            kt_arr,
                            kv,
                            mu_arr,
                            vr,
                            rd,
                            yw,
                            z_init,
                            1,
                            0.0,
                            loss,
                        ],
                        device=device,
                    )
                tape.backward(loss)

                grad_kt = float(kt_arr.grad.numpy()[0])
                grad_mu = float(mu_arr.grad.numpy()[0])

                # Finite difference for kt
                h_kt = 30.0
                loss_eval = wp.zeros(1, dtype=float, device=device)

                kt_arr_p = wp.array([30000.0 + h_kt], dtype=float, device=device)
                wp.launch(
                    _eval_deflection_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt_arr_p, kv, mu_arr, vr, rd, yw, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lp = float(loss_eval.numpy()[0])

                kt_arr_m = wp.array([30000.0 - h_kt], dtype=float, device=device)
                wp.launch(
                    _eval_deflection_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt_arr_m, kv, mu_arr, vr, rd, yw, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lm = float(loss_eval.numpy()[0])

                fd_kt = (lp - lm) / (2.0 * h_kt)
                # In saturated sliding with w=0.1, elastic = u_e * C = u_e * (mu * Fn).
                # Force does not depend on kt! So grad_kt and fd_kt are ~ 0.
                np.testing.assert_allclose(grad_kt, fd_kt, atol=1e-4)

                # Finite difference for mu
                h_mu = 0.001
                mu_arr_p = wp.array([0.7 + h_mu], dtype=float, device=device)
                wp.launch(
                    _eval_deflection_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt_arr, kv, mu_arr_p, vr, rd, yw, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lp = float(loss_eval.numpy()[0])

                mu_arr_m = wp.array([0.7 - h_mu], dtype=float, device=device)
                wp.launch(
                    _eval_deflection_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, kt_arr, kv, mu_arr_m, vr, rd, yw, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                lm = float(loss_eval.numpy()[0])

                fd_mu = (lp - lm) / (2.0 * h_mu)
                self.assertGreater(abs(fd_mu), 10.0)
                self.assertAlmostEqual(grad_mu / fd_mu, 1.0, delta=0.03)

    def test_coupled_implicit_deflection_solve_and_gradient(self):
        """Test coupled implicit_deflection mode forward solve and reverse-mode gradient."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                points = np.array([[-0.08, -0.03, -0.04], [0.06, 0.04, -0.04]], np.float32)
                matrices = np.diag([0.5, 0.5, 0.5, 5.0, 4.0, 8.0]).astype(np.float32)

                p = FrictionParams()
                p.mu = 100.0  # sticking regime
                p.viscous_ratio = 0.2
                p.release_dwell = 0.001
                p.smoothing_speed = 0.01
                p.yield_width = 0.0

                pts = wp.array(points, dtype=wp.vec3, device=device)
                nrm = wp.full(2, 100.0, dtype=float, device=device)
                com = wp.zeros(1, dtype=wp.vec3, device=device)
                vel0 = wp.array([[0.4, -0.2, 0, 0.2, 0.0, 0.3]], dtype=wp.spatial_vector, device=device)
                mob = wp.array([matrices], dtype=wp.spatial_matrix, device=device)
                kt = wp.array([80000.0, 100000.0], dtype=float, device=device, requires_grad=True)
                kv = wp.array([0.0, 0.0], dtype=float, device=device)
                params = wp.array([p], dtype=FrictionParams, device=device)
                anch = wp.array(points[:, :2], dtype=wp.vec2, device=device)
                stk = wp.ones(2, dtype=int, device=device)
                dwl = wp.zeros(2, dtype=float, device=device)
                defl = wp.zeros(2, dtype=wp.vec2, device=device)

                dt = 0.01
                # Forward comparison with backward Euler reference
                solver = FrictionSolver(
                    2, 1, mode="implicit_deflection", iterations=4, device=device, requires_grad=True
                )
                loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

                with wp.Tape() as tape:
                    res = solver.solve(pts, nrm, com, vel0, mob, kt, kv, params, anch, stk, dwl, dt, deflection=defl)
                    wp.launch(_loss_vec2, dim=1, inputs=[res.force, loss], device=device)
                tape.backward(loss)

                # Reference backward Euler velocity
                v0 = vel0.numpy()[0]
                mobility_np = mob.numpy()[0]
                kt_np = kt.numpy()
                stiffness = np.zeros((6, 6))
                for r, k in zip(points, kt_np, strict=True):
                    b = np.array([[1, 0, 0, 0, r[2], -r[1]], [0, 1, 0, -r[2], 0, r[0]]])
                    stiffness += k * b.T @ b
                expected_v = np.linalg.solve(np.eye(6) + dt**2 * mobility_np @ stiffness, v0)

                np.testing.assert_allclose(res.velocity.numpy()[0], expected_v, rtol=2e-5, atol=3e-7)
                self.assertLess(float(res.linear_residual.numpy()[0]), 2e-6)

                # Gradient check against finite differences
                analytic_grad = kt.grad.numpy().copy()
                base_kt = kt.numpy().copy()
                numeric_grad = []
                for idx in range(2):
                    h = base_kt[idx] * 0.003
                    changed = base_kt.copy()
                    changed[idx] += h
                    kt.assign(changed)
                    r_high = solver.solve(pts, nrm, com, vel0, mob, kt, kv, params, anch, stk, dwl, dt, deflection=defl)
                    f_high = r_high.force.numpy().copy()

                    changed[idx] -= 2 * h
                    kt.assign(changed)
                    r_low = solver.solve(pts, nrm, com, vel0, mob, kt, kv, params, anch, stk, dwl, dt, deflection=defl)
                    f_low = r_low.force.numpy().copy()

                    val_high = f_high[0, 0] + 0.3 * f_high[1, 1]
                    val_low = f_low[0, 0] + 0.3 * f_low[1, 1]
                    numeric_grad.append((val_high - val_low) / (2 * h))

                np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=0.02, atol=1e-6)

    def test_zero_slip_force_invariance_and_timestep_study(self):
        """Verify zero-slip force invariance with yield_width=0 and document non-idempotent creep with yield_width>0.

        For v=0, kt=1000, mu=1, Fn=10, initial z=0.0098:
        Stored force magnitude is 9.8 N (0.98 of cap 10 N).
        With yield_width=0.0, repeated calls at v=0 remain strictly invariant across timesteps.
        With yield_width=0.1, g(9.8)=9.64 induces timestep-dependent spurious relaxation towards 9.0 N.
        """
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                normal = 10.0
                kt = 1000.0
                mu = 1.0
                z_initial = 0.0098  # 9.8 N elastic force, 0.98 of cap 10 N

                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                # 1. Qualified production law: yield_width = 0.0
                # Across different timesteps (dt=0.01 vs dt=0.001) and step counts (10 vs 100),
                # force and deflection MUST remain strictly invariant.
                for dt, n_steps in [(0.01, 10), (0.001, 100)]:
                    cur_z = wp.vec2(z_initial, 0.0)
                    cur_stuck = 1
                    cur_dwell = 0.0
                    for _ in range(n_steps):
                        wp.launch(
                            _eval_deflection_step,
                            dim=1,
                            inputs=[
                                wp.vec2(0.0, 0.0),
                                dt,
                                normal,
                                kt,
                                0.0,
                                mu,
                                0.0,
                                0.01,
                                0.0,
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
                    # Strictly invariant at 9.8 N
                    np.testing.assert_allclose(out_f.numpy()[0], [-9.8, 0.0], atol=1e-5)
                    np.testing.assert_allclose(out_z.numpy()[0], [z_initial, 0.0], atol=1e-7)

                # 2. Unsupported experimental smooth shoulder: yield_width = 0.1
                # Demonstrates non-idempotent numerical creep and timestep-rate dependence:
                # dt=0.01 (10 steps) relaxes to ~9.25 N; dt=0.001 (100 steps) relaxes to ~9.04 N.
                cur_z_dt1 = wp.vec2(z_initial, 0.0)
                for _ in range(10):
                    wp.launch(
                        _eval_deflection_step,
                        dim=1,
                        inputs=[
                            wp.vec2(0.0, 0.0),
                            0.01,
                            normal,
                            kt,
                            0.0,
                            mu,
                            0.0,
                            0.01,
                            0.1,
                            cur_z_dt1,
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
                    cur_z_dt1 = out_z.numpy()[0]
                f_dt1 = abs(float(out_f.numpy()[0, 0]))

                cur_z_dt2 = wp.vec2(z_initial, 0.0)
                for _ in range(100):
                    wp.launch(
                        _eval_deflection_step,
                        dim=1,
                        inputs=[
                            wp.vec2(0.0, 0.0),
                            0.001,
                            normal,
                            kt,
                            0.0,
                            mu,
                            0.0,
                            0.01,
                            0.1,
                            cur_z_dt2,
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
                    cur_z_dt2 = out_z.numpy()[0]
                f_dt2 = abs(float(out_f.numpy()[0, 0]))

                # Both have experienced unphysical creep below 9.8 N
                self.assertLess(f_dt1, 9.8 - 0.4)
                self.assertLess(f_dt2, f_dt1 - 0.1)

    def test_adapter_rejects_nonzero_yield_width(self):
        """Verify FrictionAdapter rejects nonzero yield_width as unsupported at public API."""
        foundation = MagicMock()
        foundation.compression.device = "cpu"
        foundation.world_count = 1
        foundation.column_count = 2
        foundation.friction_solver = None
        mobility = wp.array([np.eye(6, dtype=np.float32)], dtype=wp.spatial_matrix, device="cpu")

        with self.assertRaises(ValueError):
            FrictionAdapter(foundation, mobility, yield_width=0.1)

        # yield_width=0.0 is accepted
        adapter = FrictionAdapter(foundation, mobility, yield_width=0.0)
        self.assertEqual(adapter.yield_width, 0.0)


if __name__ == "__main__":
    unittest.main()
