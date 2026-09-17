# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for provisional tangential pressure-dependent friction cap hypothesis."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.friction_deflection import bristle_deflection_step
from projects.digital_shoe.friction_pressure import (
    bristle_pressure_step,
    pressure_coefficient,
    validate_pressure_parameters,
)


@wp.kernel
def _eval_pressure_coefficient(
    normal: float,
    area: float,
    mu_low: float,
    pressure_scale_pa: float,
    out_mu: wp.array[float],
):
    out_mu[0] = pressure_coefficient(normal, area, mu_low, pressure_scale_pa)


@wp.kernel
def _eval_bristle_pressure_step(
    vel: wp.vec2,
    dt: float,
    normal: float,
    area: float,
    kt: float,
    kv: float,
    mu_low: float,
    pressure_scale_pa: float,
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
    f, J, z, s, d = bristle_pressure_step(
        vel,
        dt,
        normal,
        area,
        kt,
        kv,
        mu_low,
        pressure_scale_pa,
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
def _eval_canonical_bristle_deflection_step(
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
def _eval_pressure_param_tape(
    vel: wp.vec2,
    dt: float,
    normal: float,
    area: float,
    kt: float,
    kv: float,
    mu_low_arr: wp.array[float],
    pstar_arr: wp.array[float],
    viscous_ratio: float,
    release_dwell: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
    loss: wp.array[float],
):
    f, _J, _z, _s, _d = bristle_pressure_step(
        vel,
        dt,
        normal,
        area,
        kt,
        kv,
        mu_low_arr[0],
        pstar_arr[0],
        viscous_ratio,
        release_dwell,
        deflection,
        stuck,
        dwell,
    )
    loss[0] = f[0] * 1.5 + f[1] * 0.7


class TestDigitalShoeFrictionPressure(unittest.TestCase):
    """Test provisional tangential pressure-dependent cap hypothesis across CPU and CUDA."""

    def setUp(self):
        """Prepare execution devices."""
        self.devices = ["cpu"]
        if wp.is_cuda_available():
            self.devices.append("cuda:0")

    def test_host_validation_rejects_invalid_parameters(self):
        """Verify host validator rejects non-positive area, non-positive pstar, and negative mu."""
        with self.assertRaises(ValueError):
            validate_pressure_parameters(area=0.0, pressure_scale_pa=1e5, mu_low=0.8)
        with self.assertRaises(ValueError):
            validate_pressure_parameters(area=-0.01, pressure_scale_pa=1e5, mu_low=0.8)
        with self.assertRaises(ValueError):
            validate_pressure_parameters(area=0.001, pressure_scale_pa=0.0, mu_low=0.8)
        with self.assertRaises(ValueError):
            validate_pressure_parameters(area=0.001, pressure_scale_pa=-500.0, mu_low=0.8)
        with self.assertRaises(ValueError):
            validate_pressure_parameters(area=0.001, pressure_scale_pa=1e5, mu_low=-0.1)

        # Valid parameters should pass without error
        validate_pressure_parameters(area=0.001, pressure_scale_pa=50000.0, mu_low=0.0)
        validate_pressure_parameters(area=0.002, pressure_scale_pa=100000.0, mu_low=0.8)

    def test_coefficient_bounds_and_monotonicity(self):
        """Verify mu_eff is monotonically decreasing with normal pressure and strictly bounded in (0, mu_low]."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                area = 0.002  # 20 cm^2 [m^2]
                mu_low = 0.9
                pstar = 50000.0  # 50 kPa [Pa]

                normals = [0.0, 10.0, 50.0, 100.0, 500.0, 2000.0]
                mu_vals = []

                out = wp.zeros(1, dtype=float, device=device)
                for fn in normals:
                    wp.launch(
                        _eval_pressure_coefficient,
                        dim=1,
                        inputs=[fn, area, mu_low, pstar, out],
                        device=device,
                    )
                    mu_eff = float(out.numpy()[0])
                    mu_vals.append(mu_eff)

                    # Strictly bounded: 0 < mu_eff <= mu_low (or mu_eff == mu_low at fn=0)
                    self.assertGreater(mu_eff, 0.0)
                    self.assertLessEqual(mu_eff, mu_low + 1e-12)

                # Monotonically non-increasing with load / pressure
                for i in range(len(mu_vals) - 1):
                    self.assertGreaterEqual(mu_vals[i], mu_vals[i + 1])

                # Fn = 0 yields exactly mu_low
                self.assertAlmostEqual(mu_vals[0], mu_low, places=6)

    def test_large_pressure_scale_constant_mu_convergence(self):
        """Verify mu_eff converges to constant mu_low as pressure scale pstar approaches infinity."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                area = 0.001
                mu_low = 0.75
                normal = 200.0

                out = wp.zeros(1, dtype=float, device=device)
                for pstar in [1e6, 1e8, 1e11]:
                    wp.launch(
                        _eval_pressure_coefficient,
                        dim=1,
                        inputs=[normal, area, mu_low, pstar, out],
                        device=device,
                    )
                    mu_eff = float(out.numpy()[0])
                    p = normal / area
                    expected = mu_low / (1.0 + p / pstar)
                    self.assertAlmostEqual(mu_eff, expected, places=6)

                # For extremely large pstar, mu_eff is indistinguishable from mu_low
                self.assertAlmostEqual(mu_eff, mu_low, places=5)

    def test_low_load_zero_force(self):
        """Verify normal force <= 0 yields exactly zero tangential force, zero Jacobian, and dwell accumulation."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                dt = 0.01
                kt = 50000.0
                kv = 10.0
                mu_low = 0.8
                pstar = 50000.0
                area = 0.002
                vel = wp.vec2(1.0, -0.5)

                for normal in [0.0, -10.0]:
                    wp.launch(
                        _eval_bristle_pressure_step,
                        dim=1,
                        inputs=[
                            vel,
                            dt,
                            normal,
                            area,
                            kt,
                            kv,
                            mu_low,
                            pstar,
                            0.2,
                            0.05,
                            wp.vec2(0.0001, 0.0001),
                            1,
                            0.02,
                            out_f,
                            out_J,
                            out_z,
                            out_s,
                            out_d,
                        ],
                        device=device,
                    )
                    np.testing.assert_allclose(out_f.numpy()[0], [0.0, 0.0], atol=1e-7)
                    np.testing.assert_allclose(out_J.numpy()[0], np.zeros((2, 2)), atol=1e-7)
                    self.assertAlmostEqual(float(out_d.numpy()[0]), 0.02 + dt, places=6)

    def test_shear_traction_upper_bound(self):
        """Verify shear traction capacity C/area strictly respects the saturation asymptote mu_low * pstar."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                area = 0.002
                mu_low = 0.85
                pstar = 40000.0  # 40 kPa
                c_max_bound = mu_low * pstar * area  # Upper limit on capacity C [N]
                tau_sat = mu_low * pstar  # Upper limit on shear traction [Pa]

                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                # High loads pushing into saturation
                for normal in [100.0, 500.0, 2000.0, 10000.0, 100000.0]:
                    p = normal / area
                    mu_eff = mu_low / (1.0 + p / pstar)
                    c_eff = mu_eff * normal
                    tau_eff = c_eff / area

                    # Theoretical capacity must remain strictly below asymptote
                    self.assertLess(c_eff, c_max_bound)
                    self.assertLess(tau_eff, tau_sat)

                    # Launch bristle step in full sliding regime (kv=0 to test pure Coulomb cap)
                    wp.launch(
                        _eval_bristle_pressure_step,
                        dim=1,
                        inputs=[
                            wp.vec2(10.0, 0.0),  # large velocity to ensure full slide
                            0.01,
                            normal,
                            area,
                            50000.0,
                            0.0,
                            mu_low,
                            pstar,
                            0.0,
                            0.05,
                            wp.vec2(0.0, 0.0),
                            0,
                            0.0,
                            out_f,
                            out_J,
                            out_z,
                            out_s,
                            out_d,
                        ],
                        device=device,
                    )
                    force_mag = np.linalg.norm(out_f.numpy()[0])
                    self.assertAlmostEqual(force_mag, c_eff, places=4)
                    self.assertLess(force_mag, c_max_bound)

    def test_canonical_force_parity(self):
        """Verify bristle_pressure_step matches canonical bristle_deflection_step with mu=mu_eff exactly."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                area = 0.0015
                mu_low = 0.8
                pstar = 60000.0
                dt = 0.005
                kt = 35000.0
                kv = 15.0
                viscous_ratio = 0.25
                release_dwell = 0.01

                cases = [
                    # Sticking regime
                    {"vel": [0.01, -0.02], "normal": 60.0, "z": [0.0001, -0.0001], "stuck": 1, "dwell": 0.0},
                    # Sliding regime
                    {"vel": [1.5, -0.8], "normal": 120.0, "z": [0.0005, 0.0002], "stuck": 1, "dwell": 0.0},
                    # Unloaded separation
                    {"vel": [0.2, 0.1], "normal": 0.0, "z": [0.0002, 0.0001], "stuck": 1, "dwell": 0.005},
                    # Activation of new contact
                    {"vel": [0.5, 0.5], "normal": 40.0, "z": [0.0, 0.0], "stuck": 0, "dwell": 0.0},
                ]

                out_fp = wp.zeros(1, dtype=wp.vec2, device=device)
                out_Jp = wp.zeros(1, dtype=wp.mat22, device=device)
                out_zp = wp.zeros(1, dtype=wp.vec2, device=device)
                out_sp = wp.zeros(1, dtype=int, device=device)
                out_dp = wp.zeros(1, dtype=float, device=device)

                out_fc = wp.zeros(1, dtype=wp.vec2, device=device)
                out_Jc = wp.zeros(1, dtype=wp.mat22, device=device)
                out_zc = wp.zeros(1, dtype=wp.vec2, device=device)
                out_sc = wp.zeros(1, dtype=int, device=device)
                out_dc = wp.zeros(1, dtype=float, device=device)

                for c in cases:
                    normal = c["normal"]
                    p = max(normal, 0.0) / area
                    mu_eff = mu_low / (1.0 + p / pstar)

                    wp.launch(
                        _eval_bristle_pressure_step,
                        dim=1,
                        inputs=[
                            wp.vec2(*c["vel"]),
                            dt,
                            normal,
                            area,
                            kt,
                            kv,
                            mu_low,
                            pstar,
                            viscous_ratio,
                            release_dwell,
                            wp.vec2(*c["z"]),
                            c["stuck"],
                            c["dwell"],
                            out_fp,
                            out_Jp,
                            out_zp,
                            out_sp,
                            out_dp,
                        ],
                        device=device,
                    )

                    wp.launch(
                        _eval_canonical_bristle_deflection_step,
                        dim=1,
                        inputs=[
                            wp.vec2(*c["vel"]),
                            dt,
                            normal,
                            kt,
                            kv,
                            mu_eff,
                            viscous_ratio,
                            release_dwell,
                            0.0,  # yield_width = 0.0
                            wp.vec2(*c["z"]),
                            c["stuck"],
                            c["dwell"],
                            out_fc,
                            out_Jc,
                            out_zc,
                            out_sc,
                            out_dc,
                        ],
                        device=device,
                    )

                    np.testing.assert_allclose(out_fp.numpy()[0], out_fc.numpy()[0], atol=1e-5)
                    np.testing.assert_allclose(out_Jp.numpy()[0], out_Jc.numpy()[0], atol=1e-5)
                    np.testing.assert_allclose(out_zp.numpy()[0], out_zc.numpy()[0], atol=1e-5)
                    self.assertEqual(int(out_sp.numpy()[0]), int(out_sc.numpy()[0]))
                    self.assertAlmostEqual(float(out_dp.numpy()[0]), float(out_dc.numpy()[0]), places=7)

    def test_energy_inequality_variable_normal_load(self):
        """Verify non-positive mechanical energy residual under varying normal force."""
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
                kv = 20.0
                area = 0.002
                mu_low = 0.8
                pstar = 50000.0
                z_cur = np.array([0.0004, -0.0002], dtype=np.float32)

                test_normals = [30.0, 100.0, 300.0]
                test_vels = [
                    [0.05, 0.02],  # sticking
                    [0.8, -0.4],  # sliding
                    [2.5, 1.2],  # fast sliding
                ]

                for fn in test_normals:
                    for vel_val in test_vels:
                        wp.launch(
                            _eval_bristle_pressure_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*vel_val),
                                dt,
                                fn,
                                area,
                                kt,
                                kv,
                                mu_low,
                                pstar,
                                0.2,
                                0.01,
                                wp.vec2(*z_cur),
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

                        e_old = 0.5 * kt * np.dot(z_cur, z_cur)
                        e_new = 0.5 * kt * np.dot(z_new, z_new)
                        friction_work = np.dot(f, vel_val) * dt

                        excess = (e_new - e_old) + friction_work
                        tolerance = 1e-6 * (e_new + e_old + abs(friction_work) + 1e-5)
                        self.assertLessEqual(excess, tolerance)

    def test_zero_slip_idempotence_fixed_normal_force(self):
        """Verify zero slip velocity (v=0) produces strictly idempotent force and deflection across steps."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                normal = 80.0
                area = 0.002
                kt = 20000.0
                mu_low = 0.75
                pstar = 40000.0

                p = normal / area
                mu_eff = mu_low / (1.0 + p / pstar)
                f_cap = mu_eff * normal

                # Pick initial deflection inside stick regime (e.g. 80% of cap)
                z_init_mag = 0.8 * f_cap / kt
                z_initial = np.array([z_init_mag, 0.0], dtype=np.float32)

                out_f = wp.zeros(1, dtype=wp.vec2, device=device)
                out_J = wp.zeros(1, dtype=wp.mat22, device=device)
                out_z = wp.zeros(1, dtype=wp.vec2, device=device)
                out_s = wp.zeros(1, dtype=int, device=device)
                out_d = wp.zeros(1, dtype=float, device=device)

                for dt, n_steps in [(0.01, 10), (0.001, 100)]:
                    cur_z = wp.vec2(*z_initial)
                    cur_stuck = 1
                    cur_dwell = 0.0

                    for _ in range(n_steps):
                        wp.launch(
                            _eval_bristle_pressure_step,
                            dim=1,
                            inputs=[
                                wp.vec2(0.0, 0.0),
                                dt,
                                normal,
                                area,
                                kt,
                                0.0,
                                mu_low,
                                pstar,
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

                    # Strictly invariant at -kt * z_initial
                    expected_force = -kt * z_initial
                    np.testing.assert_allclose(out_f.numpy()[0], expected_force, atol=1e-5)
                    np.testing.assert_allclose(out_z.numpy()[0], z_initial, atol=1e-7)

    def test_finite_difference_velocity_jacobian(self):
        """Match analytical tangent Jacobian to central finite differences across sticking and sliding regimes."""
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
                area = 0.002
                kt = 30000.0
                mu_low = 0.8
                pstar = 50000.0
                vr = 0.25
                rd = 0.005

                test_configs = [
                    ([0.02, 0.01], 0.0, "sticking kv=0"),
                    ([0.02, 0.01], 50.0, "sticking kv=50"),
                    ([0.6, -0.4], 0.0, "sliding kv=0"),
                    ([0.6, -0.4], 50.0, "sliding kv=50"),
                ]

                h = 1e-6
                for vel_raw, kv, _desc in test_configs:
                    z_init = wp.vec2(0.0003, -0.0001)
                    wp.launch(
                        _eval_bristle_pressure_step,
                        dim=1,
                        inputs=[
                            wp.vec2(*vel_raw),
                            dt,
                            normal,
                            area,
                            kt,
                            kv,
                            mu_low,
                            pstar,
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
                            _eval_bristle_pressure_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*v_p),
                                dt,
                                normal,
                                area,
                                kt,
                                kv,
                                mu_low,
                                pstar,
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
                        f_pos = out_f.numpy()[0].copy()

                        v_m = list(vel_raw)
                        v_m[comp] -= h
                        wp.launch(
                            _eval_bristle_pressure_step,
                            dim=1,
                            inputs=[
                                wp.vec2(*v_m),
                                dt,
                                normal,
                                area,
                                kt,
                                kv,
                                mu_low,
                                pstar,
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
                        f_neg = out_f.numpy()[0].copy()

                        fd_J[:, comp] = (f_pos - f_neg) / (2.0 * h)

                    np.testing.assert_allclose(exact_J, fd_J, rtol=0.03, atol=2.0)

    def test_tape_parameter_gradient_mu_and_pstar(self):
        """Verify Warp Tape autodiff gradients w.r.t. mu_low and pstar match finite differences away from branch points."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                dt = 0.01
                normal = 60.0
                area = 0.002
                kt = 25000.0
                kv = 0.0
                vr = 0.2
                rd = 0.005

                # Sliding velocity so friction force is on the Coulomb cap, where dF/dmu != 0 and dF/dpstar != 0
                vel = wp.vec2(0.6, -0.3)
                z_init = wp.vec2(0.0002, -0.0001)

                mu_val = 0.8
                pstar_val = 40000.0

                mu_arr = wp.array([mu_val], dtype=float, device=device, requires_grad=True)
                pstar_arr = wp.array([pstar_val], dtype=float, device=device, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

                with wp.Tape() as tape:
                    wp.launch(
                        _eval_pressure_param_tape,
                        dim=1,
                        inputs=[
                            vel,
                            dt,
                            normal,
                            area,
                            kt,
                            kv,
                            mu_arr,
                            pstar_arr,
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

                grad_mu = float(mu_arr.grad.numpy()[0])
                grad_pstar = float(pstar_arr.grad.numpy()[0])

                # Finite difference for mu_low
                h_mu = 1e-4
                loss_eval = wp.zeros(1, dtype=float, device=device)

                mu_p = wp.array([mu_val + h_mu], dtype=float, device=device)
                wp.launch(
                    _eval_pressure_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, area, kt, kv, mu_p, pstar_arr, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                l_mu_p = float(loss_eval.numpy()[0])

                mu_m = wp.array([mu_val - h_mu], dtype=float, device=device)
                wp.launch(
                    _eval_pressure_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, area, kt, kv, mu_m, pstar_arr, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                l_mu_m = float(loss_eval.numpy()[0])
                fd_mu = (l_mu_p - l_mu_m) / (2.0 * h_mu)

                # Finite difference for pstar
                h_pstar = 10.0
                pstar_p = wp.array([pstar_val + h_pstar], dtype=float, device=device)
                wp.launch(
                    _eval_pressure_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, area, kt, kv, mu_arr, pstar_p, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                l_ps_p = float(loss_eval.numpy()[0])

                pstar_m = wp.array([pstar_val - h_pstar], dtype=float, device=device)
                wp.launch(
                    _eval_pressure_param_tape,
                    dim=1,
                    inputs=[vel, dt, normal, area, kt, kv, mu_arr, pstar_m, vr, rd, z_init, 1, 0.0, loss_eval],
                    device=device,
                )
                l_ps_m = float(loss_eval.numpy()[0])
                fd_pstar = (l_ps_p - l_ps_m) / (2.0 * h_pstar)

                self.assertGreater(abs(fd_mu), 10.0)
                self.assertAlmostEqual(grad_mu / fd_mu, 1.0, delta=0.03)
                self.assertGreater(abs(fd_pstar), 1e-5)
                self.assertAlmostEqual(grad_pstar / fd_pstar, 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
