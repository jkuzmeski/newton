# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tangential friction force continuity and derivative kinks."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.contact import bristle_step
from projects.digital_shoe.friction_deflection import bristle_deflection_step


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
def _eval_legacy_step(
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


class TestDigitalShoeFrictionContinuity(unittest.TestCase):
    """Verify raw friction force C0 continuity and derivative kinks on CPU and CUDA."""

    def setUp(self):
        self.devices = ["cpu"]
        if wp.is_cuda_available():
            self.devices.append("cuda:0")

    def _call_deflection(
        self,
        device,
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
    ):
        out_f = wp.zeros(1, dtype=wp.vec2, device=device)
        out_J = wp.zeros(1, dtype=wp.mat22, device=device)
        out_z = wp.zeros(1, dtype=wp.vec2, device=device)
        out_s = wp.zeros(1, dtype=int, device=device)
        out_d = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            _eval_deflection_step,
            dim=1,
            inputs=[
                wp.vec2(vel[0], vel[1]),
                float(dt),
                float(normal),
                float(kt),
                float(kv),
                float(mu),
                float(viscous_ratio),
                float(release_dwell),
                float(yield_width),
                wp.vec2(deflection[0], deflection[1]),
                int(stuck),
                float(dwell),
                out_f,
                out_J,
                out_z,
                out_s,
                out_d,
            ],
            device=device,
        )
        return out_f.numpy()[0], out_J.numpy()[0], out_z.numpy()[0], int(out_s.numpy()[0]), float(out_d.numpy()[0])

    def _call_legacy(
        self,
        device,
        pos,
        vel,
        dt,
        normal,
        kt,
        kv,
        mu,
        viscous_ratio,
        release_dwell,
        anchor,
        stuck,
        dwell,
    ):
        out_f = wp.zeros(1, dtype=wp.vec2, device=device)
        out_a = wp.zeros(1, dtype=wp.vec2, device=device)
        out_s = wp.zeros(1, dtype=int, device=device)
        out_d = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            _eval_legacy_step,
            dim=1,
            inputs=[
                wp.vec2(pos[0], pos[1]),
                wp.vec2(vel[0], vel[1]),
                float(dt),
                float(normal),
                float(kt),
                float(kv),
                float(mu),
                float(viscous_ratio),
                float(release_dwell),
                wp.vec2(anchor[0], anchor[1]),
                int(stuck),
                float(dwell),
                out_f,
                out_a,
                out_s,
                out_d,
            ],
            device=device,
        )
        return out_f.numpy()[0], out_a.numpy()[0], int(out_s.numpy()[0]), float(out_d.numpy()[0])

    def _eval_prescribed_trajectory(self, device, total_time, n_steps, kt, kv, mu, release_dwell):
        dt = total_time / n_steps
        times = np.linspace(0.0, total_time, n_steps + 1)

        z_cur = (0.0, 0.0)
        s_cur = 0
        d_cur = 0.0

        pos_cur = (0.0, 0.0)
        a_cur = (0.0, 0.0)
        s_cur_leg = 0
        d_cur_leg = 0.0

        forces_defl = []
        forces_leg = []

        for t in times[:-1]:
            # Prescribed smooth normal load: Fn(t) = 100 * sin(pi * t / total_time)
            Fn = max(0.0, 100.0 * np.sin(np.pi * t / total_time))
            # Prescribed smooth velocity: v(t) = 0.4 * cos(2 * pi * t / total_time)
            v_x = 0.4 * np.cos(2.0 * np.pi * t / total_time)
            vel = (v_x, 0.0)

            f_defl, _, z_cur, s_cur, d_cur = self._call_deflection(
                device, vel, dt, Fn, kt, kv, mu, 0.2, release_dwell, 0.0, z_cur, s_cur, d_cur
            )
            f_leg, a_cur, s_cur_leg, d_cur_leg = self._call_legacy(
                device, pos_cur, vel, dt, Fn, kt, kv, mu, 0.2, release_dwell, a_cur, s_cur_leg, d_cur_leg
            )

            forces_defl.append(f_defl[0])
            forces_leg.append(f_leg[0])
            pos_cur = (pos_cur[0] + vel[0] * dt, 0.0)

        return np.array(forces_defl), np.array(forces_leg)

    def test_c0_continuity_and_derivative_kink_across_stick_slide_yield(self):
        """Verify force C0 continuity and derivative kink at the Coulomb yield threshold.

        Tests epsilon limits across the slip velocity boundary for both bristle_deflection_step
        (with yield_width=0.0) and legacy bristle_step. Verifies that force difference scales
        linearly with epsilon (|f_+ - f_-| -> 0 as eps -> 0), proving C0 continuity, while
        the velocity Jacobian exhibits a discontinuous step (derivative kink, not C1).
        """
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                normal = 100.0
                mu = 0.5
                kt = 1000.0
                dt = 0.01
                # Normal cap C = mu * normal = 50.0 N.
                # For z_old = 0, trial elastic = -kt * dt * v = -10 * v.
                # Yield threshold is at v_crit = C / (kt * dt) = 5.0 m/s.
                v_crit = 5.0
                epsilons = [1e-2, 1e-3, 1e-4]

                prev_diff_defl = None
                prev_diff_leg = None

                for eps in epsilons:
                    # Below yield (stick regime)
                    v_sub = (v_crit - eps, 0.0)
                    f_sub_defl, J_sub_defl, _, _, _ = self._call_deflection(
                        device, v_sub, dt, normal, kt, 0.0, mu, 0.0, 0.01, 0.0, (0.0, 0.0), 1, 0.0
                    )
                    f_sub_leg, _, _, _ = self._call_legacy(
                        device, (0.0, 0.0), v_sub, dt, normal, kt, 0.0, mu, 0.0, 0.01, (0.0, 0.0), 1, 0.0
                    )

                    # Above yield (slide regime)
                    v_super = (v_crit + eps, 0.0)
                    f_super_defl, J_super_defl, _, _, _ = self._call_deflection(
                        device, v_super, dt, normal, kt, 0.0, mu, 0.0, 0.01, 0.0, (0.0, 0.0), 1, 0.0
                    )
                    f_super_leg, _, _, _ = self._call_legacy(
                        device, (0.0, 0.0), v_super, dt, normal, kt, 0.0, mu, 0.0, 0.01, (0.0, 0.0), 1, 0.0
                    )

                    diff_defl = float(np.linalg.norm(f_super_defl - f_sub_defl))
                    diff_leg = float(np.linalg.norm(f_super_leg - f_sub_leg))

                    # Parity between deflection step and legacy step
                    np.testing.assert_allclose(f_sub_defl, f_sub_leg, atol=1e-5)
                    np.testing.assert_allclose(f_super_defl, f_super_leg, atol=1e-5)

                    # Force jump must vanish linearly: |f_+ - f_-| <= kt * dt * eps = 10 * eps
                    self.assertLessEqual(diff_defl, kt * dt * eps * 1.05)
                    self.assertLessEqual(diff_leg, kt * dt * eps * 1.05)

                    # Verify convergence: as eps drops by 10x, diff drops by ~10x
                    if prev_diff_defl is not None:
                        self.assertLess(diff_defl, prev_diff_defl * 0.2)
                        self.assertLess(diff_leg, prev_diff_leg * 0.2)
                    prev_diff_defl = diff_defl
                    prev_diff_leg = diff_leg

                    # Derivative kink: df/dv has a jump discontinuity across yield threshold.
                    # In stick: J = -kt * dt * I = -10.0 * I.
                    # In slide (radial along x): along velocity direction, dfe/dv_x = 0.
                    # Hence J[0, 0] jumps from -10.0 to 0.0 (non-C1).
                    self.assertAlmostEqual(J_sub_defl[0, 0], -10.0, places=4)
                    self.assertAlmostEqual(J_super_defl[0, 0], 0.0, places=4)
                    jacobian_jump = abs(J_super_defl[0, 0] - J_sub_defl[0, 0])
                    self.assertGreater(jacobian_jump, 5.0)

    def test_c0_continuity_across_speed_reversal_and_zero_velocity(self):
        """Verify force continuity and bounded smooth transition across velocity reversal through v=0."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                normal = 80.0
                kt = 2000.0
                kv = 40.0
                mu = 0.6
                dt = 0.005
                # Pre-deflected bristle z = (0.01, 0.0)
                z_init = (0.01, 0.0)

                epsilons = [1e-2, 1e-3, 1e-4]
                prev_diff = None

                for eps in epsilons:
                    v_neg = (-eps, 0.0)
                    v_pos = (+eps, 0.0)

                    f_neg, _J_neg, _, _, _ = self._call_deflection(
                        device, v_neg, dt, normal, kt, kv, mu, 0.3, 0.01, 0.0, z_init, 1, 0.0
                    )
                    f_pos, _J_pos, _, _, _ = self._call_deflection(
                        device, v_pos, dt, normal, kt, kv, mu, 0.3, 0.01, 0.0, z_init, 1, 0.0
                    )

                    # Also compare with exact v = 0
                    f_zero, _, _, _, _ = self._call_deflection(
                        device, (0.0, 0.0), dt, normal, kt, kv, mu, 0.3, 0.01, 0.0, z_init, 1, 0.0
                    )

                    diff_neg = float(np.linalg.norm(f_neg - f_zero))
                    diff_pos = float(np.linalg.norm(f_pos - f_zero))
                    diff_span = float(np.linalg.norm(f_pos - f_neg))

                    # Parity with legacy helper
                    f_neg_leg, _, _, _ = self._call_legacy(
                        device, z_init, v_neg, dt, normal, kt, kv, mu, 0.3, 0.01, (0.0, 0.0), 1, 0.0
                    )
                    f_pos_leg, _, _, _ = self._call_legacy(
                        device, z_init, v_pos, dt, normal, kt, kv, mu, 0.3, 0.01, (0.0, 0.0), 1, 0.0
                    )
                    np.testing.assert_allclose(f_neg, f_neg_leg, atol=1e-5)
                    np.testing.assert_allclose(f_pos, f_pos_leg, atol=1e-5)

                    # Force vanishes to f_zero as eps -> 0
                    self.assertLess(diff_neg, (kt * dt + kv) * eps * 1.5)
                    self.assertLess(diff_pos, (kt * dt + kv) * eps * 1.5)

                    if prev_diff is not None:
                        self.assertLess(diff_span, prev_diff * 0.2)
                    prev_diff = diff_span

    def test_c0_continuity_across_normal_unload_and_reentry(self):
        """Verify force C0 continuity as normal load approaches zero and re-enters contact."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                kt = 1500.0
                kv = 30.0
                mu = 0.5
                dt = 0.005
                vel = (0.2, 0.0)
                z_init = (0.005, 0.0)

                # At Fn = 0, force must be exactly zero
                f_zero, _, z_zero, stuck_zero, dwell_zero = self._call_deflection(
                    device, vel, dt, 0.0, kt, kv, mu, 0.2, 0.02, 0.0, z_init, 1, 0.0
                )
                np.testing.assert_allclose(f_zero, [0.0, 0.0], atol=1e-6)

                # Parity with legacy
                f_zero_leg, _, stuck_zero_leg, dwell_zero_leg = self._call_legacy(
                    device, z_init, vel, dt, 0.0, kt, kv, mu, 0.2, 0.02, (0.0, 0.0), 1, 0.0
                )
                np.testing.assert_allclose(f_zero_leg, [0.0, 0.0], atol=1e-6)
                self.assertEqual(stuck_zero, stuck_zero_leg)
                self.assertAlmostEqual(dwell_zero, dwell_zero_leg, places=5)

                # As Fn -> 0+, force magnitude is bounded by mu * Fn -> 0 (under slide)
                for fn_eps in [1.0, 0.1, 0.01, 1e-3]:
                    f_eps, _, _, _, _ = self._call_deflection(
                        device, vel, dt, fn_eps, kt, kv, mu, 0.2, 0.02, 0.0, z_init, 1, 0.0
                    )
                    f_mag = float(np.linalg.norm(f_eps))
                    # Normal cone cap is mu * Fn. Force cannot exceed mu * Fn.
                    self.assertLessEqual(f_mag, mu * fn_eps + 1e-6)

                # Re-entry: normal goes from 0 back to Fn > 0
                # Contact resets / maintains deflection within dwell window
                f_reentry, _, _z_reentry, s_reentry, _ = self._call_deflection(
                    device, (0.0, 0.0), dt, 50.0, kt, kv, mu, 0.2, 0.02, 0.0, z_zero, stuck_zero, dwell_zero
                )
                self.assertEqual(s_reentry, 1)
                # Force upon re-entry is finite and continuous with preserved z
                expected_reentry_force = -kt * z_zero[0]
                self.assertAlmostEqual(f_reentry[0], expected_reentry_force, places=4)

    def test_delayed_release_dynamics_with_zero_normal(self):
        """Verify delayed bristle release after release_dwell duration under zero normal force."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                kt = 2000.0
                mu = 0.5
                release_dwell = 0.01  # 10 ms release window
                dt = 0.004  # 4 ms step
                z_init = (0.003, 0.0)

                # Step 1: dt = 0.004, dwell reaches 0.004 <= release_dwell -> retained
                f1, _, z1, s1, d1 = self._call_deflection(
                    device, (0.0, 0.0), dt, 0.0, kt, 0.0, mu, 0.0, release_dwell, 0.0, z_init, 1, 0.0
                )
                np.testing.assert_allclose(f1, [0.0, 0.0], atol=1e-6)
                np.testing.assert_allclose(z1, z_init, atol=1e-6)
                self.assertEqual(s1, 1)
                self.assertAlmostEqual(d1, 0.004, places=5)

                # Parity with legacy step 1
                f1_l, _a1_l, s1_l, d1_l = self._call_legacy(
                    device, z_init, (0.0, 0.0), dt, 0.0, kt, 0.0, mu, 0.0, release_dwell, (0.0, 0.0), 1, 0.0
                )
                np.testing.assert_allclose(f1_l, [0.0, 0.0], atol=1e-6)
                self.assertEqual(s1, s1_l)
                self.assertAlmostEqual(d1, d1_l, places=5)

                # Step 2: dwell reaches 0.008 <= release_dwell -> still retained
                _f2, _, z2, s2, d2 = self._call_deflection(
                    device, (0.0, 0.0), dt, 0.0, kt, 0.0, mu, 0.0, release_dwell, 0.0, z1, s1, d1
                )
                np.testing.assert_allclose(z2, z_init, atol=1e-6)
                self.assertEqual(s2, 1)
                self.assertAlmostEqual(d2, 0.008, places=5)

                # Step 3: dwell would reach 0.012 > release_dwell -> released, z resets to 0, stuck=0, dwell=0
                f3, _, z3, s3, d3 = self._call_deflection(
                    device, (0.0, 0.0), dt, 0.0, kt, 0.0, mu, 0.0, release_dwell, 0.0, z2, s2, d2
                )
                np.testing.assert_allclose(f3, [0.0, 0.0], atol=1e-6)
                np.testing.assert_allclose(z3, [0.0, 0.0], atol=1e-6)
                self.assertEqual(s3, 0)
                self.assertAlmostEqual(d3, 0.0, places=5)

                # Parity with legacy step 3
                f3_l, _a3_l, s3_l, d3_l = self._call_legacy(
                    device, z_init, (0.0, 0.0), dt, 0.0, kt, 0.0, mu, 0.0, release_dwell, (0.0, 0.0), s2, d2
                )
                np.testing.assert_allclose(f3_l, [0.0, 0.0], atol=1e-6)
                self.assertEqual(s3, s3_l)
                self.assertAlmostEqual(d3, d3_l, places=5)

    def test_timestep_consistency_along_smooth_prescribed_load_velocity_path(self):
        """Verify force trajectory continuity and timestep convergence along a prescribed load-velocity path."""
        for device_name in self.devices:
            with self.subTest(device=device_name):
                device = wp.get_device(device_name)
                total_time = 0.4
                kt = 2500.0
                kv = 20.0
                mu = 0.5
                release_dwell = 0.01

                # Test on coarse (dt=0.002, 200 steps) and fine (dt=0.0005, 800 steps)
                f_defl_coarse, f_leg_coarse = self._eval_prescribed_trajectory(
                    device, total_time, 200, kt, kv, mu, release_dwell
                )
                f_defl_fine, f_leg_fine = self._eval_prescribed_trajectory(
                    device, total_time, 800, kt, kv, mu, release_dwell
                )

                # Parity between deflection and legacy on coarse grid (float32 precision)
                np.testing.assert_allclose(f_defl_coarse, f_leg_coarse, atol=1e-4)
                # Parity between deflection and legacy on fine grid (float32 precision)
                np.testing.assert_allclose(f_defl_fine, f_leg_fine, atol=1e-4)

                # Trajectory continuity: step-to-step force differences are bounded
                # max |F(t_{k+1}) - F(t_k)| scales with dt
                df_coarse = np.abs(np.diff(f_defl_coarse))
                df_fine = np.abs(np.diff(f_defl_fine))
                self.assertLess(np.max(df_coarse), 12.0)
                self.assertLess(np.max(df_fine), 3.5)

                # Subsampled comparison at shared time points: f(t) converges as dt -> 0
                # Fine trajectory subsampled every 4 points matches coarse trajectory
                f_fine_sub = f_defl_fine[::4]
                l2_diff = np.mean(np.abs(f_defl_coarse - f_fine_sub))
                self.assertLess(l2_diff, 1.0)


if __name__ == "__main__":
    unittest.main()
