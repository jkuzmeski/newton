# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for 12x4 Cartesian spline linear control-polygon constraints and kernels."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from projects.impedance_instron.cartesian.gpu.constraints import (
    SplineConstraints,
    build_polytope_system,
)
from projects.impedance_instron.cartesian.trajectory import Spline

PROFILE = {
    "equilibrium_lower": [-0.18706925802385221, 0.43664648117189697, -2.240879416600583, -2.0605111429635445],
    "equilibrium_upper": [1.90736507709526, 1.5050702188249552, 1.4235366453481377, 1.7643870219635445],
    "equilibrium_rate_limit": [10.0, 10.0, 20.0, 25.0],
    "equilibrium_acceleration_limit": [200.0, 200.0, 400.0, 600.0],
}


class TestSplineConstraints(unittest.TestCase):
    """Test linear control-polygon inequalities, coordinate transforms, and Warp kernels."""

    @classmethod
    def setUpClass(cls):
        """Initialize available native test devices (CPU and CUDA)."""
        wp.init()
        cls.devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            cls.devices.append(wp.get_device("cuda:0"))

    def setUp(self):
        """Set up standard fixture profile and constraints."""
        self.duration_s = 0.36
        self.lower = np.asarray(PROFILE["equilibrium_lower"], dtype=np.float64)
        self.upper = np.asarray(PROFILE["equilibrium_upper"], dtype=np.float64)
        self.rate_limit = np.asarray(PROFILE["equilibrium_rate_limit"], dtype=np.float64)
        self.acc_limit = np.asarray(PROFILE["equilibrium_acceleration_limit"], dtype=np.float64)
        self.parameter_scale = np.array([0.2, 0.2, 0.5, 0.5], dtype=np.float64)
        self.theta_start = np.tile((self.lower + self.upper) / 2.0, (12, 1))

        self.constraints = SplineConstraints(
            self.duration_s,
            self.lower,
            self.upper,
            self.rate_limit,
            self.acc_limit,
            parameter_scale=self.parameter_scale,
            theta_start=self.theta_start,
        )

    def test_polytope_system_dimensions(self):
        """Verify polytope system constructs exactly 264 linear inequalities in 48 variables."""
        a_mat, b_vec = build_polytope_system(
            self.duration_s,
            self.lower,
            self.upper,
            self.rate_limit,
            self.acc_limit,
        )
        self.assertEqual(a_mat.shape, (264, 48))
        self.assertEqual(b_vec.shape, (264,))
        self.assertEqual(self.constraints.a_np.shape, (264, 48))
        self.assertEqual(self.constraints.b_np.shape, (264,))
        self.assertEqual(self.constraints.a_z_np.shape, (264, 48))
        self.assertEqual(self.constraints.b_z_np.shape, (264,))

    def test_random_splines_match_spline_bounds(self):
        """Compare linear polytope feasibility against Spline.bounds for random splines."""
        rng = np.random.default_rng(101)
        for _ in range(50):
            # Mix of scales to produce both feasible and infeasible splines
            scale = rng.uniform(0.1, 5.0)
            candidate = self.theta_start + rng.normal(scale=scale, size=(12, 4))
            spline = Spline(self.duration_s, candidate)
            expected = spline.bounds(self.lower, self.upper, self.rate_limit, self.acc_limit)
            actual = self.constraints.check_feasibility_host(candidate)
            self.assertEqual(expected, actual)

    def test_boundary_cases_match_spline_bounds(self):
        """Verify exact boundary feasibility matches Spline.bounds at limit surfaces."""
        # Exact upper constant spline is feasible
        theta_upper = np.tile(self.upper, (12, 1))
        spline_upper = Spline(self.duration_s, theta_upper)
        self.assertTrue(spline_upper.bounds(self.lower, self.upper, self.rate_limit, self.acc_limit))
        self.assertTrue(self.constraints.check_feasibility_host(theta_upper))

        # Exact lower constant spline is feasible
        theta_lower = np.tile(self.lower, (12, 1))
        spline_lower = Spline(self.duration_s, theta_lower)
        self.assertTrue(spline_lower.bounds(self.lower, self.upper, self.rate_limit, self.acc_limit))
        self.assertTrue(self.constraints.check_feasibility_host(theta_lower))

        # Perturbation outside position bound fails both
        theta_viol_pos = theta_upper.copy()
        theta_viol_pos[5, 1] += 1e-4
        spline_viol_pos = Spline(self.duration_s, theta_viol_pos)
        self.assertFalse(spline_viol_pos.bounds(self.lower, self.upper, self.rate_limit, self.acc_limit))
        self.assertFalse(self.constraints.check_feasibility_host(theta_viol_pos))

        # Box-feasible but rate-violating spline fails both (box clipping alone is insufficient)
        theta_rate_viol = self.theta_start.copy()
        # Set alternate rows to create large derivative while staying within [lower, upper]
        for r in range(12):
            if r % 2 == 0:
                theta_rate_viol[r, 0] = self.upper[0]
            else:
                theta_rate_viol[r, 0] = self.lower[0]
        # Position is within [lower, upper]
        self.assertTrue(np.all(theta_rate_viol >= self.lower) and np.all(theta_rate_viol <= self.upper))
        # But rate limit is exceeded
        spline_rate_viol = Spline(self.duration_s, theta_rate_viol)
        self.assertFalse(spline_rate_viol.bounds(self.lower, self.upper, self.rate_limit, self.acc_limit))
        self.assertFalse(self.constraints.check_feasibility_host(theta_rate_viol))

    def test_coordinate_mapping_and_scaled_system(self):
        """Verify theta = theta_start + S * z mapping and scaled feasibility equivalence."""
        rng = np.random.default_rng(202)
        for _ in range(20):
            theta = self.theta_start + rng.normal(scale=0.1, size=(12, 4))
            z = self.constraints.to_scaled(theta)
            theta_roundtrip = self.constraints.to_physical(z)
            np.testing.assert_allclose(theta, theta_roundtrip, rtol=1e-14, atol=1e-14)

            # Feasibility in physical coordinates matches feasibility in scaled coordinates
            feas_phys = self.constraints.check_feasibility_host(theta, scaled=False)
            feas_scaled = self.constraints.check_feasibility_host(z, scaled=True)
            self.assertEqual(feas_phys, feas_scaled)

    def test_ray_step_caps_and_boundary_exactness(self):
        """Verify maximum feasible ray step caps the direction at the exact boundary."""
        rng = np.random.default_rng(303)
        origin = self.theta_start.reshape(48)
        self.assertTrue(self.constraints.check_feasibility_host(origin))

        for _ in range(25):
            direction = rng.normal(size=48)
            alpha_max = self.constraints.max_ray_step_host(origin, direction)
            self.assertGreater(alpha_max, 0.0)

            # At alpha_max, the point is at the boundary (max slack within 1e-11)
            at_bound = origin + alpha_max * direction
            max_viol = np.max(self.constraints.a_np @ at_bound - self.constraints.b_np)
            self.assertAlmostEqual(max_viol, 0.0, places=10)

            # Point just inside is feasible
            inside = origin + (alpha_max * (1.0 - 1e-6)) * direction
            self.assertTrue(
                Spline(self.duration_s, inside.reshape(12, 4)).bounds(
                    self.lower, self.upper, self.rate_limit, self.acc_limit
                )
            )

            # Point just outside is infeasible
            outside = origin + (alpha_max * (1.0 + 1e-5) + 1e-6) * direction
            self.assertFalse(
                Spline(self.duration_s, outside.reshape(12, 4)).bounds(
                    self.lower, self.upper, self.rate_limit, self.acc_limit
                )
            )

    def test_strict_bounds_nonfinite_inputs_and_tiny_rays(self):
        """Reject nonfinite proposals and preserve strict boundaries for arbitrarily small directions."""
        upper = np.tile(self.upper, (12, 1))
        direction = np.zeros((12, 4))
        direction[0, 0] = 1e-16
        self.assertEqual(self.constraints.max_ray_step_host(upper, direction), 0.0)
        self.assertTrue(np.isinf(self.constraints.max_ray_step_host(self.theta_start, np.zeros((12, 4)))))
        candidates = np.stack([self.theta_start, upper, self.theta_start, self.theta_start, self.theta_start])
        candidates[2, 0, 0] = np.nan
        candidates[3, 0, 1] = np.inf
        candidates[4, 0, 2] = -np.inf
        expected = [self.constraints.check_control_bounds_host(c) for c in candidates]
        self.assertEqual(expected, [True, True, False, False, False])
        for device in self.devices:
            with self.subTest(device=device):
                values = wp.array(candidates, dtype=wp.float64, device=device)
                flags = wp.zeros(len(candidates), dtype=int, device=device)
                self.constraints.evaluate_control_bounds_device(values, flags)
                np.testing.assert_array_equal(flags.numpy(), expected)
                scaled = self.constraints.to_scaled(candidates)
                scaled_wp = wp.array(scaled, dtype=wp.float64, device=device)
                self.constraints.evaluate_control_bounds_device(scaled_wp, flags, scaled=True)
                np.testing.assert_array_equal(
                    flags.numpy(), [self.constraints.check_control_bounds_host(c, scaled=True) for c in scaled]
                )
                bad_shape = wp.zeros((2, 24), dtype=wp.float64, device=device)
                with self.assertRaises(ValueError):
                    self.constraints.evaluate_feasibility_device(bad_shape, flags)

    def test_warp_device_feasibility_and_ray_kernels(self):
        """Test Warp feasibility and ray-step kernels on CPU and CUDA."""
        for device in self.devices:
            with self.subTest(device=device):
                self._check_device_kernels(device)

    def _check_device_kernels(self, device):
        """Compare device results with host constraints for one execution device."""
        sc = SplineConstraints(
            self.duration_s,
            self.lower,
            self.upper,
            self.rate_limit,
            self.acc_limit,
            parameter_scale=self.parameter_scale,
            theta_start=self.theta_start,
            device=device,
        )

        candidates = np.stack(
            [
                self.theta_start.reshape(48),  # feasible
                np.tile(self.upper, 12),  # feasible (upper boundary)
                np.tile(self.lower, 12),  # feasible (lower boundary)
                np.tile(self.upper + 1.0, 12),  # infeasible (exceeds upper)
                np.tile(self.lower - 1.0, 12),  # infeasible (exceeds lower)
            ]
        )
        expected_feas = [int(sc.check_feasibility_host(c)) for c in candidates]

        cand_wp = wp.array(candidates, dtype=wp.float64, device=device)
        feas_out = wp.zeros(len(candidates), dtype=wp.int32, device=device)
        sc.evaluate_feasibility_device(cand_wp, feas_out, device=device)
        np.testing.assert_array_equal(feas_out.numpy(), expected_feas)

        # Test ray lengths
        rng = np.random.default_rng(404)
        origins = np.tile(self.theta_start.reshape(48), (5, 1))
        directions = rng.normal(size=(5, 48))
        expected_steps = [sc.max_ray_step_host(origins[i], directions[i]) for i in range(5)]

        orig_wp = wp.array(origins, dtype=wp.float64, device=device)
        dir_wp = wp.array(directions, dtype=wp.float64, device=device)
        steps_out = wp.zeros(5, dtype=wp.float64, device=device)
        sc.compute_max_ray_step_device(orig_wp, dir_wp, steps_out, device=device)
        np.testing.assert_allclose(steps_out.numpy(), expected_steps, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
