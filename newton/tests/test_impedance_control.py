# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the equilibrium-point parameterizations in projects.impedance_instron.control."""

import math
import unittest

import numpy as np

from projects.impedance_instron.control import AnkleCommand, LegCommand

STANCE_S = 0.25
SEED_STIFFNESS_N_M = 12000.0
SEED_ZETA = 0.6


def _grid(samples: int = 401) -> np.ndarray:
    """Return a uniform stance time grid [s]."""
    return np.linspace(0.0, STANCE_S, samples)


def _reference_length(times: np.ndarray) -> np.ndarray:
    """Return a smooth analytic reference leg length [m] over a stance phase."""
    phase = np.pi * (times - times[0]) / (times[-1] - times[0])
    return 0.98 - 0.05 * np.sin(phase)


class TestLegCommand(unittest.TestCase):
    """Verify the parameter layout, spline profiles and impedance coupling."""

    def test_layout_size_and_bounds_shapes(self):
        """Verify size, bound shapes and the documented parameter block order."""
        cmd = LegCommand(_grid(), length_knots=6, stiffness_knots=5, damping_knots=3)
        self.assertEqual(cmd.size, 6 + 5 + 3)

        lower, upper = cmd.bounds()
        self.assertEqual(lower.shape, (cmd.size,))
        self.assertEqual(upper.shape, (cmd.size,))
        self.assertTrue(np.all(upper > lower))

        np.testing.assert_allclose(lower[:6], 0.5)
        np.testing.assert_allclose(upper[:6], 1.6)
        np.testing.assert_allclose(np.exp(lower[6:11]), 500.0)
        np.testing.assert_allclose(np.exp(upper[6:11]), 120000.0)
        np.testing.assert_allclose(lower[11:], 0.05)
        np.testing.assert_allclose(upper[11:], 3.0)

    def test_pack_unpack_round_trip(self):
        """Verify pack and unpack are inverse and preserve the block order."""
        cmd = LegCommand(_grid(), length_knots=4, stiffness_knots=6, damping_knots=3)
        length = np.linspace(0.9, 1.0, 4)
        log_stiffness = np.linspace(math.log(1000.0), math.log(20000.0), 6)
        zeta = np.array([0.3, 0.7, 1.1])

        params = cmd.pack(length, log_stiffness, zeta)
        self.assertEqual(params.shape, (cmd.size,))
        np.testing.assert_allclose(params[:4], length)
        np.testing.assert_allclose(params[4:10], log_stiffness)
        np.testing.assert_allclose(params[10:], zeta)

        back = cmd.unpack(params)
        np.testing.assert_allclose(back[0], length)
        np.testing.assert_allclose(back[1], log_stiffness)
        np.testing.assert_allclose(back[2], zeta)

        with self.assertRaises(ValueError):
            cmd.unpack(np.zeros(cmd.size + 1))

    def test_initial_round_trip_tracks_reference(self):
        """Verify the seed reproduces a smooth analytic reference length.

        The least-squares fit of six cubic coefficients to a half-sine stance
        length is required to stay within 0.5 mm, well below the millimetre
        resolution of the rig it replaces.
        """
        times = _grid()
        reference = _reference_length(times)
        cmd = LegCommand(times)

        profile = cmd.evaluate(cmd.initial(reference, SEED_STIFFNESS_N_M, SEED_ZETA))

        self.assertLess(float(np.max(np.abs(profile.length_m - reference))), 5.0e-4)
        np.testing.assert_allclose(profile.stiffness_n_m, SEED_STIFFNESS_N_M, rtol=1.0e-9)
        implied_zeta = profile.damping_n_s_m / (2.0 * np.sqrt(profile.stiffness_n_m * cmd.mass_kg))
        np.testing.assert_allclose(implied_zeta, SEED_ZETA, rtol=1.0e-9)

    def test_initial_seed_stays_inside_bounds(self):
        """Verify the seed vector lies inside the optimizer bound box."""
        times = _grid()
        cmd = LegCommand(times)
        params = cmd.initial(_reference_length(times), SEED_STIFFNESS_N_M, SEED_ZETA)

        lower, upper = cmd.bounds()
        self.assertTrue(np.all(params >= lower))
        self.assertTrue(np.all(params <= upper))

    def test_length_rate_matches_finite_difference(self):
        """Verify the analytic length rate matches a fine central difference."""
        times = _grid()
        cmd = LegCommand(times)
        params = cmd.initial(_reference_length(times), SEED_STIFFNESS_N_M, SEED_ZETA)

        # The basis depends only on normalized time, so a denser grid over the
        # same span samples the same curve at higher resolution.
        fine_times = np.linspace(times[0], times[-1], 20001)
        fine = LegCommand(fine_times)
        profile = fine.evaluate(params)

        step = fine_times[1] - fine_times[0]
        central = (profile.length_m[2:] - profile.length_m[:-2]) / (2.0 * step)
        np.testing.assert_allclose(central, profile.length_rate_m_s[1:-1], atol=1.0e-8)

    def test_length_is_c2_without_knot_spikes(self):
        """Verify the length curvature is bounded and continuous across knots.

        A clamped cubic B-spline has piecewise-linear, continuous second
        derivative, so the sampled curvature must not jump at interior knots.
        """
        times = np.linspace(0.0, STANCE_S, 4001)
        cmd = LegCommand(times, length_knots=6)
        rng = np.random.default_rng(7)
        lower, upper = cmd.bounds()
        params = lower + rng.random(cmd.size) * (upper - lower)

        profile = cmd.evaluate(params)
        step = times[1] - times[0]
        curvature = (profile.length_m[2:] - 2.0 * profile.length_m[1:-1] + profile.length_m[:-2]) / step**2
        scale = float(np.max(np.abs(curvature)))

        self.assertGreater(scale, 0.0)
        self.assertLess(scale, 1.0e4)
        # A C1 (not C2) spline would show an O(1) curvature jump at a knot.
        self.assertLess(float(np.max(np.abs(np.diff(curvature)))), 1.0e-2 * scale)

    def test_stiffness_positive_over_the_bound_box(self):
        """Verify stiffness stays inside [500, 120000] N/m over the whole bound box."""
        cmd = LegCommand(_grid(121))
        lower, upper = cmd.bounds()
        rng = np.random.default_rng(3)

        samples = [lower, upper, 0.5 * (lower + upper)]
        for _ in range(64):
            samples.append(np.where(rng.random(cmd.size) < 0.5, lower, upper))
        for _ in range(64):
            samples.append(lower + rng.random(cmd.size) * (upper - lower))

        for params in samples:
            profile = cmd.evaluate(params)
            self.assertTrue(np.all(profile.stiffness_n_m > 0.0))
            self.assertTrue(np.all(profile.stiffness_n_m >= 500.0 - 1.0e-6))
            self.assertTrue(np.all(profile.stiffness_n_m <= 120000.0 + 1.0e-6))
            self.assertTrue(np.all(np.isfinite(profile.damping_n_s_m)))
            self.assertTrue(np.all(profile.damping_n_s_m > 0.0))

    def test_damping_follows_the_damping_ratio_law(self):
        """Verify b = 2 zeta sqrt(k m) holds elementwise for constant seeds."""
        times = _grid()
        cmd = LegCommand(times, mass_kg=64.0)
        reference = _reference_length(times)

        for zeta in (0.05, 0.6, 3.0):
            for stiffness in (500.0, 12000.0, 120000.0):
                profile = cmd.evaluate(cmd.initial(reference, stiffness, zeta))
                expected = 2.0 * zeta * np.sqrt(profile.stiffness_n_m * cmd.mass_kg)
                np.testing.assert_allclose(profile.damping_n_s_m, expected, rtol=1.0e-12)

    def test_damping_ratio_invariant_to_stiffness_scaling(self):
        """Verify scaling stiffness leaves the damping ratio unchanged."""
        cmd = LegCommand(_grid(201))
        rng = np.random.default_rng(11)
        lower, upper = cmd.bounds()
        params = lower + rng.random(cmd.size) * (upper - lower)

        base = cmd.evaluate(params)
        scaled_params = params.copy()
        scaled_params[cmd.length_knots : cmd.length_knots + cmd.stiffness_knots] += math.log(4.0)
        scaled = cmd.evaluate(scaled_params)

        np.testing.assert_allclose(scaled.stiffness_n_m, 4.0 * base.stiffness_n_m, rtol=1.0e-12)
        np.testing.assert_allclose(scaled.damping_n_s_m, 2.0 * base.damping_n_s_m, rtol=1.0e-12)
        np.testing.assert_allclose(base.length_m, scaled.length_m, rtol=1.0e-12)

    def test_profile_shapes_and_input_validation(self):
        """Verify profile array shapes and rejection of invalid constructor input."""
        times = _grid(97)
        cmd = LegCommand(times)
        profile = cmd.evaluate(cmd.initial(_reference_length(times), SEED_STIFFNESS_N_M, SEED_ZETA))

        for field in (profile.length_m, profile.length_rate_m_s, profile.stiffness_n_m, profile.damping_n_s_m):
            self.assertEqual(field.shape, times.shape)

        with self.assertRaises(ValueError):
            LegCommand(np.array([0.0]))
        with self.assertRaises(ValueError):
            LegCommand(np.array([0.0, 0.2, 0.1]))
        with self.assertRaises(ValueError):
            LegCommand(times, mass_kg=0.0)
        with self.assertRaises(ValueError):
            LegCommand(times, damping_knots=1)
        with self.assertRaises(ValueError):
            cmd.initial(_reference_length(times), -1.0, SEED_ZETA)


class TestAnkleCommand(unittest.TestCase):
    """Verify the rotational parameter layout and its impedance coupling.

    The ankle reuses the leg machinery with a rotational metric, so these checks mirror
    :class:`TestLegCommand` on the quantities that differ: the angle bounds, the ankle
    stiffness band and the damping law about the pitch inertia.
    """

    def test_layout_size_and_bounds_shapes(self):
        """Verify size, bound shapes and the documented parameter block order."""
        cmd = AnkleCommand(_grid(), angle_knots=6, stiffness_knots=5, damping_knots=3)
        self.assertEqual(cmd.size, 6 + 5 + 3)

        lower, upper = cmd.bounds()
        self.assertEqual(lower.shape, (cmd.size,))
        self.assertEqual(upper.shape, (cmd.size,))
        self.assertTrue(np.all(upper > lower))

        np.testing.assert_allclose(lower[:6], -0.8)
        np.testing.assert_allclose(upper[:6], 1.6)
        np.testing.assert_allclose(np.exp(lower[6:11]), 100.0)
        np.testing.assert_allclose(np.exp(upper[6:11]), 20000.0)
        np.testing.assert_allclose(lower[11:], 0.05)
        np.testing.assert_allclose(upper[11:], 3.0)

    def test_pack_unpack_round_trip(self):
        """Verify pack and unpack are inverse and preserve the block order."""
        cmd = AnkleCommand(_grid(), angle_knots=4, stiffness_knots=6, damping_knots=3)
        angle = np.linspace(-0.4, 0.9, 4)
        log_stiffness = np.linspace(math.log(200.0), math.log(9000.0), 6)
        zeta = np.array([0.3, 0.7, 1.1])

        params = cmd.pack(angle, log_stiffness, zeta)
        self.assertEqual(params.shape, (cmd.size,))
        np.testing.assert_allclose(params[:4], angle)
        np.testing.assert_allclose(params[4:10], log_stiffness)
        np.testing.assert_allclose(params[10:], zeta)

        back = cmd.unpack(params)
        np.testing.assert_allclose(back[0], angle)
        np.testing.assert_allclose(back[1], log_stiffness)
        np.testing.assert_allclose(back[2], zeta)

        with self.assertRaises(ValueError):
            cmd.unpack(np.zeros(cmd.size + 1))

    def test_initial_seed_stays_inside_bounds(self):
        """Verify the seeded ankle vector lies inside the optimizer bound box."""
        times = _grid()
        cmd = AnkleCommand(times)
        # A seed stiffness above the band is clipped into it rather than escaping the box.
        params = cmd.initial(1.2 * np.sin(np.pi * times / times[-1]) - 0.4, 1.0e6, SEED_ZETA)

        lower, upper = cmd.bounds()
        self.assertTrue(np.all(params >= lower))
        self.assertTrue(np.all(params <= upper))

    def test_damping_follows_the_rotational_damping_ratio_law(self):
        """Verify b_theta = 2 zeta sqrt(k_theta I) holds elementwise for constant seeds."""
        times = _grid()
        cmd = AnkleCommand(times, inertia_kg_m2=0.025)
        reference = 0.2 * np.sin(np.pi * times / times[-1])

        for zeta in (0.05, 0.6, 3.0):
            for stiffness in (100.0, 4000.0, 20000.0):
                profile = cmd.evaluate(cmd.initial(reference, stiffness, zeta))
                expected = 2.0 * zeta * np.sqrt(profile.stiffness_nm_per_rad * cmd.inertia_kg_m2)
                np.testing.assert_allclose(profile.damping_nms_per_rad, expected, rtol=1.0e-12)

    def test_profile_shapes_and_input_validation(self):
        """Verify profile array shapes and rejection of invalid constructor input."""
        times = _grid(97)
        cmd = AnkleCommand(times)
        profile = cmd.evaluate(cmd.initial(np.zeros_like(times), 4000.0, SEED_ZETA))

        for field in (
            profile.angle_rad,
            profile.angle_rate_rad_s,
            profile.stiffness_nm_per_rad,
            profile.damping_nms_per_rad,
        ):
            self.assertEqual(field.shape, times.shape)

        with self.assertRaises(ValueError):
            AnkleCommand(np.array([0.0]))
        with self.assertRaises(ValueError):
            AnkleCommand(np.array([0.0, 0.2, 0.1]))
        with self.assertRaises(ValueError):
            AnkleCommand(times, inertia_kg_m2=0.0)
        with self.assertRaises(ValueError):
            AnkleCommand(times, damping_knots=1)
        with self.assertRaises(ValueError):
            cmd.initial(np.zeros_like(times), -1.0, SEED_ZETA)
        with self.assertRaises(ValueError):
            cmd.initial(np.zeros(5), 4000.0, SEED_ZETA)

    def test_leg_layout_is_untouched_by_the_ankle(self):
        """Verify the leg parameter layout and bounds are unchanged by the shared basis."""
        cmd = LegCommand(_grid(), length_knots=6, stiffness_knots=6, damping_knots=3)
        lower, upper = cmd.bounds()
        self.assertEqual(cmd.size, 15)
        np.testing.assert_allclose(lower[:6], 0.5)
        np.testing.assert_allclose(np.exp(upper[6:12]), 120000.0)
        np.testing.assert_allclose(upper[12:], 3.0)


if __name__ == "__main__":
    unittest.main()
