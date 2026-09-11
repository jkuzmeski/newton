# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-contained CMA-ES optimizer in projects/impedance_instron."""

import json
import math
import unittest

import numpy as np

from projects.impedance_instron.cmaes import CMAES, minimize


def _sphere(x):
    """Return the sphere function, whose minimum is 0 at the origin."""
    return float(np.sum(np.asarray(x) ** 2))


def _rosenbrock(x):
    """Return the Rosenbrock function, whose minimum is 0 at the all-ones vector."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


class TestCMAESConvergence(unittest.TestCase):
    """Convergence of CMA-ES on standard unconstrained benchmarks."""

    def test_sphere_10d(self):
        """Verify CMA-ES drives the 10-D sphere function below 1e-8."""
        result = minimize(_sphere, np.full(10, 1.0), 0.5, max_evaluations=5000, seed=1)
        self.assertLess(result.fun, 1e-8)
        self.assertEqual(result.x.shape, (10,))
        np.testing.assert_allclose(result.x, np.zeros(10), atol=1e-4)
        self.assertEqual(len(result.history), result.generations)
        # best-so-far history must be monotonically non-increasing
        self.assertTrue(all(b <= a for a, b in zip(result.history, result.history[1:], strict=False)))

    def test_rosenbrock_5d(self):
        """Verify CMA-ES reaches 1e-6 on the 5-D Rosenbrock function.

        Rosenbrock has a narrow curved valley, so the threshold is deliberately
        looser than for the sphere. Measured runs reach roughly 1e-10 within
        about 2300 evaluations for several seeds, so 1e-6 leaves a wide margin
        while still failing for an optimizer stuck outside the valley (the local
        plateau near the start is above 1.0).
        """
        result = minimize(_rosenbrock, np.zeros(5), 0.5, max_evaluations=20000, seed=3)
        self.assertLess(result.fun, 1e-6)
        np.testing.assert_allclose(result.x, np.ones(5), atol=1e-3)

    def test_default_population(self):
        """Verify the default population equals 4 + floor(3 * ln(dim))."""
        for dim in (2, 5, 10, 15, 40):
            optimizer = CMAES(np.zeros(dim), 0.3, seed=0)
            self.assertEqual(optimizer.population, 4 + int(math.floor(3.0 * math.log(dim))))
            self.assertEqual(optimizer.ask().shape, (optimizer.population, dim))

    def test_callback_called_once_per_generation(self):
        """Verify the callback fires exactly once per generation with the best pair."""
        seen = []

        def callback(generation, best_x, best_f):
            seen.append((generation, np.asarray(best_x).copy(), float(best_f)))

        result = minimize(_sphere, np.full(4, 1.0), 0.3, max_evaluations=200, seed=0, callback=callback)
        self.assertEqual(len(seen), result.generations)
        self.assertEqual([entry[0] for entry in seen], list(range(1, result.generations + 1)))
        self.assertAlmostEqual(seen[-1][2], result.fun)


class TestCMAESBounds(unittest.TestCase):
    """Box-bound handling by reflection."""

    def test_candidates_stay_inside_box(self):
        """Verify every asked candidate stays inside the box on an out-of-box optimum."""
        lower = np.full(6, -1.0)
        upper = np.full(6, 1.0)
        optimizer = CMAES(np.zeros(6), 2.0, bounds=(lower, upper), seed=4)
        for _ in range(40):
            candidates = optimizer.ask()
            self.assertTrue(np.all(candidates >= lower - 0.0))
            self.assertTrue(np.all(candidates <= upper + 0.0))
            optimizer.tell(candidates, np.array([_sphere(c - 5.0) for c in candidates]))
            self.assertTrue(np.all(optimizer.mean >= lower))
            self.assertTrue(np.all(optimizer.mean <= upper))

    def test_solution_sits_on_expected_boundary(self):
        """Verify the returned solution reaches the box face nearest the outside optimum."""
        lower = np.full(4, -1.0)
        upper = np.full(4, 1.0)
        result = minimize(
            lambda x: _sphere(x - 3.0),
            np.zeros(4),
            0.3,
            bounds=(lower, upper),
            max_evaluations=3000,
            seed=2,
        )
        self.assertTrue(np.all(result.x >= lower))
        self.assertTrue(np.all(result.x <= upper))
        np.testing.assert_allclose(result.x, upper, atol=1e-4)
        self.assertAlmostEqual(result.fun, 16.0, places=4)

    def test_initial_mean_repaired_into_box(self):
        """Verify an out-of-box initial mean is folded back inside the box."""
        lower = np.zeros(3)
        upper = np.ones(3)
        optimizer = CMAES(np.array([1.25, -0.25, 0.5]), 0.1, bounds=(lower, upper), seed=0)
        np.testing.assert_allclose(optimizer.mean, [0.75, 0.25, 0.5])


class TestCMAESCheckpointing(unittest.TestCase):
    """Checkpoint round trips through JSON."""

    def test_state_restore_reproduces_next_generation(self):
        """Verify a restored optimizer asks the identical next generation."""
        optimizer = CMAES(np.full(7, 0.5), 0.4, bounds=(np.full(7, -2.0), np.full(7, 2.0)), seed=11)
        for _ in range(6):
            candidates = optimizer.ask()
            optimizer.tell(candidates, np.array([_rosenbrock(c) for c in candidates]))

        state = json.loads(json.dumps(optimizer.state()))
        restored = CMAES.restore(state)

        self.assertEqual(restored.generations, optimizer.generations)
        self.assertEqual(restored.evaluations, optimizer.evaluations)
        self.assertEqual(restored.best[1], optimizer.best[1])
        np.testing.assert_array_equal(restored.best[0], optimizer.best[0])
        np.testing.assert_array_equal(restored.ask(), optimizer.ask())

    def test_state_restore_matches_over_several_generations(self):
        """Verify a restored optimizer keeps matching the original for later generations."""
        original = CMAES(np.zeros(5), 0.3, seed=8)
        for _ in range(4):
            candidates = original.ask()
            original.tell(candidates, np.array([_sphere(c) for c in candidates]))
        restored = CMAES.restore(json.loads(json.dumps(original.state())))

        for _ in range(5):
            a = original.ask()
            b = restored.ask()
            np.testing.assert_array_equal(a, b)
            values = np.array([_sphere(c) for c in a])
            original.tell(a, values)
            restored.tell(b, values)
        np.testing.assert_array_equal(original.mean, restored.mean)
        self.assertEqual(original.sigma, restored.sigma)

    def test_state_is_json_serializable_with_unbounded_box(self):
        """Verify state() stays JSON-serializable when bounds are infinite."""
        optimizer = CMAES(np.zeros(3), 0.2, seed=0)
        text = json.dumps(optimizer.state())
        restored = CMAES.restore(json.loads(text))
        self.assertTrue(np.all(np.isinf(restored.lower)))
        self.assertTrue(np.all(np.isinf(restored.upper)))
        np.testing.assert_array_equal(restored.ask(), optimizer.ask())


class TestCMAESRobustness(unittest.TestCase):
    """Handling of non-finite objective values."""

    def test_non_finite_values_rank_worst(self):
        """Verify inf and nan candidates are ranked behind every finite candidate."""
        optimizer = CMAES(np.zeros(3), 0.5, population=6, seed=0)
        candidates = optimizer.ask()
        values = np.array([np.nan, 2.0, np.inf, 1.0, 0.5, 3.0])
        optimizer.tell(candidates, values)
        best_x, best_f = optimizer.best
        self.assertEqual(best_f, 0.5)
        np.testing.assert_array_equal(best_x, candidates[4])
        self.assertTrue(np.all(np.isfinite(optimizer.mean)))
        self.assertTrue(np.all(np.isfinite(optimizer.covariance)))

    def test_failed_evaluations_do_not_break_the_run(self):
        """Verify a run still converges when part of the domain returns inf or nan."""

        def flaky(x):
            if x[0] > 0.5:
                return math.inf
            if x[1] > 0.5:
                return math.nan
            return _sphere(x)

        result = minimize(flaky, np.full(4, 1.0), 0.5, max_evaluations=3000, seed=5)
        self.assertTrue(math.isfinite(result.fun))
        self.assertLess(result.fun, 1e-8)
        self.assertTrue(np.all(np.isfinite(result.x)))

    def test_all_values_non_finite_keeps_state_finite(self):
        """Verify a generation of only non-finite values leaves the optimizer usable."""
        optimizer = CMAES(np.zeros(4), 0.3, seed=1)
        candidates = optimizer.ask()
        optimizer.tell(candidates, np.full(optimizer.population, np.nan))
        self.assertTrue(math.isinf(optimizer.best[1]))
        self.assertTrue(np.all(np.isfinite(optimizer.mean)))
        self.assertTrue(np.all(np.isfinite(optimizer.ask())))


class TestCMAESDeterminism(unittest.TestCase):
    """Reproducibility of runs given a seed."""

    def test_same_seed_gives_identical_result(self):
        """Verify two runs with the same seed produce identical results."""
        first = minimize(_sphere, np.full(6, 0.8), 0.4, max_evaluations=600, seed=42)
        second = minimize(_sphere, np.full(6, 0.8), 0.4, max_evaluations=600, seed=42)
        np.testing.assert_array_equal(first.x, second.x)
        self.assertEqual(first.fun, second.fun)
        self.assertEqual(first.evaluations, second.evaluations)
        self.assertEqual(first.generations, second.generations)
        self.assertEqual(first.history, second.history)

    def test_different_seed_gives_different_path(self):
        """Verify a different seed follows a different search path."""
        first = minimize(_sphere, np.full(6, 0.8), 0.4, max_evaluations=600, seed=42)
        other = minimize(_sphere, np.full(6, 0.8), 0.4, max_evaluations=600, seed=43)
        self.assertNotEqual(first.history, other.history)
        self.assertFalse(np.array_equal(first.x, other.x))


if __name__ == "__main__":
    unittest.main()
