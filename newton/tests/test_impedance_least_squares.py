# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the Levenberg-Marquardt runner fit and its residual definition."""

import itertools
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.test_impedance_hogan import _chain, _tiny_shoe
from projects.impedance_instron.hogan import least_squares
from projects.impedance_instron.hogan.identify import ForceObservation, Parameterization, Trial, predict, score
from projects.impedance_instron.hogan.runner import RolloutConfig, Runner, State, Task, simulate


def _initial():
    return State(
        np.array([0.01, 0.8 * math.cos(0.12) + 0.101, math.pi / 2, 0.12, -0.24, 0.12]),
        np.array([0.12, -0.5, 0.08, 0.3, -0.25, 0.1]),
        phase_rad=2 * math.pi - 0.03,
    )


class TestLeastSquares(unittest.TestCase):
    """Keep LM residuals consistent with the reported score and fit on training data."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.shoe = _tiny_shoe(directory.name)
        self.config = RolloutConfig(dt_s=2.5e-4, contact_threshold_n=0.01)
        self.baseline = Runner.seed(reference_speed_m_s=0.3)
        self.parameters = Parameterization(self.baseline, [0.3])

    def trial(self, model=None, split="train", duration=0.008, observation=None):
        """Build a trial whose targets are an exact rollout of ``model`` on irregular clocks."""
        initial = _initial()
        time = duration * np.array([0.0, 0.127, 0.43, 0.79, 1.0])
        force_time = duration * np.array([0.0, 0.11, 0.39, 0.64, 0.96, 1.0])
        q, grf = np.tile(initial.q, (5, 1)), np.zeros((6, 2))
        if model is not None:
            trace, _ = simulate(model, _chain(), self.shoe, initial, Task(0.3), duration_s=duration, config=self.config)
            q = np.column_stack([np.interp(time, trace["time_s"], trace["state"][:, c]) for c in range(6)])
            grf = np.column_stack([np.interp(force_time, trace["time_s"][:-1], trace["grf_n"][:, c]) for c in range(2)])
        return Trial(
            split,
            split,
            _chain(),
            self.shoe,
            Task(0.3),
            initial,
            time,
            q,
            force_time,
            grf,
            {"compatibility": {"passed": True}},
            observation,
        )

    def test_residuals_match_score_sample_terms(self):
        """Sum squared residuals to the score's coordinate and GRF mean-square terms."""
        trial = self.trial()
        trial.q[1:] += np.array([0.003, -0.002, 0.01, -0.02, 0.015, 0.005])
        trial.grf_n[:, 1] += 30.0
        trace, summary = predict(self.baseline, trial, self.config)
        result = score(trace, summary, trial, self.baseline)
        tracking, force = np.asarray(result["tracking_rmse"]), np.asarray(result["grf_rmse_n"])
        expected = np.mean((tracking[:2] / 0.02) ** 2) + np.mean((tracking[2:] / 0.05) ** 2)
        expected += np.mean((force / 100.0) ** 2)
        r = least_squares.residuals(trace, summary, trial)
        self.assertAlmostEqual(float(r @ r), float(expected), places=10)
        self.assertIsNone(least_squares.residuals(trace, summary | {"status": "failed"}, trial))

    def test_score_objective_residuals_match_reported_loss(self):
        """Match every score term, including force summaries and bounded torque effort."""
        for observation in (None, ForceObservation(400.0, 1.0, 0.7)):
            with self.subTest(observation=observation):
                trial = self.trial(observation=observation)
                trial.q[1:] += np.array([0.003, -0.002, 0.01, -0.02, 0.015, 0.005])
                trial.grf_n[:, 1] += 30.0
                trace, summary = predict(self.baseline, trial, self.config)
                vector = least_squares.residuals(trace, summary, trial, self.baseline, objective="score")
                self.assertAlmostEqual(
                    float(vector @ vector), score(trace, summary, trial, self.baseline)["loss"], places=10
                )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_device_score_objective_matches_reported_loss(self):
        """Match CUDA score residual rows and sums for raw and observed force."""
        from projects.impedance_instron.hogan.gpu_residuals import GpuResiduals  # noqa: PLC0415
        from projects.impedance_instron.hogan.identify import predict_many  # noqa: PLC0415

        trials = [self.trial(), self.trial(observation=ForceObservation(400.0, 1.0, 0.7))]
        models = [self.baseline, self.parameters.model(np.full(self.parameters.size, 0.01))]
        predictions = predict_many(models, trials, self.config, device="cuda:0")
        objective = GpuResiduals(trials, self.config, rows=2, chunk=2, objective="score")
        completed, sums, _ = objective.evaluate(models, [0, 1])
        self.assertTrue(completed.all())
        rows = objective.store.numpy()[:, : objective.length]
        for index, (model, predictions_for_model) in enumerate(zip(models, predictions, strict=True)):
            expected = np.concatenate(
                [
                    least_squares.residuals(*pair, trial, model, objective="score") / math.sqrt(len(trials))
                    for trial, pair in zip(trials, predictions_for_model, strict=True)
                ]
            )
            np.testing.assert_allclose(rows[index], expected, rtol=1e-10, atol=1e-10)
            loss = np.mean(
                [score(*pair, trial, model)["loss"] for trial, pair in zip(trials, predictions_for_model, strict=True)]
            )
            self.assertAlmostEqual(sums[index], loss, delta=1e-9 * max(1.0, loss))

    def test_force_observation_reproduces_target_processing(self):
        """Filter with unit gain, zero lag, and -3 dB at the nominal two-pass cutoff, then gate both axes."""
        self.assertEqual(ForceObservation().cutoff_hz, 20.0)
        dt = 1e-3
        for cutoff in (20.0, 6.0):
            with self.subTest(cutoff_hz=cutoff):
                observation = ForceObservation(cutoff_hz=cutoff, gate_n=1.0)
                b0, b1, b2, a1, a2 = observation.coefficients(dt)

                def gain(frequency_hz, b0=b0, b1=b1, b2=b2, a1=a1, a2=a2):
                    z = np.exp(-2j * math.pi * frequency_hz * dt)
                    return abs((b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)) ** 2

                self.assertAlmostEqual(gain(0.0), 1.0, places=12)
                self.assertAlmostEqual(gain(cutoff), math.sqrt(0.5), places=3)
                # A half-sine stance padded by flight keeps its timing through the zero-lag observation.
                time = np.arange(600) * dt
                stance = (time > 0.2) & (time < 0.4)
                fz = np.where(stance, 1200.0 * np.sin(np.pi * (time - 0.2) / 0.2), 0.0)
                observed = observation.apply(np.column_stack((-0.2 * fz, fz)), dt)
                self.assertLess(abs(int(np.argmax(observed[:, 1])) - 300), 2)
                # A steady ripple at 3.5 x the cutoff keeps the two-pass gain |H|^2 of its amplitude, under 2 %.
                long_time = np.arange(2000) * dt
                ripple = 1000.0 + 80.0 * np.sin(2 * np.pi * 3.5 * cutoff * long_time)
                middle = observation.apply(np.column_stack((0.0 * ripple, ripple)), dt)[500:1500, 1]
                self.assertLess(gain(3.5 * cutoff), 0.02)
                self.assertAlmostEqual(
                    np.max(np.abs(middle - 1000.0)), 80.0 * gain(3.5 * cutoff), delta=0.02 * 80.0 * gain(3.5 * cutoff)
                )
                # Zero-lag smoothing loads before contact and unloads after it, and the gate zeroes both axes together.
                loaded = np.flatnonzero(observed[:, 1] > 0.0)
                self.assertLess(loaded[0], 200)
                self.assertGreater(loaded[-1], 400)
                np.testing.assert_array_equal(observed[observed[:, 1] < 1.0], 0.0)
        np.testing.assert_array_equal(ForceObservation().apply(np.zeros((5, 2)), dt), 0.0)
        for build in (lambda: ForceObservation(0.0), lambda: ForceObservation(20.0, -1.0)):
            with self.assertRaises(ValueError):
                build()
        with self.assertRaises(ValueError):
            ForceObservation().coefficients(0.1)

    def test_observed_residuals_match_observed_score_terms(self):
        """Compare observed simulated GRF in both the LM residual and the score, plus its weighted vibration."""
        observation = ForceObservation(cutoff_hz=400.0, gate_n=0.0, vibration_weight=0.7)
        trial = self.trial(observation=observation)
        trial.grf_n[:, 1] += 30.0
        trace, summary = predict(self.baseline, trial, self.config)
        result = score(trace, summary, trial, self.baseline)
        tracking, force = np.asarray(result["tracking_rmse"]), np.asarray(result["grf_rmse_n"])
        vibration = np.asarray(result["grf_vibration_rmse_n"])
        expected = np.mean((tracking[:2] / 0.02) ** 2) + np.mean((tracking[2:] / 0.05) ** 2)
        expected += np.mean((force / 100.0) ** 2) + np.mean((0.7 * vibration / 100.0) ** 2)
        r = least_squares.residuals(trace, summary, trial)
        self.assertAlmostEqual(float(r @ r), float(expected), places=10)
        steps = len(trace["grf_n"])
        observed = observation.apply(trace["grf_n"], summary["dt_s"])
        self.assertGreater(vibration.max(), 0.0)
        np.testing.assert_allclose(vibration, np.sqrt(np.mean((trace["grf_n"] - observed) ** 2, axis=0)))
        # The GRF block compares the observed force; the trailing block is the weighted vibration.
        raw = least_squares.residuals(trace, summary, self.trial())
        self.assertEqual(len(r), len(raw) + 2 * steps)
        self.assertGreater(np.max(np.abs(r[24 : 24 + 2 * steps] - raw[24:])), 1e-6)
        unweighted = self.trial(observation=ForceObservation(400.0, 0.0, 0.0))
        self.assertEqual(len(least_squares.residuals(trace, summary, unweighted)), len(raw))
        self.assertAlmostEqual(result["peak_fz_error_n"], float(observed[:, 1].max()) - 30.0, places=10)
        # Physical diagnostics stay on the raw contact force.
        self.assertEqual(result["peak_grf_n"], trace["grf_n"].max(0).tolist())

    def test_lm_reduces_cost_toward_known_truth(self):
        """Decrease the training cost monotonically when fitting an exact truth rollout."""
        truth = self.parameters.model(np.random.default_rng(6).normal(0, 0.05, self.parameters.size))
        search = least_squares.LMConfig(iterations=3, regularization=0.0)
        learned, report = least_squares.fit_lm(self.baseline, [self.trial(truth)], config=self.config, search=search)
        costs = [report["initial_cost"]] + [row["cost"] for row in report["history"]]
        self.assertTrue(all(b <= a for a, b in itertools.pairwise(costs)))
        self.assertLess(report["final_cost"], 0.5 * report["initial_cost"])
        self.assertTrue(any(row["accepted"] for row in report["history"]))
        self.assertEqual(report["selection_split"], "train")
        self.assertIsInstance(learned, Runner)

    def test_incompatible_data_is_not_silently_fitted(self):
        """Refuse LM fitting when input compatibility fails without an explicit override."""
        trial = self.trial()
        trial.provenance["compatibility"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "compatibility"):
            least_squares.fit_lm(self.baseline, [trial], config=self.config)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_gpu_rollouts_match_cpu_residuals(self):
        """Match CPU residual vectors from persistent padded CUDA batches."""
        trials = [self.trial(), self.trial(split="train")]
        offsets = np.random.default_rng(7).normal(0, 0.02, (3, self.parameters.size))
        cpu = least_squares._Rollouts(self.parameters, trials, self.config, "cpu")(offsets, 2)
        gpu = least_squares._Rollouts(self.parameters, trials, self.config, "cuda:0")(offsets, 2)
        for a, b in zip(cpu, gpu, strict=True):
            # The unchanged shoe law runs in float32 on both backends.
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_device_objective_matches_host_residuals_and_normal_equations(self):
        """Write CPU-equivalent residual rows, sums, and J^T J / J^T r entirely on CUDA."""
        from projects.impedance_instron.hogan.gpu_residuals import GpuResiduals  # noqa: PLC0415

        trials = [self.trial(), self.trial(split="train", duration=0.0061)]
        offsets = np.random.default_rng(7).normal(0, 0.02, (3, self.parameters.size))
        models = [self.parameters.model(x) for x in offsets]
        gpu = least_squares._Rollouts(self.parameters, trials, self.config, "cuda:0")(offsets, 3)
        cpu = least_squares._Rollouts(self.parameters, trials, self.config, "cpu")(offsets, 3)
        # Two candidate slots exercise padding and a second chunk through one captured graph.
        objective = GpuResiduals(trials, self.config, rows=6, chunk=2)
        completed, sums, motion = objective.evaluate(models, [0, 1, 5], metrics=True)
        self.assertTrue(completed.all())
        self.assertEqual(objective.length, len(gpu[0]))
        rows = objective.store.numpy()[:, : objective.length]
        for row, expected, host, total in zip((0, 1, 5), gpu, cpu, sums, strict=True):
            # Device residuals use the identical CUDA trajectory, so only rounding differs.
            np.testing.assert_allclose(rows[row], expected, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(rows[row], host, rtol=1e-4, atol=1e-5)
            self.assertAlmostEqual(total, float(expected @ expected), delta=1e-12 * max(1.0, total))
        self.assertEqual(len(motion), 3)
        self.assertEqual({m["status"] for m in motion[0]}, {"completed"})
        jacobian = np.array([(gpu[i] - gpu[2]) / 0.01 for i in range(2)])
        normal, gradient = objective.normal([0, 1], [5, 5], [0.01, 0.01], 5)
        np.testing.assert_allclose(normal, jacobian @ jacobian.T, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(gradient, jacobian @ gpu[2], rtol=1e-10, atol=1e-12)
        objective.copy_row(5, 3)
        np.testing.assert_array_equal(objective.store.numpy()[3], objective.store.numpy()[5])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_device_objective_matches_host_observed_residuals(self):
        """Filter, gate, and score the observed GRF and its vibration on CUDA exactly like the host residuals."""
        from projects.impedance_instron.hogan.gpu_residuals import GpuResiduals  # noqa: PLC0415
        from projects.impedance_instron.hogan.identify import predict_many  # noqa: PLC0415

        offsets = np.random.default_rng(9).normal(0, 0.02, (3, self.parameters.size))
        models = [self.parameters.model(x) for x in offsets]
        trace, summary = predict_many(models[:1], [self.trial()], self.config, device="cuda:0")[0][0]
        # Gate half of the peak so the device gate and its zeroed samples are exercised.
        gate = 0.5 * float(ForceObservation(400.0, 0.0).apply(trace["grf_n"], summary["dt_s"])[:, 1].max())
        self.assertGreater(gate, 0.0)
        # Observed with and without a vibration block, and raw, share one batch.
        trials = [
            self.trial(observation=ForceObservation(400.0, gate, 0.7)),
            self.trial(split="train", duration=0.0061),
            self.trial(split="train", duration=0.0071, observation=ForceObservation(300.0, 0.0, 0.0)),
        ]
        expected = least_squares._Rollouts(self.parameters, trials, self.config, "cuda:0")(offsets, 3)
        objective = GpuResiduals(trials, self.config, rows=4, chunk=2)
        self.assertEqual(objective.length, len(expected[0]))
        completed, sums, motion = objective.evaluate(models, [0, 1, 3], metrics=True)
        self.assertTrue(completed.all())
        rows = objective.store.numpy()[:, : objective.length]
        for row, vector, total in zip((0, 1, 3), expected, sums, strict=True):
            np.testing.assert_allclose(rows[row], vector, rtol=1e-12, atol=1e-15)
            self.assertAlmostEqual(total, float(vector @ vector), delta=1e-12 * max(1.0, total))
        raw_trials = [self.trial(), *trials[1:2], self.trial(split="train", duration=0.0071)]
        raw = least_squares._Rollouts(self.parameters, raw_trials, self.config, "cuda:0")(offsets, 3)
        steps = len(trace["grf_n"])
        self.assertEqual(objective.offsets[1], 24 + 4 * steps)
        self.assertGreater(np.max(np.abs(expected[0][24 : 24 + 2 * steps] - raw[0][24 : 24 + 2 * steps])), 1e-6)
        self.assertGreater(np.max(np.abs(rows[0][24 + 2 * steps : objective.offsets[1]])), 0.0)
        self.assertEqual({m["status"] for m in motion[0]}, {"completed"})

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_fast_jacobian_redoes_differences_exactly_when_its_reference_fails(self):
        """Fall back to exact finite differences in the same batch when the fast reference rollout fails."""
        trials = [self.trial(), self.trial(split="train", duration=0.0061)]
        x = np.random.default_rng(3).normal(0, 0.02, self.parameters.size)
        engines = [
            least_squares._DeviceEngine(
                self.parameters, trials, self.config, least_squares.LMConfig(fast_jacobian=fast), "cuda:0"
            )
            for fast in (False, True)
        ]
        for engine in engines:
            self.assertIsNotNone(engine.start(x))
        objective = engines[1].objective
        evaluate, modes = objective.evaluate, []

        def failing_reference(models, rows, *, metrics=False, fast=False):
            modes.append((len(models), fast))
            completed, sums, motion = evaluate(models, rows, metrics=metrics, fast=fast)
            if fast:
                completed[-1] = False
            return completed, sums, motion

        with patch.object(objective, "evaluate", side_effect=failing_reference):
            fallback = engines[1].jacobian(x)
        n = self.parameters.size + 1
        self.assertEqual(modes, [(n, True), (n, False)])
        self.assertEqual({capacity for _, capacity in objective._groups}, {1, min(n, objective.chunk)})
        for actual, expected in zip(fallback, engines[0].jacobian(x), strict=True):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
