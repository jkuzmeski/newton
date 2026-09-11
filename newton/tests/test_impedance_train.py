# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PPO trainer of the residual impedance policy.

The tests never touch the physics environment. They drive the trainer with a
deterministic linear-quadratic toy whose optimal action is known in closed
form, so a failure points at the learning algorithm and not at the rig.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from projects.impedance_instron import train as train_module
from projects.impedance_instron.objective import Verdict

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_ONNX = importlib.util.find_spec("onnx") is not None

EPISODE_FRAMES = 45
"""Fixed episode length of the impedance environment, mirrored by the toy."""

TOY_GAIN = 0.5
"""Linear map from observation to the toy's optimal action."""


class ToyEnv:
    """Deterministic linear-quadratic stand-in for ``ImpedanceEnv``.

    The first three observation channels are drawn from a standard normal and
    the fourth is the episode phase, as in the rig, where the critic needs to
    know how much of stance remains. The reward is
    ``-||a - TOY_GAIN * state||^2`` and the episode has the same fixed length
    as the rig. The optimal policy is therefore the linear map
    ``a = TOY_GAIN * state`` with an optimal return of exactly zero, which
    makes the learning test a statement about PPO rather than about a tuned
    baseline.
    """

    STATE_DIM = 3
    """Width of the random part of the observation."""

    def __init__(
        self,
        num_worlds: int,
        args=None,
        nominal: np.ndarray | None = None,
        seed: int = 0,
        shape_reward: bool = True,
        ragged: bool = False,
        verdict: Verdict | None = None,
    ):
        """Mirror the signature of the real environment.

        Args:
            num_worlds: Environments stepped in lockstep.
            args: Ignored; present for API compatibility.
            nominal: Ignored; present for API compatibility.
            seed: Seed re-applied on every reset, so episodes repeat exactly.
            shape_reward: Ignored; present for API compatibility.
            ragged: Report a single world done one frame early, to exercise the
                trainer's synchronized-episode assertion.
            verdict: Scored outcome attached to every world on the done frame,
                as the rig does; ``None`` leaves the terminal info bare.
        """
        self.num_worlds = int(num_worlds)
        self.seed = int(seed)
        self.ragged = bool(ragged)
        self.verdict = verdict
        self.materials: list | None = None
        self._frame = 0
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros((self.num_worlds, self.STATE_DIM), dtype=np.float32)

    @property
    def observation_dim(self) -> int:
        """Observation width: the random state plus the episode phase."""
        return self.STATE_DIM + 1

    def _observation(self) -> np.ndarray:
        """Return the current observation, shape [num_worlds, obs]."""
        phase = np.full((self.num_worlds, 1), self._frame / EPISODE_FRAMES, dtype=np.float32)
        return np.concatenate([self._state, phase], axis=1)

    @property
    def action_dim(self) -> int:
        """Action width of the residual impedance command."""
        return 3

    @property
    def episode_frames(self) -> int:
        """Frames per episode."""
        return EPISODE_FRAMES

    def reset(self) -> np.ndarray:
        """Restart every world from the same seeded observation sequence."""
        self._rng = np.random.default_rng(self.seed)
        self._frame = 0
        self._state = self._rng.standard_normal((self.num_worlds, self.STATE_DIM)).astype(np.float32)
        return self._observation()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Score the action against the known optimum and draw a new state.

        Args:
            actions: Residual actions, shape [num_worlds, 3].
        """
        actions = np.asarray(actions, dtype=np.float32)
        target = TOY_GAIN * self._state
        reward = -np.sum((actions - target) ** 2, axis=1).astype(np.float32)
        self._frame += 1
        done = np.zeros(self.num_worlds, dtype=bool)
        if self._frame >= EPISODE_FRAMES:
            done[:] = True
        elif self.ragged and self._frame == EPISODE_FRAMES - 1:
            done[0] = True
        self._state = self._rng.standard_normal((self.num_worlds, self.STATE_DIM)).astype(np.float32)
        info: dict = {"frame": self._frame}
        if done.all() and self.verdict is not None:
            info["verdicts"] = [self.verdict] * self.num_worlds
            info["objective_j"] = np.full(self.num_worlds, self.verdict.objective_j, dtype=np.float32)
            info["feasible"] = np.full(self.num_worlds, self.verdict.feasible, dtype=bool)
            info["on_task"] = np.full(self.num_worlds, self.verdict.on_task, dtype=bool)
        return self._observation(), reward, done, info

    def set_world_materials(self, materials: list) -> None:
        """Record the requested per-world materials.

        Args:
            materials: One material description per world.
        """
        self.materials = list(materials)

    def realised_command(self, world: int) -> dict:
        """Return the command a world realised.

        Args:
            world: World index.
        """
        return {"world": int(world), "parameters": []}


# Trace columns of the rig, mirrored here so `train.evaluation_physics` reads
# them off this module exactly as it reads them off the real environment.
TRACE_SHOE_FZ = 0
TRACE_SHOE_FX = 1
TRACE_ANKLE_Z = 2
TRACE_ANKLE_VZ = 3
TRACE_UPPER_VZ = 5
TRACE_LEG_LENGTH = 10
TRACE_COMPRESSION = 12
TRACE_COLUMNS = 15

RIG_SUBSTEPS = 4
"""Substeps per frame of the rig-shaped toy, kept small so the tests stay fast."""

RIG_STEP_S = 1.0e-3
"""Substep of the rig-shaped toy [s]."""

RIG_CONTACT = (20, 160)
"""First and last sample of the toy's half-sine contact, before thresholding."""

RIG_PEAK_FZ_N = 1050.0
"""Peak simulated vertical force of the toy [N]."""

RIG_REFERENCE_PEAK_FZ_N = 1000.0
"""Peak measured vertical force of the toy [N]."""

RIG_VZ_OFFSET_M_S = 0.1
"""Constant COM vertical velocity error built into the toy [m/s]."""

RIG_PEAK_COMPRESSION_M = 0.012
"""Peak foam compression of the toy [m]."""


class RigToyEnv(ToyEnv):
    """Toy environment that also exposes rig-shaped traces and reference rows.

    The waveforms are analytic: the simulated and the measured vertical force
    are the same half-sine scaled by 1.05, the COM vertical velocity is the
    measured one plus a constant offset, and the compression peaks at a known
    depth. Every physical metric therefore has a closed-form value that does
    not depend on the code under test.
    """

    def __init__(self, num_worlds: int, **kwargs):
        """Build the analytic trace and reference rows once.

        Args:
            num_worlds: Environments stepped in lockstep.
            **kwargs: Forwarded to :class:`ToyEnv`.
        """
        super().__init__(num_worlds, **kwargs)
        samples = EPISODE_FRAMES * RIG_SUBSTEPS + 1
        self.substeps = RIG_SUBSTEPS
        self.sample_count = samples
        self.times = np.arange(samples) * RIG_STEP_S
        self.mass = 80.0
        self.foot_mass = 8.0
        self.com_mass = self.mass - self.foot_mass
        self.body_weight_n = self.mass * 9.81

        start, end = RIG_CONTACT
        shape = np.zeros(samples)
        index = np.arange(samples)
        inside = (index >= start) & (index <= end)
        shape[inside] = np.sin(np.pi * (index[inside] - start) / (end - start))
        reference_com_vz = -1.5 + 0.02 * index
        com_vz = reference_com_vz + RIG_VZ_OFFSET_M_S
        reference_com_z = 1.0 + np.concatenate(
            [[0.0], np.cumsum(0.5 * (reference_com_vz[1:] + reference_com_vz[:-1]) * np.diff(self.times))]
        )

        self.shape = shape
        self.reference = np.zeros((samples, 28))
        self.reference[:, 1] = reference_com_z
        self.reference[:, 4] = reference_com_z
        self.reference[:, 9] = reference_com_vz
        self.reference[:, 11] = RIG_REFERENCE_PEAK_FZ_N * shape
        self.reference[:, 22] = -0.2 * RIG_REFERENCE_PEAK_FZ_N * shape

        self._trace = np.zeros((samples, TRACE_COLUMNS))
        self._trace[:, TRACE_SHOE_FZ] = RIG_PEAK_FZ_N * shape
        self._trace[:, TRACE_SHOE_FX] = -0.2 * RIG_PEAK_FZ_N * shape
        self._trace[:, TRACE_ANKLE_Z] = 0.1 + 0.01 * shape
        self._trace[:, TRACE_ANKLE_VZ] = com_vz
        self._trace[:, TRACE_UPPER_VZ] = com_vz
        self._trace[:, TRACE_LEG_LENGTH] = 1.0 - 0.02 * shape
        self._trace[:, TRACE_COMPRESSION] = RIG_PEAK_COMPRESSION_M * shape

    def trace(self, world: int) -> np.ndarray:
        """Return the analytic substep trace.

        Args:
            world: World index; every world runs the same analytic waveform.
        """
        return self._trace.copy()

    def realised_command(self, world: int) -> dict:
        """Return the impedance the analytic episode was driven with.

        Args:
            world: World index.
        """
        return {
            "time_s": self.times.copy(),
            "length_m": 1.0 - 0.01 * self.shape,
            "stiffness_n_m": np.full(self.times.size, 20000.0),
            "damping_ratio": np.full(self.times.size, 0.6),
        }


def _rig_window() -> tuple[np.ndarray, np.ndarray]:
    """Return the thresholded contact indices and the toy's half-sine shape.

    The contact gate is 2 % of body weight, so the two samples where the
    half-sine is still below 15.7 N fall outside the window: contact runs from
    sample 21 to sample 159 inclusive.
    """
    env = RigToyEnv(1)
    loaded = np.nonzero(env._trace[:, TRACE_SHOE_FZ] > 0.02 * env.body_weight_n)[0]
    return loaded, env.shape


def _toy_config(**overrides):
    """Return a small, fast PPO configuration for the toy problem.

    Args:
        **overrides: Fields of :class:`~projects.impedance_instron.train.PPOConfig` to replace.
    """
    settings = {
        "num_worlds": 16,
        "iterations": 60,
        "learning_rate": 3.0e-3,
        "clip": 0.2,
        "entropy": 0.0,
        "gamma": 0.99,
        "advantage_lambda": 0.95,
        "epochs": 8,
        "minibatches": 4,
        "seed": 0,
        "hidden_size": 64,
        "device": "cpu",
        "eval_interval": 0,
    }
    settings.update(overrides)
    return train_module.PPOConfig(**settings)


def _toy_trainer(verdict: Verdict | None = None, env_class=None, **overrides):
    """Build a trainer over the toy environment.

    Args:
        verdict: Scored outcome the environment attaches on the done frame.
        env_class: Environment class to instantiate; defaults to :class:`ToyEnv`.
        **overrides: Configuration fields to replace.
    """
    config = _toy_config(**overrides)
    env = (env_class or ToyEnv)(config.num_worlds, seed=config.seed, verdict=verdict)
    return train_module.PPOTrainer(env, config, np.arange(15, dtype=float), {"shoe": "toy"})


class TestGeneralizedAdvantage(unittest.TestCase):
    """Check the advantage estimator against hand-computed values."""

    def test_advantage_matches_hand_computation(self):
        """Reproduce a hand-computed advantage trace on a short reward sequence."""
        rewards = np.array([[1.0], [1.0], [1.0]])
        values = np.array([[0.5], [0.5], [0.5]])
        dones = np.array([[0.0], [0.0], [1.0]])
        gamma, lam = 0.9, 0.8
        advantages, returns = train_module.compute_advantages(rewards, values, dones, np.array([0.5]), gamma, lam)
        delta_2 = 1.0 - 0.5
        delta_1 = 1.0 + gamma * 0.5 - 0.5
        delta_0 = 1.0 + gamma * 0.5 - 0.5
        expected_2 = delta_2
        expected_1 = delta_1 + gamma * lam * expected_2
        expected_0 = delta_0 + gamma * lam * expected_1
        np.testing.assert_allclose(advantages[:, 0], [expected_0, expected_1, expected_2], rtol=1e-12)
        np.testing.assert_allclose(returns[:, 0], np.array([expected_0, expected_1, expected_2]) + 0.5, rtol=1e-12)

    def test_advantage_truncates_at_a_terminal_step(self):
        """Stop the trace at a terminal step instead of leaking across episodes."""
        rewards = np.array([[2.0], [3.0]])
        values = np.array([[1.0], [1.0]])
        dones = np.array([[1.0], [1.0]])
        advantages, _ = train_module.compute_advantages(rewards, values, dones, np.array([7.0]), 0.99, 0.95)
        np.testing.assert_allclose(advantages[:, 0], [1.0, 2.0], rtol=1e-12)

    def test_advantage_handles_many_worlds_independently(self):
        """Keep worlds independent when the batch has more than one column."""
        rewards = np.array([[1.0, -1.0], [1.0, -1.0]])
        values = np.zeros((2, 2))
        dones = np.array([[0.0, 0.0], [1.0, 1.0]])
        advantages, _ = train_module.compute_advantages(rewards, values, dones, np.zeros(2), 0.5, 1.0)
        np.testing.assert_allclose(advantages[:, 0], [1.5, 1.0], rtol=1e-12)
        np.testing.assert_allclose(advantages[:, 1], [-1.5, -1.0], rtol=1e-12)


class TestObservationNormalization(unittest.TestCase):
    """Check the running observation statistics and their freezing."""

    def test_running_statistics_converge(self):
        """Converge to the true mean and standard deviation of the stream."""
        rng = np.random.default_rng(0)
        truth_mean = np.array([2.0, -5.0, 0.25])
        truth_std = np.array([0.5, 3.0, 10.0])
        normalizer = train_module.RunningNormalizer(3)
        for _ in range(200):
            batch = truth_mean + truth_std * rng.standard_normal((256, 3))
            normalizer.update(batch)
        np.testing.assert_allclose(normalizer.mean, truth_mean, atol=0.02)
        np.testing.assert_allclose(normalizer.std, truth_std, rtol=0.02)

    def test_normalization_whitens_observations(self):
        """Whiten a batch to near-zero mean and unit standard deviation."""
        rng = np.random.default_rng(1)
        normalizer = train_module.RunningNormalizer(3)
        batch = 4.0 + 2.0 * rng.standard_normal((4096, 3))
        normalizer.update(batch)
        whitened = normalizer.normalize(batch)
        np.testing.assert_allclose(whitened.mean(axis=0), np.zeros(3), atol=1e-3)
        np.testing.assert_allclose(whitened.std(axis=0), np.ones(3), rtol=1e-2)

    def test_statistics_are_frozen_in_evaluation_mode(self):
        """Ignore new observations once the statistics are frozen."""
        normalizer = train_module.RunningNormalizer(2)
        stream = np.array([1.0, 2.0]) + np.linspace(-1.0, 1.0, 64)[:, None]
        normalizer.update(stream)
        frozen_mean = normalizer.mean.copy()
        frozen_count = normalizer.count
        normalizer.eval()
        normalizer.update(np.full((1000, 2), 100.0))
        np.testing.assert_allclose(normalizer.mean, frozen_mean, rtol=0.0, atol=0.0)
        self.assertEqual(normalizer.count, frozen_count)
        normalizer.train()
        normalizer.update(np.full((1000, 2), 100.0))
        self.assertGreater(normalizer.count, frozen_count)

    def test_normalizer_state_round_trips(self):
        """Rebuild identical statistics from the serialized state."""
        rng = np.random.default_rng(2)
        normalizer = train_module.RunningNormalizer(4)
        normalizer.update(rng.standard_normal((512, 4)) * 3.0 - 1.0)
        restored = train_module.RunningNormalizer.from_state(normalizer.state())
        np.testing.assert_array_equal(restored.mean, normalizer.mean)
        np.testing.assert_array_equal(restored.std, normalizer.std)
        self.assertEqual(restored.count, normalizer.count)


class TestSynchronizedEpisodes(unittest.TestCase):
    """Check that ragged episode ends are rejected, not silently averaged."""

    def test_early_termination_raises(self):
        """Reject a world that finishes before the final frame."""
        dones = np.zeros(4, dtype=bool)
        dones[2] = True
        with self.assertRaises(AssertionError):
            train_module._assert_synchronized(dones, 10, EPISODE_FRAMES)

    def test_unfinished_world_at_the_final_frame_raises(self):
        """Reject a world that is still running at the final frame."""
        dones = np.ones(4, dtype=bool)
        dones[1] = False
        with self.assertRaises(AssertionError):
            train_module._assert_synchronized(dones, EPISODE_FRAMES - 1, EPISODE_FRAMES)

    def test_synchronized_episode_is_accepted(self):
        """Accept a rollout whose worlds all finish on the final frame."""
        train_module._assert_synchronized(np.zeros(4, dtype=bool), 0, EPISODE_FRAMES)
        train_module._assert_synchronized(np.ones(4, dtype=bool), EPISODE_FRAMES - 1, EPISODE_FRAMES)


class TestExplainedVariance(unittest.TestCase):
    """Check the critic diagnostic used in the iteration report."""

    def test_perfect_prediction_explains_all_variance(self):
        """Report one for a critic that reproduces the targets."""
        targets = np.array([1.0, -2.0, 4.0, 0.5])
        self.assertAlmostEqual(train_module.explained_variance(targets, targets), 1.0, places=12)

    def test_constant_prediction_explains_nothing(self):
        """Report zero for a critic that only predicts the target mean."""
        targets = np.array([1.0, -2.0, 4.0, 0.5])
        predictions = np.full_like(targets, targets.mean())
        self.assertAlmostEqual(train_module.explained_variance(predictions, targets), 0.0, places=12)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestClippedSurrogate(unittest.TestCase):
    """Check the clipped policy objective at and outside the trust region."""

    def setUp(self):
        """Import torch once for the surrogate tests."""
        import torch

        self.torch = torch

    def test_unit_ratio_equals_the_unclipped_loss(self):
        """Reduce to the plain policy-gradient loss when the ratio is one."""
        advantages = self.torch.tensor([1.5, -2.0, 0.0, 7.25])
        ratio = self.torch.ones_like(advantages)
        loss = train_module.clipped_surrogate(ratio, advantages, 0.2)
        self.torch.testing.assert_close(loss, -advantages)

    def test_positive_advantage_clips_above_the_trust_region(self):
        """Cap the gain of a positive advantage at the upper clip."""
        advantages = self.torch.tensor([2.0])
        loss = train_module.clipped_surrogate(self.torch.tensor([1.5]), advantages, 0.2)
        self.torch.testing.assert_close(loss, -self.torch.tensor([1.2 * 2.0]))

    def test_negative_advantage_clips_below_the_trust_region(self):
        """Cap the gain of a negative advantage at the lower clip."""
        advantages = self.torch.tensor([-2.0])
        loss = train_module.clipped_surrogate(self.torch.tensor([0.5]), advantages, 0.2)
        self.torch.testing.assert_close(loss, -self.torch.tensor([0.8 * -2.0]))

    def test_ratio_inside_the_trust_region_is_not_clipped(self):
        """Leave ratios inside the trust region untouched."""
        advantages = self.torch.tensor([2.0, -2.0])
        ratio = self.torch.tensor([1.1, 0.9])
        loss = train_module.clipped_surrogate(ratio, advantages, 0.2)
        self.torch.testing.assert_close(loss, -ratio * advantages)

    def test_value_loss_clips_large_value_moves(self):
        """Penalize a value step beyond the value trust region."""
        values = self.torch.tensor([5.0])
        old_values = self.torch.tensor([0.0])
        returns = self.torch.tensor([0.0])
        clipped = train_module.value_loss(values, old_values, returns, 0.2)
        unclipped = train_module.value_loss(values, old_values, returns, None)
        self.torch.testing.assert_close(clipped, unclipped)
        equal = train_module.value_loss(old_values, old_values, returns, 0.2)
        self.torch.testing.assert_close(equal, self.torch.tensor([0.0]))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestRolloutCollection(unittest.TestCase):
    """Check rollout bookkeeping against the fixed-length environment."""

    def test_rollout_shapes_and_returns(self):
        """Collect exactly worlds x frames transitions with matching returns."""
        trainer = _toy_trainer(num_worlds=4)
        rollout = trainer.collect()
        self.assertEqual(rollout.observations.shape, (EPISODE_FRAMES, 4, ToyEnv(1).observation_dim))
        self.assertEqual(rollout.actions.shape, (EPISODE_FRAMES, 4, 3))
        self.assertEqual(rollout.rewards.shape, (EPISODE_FRAMES, 4))
        np.testing.assert_allclose(rollout.episode_returns, rollout.rewards.sum(axis=0), rtol=1e-6)
        self.assertTrue(rollout.dones[-1].all())
        self.assertFalse(rollout.dones[:-1].any())

    def test_ragged_environment_trips_the_assertion(self):
        """Fail the rollout when the environment ends episodes raggedly."""
        config = _toy_config(num_worlds=4)
        env = ToyEnv(config.num_worlds, seed=config.seed, ragged=True)
        trainer = train_module.PPOTrainer(env, config, np.zeros(15))
        with self.assertRaises(AssertionError):
            trainer.collect()

    def test_evaluation_freezes_the_normalizer(self):
        """Leave the observation statistics unchanged during evaluation."""
        trainer = _toy_trainer(num_worlds=4)
        trainer.collect()
        mean = trainer.normalizer.mean.copy()
        count = trainer.normalizer.count
        trainer.evaluate(episodes=2)
        np.testing.assert_array_equal(trainer.normalizer.mean, mean)
        self.assertEqual(trainer.normalizer.count, count)
        self.assertTrue(trainer.normalizer.training)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestDeterminism(unittest.TestCase):
    """Check that a seed pins the whole first iteration."""

    def test_same_seed_reproduces_the_first_iteration(self):
        """Reproduce identical first-iteration losses for a repeated seed."""
        first = _toy_trainer(num_worlds=4, seed=7).step_iteration()
        second = _toy_trainer(num_worlds=4, seed=7).step_iteration()
        self.assertEqual(first.policy_loss, second.policy_loss)
        self.assertEqual(first.value_loss, second.value_loss)
        self.assertEqual(first.entropy, second.entropy)
        self.assertEqual(first.return_mean, second.return_mean)

    def test_different_seeds_differ(self):
        """Produce a different first iteration for a different seed."""
        first = _toy_trainer(num_worlds=4, seed=7).step_iteration()
        other = _toy_trainer(num_worlds=4, seed=8).step_iteration()
        self.assertNotEqual(first.policy_loss, other.policy_loss)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestCheckpointRoundTrip(unittest.TestCase):
    """Check that a checkpoint is enough to deploy a frozen policy."""

    def test_round_trip_reproduces_deterministic_actions(self):
        """Reproduce identical deterministic actions from the file alone."""
        trainer = _toy_trainer(num_worlds=4)
        trainer.step_iteration()
        width = ToyEnv(1).observation_dim
        observations = np.random.default_rng(3).standard_normal((11, width)).astype(np.float32) * 2.0 + 1.0
        with trainer.torch.no_grad():
            expected = trainer.policy(trainer._tensor(trainer.normalizer.normalize(observations))).cpu().numpy()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pt"
            trainer.save(path)
            # A fresh process only has the file, so the reload must not touch
            # the trainer, the environment, or the optimizer state.
            policy = train_module.load_policy(path)
            np.testing.assert_allclose(policy.act(observations), expected, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(policy(observations), expected, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(policy.nominal, trainer.nominal, rtol=0.0, atol=0.0)
            self.assertEqual(policy.env_config, {"shoe": "toy"})
            repeated = train_module.load_policy(path)
            np.testing.assert_array_equal(repeated.act(observations), policy.act(observations))

    def test_checkpoint_carries_the_normalization_statistics(self):
        """Restore the observation statistics, which change the actions."""
        trainer = _toy_trainer(num_worlds=4)
        trainer.step_iteration()
        width = ToyEnv(1).observation_dim
        observations = np.random.default_rng(4).standard_normal((9, width)).astype(np.float32) * 5.0 + 3.0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pt"
            trainer.save(path)
            checkpoint = trainer.torch.load(path, map_location="cpu", weights_only=False)
            self.assertGreaterEqual(checkpoint["normalizer"]["count"], EPISODE_FRAMES * 4)
            np.testing.assert_allclose(checkpoint["normalizer"]["mean"], trainer.normalizer.mean, rtol=1e-12)
            policy = train_module.load_policy(path)
            stripped = dict(checkpoint)
            stripped["normalizer"] = train_module.RunningNormalizer(trainer.spec.observation_dim).state()
            naive = train_module.FrozenPolicy(stripped)
            difference = np.abs(naive.act(observations) - policy.act(observations)).max()
            self.assertGreater(difference, 1e-6, "dropping the statistics must change the deployed actions")

    def test_frozen_policy_rejects_a_foreign_format(self):
        """Refuse a checkpoint written by an incompatible layout version."""
        trainer = _toy_trainer(num_worlds=4)
        checkpoint = trainer.checkpoint()
        checkpoint["format"] = train_module.CHECKPOINT_FORMAT + 1
        with self.assertRaises(ValueError):
            train_module.FrozenPolicy(checkpoint)

    def test_frozen_policy_drives_the_environment(self):
        """Score a frozen policy on the environment without the trainer."""
        trainer = _toy_trainer(num_worlds=4)
        trainer.step_iteration()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pt"
            trainer.save(path)
            policy = train_module.load_policy(path)
        env = ToyEnv(4, seed=0)
        returns = train_module.evaluate_policy(policy, env, episodes=2)
        self.assertEqual(returns.shape, (8,))
        np.testing.assert_allclose(returns[:4], returns[4:], rtol=1e-6)


@unittest.skipUnless(_HAS_TORCH and _HAS_ONNX, "torch or onnx not installed")
class TestOnnxExport(unittest.TestCase):
    """Check the ONNX artifact consumed by Newton's inference path."""

    def test_export_declares_the_expected_io_shapes(self):
        """Export a graph whose input and output widths match the policy."""
        from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes  # noqa: PLC0415

        trainer = _toy_trainer(num_worlds=4)
        trainer.step_iteration()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "policy.pt"
            trainer.save(checkpoint)
            policy = train_module.load_policy(checkpoint)
            with warnings.catch_warnings():
                # Torch deprecates its TorchScript exporter, but the dynamo
                # exporter needs onnxscript, which Newton does not depend on.
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                exported = train_module.export_onnx(policy, Path(folder) / "policy.onnx")
            self.assertIsNotNone(exported)
            self.assertTrue(exported.exists())
            validate_policy_io_shapes(
                str(exported),
                train_module.ONNX_INPUT_NAME,
                train_module.ONNX_OUTPUT_NAME,
                obs_width=trainer.spec.observation_dim,
                action_width=trainer.spec.action_dim,
                context="impedance policy export",
            )


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestLearning(unittest.TestCase):
    """Check that PPO actually improves the policy on the toy problem."""

    def test_ppo_reduces_cost_on_the_toy(self):
        """Close most of the gap to the known optimum of the toy problem.

        The toy reward is ``-||a - 0.5 obs||^2`` with ``obs ~ N(0, I_3)``, so
        the optimal return is exactly zero and the return of the initial
        policy, whose mean output is near zero and whose standard deviation is
        ``exp(init_log_std)``, is close to
        ``-frames * (3 * 0.25 + 3 * exp(2 * init_log_std))``. The threshold is
        stated against that analytic gap rather than against a tuned number:
        the run must close at least 70% of it, which a policy that merely
        shrinks its action noise cannot reach, because the noise accounts for
        about 60% of the initial cost.
        """
        trainer = _toy_trainer()
        # The per-iteration report is exercised by its own assertions below;
        # printing sixty lines would only bury the result of the suite.
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            reports = trainer.train()
        self.assertEqual(len(captured.getvalue().strip().splitlines()), trainer.config.iterations)
        initial = reports[0].return_mean
        final = float(np.mean(trainer.evaluate(episodes=1)))
        optimum = 0.0
        analytic = -EPISODE_FRAMES * (3.0 * TOY_GAIN**2 + 3.0 * np.exp(2.0 * trainer.config.init_log_std))
        self.assertLess(abs(initial - analytic) / abs(analytic), 0.2, "initial return must match the analytic value")
        closed = (final - initial) / (optimum - initial)
        self.assertGreater(closed, 0.70, f"PPO closed only {closed:.1%} of the gap: {initial:.2f} -> {final:.2f}")
        self.assertLess(reports[-1].entropy, reports[0].entropy, "the action noise must shrink as the policy improves")
        self.assertGreater(reports[-1].explained_variance, 0.3, "the critic must explain part of the return variance")


_FLOAT = r"-?\d+\.\d{3}|nan"
"""Every numeric field is a three-decimal float, or NaN when it is unavailable."""

EVAL_LINE = re.compile(
    r"^eval iteration=(?P<iteration>\d+) "
    rf"objective_j=(?P<objective_j>{_FLOAT}) "
    r"feasible=(?P<feasible>[01]) "
    r"on_task=(?P<on_task>[01]) "
    rf"excursion_duration=(?P<excursion_duration>{_FLOAT}) "
    rf"excursion_impulse=(?P<excursion_impulse>{_FLOAT}) "
    rf"excursion_momentum=(?P<excursion_momentum>{_FLOAT}) "
    rf"violation_total=(?P<violation_total>{_FLOAT}) "
    rf"eval_return=(?P<eval_return>{_FLOAT}) "
    rf"peak_fz_n=(?P<peak_fz_n>{_FLOAT}) "
    rf"peak_fz_ref_n=(?P<peak_fz_ref_n>{_FLOAT}) "
    rf"peak_time_pct=(?P<peak_time_pct>{_FLOAT}) "
    rf"peak_time_ref_pct=(?P<peak_time_ref_pct>{_FLOAT}) "
    rf"fz_rms_n=(?P<fz_rms_n>{_FLOAT}) "
    rf"impulse_err_pct=(?P<impulse_err_pct>{_FLOAT}) "
    rf"com_vz_rms=(?P<com_vz_rms>{_FLOAT}) "
    rf"com_z_rms_mm=(?P<com_z_rms_mm>{_FLOAT}) "
    rf"contact_ms=(?P<contact_ms>{_FLOAT}) "
    rf"peak_compression_mm=(?P<peak_compression_mm>{_FLOAT}) "
    r"trace=(?P<trace>\S+)$"
)

"""Strict reader of the evaluation record, as a downstream parser must see it."""


def _eval_lines(text: str) -> list[str]:
    """Return the evaluation records printed inside captured output.

    Args:
        text: Captured stdout of a training call.
    """
    return [line for line in text.strip().splitlines() if line.startswith("eval ")]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestEvaluationLog(unittest.TestCase):
    """Check the deterministic evaluation record and its logging schedule."""

    def _train(self, verdict, iterations, eval_interval):
        """Train briefly and return the trainer with its captured output.

        Args:
            verdict: Scored outcome the environment reports on the done frame.
            iterations: Training iterations to run.
            eval_interval: Iterations between evaluations; zero disables them.
        """
        trainer = _toy_trainer(
            verdict=verdict,
            num_worlds=4,
            iterations=iterations,
            eval_interval=eval_interval,
        )
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            trainer.train()
        return trainer, captured.getvalue()

    def test_record_line_matches_the_contract(self):
        """Emit every key, in order, with three decimals and 0/1 flags."""
        verdict = Verdict(True, False, 1.5, 77.4, {}, {"momentum": 1.19})
        trainer, output = self._train(verdict, iterations=1, eval_interval=1)
        lines = _eval_lines(output)
        self.assertEqual(len(lines), 1)
        match = EVAL_LINE.match(lines[0])
        self.assertIsNotNone(match, lines[0])
        self.assertEqual(int(match["iteration"]), 1)
        self.assertEqual(float(match["objective_j"]), 77.400)
        self.assertEqual(match["feasible"], "1")
        self.assertEqual(match["on_task"], "0")
        self.assertEqual(float(match["excursion_momentum"]), 1.190)
        self.assertEqual(float(match["violation_total"]), 0.000)
        returns = trainer.eval_history[-1]["eval_return"]
        self.assertAlmostEqual(float(match["eval_return"]), round(returns, 3), places=3)

    def test_satisfied_tolerance_emits_zero(self):
        """Report a satisfied tolerance as 0.000 instead of dropping the key."""
        verdict = Verdict(True, False, 1.5, 12.0, {}, {"momentum": 2.5})
        _, output = self._train(verdict, iterations=1, eval_interval=1)
        match = EVAL_LINE.match(_eval_lines(output)[0])
        self.assertIsNotNone(match)
        self.assertEqual(float(match["excursion_duration"]), 0.000)
        self.assertEqual(float(match["excursion_impulse"]), 0.000)
        self.assertEqual(float(match["excursion_momentum"]), 2.500)

    def test_infeasible_verdict_reports_its_violations(self):
        """Sum the tier 1 violations and still emit every key."""
        verdict = Verdict(False, False, 900.0, float("nan"), {"penetration": 0.4, "saturation": 0.35}, {})
        _, output = self._train(verdict, iterations=1, eval_interval=1)
        line = _eval_lines(output)[0]
        match = EVAL_LINE.match(line)
        self.assertIsNotNone(match, line)
        self.assertEqual(match["feasible"], "0")
        self.assertEqual(match["on_task"], "0")
        self.assertAlmostEqual(float(match["violation_total"]), 0.750, places=3)
        self.assertEqual(match["objective_j"], "nan", "an unavailable objective is NaN, never a crash")
        self.assertEqual(float(match["excursion_momentum"]), 0.000)

    def test_missing_verdict_still_emits_a_full_line(self):
        """Emit a complete record when the environment reports no verdict."""
        _, output = self._train(None, iterations=1, eval_interval=1)
        line = _eval_lines(output)[0]
        match = EVAL_LINE.match(line)
        self.assertIsNotNone(match, line)
        self.assertEqual(match["objective_j"], "nan")
        self.assertEqual(match["feasible"], "0")

    def test_interval_zero_disables_evaluation(self):
        """Print no evaluation record at all when the interval is zero."""
        trainer, output = self._train(Verdict(True, True, 1.0, 5.0, {}, {}), iterations=3, eval_interval=0)
        self.assertEqual(_eval_lines(output), [])
        self.assertEqual(trainer.eval_history, [])

    def test_final_iteration_always_evaluates(self):
        """Evaluate on the last iteration even when it is not on the interval."""
        trainer, output = self._train(Verdict(True, True, 1.0, 5.0, {}, {}), iterations=3, eval_interval=25)
        lines = _eval_lines(output)
        self.assertEqual(len(lines), 1)
        self.assertEqual(int(EVAL_LINE.match(lines[0])["iteration"]), 3)
        self.assertEqual(len(trainer.eval_history), 1)

    def test_interval_and_final_iteration_both_evaluate(self):
        """Evaluate on every interval and once more on the final iteration."""
        trainer, output = self._train(Verdict(True, True, 1.0, 5.0, {}, {}), iterations=3, eval_interval=2)
        iterations = [int(EVAL_LINE.match(line)["iteration"]) for line in _eval_lines(output)]
        self.assertEqual(iterations, [2, 3])
        self.assertEqual([record["iteration"] for record in trainer.eval_history], [2, 3])

    def test_evaluation_does_not_disturb_training(self):
        """Leave the normalizer frozen and the environment ready for the next rollout."""
        verdict = Verdict(True, True, 1.0, 5.0, {}, {})
        trainer, _ = self._train(verdict, iterations=2, eval_interval=1)
        self.assertTrue(trainer.normalizer.training)
        self.assertEqual(len(trainer.reports), 2)
        # The environment is stepped again only through collect(), which resets
        # first, so a further iteration must still run.
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train(iterations=1)
        self.assertEqual(len(trainer.reports), 3)

    def test_eval_history_round_trips_through_the_checkpoint(self):
        """Carry the evaluation curve in the checkpoint and restore it intact."""
        verdict = Verdict(True, False, 1.5, 77.4, {}, {"momentum": 1.19})
        trainer, _ = self._train(verdict, iterations=2, eval_interval=1)
        self.assertEqual(len(trainer.eval_history), 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pt"
            trainer.save(path)
            stored = trainer.torch.load(path, map_location="cpu", weights_only=False)["eval_history"]
            policy = train_module.load_policy(path)
        # NaN metrics compare unequal under ==; assert_equal treats them as equal.
        np.testing.assert_equal(stored, trainer.eval_history)
        np.testing.assert_equal(policy.eval_history, trainer.eval_history)
        self.assertEqual([record["iteration"] for record in policy.eval_history], [1, 2])
        self.assertAlmostEqual(policy.eval_history[0]["excursion_momentum"], 1.19, places=9)
        self.assertAlmostEqual(policy.eval_history[0]["objective_j"], 77.4, places=6)

    def test_record_reads_world_zero(self):
        """Read the record from world 0 of a batched terminal info."""
        verdicts = [
            Verdict(True, True, 1.0, 11.0, {}, {}),
            Verdict(False, False, 900.0, 2.0, {"contact": 1.0}, {}),
        ]
        info = {"verdicts": verdicts, "objective_j": np.array([11.0, 2.0], dtype=np.float32)}
        record = train_module.evaluation_record(7, info, np.array([-3.5, -9.0]))
        self.assertEqual(record.iteration, 7)
        self.assertTrue(record.feasible)
        self.assertTrue(record.on_task)
        self.assertAlmostEqual(record.objective_j, 11.0, places=6)
        self.assertAlmostEqual(record.eval_return, -3.5, places=6)
        self.assertEqual(record.violation_total, 0.0)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestEvaluationPhysics(unittest.TestCase):
    """Check the physical metrics and the waveform artifact of an evaluation."""

    def test_metrics_match_the_analytic_waveform(self):
        """Reproduce every physical metric of the analytic rig-shaped toy.

        The simulated force is the measured half-sine scaled by 1.05 on the
        same grid, so the impulse error is exactly 5 %, both peaks sit at half
        of contact, and the COM velocity error is the built-in offset.
        """
        env = RigToyEnv(1)
        metrics, waveform = train_module.evaluation_physics(env)
        loaded, shape = _rig_window()
        window = slice(int(loaded[0]), int(loaded[-1]) + 1)
        span = env.times[window]
        self.assertEqual(int(loaded[0]), 21)
        self.assertEqual(int(loaded[-1]), 159)
        self.assertAlmostEqual(metrics["peak_fz_n"], RIG_PEAK_FZ_N, places=6)
        self.assertAlmostEqual(metrics["peak_fz_ref_n"], RIG_REFERENCE_PEAK_FZ_N, places=6)
        self.assertAlmostEqual(metrics["peak_time_pct"], 50.0, places=6)
        self.assertAlmostEqual(metrics["peak_time_ref_pct"], 50.0, places=6)
        self.assertAlmostEqual(metrics["contact_ms"], 138.0, places=6)
        self.assertAlmostEqual(metrics["impulse_err_pct"], 5.0, places=6)
        expected_rms = 50.0 * float(np.sqrt(np.mean(shape[window] ** 2)))
        self.assertAlmostEqual(metrics["fz_rms_n"], expected_rms, places=6)
        self.assertAlmostEqual(metrics["com_vz_rms"], RIG_VZ_OFFSET_M_S, places=9)
        expected_height = 1.0e3 * RIG_VZ_OFFSET_M_S * float(np.sqrt(np.mean(span**2)))
        self.assertAlmostEqual(metrics["com_z_rms_mm"], expected_height, places=6)
        self.assertAlmostEqual(metrics["peak_compression_mm"], 1.0e3 * RIG_PEAK_COMPRESSION_M, places=6)
        self.assertIsNotNone(waveform)

    def test_waveform_samples_frame_boundaries(self):
        """Sample the waveforms once per frame boundary, 46 points for 45 frames."""
        _, waveform = train_module.evaluation_physics(RigToyEnv(1))
        for key in train_module.WAVEFORM_KEYS:
            self.assertIn(key, waveform)
            self.assertEqual(waveform[key].shape, (EPISODE_FRAMES + 1,), key)
        self.assertEqual(waveform["contact_start_s"].shape, ())
        self.assertAlmostEqual(float(waveform["contact_start_s"]), 0.021, places=9)
        self.assertAlmostEqual(float(waveform["contact_end_s"]), 0.159, places=9)
        np.testing.assert_allclose(waveform["stiffness_n_m"], 20000.0)
        np.testing.assert_allclose(waveform["damping_ratio"], 0.6)

    def test_line_reports_every_physical_key(self):
        """Emit the physical metrics in the pinned order with the trace path."""
        verdict = Verdict(True, False, 1.5, 77.4, {}, {"momentum": 1.19})
        trainer = _toy_trainer(
            verdict=verdict,
            env_class=RigToyEnv,
            num_worlds=2,
            iterations=1,
            eval_interval=1,
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "policy.pt"
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                trainer.train(output=output)
            line = _eval_lines(captured.getvalue())[0]
            match = EVAL_LINE.match(line)
            self.assertIsNotNone(match, line)
            self.assertAlmostEqual(float(match["peak_fz_n"]), RIG_PEAK_FZ_N, places=3)
            self.assertAlmostEqual(float(match["peak_fz_ref_n"]), RIG_REFERENCE_PEAK_FZ_N, places=3)
            self.assertAlmostEqual(float(match["peak_time_pct"]), 50.0, places=3)
            self.assertAlmostEqual(float(match["peak_time_ref_pct"]), 50.0, places=3)
            self.assertAlmostEqual(float(match["impulse_err_pct"]), 5.0, places=3)
            self.assertAlmostEqual(float(match["com_vz_rms"]), RIG_VZ_OFFSET_M_S, places=3)
            self.assertGreater(float(match["com_z_rms_mm"]), 0.0)
            self.assertAlmostEqual(float(match["contact_ms"]), 138.0, places=3)
            self.assertAlmostEqual(float(match["peak_compression_mm"]), 12.0, places=3)
            self.assertEqual(match["trace"], str(output.with_suffix(".eval.npz")))
            self.assertEqual(trainer.eval_history[-1]["trace"], match["trace"])

    def test_waveform_file_round_trips(self):
        """Write one npz per run whose arrays reload with consistent lengths."""
        trainer = _toy_trainer(env_class=RigToyEnv, num_worlds=2, iterations=2, eval_interval=1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "policy.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train(output=output)
            path = train_module.waveform_path(output)
            self.assertTrue(path.exists())
            self.assertEqual(sorted(p.name for p in Path(folder).glob("*.npz")), ["policy.eval.npz"])
            with np.load(path) as stored:
                lengths = {key: stored[key].shape[0] for key in train_module.WAVEFORM_KEYS}
                self.assertEqual(set(lengths.values()), {EPISODE_FRAMES + 1})
                self.assertEqual(int(stored["iteration"]), 2, "the file holds the latest evaluation")
                np.testing.assert_allclose(stored["time_s"][0], 0.0)
                self.assertTrue(np.all(np.isfinite(stored["reference_fz_n"])))

    def test_unwritable_trace_reports_none(self):
        """Report trace=none instead of raising when the file cannot be written."""
        trainer = _toy_trainer(env_class=RigToyEnv, num_worlds=2, iterations=1, eval_interval=1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "policy.pt"
            # A directory where the waveform file belongs blocks only that write,
            # so the checkpoint still succeeds and the run must continue.
            train_module.waveform_path(output).mkdir()
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                trainer.train(output=output)
            match = EVAL_LINE.match(_eval_lines(captured.getvalue())[0])
            self.assertIsNotNone(match)
            self.assertEqual(match["trace"], "none")
            self.assertAlmostEqual(float(match["peak_fz_n"]), RIG_PEAK_FZ_N, places=3)

    def test_environment_without_a_trace_reports_nan(self):
        """Report NaN metrics and trace=none for an environment with no trace."""
        metrics, waveform = train_module.evaluation_physics(ToyEnv(2))
        self.assertIsNone(waveform)
        self.assertEqual(sorted(metrics), sorted(train_module.PHYSICAL_METRICS))
        self.assertTrue(all(np.isnan(value) for value in metrics.values()))
        self.assertEqual(train_module.write_waveform(waveform, "unused.pt", 1), "none")


if __name__ == "__main__":
    unittest.main(verbosity=2)
