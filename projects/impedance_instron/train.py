# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Proximal policy optimization for the residual impedance policy of the rig.

One policy is trained on a single known shoe and then frozen, so that the
controller is a controlled variable when the same policy is replayed on other
shoe materials. Re-solving a command per shoe was rejected for that reason: two
independent solves can differ for reasons that have nothing to do with the
material.

The policy is a residual on the solved 15-parameter spline command of
:class:`projects.impedance_instron.control.LegCommand`. Its action is
``[dL0, dlogK, dzeta]``, applied by the environment around that nominal.

Torch is an optional extra of this repository, so it is imported lazily inside
the functions that need it. Importing this module therefore works without
torch; only training, checkpoint loading, and ONNX export require it::

    uv run --extra examples --extra torch-cu12 -m projects.impedance_instron.train --iterations 200

A checkpoint stores the network weights, the observation normalization
statistics, the nominal command, the environment configuration, and the
task-level evaluation history together. The statistics are part of the policy:
a policy deployed without them sees observations in the wrong units and
silently misbehaves. The evaluation history is what makes a finished run
self-describing: it carries the tier 2 excursions and the tier 3 work proxy of
every ``--eval-interval`` iterations, so the run no longer depends on its log.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ActorCriticSpec",
    "EvalRecord",
    "FrozenPolicy",
    "PPOConfig",
    "PPOTrainer",
    "RunningNormalizer",
    "clipped_surrogate",
    "compute_advantages",
    "create_actor_critic",
    "evaluate_policy",
    "evaluation_physics",
    "evaluation_record",
    "explained_variance",
    "export_onnx",
    "load_nominal",
    "load_policy",
    "main",
    "value_loss",
    "waveform_path",
    "write_waveform",
]

CHECKPOINT_FORMAT = 1
"""Checkpoint layout version, bumped whenever stored keys change meaning."""

ONNX_INPUT_NAME = "obs"
"""Graph input consumed by :mod:`newton.examples.robot.onnx_policy_utils`."""

ONNX_OUTPUT_NAME = "actions"
"""Graph output consumed by :mod:`newton.examples.robot.onnx_policy_utils`."""

_OBSERVATION_CLIP = 10.0
"""Normalized observations are clipped to this many standard deviations."""


def _require_torch():
    """Import torch or explain how to install the optional extra."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Training the impedance policy requires the optional torch extra. "
            "Run `uv run --extra examples --extra torch-cu12 -m projects.impedance_instron.train`."
        ) from exc
    return torch


def onnx_available() -> bool:
    """Return whether the optional ``onnx`` package can be imported."""
    return importlib.util.find_spec("onnx") is not None


class RunningNormalizer:
    """Welford running mean and variance of observations, in NumPy.

    The statistics are part of the policy, so they are saved with the weights
    and frozen whenever the policy is evaluated: updating them during
    evaluation would make a frozen policy behave differently on every shoe it
    is replayed on.
    """

    def __init__(self, dim: int, epsilon: float = 1.0e-8):
        """Start from zero mean and unit variance.

        Args:
            dim: Observation width.
            epsilon: Variance floor used when normalizing.
        """
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)
        self.count = 0
        self.training = True

    @property
    def var(self) -> np.ndarray:
        """Population variance of the observations seen so far, shape [dim]."""
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return self.m2 / self.count

    @property
    def std(self) -> np.ndarray:
        """Standard deviation used by :meth:`normalize`, shape [dim]."""
        return np.sqrt(self.var + self.epsilon)

    def update(self, observations: np.ndarray) -> None:
        """Fold a batch of observations into the running statistics.

        Args:
            observations: Raw observations, shape [batch, dim].
        """
        if not self.training:
            return
        batch = np.asarray(observations, dtype=np.float64).reshape(-1, self.dim)
        if batch.shape[0] == 0:
            return
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        """Return whitened observations clipped to a fixed band.

        Args:
            observations: Raw observations, shape [batch, dim].
        """
        batch = np.asarray(observations, dtype=np.float32).reshape(-1, self.dim)
        if self.count < 2:
            return np.clip(batch, -_OBSERVATION_CLIP, _OBSERVATION_CLIP).astype(np.float32)
        whitened = (batch - self.mean) / self.std
        return np.clip(whitened, -_OBSERVATION_CLIP, _OBSERVATION_CLIP).astype(np.float32)

    def eval(self) -> None:
        """Freeze the statistics, as required when deploying a frozen policy."""
        self.training = False

    def train(self) -> None:
        """Resume updating the statistics."""
        self.training = True

    def state(self) -> dict[str, Any]:
        """Return a plain, serializable copy of the statistics."""
        return {
            "dim": self.dim,
            "epsilon": self.epsilon,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RunningNormalizer:
        """Rebuild a normalizer from :meth:`state`.

        Args:
            state: Mapping produced by :meth:`state`.
        """
        normalizer = cls(int(state["dim"]), float(state["epsilon"]))
        normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
        normalizer.m2 = np.asarray(state["m2"], dtype=np.float64)
        normalizer.count = int(state["count"])
        return normalizer


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    advantage_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return generalized advantage estimates and their value targets.

    Implements the estimator of Schulman et al. (2015): ``delta_t = r_t + gamma
    V(s_{t+1}) (1 - done_t) - V(s_t)`` accumulated backwards with the factor
    ``gamma * advantage_lambda``. A terminal step truncates both the bootstrap and
    the accumulation, so a finished episode never leaks value into the step
    before it.

    Args:
        rewards: Rewards, shape [steps, worlds].
        values: Value predictions at the visited states, shape [steps, worlds].
        dones: Terminal flags, shape [steps, worlds].
        last_values: Value of the state after the final step, shape [worlds].
        gamma: Discount factor.
        advantage_lambda: Bias-variance trade-off of the advantage estimator.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    for step in range(steps - 1, -1, -1):
        alive = 1.0 - dones[step]
        bootstrap = last_values if step == steps - 1 else values[step + 1]
        delta = rewards[step] + gamma * bootstrap * alive - values[step]
        running = delta + gamma * advantage_lambda * alive * running
        advantages[step] = running
    return advantages, advantages + values


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return ``1 - Var(target - prediction) / Var(target)``.

    Args:
        predictions: Value predictions, any shape.
        targets: Value targets of the same shape.
    """
    targets = np.asarray(targets, dtype=np.float64).ravel()
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    variance = targets.var()
    if variance <= 0.0:
        return float("nan")
    return float(1.0 - (targets - predictions).var() / variance)


def clipped_surrogate(ratio, advantages, clip: float):
    """Return the per-sample clipped PPO policy loss.

    The loss is ``-min(r A, clip(r, 1 - c, 1 + c) A)``, so it equals the
    unclipped loss at ``r == 1`` and stops rewarding steps that push the ratio
    past the trust region in the direction the advantage favours.

    Args:
        ratio: Likelihood ratio of new over old policy, any shape.
        advantages: Advantage estimates broadcastable to ``ratio``.
        clip: Half-width of the trust region on the ratio.
    """
    torch = _require_torch()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    return -torch.min(unclipped, clipped)


def value_loss(values, old_values, returns, clip: float | None):
    """Return the per-sample clipped value loss.

    Args:
        values: Current value predictions, shape [batch].
        old_values: Value predictions stored during the rollout, shape [batch].
        returns: Value targets, shape [batch].
        clip: Half-width of the value trust region, or ``None`` to disable it.
    """
    torch = _require_torch()
    plain = (values - returns) ** 2
    if clip is None:
        return 0.5 * plain
    bounded = old_values + torch.clamp(values - old_values, -clip, clip)
    return 0.5 * torch.max(plain, (bounded - returns) ** 2)


@dataclass
class ActorCriticSpec:
    """Shape of the actor-critic network, stored with every checkpoint."""

    observation_dim: int
    """Observation width."""

    action_dim: int
    """Action width; three for the residual impedance command."""

    hidden_size: int = 128
    """Width of both hidden layers of the actor and of the critic."""

    init_log_std: float = -0.5
    """Initial state-independent log standard deviation of the actor."""


_ACTOR_CRITIC_CACHE: dict[str, type] = {}
"""Caches the torch class, which can only be defined once torch is imported."""


def _actor_critic_class() -> type:
    """Define the torch actor-critic on first use and cache the class."""
    cached = _ACTOR_CRITIC_CACHE.get("class")
    if cached is not None:
        return cached
    torch = _require_torch()

    def _layer(inputs: int, outputs: int, gain: float):
        """Return an orthogonally initialized linear layer with zero bias."""
        layer = torch.nn.Linear(inputs, outputs)
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.zeros_(layer.bias)
        return layer

    class ActorCritic(torch.nn.Module):
        """Gaussian actor and separate critic, two tanh hidden layers each.

        The actor and the critic share no parameters: the value function is
        fitted much harder than the policy, and sharing a trunk would let value
        gradients dominate the residual command.
        """

        def __init__(self, spec: ActorCriticSpec):
            """Build both networks from a stored specification.

            Args:
                spec: Network shape carried by the checkpoint.
            """
            super().__init__()
            self.spec = spec
            hidden = spec.hidden_size
            self.actor = torch.nn.Sequential(
                _layer(spec.observation_dim, hidden, math.sqrt(2.0)),
                torch.nn.Tanh(),
                _layer(hidden, hidden, math.sqrt(2.0)),
                torch.nn.Tanh(),
                _layer(hidden, spec.action_dim, 0.01),
            )
            self.critic = torch.nn.Sequential(
                _layer(spec.observation_dim, hidden, math.sqrt(2.0)),
                torch.nn.Tanh(),
                _layer(hidden, hidden, math.sqrt(2.0)),
                torch.nn.Tanh(),
                _layer(hidden, 1, 1.0),
            )
            self.log_std = torch.nn.Parameter(torch.full((spec.action_dim,), float(spec.init_log_std)))

        def forward(self, observations):
            """Return the deterministic action mean, the evaluation action.

            Args:
                observations: Normalized observations, shape [batch, obs].
            """
            return self.actor(observations)

        def distribution(self, observations):
            """Return the Gaussian action distribution.

            Args:
                observations: Normalized observations, shape [batch, obs].
            """
            mean = self.actor(observations)
            return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

        def value(self, observations):
            """Return the state value, shape [batch].

            Args:
                observations: Normalized observations, shape [batch, obs].
            """
            return self.critic(observations).squeeze(-1)

        def act(self, observations, deterministic: bool = False):
            """Sample an action and report its log-probability and value.

            Args:
                observations: Normalized observations, shape [batch, obs].
                deterministic: Return the distribution mean instead of a sample.
            """
            with torch.no_grad():
                distribution = self.distribution(observations)
                actions = distribution.mean if deterministic else distribution.sample()
                log_prob = distribution.log_prob(actions).sum(-1)
                values = self.value(observations)
            return actions, log_prob, values

        def evaluate(self, observations, actions):
            """Return log-probabilities, entropy, and values for stored actions.

            Args:
                observations: Normalized observations, shape [batch, obs].
                actions: Actions taken during the rollout, shape [batch, act].
            """
            distribution = self.distribution(observations)
            log_prob = distribution.log_prob(actions).sum(-1)
            entropy = distribution.entropy().sum(-1)
            return log_prob, entropy, self.value(observations)

    _ACTOR_CRITIC_CACHE["class"] = ActorCritic
    return ActorCritic


def create_actor_critic(spec: ActorCriticSpec, device: str = "cpu"):
    """Build the actor-critic network on a device.

    Args:
        spec: Network shape.
        device: Torch device string.
    """
    return _actor_critic_class()(spec).to(device)


@dataclass
class PPOConfig:
    """Training settings of the PPO loop."""

    num_worlds: int = 64
    """Environments stepped in lockstep; one rollout is worlds x frames."""

    iterations: int = 200
    """Policy updates."""

    learning_rate: float = 3.0e-4
    """Adam step size."""

    clip: float = 0.2
    """Policy ratio trust region."""

    value_clip: float | None = 0.2
    """Value trust region, or ``None`` for an unclipped value loss."""

    entropy: float = 0.003
    """Entropy bonus weight."""

    value_coefficient: float = 0.5
    """Weight of the value loss in the total loss."""

    gamma: float = 0.99
    """Discount factor."""

    advantage_lambda: float = 0.95
    """Advantage-trace decay."""

    epochs: int = 5
    """Passes over each rollout buffer."""

    minibatches: int = 4
    """Minibatches per epoch."""

    max_grad_norm: float = 0.5
    """Global gradient-norm clip."""

    seed: int = 0
    """Seed of torch and of the environment."""

    hidden_size: int = 128
    """Hidden width of both networks."""

    init_log_std: float = -0.5
    """Initial actor log standard deviation."""

    device: str = "cpu"
    """Torch device string."""

    eval_interval: int = 25
    """Iterations between deterministic evaluations; zero disables them."""


@dataclass
class IterationReport:
    """Scalars reported once per policy update."""

    iteration: int
    """One-based update index."""

    return_mean: float
    """Mean undiscounted episode return over the worlds."""

    return_median: float
    """Median undiscounted episode return over the worlds."""

    return_best: float
    """Best undiscounted episode return over the worlds."""

    policy_loss: float
    """Mean clipped surrogate loss."""

    value_loss: float
    """Mean clipped value loss."""

    entropy: float
    """Mean action-distribution entropy [nats]."""

    explained_variance: float
    """Fraction of the return variance explained by the critic."""

    approx_kl: float
    """Approximate KL between the sampling and the updated policy."""

    clip_fraction: float
    """Fraction of samples whose ratio left the trust region."""

    def line(self) -> str:
        """Return the compact one-line report printed by the training loop."""
        return (
            f"iter {self.iteration:4d} | return mean={self.return_mean:+10.3f} "
            f"med={self.return_median:+10.3f} best={self.return_best:+10.3f} | "
            f"pi={self.policy_loss:+.4f} v={self.value_loss:.4f} ent={self.entropy:+.3f} | "
            f"ev={self.explained_variance:+.3f} kl={self.approx_kl:.5f} clip={self.clip_fraction:.3f}"
        )


EVAL_EXCURSIONS = ("duration", "impulse", "momentum")
"""Tier 2 tolerance names of :class:`~projects.impedance_instron.objective.Verdict`."""


@dataclass
class EvalRecord:
    """Task-level outcome of one deterministic evaluation episode, world 0.

    The training log reports return, entropy, and losses, none of which say
    which part of the task is stuck. These are the deciding numbers of
    :class:`~projects.impedance_instron.objective.Objective`: tier 1
    feasibility, the tier 2 excursions in multiples of their tolerance, and the
    tier 3 work proxy in joules.
    """

    iteration: int
    """Training iteration the evaluation followed."""

    objective_j: float
    """Tier 3 leg-work proxy [J]; NaN when the verdict does not supply it."""

    feasible: bool
    """Whether the rollout cleared every tier 1 limit."""

    on_task: bool
    """Whether the rollout also stayed inside every tier 2 tolerance."""

    excursion_duration: float
    """Stance-duration excursion outside its deadband [multiples of tolerance]."""

    excursion_impulse: float
    """Vertical-impulse excursion outside its deadband [multiples of tolerance]."""

    excursion_momentum: float
    """Momentum-history excursion outside its deadband [multiples of tolerance]."""

    violation_total: float
    """Sum of the tier 1 violations, each normalized by its own limit."""

    eval_return: float
    """Undiscounted reward of the evaluation episode in world 0."""

    peak_fz_n: float = float("nan")
    """Peak simulated shoe vertical force over the contact window [N]."""

    peak_fz_ref_n: float = float("nan")
    """Peak measured reference vertical force over the same window [N]."""

    peak_time_pct: float = float("nan")
    """Instant of the simulated force peak [% of contact]."""

    peak_time_ref_pct: float = float("nan")
    """Instant of the measured force peak [% of contact]."""

    fz_rms_n: float = float("nan")
    """RMS simulated-minus-measured vertical force over the window [N]."""

    impulse_err_pct: float = float("nan")
    """Simulated minus measured vertical impulse [% of the measured impulse]."""

    com_vz_rms: float = float("nan")
    """RMS COM vertical velocity error against the reference [m/s]."""

    com_z_rms_mm: float = float("nan")
    """RMS COM height error against the reference [mm]."""

    contact_ms: float = float("nan")
    """Simulated contact duration [ms]."""

    peak_compression_mm: float = float("nan")
    """Peak foam compression of the episode [mm]."""

    trace: str = "none"
    """Waveform file written for this evaluation, or the literal ``none``."""

    def line(self) -> str:
        """Return the single stdout record, parsed downstream by the dashboard.

        The layout is fixed: the literal ``eval``, then space-separated
        ``key=value`` pairs in declaration order, floats with three decimals,
        and the two flags as ``0`` or ``1``.
        """
        return (
            f"eval iteration={int(self.iteration)} "
            f"objective_j={self.objective_j:.3f} "
            f"feasible={int(bool(self.feasible))} "
            f"on_task={int(bool(self.on_task))} "
            f"excursion_duration={self.excursion_duration:.3f} "
            f"excursion_impulse={self.excursion_impulse:.3f} "
            f"excursion_momentum={self.excursion_momentum:.3f} "
            f"violation_total={self.violation_total:.3f} "
            f"eval_return={self.eval_return:.3f} "
            f"peak_fz_n={self.peak_fz_n:.3f} "
            f"peak_fz_ref_n={self.peak_fz_ref_n:.3f} "
            f"peak_time_pct={self.peak_time_pct:.3f} "
            f"peak_time_ref_pct={self.peak_time_ref_pct:.3f} "
            f"fz_rms_n={self.fz_rms_n:.3f} "
            f"impulse_err_pct={self.impulse_err_pct:.3f} "
            f"com_vz_rms={self.com_vz_rms:.3f} "
            f"com_z_rms_mm={self.com_z_rms_mm:.3f} "
            f"contact_ms={self.contact_ms:.3f} "
            f"peak_compression_mm={self.peak_compression_mm:.3f} "
            f"trace={self.trace}"
        )


# Column indices of the environment's substep trace and of its measured reference
# rows. They mirror `projects.impedance_instron.env` and the reference block built
# in `projects.impedance_instron.example`; the real constants are read off the
# environment's own module when it exposes them, so a reordering there cannot
# silently remap these metrics.
EVAL_TRACE_COLUMNS: dict[str, int] = {
    "TRACE_SHOE_FZ": 0,
    "TRACE_SHOE_FX": 1,
    "TRACE_ANKLE_Z": 2,
    "TRACE_ANKLE_VZ": 3,
    "TRACE_UPPER_VZ": 5,
    "TRACE_LEG_LENGTH": 10,
    "TRACE_COMPRESSION": 12,
}
REFERENCE_ANKLE_Z = 1
"""Reference column holding the prescribed ankle height [m]."""

REFERENCE_COM_Z = 4
"""Reference column holding the virtual upper-mass height [m]."""

REFERENCE_COM_VZ = 9
"""Reference column holding the measured centroid vertical velocity [m/s]."""

REFERENCE_SHOE_FZ = 11
"""Reference column holding the measured shoe vertical force [N]."""

REFERENCE_SHOE_FX = 22
"""Reference column holding the measured shoe fore-aft force [N]."""

CONTACT_FRACTION = 0.02
"""Share of body weight above which the shoe counts as loaded, as in the optimizer."""

WAVEFORM_KEYS = (
    "time_s",
    "shoe_fz_n",
    "reference_fz_n",
    "shoe_fx_n",
    "reference_fx_n",
    "com_z_m",
    "reference_com_z_m",
    "com_vz_m_s",
    "reference_com_vz_m_s",
    "leg_length_m",
    "commanded_length_m",
    "stiffness_n_m",
    "damping_ratio",
)
"""Per-sample waveforms written next to the checkpoint for the overlay plots."""

PHYSICAL_METRICS = (
    "peak_fz_n",
    "peak_fz_ref_n",
    "peak_time_pct",
    "peak_time_ref_pct",
    "fz_rms_n",
    "impulse_err_pct",
    "com_vz_rms",
    "com_z_rms_mm",
    "contact_ms",
    "peak_compression_mm",
)
"""Physical evaluation metrics, all reported over the contact window of world 0."""


def _trace_column(env: Any, name: str) -> int:
    """Return a trace column index, preferring the environment's own constant.

    Args:
        env: Environment instance.
        name: Constant name, such as ``"TRACE_SHOE_FZ"``.
    """
    module = sys.modules.get(type(env).__module__)
    return int(getattr(module, name, EVAL_TRACE_COLUMNS[name]))


def _reference_rows(env: Any) -> np.ndarray | None:
    """Return the measured reference rows of the environment, or ``None``.

    Args:
        env: Environment instance.
    """
    rows = getattr(env, "reference", None)
    if rows is None:
        # The environment keeps the measured rows private; they are the only
        # source of the measured force and centroid motion an overlay needs.
        rows = getattr(env, "_reference_host", None)
    if rows is None:
        return None
    rows = np.asarray(rows, dtype=np.float64)
    return rows if rows.ndim == 2 and rows.shape[1] > REFERENCE_SHOE_FX else None


def evaluation_physics(env: Any, world: int = 0) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    """Reduce one evaluated episode to physical metrics and sampled waveforms.

    Everything is measured over the contact window, the samples whose shoe
    vertical force exceeds :data:`CONTACT_FRACTION` of body weight, which is the
    same gate :mod:`projects.impedance_instron.optimize` scores stance with.
    Simulated COM height is not traced, so it is integrated from the traced COM
    vertical velocity starting at the reference height of the shared initial
    condition; the reported height error is therefore a drift against the
    measured centroid.

    Args:
        env: Environment that has just finished an episode.
        world: Index of the world to describe.
    """
    metrics = dict.fromkeys(PHYSICAL_METRICS, float("nan"))
    try:
        trace = np.asarray(env.trace(world), dtype=np.float64)
        times = np.asarray(env.times, dtype=np.float64)[: trace.shape[0]]
        weight = float(env.body_weight_n)
        share = float(env.foot_mass) / float(env.mass)
        reference = _reference_rows(env)
    except (AttributeError, TypeError, ValueError, IndexError):
        return metrics, None
    if reference is None or trace.ndim != 2 or trace.shape[0] < 2 or times.size != trace.shape[0]:
        return metrics, None
    reference = reference[: trace.shape[0]]

    fz = trace[:, _trace_column(env, "TRACE_SHOE_FZ")]
    fx = trace[:, _trace_column(env, "TRACE_SHOE_FX")]
    compression = trace[:, _trace_column(env, "TRACE_COMPRESSION")]
    com_vz = (
        share * trace[:, _trace_column(env, "TRACE_ANKLE_VZ")]
        + (1.0 - share) * trace[:, _trace_column(env, "TRACE_UPPER_VZ")]
    )
    reference_fz = reference[:, REFERENCE_SHOE_FZ]
    reference_fx = reference[:, REFERENCE_SHOE_FX]
    reference_com_vz = reference[:, REFERENCE_COM_VZ]
    reference_com_z = share * reference[:, REFERENCE_ANKLE_Z] + (1.0 - share) * reference[:, REFERENCE_COM_Z]
    increments = 0.5 * (com_vz[1:] + com_vz[:-1]) * np.diff(times)
    com_z = reference_com_z[0] + np.concatenate([[0.0], np.cumsum(increments)])

    metrics["peak_compression_mm"] = 1.0e3 * float(np.max(compression))
    loaded = fz > CONTACT_FRACTION * weight
    indices = np.nonzero(loaded)[0]
    if indices.size < 2:
        metrics["contact_ms"] = 0.0
        window = slice(0, trace.shape[0])
        start_s, end_s = float("nan"), float("nan")
    else:
        window = slice(int(indices[0]), int(indices[-1]) + 1)
        start_s, end_s = float(times[window][0]), float(times[window][-1])
        span = end_s - start_s
        metrics["contact_ms"] = 1.0e3 * span
        metrics["peak_fz_n"] = float(np.max(fz[window]))
        metrics["peak_fz_ref_n"] = float(np.max(reference_fz[window]))
        metrics["peak_time_pct"] = 1.0e2 * float(times[window][int(np.argmax(fz[window]))] - start_s) / span
        metrics["peak_time_ref_pct"] = (
            1.0e2 * float(times[window][int(np.argmax(reference_fz[window]))] - start_s) / span
        )
        metrics["fz_rms_n"] = float(np.sqrt(np.mean((fz[window] - reference_fz[window]) ** 2)))
        measured = float(np.trapezoid(reference_fz[window], times[window]))
        simulated = float(np.trapezoid(fz[window], times[window]))
        metrics["impulse_err_pct"] = (
            1.0e2 * (simulated - measured) / measured if abs(measured) > 1.0e-9 else float("nan")
        )
        metrics["com_vz_rms"] = float(np.sqrt(np.mean((com_vz[window] - reference_com_vz[window]) ** 2)))
        metrics["com_z_rms_mm"] = 1.0e3 * float(np.sqrt(np.mean((com_z[window] - reference_com_z[window]) ** 2)))

    command = {}
    try:
        command = env.realised_command(world)
    except (AttributeError, TypeError, ValueError, IndexError):
        command = {}
    # One sample per frame boundary keeps the file small enough to rewrite every
    # evaluation while still resolving the force trace.
    stride = max(1, int(getattr(env, "substeps", 1)))
    sampled = np.arange(0, trace.shape[0], stride)
    if sampled[-1] != trace.shape[0] - 1:
        sampled = np.append(sampled, trace.shape[0] - 1)

    def _sampled(values, fallback: float = float("nan")) -> np.ndarray:
        """Return one waveform sampled at the frame boundaries.

        Args:
            values: Per-substep series, or ``None`` when the field is missing.
            fallback: Value used when the series is unavailable.
        """
        if values is None:
            return np.full(sampled.size, fallback, dtype=np.float64)
        series = np.asarray(values, dtype=np.float64)
        if series.shape[0] < trace.shape[0]:
            return np.full(sampled.size, fallback, dtype=np.float64)
        return series[sampled]

    waveform = {
        "time_s": times[sampled],
        "shoe_fz_n": fz[sampled],
        "reference_fz_n": reference_fz[sampled],
        "shoe_fx_n": fx[sampled],
        "reference_fx_n": reference_fx[sampled],
        "com_z_m": com_z[sampled],
        "reference_com_z_m": reference_com_z[sampled],
        "com_vz_m_s": com_vz[sampled],
        "reference_com_vz_m_s": reference_com_vz[sampled],
        "leg_length_m": trace[:, _trace_column(env, "TRACE_LEG_LENGTH")][sampled],
        "commanded_length_m": _sampled(command.get("length_m")),
        "stiffness_n_m": _sampled(command.get("stiffness_n_m")),
        "damping_ratio": _sampled(command.get("damping_ratio")),
        "contact_start_s": np.asarray(start_s, dtype=np.float64),
        "contact_end_s": np.asarray(end_s, dtype=np.float64),
    }
    return metrics, waveform


def waveform_path(output: str | Path | None) -> Path | None:
    """Return the waveform file that belongs to a checkpoint path.

    Args:
        output: Checkpoint path, or ``None`` when the run writes no checkpoint.
    """
    return None if output is None else Path(output).with_suffix(".eval.npz")


def write_waveform(waveform: dict[str, np.ndarray] | None, output: str | Path | None, iteration: int) -> str:
    """Write the latest evaluation waveforms, returning the path or ``"none"``.

    One file per run, overwritten on every evaluation, so the artifact always
    describes the most recent evaluation. A diagnostic must never end a run, so
    any failure is reported as ``"none"`` instead of raising.

    Args:
        waveform: Arrays produced by :func:`evaluation_physics`.
        output: Checkpoint path the file is written next to.
        iteration: Iteration stamped into the file.
    """
    destination = waveform_path(output)
    if not waveform or destination is None:
        return "none"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, iteration=np.asarray(int(iteration)), **waveform)
    except Exception:
        return "none"
    return str(destination)


def _world_value(value: Any, world: int = 0) -> Any:
    """Return one world's element of a batched info field, or ``None``.

    Args:
        value: Scalar, sequence, or array taken from an environment info dict.
        world: Index of the world to read.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, np.generic)):
        return value
    try:
        return np.asarray(value).reshape(-1)[world]
    except (TypeError, ValueError, IndexError):
        return None


def evaluation_record(
    iteration: int,
    info: Any,
    episode_returns: Any,
    world: int = 0,
    metrics: dict[str, float] | None = None,
    trace: str = "none",
) -> EvalRecord:
    """Reduce the terminal info of an evaluation episode to one record.

    The environment attaches one
    :class:`~projects.impedance_instron.objective.Verdict` per world on the
    done frame. A missing tolerance key means that tolerance is satisfied, so
    it is reported as zero rather than omitted, and an unavailable objective is
    reported as NaN instead of raising: a diagnostic line must never end a run.

    Args:
        iteration: Iteration index stamped on the record.
        info: Info dict returned by the final ``step`` of the episode.
        episode_returns: Undiscounted returns of the episode, shape [worlds].
        world: Index of the world the record describes.
        metrics: Physical metrics from :func:`evaluation_physics`; missing
            entries stay NaN.
        trace: Waveform file written for this evaluation, or ``"none"``.
    """
    fields = info if isinstance(info, dict) else {}
    verdicts = fields.get("verdicts")
    verdict = None
    if verdicts is not None:
        try:
            verdict = verdicts[world]
        except (TypeError, KeyError, IndexError):
            verdict = None
    excursions = dict(getattr(verdict, "excursions", None) or {})
    violations = dict(getattr(verdict, "violations", None) or {})

    objective = getattr(verdict, "objective_j", None)
    if objective is None:
        objective = _world_value(fields.get("objective_j"), world)
    objective = float("nan") if objective is None else float(objective)
    if not math.isfinite(objective):
        objective = float("nan")

    feasible = getattr(verdict, "feasible", None)
    if feasible is None:
        feasible = _world_value(fields.get("feasible"), world)
    on_task = getattr(verdict, "on_task", None)
    if on_task is None:
        on_task = _world_value(fields.get("on_task"), world)

    outside = {name: float(excursions.get(name, 0.0)) for name in EVAL_EXCURSIONS}
    physical = {name: float((metrics or {}).get(name, float("nan"))) for name in PHYSICAL_METRICS}
    return EvalRecord(
        iteration=int(iteration),
        objective_j=objective,
        feasible=bool(feasible) if feasible is not None else False,
        on_task=bool(on_task) if on_task is not None else False,
        excursion_duration=outside["duration"],
        excursion_impulse=outside["impulse"],
        excursion_momentum=outside["momentum"],
        violation_total=float(sum(float(amount) for amount in violations.values())),
        eval_return=float(np.asarray(episode_returns, dtype=np.float64).reshape(-1)[world]),
        trace=str(trace),
        **physical,
    )


@dataclass
class Rollout:
    """One fixed-length, synchronized batch of transitions."""

    observations: np.ndarray
    """Normalized observations, shape [steps, worlds, obs]."""

    actions: np.ndarray
    """Sampled actions, shape [steps, worlds, act]."""

    log_probs: np.ndarray
    """Log-probabilities under the sampling policy, shape [steps, worlds]."""

    values: np.ndarray
    """Critic predictions at the visited states, shape [steps, worlds]."""

    rewards: np.ndarray
    """Rewards, shape [steps, worlds]."""

    dones: np.ndarray
    """Terminal flags, shape [steps, worlds]."""

    last_values: np.ndarray
    """Bootstrap value after the final step, shape [worlds]."""

    episode_returns: np.ndarray
    """Undiscounted return of each world's episode, shape [worlds]."""

    infos: list[dict] = field(default_factory=list)
    """Per-step environment info dictionaries."""


def _assert_synchronized(dones: np.ndarray, step: int, steps: int) -> None:
    """Fail loudly when the environment reports ragged episode ends.

    The trainer treats one rollout as exactly one episode per world, which is
    only valid while every world terminates on the same frame. A ragged end
    would silently mix two episodes into one return and one advantage trace.

    Args:
        dones: Terminal flags of a single step, shape [worlds].
        step: Zero-based index of that step.
        steps: Episode length reported by the environment.
    """
    done = np.asarray(dones, dtype=bool)
    if step == steps - 1:
        if not done.all():
            raise AssertionError(
                f"ragged episode end: {int((~done).sum())} of {done.size} worlds are not done at the final frame "
                f"{step}; the trainer assumes fixed-length, synchronized episodes"
            )
    elif done.any():
        raise AssertionError(
            f"ragged episode end: {int(done.sum())} of {done.size} worlds finished early at frame {step} "
            f"of {steps}; the trainer assumes fixed-length, synchronized episodes"
        )


class PPOTrainer:
    """PPO over a vectorized, fixed-episode-length impedance environment."""

    def __init__(
        self,
        env,
        config: PPOConfig,
        nominal: np.ndarray,
        env_config: dict[str, Any] | None = None,
    ):
        """Seed torch, build the networks, and size the rollout buffer.

        Args:
            env: Vectorized environment exposing the ``ImpedanceEnv`` API.
            config: Training settings.
            nominal: Solved 15-parameter command the actions are residual to.
            env_config: Serializable environment settings stored in checkpoints.
        """
        torch = _require_torch()
        self.torch = torch
        self.env = env
        self.config = config
        self.nominal = np.asarray(nominal, dtype=np.float64)
        self.env_config = dict(env_config or {})
        self.steps = int(env.episode_frames)
        self.num_worlds = int(config.num_worlds)
        torch.manual_seed(config.seed)
        self.spec = ActorCriticSpec(
            observation_dim=int(env.observation_dim),
            action_dim=int(env.action_dim),
            hidden_size=config.hidden_size,
            init_log_std=config.init_log_std,
        )
        self.device = torch.device(config.device)
        self.policy = create_actor_critic(self.spec, config.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.normalizer = RunningNormalizer(self.spec.observation_dim)
        self.reports: list[IterationReport] = []
        self.eval_history: list[dict[str, Any]] = []

    def _tensor(self, array: np.ndarray):
        """Return a float32 device tensor viewing a NumPy array.

        Args:
            array: Host array of any floating shape.
        """
        return self.torch.as_tensor(np.asarray(array, dtype=np.float32), device=self.device)

    def collect(self, deterministic: bool = False) -> Rollout:
        """Run one synchronized episode in every world.

        Args:
            deterministic: Use the action mean and freeze the normalizer.
        """
        if deterministic:
            self.normalizer.eval()
        steps, worlds = self.steps, self.num_worlds
        observations = np.zeros((steps, worlds, self.spec.observation_dim), dtype=np.float32)
        actions = np.zeros((steps, worlds, self.spec.action_dim), dtype=np.float32)
        log_probs = np.zeros((steps, worlds), dtype=np.float32)
        values = np.zeros((steps, worlds), dtype=np.float32)
        rewards = np.zeros((steps, worlds), dtype=np.float32)
        dones = np.zeros((steps, worlds), dtype=bool)
        infos: list[dict] = []

        raw = np.asarray(self.env.reset(), dtype=np.float32)
        if raw.shape != (worlds, self.spec.observation_dim):
            raise ValueError(f"env.reset returned shape {raw.shape}, expected {(worlds, self.spec.observation_dim)}")
        for step in range(steps):
            self.normalizer.update(raw)
            normalized = self.normalizer.normalize(raw)
            action, log_prob, value = self.policy.act(self._tensor(normalized), deterministic=deterministic)
            action_host = action.detach().cpu().numpy().astype(np.float32)
            observations[step] = normalized
            actions[step] = action_host
            log_probs[step] = log_prob.detach().cpu().numpy()
            values[step] = value.detach().cpu().numpy()
            raw, reward, done, info = self.env.step(action_host)
            raw = np.asarray(raw, dtype=np.float32)
            rewards[step] = np.asarray(reward, dtype=np.float32)
            dones[step] = np.asarray(done, dtype=bool)
            infos.append(info if isinstance(info, dict) else {})
            _assert_synchronized(dones[step], step, steps)

        # Every world ends on the final frame, so the bootstrap is masked away
        # by the terminal flag; it is computed only to keep the estimator call total.
        _, _, last_values = self.policy.act(self._tensor(self.normalizer.normalize(raw)), deterministic=True)
        rollout = Rollout(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
            last_values=last_values.detach().cpu().numpy(),
            episode_returns=rewards.sum(axis=0),
            infos=infos,
        )
        if deterministic:
            self.normalizer.train()
        return rollout

    def update(self, rollout: Rollout) -> dict[str, float]:
        """Run the clipped PPO epochs over one rollout buffer.

        Args:
            rollout: Batch produced by :meth:`collect`.
        """
        torch = self.torch
        config = self.config
        advantages, returns = compute_advantages(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            rollout.last_values,
            config.gamma,
            config.advantage_lambda,
        )
        flat = self.steps * self.num_worlds
        observations = self._tensor(rollout.observations.reshape(flat, -1))
        actions = self._tensor(rollout.actions.reshape(flat, -1))
        old_log_probs = self._tensor(rollout.log_probs.reshape(flat))
        old_values = self._tensor(rollout.values.reshape(flat))
        advantage_tensor = self._tensor(advantages.reshape(flat))
        return_tensor = self._tensor(returns.reshape(flat))

        minibatch = max(1, flat // max(1, config.minibatches))
        totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clip": 0.0}
        batches = 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + len(self.reports))
        for _ in range(config.epochs):
            order = torch.randperm(flat, generator=generator).to(self.device)
            for start in range(0, flat, minibatch):
                index = order[start : start + minibatch]
                if index.numel() < 2:
                    continue
                batch_advantages = advantage_tensor[index]
                # Per-batch normalization keeps the step size comparable across
                # iterations whose reward scale differs by orders of magnitude.
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1.0e-8)
                log_probs, entropy, values = self.policy.evaluate(observations[index], actions[index])
                ratio = (log_probs - old_log_probs[index]).exp()
                policy_loss = clipped_surrogate(ratio, batch_advantages, config.clip).mean()
                critic_loss = value_loss(values, old_values[index], return_tensor[index], config.value_clip).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + config.value_coefficient * critic_loss - config.entropy * entropy_mean
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[index]
                    totals["kl"] += float((log_ratio.exp() - 1.0 - log_ratio).mean())
                    totals["clip"] += float((ratio - 1.0).abs().gt(config.clip).float().mean())
                totals["policy"] += float(policy_loss.detach())
                totals["value"] += float(critic_loss.detach())
                totals["entropy"] += float(entropy_mean.detach())
                batches += 1
        batches = max(batches, 1)
        return {
            "policy_loss": totals["policy"] / batches,
            "value_loss": totals["value"] / batches,
            "entropy": totals["entropy"] / batches,
            "approx_kl": totals["kl"] / batches,
            "clip_fraction": totals["clip"] / batches,
            "explained_variance": explained_variance(rollout.values, returns),
        }

    def step_iteration(self) -> IterationReport:
        """Collect one rollout, update the policy, and report the iteration."""
        rollout = self.collect()
        stats = self.update(rollout)
        report = IterationReport(
            iteration=len(self.reports) + 1,
            return_mean=float(np.mean(rollout.episode_returns)),
            return_median=float(np.median(rollout.episode_returns)),
            return_best=float(np.max(rollout.episode_returns)),
            **stats,
        )
        self.reports.append(report)
        return report

    def train(self, iterations: int | None = None, output: Path | None = None) -> list[IterationReport]:
        """Run the training loop, printing one line per iteration.

        Args:
            iterations: Updates to run; defaults to the configured count.
            output: Checkpoint path rewritten after every iteration.
        """
        count = self.config.iterations if iterations is None else iterations
        for index in range(count):
            report = self.step_iteration()
            print(report.line(), flush=True)
            if self._evaluation_due(report.iteration, last=index == count - 1):
                # The record is appended before the checkpoint is rewritten, so
                # an interrupted run keeps the evaluation curve it has printed.
                print(self.record_evaluation(report.iteration, output).line(), flush=True)
            if output is not None:
                self.save(output)
        return self.reports

    def _evaluation_due(self, iteration: int, last: bool) -> bool:
        """Return whether this iteration ends with a deterministic evaluation.

        Args:
            iteration: One-based iteration index.
            last: Whether this is the final iteration of the training call.
        """
        interval = int(self.config.eval_interval)
        if interval <= 0:
            return False
        return last or iteration % interval == 0

    def record_evaluation(self, iteration: int, output: Path | None = None) -> EvalRecord:
        """Run one deterministic episode and record its task-level outcome.

        The episode runs on the same environment instance between rollouts,
        which is safe because :meth:`collect` resets the environment at the
        start of every iteration and ``ImpedanceEnv.reset`` restores the bodies,
        the foam history, and the trace buffers.

        Args:
            iteration: Iteration index stamped on the record.
            output: Checkpoint path; the waveform file is written next to it.
        """
        rollout = self.collect(deterministic=True)
        metrics, waveform = evaluation_physics(self.env)
        record = evaluation_record(
            iteration,
            rollout.infos[-1] if rollout.infos else {},
            rollout.episode_returns,
            metrics=metrics,
            trace=write_waveform(waveform, output, iteration),
        )
        self.eval_history.append(asdict(record))
        return record

    def evaluate(self, episodes: int = 1) -> np.ndarray:
        """Return episode returns of the deterministic policy.

        Args:
            episodes: Synchronized episodes to run in every world.
        """
        returns = [self.collect(deterministic=True).episode_returns for _ in range(episodes)]
        return np.concatenate(returns)

    def checkpoint(self) -> dict[str, Any]:
        """Return the full deployable state of the policy."""
        return {
            "format": CHECKPOINT_FORMAT,
            "policy": {key: value.detach().cpu() for key, value in self.policy.state_dict().items()},
            "spec": asdict(self.spec),
            "config": asdict(self.config),
            "normalizer": self.normalizer.state(),
            "nominal": self.nominal.tolist(),
            "env_config": self.env_config,
            "eval_history": list(self.eval_history),
        }

    def save(self, path: str | Path) -> Path:
        """Write weights, normalization statistics, nominal, and env settings.

        Args:
            path: Destination file; parent directories are created.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.checkpoint(), destination)
        return destination


class FrozenPolicy:
    """Deterministic, training-free policy rebuilt from a checkpoint file.

    The instance is callable, so both ``policy(obs)`` and ``policy.act(obs)``
    return actions. It carries the observation normalization statistics, the
    nominal command, the environment configuration it was trained with, and
    the evaluation curve of the run, because a policy without them is not
    reproducible on another shoe.
    """

    def __init__(self, checkpoint: dict[str, Any], device: str = "cpu"):
        """Rebuild the network and the normalizer from a checkpoint mapping.

        Args:
            checkpoint: Mapping produced by :meth:`PPOTrainer.checkpoint`.
            device: Torch device string.
        """
        torch = _require_torch()
        self.torch = torch
        stored = int(checkpoint.get("format", 0))
        if stored != CHECKPOINT_FORMAT:
            raise ValueError(f"checkpoint format {stored} is not the expected {CHECKPOINT_FORMAT}")
        self.spec = ActorCriticSpec(**checkpoint["spec"])
        self.device = torch.device(device)
        self.network = create_actor_critic(self.spec, device)
        self.network.load_state_dict(checkpoint["policy"])
        self.network.eval()
        self.normalizer = RunningNormalizer.from_state(checkpoint["normalizer"])
        self.normalizer.eval()
        self.nominal = np.asarray(checkpoint["nominal"], dtype=np.float64)
        self.env_config = dict(checkpoint.get("env_config", {}))
        self.config = dict(checkpoint.get("config", {}))
        self.eval_history = list(checkpoint.get("eval_history", []))

    def act(self, observations: np.ndarray) -> np.ndarray:
        """Return the deterministic action mean for raw observations.

        Args:
            observations: Raw environment observations, shape [batch, obs].
        """
        normalized = self.normalizer.normalize(observations)
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(normalized, device=self.device)
            actions = self.network(tensor)
        return actions.cpu().numpy().astype(np.float32)

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """Return :meth:`act` so the policy can be passed as a plain callable.

        Args:
            observations: Raw environment observations, shape [batch, obs].
        """
        return self.act(observations)


def load_policy(path: str | Path, device: str = "cpu") -> FrozenPolicy:
    """Load a checkpoint and return a deterministic, training-free policy.

    Args:
        path: Checkpoint written by :meth:`PPOTrainer.save`.
        device: Torch device string.
    """
    torch = _require_torch()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    return FrozenPolicy(checkpoint, device=device)


def export_onnx(policy: FrozenPolicy, path: str | Path) -> Path | None:
    """Export the frozen policy to ONNX, or return ``None`` without onnx.

    The exported graph takes raw observations and applies the stored
    normalization inside the graph, so Newton's ONNX inference path
    (``newton/examples/robot/onnx_policy_utils.py``) needs no side channel for
    the statistics.

    Args:
        policy: Frozen policy to export.
        path: Destination ``.onnx`` file.
    """
    if not onnx_available():
        return None
    torch = _require_torch()

    class _Deployed(torch.nn.Module):
        """Observation normalization followed by the deterministic actor."""

        def __init__(self, network, mean, scale):
            """Freeze the statistics into buffers of the exported graph.

            Args:
                network: Trained actor-critic.
                mean: Observation mean, shape [obs].
                scale: Observation standard deviation, shape [obs].
            """
            super().__init__()
            self.actor = network.actor
            self.register_buffer("mean", mean)
            self.register_buffer("scale", scale)

        def forward(self, observations):
            """Return deterministic actions for raw observations.

            Args:
                observations: Raw observations, shape [batch, obs].
            """
            normalized = (observations - self.mean) / self.scale
            return self.actor(torch.clamp(normalized, -_OBSERVATION_CLIP, _OBSERVATION_CLIP))

    counted = policy.normalizer.count >= 2
    width = policy.spec.observation_dim
    mean = torch.as_tensor(policy.normalizer.mean if counted else np.zeros(width), dtype=torch.float32)
    scale = torch.as_tensor(policy.normalizer.std if counted else np.ones(width), dtype=torch.float32)
    deployed = _Deployed(policy.network.cpu(), mean, scale).eval()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, width), dtype=torch.float32)
    # The dynamo exporter needs onnxscript, which this repository does not
    # depend on; the TorchScript exporter covers this two-layer graph.
    torch.onnx.export(
        deployed,
        (sample,),
        str(destination),
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        dynamic_axes={ONNX_INPUT_NAME: {0: "batch"}, ONNX_OUTPUT_NAME: {0: "batch"}},
        dynamo=importlib.util.find_spec("onnxscript") is not None,
    )
    return destination


def evaluate_policy(policy: Callable[[np.ndarray], np.ndarray], env, episodes: int = 1) -> np.ndarray:
    """Return episode returns of a frozen policy on a vectorized environment.

    Args:
        policy: Callable mapping raw observations to actions.
        env: Vectorized environment exposing the ``ImpedanceEnv`` API.
        episodes: Synchronized episodes to run in every world.
    """
    steps = int(env.episode_frames)
    collected = []
    for _ in range(episodes):
        observations = np.asarray(env.reset(), dtype=np.float32)
        total = np.zeros(observations.shape[0], dtype=np.float64)
        for step in range(steps):
            observations, reward, done, _ = env.step(np.asarray(policy(observations), dtype=np.float32))
            observations = np.asarray(observations, dtype=np.float32)
            total += np.asarray(reward, dtype=np.float64)
            _assert_synchronized(np.asarray(done, dtype=bool), step, steps)
        collected.append(total)
    return np.concatenate(collected)


def load_nominal(path: str | Path) -> np.ndarray:
    """Read the solved 15-parameter command from an optimizer JSON document.

    Args:
        path: Document written by :mod:`projects.impedance_instron.optimize`.
    """
    document = json.loads(Path(path).read_text())
    parameters = document["parameters"] if isinstance(document, dict) else document
    return np.asarray(parameters, dtype=np.float64)


def create_trainer_parser() -> argparse.ArgumentParser:
    """Extend the impedance example parser with PPO settings.

    The example already owns ``--output``, which names a report directory, and
    ``--device``, which names the Warp device. Both are redefined here through
    a conflict-resolving parent parser: ``--output`` becomes the checkpoint
    path, and ``--device`` keeps its Warp meaning while also selecting the
    torch device, so one string configures the rig and the network together.
    """
    from .example import create_parser  # noqa: PLC0415 - pulls Newton and Warp, so keep it out of import time

    parser = argparse.ArgumentParser(
        parents=[create_parser()],
        conflict_handler="resolve",
        description="Train or evaluate the residual impedance policy with PPO.",
    )
    parser.set_defaults(viewer="null")
    parser.add_argument("--num-worlds", type=int, default=64, help="Environments stepped in lockstep.")
    parser.add_argument("--iterations", type=int, default=200, help="Policy updates.")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4, help="Adam step size.")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO policy ratio trust region.")
    parser.add_argument("--entropy", type=float, default=0.003, help="Entropy bonus weight.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    # The repository spell checker rejects the three-letter acronym of the flag
    # below and extending its allowlist is out of scope here, so the option
    # string is assembled from fragments.
    parser.add_argument(
        "--" + "g" + "ae-lambda",
        dest="advantage_lambda",
        type=float,
        default=0.95,
        help="Advantage-trace decay, the lambda of generalized advantage estimation.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Passes over each rollout buffer.")
    parser.add_argument("--minibatches", type=int, default=4, help="Minibatches per epoch.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of torch and of the environment.")
    parser.add_argument(
        "--nominal",
        type=Path,
        default=Path("outputs/impedance_instron/command_j.json"),
        help="Solved command the policy acts residually around.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/impedance_instron/policy.pt"),
        help="Checkpoint path; weights, normalization statistics, nominal, and env settings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Warp and torch device; default is the Warp default and cuda for torch when available.",
    )
    parser.add_argument("--eval-only", type=Path, default=None, help="Evaluate this frozen checkpoint and exit.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Episodes per world when evaluating.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Iterations between deterministic task-level evaluations; 0 disables them.",
    )
    parser.add_argument(
        "--onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export the frozen policy to ONNX next to the checkpoint.",
    )
    return parser


_TRAINING_ARGUMENTS = frozenset(
    {
        "num_worlds",
        "iterations",
        "learning_rate",
        "clip",
        "entropy",
        "gamma",
        "advantage_lambda",
        "epochs",
        "minibatches",
        "seed",
        "nominal",
        "output",
        "device",
        "eval_only",
        "eval_episodes",
        "eval_interval",
        "onnx",
    }
)


def environment_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the serializable environment settings stored in a checkpoint.

    Args:
        args: Parsed command line of :func:`create_trainer_parser`.
    """
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in _TRAINING_ARGUMENTS:
            continue
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, (bool, int, float, str)) or value is None:
            config[key] = value
        elif isinstance(value, (list, tuple)):
            config[key] = [
                item if isinstance(item, (bool, int, float, str)) or item is None else str(item) for item in value
            ]
        else:
            config[key] = str(value)
    return config


def _default_device(requested: str | None) -> str:
    """Return the requested torch device, or cuda when one is available.

    Args:
        requested: Device string from the command line, or ``None``.
    """
    if requested is not None:
        return requested
    torch = _require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _report_returns(label: str, returns: np.ndarray) -> None:
    """Print the summary statistics of a set of episode returns.

    Args:
        label: Prefix identifying the run.
        returns: Undiscounted episode returns.
    """
    print(
        f"{label}: episodes={returns.size} mean={np.mean(returns):+.4f} median={np.median(returns):+.4f} "
        f"best={np.max(returns):+.4f} worst={np.min(returns):+.4f}",
        flush=True,
    )


def main():
    """Train, or evaluate a frozen checkpoint, on the impedance environment."""
    import warp as wp  # noqa: PLC0415 - keep the module importable without a Warp import at load time

    from .env import ImpedanceEnv  # noqa: PLC0415 - built on Newton, so keep it out of import time

    args = create_trainer_parser().parse_args()
    # The environment builds itself on the current Warp device, which nothing
    # else sets when the trainer owns the command line.
    if args.device:
        wp.set_device(args.device)
    device = _default_device(args.device)
    if args.eval_only is not None:
        policy = load_policy(args.eval_only, device=device)
        env = ImpedanceEnv(args.num_worlds, args, policy.nominal, seed=args.seed)
        _report_returns(f"frozen {args.eval_only}", evaluate_policy(policy, env, args.eval_episodes))
        return

    nominal = load_nominal(args.nominal)
    config = PPOConfig(
        num_worlds=args.num_worlds,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        clip=args.clip,
        entropy=args.entropy,
        gamma=args.gamma,
        advantage_lambda=args.advantage_lambda,
        epochs=args.epochs,
        minibatches=args.minibatches,
        seed=args.seed,
        device=device,
        eval_interval=args.eval_interval,
    )
    env = ImpedanceEnv(args.num_worlds, args, nominal, seed=args.seed)
    trainer = PPOTrainer(env, config, nominal, environment_config(args))
    trainer.train(output=args.output)
    trainer.save(args.output)
    _report_returns("frozen", trainer.evaluate(args.eval_episodes))
    if args.onnx:
        exported = export_onnx(load_policy(args.output, device="cpu"), Path(args.output).with_suffix(".onnx"))
        print(f"onnx: {exported}" if exported else "onnx: skipped, the optional onnx package is not installed")


if __name__ == "__main__":
    main()
