# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Neural impedance policy: proprioceptive MLP mapping to bounded joint impedance.

The actor sees only :data:`.env.OBSERVATION_NAMES`. Its 18 outputs are logits
that :meth:`.env.BilateralEnv.set_impedance` maps through sigmoids to bounded
equilibrium, stiffness, and damping per joint. The critic additionally sees
reference context during training; it is not part of a frozen policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..hogan.runner import Bounds
from .env import ACTION_NAMES, OBSERVATION_NAMES

SCHEMA = "bilateral_neural_impedance_policy_1"


def seed_action(bounds: Bounds | None = None) -> np.ndarray:
    """Return engineering seed logits, the same impedance as :meth:`..hogan.runner.Runner.seed`.

    This is an unfitted initialization, not a fitted controller.
    """
    bounds = bounds or Bounds()
    lower = np.asarray(bounds.equilibrium_lower_rad)
    span = np.asarray(bounds.equilibrium_upper_rad) - lower
    fractions = np.concatenate(
        [
            np.tile((np.array([0.2, -0.5, 0.0]) - lower) / span, 2),
            np.tile(np.array([250.0, 300.0, 200.0]) / bounds.stiffness_max_nm_rad, 2),
            np.tile(np.array([12.0, 10.0, 6.0]) / bounds.damping_max_nms_rad, 2),
        ]
    )
    return np.log(fractions / (1 - fractions))


class RunningNormalizer(nn.Module):
    """Track a running mean and variance and standardize inputs."""

    def __init__(self, size: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var.copy_(
            (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        )
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def _mlp(inputs: int, hidden: tuple[int, ...], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    size = inputs
    for width in hidden:
        layers += [nn.Linear(size, width), nn.ELU()]
        size = width
    layers.append(nn.Linear(size, outputs))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy mean with normalized proprioceptive inputs.

    Args:
        hidden: Hidden layer widths.
        seed_logits: Initial output bias; defaults to :func:`seed_action`.
        observation_names: Policy inputs; defaults to the planar environment's.
        action_names: Policy outputs; defaults to the planar environment's.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (256, 256),
        seed_logits: np.ndarray | None = None,
        observation_names: tuple[str, ...] = OBSERVATION_NAMES,
        action_names: tuple[str, ...] = ACTION_NAMES,
    ):
        super().__init__()
        self.hidden = tuple(hidden)
        self.observation_names = tuple(observation_names)
        self.action_names = tuple(action_names)
        self.normalizer = RunningNormalizer(len(self.observation_names))
        self.net = _mlp(len(self.observation_names), self.hidden, len(self.action_names))
        last = self.net[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            last.bias.copy_(torch.as_tensor(seed_logits if seed_logits is not None else seed_action()))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(observation))


class ActorCritic(nn.Module):
    """Gaussian actor in logit space and an asymmetric critic."""

    def __init__(
        self,
        privileged_size: int,
        hidden: tuple[int, ...] = (256, 256),
        log_std: float = -1.0,
        *,
        observation_names: tuple[str, ...] = OBSERVATION_NAMES,
        action_names: tuple[str, ...] = ACTION_NAMES,
        seed_logits: np.ndarray | None = None,
    ):
        super().__init__()
        self.actor = Actor(hidden, seed_logits, observation_names, action_names)
        self.log_std = nn.Parameter(torch.full((len(action_names),), float(log_std)))
        size = len(observation_names) + privileged_size
        self.critic_normalizer = RunningNormalizer(size)
        self.critic = _mlp(size, hidden, 1)

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observation)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, observation: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_normalizer(torch.cat((observation, privileged), dim=-1))).squeeze(-1)


def save_policy(actor: Actor, path: str | Path, metadata: dict) -> Path:
    """Write a frozen deterministic policy: weights plus JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": actor.state_dict(), "hidden": list(actor.hidden)}, path)
    meta = {
        "schema": SCHEMA,
        "observations": list(actor.observation_names),
        "actions": list(actor.action_names),
        "hidden": list(actor.hidden),
        "deterministic": "mean action; no exploration noise",
        **metadata,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_policy(
    path: str | Path,
    device: str = "cuda:0",
    observation_names: tuple[str, ...] = OBSERVATION_NAMES,
    action_names: tuple[str, ...] = ACTION_NAMES,
) -> tuple[Actor, dict]:
    """Load a frozen policy and verify it matches the environment's interface."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta["observations"] != list(observation_names):
        raise ValueError("Unsupported policy schema or observation interface")
    if meta["actions"] != list(action_names):
        raise ValueError("Unsupported policy action interface")
    payload = torch.load(path, map_location=device, weights_only=True)
    actor = Actor(
        tuple(payload["hidden"]), np.zeros(len(action_names)), tuple(observation_names), tuple(action_names)
    ).to(device)
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    return actor, meta


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def logits_to_impedance(logits: np.ndarray, bounds: Bounds | None = None) -> dict[str, np.ndarray]:
    """Map logits to equilibrium [rad], stiffness [N m/rad], damping [N m s/rad] per joint."""
    bounds = bounds or Bounds()
    fraction = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=float), -40, 40)))
    lower = np.tile(bounds.equilibrium_lower_rad, 2)
    upper = np.tile(bounds.equilibrium_upper_rad, 2)
    return {
        "equilibrium_rad": lower + (upper - lower) * fraction[..., :6],
        "stiffness_nm_rad": np.tile(bounds.stiffness_max_nm_rad, 2) * fraction[..., 6:12],
        "damping_nms_rad": np.tile(bounds.damping_max_nms_rad, 2) * fraction[..., 12:],
    }
