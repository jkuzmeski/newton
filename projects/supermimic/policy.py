# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic actor with bounded outputs and a value critic for the SuperMimic loop.

Learning code only: the networks take dimensions and bounds as arguments and do
not depend on Newton.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Actor", "Critic", "RunningNormalizer", "impedance_action_bounds"]


class RunningNormalizer(nn.Module):
    """Standardize inputs with a running mean and variance.

    Args:
        size: Input width.
        clip: Bound on the standardized value.
    """

    def __init__(self, size: int, *, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Merge the statistics of a batch ``x`` of shape [batch, size]."""
        x = x.reshape(-1, self.mean.shape[0])
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        count = x.shape[0]
        total = self.count + count
        delta = batch_mean - self.mean
        self.mean += delta * count / total
        self.var.copy_((self.var * self.count + batch_var * count + delta**2 * self.count * count / total) / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def _mlp(inputs: int, hidden: tuple[int, ...], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(inputs, width), nn.ELU()]
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


def impedance_action_bounds(
    q_eq_low: torch.Tensor,
    q_eq_high: torch.Tensor,
    stiffness: tuple[float, float],
    damping: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the :class:`Actor` bounds of the impedance action ``(q_eq, K, B)`` per joint.

    Stiffness and damping are bounded on a log scale, so they stay positive and
    their resolution is relative.

    Args:
        q_eq_low: Lower equilibrium angle per joint [rad], shape [joint_count].
        q_eq_high: Upper equilibrium angle per joint [rad], shape [joint_count].
        stiffness: Stiffness range ``(min, max)`` for every joint [N·m/rad].
        damping: Damping range ``(min, max)`` for every joint [N·m·s/rad].

    Returns:
        ``(low, high, log_scale)``, each of shape [joint_count, 3].
    """
    ones = torch.ones_like(q_eq_low)
    low = torch.stack([q_eq_low, stiffness[0] * ones, damping[0] * ones], dim=-1)
    high = torch.stack([q_eq_high, stiffness[1] * ones, damping[1] * ones], dim=-1)
    log_scale = torch.tensor([False, True, True], device=q_eq_low.device).expand_as(low)
    return low, high, log_scale


class Actor(nn.Module):
    """Deterministic MLP policy whose outputs lie between per-dimension bounds.

    Each output is ``low + (high - low) * sigmoid(u)`` of a network output ``u``;
    dimensions flagged in ``log_scale`` apply the same map to ``log(low)`` and
    ``log(high)`` and exponentiate, which keeps them strictly positive. The last
    layer starts near zero, so the initial action is the middle of the bounds.

    Args:
        observation_size: Input width.
        low: Lower action bound; its shape is the action shape.
        high: Upper action bound, same shape as ``low``.
        log_scale: Entries bounded on a log scale, same shape as ``low``; their
            bounds must be positive.
        hidden: Hidden layer widths.
    """

    def __init__(
        self,
        observation_size: int,
        low: torch.Tensor,
        high: torch.Tensor,
        log_scale: torch.Tensor,
        *,
        hidden: tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()
        log_scale = log_scale.bool()
        if bool((low[log_scale] <= 0).any()):
            raise ValueError("log-scale action bounds must be positive")
        self.action_shape = tuple(low.shape)
        self.register_buffer("log_scale", log_scale.flatten())
        self.register_buffer("lower", torch.where(log_scale, low.log(), low).flatten().float())
        self.register_buffer("span", (torch.where(log_scale, high.log(), high).flatten() - self.lower).float())
        self.normalizer = RunningNormalizer(observation_size)
        self.net = _mlp(observation_size, hidden, low.numel())
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def squash(self, u: torch.Tensor) -> torch.Tensor:
        """Map flat unbounded network outputs ``u`` to bounded actions of the action shape."""
        a = self.lower + self.span * torch.sigmoid(u)
        return torch.where(self.log_scale, a.exp(), a).unflatten(-1, self.action_shape)

    def logits(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the flat unbounded network output for ``observation``."""
        return self.net(self.normalizer(observation))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.squash(self.logits(observation))


class Critic(nn.Module):
    """MLP value function with normalized inputs.

    Args:
        input_size: Input width.
        hidden: Hidden layer widths.
    """

    def __init__(self, input_size: int, *, hidden: tuple[int, ...] = (256, 128)):
        super().__init__()
        self.normalizer = RunningNormalizer(input_size)
        self.net = _mlp(input_size, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(x)).squeeze(-1)
