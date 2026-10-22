# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic actor with bounded outputs and a value critic for the SuperMimic loop.

Learning code only: the networks take dimensions and bounds as arguments and do
not depend on Newton.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["DAMPING_LIMITS", "STIFFNESS_LIMITS", "Actor", "Critic", "RunningNormalizer", "impedance_action_bounds"]


# Short-time human joint impedance bounds (R3 of whitepapers/supermimic_neuromech_params.md):
# K_max = 4.8 rad^-1 times the joint torque capacity, B_max = 0.0046 s times K_max. The source has
# no hip adduction or rotation values; they take the hip flexion values. Lower bounds are the
# loosest measured swing impedance (knee in swing: K 0-3.5 N m/rad, B 0.02-0.14 N m s/rad), with
# K raised from 0 to 1 N m/rad for the log scale.
STIFFNESS_LIMITS = ((1.0, 1127.0), (1.0, 1127.0), (1.0, 1127.0), (1.0, 1141.0), (1.0, 665.0))
"""Lower and upper joint stiffness of the five joints of one side in ``JOINT_NAMES`` order [N·m/rad]; the left side is equal."""

DAMPING_LIMITS = ((0.02, 5.2), (0.02, 5.2), (0.02, 5.2), (0.02, 5.3), (0.02, 3.1))
"""Lower and upper joint damping of the five joints of one side in ``JOINT_NAMES`` order [N·m·s/rad]; the left side is equal."""


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
    joint_limits: tuple[tuple[float, float], ...],
    *,
    stiffness_limits: tuple[tuple[float, float], ...] = STIFFNESS_LIMITS,
    damping_limits: tuple[tuple[float, float], ...] = DAMPING_LIMITS,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the :class:`Actor` bounds of the impedance action ``(q_eq, K, B)`` per joint of both sides.

    Each limit table holds ``(lower, upper)`` for the joints of one side; the
    other side is equal. Stiffness and damping are bounded on a log scale, so
    they stay positive and their resolution is relative.

    Args:
        joint_limits: Equilibrium angle limits [rad], e.g. ``BodyParams.joint_limits``.
        stiffness_limits: Stiffness limits [N·m/rad].
        damping_limits: Damping limits [N·m·s/rad].
        device: Torch device of the bounds.

    Returns:
        ``(low, high, log_scale)``, each of shape [2 * joints, 3].
    """
    limits = torch.tensor((joint_limits, stiffness_limits, damping_limits), dtype=torch.float32, device=device)
    limits = limits.repeat(1, 2, 1).permute(1, 0, 2)
    log_scale = torch.tensor([False, True, True], device=device).expand(limits.shape[:2])
    return limits[..., 0], limits[..., 1], log_scale


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
