# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generative running reward of the SuperMimic runner.

The reward asks for running on the treadmill without a reference motion. It
rewards holding the treadmill speed, an upright pelvis and survival, and costs
joint work per meter travelled, the commanded stiffness and damping, and the
change of the commands between steps. Positive and negative joint work have
separate weights.

Velocities are treadmill-frame velocities (see :mod:`.sim`): a runner that holds
the belt speed has zero mean forward velocity. Every term is a rate multiplied
by the control step, so a return sums to the time integral of the rates. The
work rate is the joint work of a step divided by the step and the belt speed,
so its time integral over an episode is the episode duration times the work per
meter travelled. Positive and negative work come from the simulator, which
splits them at every substep.

Inputs are torch tensors with any leading batch shape. An action has the layout
``[..., joint, 3]`` of :class:`.sim.RunnerSim`: equilibrium angle [rad],
stiffness [N m/rad] and damping [N m s/rad] per actuated joint, before the
command filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["GenerativeCostConfig", "generative_reward", "generative_terminated", "travel_distance"]


@dataclass(frozen=True)
class GenerativeCostConfig:
    """Weights of the generative reward and its termination thresholds."""

    speed: float = 2.0
    """Weight of the speed reward ``exp(-|v_xy|^2 / speed_sigma^2)`` on the treadmill-frame pelvis velocity."""
    speed_sigma: float = 0.5
    """Width of the speed reward [m/s]."""
    upright: float = 1.0
    """Weight of the z component of the pelvis z axis, 1 when upright."""
    alive: float = 1.0
    """Reward per second of survival."""
    positive_work: float = 0.01
    """Weight of positive joint work per meter travelled [1/(J/m)]."""
    negative_work: float = 0.01
    """Weight of the magnitude of negative joint work per meter travelled [1/(J/m)]."""
    stiffness: float = 1e-4
    """Weight of the sum of commanded stiffnesses [1/(N m/rad)]."""
    damping: float = 1e-3
    """Weight of the sum of commanded dampings [1/(N m s/rad)]."""
    command_rate: tuple[float, float, float] = (1.0, 1e-5, 1e-3)
    """Weights of the squared step change of equilibrium angle, stiffness and damping, summed over joints."""
    height_min: float = 0.6
    """Pelvis height below which an episode terminates [m]."""
    up_min: float = 0.5
    """z component of the pelvis z axis below which an episode terminates."""


def _pelvis_up_z(root_quat: torch.Tensor) -> torch.Tensor:
    """z component of the pelvis z axis from a unit quaternion ``xyzw``."""
    x, y = root_quat[..., 0], root_quat[..., 1]
    return 1.0 - 2.0 * (x * x + y * y)


def generative_reward(
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    joint_work_pos: torch.Tensor,
    joint_work_neg: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    config: GenerativeCostConfig,
    *,
    dt: float,
    belt_speed: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the reward of one control step and its signed terms.

    Args:
        root_quat: Pelvis orientation ``xyzw``, shape ``[..., 4]``.
        root_lin_vel: Treadmill-frame pelvis velocity [m/s], shape ``[..., 3]``.
        joint_work_pos: Positive joint work over the step [J], shape ``[..., 10]``.
        joint_work_neg: Negative joint work over the step [J], non-positive, shape ``[..., 10]``.
        action: Commands of the step, shape ``[..., 10, 3]``.
        prev_action: Commands of the previous step, shape ``[..., 10, 3]``.
        config: Weights and thresholds.
        dt: Control step [s].
        belt_speed: Treadmill belt speed [m/s].

    Returns:
        The reward, shape ``[...]``, and its terms by name; the terms sum to the reward.
    """
    speed = torch.exp(-root_lin_vel[..., :2].square().sum(-1) / config.speed_sigma**2)
    rate = (action - prev_action).square().sum(-2) * action.new_tensor(config.command_rate)
    terms = {
        "speed": config.speed * speed,
        "upright": config.upright * _pelvis_up_z(root_quat),
        "alive": torch.full_like(speed, config.alive),
        "positive_work": -config.positive_work / (dt * belt_speed) * joint_work_pos.sum(-1),
        "negative_work": config.negative_work / (dt * belt_speed) * joint_work_neg.sum(-1),
        "stiffness": -config.stiffness * action[..., 1].sum(-1),
        "damping": -config.damping * action[..., 2].sum(-1),
        "command_rate": -rate.sum(-1),
    }
    terms = {name: dt * value for name, value in terms.items()}
    return sum(terms.values()), terms


def generative_terminated(
    root_pos: torch.Tensor, root_quat: torch.Tensor, config: GenerativeCostConfig
) -> torch.Tensor:
    """Return whether the pelvis fell below ``height_min`` or tilted past ``up_min``, shape ``[...]``."""
    return (root_pos[..., 2] < config.height_min) | (_pelvis_up_z(root_quat) < config.up_min)


def travel_distance(root_lin_vel: torch.Tensor, dt: float, belt_speed: float) -> torch.Tensor:
    """Return the distance travelled over the belt [m], shape ``[...]``.

    Args:
        root_lin_vel: Treadmill-frame pelvis velocity per step [m/s], shape ``[..., time, 3]``.
        dt: Control step [s].
        belt_speed: Treadmill belt speed [m/s].
    """
    return dt * (root_lin_vel[..., 0] + belt_speed).sum(-1)
