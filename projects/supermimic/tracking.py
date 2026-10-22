# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable F01 tracking reward and termination for SuperMimic.

States and reference frames share one layout per world (:data:`STATE_DIM`):
root position [m], root quaternion (xyzw), root linear and angular velocity in
the world frame [m/s, rad/s], the ten joint angles [rad] and joint velocities
[rad/s] in the order of :data:`projects.supermimic.body.JOINT_NAMES`.

The reward follows the BeyondMimic motion-tracking reward as used by SuperMimic
(paper Table A.2): a sum of weighted terms ``exp(-e^2 / sigma^2)`` with the
paper's weights and scales, mapped onto this body. Joint angles and velocities
stand in for the paper's body orientations and angular velocities, and the
foot positions (ankle centers) for its body positions. Body positions are
compared in each root's heading frame and the root is compared by height
only, so the reward does not depend on horizontal position or treadmill frame.
The joint-limit penalty is omitted because the body has no joint limits.
:class:`ADDiscriminator` replaces the hand weights with a learned
discriminator on the same error vector (Zhang et al. 2025, arXiv 2505.04961).

States and references must be in the same frame; the simulator reports the
treadmill frame, so the reference root velocity must exclude the belt speed.
All functions are batch-first torch code and differentiable in the state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .body import BodyParams

__all__ = [
    "ACTION_SMOOTHNESS_WEIGHT",
    "STATE_DIM",
    "TRACKING_ERROR_DIM",
    "TRACKING_ERROR_SLICES",
    "ADDiscriminator",
    "LegKinematics",
    "step_reward",
    "termination",
    "termination_probability",
    "tracking_errors",
    "tracking_reward",
]

STATE_DIM = 33
"""Root position (3), quaternion (4), linear velocity (3), angular velocity (3), joint angles (10) and velocities (10)."""

TRACKING_ERROR_SLICES = {
    "joint_angle": slice(0, 10),
    "joint_velocity": slice(10, 20),
    "root_height": slice(20, 21),
    "root_orientation": slice(21, 24),
    "root_linear_velocity": slice(24, 27),
    "root_angular_velocity": slice(27, 30),
    "foot_position": slice(30, 36),
}
"""Components of the error vector of :func:`tracking_errors`."""

TRACKING_ERROR_DIM = 36

# Per term: weight, scale sigma and the number of items the squared error is averaged over.
# Body terms carry weight 1 and root terms 0.5 as in the paper; scales are the paper's
# body position 0.3 m, orientation 0.4 rad, linear velocity 1 m/s and angular velocity 3.14 rad/s.
_TERMS = {
    "joint_angle": (1.0, 0.4, 10),
    "joint_velocity": (1.0, 3.14, 10),
    "foot_position": (1.0, 0.3, 2),
    "root_linear_velocity": (1.0, 1.0, 1),
    "root_height": (0.5, 0.3, 1),
    "root_orientation": (0.5, 0.4, 1),
    "root_angular_velocity": (0.5, 3.14, 1),
}

ACTION_SMOOTHNESS_WEIGHT = 0.1
"""Penalty weight on the squared change of the normalized policy action (paper Table A.2)."""

TERMINATION_HEIGHT = 0.25
"""Root height error that ends an episode [m] (paper Appendix A)."""

TERMINATION_TILT = 0.8
"""Difference of the root's vertical-axis z components that ends an episode, about 78 degrees from upright."""

# Logistic widths of the soft termination at the two thresholds [m] and [-].
_SOFT_HEIGHT_WIDTH = 0.02
_SOFT_TILT_WIDTH = 0.05


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def _quat_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [B, 3, 3] about a unit ``axis`` [3] by ``angle`` [B]."""
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    x, y, z = axis
    k = torch.stack((torch.stack((zero, -z, y)), torch.stack((z, zero, -x)), torch.stack((-y, x, zero))))
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return torch.eye(3, dtype=axis.dtype, device=axis.device) + s * k + (1.0 - c) * (k @ k)


class LegKinematics:
    """Forward kinematics of the ankle centers in torch.

    Args:
        params: Body geometry; only offsets and joint axes are read.
        device: Torch device of the returned positions.
    """

    def __init__(self, params: BodyParams, device: torch.device | str):
        def vec(v, mirror=(1.0, 1.0, 1.0)):
            return torch.tensor(v, dtype=torch.float32, device=device) * torch.tensor(mirror, device=device)

        self._sides = []
        for point, axis in (((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)), ((1.0, -1.0, 1.0), (-1.0, 1.0, -1.0))):
            axes = [vec(a, axis) for a in (*params.hip_axes, params.knee_axis, params.ankle_axis)]
            offsets = [vec(o, point) for o in (params.hip_offset, params.knee_offset, params.ankle_offset)]
            self._sides.append((axes, offsets))

    def foot_positions(self, root_pos: torch.Tensor, root_quat: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Ankle centers [m], shape [B, 2, 3] (right, left), of root positions [B, 3], quaternions [B, 4] and joint angles [B, 10]."""
        root = _quat_matrix(root_quat)
        feet = []
        for i, (axes, (hip, knee, ankle)) in enumerate(self._sides):
            a = q[:, 5 * i : 5 * i + 5]
            thigh = (
                root
                @ _axis_rotation(axes[0], a[:, 0])
                @ _axis_rotation(axes[1], a[:, 1])
                @ _axis_rotation(axes[2], a[:, 2])
            )
            shank = thigh @ _axis_rotation(axes[3], a[:, 3])
            feet.append(root_pos + root @ hip + thigh @ knee + shank @ ankle)
        return torch.stack(feet, dim=1)


def _heading_feet(state: torch.Tensor, kinematics: LegKinematics) -> torch.Tensor:
    """Ankle centers relative to the root in the root's heading (yaw) frame, shape [B, 6]."""
    root_pos, root_quat = state[:, :3], state[:, 3:7]
    relative = kinematics.foot_positions(root_pos, root_quat, state[:, 13:23]) - root_pos[:, None]
    forward = _quat_matrix(root_quat)[:, :, 0]
    yaw = torch.atan2(forward[:, 1], forward[:, 0])[:, None]
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y, z = relative.unbind(-1)
    return torch.stack((c * x + s * y, c * y - s * x, z), -1).flatten(1)


def tracking_errors(state: torch.Tensor, reference: torch.Tensor, kinematics: LegKinematics) -> torch.Tensor:
    """Reference minus state per tracked feature, shape [B, :data:`TRACKING_ERROR_DIM`].

    The root orientation error is ``2 sin(angle / 2)`` times the axis of the
    rotation from the state to the reference, which is smooth at zero error.

    Args:
        state: States, shape [B, :data:`STATE_DIM`].
        reference: Reference frames at the same time, shape [B, :data:`STATE_DIM`].
        kinematics: Forward kinematics of the body.
    """
    relative = _quat_multiply(reference[:, 3:7], _quat_conjugate(state[:, 3:7]))
    orientation = 2.0 * torch.where(relative[:, 3:] < 0.0, -relative[:, :3], relative[:, :3])
    return torch.cat(
        (
            reference[:, 13:33] - state[:, 13:33],
            reference[:, 2:3] - state[:, 2:3],
            orientation,
            reference[:, 7:13] - state[:, 7:13],
            _heading_feet(reference, kinematics) - _heading_feet(state, kinematics),
        ),
        dim=1,
    )


def tracking_reward(errors: torch.Tensor) -> torch.Tensor:
    """Weighted exponential tracking reward [B] of :func:`tracking_errors`; 5.5 at zero error."""
    reward = torch.zeros(errors.shape[0], dtype=errors.dtype, device=errors.device)
    for name, (weight, sigma, count) in _TERMS.items():
        squared = errors[:, TRACKING_ERROR_SLICES[name]].square().sum(-1) / count
        reward = reward + weight * torch.exp(-squared / sigma**2)
    return reward


def step_reward(task_reward: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, dt: float) -> torch.Tensor:
    """Per-step reward [B]: the task reward minus the action-smoothness penalty, times the control period.

    Args:
        task_reward: :func:`tracking_reward` or :meth:`ADDiscriminator.reward`, shape [B].
        action: Policy action in normalized units, shape [B, ...], e.g. [B, 10, 3] for (q_eq, K, B) per joint.
        prev_action: Action of the previous step, same shape.
        dt: Control period [s].
    """
    return dt * (task_reward - ACTION_SMOOTHNESS_WEIGHT * (action - prev_action).flatten(1).square().sum(-1))


def _root_deviation(state: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height = (reference[:, 2] - state[:, 2]).abs()
    tilt = (_quat_matrix(reference[:, 3:7])[:, 2, 2] - _quat_matrix(state[:, 3:7])[:, 2, 2]).abs()
    return height, tilt


def termination(state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Hard termination [B] (bool) on root height or tilt deviation from the reference (paper Appendix A)."""
    height, tilt = _root_deviation(state, reference)
    return (height > TERMINATION_HEIGHT) | (tilt > TERMINATION_TILT)


def termination_probability(state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Differentiable termination probability [B]: logistic steps at the thresholds of :func:`termination`."""
    height, tilt = _root_deviation(state, reference)
    stay = torch.sigmoid((TERMINATION_HEIGHT - height) / _SOFT_HEIGHT_WIDTH) * torch.sigmoid(
        (TERMINATION_TILT - tilt) / _SOFT_TILT_WIDTH
    )
    return 1.0 - stay


class ADDiscriminator(nn.Module):
    """Adversarial differential discriminator on tracking errors (Zhang et al. 2025).

    The only positive sample is the zero error vector and the negatives are
    the policy's errors. The discriminator replaces the hand-weighted sum of
    :func:`tracking_reward` with ``reward = -log(1 - D(errors))``. Inputs are
    divided by a running root mean square of the observed errors so that zero
    stays the positive sample.

    Args:
        error_dim: Length of the error vector.
        hidden: Hidden layer widths (ReLU).
        gradient_penalty: Weight of the squared input gradient of the logit at the negatives.
    """

    def __init__(self, error_dim: int, *, hidden: tuple[int, ...] = (1024, 512), gradient_penalty: float = 1.0):
        super().__init__()
        layers, width = [], error_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))
        self.gradient_penalty = gradient_penalty
        self.register_buffer("error_mean_square", torch.ones(error_dim))
        self.register_buffer("error_count", torch.zeros(()))

    def logit(self, errors: torch.Tensor) -> torch.Tensor:
        """Discriminator logits [B] of error vectors [B, error_dim]."""
        return self.net(errors / torch.sqrt(self.error_mean_square + 1.0e-8)).squeeze(-1)

    def reward(self, errors: torch.Tensor) -> torch.Tensor:
        """Tracking reward ``-log(1 - D(errors))`` [B]; differentiable in ``errors``."""
        return F.softplus(self.logit(errors))

    def loss(self, errors: torch.Tensor) -> torch.Tensor:
        """Discriminator loss on policy errors [B, error_dim]; updates the input scale first."""
        errors = errors.detach()
        with torch.no_grad():
            self.error_count += errors.shape[0]
            self.error_mean_square += (
                (errors.square().mean(0) - self.error_mean_square) * errors.shape[0] / self.error_count
            )
        errors.requires_grad_(True)
        negative = self.logit(errors)
        positive = self.logit(torch.zeros_like(errors[:1]))
        (gradient,) = torch.autograd.grad(negative.sum(), errors, create_graph=True)
        scaled = gradient * torch.sqrt(self.error_mean_square + 1.0e-8)
        return (
            F.softplus(-positive).mean()
            + F.softplus(negative).mean()
            + self.gradient_penalty * scaled.square().sum(-1).mean()
        )
