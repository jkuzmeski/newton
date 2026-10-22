# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learned differentiable dynamics of the runner, conditioned on the sim context.

A deterministic MLP in the style of SuperMimic (§3.2) predicts the next sim state
and the aux outputs of one control step from the state, the action, the sim
context ``xi`` and the last ``history_length`` states and actions. The foam under
each foot has rate-dependent memory that the current state does not reveal; the
history lets the model infer it.

A state row holds ``root_pos`` (3) [m], ``root_quat`` (4, xyzw), ``root_lin_vel``
(3) [m/s] and ``root_ang_vel`` (3) [rad/s] in world frame, then ``q`` and ``qd``
of the ``joint_count`` joints [rad, rad/s]. The model ignores the horizontal root
position, which does not affect the dynamics on flat ground, and predicts the
step change: position and velocity increments, and the world-frame rotation
vector that takes the root orientation to the next one.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["WorldModel"]


def _quaternion_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tables that map the outer product ``a b^T`` of xyzw quaternions,
    flattened to 16 entries, to the product ``a b`` and, for unit ``a = b``, to
    the row-major rotation matrix."""
    product = torch.zeros(4, 4, 4)
    for i, j, k, sign in (
        (3, 0, 0, 1), (0, 3, 0, 1), (1, 2, 0, 1), (2, 1, 0, -1),
        (3, 1, 1, 1), (0, 2, 1, -1), (1, 3, 1, 1), (2, 0, 1, 1),
        (3, 2, 2, 1), (0, 1, 2, 1), (1, 0, 2, -1), (2, 3, 2, 1),
        (3, 3, 3, 1), (0, 0, 3, -1), (1, 1, 3, -1), (2, 2, 3, -1),
    ):  # fmt: skip
        product[i, j, k] = sign
    rotation = torch.zeros(4, 4, 9)
    for entry, terms in enumerate(
        (
            ((0, 0, 1), (1, 1, -1), (2, 2, -1), (3, 3, 1)),
            ((0, 1, 2), (2, 3, -2)),
            ((0, 2, 2), (1, 3, 2)),
            ((0, 1, 2), (2, 3, 2)),
            ((0, 0, -1), (1, 1, 1), (2, 2, -1), (3, 3, 1)),
            ((1, 2, 2), (0, 3, -2)),
            ((0, 2, 2), (1, 3, -2)),
            ((1, 2, 2), (0, 3, 2)),
            ((0, 0, -1), (1, 1, -1), (2, 2, 1), (3, 3, 1)),
        )
    ):
        for i, j, value in terms:
            rotation[i, j, entry] = value
    return product.reshape(16, 4), rotation.reshape(16, 9)


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[..., :, None] * b[..., None, :]).flatten(-2)


def _quat_exp(v: torch.Tensor) -> torch.Tensor:
    # The epsilon keeps the gradient finite at zero rotation.
    angle = torch.sqrt((v * v).sum(-1, keepdim=True) + 1.0e-12)
    return torch.cat((v * (torch.sin(0.5 * angle) / angle), torch.cos(0.5 * angle)), -1)


def _quat_log(q: torch.Tensor) -> torch.Tensor:
    q = torch.where(q[..., 3:] < 0.0, -q, q)
    norm = torch.sqrt((q[..., :3] * q[..., :3]).sum(-1, keepdim=True) + 1.0e-12)
    return q[..., :3] * (2.0 * torch.atan2(norm, q[..., 3:]) / norm)


class WorldModel(nn.Module):
    """Deterministic MLP dynamics model of the runner.

    The model carries a Markov *model state*: the current sim state followed by
    the ``history_length`` previous states and then their actions, oldest first,
    flattened to ``[batch, model_state_dim]``. The command filter of the sim is a
    function of past actions, so a few steps of history also cover it.
    :meth:`start` builds a model state from logged steps.

    Call :meth:`fit_normalization` once on logged data before training. Windows
    for :meth:`fit_normalization` and :meth:`loss` hold consecutive control steps
    of one episode: ``states`` ``[batch, L + 1, state_dim]``, ``actions``
    ``[batch, L, joint_count, 3]`` and ``aux`` ``[batch, L, aux_dim]``, where step
    ``t`` takes ``states[:, t]`` with ``actions[:, t]`` to ``states[:, t + 1]``
    and logs ``aux[:, t]``; ``xi`` is ``[batch, xi_dim]``. Steps before an
    episode start can be filled with its first state and first action; the sim
    passes the first action through its filter unchanged.

    An action holds the equilibrium angle [rad], stiffness [N m/rad] and damping
    [N m s/rad] of each joint, shape ``[..., joint_count, 3]``, before the sim's
    command filter.

    Args:
        joint_count: Number of actuated joints ``n``; the state has ``13 + 2 n`` entries.
        xi_dim: Sim context size.
        aux_dim: Size of the aux outputs of a step, such as ground reaction forces,
            joint torques and joint work.
        history_length: Number of past states and actions the model sees.
        hidden: Hidden layer widths of the MLP.
    """

    def __init__(
        self,
        joint_count: int,
        xi_dim: int,
        aux_dim: int,
        *,
        history_length: int = 4,
        hidden: tuple[int, ...] = (2048, 2048, 2048),
    ):
        super().__init__()
        self.state_dim = 13 + 2 * joint_count
        self.joint_count = joint_count
        action_dim = 3 * joint_count
        self.history_length = history_length
        self.model_state_dim = (history_length + 1) * self.state_dim + history_length * action_dim
        feature_dim = 16 + 2 * joint_count
        change_dim = 12 + 2 * joint_count
        layers = []
        width = (history_length + 1) * feature_dim + history_length * action_dim + action_dim + xi_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SiLU()]
            width = size
        output = nn.Linear(width, change_dim + aux_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.net = nn.Sequential(*layers, output)
        product, rotation = _quaternion_tables()
        self.register_buffer("quat_product", product, persistent=False)
        self.register_buffer("quat_rotation", rotation, persistent=False)
        for name, size in (
            ("feature", feature_dim),
            ("action", action_dim),
            ("xi", xi_dim),
            ("change", change_dim),
            ("aux", aux_dim),
        ):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_std", torch.ones(size))

    def _features(self, state: torch.Tensor) -> torch.Tensor:
        quat = state[..., 3:7]
        return torch.cat((state[..., 2:3], _outer(quat, quat) @ self.quat_rotation, state[..., 7:]), -1)

    def _difference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the step change from state ``b`` to state ``a``."""
        inverse = torch.cat((-b[..., 3:6], b[..., 6:7]), -1)
        rotation = _quat_log(_outer(a[..., 3:7], inverse) @ self.quat_product)
        return torch.cat((a[..., :3] - b[..., :3], rotation, a[..., 7:] - b[..., 7:]), -1)

    def _advance(self, state: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
        quat = _outer(_quat_exp(change[..., 3:6]), state[..., 3:7]) @ self.quat_product
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        return torch.cat((state[..., :3] + change[..., :3], quat, state[..., 7:] + change[..., 6:]), -1)

    @torch.no_grad()
    def fit_normalization(
        self, states: torch.Tensor, actions: torch.Tensor, xi: torch.Tensor, aux: torch.Tensor
    ) -> None:
        """Set the input and output scales from logged windows."""
        for name, values in (
            ("feature", self._features(states)),
            ("action", actions.flatten(2)),
            ("xi", xi),
            ("change", self._difference(states[:, 1:], states[:, :-1])),
            ("aux", aux),
        ):
            rows = values.reshape(-1, values.shape[-1])
            getattr(self, f"{name}_mean").copy_(rows.mean(0))
            getattr(self, f"{name}_std").copy_(rows.std(0).clamp_min(1.0e-6))

    def start(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Build model states from logged steps.

        Args:
            states: The last ``history_length + 1`` states, oldest first, shape
                ``[batch, history_length + 1, state_dim]``.
            actions: The actions of the past steps, shape ``[batch, history_length, joint_count, 3]``.

        Returns:
            Model states, shape ``[batch, model_state_dim]``.
        """
        return torch.cat((states[:, -1], states[:, :-1].flatten(1), actions.flatten(1)), -1)

    def forward(
        self, model_state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one control step.

        Args:
            model_state: Model state, shape ``[batch, model_state_dim]``; the sim
                state is ``model_state[:, :state_dim]``.
            action: Action of the step, shape ``[batch, joint_count, 3]``.
            xi: Sim context, shape ``[batch, xi_dim]``.

        Returns:
            The next model state ``[batch, model_state_dim]`` and the step's aux
            outputs ``[batch, aux_dim]``.
        """
        s, h, a = self.state_dim, self.history_length, 3 * self.joint_count
        action = action.flatten(1)
        state = model_state[:, :s]
        past_states = model_state[:, s : (h + 1) * s].unflatten(1, (h, s))
        past_actions = model_state[:, (h + 1) * s :].unflatten(1, (h, a))
        states = torch.cat((past_states, state[:, None]), 1)
        x = torch.cat(
            (
                ((self._features(states) - self.feature_mean) / self.feature_std).flatten(1),
                ((torch.cat((past_actions, action[:, None]), 1) - self.action_mean) / self.action_std).flatten(1),
                (xi - self.xi_mean) / self.xi_std,
            ),
            -1,
        )
        out = self.net(x)
        split = self.change_mean.shape[0]
        change = out[:, :split] * self.change_std + self.change_mean
        aux = out[:, split:] * self.aux_std + self.aux_mean
        next_state = self._advance(state, change)
        if h:
            next_state = torch.cat(
                (
                    next_state,
                    model_state[:, 2 * s : (h + 1) * s],
                    state,
                    model_state[:, (h + 1) * s + a :],
                    action,
                ),
                -1,
            )
        return next_state, aux

    def rollout(
        self, model_state: torch.Tensor, actions: torch.Tensor, xi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll the model out autoregressively on an action sequence.

        The result is differentiable with respect to the start model state, the
        actions and ``xi``.

        Args:
            model_state: Start model state, shape ``[batch, model_state_dim]``.
            actions: Actions, shape ``[batch, horizon, joint_count, 3]``.
            xi: Sim context, shape ``[batch, xi_dim]``.

        Returns:
            Sim states ``[batch, horizon + 1, state_dim]`` starting with the start
            state, and aux outputs ``[batch, horizon, aux_dim]``.
        """
        states, auxes = [model_state[:, : self.state_dim]], []
        for t in range(actions.shape[1]):
            model_state, aux = self(model_state, actions[:, t], xi)
            states.append(model_state[:, : self.state_dim])
            auxes.append(aux)
        return torch.stack(states, 1), torch.stack(auxes, 1)

    def loss(self, states: torch.Tensor, actions: torch.Tensor, xi: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Return the k-step autoregressive loss on logged windows.

        The first ``history_length`` steps of each window seed the history; the
        model then rolls out the remaining ``k`` steps on the logged actions from
        the logged state. State errors are scaled by the spread of one-step changes
        and aux errors by the spread of the aux outputs.
        """
        h = self.history_length
        predicted, predicted_aux = self.rollout(self.start(states[:, : h + 1], actions[:, :h]), actions[:, h:], xi)
        state_error = self._difference(predicted[:, 1:], states[:, h + 1 :]) / self.change_std
        aux_error = (predicted_aux - aux[:, h:]) / self.aux_std
        return state_error.square().mean() + aux_error.square().mean()
