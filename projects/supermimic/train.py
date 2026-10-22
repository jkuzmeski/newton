# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Policy improvement through a frozen differentiable world model (SuperMimic §3.3).

The actor is trained by backpropagation through ``H`` imagined steps, with a
critic bootstrapping the return beyond the horizon and a soft survival weight in
place of hard termination. :func:`train` alternates collection, world-model fit
and policy improvement through functions passed in by the caller.

Learning code only: no Newton import.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .policy import Actor, Critic

__all__ = [
    "ImaginationHooks",
    "Trainer",
    "TrainerConfig",
    "cosine_lr_scale",
    "imagined_return",
    "lambda_returns",
    "train",
]

Tensor = torch.Tensor


@dataclass(frozen=True)
class ImaginationHooks:
    """Model and task functions for imagined rollouts; all tensors are batch first.

    ``k`` is the per-rollout reference frame index (long, shape [batch]); it
    advances by :attr:`frame_stride` per step.
    """

    dynamics: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
    """Differentiable world model ``(s, a, xi) -> (s_next, aux)``; ``aux`` holds per-step outputs such as contact forces."""
    observe: Callable[[Tensor, Tensor], Tensor]
    """Actor observation ``(s, xi) -> o``, including any sensor noise."""
    goal: Callable[[Tensor], Tensor]
    """Goal ``k -> g`` of shape [batch, goal_size] (``goal_size`` may be 0)."""
    reward: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
    """Reward ``(s, a, s_next, aux, k_next) -> r`` of shape [batch]."""
    termination: Callable[[Tensor, Tensor], Tensor]
    """Differentiable termination probability ``(s_next, k_next) -> d`` in [0, 1], shape [batch]."""
    frame_stride: int = 1
    """Reference frames per step."""


@dataclass(frozen=True)
class TrainerConfig:
    """Policy and critic hyperparameters (paper Table A.5)."""

    horizon: int = 32
    """Imagination horizon ``H``."""
    gamma: float = 0.99
    """Discount per control step."""
    lam: float = 0.95
    """Lambda of the critic targets."""
    actor_lr: float = 2e-4
    """Actor learning rate."""
    critic_lr: float = 2e-4
    """Critic learning rate."""
    actor_grad_norm: float = 1.0
    """Global gradient-norm clip of the actor."""
    actor_grad_value: float = 0.1
    """Element-wise gradient clip of the actor, applied after the norm clip."""
    critic_grad_norm: float = 1.0
    """Global gradient-norm clip of the critic."""
    critic_epochs: int = 5
    """Critic passes over each imagined batch."""
    critic_minibatches: int = 4
    """Shuffled minibatches per critic pass."""
    critic_target_rate: float = 1.0
    """Polyak rate per update of the target critic that gives the lambda targets; 1 makes it the critic."""
    lr_decay_iterations: int = 250_000
    """Updates over which the learning rates decay on a cosine to :attr:`lr_final_scale`, then stay."""
    lr_final_scale: float = 0.03
    """Learning-rate factor at the end of the decay."""


def cosine_lr_scale(update: int, config: TrainerConfig) -> float:
    """Return the learning-rate factor after ``update`` updates (cosine decay, then constant)."""
    progress = min(update / config.lr_decay_iterations, 1.0)
    return config.lr_final_scale + (1.0 - config.lr_final_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


def imagined_return(rewards: Tensor, done: Tensor, value_last: Tensor, gamma: float) -> Tensor:
    """Return the survival-weighted imagined return of Eq. 2 per rollout.

    ``sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H`` with ``c_0 = 1`` and
    ``c_{t+1} = c_t (1 - d_{t+1})``.

    Args:
        rewards: ``r_{t+1}``, shape [batch, H].
        done: Termination probability ``d_{t+1}``, shape [batch, H].
        value_last: Critic value ``V_H``, shape [batch].
        gamma: Discount.

    Returns:
        Return per rollout, shape [batch].
    """
    horizon = rewards.shape[1]
    discount = gamma ** torch.arange(horizon + 1, device=rewards.device, dtype=rewards.dtype)
    survival = torch.cumprod(torch.cat([torch.ones_like(done[:, :1]), 1.0 - done], dim=1), dim=1)
    weighted = discount * survival
    return (weighted[:, :-1] * rewards).sum(1) + weighted[:, -1] * value_last


def lambda_returns(rewards: Tensor, done: Tensor, values: Tensor, gamma: float, lam: float) -> Tensor:
    """Return the critic targets of Eq. 3.

    ``R_t = r_{t+1} + gamma (1 - d_{t+1}) ((1 - lam) V_{t+1} + lam R_{t+1})`` with
    ``R_H = V_H``.

    Args:
        rewards: ``r_{t+1}``, shape [batch, H].
        done: Termination probability ``d_{t+1}``, shape [batch, H].
        values: Critic values ``V_0 .. V_H``, shape [batch, H + 1].
        gamma: Discount.
        lam: Lambda.

    Returns:
        Targets ``R_0 .. R_{H-1}``, shape [batch, H].
    """
    target = values[:, -1]
    targets = []
    for t in reversed(range(rewards.shape[1])):
        target = rewards[:, t] + gamma * (1.0 - done[:, t]) * ((1.0 - lam) * values[:, t + 1] + lam * target)
        targets.append(target)
    return torch.stack(targets[::-1], dim=1)


class Trainer:
    """Actor and critic updates on imagined rollouts through a frozen world model.

    Args:
        actor: Policy; its input is ``observe(s, xi)`` followed by ``goal(k)``.
        critic: Value function; its input is ``s``, ``goal(k)`` and ``xi``.
        config: Hyperparameters.
    """

    def __init__(self, actor: Actor, critic: Critic, config: TrainerConfig):
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(critic).requires_grad_(False)
        self.config = config
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr, betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
        self.schedules = [
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: cosine_lr_scale(update, config))
            for optimizer in (self.actor_optimizer, self.critic_optimizer)
        ]

    def state_dict(self) -> dict[str, dict]:
        """Return the networks, the target critic, the optimizers and the learning-rate schedules."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "schedules": [schedule.state_dict() for schedule in self.schedules],
        }

    def load_state_dict(self, state: dict[str, dict]) -> None:
        """Restore a :meth:`state_dict`, so that training continues as if uninterrupted."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for schedule, schedule_state in zip(self.schedules, state["schedules"], strict=True):
            schedule.load_state_dict(schedule_state)

    def act(self, hooks: ImaginationHooks, s: Tensor, xi: Tensor, k: Tensor) -> Tensor:
        """Return the actor's action in state ``s`` at reference frame ``k``."""
        return self.actor(torch.cat([hooks.observe(s, xi), hooks.goal(k)], dim=-1))

    def value(self, hooks: ImaginationHooks, s: Tensor, xi: Tensor, k: Tensor) -> Tensor:
        """Return the critic's value of state ``s`` at reference frame ``k``."""
        return self.critic(torch.cat([s, hooks.goal(k), xi], dim=-1))

    def update(self, hooks: ImaginationHooks, s0: Tensor, xi: Tensor, k0: Tensor) -> dict[str, float]:
        """Improve the actor by BPTT over one imagined batch, then fit the critic.

        Args:
            hooks: World model and task.
            s0: Start states, shape [batch, state_size].
            xi: Sim context per rollout, shape [batch, context_size].
            k0: Reference frame index of the start states, shape [batch].

        Returns:
            Scalar diagnostics.
        """
        cfg = self.config
        s = s0
        states, observations, rewards, done = [s0], [], [], []
        stride = hooks.frame_stride
        for t in range(cfg.horizon):
            k, k_next = k0 + t * stride, k0 + (t + 1) * stride
            o = torch.cat([hooks.observe(s, xi), hooks.goal(k)], dim=-1)
            a = self.actor(o)
            s_next, aux = hooks.dynamics(s, a, xi)
            rewards.append(hooks.reward(s, a, s_next, aux, k_next))
            done.append(hooks.termination(s_next, k_next))
            observations.append(o.detach())
            states.append(s_next)
            s = s_next
        rewards = torch.stack(rewards, dim=1)
        done = torch.stack(done, dim=1)
        value_last = self.value(hooks, s, xi, k0 + cfg.horizon * stride)
        objective = imagined_return(rewards, done, value_last, cfg.gamma).mean() / cfg.horizon

        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        # Only the actor's gradient: the world model and critic stay untouched.
        grads = torch.autograd.grad(-objective, actor_params)
        for p, g in zip(actor_params, grads, strict=True):
            p.grad = g
        actor_grad_norm = nn.utils.clip_grad_norm_(actor_params, cfg.actor_grad_norm)
        nn.utils.clip_grad_value_(actor_params, cfg.actor_grad_value)
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor.normalizer.update(torch.stack(observations))

        critic_loss, target_max = self._fit_critic(
            hooks, torch.stack(states, dim=1).detach(), xi, k0, rewards.detach(), done.detach()
        )
        for schedule in self.schedules:
            schedule.step()
        survival = torch.cumprod(1.0 - done.detach(), dim=1)[:, -1]
        return {
            "objective": objective.item(),
            "reward": rewards.detach().mean().item(),
            "survival": survival.mean().item(),
            "value": value_last.detach().mean().item(),
            "value_max_abs": value_last.detach().abs().max().item(),
            "critic_target_max_abs": target_max,
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_loss": critic_loss,
        }

    def _fit_critic(
        self, hooks: ImaginationHooks, states: Tensor, xi: Tensor, k0: Tensor, rewards: Tensor, done: Tensor
    ) -> tuple[float, float]:
        cfg = self.config
        batch, steps = states.shape[:2]
        frames = k0[:, None] + hooks.frame_stride * torch.arange(steps, device=k0.device)
        inputs = torch.cat(
            [states, hooks.goal(frames.reshape(-1)).reshape(batch, steps, -1), xi[:, None].expand(-1, steps, -1)],
            dim=-1,
        )
        self.critic.normalizer.update(inputs)
        self.target_critic.normalizer.load_state_dict(self.critic.normalizer.state_dict())
        with torch.no_grad():
            targets = lambda_returns(rewards, done, self.target_critic(inputs), cfg.gamma, cfg.lam)
        inputs = inputs[:, :-1].reshape(-1, inputs.shape[-1])
        targets = targets.reshape(-1)
        size = inputs.shape[0] // cfg.critic_minibatches
        for _ in range(cfg.critic_epochs):
            order = torch.randperm(inputs.shape[0], device=inputs.device)
            for i in range(cfg.critic_minibatches):
                index = order[i * size : (i + 1) * size]
                loss = (self.critic(inputs[index]) - targets[index]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_norm)
                self.critic_optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target.lerp_(source, cfg.critic_target_rate)
        return loss.item(), targets.abs().max().item()


def train(
    trainer: Trainer,
    hooks: ImaginationHooks,
    iterations: int,
    *,
    collect: Callable[[Trainer], dict[str, float]],
    fit_world_model: Callable[[], dict[str, float]],
    sample_starts: Callable[[], tuple[Tensor, Tensor, Tensor]],
    log: Callable[[int, dict[str, float]], None] | None = None,
):
    """Run the SuperMimic loop: collect, fit the world model, improve the policy.

    Args:
        trainer: Actor-critic to improve.
        hooks: Imagination functions; ``hooks.dynamics`` must use the model that
            ``fit_world_model`` trains.
        iterations: Loop iterations.
        collect: Rolls out ``trainer`` (with action noise) in the simulator,
            stores the transitions and returns diagnostics.
        fit_world_model: Updates the world model on stored windows and returns
            diagnostics.
        sample_starts: Returns imagination start states, sim context and
            reference frame indices ``(s0, xi, k0)``.
        log: Receives the iteration and its diagnostics, with the wall time of
            each phase [s]; a phase ends when its scalar diagnostics reach the host.
    """
    for iteration in range(iterations):
        start = time.perf_counter()
        metrics = collect(trainer)
        collected = time.perf_counter()
        metrics.update(fit_world_model())
        fitted = time.perf_counter()
        metrics.update(trainer.update(hooks, *sample_starts()))
        metrics["time_collect_s"] = collected - start
        metrics["time_world_model_s"] = fitted - collected
        metrics["time_policy_s"] = time.perf_counter() - fitted
        if log is not None:
            log(iteration, metrics)
