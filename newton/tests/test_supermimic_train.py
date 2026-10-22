# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import math
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch
    from torch import nn

    from projects.supermimic.policy import Actor, Critic
    from projects.supermimic.train import (
        ImaginationHooks,
        Trainer,
        TrainerConfig,
        cosine_lr_scale,
        imagined_return,
        lambda_returns,
        train,
    )

# Toy task: a torque-limited pendulum (angle from hanging) under an impedance action
# (q_eq, K, B) tracks a large swing that needs saturated torque; tracking error
# beyond 0.8 rad terminates.
DT = 0.05
TORQUE_MAX = 5.0
GRAVITY = 9.81
PERIOD = 3.0
AMPLITUDE = 2.0


def _pendulum(s, a, xi):
    theta, omega = s[:, 0], s[:, 1]
    a = a[:, 0]
    torque = TORQUE_MAX * torch.tanh((a[:, 1] * (a[:, 0] - theta) - a[:, 2] * omega) / TORQUE_MAX)
    omega = omega + DT * (torque - GRAVITY * torch.sin(theta))
    return torch.stack([theta + DT * omega, omega], dim=-1), torque[:, None]


def _reference(k):
    w = 2.0 * math.pi / PERIOD
    t = k.float() * DT
    return torch.stack([AMPLITUDE * torch.sin(w * t), AMPLITUDE * w * torch.cos(w * t)], dim=-1)


def _goal(k):
    return torch.cat([_reference(k), _reference(k + 5)], dim=-1)


def _observe(s, xi):
    return s


def _reward(s, a, s_next, aux, k):
    e = s_next - _reference(k)
    return torch.exp(-((e[:, 0] / 0.3) ** 2)) + 0.5 * torch.exp(-((e[:, 1] / 1.0) ** 2))


def _tracking_error(s, k):
    return (s[:, 0] - _reference(k)[:, 0]).abs()


def _termination(s_next, k):
    return torch.sigmoid((_tracking_error(s_next, k) - 0.8) / 0.05)


def _trainer(device, **config):
    low = torch.tensor([[-math.pi, 0.5, 0.05]], device=device)
    high = torch.tensor([[math.pi, 100.0, 10.0]], device=device)
    log_scale = torch.tensor([[False, True, True]], device=device)
    actor = Actor(6, low, high, log_scale, hidden=(64, 64)).to(device)
    critic = Critic(6, hidden=(64, 64)).to(device)
    return Trainer(actor, critic, TrainerConfig(actor_lr=1e-3, critic_lr=1e-3, **config))


def _starts(n, device):
    k = torch.randint(0, round(PERIOD / DT), (n,), device=device)
    s = _reference(k) + torch.randn(n, 2, device=device) * torch.tensor([0.2, 0.5], device=device)
    return s, torch.zeros(n, 0, device=device), k


def _evaluate(trainer, hooks, device, steps=400, n=64):
    """Return the surviving fraction and RMS angle error over ``steps`` in the true dynamics."""
    s, xi, k = _starts(n, device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    errors = []
    with torch.no_grad():
        for t in range(steps):
            s, _ = _pendulum(s, trainer.act(hooks, s, xi, k + t), xi)
            errors.append(_tracking_error(s, k + t + 1))
            alive &= errors[-1] < 0.8
    return alive.float().mean().item(), torch.stack(errors).square().mean().sqrt().item()


ITERATIONS_TRUE = 200
ITERATIONS_FITTED = 200

if _HAS_TORCH:

    class _Model(nn.Module):
        """Residual MLP world model of the pendulum."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 2))

        def forward(self, s, a, xi):
            a = a[:, 0]
            x = torch.cat([s[:, :1].sin(), s[:, :1].cos(), s[:, 1:] / 5, a[:, :1] / math.pi, a[:, 1:].log()], dim=-1)
            delta = self.net(x)
            omega = s[:, 1] + delta[:, 1]
            return torch.stack([s[:, 0] + DT * omega + 0.01 * delta[:, 0], omega], dim=-1), s[:, :0]

    class _Collector:
        """Noisy-policy rollouts in the true dynamics stored in a ring buffer of episodes."""

        def __init__(self, envs, capacity, device):
            self.envs, self.capacity, self.device = envs, capacity, device
            self.s, self.xi, self.k = _starts(envs, device)
            self.episode = torch.arange(envs, device=device)
            self.episode_count = envs
            self.state = torch.zeros(capacity, envs, 2, device=device)
            self.action = torch.zeros(capacity, envs, 1, 3, device=device)
            self.frame = torch.zeros(capacity, envs, dtype=torch.long, device=device)
            self.episode_id = torch.full((capacity, envs), -1, dtype=torch.long, device=device)
            self.size = 0
            self.cursor = 0

        @torch.no_grad()
        def __call__(self, trainer, steps=8, noise=0.3):
            actor = trainer.actor
            for _ in range(steps):
                u = actor.logits(torch.cat([_observe(self.s, self.xi), _goal(self.k)], dim=-1))
                a = actor.squash(u + noise * torch.randn_like(u))
                i = self.cursor
                self.state[i], self.action[i], self.frame[i], self.episode_id[i] = self.s, a, self.k, self.episode
                self.cursor = (i + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)
                self.s, _ = _pendulum(self.s, a, self.xi)
                self.k = self.k + 1
                reset = (_tracking_error(self.s, self.k) > 0.8) | (self.k > 400)
                count = int(reset.sum())
                if count:
                    self.s[reset], _, self.k[reset] = _starts(count, self.device)
                    self.episode[reset] = self.episode_count + torch.arange(count, device=self.device)
                    self.episode_count += count
            return {}

        def windows(self, batch, length):
            """Return up to ``batch`` windows of ``length`` transitions inside one episode."""
            t = (
                self.cursor - self.size + torch.randint(0, self.size - length - 1, (2 * batch,), device=self.device)
            ) % self.capacity
            env = torch.randint(0, self.envs, (2 * batch,), device=self.device)
            steps = (t[:, None] + torch.arange(length + 1, device=self.device)) % self.capacity
            ids = self.episode_id[steps, env[:, None]]
            keep = (ids == ids[:, :1]).all(1) & (ids[:, 0] >= 0)
            steps, env = steps[keep][:batch], env[keep][:batch]
            return self.state[steps, env[:, None]], self.action[steps, env[:, None]], self.frame[steps[:, 0], env]


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestSupermimicTrain(unittest.TestCase):
    def test_lambda_returns_follow_recursion(self):
        """Match the Eq. 3 recursion, one-step targets at lambda 0 and Eq. 2 at lambda 1."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        done = torch.tensor([[0.0, 0.5, 0.0]])
        values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        gamma, lam = 0.9, 0.5

        r2 = 3.0 + gamma * 40.0
        r1 = 2.0 + gamma * 0.5 * ((1 - lam) * 30.0 + lam * r2)
        r0 = 1.0 + gamma * ((1 - lam) * 20.0 + lam * r1)
        torch.testing.assert_close(lambda_returns(rewards, done, values, gamma, lam), torch.tensor([[r0, r1, r2]]))

        one_step = rewards + gamma * (1 - done) * values[:, 1:]
        torch.testing.assert_close(lambda_returns(rewards, done, values, gamma, 0.0), one_step)

        full = lambda_returns(rewards, done, values, gamma, 1.0)[:, 0]
        torch.testing.assert_close(full, imagined_return(rewards, done, values[:, -1], gamma))

    def test_imagined_return_drops_terms_after_termination(self):
        """Count the reward of the terminating step and nothing after it."""
        rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        done = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        result = imagined_return(rewards, done, torch.tensor([100.0]), 0.5)

        torch.testing.assert_close(result, torch.tensor([1.0 + 0.5]))

    def test_update_changes_only_actor_and_critic(self):
        """Leave world-model weights and gradients untouched by the policy update."""
        torch.manual_seed(0)
        model = nn.Linear(5, 2)
        weight = model.weight.detach().clone()

        def dynamics(s, a, xi):
            return s + DT * model(torch.cat([s, a[:, 0]], dim=-1)), torch.zeros(s.shape[0], 0)

        hooks = ImaginationHooks(dynamics, _observe, _goal, _reward, _termination)
        trainer = _trainer("cpu", horizon=4)
        actor = [p.detach().clone() for p in trainer.actor.parameters()]
        critic = [p.detach().clone() for p in trainer.critic.parameters()]

        trainer.update(hooks, *_starts(32, "cpu"))

        torch.testing.assert_close(model.weight, weight, atol=0, rtol=0)
        self.assertIsNone(model.weight.grad)
        self.assertTrue(any(not torch.equal(a, p) for a, p in zip(actor, trainer.actor.parameters(), strict=True)))
        self.assertTrue(any(not torch.equal(c, p) for c, p in zip(critic, trainer.critic.parameters(), strict=True)))

    def test_resume_continues_like_an_uninterrupted_run(self):
        """Restore networks, target critic, optimizers and schedules, so a resumed update matches an uninterrupted one."""
        hooks = ImaginationHooks(_pendulum, _observe, _goal, _reward, _termination)
        torch.manual_seed(0)
        trainer = _trainer("cpu", horizon=4, critic_target_rate=0.5)
        s0, xi, k0 = _starts(16, "cpu")
        trainer.update(hooks, s0, xi, k0)
        checkpoint = io.BytesIO()
        torch.save(trainer.state_dict(), checkpoint)
        checkpoint.seek(0)
        resumed = _trainer("cpu", horizon=4, critic_target_rate=0.5)
        resumed.load_state_dict(torch.load(checkpoint))

        torch.manual_seed(1)
        trainer.update(hooks, s0, xi, k0)
        torch.manual_seed(1)
        resumed.update(hooks, s0, xi, k0)

        for name in ("actor", "critic", "target_critic"):
            expected, actual = getattr(trainer, name).state_dict(), getattr(resumed, name).state_dict()
            for key, value in expected.items():
                torch.testing.assert_close(actual[key], value, atol=0, rtol=0)

    def test_learning_rates_decay_on_a_cosine_then_stay(self):
        """Decay both learning rates to the final factor over the decay updates, then hold them."""
        hooks = ImaginationHooks(_pendulum, _observe, _goal, _reward, _termination)
        trainer = _trainer("cpu", horizon=4, lr_decay_iterations=2)
        s0, xi, k0 = _starts(8, "cpu")
        lrs = []
        for _ in range(3):
            trainer.update(hooks, s0, xi, k0)
            lrs.append(
                [group["lr"] for o in (trainer.actor_optimizer, trainer.critic_optimizer) for group in o.param_groups]
            )

        torch.testing.assert_close(torch.tensor(lrs[0]), torch.full((2,), 1e-3 * cosine_lr_scale(1, trainer.config)))
        torch.testing.assert_close(torch.tensor(lrs[1]), torch.full((2,), 3e-5))
        torch.testing.assert_close(torch.tensor(lrs[2]), torch.full((2,), 3e-5))
        self.assertAlmostEqual(cosine_lr_scale(1, trainer.config), 0.515)

    def test_update_advances_reference_by_frame_stride(self):
        """Advance the reference frame of goal, reward and termination by the frame stride per step."""
        seen = {"goal": [], "reward": [], "termination": []}

        def goal(k):
            seen["goal"].append(k[0].item())
            return _goal(k)

        def reward(s, a, s_next, aux, k):
            seen["reward"].append(k[0].item())
            return _reward(s, a, s_next, aux, k)

        def termination(s_next, k):
            seen["termination"].append(k[0].item())
            return _termination(s_next, k)

        hooks = ImaginationHooks(_pendulum, _observe, goal, reward, termination, frame_stride=3)
        trainer = _trainer("cpu", horizon=4)
        s0, xi, _ = _starts(8, "cpu")

        trainer.update(hooks, s0, xi, torch.full((8,), 7))

        # Actor inputs at 7, 10, 13, 16, then the bootstrap value at 19 and the critic inputs at 7 .. 19.
        self.assertEqual(seen["goal"][:5], [7, 10, 13, 16, 19])
        self.assertEqual(seen["reward"], [10, 13, 16, 19])
        self.assertEqual(seen["termination"], [10, 13, 16, 19])

    def test_actor_learns_tracking_through_true_dynamics(self):
        """Learn to track the swing when imagination uses the true dynamics."""
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hooks = ImaginationHooks(_pendulum, _observe, _goal, _reward, _termination)
        trainer = _trainer(device)
        survived, _ = _evaluate(trainer, hooks, device)
        self.assertLess(survived, 0.1)

        for _ in range(ITERATIONS_TRUE):
            trainer.update(hooks, *_starts(512, device))

        survived, rms = _evaluate(trainer, hooks, device)
        self.assertGreaterEqual(survived, 0.95)
        self.assertLess(rms, 0.1)

    def test_train_loop_learns_through_fitted_model(self):
        """Learn to track the swing in the true dynamics with a world model fitted by the loop."""
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _Model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        buffer = _Collector(256, 400, device)
        hooks = ImaginationHooks(model, _observe, _goal, _reward, _termination)
        trainer = _trainer(device)
        scale = torch.tensor([0.05, 0.5], device=device)

        def fit_world_model(window=4):
            for _ in range(5):
                states, actions, _ = buffer.windows(512, window)
                s = states[:, 0]
                loss = 0.0
                for j in range(window):
                    s, _ = model(s, actions[:, j], None)
                    loss = loss + ((s - states[:, j + 1]) / scale).square().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            return {"model_loss": loss.item() / window}

        def sample_starts():
            states, _, k = buffer.windows(512, 0)
            return states[:, 0], torch.zeros(states.shape[0], 0, device=device), k

        buffer(trainer, steps=32)
        train(
            trainer,
            hooks,
            ITERATIONS_FITTED,
            collect=buffer,
            fit_world_model=fit_world_model,
            sample_starts=sample_starts,
        )

        survived, rms = _evaluate(trainer, hooks, device)
        self.assertGreaterEqual(survived, 0.95)
        self.assertLess(rms, 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
