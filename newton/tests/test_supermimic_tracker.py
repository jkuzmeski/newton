# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    from projects.supermimic.body import BodyParams
    from projects.supermimic.buffer import ReplayBuffer
    from projects.supermimic.policy import Critic
    from projects.supermimic.tracker import TrackerConfig, TrackerTask
    from projects.supermimic.train import Trainer, TrainerConfig
    from projects.supermimic.world_model import WorldModel

WINDOWS = 2
FRAMES = 40
STRIDE = 2
XI = 3
AUX = 4


def _reference():
    """Running-like reference: upright root at 0.9 m moving forward, joints swinging inside their limits."""
    t = torch.arange(FRAMES, dtype=torch.float32) * 0.005
    frames = torch.zeros(WINDOWS, FRAMES, 33)
    frames[..., 0] = 3.7 * t
    frames[..., 2] = 0.9
    frames[..., 6] = 1.0
    frames[..., 7] = 3.7
    phase = 2 * math.pi * t / 0.7
    frames[..., 13:23] = 0.2 * torch.sin(phase)[:, None] + 0.2
    frames[..., 23:33] = 0.2 * (2 * math.pi / 0.7) * torch.cos(phase)[:, None]
    frames[1, :, 13:23] += 0.1
    return frames


def _task(history=2):
    torch.manual_seed(0)
    model = WorldModel(10, XI, AUX, history_length=history, hidden=(32,))
    return TrackerTask(model, _reference(), STRIDE, BodyParams(), 0.01, TrackerConfig(goal_frames=3))


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestSupermimicTracker(unittest.TestCase):
    def test_reward_is_maximal_on_the_reference(self):
        """Give the full tracking reward and no termination when the state matches the reference."""
        task = _task()
        k = torch.tensor([5, FRAMES + 9])
        state = task.reference[k]
        action = task.actor()(torch.zeros(2, task.observation_size + task.goal_size)).detach()
        model_state = task.world_model.start(state[:, None].expand(-1, 3, -1), action[:, None].expand(-1, 2, -1, -1))

        reward = task.reward(model_state, action, model_state, torch.zeros(2, AUX), k)

        torch.testing.assert_close(reward, torch.full((2,), 0.01 * 5.5))
        self.assertLess(float(task.termination(model_state, k).max()), 1e-3)

    def test_goal_holds_the_last_frame_of_the_window(self):
        """Read goal frames ahead by the frame stride and hold the window's last frame."""
        task = _task()
        k = torch.tensor([FRAMES + 10, 2 * FRAMES - 3])

        goal = task.goal(k).unflatten(1, (3, 31))

        torch.testing.assert_close(goal[0], task.reference[[FRAMES + 12, FRAMES + 14, FRAMES + 16], 2:])
        torch.testing.assert_close(goal[1], task.reference[[2 * FRAMES - 1] * 3, 2:])

    def test_normalized_action_maps_impedance_bounds_to_unit_range(self):
        """Keep q_eq in rad and map K and B bounds to 0 and 1."""
        task = _task()
        action = torch.stack((task.action_low, task.action_high))

        normalized = task.normalized_action(action)

        torch.testing.assert_close(normalized[..., 0], action[..., 0])
        torch.testing.assert_close(normalized[0, :, 1:], torch.zeros(10, 2))
        torch.testing.assert_close(normalized[1, :, 1:], torch.ones(10, 2))

    def test_start_action_carries_the_reference_moment(self):
        """Carry the reference moment with a lead of at most 0.35 rad below the stiffness cap, and loose K without moment."""
        task = _task()
        q_ref = task.reference[[3, FRAMES + 5], 13:23]
        moment = torch.tensor([[0.0] * 10, [50.0, -20.0, 5.0, -150.0, 2000.0] * 2])

        action = task.start_action(q_ref, moment)
        q_eq, stiffness = action[..., 0], action[..., 1]

        torch.testing.assert_close(q_eq[0], q_ref[0])
        torch.testing.assert_close(stiffness[0], task.action_low[:, 1])
        self.assertTrue(bool(((q_eq >= task.action_low[:, 0]) & (q_eq <= task.action_high[:, 0])).all()))
        self.assertTrue(bool((stiffness <= task.action_high[:, 1]).all()))
        # Below the stiffness cap the posture lead carries the moment exactly.
        carried = stiffness[1] < task.action_high[:, 1]
        lead = (q_eq - q_ref)[1]
        torch.testing.assert_close((stiffness[1] * lead)[carried], moment[1][carried])
        self.assertTrue(bool((lead[carried].abs() <= 0.35 + 1e-6).all()))
        self.assertTrue(bool((~carried).any()))

    def test_starts_leave_room_for_the_horizon(self):
        """Sample only non-terminal starts whose horizon stays inside the reference window."""
        task = _task()
        buffer = ReplayBuffer(4, 60, state_shape=33, action_shape=(10, 3), xi_shape=XI, aux_shape=AUX, device="cpu")
        for step in range(60):
            frame = torch.full((4,), (step % 20) * STRIDE)
            buffer.push(
                task.reference[frame],
                torch.ones(4, 10, 3),
                torch.zeros(4, XI),
                torch.zeros(4, AUX),
                frame,
                torch.full((4,), step % 20 == 0),
                torch.full((4,), step % 7 == 0),
            )

        model_state, _, k = task.sample_starts(buffer, 256, 8)

        self.assertGreater(len(k), 0)
        self.assertEqual(model_state.shape, (len(k), task.world_model.model_state_dim))
        self.assertTrue(bool((k % FRAMES + 8 * STRIDE < FRAMES).all()))
        torch.testing.assert_close(model_state[:, :33], task.reference[k])

    def test_update_through_the_world_model_is_finite(self):
        """Run one actor-critic update through the world model with finite results."""
        task = _task()
        model = task.world_model
        with torch.no_grad():
            model.net[-1].weight.normal_(std=0.01)
        k = torch.randint(0, FRAMES - 8 * STRIDE, (16,)) + FRAMES * torch.randint(0, WINDOWS, (16,))
        actor = task.actor(hidden=(32,))
        action = actor(torch.zeros(16, task.observation_size + task.goal_size)).detach()
        model_state = model.start(task.reference[k][:, None].expand(-1, 3, -1), action[:, None].expand(-1, 2, -1, -1))
        trainer = Trainer(
            actor, Critic(model.model_state_dim + task.goal_size + XI, hidden=(32,)), TrainerConfig(horizon=8)
        )

        metrics = trainer.update(task.hooks, model_state, torch.zeros(16, XI), k)

        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertGreater(metrics["actor_grad_norm"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
