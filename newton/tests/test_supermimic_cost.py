# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch

    from projects.supermimic.cost import (
        GenerativeCostConfig,
        generative_reward,
        generative_terminated,
        travel_distance,
    )

DT = 0.01
BELT = 4.0
UPRIGHT = (0.0, 0.0, 0.0, 1.0)


def _action(k: float, b: float, q_eq: float = 0.0) -> "torch.Tensor":
    return torch.tensor([q_eq, k, b]).repeat(10, 1)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestSupermimicCost(unittest.TestCase):
    def reward(self, quat=UPRIGHT, vel=(0.0, 0.0, 0.0), work=None, action=None, prev=None, config=None):
        action = _action(200.0, 10.0) if action is None else action
        pos, neg = (torch.zeros(10), torch.zeros(10)) if work is None else work
        return generative_reward(
            torch.tensor(quat),
            torch.tensor(vel),
            pos,
            neg,
            action,
            action if prev is None else prev,
            config or GenerativeCostConfig(),
            dt=DT,
            belt_speed=BELT,
        )

    def test_steady_upright_runner(self):
        """Earn speed, upright and survival and pay only the gains when holding the belt speed at rest power."""
        c = GenerativeCostConfig()
        reward, terms = self.reward()
        expected = DT * (c.speed + c.upright + c.alive - c.stiffness * 2000.0 - c.damping * 100.0)
        self.assertAlmostEqual(reward.item(), expected, places=6)
        self.assertAlmostEqual(sum(terms.values()).item(), reward.item(), places=6)
        for name in ("positive_work", "negative_work", "command_rate"):
            self.assertEqual(terms[name].item(), 0.0)

    def test_speed_and_upright(self):
        """Score a planar speed error of one sigma as exp(-1) and a pelvis lying on its side as 0."""
        c = GenerativeCostConfig()
        h = math.sqrt(0.5)
        _, terms = self.reward(quat=(h, 0.0, 0.0, h), vel=(0.3, -0.4, 2.0))
        self.assertAlmostEqual(terms["speed"].item(), DT * c.speed * math.exp(-1.0), places=6)
        self.assertAlmostEqual(terms["upright"].item(), 0.0, places=6)

    def test_work_weights_per_meter(self):
        """Weight positive and negative work separately, per meter of belt travel in the step."""
        c = GenerativeCostConfig(positive_work=0.02, negative_work=0.005)
        pos, neg = torch.zeros(10), torch.zeros(10)
        pos[0], pos[8], neg[3] = 1.0, 0.2, -0.6
        _, terms = self.reward(work=(pos, neg), config=c)
        self.assertAlmostEqual(terms["positive_work"].item(), -0.02 * 1.2 / BELT, places=7)
        self.assertAlmostEqual(terms["negative_work"].item(), -0.005 * 0.6 / BELT, places=7)

    def test_command_rate(self):
        """Sum the weighted squared change of each command channel over the joints."""
        c = GenerativeCostConfig(command_rate=(2.0, 0.1, 0.5))
        _, terms = self.reward(action=_action(203.0, 11.0, 0.1), prev=_action(200.0, 10.0, 0.0), config=c)
        expected = -DT * 10.0 * (2.0 * 0.01 + 0.1 * 9.0 + 0.5 * 1.0)
        self.assertAlmostEqual(terms["command_rate"].item(), expected, places=6)

    def test_reward_gradients(self):
        """Pass the hand-derived gradient to work, stiffness and equilibrium angle."""
        c = GenerativeCostConfig(command_rate=(3.0, 0.0, 0.0))
        pos = torch.full((10,), 0.5, requires_grad=True)
        neg = torch.full((10,), -0.5, requires_grad=True)
        action = _action(200.0, 10.0, 0.2).requires_grad_(True)
        reward, _ = self.reward(work=(pos, neg), action=action, prev=_action(200.0, 10.0, 0.0), config=c)
        reward.backward()
        self.assertAlmostEqual(pos.grad[0].item(), -c.positive_work / BELT, places=8)
        self.assertAlmostEqual(neg.grad[1].item(), c.negative_work / BELT, places=8)
        self.assertAlmostEqual(action.grad[4, 1].item(), -DT * c.stiffness, places=8)
        self.assertAlmostEqual(action.grad[4, 0].item(), -DT * 3.0 * 2.0 * 0.2, places=7)

    def test_travel_distance(self):
        """Measure the distance over the belt from treadmill-frame velocities."""
        vel = torch.tensor([[0.1, 0.3, 0.0], [-0.3, 0.0, 1.0]])
        self.assertAlmostEqual(travel_distance(vel, 0.1, BELT).item(), 0.1 * (4.1 + 3.7), places=6)

    def test_work_return_matches_work_per_meter(self):
        """Sum the work terms of an episode to its duration times the work per meter travelled."""
        c = GenerativeCostConfig(positive_work=0.03, negative_work=0.01)
        steps = 50
        pos, neg = torch.rand(steps, 10), -torch.rand(steps, 10)
        vel = torch.zeros(steps, 3)
        action = _action(200.0, 10.0).expand(steps, 10, 3)
        _, terms = generative_reward(
            torch.tensor(UPRIGHT).expand(steps, 4), vel, pos, neg, action, action, c, dt=DT, belt_speed=BELT
        )
        distance = travel_distance(vel, DT, BELT)
        expected = -steps * DT * (c.positive_work * pos.sum() - c.negative_work * neg.sum()) / distance
        total = terms["positive_work"].sum() + terms["negative_work"].sum()
        self.assertAlmostEqual(total.item(), expected.item(), places=5)

    def test_terminated(self):
        """End an episode on a low pelvis or a pelvis tilted past the threshold."""
        c = GenerativeCostConfig(height_min=0.6, up_min=0.5)
        tilt = math.radians(70.0)
        quat = torch.tensor([UPRIGHT, UPRIGHT, (math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))])
        pos = torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.0, 0.5], [0.0, 0.0, 0.9]])
        self.assertEqual(generative_terminated(pos, quat, c).tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main(verbosity=2)
