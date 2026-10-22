# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    from projects.supermimic.buffer import ReplayBuffer
    from projects.supermimic.m3 import evaluate, steps_since_touchdown
    from projects.supermimic.world_model import WorldModel


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestSupermimicM3(unittest.TestCase):
    def test_steps_since_touchdown(self):
        """Count steps from the first loaded step and skip contact carried over an episode start."""
        force = torch.tensor([0.0, 20.0, 300.0, 900.0, 0.0, 400.0, 800.0, 700.0, 0.0])
        reset = torch.zeros(9, dtype=torch.bool)
        reset[6] = True
        since = steps_since_touchdown(force, reset)
        self.assertEqual(since.tolist(), [-1, -1, 0, 1, -1, 0, -1, -1, -1])

    def test_evaluate_scores_every_post_touchdown_step(self):
        """Score each step in the first five after touchdown once at horizon 1 and only full windows at 8."""
        torch.manual_seed(0)
        buffer = ReplayBuffer(3, 40, state_shape=15, action_shape=(1, 3), xi_shape=2, aux_shape=6, device="cpu")
        for t in range(40):
            state = torch.randn(3, 15)
            state[:, 3:7] = torch.nn.functional.normalize(state[:, 3:7], dim=-1)
            aux = torch.zeros(3, 6)
            aux[:, 2] = 1000.0 * ((t % 10) >= 3)
            reset = torch.full((3,), t % 20 == 0)
            buffer.push(
                state,
                torch.rand(3, 1, 3),
                torch.rand(3, 2),
                aux,
                torch.zeros(3),
                reset,
                torch.zeros(3, dtype=torch.bool),
            )
        model = WorldModel(1, 2, 6, history_length=2, hidden=(16,))
        result = evaluate(model, buffer)
        # Touchdowns at episode steps 3 and 13 of two episodes per world; an 8-step window needs episode step >= 7.
        self.assertEqual([b["count"] for b in result["1"]], [12] * 5)
        self.assertEqual([b["count"] for b in result["8"]], [6, 6, 6, 6, 12])
        self.assertEqual([b["count"] for b in result["32"]], [0] * 5)
        self.assertTrue(all(b["mean_n"] == 1000.0 for b in result["1"]))
