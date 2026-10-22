# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    from projects.supermimic.body import JOINT_NAMES, BodyParams
    from projects.supermimic.policy import Actor, RunningNormalizer, impedance_action_bounds


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestSupermimicPolicy(unittest.TestCase):
    def test_actor_actions_stay_inside_impedance_bounds(self):
        """Keep equilibrium, stiffness and damping inside their bounds for extreme inputs."""
        torch.manual_seed(0)
        joint_limits = tuple((lower, lower + 1.5) for lower in (-1.0, -0.8, -0.6, -0.4, -0.2))
        low, high, log_scale = impedance_action_bounds(
            joint_limits, stiffness_limits=((5.0, 500.0),) * 5, damping_limits=((0.1, 20.0),) * 5
        )
        actor = Actor(7, low, high, log_scale, hidden=(32,))
        with torch.no_grad():
            actor.net[-1].weight.normal_(std=100.0)
        with torch.no_grad():
            action = actor(1e3 * torch.randn(256, 7))

        self.assertEqual(action.shape, (256, 10, 3))
        # Exact up to float32 rounding of exp(log(bound)).
        self.assertTrue(bool((action >= low * (1 - 1e-6) - 1e-6).all() and (action <= high * (1 + 1e-6) + 1e-6).all()))
        self.assertGreater(float(action[..., 1:].min()), 0.0)
        # Saturated outputs reach both ends of each range.
        self.assertTrue(bool(torch.isclose(action.amin(0), low, rtol=1e-3, atol=1e-4).all()))
        self.assertTrue(bool(torch.isclose(action.amax(0), high, rtol=1e-3, atol=1e-4).all()))

    def test_actor_starts_at_middle_of_bounds(self):
        """Start at the linear midpoint of q_eq and the geometric midpoint of K and B."""
        low, high, log_scale = impedance_action_bounds(
            ((-1.0, 1.0),) * 5, stiffness_limits=((4.0, 400.0),) * 5, damping_limits=((0.5, 50.0),) * 5
        )
        actor = Actor(3, low, high, log_scale, hidden=(16,))
        with torch.no_grad():
            actor.net[-1].weight.zero_()
        action = actor(torch.randn(5, 3))

        torch.testing.assert_close(action[..., 0], torch.zeros(5, 10))
        torch.testing.assert_close(action[..., 1], torch.full((5, 10), 40.0))
        torch.testing.assert_close(action[..., 2], torch.full((5, 10), 5.0))

    def test_default_bounds_are_human_impedance_limits_on_both_sides(self):
        """Bound K and B of every joint by the human short-time impedance table, equal on both sides."""
        torch.manual_seed(0)
        low, high, log_scale = impedance_action_bounds(BodyParams().joint_limits)
        actor = Actor(7, low, high, log_scale, hidden=(32,))
        with torch.no_grad():
            actor.net[-1].weight.normal_(std=100.0)
            action = actor(1e3 * torch.randn(256, 7))

        stiffness, damping = action[..., 1].amax(0), action[..., 2].amax(0)
        for side in ("r", "l"):
            self.assertAlmostEqual(float(stiffness[JOINT_NAMES.index(f"knee_flexion_{side}")]), 1141.0, delta=1.0)
            self.assertAlmostEqual(float(stiffness[JOINT_NAMES.index(f"ankle_dorsiflexion_{side}")]), 665.0, delta=1.0)
            self.assertAlmostEqual(float(damping[JOINT_NAMES.index(f"knee_flexion_{side}")]), 5.3, delta=1e-2)
            self.assertAlmostEqual(float(damping[JOINT_NAMES.index(f"ankle_dorsiflexion_{side}")]), 3.1, delta=1e-2)
        torch.testing.assert_close(high[:5], high[5:])
        self.assertLessEqual(float(stiffness.max()), 1141.0 * (1 + 1e-6))
        self.assertLessEqual(float(damping.max()), 5.3 * (1 + 1e-6))

    def test_running_normalizer_matches_batch_statistics(self):
        """Standardize with the mean and variance of all updates combined."""
        torch.manual_seed(0)
        x = 3.0 + 2.0 * torch.randn(3000, 4)
        normalizer = RunningNormalizer(4)
        for chunk in x.split(1000):
            normalizer.update(chunk)

        torch.testing.assert_close(normalizer.mean, x.mean(0), atol=1e-4, rtol=0)
        torch.testing.assert_close(normalizer.var, x.var(0, unbiased=False), atol=1e-3, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
