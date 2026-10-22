# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import unittest

import numpy as np

import newton
from projects.supermimic.body import BODY_NAMES, BodyParams, add_body

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch

    from projects.supermimic.tracking import (
        STATE_DIM,
        TRACKING_ERROR_DIM,
        ADDiscriminator,
        LegKinematics,
        termination,
        termination_probability,
        tracking_errors,
        tracking_reward,
    )


def _states(generator, count: int) -> "torch.Tensor":
    """Random upright-ish states with normalized quaternions."""
    state = torch.randn(count, STATE_DIM, generator=generator) * 0.3
    state[:, 2] += 0.95
    state[:, 6] += 3.0
    state[:, 3:7] = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True)
    return state


@unittest.skipUnless(TORCH, "torch is not installed")
class TestSupermimicTracking(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(0)
        self.kinematics = LegKinematics(BodyParams(), "cpu")

    def test_foot_positions_match_newton(self):
        """Place the ankle centers where Newton forward kinematics puts the foot origins."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        add_body(builder, BodyParams())
        model = builder.finalize(device="cpu")
        newton_state = model.state()
        state = _states(self.generator, 4)
        for world in range(4):
            joint_q = np.concatenate((state[world, :7].numpy(), state[world, 13:23].numpy())).astype(np.float32)
            newton_state.joint_q.assign(joint_q)
            newton.eval_fk(model, newton_state.joint_q, newton_state.joint_qd, newton_state)
            body_q = newton_state.body_q.numpy()
            expected = body_q[[BODY_NAMES.index("foot_r"), BODY_NAMES.index("foot_l")], :3]
            feet = self.kinematics.foot_positions(
                state[world : world + 1, :3], state[world : world + 1, 3:7], state[world : world + 1, 13:23]
            )
            np.testing.assert_allclose(feet[0].numpy(), expected, atol=1.0e-5)

    def test_zero_error_gives_the_maximum(self):
        """Give zero errors and the 5.5 maximum for a state equal to its reference, less otherwise."""
        state = _states(self.generator, 64)
        errors = tracking_errors(state, state, self.kinematics)
        self.assertEqual(errors.shape, (64, TRACKING_ERROR_DIM))
        self.assertLess(errors.abs().max().item(), 1.0e-6)
        torch.testing.assert_close(tracking_reward(errors), torch.full((64,), 5.5))
        other = tracking_reward(tracking_errors(state, _states(self.generator, 64), self.kinematics))
        self.assertTrue(bool((other < 5.5).all()))

    def test_reward_ignores_horizontal_position_and_quaternion_sign(self):
        """Keep the reward unchanged under a horizontal root shift and a flipped root quaternion."""
        state, reference = _states(self.generator, 16), _states(self.generator, 16)
        moved = state.clone()
        moved[:, :2] += 2.0
        moved[:, 3:7] *= -1.0
        torch.testing.assert_close(
            tracking_reward(tracking_errors(moved, reference, self.kinematics)),
            tracking_reward(tracking_errors(state, reference, self.kinematics)),
        )

    def test_gradients_are_finite(self):
        """Backpropagate finite gradients of the reward and soft termination at zero and nonzero error."""
        reference = _states(self.generator, 32)
        for state in (reference.clone(), _states(self.generator, 32)):
            state.requires_grad_(True)
            objective = tracking_reward(tracking_errors(state, reference, self.kinematics)) - termination_probability(
                state, reference
            )
            (gradient,) = torch.autograd.grad(objective.sum(), state)
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_termination_thresholds(self):
        """Terminate beyond 0.25 m of root height error or 78 degrees of extra tilt, not before."""
        reference = _states(self.generator, 1)
        reference[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        state = reference.repeat(4, 1)
        state[1, 2] -= 0.2
        state[2, 2] -= 0.3
        angle = math.radians(85.0)
        state[3, 3:7] = torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])
        self.assertEqual(termination(state, reference).tolist(), [False, False, True, True])
        probability = termination_probability(state, reference)
        self.assertLess(probability[:2].max().item(), 0.1)
        self.assertGreater(probability[2:].min().item(), 0.9)

    def test_add_discriminator_learns_zero_error_as_best(self):
        """Train the ADD discriminator on toy errors so its reward peaks at zero and pulls errors toward it."""
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scale = torch.tensor([0.05, 1.0, 3.0, 0.2], device=device)
        discriminator = ADDiscriminator(4, hidden=(64, 64)).to(device)
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=1.0e-3)
        for _ in range(500):
            loss = discriminator.loss(torch.randn(256, 4, device=device) * scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        errors = torch.randn(1024, 4, device=device) * scale
        zero = discriminator.reward(torch.zeros(1, 4, device=device))
        self.assertGreater(zero.item(), discriminator.reward(errors).mean().item())
        self.assertGreater(discriminator.reward(0.2 * errors).mean().item(), discriminator.reward(errors).mean().item())
        errors.requires_grad_(True)
        (gradient,) = torch.autograd.grad(discriminator.reward(errors).sum(), errors)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        shrinking = ((gradient * errors).sum(-1) < 0.0).float().mean().item()
        self.assertGreater(shrinking, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
