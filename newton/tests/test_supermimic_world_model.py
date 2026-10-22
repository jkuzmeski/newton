# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    from projects.supermimic.world_model import WorldModel


def _hopper(episode_count: int, step_count: int, seed: int, device: str):
    """Simulate a hopper on a foam with hidden memory.

    A 60 kg body sits on a sliding leg (one joint) driven by ``K (q_eq - q) - B qd``
    over a 6 kg foot. Under the foot lies a spring in parallel with a Maxwell
    element (spring ``k1`` in series with a damper of relaxation time ``tau``). The
    damper deflection is hidden from the state and relaxes only slowly between
    contacts. The contact force also pitches the body. Each episode draws its foam
    ``xi = (k1 / 1e5 N/m, tau / 0.1 s)``.

    Returns:
        ``states [E, T + 1, 15]``, ``actions [E, T, 1, 3]``, ``xi [E, 2]`` and
        ``aux [E, T, 3]`` (contact force [N], joint force [N], joint power [W]).
    """
    g = torch.Generator(device).manual_seed(seed)

    def uniform(low, high, *shape):
        return low + (high - low) * torch.rand(*shape, generator=g, device=device)

    e = episode_count
    k1, tau = uniform(1.0e5, 3.0e5, e), uniform(0.02, 0.12, e)
    stiffness, damping = uniform(8.0e3, 1.6e4, e), uniform(50.0, 200.0, e)
    amplitude, frequency, phase = uniform(0.04, 0.1, e), uniform(1.8, 3.0, e), uniform(0.0, 2.0 * math.pi, e)
    body_mass, foot_mass, leg_length, inertia, lever = 60.0, 6.0, 0.9, 10.0, 0.02
    z, vz = leg_length + uniform(0.0, 0.1, e), uniform(-1.0, 0.5, e)
    x, vx = torch.zeros(e, device=device), uniform(-0.5, 0.5, e)
    pitch, pitch_rate = torch.zeros(e, device=device), torch.zeros(e, device=device)
    q, qd, memory = torch.zeros(e, device=device), torch.zeros(e, device=device), torch.zeros(e, device=device)
    dt, substeps = 1.0e-3, 10

    def state():
        zeros = torch.zeros_like(z)
        half = 0.5 * pitch
        return torch.stack(
            (
                x,
                zeros,
                z,
                zeros,
                torch.sin(half),
                zeros,
                torch.cos(half),
                vx,
                zeros,
                vz,
                zeros,
                pitch_rate,
                zeros,
                q,
                qd,
            ),
            -1,
        )

    states, actions, auxes = [state()], [], []
    for t in range(step_count):
        q_eq = 0.05 + amplitude * torch.sin(2.0 * math.pi * frequency * t * substeps * dt + phase)
        action = torch.stack(
            (
                q_eq + 0.01 * torch.randn(e, generator=g, device=device),
                stiffness * (1.0 + 0.1 * torch.randn(e, generator=g, device=device)),
                damping * (1.0 + 0.1 * torch.randn(e, generator=g, device=device)),
            ),
            -1,
        )
        aux = torch.zeros(e, 3, device=device)
        for _ in range(substeps):
            force = action[:, 1] * (action[:, 0] - q) - action[:, 2] * qd
            depth = (leg_length + q - z).clamp_min(0.0)
            memory = depth + (memory - depth) * torch.exp(-dt / tau)
            contact = (3.0e4 * depth + k1 * (depth - memory)).clamp_min(0.0) * (depth > 0.0)
            body_acc = force / body_mass - 9.81
            foot_acc = (contact - force) / foot_mass - 9.81
            vz = vz + body_acc * dt
            qd = qd + (body_acc - foot_acc) * dt
            vx = vx - 2.0 * contact / (body_mass * 9.81) * vx * dt
            pitch_rate = pitch_rate + (lever * contact - 100.0 * pitch - 10.0 * pitch_rate) / inertia * dt
            z, q, x, pitch = z + vz * dt, q + qd * dt, x + vx * dt, pitch + pitch_rate * dt
            aux += torch.stack((contact, force, force * qd), -1) / substeps
        states.append(state())
        actions.append(action)
        auxes.append(aux)
    xi = torch.stack((k1 / 1.0e5, tau / 0.1), -1)
    return torch.stack(states, 1), torch.stack(actions, 1)[:, :, None], xi, torch.stack(auxes, 1)


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestSupermimicWorldModel(unittest.TestCase):
    def test_step_change_round_trip(self):
        """Recover the step change, including the root rotation vector, from the advanced state."""
        torch.manual_seed(0)
        model = WorldModel(2, 1, 1, hidden=(8,))
        state = torch.randn(64, 17)
        state[:, 3:7] = torch.nn.functional.normalize(state[:, 3:7], dim=-1)
        change = 0.5 * torch.randn(64, 16)
        recovered = model._difference(model._advance(state, change), state)
        self.assertLess((recovered - change).abs().max().item(), 1.0e-5)
        rotation = model._features(state)[:, 1:10].unflatten(1, (3, 3))
        self.assertLess((rotation @ rotation.transpose(1, 2) - torch.eye(3)).abs().max().item(), 1.0e-5)

    def test_constant_inputs_keep_unit_scale(self):
        """Scale an xi entry that is constant in the data by one, not by its zero spread."""
        model = WorldModel(1, 3, 3, history_length=1, hidden=(8,))
        states, actions, xi, aux = _hopper(16, 6, 0, "cpu")
        xi = torch.cat((xi, torch.ones(16, 1)), -1)
        model.fit_normalization(states, actions, xi, aux)
        self.assertEqual(model.xi_std[2].item(), 1.0)
        self.assertGreater(model.xi_std[:2].min().item(), 0.0)
        self.assertLess(model.xi_std[:2].max().item(), 1.0)

    def test_rollout_is_differentiable_and_shifts_history(self):
        """Propagate gradients to actions and xi and carry past states and actions in the model state."""
        torch.manual_seed(0)
        model = WorldModel(1, 2, 3, history_length=2, hidden=(32, 32))
        states, actions, xi, aux = _hopper(16, 12, 0, "cpu")
        model.fit_normalization(states, actions, xi, aux)
        torch.nn.init.normal_(model.net[-1].weight, std=0.1)
        actions = actions.clone().requires_grad_()
        xi = xi.clone().requires_grad_()
        start = model.start(states[:, :3], actions[:, :2])
        predicted, predicted_aux = model.rollout(start, actions[:, 2:5], xi)
        (predicted[:, -1].square().sum() + predicted_aux.square().sum()).backward()
        for grad in (actions.grad[:, 2:5], xi.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.0)
        self.assertLess((predicted[..., 3:7].norm(dim=-1) - 1.0).abs().max().item(), 1.0e-6)
        next_state, _ = model(start, actions[:, 2], xi)
        expected = model.start(torch.cat((states[:, 1:3], next_state[:, None, :15]), 1), actions[:, 1:3])
        self.assertTrue(torch.equal(next_state, expected))

    def test_history_recovers_hidden_foam_memory(self):
        """Cut the held-out contact force error by a third with four steps of history.

        The current state does not reveal the deflection of the foam's Maxwell
        element, so a model without history misses part of the contact force.
        """
        torch.manual_seed(0)
        train, held_out = _hopper(512, 150, 0, "cpu"), _hopper(64, 150, 1, "cpu")
        generator = torch.Generator().manual_seed(2)

        def sample(data, count):
            states, actions, xi, aux = data
            episode = torch.randint(0, states.shape[0], (count, 1), generator=generator)
            steps = torch.randint(4, 150, (count, 1), generator=generator) + torch.arange(-4, 2)
            return (
                states[episode, steps],
                actions[episode, steps[:, :-1]],
                xi[episode[:, 0]],
                aux[episode, steps[:, :-1]],
            )

        test = sample(held_out, 2048)
        errors = {}
        for history_length in (0, 4):
            model = WorldModel(1, 2, 3, history_length=history_length, hidden=(128, 128))
            skip = 4 - history_length
            model.fit_normalization(*(x[:, skip:] if x.ndim > 2 else x for x in sample(train, 2048)))
            optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
            for _ in range(1500):
                loss = model.loss(*(x[:, skip:] if x.ndim > 2 else x for x in sample(train, 512)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            states, actions, xi, aux = test
            with torch.no_grad():
                start = model.start(states[:, skip:5], actions[:, skip:4])
                _, predicted = model.rollout(start, actions[:, 4:], xi)
            contact = aux[:, 4:, 0] > 0.0
            errors[history_length] = (predicted[..., 0] - aux[:, 4:, 0])[contact].square().mean().sqrt().item()
        self.assertLess(errors[4], 0.67 * errors[0], errors)
