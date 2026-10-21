# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""History-conditioned shoe adaptation: the student of the design document.

The teacher sees the shoe parameters (theta) as an input; a deployed runner cannot, so
the student infers them from its own recent contact-interface history. The recipe is
RMA (Kumar et al., arXiv 2107.04034) as verified from the full text: a per-scalar
projection, three 1-D convolutions over a 50-step state-action history, a max over time,
and a small embedding that the policy consumes; the estimator is queried slower than the
policy (10 Hz here, 100 Hz for the gait); and the loss sits on the embedding, with the
physical parameters as an auxiliary readout, because RMA\'s variant that regressed the
parameters directly lost about 17 points of success. Never a flattened-history MLP - the
in-hand RMA ablation (arXiv 2210.04887) shows PPO collapsing as the flattened history
grows.

Two differences from RMA, both deliberate and both reported with the results:

* the readout regresses theta itself (:func:`~.shoe_randomization.normalize`), because
  here the privileged vector *is* a physical parameter vector that the evaluation can
  score. Whether an implicit student would do as well is the identifiability ablation of
  gate G6, run by setting the auxiliary weight to zero.
* the estimator is trained by supervised regression on snapshots taken during collection,
  while the policy treats the embedding as an observation feature. Gradients do not flow
  from the PPO loss through the encoder, so the collection loop keeps no 50-step graph.
  A-RMA\'s phase 3 (arXiv 2205.15299) - fine-tune the policy against the estimator\'s own
  wrong estimate with the estimator frozen - is the ``frozen`` mode of the trainer, and it
  is also what closes the gap between the estimate used to collect and the estimate used
  to deploy.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .env3d import ACTION_NAMES, BilateralEnv3D
from .policy import Actor
from .shoe_randomization import N_THETA

HISTORY_STEPS = 50
"""Steps of state-action history the encoder reads: 0.5 s at the 100 Hz policy rate, the
window RMA used at 100 Hz. Half a second is a bit more than one stance and less than a
full stride, which is the honest size of the information: the estimator sees one footfall."""

QUERY_EVERY = 10
"""Control steps between estimator queries: the encoder runs at 10 Hz, the policy at 100 Hz."""

EMBEDDING_SIZE = 8
"""Width of the adaptation embedding the policy consumes."""

EMBEDDING_NAMES = tuple(f"shoe_embedding_{index}" for index in range(EMBEDDING_SIZE))
"""Observation columns the student gets on top of the environment\'s own."""


class HistoryBuffer:
    """A fixed-depth per-world ring of the encoder inputs, with a reset mask.

    A world that restarts must not feed the encoder the history of the episode that
    ended: the buffer is cleared for those rows, and the encoder sees its own
    zero-padding, which is also what it will see at the start of every deployment.
    """

    def __init__(self, worlds: int, channels: int, steps: int = HISTORY_STEPS, device: str = "cuda:0") -> None:
        self.worlds, self.channels, self.steps = int(worlds), int(channels), int(steps)
        self.data = torch.zeros(self.worlds, self.channels, self.steps, device=device)
        self.filled = torch.zeros(self.worlds, dtype=torch.long, device=device)

    def push(self, values: torch.Tensor) -> None:
        """Append one control step of encoder inputs ``[worlds, channels]``, oldest first."""
        self.data.copy_(torch.roll(self.data, -1, dims=-1))
        self.data[:, :, -1].copy_(values)
        self.filled.add_(1)

    def windows(self) -> torch.Tensor:
        """The current history, ``[worlds, channels, steps]``, oldest step first."""
        return self.data

    def snapshot(self, rows: torch.Tensor) -> torch.Tensor:
        """Copy the history of ``rows`` out of the buffer, for a supervised target later."""
        return self.data[rows].clone()

    def clear(self, mask: torch.Tensor) -> None:
        self.data.masked_fill_(mask.reshape(self.worlds, 1, 1), 0.0)
        self.filled.masked_fill_(mask, 0)


class ShoeEncoder(nn.Module):
    """Infer the shoe from a step history of contact-interface channels.

    Args:
        channels: Encoder inputs per step (observation plus the previous action).
        steps: History depth.
        scalar_width: Width of the per-scalar projection of every input value.
        embedding: Width of the returned embedding.
        theta: Width of the auxiliary readout; zero disables it.
    """

    def __init__(
        self,
        channels: int,
        steps: int = HISTORY_STEPS,
        *,
        scalar_width: int = 32,
        embedding: int = EMBEDDING_SIZE,
        theta: int = N_THETA,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.steps = int(steps)
        self.scalar_width = int(scalar_width)
        self.project = nn.Sequential(nn.Linear(1, self.scalar_width), nn.LeakyReLU())
        self.convolutions = nn.Sequential(
            nn.Conv1d(self.channels * self.scalar_width, 32, kernel_size=5),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.LeakyReLU(),
        )
        self.head = nn.Linear(32, embedding)
        self.readout = nn.Linear(embedding, theta) if theta else None

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(embedding, theta_hat)`` for histories ``[worlds, channels, steps]``.

        Every scalar of the history is projected on its own, which is what lets one
        network read a travel channel and a torque channel the same way; the projection
        is the widest tensor in the model, so :func:`encode` feeds it in slices.
        """
        batch = history.shape[0]
        values = history.permute(0, 2, 1).reshape(-1, 1)
        projected = self.project(values).view(batch, self.steps, self.channels, self.scalar_width)
        widened = projected.permute(0, 2, 3, 1).reshape(batch, self.channels * self.scalar_width, self.steps)
        convolved = self.convolutions(widened).amax(dim=-1)
        embedding = self.head(convolved)
        return embedding, None if self.readout is None else self.readout(embedding)

    def describe(self) -> dict:
        return {
            "channels": self.channels,
            "steps": self.steps,
            "scalar_width": self.scalar_width,
            "embedding": int(self.head.out_features),
            "readout": None if self.readout is None else int(self.readout.out_features),
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def grow_actor(actor: Actor, add_names: tuple[str, ...], drop_names: tuple[str, ...] = ()) -> Actor:
    """Rebuild an actor with extra observation columns, keeping what it already knows.

    Fine-tuning a tracker into an adaptive policy changes the actor\'s input width: the
    teacher gains the shoe parameters, the student gains the contact channels and the
    embedding. The first layer is copied column by column for the channels the two
    interfaces share, and the new columns start at zero weight with the running
    normalizer assuming mean zero and unit variance, so the grown policy is exactly the
    policy it was until the new channels earn their weight. This is what makes the
    ablation ladder comparable: every arm starts from the same behaviour.
    """
    old_names = list(actor.observation_names)
    new_names = [name for name in old_names if name not in set(drop_names)]
    keep = [old_names.index(name) for name in new_names]
    grown = Actor(
        actor.hidden,
        seed_logits=actor.net[-1].bias.detach().cpu().numpy(),
        observation_names=tuple(new_names) + tuple(add_names),
        action_names=actor.action_names,
    ).to(next(actor.parameters()).device)
    with torch.no_grad():
        # Everything the two networks agree on by shape is carried over as it stands - the
        # hidden layers and the output bias - and only the input layer needs remapping,
        # because it is the one whose width changes. Copying the input layer alone would
        # leave the trunk freshly randomised, which is a policy that does not run at all.
        state = grown.net.state_dict()
        for key, value in actor.net.state_dict().items():
            if key in state and state[key].shape == value.shape:
                state[key] = value.clone()
        source = actor.net[0]
        # ``keep`` indexes the shared channels on both sides: the dropped columns are gone
        # from the target, whose remaining inputs are in the same order and start at zero.
        state["0.weight"][:, keep] = source.weight[:, keep]
        state["0.weight"][:, len(keep) :] = 0.0
        grown.net.load_state_dict(state)
        grown.normalizer.mean[: len(keep)].copy_(actor.normalizer.mean[keep])
        grown.normalizer.var[: len(keep)].copy_(actor.normalizer.var[keep])
        grown.normalizer.count.copy_(actor.normalizer.count)
    return grown


@torch.no_grad()
def encode(encoder: ShoeEncoder, history: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Embed every world's history, a slice of worlds at a time.

    The per-scalar projection is ``worlds x steps x channels x width`` wide: at the four
    thousand worlds of a training run and RMA's own 32-unit projection that is gigabytes,
    more than the simulation itself holds. Chunking the batch changes nothing but the
    arithmetic order, and the estimate is not differentiated through anyway.
    """
    if history.shape[0] <= chunk:
        return encoder(history)[0]
    return torch.cat([encoder(history[i : i + chunk])[0] for i in range(0, history.shape[0], chunk)])


class AdaptiveRunner(nn.Module):
    """The deployed student: policy plus estimator plus the history the estimator reads.

    Args:
        actor: Policy whose inputs are the environment observation followed by
            :data:`EMBEDDING_NAMES`.
        encoder: The trained :class:`ShoeEncoder` over the same observation plus the
            previous action.
        history_steps: Depth of the buffer, and the stride count it covers.
        query_every: Control steps between estimator evaluations.
    """

    def __init__(
        self,
        actor: Actor,
        encoder: ShoeEncoder,
        *,
        history_steps: int = HISTORY_STEPS,
        query_every: int = QUERY_EVERY,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.encoder = encoder
        self.query_every = int(query_every)
        self.history_steps = int(history_steps)
        self.buffer: HistoryBuffer | None = None
        self.embedding: torch.Tensor | None = None
        self._action: torch.Tensor | None = None
        self._steps = 0

    def observation_names(self, env: BilateralEnv3D) -> tuple[str, ...]:
        return tuple(env.observation_names) + EMBEDDING_NAMES

    def channel_names(self, env: BilateralEnv3D) -> tuple[str, ...]:
        """The encoder's inputs: the environment observation, then the last action.

        Not the actor's inputs -- :meth:`observation_names` adds the embedding, which is an
        output of the encoder and never one of its readings -- and the order is the one the
        trainer used (:meth:`push`), so a checkpoint and its channel list cannot drift.
        """
        return tuple(env.observation_names) + ACTION_NAMES

    def reset(self, env: BilateralEnv3D, worlds: int) -> None:
        """Start a fresh rollout with an empty history and no estimate."""
        channels = len(self.channel_names(env))
        self.buffer = HistoryBuffer(worlds, channels, self.history_steps, env.torch_device)
        self.embedding = None
        self._action = torch.zeros(worlds, env.action_size, device=env.torch_device)
        self._steps = 0

    def reset_worlds(self, mask: torch.Tensor) -> None:
        if self.buffer is not None:
            self.buffer.clear(mask)

    def push(self, observation: torch.Tensor, action: torch.Tensor) -> None:
        """Record the state that produced ``action``; call before the next query."""
        assert self.buffer is not None and self._action is not None
        self.buffer.push(torch.cat((observation, self._action), dim=-1).detach())
        self._action.copy_(action)

    def estimate(self) -> torch.Tensor:
        """The current embedding, refreshed every :attr:`query_every` steps."""
        assert self.buffer is not None
        if self.embedding is None or self._steps % self.query_every == 0:
            self.embedding = encode(self.encoder, self.buffer.windows())
        self._steps += 1
        return self.embedding

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """One control step: refresh the estimate, act, and remember what was seen."""
        assert self.embedding is not None or self._steps == 0
        embedding = self.estimate()
        action = self.actor(torch.cat((observation, embedding), dim=-1))
        return action


def readout_error(theta_hat: torch.Tensor, theta: torch.Tensor) -> dict:
    """Estimator accuracy as a fraction of the prior width, per theta component.

    Gate G2 of the evaluation protocol asks for less than a quarter of the prior width,
    and asks it per component, because a parameter that cannot be identified from one
    second of stance should be marginalised over rather than regressed.
    """
    error = (theta_hat - theta).abs()
    return {
        "mean": float(error.mean()),
        "median": float(error.median()),
        "p90": float(error.quantile(0.90)),
        "per_component": [round(float(column), 4) for column in error.mean(dim=0)],
    }


def snapshot_rows(worlds: int, count: int, rng: np.random.Generator | None = None) -> torch.Tensor:
    """Rows whose history is copied out for the supervised step, at most ``count`` of them."""
    rng = rng or np.random.default_rng(0)
    if worlds <= count:
        return torch.arange(worlds)
    return torch.as_tensor(rng.choice(worlds, size=count, replace=False), dtype=torch.long)


def theta_target(env: BilateralEnv3D) -> torch.Tensor:
    """The supervised target: theta in the encoding the readout is scored in."""
    return env.shoe_randomizer.privileged()


def cosine_drift(iteration: int, total: int, floor: float) -> float:
    """Auxiliary-loss weight schedule: full weight, eased to ``floor`` by the last iteration."""
    if total <= 0:
        return 1.0
    progress = min(1.0, max(0.0, iteration / total))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
