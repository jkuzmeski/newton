# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU replay buffer of per-world SuperMimic trajectories.

Every push appends one row per world: the state ``s_t``, the action ``a_t``
executed from it, the sim context ``xi``, the auxiliary outputs of the step
``s_t -> s_{t+1}``, the reference index, and two flags. ``reset`` marks the
first row of an episode. ``done`` marks a terminal state; its row still holds the
action and outputs of one more step, and the world is reset before the next row.
Rows of one world are chronological, so ``s_{t+1}`` is the next row unless that
row is a reset. When the buffer is full, new rows replace the oldest.

A ``k``-step window is ``k + 1`` consecutive rows of one episode: the model rolls
from the first state on the first ``k`` actions, and its last state may be
terminal. An optional history of ``h`` earlier rows precedes the window; rows
before the episode start repeat the episode's first row, as a policy's history
is filled at reset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["ReplayBuffer"]

_FLOAT_FIELDS = ("state", "action", "xi", "aux")


class ReplayBuffer:
    """Fixed-size ring buffer of per-world rows on one device.

    Args:
        world_count: Number of simulated worlds.
        capacity: Rows kept per world.
        state_shape: Shape of a state row.
        action_shape: Shape of an action row.
        xi_shape: Shape of a sim-context row.
        aux_shape: Shape of an auxiliary-output row.
        device: Torch device of the storage.
    """

    def __init__(
        self,
        world_count: int,
        capacity: int,
        *,
        state_shape: int | tuple[int, ...],
        action_shape: int | tuple[int, ...],
        xi_shape: int | tuple[int, ...],
        aux_shape: int | tuple[int, ...],
        device: torch.device | str = "cuda",
    ):
        self.world_count = world_count
        self.capacity = capacity
        self.device = torch.device(device)
        self.rows = 0
        """Rows pushed per world since creation."""
        rows = {"state": state_shape, "action": action_shape, "xi": xi_shape, "aux": aux_shape}
        shape = (world_count, capacity)
        self.data = {
            k: torch.zeros((*shape, *torch.Size(np.atleast_1d(s))), dtype=torch.float32, device=self.device)
            for k, s in rows.items()
        }
        self.data["reference"] = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.data["reset"] = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.data["done"] = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.data["episode_step"] = torch.zeros(shape, dtype=torch.int64, device=self.device)
        """Rows since the episode's first row."""

    @property
    def size(self) -> int:
        """Rows held per world."""
        return min(self.rows, self.capacity)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        xi: torch.Tensor,
        aux: torch.Tensor,
        reference: torch.Tensor,
        reset: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Append one row per world; every argument has the world count as its first dimension."""
        col = self.rows % self.capacity
        if self.rows == 0 and not bool(reset.all()):
            raise ValueError("The first row of every world must be a reset")
        previous = self.data["episode_step"][:, (self.rows - 1) % self.capacity]
        step = torch.where(reset, 0, previous + 1)
        row = {
            "state": state,
            "action": action,
            "xi": xi,
            "aux": aux,
            "reference": reference,
            "reset": reset,
            "done": done,
        }
        for key, value in row.items():
            self.data[key][:, col] = value
        self.data["episode_step"][:, col] = step
        self.rows += 1

    def valid_starts(self, steps: int, history: int = 0) -> torch.Tensor:
        """Return a mask ``[world_count, rows]`` of window starts by chronological row.

        Row ``i`` of the result refers to the ``i``-th oldest row held. A start is
        valid if its window of ``steps + 1`` rows lies in one episode and its
        history of ``history`` rows is held.
        """
        n = self.size
        oldest = self.rows - n
        order = (oldest + torch.arange(n, device=self.device)) % self.capacity
        episode_step = self.data["episode_step"][:, order]
        valid = torch.zeros((self.world_count, n), dtype=torch.bool, device=self.device)
        last = n - steps
        if last > history:
            valid[:, history:last] = episode_step[:, history + steps :] == episode_step[:, history:last] + steps
        return valid

    def sample(
        self, batch_size: int, steps: int, *, history: int = 0, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample windows uniformly over valid starts.

        Args:
            batch_size: Number of windows.
            steps: Window length ``k`` in steps.
            history: Rows of history before each window.
            generator: Random generator on the buffer's device.

        Returns:
            Every stored field, shape ``[batch_size, history + steps + 1, ...]``;
            the window's first state is at index ``history``.
        """
        valid = self.valid_starts(steps, history)
        world, start = torch.nonzero(valid, as_tuple=True)
        if len(world) == 0:
            raise ValueError(f"No valid {steps}-step window with {history} history rows")
        pick = torch.randint(len(world), (batch_size,), device=self.device, generator=generator)
        world, start = world[pick], start[pick] + (self.rows - self.size)
        offset = torch.arange(-history, steps + 1, device=self.device)
        episode_step = self.data["episode_step"][world, start % self.capacity]
        rows = start[:, None] + torch.maximum(offset, -episode_step[:, None])
        cols = rows % self.capacity
        return {key: value[world[:, None], cols] for key, value in self.data.items()}

    def save(self, path: str | Path) -> None:
        """Write the held rows in chronological order to an ``.npz`` file."""
        n = self.size
        order = ((self.rows - n + torch.arange(n, device=self.device)) % self.capacity).cpu()
        np.savez(path, **{key: value.cpu()[:, order].numpy() for key, value in self.data.items()})

    @classmethod
    def load(
        cls, path: str | Path, *, capacity: int | None = None, device: torch.device | str = "cuda"
    ) -> ReplayBuffer:
        """Read a buffer written by :meth:`save`.

        Args:
            path: File written by :meth:`save`.
            capacity: Rows kept per world; defaults to the rows in the file.
            device: Torch device of the storage.
        """
        with np.load(path) as f:
            arrays = {key: f[key] for key in f.files}
        world_count, n = arrays["reset"].shape
        shapes = {f"{key}_shape": arrays[key].shape[2:] for key in _FLOAT_FIELDS}
        buffer = cls(world_count, capacity or n, **shapes, device=device)
        keep = min(n, buffer.capacity)
        for key, value in arrays.items():
            buffer.data[key][:, :keep] = torch.from_numpy(value[:, n - keep :]).to(buffer.device)
        buffer.rows = keep
        return buffer
