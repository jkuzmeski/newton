# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from projects.supermimic.buffer import ReplayBuffer
except ImportError:
    torch = None


def _fill(buffer: ReplayBuffer, steps: int, reset_at: dict[int, list[int]], done_at: dict[int, list[int]]):
    """Push rows whose state holds (world, push count) and whose action holds the push count."""
    w = buffer.world_count
    for t in range(steps):
        reset = torch.zeros(w, dtype=torch.bool, device=buffer.device)
        done = torch.zeros_like(reset)
        reset[reset_at.get(t, [])] = True
        done[done_at.get(t, [])] = True
        if t == 0:
            reset[:] = True
        state = torch.stack(
            (torch.arange(w, device=buffer.device).float(), torch.full((w,), float(t), device=buffer.device)), 1
        )
        buffer.push(
            state,
            torch.full((w, 1), float(t), device=buffer.device),
            torch.zeros((w, 2), device=buffer.device),
            torch.zeros((w, 3), device=buffer.device),
            torch.full((w,), t, dtype=torch.int64, device=buffer.device),
            reset,
            done,
        )


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "PyTorch with CUDA is required")
class TestSupermimicBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(3, 20, state_dim=2, action_dim=1, xi_dim=2, aux_dim=3)

    def _check_windows(self, batch, steps, history):
        """Assert that windows are contiguous rows of one world and one episode."""
        state, reset = batch["state"], batch["reset"]
        self.assertTrue(torch.all(state[:, :, 0] == state[:, :1, 0]))
        t = state[:, history:, 1]
        self.assertTrue(torch.all(t[:, 1:] - t[:, :-1] == 1))
        self.assertFalse(reset[:, history + 1 :].any())
        self.assertFalse(batch["done"][:, history : history + steps].any())

    def test_windows_never_cross_resets(self):
        """Sample k-step windows inside episodes, with terminal states only at the window end."""
        _fill(self.buffer, 15, {6: [0], 7: [1, 2], 11: [0]}, {5: [0], 6: [1]})
        batch = self.buffer.sample(4096, 4, generator=torch.Generator("cuda").manual_seed(0))
        self.assertEqual(tuple(batch["state"].shape), (4096, 5, 2))
        self._check_windows(batch, 4, 0)
        starts = {(int(w), int(t)) for w, t in batch["state"][:, 0].tolist()}
        valid = (
            {(0, t) for t in (0, 1, 6)}
            | {(1, t) for t in (0, 1, 2, 7, 8, 9, 10)}
            | {(2, t) for t in (0, 1, 2, 7, 8, 9, 10)}
        )
        self.assertEqual(starts, valid)
        self.assertTrue(batch["done"][batch["state"][:, 0, 0] == 0][:, -1].any())

    def test_history_repeats_the_episode_start(self):
        """Fill history rows before an episode start with the episode's first row."""
        _fill(self.buffer, 15, {6: [0, 1, 2]}, {})
        batch = self.buffer.sample(4096, 2, history=3, generator=torch.Generator("cuda").manual_seed(1))
        self._check_windows(batch, 2, 3)
        t = batch["state"][:, :, 1]
        first = torch.where(t[:, 3] >= 6, 6.0, 0.0)
        self.assertTrue(
            torch.all(t[:, :3] == torch.maximum(t[:, 3:4] + torch.arange(-3, 0, device="cuda"), first[:, None]))
        )
        self.assertEqual(int(t[:, 3].min()), 3)
        self.assertTrue(bool((t[:, 3] == 6).any()))

    def test_ring_replaces_oldest_rows(self):
        """Keep the newest rows after wrapping and never start a window before the oldest row held."""
        _fill(self.buffer, 47, {30: [1]}, {})
        self.assertEqual(self.buffer.size, 20)
        batch = self.buffer.sample(4096, 3, history=2, generator=torch.Generator("cuda").manual_seed(2))
        self._check_windows(batch, 3, 2)
        t = batch["state"][:, :, 1]
        self.assertEqual(int(t.min()), 27)
        self.assertEqual(int(t.max()), 46)

    def test_save_and_load_round_trip(self):
        """Restore the held rows in chronological order and sample the same windows."""
        _fill(self.buffer, 33, {25: [2]}, {24: [2]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "buffer.npz"
            self.buffer.save(path)
            loaded = ReplayBuffer.load(path)
        self.assertEqual((loaded.world_count, loaded.capacity, loaded.size), (3, 20, 20))
        self.assertTrue(torch.equal(loaded.valid_starts(4, 1), self.buffer.valid_starts(4, 1)))
        a = self.buffer.sample(256, 4, history=1, generator=torch.Generator("cuda").manual_seed(3))
        b = loaded.sample(256, 4, history=1, generator=torch.Generator("cuda").manual_seed(3))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
