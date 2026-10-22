# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the merge of stage-0 collect shards."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import torch

    from projects.supermimic.body import STATE_DIM
    from projects.supermimic.buffer import ReplayBuffer
    from projects.supermimic.collect import AUX_DIM, JOINT_COUNT, Stage0Config
    from projects.supermimic.context import XI_COORDINATES, XI_DIM
    from projects.supermimic.grid.stage0 import merge
except ImportError:
    torch = None


def _shard(folder: Path, worlds: int, seed: int) -> None:
    buffer = ReplayBuffer(
        worlds,
        5,
        state_shape=STATE_DIM,
        action_shape=(JOINT_COUNT, 3),
        xi_shape=XI_DIM,
        aux_shape=AUX_DIM,
        device="cpu",
    )
    for row in range(5):
        buffer.push(
            torch.full((worlds, STATE_DIM), float(seed)),
            torch.zeros((worlds, JOINT_COUNT, 3)),
            torch.zeros((worlds, XI_DIM)),
            torch.zeros((worlds, AUX_DIM)),
            reference=torch.zeros(worlds, dtype=torch.int64),
            reset=torch.full((worlds,), row == 0),
            done=torch.zeros(worlds, dtype=torch.bool),
        )
    folder.mkdir()
    buffer.save(folder / "buffer.npz")
    meta = {
        "dt_s": 0.01,
        "shoe": ["puma-fast-r-nitro-elite-3"] * worlds,
        "split": "train",
        "seed": seed,
        "reference": "reference",
        "reference_frames_per_window": 100,
        "xi_coordinates": [c[0] for c in XI_COORDINATES],
        "config": asdict(Stage0Config()),
        "failures": [seed] * worlds,
        "stiffness_short_steps": {"knee_flexion_r": seed + 1},
        "unloaded_stance_steps": [seed, 1],
        "xi": np.zeros((worlds, XI_DIM)).tolist(),
    }
    (folder / "meta.json").write_text(json.dumps(meta))


@unittest.skipUnless(torch is not None, "PyTorch is required")
class TestStage0Merge(unittest.TestCase):
    def test_merge_concatenates_worlds(self):
        """Stack the worlds of the shards in order and add up the per-shard counts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _shard(root / "a", 2, 0)
            _shard(root / "b", 3, 1)
            meta = merge([root / "a", root / "b"], root / "merged")
            buffer = ReplayBuffer.load(root / "merged" / "buffer.npz", device="cpu")
        self.assertEqual(buffer.world_count, 5)
        np.testing.assert_array_equal(buffer.data["state"][:, 0, 0].numpy(), [0, 0, 1, 1, 1])
        self.assertEqual(meta["seed"], [0, 1])
        self.assertEqual(meta["failures"], [0, 0, 1, 1, 1])
        self.assertEqual(meta["stiffness_short_steps"], {"knee_flexion_r": 3})
        self.assertEqual(meta["unloaded_stance_steps"], [1, 2])
        self.assertEqual(meta["summary"]["rows"], 25)
        self.assertEqual(meta["summary"]["sim_failures"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
