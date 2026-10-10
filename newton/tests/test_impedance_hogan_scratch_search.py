# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify independent Hogan fits never initialize from learned checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.hogan import scratch_search
from projects.impedance_instron.hogan.runner import Runner


class TestHoganScratchSearch(unittest.TestCase):
    """Verify unfitted model identities throughout the independent search."""

    def test_every_initialization_retains_unfitted_engineering_weights(self):
        """Construct all initializations without reading any saved runner."""
        expected = Runner.seed(reference_speed_m_s=3.65).weights
        with patch.object(Runner, "load", side_effect=AssertionError("Must not read a fitted model")):
            grid = scratch_search.candidates()
            self.assertEqual(len(grid), 16)
            for candidate in grid:
                model = scratch_search.initial_model(candidate)
                np.testing.assert_array_equal(model.weights, expected)
                self.assertEqual(model.frequency_hz, candidate["frequency_hz"])
                self.assertEqual(model.response_time_s, candidate["response_time_s"])
                self.assertEqual(model.immediate_damping, candidate["damping_mode"] == "immediate")

    def test_full_search_never_reuses_a_fitted_output_as_initialization(self):
        """Fit each candidate from its raw seed while retaining failed attempts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text("{}")
            output = root / "run"
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                source = Path(command[command.index("--model") + 1])
                destination = Path(command[command.index("--output") + 1])
                if command[3] == "fit":
                    self.assertEqual(source.parent, output / "initial_models")
                    np.testing.assert_array_equal(
                        Runner.load(source).weights, Runner.seed(reference_speed_m_s=3.65).weights
                    )
                    if len(calls) == 2:
                        return type("Process", (), {"returncode": 1})()
                destination.mkdir(parents=True)
                (destination / "runner.json").write_bytes(source.read_bytes())
                loss = 100.0 if len(calls) == 1 else 10.0
                (destination / "summary.json").write_text(
                    json.dumps(
                        {
                            "splits": {
                                "train": {"mean_loss": loss, "failed": 0},
                                "eval": {"mean_loss": 999.0, "failed": 0},
                            }
                        }
                    )
                )
                return type("Process", (), {"returncode": 0})()

            with patch.object(scratch_search.subprocess, "run", side_effect=fake_run):
                scratch_search.main(["--dataset", str(dataset), "--output", str(output), "--run"])
            self.assertEqual(len(calls), 17)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(summary["coarse_grid_complete"])
            self.assertEqual(summary["best_common_train_loss"], 10.0)
            self.assertEqual(sum(c["exit_code"] != 0 for c in summary["candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
