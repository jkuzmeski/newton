# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hogan tuning sweep planner."""

import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from projects.impedance_instron.hogan import sweep


class TestHoganSweep(unittest.TestCase):
    """Verify safe planning and comparable command construction."""

    def test_from_scratch_fits_seed_before_evaluation_and_sweep(self):
        """Initialize all candidates from a fresh lagged seed fit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = Namespace(
                dataset=root / "data",
                model=sweep.BASELINE,
                output=root / "sweep",
                device="cuda:0",
                iterations=3,
                chunk=128,
                run=False,
                from_scratch=True,
            )
            plan, commands = sweep.build_plan(args)
            seed = commands[0]["argv"]
            self.assertEqual(commands[0]["id"], "seed_fit")
            self.assertNotIn("--model", seed)
            self.assertNotIn("--immediate-damping", seed)
            self.assertEqual(seed[seed.index("--iterations") + 1], "15")
            self.assertEqual(len(plan["inputs"]), 1)
            baseline = commands[1]["argv"]
            self.assertEqual(Path(baseline[baseline.index("--model") + 1]), root / "sweep/seed_fit/runner.json")

    def test_plan_contains_three_vibration_weights_and_common_evaluations(self):
        """Build three fits and evaluate each fit at the same force weight."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = Namespace(
                dataset=root / "dataset",
                model=sweep.BASELINE,
                output=root / "pilot",
                device="cuda:0",
                iterations=3,
                chunk=128,
                run=False,
            )
            plan, commands = sweep.build_plan(args)
        fits = [
            entry for entry in commands if entry["id"].startswith("fit_vibration_") and "_common_" not in entry["id"]
        ]
        self.assertEqual(
            [entry["force_vibration_weight"] for entry in plan["effective_configuration"]["candidates"]],
            [0.5, 1.0, 2.0],
        )
        self.assertEqual(len(fits), 3)
        for entry in commands:
            argv = entry.get("argv", entry.get("argv_template", []))
            if "_common_weight_evaluation" in entry["id"] or entry["id"] == "frozen_baseline_evaluation":
                self.assertEqual(argv[argv.index("--force-vibration-weight") + 1], "1.0")
                self.assertEqual(argv[argv.index("--speed") + 1], "3.65")
                self.assertEqual(argv[argv.index("--compression-limit") + 1], "0.99")

    def test_plan_only_missing_dataset_saves_incomplete_inventory(self):
        """Save a truthful plan without running when the dataset is unavailable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "plan"
            with patch("sys.argv", ["sweep", "--dataset", str(root / "absent"), "--output", str(output)]):
                self.assertEqual(sweep.main(), 0)
            inventory = json.loads((output / "experiment-artifacts.json").read_text())
            self.assertEqual(inventory["artifacts"]["deformation"]["status"], "missing")
            self.assertFalse(json.loads((output / "summary.json").read_text())["inputs"][0]["exists"])
            self.assertFalse((output / "baseline_eval").exists())

    def test_run_requires_manifest_and_does_not_start_subprocesses(self):
        """Block execution when the dataset manifest is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "run"
            with patch("sys.argv", ["sweep", "--dataset", str(root / "dataset"), "--output", str(output), "--run"]):
                with patch(
                    "projects.impedance_instron.hogan.sweep.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "revision\n", ""),
                ) as run:
                    with self.assertRaises(SystemExit):
                        sweep.main()
                    self.assertEqual(run.call_count, 1)
                    self.assertEqual(run.call_args.args[0][:2], ["git", "rev-parse"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "blocked_missing_inputs")

    def test_refuses_existing_output(self):
        """Preserve an existing sweep directory without modifying its files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep")
            with patch("sys.argv", ["sweep", "--dataset", str(root / "dataset"), "--output", str(output)]):
                with self.assertRaises(SystemExit):
                    sweep.main()
            self.assertEqual(marker.read_text(), "keep")

    def test_run_executes_serial_commands_and_preserves_unique_logs(self):
        """Run the complete orchestration with mocked producer commands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text("{}")
            output = root / "run"
            calls = []

            def fake_run(argv, **kwargs):
                calls.append((argv, kwargs))
                if argv[:2] == ["git", "rev-parse"]:
                    return subprocess.CompletedProcess(argv, 0, stdout="revision\n", stderr="")
                destination = Path(argv[argv.index("--output") + 1])
                destination.mkdir()
                (destination / "summary.json").write_text(
                    json.dumps({"splits": {"train": {"mean_loss": 2.0}, "eval": {"mean_loss": 3.0}}})
                )
                return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

            with patch("sys.argv", ["sweep", "--dataset", str(dataset), "--output", str(output), "--run"]):
                with patch("projects.impedance_instron.hogan.sweep.subprocess.run", side_effect=fake_run):
                    self.assertEqual(sweep.main(), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(summary["commands"]), 7)
            log_paths = [entry[key] for entry in summary["commands"] for key in ("stdout", "stderr")]
            self.assertEqual(len(log_paths), len(set(log_paths)))
            self.assertTrue(all(Path(path).is_file() for path in log_paths))
            eval_commands = [argv for argv, _ in calls if "evaluate" in argv]
            self.assertEqual(len(eval_commands), 4)
            self.assertTrue(all(argv[argv.index("--force-vibration-weight") + 1] == "1.0" for argv in eval_commands))
            evaluations = [entry for entry in summary["commands"] if "common_score_splits" in entry]
            self.assertEqual(len(evaluations), 4)
            self.assertEqual(evaluations[0]["common_score_splits"]["train"]["mean_loss"], 2.0)
            inventory = json.loads((output / "experiment-artifacts.json").read_text())
            self.assertEqual(inventory["experiment_status"], "completed")

    def test_failed_producer_stops_following_commands(self):
        """Preserve a failed command and stop before launching later candidates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text("{}")
            output = root / "run"
            responses = [
                subprocess.CompletedProcess([], 0, "revision\n", ""),
                subprocess.CompletedProcess([], 1, "", "Input incompatibility"),
            ]
            with patch("projects.impedance_instron.hogan.sweep.subprocess.run", side_effect=responses) as run:
                self.assertEqual(sweep.main(["--dataset", str(dataset), "--output", str(output), "--run"]), 1)
            self.assertEqual(run.call_count, 2)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(len(summary["commands"]), 1)
            self.assertEqual(Path(summary["commands"][0]["stderr"]).read_text(), "Input incompatibility")


if __name__ == "__main__":
    unittest.main()
