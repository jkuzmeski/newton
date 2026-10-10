# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic full Hogan parameter search."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from projects.impedance_instron.hogan import search
from projects.impedance_instron.hogan.runner import Runner


class TestHoganSearch(unittest.TestCase):
    """Verify grid coverage, initialization controls, and safe selection."""

    def _models(self, root: Path) -> list[Path]:
        paths = [root / "seed.json", root / "full.json"]
        for index, path in enumerate(paths):
            runner = Runner.seed(reference_speed_m_s=3.65)
            data = runner.to_dict()
            data["response_time_s"] = 0.011669946 if index == 0 else 0.0061578315
            data["immediate_damping"] = bool(index)
            path.write_text(json.dumps(data), encoding="utf-8")
        return paths

    def test_builds_full_24_configuration_grid(self):
        """Build all source, damping, response, and regularization combinations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = self._models(root)
            candidates = search.make_candidates(models)
            self.assertEqual(len(candidates), 24)
            self.assertEqual({c["damping_mode"] for c in candidates}, {"lagged", "immediate"})
            self.assertEqual({c["response_time_multiplier"] for c in candidates}, {0.5, 1.0, 2.0})
            self.assertEqual({c["regularization"] for c in candidates}, {0.001, 0.01})
            self.assertEqual({c["source_index"] for c in candidates}, {0, 1})

    def test_candidate_dynamics_override_source_and_preserve_model_fields(self):
        """Apply selected dynamics while retaining all other source model values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = self._models(root)[1]
            source = Runner.load(model).to_dict()
            candidate = {"damping_mode": "lagged", "response_time_multiplier": 0.5}
            output = search.candidate_initial_model(model, candidate, root / "candidate.json")
            modified = json.loads(output.read_text())
            self.assertFalse(modified["immediate_damping"])
            self.assertEqual(modified["response_time_s"], 0.005)
            loaded = Runner.from_dict(modified).to_dict()
            for key, value in Runner.from_dict(source).to_dict().items():
                if key not in ("immediate_damping", "response_time_s"):
                    self.assertEqual(loaded[key], value)

    def test_selection_uses_only_common_training_loss(self):
        """Ignore held-out loss and reject candidates with training failures."""
        records = [
            {"id": "winner", "train_mean_loss": 1.0, "train_failures": 0, "eval_metrics": {"mean_loss": 999999}},
            {"id": "failed", "train_mean_loss": 0.01, "train_failures": 1, "eval_metrics": {"mean_loss": 0}},
            {"id": "worse", "train_mean_loss": 2.0, "train_failures": 0, "eval_metrics": {"mean_loss": 0}},
        ]
        self.assertEqual(search._best(records)["id"], "winner")

    def test_invalid_settings_and_missing_inputs_launch_nothing(self):
        """Reject malformed search settings or absent inputs before subprocess work."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = self._models(root)
            with patch("projects.impedance_instron.hogan.search.subprocess.run") as run:
                with self.assertRaises(SystemExit):
                    search.main(
                        [
                            "--dataset",
                            str(root / "absent"),
                            "--models",
                            *(str(p) for p in models),
                            "--output",
                            str(root / "bad"),
                            "--iterations",
                            "0",
                            "--run",
                        ]
                    )
                self.assertEqual(run.call_count, 0)
                result = search.main(
                    [
                        "--dataset",
                        str(root / "absent"),
                        "--models",
                        *(str(p) for p in models),
                        "--output",
                        str(root / "missing"),
                        "--run",
                    ]
                )
                self.assertEqual(result, 2)
                self.assertEqual(run.call_count, 0)

    def test_failed_candidate_is_recorded_and_following_fit_runs(self):
        """Continue after a fit failure and retain per-stage logs and checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = self._models(root)
            dataset = root / "data"
            dataset.mkdir()
            (dataset / "manifest.json").write_text("{}")
            output = root / "result"
            output.mkdir()
            args = Namespace(
                dataset=dataset,
                models=models,
                output=output,
                device="cpu",
                chunk=2,
                iterations=1,
                finalists=1,
                refine_iterations=1,
                budget_minutes=10,
            )
            plan = search.build_plan(args)
            plan["candidates"] = plan["candidates"][:2]
            calls = []

            def fake_run(argv, **kwargs):
                calls.append(argv)
                self.assertNotIn("capture_output", kwargs)
                destination = Path(argv[argv.index("--output") + 1])
                if "coarse/" in str(destination) and len([x for x in calls if "coarse/" in str(x)]) == 1:
                    return type("Process", (), {"returncode": 1})()
                destination.mkdir(parents=True)
                source_arg = argv[argv.index("--model") + 1]
                (destination / "runner.json").write_text(Path(source_arg).read_text())
                (destination / "summary.json").write_text(
                    json.dumps(
                        {
                            "splits": {
                                "train": {"mean_loss": 2.0, "failed": 0},
                                "eval": {"mean_loss": 99.0, "failed": 0},
                            }
                        }
                    )
                )
                return type("Process", (), {"returncode": 0})()

            with patch("projects.impedance_instron.hogan.search.subprocess.run", side_effect=fake_run):
                result = search.run(args, plan)
            self.assertEqual(len(calls), 5)  # 2 baselines, failed + passing fit, finalist refinement
            failed = next(row for row in result["candidates"] if row["stage"] == "coarse" and row["exit_code"] == 1)
            self.assertTrue(Path(failed["stderr"]).is_file())
            self.assertTrue((output / "best.json").is_file())
            self.assertEqual(result["best"]["train_mean_loss"], 2.0)


if __name__ == "__main__":
    unittest.main()
