# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the single Hogan command surface and preserve the local full-run baseline."""

import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.impedance_instron import __main__ as pipeline
from projects.impedance_instron.hogan import __main__ as hogan_pipeline
from projects.impedance_instron.hogan.fit_report import write_report
from projects.impedance_instron.hogan.identify import load_trials, predict_many, score
from projects.impedance_instron.hogan.runner import RolloutConfig, Runner

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "impedance_instron" / "generative_fit_lm_flightcom_20261008"
MODEL = ROOT / "projects" / "impedance_instron" / "hogan" / "baselines" / "generative_runner_f01_20261008.json"


class TestHoganCommands(unittest.TestCase):
    def test_forward_commands_without_changing_options(self):
        """Forward each command to its existing parser without altering its options."""
        for command, (name, _) in pipeline._COMMANDS.items():
            with self.subTest(command=command):
                module = ModuleType(name)
                seen = []
                module.main = lambda seen=seen: seen.append(sys.argv[1:])
                arguments = [command, "--output", "new-run"]
                original = sys.argv
                with patch.dict(sys.modules, {name: module}):
                    pipeline.main(arguments)
                prefix = [command] if command in pipeline._IDENTIFY_COMMANDS else []
                self.assertEqual(seen, [[*prefix, "--output", "new-run"]])
                self.assertEqual(arguments, [command, "--output", "new-run"])
                self.assertIs(sys.argv, original)

    def test_help_advertises_only_active_workflows(self):
        """Show the generative commands rather than the retired controller pipeline."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            pipeline.main([])
        for command in pipeline._COMMANDS:
            self.assertIn(command, stream.getvalue())
        self.assertNotIn("baseline12", stream.getvalue())
        self.assertIs(hogan_pipeline.main, pipeline.main)

    def test_reject_unknown_or_retired_commands(self):
        """Reject retired tracking commands instead of silently launching another model."""
        for command in ("learn", "control", "quick-fit", "--output"):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    pipeline.main([command])


@unittest.skipUnless((RUN / "summary.json").is_file() and wp.is_cuda_available(), "Local full run and CUDA required")
class TestHoganFullRun(unittest.TestCase):
    """Replay the original local artifacts without refitting or overwriting them."""

    def test_preserve_all_107_predictions_and_report(self):
        """Match every saved full-run trace, split metric, and rendered report figure exactly."""
        artifacts = [path for path in RUN.iterdir() if path.is_file()]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in [MODEL, *artifacts]}
        summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
        command = summary["command"]
        trials = load_trials(
            command["dataset"],
            mount_m=command["mount"],
            pitch_rad=command["pitch"],
            speed_m_s=command["speed"],
            height_offset_m=command["height_offset"],
            friction_model=command["friction_model"],
            limit_per_split=command["limit_per_split"],
        )
        model = Runner.load(MODEL)
        self.assertEqual(model.to_dict(), Runner.load(RUN / "runner.json").to_dict())
        self.assertEqual(len(trials), 107)
        self.assertEqual(sum(trial.split == "train" for trial in trials), 98)
        self.assertEqual([trial.id for trial in trials], [entry["id"] for entry in summary["trials"]])
        predictions = predict_many([model], trials, RolloutConfig(**summary["rollout"]), device=command["device"])[0]
        losses = {"train": [], "eval": []}
        for trial, entry, (trace, info) in zip(trials, summary["trials"], predictions, strict=True):
            with self.subTest(trial=trial.id), np.load(RUN / entry["trace"]) as saved:
                self.assertEqual(set(trace), set(saved.files))
                for key, value in trace.items():
                    np.testing.assert_array_equal(value, saved[key], err_msg=f"{trial.id}: {key}")
                result = score(trace, info, trial, model)
                expected = next(
                    row for row in summary["splits"][trial.split]["learned"]["trials"] if row["id"] == trial.id
                )
                self.assertEqual(result, expected)
                losses[trial.split].append(result["loss"])
        self.assertEqual(float(np.mean(losses["train"])), 16.079667862151016)
        self.assertEqual(float(np.mean(losses["eval"])), 14.796461691170371)
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)
            shutil.copy2(RUN / "summary.json", copy)
            for entry in summary["trials"]:
                shutil.copy2(RUN / entry["trace"], copy)
            report = write_report(copy, device=command["device"]).read_text(encoding="utf-8")
            original = (RUN / "report.html").read_text(encoding="utf-8")
            self.assertEqual(
                re.findall(r"<svg.*?</svg>", report, re.DOTALL), re.findall(r"<svg.*?</svg>", original, re.DOTALL)
            )
            self.assertIn("16.08", report)
            self.assertIn("14.80", report)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})


if __name__ == "__main__":
    unittest.main(verbosity=2)
