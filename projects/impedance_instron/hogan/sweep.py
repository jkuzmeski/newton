# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan and run a small serial force-vibration-weight sweep for Hogan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WEIGHTS = (0.5, 1.0, 2.0)
BASELINE = Path(__file__).with_name("baselines") / "generative_runner_f01_20261008.json"
MOUNT_M = (-0.03186147427106201, 0.0, 0.10943209684347802)
TASK_SPEED_M_S = 3.65
COMPRESSION_LIMIT = 0.99
REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(
    args: argparse.Namespace,
    subcommand: str,
    destination: Path,
    *,
    model: Path | None,
    weight: float | None = None,
    immediate: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "projects.impedance_instron.hogan.identify",
        subcommand,
        "--dataset",
        str(args.dataset.resolve()),
        "--output",
        str(destination.resolve()),
        "--device",
        args.device,
        "--mount",
        *(str(value) for value in MOUNT_M),
        "--speed",
        str(TASK_SPEED_M_S),
        "--compression-limit",
        str(COMPRESSION_LIMIT),
    ]
    if model is not None:
        command += ["--model", str(model.resolve())]
    if subcommand == "fit":
        command += ["--iterations", str(args.iterations), "--chunk", str(args.chunk)]
    command += ["--force-filter-hz", "20", "--force-vibration-weight", str(weight if weight is not None else 1.0)]
    if immediate:
        command.append("--immediate-damping")
    return command


def build_plan(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Build plan and serial commands without starting simulation work."""
    output = args.output.resolve()
    from_scratch = getattr(args, "from_scratch", False)
    baseline = output / "seed_fit" / "runner.json" if from_scratch else args.model.resolve()
    candidates = []
    commands = []
    if from_scratch:
        seed_command = _command(args, "fit", output / "seed_fit", model=None, weight=1.0)
        seed_command[seed_command.index("--iterations") + 1] = "15"
        commands.append(
            {
                "id": "seed_fit",
                "purpose": "Fit a fresh lagged-damping seed for 15 iterations before switching damping.",
                "argv": seed_command,
            }
        )
    commands.append(
        {
            "id": "frozen_baseline_evaluation",
            "purpose": "Evaluate the exact frozen baseline at common force weight 1.",
            "argv": _command(args, "evaluate", output / "baseline_eval", model=baseline, weight=1.0),
        }
    )
    for weight in WEIGHTS:
        run_id = f"fit_vibration_{weight:g}"
        fit_dir = output / run_id
        candidates.append(
            {
                "id": run_id,
                "force_vibration_weight": weight,
                "fit_output": str(fit_dir),
                "common_weight_evaluation": str(output / f"{run_id}_eval_common_w1"),
            }
        )
        commands.append(
            {
                "id": run_id,
                "purpose": f"Fit with vibration weight {weight:g}; immediate damping, 20 Hz force filter.",
                "argv": _command(args, "fit", fit_dir, model=baseline, weight=weight, immediate=True),
            }
        )
        commands.append(
            {
                "id": f"{run_id}_common_weight_evaluation",
                "purpose": "Evaluate the fitted runner at common force weight 1 for comparable reporting.",
                "after": run_id,
                "model_from": str(fit_dir / "runner.json"),
                "argv_template": _command(
                    args, "evaluate", output / f"{run_id}_eval_common_w1", model=fit_dir / "runner.json", weight=1.0
                ),
            }
        )
    dataset_manifest = args.dataset.resolve() / "manifest.json"
    inputs = [dataset_manifest] if from_scratch else [dataset_manifest, baseline]
    input_info = [
        {"path": str(path), "exists": path.is_file(), "sha256": _sha256(path) if path.is_file() else None}
        for path in inputs
    ]
    revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        or None
    )
    source_files = [
        Path(__file__),
        Path(__file__).with_name("identify.py"),
        Path(__file__).with_name("runner.py"),
        Path(__file__).with_name("least_squares.py"),
    ]
    plan = {
        "schema": "hogan-tuning-sweep-v1",
        "run_id": output.name,
        "domain": "impedance-instron",
        "question": "How does explicit stance force-vibration penalty weight affect balanced motion and ground-reaction-force fit?",
        "baseline": {
            "model": str(baseline),
            "frozen_evaluation": "baseline_eval",
            "matched_fit_initialization": True,
            "from_scratch": from_scratch,
        },
        "changed_variables": {"force_vibration_weight": list(WEIGHTS)},
        "matched_configuration": {
            "force_filter_hz": 20,
            "immediate_damping": True,
            "mount_m": list(MOUNT_M),
            "speed_m_s": TASK_SPEED_M_S,
            "compression_limit": COMPRESSION_LIMIT,
            "device": args.device,
            "iterations": args.iterations,
            "chunk": args.chunk,
        },
        "command": "uv run --no-sync -m projects.impedance_instron.hogan.sweep --dataset DATASET --output OUTPUT [--run]",
        "effective_configuration": {
            "dataset": str(args.dataset.resolve()),
            "model": str(baseline),
            "candidates": candidates,
            "run_requested": args.run,
        },
        "seed": "Inherited from deterministic dataset and LM procedure; no additional random seed is set.",
        "splits": {
            "fit": "train only (identify pipeline contract)",
            "reporting": "train and eval separately",
            "selection": "undecided; eval is reporting only",
        },
        "selection_rule": "No automatic selection. Compare training objective and common-weight train/eval metrics; held-out metrics are reporting only.",
        "budget": {
            "candidates": len(WEIGHTS),
            "iterations_each": args.iterations,
            "chunk": args.chunk,
            "execution": "serial GPU fits; baseline and common-weight evaluations included",
            "seed_fit_iterations": 15 if from_scratch else 0,
        },
        "stopping_conditions": "Complete each fit up to the configured LM iterations; preserve failures and stop on first failed subprocess.",
        "qualification_checks": [
            "Input and source hashes recorded",
            "Fit summaries preserve split metrics",
            "Physical/report qualification not performed by this sweep planner",
        ],
        "replay_samples": [],
        "capture_plan": [
            {
                "producer": "hogan.identify fit/evaluate",
                "destination": str(output / "<run>/"),
                "verification": "Retain producer summary, runner, traces, scenarios and report.",
            },
            {
                "producer": "pending saved-contact/deformation capture",
                "destination": str(output / "<run>/deformation/"),
                "verification": "Capture and verify per-column deformation aligned to the selected replay before any report can be marked ready.",
            },
        ],
        "source_revision": revision,
        "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_files if path.is_file()},
        "inputs": input_info,
        "artifact_inventory": "experiment-artifacts.json records required evidence roles; missing evidence remains explicitly missing.",
    }
    return plan, commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--model", type=Path, default=BASELINE)
    initialization.add_argument(
        "--from-scratch", action="store_true", help="Fit a fresh lagged seed for 15 iterations before the sweep"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--run", action="store_true", help="Execute serially; without this option only save the plan")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.chunk < 1:
        parser.error("--iterations and --chunk must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    plan, commands = build_plan(args)
    (output / "experiment-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    roles = {
        "provenance": "Record dataset and source identities and hashes.",
        "configuration": "Save effective fit and evaluation settings.",
        "selected_model": "No candidate is selected by this sweep.",
        "motion_trace": "Capture a representative replay for the selected candidate.",
        "motion_viewer": "Export a playable viewer for the matching replay.",
        "geometry": "Record the actual shoe and limb geometry.",
        "deformation": "Pending aligned per-column deformation capture.",
        "deformation_validation": "Pending verification of replay-aligned deformation.",
        "biomechanics": "Save native measured and simulated channels and comparisons.",
        "training_history": "Fit history is produced by each identify fit command.",
        "metrics": "Save train/eval metrics with units and channel names.",
        "qualification": "Record applicable numerical and physical qualification checks.",
        "report_spec": "Create a spec from complete, identity-matched evidence.",
        "report_html": "Render only after required evidence is complete.",
    }
    inventory = {
        "schema": "instron-experiment-artifacts-v1",
        "domain": "impedance-instron",
        "run_id": output.name,
        "experiment_status": "planned",
        "fitting_performed": True,
        "approved_roots": [str(output)],
        "replay": {
            "model_identity": plan["baseline"]["model"],
            "sample_id": "undetermined until dataset inspection",
            "split": "eval",
            "shoe_identity": "dataset-defined",
            "alignment": {"declared": False, "basis": "Sweep setup does not capture or verify replay alignment."},
        },
        "artifacts": {role: {"status": "missing", "reason": reason} for role, reason in roles.items()},
    }
    for role in ("provenance", "configuration"):
        inventory["artifacts"][role] = {"status": "available", "path": "experiment-plan.json"}
    (output / "experiment-artifacts.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    readiness = [{"path": item["path"], "exists": item["exists"], "sha256": item["sha256"]} for item in plan["inputs"]]
    missing = [item["path"] for item in readiness if not item["exists"]]
    summary = {
        "status": "planned",
        "started_at": None,
        "completed_at": None,
        "source_revision": plan["source_revision"],
        "inputs": readiness,
        "commands": [],
        "failures": [],
        "selection": "undecided; eval is reporting only",
    }
    if args.run:
        dataset_path = args.dataset.resolve()
        readiness_errors = list(missing)
        if not dataset_path.is_dir() or not (dataset_path / "manifest.json").is_file():
            readiness_errors.append(str(dataset_path / "manifest.json"))
        if readiness_errors:
            summary.update(
                status="blocked_missing_inputs",
                failures=[f"Required input missing: {path}" for path in dict.fromkeys(readiness_errors)],
            )
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            raise SystemExit("Cannot run sweep; missing inputs: " + ", ".join(dict.fromkeys(readiness_errors)))
        summary["started_at"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "running"
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for item in commands:
            argv_command = list(item.get("argv", item.get("argv_template", [])))
            if "argv_template" in item:
                prior = next((entry for entry in summary["commands"] if entry["id"] == item["after"]), None)
                model_path = Path(prior["output"]) / "runner.json" if prior else Path(item["model_from"])
                argv_command[argv_command.index("--model") + 1] = str(model_path)
            log_stem = output / item["id"]
            proc = subprocess.run(argv_command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            stdout_path = Path(f"{log_stem}.stdout.txt")
            stderr_path = Path(f"{log_stem}.stderr.txt")
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            record = {
                "id": item["id"],
                "argv": argv_command,
                "returncode": proc.returncode,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "output": str(Path(argv_command[argv_command.index("--output") + 1])),
            }
            producer_summary = Path(record["output"]) / "summary.json"
            if producer_summary.is_file():
                record["summary"] = str(producer_summary)
                if "evaluate" in argv_command:
                    record["common_score_splits"] = json.loads(producer_summary.read_text())["splits"]
            summary["commands"].append(record)
            if proc.returncode:
                summary["failures"].append({"command": item["id"], "returncode": proc.returncode})
                summary["status"] = "failed"
                break
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not summary["failures"]:
            summary["status"] = "completed"
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    inventory["experiment_status"] = summary["status"]
    (output / "experiment-artifacts.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    if args.run and summary["failures"]:
        return 1
    print(f"Sweep {summary['status']}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
