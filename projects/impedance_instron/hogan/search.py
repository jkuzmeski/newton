# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan or run a deterministic full-parameter Hogan impedance search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .runner import Runner

MOUNT_M = (-0.03186147427106201, 0.0, 0.10943209684347802)
SPEED_M_S = 3.65
COMPRESSION_LIMIT = 0.99
FORCE_FILTER_HZ = 20.0
FORCE_VIBRATION_WEIGHT = 1.0
DAMPING_MODES = ("lagged", "immediate")
RESPONSE_MULTIPLIERS = (0.5, 1.0, 2.0)
REGULARIZATIONS = (0.001, 0.01)
REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    """Hash a file without loading large inputs into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_candidates(models: list[Path]) -> list[dict]:
    """Return the fixed 24-point source, dynamics, response, and regularization grid."""
    if len(models) != 2:
        raise ValueError("Exactly two --models are required: seed fit and vibration-1 full fit")
    candidates = []
    for source_index, source in enumerate(models):
        for damping in DAMPING_MODES:
            for response_multiplier in RESPONSE_MULTIPLIERS:
                for regularization in REGULARIZATIONS:
                    candidates.append(
                        {
                            "id": f"s{source_index}_{damping}_r{response_multiplier:g}_reg{regularization:g}",
                            "source_model": str(source.resolve()),
                            "source_index": source_index,
                            "damping_mode": damping,
                            "response_time_multiplier": response_multiplier,
                            "regularization": regularization,
                        }
                    )
    return candidates


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command(
    args, action: str, output: Path, model: Path, *, iterations: int, regularization=0.01, bound=1.5
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "projects.impedance_instron.hogan.identify",
        action,
        "--dataset",
        str(args.dataset.resolve()),
        "--output",
        str(output.resolve()),
        "--model",
        str(model.resolve()),
        "--device",
        args.device,
        "--dt",
        "0.000125",
        "--mount",
        *(str(x) for x in MOUNT_M),
        "--speed",
        str(SPEED_M_S),
        "--compression-limit",
        str(COMPRESSION_LIMIT),
        "--force-filter-hz",
        str(FORCE_FILTER_HZ),
        "--force-vibration-weight",
        str(FORCE_VIBRATION_WEIGHT),
    ]
    if action == "fit":
        argv += [
            "--iterations",
            str(iterations),
            "--chunk",
            str(args.chunk),
            "--lm-bound",
            str(bound),
            "--lm-regularization",
            str(regularization),
        ]
    return argv


def build_plan(args) -> dict:
    """Create the immutable search plan and its full candidate matrix."""
    candidates = make_candidates(args.models)
    for candidate in candidates:
        source = Path(candidate["source_model"])
        candidate["source_sha256"] = sha256(source) if source.is_file() else None
        try:
            response = Runner.load(source).response_time_s
            candidate["response_time_initial_s"] = response
            candidate["response_time_effective_s"] = min(
                0.15, max(0.005, response * candidate["response_time_multiplier"])
            )
        except (OSError, ValueError, KeyError, TypeError):
            candidate["response_time_initial_s"] = None
            candidate["response_time_effective_s"] = None
    manifest = args.dataset.resolve() / "manifest.json"
    inputs = [manifest, *(path.resolve() for path in args.models)]
    source_files = [
        Path(__file__),
        Path(__file__).with_name("identify.py"),
        Path(__file__).with_name("runner.py"),
        Path(__file__).with_name("least_squares.py"),
    ]
    return {
        "schema": "hogan-full-parameter-search-v1",
        "domain": "impedance-instron",
        "run_id": args.output.name,
        "created_at": _stamp(),
        "question": "Find the lowest common training score across a deterministic broad initialization and dynamics grid.",
        "baseline": {"models": [str(p.resolve()) for p in args.models], "evaluated_frozen": True},
        "changed_variables": {
            "initialization_source": [str(p.resolve()) for p in args.models],
            "damping_mode": list(DAMPING_MODES),
            "response_time_multiplier": list(RESPONSE_MULTIPLIERS),
            "lm_regularization": list(REGULARIZATIONS),
        },
        "matched_configuration": {
            "dt_s": 0.000125,
            "lm_step": 0.01,
            "lm_initial_damping": 0.01,
            "lm_tolerance": 0.0001,
            "mount_m": list(MOUNT_M),
            "speed_m_s": SPEED_M_S,
            "compression_limit": COMPRESSION_LIMIT,
            "force_filter_hz": FORCE_FILTER_HZ,
            "force_vibration_weight": FORCE_VIBRATION_WEIGHT,
            "device": args.device,
            "chunk": args.chunk,
            "coarse_iterations": args.iterations,
            "refine_iterations": args.refine_iterations,
            "refine_bound": 1.5,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "effective_configuration": {"dataset": str(args.dataset.resolve()), "candidates": candidates},
        "command": "uv run --no-sync -m projects.impedance_instron.hogan.search --dataset DATASET --models SEED FULL --output OUTPUT --run",
        "stages": [
            {
                "name": "frozen_baselines",
                "count": 2,
                "configuration": "Evaluate each supplied model at common 20 Hz / vibration weight 1.",
            },
            {
                "name": "coarse_grid",
                "count": len(candidates),
                "iterations": args.iterations,
                "configuration": "Fit each grid point on all train stances; stop launching at 70% of total budget.",
            },
            {
                "name": "refinement",
                "count": args.finalists,
                "iterations": args.refine_iterations,
                "configuration": "Refit top eligible coarse candidates from their saved runners with bound 1.5.",
            },
        ],
        "seed": "Deterministic initial models and LM; no random sampling.",
        "splits": {
            "fit": "all train stances",
            "evaluation": "eval metrics reported separately",
            "selection": "common identify.score mean_loss on train only",
        },
        "selection_rule": (
            "Select the lowest finite common training mean_loss among zero-training-failure frozen baselines, "
            "coarse fits, and refinements; tie-break by failure count, loss, then candidate id. Eval is never used. "
            "This finite search does not establish a global optimum."
        ),
        "budget": {
            "budget_minutes": args.budget_minutes,
            "reserved_fraction": 0.15,
            "coarse_stop_fraction": 0.70,
            "capture_and_export_reserve_minutes": args.budget_minutes * 0.15,
            "execution": "serial fits/evaluations; each completed stage is checkpointed",
        },
        "stopping_conditions": "At each stage boundary, stop launching jobs once elapsed time reaches 85% of budget.",
        "qualification_checks": [
            "Input compatibility remains enforced by identify.fit",
            "Only zero-failure training results are eligible",
        ],
        "replay_samples": [
            {
                "sample_id": "FR3_1_eval_000",
                "split": "eval",
                "selection": "First compatible held-out manifest member, fixed before search.",
            }
        ],
        "capture_plan": [
            {
                "producer": "hogan.identify",
                "destination": str(args.output.resolve()),
                "verification": "Retain runner, summary, traces and per-stage logs. Replay saved native poses with the shared contact solver; verify shoe/source/initial state and native-step forces, moments and compression before exporting per-column deformation and players.",
            }
        ],
        "inputs": [
            {"path": str(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}
            for path in inputs
        ],
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files if path.is_file()},
    }


def _read_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _training_record(run_id: str, source: Path, summary: dict, stage: str) -> dict:
    train = summary.get("splits", {}).get("train", {})
    scores = train.get("learned", train)
    loss = scores.get("mean_loss")
    return {
        "id": run_id,
        "stage": stage,
        "model": str(source),
        "train_mean_loss": loss if isinstance(loss, (int, float)) and math.isfinite(loss) else None,
        "train_failures": scores.get("failed"),
        "train_metrics": scores,
        "eval_metrics": summary.get("splits", {}).get("eval", {}).get("learned", summary.get("splits", {}).get("eval")),
        "summary": str(source.parent / "summary.json"),
        "exit_code": 0,
    }


def candidate_initial_model(source: Path, candidate: dict, destination: Path) -> Path:
    """Apply candidate dynamics to a complete source model and save its initialization."""
    runner = Runner.load(source)
    model_data = runner.to_dict()
    model_data["immediate_damping"] = candidate["damping_mode"] == "immediate"
    model_data["response_time_s"] = min(
        0.15, max(0.005, runner.response_time_s * candidate["response_time_multiplier"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model_data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def _eligible(records: list[dict]) -> list[dict]:
    return [
        r
        for r in records
        if r.get("train_failures") == 0
        and isinstance(r.get("train_mean_loss"), (int, float))
        and math.isfinite(r["train_mean_loss"])
    ]


def _best(records: list[dict]) -> dict | None:
    viable = _eligible(records)
    return min(viable, key=lambda r: (r["train_failures"], r["train_mean_loss"], r["id"])) if viable else None


def _write_state(output: Path, state: dict) -> None:
    state["updated_at"] = _stamp()
    (output / "summary.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args, plan: dict) -> dict:
    """Run stages serially, checkpointing all outcomes and respecting the time reserve."""
    output = args.output.resolve()
    started = time.monotonic()
    coarse_deadline_s = args.budget_minutes * 60 * 0.70
    deadline_s = args.budget_minutes * 60 * 0.85
    state = {
        "schema": "hogan-full-parameter-search-results-v1",
        "status": "running",
        "started_at": _stamp(),
        "plan": "experiment-plan.json",
        "stages": [],
        "candidates": [],
    }
    _write_state(output, state)

    def remaining() -> bool:
        return time.monotonic() - started < deadline_s

    def execute(run_id: str, argv: list[str], stage: str, record_meta: dict) -> dict | None:
        if not remaining():
            state["status"] = "budget_exhausted"
            return None
        log = output / "logs" / run_id
        log.mkdir(parents=True, exist_ok=False)
        begin = time.monotonic()
        with (
            (log / "stdout.log").open("w", encoding="utf-8") as stdout,
            (log / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            try:
                exit_code = subprocess.run(argv, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False).returncode
            except OSError as exc:
                stderr.write(f"Could not launch candidate command: {exc}\n")
                exit_code = -1
        elapsed = time.monotonic() - begin
        row = {
            "id": run_id,
            "stage": stage,
            **record_meta,
            "argv": argv,
            "stdout": str(log / "stdout.log"),
            "stderr": str(log / "stderr.log"),
            "wall_s": elapsed,
            "exit_code": exit_code,
        }
        state["stages"].append(row)
        if exit_code == 0:
            try:
                parsed = _training_record(
                    run_id,
                    Path(argv[argv.index("--output") + 1]) / "runner.json",
                    _read_summary(Path(argv[argv.index("--output") + 1])),
                    stage,
                )
                parsed.update({k: v for k, v in record_meta.items() if k not in parsed})
                parsed["wall_s"] = elapsed
                state["candidates"].append(parsed)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["parse_error"] = str(exc)
        else:
            state["candidates"].append(
                {
                    "id": run_id,
                    "stage": stage,
                    **record_meta,
                    "train_mean_loss": None,
                    "train_failures": None,
                    "exit_code": exit_code,
                    "wall_s": elapsed,
                    "stdout": row["stdout"],
                    "stderr": row["stderr"],
                }
            )
        _write_state(output, state)
        return row

    # Evaluate the supplied frozen baselines using the identical common score.
    for i, model in enumerate(args.models):
        dest = output / "baselines" / f"source_{i}"
        execute(
            f"baseline_{i}",
            _command(args, "evaluate", dest, model, iterations=0),
            "baseline",
            {"source_model": str(model.resolve()), "source_sha256": sha256(model)},
        )
        if state["status"] == "budget_exhausted":
            break

    # Fit every planned candidate, independently initialized from its declared source.
    coarse_stopped = False
    for candidate in plan["candidates"]:
        if state["status"] == "budget_exhausted":
            break
        if time.monotonic() - started >= coarse_deadline_s:
            coarse_stopped = True
            state["coarse_status"] = "stopped_to_reserve_refinement_and_capture_time"
            _write_state(output, state)
            break
        source = Path(candidate["source_model"])
        init_dir = output / "initial_models"
        init_path = init_dir / f"{candidate['id']}.json"
        candidate_initial_model(source, candidate, init_path)
        dest = output / "coarse" / candidate["id"]
        argv = _command(
            args, "fit", dest, init_path, iterations=args.iterations, regularization=candidate["regularization"]
        )
        execute(candidate["id"], argv, "coarse", candidate)

    coarse = [r for r in state["candidates"] if r["stage"] == "coarse"]
    if state["status"] == "budget_exhausted" and len(coarse) < len(plan["candidates"]):
        coarse_stopped = True
    for rank, candidate in enumerate(
        sorted(_eligible(coarse), key=lambda r: (r["train_mean_loss"], r["id"]))[: args.finalists]
    ):
        if state["status"] == "budget_exhausted":
            break
        source = Path(candidate["model"])
        dest = output / "refined" / candidate["id"]
        execute(
            f"refine_{rank}_{candidate['id']}",
            _command(
                args,
                "fit",
                dest,
                source,
                iterations=args.refine_iterations,
                regularization=candidate["regularization"],
                bound=1.5,
            ),
            "refinement",
            {
                "coarse_candidate": candidate["id"],
                "regularization": candidate["regularization"],
                "source_model": candidate.get("source_model"),
                "source_sha256": candidate.get("source_sha256"),
            },
        )

    winner = _best(state["candidates"])
    state["best"] = winner
    if winner:
        best_path = output / "best.json"
        shutil.copyfile(winner["model"], best_path)
        state["best_model"] = str(best_path)
        state["best_model_sha256"] = sha256(best_path)
        state["best_source_sha256"] = winner.get("source_sha256")
        state["best_common_train_loss"] = winner["train_mean_loss"]
        state["selection_rule"] = plan["selection_rule"]
    state["coarse_grid_complete"] = not coarse_stopped
    if state["status"] != "budget_exhausted":
        state["status"] = "budget_exhausted" if coarse_stopped else "completed"
    state["finished_at"] = _stamp()
    state["wall_s"] = time.monotonic() - started
    _write_state(output, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--models", type=Path, nargs="+", required=True, help="Seed fit and full fit_vibration_1 runner.json paths"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Execute after saving the experiment plan")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--refine-iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--budget-minutes", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.finalists < 1 or args.refine_iterations < 1 or args.chunk < 1:
        parser.error("iterations, finalists, refine-iterations, and chunk must be positive")
    if not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0:
        parser.error("budget-minutes must be finite and positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output {args.output}")
    try:
        plan = build_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True)
    (args.output / "experiment-plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not args.run:
        print(f"planned {len(plan['candidates'])} candidates in {args.output}")
        return 0
    missing = [entry["path"] for entry in plan["inputs"] if not entry["exists"]]
    if missing:
        state = {"status": "blocked_missing_inputs", "missing_inputs": missing, "plan": "experiment-plan.json"}
        _write_state(args.output, state)
        print("missing required inputs; no jobs launched")
        return 2
    try:
        for model in args.models:
            Runner.load(model)
        json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        state = {"status": "blocked_invalid_inputs", "error": str(exc), "plan": "experiment-plan.json"}
        _write_state(args.output, state)
        print("invalid model or dataset manifest; no jobs launched")
        return 2
    run(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
