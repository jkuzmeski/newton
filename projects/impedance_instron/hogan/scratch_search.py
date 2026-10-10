# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit independent unfitted engineering seeds without loading learned checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .runner import Runner
from .search import REPO_ROOT, _best, _command, _training_record, _write_state, sha256


def candidates() -> list[dict]:
    """Return 16 independent engineering initializations and optimizer settings."""
    return [
        {
            "id": f"scratch_f{frequency:g}_t{response:g}_{mode}_reg{regularization:g}",
            "frequency_hz": frequency,
            "response_time_s": response,
            "damping_mode": mode,
            "regularization": regularization,
        }
        for frequency in (1.5, 2.0)
        for response in (0.015, 0.035)
        for mode in ("lagged", "immediate")
        for regularization in (0.0, 0.001)
    ]


def initial_model(candidate: dict) -> Runner:
    """Construct the initialization directly from Runner.seed and fixed settings."""
    data = Runner.seed(reference_speed_m_s=3.65).to_dict()
    data.update(
        frequency_hz=candidate["frequency_hz"],
        response_time_s=candidate["response_time_s"],
        immediate_damping=candidate["damping_mode"] == "immediate",
    )
    return Runner.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--budget-minutes", type=float, default=85.0)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.chunk < 1 or not 0 < args.budget_minutes < float("inf"):
        parser.error("iterations, chunk and budget must be finite and positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    manifest = args.dataset.resolve() / "manifest.json"
    if not manifest.is_file():
        parser.error(f"missing dataset manifest: {manifest}")
    json.loads(manifest.read_text())
    args.output.mkdir(parents=True)
    root = args.output.resolve()
    grid = candidates()
    for candidate in grid:
        path = root / "initial_models" / f"{candidate['id']}.json"
        path.parent.mkdir(exist_ok=True)
        initial_model(candidate).save(path)
        candidate.update(initial_model=str(path), initial_model_sha256=sha256(path))

    def now():
        return datetime.now(timezone.utc).isoformat()

    plan = {
        "schema": "hogan-independent-scratch-search-v1",
        "domain": "impedance-instron",
        "run_id": root.name,
        "created_at": now(),
        "question": "Find the best loss from independently unfitted engineering initializations.",
        "initialization": "Every candidate is constructed directly from Runner.seed; no prior fitted model is read.",
        "candidates": grid,
        "candidate_count": len(grid),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(manifest),
        "iterations_per_independent_fit": args.iterations,
        "lm_bound": 3.0,
        "active_parameters": 119,
        "seed": "Deterministic engineering seeds; no random sampling.",
        "budget_minutes": args.budget_minutes,
        "stopping_conditions": "Each independent fit runs to its iteration cap or native convergence; stop launching at 85% of budget.",
        "selection_rule": "Lowest finite common training mean_loss with zero training failures; held-out metrics never select. No global optimum claim.",
        "replay_samples": [
            {
                "sample_id": "FR3_1_eval_000",
                "split": "eval",
                "selection": "First compatible held-out member fixed before fitting.",
            }
        ],
        "capture_plan": "Save all native traces/scenarios/history; source/shoe/state-checked native-step contact replay exports verified per-column deformation and players per completed condition.",
        "qualification_checks": [
            "Input compatibility remains enforced",
            "Training failures disqualify a model",
            "Replay force/moment/compression agreement",
        ],
        "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    (root / "experiment-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.run:
        print(f"Planned {len(grid)} independent from-scratch fits in {root}")
        return 0
    start = time.monotonic()
    state = {
        "schema": "hogan-independent-scratch-results-v1",
        "status": "running",
        "started_at": now(),
        "plan": "experiment-plan.json",
        "stages": [],
        "candidates": [],
    }
    _write_state(root, state)

    def execute(run_id, action, model, destination, stage, metadata, regularization=0.0):
        log = root / "logs" / run_id
        log.mkdir(parents=True)
        command = _command(
            args, action, destination, model, iterations=args.iterations, regularization=regularization, bound=3.0
        )
        begin = time.monotonic()
        with (log / "stdout.log").open("w") as out, (log / "stderr.log").open("w") as err:
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=out, stderr=err, check=False)
        row = {
            "id": run_id,
            "stage": stage,
            "argv": command,
            "exit_code": result.returncode,
            "wall_s": time.monotonic() - begin,
            "stdout": str(log / "stdout.log"),
            "stderr": str(log / "stderr.log"),
            **metadata,
        }
        state["stages"].append(row)
        if result.returncode == 0:
            summary = json.loads((destination / "summary.json").read_text())
            record = _training_record(run_id, destination / "runner.json", summary, stage)
            record.update(metadata, wall_s=row["wall_s"])
        else:
            record = {**row, "train_mean_loss": None, "train_failures": None}
        state["candidates"].append(record)
        _write_state(root, state)

    baseline = root / "engineering_seed.json"
    Runner.seed(reference_speed_m_s=3.65).save(baseline)
    execute(
        "unfitted_baseline",
        "evaluate",
        baseline,
        root / "baseline",
        "baseline",
        {"initialization": "Unfitted Runner.seed"},
    )
    for candidate in grid:
        if time.monotonic() - start >= args.budget_minutes * 60 * 0.85:
            break
        execute(
            candidate["id"],
            "fit",
            Path(candidate["initial_model"]),
            root / "fits" / candidate["id"],
            "coarse",
            candidate,
            candidate["regularization"],
        )
    winner = _best(state["candidates"])
    attempted = sum(c["stage"] == "coarse" for c in state["candidates"])
    state.update(
        best=winner,
        coarse_grid_complete=attempted == len(grid),
        status="completed" if attempted == len(grid) else "budget_exhausted",
        finished_at=now(),
        wall_s=time.monotonic() - start,
        selection_rule=plan["selection_rule"],
    )
    if winner:
        (root / "best.json").write_bytes(Path(winner["model"]).read_bytes())
        state.update(
            best_model=str(root / "best.json"),
            best_model_sha256=sha256(root / "best.json"),
            best_common_train_loss=winner["train_mean_loss"],
        )
    _write_state(root, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
