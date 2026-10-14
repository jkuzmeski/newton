# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Execute one indexed run of a grid experiment inside a SLURM array task.

The run's output directory is ``<out-root>/<experiment>/<array job id>/<index>_<name>``,
so resubmitting an experiment never overwrites an earlier job. A record
beside it (``<index>_<name>.run.json``) records the code snapshot, the job, the node, and the exact arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from importlib import import_module
from pathlib import Path

from .experiments import get


def run_directory(out_root: Path, experiment: str, index: int, name: str, smoke: bool) -> Path:
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or time.strftime("local-%Y%m%d-%H%M%S")
    group = f"{experiment}-smoke" if smoke else experiment
    return out_root / group / job / f"{index:02d}_{name}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data-root", type=Path, help="Directory holding the prepared datasets.")
    parser.add_argument("--out-root", type=Path, help="Directory for run outputs.")
    parser.add_argument("--tensorboard-root", type=Path, help="Project TensorBoard log directory.")
    parser.add_argument("--smoke", action="store_true", help="Append the run's smoke-test arguments.")
    parser.add_argument("--print-dataset", action="store_true", help="Print the run's dataset name and exit.")
    parser.add_argument("--no-report", action="store_true", help="Skip the evaluation report after training.")
    parser.add_argument(
        "--report-job", help="Only (re)build the report of this index's run in an earlier array job; no training."
    )
    args = parser.parse_args(argv)
    runs = get(args.experiment)
    if not 0 <= args.index < len(runs):
        raise SystemExit(f"index {args.index} is outside {args.experiment} (0-{len(runs) - 1})")
    run = runs[args.index]
    if args.print_dataset:
        print(run.dataset)
        return
    if args.data_root is None or args.out_root is None:
        parser.error("--data-root and --out-root are required to run")
    if args.report_job:
        group = f"{args.experiment}-smoke" if args.smoke else args.experiment
        report(
            args.out_root / group / args.report_job / f"{args.index:02d}_{run.name}",
            args.data_root / run.dataset,
            args.smoke,
        )
        return
    output = run_directory(args.out_root, args.experiment, args.index, run.name, args.smoke)
    command = ["--dataset", str(args.data_root / run.dataset), "--output", str(output), *run.args]
    if args.smoke:
        command += list(run.smoke_args)
    if args.tensorboard_root is not None and run.tensorboard:
        command += ["--tensorboard", str(args.tensorboard_root / output.relative_to(args.out_root))]
    if output.exists():
        raise SystemExit(f"{output} already exists; each array task needs a fresh run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "experiment": args.experiment,
        "index": args.index,
        "run": run.name,
        "entry": run.entry,
        "arguments": command,
        "smoke": args.smoke,
        "code_commit": os.environ.get("GRID_CODE_COMMIT"),
        "code_label": os.environ.get("GRID_CODE_LABEL"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    # Beside, not inside, the run directory: the trainer requires a fresh output directory.
    output.with_name(f"{output.name}.run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info), flush=True)
    import_module(run.entry).main(command)
    if not args.no_report:
        report(output, args.data_root / run.dataset, args.smoke)


def report(output: Path, dataset: Path, smoke: bool) -> None:
    """Write ``<output>/report``; a failed report is logged in ``report/ERROR.txt`` and does not fail the task."""
    from ..report import run_report  # noqa: PLC0415

    if not (output / "policy.pt").is_file():
        print(f"report skipped: {output / 'policy.pt'} does not exist", flush=True)
        return
    started = time.time()
    try:
        run_report(output, dataset, splits=("eval",) if smoke else ("train", "eval"))
    except Exception:
        (output / "report").mkdir(exist_ok=True)
        (output / "report" / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"report failed:\n{traceback.format_exc()}", flush=True)
        return
    print(f"report: {output / 'report'} ({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
