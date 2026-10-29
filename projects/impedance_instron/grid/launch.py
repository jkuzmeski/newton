# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run Hogan pipeline tasks on the LLSC grid, one H100 per task.

The ``Z:`` share is the grid home, so files move through ``Z:`` and only short
commands go over ssh. ``submit`` exports one *commit* (never the working tree)
with ``git archive`` to ``code/<sha12>`` once, copies each input folder to
``data/<name>`` once (inputs are immutable: a changed input needs a new name),
writes the task list to the run folder and queues one SLURM array task per
entry. A task runs ``python -u -m <args>`` in the code folder with the grid uv
environment; ``{data}`` and ``{out}`` in its arguments become the data root and
the task's own output folder.

From the repository root::

    uv run --no-sync python -m projects.impedance_instron.grid.launch setup
    uv run --no-sync python -m projects.impedance_instron.grid.launch submit --name hogan_fit --commit <sha> \
        --tasks tasks.json --data f01_v1=<folder> --time 04:00:00
    uv run --no-sync python -m projects.impedance_instron.grid.launch status

``tasks.json`` is a list of ``{"name": ..., "args": [...]}``. Each submission is
appended to ``submissions.jsonl`` with its job id, commit, run folder and log
pattern.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

HOST = "txg-login.llgrid.ll.mit.edu"
REMOTE_ROOT = "/home/gridsan/JO31399/newton_grid/hogan"
LOCAL_ROOT = Path("Z:/newton_grid/hogan")
SNAPSHOT_PATHS = ("newton", "projects", "pyproject.toml", ":(exclude)newton/examples")
"""Snapshot contents; ``newton/tests`` is kept so the impedance regression tests run on the grid."""
TASK_SCRIPT = "projects/impedance_instron/grid/task.sbatch"
PACKAGES = ("warp-lang", "mujoco", "mujoco-warp", "numpy", "scipy", "trimesh", "rtree", "matplotlib")
"""Grid environment packages, pinned to the versions of the local environment."""
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
"""PyTorch wheel index of the local ``torch`` build; the H100 nodes run CUDA 12.8 wheels."""
MAX_GPUS = 8
"""Tasks per submission without asking the user first (the user allowed up to about 8 nodes)."""


def ssh(command: str) -> str:
    """Run one command on the login node and return its stdout."""
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh failed ({result.returncode}): {command}\n{result.stderr}")
    return result.stdout


def setup() -> None:
    """Create the grid folders and the uv environment on the login node (compute nodes are offline)."""
    for name in ("code", "data", "runs", "logs", "warp_cache"):
        (LOCAL_ROOT / name).mkdir(parents=True, exist_ok=True)
    pins = " ".join(f"'{name}=={version(name)}'" for name in PACKAGES)
    torch = f"'torch=={version('torch')}'"
    print(
        ssh(
            "source /etc/profile >/dev/null 2>&1; module load proxy-mitll >/dev/null 2>&1; "
            f"cd {REMOTE_ROOT} && ~/.local/bin/uv venv --allow-existing --python 3.12 venv && "
            f"~/.local/bin/uv pip install --python venv/bin/python {pins} && "
            f"~/.local/bin/uv pip install --python venv/bin/python --index-url {TORCH_INDEX} {torch} && "
            "venv/bin/python -c 'import warp, numpy, scipy, torch; "
            "print(warp.__version__, numpy.__version__, scipy.__version__, torch.__version__)'"
        )
    )


def push_code(commit: str) -> str:
    """Export ``commit`` once to a read-only ``code/<sha12>`` folder; return the full hash."""
    sha = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    short = sha[:12]
    if (LOCAL_ROOT / "code" / short / "GRID_COMMIT").exists():
        return sha
    # Unique archive and folder names let submissions of the same commit push at once.
    archive = LOCAL_ROOT / "code" / f"{short}.{os.getpid()}.tar.gz"
    # Export with the repository's LF endings; core.autocrlf would write CRLF on Windows.
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "archive", "--format=tar.gz", "-o", str(archive),
         sha, "--", *SNAPSHOT_PATHS],
        check=True,
    )  # fmt: skip
    d = f"{REMOTE_ROOT}/code/{short}"
    ssh(
        f"set -e; t=$(mktemp -d {d}.tmp.XXXXXX); tar -xzf {REMOTE_ROOT}/code/{archive.name} -C $t; "
        f"rm -f {REMOTE_ROOT}/code/{archive.name}; echo {sha} > $t/GRID_COMMIT; chmod -R a-w $t; "
        f"if [ -e {d} ]; then chmod -R u+w $t; rm -rf $t; else mv $t {d}; fi"
    )
    return sha


def digest(folder: Path) -> str:
    """Return the SHA-256 of the relative paths and contents of the files in ``folder``."""
    sha = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        sha.update(path.relative_to(folder).as_posix().encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def stage(name: str, source: Path) -> None:
    """Copy ``source`` once to ``data/<name>``; an existing input with the same name is kept as is."""
    target = LOCAL_ROOT / "data" / name
    if target.exists():
        return
    partial = target.with_name(f".{name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(source, partial)
    partial.rename(target)


def submit(
    name: str,
    commit: str,
    tasks: list[dict],
    data: dict[str, Path],
    time_limit: str,
    after: str | None = None,
    exclude: str | None = None,
    memory: str | None = None,
) -> dict:
    """Push the code and inputs, write the task list and queue the array; return the submission record.

    The array waits until every task of job ``after`` succeeded, if given, and
    avoids the comma-separated nodes ``exclude``, if given. ``memory`` replaces
    the host memory per task of the task script, such as ``"128G"``.
    """
    if not 0 < len(tasks) <= MAX_GPUS:
        raise SystemExit(f"{len(tasks)} tasks; at most {MAX_GPUS} GPUs without asking the user")
    sha = push_code(commit)
    for key, source in data.items():
        stage(key, source)
    run = f"{time.strftime('%Y%m%d-%H%M%S')}_{sha[:8]}"
    (LOCAL_ROOT / "runs" / name / run).mkdir(parents=True)
    (LOCAL_ROOT / "runs" / name / run / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    run_dir = f"{REMOTE_ROOT}/runs/{name}/{run}"
    log = f"{REMOTE_ROOT}/logs/{name}-%A_%a.out"
    exports = f"ALL,GRID_ROOT={REMOTE_ROOT},GRID_CODE_DIR={REMOTE_ROOT}/code/{sha[:12]},RUN_DIR={run_dir}"
    dependency = f"--dependency=afterok:{after} --kill-on-invalid-dep=yes " if after else ""
    options = (f"--exclude={exclude} " if exclude else "") + (f"--mem={memory} " if memory else "")
    output = ssh(
        f"cd {REMOTE_ROOT} && sbatch --parsable --job-name=hogan-{name} --array=0-{len(tasks) - 1} --time={time_limit} "
        f"{dependency}{options}"
        f"--output={log} --error={log} --export={exports} {REMOTE_ROOT}/code/{sha[:12]}/{TASK_SCRIPT}"
    )
    job = output.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    record = {
        "job": job,
        "name": name,
        "commit": sha,
        "run_dir": run_dir,
        "log": log,
        "tasks": [t["name"] for t in tasks],
        "after": after,
    }
    with (LOCAL_ROOT / "submissions.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup")
    commands.add_parser("status")
    sub = commands.add_parser("submit")
    sub.add_argument("--name", required=True, help="Experiment name; runs land in runs/<name>/<stamp>_<sha8>.")
    sub.add_argument("--commit", required=True, help="Commit to run; the working tree is never shipped.")
    sub.add_argument("--tasks", type=Path, required=True, help='JSON list of {"name": ..., "args": [...]}.')
    sub.add_argument("--data", action="append", default=[], help="NAME=FOLDER input staged to data/NAME.")
    sub.add_argument("--time", default="01:00:00")
    sub.add_argument("--after", help="Job id whose tasks must all succeed first.")
    sub.add_argument("--exclude", help="Comma-separated nodes to avoid, such as nodes with a broken GPU.")
    args = parser.parse_args()
    if args.command == "setup":
        setup()
    elif args.command == "status":
        print(ssh("source /etc/profile >/dev/null 2>&1; LLstat"))
    else:
        data = {key: Path(value) for key, value in (item.split("=", 1) for item in args.data)}
        tasks = json.loads(args.tasks.read_text())
        print(json.dumps(submit(args.name, args.commit, tasks, data, args.time, args.after, args.exclude)))


if __name__ == "__main__":
    main()
