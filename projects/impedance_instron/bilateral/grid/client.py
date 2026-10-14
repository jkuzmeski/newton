# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Workstation client: push code snapshots and datasets to the grid and submit experiments.

The ``Z:`` share and the grid home are the same files, so code snapshots and
datasets are written through ``Z:`` and only small commands go over ssh.

Typical use from the repository root::

    uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client setup
    uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client submit pm_ablation --indices 0 --smoke
    uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client submit pm_ablation --time 04:00:00
    uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client status
    uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client tail <job> --task 0
    uv run --no-sync python -m projects.impedance_instron.bilateral.grid.client report <job>

Every training task evaluates its policy and writes graphs to ``<run>/report``;
``submit`` also queues a CPU job that, after the array ends, compares the runs
in ``<runs>/<experiment>/<job>/report/report.md``. ``report <job>`` rebuilds
that comparison locally; ``report <job> --evaluate`` re-evaluates an earlier
job's runs with the current code (for jobs trained before the reports existed).

``push`` (and every ``submit``) snapshots the working tree, including
uncommitted and untracked, non-ignored files, as a commit object kept under
``refs/grid/`` locally. The snapshot extracts once to an immutable
``code/<commit>`` directory on the grid, so later dev pushes never change the
code of queued or running jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from .experiments import get

HOST = os.environ.get("GRID_HOST", "txg-login.llgrid.ll.mit.edu")
USER = os.environ.get("GRID_USER", os.environ.get("USERNAME", os.environ.get("USER", ""))).upper()
REMOTE_ROOT = os.environ.get("GRID_REMOTE_ROOT", f"/home/gridsan/{USER}/newton_grid")
LOCAL_ROOT = Path(os.environ.get("GRID_LOCAL_ROOT", "Z:/newton_grid"))
DATA_SOURCE = Path(os.environ.get("GRID_DATA_SOURCE", "outputs/bilateral"))
MODULE = "conda/Python-ML-2026a-pytorch"
SNAPSHOT_PATHS = (
    "newton",
    "projects",
    "pyproject.toml",
    ":(exclude)newton/tests",
    ":(exclude)newton/examples",
    ":(exclude)projects/gait_c3d/assets",
)
JOB_SCRIPT = "projects/impedance_instron/bilateral/grid/job.sbatch"
REPORT_SCRIPT = "projects/impedance_instron/bilateral/grid/report.sbatch"
TENSORBOARD_SCRIPT = "projects/impedance_instron/bilateral/grid/tensorboard.sbatch"
TENSORBOARD_JOB = "bil-tensorboard"
MAX_GPUS_WITHOUT_CONFIRMATION = 4


def _git(*args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def ssh(command: str, *, check: bool = True) -> str:
    """Run one command on the login node and return its stdout."""
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, command],
        check=False,
        capture_output=True,
        text=True,
    )
    stderr = "\n".join(
        line for line in result.stderr.splitlines() if "post-quantum" not in line and "store now" not in line
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"ssh command failed ({result.returncode}): {command}\n{stderr}")
    if stderr.strip():
        print(stderr, file=sys.stderr)
    return result.stdout


@dataclass(frozen=True)
class Snapshot:
    commit: str
    label: str

    @property
    def short(self) -> str:
        return self.commit[:12]

    @property
    def remote_dir(self) -> str:
        return f"{REMOTE_ROOT}/code/{self.short}"


def snapshot() -> Snapshot:
    """Return a commit of the current working tree without touching the index, HEAD, or stash."""
    head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    index = Path(_git("rev-parse", "--git-path", "index"))
    with tempfile.TemporaryDirectory() as scratch:
        temporary = Path(scratch) / "index"
        # Copying the real index keeps its stat cache, so only changed files are hashed.
        shutil.copyfile(index, temporary)
        env = {**os.environ, "GIT_INDEX_FILE": str(temporary)}
        _git("add", "-A", env=env)
        tree = _git("write-tree", env=env)
    if tree == _git("rev-parse", "HEAD^{tree}"):
        return Snapshot(head, f"{branch}@{head[:12]}")
    message = f"grid snapshot of {branch}@{head[:12]} with uncommitted changes"
    commit = _git("commit-tree", tree, "-p", head, "-m", message)
    # Keep the snapshot reachable so the exact code of a grid run stays reproducible.
    _git("update-ref", f"refs/grid/{commit[:12]}", commit)
    return Snapshot(commit, f"{branch}@{head[:12]}+dev-{commit[:12]}")


def push_code() -> Snapshot:
    """Snapshot the working tree and extract it once into an immutable grid code directory."""
    snap = snapshot()
    marker = LOCAL_ROOT / "code" / snap.short / "GRID_COMMIT"
    if marker.exists():
        print(f"code {snap.short} already on the grid ({snap.label})")
        return snap
    (LOCAL_ROOT / "code").mkdir(parents=True, exist_ok=True)
    archive = LOCAL_ROOT / "code" / f"{snap.short}.tar.gz"
    # Archive with the repository's LF line endings; core.autocrlf would export CRLF on Windows.
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "archive", "--format=tar.gz", "-o", str(archive),
         snap.commit, "--", *SNAPSHOT_PATHS],
        check=True,
    )  # fmt: skip
    remote = snap.remote_dir
    ssh(
        f"set -e; d={remote}; rm -rf $d.tmp; mkdir -p $d.tmp; tar -xzf $d.tar.gz -C $d.tmp; "
        f"echo {snap.commit} > $d.tmp/GRID_COMMIT; echo '{snap.label}' > $d.tmp/GRID_LABEL; "
        f"chmod -R a-w $d.tmp; mv $d.tmp $d; rm -f $d.tar.gz"
    )
    print(f"pushed code {snap.short} ({snap.label}) -> {remote}")
    return snap


def _manifest_hash(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()


def sync_data(names: list[str]) -> None:
    """Copy prepared datasets to the grid once; a changed dataset needs a new name."""
    for name in names:
        source = DATA_SOURCE / name
        target = LOCAL_ROOT / "data" / name
        if not (source / "manifest.json").is_file():
            raise FileNotFoundError(f"no prepared dataset at {source}")
        if target.exists():
            if _manifest_hash(source) != _manifest_hash(target):
                raise RuntimeError(
                    f"{target} differs from {source}; datasets on the grid are immutable, prepare it under a new name"
                )
            continue
        partial = target.with_name(f".{name}.partial")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(source, partial)
        partial.rename(target)
        print(f"copied dataset {name} -> {target}")


def setup() -> None:
    """Create the grid directories and the venv (LLSC PyTorch module + matching warp-lang)."""
    for name in ("code", "data", "runs", "tb", "logs", "warp_cache"):
        (LOCAL_ROOT / name).mkdir(parents=True, exist_ok=True)
    warp, mujoco, mujoco_warp = version("warp-lang"), version("mujoco"), version("mujoco-warp")
    ssh(
        "source /etc/profile >/dev/null 2>&1; "
        f"module load {MODULE} >/dev/null 2>&1; "
        f"if [ ! -x {REMOTE_ROOT}/venv/bin/python ]; then python -m venv --system-site-packages {REMOTE_ROOT}/venv; fi; "
        "module load proxy-mitll >/dev/null 2>&1; unset PYTHONPATH; "
        f"{REMOTE_ROOT}/venv/bin/python -m pip install --quiet --no-cache-dir 'warp-lang=={warp}' "
        f"'mujoco=={mujoco}' 'mujoco-warp=={mujoco_warp}'; "
        f"{REMOTE_ROOT}/venv/bin/python -c 'import warp, torch, numpy, mujoco_warp; "
        'print("warp", warp.__version__, "torch", torch.__version__, "numpy", numpy.__version__)\''
    )
    print(ssh(f"{REMOTE_ROOT}/venv/bin/python -c 'import warp; print(warp.__version__)' 2>/dev/null").strip())
    print(f"grid root ready: {REMOTE_ROOT} (local {LOCAL_ROOT})")


def free_h100_gpus() -> int | None:
    """Return the free H100 GPU count reported by LLfree, if it can be parsed."""
    output = ssh("source /etc/profile >/dev/null 2>&1; LLfree", check=False)
    for line in output.splitlines():
        if "e9-h100nvl" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            try:
                return int(cells[-1])
            except ValueError:
                return None
    return None


def _parse_indices(text: str | None, count: int) -> list[int]:
    if text is None:
        return list(range(count))
    indices: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = (int(value) for value in part.split("-"))
            indices.extend(range(low, high + 1))
        else:
            indices.append(int(part))
    if any(not 0 <= index < count for index in indices):
        raise SystemExit(f"indices must be within 0-{count - 1}")
    return sorted(set(indices))


def _array_job(
    experiment: str,
    indices: list[int],
    *,
    smoke: bool,
    time_limit: str,
    throttle: int,
    snap: Snapshot,
    report_job: str | None = None,
) -> str:
    """Submit one array of training tasks, or of report-only tasks for ``report_job``; return the job id."""
    array = f"{','.join(str(index) for index in indices)}%{throttle}"
    group = experiment + ("-smoke" if smoke else "")
    name = f"bil-{group}" + ("-report" if report_job else "")
    log = f"{REMOTE_ROOT}/logs/{group}{'-report' if report_job else ''}-%A_%a.out"
    exports = [
        "ALL",
        f"GRID_ROOT={REMOTE_ROOT}",
        f"GRID_CODE_DIR={snap.remote_dir}",
        f"GRID_CODE_COMMIT={snap.commit}",
        f"GRID_CODE_LABEL={snap.label}",
        f"EXPERIMENT={experiment}",
        f"SMOKE={int(smoke)}",
    ]
    if report_job:
        exports.append(f"REPORT_JOB={int(report_job)}")
    command = (
        f"cd {REMOTE_ROOT} && sbatch --parsable --job-name={name} --array={array} --time={time_limit} "
        f"--output={log} --error={log} --export={','.join(exports)} {snap.remote_dir}/{JOB_SCRIPT}"
    )
    return _job_id(ssh(command))


def _experiment_report_job(experiment: str, runs_job: str, *, smoke: bool, snap: Snapshot, after: str) -> str:
    """Submit the CPU job that compares the runs of ``runs_job`` once every task of ``after`` has ended."""
    group = experiment + ("-smoke" if smoke else "")
    log = f"{REMOTE_ROOT}/logs/{group}-experiment-report-{runs_job}-%j.out"
    exports = (
        f"ALL,GRID_ROOT={REMOTE_ROOT},GRID_CODE_DIR={snap.remote_dir},RUNS_DIR={REMOTE_ROOT}/runs/{group}/{runs_job}"
    )
    command = (
        f"cd {REMOTE_ROOT} && sbatch --parsable --job-name=bil-{group}-compare --dependency=afterany:{int(after)} "
        f"--time=00:20:00 --output={log} --error={log} --export={exports} {snap.remote_dir}/{REPORT_SCRIPT}"
    )
    return _job_id(ssh(command))


def _job_id(output: str) -> str:
    job = output.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    return job


def _check_gpus(count: int, time_limit: str, allow_large: bool, queue_anyway: bool) -> None:
    if count > MAX_GPUS_WITHOUT_CONFIRMATION and not allow_large:
        raise SystemExit(
            f"{count} GPUs requested; confirm with the team and pass --allow-large "
            f"(more than {MAX_GPUS_WITHOUT_CONFIRMATION} GPUs)"
        )
    hours, minutes, _ = (int(part) for part in time_limit.split(":"))
    if hours * 60 + minutes > 8 * 60 and not allow_large:
        raise SystemExit("more than 8 h per task; pass --allow-large after confirming")
    free = free_h100_gpus()
    print(f"LLfree: {free if free is not None else 'unknown'} free H100 GPUs on e9-h100nvl")
    if free == 0 and not queue_anyway:
        raise SystemExit("no free H100 GPUs; wait, or pass --queue-anyway")


def _snapshot(code: str | None) -> Snapshot:
    if not code:
        return push_code()
    label = (LOCAL_ROOT / "code" / code / "GRID_LABEL").read_text(encoding="utf-8").strip()
    commit = (LOCAL_ROOT / "code" / code / "GRID_COMMIT").read_text(encoding="utf-8").strip()
    return Snapshot(commit, label)


def _record(record: dict) -> None:
    with (LOCAL_ROOT / "submissions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def submit(args: argparse.Namespace) -> None:
    runs = get(args.experiment)
    indices = _parse_indices(args.indices, len(runs))
    _check_gpus(len(indices), args.time, args.allow_large, args.queue_anyway)
    snap = _snapshot(args.code)
    sync_data(sorted({runs[index].dataset for index in indices}))
    throttle = args.throttle or len(indices)
    job = _array_job(args.experiment, indices, smoke=args.smoke, time_limit=args.time, throttle=throttle, snap=snap)
    compare = None
    if not args.no_report:
        compare = _experiment_report_job(args.experiment, job, smoke=args.smoke, snap=snap, after=job)
    _record(
        {
            "job": job,
            "experiment": args.experiment,
            "indices": indices,
            "runs": [runs[index].name for index in indices],
            "smoke": args.smoke,
            "time": args.time,
            "throttle": throttle,
            "code_commit": snap.commit,
            "code_label": snap.label,
            "report_job": compare,
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    group = args.experiment + ("-smoke" if args.smoke else "")
    print(f"submitted job {job}: {args.experiment} indices {indices} code {snap.short} ({snap.label})")
    if compare:
        print(f"experiment report job {compare} runs after {job} ends")
    print(f"logs: {LOCAL_ROOT / 'logs'} ; outputs: {LOCAL_ROOT / 'runs' / group / job}")
    print(f"graphs: <run>/report/*.png per run; {LOCAL_ROOT / 'runs' / group / job / 'report' / 'report.md'}")


def report(args: argparse.Namespace) -> None:
    """Rebuild an earlier job's experiment report locally, or with --evaluate re-evaluate its runs on the grid."""
    records = [record for record in submissions() if record["job"] == str(int(args.job)) and "report_of" not in record]
    if not records:
        raise SystemExit(f"job {args.job} is not in {LOCAL_ROOT / 'submissions.jsonl'}")
    record = records[-1]
    group = record["experiment"] + ("-smoke" if record["smoke"] else "")
    directory = LOCAL_ROOT / "runs" / group / record["job"]
    if not args.evaluate:
        from ..report import experiment_report  # noqa: PLC0415

        print(experiment_report(directory))
        return
    indices = record["indices"]
    _check_gpus(len(indices), args.time, args.allow_large, args.queue_anyway)
    snap = _snapshot(args.code)
    runs = get(record["experiment"])
    sync_data(sorted({runs[index].dataset for index in indices}))
    job = _array_job(
        record["experiment"],
        indices,
        smoke=record["smoke"],
        time_limit=args.time,
        throttle=len(indices),
        snap=snap,
        report_job=record["job"],
    )
    compare = _experiment_report_job(record["experiment"], record["job"], smoke=record["smoke"], snap=snap, after=job)
    _record(
        {
            "job": job,
            "experiment": record["experiment"],
            "indices": indices,
            "runs": record["runs"],
            "smoke": record["smoke"],
            "time": args.time,
            "report_of": record["job"],
            "report_job": compare,
            "code_commit": snap.commit,
            "code_label": snap.label,
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    print(f"submitted report job {job} for {directory} (code {snap.short}); experiment report job {compare}")


def submissions() -> list[dict]:
    path = LOCAL_ROOT / "submissions.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def status(job: str | None) -> None:
    if job is None:
        print(ssh("source /etc/profile >/dev/null 2>&1; LLstat", check=False))
        return
    print(
        ssh(
            f"sacct -j {int(job)} --format=JobID%18,JobName%24,State,Elapsed,Timelimit,MaxRSS,ReqMem,NodeList",
            check=False,
        )
    )


def tail(job: str, task: int, lines: int) -> None:
    logs = sorted((LOCAL_ROOT / "logs").glob(f"*-{int(job)}_{task}.out"))
    if not logs:
        raise SystemExit(f"no log for job {job} task {task} yet in {LOCAL_ROOT / 'logs'}")
    # Warp's per-module load lines drown the training progress.
    text = [
        line
        for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
        if "load on device" not in line
    ]
    print(f"== {logs[-1]}")
    print("\n".join(text[-lines:]))


def tensorboard(time_limit: str, code: str | None, stop: bool) -> None:
    """Serve the TensorBoard of every run on the LLSC portal, reusing a running instance."""
    running = ssh(f"squeue -h -u $USER -n {TENSORBOARD_JOB} -t PENDING,RUNNING -o %i", check=False).split()
    url = f"https://{USER.lower()}-tensorboard.fn.txg-portal.llgrid.ll.mit.edu/"
    if stop:
        for job in running:
            print(ssh(f"scancel {int(job)}", check=False) or f"cancelled TensorBoard job {job}")
        return
    if running:
        print(f"TensorBoard job {running[0]} already serves {url}")
        return
    snap = _snapshot(code)
    log = f"{REMOTE_ROOT}/logs/tensorboard-%j.out"
    command = (
        f"cd {REMOTE_ROOT} && sbatch --parsable --job-name={TENSORBOARD_JOB} --time={time_limit} "
        f"--output={log} --error={log} --export=ALL,GRID_ROOT={REMOTE_ROOT} {snap.remote_dir}/{TENSORBOARD_SCRIPT}"
    )
    job = _job_id(ssh(command))
    print(f"TensorBoard job {job} serves {REMOTE_ROOT}/tb at {url} once it starts (log in to the portal first)")


def cancel(job: str, force: bool) -> None:
    known = {record["job"] for record in submissions()}
    if str(int(job)) not in known and not force:
        raise SystemExit(f"job {job} was not submitted by this client; pass --force after confirming")
    print(ssh(f"scancel {int(job)}", check=False) or f"cancelled {job}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Create grid directories and the Python environment.")
    commands.add_parser("push", help="Snapshot the working tree to an immutable grid code directory.")
    data = commands.add_parser("sync-data", help="Copy prepared datasets to the grid.")
    data.add_argument("names", nargs="+")
    sub = commands.add_parser("submit", help="Submit an experiment as a throttled SLURM array.")
    sub.add_argument("experiment")
    sub.add_argument("--indices", help="Runs to submit, e.g. '0-3' or '0,2'; default all.")
    sub.add_argument("--time", default="04:00:00", help="Time limit per task (HH:MM:SS).")
    sub.add_argument("--throttle", type=int, help="Maximum concurrent tasks; default all.")
    sub.add_argument("--smoke", action="store_true", help="Run each index with its short smoke-test arguments.")
    sub.add_argument("--code", help="Reuse an existing grid code snapshot (12-character commit) instead of pushing.")
    sub.add_argument("--allow-large", action="store_true", help="Permit > 4 GPUs or > 8 h per task.")
    sub.add_argument("--queue-anyway", action="store_true", help="Submit even if LLfree reports no free H100.")
    sub.add_argument("--no-report", action="store_true", help="Do not queue the experiment comparison report job.")
    rp = commands.add_parser("report", help="Rebuild a job's experiment report from Z:, or --evaluate its runs again.")
    rp.add_argument("job")
    rp.add_argument("--evaluate", action="store_true", help="Re-evaluate every run on the grid (GPU) first.")
    rp.add_argument("--time", default="00:30:00", help="Time limit per report task with --evaluate.")
    rp.add_argument("--code", help="Grid code snapshot to evaluate with; default pushes the working tree.")
    rp.add_argument("--allow-large", action="store_true", help="Permit > 4 GPUs.")
    rp.add_argument("--queue-anyway", action="store_true", help="Submit even if LLfree reports no free H100.")
    st = commands.add_parser("status", help="LLstat, or sacct for one job.")
    st.add_argument("job", nargs="?")
    tl = commands.add_parser("tail", help="Show the end of a task log.")
    tl.add_argument("job")
    tl.add_argument("--task", type=int, default=0)
    tl.add_argument("--lines", type=int, default=40)
    commands.add_parser("load", help="LLload -g GPU utilization of running jobs.")
    cn = commands.add_parser("cancel", help="Cancel one job submitted by this client.")
    cn.add_argument("job")
    cn.add_argument("--force", action="store_true")
    commands.add_parser("list", help="List experiments and past submissions.")
    tb = commands.add_parser("tensorboard", help="Serve TensorBoard of all runs on the LLSC portal (CPU job).")
    tb.add_argument("--time", default="08:00:00", help="Time limit of the TensorBoard job (HH:MM:SS).")
    tb.add_argument("--code", help="Grid code snapshot holding the job script; default pushes the working tree.")
    tb.add_argument("--stop", action="store_true", help="Cancel the running TensorBoard job.")
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup()
    elif args.command == "push":
        push_code()
    elif args.command == "sync-data":
        sync_data(args.names)
    elif args.command == "submit":
        submit(args)
    elif args.command == "report":
        report(args)
    elif args.command == "status":
        status(args.job)
    elif args.command == "tail":
        tail(args.job, args.task, args.lines)
    elif args.command == "load":
        print(ssh("source /etc/profile >/dev/null 2>&1; LLload -g", check=False))
    elif args.command == "cancel":
        cancel(args.job, args.force)
    elif args.command == "tensorboard":
        tensorboard(args.time, args.code, args.stop)
    elif args.command == "list":
        from .experiments import main as list_experiments  # noqa: PLC0415

        list_experiments([])
        for record in submissions():
            print(
                f"{record['submitted']} job {record['job']} {record['experiment']} {record['indices']} "
                f"smoke={record['smoke']} code={record['code_label']}"
            )


if __name__ == "__main__":
    main()
