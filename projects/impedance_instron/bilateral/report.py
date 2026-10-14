# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graphs and summaries for trained bilateral runs and for whole grid experiments.

``run`` evaluates one trained policy on its dataset's splits and writes, under
``<run>/report/``:

- ``training.png``: training curves;
- ``segment_<split>_<i>.png``: joint angles and per-foot GRF against the measurement;
- ``ankle_<split>.png``: ankle error split into command and following error by gait phase;
- ``touchdowns_<split>_0.png``: ankle impedance, torque, and Fz around each touchdown;
- ``summary.json``, ``metrics.json``, and ``traces_<split>.npz``.

``experiment`` collects the reports of an experiment's runs into
``<experiment job>/report/``: overlaid training curves, a metric comparison,
the ankle phase comparison, and ``report.md``. It needs no GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np

TRAINING_PANELS = (
    ("mean_step_reward", "Mean step reward"),
    ("episode_length_s", "Episode length [s]"),
    ("termination_fraction", "Termination fraction"),
    ("reward_pose", "Pose kernel"),
    ("reward_force", "Force kernel"),
    ("reward_position", "Root position kernel"),
    ("action_std", "Action std"),
    ("kl", "KL"),
    ("samples_per_s", "Samples per second"),
)
PHASES = ("stance", "early_swing", "late_swing", "pre_touchdown")
JOINTS = ("hip R", "knee R", "ankle R", "hip L", "knee L", "ankle L")


def _pyplot():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def read_progress(path: Path) -> dict[str, np.ndarray]:
    """Read a PPO ``progress.csv`` into float columns."""
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def plot_training(runs: dict[str, dict[str, np.ndarray]], path: Path, title: str) -> Path:
    """Plot training curves of one or more runs against samples."""
    plt = _pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for (key, label), ax in zip(TRAINING_PANELS, axes.ravel(), strict=True):
        for name, progress in runs.items():
            if key in progress:
                ax.plot(progress["samples"] / 1e6, progress[key], lw=1, label=name)
        ax.set_title(label)
        ax.set_xlabel("samples [M]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _ankle_phase(summary: dict, phase: str, key: str) -> float:
    values = [summary[phase][side][key] for side in ("right", "left") if side in summary.get(phase, {})]
    return float(np.mean(values)) if values else float("nan")


def plot_ankle_phases(runs: dict[str, dict], path: Path, title: str) -> Path:
    """Grouped bars of ankle RMSE, q_eq bias, and following bias per gait phase, averaged over feet."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    panels = (
        ("ankle_rmse_deg", "Ankle RMSE [deg]"),
        ("command_bias_deg", "q_eq - q_ref bias [deg]"),
        ("following_bias_deg", "q - q_eq bias [deg]"),
        ("stiffness_median_nm_rad", "Median ankle K [N m/rad]"),
    )
    width = 0.8 / max(len(runs), 1)
    x = np.arange(len(PHASES))
    for (key, label), ax in zip(panels, axes, strict=True):
        for i, (name, summary) in enumerate(runs.items()):
            ax.bar(x + i * width, [_ankle_phase(summary, phase, key) for phase in PHASES], width, label=name)
        ax.set_xticks(x + 0.4 - width / 2, [phase.replace("_", "\n") for phase in PHASES])
        ax.set_title(label)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_touchdowns(traces: dict, dataset: Path, split: str, segment: int, path: Path) -> Path | None:
    """Plot ankle angle, impedance, torque, and Fz from 200 ms before to 100 ms after each simulated touchdown."""
    from .env import ReferenceLibrary  # noqa: PLC0415
    from .rollout import initial_state  # noqa: PLC0415

    if "impedance" not in traces or "torque_nm" not in traces:
        return None
    dt = float(traces["dt_s"])
    library = ReferenceLibrary(dataset, (split,), dt, "cpu")
    reference, start = library.references[segment], initial_state(library, segment)
    failed = int(traces["failed_step"][segment])
    n = int(traces["steps"][segment]) if failed < 0 else failed
    clock = np.arange(n + 1) * dt
    q_ref = np.column_stack(
        [np.interp(clock, reference["time_s"] - start.origin_s, reference["q"][:, c]) for c in range(9)]
    )
    force_time = reference["grf_time_s"] - start.origin_s
    impedance, torque = traces["impedance"][:n, segment], traces["torque_nm"][:n, segment]
    q, grf = traces["state"][: n + 1, segment], traces["grf_n"][:n, segment].reshape(n, 2, 2)
    touchdowns = sorted(
        (clock[k], s)
        for s in range(2)
        for k in np.flatnonzero((grf[1:, s, 1] > 50) & (grf[:-1, s, 1] <= 50)) + 1
        if grf[k : k + 30, s, 1].max() > 500
    )
    if not touchdowns:
        return None
    plt = _pyplot()
    fig, axes = plt.subplots(5, len(touchdowns), figsize=(3.2 * len(touchdowns), 12), sharey="row", squeeze=False)
    for col, (td, s) in enumerate(touchdowns):
        window = (clock[:-1] > td - 0.2) & (clock[:-1] < td + 0.1)
        x = (clock[:-1][window] - td) * 1e3
        a = 5 + 3 * s
        axes[0, col].plot(x, np.degrees(q_ref[:-1][window, a]), "k", label="measured")
        axes[0, col].plot(x, np.degrees(q[:-1][window, a]), "C3", label="q")
        axes[0, col].plot(x, np.degrees(impedance[window, 2 + 3 * s]), "C2--", label="q_eq")
        axes[0, col].set_title(f"{'R' if s == 0 else 'L'} touchdown t={td:.2f} s")
        axes[1, col].plot(x, impedance[window, 8 + 3 * s], "C0")
        axes[2, col].plot(x, impedance[window, 14 + 3 * s], "C1")
        spring = impedance[window, 8 + 3 * s] * (impedance[window, 2 + 3 * s] - q[:-1][window, a])
        axes[3, col].plot(x, torque[window, 2 + 3 * s], "C4", label="delivered torque")
        axes[3, col].plot(x, spring, "C4:", label="K(q_eq - q)")
        axes[4, col].plot(x, grf[window, s, 1], "C3", label="model")
        axes[4, col].plot(x, np.interp(clock[:-1][window], force_time, reference["grf_side_n"][s, :, 1]), "k")
        axes[4, col].set_xlabel("ms from touchdown")
        for row in range(5):
            axes[row, col].axvline(0, color="0.6", lw=0.5)
    for row, label in enumerate(
        ("ankle [deg]", "ankle K [N m/rad]", "ankle D [N m s/rad]", "ankle torque [N m]", "Fz [N]")
    ):
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(fontsize=7)
    axes[3, 0].legend(fontsize=7)
    fig.suptitle(f"{library.members[segment]['id']} ({split}): ankle impedance around each touchdown")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def _compact(scores: dict, ankle: dict | None) -> dict:
    metrics = dict(scores["summary"])
    metrics["grf_vibration_rmse_n"] = np.mean([row["grf_vibration_rmse_n"] for row in scores["segments"]], 0).tolist()
    if ankle is not None:
        metrics["ankle_phases"] = {
            phase: {
                key: _ankle_phase(ankle, phase, key)
                for key in ("ankle_rmse_deg", "command_bias_deg", "following_bias_deg", "stiffness_median_nm_rad")
            }
            for phase in PHASES
            if phase in ankle
        }
        metrics["touchdown_peak_excess_n"] = ankle.get("touchdown", {}).get("peak_excess_n_mean")
    return metrics


def run_report(
    run_dir: Path, dataset: Path, *, device: str = "cuda:0", splits=("train", "eval"), segments: int = 3
) -> dict:
    """Evaluate a trained run and write its graphs and metrics under ``<run>/report``."""
    run_dir, dataset = Path(run_dir), Path(dataset)
    report = run_dir / "report"
    report.mkdir(exist_ok=True)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    plot_training({run_dir.name: read_progress(run_dir / "progress.csv")}, report / "training.png", run_dir.name)
    metrics: dict = {"run": run_dir.name, "kind": config.get("kind", "planar"), "splits": {}}
    if metrics["kind"] == "3d":
        from .evaluate3d import evaluate  # noqa: PLC0415

        for split in splits:
            metrics["splits"][split] = evaluate(run_dir / "policy.pt", dataset, split, device)["summary"]
    else:
        from .ankle_diagnosis import analyze  # noqa: PLC0415
        from .evaluate import evaluate_split, figure  # noqa: PLC0415

        scores_by_split = {}
        for split in splits:
            scores, traces = evaluate_split(run_dir / "policy.pt", dataset, split, device=device)
            scores_by_split[split] = scores
            traces_path = report / f"traces_{split}.npz"
            np.savez_compressed(traces_path, **traces)
            for segment in range(min(segments, len(scores["segments"]))):
                figure(traces, scores, dataset, split, report / f"segment_{split}_{segment}.png", segment)
            ankle = analyze(traces_path, dataset, split)
            (report / f"ankle_{split}.json").write_text(json.dumps(ankle, indent=1) + "\n", encoding="utf-8")
            plot_ankle_phases(
                {run_dir.name: ankle["summary"]}, report / f"ankle_{split}.png", f"{run_dir.name} ({split})"
            )
            plot_touchdowns(traces, dataset, split, 0, report / f"touchdowns_{split}_0.png")
            metrics["splits"][split] = _compact(scores, ankle["summary"])
        (report / "summary.json").write_text(json.dumps(scores_by_split, indent=1) + "\n", encoding="utf-8")
    (report / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def plot_comparison(metrics: dict[str, dict], split: str, path: Path, title: str) -> Path:
    """Bar charts comparing runs on one split."""
    plt = _pyplot()
    names = list(metrics)
    rows = [metrics[name]["splits"].get(split, {}) for name in names]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    x = np.arange(len(names))

    def bars(ax, values, label, log=False):
        ax.bar(x, values, color=[f"C{i}" for i in range(len(names))])
        ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
        ax.set_title(label)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    bars(axes[0, 0], [row.get("mean_loss", np.nan) for row in rows], "Mean loss (log)", log=True)
    segments = [row.get("segments", 0) for row in rows]
    bars(axes[0, 1], [row.get("failed", np.nan) for row in rows], f"Failed free runs (of {max(segments, default=0)})")
    bars(axes[0, 2], [row.get("mean_abs_peak_fz_error_n") or np.nan for row in rows], "Mean |peak Fz error| [N]")
    width = 0.8 / max(len(names), 1)
    for i, row in enumerate(rows):
        joints = row.get("joint_rmse_deg", [])
        if len(joints) == len(JOINTS):
            axes[1, 0].bar(np.arange(6) + i * width, joints, width, label=names[i])
        grf = row.get("grf_rmse_n", [np.nan] * 4)
        axes[1, 1].bar(np.arange(4) + i * width, grf, width, label=names[i])
        vibration = row.get("grf_vibration_rmse_n", [np.nan] * 4)
        axes[1, 2].bar(np.arange(4) + i * width, vibration, width, label=names[i])
    axes[1, 0].set_xticks(np.arange(6) + 0.4 - width / 2, JOINTS)
    axes[1, 0].set_title("Joint RMSE [deg]")
    for ax, label in ((axes[1, 1], "GRF RMSE, 20 Hz observed [N]"), (axes[1, 2], "GRF above 20 Hz, RMS [N]")):
        ax.set_xticks(np.arange(4) + 0.4 - width / 2, ("Fx R", "Fz R", "Fx L", "Fz L"))
        ax.set_title(label)
    for ax in axes[1]:
        ax.grid(axis="y", alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def experiment_report(directory: Path) -> Path:
    """Compare every run of one experiment job; runs without a report are listed as missing."""
    directory = Path(directory)
    out = directory / "report"
    out.mkdir(exist_ok=True)
    runs = sorted(path for path in directory.iterdir() if path.is_dir() and path.name != "report")
    progress, metrics, ankles, missing = {}, {}, {}, []
    for run in runs:
        if (run / "progress.csv").is_file():
            progress[run.name] = read_progress(run / "progress.csv")
        if (run / "report" / "metrics.json").is_file():
            metrics[run.name] = json.loads((run / "report" / "metrics.json").read_text(encoding="utf-8"))
            ankle = run / "report" / "ankle_eval.json"
            if ankle.is_file():
                ankles[run.name] = json.loads(ankle.read_text(encoding="utf-8"))["summary"]
        else:
            missing.append(run.name)
    title = f"{directory.parent.name} job {directory.name}"
    images = []
    if progress:
        images.append(plot_training(progress, out / "training.png", f"{title}: training"))
    splits = sorted({split for m in metrics.values() for split in m["splits"]}, key=lambda s: s != "eval")
    for split in splits:
        images.append(plot_comparison(metrics, split, out / f"comparison_{split}.png", f"{title}: {split}"))
    if ankles:
        images.append(plot_ankle_phases(ankles, out / "ankle_eval.png", f"{title}: ankle by gait phase (eval)"))
    lines = [f"# {title}", ""]
    commit = None
    for run in runs:
        record = directory / f"{run.name}.run.json"
        if record.is_file():
            commit = json.loads(record.read_text(encoding="utf-8")).get("code_label")
            break
    if commit:
        lines += [f"Code: `{commit}`", ""]
    for split in splits:
        lines += [
            f"## {split}",
            "",
            "| run | mean loss | failed | peak Fz error [N] | contact duration error [ms] | late-swing ankle RMSE [deg] | touchdown peak excess [N] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in metrics.items():
            row = m["splits"].get(split, {})
            late = row.get("ankle_phases", {}).get("late_swing", {}).get("ankle_rmse_deg")
            fmt = lambda value, spec: "-" if value is None else format(value, spec)  # noqa: E731
            lines.append(
                f"| {name} | {fmt(row.get('mean_loss'), '.1f')} | {row.get('failed', '-')}/{row.get('segments', '-')} "
                f"| {fmt(row.get('mean_abs_peak_fz_error_n'), '.0f')} | {fmt(row.get('mean_contact_duration_error_ms'), '.1f')} "
                f"| {fmt(late, '.1f')} | {fmt(row.get('touchdown_peak_excess_n'), '.0f')} |"
            )
        lines.append("")
    if missing:
        lines += [f"Runs without a report (failed or still running): {', '.join(missing)}", ""]
    lines += [f"![{image.stem}]({image.name})" for image in images]
    lines += ["", "Per-run graphs are in each run's `report/` directory."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "report.md"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Evaluate one trained run and write its graphs.")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--splits", nargs="+", default=["train", "eval"])
    run.add_argument("--segments", type=int, default=3, help="Segment figures per split.")
    run.add_argument("--device", default="cuda:0")
    experiment = commands.add_parser("experiment", help="Compare the runs of one experiment job.")
    experiment.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            print(json.dumps(run_report(args.run_dir, args.dataset, device=args.device, splits=tuple(args.splits),
                                        segments=args.segments)))  # fmt: skip
        except Exception:
            (Path(args.run_dir) / "report").mkdir(exist_ok=True)
            (Path(args.run_dir) / "report" / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
    else:
        print(experiment_report(args.dir))


if __name__ == "__main__":
    main()
