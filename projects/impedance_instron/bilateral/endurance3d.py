# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a frozen three-dimensional bilateral policy open-ended over a target distance.

:mod:`.evaluate3d` scores one measured segment at a time and stops at the end of
that window. This module asks how far a policy runs before it falls. Each world
starts once from a flight initial state of the dataset, exactly as a training
episode does, and then runs on the policy's mean action with no termination and
no reset: the loop calls :meth:`.env3d.BilateralEnv3D.advance` directly, so the
reference departure rules and the episode clock of
:meth:`.env3d.BilateralEnv3D.step` never interrupt a run. Only the fall rules
below retire a world, and a world that has covered the distance keeps being
integrated, so its last frames show whether it held its speed to the end.

The nominal target of 1 km is 274 s and 137k control steps at 3.65 m/s, so the
loop keeps host work small: it checks falls and distance every
:attr:`EnduranceConfig.check_every_steps` steps and writes one stacked record
every :attr:`EnduranceConfig.record_every_steps` steps. Fall times and finish
times are exact control steps; recorded positions are quantized to the recording
cadence.

A world has fallen when the environment screens it (a non-finite state, a shoe
force or compression limit, or a hip below
:attr:`.env3d.Env3DConfig.minimum_hip_height_m`), when its pelvis drops below
:attr:`EnduranceConfig.minimum_pelvis_height_m`, or when its pelvis tilts past
:attr:`EnduranceConfig.maximum_pelvis_tilt_rad`. The measured trials keep the hip
between 0.757 and 0.851 m and the pelvis tilt under 0.25 rad, so the defaults of
0.55 m and 60 deg sit far outside the measured stride and fire only on a genuine
fall. A fallen world is held in place, its impedance frozen and its coordinate
rates zeroed at each check, so its trace stays finite; a status of 1 means this
module stopped the world, and :data:`CAUSES` names the reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .env3d import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    BilateralEnv3D,
    Bounds3D,
    Env3DConfig,
    ReferenceLibrary3D,
    seed_action,
)
from .policy import load_policy

CAUSES = {
    1: "held by the endurance rules",
    4: "contact wrench not finite",
    5: "shoe force limit",
    6: "shoe compression limit",
    8: "state not finite",
    9: "hip below the environment floor",
    100: "pelvis below the height limit",
    101: "pelvis past the tilt limit",
}
"""Screen codes of :mod:`.env3d`, plus the 100 and 101 codes this module records."""

RECORDS = (
    "x_m",
    "y_m",
    "z_m",
    "yaw_rad",
    "tilt_rad",
    "forward_m_s",
    "lateral_m_s",
    "vertical_m_s",
    "pacer_x_m",
    "pacer_y_m",
    "fz_r_n",
    "fz_l_n",
    "alive",
)
"""Columns of one recorded frame; ``pacer_*_m`` is the pacer position minus the pelvis."""


@dataclass(frozen=True)
class EnduranceConfig:
    """Distance, fall rules, and recording cadence of one open-ended run."""

    distance_m: float = 1000.0
    """Forward distance along +x at which a world is retired."""
    time_limit_s: float | None = None
    """Longest run [s]; defaults to 1.5 times the nominal time at the dataset speed."""
    minimum_pelvis_height_m: float = 0.55
    """Hip height [m] below which a world has fallen; the measured stride stays above 0.757 m."""
    maximum_pelvis_tilt_rad: float = math.radians(60.0)
    """Pelvis tilt [rad] past which a world has fallen; the measured trials stay under 0.25 rad."""
    stance_force_n: float = 50.0
    """Interval-mean vertical foot force [N] above which a foot is counted in stance."""
    stance_minimum_s: float = 0.05
    """Shortest counted stance [s], which rejects force chatter."""
    check_every_steps: int = 10
    """Control steps between fall and distance checks; 10 is 0.02 s at the 2 ms control step."""
    record_every_steps: int = 2
    """Control steps between recorded frames; 2 is 250 Hz at the 2 ms control step."""


class ConstantImpedance:
    """Unfitted seed impedance logits as a stand-in actor, to exercise the harness.

    The observation is read only for its batch size, so a run still pays for
    :meth:`.env3d.BilateralEnv3D.observe` and reports a representative rate.
    """

    def __init__(self, bounds: Bounds3D | None = None, device: str = "cpu"):
        self.logits = torch.tensor(seed_action(bounds), dtype=torch.float32, device=device)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(observation.shape[0], -1)


def stored(kind: type, saved: dict) -> Env3DConfig | Bounds3D:
    """Rebuild a stored environment dataclass, ignoring keys of a newer or older environment."""
    values = {key: value for key, value in saved.items() if key in {f.name for f in fields(kind)}}
    if isinstance(values.get("reward_weights"), list):
        values["reward_weights"] = tuple(values["reward_weights"])
    return kind(**values)


def heading(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Yaw of the body +x axis and its tilt from world +z [rad] for xyzw quaternions.

    Matches :func:`.env3d._heading`, so the recorded heading is the one the actor sees.
    """
    x, y, z, w = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (x * y + z * w), 1.0 - 2.0 * (y * y + z * z))
    tilt = torch.arccos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0))
    return yaw, tilt


def _state(env: BilateralEnv3D, alive: torch.Tensor) -> torch.Tensor:
    """Frame of every world: pelvis pose, heading-frame velocity, pacer offset, and foot loads."""
    q, qd = env.joint_q, env.joint_qd
    yaw, tilt = heading(q[:, 3:7])
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    velocity = qd[:, :3]
    force = env.t.grf[:, :, 2] / env.t.substeps.clamp(min=1)[:, None]
    pacer = env.pacer() - q[:, :2]
    return torch.stack(
        (
            q[:, 0],
            q[:, 1],
            q[:, 2],
            yaw,
            tilt,
            velocity[:, 0] * cosine + velocity[:, 1] * sine,
            -velocity[:, 0] * sine + velocity[:, 1] * cosine,
            velocity[:, 2],
            pacer[:, 0],
            pacer[:, 1],
            force[:, 0],
            force[:, 1],
            alive.float(),
        ),
        dim=-1,
    )


def reset_once(env: BilateralEnv3D) -> None:
    """Start every world from a distinct flight initial state, which also places its pacer.

    :meth:`.env3d.BilateralEnv3D.reset` draws initial states with replacement, so two worlds
    of the same trial can land on the same frame of the same segment and then run identical
    simulations. Each world re-draws a bounded number of times to avoid that.
    """
    seen: set[tuple[int, int]] = set()
    for world in range(env.worlds):
        mask = torch.zeros(env.worlds, dtype=torch.bool, device=env.torch_device)
        mask[world] = True
        for _ in range(8):
            env.reset(mask)
            key = (int(env.segment[world]), int(env.index[world]))
            if key not in seen:
                break
        seen.add(key)


@torch.no_grad()
def run(
    env: BilateralEnv3D,
    actor: Callable[[torch.Tensor], torch.Tensor],
    config: EnduranceConfig | None = None,
    *,
    progress: bool = True,
) -> dict:
    """Drive every world open-ended until each has fallen or covered the distance.

    Args:
        env: Environment in any state; :func:`reset_once` starts its worlds here.
        actor: Maps a batch of observations to impedance logits, the frozen policy mean action.
        config: Distance, fall rules, and recording cadence.
        progress: Print one line every 5 s of simulated time.

    Returns:
        Recorded frames, terminal codes, per-world start states, and wall-clock timing.
    """
    cfg = config or EnduranceConfig()
    dt = env.config.control_dt_s
    device, worlds = env.torch_device, env.worlds
    reset_once(env)
    nominal = float(env.speed.mean())
    limit = cfg.time_limit_s if cfg.time_limit_s is not None else 1.5 * cfg.distance_m / max(nominal, 1e-6)
    max_steps = max(int(math.ceil(limit / dt)), 1)
    records = torch.zeros(max_steps // cfg.record_every_steps + 1, worlds, len(RECORDS), device=device)
    alive = torch.ones(worlds, dtype=torch.bool, device=device)
    causes = torch.zeros(worlds, dtype=torch.long, device=device)
    falls = torch.full((worlds,), -1, dtype=torch.long, device=device)
    finishes = torch.full((worlds,), -1, dtype=torch.long, device=device)
    start = env.joint_q[:, 0].clone()
    records[0] = _state(env, alive)
    every = max(1, int(round(5.0 / (dt * cfg.check_every_steps))))
    wall = time.perf_counter()
    steps = 0
    for k in range(1, max_steps + 1):
        env.advance(actor(env.observe()))
        steps = k
        if k % cfg.record_every_steps == 0:
            records[k // cfg.record_every_steps] = _state(env, alive)
        if k % cfg.check_every_steps:
            continue
        q = env.joint_q
        _, tilt = heading(q[:, 3:7])
        traveled = q[:, 0] - start
        low = q[:, 2] < cfg.minimum_pelvis_height_m
        fallen = (env.t.status != 0) | low | (tilt > cfg.maximum_pelvis_tilt_rad)
        fresh = fallen & alive
        reason = torch.where(env.t.status != 0, env.t.status.long(), torch.where(low, 100, 101))
        causes = torch.where(fresh & (causes == 0), reason, causes)
        falls = torch.where(fresh, torch.full_like(falls, k), falls)
        alive &= ~fallen
        reached = traveled >= cfg.distance_m
        finishes = torch.where(alive & reached & (finishes < 0), torch.full_like(finishes, k), finishes)
        # A retired world is held: a nonzero status makes the environment's kernels skip it, and
        # zeroing its rates at each check keeps its held pose from drifting down under gravity.
        env.joint_qd.mul_(alive.to(env.joint_qd.dtype)[:, None])
        env.t.status.copy_(torch.where(alive, env.t.status, env.t.status.clamp(min=1)))
        done = bool((~alive | reached).all())
        if progress and (k // cfg.check_every_steps) % every == 0:
            rate = k / (time.perf_counter() - wall)
            print(
                f"  t {k * dt:6.1f} s  x {float(traveled.max()):6.1f} m  alive {int(alive.sum()):3d}/{worlds}"
                f"  {rate:6.1f} steps/s  eta {(max_steps - k) / rate / 60:5.1f} min",
                flush=True,
            )
        if done:
            break
    elapsed = time.perf_counter() - wall
    return {
        "records": records[: steps // cfg.record_every_steps + 1].cpu().numpy(),
        "record_dt_s": cfg.record_every_steps * dt,
        "causes": causes.cpu().tolist(),
        "fall_steps": falls.cpu().tolist(),
        "finish_steps": finishes.cpu().tolist(),
        "starts": [
            {
                "world": w,
                "trial": env.library.trials[int(env.world_trials[w])],
                "segment": env.library.members[int(env.segment[w])]["id"],
                "speed_m_s": round(float(env.speed[w]), 3),
                "pacer_start_m": [round(float(v), 4) for v in env.pacer_start[w]],
            }
            for w in range(worlds)
        ],
        "timing": {
            "wall_s": elapsed,
            "simulated_s": steps * dt,
            "control_steps": steps,
            "steps_per_s": steps / elapsed,
            "realtime_factor": steps * dt / elapsed,
            "control_dt_s": dt,
        },
    }


def _stances(force: np.ndarray, config: EnduranceConfig, record_dt_s: float) -> tuple[np.ndarray, float]:
    """Vertical peaks [N] of one foot's stances and the loaded fraction of the window.

    Args:
        force: Interval-mean vertical force [N] of one foot over one world's run.
        config: Stance threshold and shortest counted stance.
        record_dt_s: Interval [s] between recorded frames.

    Returns:
        One peak per counted stance, and the fraction of frames above the threshold.
    """
    loaded = force > config.stance_force_n
    change = np.diff(loaded.astype(np.int8))
    on, off = np.flatnonzero(change == 1) + 1, np.flatnonzero(change == -1)
    if loaded.size and loaded[0]:
        on = np.concatenate(([0], on))
    if loaded.size and loaded[-1]:
        off = np.concatenate((off, [loaded.size - 1]))
    shortest = max(1, int(round(config.stance_minimum_s / record_dt_s)))
    ends = min(on.size, off.size)
    peaks = [
        float(force[first : last + 1].max())
        for first, last in zip(on[:ends], off[:ends], strict=True)
        if last - first + 1 >= shortest
    ]
    return np.array(peaks), float(loaded.mean())


def _peak_stats(peaks: np.ndarray) -> dict:
    """Median, 99th percentile, and maximum of a set of stance peaks, or ``None``."""
    if not peaks.size:
        return {"peak_n_median": None, "peak_n_p99": None, "peak_n_max": None}
    return {
        "peak_n_median": round(float(np.median(peaks)), 1),
        "peak_n_p99": round(float(np.quantile(peaks, 0.99)), 1),
        "peak_n_max": round(float(peaks.max()), 1),
    }


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles [rad] to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def summarize(traces: dict, config: EnduranceConfig) -> list[dict]:
    """Summarize each world up to its fall, its finish, or the end of the run.

    The reported distance is the pelvis position at the recorded frame of the fall
    or of the distance crossing, and the stance statistics cover that window only.

    Args:
        traces: Output of :func:`run`.
        config: The configuration the run used, for the distance and stance rules.

    Returns:
        One JSON-ready row per world, in world order.
    """
    dt = float(traces["record_dt_s"])
    control_dt = float(traces["timing"]["control_dt_s"])
    rows = []
    for world in range(len(traces["starts"])):
        record = traces["records"][:, world]
        column = dict(zip(RECORDS, range(len(RECORDS)), strict=True))
        alive = record[:, column["alive"]] > 0.5
        last = len(record) - 1
        finish = int(traces["finish_steps"][world])
        row = last if alive.all() else int(np.argmax(~alive))
        if alive.all() and finish >= 0:
            row = min(row, int(math.ceil(finish / config.record_every_steps)))
        window = record[: row + 1]
        x, y, z = (window[:, column[name]] for name in ("x_m", "y_m", "z_m"))
        yaw = _wrap(window[:, column["yaw_rad"]] - record[0, column["yaw_rad"]])
        stance = {side: _stances(window[:, column[f"fz_{side}_n"]], config, dt) for side in ("r", "l")}
        peaks = np.concatenate([stance[side][0] for side in ("r", "l")])
        time_s = row * dt
        fall = int(traces["fall_steps"][world])
        cause = int(traces["causes"][world])
        rows.append(
            {
                **traces["starts"][world],
                "fell": fall >= 0,
                "cause": CAUSES.get(cause, f"code {cause}") if cause else None,
                "cause_code": cause or None,
                "fall_time_s": None if fall < 0 else round(fall * control_dt, 3),
                "reached_distance": finish >= 0,
                "finish_time_s": None if finish < 0 else round(finish * control_dt, 3),
                "distance_x_m": round(float(x[-1] - x[0]), 2),
                "distance_path_m": round(float(np.hypot(*np.diff(np.column_stack((x, y)), axis=0).T).sum()), 2),
                "time_s": round(time_s, 3),
                "mean_forward_speed_m_s": round(float(window[:, column["forward_m_s"]].mean()), 3),
                "progress_speed_m_s": round(float((x[-1] - x[0]) / time_s), 3) if time_s else None,
                "heading_drift_rad": round(float(yaw[-1]), 3),
                "heading_deviation_rad": round(float(np.abs(yaw).max()), 3),
                "lateral_drift_m": round(float(y[-1] - y[0]), 3),
                "lateral_deviation_m": round(float(np.abs(y - y[0]).max()), 3),
                "height_min_m": round(float(z.min()), 3),
                "height_mean_m": round(float(z.mean()), 3),
                "tilt_max_rad": round(float(window[:, column["tilt_rad"]].max()), 3),
                "pacer_offset_x_m": round(float(window[-1, column["pacer_x_m"]]), 3),
                "pacer_offset_y_m": round(float(window[-1, column["pacer_y_m"]]), 3),
                "cadence_steps_per_min": round(60.0 * peaks.size / time_s, 1) if time_s else None,
                "stances": int(peaks.size),
                "grf_peak": _peak_stats(peaks),
                "feet": {
                    side: {
                        "stances": int(stance[side][0].size),
                        "loaded_fraction": round(stance[side][1], 3),
                        **_peak_stats(stance[side][0]),
                    }
                    for side in ("r", "l")
                },
            }
        )
    return rows


def aggregate(rows: list[dict], config: EnduranceConfig) -> dict:
    """Pool per-world rows into the headline numbers of one run."""

    def median(values: list[float]) -> float | None:
        return round(float(np.median(values)), 3) if values else None

    reached = [row for row in rows if row["reached_distance"]]
    fell = [row for row in rows if row["fell"]]
    running = [row for row in rows if row["time_s"] > 1.0]
    causes: dict[str, int] = {}
    for row in fell:
        causes[row["cause"]] = causes.get(row["cause"], 0) + 1
    return {
        "worlds": len(rows),
        "distance_m": config.distance_m,
        "reached_distance": len(reached),
        "fell": len(fell),
        "fall_causes": causes,
        "distance_x_m_median": median([row["distance_x_m"] for row in rows]),
        "distance_x_m_max": round(max((row["distance_x_m"] for row in rows), default=0.0), 2),
        "time_s_median": median([row["time_s"] for row in rows]),
        "survival_s_median": median([row["fall_time_s"] for row in fell]),
        "finish_time_s_median": median([row["finish_time_s"] for row in reached]),
        "progress_speed_m_s_median": median(
            [row["progress_speed_m_s"] for row in running if row["progress_speed_m_s"]]
        ),
        "cadence_steps_per_min_median": median(
            [row["cadence_steps_per_min"] for row in running if row["cadence_steps_per_min"]]
        ),
        "grf_peak_n_median": median(
            [row["grf_peak"]["peak_n_median"] for row in rows if row["grf_peak"]["peak_n_median"]]
        ),
        "grf_peak_n_max": round(
            max((row["grf_peak"]["peak_n_max"] for row in rows if row["grf_peak"]["peak_n_max"]), default=0.0), 1
        ),
        "heading_drift_rad_median": median([row["heading_drift_rad"] for row in running]),
        "lateral_deviation_m_median": median([row["lateral_deviation_m"] for row in running]),
        "height_min_m_min": round(min((row["height_min_m"] for row in rows), default=0.0), 3),
    }


def _pyplot():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def plot(traces: dict, rows: list[dict], config: EnduranceConfig, path: Path) -> Path:
    """Write the pelvis path, forward speed, and hip height of every world to one figure.

    Args:
        traces: Output of :func:`run`.
        rows: Output of :func:`summarize`, which supplies each world's fall marker.
        config: Fall thresholds, drawn as reference lines.
        path: Where to write the PNG.

    Returns:
        The written path.
    """
    plt = _pyplot()
    dt = float(traces["record_dt_s"])
    world_count = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": (1.1, 1, 1)})
    path_axis, speed_axis, height_axis = axes
    for world in range(world_count):
        record = traces["records"][:, world]
        x, y, z = (record[:, RECORDS.index(name)] for name in ("x_m", "y_m", "z_m"))
        speed = record[:, RECORDS.index("forward_m_s")]
        row = rows[world]
        color = f"C{world}"
        label = f"{row['trial']} {row['segment']}"
        path_axis.plot(x, y, color=color, lw=1.0, label=label)
        path_axis.plot(x[0], y[0], "o", color=color, ms=4)
        path_axis.plot(x[-1], y[-1], "x" if row["fell"] else "s", color=color, ms=7)
        speed_axis.plot(np.arange(len(speed)) * dt, speed, color=color, lw=1.0)
        height_axis.plot(np.arange(len(z)) * dt, z, color=color, lw=1.0, label=label if row["fell"] else None)
        if row["fell"]:
            for axis in (speed_axis, height_axis):
                axis.axvline(row["fall_time_s"] or 0.0, color=color, ls=":", lw=0.8)
    speed_axis.axhline(float(np.mean([row["speed_m_s"] for row in rows])), color="k", ls="--", lw=0.8)
    speed_axis.set_ylabel("forward speed [m/s]")
    height_axis.axhline(config.minimum_pelvis_height_m, color="k", ls="--", lw=0.8)
    height_axis.set_ylabel("hip height [m]")
    for axis in (speed_axis, height_axis):
        axis.set_xlabel("time [s]")
        axis.grid(alpha=0.3)
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.set_aspect("equal", adjustable="datalim")
    path_axis.grid(alpha=0.3)
    path_axis.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{world_count} worlds, {config.distance_m:g} m target, "
        f"{sum(row['reached_distance'] for row in rows)} reached the distance",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _timeseries(traces: dict, path: Path, every_s: float = 0.1) -> Path:
    """Write each world's decimated records as one long CSV table."""
    stride = max(1, int(round(every_s / float(traces["record_dt_s"]))))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("world", "t_s", *RECORDS))
        records = traces["records"][::stride]
        for row, frame in enumerate(records):
            for world, values in enumerate(frame):
                writer.writerow((world, round(row * stride * float(traces["record_dt_s"]), 4), *values.tolist()))
    return path


def _markdown(report: dict, rows: list[dict], files: dict[str, str]) -> str:
    """Render the short human-readable summary of one run."""
    summary, timing, inputs = report["summary"], report["timing"], report["inputs"]
    lines = [
        f"# 3D endurance run: {summary['reached_distance']}/{summary['worlds']} worlds reached "
        f"{summary['distance_m']:g} m",
        "",
        f"- policy: `{inputs['policy'] or 'unfitted seed impedance'}`",
        f"- dataset: `{inputs['dataset']}` (splits {inputs['splits']}, device {inputs['device']}, seed {inputs['seed']})",
        f"- {timing['control_steps']} control steps, {timing['simulated_s']:.1f} s simulated in "
        f"{timing['wall_s']:.1f} s wall ({timing['steps_per_s']:.1f} steps/s, "
        f"{timing['realtime_factor']:.3f}x realtime)",
        f"- fall rules: hip below {report['endurance']['minimum_pelvis_height_m']} m, tilt above "
        f"{math.degrees(report['endurance']['maximum_pelvis_tilt_rad']):g} deg, or the environment screen",
        f"- median distance along x {summary['distance_x_m_median']} m, median run {summary['time_s_median']} s"
        + (f", median survival {summary['survival_s_median']} s" if summary["fell"] else ""),
        "",
        "| world | trial | segment | distance x [m] | path [m] | time [s] | cause | speed [m/s] | cadence "
        "[steps/min] | GRF median [N] | GRF p99 [N] | heading [rad] | lateral [m] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        cause = row["cause"] or ("reached the distance" if row["reached_distance"] else "ran to the time limit")
        speed = row["progress_speed_m_s"]
        lines.append(
            f"| {row['world']} | {row['trial']} | {row['segment']} | {row['distance_x_m']:.2f} "
            f"| {row['distance_path_m']:.2f} | {row['time_s']:.2f} | {cause} "
            f"| {'-' if speed is None else f'{speed:.2f}'} | {row['cadence_steps_per_min'] or '-'} "
            f"| {row['grf_peak']['peak_n_median'] or '-'} | {row['grf_peak']['peak_n_p99'] or '-'} "
            f"| {row['heading_drift_rad']:.2f} | {row['lateral_drift_m']:.2f} |"
        )
    lines += ["", "## Files", ""]
    lines += [f"- `{name}`: {description}" for name, description in files.items()]
    lines.append("")
    return "\n".join(lines)


def write_outputs(directory: Path, report: dict, traces: dict, rows: list[dict]) -> None:
    """Write ``summary.json``, ``summary.md``, ``endurance.png``, ``timeseries.csv``, and ``traces.npz``."""
    directory.mkdir(parents=True, exist_ok=True)
    _timeseries(traces, directory / "timeseries.csv")
    np.savez_compressed(
        directory / "traces.npz",
        records=traces["records"],
        columns=np.array(RECORDS),
        record_dt_s=np.array(float(traces["record_dt_s"])),
    )
    plot(traces, rows, EnduranceConfig(**report["endurance"]), directory / "endurance.png")
    files = {
        "summary.json": "inputs, environment, fall rules, timing, per-world rows, and pooled summary",
        "summary.md": "the table above",
        "endurance.png": "pelvis path in x-y, forward speed, and hip height per world",
        "timeseries.csv": "records decimated to 0.1 s, one row per world and frame",
        "traces.npz": f"full records array of shape (frames, worlds, {len(RECORDS)})",
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "summary.md").write_text(_markdown(report, rows, files), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="3D bilateral dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the reports.")
    parser.add_argument("--policy", type=Path, help="policy.pt from train-policy --kind 3d.")
    parser.add_argument("--distance-m", type=float, default=EnduranceConfig.distance_m)
    parser.add_argument("--time-limit-s", type=float, default=None, help="Defaults to 1.5x the nominal time.")
    parser.add_argument("--worlds", type=int, default=8, help="Worlds: trial geometries and start states.")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits supplying the initial states.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the initial states.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-pelvis-height-m", type=float, default=EnduranceConfig.minimum_pelvis_height_m)
    parser.add_argument("--maximum-pelvis-tilt-deg", type=float, default=60.0)
    parser.add_argument("--stance-force-n", type=float, default=EnduranceConfig.stance_force_n)
    parser.add_argument("--check-every-steps", type=int, default=EnduranceConfig.check_every_steps)
    parser.add_argument("--record-every-steps", type=int, default=EnduranceConfig.record_every_steps)
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one open-ended distance test and write its reports."""
    args = _parser().parse_args(argv)
    config, bounds, meta = Env3DConfig(), Bounds3D(), {}
    actor = None
    if args.policy is not None:
        actor, meta = load_policy(args.policy, args.device, OBSERVATION_NAMES, ACTION_NAMES)
        environment = meta["environment"]
        config = stored(Env3DConfig, environment["config"])
        bounds = stored(Bounds3D, environment["bounds"])
    library = ReferenceLibrary3D(args.dataset, tuple(args.splits), config.control_dt_s, args.device)
    env = BilateralEnv3D(library, args.worlds, config, bounds, device=args.device, seed=args.seed)
    if actor is None:
        actor, report_actor = ConstantImpedance(bounds, env.torch_device), "unfitted seed impedance"
    else:
        report_actor = str(args.policy.resolve())
    endurance = EnduranceConfig(
        distance_m=args.distance_m,
        time_limit_s=args.time_limit_s,
        minimum_pelvis_height_m=args.minimum_pelvis_height_m,
        maximum_pelvis_tilt_rad=math.radians(args.maximum_pelvis_tilt_deg),
        stance_force_n=args.stance_force_n,
        check_every_steps=args.check_every_steps,
        record_every_steps=args.record_every_steps,
    )
    print(
        f"{args.worlds} worlds, {endurance.distance_m:g} m target on {args.dataset}: {report_actor}",
        flush=True,
    )
    traces = run(env, actor, endurance, progress=not args.quiet)
    rows = summarize(traces, endurance)
    report = {
        "inputs": {
            "policy": None if args.policy is None else str(args.policy.resolve()),
            "dataset": str(library.root),
            "splits": list(args.splits),
            "device": args.device,
            "seed": args.seed,
            "actor": "policy mean action" if meta else "unfitted seed impedance",
        },
        "environment": env.describe(),
        "endurance": asdict(endurance),
        "timing": traces["timing"],
        "summary": aggregate(rows, endurance),
        "worlds": rows,
    }
    write_outputs(args.output, report, traces, rows)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {args.output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
