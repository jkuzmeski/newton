# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton verification of SuperMimic F01 trackers (M4 exit gate).

A policy runs in :class:`~projects.supermimic.sim.RunnerSim` worlds that all
wear the subject's fitted shoe (:func:`~projects.supermimic.tracker.fixed_shoe_sim`). Each world starts at one
reference flight frame of the first flight phase of a reference window and
runs to the end of the window, a fall (:func:`~projects.supermimic.tracking.termination`)
or a sim failure (non-finite outputs or a joint speed above
:attr:`~projects.supermimic.collect.Stage0Config.failure_joint_speed_rad_s`).

The report compares the rollouts with F01:

- survival: completed, fallen and failed episodes and complete stances per episode;
- joint-angle RMSE per joint and pelvis height and tilt against the reference
  frames at the same time;
- per stance, from the 1 kHz vertical and fore-aft ground reaction of each foot:
  contact time, impact peak, active peak and loading rate
  (:func:`~projects.supermimic.analysis.impact_metrics`) and the braking and
  propulsive impulse; flight time and cadence over both feet. F01 values come
  from the unfiltered 1 kHz plate signals of every trial;
- the commanded stiffness and damping against the human bounds of :mod:`.policy`;
- the work of the ground on the body per stance, the change of mechanical energy
  minus the joint work, which a passive bed keeps negative; its error is the
  same balance over flight steps, where it should vanish;
- positive and negative joint work as mean power, and stance peak moments and
  quasi-stiffness (:func:`~projects.supermimic.analysis.joint_summary`), against
  F01 inverse dynamics.

The runner faces +x of the treadmill frame, so a braking force is negative x.
Policies are a trained tracker checkpoint (:func:`actor_policy`) or, as a
baseline, the noise-free stage-0 law of :mod:`.collect` (:func:`stage0_policy`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .analysis import GRAVITY, impact_metrics, inverse_dynamics, joint_summary, stance_intervals
from .body import JOINT_COUNT, JOINT_NAMES, STATE_DIM
from .collect import (
    Stage0Config,
    episode_starts,
    loaded_moment,
    reference_action,
    reference_moments,
    sim_aux,
    sim_state,
)
from .context import XI_DIM
from .policy import DAMPING_LIMITS, STIFFNESS_LIMITS
from .reference import Subject, load_body_params, load_shoe, load_subject, trial_reference
from .sim import RunnerSim
from .tracker import TrackerConfig, TrackerTask, fixed_shoe_sim, reference_states
from .tracking import termination
from .world_model import WorldModel

__all__ = [
    "Policy",
    "actor_policy",
    "f01_summary",
    "first_flight_starts",
    "main",
    "markdown",
    "mechanical_energy",
    "rollout",
    "rollout_summary",
    "stage0_policy",
    "stance_metrics",
]

Policy = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""Map states [W, STATE_DIM], previous actions [W, 10, 3], flat reference frames ``k`` [W] and
loaded feet [W, 2] (right, left) to actions [W, 10, 3]."""

OUTCOMES = ("completed", "fall", "failure")
STANCE_KEYS = (
    "contact_time_s",
    "impact_bw",
    "peak_bw",
    "loading_rate_bw_s",
    "braking_impulse_bw_s",
    "propulsive_impulse_bw_s",
)
BED_WORK_TOLERANCE_J = 1.0
"""Ground work over a stance above which the bed counts as adding energy [J]."""
BOUND_FRACTION = 0.02
"""Commands within this log-fraction of a stiffness or damping bound count as at the bound."""


SPLITS = {"eval": ("eval",), "train": ("train",), "all": ("eval", "train")}


def first_flight_starts(starts: np.ndarray) -> np.ndarray:
    """Return the (window, frame) starts in the first flight phase of each window, the longest episodes.

    Args:
        starts: Reference flight frames as (window, frame) rows [n, 2], from
            :func:`~projects.supermimic.collect.episode_starts`.
    """
    keep = []
    for window in np.unique(starts[:, 0]):
        frames = np.sort(starts[starts[:, 0] == window, 1])
        end = np.flatnonzero(np.diff(frames) > 1)
        frames = frames[: end[0] + 1] if len(end) else frames
        keep.append(np.stack((np.full_like(frames, window), frames), 1))
    return np.concatenate(keep)


def mechanical_energy(sim: RunnerSim) -> torch.Tensor:
    """Return the kinetic plus gravitational energy [J] of each world's bodies, shape [world_count].

    It is computed in the belt frame, the inertial frame the sim integrates in.
    """
    model, state = sim.model, sim.state_0
    pose, twist = wp.to_torch(state.body_q), wp.to_torch(state.body_qd)
    mass, inertia = wp.to_torch(model.body_mass), wp.to_torch(model.body_inertia)
    x, y, z, w = pose[:, 3:7].unbind(-1)
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )
    com_height = pose[:, 2] + (rotation[:, 2] * wp.to_torch(model.body_com)).sum(-1)
    omega = (rotation.transpose(1, 2) @ twist[:, 3:, None]).squeeze(-1)
    energy = (
        0.5 * mass * twist[:, :3].square().sum(-1)
        + 0.5 * (omega * (inertia @ omega[:, :, None]).squeeze(-1)).sum(-1)
        + mass * GRAVITY * com_height
    )
    return energy.reshape(sim.world_count, -1).sum(1)


def actor_policy(task: TrackerTask, weights: dict[str, torch.Tensor]) -> Policy:
    """Return the deterministic actor of a :mod:`.tracker` checkpoint's ``actor`` state dict.

    The hidden widths are read from the weights.
    """
    linear = sorted(int(k.split(".")[1]) for k in weights if k.startswith("net.") and k.endswith(".weight"))
    actor = task.actor(hidden=tuple(weights[f"net.{i}.weight"].shape[0] for i in linear[:-1]))
    actor.load_state_dict(weights)
    actor.eval()

    def policy(state, previous, k, loaded):
        return actor(torch.cat((task.observation(state, previous), task.goal(k)), -1))

    return policy


def stage0_policy(task: TrackerTask, contact: torch.Tensor, stiffness_quantile: float = 0.5) -> Policy:
    """Return the stage-0 law of :mod:`.collect` without noise, the open-loop baseline.

    The stiffness sits at ``stiffness_quantile`` between its moment floor and
    upper bound and the damping at the geometric middle of its bounds.

    Args:
        task: Tracker task, for the reference, its moments and the bounds.
        contact: Reference foot contact, shape [windows * frames, 2].
        stiffness_quantile: Log-scale quantile of the stiffness between its floor and upper bound.
    """
    low, high = task.action_low, task.action_high
    q_limits, k_limits = torch.stack((low[:, 0], high[:, 0]), -1), torch.stack((low[:, 1], high[:, 1]), -1)
    damping = (low[:, 2] * high[:, 2]).sqrt()

    def policy(state, previous, k, loaded):
        q_ref = task.reference[k, 13 : 13 + JOINT_COUNT]
        action, _ = reference_action(
            q_ref,
            task.reference[k, 13 + JOINT_COUNT : 13 + 2 * JOINT_COUNT],
            loaded_moment(task.moment[k], contact[k], loaded),
            damping.expand_as(q_ref),
            torch.full_like(q_ref, stiffness_quantile),
            torch.zeros_like(q_ref),
            q_limits,
            k_limits,
            Stage0Config.max_posture_lead_rad,
        )
        return action

    return policy


@torch.no_grad()
def rollout(
    sim: RunnerSim,
    task: TrackerTask,
    joint_q: torch.Tensor,
    joint_qd: torch.Tensor,
    starts: torch.Tensor,
    policy: Policy,
) -> dict[str, np.ndarray]:
    """Run ``policy`` once from every start, one world per start, and return the steps.

    Args:
        sim: Sim worlds, one per start.
        task: Tracker task of the reference.
        joint_q: Reference coordinates, shape [windows, frames, 17].
        joint_qd: Reference velocities, shape [windows, frames, 16].
        starts: Episode starts as (window, frame) rows, shape [world_count, 2].
        policy: Controller.

    Returns:
        Arrays over [world, step]: ``state`` after the step, ``k`` the flat reference
        frame of that state, ``action``, ``grf`` (step mean) and ``grf_fine``
        [W, T, fine, 2, 3], ``torque``, ``work_pos`` and ``work_neg`` [W, T, 10],
        ``energy`` (:func:`mechanical_energy` after the step), and per world
        ``start_energy``, ``steps`` (recorded steps) and ``outcome`` (index into :data:`OUTCOMES`).
    """
    w, device = sim.world_count, joint_q.device
    window, frame = starts[:, 0].clone(), starts[:, 1].clone()
    steps = int(((task.frames - 1 - frame) // task.frame_stride).max())
    all_worlds = torch.ones(w, dtype=torch.bool, device=device)
    record = {
        "state": torch.zeros((steps, w, STATE_DIM), device=device),
        "k": torch.zeros((steps, w), dtype=torch.int64, device=device),
        "action": torch.zeros((steps, w, JOINT_COUNT, 3), device=device),
        "grf": torch.zeros((steps, w, 2, 3), device=device),
        "grf_fine": torch.zeros((steps, *sim.grf_fine.shape), device=device),
        "torque": torch.zeros((steps, w, JOINT_COUNT), device=device),
        "work_pos": torch.zeros((steps, w, JOINT_COUNT), device=device),
        "work_neg": torch.zeros((steps, w, JOINT_COUNT), device=device),
        "energy": torch.zeros((steps, w), device=device),
    }
    count = torch.zeros(w, dtype=torch.int64, device=device)
    outcome = torch.full((w,), -1, dtype=torch.int64, device=device)
    start_q, start_qd = joint_q[window, frame], joint_qd[window, frame]
    with torch.cuda.stream(wp.stream_to_torch(sim.device)):
        sim.reset(wp.from_torch(all_worlds), wp.from_torch(start_q), wp.from_torch(start_qd))
        start_energy = mechanical_energy(sim)
        previous = task.start_action(window * task.frames + frame)
        loaded = torch.zeros((w, 2), dtype=torch.bool, device=device)
        for t in range(steps):
            alive = outcome < 0
            state = sim_state(sim)
            k = window * task.frames + frame
            action = policy(state, previous, k, loaded)
            sim.step(wp.from_torch(action.contiguous()))
            aux, after = sim_aux(sim, state), sim_state(sim)
            frame = torch.where(alive, frame + task.frame_stride, frame)
            k_next = window * task.frames + frame
            for name, value in (
                ("state", after),
                ("k", k_next),
                ("action", action),
                ("grf", wp.to_torch(sim.grf)),
                ("grf_fine", wp.to_torch(sim.grf_fine)),
                ("torque", wp.to_torch(sim.joint_torque)),
                ("work_pos", wp.to_torch(sim.joint_work_pos)),
                ("work_neg", wp.to_torch(sim.joint_work_neg)),
                ("energy", mechanical_energy(sim)),
            ):
                record[name][t] = value
            failed = ~(
                torch.isfinite(aux).all(1)
                & (after[:, -JOINT_COUNT:].abs().amax(1) < Stage0Config.failure_joint_speed_rad_s)
            )
            fell = termination(after, task.reference[k_next])
            completed = frame + task.frame_stride >= task.frames
            count += alive & ~failed
            outcome[alive & completed] = 0
            outcome[alive & fell] = 1
            outcome[alive & failed] = 2
            # Ended worlds go back to their start so that they keep stepping finite states.
            ended = alive & (outcome >= 0)
            if ended.any():
                sim.reset(wp.from_torch(ended), wp.from_torch(start_q), wp.from_torch(start_qd))
            previous = action
            loaded = aux[:, [2, 5]] > Stage0Config.touchdown_force_n
    result = {name: value.transpose(0, 1).cpu().numpy() for name, value in record.items()}
    result["start_energy"] = start_energy.cpu().numpy()
    result["steps"] = count.cpu().numpy()
    result["outcome"] = outcome.cpu().numpy()
    return result


def stance_metrics(grf: np.ndarray, dt: float, body_weight: float) -> dict:
    """Return per-stance ground reaction metrics of one continuous record of both feet.

    Args:
        grf: Ground reaction per foot (right, left) [samples, 2, 3] [N], x forward.
        dt: Sample interval [s].
        body_weight: Body weight [N].

    Returns:
        Per foot (``right``, ``left``) a list per complete stance of
        :data:`STANCE_KEYS`, the stance touchdowns ``touchdown_s`` of both feet
        in time order, and ``rejected``, the stances whose impact
        :func:`~projects.supermimic.analysis.impact_metrics` cannot measure.
    """
    result = {"right": [], "left": [], "touchdown_s": [], "rejected": 0}
    for side, name in enumerate(("right", "left")):
        z, x = grf[:, side, 2], grf[:, side, 0]
        for a, b in stance_intervals(z):
            result["touchdown_s"].append(a * dt)
            try:
                impact = impact_metrics(z[a - 1 : b + 1], dt, body_weight)
            except ValueError:
                result["rejected"] += 1
                continue
            result[name].append(
                {
                    "contact_time_s": (b - a) * dt,
                    "impact_bw": impact["impact_bw"],
                    "peak_bw": impact["peak_bw"],
                    "loading_rate_bw_s": impact["loading_rate_bw_s"],
                    "braking_impulse_bw_s": float(-np.minimum(x[a:b], 0.0).sum() * dt / body_weight),
                    "propulsive_impulse_bw_s": float(np.maximum(x[a:b], 0.0).sum() * dt / body_weight),
                }
            )
    result["touchdown_s"].sort()
    return result


def _stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(values.mean()), "sd": float(values.std()), "n": len(values)}


def _stance_summary(records: list[dict]) -> dict:
    """Pool :func:`stance_metrics` of several records into per-foot statistics and timing."""
    summary = {}
    for side in ("right", "left"):
        stances = [s for r in records for s in r[side]]
        summary[side] = {key: _stats([s[key] for s in stances]) for key in STANCE_KEYS}
    steps = np.concatenate([np.diff(r["touchdown_s"]) for r in records if len(r["touchdown_s"]) > 1] or [[]])
    summary["step_time_s"] = _stats(steps)
    summary["cadence_steps_per_min"] = _stats(60.0 / steps) if len(steps) else _stats([])
    summary["rejected_stances"] = sum(r["rejected"] for r in records)
    return summary


def _tilt_deg(quat: np.ndarray) -> np.ndarray:
    """Angle of the pelvis vertical axis from the world vertical [deg] of quaternions (xyzw)."""
    x, y = quat[..., 0], quat[..., 1]
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def _joint_power(moments: np.ndarray, rates: np.ndarray, mass: float) -> dict:
    power = moments * rates / mass
    return {
        "positive_w_kg": np.maximum(power, 0.0).mean(0).tolist(),
        "negative_w_kg": np.minimum(power, 0.0).mean(0).tolist(),
    }


def f01_summary(reference: Path, subject: Subject) -> dict:
    """Return the F01 values of the report from every trial of ``subject``, the reference's runner.

    Stance metrics use the unfiltered 1 kHz plate force; joint work, peak
    moments and quasi-stiffness use inverse dynamics on the marker-rate
    reference, over frames with a valid marker fit, as
    :func:`~projects.supermimic.analysis.subject_summary`.
    """
    body = load_body_params(reference / "body_params.json")
    shoe = load_shoe(subject.shoe)
    mass = body.mass + shoe.pair_mass_kg
    records, power, joints = [], [], {}
    for trial in subject.trials:
        data = trial_reference(subject, trial, body)
        analog_dt = float(np.mean(np.diff(data["analog_time_s"])))
        records.append(stance_metrics(data["analog_grf"], analog_dt, mass * GRAVITY))
        dt = float(np.mean(np.diff(data["time_s"])))
        grf = data["grf"]
        moments = inverse_dynamics(data["joint_q"], dt, body, grf, np.nan_to_num(data["cop"]), np.zeros_like(grf), shoe)
        valid = data["valid"]
        power.append((moments[valid], data["joint_qd"][valid, 6:]))
        stances = tuple(
            np.array([(a, b) for a, b in stance_intervals(grf[:, side, 2]) if valid[a:b].all()]) for side in range(2)
        )
        joints[trial] = joint_summary(data["joint_q"][:, 7:], moments, stances, mass)
    return {
        "stance": _stance_summary(records),
        "joint_power": _joint_power(np.concatenate([m for m, _ in power]), np.concatenate([r for _, r in power]), mass),
        "joints": joints,
    }


def rollout_summary(result: dict, task: TrackerTask, dt: float, mass: float) -> dict:
    """Return the report values of a :func:`rollout`.

    Args:
        result: Output of :func:`rollout`.
        task: Tracker task of the reference.
        dt: Control period [s].
        mass: Body mass with shoes [kg].
    """
    w = len(result["steps"])
    fine = result["grf_fine"].shape[2]
    reference = task.reference.cpu().numpy()
    alive = np.arange(result["k"].shape[1])[None] < result["steps"][:, None]
    state, ref = result["state"][alive], reference[result["k"][alive]]
    q_error = np.degrees(state[:, 13 : 13 + JOINT_COUNT] - ref[:, 13 : 13 + JOINT_COUNT])

    records, stance_counts, joint_rows, joint_stances = [], [], [], ([], [])
    bed_work, flight_work, step_max = [], [], []
    offset = 0
    for i in range(w):
        n = result["steps"][i]
        records.append(stance_metrics(result["grf_fine"][i, :n].reshape(-1, 2, 3), dt / fine, mass * GRAVITY))
        stance_counts.append(sum(len(records[-1][s]) for s in ("right", "left")) + records[-1]["rejected"])
        for side in range(2):
            joint_stances[side].extend(
                (a + offset, b + offset) for a, b in stance_intervals(result["grf"][i, :n, side, 2])
            )
        joint_rows.append((result["state"][i, :n, 13 : 13 + JOINT_COUNT], result["torque"][i, :n]))
        offset += n
        # Work of the ground on the body per step: the energy change minus the joint work.
        energy = np.concatenate(([result["start_energy"][i]], result["energy"][i, :n]))
        ground = np.diff(energy) - (result["work_pos"][i, :n] + result["work_neg"][i, :n]).sum(-1)
        flight_work.extend(ground[(result["grf"][i, :n, :, 2] == 0.0).all(-1)])
        step_max.append(ground.max(initial=0.0))
        for side in range(2):
            bed_work.extend(ground[a:b].sum() for a, b in stance_intervals(result["grf"][i, :n, side, 2]))
    angles = np.concatenate([a for a, _ in joint_rows])
    torques = np.concatenate([m for _, m in joint_rows])
    stances = tuple(np.array(s, dtype=np.int64).reshape(-1, 2) for s in joint_stances)

    action = result["action"][alive]
    bounds = {"stiffness": STIFFNESS_LIMITS * 2, "damping": DAMPING_LIMITS * 2}
    impedance = {}
    for c, (name, table) in enumerate(bounds.items(), start=1):
        limits = np.asarray(table)
        log_span = np.log(limits[:, 1] / limits[:, 0])
        position = np.log(action[:, :, c] / limits[:, 0]) / log_span
        impedance[name] = {
            "bounds": limits.tolist(),
            "percentiles_0_5_50_95_100": np.percentile(action[:, :, c], (0, 5, 50, 95, 100), axis=0).T.tolist(),
            "at_lower": (position < BOUND_FRACTION).mean(0).tolist(),
            "at_upper": (position > 1.0 - BOUND_FRACTION).mean(0).tolist(),
        }

    seconds = result["steps"].sum() * dt
    work_pos = result["work_pos"][alive].sum(0) / seconds / mass
    work_neg = result["work_neg"][alive].sum(0) / seconds / mass
    outcome = result["outcome"]
    return {
        "worlds": w,
        "survival": {
            **{f"{name}_fraction": float((outcome == i).mean()) for i, name in enumerate(OUTCOMES)},
            "falls_per_s": float((outcome == 1).sum() / seconds),
            "failures_per_s": float((outcome == 2).sum() / seconds),
            "episode_s": _stats(result["steps"] * dt),
            "stances_per_episode": _stats(stance_counts),
        },
        "joint_rmse_deg": dict(zip(JOINT_NAMES, np.sqrt(np.square(q_error).mean(0)).tolist(), strict=True)),
        "pelvis": {
            "height_m": _stats(state[:, 2]),
            "height_reference_m": _stats(ref[:, 2]),
            "height_rmse_m": float(np.sqrt(np.square(state[:, 2] - ref[:, 2]).mean())),
            "tilt_deg": _stats(_tilt_deg(state[:, 3:7])),
            "tilt_max_deg": float(_tilt_deg(state[:, 3:7]).max()),
            "tilt_reference_deg": _stats(_tilt_deg(ref[:, 3:7])),
            "tilt_rmse_deg": float(np.sqrt(np.square(_tilt_deg(state[:, 3:7]) - _tilt_deg(ref[:, 3:7])).mean())),
        },
        "stance": _stance_summary(records),
        "impedance": impedance,
        "joint_power": {"positive_w_kg": work_pos.tolist(), "negative_w_kg": work_neg.tolist()},
        "bed_work": {
            "stance_j": _stats(bed_work),
            "stance_percentiles_0_50_95_100_j": np.percentile(bed_work, (0, 50, 95, 100)).tolist()
            if bed_work
            else None,
            "stances_adding_energy": int((np.asarray(bed_work) > BED_WORK_TOLERANCE_J).sum()),
            "step_max_j": float(np.max(step_max)),
            "flight_step_abs_percentiles_50_95_100_j": np.percentile(np.abs(flight_work), (50, 95, 100)).tolist()
            if flight_work
            else None,
        },
        "joints": joint_summary(angles, torques, stances, mass) if all(len(s) for s in stances) else None,
    }


def _cell(stats: dict, digits: int) -> str:
    if stats["mean"] is None:
        return "-"
    return f"{stats['mean']:.{digits}f} ± {stats['sd']:.{digits}f} (n={stats['n']})"


def markdown(report: dict) -> str:
    """Return the report as markdown tables."""
    sim, f01 = report["sim"], report["f01"]
    survival = sim["survival"]
    lines = [
        f"# Newton verification: {report['policy']}",
        "",
        f"{sim['worlds']} worlds on {report['split']} windows, shoe {report['shoe']}.",
        "",
        "| survival | value |",
        "| --- | --- |",
        *(f"| {name} | {survival[f'{name}_fraction']:.3f} |" for name in OUTCOMES),
        f"| falls per s | {survival['falls_per_s']:.3f} |",
        f"| failures per s | {survival['failures_per_s']:.3f} |",
        f"| episode [s] | {_cell(survival['episode_s'], 3)} |",
        f"| stances per episode | {_cell(survival['stances_per_episode'], 2)} |",
        f"| stances where the bed adds > {BED_WORK_TOLERANCE_J:g} J | {sim['bed_work']['stances_adding_energy']} "
        f"of {sim['bed_work']['stance_j']['n']} |",
        f"| bed work per stance [J] | {_cell(sim['bed_work']['stance_j'], 1)} |",
        f"| largest ground work in one step [J] | {sim['bed_work']['step_max_j']:.1f} |",
        f"| flight-step energy error p50/p95/max [J] | "
        f"{'/'.join(f'{e:.2f}' for e in sim['bed_work']['flight_step_abs_percentiles_50_95_100_j'] or [])} |",
        "",
        "| stance metric | foot | sim | F01 | sim in F01 ± 2 SD |",
        "| --- | --- | --- | --- | --- |",
    ]
    for side in ("right", "left"):
        for key in STANCE_KEYS:
            s, f = sim["stance"][side][key], f01["stance"][side][key]
            inside = s["mean"] is not None and abs(s["mean"] - f["mean"]) <= 2.0 * f["sd"]
            lines.append(f"| {key} | {side} | {_cell(s, 3)} | {_cell(f, 3)} | {'yes' if inside else 'no'} |")
    for key in ("step_time_s", "cadence_steps_per_min"):
        s, f = sim["stance"][key], f01["stance"][key]
        inside = s["mean"] is not None and abs(s["mean"] - f["mean"]) <= 2.0 * f["sd"]
        lines.append(f"| {key} | both | {_cell(s, 3)} | {_cell(f, 3)} | {'yes' if inside else 'no'} |")
    pelvis = sim["pelvis"]
    lines += [
        "",
        "| pelvis | sim | reference | RMSE |",
        "| --- | --- | --- | --- |",
        f"| height [m] | {_cell(pelvis['height_m'], 3)} | {_cell(pelvis['height_reference_m'], 3)} "
        f"| {pelvis['height_rmse_m']:.3f} |",
        f"| tilt [deg] | {_cell(pelvis['tilt_deg'], 1)}, max {pelvis['tilt_max_deg']:.1f} "
        f"| {_cell(pelvis['tilt_reference_deg'], 1)} | {pelvis['tilt_rmse_deg']:.1f} |",
        "",
        "| joint | angle RMSE [deg] | K p5-p50-p95 [N m/rad] | K at low/high | B p5-p50-p95 [N m s/rad] "
        "| B at low/high | +power sim/F01 [W/kg] | -power sim/F01 [W/kg] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    stiffness, damping = sim["impedance"]["stiffness"], sim["impedance"]["damping"]
    for j, name in enumerate(JOINT_NAMES):
        k, b = stiffness["percentiles_0_5_50_95_100"][j], damping["percentiles_0_5_50_95_100"][j]
        lines.append(
            f"| {name} | {sim['joint_rmse_deg'][name]:.1f} | {k[1]:.0f}-{k[2]:.0f}-{k[3]:.0f} "
            f"| {stiffness['at_lower'][j]:.2f}/{stiffness['at_upper'][j]:.2f} "
            f"| {b[1]:.2f}-{b[2]:.2f}-{b[3]:.2f} | {damping['at_lower'][j]:.2f}/{damping['at_upper'][j]:.2f} "
            f"| {sim['joint_power']['positive_w_kg'][j]:.2f}/{f01['joint_power']['positive_w_kg'][j]:.2f} "
            f"| {sim['joint_power']['negative_w_kg'][j]:.2f}/{f01['joint_power']['negative_w_kg'][j]:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify a SuperMimic F01 tracker in Newton (M4 exit gate).")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("outputs/supermimic/reference_F01_2026-10-22-v3"),
        help="reference folder",
    )
    parser.add_argument(
        "--subject", type=Path, help="subject spec whose data folder holds the trials; the reference's by default"
    )
    parser.add_argument("--checkpoint", type=Path, help="tracker checkpoint.pt; the stage-0 law when omitted")
    parser.add_argument("--split", choices=("eval", "train", "all"), default="eval", help="reference windows")
    parser.add_argument("--plot", action="store_true", help="plot the longest episode against the reference")
    parser.add_argument("--output", type=Path, required=True, help="report folder")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    with np.load(args.reference / "reference.npz") as f:
        ref = {k: f[k] for k in f.files}
    body = load_body_params(args.reference / "body_params.json")
    subject = load_subject(args.subject or args.reference / "subject.json")
    shoe = load_shoe(subject.shoe)
    starts = first_flight_starts(np.concatenate([episode_starts(ref, s, "cpu").numpy() for s in SPLITS[args.split]]))
    starts = torch.as_tensor(starts, device=device)
    sim, _ = fixed_shoe_sim(body, shoe, len(starts), float(ref["belt_speed_m_s"]))
    joint_q = torch.as_tensor(ref["joint_q"], dtype=torch.float32, device=device)
    joint_qd = torch.as_tensor(ref["joint_qd"], dtype=torch.float32, device=device)
    moment = torch.as_tensor(reference_moments(ref, body, shoe), dtype=torch.float32, device=device)
    stride = round(sim.config.control_dt_s / float(ref["frame_dt"]))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True) if args.checkpoint else {}
    config = TrackerConfig(goal_frames=checkpoint.get("goal_frames", TrackerConfig.goal_frames))
    # The task reads only the history length of its world model; the sim replaces the dynamics.
    world_model = WorldModel(JOINT_COUNT, XI_DIM, 0, hidden=()).to(device)
    task = TrackerTask(
        world_model, reference_states(joint_q, joint_qd), moment, stride, body, sim.config.control_dt_s, config
    )
    if args.checkpoint:
        policy, name = actor_policy(task, checkpoint["actor"]), str(args.checkpoint)
    else:
        contact = torch.as_tensor(ref["contact"], device=device).reshape(-1, 2)
        policy, name = stage0_policy(task, contact), "stage-0 law"

    result = rollout(sim, task, joint_q, joint_qd, starts, policy)
    mass = body.mass + shoe.pair_mass_kg
    report = {
        "policy": name,
        "reference": str(args.reference),
        "split": args.split,
        "shoe": shoe.name,
        "sim": rollout_summary(result, task, sim.config.control_dt_s, mass),
        "f01": f01_summary(args.reference, subject),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    if args.plot:
        from .viz import plot_trajectory, reference_trajectory  # noqa: PLC0415, needs matplotlib

        i = int(np.argmax(result["steps"]))
        n = result["steps"][i]
        trajectory = {
            "joint_q": np.concatenate((result["state"][i, :n, :7], result["state"][i, :n, 13 : 13 + JOINT_COUNT]), 1),
            "grf": result["grf"][i, :n],
            "action": result["action"][i, :n],
            "dt": sim.config.control_dt_s,
        }
        window, frame = (int(v) for v in starts[i])
        reference, _ = reference_trajectory(args.reference, window, frame + stride)
        plot_trajectory(trajectory, reference, args.output / f"window{window}_frame{frame}.png")


if __name__ == "__main__":
    main()
