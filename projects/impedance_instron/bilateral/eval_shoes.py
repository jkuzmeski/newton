# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Score the shoe-adaptation ladder on shoes it was never trained on.

The ladder arms differ only in what they are told about the shoe, so the question this
script answers is not "who tracks the reference best" but "who is still tracking when the
foam changes". One command runs every arm over every shoe of the protocol and writes one
row per arm, shoe and world plus a summary, so the report builder never re-simulates.

Protocol (design document section 6, gates G2 and G3):

* ``--arm name=POLICY.pt`` is repeatable and one process scores them all. Each arm runs in
  the environment recorded beside its own checkpoint, so it keeps the physics, the solver
  and the observation interface it was trained in; only the shoe, the fall rules and the
  run length belong to this harness. A student arm adds ``--encoder name=ENCODER.pt``,
  which is the other half of it.
* Shoes come from :func:`.shoe_randomization.heldout` and :func:`.shoe_randomization.archetypes`:
  ``interp`` points sit inside the trained prior but off the manifold, ``extrap`` points
  leave it, ``archetype`` points are the named compositions the prior draws half its
  samples from, and ``swap`` changes the shoe mid run. ``nominal`` -- the measured artifact
  in every world -- is always run, because every other number is a paired delta against it.
* Pairing is by world, not by median. The environment is rebuilt per arm and condition with
  one seed and :func:`.endurance3d.reset_once` draws start states from a generator seeded
  with it, so world *w* starts from the same airborne frame of the same measured stride in
  every arm and every shoe. World *w* of a condition then differs from world *w* of
  ``nominal`` by the shoe and nothing else.
* Force is measured twice. The interval-mean force at the control rate is what the training
  reward and the pacer see, so its peaks and loading rates are the ones the policy was
  shaped against. The substep force is recorded too, over a short window, by replaying the
  same kernels eagerly and differencing the accumulator: that is the series
  :mod:`.force_quality` was written for and the only one on which a loading rate faster than
  the control rate exists at all.

Usage (worktree root):

    uv run --no-sync python -m projects.impedance_instron.bilateral.eval_shoes \\
        --arm robust=Z:/newton_grid/runs/shoe_adapt/JOB/00_robust_asym/policy.pt \\
        --arm teacher=Z:/newton_grid/runs/shoe_adapt/JOB/01_teacher_asym/policy.pt \\
        --shoes nominal,interp,extrap,archetype,swap --seconds 10 --raw-seconds 1.5
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from . import force_quality, model3d
from .adaptation import AdaptiveRunner, ShoeEncoder, readout_error
from .endurance3d import CAUSES, heading, reset_once, stored
from .env3d import ACTION_NAMES, BilateralEnv3D, Env3DConfig, ReferenceLibrary3D
from .policy import load_policy
from .shoe_randomization import CONTACT_OBSERVATION_NAMES, THETA_NAMES, ShoeTheta

SETTLE_S = 1.5
"""Time ignored at the start of every condition: the runner is still catching the pacer."""

STANCE_FORCE_N = 50.0
MIN_STANCE_S = 0.06

WIDTHS = {
    "q": 17,
    "qd": 16,
    "q_ref": 17,
    "grf": 6,
    "grf_ref": 6,
    "power": model3d.ACTUATED,
    "position": 2,
    "alive": 1,
    "cadence": 1,
    "crossing": 1,
    "jump": model3d.ACTUATED,
}
"""Channels traced every control step, in trace order. ``cadence`` is the environment's own
estimate and ``crossing`` the inter-leg penetration of the step; the swap probe reads both,
and ``jump`` is recorded because the renderer of these traces needs it."""

RECORDED = tuple(WIDTHS)

JOINTS = tuple(name.removeprefix("equilibrium_") for name in ACTION_NAMES[: model3d.ACTUATED])
"""Actuated joint names, in the order of the recorded joint block."""

GROUPS = ("nominal", "interp", "extrap", "archetype", "swap")

SWAP_AT_S = 4.0
SWAP_EXTRA_S = 7.0
"""When the mid-run swap lands: late enough that the pre-swap gait is measured."""

"""Extra length of the swap recording: adaptation is counted in strides after the swap, and
three consecutive in-band strides is the rule, so the run has to outlive the swap by seconds."""

SWAP_POINT = ShoeTheta(g1_scale=1.6, feq=0.5, tau_scale=3.0)
"""The swap target: stiffer, springier and slower to settle than the measured shoe, the
``eva_like`` archetype without its jitter, so a lag in adaptation shows up in force."""

CAPTIONS = {
    "nominal": "the measured shoe, which every delta is taken against",
    "interp": "inside the trained prior, off the archetype manifold",
    "extrap": "outside the trained prior",
    "archetype": "a named composition the prior draws from",
    "swap": "the shoe changes mid-run",
}


def conditions(groups: tuple[str, ...], named: dict[str, ShoeTheta]) -> dict[str, tuple[ShoeTheta, str]]:
    """The shoes to run, keyed by condition name, each with its protocol group."""
    from .shoe_randomization import heldout  # noqa: PLC0415

    out: dict[str, tuple[ShoeTheta, str]] = {}
    if "nominal" in groups:
        out["nominal"] = (ShoeTheta(), "nominal")
    for name, (theta, long) in heldout().items():
        # The protocol names the groups by what they test; the held-out table spells them
        # out, so the two vocabularies are joined here rather than in both places.
        group = {"interpolation": "interp", "extrapolation": "extrap"}.get(long, long)
        if group in groups:
            out[name] = (theta, group)
    if "archetype" in groups:
        out.update({name: (theta, "archetype") for name, theta in named.items()})
    if "swap" in groups:
        out["swap_to_eva"] = (SWAP_POINT, "swap")
    return out


@dataclass
class Arm:
    """One policy of the ladder, plus the estimator half when it has one."""

    name: str
    policy: Path
    encoder: Path | None = None
    actor: object = None
    runner: AdaptiveRunner | None = None
    config: Env3DConfig | None = None
    meta: dict | None = None


def load_arm(arm: Arm, device: str, shoe_dr: str, leg_contact: str | None = None, actuation: str | None = None) -> Arm:
    """Read a checkpoint and the environment it was trained in.

    The observation names come from the checkpoint rather than from :mod:`.env3d`, because a
    teacher is trained on the shoe parameters and a student on the estimator embedding; both
    are wider than the base interface, and the width has to match exactly.
    """
    meta = json.loads(arm.policy.with_suffix(".json").read_text(encoding="utf-8"))
    arm.actor, arm.meta = load_policy(arm.policy, device, tuple(meta["observations"]), tuple(meta["actions"]))
    overrides: dict[str, object] = {"shoe_dr": shoe_dr, "shoe_swap_probability": 0.0}
    stored_config = stored(Env3DConfig, meta["environment"]["config"])
    if leg_contact is not None:
        overrides["leg_contact"] = leg_contact
    elif stored_config.leg_contact == "off":
        # Crossing is read from the capsule penetration record, which only exists once the
        # capsules are built, so an arm trained without the collision law would report no
        # crossing whatever its legs did. "probe" builds the same capsules with zero stiffness
        # and zero friction: the depth is measured, the dynamics are untouched, and the arm is
        # scored on the plant it trained on. The manifest carries the mode per row.
        overrides["leg_contact"] = "probe"
    if actuation is not None:
        # The impedance setpoints are held across the substeps unless the run asked for a
        # profile, so scoring an arm under a named profile moves it onto that plant.
        overrides["actuation"] = actuation
    arm.config = replace(stored_config, **overrides)
    if arm.encoder is not None:
        payload = torch.load(arm.encoder, map_location=device, weights_only=False)
        described = payload["describe"]
        encoder = ShoeEncoder(
            described["channels"],
            described["steps"],
            scalar_width=described["scalar_width"],
            embedding=described["embedding"],
            theta=described["readout"] or 0,
        ).to(device)
        encoder.load_state_dict(payload["encoder"])
        encoder.eval()
        settings = payload["config"]
        arm.runner = AdaptiveRunner(
            arm.actor, encoder, history_steps=settings["history_steps"], query_every=settings["query_every"]
        )
    elif any(name.startswith("shoe_embedding") for name in meta["observations"]):
        raise ValueError(f"{arm.name}: this policy reads the estimator embedding, so pass --encoder {arm.name}=...")
    return arm


def build(arm: Arm, library: ReferenceLibrary3D, worlds: int, seed: int, compression_limit: float) -> BilateralEnv3D:
    """Build the arm's own environment, with the shoe under the randomizer's control.

    ``shoe_dr`` is forced on even for an arm trained without it, because the randomizer is
    the only mechanism that gives each world its own foam. That does not change what an arm
    can see: the observation interface depends on ``contact_observation`` and
    ``theta_observation``, both of which stay exactly as the checkpoint stored them.
    """
    env = BilateralEnv3D(library, worlds, replace(arm.config, compression_limit=compression_limit), seed=seed)
    if env.shoe_randomizer is None:
        raise ValueError("the environment was built without shoe randomization")
    return env


def controller(arm: Arm, env: BilateralEnv3D) -> Callable[[torch.Tensor], torch.Tensor]:
    """The step-time controller, holding its own history when the arm is a student."""
    if arm.runner is None:
        actor = arm.actor

        def act(observation: torch.Tensor) -> torch.Tensor:
            return actor(observation)

        return act
    arm.runner.reset(env, env.worlds)

    def act_student(observation: torch.Tensor) -> torch.Tensor:
        action = arm.runner.act(observation)
        arm.runner.push(observation, action)
        return action

    return act_student


@torch.no_grad()
def raw_substeps(env: BilateralEnv3D, act: Callable[[torch.Tensor], torch.Tensor], steps: int) -> torch.Tensor:
    """Vertical foot force per substep, ``[steps, substeps, worlds, sides]``.

    :meth:`.env3d.BilateralEnv3D.integrate` records its substeps into one CUDA graph, so the
    accumulator can only be read once per control step. This replays the same kernels from
    Python -- :meth:`._substep` in the same alternating state order -- and differences the
    accumulator, which is the force of that substep alone. Identical launches, so identical
    numbers; slow enough that it is only run over the window scoring asks for.
    """
    states = env.states
    substeps = env.config.substeps
    out = torch.zeros(steps, substeps, env.worlds, 2, device=env.torch_device)
    for step in range(steps):
        env._stage(act(env.observe()))
        env._pre()
        env._clear_interval()
        previous = torch.zeros(env.worlds, 2, device=env.torch_device)
        for k in range(substeps):
            env._substep(states[k % 2], states[(k + 1) % 2])
            cumulative = env.t.grf[:, :, 2]
            out[step, k] = cumulative - previous
            previous = cumulative.clone()
        env._post()
    return out


@torch.no_grad()
def rollout(
    env: BilateralEnv3D,
    act: Callable[[torch.Tensor], torch.Tensor],
    seconds: float,
    raw_seconds: float,
    swap: tuple[int, ShoeTheta] | None = None,
    fall_height_m: float = 0.55,
    fall_tilt_rad: float = math.radians(60.0),
) -> dict:
    """Run every world until it falls or the clock runs out, tracing :data:`RECORDED`.

    Args:
        env: Environment whose shoe is pinned and whose worlds are already reset.
        act: Maps the observation to impedance logits: the frozen deterministic mean.
        seconds: Length of the run.
        raw_seconds: Length of the substep force window, taken just after settling.
        swap: ``(control_step, theta)`` to install mid-run, for the adaptation-lag probe.
        fall_height_m: Pelvis height below which a world has fallen.
        fall_tilt_rad: Pelvis tilt past which a world has fallen.

    Returns:
        The trace as ``[steps + 1, worlds, sum(WIDTHS)]`` with the column spans in
        ``columns``, the fall bookkeeping, and the substep window if it was asked for.
    """
    dt = env.config.control_dt_s
    steps = int(round(seconds / dt))
    device = env.torch_device
    trace = torch.zeros(steps + 1, env.worlds, sum(WIDTHS.values()), device=device)
    # The per-foot contact block, recorded beside the trace on the same clock: travel as a
    # fraction of foam thickness against normal load is the loss loop, and neither half of it
    # is recoverable afterwards. The channels exist only for an arm built with the contact
    # observation, so an arm without it records nothing rather than a block of zeros that a
    # reader could plot as if it were a measurement.
    width = env._contact_channels().shape[1]
    contact = torch.zeros(steps + 1, env.worlds, width, device=device) if width else None
    alive = torch.ones(env.worlds, dtype=torch.bool, device=device)
    causes = torch.zeros(env.worlds, dtype=torch.long, device=device)
    falls = torch.full((env.worlds,), -1, dtype=torch.long, device=device)
    start_x = env.position()[:, 0].clone()
    raw_at = int(round((SETTLE_S + 0.5) / dt))
    raw_wanted = min(int(round(raw_seconds / dt)), max(steps - raw_at - 1, 0))
    raw = torch.zeros(max(raw_wanted, 0), env.config.substeps, env.worlds, 2, device=device)
    columns = np.cumsum([0] + [WIDTHS[name] for name in RECORDED])

    def write(frame: int) -> None:
        q, qd = env.joint_q, env.joint_qd
        q_ref, _, grf_ref = env._reference()
        substeps = env.t.substeps.clamp(min=1)[:, None]
        values = {
            "q": q,
            "qd": qd,
            "q_ref": q_ref,
            "grf": env._interval_force(),
            "grf_ref": grf_ref,
            "power": env.t.power_positive / substeps,
            "position": env.position(),
            "alive": alive.to(q.dtype),
            "cadence": env.cadence_spm,
            "crossing": env.leg_contact_depth
            if env.leg_contact is not None
            else torch.zeros(env.worlds, device=device),
            "jump": env.jump,
        }
        trace[frame] = torch.cat([values[name].reshape(env.worlds, WIDTHS[name]) for name in RECORDED], dim=-1)
        if contact is not None:
            contact[frame] = env._contact_channels()

    write(0)
    for step in range(1, steps + 1):
        if swap is not None and step == swap[0]:
            env.shoe_randomizer.pin([swap[1]] * env.worlds)
        if raw_wanted and raw_at <= step < raw_at + raw_wanted:
            raw[step - raw_at] = raw_substeps(env, act, 1)[0]
        else:
            env.advance(act(env.observe()))
        low = env.joint_q[:, 2] < fall_height_m
        tilt = heading(env.joint_q[:, 3:7])[1] > fall_tilt_rad
        fallen = (env.t.status != 0) | low | tilt
        fresh = fallen & alive
        reason = torch.where(env.t.status != 0, env.t.status.long(), torch.where(low, 100, 101))
        causes = torch.where(fresh & (causes == 0), reason, causes)
        falls = torch.where(fresh, torch.full_like(falls, step), falls)
        alive &= ~fallen
        write(step)
    return {
        "trace": trace.cpu().numpy(),
        "contact": None if contact is None else contact.cpu().numpy(),
        "columns": columns,
        "causes": causes.cpu().tolist(),
        "fall_steps": falls.cpu().tolist(),
        "start_x": start_x.cpu().numpy(),
        "raw": raw.cpu().numpy(),
        "raw_start_step": raw_at,
        "steps": steps,
    }


def touchdowns(force: np.ndarray, threshold_n: float = STANCE_FORCE_N) -> np.ndarray:
    """Frames at which one foot's vertical force rises through the stance threshold."""
    return np.flatnonzero(np.diff((force > threshold_n).astype(int)) == 1) + 1


def loading_rate(force: np.ndarray, dt_s: float, threshold_n: float = STANCE_FORCE_N) -> tuple[float, float]:
    """Median average loading rate and maximum instantaneous rate of one trace [N/s].

    The average is the peak divided by the time from touchdown to that peak, the definition
    that survives sampling: at the 100 Hz control rate it is what the training reward saw,
    and on the substep trace it is the impact slope the roughness studies quote.
    """
    averages, instant = [], []
    for start, end in force_quality.stances(force, dt_s, threshold_n, MIN_STANCE_S):
        window = force[start:end]
        peak = int(np.argmax(window))
        if peak < 1:
            continue
        averages.append(float(window[peak] / (peak * dt_s)))
        instant.append(float(np.max(np.diff(window)) / dt_s))
    if not averages:
        return float("nan"), float("nan")
    return float(np.median(averages)), float(np.max(instant))


def score_world(rec: dict, world: int, env: BilateralEnv3D, start_frame: int | None = None) -> dict:
    """One world's gait in one shoe, scored between settling and its fall.

    Cadence and peaks are computed from the interval-mean force, which is what the reward
    and the pacer react to; the same quantities on the substep trace are added under a
    ``raw_`` prefix, together with the roughness measures of :mod:`.force_quality`, which
    only mean anything at that rate.
    """
    dt, dt_s = env.config.control_dt_s, env.config.dt_s
    substeps = env.config.substeps
    column = {name: int(index) for name, index in zip(RECORDED, rec["columns"][:-1], strict=True)}
    trace = rec["trace"]
    fall = int(rec["fall_steps"][world])
    first = int(round(SETTLE_S / dt)) if start_frame is None else start_frame
    last = fall if fall > 0 else rec["steps"]
    # A world that falls during settling is scored from the start rather than dropped: its
    # fall is the finding, and an empty window would report a negative time and zero distance.
    early_fall = last <= first
    if early_fall:
        first, last = 0, max(last, 1)
    window = slice(first, last)
    time_s = (last - first) * dt

    def frame(name: str) -> np.ndarray:
        return trace[window, world, column[name] : column[name] + WIDTHS[name]]

    q, q_ref = frame("q"), frame("q_ref")
    grf, grf_ref, power, position = frame("grf"), frame("grf_ref"), frame("power"), frame("position")
    distance = float(position[-1, 0] - position[0, 0])
    joints = q[:, 7:] - q_ref[:, 7:]
    per_foot, cadence, cadence_ref, peaks, rates = {}, [], [], [], []
    for side, suffix in ((0, "right"), (1, "left")):
        fz = grf[:, 3 * side + 2]
        found = force_quality.stances(fz, dt, STANCE_FORCE_N, MIN_STANCE_S)
        touch = touchdowns(fz)
        if len(touch) > 1:
            cadence.append(120.0 / (np.diff(touch).mean() * dt))
        reference = touchdowns(grf_ref[:, 3 * side + 2])
        if len(reference) > 1:
            cadence_ref.append(120.0 / (np.diff(reference).mean() * dt))
        peaks.extend(float(fz[a:b].max()) for a, b in found)
        average, _ = loading_rate(fz, dt)
        rates.append(average)
        per_foot[suffix] = {
            "stances": len(found),
            "loaded_fraction": round(float((fz > STANCE_FORCE_N).mean()), 3),
            "peak_n": round(float(np.median([fz[a:b].max() for a, b in found])), 1) if found else None,
            "loading_rate_n_s": round(average, 1) if np.isfinite(average) else None,
        }
    steps_counted = max(sum(v["stances"] for v in per_foot.values()), 1)
    work_j = float(power.sum()) * dt
    cause = int(rec["causes"][world])
    row = {
        "world": world,
        "fell": fall > 0,
        "cause": CAUSES.get(cause, f"code {cause}") if cause else None,
        "cause_code": cause or None,
        "fall_time_s": round(fall * dt, 3) if fall > 0 else None,
        "scored_s": round(time_s, 3),
        "fell_early": early_fall,
        "cadence_steps_per_min": round(float(np.mean(cadence)), 1) if cadence else None,
        "cadence_reference_steps_per_min": round(float(np.mean(cadence_ref)), 1) if cadence_ref else None,
        "distance_x_m": round(distance, 3),
        "progress_speed_m_s": round(distance / time_s, 3) if time_s > 0 else None,
        "stride_length_m": round(distance / (steps_counted / 2.0), 3) if time_s > 0 else None,
        "grf_peak_n": round(float(np.median(peaks)), 1) if peaks else None,
        "loading_rate_n_s": round(float(np.nanmean(rates)), 1) if np.isfinite(rates).any() else None,
        "joint_rms_mean_deg": round(float(np.degrees(np.sqrt((joints**2).mean()))), 2),
        "joint_rms_deg": dict(
            zip(JOINTS, np.round(np.degrees(np.sqrt((joints**2).mean(axis=0))), 2).tolist(), strict=True)
        ),
        "positive_work_j": round(work_j, 2),
        "positive_work_j_per_m": round(work_j / distance, 2) if distance > 0.5 else None,
        "crossing_fraction": round(float((frame("crossing") > 1e-4).mean()), 3),
        "torque_jump_p99": round(float(np.percentile(np.abs(frame("jump")), 99)), 3),
        "feet": per_foot,
    }
    raw = rec["raw"]
    if raw.size:
        for side, suffix in ((0, "right"), (1, "left")):
            trace_raw = raw[:, :, world, side].reshape(-1).astype(float)
            quality = force_quality.quality(trace_raw, dt_s, substeps)
            average, instant = loading_rate(trace_raw, dt_s)
            row[f"raw_{suffix}"] = {
                "peak_n": quality.get("peak_raw_n"),
                "hf20_rms_n": quality.get("hf20_rms_n"),
                "hf30_rms_n": quality.get("hf30_rms_n"),
                "hf30_midstance_rms_n": quality.get("hf30_midstance_rms_n"),
                "jerk_rms_n_s3": quality.get("jerk_rms_n_s3"),
                "loading_rate_n_s": average if np.isfinite(average) else None,
                "loading_rate_max_n_s": instant if np.isfinite(instant) else None,
                "stances": quality.get("stances"),
            }
        for metric in (
            "peak_n",
            "hf20_rms_n",
            "hf30_rms_n",
            "hf30_midstance_rms_n",
            "jerk_rms_n_s3",
            "loading_rate_max_n_s",
        ):
            pooled = [row[f"raw_{s}"][metric] for s in ("right", "left") if row[f"raw_{s}"][metric] is not None]
            row[f"raw_{metric}"] = (
                round(float(np.median(pooled) if metric == "peak_n" else np.mean(pooled)), 1) if pooled else None
            )
        rates_raw = [v for v in (row[f"raw_{s}"]["loading_rate_n_s"] for s in ("right", "left")) if v is not None]
        row["raw_loading_rate_n_s"] = round(float(np.mean(rates_raw)), 1) if rates_raw else None
    return row


def adaptation_lag(rec: dict, world: int, env: BilateralEnv3D, swap_step: int) -> dict:
    """How many strides it takes to get the old gait back after a mid-run shoe change.

    Each stride is scored on the vertical peak, its own cadence, and the pose error against
    the reference. A world is back when all three sit inside the pre-swap band for three
    consecutive strides; a world that never returns is reported as ``None`` with the number
    of strides it did get, so the table cannot silently average away a failure to adapt.
    """
    dt = env.config.control_dt_s
    column = {name: int(index) for name, index in zip(RECORDED, rec["columns"][:-1], strict=True)}
    trace = rec["trace"]
    fz = trace[:, world, column["grf"] + 2]
    touch = touchdowns(fz)
    q, q_ref = trace[:, world, column["q"] : column["q"] + 17], trace[:, world, column["q_ref"] : column["q_ref"] + 17]

    def stride_metrics(first: int, last: int) -> tuple[float, float, float]:
        window = slice(first, last)
        pose = float(np.degrees(np.sqrt(((q[window, 7:] - q_ref[window, 7:]) ** 2).mean())))
        return float(fz[first:last].max()), 120.0 / max((last - first) * dt, 1e-6), pose

    usable = touch[(touch > int(round(SETTLE_S / dt))) & (touch < rec["steps"])]
    if len(usable) < 4:
        return {"strides_scored": int(max(len(usable) - 1, 0))}
    strides = list(pairwise(usable))
    before = [stride_metrics(a, b) for a, b in strides if b <= swap_step][-4:]
    after = [stride_metrics(a, b) for a, b in strides if a >= swap_step]
    if not before or not after:
        return {"strides_scored": len(usable) - 1, "note": "swap left no window on one side"}

    def crossing_index(values: list[float], baseline: float, band: float, floor: float) -> int | None:
        errors = [abs(value - baseline) for value in values]
        tolerance = max(band * abs(baseline), floor)
        for index in range(len(errors) - 2):
            if all(error <= tolerance for error in errors[index : index + 3]):
                return index + 1
        return None

    pre = [float(np.median([stride[k] for stride in before])) for k in range(3)]
    return {
        "pre_peak_n": round(pre[0], 1),
        "pre_cadence_steps_per_min": round(pre[1], 1),
        "pre_pose_deg": round(pre[2], 2),
        "post_peak_n": round(float(np.median([stride[0] for stride in after])), 1),
        "post_cadence_steps_per_min": round(float(np.median([stride[1] for stride in after])), 1),
        "post_pose_deg": round(float(np.median([stride[2] for stride in after])), 2),
        "peak_change_fraction": round(float(after[-1][0] / pre[0] - 1.0), 3),
        "strides_to_adapt_peak": crossing_index([stride[0] for stride in after], pre[0], 0.1, 5.0),
        "strides_to_adapt_cadence": crossing_index([stride[1] for stride in after], pre[1], 0.05, 5.0),
        "strides_to_adapt_pose": crossing_index([stride[2] for stride in after], pre[2], 0.25, 1.0),
        "strides_after_swap": len(after),
    }


def flatten(row: dict) -> dict:
    """Scalar metrics of one row, with the per-foot and per-rate blocks folded in."""
    out = {key: value for key, value in row.items() if isinstance(value, (int, float, bool)) or value is None}
    for side, block in row.get("feet", {}).items():
        out[f"{side}_peak_n"] = block.get("peak_n")
        out[f"{side}_loading_rate_n_s"] = block.get("loading_rate_n_s")
        out[f"{side}_loaded_fraction"] = block.get("loaded_fraction")
    for key, value in row.items():
        if key.startswith("raw_") and isinstance(value, dict):
            for name, number in value.items():
                out[f"{key}_{name}"] = number
    for key, value in row.get("lag", {}).items():
        out[f"lag_{key}"] = value
    return out


SEMANTICS = {
    "g1_scale": "Ogden-Hill shear modulus multiplier on instantaneous_shear_modulus_pa: the bed stiffness",
    "alpha1": "first Hyperfoam exponent",
    "feq": "equilibrium fraction; 1 - feq is the share that relaxes away during stance",
    "tau_scale": "Maxwell relaxation time multiplier",
    "g2_scale": "second Ogden-Hill shear modulus multiplier; zero disables the term",
    "normal_damping_n_s_m": "per-column Kelvin-Voigt normal damping [N s/m]",
    "mu": "elastic-Coulomb friction coefficient against the belt",
    "stack_dz_m": "stack-height offset [m], positive is a thicker sole: the free-height knob, and the only theta component that moves geometry",
    "poisson": "effective Poisson ratio, the lateral bulge under compression",
}
"""What each theta component does, written into every recording so a viewer can label a column."""

METRICS = (
    "cadence_steps_per_min",
    "stride_length_m",
    "distance_x_m",
    "progress_speed_m_s",
    "grf_peak_n",
    "raw_peak_n",
    "loading_rate_n_s",
    "raw_loading_rate_n_s",
    "raw_loading_rate_max_n_s",
    "raw_hf30_rms_n",
    "raw_jerk_rms_n_s3",
    "positive_work_j_per_m",
    "joint_rms_mean_deg",
    "torque_jump_p99",
    "crossing_fraction",
    "scored_s",
    "fell",
    "readout_mean_halfwidths",
    "readout_median_halfwidths",
    "readout_p90_halfwidths",
    "lag_strides_to_adapt_peak",
    "lag_strides_to_adapt_cadence",
    "lag_strides_to_adapt_pose",
)
"""The columns of the summary and of the report table."""


def median(values: list) -> float | None:
    numbers = [float(value) for value in values if value is not None and np.isfinite(value)]
    return round(float(np.median(numbers)), 4) if numbers else None


def summarize(rows: list[dict]) -> dict:
    """Median per arm and shoe, and the paired delta of every shoe against ``nominal``.

    The delta is taken world by world before the median, which is the point of the pairing:
    it is the change the same runner at the same start state experienced, not the difference
    of two pooled populations whose spread would swallow it.
    """
    flat = {(row["arm"], row["condition"], row["world"]): flatten(row) for row in rows}
    keys = sorted({(arm, condition) for arm, condition, _ in flat})
    report: dict[str, dict] = {}
    for arm, condition in keys:
        worlds = sorted({world for (a, c, world) in flat if (a, c) == (arm, condition)})
        baseline = {(a, c, w): value for (a, c, w), value in flat.items() if c == "nominal" and a == arm}
        cells: dict[str, dict] = {}
        for metric in METRICS:
            values, deltas = [], []
            for world in worlds:
                here = flat[(arm, condition, world)].get(metric)
                values.append(here)
                reference = baseline.get((arm, "nominal", world), {}).get(metric)
                if here is not None and reference is not None:
                    deltas.append(here - reference)
            cells[metric] = {
                "median": median(values),
                "delta_vs_nominal": None if condition == "nominal" or not deltas else median(deltas),
                "worlds": len([v for v in values if v is not None]),
            }
        # Falls are a count. A median of booleans is 0 or 1, and a reader who finds a column
        # named ``fell`` beside a column of medians reads that 1 as one world and the 0 as none,
        # where the softest extrapolation's truth was that all thirty-two worlds fell.
        fallen = [world for world in worlds if flat[(arm, condition, world)].get("fell")]
        cells["fell"] = {
            "median": round(len(fallen) / len(worlds), 3) if worlds else None,
            "count": f"{len(fallen)}/{len(worlds)}",
            "delta_vs_nominal": None,
            "worlds": len(worlds),
        }
        report[f"{arm}/{condition}"] = cells
    return report


def markdown(rows: list[dict], report: dict) -> str:
    """A table per arm: one row per shoe, one column per :data:`METRICS` entry."""
    lines = ["# Held-out shoe evaluation", ""]
    for arm in sorted({row["arm"] for row in rows}):
        conditions_run = sorted({row["condition"] for row in rows if row["arm"] == arm})
        lines += [f"## {arm}", "", "| shoe | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for condition in conditions_run:
            cells = report[f"{arm}/{condition}"]
            body = []
            for metric in METRICS:
                if metric == "fell":
                    body.append(cells[metric]["count"])
                    continue
                value = cells[metric]["median"]
                delta = cells[metric]["delta_vs_nominal"]
                shown = "-" if value is None else f"{value:g}"
                if delta not in (None, 0.0):
                    shown += f" ({delta:+g})"
                body.append(shown)
            lines.append(f"| {condition} | " + " | ".join(body) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_series(
    path: Path,
    rec: dict,
    env: BilateralEnv3D,
    theta_rows: np.ndarray,
    swap_step: int,
    theta_before: np.ndarray,
    dataset: str,
) -> dict:
    """Write the recording for the renderer, and return its manifest entry.

    The key names and layouts are the ones :mod:`.gait3d` already writes, so a renderer that
    reads a training recording reads this one too; ``position`` is added because the
    environment recenters and ``joint_q`` alone cannot say how far the runner has gone.
    """
    trace = rec["trace"]
    arrays = {}
    for name, width in WIDTHS.items():
        start = int(rec["columns"][list(WIDTHS).index(name)])
        arrays[name] = trace[..., start : start + width]
    series = {}
    if rec["contact"] is not None:
        series["contact"] = rec["contact"].transpose(1, 0, 2).astype(np.float32)
        series["contact_names"] = np.array(CONTACT_OBSERVATION_NAMES)
    np.savez_compressed(
        path,
        **series,
        control_dt_s=np.float32(env.config.control_dt_s),
        substep_dt_s=np.float32(env.config.dt_s),
        q=arrays["q"],
        qd=arrays["qd"],
        q_ref=arrays["q_ref"],
        grf=arrays["grf"],
        grf_ref=arrays["grf_ref"],
        power=arrays["power"],
        position=arrays["position"],
        cadence=arrays["cadence"],
        crossing=arrays["crossing"],
        jump=arrays["jump"],
        alive=arrays["alive"][..., 0] > 0.5,
        raw_fz=rec["raw"],
        raw_start_step=np.int64(rec["raw_start_step"]),
        theta=theta_rows.astype(np.float32),
        theta_before=np.asarray(theta_before, dtype=np.float32),
        theta_after=theta_rows.astype(np.float32),
        theta_names=np.array(THETA_NAMES),
        theta_semantics=json.dumps(SEMANTICS),
        shoe_dataset=dataset,
        shoe_geometry="identical bed and shell asset in every condition; only stack_dz_m moves geometry",
        world_trial=env.world_trials.cpu().numpy().astype(np.int64),
        segment=env.segment.cpu().numpy().astype(np.int64),
        start_index=env.index.cpu().numpy().astype(np.int64),
        swap_step=np.int64(swap_step),
    )
    return {
        "file": path.name,
        "frames": int(trace.shape[0]),
        "worlds": int(trace.shape[1]),
        "trials": sorted({int(v) for v in env.world_trials.cpu().numpy().tolist()}),
        **({"contact_channels": list(CONTACT_OBSERVATION_NAMES)} if rec["contact"] is not None else {}),
    }


def parse_arm(spec: str, encoders: dict[str, str]) -> Arm:
    name, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"--arm wants name=path, got {spec!r}")
    return Arm(name=name, policy=Path(path), encoder=Path(encoders[name]) if name in encoders else None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        default=None,
        help="Skip simulation: re-read rows.json from these directories and write one combined "
        "summary. One array index per arm is how a ladder is evaluated in parallel, and the "
        "report wants one table.",
    )
    parser.add_argument("--arm", action="append", required=False, help="name=POLICY.pt, repeatable.")
    parser.add_argument("--encoder", action="append", default=[], help="name=ENCODER.pt for a student arm.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/bilateral/dataset_3d_static_sidefix"),
        help="Reference dataset; use the one the arms trained on.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=8, help="Worlds per shoe, each a different start state.")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--raw-seconds", type=float, default=1.5)
    parser.add_argument("--shoes", default="nominal,interp,extrap,archetype,swap")
    parser.add_argument(
        "--only",
        default="",
        help="Comma list of condition names to keep; nominal is always kept. For rerunning one shoe.",
    )
    parser.add_argument("--seed", type=int, default=3, help="Same seed everywhere: it is what pairs the worlds.")
    parser.add_argument(
        "--compression-limit",
        type=float,
        default=0.999,
        help="Foam crush fraction that retires a world. Evaluation raises it above the training "
        "screen so a very soft shoe is scored rather than aborted; the training value is 0.99.",
    )
    parser.add_argument("--shoe-dr", default="wide", help="Preset the theta encoding is normalized by.")
    parser.add_argument(
        "--actuation",
        default=None,
        help="Score every arm under this substep actuation profile of the impedance setpoints "
        "rather than the one stored with its checkpoint, which is the zero-order hold.",
    )
    parser.add_argument(
        "--leg-contact",
        default=None,
        help="Score every arm on this inter-leg contact law rather than the one stored with its "
        "checkpoint. The crossing metric reads the contact depth, so an arm trained without the "
        "law would report no crossing by construction.",
    )
    parser.add_argument("--fall-height-m", type=float, default=0.55)
    parser.add_argument("--fall-tilt-deg", type=float, default=60.0)
    parser.add_argument("--no-series", action="store_true", help="Score only; skip the npz recordings.")
    args = parser.parse_args(argv)

    if args.merge is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        merged: list[dict] = []
        manifest: list[dict] = []
        for directory in args.merge:
            # The flat columns the summary is built from go into the file as well, so a
            # reader who only opens rows.json - the paired bootstrap, for instance - can see
            # the per-world swap lags and per-foot peaks instead of only the medians.
            merged += [
                {**row, **flatten(row)} for row in json.loads((directory / "rows.json").read_text(encoding="utf-8"))
            ]
            path = directory / "manifest.json"
            if path.exists():
                manifest += json.loads(path.read_text(encoding="utf-8"))
        report = summarize(merged)
        (args.output / "rows.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.output / "summary.md").write_text(markdown(merged, report), encoding="utf-8")
        print(f"merged {len(merged)} rows from {len(args.merge)} runs into {args.output}")
        return

    if not args.arm:
        raise SystemExit("pass --arm name=policy.pt, or --merge DIR ...")
    encoders = {}
    for spec in args.encoder:
        name, _, path = spec.partition("=")
        encoders[name] = path
    arms = [parse_arm(spec, encoders) for spec in args.arm]
    groups = tuple(part for part in args.shoes.split(",") if part)
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise SystemExit(f"unknown shoe groups {sorted(unknown)}; expected some of {list(GROUPS)}")

    only = {part for part in args.only.split(",") if part}
    known = set(conditions(tuple(set(groups) | {"interp", "extrap", "nominal", "swap"}), {}))
    unknown = only - known
    if unknown:
        raise SystemExit(f"unknown shoe names {sorted(unknown)}; expected some of {sorted(known)}")

    args.output.mkdir(parents=True, exist_ok=True)
    libraries: dict[float, ReferenceLibrary3D] = {}
    named: dict[str, ShoeTheta] = {}
    rows: list[dict] = []
    manifest: list[dict] = []
    for arm in arms:
        load_arm(arm, args.device, args.shoe_dr, args.leg_contact, args.actuation)
        control_dt_s = arm.config.control_dt_s
        if control_dt_s not in libraries:
            libraries[control_dt_s] = ReferenceLibrary3D(args.dataset, (args.split,), control_dt_s, args.device)
        if not named:
            from .shoe_randomization import archetypes, reference_of  # noqa: PLC0415

            probe = build(arm, libraries[control_dt_s], 1, args.seed, args.compression_limit)
            named.update(archetypes(reference_of(probe)))
        wanted = conditions(groups, named)
        missing = only - set(wanted)
        if missing:
            raise SystemExit(f"unknown shoe names {sorted(missing)}; expected some of {sorted(wanted)}")
        for condition, (theta, group) in wanted.items():
            if only and condition not in only and group != "nominal":
                continue
            swap_step = int(round(SWAP_AT_S / control_dt_s)) if group == "swap" else -1
            # A swap is measured against the shoe the arm was anchored on, so it starts on the
            # measured artifact and changes to the condition's shoe at swap_step.
            pin_first = ShoeTheta() if swap_step > 0 else theta
            env = build(arm, libraries[control_dt_s], args.replicates, args.seed, args.compression_limit)
            env.capture()
            reset_once(env)
            truth = env.shoe_randomizer.pin([pin_first] * env.worlds)
            rec = rollout(
                env,
                controller(arm, env),
                args.seconds + (SWAP_EXTRA_S if swap_step > 0 else 0.0),
                args.raw_seconds if swap_step < 0 else 0.0,
                swap=(swap_step, theta) if swap_step > 0 else None,
                fall_height_m=args.fall_height_m,
                fall_tilt_rad=math.radians(args.fall_tilt_deg),
            )
            for world in range(env.worlds):
                row = score_world(rec, world, env)
                if swap_step > 0:
                    row["lag"] = adaptation_lag(rec, world, env, swap_step)
                if arm.runner is not None:
                    _, theta_hat = arm.runner.encoder(arm.runner.buffer.windows())
                    row["readout"] = readout_error(theta_hat, env.shoe_randomizer.privileged())
                    row["readout_mean_halfwidths"] = row["readout"]["mean"]
                    row["readout_median_halfwidths"] = row["readout"]["median"]
                    row["readout_p90_halfwidths"] = row["readout"]["p90"]
                rows.append(
                    {
                        "arm": arm.name,
                        "condition": condition,
                        "group": group,
                        "theta": {
                            name: round(float(value), 5) for name, value in zip(THETA_NAMES, truth[world], strict=True)
                        },
                        "theta_normalized": [round(float(v), 4) for v in env.shoe_randomizer.privileged()[world]],
                        **row,
                    }
                )
            entry = {
                "arm": arm.name,
                "condition": condition,
                "group": group,
                "caption": CAPTIONS[group],
                "policy": str(arm.policy),
                "encoder": None if arm.encoder is None else str(arm.encoder),
                "dataset": str(args.dataset),
                "split": args.split,
                "seed": args.seed,
                "seconds": args.seconds,
                "control_dt_s": control_dt_s,
                "solver": arm.config.solver,
                "actuation": arm.config.actuation,
                "leg_contact": arm.config.leg_contact,
                "compression_limit": args.compression_limit,
                "swap_step": swap_step,
                "shoe_dataset": str(args.dataset),
                "theta_names": list(THETA_NAMES),
                "theta_before": [round(float(v), 5) for v in truth[0]],
                "theta_after": [round(float(v), 5) for v in env.shoe_randomizer.host[0]],
            }
            if not args.no_series:
                entry.update(
                    write_series(
                        args.output / f"series_{arm.name}__{condition}.npz",
                        rec,
                        env,
                        env.shoe_randomizer.host.copy(),
                        swap_step,
                        truth,
                        str(args.dataset),
                    )
                )
            manifest.append(entry)
            fell = sum(1 for row in rows if row["arm"] == arm.name and row["condition"] == condition and row["fell"])
            print(
                f"  {arm.name:>16s} {condition:>18s}  {args.replicates:2d} worlds  fell {fell}  "
                f"cadence {median([r['cadence_steps_per_min'] for r in rows if r['arm'] == arm.name and r['condition'] == condition])}  "
                f"peak {median([r['grf_peak_n'] for r in rows if r['arm'] == arm.name and r['condition'] == condition])} N",
                flush=True,
            )
    report = summarize(rows)
    (args.output / "rows.json").write_text(
        json.dumps([{**r, **flatten(r)} for r in rows], indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(rows, report), encoding="utf-8")
    print(f"wrote {len(rows)} rows and {len(manifest)} recordings to {args.output}")


if __name__ == "__main__":
    main()
