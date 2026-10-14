# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic free rollouts of bilateral controllers from measured segment starts.

Each world starts at the end of a segment's three-frame observed-position prefix,
in flight, and runs open loop to the segment's end with no reference input,
departure termination, or reset. Initial joint rates come from backward
quadratic differentiation of that prefix; where the dataset provides the
preceding stride's plate-integrated COM velocity, the root velocity is chosen so
the model's COM moves with it. The oscillator phase follows the kinematic
convention of :func:`..hogan.identify.initialize`. Only screen failures stop a
world early. Measurements then score the traces offline.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..hogan.identify import ForceObservation
from .env import BilateralEnv, ReferenceLibrary
from .gpu import FAILURES

PREFIX_FRAMES = 3


@dataclass(frozen=True)
class Start:
    """A deterministic initial state at the end of a segment's observation prefix."""

    q: np.ndarray
    v: np.ndarray
    origin_s: float
    """Prediction origin on the segment's motion clock [s]."""
    phase_rad: float


def initial_state(library: ReferenceLibrary, segment: int) -> Start:
    """Initialize from the first three measured frames of a segment."""
    reference = library.references[segment]
    member = library.members[segment]
    chain = library.chains[segment]
    time = reference["time_s"][:PREFIX_FRAMES]
    q = reference["q"][:PREFIX_FRAMES]
    t = time - time[-1]
    coefficients = np.linalg.solve(np.column_stack((np.ones(3), t, t * t)), q)
    velocity = coefficients[1].copy()
    flight = member.get("flight_velocity_m_s")
    if flight is not None:
        if member.get("flight_velocity_frame") != PREFIX_FRAMES - 1:
            raise ValueError("flight_velocity_m_s must be given at the prefix end")
        velocity[:2] = np.asarray(flight) - chain.com_jacobian(q[-1])[:, 2:] @ velocity[2:]
    phase = math.atan2(-velocity[3] / 8.0, q[-1, 3]) % (2 * math.pi)
    return Start(q[-1].copy(), velocity, float(time[-1]), phase)


class SegmentRollouts:
    """Run deterministic rollouts of ``candidates`` controllers over a library's segments.

    World ``w`` runs candidate ``w // segments`` on segment ``w % segments``.
    """

    def __init__(self, env: BilateralEnv, candidates: int = 1, horizon_s: float | None = None):
        self.env = env
        self.library = env.library
        self.segments = len(self.library)
        self.candidates = int(candidates)
        if env.worlds != self.segments * self.candidates:
            raise ValueError("Environment worlds must equal segments times candidates")
        dt = env.config.control_dt_s
        starts = [initial_state(self.library, s) for s in range(self.segments)]
        self.starts = starts
        durations = np.array(
            [
                reference["time_s"][-1] - start.origin_s
                for reference, start in zip(self.library.references, starts, strict=True)
            ]
        )
        if horizon_s is not None:
            durations = np.minimum(durations, horizon_s)
        self.durations_s = durations
        self.steps = np.floor(durations / dt).astype(int)
        self.max_steps = int(self.steps.max())
        dev = env.torch_device
        self.q0 = torch.tensor(np.array([s.q for s in starts]), dtype=torch.float64, device=dev)
        self.v0 = torch.tensor(np.array([s.v for s in starts]), dtype=torch.float64, device=dev)
        self.phase0 = torch.tensor([s.phase_rad for s in starts], dtype=torch.float64, device=dev)
        self.origin_index = torch.tensor([int(round(s.origin_s / dt)) for s in starts], dtype=torch.long, device=dev)

    def reset(self):
        env = self.env
        segment = torch.arange(env.worlds, device=env.torch_device) % self.segments
        q = self.q0[segment].clone()
        origin = q[:, :2].clone()
        origin[:, 1] = 0.0
        q[:, :2] -= origin
        env.segment[:] = segment
        env.index[:] = self.origin_index[segment]
        env.origin[:] = origin
        env.steps[:] = 0
        env.t.speed[:] = self.library.speeds[segment]
        mask = torch.ones(env.worlds, dtype=torch.bool, device=env.torch_device)
        env._write_state(mask, q, self.v0[segment], self.phase0[segment])

    def run(self, controller: Callable[[torch.Tensor], torch.Tensor] | None, *, record: bool = True) -> dict:
        """Integrate all worlds; ``controller`` maps observations to actions, or ``None`` for linear weights.

        Returns:
            Control-rate traces as NumPy arrays: ``state`` and ``velocity``
            [steps + 1, worlds, 9] in the measured frame, ``grf_n`` interval-mean
            right/left fx, fz [steps, worlds, 4], ``load_nm``, ``effort``,
            ``power_positive_w``, ``power_negative_w`` [steps, worlds, 6], the
            commanded ``impedance`` [steps, worlds, 18] in neural mode, and
            ``status`` [worlds] with ``failed_step``.
        """
        env, dev = self.env, self.env.torch_device
        self.reset()
        steps = self.max_steps
        w = env.worlds
        horizon = torch.as_tensor(np.tile(self.steps, self.candidates), device=dev)
        failed_step = torch.full((w,), -1, dtype=torch.long, device=dev)
        traces = {
            "state": [env.t.q.clone() + _pad(env.origin)],
            "velocity": [env.t.v.clone()],
            "grf_n": [],
            "effort": [],
            "power_positive_w": [],
            "power_negative_w": [],
            "phase_rad": [env.t.phase.clone()],
            "torque_nm": [],
            "sense_bw": [],
        }
        if controller is not None:
            traces["impedance"] = []
        with torch.no_grad():
            for k in range(steps):
                # Worlds past their own horizon stop integrating.
                stopped = (horizon <= k) & (env.t.status == 0)
                env.t.status[stopped] = 1
                if controller is not None:
                    action = controller(env.observe())
                    env.set_impedance(action)
                    traces["impedance"].append(
                        torch.cat((env.t.equilibrium, env.t.stiffness, env.t.damping), dim=1).clone()
                    )
                env.integrate()
                newly = (env.t.status > 1) & (failed_step < 0)
                failed_step[newly] = k
                env.index += 1
                if record:
                    traces["state"].append(env.t.q.clone() + _pad(env.origin))
                    traces["velocity"].append(env.t.v.clone())
                    traces["phase_rad"].append(env.t.phase.clone())
                    count = env.t.substeps.clamp(min=1).to(torch.float64)[:, None]
                    traces["grf_n"].append(env.t.grf_sum / count)
                    traces["effort"].append(env.t.effort / count)
                    traces["power_positive_w"].append(env.t.power_positive / count)
                    traces["power_negative_w"].append(env.t.power_negative / count)
                    traces["torque_nm"].append(env.t.torque.clone())
                    traces["sense_bw"].append(env.t.sense.clone())
        result = {name: torch.stack(values).cpu().numpy() for name, values in traces.items() if values}
        result["status"] = env.t.status.cpu().numpy()
        result["failed_step"] = failed_step.cpu().numpy()
        result["steps"] = np.tile(self.steps, self.candidates)
        result["dt_s"] = env.config.control_dt_s
        return result


def _pad(origin: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(origin, (0, 7))


def _contacts(member: dict, origin_s: float, duration_s: float) -> list[dict]:
    result = []
    for contact in member["contacts"]:
        start = contact["start_s"] - member["start_s"] - origin_s
        end = contact["end_s"] - member["start_s"] - origin_s
        if start > 0 and end < duration_s:
            result.append({"side": 0 if contact["side"] == "right" else 1, "start_s": start, "end_s": end})
    return result


def score_world(
    traces: dict,
    world: int,
    library: ReferenceLibrary,
    segment: int,
    start: Start,
    force_observation: ForceObservation | None = None,
    threshold_n: float = 50.0,
    torque_cap: np.ndarray | None = None,
) -> dict:
    """Score one world's free prediction against its segment, offline.

    The loss mirrors :func:`..hogan.identify.score` for two feet: hip position
    and angle tracking, per-foot GRF, per-contact peak force, impulse, and
    contact duration, effort, unobservable force vibration, and an unfinished
    window penalty.
    """
    force_observation = force_observation or ForceObservation()
    reference = library.references[segment]
    member = library.members[segment]
    dt = float(traces["dt_s"])
    steps = int(traces["steps"][world])
    failed = int(traces["failed_step"][world])
    completed = steps if failed < 0 else failed
    duration = steps * dt
    time = np.arange(completed + 1) * dt
    measured_time = reference["time_s"] - start.origin_s
    observed = (measured_time > 0) & (measured_time <= time[-1] + 1e-12)
    state = traces["state"][: completed + 1, world]
    tracking = np.zeros(9)
    if observed.any():
        simulated = np.column_stack([np.interp(measured_time[observed], time, state[:, c]) for c in range(9)])
        tracking = np.sqrt(np.mean((simulated - reference["q"][observed]) ** 2, axis=0))
    grf = traces["grf_n"][:completed, world].reshape(-1, 2, 2)
    force_time = reference["grf_time_s"] - start.origin_s
    interval = (np.arange(completed) + 0.5) * dt
    measured = np.stack(
        [
            np.column_stack([np.interp(interval, force_time, reference["grf_side_n"][s, :, a]) for a in range(2)])
            for s in range(2)
        ],
        axis=1,
    )
    if force_observation is not None and completed > 8:
        observed_grf = np.stack([force_observation.apply(grf[:, s], dt) for s in range(2)], axis=1)
    else:
        observed_grf = grf
    force_rmse = np.sqrt(np.mean((observed_grf - measured) ** 2, axis=0)).ravel() if completed else np.zeros(4)
    vibration = np.sqrt(np.mean((grf - observed_grf) ** 2, axis=0)).ravel() if completed else np.zeros(4)
    peaks, impulses, durations, sides = [], [], [], []
    for contact in _contacts(member, start.origin_s, duration):
        side = contact["side"]
        window = (interval >= contact["start_s"] - 0.1) & (interval <= contact["end_s"] + 0.1)
        fine = (force_time >= contact["start_s"] - 0.1) & (force_time <= contact["end_s"] + 0.1)
        target = reference["grf_side_n"][side][fine]
        sim = observed_grf[window, side]
        if not len(sim):
            continue
        sides.append(side)
        peaks.append(float(sim[:, 1].max() - target[:, 1].max()))
        target_impulse = np.trapezoid(target, force_time[fine], axis=0)
        impulses.append(sim.sum(0) * dt - target_impulse)
        target_duration = float(np.count_nonzero(target[:, 1] > threshold_n) * np.median(np.diff(force_time)))
        durations.append(float(np.count_nonzero(sim[:, 1] > threshold_n) * dt) - target_duration)
    cap = np.ones(6) if torque_cap is None else torque_cap
    effort = float(np.mean(traces["effort"][:completed, world] / cap**2)) if completed else 0.0
    loss = (
        float(np.mean((tracking[:2] / 0.02) ** 2))
        + float(np.mean((tracking[2:] / 0.05) ** 2))
        + float(np.mean((force_rmse / 100.0) ** 2))
        + (float(np.mean(np.square(peaks))) / 100.0**2 if peaks else 0.0)
        + (float(np.mean(np.square(impulses))) / 20.0**2 if impulses else 0.0)
        + (float(np.mean(np.square(durations))) / 0.02**2 if durations else 0.0)
        + 0.01 * effort
        + float(np.mean((vibration / 100.0) ** 2))
    )
    # The single-leg runner's loss restricted to the right leg and right foot, for its stance windows.
    right = [peak for peak, contact in zip(peaks, sides, strict=True) if contact == 0]
    right_impulse = [value for value, contact in zip(impulses, sides, strict=True) if contact == 0]
    right_duration = [value for value, contact in zip(durations, sides, strict=True) if contact == 0]
    single_leg = (
        float(np.mean((tracking[:2] / 0.02) ** 2))
        + float(np.mean((tracking[2:6] / 0.05) ** 2))
        + float(np.mean((force_rmse[:2] / 100.0) ** 2))
        + (float(np.mean(np.square(right))) / 100.0**2 if right else 0.0)
        + (float(np.mean(np.square(right_impulse))) / 20.0**2 if right_impulse else 0.0)
        + (float(np.mean(np.square(right_duration))) / 0.02**2 if right_duration else 0.0)
        + 0.01 * (float(np.mean(traces["effort"][:completed, world, :3] / cap[:3] ** 2)) if completed else 0.0)
        + float(np.mean((vibration[:2] / 100.0) ** 2))
    )
    if failed >= 0:
        loss += 1000.0 * (2.0 - completed / max(steps, 1))
        single_leg += 1000.0 * (2.0 - completed / max(steps, 1))
    positive = traces["power_positive_w"][:completed, world].sum(0) * dt
    negative = traces["power_negative_w"][:completed, world].sum(0) * dt
    return {
        "id": member["id"],
        "split": member["split"],
        "loss": loss,
        "single_leg_loss": single_leg,
        "status": "completed" if failed < 0 else "failed",
        "failure": None if failed < 0 else FAILURES.get(int(traces["status"][world]), "unknown"),
        "failure_time_s": None if failed < 0 else completed * dt,
        "duration_s": duration,
        "tracking_rmse": tracking.tolist(),
        "grf_rmse_n": force_rmse.tolist(),
        "grf_vibration_rmse_n": vibration.tolist(),
        "peak_fz_error_n": peaks,
        "impulse_error_ns": [list(map(float, i)) for i in impulses],
        "contact_duration_error_s": durations,
        "effort": effort,
        "joint_positive_work_j": positive.tolist(),
        "joint_negative_work_j": negative.tolist(),
        "contacts_scored": len(peaks),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-segment scores."""
    if not rows:
        return {}
    tracking = np.array([row["tracking_rmse"] for row in rows])
    force = np.array([row["grf_rmse_n"] for row in rows])
    peaks = [p for row in rows for p in row["peak_fz_error_n"]]
    durations = [d for row in rows for d in row["contact_duration_error_s"]]
    return {
        "segments": len(rows),
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "mean_single_leg_loss": float(np.mean([row["single_leg_loss"] for row in rows])),
        "failed": sum(row["status"] != "completed" for row in rows),
        "root_rmse_mm": (1000 * tracking[:, :2].mean(0)).tolist(),
        "pelvis_rmse_deg": float(np.degrees(tracking[:, 2].mean())),
        "joint_rmse_deg": np.degrees(tracking[:, 3:].mean(0)).tolist(),
        "grf_rmse_n": force.mean(0).tolist(),
        "mean_abs_peak_fz_error_n": float(np.mean(np.abs(peaks))) if peaks else None,
        "mean_contact_duration_error_ms": float(1000 * np.mean(durations)) if durations else None,
    }
