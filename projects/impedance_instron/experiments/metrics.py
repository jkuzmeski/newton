# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure frozen-rollout changes without rewarding truncated trajectories."""

from __future__ import annotations

import numpy as np

from ..cartesian.fit import FitConfig, _refinement


def _integral(values, times):
    """Integrate only observed time support with trapezoidal quadrature."""
    return np.trapezoid(values, x=times, axis=0) if len(times) > 1 else np.zeros(np.shape(values)[1:])


def motion_series(trace: dict, run: dict) -> tuple[np.ndarray, np.ndarray]:
    """Include the saved terminal state without padding a terminated rollout."""
    time = np.asarray(trace["time_s"])
    state = np.asarray(trace["state"])
    terminal_time = float(run["integrated_duration_s"])
    terminal = np.asarray(run["terminal_state"])
    if np.isfinite(terminal).all() and (not len(time) or terminal_time > time[-1] + 1e-12):
        time = np.r_[time, terminal_time]
        state = np.vstack((state.reshape(-1, 5), terminal))
    return time, state


def observations(trace: dict, run: dict, *, ground_height_m: float = 0.0, contact_threshold_n: float = 1.0) -> dict:
    """Report SI motion, force, and work quantities on the actual saved prefix."""
    time = np.asarray(trace["time_s"])
    result = {
        "sample_count": len(time),
        "completed_fraction": run["integrated_duration_s"] / run["requested_duration_s"],
        "full_stance": run["status"] == "completed",
        "contact_threshold_n": contact_threshold_n,
        "slip": {"available": False, "reason": "Per-column stick/slip history is not part of the leg trace."},
    }
    if not len(time):
        return result
    force = np.asarray(trace["grf_n"])
    velocity = np.asarray(trace["velocity"])
    power = np.sum(trace["hip_force_n"] * velocity[:, :2], axis=1)
    power += np.sum(trace["joint_torque_nm"] * velocity[:, 3:5], axis=1)
    contact = force[:, 1] > contact_threshold_n
    indices = np.flatnonzero(contact)
    ankle = np.asarray(trace["joints_m"])[:, 2]
    cop = np.full(len(time), np.nan)
    cop[contact] = (
        ankle[contact, 0]
        + (trace["ankle_contact_moment_nm"][contact] + (ground_height_m - ankle[contact, 1]) * force[contact, 0])
        / force[contact, 1]
    )
    peak = int(np.argmax(force[:, 1]))
    result.update(
        force_support_s=[float(time[0]), float(time[-1])],
        grf_impulse_ns=_integral(force, time).tolist(),
        peak_vertical_grf_n=float(force[peak, 1]),
        peak_vertical_grf_time_s=float(time[peak]),
        peak_horizontal_grf_abs_n=float(np.max(np.abs(force[:, 0]))),
        minimum_hip_height_m=float(np.min(trace["state"][:, 1])),
        foot_pitch_range_rad=[
            float(np.min(np.sum(trace["state"][:, 2:], axis=1))),
            float(np.max(np.sum(trace["state"][:, 2:], axis=1))),
        ],
        maximum_hip_force_n=float(np.max(np.linalg.norm(trace["hip_force_n"], axis=1))),
        maximum_joint_torque_abs_nm=np.max(np.abs(trace["joint_torque_nm"]), axis=0).tolist(),
        actuator_work_signed_j=float(_integral(power, time)),
        actuator_work_positive_j=float(_integral(np.maximum(power, 0), time)),
        actuator_work_negative_j=float(_integral(np.minimum(power, 0), time)),
        first_contact_time_s=float(time[indices[0]]) if len(indices) else None,
        last_contact_time_s=float(time[indices[-1]]) if len(indices) else None,
        contact_sample_duration_s=float(np.count_nonzero(contact) * run["actual_dt_s"]),
        contact_loss_times_s=time[1:][contact[:-1] & ~contact[1:]].tolist(),
        cop_x_range_m=[float(np.nanmin(cop)), float(np.nanmax(cop))] if len(indices) else None,
        maximum_driven_compression_fraction=float(np.max(trace["driven_compression_fraction"])),
        maximum_passive_compression_fraction=float(np.max(trace["passive_compression_fraction"])),
        passive_cap_steps=int(np.count_nonzero(trace["passive_cap_column_count"])),
        qualification="Impulse and work cover saved force support only; no force extrapolation to the terminal state.",
    )
    return result


def _difference(ta, a, tb, b) -> dict:
    """Compare finite shared support, never extrapolating missing motion or force."""
    if not len(ta) or not len(tb):
        return {"available": False, "reason": "No shared samples"}
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if hi < lo:
        return {"available": False, "reason": "No shared time support"}
    grid = np.unique(np.r_[ta[(ta >= lo) & (ta <= hi)], tb[(tb >= lo) & (tb <= hi)]])
    aa, bb = np.asarray(a).reshape(len(ta), -1), np.asarray(b).reshape(len(tb), -1)
    delta = np.column_stack([np.interp(grid, ta, aa[:, i]) - np.interp(grid, tb, bb[:, i]) for i in range(aa.shape[1])])
    rms = np.sqrt(_integral(delta**2, grid) / (hi - lo)) if hi > lo else np.abs(delta[0])
    return {
        "available": True,
        "support_s": [float(lo), float(hi)],
        "rms": rms.tolist(),
        "maximum_abs": np.max(np.abs(delta), axis=0).tolist(),
    }


def compare(trace: dict, run: dict, baseline: dict, baseline_run: dict) -> dict:
    """Return baseline-relative changes with explicit censoring and units."""
    t, q = motion_series(trace, run)
    bt, bq = motion_series(baseline, baseline_run)
    return {
        "full_stance_comparison": run["status"] == baseline_run["status"] == "completed",
        "motion_units": ["m", "m", "rad", "rad", "rad"],
        "motion": _difference(t, q, bt, bq),
        "force_units": ["N", "N"],
        "grf": _difference(trace["time_s"], trace["grf_n"], baseline["time_s"], baseline["grf_n"]),
        "qualification": "Prefix changes are not full-stance tracking scores and cannot rank terminated cases as better fits.",
    }


def refinement(coarse, coarse_run, fine, fine_run, *, settings: FitConfig, event_tolerance_s: float = 0.001) -> dict:
    """Check complete trajectories or separately qualify terminated-event agreement."""
    if coarse_run["status"] == fine_run["status"] == "completed":
        return _refinement(coarse, coarse_run, fine, fine_run, coarse_run["requested_duration_s"], settings)
    shared = compare(coarse, coarse_run, fine, fine_run)
    cf, ff = coarse_run.get("failure"), fine_run.get("failure")
    same = bool(cf and ff and sorted(cf["reasons"]) == sorted(ff["reasons"]))
    event_delta = abs(cf["time_s"] - ff["time_s"]) if cf and ff else None
    motion, force = shared["motion"], shared["grf"]
    prefix_passed = False
    if motion["available"] and force["available"]:
        m, f = np.asarray(motion["maximum_abs"]), np.asarray(force["maximum_abs"])
        prefix_passed = bool(
            np.linalg.norm(m[:2]) <= settings.refinement_position_m
            and np.max(m[2:]) <= settings.refinement_angle_rad
            and np.linalg.norm(f) <= settings.refinement_force_n
        )
    numerical = any(
        "Nonfinite" in reason or "Numerical speed" in reason
        for failure in (cf, ff)
        if failure
        for reason in failure["reasons"]
    )
    event_agreement = bool(same and event_delta <= event_tolerance_s)
    return {
        "performed": True,
        "complete": False,
        "passed": False,
        "terminated_event_agreement": event_agreement,
        "prefix_within_tolerances": prefix_passed,
        "screen_event_supported": event_agreement and prefix_passed and not numerical,
        "event_time_difference_s": event_delta,
        "event_time_tolerance_s": event_tolerance_s,
        "shared_prefix": shared,
        "qualification": "Consistent screen termination is not a physiological fall or complete-stance acceptance.",
    }
