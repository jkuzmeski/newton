# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actuator power, work, and sensitivity analysis for Cartesian leg rollouts.

Computes four-channel and combined mechanical power and work histories from saved
simulation traces without state extrapolation or physiological claims.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validate_trace_and_run(trace: Any, run: Any) -> tuple[bool, str | None]:
    """Validate that trace and run structures exist, match dimensions, and are finite."""
    if trace is None or run is None:
        return False, "Trace or run dictionary is None"
    if not isinstance(run, dict):
        return False, "run summary must be a dict"
    if not isinstance(trace, (dict, np.lib.npyio.NpzFile)):
        return False, "trace must be a dict or NpzFile"

    for key in ("status", "integrated_duration_s", "actual_dt_s"):
        if key not in run:
            return False, f"Missing required run summary key: '{key}'"

    for key in ("time_s", "velocity", "joint_torque_nm", "hip_force_n"):
        if key not in trace:
            return False, f"Missing required trace array: '{key}'"

    try:
        dur = float(run["integrated_duration_s"])
        dt = float(run["actual_dt_s"])
    except (TypeError, ValueError):
        return False, "Scalar run quantities cannot be converted to float"

    if not math.isfinite(dur) or dur < 0.0:
        return False, f"run['integrated_duration_s'] must be finite and >= 0, got {dur}"
    if not math.isfinite(dt) or dt <= 0.0:
        return False, f"run['actual_dt_s'] must be finite and > 0, got {dt}"

    try:
        time = np.asarray(trace["time_s"], dtype=np.float64)
        velocity = np.asarray(trace["velocity"], dtype=np.float64)
        torque = np.asarray(trace["joint_torque_nm"], dtype=np.float64)
        hip_force = np.asarray(trace["hip_force_n"], dtype=np.float64)
    except (TypeError, ValueError) as err:
        return False, f"Trace arrays cannot be converted to float64: {err}"

    if time.ndim != 1 or len(time) < 2:
        return False, f"trace['time_s'] must be 1-D with at least 2 points, got shape {time.shape}"

    if not np.all(np.isfinite(time)):
        return False, "Nonfinite values detected in trace['time_s']"

    if np.any(np.diff(time) <= 0.0):
        return False, "trace['time_s'] values must be strictly increasing"

    count = len(time)
    if velocity.ndim != 2 or velocity.shape[0] != count or velocity.shape[1] < 5:
        return False, f"trace['velocity'] shape must be ({count}, >=5), got {velocity.shape}"

    if torque.ndim != 2 or torque.shape[0] != count or torque.shape[1] < 2:
        return False, f"trace['joint_torque_nm'] shape must be ({count}, >=2), got {torque.shape}"

    if hip_force.ndim != 2 or hip_force.shape[0] != count or hip_force.shape[1] < 2:
        return False, f"trace['hip_force_n'] shape must be ({count}, >=2), got {hip_force.shape}"

    if not (
        np.all(np.isfinite(velocity[:, :5]))
        and np.all(np.isfinite(torque[:, :2]))
        and np.all(np.isfinite(hip_force[:, :2]))
    ):
        return False, "Nonfinite values detected in trace velocity, torque, or hip force"

    lo = float(time[0])
    hi = min(float(time[-1]), dur)
    if hi <= lo:
        return False, f"No positive-duration integrated prefix: time[0]={lo}, effective hi={hi}"

    return True, None


def _prepare_signals(trace: dict | np.lib.npyio.NpzFile, run: dict) -> tuple[dict[str, np.ndarray], float, float]:
    """Extract and prepare 1-D force, torque, and power signals on the integrated prefix.

    Calculates power directly from raw trace kinematics before any support clipping.
    """
    time = np.asarray(trace["time_s"], dtype=np.float64)
    velocity = np.asarray(trace["velocity"], dtype=np.float64)
    torque = np.asarray(trace["joint_torque_nm"], dtype=np.float64)
    hip_force = np.asarray(trace["hip_force_n"], dtype=np.float64)

    dur = float(run["integrated_duration_s"])
    if len(time) > 0 and time[-1] > dur + 1e-12:
        mask = time <= dur + 1e-12
        time = time[mask]
        velocity = velocity[mask]
        torque = torque[mask]
        hip_force = hip_force[mask]

    knee_power = torque[:, 0] * velocity[:, 3]
    ankle_power = torque[:, 1] * velocity[:, 4]
    hip_x_power = hip_force[:, 0] * velocity[:, 0]
    hip_z_power = hip_force[:, 1] * velocity[:, 1]
    hip_power = hip_x_power + hip_z_power
    total_power = knee_power + ankle_power + hip_x_power + hip_z_power

    hip_force_mag = np.linalg.norm(hip_force[:, :2], axis=1)

    signals = {
        "time_s": time,
        "knee_torque_nm": torque[:, 0],
        "ankle_torque_nm": torque[:, 1],
        "hip_force_n": hip_force_mag,
        "hip_x_force_n": hip_force[:, 0],
        "hip_z_force_n": hip_force[:, 1],
        "knee_power_w": knee_power,
        "ankle_power_w": ankle_power,
        "hip_power_w": hip_power,
        "hip_x_power_w": hip_x_power,
        "hip_z_power_w": hip_z_power,
        "total_power_w": total_power,
    }
    return signals, float(time[0]), float(time[-1])


def _slice_signals(signals: dict[str, np.ndarray], lo: float, hi: float) -> dict[str, np.ndarray] | None:
    """Slice prepared signals to [lo, hi], interpolating power and load directly at boundaries.

    Directly interpolating prepared power preserves the exact piecewise-linear power
    quadrature model without introducing nonlinear boundary products.
    """
    time = signals["time_s"]
    if len(time) == 0 or hi <= lo:
        return None

    lo = max(lo, float(time[0]))
    hi = min(hi, float(time[-1]))
    if hi <= lo:
        return None

    mask = (time >= lo - 1e-12) & (time <= hi + 1e-12)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return None

    sub_time = time[indices].copy()
    keys = [k for k in signals if k != "time_s"]
    sub = {k: signals[k][indices].copy() for k in keys}

    # Interpolate exact lower boundary if needed
    if sub_time[0] > lo + 1e-12:
        i1 = indices[0]
        i0 = i1 - 1
        if i0 >= 0:
            dt = time[i1] - time[i0]
            frac = (lo - time[i0]) / dt if dt > 0 else 0.0
            sub_time = np.r_[lo, sub_time]
            for k in keys:
                v0 = signals[k][i0]
                v1 = signals[k][i1]
                v_lo = v0 + frac * (v1 - v0)
                sub[k] = np.r_[v_lo, sub[k]]

    # Interpolate exact upper boundary if needed
    if sub_time[-1] < hi - 1e-12:
        i0 = indices[-1]
        i1 = i0 + 1
        if i1 < len(time):
            dt = time[i1] - time[i0]
            frac = (hi - time[i0]) / dt if dt > 0 else 0.0
            sub_time = np.r_[sub_time, hi]
            for k in keys:
                v0 = signals[k][i0]
                v1 = signals[k][i1]
                v_hi = v0 + frac * (v1 - v0)
                sub[k] = np.r_[sub[k], v_hi]

    sub["time_s"] = sub_time
    return sub


def _integrate_piecewise_linear(
    power: np.ndarray, time: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate piecewise-linear power splitting zero crossings consistently.

    Guarantees positive_work_j - absorbed_work_j == net_work_j == integral(P dt)
    to machine precision.
    """
    n = len(time)
    cum_pos = np.zeros(n, dtype=np.float64)
    cum_abs = np.zeros(n, dtype=np.float64)
    cum_net = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return 0.0, 0.0, 0.0, cum_pos, cum_abs, cum_net

    dt = np.diff(time)
    p0 = power[:-1]
    p1 = power[1:]

    int_pos = np.zeros(n - 1, dtype=np.float64)
    int_abs = np.zeros(n - 1, dtype=np.float64)

    # Case 1: Both non-negative
    both_pos = (p0 >= 0.0) & (p1 >= 0.0)
    int_pos[both_pos] = 0.5 * (p0[both_pos] + p1[both_pos]) * dt[both_pos]

    # Case 2: Both non-positive
    both_neg = (p0 <= 0.0) & (p1 <= 0.0)
    int_abs[both_neg] = 0.5 * (-p0[both_neg] - p1[both_neg]) * dt[both_neg]

    # Case 3: Positive to negative crossing
    pos_neg = (p0 > 0.0) & (p1 < 0.0)
    if np.any(pos_neg):
        dt_sub = dt[pos_neg]
        y0 = p0[pos_neg]
        y1 = p1[pos_neg]
        t_frac = y0 / (y0 - y1)
        int_pos[pos_neg] = 0.5 * y0 * (t_frac * dt_sub)
        int_abs[pos_neg] = 0.5 * (-y1) * ((1.0 - t_frac) * dt_sub)

    # Case 4: Negative to positive crossing
    neg_pos = (p0 < 0.0) & (p1 > 0.0)
    if np.any(neg_pos):
        dt_sub = dt[neg_pos]
        y0 = p0[neg_pos]
        y1 = p1[neg_pos]
        t_frac = -y0 / (y1 - y0)
        int_abs[neg_pos] = 0.5 * (-y0) * (t_frac * dt_sub)
        int_pos[neg_pos] = 0.5 * y1 * ((1.0 - t_frac) * dt_sub)

    cum_pos[1:] = np.cumsum(int_pos)
    cum_abs[1:] = np.cumsum(int_abs)
    cum_net[1:] = cum_pos[1:] - cum_abs[1:]

    return float(cum_pos[-1]), float(cum_abs[-1]), float(cum_net[-1]), cum_pos, cum_abs, cum_net


def _time_weighted_rms(signal: np.ndarray, time: np.ndarray) -> float:
    """Compute time-weighted root-mean-square: sqrt(1/T * integral(x^2 dt))."""
    if len(time) <= 1:
        return float(np.abs(signal[0])) if len(signal) > 0 else 0.0
    duration = float(time[-1] - time[0])
    if duration <= 0.0:
        return float(np.abs(signal[0]))
    integral_sq = np.trapezoid(signal**2, x=time)
    return float(np.sqrt(max(0.0, integral_sq / duration)))


def _compute_channel_from_signals(
    load: np.ndarray,
    power: np.ndarray,
    time: np.ndarray,
    *,
    is_rotational: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Compute metrics and cumulative work curves for a 1-D actuator channel."""
    pos_w, abs_w, net_w, cum_pos, cum_abs, cum_net = _integrate_piecewise_linear(power, time)
    rms_val = _time_weighted_rms(load, time)

    metrics: dict[str, float] = {
        "peak_positive_power_w": float(np.max(np.maximum(power, 0.0))),
        "peak_absorbed_power_w": float(np.max(np.maximum(-power, 0.0))),
        "positive_work_j": pos_w,
        "absorbed_work_j": abs_w,
        "net_work_j": net_w,
    }

    if is_rotational:
        metrics.update(
            {
                "peak_abs_torque_nm": float(np.max(np.abs(load))),
                "rms_torque_nm": rms_val,
                "min_torque_nm": float(np.min(load)),
                "max_torque_nm": float(np.max(load)),
            }
        )
    else:
        metrics.update(
            {
                "peak_force_n": float(np.max(np.abs(load))),
                "rms_force_n": rms_val,
                "min_force_n": float(np.min(load)),
                "max_force_n": float(np.max(load)),
            }
        )

    return metrics, cum_pos, cum_abs, cum_net


def _compute_from_prepared(
    signals: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Compute channel metrics and full-resolution curves from prepared signals."""
    time = signals["time_s"]

    # Knee and ankle
    k_met, k_pos, k_abs, k_net = _compute_channel_from_signals(
        signals["knee_torque_nm"], signals["knee_power_w"], time, is_rotational=True
    )
    a_met, a_pos, a_abs, a_net = _compute_channel_from_signals(
        signals["ankle_torque_nm"], signals["ankle_power_w"], time, is_rotational=True
    )

    # Hip translational actuators
    hx_met, hx_pos, hx_abs, hx_net = _compute_channel_from_signals(
        signals["hip_x_force_n"], signals["hip_x_power_w"], time, is_rotational=False
    )
    hz_met, hz_pos, hz_abs, hz_net = _compute_channel_from_signals(
        signals["hip_z_force_n"], signals["hip_z_power_w"], time, is_rotational=False
    )

    # Combined hip point
    hip_pow = signals["hip_power_w"]
    h_pos_w, h_abs_w, h_net_w, hip_pos, hip_abs, hip_net = _integrate_piecewise_linear(hip_pow, time)
    hip_f_rms = _time_weighted_rms(signals["hip_force_n"], time)
    hip_met = {
        "peak_force_n": float(np.max(signals["hip_force_n"])),
        "rms_force_n": hip_f_rms,
        "min_force_n": float(np.min(signals["hip_force_n"])),
        "max_force_n": float(np.max(signals["hip_force_n"])),
        "peak_positive_power_w": float(np.max(np.maximum(hip_pow, 0.0))),
        "peak_absorbed_power_w": float(np.max(np.maximum(-hip_pow, 0.0))),
        "positive_work_j": h_pos_w,
        "absorbed_work_j": h_abs_w,
        "net_work_j": h_net_w,
    }

    # Total actuator work: sum across the four physical actuators
    # Summing work per actuator avoids sign cancellation between actuators.
    tot_pos = (
        k_met["positive_work_j"] + a_met["positive_work_j"] + hx_met["positive_work_j"] + hz_met["positive_work_j"]
    )
    tot_abs = (
        k_met["absorbed_work_j"] + a_met["absorbed_work_j"] + hx_met["absorbed_work_j"] + hz_met["absorbed_work_j"]
    )
    tot_net = k_met["net_work_j"] + a_met["net_work_j"] + hx_met["net_work_j"] + hz_met["net_work_j"]

    tot_pos_curve = k_pos + a_pos + hx_pos + hz_pos
    tot_abs_curve = k_abs + a_abs + hx_abs + hz_abs
    tot_net_curve = k_net + a_net + hx_net + hz_net

    tot_pow = signals["total_power_w"]
    tot_met = {
        "positive_work_j": float(tot_pos),
        "absorbed_work_j": float(tot_abs),
        "net_work_j": float(tot_net),
        "peak_positive_power_w": float(np.max(np.maximum(tot_pow, 0.0))),
        "peak_absorbed_power_w": float(np.max(np.maximum(-tot_pow, 0.0))),
    }

    channels: dict[str, dict[str, float]] = {
        "knee": k_met,
        "ankle": a_met,
        "hip": hip_met,
        "hip_x": hx_met,
        "hip_z": hz_met,
        "total": tot_met,
    }

    curves: dict[str, np.ndarray] = {
        "time_s": time,
        "knee_torque_nm": signals["knee_torque_nm"],
        "ankle_torque_nm": signals["ankle_torque_nm"],
        "knee_power_w": signals["knee_power_w"],
        "ankle_power_w": signals["ankle_power_w"],
        "hip_power_w": hip_pow,
        "total_power_w": tot_pow,
        "knee_positive_work_j": k_pos,
        "knee_absorbed_work_j": k_abs,
        "knee_net_work_j": k_net,
        "ankle_positive_work_j": a_pos,
        "ankle_absorbed_work_j": a_abs,
        "ankle_net_work_j": a_net,
        "hip_positive_work_j": hip_pos,
        "hip_absorbed_work_j": hip_abs,
        "hip_net_work_j": hip_net,
        "total_positive_work_j": tot_pos_curve,
        "total_absorbed_work_j": tot_abs_curve,
        "total_net_work_j": tot_net_curve,
    }

    return channels, curves


def _difference_channels(
    chans_a: dict[str, dict[str, float]],
    chans_b: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    """Compute nested delta A - B for all channels and shared metric keys."""
    deltas: dict[str, dict[str, float | None]] = {}
    for ch_name, ch_a in chans_a.items():
        if ch_name not in chans_b:
            continue
        ch_b = chans_b[ch_name]
        ch_delta: dict[str, float | None] = {}
        for k, val_a in ch_a.items():
            if k in ch_b and val_a is not None and ch_b[k] is not None:
                ch_delta[k] = float(val_a - ch_b[k])
            else:
                ch_delta[k] = None
        deltas[ch_name] = ch_delta
    return deltas


def _percent_change(delta: float | None, baseline: float | None) -> float | None:
    """Normalize a signed change by baseline magnitude, omitting near-zero denominators."""
    if (
        delta is None
        or baseline is None
        or not np.isfinite(delta)
        or not np.isfinite(baseline)
        or abs(baseline) <= 1e-9
    ):
        return None
    return float(100.0 * delta / abs(baseline))


def _compute_baseline_delta(
    signals_cond: dict[str, np.ndarray],
    run_cond: dict,
    signals_base: dict[str, np.ndarray],
    run_base: dict,
) -> dict[str, Any]:
    """Compare native condition minus current baseline primary replay on common saved support."""
    lo = max(float(signals_cond["time_s"][0]), float(signals_base["time_s"][0]))
    hi = min(float(signals_cond["time_s"][-1]), float(signals_base["time_s"][-1]))

    if hi <= lo:
        return {"available": False, "reason": "No common saved time support between condition and baseline"}

    sub_cond = _slice_signals(signals_cond, lo, hi)
    sub_base = _slice_signals(signals_base, lo, hi)

    if sub_cond is None or sub_base is None:
        return {"available": False, "reason": "Unable to extract common saved time support"}

    chans_cond, _ = _compute_from_prepared(sub_cond)
    chans_base, _ = _compute_from_prepared(sub_base)

    full_stance_comp = (run_cond["status"] == "completed") and (run_base["status"] == "completed")
    deltas = _difference_channels(chans_cond, chans_base)

    warnings: list[str] = []
    if not full_stance_comp:
        warnings.append(
            "Rollouts truncated before completion; delta covers common observed prefix only. "
            "Shorter truncated work is not ranked as an efficiency benefit."
        )

    return {
        "available": True,
        "support_s": [float(lo), float(hi)],
        "full_stance_comparison": full_stance_comp,
        "channels": deltas,
        "percent_change_channels": {
            channel: {key: _percent_change(delta, chans_base[channel][key]) for key, delta in metrics.items()}
            for channel, metrics in deltas.items()
        },
        "percent_change_definition": "100 * (condition - baseline) / abs(baseline), on the same shared support",
        "warnings": warnings,
        "qualification": (
            "Prefix comparisons cover observed force support only. Truncated trajectories cannot rank "
            "as improvements over full-stance baselines."
        ),
    }


def _compute_refinement(
    signals_nat: dict[str, np.ndarray],
    run_nat: dict,
    signals_ref: dict[str, np.ndarray],
    run_ref: dict,
) -> dict[str, Any]:
    """Compare native vs half-step resolution on common saved support."""
    lo = max(float(signals_nat["time_s"][0]), float(signals_ref["time_s"][0]))
    hi = min(float(signals_nat["time_s"][-1]), float(signals_ref["time_s"][-1]))

    if hi <= lo:
        return {"performed": True, "available": False, "reason": "No common saved time support for refinement"}

    sub_nat = _slice_signals(signals_nat, lo, hi)
    sub_ref = _slice_signals(signals_ref, lo, hi)

    if sub_nat is None or sub_ref is None:
        return {"performed": True, "available": False, "reason": "Unable to extract common support for refinement"}

    chans_nat, _ = _compute_from_prepared(sub_nat)
    chans_ref, _ = _compute_from_prepared(sub_ref)

    full_stance_comp = (run_nat["status"] == "completed") and (run_ref["status"] == "completed")

    channels_refinement: dict[str, dict[str, dict[str, float | None]]] = {}
    for ch_name, ch_nat in chans_nat.items():
        if ch_name not in chans_ref:
            continue
        ch_ref = chans_ref[ch_name]
        ch_dict: dict[str, dict[str, float | None]] = {}
        for k, val_nat in ch_nat.items():
            val_ref = ch_ref.get(k)
            if val_nat is not None and val_ref is not None:
                change = float(val_ref - val_nat)
                ch_dict[k] = {
                    "native": float(val_nat),
                    "half_step": float(val_ref),
                    "change": change,
                    "absolute_change": float(abs(change)),
                }
            else:
                ch_dict[k] = {
                    "native": val_nat,
                    "half_step": val_ref,
                    "change": None,
                    "absolute_change": None,
                }
        channels_refinement[ch_name] = ch_dict

    return {
        "performed": True,
        "available": True,
        "support_s": [float(lo), float(hi)],
        "full_stance_comparison": full_stance_comp,
        "channels": channels_refinement,
        "qualification": (
            "Refinement difference indicates numerical sensitivity between base and halved timesteps on shared "
            "support. It is not a formal error bound, confidence interval, or physiological qualification."
        ),
    }


def _compute_effect_refinement(
    signals_cn: dict[str, np.ndarray],
    run_cn: dict,
    signals_cr: dict[str, np.ndarray],
    run_cr: dict,
    signals_bn: dict[str, np.ndarray],
    run_bn: dict,
    signals_br: dict[str, np.ndarray],
    run_br: dict,
) -> dict[str, Any]:
    """Compare native baseline-relative effect vs half-step baseline-relative effect across all 4 traces."""
    all_signals = [signals_cn, signals_cr, signals_bn, signals_br]
    lo = max(float(s["time_s"][0]) for s in all_signals)
    hi = min(float(s["time_s"][-1]) for s in all_signals)

    if hi <= lo:
        return {"performed": True, "available": False, "reason": "No common 4-way time support"}

    sub_cn = _slice_signals(signals_cn, lo, hi)
    sub_cr = _slice_signals(signals_cr, lo, hi)
    sub_bn = _slice_signals(signals_bn, lo, hi)
    sub_br = _slice_signals(signals_br, lo, hi)

    if any(s is None for s in (sub_cn, sub_cr, sub_bn, sub_br)):
        return {"performed": True, "available": False, "reason": "Failed slicing 4-way common support"}

    c_cn, _ = _compute_from_prepared(sub_cn)
    c_cr, _ = _compute_from_prepared(sub_cr)
    c_bn, _ = _compute_from_prepared(sub_bn)
    c_br, _ = _compute_from_prepared(sub_br)

    delta_native = _difference_channels(c_cn, c_bn)
    delta_half = _difference_channels(c_cr, c_br)

    channels_effect: dict[str, dict[str, dict[str, Any]]] = {}
    warnings: list[str] = []

    for ch_name, ch_dn in delta_native.items():
        if ch_name not in delta_half:
            continue
        ch_dh = delta_half[ch_name]
        ch_dict: dict[str, dict[str, Any]] = {}
        for k, dn in ch_dn.items():
            dh = ch_dh.get(k)
            if dn is not None and dh is not None:
                drift = float(abs(dh - dn))
                sign_changed = bool((dn > 0.0 and dh < 0.0) or (dn < 0.0 and dh > 0.0))
                ratio = float(abs(dn) / drift) if drift > 1e-12 else None
                ch_dict[k] = {
                    "delta_native": float(dn),
                    "delta_half_step": float(dh),
                    "baseline_native": float(c_bn[ch_name][k]),
                    "percent_change_native": _percent_change(dn, c_bn[ch_name][k]),
                    "drift_abs": drift,
                    "sign_changed": sign_changed,
                    "effect_to_drift_ratio": ratio,
                }
                if sign_changed or (drift > 0.0 and abs(dh) <= drift):
                    warnings.append(
                        f"Channel '{ch_name}' metric '{k}' changes sign or its half-step effect "
                        f"(|{dh:.3g}|) is no larger than the observed timestep change ({drift:.3g})."
                    )
            else:
                ch_dict[k] = {
                    "delta_native": dn,
                    "delta_half_step": dh,
                    "baseline_native": None,
                    "percent_change_native": None,
                    "drift_abs": None,
                    "sign_changed": None,
                    "effect_to_drift_ratio": None,
                }
        channels_effect[ch_name] = ch_dict

    full_stance_comp = all(rn["status"] == "completed" for rn in (run_cn, run_cr, run_bn, run_br))

    return {
        "performed": True,
        "available": True,
        "support_s": [float(lo), float(hi)],
        "full_stance_comparison": full_stance_comp,
        "channels": channels_effect,
        "warnings": warnings,
        "qualification": (
            "Effect refinement checks stability of baseline-relative contrasts across discretization levels. "
            "Small effect-to-drift ratios indicate condition effects may be sensitive to integrator timestep."
        ),
    }


def analyze_actuation(
    trace: Any,
    run: Any,
    *,
    refined_trace: Any = None,
    refined_run: Any = None,
    baseline_trace: Any = None,
    baseline_run: Any = None,
    baseline_refined_trace: Any = None,
    baseline_refined_run: Any = None,
) -> dict[str, Any]:
    """Analyze actuator work, power, and numerical refinement for saved Cartesian leg traces.

    Calculates knee, ankle, hip translational (x, z), combined hip point, and 4-actuator total
    mechanical power and work curves at full saved resolution.

    Args:
        trace: Native simulation trace dict or NpzFile containing time_s, velocity, joint_torque_nm, hip_force_n.
        run: Run summary dict containing status, integrated_duration_s, actual_dt_s.
        refined_trace: Optional half-step simulation trace.
        refined_run: Optional half-step run summary dict.
        baseline_trace: Optional current baseline primary native trace.
        baseline_run: Optional current baseline primary native run summary dict.
        baseline_refined_trace: Optional current baseline primary half-step trace.
        baseline_refined_run: Optional current baseline primary half-step run summary dict.

    Returns:
        Structured analysis dict containing available status, support_s, full_stance flag,
        channel metrics, full-resolution curves, and optional baseline_delta, refinement,
        and effect_refinement diagnostics.
    """
    valid, reason = _validate_trace_and_run(trace, run)
    if not valid:
        return {
            "available": False,
            "reason": reason,
            "support_s": None,
            "full_stance": False,
            "channels": None,
            "curves": None,
            "baseline_delta": None,
            "refinement": None,
            "effect_refinement": None,
            "qualification": "Actuator analysis unavailable due to missing, invalid, or nonfinite trace data.",
        }

    signals_nat, lo_nat, hi_nat = _prepare_signals(trace, run)
    support_s = [lo_nat, hi_nat]
    full_stance = bool(run["status"] == "completed")

    channels, curves = _compute_from_prepared(signals_nat)

    # Baseline comparison (native condition vs native baseline)
    baseline_delta_result = None
    if baseline_trace is not None and baseline_run is not None:
        ok_base, reason_base = _validate_trace_and_run(baseline_trace, baseline_run)
        if ok_base:
            signals_base, _, _ = _prepare_signals(baseline_trace, baseline_run)
            baseline_delta_result = _compute_baseline_delta(signals_nat, run, signals_base, baseline_run)
        else:
            baseline_delta_result = {
                "available": False,
                "reason": f"Baseline trace invalid: {reason_base}",
            }

    # Refinement comparison (native condition vs refined condition)
    refinement_result = None
    if refined_trace is not None and refined_run is not None:
        ok_ref, reason_ref = _validate_trace_and_run(refined_trace, refined_run)
        if ok_ref:
            signals_ref, _, _ = _prepare_signals(refined_trace, refined_run)
            refinement_result = _compute_refinement(signals_nat, run, signals_ref, refined_run)
        else:
            refinement_result = {
                "performed": False,
                "available": False,
                "reason": f"Refined trace invalid: {reason_ref}",
            }

    # Effect refinement comparison (condition effect vs baseline effect across native and refined)
    effect_refinement_result = None
    if (
        refined_trace is not None
        and refined_run is not None
        and baseline_trace is not None
        and baseline_run is not None
        and baseline_refined_trace is not None
        and baseline_refined_run is not None
    ):
        traces_and_runs = [
            (trace, run, "Condition native"),
            (refined_trace, refined_run, "Condition refined"),
            (baseline_trace, baseline_run, "Baseline native"),
            (baseline_refined_trace, baseline_refined_run, "Baseline refined"),
        ]
        all_valid = True
        err_msg = ""
        for tr, rn, label in traces_and_runs:
            ok, reason = _validate_trace_and_run(tr, rn)
            if not ok:
                all_valid = False
                err_msg = f"{label} invalid: {reason}"
                break

        if all_valid:
            sigs_cn, _, _ = _prepare_signals(trace, run)
            sigs_cr, _, _ = _prepare_signals(refined_trace, refined_run)
            sigs_bn, _, _ = _prepare_signals(baseline_trace, baseline_run)
            sigs_br, _, _ = _prepare_signals(baseline_refined_trace, baseline_refined_run)
            effect_refinement_result = _compute_effect_refinement(
                sigs_cn,
                run,
                sigs_cr,
                refined_run,
                sigs_bn,
                baseline_run,
                sigs_br,
                baseline_refined_run,
            )
        else:
            effect_refinement_result = {
                "performed": False,
                "available": False,
                "reason": err_msg,
            }

    return {
        "available": True,
        "reason": None,
        "support_s": support_s,
        "full_stance": full_stance,
        "channels": channels,
        "curves": curves,
        "baseline_delta": baseline_delta_result,
        "refinement": refinement_result,
        "effect_refinement": effect_refinement_result,
        "qualification": (
            "Mechanical work and power computed from actuator load and joint/hip kinematics on saved force support. "
            "No muscle models, metabolic costs, or physiological efficiency claims are made."
        ),
    }
