# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure NumPy braking, propulsive, and horizontal ground reaction force metrics.

This module provides tools to score predicted leg ground reaction force (GRF)
traces against a measured Cartesian reference, with:
- Strict finite units and clock validation (no silent extrapolation)
- Explicit forward sign parameter (+/-1)
- Piecewise-linear zero-crossing integration for braking and propulsive impulses
- Stance normal masking with contiguous interval preservation
- Separate full horizontal force RMSE and stance horizontal force RMSE
- Incomplete trace detection without fake passing scores
- Common declared force support scoring and excluded endpoint accounting
- Schema compatibility with cartesian.data.load and existing impedance Instron baselines
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from projects.impedance_instron.cartesian import data


def _file_sha256(path: Path | str) -> str:
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _check_strict_increasing_finite(time: np.ndarray, name: str = "time") -> None:
    """Ensure 1D time array is strictly increasing and all elements are finite."""
    if time.ndim != 1 or len(time) < 2:
        raise ValueError(f"{name} must be a 1D array of at least 2 samples")
    if not np.isfinite(time).all():
        raise ValueError(f"{name} must contain only finite numbers")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _find_contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of [start, end] inclusive index slices where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _validate_intervals(intervals: list[tuple[int, int]], n_samples: int) -> None:
    """Validate that interval slices are non-empty, in-bounds, and non-overlapping in order."""
    prev_end = -1
    for start, end in intervals:
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
            raise ValueError(f"Interval indices must be integers: ({start}, {end})")
        if start < 0 or end >= n_samples:
            raise ValueError(f"Interval ({start}, {end}) is out of bounds for {n_samples} samples")
        if start > end:
            raise ValueError(f"Interval start {start} exceeds end {end}")
        if start <= prev_end:
            raise ValueError(f"Interval ({start}, {end}) overlaps or is out of order with previous end {prev_end}")
        prev_end = end


def _segment_piecewise_linear_impulses(
    t: np.ndarray,
    f: np.ndarray,
) -> tuple[float, float, float]:
    """Compute positive, negative, and net impulses for a contiguous (t, f) segment.

    Correctly splits piecewise-linear segments at exact zero crossings to avoid
    the systematic overestimation of naive clipped trapezoids.

    Returns:
        (positive_impulse, negative_impulse, net_impulse)
        where:
            positive_impulse = integral max(f, 0) dt >= 0
            negative_impulse = -integral min(f, 0) dt >= 0
            net_impulse = integral f dt = positive_impulse - negative_impulse
    """
    if len(t) < 2:
        return 0.0, 0.0, 0.0

    dt = np.diff(t)
    f0 = f[:-1]
    f1 = f[1:]

    pos_mask = (f0 >= 0.0) & (f1 >= 0.0)
    neg_mask = (f0 <= 0.0) & (f1 <= 0.0)
    cross_mask = ~(pos_mask | neg_mask)

    pos_impulse = float(np.sum(0.5 * (f0[pos_mask] + f1[pos_mask]) * dt[pos_mask]))
    neg_impulse = float(-np.sum(0.5 * (f0[neg_mask] + f1[neg_mask]) * dt[neg_mask]))

    if np.any(cross_mask):
        c_f0 = f0[cross_mask]
        c_f1 = f1[cross_mask]
        c_dt = dt[cross_mask]

        # Exact linear zero crossing fraction: c_f0 + alpha * (c_f1 - c_f0) = 0
        alpha = -c_f0 / (c_f1 - c_f0)
        dt0 = alpha * c_dt
        dt1 = (1.0 - alpha) * c_dt

        from_pos = c_f0 > 0.0
        from_neg = ~from_pos

        pos_impulse += float(np.sum(0.5 * c_f0[from_pos] * dt0[from_pos]))
        neg_impulse += float(-np.sum(0.5 * c_f1[from_pos] * dt1[from_pos]))

        neg_impulse += float(-np.sum(0.5 * c_f0[from_neg] * dt0[from_neg]))
        pos_impulse += float(np.sum(0.5 * c_f1[from_neg] * dt1[from_neg]))

    return pos_impulse, neg_impulse, pos_impulse - neg_impulse


def compute_braking_propulsive_impulses(
    time: np.ndarray,
    force_forward: np.ndarray,
    intervals: list[tuple[int, int]] | None = None,
) -> tuple[float, float, float]:
    """Compute braking, propulsive, and net impulses over defined intervals.

    Args:
        time: 1D strictly increasing time array [s].
        force_forward: 1D forward ground reaction force array [N] (positive = forward).
        intervals: Optional list of (start_idx, end_idx) index ranges (inclusive).
            If None, evaluates the entire time array as a single contiguous segment.
            If multiple intervals are provided, impulses are accumulated across
            each interval independently without connecting flight or noncontiguous gaps.
            Overlapping, out-of-order, or out-of-bounds intervals raise ValueError.

    Returns:
        (braking_impulse, propulsive_impulse, net_impulse)
        where:
            braking_impulse = -integral min(Fforward, 0) dt >= 0
            propulsive_impulse = integral max(Fforward, 0) dt >= 0
            net_impulse = integral Fforward dt = propulsive_impulse - braking_impulse
    """
    time = np.asarray(time, dtype=float)
    force_forward = np.asarray(force_forward, dtype=float)
    _check_strict_increasing_finite(time, "time")
    if force_forward.shape != time.shape or not np.isfinite(force_forward).all():
        raise ValueError("force_forward must match time shape and be finite")

    if intervals is None:
        intervals = [(0, len(time) - 1)]
    else:
        _validate_intervals(intervals, len(time))

    total_propulsive = 0.0
    total_braking = 0.0

    for start, end in intervals:
        if start == end:
            continue
        t_seg = time[start : end + 1]
        f_seg = force_forward[start : end + 1]
        pos, neg, _ = _segment_piecewise_linear_impulses(t_seg, f_seg)
        total_propulsive += pos
        total_braking += neg

    return total_braking, total_propulsive, total_propulsive - total_braking


def compute_force_peaks(
    time: np.ndarray,
    force_forward: np.ndarray,
    intervals: list[tuple[int, int]] | None = None,
) -> dict[str, float | None]:
    """Find peak magnitudes and timing of braking (negative) and propulsive (positive) forces.

    Args:
        time: 1D strictly increasing time array [s].
        force_forward: 1D forward ground reaction force array [N].
        intervals: Optional list of (start_idx, end_idx) index ranges.
            Overlapping, out-of-order, or out-of-bounds intervals raise ValueError.

    Returns:
        dict with keys:
            braking_peak_magnitude_n: Peak braking magnitude (abs(min force)), or None if no negative force.
            braking_peak_signed_n: Signed peak braking force (min force <= 0), or None if no negative force.
            braking_peak_time_s: Time [s] of peak braking force, or None if no negative force.
            propulsive_peak_magnitude_n: Peak propulsive magnitude (max force), or None if no positive force.
            propulsive_peak_signed_n: Signed peak propulsive force (max force >= 0), or None if no positive force.
            propulsive_peak_time_s: Time [s] of peak propulsive force, or None if no positive force.
    """
    time = np.asarray(time, dtype=float)
    force_forward = np.asarray(force_forward, dtype=float)
    _check_strict_increasing_finite(time, "time")
    if force_forward.shape != time.shape or not np.isfinite(force_forward).all():
        raise ValueError("force_forward must match time shape and be finite")

    if intervals is None:
        idx_pool = np.arange(len(time))
    else:
        _validate_intervals(intervals, len(time))
        indices: list[int] = []
        for start, end in intervals:
            indices.extend(range(start, end + 1))
        idx_pool = np.array(indices, dtype=int)

    if len(idx_pool) == 0:
        return {
            "braking_peak_magnitude_n": None,
            "braking_peak_signed_n": None,
            "braking_peak_time_s": None,
            "propulsive_peak_magnitude_n": None,
            "propulsive_peak_signed_n": None,
            "propulsive_peak_time_s": None,
        }

    sub_time = time[idx_pool]
    sub_force = force_forward[idx_pool]

    # Braking (negative force)
    min_idx = int(np.argmin(sub_force))
    min_val = float(sub_force[min_idx])
    if min_val < 0.0:
        braking_peak_mag = float(abs(min_val))
        braking_peak_signed = min_val
        braking_peak_time = float(sub_time[min_idx])
    else:
        braking_peak_mag = None
        braking_peak_signed = None
        braking_peak_time = None

    # Propulsive (positive force)
    max_idx = int(np.argmax(sub_force))
    max_val = float(sub_force[max_idx])
    if max_val > 0.0:
        propulsive_peak_mag = max_val
        propulsive_peak_signed = max_val
        propulsive_peak_time = float(sub_time[max_idx])
    else:
        propulsive_peak_mag = None
        propulsive_peak_signed = None
        propulsive_peak_time = None

    return {
        "braking_peak_magnitude_n": braking_peak_mag,
        "braking_peak_signed_n": braking_peak_signed,
        "braking_peak_time_s": braking_peak_time,
        "propulsive_peak_magnitude_n": propulsive_peak_mag,
        "propulsive_peak_signed_n": propulsive_peak_signed,
        "propulsive_peak_time_s": propulsive_peak_time,
    }


def _interp_linear_strict(source_t: np.ndarray, source_f: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Linearly interpolate source_f at query_t without extrapolation."""
    if len(query_t) == 0:
        return np.empty((0, *source_f.shape[1:]), dtype=source_f.dtype)
    tol = 1e-12
    if query_t[0] < source_t[0] - tol or query_t[-1] > source_t[-1] + tol:
        raise ValueError(
            f"Query times [{query_t[0]}, {query_t[-1]}] extend outside source support [{source_t[0]}, {source_t[-1]}]"
        )
    query_clipped = np.clip(query_t, source_t[0], source_t[-1])
    if len(source_t) == 1:
        return np.repeat(source_f[:1], len(query_t), axis=0)

    upper = np.clip(np.searchsorted(source_t, query_clipped, side="right"), 1, len(source_t) - 1)
    lower = upper - 1
    denom = source_t[upper] - source_t[lower]
    fraction = (query_clipped - source_t[lower]) / denom
    if source_f.ndim > 1:
        fraction = fraction[:, None]
    return source_f[lower] * (1.0 - fraction) + source_f[upper] * fraction


def _check_rollout_completion(
    trace: dict[str, np.ndarray],
    summary: dict[str, Any] | None,
    expected_duration: float,
) -> tuple[bool, str | None]:
    """Require complete force support or explicit, duration-matched rollout evidence.

    A successful preintegration rollout may lack its terminal force sample.
    Completion permits scoring its declared common support, never fabricating
    a terminal force. An optimizer exhausting its budget is not a failed rollout.
    """
    time = np.asarray(trace["time_s"], dtype=float)
    if "state" in trace and not np.isfinite(trace["state"]).all():
        return False, "Trace state contains non-finite values"
    span = float(time[-1] - time[0])
    tolerance = 1e-9
    blocks = []
    if summary is not None:
        if not isinstance(summary, dict):
            return False, "Summary must be a JSON object"
        blocks = [summary]
        blocks.extend(summary[key] for key in ("run", "best") if isinstance(summary.get(key), dict))
    steps = None
    duration = None
    dt = None
    declared_complete = False
    for block in blocks:
        if block.get("complete") is False or block.get("status") in ("failed", "incomplete", "aborted"):
            return False, "Summary explicitly reports an incomplete or failed rollout"
        if block.get("failure") is not None or block.get("failure_code", 0) not in (0, None):
            return False, "Summary reports a simulation failure"
        declared_complete |= block.get("complete") is True or block.get("status") == "completed"
        if "integrated_steps" in block:
            value = block["integrated_steps"]
            if isinstance(value, bool) or value != len(time):
                return False, "Summary integrated_steps does not match trace length"
            steps = int(value)
        if "integrated_duration_s" in block:
            value = block["integrated_duration_s"]
            if not np.isfinite(value) or abs(float(value) - expected_duration) > tolerance:
                return False, "Summary integrated duration does not match reference duration"
            duration = float(value)
        if "actual_dt_s" in block:
            value = block["actual_dt_s"]
            if not np.isfinite(value) or float(value) <= 0:
                return False, "Summary timestep must be finite and positive"
            dt = float(value)
    if span >= expected_duration - tolerance:
        return True, None
    if not declared_complete:
        return False, "Force trace ends early without explicit rollout completion evidence"
    differences = np.diff(time)
    if dt is None and steps is not None and np.allclose(differences, differences[0], rtol=1e-6, atol=1e-12):
        dt = float(differences[0])
    if dt is None or not np.allclose(differences, dt, rtol=1e-6, atol=1e-12):
        return False, "Preintegration completion requires a consistent recorded timestep"
    integrated_duration = span + dt
    if abs(integrated_duration - expected_duration) > tolerance:
        return False, "Preintegration trace does not span the declared full duration"
    if steps is None and duration is None:
        return False, "Preintegration completion requires integrated step or duration evidence"
    return True, None


def score_friction_trace(
    reference: dict[str, Any] | Path | str,
    trace: dict[str, Any] | Path | str,
    forward_sign: int,
    normal_threshold_n: float | None = 50.0,
    summary: dict[str, Any] | Path | str | None = None,
    *,
    stance_normal_n: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score a simulated or predicted trace against a measured reference.

    Args:
        reference: Reference dictionary loaded via ``cartesian.data.load()`` or path to NPZ.
            If a path is provided, it is validated and loaded with ``cartesian.data.load``.
        trace: Simulated trace dictionary or path to NPZ. Must contain keys
            ``time_s`` and ``grf_n``.
        forward_sign: Explicit direction factor (+1 or -1) such that
            ``Fforward = forward_sign * grf[:, 0]``.
        normal_threshold_n: Optional normal force threshold [N] to define stance.
            When provided, stance metrics are computed exclusively on intervals where
            the MEASURED reference upward force exceeds this threshold. Noncontiguous
            intervals are preserved without joining flight phases.
        summary: Optional simulation summary dictionary or path to JSON. Used to verify
            rollout completion without confusing optimizer search status.
        stance_normal_n: Optional measured normal signal [N] on the reference clock,
            used only to define stance. This separates pre-filter contact events from
            signed, filtered force observations without clipping either force component.

    Returns:
        Structured evaluation dictionary containing:
            - complete: bool (True only if clocks, shapes, rollout, and common support succeed)
            - rollout_complete: bool
            - failure_reason: str | None
            - metadata: dict of inputs, hashes, forward_sign, thresholds
            - support: declared common force support interval, covered count, and excluded endpoint count
            - reference_metrics: braking/propulsive impulses, peaks, timings on reference
            - trace_metrics: braking/propulsive impulses, peaks, timings on trace
            - comparison_metrics: impulse errors, peak errors, timing errors, stance RMSE, full RMSE
    """
    if forward_sign not in (1, -1):
        raise ValueError("forward_sign must be explicitly +1 or -1")

    # Load reference
    ref_path = None
    if isinstance(reference, (str, Path)):
        ref_path = Path(reference)
        reference = data.load(ref_path)
    elif not isinstance(reference, dict):
        raise ValueError("reference must be a dict or path to NPZ")

    # Load trace
    trace_path = None
    if isinstance(trace, (str, Path)):
        trace_path = Path(trace)
        with np.load(trace_path, allow_pickle=False) as arch:
            trace = {k: arch[k].copy() for k in arch.files}
    elif not isinstance(trace, dict):
        raise ValueError("trace must be a dict or path to NPZ")

    # Load summary
    sum_path = None
    if isinstance(summary, (str, Path)):
        sum_path = Path(summary)
        with open(sum_path, encoding="utf-8") as f:
            summary = json.load(f)

    # Initialize report
    result: dict[str, Any] = {
        "schema": "digital_shoe_friction_metrics_1",
        "complete": False,
        "rollout_complete": False,
        "failure_reason": None,
        "metadata": {
            "forward_sign": forward_sign,
            "normal_threshold_n": normal_threshold_n,
            "rmse_definition": "Root mean square over covered native reference samples; separate full and measured-stance masks.",
            "signal_policy": "Caller-supplied signals; no automatic filtering or clipping. Use score_observed_forces or friction-report for the matched observation policy.",
            "reference_file": str(ref_path) if ref_path else None,
            "reference_sha256": _file_sha256(ref_path) if ref_path and ref_path.exists() else None,
            "trace_file": str(trace_path) if trace_path else None,
            "trace_sha256": _file_sha256(trace_path) if trace_path and trace_path.exists() else None,
            "summary_file": str(sum_path) if sum_path else None,
        },
        "support": {},
        "reference_metrics": {},
        "trace_metrics": {},
        "comparison_metrics": {},
    }

    # Validate reference arrays
    for req in ("grf_time_s", "grf_target_n"):
        if req not in reference:
            result["failure_reason"] = f"Missing reference key: {req}"
            return result

    ref_time = np.asarray(reference["grf_time_s"], dtype=float)
    ref_grf = np.asarray(reference["grf_target_n"], dtype=float)
    try:
        _check_strict_increasing_finite(ref_time, "reference grf_time_s")
    except ValueError as err:
        result["failure_reason"] = f"Invalid reference grf_time_s: {err}"
        return result

    if ref_grf.ndim != 2 or ref_grf.shape != (len(ref_time), 2) or not np.isfinite(ref_grf).all():
        result["failure_reason"] = "reference grf_target_n must be finite with shape [N, 2]"
        return result

    # Validate trace arrays
    for req in ("time_s", "grf_n"):
        if req not in trace:
            result["failure_reason"] = f"Missing trace key: {req}"
            return result

    tr_time = np.asarray(trace["time_s"], dtype=float)
    tr_grf = np.asarray(trace["grf_n"], dtype=float)
    try:
        _check_strict_increasing_finite(tr_time, "trace time_s")
    except ValueError as err:
        result["failure_reason"] = f"Invalid trace time_s: {err}"
        return result

    if tr_grf.ndim != 2 or tr_grf.shape != (len(tr_time), 2) or not np.isfinite(tr_grf).all():
        result["failure_reason"] = "trace grf_n must be finite with shape [M, 2]"
        return result

    # Check rollout completion against expected duration
    ref_duration = float(ref_time[-1] - ref_time[0])
    rollout_ok, rollout_err = _check_rollout_completion(trace, summary, ref_duration)
    result["rollout_complete"] = rollout_ok
    if not rollout_ok:
        result["failure_reason"] = f"Incomplete rollout: {rollout_err}"
        return result

    # Check force support coverage:
    # Under the leg impedance objective contract, simulation trace samples are strictly preintegration.
    # No extrapolation: native measured samples outside simulated support [tr_time[0], tr_time[-1]]
    # are excluded and explicitly counted.
    tol = 1e-12
    support_mask = (ref_time >= tr_time[0] - tol) & (ref_time <= tr_time[-1] + tol)
    covered_count = int(np.count_nonzero(support_mask))
    total_count = len(ref_time)
    uncovered_count = total_count - covered_count

    if not np.any(support_mask):
        result["failure_reason"] = "No native reference samples lie within simulated trace support"
        return result

    covered_ref_time = ref_time[support_mask]
    common_support_start = float(covered_ref_time[0])
    common_support_end = float(covered_ref_time[-1])

    result["support"] = {
        "reference_interval_s": [float(ref_time[0]), float(ref_time[-1])],
        "reference_sample_count": total_count,
        "trace_interval_s": [float(tr_time[0]), float(tr_time[-1])],
        "trace_sample_count": len(tr_time),
        "declared_common_support_s": [common_support_start, common_support_end],
        "covered_native_sample_count": covered_count,
        "uncovered_native_sample_count": uncovered_count,
        "native_force_samples_outside_simulated_support": uncovered_count,
    }

    result["support"]["force_support_complete"] = uncovered_count == 0
    if tr_time[0] > ref_time[0] + tol:
        result["failure_reason"] = "Trace omits the beginning of the reference window"
        return result

    # Extract forward and normal forces
    ref_fwd = forward_sign * ref_grf[:, 0]
    ref_up = ref_grf[:, 1] if stance_normal_n is None else np.asarray(stance_normal_n, dtype=float)
    if ref_up.shape != (len(ref_time),) or not np.isfinite(ref_up).all():
        raise ValueError("stance_normal_n must be finite and match the reference force clock")
    result["metadata"]["stance_source"] = "reference_grf" if stance_normal_n is None else "explicit_normal_signal"

    covered_ref_fwd = ref_fwd[support_mask]
    covered_ref_up = ref_up[support_mask]

    # Interpolate trace forward force onto covered native reference grid without extrapolation
    try:
        tr_fwd_interp = forward_sign * _interp_linear_strict(tr_time, tr_grf[:, 0], covered_ref_time)
    except ValueError as err:
        result["failure_reason"] = f"Interpolation error: {err}"
        return result

    # Compute full horizontal force RMSE over the entire common support
    full_diff = tr_fwd_interp - covered_ref_fwd
    full_fwd_rmse = float(np.sqrt(np.mean(full_diff**2)))
    full_fwd_max_abs_err = float(np.max(np.abs(full_diff)))

    # Apply optional positive threshold normal mask from MEASURED normals
    if normal_threshold_n is not None:
        if not np.isfinite(normal_threshold_n) or normal_threshold_n <= 0.0:
            raise ValueError("normal_threshold_n must be finite and positive if specified")
        normal_mask = covered_ref_up >= normal_threshold_n
        intervals = _find_contiguous_intervals(normal_mask)
        active_sample_count = int(np.count_nonzero(normal_mask))
    else:
        intervals = [(0, len(covered_ref_time) - 1)]
        active_sample_count = len(covered_ref_time)

    result["support"]["active_stance_sample_count"] = active_sample_count
    result["support"]["contiguous_stance_interval_count"] = len(intervals)
    result["support"]["stance_intervals_s"] = [
        [float(covered_ref_time[s]), float(covered_ref_time[e])] for s, e in intervals
    ]

    if len(intervals) == 0 or active_sample_count < 2:
        result["failure_reason"] = "No contiguous contact intervals meet normal force threshold"
        return result

    # Compute reference metrics on stance intervals
    ref_braking_imp, ref_prop_imp, ref_net_imp = compute_braking_propulsive_impulses(
        covered_ref_time, covered_ref_fwd, intervals
    )
    ref_peaks = compute_force_peaks(covered_ref_time, covered_ref_fwd, intervals)

    result["reference_metrics"] = {
        "braking_impulse_ns": ref_braking_imp,
        "propulsive_impulse_ns": ref_prop_imp,
        "net_impulse_ns": ref_net_imp,
        **ref_peaks,
    }

    # Compute trace metrics on the same stance intervals
    tr_braking_imp, tr_prop_imp, tr_net_imp = compute_braking_propulsive_impulses(
        covered_ref_time, tr_fwd_interp, intervals
    )
    tr_peaks = compute_force_peaks(covered_ref_time, tr_fwd_interp, intervals)

    result["trace_metrics"] = {
        "braking_impulse_ns": tr_braking_imp,
        "propulsive_impulse_ns": tr_prop_imp,
        "net_impulse_ns": tr_net_imp,
        **tr_peaks,
    }

    # Stance horizontal force RMSE on the active stance intervals
    active_indices: list[int] = []
    for s, e in intervals:
        active_indices.extend(range(s, e + 1))
    act_idx = np.array(active_indices, dtype=int)

    stance_diff = tr_fwd_interp[act_idx] - covered_ref_fwd[act_idx]
    stance_fwd_rmse = float(np.sqrt(np.mean(stance_diff**2)))
    stance_fwd_max_abs_err = float(np.max(np.abs(stance_diff)))

    # Missing predicted phase flags: True when reference has phase but prediction lacks it
    missing_predicted_braking = bool(
        ref_peaks["braking_peak_magnitude_n"] is not None and tr_peaks["braking_peak_magnitude_n"] is None
    )
    missing_predicted_propulsion = bool(
        ref_peaks["propulsive_peak_magnitude_n"] is not None and tr_peaks["propulsive_peak_magnitude_n"] is None
    )

    def _peak_rel_err(pred: float | None, target: float | None) -> float | None:
        """Return relative peak error |pred - target| / |target|.

        If target is absent (None or ~0), returns None.
        If target exists but predicted phase is absent (pred is None), returns 1.0
        (100% missing-phase penalty) rather than None.
        """
        if target is None or abs(target) < 1e-12:
            return None
        if pred is None:
            return 1.0
        return float(abs(pred - target) / abs(target))

    def _diff_or_none(pred: float | None, target: float | None) -> float | None:
        """Return (pred - target), or None if either phase is absent."""
        if pred is None or target is None:
            return None
        return float(pred - target)

    comp: dict[str, Any] = {
        "full_horizontal_force_rmse_n": full_fwd_rmse,
        "full_horizontal_force_max_abs_error_n": full_fwd_max_abs_err,
        "stance_horizontal_force_rmse_n": stance_fwd_rmse,
        "stance_horizontal_force_max_abs_error_n": stance_fwd_max_abs_err,
        "missing_predicted_braking_phase": missing_predicted_braking,
        "missing_predicted_propulsive_phase": missing_predicted_propulsion,
        "braking_impulse_diff_ns": float(tr_braking_imp - ref_braking_imp),
        "braking_impulse_relative_error": float(abs(tr_braking_imp - ref_braking_imp) / ref_braking_imp)
        if ref_braking_imp > 1e-12
        else None,
        "propulsive_impulse_diff_ns": float(tr_prop_imp - ref_prop_imp),
        "propulsive_impulse_relative_error": float(abs(tr_prop_imp - ref_prop_imp) / ref_prop_imp)
        if ref_prop_imp > 1e-12
        else None,
        "net_impulse_diff_ns": float(tr_net_imp - ref_net_imp),
        "braking_peak_diff_n": _diff_or_none(
            tr_peaks["braking_peak_magnitude_n"], ref_peaks["braking_peak_magnitude_n"]
        ),
        "braking_peak_relative_error": _peak_rel_err(
            tr_peaks["braking_peak_magnitude_n"], ref_peaks["braking_peak_magnitude_n"]
        ),
        "braking_peak_timing_diff_s": _diff_or_none(tr_peaks["braking_peak_time_s"], ref_peaks["braking_peak_time_s"]),
        "propulsive_peak_diff_n": _diff_or_none(
            tr_peaks["propulsive_peak_magnitude_n"], ref_peaks["propulsive_peak_magnitude_n"]
        ),
        "propulsive_peak_relative_error": _peak_rel_err(
            tr_peaks["propulsive_peak_magnitude_n"], ref_peaks["propulsive_peak_magnitude_n"]
        ),
        "propulsive_peak_timing_diff_s": _diff_or_none(
            tr_peaks["propulsive_peak_time_s"], ref_peaks["propulsive_peak_time_s"]
        ),
    }

    result["comparison_metrics"] = comp
    result["complete"] = True
    return result


def score_observed_forces(reference: dict, time_s: np.ndarray, forces: np.ndarray) -> dict:
    """Score signed, bandwidth-matched force observations without altering physical force.

    Args:
        reference: Reference with pre-20 Hz force and declared processing metadata.
        time_s: Complete preintegration simulation clock [s].
        forces: Physical forces [N], shape [steps, 2] or [batch, steps, 2].

    Returns:
        Observation arrays, per-run phase metrics, two-component force RMSE [N],
        and the first six entries of the existing friction loss/metric convention.
        Mechanical energy and cone diagnostics must still come from the raw rollout.
    """
    from .friction_observation import observe_friction_comparison  # noqa: PLC0415

    observation = observe_friction_comparison(reference, time_s, forces)
    if not observation["complete"]:
        raise ValueError(observation["failure_reason"])
    clock = observation["clock"]
    target = observation["observed_target"]
    predicted = observation["observed_prediction"]
    if predicted.ndim == 2:
        predicted = predicted[None, ...]
    if predicted.ndim != 3:
        raise ValueError("Force scoring supports a single trace or one batch dimension")
    ref = {"grf_time_s": clock, "grf_target_n": target}
    force_scale = max(float(np.max(np.abs(target[:, 0]))), 1.0)
    scores = np.zeros((len(predicted), 6), dtype=float)
    metrics = []
    for index, prediction in enumerate(predicted):
        scored = score_friction_trace(
            ref, {"time_s": clock, "grf_n": prediction}, 1, stance_normal_n=observation["pre20hz_normal"]
        )
        if not scored["complete"]:
            raise ValueError(scored["failure_reason"])
        observed, measured = scored["trace_metrics"], scored["reference_metrics"]
        keys = (
            "braking_impulse_ns",
            "propulsive_impulse_ns",
            "braking_peak_magnitude_n",
            "propulsive_peak_magnitude_n",
        )
        values = np.array([observed[name] or 0.0 for name in keys])
        targets = np.array([measured[name] or 0.0 for name in keys])
        error = (values - targets) / np.maximum(targets, 1.0)
        rmse = scored["comparison_metrics"]["full_horizontal_force_rmse_n"]
        loss = (rmse / force_scale) ** 2 + 0.25 * np.sum(error[:2] ** 2) + 0.1 * np.sum(error[2:] ** 2)
        scores[index] = [loss, rmse, *values]
        metrics.append(scored)
    return {
        "observation": observation,
        "metrics": metrics,
        "scores": scores,
        "force_rmse_n": np.sqrt(np.mean((predicted - target) ** 2, axis=1)),
        "force_scale_n": force_scale,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for scoring trace against reference."""
    parser = argparse.ArgumentParser(
        description="Score simulated leg ground reaction force trace against measured reference.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to Cartesian reference NPZ (schema cartesian_single_leg_1).",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to predicted trace NPZ containing time_s and grf_n.",
    )
    parser.add_argument(
        "--forward-sign",
        type=int,
        required=True,
        choices=[1, -1],
        help="Explicit direction sign (+1 or -1) such that Fforward = forward_sign * grf[:, 0].",
    )
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=50.0,
        help="Normal force threshold [N] to define contact stance intervals (default: 50.0).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to simulation summary.json for rollout completion and status checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSON score report.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    res = score_friction_trace(
        reference=args.reference,
        trace=args.trace,
        forward_sign=args.forward_sign,
        normal_threshold_n=args.normal_threshold,
        summary=args.summary,
    )

    out_str = json.dumps(res, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"Metrics written to {args.output}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
