# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction-only force observation and comparison module.

This module provides matched observation of predicted leg ground reaction forces
against a measured reference:
- High-rate simulation forces are smoothed with a physical-duration Hann filter
  (matching the native reference preparation) to prevent aliasing before resampling.
- Forces are resampled onto the common supported native reference clock without
  extrapolation.
- Full preintegration support is strictly validated: matching start time within tolerance,
  end time covering the reference up to one preintegration timestep, uniform clock
  without interior gaps. Partial runs must not complete just because a common segment exists.
- Reconstructs target strictly from ``unfiltered_grf_target_n`` (already 21-sample Hann
  filtered); rejects falling back to ``grf_target_n`` to prevent double-filtering of
  clipped targets.
- Requires declared filter metadata or an explicit synthetic filter specification,
  strictly validating family ("Butterworth"), order (4), cutoff (20.0 Hz), passes
  ("forward/backward"), and finite parameters.
- Reconstructs and applies ``forward_sign`` consistently to forward force components
  (Fx) of both target and prediction copies, never mutating inputs in-place.
- Both predicted and reference forces are filtered through the frozen 20 Hz zero-phase
  Butterworth low-pass filter (4th order forward/backward, effective order 8).
- Signed vertical forces (Fz) are preserved (no component-wise clamping in comparison).
- Pre-20 Hz source vs. observed vs. raw diagnostics remain cleanly separated.
- High-rate chatter and raw force/stability/energy diagnostics remain visible and are
  never obscured, serializable to JSON.
- Never filters forces applied to the physical body or overwrites baselines.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

FORCE_FILTER_SAMPLES = 21
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = 20.0
FILTER_PAD_MULTIPLE = 3.0
FILTER_MIN_PAD = 15


def _check_strict_increasing_finite(
    time: np.ndarray, name: str = "time", *, require_uniform: bool = True, tolerance: float = 1e-4
) -> float:
    """Ensure 1-D time array is strictly increasing, finite, and uniform.

    Returns:
        The median sampling interval dt [s].
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError(f"{name} must be a 1-D array of at least 2 samples")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{name} must contain only finite numbers")
    diffs = np.diff(time)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{name} must be strictly increasing without duplicates or backward steps")
    dt = float(np.median(diffs))
    if dt <= 0.0:
        raise ValueError(f"{name} timestep must be positive")
    if require_uniform:
        max_rel_diff = np.max(np.abs(diffs - dt)) / dt
        if max_rel_diff > tolerance:
            raise ValueError(
                f"{name} has non-uniform spacing or interior gaps (max relative deviation: {max_rel_diff:.4e})"
            )
    return dt


def _validate_filter_metadata(ref_meta: Any) -> dict[str, Any]:
    """Strictly validate reference filter metadata."""
    if not isinstance(ref_meta, dict):
        raise ValueError(f"filter metadata must be a dictionary, got {type(ref_meta).__name__}")

    family = ref_meta.get("family")
    if family != "Butterworth":
        raise ValueError(f"filter family must be 'Butterworth', got {family!r}")

    order = ref_meta.get("order")
    if order != FILTER_ORDER:
        raise ValueError(f"filter order must be {FILTER_ORDER}, got {order!r}")

    cutoff_hz = ref_meta.get("cutoff_hz")
    if cutoff_hz is None or not np.isfinite(cutoff_hz) or abs(cutoff_hz - FILTER_CUTOFF_HZ) > 1e-6:
        raise ValueError(f"filter cutoff_hz must be {FILTER_CUTOFF_HZ}, got {cutoff_hz!r}")

    passes = ref_meta.get("passes")
    if passes != "forward/backward":
        raise ValueError(f"filter passes must be 'forward/backward', got {passes!r}")

    return ref_meta


def _padlen(rate_hz: float) -> int:
    """Calculate padlen for Butterworth filter matching prepare_subject.py."""
    return max(FILTER_MIN_PAD, int(math.ceil(FILTER_PAD_MULTIPLE * rate_hz / FILTER_CUTOFF_HZ)))


def design_butterworth_20hz(rate_hz: float) -> tuple[np.ndarray, int]:
    """Design frozen 4th-order 20 Hz Butterworth low-pass filter (SOS format) and pad length.

    Args:
        rate_hz: Sampling rate in Hz.

    Returns:
        tuple of (sos, padlen).
    """
    if rate_hz <= 2.0 * FILTER_CUTOFF_HZ:
        raise ValueError(f"Sampling rate {rate_hz:.2f} Hz must be greater than twice the 20 Hz cutoff")
    from scipy.signal import butter

    sos = butter(FILTER_ORDER, FILTER_CUTOFF_HZ, btype="low", fs=rate_hz, output="sos")
    padlen = _padlen(rate_hz)
    return sos, padlen


def compute_chatter_diagnostics(
    time_s: np.ndarray,
    forces: np.ndarray,
) -> dict[str, Any]:
    """Compute raw high-rate force variation and chatter metrics.

    Evaluates the first and second time derivatives of raw predicted forces to ensure
    high-frequency oscillations, discretization artifacts, or solver chatter remain visible
    and are never obscured by filtering. All return values are JSON-serializable Python types.

    Args:
        time_s: 1-D time array [s] of shape [T].
        forces: Array of shape [..., T, 2].

    Returns:
        dict of chatter diagnostics including RMS and peak force rates (dF/dt) and
        accelerations (d^2F/dt^2).
    """
    _check_strict_increasing_finite(time_s, "time_s")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.ndim < 2 or forces.shape[-1] != 2:
        raise ValueError(f"forces must have shape [..., T, 2], got {forces.shape}")
    if forces.shape[-2] != len(time_s):
        raise ValueError(f"forces time axis length {forces.shape[-2]} does not match time_s {len(time_s)}")
    if not np.all(np.isfinite(forces)):
        raise ValueError("forces contain non-finite values")

    dt = np.diff(time_s)
    dt_shape = [1] * forces.ndim
    dt_shape[-2] = len(dt)
    dt_b = dt.reshape(dt_shape)

    diff1 = np.diff(forces, axis=-2)
    f_rate = diff1 / dt_b  # [..., T-1, 2]

    rms_rate = np.sqrt(np.mean(f_rate**2, axis=-2))
    max_rate = np.max(np.abs(f_rate), axis=-2)

    diag: dict[str, Any] = {
        "max_abs_force_rate_n_s": float(np.max(max_rate)),
        "mean_rms_force_rate_n_s": float(np.mean(rms_rate)),
        "components": {
            "fx_max_rate_n_s": float(np.max(max_rate[..., 0])),
            "fz_max_rate_n_s": float(np.max(max_rate[..., 1])),
            "fx_rms_rate_n_s": float(np.mean(rms_rate[..., 0])),
            "fz_rms_rate_n_s": float(np.mean(rms_rate[..., 1])),
        },
    }

    if forces.shape[-2] >= 3:
        dt_mid = 0.5 * (dt[:-1] + dt[1:])
        dt_mid_shape = [1] * forces.ndim
        dt_mid_shape[-2] = len(dt_mid)
        dt_mid_b = dt_mid.reshape(dt_mid_shape)

        diff2 = np.diff(f_rate, axis=-2)
        f_acc = diff2 / dt_mid_b  # [..., T-2, 2]
        rms_acc = np.sqrt(np.mean(f_acc**2, axis=-2))
        max_acc = np.max(np.abs(f_acc), axis=-2)
        diag["max_abs_force_acc_n_s2"] = float(np.max(max_acc))
        diag["mean_rms_force_acc_n_s2"] = float(np.mean(rms_acc))

    return diag


def approximate_hann_filter_highrate(
    time_s: np.ndarray,
    forces: np.ndarray,
    target_duration_s: float = 0.010,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply physical-duration Hann smoothing to high-rate predicted forces to prevent aliasing.

    Approximates the physical Hann window duration (from the native 21-sample Hann filter at
    2000 Hz, duration (21 - 1)/2000 = 0.010 s) on the finer simulation grid.

    Args:
        time_s: 1-D strictly increasing time array [s] of length T.
        forces: Array of shape [..., T, 2] containing raw predicted forces.
        target_duration_s: Target physical window duration in seconds (must be finite and > 0).

    Returns:
        tuple of (smoothed_forces, metadata_dict).
    """
    if not np.isfinite(target_duration_s) or target_duration_s <= 0.0:
        raise ValueError(f"target_duration_s must be finite and positive, got {target_duration_s}")

    dt = _check_strict_increasing_finite(time_s, "time_s")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.ndim < 2 or forces.shape[-1] != 2:
        raise ValueError(f"forces must have shape [..., T, 2], got {forces.shape}")
    if forces.shape[-2] != len(time_s):
        raise ValueError(f"forces time axis length {forces.shape[-2]} does not match time_s {len(time_s)}")
    if not np.all(np.isfinite(forces)):
        raise ValueError("forces contain non-finite values")

    n_intervals = int(round(target_duration_s / dt))
    n_samples = n_intervals + 1
    if n_samples % 2 == 0:
        n_samples += 1

    actual_duration_s = float((n_samples - 1) * dt)
    effective_rate_hz = 1.0 / dt

    if n_samples > len(time_s):
        raise ValueError(f"High-rate trace length ({len(time_s)}) is shorter than Hann window ({n_samples})")

    kernel = np.hanning(n_samples)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        raise ValueError("Hann kernel sum must be positive")
    kernel /= kernel_sum

    # Convolve along time axis (-2) matching prepare_subject.py mode="same" (zero-padded boundaries)
    from scipy.ndimage import convolve1d

    smoothed = convolve1d(forces, kernel, axis=-2, mode="constant", cval=0.0)

    meta = {
        "method": "symmetric Hann",
        "target_duration_s": float(target_duration_s),
        "actual_duration_s": actual_duration_s,
        "window_samples": n_samples,
        "grid_dt_s": dt,
        "effective_rate_hz": effective_rate_hz,
        "approximation_rule": "n_intervals = round(target_duration / dt); n_samples = n_intervals + 1 (odd)",
        "kernel_normalized": True,
        "boundary_limitation": (
            "Zero-padding outside supported simulation trace (convolve1d mode=constant). "
            "Original reference preparation Hann filter ran on longer trial source context before window selection; "
            "constant/DC response matches only on the interior window away from boundary padding."
        ),
    }
    return smoothed, meta


def resample_to_reference_clock(
    source_time_s: np.ndarray,
    source_forces: np.ndarray,
    reference_time_s: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resample high-rate forces onto common supported native reference clock without extrapolation.

    Args:
        source_time_s: 1-D strictly increasing source time array [s] of length T.
        source_forces: Array of shape [..., T, 2].
        reference_time_s: 1-D strictly increasing reference time array [s] of length N.
        tolerance: Floating point tolerance for bounds checking.

    Returns:
        tuple of (resampled_forces, support_mask, metadata_dict).
        resampled_forces has shape [..., M, 2] where M = count(support_mask).
    """
    _check_strict_increasing_finite(source_time_s, "source_time_s")
    _check_strict_increasing_finite(reference_time_s, "reference_time_s")
    source_forces = np.asarray(source_forces, dtype=np.float64)
    if source_forces.ndim < 2 or source_forces.shape[-1] != 2:
        raise ValueError(f"source_forces must have shape [..., T, 2], got {source_forces.shape}")
    if source_forces.shape[-2] != len(source_time_s):
        raise ValueError("source_forces time dimension does not match source_time_s")

    t_min = float(source_time_s[0])
    t_max = float(source_time_s[-1])

    support_mask = (reference_time_s >= t_min - tolerance) & (reference_time_s <= t_max + tolerance)
    covered_count = int(np.count_nonzero(support_mask))
    total_count = len(reference_time_s)
    excluded_count = total_count - covered_count

    if covered_count == 0:
        raise ValueError(f"No reference samples lie within source time support [{t_min:.6f}, {t_max:.6f}]")

    covered_time = reference_time_s[support_mask]
    query_clipped = np.clip(covered_time, t_min, t_max)

    upper = np.clip(np.searchsorted(source_time_s, query_clipped, side="right"), 1, len(source_time_s) - 1)
    lower = upper - 1
    denom = source_time_s[upper] - source_time_s[lower]
    fraction = (query_clipped - source_time_s[lower]) / denom

    frac_shape = [1] * source_forces.ndim
    frac_shape[-2] = len(query_clipped)
    fraction_b = fraction.reshape(frac_shape)

    f_lower = np.take(source_forces, lower, axis=-2)
    f_upper = np.take(source_forces, upper, axis=-2)

    resampled = f_lower * (1.0 - fraction_b) + f_upper * fraction_b

    meta = {
        "source_interval_s": [t_min, t_max],
        "reference_interval_s": [float(reference_time_s[0]), float(reference_time_s[-1])],
        "common_support_interval_s": [float(covered_time[0]), float(covered_time[-1])],
        "total_reference_samples": total_count,
        "covered_reference_samples": covered_count,
        "excluded_reference_samples": excluded_count,
        "extrapolation_permitted": False,
    }
    return resampled, support_mask, meta


def observe_friction_comparison(
    reference: dict[str, Any],
    pred_time_s: np.ndarray,
    pred_forces: np.ndarray,
    *,
    forward_sign: int = 1,
    normal_threshold_n: float | None = 50.0,
    filter_spec: dict[str, Any] | None = None,
    hann_duration_s: float | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Generate paired reference/prediction force observations on common native clock.

    Transforms raw high-rate predicted forces and measured reference inputs into matched
    20 Hz zero-phase filtered observations while retaining signed vertical forces (no Fz clamp)
    and tracking full-rate raw chatter/stability metrics.

    Strict validation requirements:
    - Pre-20 Hz reference target must come from ``unfiltered_grf_target_n``. No fallback
      to ``grf_target_n`` is allowed because that would double-filter a clipped target.
    - Reference filter metadata (or explicit ``filter_spec``) is strictly validated:
      family='Butterworth', order=4, cutoff_hz=20.0, passes='forward/backward'.
    - Hann smoothing window duration is derived from declared reference metadata or explicit
      ``hann_duration_s`` (positive finite float); never silently assumed.
    - Force shapes must be [..., T, 2]; invalid rank, scalar, or channel count raises ValueError.
    - Full preintegration trace support is required: simulation start must match reference start
      within tolerance, and simulation span must reach within one simulation timestep of the
      reference endpoint (``ref_time[-1] <= pred_time_s[-1] + pred_dt + tolerance``). Partial
      runs are rejected with complete=False and do not pass just because a common segment exists.
    - Clocks must be strictly increasing, finite, and uniform without interior gaps.
    - ``forward_sign`` (+1 or -1) is applied consistently to the forward force component (Fx)
      of newly constructed target and prediction observation copies; inputs are never mutated.

    Args:
        reference: Reference dictionary (must contain 'grf_time_s' and 'unfiltered_grf_target_n').
        pred_time_s: 1-D strictly increasing predicted time array [s] of length T.
        pred_forces: Raw predicted ground reaction forces of shape [..., T, 2] [N] (Fx, Fz).
        forward_sign: Direction factor (+1 or -1) such that F_forward = forward_sign * Fx.
        normal_threshold_n: Normal force threshold [N] to define contact stance intervals
            from pre-20 Hz measured normal force (must be finite positive if specified).
        filter_spec: Optional explicit filter specification dict for synthetic references.
        hann_duration_s: Optional explicit physical Hann smoothing duration in seconds.
            If None, derived from declared source metadata or filter_spec["hann_duration_s"].
        tolerance: Clock matching tolerance in seconds (must be finite >= 0).

    Returns:
        Structured observation dictionary containing:
            - complete: bool
            - failure_reason: str | None
            - clock: 1-D array of covered native reference sample times [s]
            - support_mask: boolean mask on original reference.grf_time_s
            - stance_mask: boolean mask on covered native clock [M]
            - observed_target: 2-D array [M, 2] of matched 20 Hz filtered signed reference forces [N]
            - observed_prediction: Array [..., M, 2] of matched 20 Hz filtered predicted forces [N]
            - pre20hz_target: 2-D array [M, 2] of pre-20 Hz reference forces on common support
            - pre20hz_normal: 1-D array [M] of pre-20 Hz reference normal forces (for stance gating)
            - raw_prediction_chatter: Chatter diagnostics of the raw predicted traces
            - raw_prediction: Unmodified input forces
            - metadata: Complete filter, window, padding, and provenance metadata
    """
    if forward_sign not in (1, -1):
        raise ValueError("forward_sign must be explicitly +1 or -1")

    if normal_threshold_n is not None:
        if not np.isfinite(normal_threshold_n) or normal_threshold_n <= 0.0:
            raise ValueError(f"normal_threshold_n must be finite and positive, got {normal_threshold_n}")

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}")

    # Validate reference keys: strictly require unfiltered_grf_target_n (NO fallback to grf_target_n)
    if "grf_time_s" not in reference:
        raise ValueError("reference must contain 'grf_time_s'")
    if "unfiltered_grf_target_n" not in reference:
        raise ValueError(
            "reference must contain 'unfiltered_grf_target_n' (unfiltered pre-20 Hz source); "
            "fallback to grf_target_n is prohibited to prevent double-filtering clipped targets"
        )

    ref_time = np.asarray(reference["grf_time_s"], dtype=np.float64)
    ref_dt = _check_strict_increasing_finite(ref_time, "reference grf_time_s", require_uniform=True)
    ref_rate_hz = 1.0 / ref_dt

    pre20hz_source = np.asarray(reference["unfiltered_grf_target_n"], dtype=np.float64)
    if pre20hz_source.ndim != 2 or pre20hz_source.shape != (len(ref_time), 2):
        raise ValueError(f"pre-20 Hz source must have shape [N, 2], got {pre20hz_source.shape}")
    if not np.all(np.isfinite(pre20hz_source)):
        raise ValueError("pre-20 Hz source contains non-finite values")

    # Strictly validate declared filter metadata
    ref_filter_meta = None
    ref_raw_meta = None
    if filter_spec is not None:
        ref_filter_meta = _validate_filter_metadata(filter_spec)
    elif "metadata_json" in reference:
        raw_meta = reference["metadata_json"]
        meta_dict = json.loads(str(raw_meta)) if isinstance(raw_meta, (str, bytes, np.ndarray)) else raw_meta
        if isinstance(meta_dict, dict) and "reference_filter" in meta_dict:
            ref_raw_meta = meta_dict
            ref_filter_meta = _validate_filter_metadata(meta_dict["reference_filter"])

    if ref_filter_meta is None:
        raise ValueError(
            "Reference is missing declared filter metadata ('reference_filter' in metadata_json) "
            "and no explicit filter_spec was provided"
        )

    # Determine Hann smoothing window duration
    target_hann_dur = None
    if hann_duration_s is not None:
        if not np.isfinite(hann_duration_s) or hann_duration_s <= 0.0:
            raise ValueError(f"hann_duration_s must be finite and positive, got {hann_duration_s}")
        target_hann_dur = float(hann_duration_s)
    elif ref_raw_meta is not None:
        # Check if declared in native_grf provenance
        try:
            diag_filt = ref_raw_meta["native_grf"]["original"]["provenance"]["metadata"]["diagnostics"]["filter"]
            if "duration_s" in diag_filt and np.isfinite(diag_filt["duration_s"]):
                # Known primary source uses (samples - 1) / rate_hz:
                samples = diag_filt.get("samples", FORCE_FILTER_SAMPLES)
                target_hann_dur = float((samples - 1) * ref_dt)
        except (KeyError, TypeError):
            pass

    if target_hann_dur is None and filter_spec is not None:
        target_hann_dur = filter_spec.get("hann_duration_s")
    if target_hann_dur is None:
        raise ValueError("Hann preprocessing is undeclared; provide hann_duration_s or source Hann metadata")
    if not np.isfinite(target_hann_dur) or target_hann_dur <= 0:
        raise ValueError("Hann duration must be finite and positive")

    # Validate simulation trace array and shape [..., T, 2]
    pred_time_s = np.asarray(pred_time_s, dtype=np.float64)
    pred_dt = _check_strict_increasing_finite(pred_time_s, "pred_time_s", require_uniform=True)

    pred_forces = np.asarray(pred_forces, dtype=np.float64)
    if pred_forces.ndim < 2 or pred_forces.shape[-1] != 2:
        raise ValueError(f"pred_forces must have shape [..., T, 2], got {pred_forces.shape}")
    if pred_forces.shape[-2] != len(pred_time_s):
        raise ValueError(
            f"pred_forces time dimension {pred_forces.shape[-2]} does not match pred_time_s {len(pred_time_s)}"
        )
    if not np.all(np.isfinite(pred_forces)):
        raise ValueError("pred_forces contain non-finite values")

    # Validate full preintegration support
    if abs(pred_time_s[0] - ref_time[0]) > tolerance:
        return {
            "complete": False,
            "failure_reason": (
                f"Simulation start time {pred_time_s[0]:.6f} s does not match reference start time {ref_time[0]:.6f} s"
            ),
        }

    if ref_time[-1] > pred_time_s[-1] + pred_dt + tolerance:
        return {
            "complete": False,
            "failure_reason": (
                f"Simulation trace ends at {pred_time_s[-1]:.6f} s; fails full preintegration support "
                f"for reference ending at {ref_time[-1]:.6f} s (pred_dt={pred_dt:.6e} s)"
            ),
        }

    # 1. Chatter diagnostics on raw high-rate predicted forces
    chatter_diag = compute_chatter_diagnostics(pred_time_s, pred_forces)

    # 2. Approximate physical Hann window smoothing on high-rate predicted forces
    hann_smoothed_pred, hann_meta = approximate_hann_filter_highrate(
        pred_time_s, pred_forces, target_duration_s=target_hann_dur
    )

    # 3. Resample smoothed prediction onto common supported native reference clock
    resampled_pred, support_mask, resample_meta = resample_to_reference_clock(pred_time_s, hann_smoothed_pred, ref_time)

    covered_ref_time = ref_time[support_mask]
    covered_pre20hz_source = pre20hz_source[support_mask].copy()

    # 4. Filter both target and prediction through identical frozen 20 Hz zero-phase Butterworth filter
    sos, padlen = design_butterworth_20hz(ref_rate_hz)

    if len(covered_ref_time) <= padlen:
        raise ValueError(
            f"Common covered reference samples ({len(covered_ref_time)}) is too short for 20 Hz filter padlen ({padlen})"
        )

    # Filter target with signed Fz preserved
    from scipy.signal import sosfiltfilt

    target_filtered = sosfiltfilt(sos, covered_pre20hz_source, axis=-2, padtype="odd", padlen=padlen)

    # Filter resampled prediction with identical filter and padding
    pred_filtered = sosfiltfilt(sos, resampled_pred, axis=-2, padtype="odd", padlen=padlen)

    # Apply forward_sign (+1 or -1) consistently to forward forces (index 0) of copies
    if forward_sign == -1:
        target_filtered = target_filtered.copy()
        pred_filtered = pred_filtered.copy()
        covered_pre20hz_source = covered_pre20hz_source.copy()

        target_filtered[..., 0] *= -1.0
        pred_filtered[..., 0] *= -1.0
        covered_pre20hz_source[..., 0] *= -1.0

    # Butterworth metadata
    butter_meta = {
        "family": "Butterworth",
        "order": FILTER_ORDER,
        "effective_order": 2 * FILTER_ORDER,
        "cutoff_hz": FILTER_CUTOFF_HZ,
        "cutoff_definition": "single-pass -3 dB; combined -6 dB; zero-phase forward/backward",
        "passes": "forward/backward",
        "phase": "zero",
        "implementation": "scipy.signal.butter(output=sos) + sosfiltfilt(axis=-2)",
        "sos": sos.tolist(),
        "sampling_rate_hz": ref_rate_hz,
        "padlen_samples": padlen,
        "padtype": "odd",
        "padding_rule": "max(15 samples, ceil(3 * native_rate / cutoff)); reject shorter inputs",
        "signed_fz_preserved": True,
        "clamp_vertical_zero": False,
        "context_limitation": "Zero-phase odd padding on truncated window endpoints; endpoint edge artifacts can remain.",
    }

    # Stance normal signal for event masking (vertical normal is index 1, unscaled by forward_sign)
    pre20hz_normal = covered_pre20hz_source[..., 1].copy()

    # Stance mask based on pre-20 Hz measured normal force
    if normal_threshold_n is not None:
        stance_mask = pre20hz_normal >= normal_threshold_n
    else:
        stance_mask = np.ones(len(covered_ref_time), dtype=bool)

    metadata = {
        "schema": "digital_shoe_friction_observation_1",
        "forward_sign": forward_sign,
        "normal_threshold_n": normal_threshold_n,
        "reference_filter_declared": ref_filter_meta,
        "hann_presmoothing": hann_meta,
        "clock_resampling": resample_meta,
        "butterworth_20hz": butter_meta,
        "diagnostics": {
            "chatter": chatter_diag,
            "raw_prediction_min": float(np.min(pred_forces)),
            "raw_prediction_max": float(np.max(pred_forces)),
            "observed_prediction_min": float(np.min(pred_filtered)),
            "observed_prediction_max": float(np.max(pred_filtered)),
            "observed_target_min": float(np.min(target_filtered)),
            "observed_target_max": float(np.max(target_filtered)),
            "observed_target_fz_min": float(np.min(target_filtered[..., 1])),
            "target_negative_fz_retained": bool(np.any(target_filtered[..., 1] < 0.0)),
        },
    }

    return {
        "complete": True,
        "failure_reason": None,
        "clock": covered_ref_time,
        "support_mask": support_mask,
        "stance_mask": stance_mask,
        "observed_target": target_filtered,
        "observed_prediction": pred_filtered,
        "pre20hz_target": covered_pre20hz_source,
        "pre20hz_normal": pre20hz_normal,
        "raw_prediction_chatter": chatter_diag,
        "raw_prediction": pred_forces,
        "metadata": metadata,
    }
