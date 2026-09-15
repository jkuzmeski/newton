# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Screen paired finite-window return using native positions and rates.

The bands are engineering settings, not human-identified tolerances. A return
means only that sampled planar body positions and rates, including the actual
terminal, meet the bands. Foam, Maxwell, and friction-history states are not
screened. It does not establish full physical-state return, inter-sample
behavior, asymptotic settling, frequency, passivity, or recovery after a
material change that remains in place.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from numbers import Real

import numpy as np

_POSITION_CHANNELS = ("pelvis_x_m", "pelvis_z_m", "foot_x_m", "foot_z_m", "leg_length_m")
_VELOCITY_CHANNELS = ("pelvis_vx_m_s", "pelvis_vz_m_s", "foot_vx_m_s", "foot_vz_m_s", "leg_rate_m_s")
_REQUIRED_CHANNELS = (*_POSITION_CHANNELS, "pitch_rad", *_VELOCITY_CHANNELS, "pitch_rate_rad_s")


@dataclass(frozen=True)
class RecoveryConfig:
    """Set positive finite engineering bands and observation durations in SI units.

    These settings are not identified from human recovery data. Position and
    rate bands apply to every required paired channel, not only pelvis height.
    """

    minimum_window_s: float = 0.15
    dwell_s: float = 0.05
    position_tolerance_m: float = 0.001
    angle_tolerance_rad: float = 0.001
    velocity_tolerance_m_s: float = 0.01
    angular_velocity_tolerance_rad_s: float = 0.01

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{field.name} must be a positive finite number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be a positive finite number")
            object.__setattr__(self, field.name, float(value))

    def to_dict(self) -> dict:
        """Return the complete engineering screen settings."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> RecoveryConfig:
        """Restore settings, using defaults for omissions and rejecting unknown fields."""
        if not isinstance(value, dict):
            raise ValueError("Recovery settings must be a dictionary")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown recovery settings: {sorted(unknown, key=str)}")
        return cls(**value)


def terminal_state(rig) -> tuple[dict, dict[str, np.ndarray]]:
    """Read world zero after the final integration without advancing the solver.

    Newton body spatial velocities store linear components first, then angular
    components. Leg rate is the leg-axis projection of relative body velocity,
    not a finite difference of trace samples. A zero-length leg has no defined
    axis and yields a nonfinite rate so it cannot pass the recovery screen.

    Args:
        rig: Rig whose ``state_0`` is the actual current state and ``sim_time``
            is its physical clock [s]. Other batch worlds remain in raw arrays.

    Returns:
        Scalar position/rate channels for world zero and independent numeric
        arrays ``terminal_body_q``, ``terminal_body_qd``, ``terminal_time_s``
        suitable for ``numpy.savez``. Positions retain the response report's
        original names. Raw state arrays retain all body components and worlds.
    """
    q = np.asarray(rig.state_0.body_q.numpy()).copy()
    qd = np.asarray(rig.state_0.body_qd.numpy()).copy()
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != 7 or qd.shape != (q.shape[0], 6):
        raise ValueError("Terminal state requires matching body_q [body,7] and body_qd [body,6]")
    foot, pelvis = q[:2].astype(float)
    vf, vp = qd[:2, :3].astype(float)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        separation = pelvis[:3] - foot[:3]
        length = float(np.linalg.norm(separation))
        rate = float(np.dot(separation / length, vp - vf)) if length > 0 else float("nan")
        result = {
            "time_s": float(rig.sim_time),
            "pelvis_z_m": float(pelvis[2]),
            "pitch_rad": float(2 * np.arctan2(foot[4], foot[6])),
            "leg_length_m": length,
            "foot_x_m": float(foot[0]),
            "pelvis_x_m": float(pelvis[0]),
            "foot_z_m": float(foot[2]),
            "pelvis_vx_m_s": float(vp[0]),
            "pelvis_vz_m_s": float(vp[2]),
            "foot_vx_m_s": float(vf[0]),
            "foot_vz_m_s": float(vf[2]),
            "pitch_rate_rad_s": float(qd[0, 4]),
            "leg_rate_m_s": rate,
        }
    return result, {"terminal_body_q": q, "terminal_body_qd": qd, "terminal_time_s": np.asarray(result["time_s"])}


def _numeric_fields(record, *, required):
    arrays, missing, nonfinite, malformed = {}, [], [], []
    if not isinstance(record, dict):
        return arrays, list(required), nonfinite, ["record"]
    missing = [name for name in required if name not in record]
    for name, value in record.items():
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError):
            malformed.append(name)
            continue
        arrays[name] = array
        if not np.isfinite(array).all():
            nonfinite.append(name)
    return arrays, missing, nonfinite, malformed


def _scalar(record, name):
    value = record.get(name)
    return float(value) if value is not None and value.shape == () and np.isfinite(value) else None


def _delta(value, baseline, *, angle=False):
    with np.errstate(invalid="ignore", over="ignore"):
        delta = value - baseline
        return np.arctan2(np.sin(delta), np.cos(delta)) if angle else delta


def summarize_recovery(
    trace: dict,
    baseline_trace: dict,
    dt: float,
    *,
    terminal: dict,
    baseline_terminal: dict,
    push_end_s: float | None,
    config: RecoveryConfig | None = None,
    pair_valid: bool = True,
) -> dict:
    """Compare paired position and rate return on unchanged physical clocks.

    Args:
        trace: Full pre-integration numeric channels for the perturbed world.
        baseline_trace: Unperturbed controller/material pair on the exact same
            clocks. No interpolation or contact-triggered retiming is allowed.
        dt: Positive solver interval [s]. Uniform sampling is checked against
            this interval with a reported float32 storage-roundoff allowance.
        terminal: Actual post-integration position and rate channels.
        baseline_terminal: Matching baseline terminal channels at exactly the
            same time. The terminal must be one solver step after the trace.
        push_end_s: Removed pulse end [s]. ``None`` denotes a persistent material
            change and cannot support a recovery-to-baseline claim.
        config: Engineering bands and required observation durations.
        pair_valid: Separate safety, numerical, and matching-pair qualification.
            Only a boolean true permits a successful return classification.

    Returns:
        JSON-safe deviations, clock checks, window/dwell evidence, and status.
        Trace peaks remain separate from terminal-inclusive peaks. A successful
        return requires an adequate observed post-pulse window and a final
        uninterrupted sampled in-band suffix lasting at least ``dwell_s``,
        including the actual terminal. Suffix entry is an observed sample time,
        never an estimated settling time. Nonfinite values in either full input
        record invalidate the pair; no finite-only selection is performed.
    """
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    dt = float(dt)
    if push_end_s is not None:
        if (
            isinstance(push_end_s, (bool, np.bool_))
            or not isinstance(push_end_s, Real)
            or not math.isfinite(push_end_s)
            or push_end_s < 0
        ):
            raise ValueError("push_end_s must be nonnegative and finite, or None for a persistent material change")
        push_end_s = float(push_end_s)
    config = RecoveryConfig() if config is None else config
    if not isinstance(config, RecoveryConfig):
        raise ValueError("config must be a RecoveryConfig")

    records, missing, nonfinite, malformed, shape_errors = {}, {}, {}, {}, {}
    required = ("time_s", *_REQUIRED_CHANNELS)
    for label, record in (
        ("trace", trace),
        ("baseline_trace", baseline_trace),
        ("terminal", terminal),
        ("baseline_terminal", baseline_terminal),
    ):
        records[label], missing[label], nonfinite[label], malformed[label] = _numeric_fields(record, required=required)
        arrays = records[label]
        shape = arrays.get("time_s", np.empty(0)).shape if "trace" in label else ()
        shape_errors[label] = [name for name in required if name in arrays and arrays[name].shape != shape]
    t = records["trace"].get("time_s", np.empty(0))
    bt = records["baseline_trace"].get("time_s", np.empty(0))
    clock_shape_valid = t.ndim == bt.ndim == 1 and t.size > 0 and bt.size > 0
    clocks_match = bool(clock_shape_valid and np.isfinite(t).all() and np.array_equal(t, bt))
    final_time = _scalar(records["terminal"], "time_s")
    base_final_time = _scalar(records["baseline_terminal"], "time_s")
    terminal_clock_match = final_time is not None and base_final_time is not None and final_time == base_final_time
    last_time = float(t[-1]) if t.ndim == 1 and t.size and np.isfinite(t[-1]) else None
    # The native trace stores float32 t = index * float32(dt), then exports
    # float64. This allowance checks that grid; paired clocks still match exactly.
    clock_scale = max(abs(final_time or 0.0), dt)
    if t.ndim == 1 and t.size and np.isfinite(t).all():
        clock_scale = max(clock_scale, float(np.max(np.abs(t))))
    clock_atol = min(4 * float(np.finfo(np.float32).eps) * clock_scale, dt * 0.001)
    sampling_valid = False
    if clocks_match and terminal_clock_match:
        expected = float(t[0]) + np.arange(t.size, dtype=float) * dt
        sampling_valid = bool(
            t[0] >= 0
            and np.all(np.diff(t) > 0)
            and np.all(np.abs(t - expected) <= clock_atol)
            and final_time > t[-1]
            and abs(final_time - (float(t[0]) + t.size * dt)) <= clock_atol
        )

    reasons = []
    external_valid = isinstance(pair_valid, (bool, np.bool_)) and bool(pair_valid)
    if not external_valid:
        reasons.append("pair_not_qualified")
    for label, problems in (
        ("missing", missing),
        ("nonfinite", nonfinite),
        ("malformed", malformed),
        ("shape", shape_errors),
    ):
        reasons.extend(f"{label}:{source}:{name}" for source, names in problems.items() for name in names)
    if not clocks_match:
        reasons.append("paired_trace_clock_mismatch_or_invalid")
    if not terminal_clock_match:
        reasons.append("paired_terminal_clock_mismatch_or_invalid")
    if not sampling_valid:
        reasons.append("nonuniform_or_nonmonotonic_clock_or_inconsistent_dt_or_terminal")

    tolerances = dict.fromkeys(_POSITION_CHANNELS, config.position_tolerance_m)
    tolerances.update(dict.fromkeys(_VELOCITY_CHANNELS, config.velocity_tolerance_m_s))
    tolerances.update(pitch_rad=config.angle_tolerance_rad, pitch_rate_rad_s=config.angular_velocity_tolerance_rad_s)
    deviations, deltas, finals = {}, {}, {}
    for name in _REQUIRED_CHANNELS:
        delta, final = None, None
        value, baseline = records["trace"].get(name), records["baseline_trace"].get(name)
        if clocks_match and value is not None and baseline is not None and value.shape == baseline.shape == t.shape:
            candidate = _delta(value, baseline, angle=name == "pitch_rad")
            if np.isfinite(candidate).all():
                delta = candidate
            elif name not in nonfinite["trace"] and name not in nonfinite["baseline_trace"]:
                reasons.append(f"nonfinite_difference:trace:{name}")
        value, baseline = _scalar(records["terminal"], name), _scalar(records["baseline_terminal"], name)
        if terminal_clock_match and value is not None and baseline is not None:
            candidate = _delta(np.asarray(value), np.asarray(baseline), angle=name == "pitch_rad")
            if np.isfinite(candidate):
                final = float(candidate)
            else:
                reasons.append(f"nonfinite_difference:terminal:{name}")
        deltas[name], finals[name] = delta, final
        both = np.append(delta, final) if delta is not None and final is not None else None
        post = (
            both[np.append(t >= push_end_s, final_time >= push_end_s)]
            if both is not None and push_end_s is not None
            else None
        )
        deviations[name] = {
            "peak_abs_deviation": float(np.max(np.abs(delta))) if delta is not None else None,
            "peak_abs_including_terminal_deviation": float(np.max(np.abs(both))) if both is not None else None,
            "peak_abs_post_pulse_deviation": float(np.max(np.abs(post))) if post is not None and post.size else None,
            "last_trace_deviation": float(delta[-1]) if delta is not None else None,
            "final_deviation": final,
            "tolerance": tolerances[name],
            "terminal_within_tolerance": abs(final) <= tolerances[name] if final is not None else None,
        }

    available = max(0.0, final_time - push_end_s) if final_time is not None and push_end_s is not None else None
    observed = (
        max(0.0, final_time - max(push_end_s, float(t[0]))) if sampling_valid and push_end_s is not None else None
    )
    window_atol = min(16 * float(np.finfo(float).eps) * clock_scale, config.minimum_window_s * 1e-6)
    adequate = observed is not None and observed + window_atol >= max(config.minimum_window_s, config.dwell_s)
    terminal_in_band, dwell_in_band, suffix_duration, suffix_entry = None, None, None, None
    if not reasons:
        terminal_in_band = all(abs(finals[name]) <= tolerances[name] for name in _REQUIRED_CHANNELS)
        if push_end_s is not None:
            all_times = np.append(t, final_time)
            in_band = np.ones(all_times.shape, dtype=bool)
            for name in _REQUIRED_CHANNELS:
                in_band &= np.abs(np.append(deltas[name], finals[name])) <= tolerances[name]
            in_band &= all_times >= push_end_s
            failures = np.flatnonzero(~in_band)
            start = int(failures[-1] + 1) if failures.size else 0
            suffix_duration = float(final_time - all_times[start]) if start < all_times.size else 0.0
            dwell_atol = min(clock_atol, config.dwell_s * 1e-5)
            dwell_in_band = bool(
                terminal_in_band and suffix_duration > 0 and suffix_duration + dwell_atol >= config.dwell_s
            )
            if adequate and dwell_in_band:
                suffix_entry = float(all_times[start])

    if reasons:
        status = "invalid_pair"
    elif push_end_s is None:
        status = "persistent_material_change"
    elif not adequate:
        status = "insufficient_window"
    elif dwell_in_band and terminal_in_band:
        status = "returned_within_window"
    else:
        status = "not_returned_within_window"
    return {
        "schema": "impedance_finite_window_recovery_1",
        "status": status,
        "returned_within_window": status == "returned_within_window",
        "engineering_screen": True,
        "state_scope": "planar_body_positions_and_rates_only",
        "interpretation": (
            "Sampled finite-window planar body position/rate return only; "
            "foam, Maxwell, and friction-history states are not screened. "
            "Not full physical-state return, human-identified recovery, asymptotic settling, frequency, or passivity."
        ),
        "config": config.to_dict(),
        "pair_valid": external_valid,
        "paired_clock_match": clocks_match,
        "terminal_clock_match": terminal_clock_match,
        "sampling_clock_valid": sampling_valid,
        "dt_grid_roundoff_tolerance_s": clock_atol,
        "invalid_reasons": reasons,
        "missing_fields": missing,
        "nonfinite_fields": nonfinite,
        "malformed_fields": malformed,
        "shape_errors": shape_errors,
        "sample_count": int(t.size),
        "last_trace_time_s": last_time,
        "final_state_time_s": final_time,
        "peak_includes_terminal_state": False,
        "deviations": deviations,
        "window": {
            "push_end_s": push_end_s,
            "available_post_pulse_s": available,
            "observed_post_pulse_s": observed,
            "required_window_s": config.minimum_window_s,
            "required_dwell_s": config.dwell_s,
            "window_adequate": adequate,
            "post_pulse_trace_sample_count": int(np.count_nonzero(t >= push_end_s))
            if clocks_match and push_end_s is not None
            else None,
            "final_dwell_within_tolerances": dwell_in_band,
            "terminal_within_tolerances": terminal_in_band,
            "observed_suffix_entry_time_s": suffix_entry,
            "observed_suffix_entry_after_pulse_s": suffix_entry - push_end_s if suffix_entry is not None else None,
            "observed_suffix_duration_s": suffix_duration,
        },
    }
