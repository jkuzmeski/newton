# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent CPU validation of refit controller equilibrium splines under fixed Maxwell friction.

This module evaluates refit four-channel Cartesian and joint equilibrium commands
against the baseline controller under identical frozen physical contact conditions:
- Fixed Maxwell tangential friction (method 7) with unchanged normal contact laws.
- Unchanged Cartesian hip impedance and joint impedance gains.
- Unchanged measured kinematics, reference frames, and profile bounds.
- Raw physical forces on the native simulation clock without output force filtering.
- Dual scoring: raw unfiltered ground reaction forces (peaks, impulses, timing)
  alongside the original six-channel measured fit metrics against sealed targets.
- Actuator load extrema and rate diagnostics across all four effort channels
  with declared engineering guards (1.5x peak and 2.0x rate).
- Foot pitch and planar translational velocity diagnostics.
- Strict non-mutation checks for inputs, sources, and coefficient files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from projects.digital_shoe.friction_dynamic import (
    DEFAULT_BASELINE_DIR,
    _finite_report,
    compute_source_hashes,
    configure_candidate_friction,
    load_baseline_bundle,
    parse_candidate,
)
from projects.digital_shoe.friction_leg import DEFAULT_BASELINE_MANIFEST, verify_baseline_inputs
from projects.digital_shoe.friction_metrics import score_friction_trace
from projects.impedance_instron.cartesian.fit import FitConfig, _Objective
from projects.impedance_instron.cartesian.mechanics import Body
from projects.impedance_instron.cartesian.run import Config, simulate
from projects.impedance_instron.cartesian.shoe import Shoe
from projects.impedance_instron.cartesian.trajectory import Spline, _derivative_control_polygons

EFFORT_CHANNELS = ("hip_force_x_n", "hip_force_z_n", "knee_torque_nm", "ankle_torque_nm")
RATE_CHANNELS = (
    "hip_force_x_rate_n_s",
    "hip_force_z_rate_n_s",
    "knee_torque_rate_nm_s",
    "ankle_torque_rate_nm_s",
)


def sha256_file(path: Path | str) -> str:
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_artifact_maxwell_candidate(baseline_dir: Path | str) -> dict[str, Any]:
    """Derive exact fixed Maxwell friction settings using relaxation time from digital shoe artifact."""
    shoe_path = Path(baseline_dir) / "digital_shoe.json"
    with open(shoe_path, encoding="utf-8") as f:
        artifact = json.load(f)
    tau = float(artifact["constitutive_model"]["parameters"]["maxwell_relaxation_time_s"])
    return {
        "method": 7,
        "mu": 0.8,
        "kt_scale": 0.1,
        "kv_scale": 1.0,
        "viscous_ratio": 0.0,
        "release_dwell_s": 0.0005,
        "yield_width": 0.0,
        "shear_relaxation_time_s": tau,
    }


def load_coefficients(
    coefficients_path: Path | str, expected_duration_s: float | None = None
) -> tuple[np.ndarray, float, str]:
    """Load and validate equilibrium spline coefficients from an NPZ file.

    Contract:
        - Must contain 'coefficients' array of shape (12, 4) and finite floats.
        - Must contain 'duration_s' scalar float.
        - If expected_duration_s is provided, duration_s must match within numerical tolerance.

    Args:
        coefficients_path: Path to the .npz archive.
        expected_duration_s: Optional reference duration [s] to verify against.

    Returns:
        Tuple of (coefficients [12, 4], duration_s, file_sha256).
    """
    coeff_p = Path(coefficients_path).resolve()
    if not coeff_p.is_file():
        raise FileNotFoundError(f"Coefficients file not found: {coeff_p}")

    file_sha256 = sha256_file(coeff_p)
    with np.load(coeff_p, allow_pickle=False) as arch:
        if "coefficients" not in arch:
            raise KeyError(f"Coefficients file {coeff_p} missing required array 'coefficients'")
        if "duration_s" not in arch:
            raise KeyError(f"Coefficients file {coeff_p} missing required scalar 'duration_s'")

        coefficients = np.asarray(arch["coefficients"], dtype=float)
        duration_s = float(arch["duration_s"])

    if coefficients.shape != (12, 4):
        raise ValueError(f"Equilibrium coefficients must have shape (12, 4), got {coefficients.shape}")
    if not np.isfinite(coefficients).all():
        raise ValueError("Equilibrium coefficients contain non-finite values")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"Equilibrium duration_s must be finite and positive, got {duration_s}")

    if expected_duration_s is not None:
        if not np.isclose(duration_s, expected_duration_s, rtol=0, atol=1e-9):
            raise ValueError(
                f"Equilibrium duration {duration_s:.9f} s does not match expected duration {expected_duration_s:.9f} s"
            )

    return coefficients, duration_s, file_sha256


def verify_spline_profile_bounds(spline: Spline, profile: dict[str, Any]) -> dict[str, Any]:
    """Verify that an equilibrium spline respects the original profile bounds and rate limits.

    Checks:
        - Position bounds: equilibrium_lower <= coefficients <= equilibrium_upper
        - Rate limits: first derivative control polygon <= equilibrium_rate_limit * duration_s
        - Acceleration limits: second derivative control polygon <= equilibrium_acceleration_limit * duration_s^2

    Args:
        spline: Equilibrium Spline instance.
        profile: Pinned profile dictionary containing bounds.

    Returns:
        Dictionary of bounds verification diagnostics.
    """
    lower = np.asarray(profile["equilibrium_lower"], dtype=float)
    upper = np.asarray(profile["equilibrium_upper"], dtype=float)
    rate = np.asarray(profile["equilibrium_rate_limit"], dtype=float)
    accel = np.asarray(profile["equilibrium_acceleration_limit"], dtype=float)

    is_bounded = spline.bounds(lower, upper, rate, accel)

    # Detailed per-channel diagnostics
    coeffs = spline.coefficients
    pos_lower_violation = (coeffs < lower).any(axis=0).tolist()
    pos_upper_violation = (coeffs > upper).any(axis=0).tolist()

    first, second = _derivative_control_polygons(coeffs)
    max_rate_scaled = np.max(np.abs(first), axis=0) / spline.duration_s
    max_accel_scaled = np.max(np.abs(second), axis=0) / (spline.duration_s**2)

    rate_violation = (max_rate_scaled > rate).tolist()
    accel_violation = (max_accel_scaled > accel).tolist()

    return {
        "within_bounds": bool(is_bounded),
        "position_lower_violation": pos_lower_violation,
        "position_upper_violation": pos_upper_violation,
        "rate_violation": rate_violation,
        "acceleration_violation": accel_violation,
        "max_rate_observed": max_rate_scaled.tolist(),
        "rate_limits": rate.tolist(),
        "max_accel_observed": max_accel_scaled.tolist(),
        "accel_limits": accel.tolist(),
    }


def compute_actuator_diagnostics(trace: dict[str, Any], dt_s: float) -> dict[str, Any]:
    """Compute peak actuator loads and load rates from a simulation trace across all 4 channels.

    Channels:
        - Hip force X, Z, and Euclidean norm [N]
        - Knee joint torque [N*m]
        - Ankle joint torque [N*m]
        - Component-wise 4-vector: [hip_x, hip_z, knee_torque, ankle_torque]

    Args:
        trace: Simulation trace containing 'hip_force_n' and 'joint_torque_nm'.
        dt_s: Simulation timestep [s].

    Returns:
        Dictionary of extrema, rates, and 4-channel peak vectors.
    """
    hip_f = np.asarray(trace["hip_force_n"], dtype=float)
    torque = np.asarray(trace["joint_torque_nm"], dtype=float)

    hip_norm = np.linalg.norm(hip_f, axis=1)

    time_s = np.asarray(trace.get("time_s", []))
    if len(time_s) > 1:
        dt_series = np.diff(time_s)
        dt_series = np.where(dt_series > 0, dt_series, dt_s)
    else:
        dt_series = np.full(max(0, len(hip_f) - 1), dt_s)

    hip_f_rate = np.diff(hip_f, axis=0) / dt_series[:, None] if len(hip_f) > 1 else np.zeros((0, 2))
    hip_norm_rate = np.diff(hip_norm) / dt_series if len(hip_norm) > 1 else np.zeros(0)
    torque_rate = np.diff(torque, axis=0) / dt_series[:, None] if len(torque) > 1 else np.zeros((0, 2))

    effort_4ch = np.column_stack((hip_f, torque)) if len(hip_f) else np.zeros((0, 4))
    effort_rate_4ch = np.column_stack((hip_f_rate, torque_rate)) if len(hip_f_rate) else np.zeros((0, 4))

    max_effort_4ch = np.max(np.abs(effort_4ch), axis=0).tolist() if len(effort_4ch) else [None] * 4
    max_rate_4ch = np.max(np.abs(effort_rate_4ch), axis=0).tolist() if len(effort_rate_4ch) else [None] * 4

    def _stats(arr: np.ndarray) -> dict[str, float | None]:
        if len(arr) == 0:
            return {"min": None, "max": None, "max_abs": None}
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "max_abs": float(np.max(np.abs(arr))),
        }

    return {
        "extrema": {
            "hip_force_x_n": _stats(hip_f[:, 0]),
            "hip_force_z_n": _stats(hip_f[:, 1]),
            "hip_force_norm_n": _stats(hip_norm),
            "knee_torque_nm": _stats(torque[:, 0]),
            "ankle_torque_nm": _stats(torque[:, 1]),
        },
        "rates": {
            "hip_force_x_rate_n_s": _stats(hip_f_rate[:, 0]),
            "hip_force_z_rate_n_s": _stats(hip_f_rate[:, 1]),
            "hip_force_norm_rate_n_s": _stats(hip_norm_rate),
            "knee_torque_rate_nm_s": _stats(torque_rate[:, 0]),
            "ankle_torque_rate_nm_s": _stats(torque_rate[:, 1]),
        },
        "effort_4ch_max_abs": max_effort_4ch,
        "effort_rate_4ch_max_abs": max_rate_4ch,
        "channel_names": list(EFFORT_CHANNELS),
    }


def compute_engineering_guards(
    base_diag: dict[str, Any],
    refit_diag: dict[str, Any],
) -> dict[str, Any]:
    """Check component-wise 4-channel effort peak (1.5x) and rate (2.0x) engineering guards."""
    base_peaks = base_diag["effort_4ch_max_abs"]
    refit_peaks = refit_diag["effort_4ch_max_abs"]
    base_rates = base_diag["effort_rate_4ch_max_abs"]
    refit_rates = refit_diag["effort_rate_4ch_max_abs"]

    peak_ratios: dict[str, float | None] = {}
    rate_ratios: dict[str, float | None] = {}
    peak_passed: dict[str, bool | None] = {}
    rate_passed: dict[str, bool | None] = {}

    for i, ch in enumerate(EFFORT_CHANNELS):
        bp, rp = base_peaks[i], refit_peaks[i]
        if bp is not None and bp > 0 and rp is not None:
            ratio = float(rp / bp)
            peak_ratios[ch] = ratio
            peak_passed[ch] = bool(ratio <= 1.5)
        else:
            peak_ratios[ch] = None
            peak_passed[ch] = None

    for i, ch in enumerate(RATE_CHANNELS):
        br, rr = base_rates[i], refit_rates[i]
        if br is not None and br > 0 and rr is not None:
            ratio = float(rr / br)
            rate_ratios[ch] = ratio
            rate_passed[ch] = bool(ratio <= 2.0)
        else:
            rate_ratios[ch] = None
            rate_passed[ch] = None

    all_peaks_pass = all(v is True for v in peak_passed.values()) if peak_passed else False
    all_rates_pass = all(v is True for v in rate_passed.values()) if rate_passed else False
    overall_guard_pass = bool(all_peaks_pass and all_rates_pass)

    return {
        "overall_guard_passed": overall_guard_pass,
        "peak_threshold_multiplier": 1.5,
        "rate_threshold_multiplier": 2.0,
        "peak_ratios": peak_ratios,
        "peak_guards_passed": peak_passed,
        "all_peaks_passed": all_peaks_pass,
        "rate_ratios": rate_ratios,
        "rate_guards_passed": rate_passed,
        "all_rates_passed": all_rates_pass,
        "guard_qualification": (
            "Declared engineering guards on actuator command effort: 1.5x component-wise baseline peak "
            "and 2.0x component-wise baseline rate across all four actuator channels. "
            "These are optimization safety screens, not physiological acceptance limits."
        ),
    }


def compute_foot_kinematics_diagnostics(
    body: Body,
    trace: dict[str, Any],
    reference: dict[str, Any],
    static_pitch_rad: float,
) -> dict[str, Any]:
    """Compute foot pitch angle, angular velocity, and planar translational velocity."""
    states = np.asarray(trace["state"], dtype=float)
    velocities = np.asarray(trace["velocity"], dtype=float)
    times = np.asarray(trace["time_s"], dtype=float)
    n = len(states)

    angular_jacobian = body.angular_jacobian(2)
    ankle_local = np.zeros(2)

    pitches_rad = np.empty(n, dtype=float)
    relative_pitches_rad = np.empty(n, dtype=float)
    angular_vels_rad_s = np.empty(n, dtype=float)
    ankle_vels_m_s = np.empty((n, 2), dtype=float)
    plane_vx_m_s = np.empty(n, dtype=float)

    for i in range(n):
        s = states[i]
        v = velocities[i]
        ankle, jacobian, _ = body.point(s, 2, ankle_local)
        ankle_vels_m_s[i] = jacobian @ v
        pitch = body.angle(s, 2)
        pitches_rad[i] = pitch
        relative_pitches_rad[i] = pitch - static_pitch_rad
        angular_vels_rad_s[i] = float(angular_jacobian @ v)
        plane_vx_m_s[i] = ankle_vels_m_s[i, 0] + angular_vels_rad_s[i] * ankle[1]

    speeds_m_s = np.linalg.norm(ankle_vels_m_s, axis=1)
    plane_diagnostics = {}
    if n >= 2:
        from .friction_onset import compute_reference_derived_kinematics  # noqa: PLC0415

        reference_plane = compute_reference_derived_kinematics(reference, times, body)["plane_vel_x"]
        early = times <= 0.08
        plane_diagnostics = {
            "early_rmse_m_s": float(np.sqrt(np.mean((plane_vx_m_s[early] - reference_plane[early]) ** 2)))
            if np.any(early)
            else None,
            "samples": {
                str(t): {
                    "actual_m_s": float(plane_vx_m_s[np.argmin(abs(times - t))]),
                    "reference_m_s": float(reference_plane[np.argmin(abs(times - t))]),
                }
                for t in (0.0, 0.02, 0.03, 0.04, 0.05, 0.08)
                if times[0] <= t <= times[-1]
            },
            "qualification": "Rigid projected-ground-point velocity from filtered marker reference, not directly measured outsole slip.",
        }

    pitch_rmse_filtered = None
    pitch_rmse_unfiltered = None
    if "foot_pitch_target_rad" in reference and n:
        ref_time = np.asarray(reference["time_s"])
        target_pitch = np.interp(times, ref_time, reference["foot_pitch_target_rad"])
        pitch_rmse_filtered = float(np.sqrt(np.mean((pitches_rad - target_pitch) ** 2)))
    if "unfiltered_foot_pitch_target_rad" in reference and n:
        ref_time = np.asarray(reference["time_s"])
        target_unfiltered = np.interp(times, ref_time, reference["unfiltered_foot_pitch_target_rad"])
        pitch_rmse_unfiltered = float(np.sqrt(np.mean((pitches_rad - target_unfiltered) ** 2)))

    return {
        "ground_plane_velocity": plane_diagnostics,
        "foot_pitch_rad": {
            "min": float(np.min(pitches_rad)) if n else None,
            "max": float(np.max(pitches_rad)) if n else None,
            "mean": float(np.mean(pitches_rad)) if n else None,
            "rmse_vs_filtered_target_rad": pitch_rmse_filtered,
            "rmse_vs_unfiltered_target_rad": pitch_rmse_unfiltered,
        },
        "foot_relative_pitch_rad": {
            "min": float(np.min(relative_pitches_rad)) if n else None,
            "max": float(np.max(relative_pitches_rad)) if n else None,
            "mean": float(np.mean(relative_pitches_rad)) if n else None,
        },
        "foot_angular_velocity_rad_s": {
            "min": float(np.min(angular_vels_rad_s)) if n else None,
            "max": float(np.max(angular_vels_rad_s)) if n else None,
            "max_abs": float(np.max(np.abs(angular_vels_rad_s))) if n else None,
        },
        "ankle_plane_velocity_m_s": {
            "x_min": float(np.min(ankle_vels_m_s[:, 0])) if n else None,
            "x_max": float(np.max(ankle_vels_m_s[:, 0])) if n else None,
            "z_min": float(np.min(ankle_vels_m_s[:, 1])) if n else None,
            "z_max": float(np.max(ankle_vels_m_s[:, 1])) if n else None,
            "max_speed_m_s": float(np.max(speeds_m_s)) if n else None,
        },
    }


def validate_controller(
    baseline_dir: Path | str = DEFAULT_BASELINE_DIR,
    coefficients_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    *,
    device: str = "cpu",
    dt_scale: float = 1.0,
    friction_candidate: dict[str, Any] | Path | str | None = None,
    manifest_path: Path | str = DEFAULT_BASELINE_MANIFEST,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Execute independent CPU validation comparing a refit controller against baseline under fixed Maxwell friction.

    Enforces:
        - Output directory must not exist prior to validation (raises FileExistsError).
        - Source files and baseline inputs are fingerprinted before and verified after.
        - Refit spline coefficients must satisfy original profile bounds; bounds violations NEVER simulate.
        - Fixed Maxwell friction settings (method 7) with relaxation time derived from shoe artifact.
        - Dual scoring of raw physical forces and original six-channel target metrics.
        - Actuator load extrema and 4-channel engineering guard checks.
        - Strict completion criteria: actual step count, raw metric complete flag, and finite Objective residual.

    Args:
        baseline_dir: Directory containing sealed baseline files.
        coefficients_path: Path to refit equilibrium .npz file (coefficients [12, 4], duration_s).
        output_dir: Output directory to write traces and validation reports (MUST NOT exist).
        device: Device to simulate on (default: cpu).
        dt_scale: Timestep multiplier relative to baseline actual_dt_s.
        friction_candidate: Friction candidate specification (must be fixed Maxwell method 7).
        manifest_path: Path to sealed baseline manifest JSON.
        raise_on_failure: Whether to raise RuntimeError/ValueError if validation checks fail.

    Returns:
        Structured validation report dictionary.
    """
    if dt_scale <= 0.0 or not np.isfinite(dt_scale):
        raise ValueError("dt_scale must be positive and finite")

    # Reject existing output directory BEFORE executing any check or simulation
    out_p: Path | None = None
    if output_dir is not None:
        out_p = Path(output_dir).resolve()
        if out_p.exists():
            raise FileExistsError(f"Output directory already exists: {out_p}")
        out_p.parent.mkdir(parents=True, exist_ok=True)

    # Check source files before simulation
    source_hashes_before = compute_source_hashes()
    source_hashes_before[str(Path(__file__).resolve())] = sha256_file(Path(__file__))

    # Load sealed baseline bundle
    ref, prof, spline_base, _shoe_base, base_cfg, input_hashes, summary_raw, base_actual_dt_s = load_baseline_bundle(
        baseline_dir, manifest_path=manifest_path, device=device
    )

    # Obtain exact fixed Maxwell candidate with tau from shoe artifact
    default_fric = get_artifact_maxwell_candidate(baseline_dir)
    if friction_candidate is None:
        friction_candidate = default_fric
    fric_params, fric_meta = parse_candidate(friction_candidate)

    # Explicitly enforce fixed Maxwell contract
    if fric_params.method != 7:
        raise ValueError(
            f"Only fixed Maxwell friction (method 7) is authorized for controller validation, got {fric_params.method}"
        )
    if (
        fric_params.mu != 0.8
        or fric_params.kt_scale != 0.1
        or fric_params.kv_scale != 1.0
        or fric_params.viscous_ratio != 0.0
        or fric_params.release_dwell_s != 0.0005
        or fric_params.yield_width != 0.0
    ):
        raise ValueError(
            f"Friction candidate parameters deviate from authorized fixed Maxwell settings: {asdict(fric_params)}"
        )
    if not np.isclose(fric_params.shear_relaxation_time_s, default_fric["shear_relaxation_time_s"], rtol=0, atol=1e-12):
        raise ValueError(
            f"Maxwell shear_relaxation_time_s ({fric_params.shear_relaxation_time_s}) must match shoe artifact ({default_fric['shear_relaxation_time_s']})"
        )

    # Build simulation configuration
    duration_expected = float(ref["time_s"][-1])
    steps_expected = math.ceil(duration_expected / (base_actual_dt_s * dt_scale))
    actual_dt_s = duration_expected / steps_expected
    sim_config = Config(
        dt_s=actual_dt_s,
        gravity_m_s2=base_cfg.gravity_m_s2,
        compression_limit=base_cfg.compression_limit,
        maximum_force_n=base_cfg.maximum_force_n,
        minimum_hip_height_m=base_cfg.minimum_hip_height_m,
        maximum_speed=base_cfg.maximum_speed,
        joint_limits_diagnostic=base_cfg.joint_limits_diagnostic,
    )

    # Load and validate refit coefficients
    coeff_sha_initial = None
    if coefficients_path is None:
        refit_coeffs = spline_base.coefficients.copy()
        refit_duration = spline_base.duration_s
        coeff_sha_initial = "baseline_equilibrium_self_test"
        spline_refit = spline_base
    else:
        refit_coeffs, refit_duration, coeff_sha_initial = load_coefficients(
            coefficients_path, expected_duration_s=duration_expected
        )
        spline_refit = Spline(refit_duration, refit_coeffs)

    # Verify bounds of refit spline against original profile: BOUNDS VIOLATIONS MUST NOT SIMULATE
    bounds_check = verify_spline_profile_bounds(spline_refit, prof)
    if not bounds_check["within_bounds"]:
        msg = f"Refit equilibrium spline exceeds original profile bounds: {bounds_check}"
        if raise_on_failure:
            raise ValueError(msg)
        # Even if raise_on_failure is False, do NOT simulate out-of-bounds splines
        report_failed: dict[str, Any] = {
            "schema": "digital_shoe_friction_controller_validation_1",
            "status": "bounds_violated",
            "complete": False,
            "failure_reason": msg,
            "coefficients": {
                "path": str(coefficients_path) if coefficients_path else "baseline_self_test",
                "sha256": coeff_sha_initial,
                "shape": list(refit_coeffs.shape),
                "duration_s": refit_duration,
                "bounds_check": bounds_check,
            },
            "source_hashes": source_hashes_before,
            "input_hashes": input_hashes,
        }
        if out_p is not None:
            out_p.mkdir(parents=False, exist_ok=False)
            with open(out_p / "validation.json", "w", encoding="utf-8") as f:
                json.dump(_finite_report(report_failed), f, indent=2, allow_nan=False)
        return _finite_report(report_failed)

    # Initialize leg mechanics body for kinematics diagnostics
    body = Body(
        ref["lengths_m"],
        ref["endpoint_local_m"],
        prof["masses_kg"],
        prof["com_local_m"],
        prof["inertias_kg_m2"],
    )

    mount_m = summary_raw["shoe"]["mount_m"]
    static_pitch_rad = float(summary_raw["shoe"]["static_pitch_rad"])

    # 1. Run baseline controller + fixed Maxwell friction
    shoe_run_base = Shoe(
        Path(baseline_dir) / "digital_shoe.json", mount_m, static_pitch_rad, device=device, friction_model="legacy"
    )
    configure_candidate_friction(shoe_run_base, fric_params)
    shoe_run_base.foundation.reset()
    trace_base, sum_base = simulate(ref, prof, spline_base, shoe_run_base, config=sim_config)

    # 2. Run refit controller + identical fixed Maxwell friction
    shoe_run_refit = Shoe(
        Path(baseline_dir) / "digital_shoe.json", mount_m, static_pitch_rad, device=device, friction_model="legacy"
    )
    configure_candidate_friction(shoe_run_refit, fric_params)
    shoe_run_refit.foundation.reset()
    trace_refit, sum_refit = simulate(ref, prof, spline_refit, shoe_run_refit, config=sim_config)

    # Verify source files, inputs, and coefficient file were NOT modified during simulation
    source_hashes_after = compute_source_hashes()
    source_hashes_after[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Tracked source files were modified during controller validation!")

    verified_inputs_after = verify_baseline_inputs(baseline_dir, manifest_path=manifest_path)
    if input_hashes != verified_inputs_after:
        raise RuntimeError("Baseline input files were modified during controller validation!")

    if coefficients_path is not None:
        coeff_sha_after = sha256_file(coefficients_path)
        if coeff_sha_initial != coeff_sha_after:
            raise RuntimeError("Coefficients file was modified during controller validation!")
    else:
        coeff_sha_after = coeff_sha_initial

    # 3. Score raw physical friction metrics (pre-20Hz unfiltered target)
    raw_time_base = trace_base["time_s"]
    raw_time_refit = trace_refit["time_s"]
    pre_target = ref["unfiltered_grf_target_n"]

    raw_grf_base = np.column_stack(
        [np.interp(raw_time_base, ref["grf_time_s"], pre_target[:, axis]) for axis in range(2)]
    )
    raw_grf_refit = np.column_stack(
        [np.interp(raw_time_refit, ref["grf_time_s"], pre_target[:, axis]) for axis in range(2)]
    )

    raw_metric_base = score_friction_trace(
        {"grf_time_s": raw_time_base, "grf_target_n": raw_grf_base},
        trace_base,
        forward_sign=1,
        normal_threshold_n=50.0,
        summary=None,
    )
    raw_metric_refit = score_friction_trace(
        {"grf_time_s": raw_time_refit, "grf_target_n": raw_grf_refit},
        trace_refit,
        forward_sign=1,
        normal_threshold_n=50.0,
        summary=None,
    )

    # 4. Score original six-channel metrics against sealed measured targets
    fit_cfg_raw = summary_raw.get("fit_config", {})
    fit_config = FitConfig(**fit_cfg_raw) if fit_cfg_raw else FitConfig()
    obj = _Objective(ref, fit_config)
    res_base, six_base, costs_base = obj.evaluate(trace_base, sum_base)
    res_refit, six_refit, costs_refit = obj.evaluate(trace_refit, sum_refit)

    # Strict rollout completion check: requires actual steps, raw metric complete, and valid residual
    base_completed = bool(
        sum_base.get("status") == "completed"
        and sum_base.get("integrated_steps") == steps_expected
        and raw_metric_base.get("complete") is True
        and res_base is not None
        and np.isfinite(res_base).all()
    )
    refit_completed = bool(
        sum_refit.get("status") == "completed"
        and sum_refit.get("integrated_steps") == steps_expected
        and raw_metric_refit.get("complete") is True
        and res_refit is not None
        and np.isfinite(res_refit).all()
    )
    overall_completed = bool(base_completed and refit_completed)
    for metric, complete in ((raw_metric_base, base_completed), (raw_metric_refit, refit_completed)):
        if not complete:
            metric["complete"] = False
            metric["failure_reason"] = "Full original-clock rollout completion check failed"
            metric["trace_metrics"] = {}
            metric["comparison_metrics"] = {}

    # 5. Compute actuator load extrema and load rates across 4 channels
    actuator_diag_base = compute_actuator_diagnostics(trace_base, actual_dt_s)
    actuator_diag_refit = compute_actuator_diagnostics(trace_refit, actual_dt_s)

    # 6. Engineering guard checks (1.5x component peak and 2.0x component rate)
    engineering_guards = compute_engineering_guards(actuator_diag_base, actuator_diag_refit)

    # 7. Compute foot kinematics and plane velocity diagnostics
    foot_diag_base = compute_foot_kinematics_diagnostics(body, trace_base, ref, static_pitch_rad)
    foot_diag_refit = compute_foot_kinematics_diagnostics(body, trace_refit, ref, static_pitch_rad)

    # Assemble comprehensive validation report
    report: dict[str, Any] = {
        "schema": "digital_shoe_friction_controller_validation_1",
        "status": "completed" if overall_completed else "failed",
        "complete": overall_completed,
        "dt_scale": dt_scale,
        "actual_dt_s": actual_dt_s,
        "expected_steps": steps_expected,
        "baseline_actual_dt_s": base_actual_dt_s,
        "source_hashes": source_hashes_after,
        "input_hashes": verified_inputs_after,
        "coefficients": {
            "path": str(coefficients_path) if coefficients_path else "baseline_self_test",
            "sha256": coeff_sha_after,
            "shape": list(refit_coeffs.shape),
            "duration_s": refit_duration,
            "bounds_check": bounds_check,
        },
        "friction_parameters": {
            "source": fric_meta.get("source"),
            "sha256": fric_meta.get("sha256"),
            "parameters": asdict(fric_params),
        },
        "engineering_guards": engineering_guards,
        "baseline_controller": {
            "status": sum_base.get("status"),
            "complete": base_completed,
            "integrated_steps": sum_base.get("integrated_steps"),
            "integrated_duration_s": sum_base.get("integrated_duration_s"),
            "raw_physical_friction_metrics": raw_metric_base,
            "original_six_channel_metrics": six_base,
            "original_six_channel_costs": costs_base,
            "actuator_diagnostics": actuator_diag_base,
            "foot_diagnostics": foot_diag_base,
            "summary": sum_base,
        },
        "refit_controller": {
            "status": sum_refit.get("status"),
            "complete": refit_completed,
            "integrated_steps": sum_refit.get("integrated_steps"),
            "integrated_duration_s": sum_refit.get("integrated_duration_s"),
            "raw_physical_friction_metrics": raw_metric_refit,
            "original_six_channel_metrics": six_refit,
            "original_six_channel_costs": costs_refit,
            "actuator_diagnostics": actuator_diag_refit,
            "foot_diagnostics": foot_diag_refit,
            "summary": sum_refit,
        },
        "comparison": {
            "six_channel_metrics_diff": {
                k: (np.array(six_refit[k]) - np.array(six_base[k])).tolist() for k in six_base if k in six_refit
            }
            if six_base and six_refit
            else {},
            "raw_braking_impulse_diff_ns": (
                raw_metric_refit["trace_metrics"]["braking_impulse_ns"]
                - raw_metric_base["trace_metrics"]["braking_impulse_ns"]
                if raw_metric_base.get("complete") and raw_metric_refit.get("complete")
                else None
            ),
            "raw_propulsive_impulse_diff_ns": (
                raw_metric_refit["trace_metrics"]["propulsive_impulse_ns"]
                - raw_metric_base["trace_metrics"]["propulsive_impulse_ns"]
                if raw_metric_base.get("complete") and raw_metric_refit.get("complete")
                else None
            ),
        },
        "qualification": (
            "Independent CPU validation of refit controller equilibrium splines under fixed Maxwell friction. "
            "Normal contact mechanics and impedance gains are unchanged. Raw unfiltered contact forces and original "
            "sealed six-channel target metrics are reported separately. Process completion does not imply physiological acceptance."
        ),
    }

    limits = np.array(
        [fit_config.hip_tolerance_m] * 2 + [fit_config.joint_tolerance_rad] * 2 + [fit_config.force_tolerance_n] * 2
    )
    if six_refit:
        values = np.r_[six_refit["hip_rmse_m"], six_refit["joint_rmse_rad"], six_refit["force_rmse_n"]]
        report["original_six_gate_passed"] = (values < limits).tolist()
        report["all_original_six_gates_passed"] = bool(np.all(values < limits))
    else:
        report["all_original_six_gates_passed"] = False
    report["qualification"] += (
        " Completion certifies only a full numerical rollout, not experimental acceptance or calibration."
    )
    report = _finite_report(report)

    # Save outputs if requested
    if out_p is not None:
        out_p.mkdir(parents=False, exist_ok=False)
        np.savez_compressed(out_p / "trace_baseline_controller.npz", **trace_base)
        np.savez_compressed(out_p / "trace_refit_controller.npz", **trace_refit)
        with open(out_p / "run_baseline_controller.json", "w", encoding="utf-8") as f:
            json.dump(_finite_report(sum_base), f, indent=2, allow_nan=False)
        with open(out_p / "run_refit_controller.json", "w", encoding="utf-8") as f:
            json.dump(_finite_report(sum_refit), f, indent=2, allow_nan=False)
        with open(out_p / "validation.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)

    if raise_on_failure and not overall_completed:
        raise RuntimeError(
            f"Controller validation failed: base_completed={base_completed}, refit_completed={refit_completed}"
        )

    return report


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for controller validation."""
    parser = argparse.ArgumentParser(
        description="Independent CPU validation of refit controller equilibrium splines under fixed Maxwell friction."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Path to baseline directory containing sealed inputs.",
    )
    parser.add_argument(
        "--coefficients-path",
        type=Path,
        required=True,
        help="Path to refit coefficients .npz file (containing 'coefficients' and 'duration_s').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write validation traces and report JSON (MUST NOT exist).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to simulate on (default: cpu).",
    )
    parser.add_argument(
        "--dt-scale",
        type=float,
        default=1.0,
        help="Timestep scaling factor relative to baseline actual_dt_s (default: 1.0).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_BASELINE_MANIFEST,
        help="Path to sealed baseline manifest JSON.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for controller validation."""
    parser = build_parser()
    args = parser.parse_args()
    report = validate_controller(
        baseline_dir=args.baseline_dir,
        coefficients_path=args.coefficients_path,
        output_dir=args.output_dir,
        device=args.device,
        dt_scale=args.dt_scale,
        manifest_path=args.manifest_path,
    )
    print(f"Controller validation finished with status: {report['status']}, complete: {report['complete']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
