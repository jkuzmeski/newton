# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate offline campaign report, results.json, and results.csv for frozen shoe experiments."""

from __future__ import annotations

import copy
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from projects.digital_shoe.material import hyperfoam_pressure_numpy

from .actuation import analyze_actuation


def _plain(value: Any) -> Any:
    """Recursively convert numpy types to Python native types and nonfinite floats to None."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_float(val: Any) -> float | None:
    """Return a finite float or None."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _format_cell(val: Any) -> str:
    """Format scalar or structure for CSV cell output."""
    if val is None:
        return ""
    if isinstance(val, (int, str)):
        return str(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            return ""
        return f"{val:.6g}"
    if isinstance(val, (list, tuple, dict)):
        return json.dumps(_plain(val))
    return str(val)


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Load an npz file safely as a dict of numpy arrays, distinguishing missing from corrupt."""
    if not path.is_file():
        return None, None
    try:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}, None
    except Exception as exc:
        return None, str(exc)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load a json file safely, distinguishing missing from corrupt."""
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def _decimate_series(t: np.ndarray, y: np.ndarray, max_points: int = 150) -> tuple[list[float], list[Any]]:
    """Decimate a time series to at most max_points, always keeping the first and last point."""
    n = len(t)
    if n <= max_points:
        return [_plain(float(x)) for x in t], _plain(y)
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    indices = np.unique(indices)
    return [_plain(float(x)) for x in t[indices]], _plain(y[indices])


def _decimate_curve(x: np.ndarray, y: np.ndarray, max_points: int = 150) -> tuple[list[float], list[float]]:
    """Decimate an (x, y) 1D curve to at most max_points, keeping endpoints."""
    n = len(x)
    if n <= max_points:
        return [_plain(float(v)) for v in x], [_plain(float(v)) for v in y]
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    indices = np.unique(indices)
    return [_plain(float(v)) for v in x[indices]], [_plain(float(v)) for v in y[indices]]


def _calc_ogden_hill_pressure_kpa(
    strains: list[float] | np.ndarray, mu1: float, a1: float, mu2: float, a2: float
) -> list[float]:
    """Calculate 2-term compressible Ogden-Hill pressure curve [kPa] via shared hyperfoam_pressure_numpy."""
    st = np.asarray(strains, dtype=float)
    p_pa = hyperfoam_pressure_numpy(st, float(mu1), float(a1), float(mu2), float(a2), 0.0, 1.0, 0.05)
    return [float(v) for v in (p_pa / 1000.0)]


def _compute_trace_domain_evidence(trace_data: dict[str, Any] | None) -> dict[str, Any]:
    """Compute strain domain evidence (literature limit = 0.60) from a simulation trace."""
    if trace_data is None or "time_s" not in trace_data:
        return {"available": False, "reason": "Simulation trace unavailable"}

    time_s = np.asarray(trace_data["time_s"], dtype=float)
    total_steps = len(time_s)
    if total_steps == 0 or not np.isfinite(time_s).all():
        return {"available": False, "reason": "Simulation trace time missing, empty, or nonfinite"}

    has_driven = "driven_compression_fraction" in trace_data
    has_passive = "passive_compression_fraction" in trace_data
    has_all = "compression_fraction" in trace_data

    if not (has_driven or has_passive or has_all):
        return {"available": False, "reason": "No compression fraction channels found in simulation trace"}

    driven_arr = np.asarray(trace_data["driven_compression_fraction"], dtype=float) if has_driven else None
    passive_arr = np.asarray(trace_data["passive_compression_fraction"], dtype=float) if has_passive else None
    all_arr = np.asarray(trace_data["compression_fraction"], dtype=float) if has_all else None

    for name, arr in [("driven", driven_arr), ("passive", passive_arr), ("all", all_arr)]:
        if arr is not None and (len(arr) != total_steps or not np.isfinite(arr).all()):
            return {
                "available": False,
                "reason": f"Simulation trace {name} strain array has wrong length or nonfinite values",
            }

    if all_arr is None and driven_arr is not None and passive_arr is not None:
        all_arr = np.maximum(driven_arr, passive_arr)
    elif all_arr is None and driven_arr is not None:
        all_arr = driven_arr

    limit = 0.60
    max_all = float(np.max(all_arr)) if all_arr is not None and len(all_arr) else None
    max_driven = float(np.max(driven_arr)) if driven_arr is not None and len(driven_arr) else None
    max_passive = float(np.max(passive_arr)) if passive_arr is not None and len(passive_arr) else None

    all_over = np.where(all_arr > limit)[0] if all_arr is not None else np.array([])
    exceeds_any = bool(len(all_over) > 0)
    first_exceed_time_s = float(time_s[all_over[0]]) if exceeds_any else None
    exceed_step_count = int(len(all_over))
    exceed_step_fraction = float(len(all_over) / total_steps) if total_steps > 0 else 0.0

    driven_over = np.where(driven_arr > limit)[0] if driven_arr is not None else np.array([])
    driven_exceeds = bool(len(driven_over) > 0)
    driven_first_time_s = float(time_s[driven_over[0]]) if driven_exceeds else None
    driven_exceed_count = int(len(driven_over))
    driven_exceed_frac = float(len(driven_over) / total_steps) if total_steps > 0 else 0.0

    passive_over = np.where(passive_arr > limit)[0] if passive_arr is not None else np.array([])
    passive_exceeds = bool(len(passive_over) > 0)
    passive_first_time_s = float(time_s[passive_over[0]]) if passive_exceeds else None
    passive_exceed_count = int(len(passive_over))
    passive_exceed_frac = float(len(passive_over) / total_steps) if total_steps > 0 else 0.0

    return {
        "available": True,
        "literature_limit_strain": limit,
        "total_saved_steps": total_steps,
        "max_any_strain": max_all,
        "max_driven_strain": max_driven,
        "max_passive_strain": max_passive,
        "exceeds_literature_limit": exceeds_any,
        "first_exceed_time_s": first_exceed_time_s,
        "exceed_step_count": exceed_step_count,
        "exceed_step_fraction": exceed_step_fraction,
        "driven": {
            "max_strain": max_driven,
            "exceeds": driven_exceeds,
            "first_exceed_time_s": driven_first_time_s,
            "exceed_step_count": driven_exceed_count,
            "exceed_step_fraction": driven_exceed_frac,
        },
        "passive": {
            "max_strain": max_passive,
            "exceeds": passive_exceeds,
            "first_exceed_time_s": passive_first_time_s,
            "exceed_step_count": passive_exceed_count,
            "exceed_step_fraction": passive_exceed_frac,
        },
    }


def _compute_hys_domain_evidence(raw_npz_data: dict[str, Any] | None) -> dict[str, Any]:
    """Compute strain domain evidence (literature limit = 0.60) from bench hysteresis raw NPZ."""
    if raw_npz_data is None or "time_s" not in raw_npz_data:
        return {"available": False, "reason": "Hysteresis raw data unavailable"}

    time_s = np.asarray(raw_npz_data["time_s"], dtype=float)
    total_steps = len(time_s)
    if total_steps == 0 or not np.isfinite(time_s).all():
        return {"available": False, "reason": "Hysteresis time array missing, empty, or nonfinite"}

    has_all = "max_column_strain" in raw_npz_data
    has_passive = "max_passive_strain" in raw_npz_data

    if not has_all:
        return {"available": False, "reason": "Missing max_column_strain in hysteresis raw data"}

    all_arr = np.asarray(raw_npz_data["max_column_strain"], dtype=float)
    passive_arr = np.asarray(raw_npz_data["max_passive_strain"], dtype=float) if has_passive else None

    if len(all_arr) != total_steps or not np.isfinite(all_arr).all():
        return {"available": False, "reason": "Hysteresis max_column_strain array has wrong length or nonfinite values"}
    if passive_arr is not None and (len(passive_arr) != total_steps or not np.isfinite(passive_arr).all()):
        return {
            "available": False,
            "reason": "Hysteresis max_passive_strain array has wrong length or nonfinite values",
        }

    limit = 0.60
    max_all = float(np.max(all_arr)) if len(all_arr) else None
    max_passive = float(np.max(passive_arr)) if passive_arr is not None and len(passive_arr) else None

    all_over = np.where(all_arr > limit)[0]
    exceeds_any = bool(len(all_over) > 0)
    first_exceed_time_s = float(time_s[all_over[0]]) if exceeds_any else None
    exceed_step_count = int(len(all_over))
    exceed_step_fraction = float(len(all_over) / total_steps) if total_steps > 0 else 0.0

    passive_over = np.where(passive_arr > limit)[0] if passive_arr is not None else np.array([])
    passive_exceeds = bool(len(passive_over) > 0)
    passive_first_time_s = float(time_s[passive_over[0]]) if passive_exceeds else None
    passive_exceed_count = int(len(passive_over))
    passive_exceed_frac = float(len(passive_over) / total_steps) if total_steps > 0 else 0.0

    return {
        "available": True,
        "literature_limit_strain": limit,
        "total_saved_steps": total_steps,
        "max_any_strain": max_all,
        "max_all_columns_strain": max_all,
        "max_driven_strain": None,  # Bench NPZ records all columns and passive columns, no separate driven-only
        "max_passive_strain": max_passive,
        "exceeds_literature_limit": exceeds_any,
        "first_exceed_time_s": first_exceed_time_s,
        "exceed_step_count": exceed_step_count,
        "exceed_step_fraction": exceed_step_fraction,
        "passive": {
            "max_strain": max_passive,
            "exceeds": passive_exceeds,
            "first_exceed_time_s": passive_first_time_s,
            "exceed_step_count": passive_exceed_count,
            "exceed_step_fraction": passive_exceed_frac,
        },
    }


def _extract_protocol_data(
    protocol_dir: Path,
    cond_id: str,
    protocol_name: str,
    campaign_dir: Path,
    *,
    baseline_actuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract native, refined, and comparison data for a protocol (primary or clearance_matched)."""
    native_dir = protocol_dir / "native"
    refined_dir = protocol_dir / "refined"
    comparison_file = protocol_dir / "comparison.json"

    native_summary, native_json_err = _load_json(native_dir / "summary.json")
    refined_summary, _ = _load_json(refined_dir / "summary.json")
    comparison, _ = _load_json(comparison_file)

    native_trace, native_npz_err = _load_npz(native_dir / "trace.npz")
    refined_trace, _ = _load_npz(refined_dir / "trace.npz")

    # Check for report.html generated per condition by root/runner
    native_report_path = native_dir / "report.html"
    relative_report_link = None
    if native_report_path.is_file():
        try:
            relative_report_link = str(native_report_path.relative_to(campaign_dir))
        except ValueError:
            relative_report_link = str(native_report_path)

    # Check for corruption first
    if native_json_err is not None:
        return {
            "status": "corrupt",
            "failure_reason": f"Corrupt summary.json: {native_json_err}",
            "last_valid_time_s": None,
            "completed_fraction": None,
            "integrated_duration_s": None,
            "requested_duration_s": None,
            "native_report_link": relative_report_link,
            "summary": None,
            "refined_summary": None,
            "comparison": None,
            "refinement": None,
            "baseline_delta": None,
            "motion": {"available": False, "reason": "Corrupt summary.json"},
            "plot_curves": {"available": False},
            "metrics": None,
            "observations": None,
            "run": None,
        }

    if native_npz_err is not None:
        return {
            "status": "corrupt",
            "failure_reason": f"Corrupt trace.npz: {native_npz_err}",
            "last_valid_time_s": None,
            "completed_fraction": None,
            "integrated_duration_s": None,
            "requested_duration_s": None,
            "native_report_link": relative_report_link,
            "summary": native_summary,
            "refined_summary": None,
            "comparison": None,
            "refinement": None,
            "baseline_delta": None,
            "motion": {"available": False, "reason": "Corrupt trace.npz"},
            "plot_curves": {"available": False},
            "metrics": None,
            "observations": None,
            "run": None,
        }

    # Determine status and failure details
    status = "not_run"
    failure_reason = None
    last_valid_time_s = None
    completed_fraction = None
    integrated_duration_s = None
    requested_duration_s = None

    if native_summary is not None:
        run = native_summary.get("run", {})
        status = run.get("status", "unknown")
        failure = run.get("failure")
        if failure:
            reasons = failure.get("reasons", [])
            failure_reason = "; ".join(reasons) if isinstance(reasons, list) else str(reasons)
            last_valid_time_s = _safe_float(failure.get("time_s"))
        integrated_duration_s = _safe_float(run.get("integrated_duration_s"))
        requested_duration_s = _safe_float(run.get("requested_duration_s"))
        if requested_duration_s and requested_duration_s > 0 and integrated_duration_s is not None:
            completed_fraction = integrated_duration_s / requested_duration_s
        if last_valid_time_s is None and integrated_duration_s is not None:
            last_valid_time_s = integrated_duration_s

    # Motion decimated data
    motion_data: dict[str, Any] = {"available": False}
    if (
        native_trace is not None
        and "time_s" in native_trace
        and "state" in native_trace
        and len(native_trace["time_s"]) > 0
    ):
        t_arr = np.asarray(native_trace["time_s"], dtype=float)
        state_arr = np.asarray(native_trace["state"], dtype=float)
        # Handle failure stop time truncation if trace wasn't stopped
        if last_valid_time_s is not None and len(t_arr) > 0 and t_arr[-1] > last_valid_time_s + 1e-9:
            valid_mask = t_arr <= last_valid_time_s + 1e-9
            if np.any(valid_mask):
                t_arr = t_arr[valid_mask]
                state_arr = state_arr[valid_mask]

        dec_t, dec_state = _decimate_series(t_arr, state_arr, max_points=120)
        motion_data = {
            "available": True,
            "time_s": dec_t,
            "state": dec_state,  # shape [N, 5]
            "last_valid_time_s": last_valid_time_s,
        }

    # Overlay curves for plots: Hip forward (x), Hip vertical (z), Knee/Ankle angle, GRF x, GRF z
    plot_curves: dict[str, Any] = {"available": False}
    if native_trace is not None and "time_s" in native_trace and len(native_trace["time_s"]) > 0:
        t_arr = np.asarray(native_trace["time_s"], dtype=float)
        if last_valid_time_s is not None and len(t_arr) > 0 and t_arr[-1] > last_valid_time_s + 1e-9:
            valid_mask = t_arr <= last_valid_time_s + 1e-9
            t_arr = t_arr[valid_mask]
            st = np.asarray(native_trace["state"])[valid_mask] if "state" in native_trace else None
            grf = np.asarray(native_trace["grf_n"])[valid_mask] if "grf_n" in native_trace else None
        else:
            st = np.asarray(native_trace["state"]) if "state" in native_trace else None
            grf = np.asarray(native_trace["grf_n"]) if "grf_n" in native_trace else None

        t_dec, _ = _decimate_series(t_arr, t_arr, max_points=120)
        curves: dict[str, Any] = {"time_s": t_dec}
        if st is not None and st.ndim == 2 and st.shape[1] >= 5:
            _, hip_x = _decimate_series(t_arr, st[:, 0], max_points=120)
            _, hip_z = _decimate_series(t_arr, st[:, 1], max_points=120)
            _, knee = _decimate_series(t_arr, st[:, 3], max_points=120)
            _, ankle = _decimate_series(t_arr, st[:, 4], max_points=120)
            curves["hip_x_m"] = hip_x
            curves["hip_z_m"] = hip_z
            curves["knee_rad"] = knee
            curves["ankle_rad"] = ankle
        if grf is not None and grf.ndim == 2 and grf.shape[1] >= 2:
            _, grf_x = _decimate_series(t_arr, grf[:, 0], max_points=120)
            _, grf_z = _decimate_series(t_arr, grf[:, 1], max_points=120)
            curves["grf_x_n"] = grf_x
            curves["grf_z_n"] = grf_z
        plot_curves = {"available": True, **curves}

    # Extract metrics and observations
    metrics = native_summary.get("metrics") if native_summary else None
    obs = native_summary.get("observations") if native_summary else None
    run = native_summary.get("run") if native_summary else None

    # Refinement and delta diagnostics
    refinement_info = None
    baseline_delta = None
    if comparison is not None:
        refinement_info = comparison.get("refinement")
        baseline_delta = comparison.get("baseline_delta") or comparison.get("baseline")

    if refinement_info is None and refined_summary is not None:
        refinement_info = refined_summary.get("refinement")

    actuation = analyze_actuation(
        native_trace,
        run,
        refined_trace=refined_trace,
        refined_run=refined_summary.get("run") if refined_summary else None,
        **(baseline_actuation or {}),
    )
    if actuation.get("available"):
        actuation["curves"] = _actuation_plot_curves(actuation["curves"])

    native_domain = _compute_trace_domain_evidence(native_trace)
    refined_domain = _compute_trace_domain_evidence(refined_trace)

    return {
        "status": status,
        "failure_reason": failure_reason,
        "last_valid_time_s": last_valid_time_s,
        "completed_fraction": completed_fraction,
        "integrated_duration_s": integrated_duration_s,
        "requested_duration_s": requested_duration_s,
        "native_report_link": relative_report_link,
        "summary": native_summary,
        "refined_summary": refined_summary,
        "comparison": comparison,
        "refinement": refinement_info,
        "baseline_delta": baseline_delta,
        "motion": motion_data,
        "plot_curves": plot_curves,
        "metrics": metrics,
        "observations": obs,
        "run": run,
        "actuation": actuation,
        "domain_evidence": native_domain,
        "refined_domain_evidence": refined_domain,
    }


def _actuation_plot_curves(curves: dict[str, np.ndarray]) -> dict[str, Any]:
    """Reduce display points after analysis while retaining endpoints and each series' extrema."""
    displayed = {"time_s", "knee_torque_nm", "ankle_torque_nm"}
    for channel in ("knee", "ankle", "hip", "total"):
        displayed.add(f"{channel}_power_w")
        displayed.update(f"{channel}_{kind}_work_j" for kind in ("positive", "absorbed", "net"))
    curves = {key: value for key, value in curves.items() if key in displayed}
    count = len(curves["time_s"])
    indices = set(np.linspace(0, count - 1, min(count, 360), dtype=int).tolist())
    for key, values in curves.items():
        if key != "time_s" and len(values):
            indices.update((int(np.argmin(values)), int(np.argmax(values))))
    selected = np.asarray(sorted(indices), dtype=int)
    return {key: _plain(np.asarray(values)[selected]) for key, values in curves.items()}


def _actuation_record(condition_id: str, protocol: str, data: dict[str, Any]) -> dict[str, Any]:
    """Flatten native values and common-support sensitivity diagnostics for CSV export."""
    analysis = data.get("actuation") or {}
    record = {
        "condition_id": condition_id,
        "protocol": protocol,
        "status": data.get("status"),
        "available": analysis.get("available", False),
        "support_s": analysis.get("support_s"),
        "full_stance": analysis.get("full_stance", False),
        "native_dt_s": (data.get("run") or {}).get("actual_dt_s"),
        "refined_dt_s": ((data.get("refined_summary") or {}).get("run") or {}).get("actual_dt_s"),
        "baseline_support_s": (analysis.get("baseline_delta") or {}).get("support_s"),
        "refinement_support_s": (analysis.get("refinement") or {}).get("support_s"),
        "effect_refinement_support_s": (analysis.get("effect_refinement") or {}).get("support_s"),
    }
    for channel in ("knee", "ankle", "hip", "hip_x", "hip_z", "total"):
        for metric, value in (analysis.get("channels") or {}).get(channel, {}).items():
            if isinstance(value, (int, float, np.generic)) or value is None:
                record[f"{channel}_{metric}"] = value
        for metric, value in (
            (analysis.get("baseline_delta") or {}).get("percent_change_channels", {}).get(channel, {}).items()
        ):
            record[f"{channel}_{metric}_baseline_percent_change"] = value
        for section in ("baseline_delta", "refinement", "effect_refinement"):
            for metric, value in (analysis.get(section) or {}).get("channels", {}).get(channel, {}).items():
                if isinstance(value, dict):
                    for key, scalar in value.items():
                        record[f"{channel}_{metric}_{section}_{key}"] = scalar
                else:
                    record[f"{channel}_{metric}_{section}"] = value
    return record


def _extract_hysteresis_data(hys_dir: Path, requested_fixtures: list[str]) -> dict[str, Any]:
    """Extract hysteresis summary, raw curves, work statistics, and status for requested fixtures."""
    summary_file = hys_dir / "summary.json"
    summary, json_err = _load_json(summary_file)
    if json_err is not None:
        return {
            "status": "corrupt",
            "reason": f"Corrupt summary.json: {json_err}",
            "fixtures": {
                fix: {"status": "corrupt", "reason": json_err, "metrics": None, "curve": {"available": False}}
                for fix in requested_fixtures
            },
        }

    summary = summary or {}
    comp_file = hys_dir / "comparison.json"
    comp_data, _ = _load_json(comp_file) if comp_file.is_file() else (None, None)
    comp_fixtures = comp_data.get("fixtures", {}) if comp_data else {}

    fixtures_data: dict[str, Any] = {}
    fixtures_in_summary = summary.get("fixtures", {})

    for fix_name in requested_fixtures:
        fix_info = fixtures_in_summary.get(fix_name)
        if fix_info is None:
            # Check if files exist on disk even if not in summary.json
            raw_npz_path = hys_dir / f"{fix_name}_raw.npz"
            curve_json_path = hys_dir / f"{fix_name}_curve.json"
            if raw_npz_path.is_file() or curve_json_path.is_file():
                fix_info = {"status": "completed", "files": {}}
                if raw_npz_path.is_file():
                    fix_info["files"]["npz"] = raw_npz_path.name
                if curve_json_path.is_file():
                    fix_info["files"]["curve_json"] = curve_json_path.name
            else:
                if fix_name == "forefoot_last":
                    fix_info = {
                        "status": "blocked",
                        "reason": "Fixture 'forefoot_last' not present in artifact instron_fixtures",
                    }
                else:
                    fix_info = {"status": "not_run", "reason": "No test executed"}

        status = fix_info.get("status", "unknown")
        reason = fix_info.get("reason")
        metrics = fix_info.get("metrics")
        protocol = fix_info.get("protocol")

        # Load curves
        curve_data: dict[str, Any] = {"available": False}
        files = fix_info.get("files", {})
        curve_json_file = hys_dir / files.get("curve_json", f"{fix_name}_curve.json")
        raw_npz_file = hys_dir / files.get("npz", f"{fix_name}_raw.npz")

        # Raw npz for curves and domain evidence
        raw_npz_dict = None
        if raw_npz_file.is_file():
            raw_npz_dict, npz_err = _load_npz(raw_npz_file)
            if npz_err is not None and not curve_json_file.is_file():
                status = "corrupt"
                reason = f"Corrupt {raw_npz_file.name}: {npz_err}"

        refined_raw_npz_file = hys_dir / "refined" / f"{fix_name}_raw.npz"
        refined_npz_dict = None
        if refined_raw_npz_file.is_file():
            refined_npz_dict, _ = _load_npz(refined_raw_npz_file)

        hys_domain = _compute_hys_domain_evidence(raw_npz_dict)
        refined_hys_domain = _compute_hys_domain_evidence(refined_npz_dict)

        if curve_json_file.is_file():
            c_json, _ = _load_json(curve_json_file)
            if c_json:
                curve_data = {"available": True, **c_json}
        elif raw_npz_dict is not None and "displacement_m" in raw_npz_dict and "force_n" in raw_npz_dict:
            d = np.asarray(raw_npz_dict["displacement_m"], dtype=float)
            f = np.asarray(raw_npz_dict["force_n"], dtype=float)
            first_loop = None
            if "first_loop_displacement_m" in raw_npz_dict and "first_loop_force_n" in raw_npz_dict:
                fd, ff = _decimate_curve(raw_npz_dict["first_loop_displacement_m"], raw_npz_dict["first_loop_force_n"])
                first_loop = {"displacement_m": fd, "force_n": ff}
            final_loop = None
            if "final_loop_displacement_m" in raw_npz_dict and "final_loop_force_n" in raw_npz_dict:
                ld, lf = _decimate_curve(raw_npz_dict["final_loop_displacement_m"], raw_npz_dict["final_loop_force_n"])
                final_loop = {"displacement_m": ld, "force_n": lf}

            all_d, all_f = _decimate_curve(d, f, max_points=250)
            curve_data = {
                "available": True,
                "fixture": fix_name,
                "first_loop": first_loop,
                "final_loop": final_loop,
                "all_cycles": {"displacement_m": all_d, "force_n": all_f},
                "metrics": metrics,
            }

        fixtures_data[fix_name] = {
            "status": status,
            "reason": reason,
            "protocol": protocol,
            "metrics": metrics,
            "curve": curve_data,
            "comparison": comp_fixtures.get(fix_name),
            "domain_evidence": hys_domain,
            "refined_domain_evidence": refined_hys_domain,
        }

    return {
        "status": summary.get("status", "unknown" if fixtures_data else "not_run"),
        "fixtures": fixtures_data,
        "comparison": comp_data,
    }


def write_report(campaign: Path) -> Path:
    """Read experiment campaign results and render offline HTML report, results.json, and results.csv.

    Args:
        campaign: Directory containing plan.json, baseline, and conditions.

    Returns:
        Path to generated report.html.
    """
    campaign = Path(campaign).resolve()
    plan_file = campaign / "plan.json"
    if not plan_file.is_file():
        raise FileNotFoundError(f"Missing plan.json in campaign directory: {campaign}")

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    conditions_plan = plan.get("conditions", [])

    # Fixtures requested by plan
    requested_fixtures = plan.get("hysteresis", {}).get("requested_fixtures")
    if not requested_fixtures:
        requested_fixtures = ["rearfoot_punch", "fullfoot_last"]

    # Load baseline reference lengths from campaign/baseline/reference.npz (or campaign/reference.npz)
    reference_lengths_m = None
    motion_length_reason = None
    ref_npz_path = campaign / "baseline/reference.npz"
    if not ref_npz_path.is_file():
        ref_npz_path = campaign / "reference.npz"

    if ref_npz_path.is_file():
        ref_data, ref_err = _load_npz(ref_npz_path)
        if ref_err is not None:
            motion_length_reason = f"Corrupt reference.npz: {ref_err}"
        elif ref_data and "lengths_m" in ref_data:
            reference_lengths_m = np.asarray(ref_data["lengths_m"], dtype=float).tolist()
        else:
            motion_length_reason = "Missing 'lengths_m' in reference.npz"
    else:
        motion_length_reason = "Reference file campaign/baseline/reference.npz is absent; no fallback lengths assumed."

    # Baseline overlay MUST use conditions/baseline/primary/native current replay
    conditions_dir = campaign / "conditions"
    current_baseline_primary = conditions_dir / "baseline/primary"
    baseline_motion: dict[str, Any] = {"available": False}
    b_trace = None

    if (current_baseline_primary / "native/trace.npz").is_file():
        b_trace, b_err = _load_npz(current_baseline_primary / "native/trace.npz")
        if b_err is not None:
            baseline_motion = {"available": False, "reason": f"Corrupt baseline trace.npz: {b_err}"}
        elif b_trace is not None and "time_s" in b_trace and "state" in b_trace and len(b_trace["time_s"]) > 0:
            b_t = np.asarray(b_trace["time_s"], dtype=float)
            b_st = np.asarray(b_trace["state"], dtype=float)
            b_grf = np.asarray(b_trace.get("grf_n", []), dtype=float)
            dec_bt, dec_bst = _decimate_series(b_t, b_st, max_points=120)
            curves: dict[str, Any] = {"time_s": dec_bt}
            if b_st.ndim == 2 and b_st.shape[1] >= 5:
                _, hip_x = _decimate_series(b_t, b_st[:, 0], max_points=120)
                _, hip_z = _decimate_series(b_t, b_st[:, 1], max_points=120)
                _, knee = _decimate_series(b_t, b_st[:, 3], max_points=120)
                _, ankle = _decimate_series(b_t, b_st[:, 4], max_points=120)
                curves["hip_x_m"] = hip_x
                curves["hip_z_m"] = hip_z
                curves["knee_rad"] = knee
                curves["ankle_rad"] = ankle
            if b_grf.ndim == 2 and b_grf.shape[1] >= 2:
                _, grf_x = _decimate_series(b_t, b_grf[:, 0], max_points=120)
                _, grf_z = _decimate_series(b_t, b_grf[:, 1], max_points=120)
                curves["grf_x_n"] = grf_x
                curves["grf_z_n"] = grf_z

            baseline_motion = {
                "available": True,
                "time_s": dec_bt,
                "state": dec_bst,
                "curves": curves,
            }
    else:
        baseline_motion = {
            "available": False,
            "reason": "Current baseline replay conditions/baseline/primary/native is absent; historical baseline cannot be substituted.",
        }

    baseline_summary, _ = _load_json(current_baseline_primary / "native/summary.json")
    baseline_refined_trace, _ = _load_npz(current_baseline_primary / "refined/trace.npz")
    baseline_refined_summary, _ = _load_json(current_baseline_primary / "refined/summary.json")
    baseline_actuation = {
        "baseline_trace": b_trace,
        "baseline_run": baseline_summary.get("run") if baseline_summary else None,
        "baseline_refined_trace": baseline_refined_trace,
        "baseline_refined_run": baseline_refined_summary.get("run") if baseline_refined_summary else None,
    }

    # Baseline hysteresis MUST use conditions/baseline/hysteresis
    current_baseline_hys_dir = conditions_dir / "baseline/hysteresis"
    if current_baseline_hys_dir.exists():
        baseline_hys = _extract_hysteresis_data(current_baseline_hys_dir, requested_fixtures)
    else:
        baseline_hys = {
            "status": "not_run",
            "reason": "Current baseline hysteresis conditions/baseline/hysteresis is absent.",
            "fixtures": {
                fix: {
                    "status": "not_run",
                    "reason": "Baseline hysteresis not run",
                    "metrics": None,
                    "curve": {"available": False},
                }
                for fix in requested_fixtures
            },
        }

    # Read each planned condition
    conditions_data: list[dict[str, Any]] = []
    results_records: list[dict[str, Any]] = []
    actuation_records: list[dict[str, Any]] = []
    actuation_results: list[dict[str, Any]] = []

    for cond_plan in conditions_plan:
        cond_id = str(cond_plan.get("id"))
        cond_dir = conditions_dir / cond_id

        # Read condition.json if available, or fall back to plan entry
        cond_json_file = cond_dir / "condition.json"
        cond_meta, _ = _load_json(cond_json_file)
        if cond_meta is None:
            cond_meta = dict(cond_plan)
        else:
            cond_meta = copy.deepcopy(cond_meta)

        # Sanitize any manuscript_note in metadata to avoid local file paths
        if "paper_material" in cond_meta and isinstance(cond_meta["paper_material"], dict):
            sources = cond_meta["paper_material"].get("sources")
            if isinstance(sources, dict) and "manuscript_note" in sources:
                sources["manuscript_note"] = (
                    "Verified from authors' arXiv manuscript source (EWC26.tex); publisher automated access blocked."
                )

        # Primary protocol
        primary_dir = cond_dir / "primary"
        if primary_dir.exists():
            primary_data = _extract_protocol_data(
                primary_dir, cond_id, "primary", campaign, baseline_actuation=baseline_actuation
            )
        else:
            primary_data = {
                "status": "not_run",
                "failure_reason": None,
                "last_valid_time_s": None,
                "completed_fraction": None,
                "integrated_duration_s": None,
                "requested_duration_s": None,
                "native_report_link": None,
                "summary": None,
                "refined_summary": None,
                "comparison": None,
                "refinement": None,
                "baseline_delta": None,
                "motion": {"available": False, "reason": "Condition not run"},
                "plot_curves": {"available": False},
                "metrics": None,
                "observations": None,
                "run": None,
            }

        # Clearance matched protocol (optional)
        clearance_dir = cond_dir / "clearance_matched"
        clearance_data = None
        if clearance_dir.exists():
            clearance_data = _extract_protocol_data(
                clearance_dir, cond_id, "clearance_matched", campaign, baseline_actuation=baseline_actuation
            )

        # Hysteresis bench
        hys_dir = cond_dir / "hysteresis"
        if hys_dir.exists():
            hys_data = _extract_hysteresis_data(hys_dir, requested_fixtures)
        else:
            hys_data = {
                "status": "not_run",
                "fixtures": {
                    fix: {
                        "status": "blocked" if fix == "forefoot_last" else "not_run",
                        "reason": "Fixture 'forefoot_last' not present in artifact instron_fixtures"
                        if fix == "forefoot_last"
                        else "Bench test not run",
                        "metrics": None,
                        "curve": {"available": False},
                    }
                    for fix in requested_fixtures
                },
            }

        # Check for condition-level native report.html if at cond_dir / report.html
        cond_report_file = cond_dir / "report.html"
        cond_report_link = primary_data.get("native_report_link")
        if cond_report_file.is_file() and not cond_report_link:
            try:
                cond_report_link = str(cond_report_file.relative_to(campaign))
            except ValueError:
                cond_report_link = str(cond_report_file)

        paper_mat = cond_meta.get("paper_material")
        if paper_mat and isinstance(paper_mat, dict):
            paper_mat = copy.deepcopy(paper_mat)
            if "sources" in paper_mat and isinstance(paper_mat["sources"], dict):
                paper_mat["sources"]["manuscript_note"] = (
                    "Verified from authors' arXiv manuscript source (EWC26.tex); publisher automated access blocked."
                )

        condition_entry = {
            "id": cond_id,
            "family": cond_meta.get("family", "unknown"),
            "factor": cond_meta.get("factor", "unknown"),
            "scale": cond_meta.get("scale"),
            "description": cond_meta.get("description", ""),
            "metadata": cond_meta,
            "paper_material": paper_mat,
            "report_link": cond_report_link,
            "primary": primary_data,
            "clearance_matched": clearance_data,
            "hysteresis": hys_data,
        }
        conditions_data.append(condition_entry)
        for protocol_name, protocol_data in (("primary", primary_data), ("clearance_matched", clearance_data)):
            if protocol_data is None:
                continue
            actuation_records.append(_actuation_record(cond_id, protocol_name, protocol_data))
            actuation_results.append(
                {"condition_id": cond_id, "protocol": protocol_name, "actuation": protocol_data.get("actuation")}
            )

        # Classification outcome distinguishing numerically qualified vs unqualified / unresolved
        p_status = primary_data["status"]
        p_fail = primary_data["failure_reason"]
        p_last_t = primary_data["last_valid_time_s"]
        p_frac = primary_data["completed_fraction"]

        ref = primary_data.get("refinement")
        outcome = "not_run"
        if p_status == "corrupt":
            outcome = "corrupt"
        elif p_status == "completed":
            if ref is None:
                outcome = "completed_unrefined"
            elif ref.get("passed") is True:
                outcome = (
                    "diagnostic_step_consistent"
                    if (primary_data.get("summary") or {}).get("diagnostic_unqualified")
                    else "completed_step_consistent"
                )
            else:
                outcome = "numerically_unresolved"
        elif p_status in ("stopped", "failed", "terminated"):
            outcome = "stopped"
        elif p_status != "not_run":
            outcome = p_status

        # Motion metrics
        obs = primary_data.get("observations") or {}
        b_delta = primary_data.get("baseline_delta") or {}

        rec = {
            "condition_id": cond_id,
            "family": cond_meta.get("family"),
            "factor": cond_meta.get("factor"),
            "scale": cond_meta.get("scale"),
            "outcome": outcome,
            "primary_status": p_status,
            "diagnostic_unqualified": (primary_data.get("summary") or {}).get("diagnostic_unqualified", False),
            "failure_reason": p_fail or "",
            "last_valid_time_s": p_last_t,
            "completed_fraction": p_frac,
            "peak_vertical_grf_z_n": _safe_float(obs.get("peak_vertical_grf_n")),
            "minimum_hip_z_m": _safe_float(obs.get("minimum_hip_height_m")),
            "actuator_work_signed_j": _safe_float(obs.get("actuator_work_signed_j")),
            "passive_cap_steps": obs.get("passive_cap_steps"),
            "refinement_passed": ref.get("passed") if ref else None,
            "refinement_max_hip_pos_diff_m": _safe_float(ref.get("maximum_hip_position_difference_m")) if ref else None,
            "baseline_motion_rms": _plain(b_delta.get("motion", {}).get("rms")) if isinstance(b_delta, dict) else None,
            "baseline_grf_rms": _plain(b_delta.get("grf", {}).get("rms")) if isinstance(b_delta, dict) else None,
        }

        for channel in ("knee", "ankle", "hip", "total"):
            for metric in ("peak_abs_torque_nm", "rms_torque_nm", "positive_work_j", "absorbed_work_j", "net_work_j"):
                rec[f"{channel}_{metric}"] = _safe_float(
                    ((primary_data.get("actuation") or {}).get("channels") or {}).get(channel, {}).get(metric)
                )

        # Add hysteresis metrics for requested fixtures
        for fix_name in requested_fixtures:
            f_data = hys_data.get("fixtures", {}).get(fix_name, {})
            f_mets = f_data.get("metrics") or {}
            f_comp = f_data.get("comparison") or {}
            rec[f"{fix_name}_status"] = f_data.get("status", "not_run")
            rec[f"{fix_name}_work_net_j"] = _safe_float(f_mets.get("work_net_j"))
            rec[f"{fix_name}_loss_ratio"] = _safe_float(f_mets.get("hysteresis_loss_ratio"))
            rec[f"{fix_name}_peak_force_n"] = _safe_float(f_mets.get("peak_force_n"))
            rec[f"{fix_name}_refinement_passed"] = f_comp.get("passed") if f_comp else None
            rec[f"{fix_name}_max_force_diff_n"] = (
                _safe_float(f_comp.get("maximum_force_difference_n")) if f_comp else None
            )
            if f_data.get("reason"):
                rec[f"{fix_name}_reason"] = f_data["reason"]

        # Strain domain evidence (literature limit 0.60 vs runtime solver cap 0.90)
        p_dom = primary_data.get("domain_evidence") or {}
        if p_dom.get("available"):
            rec["primary_domain_max_driven_strain"] = _safe_float(p_dom.get("max_driven_strain"))
            rec["primary_domain_max_passive_strain"] = _safe_float(p_dom.get("max_passive_strain"))
            rec["primary_domain_max_any_strain"] = _safe_float(p_dom.get("max_any_strain"))
            rec["primary_domain_exceeds_limit"] = p_dom.get("exceeds_literature_limit")
            rec["primary_domain_first_exceed_time_s"] = _safe_float(p_dom.get("first_exceed_time_s"))
            rec["primary_domain_exceed_step_fraction"] = _safe_float(p_dom.get("exceed_step_fraction"))

        for fix_name in requested_fixtures:
            f_dom = (hys_data.get("fixtures", {}).get(fix_name, {}) or {}).get("domain_evidence") or {}
            if f_dom.get("available"):
                rec[f"{fix_name}_domain_max_all_strain"] = _safe_float(f_dom.get("max_all_columns_strain"))
                rec[f"{fix_name}_domain_max_passive_strain"] = _safe_float(f_dom.get("max_passive_strain"))
                rec[f"{fix_name}_domain_exceeds_limit"] = f_dom.get("exceeds_literature_limit")
                rec[f"{fix_name}_domain_first_exceed_time_s"] = _safe_float(f_dom.get("first_exceed_time_s"))
                rec[f"{fix_name}_domain_exceed_step_fraction"] = _safe_float(f_dom.get("exceed_step_fraction"))

        results_records.append(rec)

    # Clean up results for results.json and results.csv (guarantee no NaN)
    results_json_path = campaign / "results.json"
    results_csv_path = campaign / "results.csv"

    plain_records = _plain(results_records)
    results_json_path.write_text(json.dumps(plain_records, indent=2, allow_nan=False), encoding="utf-8")

    # Write CSV
    if results_records:
        fieldnames = list(results_records[0].keys())
        with open(results_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for r in plain_records:
                writer.writerow([_format_cell(r.get(f)) for f in fieldnames])
    else:
        results_csv_path.write_text("", encoding="utf-8")

    (campaign / "actuation.json").write_text(json.dumps(_plain(actuation_results), allow_nan=False), encoding="utf-8")
    with (campaign / "actuation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        columns = list(dict.fromkeys(key for row in actuation_records for key in row))
        writer.writerow(columns)
        for row in actuation_records:
            writer.writerow([_format_cell(_plain(row.get(key))) for key in columns])

    # Render HTML report
    report_html_path = campaign / "report.html"

    # Baseline constitutive parameters and compression curve for reference
    baseline_shoe_file = campaign / "baseline/digital_shoe.json"
    if not baseline_shoe_file.is_file():
        baseline_shoe_file = conditions_dir / "baseline/digital_shoe.json"
    b_shoe_dict, _ = _load_json(baseline_shoe_file) if baseline_shoe_file.is_file() else (None, None)
    b_params = (b_shoe_dict or {}).get("constitutive_model", {}).get("parameters")

    if (
        b_params is not None
        and "equilibrium_fraction" in b_params
        and "instantaneous_shear_modulus_pa" in b_params
        and "instantaneous_shear_modulus_2_pa" in b_params
        and "hyperfoam_exponent" in b_params
        and "hyperfoam_exponent_2" in b_params
    ):
        b_feq = float(b_params["equilibrium_fraction"])
        b_mu1_inst = float(b_params["instantaneous_shear_modulus_pa"])
        b_mu2_inst = float(b_params["instantaneous_shear_modulus_2_pa"])
        b_alpha1 = float(b_params["hyperfoam_exponent"])
        b_alpha2 = float(b_params["hyperfoam_exponent_2"])
        b_mu1_eq = b_mu1_inst * b_feq
        b_mu2_eq = b_mu2_inst * b_feq
        b_tau = float(b_params.get("maxwell_relaxation_time_s", 0.0))

        std_strains = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        b_curve_kpa = _calc_ogden_hill_pressure_kpa(std_strains, b_mu1_eq, b_alpha1, b_mu2_eq, b_alpha2)

        baseline_paper_compression = {
            "available": True,
            "label": "Baseline Shoe Artifact",
            "parameters": {
                "mu1_eq_pa": b_mu1_eq,
                "alpha1": b_alpha1,
                "mu2_eq_pa": b_mu2_eq,
                "alpha2": b_alpha2,
                "mu_eq_sum_pa": b_mu1_eq + b_mu2_eq,
                "mu1_inst_pa": b_mu1_inst,
                "mu2_inst_pa": b_mu2_inst,
                "feq": b_feq,
                "tau_s": b_tau,
            },
            "table_strain": std_strains,
            "pressure_curve_kpa": b_curve_kpa,
            "stress_50pct_kpa": b_curve_kpa[10] if len(b_curve_kpa) > 10 else None,
            "note": "Reference unconfined compression response from baseline shoe artifact parameters.",
        }
    else:
        baseline_paper_compression = {
            "available": False,
            "label": "Baseline Shoe Artifact",
            "parameters": {},
            "table_strain": [],
            "pressure_curve_kpa": [],
            "stress_50pct_kpa": None,
            "note": "Baseline shoe artifact constitutive parameters unavailable.",
        }

    # Build paper_materials.json and paper_materials.csv summary
    paper_materials_records: list[dict[str, Any]] = []
    paper_materials_summary: dict[str, Any] = {
        "campaign_id": plan.get("campaign_id", campaign.name),
        "suite": plan.get("suite", "unknown"),
        "baseline_reference": baseline_paper_compression,
        "conditions": {},
    }

    for c in conditions_data:
        cid = c["id"]
        p_mat = c.get("paper_material")
        prim = c.get("primary") or {}
        p_dom = prim.get("domain_evidence") or {}
        p_dom_ref = prim.get("refined_domain_evidence") or {}

        cond_entry = {
            "condition_id": cid,
            "family": c.get("family"),
            "paper_material": p_mat,
            "primary": {
                "native": p_dom,
                "refined": p_dom_ref,
            },
            "hysteresis": {},
        }
        for fix_name in requested_fixtures:
            f_data = (c.get("hysteresis") or {}).get("fixtures", {}).get(fix_name, {})
            cond_entry["hysteresis"][fix_name] = {
                "native": f_data.get("domain_evidence") or {},
                "refined": f_data.get("refined_domain_evidence") or {},
            }
        paper_materials_summary["conditions"][cid] = cond_entry

        # Flatten records for CSV
        # Record for primary native
        paper_materials_records.append(
            {
                "condition_id": cid,
                "protocol": "primary",
                "resolution": "native",
                "fixture": "stance",
                "material_name": (p_mat or {}).get(
                    "material_name", "baseline" if c.get("family") == "baseline" else "-"
                ),
                "literature_limit_strain": 0.60,
                "max_driven_strain": p_dom.get("max_driven_strain"),
                "max_passive_strain": p_dom.get("max_passive_strain"),
                "max_any_strain": p_dom.get("max_any_strain"),
                "exceeds_literature_limit": p_dom.get("exceeds_literature_limit"),
                "first_exceed_time_s": p_dom.get("first_exceed_time_s"),
                "exceed_step_fraction": p_dom.get("exceed_step_fraction"),
            }
        )
        # Record for primary refined
        if p_dom_ref.get("available"):
            paper_materials_records.append(
                {
                    "condition_id": cid,
                    "protocol": "primary",
                    "resolution": "half_step",
                    "fixture": "stance",
                    "material_name": (p_mat or {}).get(
                        "material_name", "baseline" if c.get("family") == "baseline" else "-"
                    ),
                    "literature_limit_strain": 0.60,
                    "max_driven_strain": p_dom_ref.get("max_driven_strain"),
                    "max_passive_strain": p_dom_ref.get("max_passive_strain"),
                    "max_any_strain": p_dom_ref.get("max_any_strain"),
                    "exceeds_literature_limit": p_dom_ref.get("exceeds_literature_limit"),
                    "first_exceed_time_s": p_dom_ref.get("first_exceed_time_s"),
                    "exceed_step_fraction": p_dom_ref.get("exceed_step_fraction"),
                }
            )
        # Records for hysteresis fixtures
        for fix_name in requested_fixtures:
            f_data = (c.get("hysteresis") or {}).get("fixtures", {}).get(fix_name, {})
            f_dom = f_data.get("domain_evidence") or {}
            f_dom_ref = f_data.get("refined_domain_evidence") or {}
            if f_dom.get("available"):
                paper_materials_records.append(
                    {
                        "condition_id": cid,
                        "protocol": "hysteresis",
                        "resolution": "native",
                        "fixture": fix_name,
                        "material_name": (p_mat or {}).get(
                            "material_name", "baseline" if c.get("family") == "baseline" else "-"
                        ),
                        "literature_limit_strain": 0.60,
                        "max_driven_strain": None,
                        "max_passive_strain": f_dom.get("max_passive_strain"),
                        "max_any_strain": f_dom.get("max_all_columns_strain"),
                        "exceeds_literature_limit": f_dom.get("exceeds_literature_limit"),
                        "first_exceed_time_s": f_dom.get("first_exceed_time_s"),
                        "exceed_step_fraction": f_dom.get("exceed_step_fraction"),
                    }
                )
            if f_dom_ref.get("available"):
                paper_materials_records.append(
                    {
                        "condition_id": cid,
                        "protocol": "hysteresis",
                        "resolution": "half_step",
                        "fixture": fix_name,
                        "material_name": (p_mat or {}).get(
                            "material_name", "baseline" if c.get("family") == "baseline" else "-"
                        ),
                        "literature_limit_strain": 0.60,
                        "max_driven_strain": None,
                        "max_passive_strain": f_dom_ref.get("max_passive_strain"),
                        "max_any_strain": f_dom_ref.get("max_all_columns_strain"),
                        "exceeds_literature_limit": f_dom_ref.get("exceeds_literature_limit"),
                        "first_exceed_time_s": f_dom_ref.get("first_exceed_time_s"),
                        "exceed_step_fraction": f_dom_ref.get("exceed_step_fraction"),
                    }
                )

    (campaign / "paper_materials.json").write_text(
        json.dumps(_plain(paper_materials_summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    if paper_materials_records:
        pm_cols = list(dict.fromkeys(key for row in paper_materials_records for key in row))
        with (campaign / "paper_materials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(pm_cols)
            for row in paper_materials_records:
                writer.writerow([_format_cell(_plain(row.get(k))) for k in pm_cols])
    has_paper_materials = any(bool(c.get("paper_material")) for c in conditions_data)

    report_payload = _plain(
        {
            "campaign_id": plan.get("campaign_id", campaign.name),
            "suite": plan.get("suite", "unknown"),
            "reference_lengths_m": reference_lengths_m,
            "motion_length_reason": motion_length_reason,
            "baseline_motion": baseline_motion,
            "baseline_hys": baseline_hys,
            "baseline_paper_compression": baseline_paper_compression,
            "has_paper_materials": has_paper_materials,
            "requested_fixtures": requested_fixtures,
            "conditions": conditions_data,
            "results": plain_records,
        }
    )

    # Literal replacement with "\u003c" and "\u0026" so browser JS receives clean escaped characters
    # and prevents </script> tags in payload from breaking HTML script block
    raw_json_str = json.dumps(report_payload, allow_nan=False)
    encoded_data = raw_json_str.replace("<", r"\u003c").replace("&", r"\u0026")

    # Generate static summary tables for pure offline HTML view
    # Dynamic table headers based on requested_fixtures
    fixture_headers = "".join(f"<th>{html.escape(fix)} Hys</th>" for fix in requested_fixtures)

    summary_rows = []
    for c in conditions_data:
        cid = html.escape(str(c["id"]))
        fam = html.escape(str(c["family"]))
        fac = html.escape(str(c["factor"]))
        sc = html.escape(str(c["scale"]) if c["scale"] is not None else "-")

        fail = html.escape(str(c["primary"]["failure_reason"] or "-"))
        last_t = f"{c['primary']['last_valid_time_s']:.3f} s" if c["primary"]["last_valid_time_s"] is not None else "-"
        frac = (
            f"{c['primary']['completed_fraction'] * 100:.1f}%"
            if c["primary"]["completed_fraction"] is not None
            else "-"
        )

        # Determine outcome badge
        rec_for_c = next((r for r in results_records if r["condition_id"] == c["id"]), {})
        outcome_val = rec_for_c.get("outcome", "not_run")
        outcome_badge = f"<span class='badge badge-{outcome_val}'>{outcome_val.replace('_', ' ')}</span>"

        # Fixture cells
        fix_cells = []
        for fix in requested_fixtures:
            f_st = c["hysteresis"]["fixtures"].get(fix, {}).get("status", "not_run")
            fix_cells.append(f"<td><span class='badge badge-{f_st}'>{f_st}</span></td>")
        fix_cells_html = "".join(fix_cells)

        rlink = (
            f"<a href='{html.escape(c['report_link'])}' target='_blank'>Native Report</a>"
            if c.get("report_link")
            else "-"
        )

        row_cls = f"row-{outcome_val}"
        summary_rows.append(
            f"<tr class='{row_cls}'>"
            f"<td><b>{cid}</b></td><td>{fam}</td><td>{fac}</td><td>{sc}</td>"
            f"<td>{outcome_badge}</td>"
            f"<td>{frac}</td><td>{last_t}</td><td>{fail}</td>"
            f"{fix_cells_html}<td>{rlink}</td>"
            f"</tr>"
        )

    shared_map_js = (Path(__file__).resolve().parents[1] / "cartesian/shoe_map.js").read_text()
    rendered_html = _HTML_TEMPLATE.replace("__SHOE_MAP_JS__", shared_map_js)
    rendered_html = rendered_html.replace("__PAYLOAD__", encoded_data)
    rendered_html = rendered_html.replace("__FIXTURE_HEADERS__", fixture_headers)
    rendered_html = rendered_html.replace("__TABLE_ROWS__", "\n".join(summary_rows))
    rendered_html = rendered_html.replace("__CAMPAIGN_NAME__", html.escape(campaign.name))

    report_html_path.write_text(rendered_html, encoding="utf-8")
    return report_html_path


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Campaign Report — __CAMPAIGN_NAME__</title>
<style>
:root{--bg:#f8fafc;--card:#ffffff;--border:#e2e8f0;--text:#1e293b;--text-muted:#64748b;--primary:#2563eb;--green:#16a34a;--red:#dc2626;--orange:#ea580c;--blue:#0284c7}
*{box-sizing:border-box}
body{font:14px system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}
main{max-width:1400px;margin:auto}
h1{font-size:24px;margin:0 0 8px;color:#0f172a}
h2{font-size:18px;margin:0 0 12px;color:#1e293b}
h3{font-size:15px;margin:0 0 8px;color:#334155}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:18px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.header-bar{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid var(--border);padding-bottom:12px;margin-bottom:18px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}
.badge-completed_qualified{background:#dcfce7;color:#15803d}
.badge-completed_unrefined{background:#e0f2fe;color:#0369a1}
.badge-numerically_unresolved{background:#fef3c7;color:#b45309}
.badge-completed{background:#dcfce7;color:#15803d}
.badge-stopped,.badge-failed,.badge-terminated{background:#fee2e2;color:#b91c1c}
.badge-corrupt{background:#fce7f3;color:#be185d}
.badge-not_run{background:#f1f5f9;color:#64748b}
.badge-blocked{background:#fef3c7;color:#b45309}
table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
th{background:#f8fafc;font-weight:600;color:var(--text-muted)}
tr:hover{background:#f1f5f9}
.row-stopped td{background:#fff5f5}
.row-numerically_unresolved td{background:#fffbeb}
.row-corrupt td{background:#fdf2f8}
.row-notrun td,.row-not_run td{color:#94a3b8}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px;background:#f1f5f9;border-radius:6px;margin-bottom:16px}
select,input[type=range],button{padding:6px 10px;border:1px solid #cbd5e1;border-radius:4px;background:white;font:inherit}
button{cursor:pointer;background:var(--primary);color:white;border:0;font-weight:600}
button:hover{background:#1d4ed8}
.grid-2col{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:960px){.grid-2col{grid-template-columns:1fr}}
canvas{width:100%;background:#ffffff;border:1px solid var(--border);border-radius:6px}
.chart-container{width:100%;height:220px;background:#ffffff;border:1px solid var(--border);border-radius:6px;margin-bottom:12px}
.legend{font-size:12px;color:var(--text-muted);margin-top:6px}
pre{background:#f8fafc;border:1px solid var(--border);padding:10px;border-radius:4px;overflow-x:auto;font:12px ui-monospace,monospace}
.tag{display:inline-block;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px}
.warning-box{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;border-radius:4px;margin-bottom:12px;font-size:13px}
.danger-box{background:#fef2f2;border-left:4px solid #ef4444;padding:10px 14px;border-radius:4px;margin-bottom:12px;font-size:13px}
.overlay-guide{background:#f8fafc;border:1px solid var(--border);padding:10px 14px;margin:12px 0;font-size:13px;line-height:1.5}
.overlay-guide p{margin:0 0 8px}.overlay-guide ul{margin:0;padding-left:20px}
.playback-controls{margin:6px 0 10px;padding:10px 12px;background:#eef3f7;border:1px solid var(--border);border-radius:6px}
.heatmap-pair{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:12px;overflow-x:auto}
.heatmap-panel{min-width:0;border:1px solid var(--border);border-radius:6px;padding:10px;background:white}
.heatmap-panel h3{margin:0 0 6px;font-size:14px;overflow-wrap:anywhere}
.heatmap-panel canvas{display:block;width:100%;height:260px}
.heatmap-panel .legend{min-height:2.2em}
.heatmap-color-ramp{height:10px;max-width:380px;background:linear-gradient(to right,#0000ff 0%,#00ffff 33.333%,#ffff00 66.667%,#ff0000 100%);border:1px solid #a2b1b9}
.heatmap-color-ticks{display:flex;justify-content:space-between;max-width:380px;font-size:11px;margin:3px 0 10px}
.table-scroll{overflow-x:auto}.metric-delta{display:block;color:#64748b;font-size:11px}.selected-metric-row{background:#eef5ff}.sensitive-metric{color:#b45309;font-weight:600}
.overlay-guide summary{cursor:pointer}.overlay-guide[open] summary{margin-bottom:10px}
.native-scene-wrap{position:relative;display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.native-scene-wrap.side-by-side{grid-template-columns:repeat(2,minmax(0,1fr))}
.native-scene-wrap iframe{width:100%;height:1100px;border:1px solid var(--border);border-radius:6px;background:white}
.native-scene-wrap iframe.native-data-source{position:absolute;width:100%;height:1100px;opacity:0;pointer-events:none;left:0;top:0}
@media(max-width:900px){.native-scene-wrap.side-by-side{grid-template-columns:minmax(0,1fr)}}
</style>
</head>
<body>
<main>
<div class="header-bar">
  <div>
    <h1>Campaign Report: __CAMPAIGN_NAME__</h1>
    <div class="legend">Offline multi-condition sensitivity evaluation with side-by-side motion playback, baseline overlays, and raw bench hysteresis curves.</div>
  </div>
</div>


<div class="card" id="paper_material_card" style="display:none;">
  <h2>Paper-Informed Midsole Compression Surrogates & Strain Domain Evidence</h2>
  <div class="warning-box" id="paper_assumptions_box" style="margin-bottom:14px;">
    <b>Surrogate Assumptions & Provenance:</b>
    <ul style="margin:6px 0 0;padding-left:20px;font-size:12.5px;line-height:1.5;">
      <li><b>Hyperelastic surrogate:</b> 2-term compressible Ogden-Hill parameters fit to unconfined compression mean loading/unloading response at 0.25 s<sup>-1</sup> (McCulloch, Delp, Kuhl 2026; Engineering with Computers / arXiv:2602.12694v2; verified from authors' arXiv manuscript source; publisher automated access blocked).</li>
      <li><b>Not authors' full CANN:</b> This is an engineering Ogden-Hill surrogate approximation, not the authors' richer CANN models (which screened 14 candidate strain energy terms down to 3 terms under sparse regularization; publisher automated access blocked).</li>
      <li><b>Viscoelasticity:</b> The paper did not model Maxwell branch relaxation; baseline shoe viscoelasticity (equilibrium fraction f<sub>eq</sub> and relaxation time &tau;) is held fixed as an explicit assumption. Instantaneous moduli are converted via &mu;<sub>inst</sub> = &mu;<sub>eq</sub> / f<sub>eq</sub>.</li>
      <li><b>Shear coupling:</b> Inferred Pasternak foundation shear modulus is pinned to the surrogate equilibrium modulus sum &mu;<sub>eq</sub>, not the paper's full nonlinear simple shear response.</li>
      <li><b>Literature strain domain (60%):</b> Experimental calibration range covers compressive strain &epsilon; &le; 0.60 (&lambda; &ge; 0.40). The runtime simulation solver cap remains 0.90 (90% compression). Strains above 0.60 represent model extrapolation beyond experimental calibration.</li>
    </ul>
  </div>

  <div id="paper_domain_warning_box" class="warning-box" style="display:none;background:#fff7ed;border-left:4px solid #ea580c;color:#9a3412;"></div>

  <div class="grid-2col">
    <div>
      <h3>Unconfined Compression Response [kPa vs Strain]</h3>
      <svg id="plot_paper_compression" class="chart-container" style="height:280px;" viewBox="0 0 540 280"></svg>
      <div class="legend" id="paper_plot_legend" style="text-align:center;">
        Blue: Left surrogate · Orange: Right surrogate · Gray dashed: Baseline shoe artifact · Red dashed: 60% literature domain limit (lines: surrogate equilibrium curves; dots: published cycle-averaged stresses)
      </div>
    </div>
    <div>
      <h3>Material Parameters, Fit Quality & Literature Limits</h3>
      <div id="paper_param_table_pane" class="table-scroll"></div>
    </div>
  </div>
</div>

<div class="card">
  <h2>Condition Selection & Inspection</h2>
  <div class="controls">
    <label><b>Left condition:</b> <select id="condition_select"></select></label>
    <label><b>Right condition:</b> <select id="comparison_select"></select></label>
    <label><b>Protocol:</b> <select id="protocol_select"><option value="primary">Primary</option><option value="clearance_matched">Clearance Matched</option></select></label>
    <span id="condition_info" style="font-weight:600;margin-left:auto;"></span>
  </div>
  <div id="condition_meta_pane"></div>
</div>

<div class="card">
  <h2>Native Motion and Shoe Visuals</h2>
  <div class="warning-box" id="failure_warning" style="display:none;"></div>
  <div class="danger-box" id="motion_warning" style="display:none;"></div>
  <div class="controls">
    <label><b>View:</b> <select id="native_view_mode"><option value="overlay">Overlay</option><option value="side_by_side">Side by side</option></select></label>
    <label><input type="checkbox" id="show_comparison" checked> Show comparison layer</label>
    <label><b>Comparison opacity:</b> <input type="range" id="comparison_opacity" min="5" max="100" value="25" style="width:130px;"> <span id="opacity_label">25%</span></label>
  </div>
  <details class="overlay-guide">
    <summary><b>How to read the animation and heat maps</b></summary>
    <p><b id="primary_layer_label">Solid layer: left condition.</b> <span id="comparison_layer_label">Faded layer: right condition.</span> Both layers show simulated motion. <b>No mocap layer is shown here.</b></p>
    <ul>
      <li><b>Opacity changes visibility only.</b> At 25% the comparison is faint; at 100% it is opaque. Both conditions use the same time and world coordinates. Overlay mode shares one camera; Side-by-side uses native auto-framing. No pose alignment hides their differences.</li>
      <li><b>Native shoe controls:</b> Mesh, Springs, or Both; compression in mm or strain in %; shoe-width slice; and the solved deformation map. Each layer uses its own shoe geometry and spring lengths.</li>
      <li><b>Spring color means compression, not force.</b> The mm scale is shared across the selected pair. Strain uses each spring's own rest length. The two heat maps always show the two conditions separately, at full opacity and at the displayed stance time.</li>
      <li><b>Gold is the rigid last.</b> Green CAD is undeformed context, not a deformed material surface. The solved springs show the simulated foundation, including modified geometry.</li>
      <li><b>Purple cross and dashed connector:</b> the solid condition's controller equilibrium. <b>Purple arrow:</b> its hip force. <b>Orange arrow:</b> its ground reaction force. These are not mocap inputs or measured loads.</li>
      <li><b>A stopped trajectory stays labeled stopped.</b> Saved states are not extrapolated. Both heat maps stay side by side in either animation mode.</li>
    </ul>
  </details>
  <div id="native_view_status" class="legend" role="status">Loading native reports…</div>
  <div id="native_scene_wrap" class="native-scene-wrap">
    <iframe id="native_primary_frame" title="Selected condition native report" sandbox="allow-scripts"></iframe>
    <iframe id="native_comparison_frame" title="Comparison condition native report" sandbox="allow-scripts" class="native-data-source" aria-hidden="true"></iframe>
  </div>
  <div id="playback_controls" class="controls playback-controls">
    <button id="play_btn">Play</button>
    <label style="flex:1;display:flex;align-items:center;gap:8px;">
      <b>Stance time:</b><input type="range" id="time_slider" min="0" max="1000" value="0" style="flex:1;">
      <span id="time_label" style="min-width:90px;font-family:monospace;">0.000 s</span>
    </label>
  </div>
  <section id="paired_heatmaps" aria-label="Synchronized shoe heat maps">
    <div class="controls">
      <h3 style="margin:0;">Both shoes — same stance time</h3>
      <label><b>Spring color:</b> <select id="heatmap_metric"><option value="mm">Compression [mm]</option><option value="strain">Compression / rest length [%]</option></select></label>
      <span id="heatmap_stance_time" class="legend">Waiting for native animation</span>
    </div>
    <p id="heatmap_scale_label" class="legend">Both maps share a color scale. Colors show compression, not force.</p>
    <div class="heatmap-color-ramp" aria-hidden="true"></div>
    <div id="heatmap_color_ticks" class="heatmap-color-ticks"></div>
    <div class="heatmap-pair">
      <div class="heatmap-panel">
        <h3 id="heatmap_primary_title">Left shoe</h3>
        <canvas id="heatmap_primary" aria-label="Left shoe compression map"></canvas>
        <p id="heatmap_primary_status" class="legend">Waiting for saved spring history.</p>
        <p id="heatmap_primary_hover" class="legend">Hover over a spring for its compression and rest length.</p>
      </div>
      <div class="heatmap-panel">
        <h3 id="heatmap_comparison_title">Right shoe</h3>
        <canvas id="heatmap_comparison" aria-label="Right shoe compression map"></canvas>
        <p id="heatmap_comparison_status" class="legend">Waiting for saved spring history.</p>
        <p id="heatmap_comparison_hover" class="legend">Hover over a spring for its compression and rest length.</p>
      </div>
    </div>
    <p class="legend">Maps stay side by side in both animation modes. Each uses its own spring geometry and rest lengths. Colors update after the native animation renders the matching stance frame. Squares are driven columns; circles are passive surround.</p>
  </section>
</div>

<div class="card" id="actuation_section">
  <h2>Ankle and Knee Torque, Power and Work</h2>
  <p>These are simulated controller loads and mechanical work, not muscle effort or metabolic cost. The shoe-contact moment about the ankle is a different quantity. Check hip work as well: lower ankle work can shift demand elsewhere.</p>
  <p id="actuation_pair_label" class="legend"></p>
  <div id="actuation_support" class="legend"></div>
  <div class="grid-2col">
    <div><h3>Knee actuator torque [N·m]</h3><svg id="plot_knee_torque" class="chart-container" viewBox="0 0 540 220"></svg></div>
    <div><h3>Ankle actuator torque [N·m]</h3><svg id="plot_ankle_torque" class="chart-container" viewBox="0 0 540 220"></svg></div>
    <div><h3>Knee actuator power [W]</h3><svg id="plot_knee_power" class="chart-container" viewBox="0 0 540 220"></svg></div>
    <div><h3>Ankle actuator power [W]</h3><svg id="plot_ankle_power" class="chart-container" viewBox="0 0 540 220"></svg></div>
  </div>
  <p class="legend">Solid blue: left condition. Dashed gray: right condition. Purple cursor: displayed stance time. Positive knee torque extends the knee; positive ankle torque dorsiflexes. Power = torque &times; relative joint angular velocity. Positive power produces work; negative power absorbs it.</p>
  <details>
    <summary><b>Hip support and combined actuator power</b></summary>
    <div class="grid-2col">
      <div><h3>Hip point power, F · v [W]</h3><svg id="plot_hip_power" class="chart-container" viewBox="0 0 540 220"></svg></div>
      <div><h3>Combined actuator net power [W]</h3><svg id="plot_total_power" class="chart-container" viewBox="0 0 540 220"></svg></div>
    </div>
  </details>
  <div class="controls">
    <h3 style="margin:0;">Cumulative mechanical work</h3>
    <label><b>Work:</b> <select id="work_kind"><option value="positive">Produced</option><option value="absorbed">Absorbed (positive magnitude)</option><option value="net">Net (produced minus absorbed)</option></select></label>
  </div>
  <div class="grid-2col">
    <div><h3 id="knee_work_title">Knee cumulative produced work [J]</h3><svg id="plot_knee_work" class="chart-container" viewBox="0 0 540 220"></svg></div>
    <div><h3 id="ankle_work_title">Ankle cumulative produced work [J]</h3><svg id="plot_ankle_work" class="chart-container" viewBox="0 0 540 220"></svg></div>
    <div><h3 id="hip_work_title">Hip point cumulative produced work [J]</h3><svg id="plot_hip_work" class="chart-container" viewBox="0 0 540 220"></svg></div>
    <div><h3 id="total_work_title">All actuators cumulative produced work [J]</h3><svg id="plot_total_work" class="chart-container" viewBox="0 0 540 220"></svg></div>
  </div>
  <h3>Selected-condition summary</h3>
  <p class="legend">Values use the native saved trace. Parentheses show change from the fixed baseline primary replay, not necessarily the right selection. Baseline changes use common saved support and are withheld for incomplete stances.</p>
  <div class="table-scroll"><table id="actuation_summary_table">
    <thead><tr><th>Condition</th><th>Actuator</th><th>Peak / RMS load</th><th>Produced work [J]<br>(Δ baseline)</th><th>Absorbed work [J]<br>(Δ baseline)</th><th>Net work [J]<br>(Δ baseline)</th></tr></thead>
    <tbody id="actuation_summary_rows"></tbody>
  </table></div>
  <h3>Differences across the campaign</h3>
  <div class="controls">
    <label><b>Actuator:</b> <select id="actuation_channel"><option value="ankle">Ankle</option><option value="knee">Knee</option><option value="hip">Hip point</option><option value="total">All four actuators</option></select></label>
    <label><b>Metric:</b> <select id="actuation_metric"></select></label>
    <label><b>Order:</b> <select id="actuation_sort"><option value="difference">Largest absolute baseline change</option><option value="condition">Campaign order</option></select></label>
    <a href="actuation.csv" download>Download actuator CSV</a><a href="actuation.json" download>JSON</a>
  </div>
  <p id="actuation_campaign_note" class="legend"></p>
  <div class="table-scroll"><table id="actuation_campaign_table">
    <thead><tr><th>Condition</th><th>Native value</th><th>Δ baseline<br>native step</th><th>Change from<br>baseline [%]</th><th>Change in Δ<br>across timesteps</th><th>Numerical sensitivity</th></tr></thead>
    <tbody id="actuation_campaign_rows"></tbody>
  </table></div>
  <details>
    <summary><b>Calculation and numerical limits</b></summary>
    <ul>
      <li>Statistics use every saved sample, before display downsampling. Display curves retain endpoints and each series' global extrema.</li>
      <li>Power is integrated over observed force support only. No force or power is extrapolated to an unsampled terminal state. Produced and absorbed work split piecewise-linear power at zero crossings; absorbed work is a positive magnitude.</li>
      <li>Hip point power is Fx·vx + Fz·vz. All-actuator produced and absorbed work sum the separate hip-x, hip-z, knee and ankle actuator contributions, so simultaneous production and absorption do not cancel across actuators. This need not equal the sum of the three displayed joint/hip-point produced-work values. The combined power curve and its peak values are net power; they can cancel across actuators.</li>
      <li>Native/half-step effects use a common interval across condition and baseline at both timesteps. A sign change or an effect no larger than its observed timestep change is flagged. Two timesteps give a sensitivity diagnostic, not a proven error bound, confidence interval, or new acceptance gate.</li>
      <li>Stopped cases show their saved prefix only. Shorter work is not evidence of a better shoe. Existing numerical and physical qualification warnings still apply.</li>
    </ul>
  </details>
</div>

<div class="card">
  <h2>Force & Kinematic Condition Overlays</h2>
  <div class="grid-2col">
    <div>
      <h3>Vertical Ground Reaction Force (GRF Fz) [N]</h3>
      <svg id="plot_grf_z" class="chart-container" viewBox="0 0 540 220"></svg>
    </div>
    <div>
      <h3>Fore-Aft Ground Reaction Force (GRF Fx) [N]</h3>
      <svg id="plot_grf_x" class="chart-container" viewBox="0 0 540 220"></svg>
    </div>
    <div>
      <h3>Hip Vertical Position (z) [m]</h3>
      <svg id="plot_hip_z" class="chart-container" viewBox="0 0 540 220"></svg>
    </div>
    <div>
      <h3>Knee Angle [rad]</h3>
      <svg id="plot_knee" class="chart-container" viewBox="0 0 540 220"></svg>
    </div>
  </div>
  <p class="legend">Solid blue line: left condition. Dashed gray line: right condition. Red cross: Premature termination / model screen event.</p>
</div>

<div class="card">
  <h2>Raw Bench Hysteresis Curves (F-vs-Displacement)</h2>
  <div class="controls">
    <label><b>Fixture:</b>
      <select id="fixture_select"></select>
    </label>
    <label><input type="checkbox" id="overlay_baseline_hys" checked> Overlay Right-Condition Hysteresis</label>
    <span id="fixture_status_badge" style="margin-left:auto;"></span>
  </div>
  <div class="grid-2col">
    <div>
      <svg id="plot_hysteresis" class="chart-container" style="height:320px;" viewBox="0 0 540 320"></svg>
      <div class="legend" id="hys_plot_legend" style="text-align:center;">Blue: left final loop · Orange: left first loop · Gray dashed: right final loop</div>
    </div>
    <div>
      <h3>Hysteresis Work & Dissipation Diagnostics</h3>
      <div id="hys_diagnostics_pane"></div>
    </div>
  </div>
</div>

<div class="card">
  <h2>All Planned Campaign Conditions</h2>
  <div style="overflow-x:auto;">
    <table>
      <thead>
        <tr>
          <th>Condition ID</th>
          <th>Family</th>
          <th>Factor</th>
          <th>Scale</th>
          <th>Outcome</th>
          <th>Completed %</th>
          <th>Valid Duration</th>
          <th>Failure / Termination</th>
          __FIXTURE_HEADERS__
          <th>Link</th>
        </tr>
      </thead>
      <tbody>
__TABLE_ROWS__
      </tbody>
    </table>
  </div>
</div>
</main>

<script>
__SHOE_MAP_JS__
</script>
<script>
const DATA = __PAYLOAD__;
const reqFixtures = DATA.requested_fixtures || ["rearfoot_punch", "fullfoot_last"];
const fixSelect = document.getElementById("fixture_select");
reqFixtures.forEach(fix => {
  const opt = document.createElement("option");
  opt.value = fix;
  opt.textContent = fix + (fix === "forefoot_last" ? " (Missing/Blocked)" : "");
  fixSelect.appendChild(opt);
});

const condSelect = document.getElementById("condition_select");
const compareSelect = document.getElementById("comparison_select");
const hasResults = c => Boolean(c.primary?.motion?.available || c.clearance_matched?.motion?.available ||
  Object.values(c.hysteresis?.fixtures || {}).some(f => f.curve?.available));
DATA.conditions.forEach((c, idx) => {
  for (const select of [condSelect, compareSelect]) {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${c.id} — ${c.primary.status}${hasResults(c) ? "" : " (not run)"}`;
    opt.disabled = !hasResults(c);
    select.appendChild(opt);
  }
});
const firstRun = DATA.conditions.findIndex(hasResults);
const baselineIndex = DATA.conditions.findIndex(c => c.family === "baseline" && hasResults(c));
const variantIndex = DATA.conditions.findIndex(c => c.family !== "baseline" && hasResults(c));
condSelect.value = String(variantIndex >= 0 ? variantIndex : Math.max(0, firstRun));
compareSelect.value = String(baselineIndex >= 0 ? baselineIndex : Math.max(0, firstRun));
let currentCond = DATA.conditions[Number(condSelect.value)] || null;
let currentCompare = DATA.conditions[Number(compareSelect.value)] || null;
let currentProtocol = "primary";
const hasClearanceData = DATA.conditions.some(c => c.clearance_matched && c.clearance_matched.status !== "not_run");
if (!hasClearanceData) {
  const pSelect = document.getElementById("protocol_select");
  if (pSelect && pSelect.closest("label")) pSelect.closest("label").style.display = "none";
}
function selectedProtocol(condition) {
  if (!condition) return null;
  return currentProtocol === "primary" || condition.family === "baseline"
    ? condition.primary : condition.clearance_matched;
}
let currentTime = 0;
let isPlaying = false;
let animReq = null;
let lastTimestamp = null;

const nativeFrames = {
  primary: {element: document.getElementById("native_primary_frame"), path:null, data:null, sequence:0},
  comparison: {element: document.getElementById("native_comparison_frame"), path:null, data:null, sequence:0}
};
function sendNative(slot, message) {
  if (slot.path) slot.element.contentWindow.postMessage(message, "*");
}
function updateNativeStatus() {
  const status = document.getElementById("native_view_status");
  const labels = [];
  for (const [role, slot] of Object.entries(nativeFrames)) {
    labels.push(`${role}: ${!slot.path ? "native report not yet generated for this protocol" : slot.data ? "native visuals loaded" : "loading"}`);
  }
  status.textContent = labels.join(" · ");
}
function configureNative() {
  const sideBySide = document.getElementById("native_view_mode").value === "side_by_side";
  const opacity = Number(document.getElementById("comparison_opacity").value) / 100;
  const show = document.getElementById("show_comparison").checked;
  document.getElementById("opacity_label").textContent = `${Math.round(opacity*100)}%`;
  document.getElementById("primary_layer_label").textContent = `Solid layer: ${currentCond?.id || "none"}.`;
  document.getElementById("comparison_layer_label").textContent = sideBySide
    ? `Right panel: ${currentCompare?.id || "none"} (solid in its own view).`
    : `Comparison layer: ${currentCompare?.id || "none"} (${show ? Math.round(opacity*100)+"% opacity" : "hidden"}).`;
  document.getElementById("native_scene_wrap").classList.toggle("side-by-side", sideBySide);
  nativeFrames.comparison.element.classList.toggle("native-data-source", !sideBySide);
  nativeFrames.comparison.element.setAttribute("aria-hidden", String(!sideBySide));
  nativeFrames.comparison.element.setAttribute("tabindex", sideBySide ? "0" : "-1");
  for (const [role, slot] of Object.entries(nativeFrames)) {
    if (!slot.data) continue;
    const isPrimary = role === "primary";
    const other = isPrimary ? nativeFrames.comparison : nativeFrames.primary;
    const config = {
      type:"newton-native-configure", experiment:true, compact:true, external_heatmaps:true,
      compression_metric: document.getElementById("heatmap_metric").value,
      primary_label: (isPrimary ? currentCond : currentCompare)?.id || "Unavailable",
      comparison_label: (isPrimary ? currentCompare : currentCond)?.id || "Unavailable",
      comparison_opacity: opacity,
      show_comparison: !sideBySide && isPrimary && show
    };
    if (slot.lastComparison !== other.data) {
      config.comparison = other.data;
      slot.lastComparison = other.data;
    }
    sendNative(slot, config);
  }
  pendingNativeFrame = null;
  renderMotionFrames();
  updateNativeStatus();
}
function loadNativeViews() {
  for (const [role, slot] of Object.entries(nativeFrames)) {
    const condition = role === "primary" ? currentCond : currentCompare;
    const data = selectedProtocol(condition);
    const path = data?.native_report_link || null;
    if (path === slot.path) continue;
    slot.path = path;
    slot.data = null;
    slot.heatmapLayer = null;
    slot.lastComparison = undefined;
    slot.sequence++;
    slot.element.style.visibility = "hidden";
    if (path) {
      const url = new URL(path, document.baseURI);
      url.hash = "experiment";
      slot.element.src = url.href;
    } else slot.element.removeAttribute("src");
  }
  pendingNativeFrame = null;
  displayedStanceTime = null;
  displayedFrameTimes = null;
  drawHeatmaps(null);
  configureNative();
}
window.addEventListener("message", event => {
  const entry = Object.entries(nativeFrames).find(([,slot]) => event.source === slot.element.contentWindow);
  if (!entry || !event.data || typeof event.data !== "object") return;
  const [role, slot] = entry;
  const message = event.data;
  if (message.type === "newton-native-ready") {
    sendNative(slot, {type:"newton-native-request", request_id:`${role}:${slot.sequence}`});
  } else if (message.type === "newton-native-data") {
    if (message.request_id !== `${role}:${slot.sequence}`) return;
    const condition = role === "primary" ? currentCond : currentCompare;
    const expected = selectedProtocol(condition)?.summary;
    if (!expected || message.data?.summary?.condition !== expected.condition || message.data?.summary?.protocol !== expected.protocol) return;
    slot.data = message.data;
    slot.heatmapLayer = heatmapLayer(message.data);
    configureNative();
  } else if (message.type === "newton-native-frame") {
    if (!pendingNativeFrame || message.frame_id !== pendingNativeFrame.id || !pendingNativeFrame.waiting.has(role)) return;
    const condition = role === "primary" ? currentCond : currentCompare;
    if (message.primary_label !== condition?.id) return;
    pendingNativeFrame.waiting.delete(role);
    pendingNativeFrame.stamps[role] = message;
    if (pendingNativeFrame.waiting.size === 0) {
      const renderedTime = pendingNativeFrame.time;
      displayedFrameTimes = {
        primary:pendingNativeFrame.stamps.primary?.primary_frame_time_s,
        comparison:pendingNativeFrame.stamps.comparison?.primary_frame_time_s ?? pendingNativeFrame.stamps.primary?.comparison_frame_time_s
      };
      pendingNativeFrame = null;
      displayedStanceTime = renderedTime;
      drawHeatmaps(renderedTime);
      updatePlotCursors(renderedTime);
      document.getElementById("time_label").textContent = renderedTime.toFixed(3) + " s";
      dispatchNativeFrame();
    }
  } else if (message.type === "newton-native-configured") {
    if (slot.data) slot.element.style.visibility = "visible";
  } else if (message.type === "newton-native-height" && Number.isFinite(message.height)) {
    if (!slot.element.classList.contains("native-data-source")) {
      const height = Math.max(280,Math.min(2400,Math.ceil(message.height)));
      if (Math.abs(parseFloat(slot.element.style.height || "1100") - height) > 2) slot.element.style.height = `${height}px`;
    }
  }
});
for (const id of ["native_view_mode", "comparison_opacity", "show_comparison", "heatmap_metric"]) {
  document.getElementById(id).oninput = configureNative;
}
let nativeFrameSequence = 0;
let pendingNativeFrame = null;
let queuedStanceTime = null;
let displayedStanceTime = null;
let displayedFrameTimes = null;
function renderMotionFrames() {
  queuedStanceTime = currentTime;
  document.getElementById("time_slider").value = Math.min(1000,Math.max(0,currentTime/0.36*1000));
  dispatchNativeFrame();
}
function dispatchNativeFrame() {
  if (pendingNativeFrame || queuedStanceTime === null) return;
  const roles = document.getElementById("native_view_mode").value === "side_by_side"
    ? ["primary", "comparison"] : ["primary"];
  if (roles.some(role => !nativeFrames[role].data)) return;
  const time = queuedStanceTime;
  queuedStanceTime = null;
  const id = ++nativeFrameSequence;
  pendingNativeFrame = {id,time,waiting:new Set(roles),stamps:{}};
  for (const role of roles) sendNative(nativeFrames[role], {type:"newton-native-time",time_s:time,frame_id:id});
}
function heatmapLayer(payload) {
  const springs = payload?.springs;
  if (!springs?.available || !springs.time_s?.length) return null;
  return {
    anchor_local_m:springs.anchor_local_m,rest_length_m:springs.rest_length_m,
    driven:springs.driven,spacing_m:springs.spacing_m,
    mount_m:payload.geometry?.mount_m || [0,0,0],
    compression:NativeShoeMap.unpack(springs.compression_m),time_s:springs.time_s,
    label:payload.summary?.condition || "Condition", mm_max:springs.color_max_mm,
    run:payload.summary?.run || {}
  };
}
function emptyHeatmap(canvas, text) {
  const ratio=window.devicePixelRatio||1,width=Math.max(1,canvas.clientWidth);
  canvas.width=Math.round(width*ratio);canvas.height=Math.round(260*ratio);
  const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.clearRect(0,0,width,260);ctx.fillStyle="#64748b";ctx.font="12px system-ui";
  ctx.textAlign="center";ctx.fillText(text,width/2,130);
}
function drawHeatmaps(time) {
  const metric=document.getElementById("heatmap_metric").value;
  const scale=Math.max(1e-9,...Object.values(nativeFrames).map(slot=>Number(slot.heatmapLayer?.mm_max)||0));
  document.getElementById("heatmap_stance_time").textContent = time === null ? "Waiting for native animation" : `Displayed stance: ${time.toFixed(3)} s`;
  document.getElementById("heatmap_scale_label").textContent = metric === "strain"
    ? "Both maps: 0 to 100% compression / each spring's own rest length. Colors show compression, not force."
    : `Both maps: 0 to ${scale.toFixed(1)} mm compression, using the same scale as the animation. Colors are not force.`;
  const colorMax=metric==="strain"?100:scale;
  document.getElementById("heatmap_color_ticks").replaceChildren(...[0,1/3,2/3,1].map(fraction=>{
    const tick=document.createElement("span");tick.textContent=(colorMax*fraction).toFixed(1)+(metric==="strain"?"%":" mm");return tick;
  }));
  for (const [role, slot] of Object.entries(nativeFrames)) {
    const condition=role==="primary"?currentCond:currentCompare;
    const canvas=document.getElementById(`heatmap_${role}`);
    const status=document.getElementById(`heatmap_${role}_status`);
    document.getElementById(`heatmap_${role}_title`).textContent = `${role==="primary"?"Left":"Right"}: ${condition?.id || "No selection"}`;
    const layer=slot.heatmapLayer;
    if (!layer || time === null) {
      slot.heatmapProjection=null;
      emptyHeatmap(canvas,!layer?"Waiting for saved spring history":"Waiting for matching animation frame");
      status.textContent=!slot.path?"No native report for this protocol.":!slot.data?"Loading native result.":!layer?(slot.data.springs?.reason || "Audited spring history unavailable."):"Waiting for rendered stance frame.";
      continue;
    }
    const savedFrameTime=displayedFrameTimes?.[role];
    const frame=NativeShoeMap.frameIndex(layer.time_s,Number.isFinite(savedFrameTime)?savedFrameTime:time);
    slot.heatmapProjection=NativeShoeMap.draw(canvas,layer,{frame,metric,mm_max:scale,height:260,slice_y_m:null});
    const stopped=layer.run.status!=="completed" && time>=Number(layer.run.integrated_duration_s||0);
    status.textContent=`${layer.rest_length_m.length} springs · saved stance frame ${layer.time_s[frame].toFixed(4)} s${stopped?" · stopped; last valid frame":""}`;
  }
}
for (const role of ["primary","comparison"]) {
  const canvas=document.getElementById(`heatmap_${role}`);
  canvas.onmousemove=event=>{
    const slot=nativeFrames[role],layer=slot.heatmapLayer,projection=slot.heatmapProjection;
    if (!layer||!projection) return;
    const rect=canvas.getBoundingClientRect();
    const x=(event.clientX-rect.left-projection.ox)/projection.scale-projection.offsetX;
    const y=(projection.oy-(event.clientY-rect.top))/projection.scale-projection.offsetY;
    let best=-1,distance=Infinity;
    layer.anchor_local_m.forEach((point,index)=>{const d=Math.hypot(point[0]-x,point[1]-y);if(d<distance){best=index;distance=d;}});
    const label=document.getElementById(`heatmap_${role}_hover`);
    if(best<0||distance>layer.spacing_m*.75){label.textContent="Hover over a spring for its compression and rest length.";return;}
    const rest=layer.rest_length_m[best],compression=layer.compression[projection.frame*layer.rest_length_m.length+best];
    label.textContent=`Spring ${best}: ${(compression*1000).toFixed(2)} mm · ${(compression/rest*100).toFixed(1)}% · rest ${(rest*1000).toFixed(2)} mm · ${layer.driven[best]?"driven":"passive"}`;
  };
}


function drawPaperCompressionPlot(leftCond, rightCond) {
  const svg = document.getElementById("plot_paper_compression");
  if (!svg) return;
  svg.innerHTML = "";
  const svgNS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };

  const limitStrain = 0.60;
  const maxStress = 350.0;
  const toX = s => 55 + (Math.max(0, Math.min(0.65, s)) / 0.60) * 440;
  const toY = p => 240 - (Math.max(0, Math.min(maxStress, p)) / maxStress) * 215;

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const sVal = frac * limitStrain;
    const xPx = toX(sVal);
    svg.appendChild(el("line", {x1: xPx, y1: 25, x2: xPx, y2: 240, stroke: "#f1f5f9", "stroke-width": 1}));
    const sTxt = el("text", {x: xPx, y: 255, "text-anchor": "middle", fill: "#64748b", "font-size": 10});
    sTxt.textContent = `${Math.round(sVal * 100)}%`;
    svg.appendChild(sTxt);

    const pVal = frac * maxStress;
    const yPx = toY(pVal);
    svg.appendChild(el("line", {x1: 55, y1: yPx, x2: 495, y2: yPx, stroke: "#f1f5f9", "stroke-width": 1}));
    const pTxt = el("text", {x: 48, y: yPx + 4, "text-anchor": "end", fill: "#64748b", "font-size": 10});
    pTxt.textContent = `${Math.round(pVal)}`;
    svg.appendChild(pTxt);
  }

  // Axes labels
  const xlab = el("text", {x: 275, y: 272, "text-anchor": "middle", fill: "#334155", "font-size": 11, "font-weight": "600"});
  xlab.textContent = "Compressive Strain ε [-]";
  svg.appendChild(xlab);

  const ylab = el("text", {x: 18, y: 130, "text-anchor": "middle", fill: "#334155", "font-size": 11, "font-weight": "600", transform: "rotate(-90 18 130)"});
  ylab.textContent = "Compression stress [kPa]";
  svg.appendChild(ylab);

  // Literature limit vertical line (0.60 strain)
  const limX = toX(limitStrain);
  svg.appendChild(el("line", {x1: limX, y1: 20, x2: limX, y2: 240, stroke: "#dc2626", "stroke-width": 1.5, "stroke-dasharray": "4 4"}));
  const limTxt = el("text", {x: limX - 4, y: 18, "text-anchor": "end", fill: "#dc2626", "font-size": 10, "font-weight": "bold"});
  limTxt.textContent = "Literature Limit (60%)";
  svg.appendChild(limTxt);

  // Baseline reference curve
  const baseData = DATA.baseline_paper_compression;
  if (baseData && baseData.table_strain && baseData.pressure_curve_kpa) {
    let d = "";
    for (let i = 0; i < baseData.table_strain.length; i++) {
      d += (d === "" ? "M" : "L") + toX(baseData.table_strain[i]) + " " + toY(baseData.pressure_curve_kpa[i]) + " ";
    }
    svg.appendChild(el("path", {d, fill: "none", stroke: "#94a3b8", "stroke-width": 2, "stroke-dasharray": "5 4"}));
  }

  // Draw curves and points for right condition (orange) if paper_material present
  const rMat = rightCond?.paper_material;
  if (rMat && rMat.fit_metrics && rMat.fit_metrics.pressure_curve_kpa) {
    let d = "";
    const pCurve = rMat.fit_metrics.pressure_curve_kpa;
    const strains = rMat.table_strain;
    for (let i = 0; i < strains.length; i++) {
      d += (d === "" ? "M" : "L") + toX(strains[i]) + " " + toY(pCurve[i]) + " ";
    }
    svg.appendChild(el("path", {d, fill: "none", stroke: "#ea580c", "stroke-width": 2, "stroke-dasharray": "4 3"}));
    if (rMat.table_stress_kpa) {
      for (let i = 0; i < strains.length; i++) {
        svg.appendChild(el("circle", {cx: toX(strains[i]), cy: toY(rMat.table_stress_kpa[i]), r: 3, fill: "#ea580c"}));
      }
    }
  }

  // Draw curves and points for left condition (blue) if paper_material present
  const lMat = leftCond?.paper_material;
  if (lMat && lMat.fit_metrics && lMat.fit_metrics.pressure_curve_kpa) {
    let d = "";
    const pCurve = lMat.fit_metrics.pressure_curve_kpa;
    const strains = lMat.table_strain;
    for (let i = 0; i < strains.length; i++) {
      d += (d === "" ? "M" : "L") + toX(strains[i]) + " " + toY(pCurve[i]) + " ";
    }
    svg.appendChild(el("path", {d, fill: "none", stroke: "#2563eb", "stroke-width": 2.5}));
    if (lMat.table_stress_kpa) {
      for (let i = 0; i < strains.length; i++) {
        svg.appendChild(el("circle", {cx: toX(strains[i]), cy: toY(lMat.table_stress_kpa[i]), r: 3.5, fill: "#2563eb"}));
      }
    }
  }
}

function updatePaperMaterialSection(leftCond, rightCond) {
  const card = document.getElementById("paper_material_card");
  if (!card) return;

  const hasPaper = Boolean(DATA.has_paper_materials || leftCond?.paper_material || rightCond?.paper_material);
  if (!hasPaper) {
    card.style.display = "none";
    return;
  }
  card.style.display = "block";

  // Build provenance and parameter table
  const pane = document.getElementById("paper_param_table_pane");
  const baseData = DATA.baseline_paper_compression;
  const lMat = leftCond?.paper_material;
  const rMat = rightCond?.paper_material;

  const fmt = (v, digits=3) => (v !== null && v !== undefined && isFinite(v)) ? Number(v).toFixed(digits) : "-";
  const fmtExp = (v) => (v !== null && v !== undefined && isFinite(v)) ? Number(v).toExponential(3) : "-";

  let html = `<table style="font-size:12px;margin:0;">
    <thead>
      <tr>
        <th>Quantity / Parameter</th>
        <th>Baseline Shoe</th>
        <th>Left: ${leftCond ? leftCond.id : "-"}</th>
        <th>Right: ${rightCond ? rightCond.id : "-"}</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>Material / Origin</b></td>
        <td>Fixed baseline artifact</td>
        <td>${lMat ? `<b>${lMat.material_name}</b> (${lMat.label})` : "Baseline / Sensitivity"}</td>
        <td>${rMat ? `<b>${rMat.material_name}</b> (${rMat.label})` : "Baseline / Sensitivity"}</td>
      </tr>
      <tr>
        <td><b>Fit RMSE [kPa]</b></td>
        <td>Not fit to paper data</td>
        <td>${lMat?.fit_metrics ? `<b>${fmt(lMat.fit_metrics.rmse_kpa, 4)}</b>` : "-"}</td>
        <td>${rMat?.fit_metrics ? `<b>${fmt(rMat.fit_metrics.rmse_kpa, 4)}</b>` : "-"}</td>
      </tr>
      <tr>
        <td><b>Max Fit Error [kPa]</b></td>
        <td>-</td>
        <td>${lMat?.fit_metrics ? `${fmt(lMat.fit_metrics.max_err_kpa, 3)} (at &epsilon;=${fmt(lMat.fit_metrics.max_err_strain, 2)})` : "-"}</td>
        <td>${rMat?.fit_metrics ? `${fmt(rMat.fit_metrics.max_err_kpa, 3)} (at &epsilon;=${fmt(rMat.fit_metrics.max_err_strain, 2)})` : "-"}</td>
      </tr>
      <tr>
        <td><b>Pressure at 50% Strain [kPa]</b></td>
        <td>${fmt(baseData?.stress_50pct_kpa, 1)}</td>
        <td>${lMat?.fit_metrics ? `<b>${fmt(lMat.fit_metrics.stress_50pct_kpa, 1)}</b>` : fmt(baseData?.stress_50pct_kpa, 1)}</td>
        <td>${rMat?.fit_metrics ? `<b>${fmt(rMat.fit_metrics.stress_50pct_kpa, 1)}</b>` : fmt(baseData?.stress_50pct_kpa, 1)}</td>
      </tr>
      <tr>
        <td><b>&mu;<sub>1,eq</sub> [Pa] / &alpha;<sub>1</sub></b></td>
        <td>${fmt(baseData?.parameters?.mu1_eq_pa, 0)} / ${fmt(baseData?.parameters?.alpha1, 2)}</td>
        <td>${lMat?.surrogate_params ? `${fmt(lMat.surrogate_params.mu1_eq_pa, 0)} / ${fmt(lMat.surrogate_params.alpha1, 2)}` : "-"}</td>
        <td>${rMat?.surrogate_params ? `${fmt(rMat.surrogate_params.mu1_eq_pa, 0)} / ${fmt(rMat.surrogate_params.alpha1, 2)}` : "-"}</td>
      </tr>
      <tr>
        <td><b>&mu;<sub>2,eq</sub> [Pa] / &alpha;<sub>2</sub></b></td>
        <td>${fmt(baseData?.parameters?.mu2_eq_pa, 0)} / ${fmt(baseData?.parameters?.alpha2, 2)}</td>
        <td>${lMat?.surrogate_params ? `${fmt(lMat.surrogate_params.mu2_eq_pa, 0)} / ${fmt(lMat.surrogate_params.alpha2, 2)}` : "-"}</td>
        <td>${rMat?.surrogate_params ? `${fmt(rMat.surrogate_params.mu2_eq_pa, 0)} / ${fmt(rMat.surrogate_params.alpha2, 2)}` : "-"}</td>
      </tr>
      <tr>
        <td><b>&Sigma; &mu;<sub>eq</sub> (Pasternak shear G) [Pa]</b></td>
        <td>${fmt(baseData?.parameters?.mu_eq_sum_pa, 0)}</td>
        <td>${lMat?.surrogate_params ? `${fmt(lMat.surrogate_params.mu_eq_sum_pa, 0)}` : "-"}</td>
        <td>${rMat?.surrogate_params ? `${fmt(rMat.surrogate_params.mu_eq_sum_pa, 0)}` : "-"}</td>
      </tr>
      <tr>
        <td><b>f<sub>eq</sub> / &tau; [s] (Held Fixed)</b></td>
        <td>${fmt(baseData?.parameters?.feq, 4)} / ${fmt(baseData?.parameters?.tau_s, 5)}</td>
        <td>${lMat?.surrogate_params ? `${fmt(lMat.surrogate_params.feq, 4)} / ${fmt(lMat.surrogate_params.tau_s, 5)}` : "-"}</td>
        <td>${rMat?.surrogate_params ? `${fmt(rMat.surrogate_params.feq, 4)} / ${fmt(rMat.surrogate_params.tau_s, 5)}` : "-"}</td>
      </tr>
      <tr>
        <td><b>&mu;<sub>1,inst</sub> / &mu;<sub>2,inst</sub> [Pa]</b></td>
        <td>${fmt(baseData?.parameters?.mu1_inst_pa, 0)} / ${fmt(baseData?.parameters?.mu2_inst_pa, 0)}</td>
        <td>${lMat?.surrogate_params ? `${fmt(lMat.surrogate_params.mu1_inst_pa, 0)} / ${fmt(lMat.surrogate_params.mu2_inst_pa, 0)}` : "-"}</td>
        <td>${rMat?.surrogate_params ? `${fmt(rMat.surrogate_params.mu1_inst_pa, 0)} / ${fmt(rMat.surrogate_params.mu2_inst_pa, 0)}` : "-"}</td>
      </tr>
      <tr>
        <td><b>Literature Strain Limit</b></td>
        <td>0.60 (60% strain)</td>
        <td>${lMat ? `<b>0.60</b> (test stretch &lambda;&ge;0.40)` : "-"}</td>
        <td>${rMat ? `<b>0.60</b> (test stretch &lambda;&ge;0.40)` : "-"}</td>
      </tr>
      <tr>
        <td><b>Manuscript Source</b></td>
        <td>Baseline design</td>
        <td>${lMat?.sources?.doi ? `<a href="${lMat.sources.doi}" target="_blank" style="color:var(--primary);">Springer DOI</a> · <a href="${lMat.sources.arxiv}" target="_blank" style="color:var(--primary);">arXiv</a>` : "-"}</td>
        <td>${rMat?.sources?.doi ? `<a href="${rMat.sources.doi}" target="_blank" style="color:var(--primary);">Springer DOI</a> · <a href="${rMat.sources.arxiv}" target="_blank" style="color:var(--primary);">arXiv</a>` : "-"}</td>
      </tr>
    </tbody>
  </table>`;
  pane.innerHTML = html;

  drawPaperCompressionPlot(leftCond, rightCond);

  // Update domain warnings for selected conditions
  const warnBox = document.getElementById("paper_domain_warning_box");
  const warnings = [];

  for (const [role, cond] of [["Left", leftCond], ["Right", rightCond]]) {
    if (!cond) continue;
    const pData = selectedProtocol(cond);
    const pDom = pData?.domain_evidence;
    if (pDom && pDom.available) {
      if (pDom.exceeds_literature_limit) {
        warnings.push(`<b>${role} Condition (${cond.id}): Stance Strain Domain Limit Exceeded.</b>
          Maximum any-column compressive strain reached <b>${(pDom.max_any_strain * 100).toFixed(1)}%</b>
          (driven: <b>${pDom.max_driven_strain !== null ? (pDom.max_driven_strain * 100).toFixed(1) + "%" : "N/A"}</b>,
           passive: <b>${(pDom.max_passive_strain * 100).toFixed(1)}%</b>),
          exceeding the experimental literature limit of 60.0% (&lambda;=0.40) at stance time <b>${pDom.first_exceed_time_s.toFixed(3)} s</b>
          across <b>${pDom.exceed_step_count} saved stance steps (${(pDom.exceed_step_fraction * 100).toFixed(1)}% of steps)</b>.
          <i>Note: Count and fraction reflect simulation time steps where &ge;1 column exceeded 60% strain, not the spatial percentage of columns.
          Runtime physics solver cap remains 90.0% (0.90 strain); solver behavior is unclipped by reporting domain warnings.</i>`);
      }
    }

    // Check bench hysteresis fixtures
    const hysFixes = cond.hysteresis?.fixtures || {};
    for (const [fixName, fixObj] of Object.entries(hysFixes)) {
      const fDom = fixObj.domain_evidence;
      if (fDom && fDom.available && fDom.exceeds_literature_limit) {
        warnings.push(`<b>${role} Condition (${cond.id}) — ${fixName} Bench:</b>
          Maximum column compressive strain reached <b>${(fDom.max_all_columns_strain * 100).toFixed(1)}%</b>
          (passive: <b>${(fDom.max_passive_strain * 100).toFixed(1)}%</b>, driven-only: not tracked separately in bench NPZ),
          exceeding literature calibration domain (60.0%) at <b>${fDom.first_exceed_time_s.toFixed(3)} s</b>
          across <b>${fDom.exceed_step_count} bench steps (${(fDom.exceed_step_fraction * 100).toFixed(1)}% of steps with &ge;1 column over limit)</b>.`);
      }
    }
  }

  if (warnings.length > 0) {
    warnBox.style.display = "block";
    warnBox.innerHTML = warnings.join("<br><br>");
  } else {
    warnBox.style.display = "none";
  }
}

function drawSvgPlot(svgId, title, yUnit, varTimes, varVals, baseTimes, baseVals, failTime, comparisonFailTime=null) {
  const svg = document.getElementById(svgId);
  svg.innerHTML = "";
  const svgNS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };

  const allY = [...(varVals || []), ...(baseVals || [])].filter(v => v !== null && isFinite(v));
  if (allY.length === 0) {
    const txt = el("text", {x: 270, y: 110, "text-anchor": "middle", fill: "#94a3b8", "font-size": 13});
    txt.textContent = "Data unavailable";
    svg.appendChild(txt);
    return;
  }

  let ymin = Math.min(...allY);
  let ymax = Math.max(...allY);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const pad = (ymax - ymin) * 0.1;
  ymin -= pad; ymax += pad;
  const tmax = 0.36;

  const toX = t => 55 + (t / tmax) * 460;
  const toY = y => 190 - ((y - ymin) / (ymax - ymin)) * 165;

  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yVal = ymin + frac * (ymax - ymin);
    const yPx = toY(yVal);
    svg.appendChild(el("line", {x1: 55, y1: yPx, x2: 515, y2: yPx, stroke: "#e2e8f0", "stroke-width": 1}));
    const txt = el("text", {x: 48, y: yPx + 4, "text-anchor": "end", fill: "#64748b", "font-size": 10});
    txt.textContent = yVal.toPrecision(3);
    svg.appendChild(txt);

    const tVal = frac * tmax;
    const xPx = toX(tVal);
    const tTxt = el("text", {x: xPx, y: 205, "text-anchor": "middle", fill: "#64748b", "font-size": 10});
    tTxt.textContent = tVal.toFixed(2) + "s";
    svg.appendChild(tTxt);
  }

  if (ymin < 0 && ymax > 0) svg.appendChild(el("line",{x1:55,x2:515,y1:toY(0),y2:toY(0),stroke:"#94a3b8","stroke-width":1}));

  if (baseTimes && baseVals && baseTimes.length > 0) {
    let d = "";
    for (let i = 0; i < baseTimes.length; i++) {
      if (baseVals[i] !== null && isFinite(baseVals[i])) {
        d += (d === "" ? "M" : "L") + toX(baseTimes[i]) + " " + toY(baseVals[i]) + " ";
      }
    }
    svg.appendChild(el("path", {d, fill: "none", stroke: "#94a3b8", "stroke-width": 1.5, "stroke-dasharray": "4 4"}));
  }

  if (varTimes && varVals && varTimes.length > 0) {
    let d = "";
    for (let i = 0; i < varTimes.length; i++) {
      if (varVals[i] !== null && isFinite(varVals[i])) {
        d += (d === "" ? "M" : "L") + toX(varTimes[i]) + " " + toY(varVals[i]) + " ";
      }
    }
    svg.appendChild(el("path", {d, fill: "none", stroke: "#2563eb", "stroke-width": 2}));
  }

  if (failTime !== null && failTime <= tmax) {
    const fx = toX(failTime);
    svg.appendChild(el("line", {x1: fx, y1: 25, x2: fx, y2: 190, stroke: "#dc2626", "stroke-width": 1.5, "stroke-dasharray": "3 3"}));
    const cross = el("text", {x: fx, y: 20, "text-anchor": "middle", fill: "#dc2626", "font-size": 12, "font-weight": "bold"});
    cross.textContent = "✖ Left stop";
    svg.appendChild(cross);
  }
  if (comparisonFailTime !== null && comparisonFailTime <= tmax) {
    const fx=toX(comparisonFailTime);
    svg.appendChild(el("line",{x1:fx,x2:fx,y1:25,y2:190,stroke:"#b45309","stroke-dasharray":"4 4"}));
    const text=el("text",{x:fx,y:12,"font-size":10,fill:"#b45309"});text.textContent="Right stop";svg.appendChild(text);
  }
  const x=toX(displayedStanceTime ?? currentTime);
  svg.appendChild(el("line",{class:"stance-cursor",x1:x,x2:x,y1:25,y2:190,stroke:"#9333ea","stroke-width":1,"stroke-dasharray":"2 3"}));
}
function updatePlotCursors(time) {
  const x=55+Math.min(.36,Math.max(0,time))/.36*460;
  document.querySelectorAll(".stance-cursor").forEach(line=>{line.setAttribute("x1",x);line.setAttribute("x2",x);});
}

function updatePlots() {
  const pData = selectedProtocol(currentCond);
  const vCurves = pData && pData.plot_curves && pData.plot_curves.available ? pData.plot_curves : null;
  const bData = selectedProtocol(currentCompare);
  const bCurves = bData?.plot_curves?.available ? bData.plot_curves : null;
  const failT = pData && ["stopped", "failed", "terminated"].includes(pData.status) ? pData.last_valid_time_s : null;

  const vT = vCurves ? vCurves.time_s : null;
  const bT = bCurves ? bCurves.time_s : null;

  drawSvgPlot("plot_grf_z", "Vertical GRF (Fz)", "N", vT, vCurves ? vCurves.grf_z_n : null, bT, bCurves ? bCurves.grf_z_n : null, failT);
  drawSvgPlot("plot_grf_x", "Fore-Aft GRF (Fx)", "N", vT, vCurves ? vCurves.grf_x_n : null, bT, bCurves ? bCurves.grf_x_n : null, failT);
  drawSvgPlot("plot_hip_z", "Hip Vertical (z)", "m", vT, vCurves ? vCurves.hip_z_m : null, bT, bCurves ? bCurves.hip_z_m : null, failT);
  drawSvgPlot("plot_knee", "Knee Angle", "rad", vT, vCurves ? vCurves.knee_rad : null, bT, bCurves ? bCurves.knee_rad : null, failT);
  updateActuation();
}

const actuationNames={knee:"Knee",ankle:"Ankle",hip:"Hip point",total:"All four actuators"};
const workNames={positive:"produced",absorbed:"absorbed",net:"net"};
const actuationMetricNames={
  positive_work_j:["Produced work","J"],absorbed_work_j:["Absorbed work","J"],net_work_j:["Net work","J"],
  peak_abs_torque_nm:["Peak absolute torque","N·m"],rms_torque_nm:["RMS torque","N·m"],
  peak_force_n:["Peak force magnitude","N"],rms_force_n:["RMS force magnitude","N"],
  peak_positive_power_w:["Peak produced power","W"],peak_absorbed_power_w:["Peak absorbed power","W"]
};
function actuationMetricLabel(metric,channel) {
  if(channel==="total"&&metric==="peak_positive_power_w")return "Peak positive combined net power";
  if(channel==="total"&&metric==="peak_absorbed_power_w")return "Peak negative combined net power magnitude";
  return actuationMetricNames[metric]?.[0]||metric;
}
function metricText(value,signed=false) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value)<1e-10) return "0";
  return (signed&&value>0?"+":"")+Number(value).toPrecision(4);
}
function percentText(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value)<1e-12) return "0.00%";
  const digits=Math.abs(value)<.01?value.toPrecision(2):value.toFixed(2);
  return (value>0?"+":"")+digits+"%";
}
function supportText(support) {return support?.length===2?`${support[0].toFixed(4)} to ${support[1].toFixed(4)} s`:"unavailable";}
function actuationMetricOptions() {
  const channel=document.getElementById("actuation_channel").value;
  const select=document.getElementById("actuation_metric"),old=select.value;
  const keys=["positive_work_j","absorbed_work_j","net_work_j"];
  if(channel==="ankle"||channel==="knee")keys.push("peak_abs_torque_nm","rms_torque_nm");
  if(channel==="hip")keys.push("peak_force_n","rms_force_n");
  keys.push("peak_positive_power_w","peak_absorbed_power_w");
  select.replaceChildren(...keys.map(key=>{const option=document.createElement("option");option.value=key;option.textContent=`${actuationMetricLabel(key,channel)} [${actuationMetricNames[key][1]}]`;return option;}));
  if(keys.includes(old))select.value=old;
}
function updateActuation() {
  const left=selectedProtocol(currentCond),right=selectedProtocol(currentCompare);
  const la=left?.actuation,ra=right?.actuation;
  const lc=la?.available?la.curves:null,rc=ra?.available?ra.curves:null;
  const failed=data=>["failed","stopped","terminated"].includes(data?.status)?data.last_valid_time_s:null;
  const plot=(id,key,unit)=>drawSvgPlot(id,"",unit,lc?.time_s,lc?.[key],rc?.time_s,rc?.[key],failed(left),failed(right));
  document.getElementById("actuation_pair_label").textContent=`Left: ${currentCond?.id||"none"} · Right: ${currentCompare?.id||"none"} · ${currentProtocol.replaceAll("_"," ")} protocol`;
  const support=document.getElementById("actuation_support");support.replaceChildren();
  for(const [role,analysis] of [["Left",la],["Right",ra]]) {
    const line=document.createElement("p");
    line.textContent=!analysis?.available?`${role}: ${analysis?.reason||"No saved actuator data for this protocol."}`:`${role}: saved force support ${supportText(analysis.support_s)}; ${analysis.full_stance?"completed stance":"INCOMPLETE stance — prefix only"}. Native/half-step comparison: ${analysis.refinement?.performed?supportText(analysis.refinement.support_s):"unavailable"}.`;
    support.appendChild(line);
  }
  for(const joint of ["knee","ankle"]){plot(`plot_${joint}_torque`,`${joint}_torque_nm`,"N·m");plot(`plot_${joint}_power`,`${joint}_power_w`,"W");}
  plot("plot_hip_power","hip_power_w","W");plot("plot_total_power","total_power_w","W");
  const kind=document.getElementById("work_kind").value;
  for(const channel of ["knee","ankle","hip","total"]){
    document.getElementById(`${channel}_work_title`).textContent=`${actuationNames[channel]} cumulative ${workNames[kind]} work [J]`;
    plot(`plot_${channel}_work`,`${channel}_${kind}_work_j`,"J");
  }
  const rows=document.getElementById("actuation_summary_rows");rows.replaceChildren();
  for(const [role,condition,analysis] of [["Left",currentCond,la],["Right",currentCompare,ra]]) {
    if(!analysis?.available)continue;
    for(const channel of ["knee","ankle","hip","total"]){
      const metrics=analysis.channels[channel]||{},delta=analysis.baseline_delta?.full_stance_comparison?analysis.baseline_delta.channels?.[channel]:null;
      const row=document.createElement("tr");
      const add=text=>{const td=document.createElement("td");td.textContent=text;row.appendChild(td);return td;};
      add(`${role}: ${condition.id}`);add(actuationNames[channel]);
      const isJoint=channel==="knee"||channel==="ankle",peak=isJoint?"peak_abs_torque_nm":"peak_force_n",rms=isJoint?"rms_torque_nm":"rms_force_n",unit=isJoint?"N·m":"N";
      const load=add(channel==="total"?"Not summed across units":`${metricText(metrics[peak])} / ${metricText(metrics[rms])} ${unit}`);
      if(channel!=="total"){
        const change=document.createElement("span");change.className="metric-delta";change.textContent=`Δ ${metricText(delta?.[peak],true)} / ${metricText(delta?.[rms],true)} ${unit}`;load.appendChild(change);
      }
      for(const key of ["positive_work_j","absorbed_work_j","net_work_j"]){
        const cell=add(metricText(metrics[key])),change=document.createElement("span");change.className="metric-delta";change.textContent=`(Δ ${metricText(delta?.[key],true)})`;cell.appendChild(change);
      }
      rows.appendChild(row);
    }
  }
  if(!rows.children.length){const row=rows.insertRow(),cell=row.insertCell();cell.colSpan=6;cell.textContent="Saved actuator data unavailable.";}
  updateActuationCampaign();
}
function updateActuationCampaign() {
  const channel=document.getElementById("actuation_channel").value,metric=document.getElementById("actuation_metric").value;
  const unit=actuationMetricNames[metric]?.[1]||"";
  const records=DATA.conditions.map(condition=>({condition,data:selectedProtocol(condition)})).filter(record=>record.data);
  const baselineChange=analysis=>analysis?.baseline_delta?.full_stance_comparison?analysis.baseline_delta.channels?.[channel]?.[metric]:null;
  if(document.getElementById("actuation_sort").value==="difference")records.sort((a,b)=>{
    if(a.condition.family==="baseline")return -1;if(b.condition.family==="baseline")return 1;
    const av=baselineChange(a.data.actuation),bv=baselineChange(b.data.actuation);
    return (Number.isFinite(bv)?Math.abs(bv):-1)-(Number.isFinite(av)?Math.abs(av):-1);
  });
  document.getElementById("actuation_campaign_note").textContent=`${actuationNames[channel]} — ${actuationMetricLabel(metric,channel)} [${unit}]. ${records.length} protocol entries. Δ means condition minus the fixed baseline primary replay. Native values use their own saved interval. Δ and percent changes use matching shared support; percent = 100 * Δ / |baseline|. Percent is unavailable for zero/near-zero baselines or incomplete comparisons. Half-step values remain in row tooltips and downloads. This table does not establish physical or numerical acceptance.`;
  const rows=document.getElementById("actuation_campaign_rows");rows.replaceChildren();
  for(const {condition,data} of records){
    const analysis=data.actuation,full=analysis?.full_stance,base=baselineChange(analysis),effect=analysis?.effect_refinement;
    const check=effect?.performed&&effect.full_stance_comparison?effect.channels?.[channel]?.[metric]:null;
    const deltaNative=check?.delta_native??base,deltaHalf=check?.delta_half_step,drift=check?.drift_abs;
    const percent=check ? check.percent_change_native : analysis?.baseline_delta?.full_stance_comparison
      ? analysis.baseline_delta.percent_change_channels?.[channel]?.[metric] : null;
    const row=document.createElement("tr");row.dataset.condition=condition.id;
    if(condition.id===currentCond?.id||condition.id===currentCompare?.id)row.className="selected-metric-row";
    const add=text=>{const cell=document.createElement("td");cell.textContent=text;row.appendChild(cell);return cell;};
    add(condition.id);add(metricText(analysis?.channels?.[channel]?.[metric]));add(metricText(deltaNative,true));add(percentText(percent));add(metricText(drift));
    let note="Half-step effect unavailable",sensitive=false;
    if(!analysis?.available)note=analysis?.reason||"Not available";
    else if(!full)note="Incomplete stance: do not rank prefix work";
    else if(condition.family==="baseline")note="Baseline reference";
    else if(check){
      if(deltaNative===0&&deltaHalf===0)note="No observed difference at either timestep";
      else if(check.sign_changed){note="Effect changes sign at half step";sensitive=true;}
      else if(Math.abs(deltaHalf)<=drift){note="Effect no larger than observed timestep change";sensitive=true;}
      else note="Effect larger than observed timestep change";
    }
    const cell=add(note);if(sensitive)cell.className="sensitive-metric";
    row.title=`Native saved support: ${supportText(analysis?.support_s)}. Four-trace effect support: ${supportText(effect?.support_s)}. Half-step baseline change: ${metricText(deltaHalf,true)} ${unit}. Percent uses absolute baseline magnitude; undefined near zero.`;
    rows.appendChild(row);
  }
}
actuationMetricOptions();
document.getElementById("work_kind").onchange=updateActuation;
document.getElementById("actuation_channel").onchange=()=>{actuationMetricOptions();updateActuationCampaign();};
for(const id of ["actuation_metric","actuation_sort"])document.getElementById(id).onchange=updateActuationCampaign;

function drawHysteresisPlot() {
  const svg = document.getElementById("plot_hysteresis");
  svg.innerHTML = "";
  const svgNS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };

  const fixName = document.getElementById("fixture_select").value;
  const fixData = currentCond && currentCond.hysteresis && currentCond.hysteresis.fixtures ? currentCond.hysteresis.fixtures[fixName] : null;

  const statusBadge = document.getElementById("fixture_status_badge");
  const diagPane = document.getElementById("hys_diagnostics_pane");

  if (!fixData || fixData.status === "blocked") {
    statusBadge.innerHTML = `<span class="badge badge-blocked">BLOCKED / UNAVAILABLE</span>`;
    const reason = fixData ? fixData.reason : "Fixture not present";
    diagPane.innerHTML = `<div class="warning-box"><b>Fixture '${fixName}' is unavailable.</b><br>${reason}<br><i>Forefoot last geometry is not present in local bundle. Never substituted by fullfoot.</i></div>`;
    const txt = el("text", {x: 270, y: 160, "text-anchor": "middle", fill: "#94a3b8", "font-size": 14});
    txt.textContent = `Fixture '${fixName}' unavailable (${reason})`;
    svg.appendChild(txt);
    return;
  }

  statusBadge.innerHTML = `<span class="badge badge-${fixData.status}">${fixData.status.toUpperCase()}</span>`;

  const c = fixData.curve;
  const baseFix = currentCompare?.hysteresis?.fixtures?.[fixName] || null;
  const overlayBase = document.getElementById("overlay_baseline_hys").checked;

  let allD = [], allF = [];
  if (c && c.available) {
    if (c.final_loop) { allD.push(...c.final_loop.displacement_m); allF.push(...c.final_loop.force_n); }
    if (c.first_loop) { allD.push(...c.first_loop.displacement_m); allF.push(...c.first_loop.force_n); }
    if (c.all_cycles) { allD.push(...c.all_cycles.displacement_m); allF.push(...c.all_cycles.force_n); }
  }
  if (overlayBase && baseFix && baseFix.curve && baseFix.curve.available) {
    const bc = baseFix.curve;
    if (bc.final_loop) { allD.push(...bc.final_loop.displacement_m); allF.push(...bc.final_loop.force_n); }
  }

  if (allD.length === 0) {
    diagPane.innerHTML = `<p class="legend">No curve data available for fixture '${fixName}'. Status: ${fixData.status}</p>`;
    const txt = el("text", {x: 270, y: 160, "text-anchor": "middle", fill: "#94a3b8", "font-size": 13});
    txt.textContent = "No curve points recorded";
    svg.appendChild(txt);
    return;
  }

  const m = fixData.metrics || {};
  const isClosed = m.is_state_closed;
  const qualNote = m.qualification === "valid_closed_loop" ? "Closed loop verified" : "Residual state / unclosed cycle (dissipation claim not certified)";
  let compRow = "";
    const comp = fixData.comparison;
    if (comp && comp.performed) {
      const compPassed = comp.passed ? "<span style='color:green;font-weight:600;'>Passed</span>" : "<span style='color:red;font-weight:600;'>Failed</span>";
      const fDiff = comp.maximum_force_difference_n !== undefined && comp.maximum_force_difference_n !== null ? comp.maximum_force_difference_n.toFixed(3) + " N (tol: " + (comp.force_limit_n !== undefined ? comp.force_limit_n.toFixed(3) : "-") + " N)" : "-";
      compRow = `<tr><th>Bench Refinement</th><td>${compPassed} (max ΔF: ${fDiff})</td></tr>`;
    }
    diagPane.innerHTML = `
      <table style="font-size:12px;">
        <tr><th>Work Input</th><td>${m.work_input_j !== undefined && m.work_input_j !== null ? m.work_input_j.toFixed(3) + " J" : "-"}</td></tr>
        <tr><th>Work Returned</th><td>${m.work_returned_j !== undefined && m.work_returned_j !== null ? m.work_returned_j.toFixed(3) + " J" : "-"}</td></tr>
        <tr><th>Work Net</th><td>${m.work_net_j !== undefined && m.work_net_j !== null ? m.work_net_j.toFixed(3) + " J" : "-"}</td></tr>
        <tr><th>Hysteresis Loss Ratio</th><td>${m.hysteresis_loss_ratio !== undefined && m.hysteresis_loss_ratio !== null ? (m.hysteresis_loss_ratio*100).toFixed(1) + "%" : "-"}</td></tr>
        <tr><th>Peak Force</th><td>${m.peak_force_n !== undefined && m.peak_force_n !== null ? m.peak_force_n.toFixed(1) + " N" : "-"}</td></tr>
        <tr><th>Peak Displacement</th><td>${m.peak_displacement_m !== undefined && m.peak_displacement_m !== null ? (m.peak_displacement_m*1000).toFixed(2) + " mm" : "-"}</td></tr>
        <tr><th>Loop Closure Status</th><td><b>${qualNote}</b></td></tr>
        ${compRow}
      </table>
    `;

  const xmin = 0, xmax = Math.max(...allD) * 1000 * 1.08;
  const ymin = 0, ymax = Math.max(...allF) * 1.1;

  const toX = d_m => 60 + ((d_m * 1000 - xmin) / (xmax - xmin)) * 440;
  const toY = f_n => 280 - ((f_n - ymin) / (ymax - ymin)) * 240;

  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yVal = ymin + frac * (ymax - ymin);
    const yPx = toY(yVal);
    svg.appendChild(el("line", {x1: 60, y1: yPx, x2: 500, y2: yPx, stroke: "#e2e8f0", "stroke-width": 1}));
    const txt = el("text", {x: 52, y: yPx + 4, "text-anchor": "end", fill: "#64748b", "font-size": 10});
    txt.textContent = yVal.toFixed(0) + " N";
    svg.appendChild(txt);

    const xVal = xmin + frac * (xmax - xmin);
    const xPx = toX(xVal / 1000);
    const xTxt = el("text", {x: xPx, y: 298, "text-anchor": "middle", fill: "#64748b", "font-size": 10});
    xTxt.textContent = xVal.toFixed(1) + " mm";
    svg.appendChild(xTxt);
  }

  const xlab = el("text", {x: 280, y: 315, "text-anchor": "middle", fill: "#334155", "font-size": 11, "font-weight": "600"});
  xlab.textContent = "Displacement [mm]";
  svg.appendChild(xlab);

  if (overlayBase && baseFix && baseFix.curve && baseFix.curve.available && baseFix.curve.final_loop) {
    const bd = baseFix.curve.final_loop.displacement_m;
    const bf = baseFix.curve.final_loop.force_n;
    let pathD = "";
    for (let i = 0; i < bd.length; i++) pathD += (pathD === "" ? "M" : "L") + toX(bd[i]) + " " + toY(bf[i]) + " ";
    svg.appendChild(el("path", {d: pathD, fill: "none", stroke: "#94a3b8", "stroke-width": 2, "stroke-dasharray": "5 4"}));
  }

  if (c && c.first_loop) {
    const fd = c.first_loop.displacement_m;
    const ff = c.first_loop.force_n;
    let pathD = "";
    for (let i = 0; i < fd.length; i++) pathD += (pathD === "" ? "M" : "L") + toX(fd[i]) + " " + toY(ff[i]) + " ";
    svg.appendChild(el("path", {d: pathD, fill: "none", stroke: "#ea580c", "stroke-width": 1.5, "stroke-dasharray": "2 2"}));
  }

  if (c && c.final_loop) {
    const ld = c.final_loop.displacement_m;
    const lf = c.final_loop.force_n;
    let pathD = "";
    for (let i = 0; i < ld.length; i++) pathD += (pathD === "" ? "M" : "L") + toX(ld[i]) + " " + toY(lf[i]) + " ";
    svg.appendChild(el("path", {d: pathD, fill: "none", stroke: "#2563eb", "stroke-width": 2.5}));

    if (ld.length > 10) {
      const midUp = Math.floor(ld.length * 0.25);
      const midDown = Math.floor(ld.length * 0.75);
      [midUp, midDown].forEach((idx, k) => {
        const x1 = toX(ld[idx]), y1 = toY(lf[idx]);
        const x2 = toX(ld[idx+1]), y2 = toY(lf[idx+1]);
        const ang = Math.atan2(y2 - y1, x2 - x1);
        const arr = el("polygon", {
          points: `${x1},${y1} ${x1 - 8*Math.cos(ang-0.5)},${y1 - 8*Math.sin(ang-0.5)} ${x1 - 8*Math.cos(ang+0.5)},${y1 - 8*Math.sin(ang+0.5)}`,
          fill: "#2563eb"
        });
        svg.appendChild(arr);
      });
    }
  }
}

function updateConditionMeta() {
  const pData = selectedProtocol(currentCond);
  const metaPane = document.getElementById("condition_meta_pane");
  const failBox = document.getElementById("failure_warning");

  if (!currentCond || !pData) {
    metaPane.innerHTML = "<p>No condition selected.</p>";
    failBox.style.display = "none";
    return;
  }

  const stat = pData.status;
  const isTerm = stat === "stopped" || stat === "failed" || stat === "terminated";
  if (isTerm) {
    failBox.style.display = "block";
    failBox.innerHTML = `<b>Simulation Terminated Prematurely:</b> ${pData.failure_reason || "Model screen triggered"}. Valid simulated prefix held at ${pData.last_valid_time_s !== null ? pData.last_valid_time_s.toFixed(3) + " s" : "unknown"}. (Not padded with frozen states).`;
  } else {
    failBox.style.display = "none";
  }

  const repLink = pData.native_report_link ? `<a href="${pData.native_report_link}" target="_blank" style="margin-left:12px;font-weight:600;color:var(--primary);">Open selected native report ↗</a>` : "";
  document.getElementById("condition_info").innerHTML = `Status: <span class="badge badge-${stat}">${stat.toUpperCase()}</span> ${repLink}`;

  const obs = pData.observations || {};
  const ref = pData.refinement || {};
  metaPane.innerHTML = `
    <div style="display:flex;flex-wrap:wrap;gap:20px;font-size:13px;">
      <div><b>Condition ID:</b> ${currentCond.id}</div>
      <div><b>Family:</b> ${currentCond.family}</div>
      <div><b>Factor:</b> ${currentCond.factor}</div>
      <div><b>Scale:</b> ${currentCond.scale !== null && currentCond.scale !== undefined ? currentCond.scale : "None"}</div>
      <div><b>Completed Duration:</b> ${pData.integrated_duration_s !== null && pData.integrated_duration_s !== undefined ? pData.integrated_duration_s.toFixed(3) + " s" : "-"}</div>
      <div><b>Peak Vertical GRF:</b> ${obs.peak_vertical_grf_n ? obs.peak_vertical_grf_n.toFixed(1) + " N" : "-"}</div>
      <div><b>Refinement Passed:</b> ${ref.passed !== undefined ? (ref.passed ? "Yes" : "No") : "-"}</div>
    </div>
  `;
}

function selectCondition(idx) {
  currentCond = DATA.conditions[idx];
  currentTime = 0;
  updateConditionMeta();
  updatePaperMaterialSection(currentCond, currentCompare);
  loadNativeViews();
  renderMotionFrames();
  updatePlots();
  drawHysteresisPlot();
}

condSelect.onchange = e => selectCondition(Number(e.target.value));
compareSelect.onchange = e => {
  currentCompare = DATA.conditions[Number(e.target.value)];
  updatePaperMaterialSection(currentCond, currentCompare);
  loadNativeViews();
  renderMotionFrames();
  updatePlots();
  drawHysteresisPlot();
};
document.getElementById("protocol_select").onchange = e => {
  currentProtocol = e.target.value;
  updateConditionMeta();
  loadNativeViews();
  renderMotionFrames();
  updatePlots();
};
document.getElementById("fixture_select").onchange = () => drawHysteresisPlot();
document.getElementById("overlay_baseline_hys").onchange = () => drawHysteresisPlot();

const slider = document.getElementById("time_slider");
slider.oninput = e => {
  currentTime = (Number(e.target.value) / 1000) * 0.36;
  renderMotionFrames();
};

const playBtn = document.getElementById("play_btn");
playBtn.onclick = () => {
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? "Pause" : "Play";
  lastTimestamp = null;
  if (isPlaying) requestAnimationFrame(stepAnimation);
};

function stepAnimation(ts) {
  if (!isPlaying) return;
  if (lastTimestamp !== null) {
    const dt = (ts - lastTimestamp) / 1000;
    currentTime = (currentTime + dt * 0.5) % 0.36;
    renderMotionFrames();
  }
  lastTimestamp = ts;
  requestAnimationFrame(stepAnimation);
}

if (DATA.conditions.length > 0) selectCondition(Number(condSelect.value));
window.onresize = () => { if (displayedStanceTime !== null) drawHeatmaps(displayedStanceTime);renderMotionFrames(); };
</script>
</body>
</html>
"""
