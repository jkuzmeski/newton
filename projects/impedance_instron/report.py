# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write portable, offline diagnostics for the vertical impedance test rig."""

import csv
import html
import json
from pathlib import Path

import numpy as np

_REQUIRED = (
    "time_s",
    "reference_fz_n",
    "shoe_fz_n",
    "foot_x_m",
    "foot_z_m",
    "pitch_rad",
    "com_x_m",
    "com_z_m",
    "com_vz_m_s",
    "leg_force_n",
    "active_power_w",
    "damping_power_w",
    "pitch_power_w",
    "shoe_contact_power_w",
    "com_energy_j",
    "max_compression_m",
    "controller_clipped",
)
_AUDIT_FIELDS = {
    "profile_hash": ("profile_hash", "profile_sha256"),
    "artifact_hash": ("artifact_hash", "artifact_sha256", "shoe_artifact_hash"),
    "mass_kg": ("mass_kg", "body_mass_kg"),
    "foot_mass_kg": ("foot_mass_kg",),
    "registration": ("registration",),
    "pitch_inertia_kg_m2": ("pitch_inertia_kg_m2",),
    "gravity_m_s2": ("gravity_m_s2",),
    "processed_reference_hash": ("processed_reference_hash",),
    "runtime_source_hash": ("runtime_source_hash",),
    "initial_prescribed_state": ("initial_prescribed_state",),
    "solver": ("solver",),
    "kinematic_rate_hz": ("kinematic_rate_hz",),
    "force_limit_n": ("force_limit_n",),
    "normal_damping_n_s_m_per_column": ("normal_damping_n_s_m_per_column",),
    "stiffness_n_m": ("stiffness_n_m", "leg_stiffness_n_m", "vertical_stiffness_n_m"),
    "damping_n_s_m": ("damping_n_s_m", "leg_damping_n_s_m", "vertical_damping_n_s_m"),
    "dt_s": ("dt_s", "dt"),
    "mode": ("mode",),
    "reference_mode": ("reference_mode",),
    "ankle_mount_m": ("ankle_mount_m",),
    "shoe_orientation": ("shoe_orientation",),
    "initial_state": ("initial_state",),
    "shoe_stiffness_scale": ("shoe_stiffness_scale",),
}
_LIMITS = (
    "Vertical impedance only. Fore-aft motion and foot pitch are prescribed, not predicted.",
    "COM is a force-integrated surrogate, not measured whole-body COM.",
    "Synthetic material scales are controlled scenarios, not identified new shoes.",
    "This is an engineering demonstration, not human validation or evidence of metabolic savings.",
    "Shoe contact work includes imposed foot translation and pitch. It is not a closed-loop material hysteresis measurement.",
    "Active leg work, damping loss, pitch drive, replay guides, and horizontal track are separate channels. The rig energy balance is not a whole-human energy balance.",
)
_METRIC_LABELS = {
    "peak_shoe_force_n": "Peak shoe vertical force [N]",
    "final_shoe_force_n": "Final shoe force after the supplied window [N]",
    "shoe_impulse_n_s": "Shoe vertical impulse [N s]",
    "reference_impulse_n_s": "Reference vertical impulse [N s]",
    "force_rmse_n": "Time-weighted vertical force RMSE [N]",
    "positive_active_leg_work_j": "Positive active-source work [J]",
    "negative_active_leg_work_j": "Negative active leg work (signed) [J]",
    "net_active_leg_work_j": "Net active leg work [J]",
    "damping_dissipation_j": "Leg damping dissipation (positive loss) [J]",
    "pitch_drive_work_j": "Net prescribed pitch drive work [J]",
    "positive_pitch_drive_work_j": "Positive prescribed pitch drive work [J]",
    "negative_pitch_drive_work_j": "Negative prescribed pitch drive work [J]",
    "shoe_contact_work_j": "Net shoe contact work [J]",
    "positive_shoe_contact_work_j": "Positive shoe contact work [J]",
    "negative_shoe_contact_work_j": "Negative shoe contact work [J]",
    "com_height_change_m": "COM height: end minus start [m]",
    "com_velocity_change_m_s": "COM vertical velocity: end minus start [m/s]",
    "com_height_change_relative_reference_m": "COM height change minus reference change [m]",
    "com_velocity_change_relative_reference_m_s": "COM velocity change minus reference change [m/s]",
    "com_endpoint_height_error_m": "COM final height minus reference [m]",
    "com_endpoint_velocity_error_m_s": "COM final velocity minus reference [m/s]",
    "com_energy_change_j": "COM energy: end minus start [J]",
    "max_compression_m": "Peak shoe compression [m]",
    "minimum_last_height_m": "Minimum rigid-last height above ground [m]",
    "peak_ankle_torque_nm": "Peak absolute ankle drive torque [N m]",
    "peak_pitch_acceleration_rad_s2": "Peak absolute pitch acceleration [rad/s^2]",
    "controller_clipped_time_fraction": "Controller-clipped time fraction [0-1]",
    "rig_energy_change_j": "Rig energy: end minus start [J]",
    "rig_energy_balance_residual_j": "Rig mechanical energy balance residual [J]",
    "rig_energy_balance_relative_residual": "Energy residual / absolute power throughput [0-1]",
    "replay_vertical_drive_work_j": "Net vertical replay-guide work [J]",
    "track_drive_work_j": "Net horizontal track-drive work [J]",
    "other_support_work_j": "Net opposite-support boundary work [J]",
}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _columns(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    if len(rows) < 2:
        raise ValueError("A report needs at least two trace samples")
    names = list(dict.fromkeys(key for row in rows for key in row))
    missing = [key for key in _REQUIRED if any(key not in row for row in rows)]
    if missing:
        raise ValueError(f"Missing required trace columns: {', '.join(missing)}")
    columns = {key: np.asarray([row.get(key, float("nan")) for row in rows], dtype=float) for key in names}
    for key in _REQUIRED:
        if not np.all(np.isfinite(columns[key])):
            raise ValueError(f"Nonfinite trace values in {key}")
    if np.any(np.diff(columns["time_s"]) <= 0.0):
        raise ValueError("Trace time_s must be strictly increasing")
    if np.any(columns["damping_power_w"] > 1.0e-8):
        raise ValueError("damping_power_w must be nonpositive")
    if not np.all(np.isin(columns["controller_clipped"], (0.0, 1.0))):
        raise ValueError("controller_clipped must contain only 0 or 1")
    return columns


def _integral(time: np.ndarray, values: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))


def _work(time: np.ndarray, power: np.ndarray) -> tuple[float, float, float]:
    """Split work at zero crossings of linearly interpolated power samples."""
    left, right = power[:-1], power[1:]
    positive = 0.5 * (np.maximum(left, 0.0) + np.maximum(right, 0.0)) * np.diff(time)
    crossing = left * right < 0.0
    # Split crossing segments rather than overcounting both signed work channels.
    positive[crossing] = (
        0.5
        * np.maximum(left[crossing], right[crossing]) ** 2
        / np.abs(right[crossing] - left[crossing])
        * np.diff(time)[crossing]
    )
    net = _integral(time, power)
    pos = float(np.sum(positive))
    return pos, net - pos, net


def _metadata_value(metadata: dict, aliases: tuple[str, ...]):
    return next((metadata[key] for key in aliases if key in metadata), None)


def _summarize(columns: dict[str, np.ndarray], metadata: dict) -> dict:
    time = columns["time_s"]
    duration = float(time[-1] - time[0])
    active = _work(time, columns["active_power_w"])
    pitch = _work(time, columns["pitch_power_w"])
    contact = _work(time, columns["shoe_contact_power_w"])
    metrics = {
        "peak_shoe_force_n": float(np.max(columns["shoe_fz_n"])),
        "final_shoe_force_n": float(columns["shoe_fz_n"][-1]),
        "shoe_impulse_n_s": _integral(time, columns["shoe_fz_n"]),
        "reference_impulse_n_s": _integral(time, columns["reference_fz_n"]),
        "force_rmse_n": float(
            np.sqrt(_integral(time, (columns["shoe_fz_n"] - columns["reference_fz_n"]) ** 2) / duration)
        ),
        "positive_active_leg_work_j": active[0],
        "negative_active_leg_work_j": active[1],
        "net_active_leg_work_j": active[2],
        "damping_dissipation_j": -_integral(time, np.minimum(columns["damping_power_w"], 0.0)),
        "pitch_drive_work_j": pitch[2],
        "positive_pitch_drive_work_j": pitch[0],
        "negative_pitch_drive_work_j": pitch[1],
        "shoe_contact_work_j": contact[2],
        "positive_shoe_contact_work_j": contact[0],
        "negative_shoe_contact_work_j": contact[1],
        "com_height_change_m": float(columns["com_z_m"][-1] - columns["com_z_m"][0]),
        "com_velocity_change_m_s": float(columns["com_vz_m_s"][-1] - columns["com_vz_m_s"][0]),
        "com_energy_change_j": float(columns["com_energy_j"][-1] - columns["com_energy_j"][0]),
        "max_compression_m": float(np.max(columns["max_compression_m"])),
        "controller_clipped_time_fraction": _integral(time, columns["controller_clipped"]) / duration,
    }
    for channel, metric, reduction in (
        ("last_min_height_m", "minimum_last_height_m", np.min),
        ("ankle_torque_nm", "peak_ankle_torque_nm", lambda v: np.max(np.abs(v))),
        ("pitch_acceleration_rad_s2", "peak_pitch_acceleration_rad_s2", lambda v: np.max(np.abs(v))),
    ):
        if channel in columns:
            if not np.all(np.isfinite(columns[channel])):
                raise ValueError(f"Nonfinite trace values in {channel}")
            metrics[metric] = float(reduction(columns[channel]))
    for channel, prefix in (
        ("replay_vertical_power_w", "replay_vertical_drive"),
        ("track_power_w", "track_drive"),
        ("other_support_power_w", "other_support"),
    ):
        if channel in columns:
            if not np.all(np.isfinite(columns[channel])):
                raise ValueError(f"Nonfinite trace values in {channel}")
            pos, neg, net = _work(time, columns[channel])
            metrics[f"positive_{prefix}_work_j"] = pos
            metrics[f"negative_{prefix}_work_j"] = neg
            metrics[f"{prefix}_work_j"] = net
    balance_channels = (
        "active_power_w",
        "damping_power_w",
        "pitch_power_w",
        "shoe_contact_power_w",
        "replay_vertical_power_w",
        "track_power_w",
        "other_support_power_w",
    )
    if "rig_energy_j" in columns and all(key in columns for key in balance_channels):
        balance_power = sum(columns[key] for key in balance_channels)
        delta_energy = float(columns["rig_energy_j"][-1] - columns["rig_energy_j"][0])
        metrics["rig_energy_change_j"] = delta_energy
        metrics["rig_energy_balance_residual_j"] = delta_energy - _integral(time, balance_power)
        metrics["rig_energy_balance_relative_residual"] = abs(metrics["rig_energy_balance_residual_j"]) / max(
            _integral(time, sum(np.abs(columns[key]) for key in balance_channels)), 1.0
        )
    velocity_source = "unavailable"
    reference_height = columns.get("reference_com_z_m")
    reference_velocity = columns.get("reference_com_vz_m_s")
    if reference_height is not None and np.all(np.isfinite(reference_height)):
        metrics["com_height_change_relative_reference_m"] = metrics["com_height_change_m"] - float(
            reference_height[-1] - reference_height[0]
        )
        metrics["com_endpoint_height_error_m"] = float(columns["com_z_m"][-1] - reference_height[-1])
        if reference_velocity is None:
            reference_velocity = np.gradient(reference_height, time)
            velocity_source = "finite-difference estimate from reference_com_z_m (one-sided endpoints)"
    if reference_velocity is not None and np.all(np.isfinite(reference_velocity)):
        if velocity_source == "unavailable":
            velocity_source = "reference_com_vz_m_s"
        metrics["com_velocity_change_relative_reference_m_s"] = metrics["com_velocity_change_m_s"] - float(
            reference_velocity[-1] - reference_velocity[0]
        )
        metrics["com_endpoint_velocity_error_m_s"] = float(columns["com_vz_m_s"][-1] - reference_velocity[-1])
    expected = _metadata_value(metadata, ("expected_duration_s", "expected_stance_duration_s", "profile_duration_s"))
    warnings = []
    complete = None
    status = "unknown"
    if expected is None:
        warnings.append("Expected duration is absent. Stance-window completeness cannot be verified.")
    else:
        expected = float(expected)
        if not np.isfinite(expected) or expected <= 0.0:
            raise ValueError("metadata expected_duration_s must be finite and positive")
        tolerance = max(1.0e-9, expected * 1.0e-6)
        complete = abs(float(time[0])) <= tolerance and duration >= expected - tolerance
        status = "complete" if complete else "incomplete"
        if not complete:
            warnings.append(
                "INCOMPLETE STANCE WINDOW: totals cover only the saved interval. Do not compare them with full-window totals."
            )
        elif duration > expected + tolerance:
            status = "overrun"
            warnings.append(
                "Trace extends beyond the expected stance window. Work and impulse include the extra interval."
            )
    qualification = metadata.get("engineering_qualification", {})
    if qualification.get("passed") is not True:
        warnings.append(
            "ENGINEERING QUALIFICATION NOT PASSED: " + "; ".join(qualification.get("reasons", ["not recorded"]))
        )
    if metadata.get("shoe_identification_passed") is not True:
        warnings.append(
            "The source shoe identification has not passed all declared validation gates. These runs are exploratory mechanics, not validated footwear predictions."
        )
    if metrics["controller_clipped_time_fraction"] > 0.0:
        warnings.append(
            "The controller clipped during this run. Prescribed impedance commands were not always achieved."
        )
    if velocity_source.startswith("finite-difference"):
        warnings.append("Reference COM endpoint velocities are finite-difference estimates, not measured velocities.")
    return {
        "schema_version": 1,
        "metadata": metadata,
        "sample_count": len(time),
        "window": {
            "start_time_s": float(time[0]),
            "end_time_s": float(time[-1]),
            "duration_s": duration,
            "expected_duration_s": expected,
            "completion_fraction": duration / expected if expected is not None else None,
            "complete": complete,
            "status": status,
        },
        "metrics": metrics,
        "reference_com_velocity_source": velocity_source,
        "integration": "Trapezoidal time integration; signed work splits linear-power zero crossings. Force RMSE is time-weighted.",
        "warnings": warnings,
    }


def _audit(current: dict, previous: dict) -> dict:
    now, before = current["metadata"], previous["metadata"]
    changes = now.get("scenario_changes", now.get("scenario_change", []))
    if isinstance(changes, str):
        changes = [changes]
    explicit_scale_change = "shoe_stiffness_scale" in (changes or [])
    scale_now = now.get("shoe_stiffness_scale")
    scale_before = before.get("shoe_stiffness_scale")
    allowed_scale = (
        explicit_scale_change and scale_now is not None and scale_before is not None and scale_now != scale_before
    )
    checks = []
    for field, aliases in _AUDIT_FIELDS.items():
        value, old_value = _metadata_value(now, aliases), _metadata_value(before, aliases)
        if value is None or old_value is None:
            status = "missing"
        elif value == old_value:
            status = "match"
        elif field == "shoe_stiffness_scale" and allowed_scale:
            status = "declared scale scenario"
        else:
            status = "different"
        checks.append({"field": field, "current": value, "previous": old_value, "status": status})
    for field in ("start_time_s", "end_time_s", "expected_duration_s", "status"):
        value, old_value = current["window"][field], previous["window"][field]
        status = "missing" if value is None or old_value is None else ("match" if value == old_value else "different")
        checks.append({"field": f"window.{field}", "current": value, "previous": old_value, "status": status})
    for label, metadata in (("current", now), ("previous", before)):
        passed = metadata.get("engineering_qualification", {}).get("passed") is True
        checks.append(
            {
                "field": f"{label}.engineering_qualification",
                "current": passed,
                "previous": True,
                "status": "match" if passed else "not qualified",
            }
        )
    eligible = all(check["status"] in ("match", "declared scale scenario") for check in checks)
    eligible = eligible and current["window"]["status"] == previous["window"]["status"] == "complete"
    return {
        "eligible": eligible,
        "status": ("declared scale scenario" if allowed_scale else "matched settings")
        if eligible
        else "not a fair full-window comparison",
        "checks": checks,
        "note": "A scale scenario keeps the same source artifact and all rig settings; changing artifact bytes is not silently accepted as a pure stiffness change.",
    }


def _number(value) -> str:
    return "—" if value is None else f"{float(value):.6g}"


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _plot(title: str, unit: str, series: list[tuple]) -> str:
    """Draw all finite segments directly into an accessible inline SVG."""
    series = [(label, x, y, color, dashed) for label, x, y, color, dashed in series if np.any(np.isfinite(y))]
    if not series:
        return ""
    finite_y = np.concatenate([y[np.isfinite(y)] for _, _, y, _, _ in series])
    low, high = float(np.min(finite_y)), float(np.max(finite_y))
    margin = max((high - low) * 0.08, abs(high) * 0.005, 1.0e-6)
    low, high = low - margin, high + margin
    xmin = min(float(x[0]) for _, x, _, _, _ in series)
    xmax = max(float(x[-1]) for _, x, _, _, _ in series)
    width, height, left, top, plot_width, plot_height = 640, 285, 76, 18, 548, 217
    content = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}"><title>{_escape(title)}</title>'
    ]
    for fraction in np.linspace(0.0, 1.0, 5):
        sx, sy = left + fraction * plot_width, top + (1.0 - fraction) * plot_height
        content.append(f'<path d="M{left},{sy:.2f}h{plot_width}" class="grid"/>')
        content.append(
            f'<text x="{left - 8}" y="{sy + 4:.2f}" text-anchor="end">{_number(low + fraction * (high - low))}</text>'
        )
        content.append(
            f'<text x="{sx:.2f}" y="{top + plot_height + 20}" text-anchor="middle">{_number(xmin + fraction * (xmax - xmin))}</text>'
        )
    if low <= 0.0 <= high:
        zero = top + high / (high - low) * plot_height
        content.append(f'<path d="M{left},{zero:.2f}h{plot_width}" stroke="#99a7b9" stroke-dasharray="3 4"/>')
    for label, x, y, color, dashed in series:
        sx = left + (x - xmin) / (xmax - xmin) * plot_width
        sy = top + (high - y) / (high - low) * plot_height
        # Preserve every sample, including short force peaks; gaps mean unavailable data.
        pieces = []
        active = False
        for px, py in zip(sx, sy, strict=True):
            if np.isfinite(py):
                pieces.append(f"{'L' if active else 'M'}{px:.2f},{py:.2f}")
                active = True
            else:
                active = False
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        content.append(
            f'<path d="{" ".join(pieces)}" fill="none" stroke="{color}" stroke-width="2"{dash}><title>{_escape(label)}</title></path>'
        )
    content.append(f'<text x="{left + plot_width / 2}" y="277" text-anchor="middle">Elapsed time [s]</text></svg>')
    legend = " ".join(
        f'<span><i style="background:{color}"></i>{_escape(label)}{" (dashed)" if dashed else ""}</span>'
        for label, _, _, color, dashed in series
    )
    return f'<section class="plot"><h3>{_escape(title)} <small>[{_escape(unit)}]</small></h3>{"".join(content)}<div class="legend">{legend}</div></section>'


def _plots(columns: dict[str, np.ndarray], previous: dict[str, np.ndarray] | None) -> str:
    time = columns["time_s"] - columns["time_s"][0]
    old_time = previous["time_s"] - previous["time_s"][0] if previous is not None else None
    definitions = [
        (
            "Ground reaction force",
            "N",
            [
                ("shoe_fz_n", "Current shoe Fz", "#1967b3", False),
                ("reference_fz_n", "Reference Fz", "#b55214", True),
                ("reference_fx_n", "Reference Fx: context only", "#737e88", True),
            ],
        ),
        (
            "Fore-aft center of pressure",
            "m",
            [
                ("shoe_cop_x_m", "Current shoe CoP", "#1967b3", False),
                ("reference_cop_x_m", "Reference CoP", "#b55214", True),
            ],
        ),
        (
            "Fixture/ankle height (free in impedance mode)",
            "m",
            [
                ("foot_z_m", "Current fixture", "#1967b3", False),
                ("reference_foot_z_m", "Nominal inertial reference, not imposed", "#b55214", True),
            ],
        ),
        (
            "Force-integrated surrogate COM height",
            "m",
            [
                ("com_z_m", "Current surrogate COM", "#1967b3", False),
                ("reference_com_z_m", "Reference surrogate COM", "#b55214", True),
            ],
        ),
        (
            "Surrogate COM vertical velocity",
            "m/s",
            [
                ("com_vz_m_s", "Current surrogate COM", "#1967b3", False),
                ("reference_com_vz_m_s", "Reference surrogate COM", "#b55214", True),
            ],
        ),
        (
            "Mechanical power channels",
            "W",
            [
                ("active_power_w", "Active leg", "#1967b3", False),
                ("damping_power_w", "Leg damping", "#b55214", False),
                ("pitch_power_w", "Prescribed pitch drive", "#8054a0", False),
                ("shoe_contact_power_w", "Shoe contact", "#16806a", False),
            ],
        ),
        ("Vertical leg force", "N", [("leg_force_n", "Current leg force", "#1967b3", False)]),
        ("Vertical impedance engagement", "0-1", [("impedance_gain", "Quintic toe-off release", "#1967b3", False)]),
        (
            "Fixture release force",
            "N",
            [("retraction_force_n", "Gravity compensation plus optional lift", "#1967b3", False)],
        ),
        ("Peak spring compression", "m", [("max_compression_m", "Current shoe", "#1967b3", False)]),
        (
            "Prescribed fore-aft positions",
            "m",
            [("foot_x_m", "Foot", "#1967b3", False), ("com_x_m", "Surrogate COM", "#b55214", True)],
        ),
        (
            "Prescribed foot pitch",
            "rad",
            [
                ("pitch_rad", "Applied angle", "#1967b3", False),
                ("raw_pitch_rad", "Source angle before command smoothing", "#b55214", True),
            ],
        ),
        ("Ankle drive torque", "N m", [("ankle_torque_nm", "Pitch motor", "#1967b3", False)]),
        ("Rigid last ground clearance", "m", [("last_min_height_m", "Lowest rigid-last vertex", "#1967b3", False)]),
        (
            "Measured COP context (original heel-origin frame)",
            "m",
            [("source_cop_x_m", "Not position-registered to pitch-only rig", "#737e88", True)],
        ),
    ]
    plots = []
    for title, unit, channels in definitions:
        series = [
            (label, time, columns[key], color, dashed) for key, label, color, dashed in channels if key in columns
        ]
        first = channels[0][0]
        if previous is not None and first in previous:
            series.append(("Previous: " + channels[0][1], old_time, previous[first], "#b31b69", True))
        plots.append(_plot(title, unit, series))
    work_series = []
    for key, label, color in (
        ("active_power_w", "Net active leg", "#1967b3"),
        ("damping_power_w", "Damping loss", "#b55214"),
        ("pitch_power_w", "Net pitch drive", "#8054a0"),
        ("shoe_contact_power_w", "Net shoe contact", "#16806a"),
    ):
        power = columns[key] * (-1.0 if key == "damping_power_w" else 1.0)
        cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (power[:-1] + power[1:]) * np.diff(time))))
        work_series.append((label, time, cumulative, color, False))
    plots.insert(6, _plot("Cumulative work: separate channels", "J", work_series))
    plots.append(
        _plot("Surrogate COM energy", "J", [("Current COM energy", time, columns["com_energy_j"], "#1967b3", False)])
    )
    return "".join(plots)


def _render(summary: dict, columns: dict[str, np.ndarray], previous: dict | None, previous_columns: dict | None) -> str:
    metadata, metrics = summary["metadata"], summary["metrics"]
    audit = summary.get("comparison", {}).get("audit")
    table = []
    for key, label in _METRIC_LABELS.items():
        if key not in metrics and (previous is None or key not in previous["metrics"]):
            continue
        value = metrics.get(key)
        cells = f'<th scope="row">{_escape(label)}</th><td>{_number(value)}</td>'
        if previous is not None:
            old = previous["metrics"].get(key)
            delta = value - old if audit["eligible"] and value is not None and old is not None else None
            cells += f"<td>{_number(old)}</td><td>{_number(delta)}</td>"
        table.append(f"<tr>{cells}</tr>")
    comparison_html = "<p>No previous run was supplied.</p>"
    if audit is not None:
        checks = "".join(
            f'<tr><th scope="row">{_escape(check["field"])}</th><td>{_escape(check["status"])}</td>'
            f"<td><code>{_escape(json.dumps(check['current'], sort_keys=True))}</code></td>"
            f"<td><code>{_escape(json.dumps(check['previous'], sort_keys=True))}</code></td></tr>"
            for check in audit["checks"]
        )
        comparison_html = (
            f'<p class="{"ok" if audit["eligible"] else "warning"}"><strong>{_escape(audit["status"])}</strong>. '
        )
        comparison_html += (
            "Deltas are current minus previous."
            if audit["eligible"]
            else "Previous curves and values are context only. Metric deltas are suppressed."
        )
        comparison_html += f'</p><p>{_escape(audit["note"])}</p><div class="scroll"><table><thead><tr><th>Audit field</th><th>Status</th><th>Current</th><th>Previous</th></tr></thead><tbody>{checks}</tbody></table></div>'
    profile_limits = metadata.get(
        "profile_limits",
        [
            "Input locomotion classification must be checked in the profile provenance; this rig runs one stance, not a complete stride."
        ],
    )
    if isinstance(profile_limits, str):
        profile_limits = [profile_limits]
    limits = "".join(f"<li>{_escape(limit)}</li>" for limit in (*profile_limits, *_LIMITS))
    warnings = "".join(f"<li>{_escape(warning)}</li>" for warning in summary["warnings"])
    window = summary["window"]
    duration = f"{_number(window['duration_s'])} / {_number(window['expected_duration_s'])} s"
    compare_heading = "<th>Previous</th><th>Delta</th>" if previous is not None else ""
    metadata_json = _escape(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertical impedance Instron comparison</title><style>
:root{{font-family:system-ui,sans-serif;color:#192d42;background:#f2f5f8;line-height:1.5}}
body{{max-width:1320px;margin:auto;padding:24px}}h1,h2,h3{{line-height:1.25}}h1{{margin-bottom:8px}}
a{{color:#125c9b}}header,article,.plot{{background:white;border:1px solid #dce4ec;border-radius:10px;padding:20px;margin-bottom:18px}}
.subtitle,small{{color:#536577}}.badge{{display:inline-block;padding:4px 10px;background:#e7edf4;border-radius:6px;margin-right:8px}}
.warning{{background:#fff0d5;border-left:4px solid #bd6b00;padding:12px}}.ok{{background:#e6f3eb;padding:12px}}
.panels{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.plot{{min-width:0;margin:0;padding:14px}}
.plot h3{{font-size:1rem;margin:0 0 8px}}svg{{width:100%;height:auto}}svg text{{font:11px system-ui;fill:#536577}}.grid{{stroke:#e5eaf0;fill:none}}
.legend{{font-size:.78rem;display:flex;flex-wrap:wrap;gap:4px 14px}}.legend i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}th,td{{padding:8px 10px;border-bottom:1px solid #e5eaf0;text-align:left;vertical-align:top}}
td{{font-variant-numeric:tabular-nums}}tbody th{{font-weight:500}}thead{{background:#edf2f7}}code{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem}}li{{margin:5px 0}}.panels+article{{margin-top:18px}}
@media(max-width:800px){{body{{padding:10px}}.panels{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}.plot{{break-inside:avoid}}}}
</style></head><body><header><h1>Vertical impedance Instron</h1>
<p class="subtitle">Offline mechanical comparison · prescribed motion context · separate actuator and contact work</p>
<span class="badge">Window: {_escape(window["status"])}</span><span class="badge">Saved / expected: {duration}</span>
<span class="badge">{summary["sample_count"]} samples</span><p><a href="trace.csv">Download trace CSV</a> · <a href="summary.json">Download summary JSON</a></p>
</header><article><h2>Scope and limits</h2><ul>{limits}</ul>{f'<ul class="warning">{warnings}</ul>' if warnings else ""}</article>
<article><h2>Fair-comparison audit</h2>{comparison_html}</article>
<article><h2>Window metrics</h2><p>{_escape(summary["integration"])}</p>
<p>Negative work is signed; damping dissipation is positive. Reference COM velocity source: {_escape(summary["reference_com_velocity_source"])}.</p>
<div class="scroll"><table><thead><tr><th>Metric</th><th>Current</th>{compare_heading}</tr></thead><tbody>{"".join(table)}</tbody></table></div></article>
<div class="panels">{_plots(columns, previous_columns)}</div>
<article><details><summary>Run metadata and provenance</summary><pre>{metadata_json}</pre></details></article>
<footer>All plots are inline SVG. No network access, JavaScript, or plotting library is required. Curves use elapsed time from each trace start; the audit checks absolute window bounds.</footer>
</body></html>"""


def write_report(
    output_dir: Path, rows: list[dict[str, float]], metadata: dict, comparison: Path | None = None
) -> Path:
    """Export the trace, mechanical summary, and self-contained offline report.

    Args:
        output_dir: Directory for report.html, trace.csv, and summary.json.
        rows: Time-ordered SI trace samples; power is positive into its named system.
        metadata: Run settings and provenance. Fair comparisons require profile_hash,
            artifact_hash, mass_kg, stiffness_n_m, damping_n_s_m, dt_s, mode,
            initial_state, and shoe_stiffness_scale. expected_duration_s defines the
            complete saved window. profile_limits describes the source selection
            and locomotion classification. scenario_changes=["shoe_stiffness_scale"] explicitly
            declares a synthetic scale comparison; other differences remain flagged.
        comparison: Previous output directory containing trace.csv and summary.json.

    Returns:
        Path to report.html. The HTML has no external rendering dependencies.

    Raises:
        ValueError: Required channels, finite values, time order, or metadata are invalid.
    """
    output_dir = Path(output_dir)
    columns = _columns(rows)
    metadata = json.loads(json.dumps(metadata, default=_json_default, allow_nan=False))
    summary = _summarize(columns, metadata)
    previous = previous_columns = None
    if comparison is not None:
        comparison = Path(comparison)
        with (comparison / "trace.csv").open(newline="", encoding="utf-8") as stream:
            previous_rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        previous_columns = _columns(previous_rows)
        saved = json.loads((comparison / "summary.json").read_text(encoding="utf-8"))
        # Recompute metrics from the actual previous trace, not a potentially stale summary.
        previous = _summarize(previous_columns, saved["metadata"])
        summary["comparison"] = {
            "directory": str(comparison.resolve()),
            "audit": _audit(summary, previous),
            "previous_metrics": previous["metrics"],
        }
        if saved.get("metrics") != previous["metrics"]:
            summary["comparison"]["audit"]["eligible"] = False
            summary["comparison"]["audit"]["status"] = "saved previous metrics do not match trace"
            summary["warnings"].append(
                "Previous summary metrics differ from its trace. Previous values were recomputed and comparison deltas are suppressed."
            )
    document = _render(summary, columns, previous, previous_columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for index in range(len(rows)):
            writer.writerow({key: float(values[index]) for key, values in columns.items()})
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
