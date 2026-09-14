# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write a local, dependency-free kinematics and mechanics report."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def summarize_trace(trace: dict, dt: float) -> dict:
    """Report kinematic errors and mechanical outcomes without an effort objective."""
    z = np.asarray(trace["pelvis_z_m"]) - trace["reference_pelvis_z_m"]
    angle = np.asarray(trace["pitch_rad"]) - trace["reference_pitch_rad"]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    result = {
        "tracking_loss": float(np.sum(trace["tracking_error"]) * dt),
        "pelvis_rms_mm": float(np.sqrt(np.mean(z * z)) * 1000),
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(angle * angle)))),
        "peak_fz_n": float(np.max(trace["shoe_fz_n"])),
        "vertical_impulse_n_s": float(np.sum(trace["shoe_fz_n"]) * dt),
        "peak_compression_mm": float(np.max(trace["compression_m"]) * 1000),
        "minimum_last_clearance_mm": float(np.min(trace["last_clearance_m"]) * 1000),
    }
    for actuator in ("leg", "ankle"):
        power = np.asarray(trace[f"{actuator}_source_power_w"])
        result[f"{actuator}_positive_work_j"] = float(np.maximum(power, 0).sum() * dt)
        result[f"{actuator}_negative_work_j"] = float(np.minimum(power, 0).sum() * dt)
        result[f"{actuator}_net_work_j"] = float(power.sum() * dt)
        result[f"{actuator}_damper_work_j"] = float(np.asarray(trace[f"{actuator}_damping_power_w"]).sum() * dt)
    return result


def _panel(time, curves, title, unit):
    width, height = 800, 230
    left, right, top, bottom = 80, 20, 30, 35
    finite = [np.asarray(y)[np.isfinite(y)] for _, y in curves]
    values = np.concatenate([a for a in finite if a.size]) if any(a.size for a in finite) else np.array([0.0, 1.0])
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.08, abs(high) * 0.01, 1e-6)
    low, high = low - padding, high + padding
    t0, t1 = float(time[0]), float(time[-1])

    def sx(t):
        return left + (width - left - right) * (t - t0) / max(t1 - t0, 1e-12)

    def sy(y):
        return height - bottom - (height - top - bottom) * (y - low) / (high - low)

    parts = [
        f'<section><h3>{html.escape(title)}</h3><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
    ]
    for y in np.linspace(low, high, 5):
        yp = sy(y)
        parts.append(
            f'<path d="M{left},{yp:.2f}H{width - right}" stroke="#ddd"/><text x="{left - 8}" y="{yp + 4:.2f}" text-anchor="end">{y:.4g}</text>'
        )
    for t in np.linspace(t0, t1, 6):
        xp = sx(t)
        parts.append(f'<text x="{xp:.2f}" y="{height - 12}" text-anchor="middle">{t:.3f}</text>')
    # Downsample display only. The NPZ retains every solver sample.
    indices = np.unique(np.r_[np.arange(0, len(time), max(1, len(time) // 1600)), len(time) - 1])
    for j, (label, data) in enumerate(curves):
        points = " ".join(f"{sx(time[i]):.2f},{sy(data[i]):.2f}" for i in indices if np.isfinite(data[i]))
        dash = ' stroke-dasharray="6 4"' if "reference" in label.lower() else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[j % len(COLORS)]}" stroke-width="1.7"{dash}/>'
        )
    parts.append(f'<text x="12" y="18">{html.escape(unit)}</text></svg><p class="legend">')
    parts.extend(
        f'<span style="color:{COLORS[j % len(COLORS)]}">{html.escape(label)}</span>'
        for j, (label, _) in enumerate(curves)
    )
    parts.append("</p></section>")
    return "".join(parts)


def write_report(rig, output: str | Path, summary: dict | None = None) -> Path:
    """Save full traces and one offline HTML block of kinematics and mechanics.

    Args:
        rig: Completed simple rig, exposing named traces and frame timing.
        output: Report directory.
        summary: Optional training or frozen-policy evaluation metadata.
    """
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    trace = rig.trace(0)
    time = np.asarray(trace["time_s"])
    if len(time) < 2:
        raise ValueError("A report needs at least two simulated samples")
    dt = float(rig.frame_dt) / rig.config.substeps
    metrics = summarize_trace(trace, dt)
    for world in range(rig.num_worlds):
        np.savez_compressed(destination / f"trace_world_{world}.npz", **rig.trace(world))
    record = {"contract": "two-stiffness-pelvis-pitch-v1", "metrics_world_0": metrics, "evaluation": summary or {}}
    (destination / "summary.json").write_text(json.dumps(_jsonable(record), indent=2, allow_nan=False) + "\n")
    panels = [
        (
            "Pelvis centroid height — training target",
            "m",
            [("Upper mass", trace["pelvis_z_m"]), ("Measured reference", trace["reference_pelvis_z_m"])],
        ),
        (
            "Foot pitch — training target, not anatomical ankle angle",
            "rad",
            [
                ("Achieved", trace["pitch_rad"]),
                ("Measured reference", trace["reference_pitch_rad"]),
                ("Frozen equilibrium", trace["ankle_equilibrium_rad"]),
            ],
        ),
        ("Leg stiffness — policy output 1", "N/m", [("K leg", trace["leg_stiffness_n_m"])]),
        ("Ankle rotational stiffness — policy output 2", "N·m/rad", [("K ankle", trace["ankle_stiffness_n_m_rad"])]),
        ("Leg damping — fixed rule, not another policy output", "N·s/m", [("B leg", trace["leg_damping_n_s_m"])]),
        (
            "Ankle damping — fixed rule, not another policy output",
            "N·m·s/rad",
            [("B ankle", trace["ankle_damping_n_m_s_rad"])],
        ),
        (
            "Vertical GRF — evaluation only",
            "N",
            [("Simulated", trace["shoe_fz_n"]), ("Recorded reference", trace["reference_fz_n"])],
        ),
        (
            "Fore-aft GRF — evaluation only",
            "N",
            [("Simulated", trace["shoe_fx_n"]), ("Recorded reference", trace["reference_fx_n"])],
        ),
        (
            "Cumulative signed source work — evaluation only",
            "J",
            [
                ("Leg", np.cumsum(trace["leg_source_power_w"]) * dt),
                ("Ankle", np.cumsum(trace["ankle_source_power_w"]) * dt),
            ],
        ),
        ("Maximum column compression", "mm", [("Compression", np.asarray(trace["compression_m"]) * 1000)]),
    ]
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.6g}</td></tr>" for k, v in metrics.items())
    metadata = html.escape(json.dumps(_jsonable(summary or {}), indent=2, allow_nan=False))
    page = '<!doctype html><meta charset="utf-8"><title>Two-stiffness impedance</title>'
    page += "<style>body{font:15px system-ui;margin:2rem auto;max-width:1000px;padding:0 1rem;color:#222}svg{width:100%;font:12px system-ui}section{border-top:1px solid #bbb;padding-top:.5rem}.legend span{margin-right:1.5rem}td{padding:.2rem 1rem}pre{white-space:pre-wrap}</style>"
    page += "<h1>Two-stiffness impedance</h1><p><b>Reward:</b> measured pelvis height and foot pitch only. <b>Evaluation:</b> GRF and work. Equilibrium schedules and the damping rule are frozen. Time [s] is the same nominal physical clock in every panel.</p>"
    page += "<p>The upper lump represents the pelvis centroid as an engineering surrogate, not true whole-body COM. The angle is ground-relative foot pitch. Matching these signals does not establish physiological stiffness or same-shoe validation.</p>"
    page += f"<table>{rows}</table><details><summary>Run metadata and safety</summary><pre>{metadata}</pre></details>"
    page += "".join(_panel(time, curves, title, unit) for title, unit, curves in panels)
    page += "<p>Full-resolution data: trace_world_0.npz. Display curves may be decimated; saved data are not.</p>"
    path = destination / "report.html"
    path.write_text(page)
    return path
