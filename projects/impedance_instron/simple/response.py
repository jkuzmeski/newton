# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run paired fixed-gain response tests without training or changing the reference.

The suite compares each disturbed rollout with its own undisturbed mode/gain
pair. Damping stays identical across the stiffness sweep. Reports retain failed
cases and raw solver samples. They do not rank stiffness or identify frequency,
passivity, or settling properties from a short stride.
"""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import platform
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .policy import evaluate_policy
from .reference import Reference
from .report import _jsonable

if TYPE_CHECKING:
    from .rig import RigConfig

SCHEMA = "impedance_paired_response_1"
_CHANNELS = {
    "pelvis_z_m": ("Pelvis height", "m"),
    "pitch_rad": ("Foot pitch (not anatomical ankle angle)", "rad"),
    "leg_length_m": ("Leg length", "m"),
    "foot_x_m": ("Foot horizontal position", "m"),
    "pelvis_x_m": ("Upper-body horizontal position", "m"),
}


def _response_types():
    module = importlib.import_module(".response_control", __package__)
    return module.RigResponse, module.ResponseConfig


def _source_fingerprints():
    local = Path(__file__).parent
    paths = [local / name for name in ("response.py", "response_control.py", "rig.py", "reference.py", "policy.py")]
    shoe = local.parents[1] / "digital_shoe"
    paths.extend(shoe / name for name in ("runtime.py", "material.py", "contact.py"))
    root = local.parents[2]
    paths.extend((root / "newton" / "_src" / "solvers" / "semi_implicit").glob("*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _check_replay_sources(expected, *, allow_physics_update=False):
    current = _source_fingerprints()
    changed = {
        name: {"saved": expected.get(name), "current": current.get(name)}
        for name in expected.keys() | current.keys()
        if expected.get(name) != current.get(name)
    }
    if changed and not allow_physics_update:
        raise ValueError(
            "Response replay sources changed; old results do not apply. "
            "Use --allow-physics-update to run a new, explicitly marked experiment."
        )
    return changed


def _finite_stat(values, operation):
    values = np.asarray(values, dtype=float)
    return float(operation(values)) if values.size and np.isfinite(values).all() else None


def _integral(values, dt):
    return _finite_stat(values, lambda value: np.sum(value) * dt)


def _difference(value, baseline, *, angle=False):
    with np.errstate(invalid="ignore", over="ignore"):
        delta = np.asarray(value, dtype=float) - np.asarray(baseline, dtype=float)
        return np.arctan2(np.sin(delta), np.cos(delta)) if angle else delta


def _touchdown(trace, threshold):
    if "detected_touchdown_s" in trace:
        values = np.asarray(trace["detected_touchdown_s"], dtype=float)
        detected = values[np.isfinite(values) & (values >= 0)]
        return float(detected[0]) if detected.size else None
    time = np.asarray(trace.get("time_s", []))
    force = np.asarray(trace.get("shoe_fz_n", []))
    ids = np.flatnonzero(force >= threshold)
    return float(time[ids[0]]) if ids.size else None


def summarize_response(
    trace: dict,
    baseline_trace: dict,
    dt: float,
    *,
    terminal: dict | None = None,
    baseline_terminal: dict | None = None,
    evaluation: dict | None = None,
    contact_start_s: float = 0.0,
    contact_threshold_n: float = 20.0,
) -> dict:
    """Measure a fixed-clock perturbation relative to its matching baseline.

    Args:
        trace: Full pre-integration numeric traces for one disturbed world.
        baseline_trace: Undisturbed trace at the same mode, gains and damping.
        dt: Solver interval [s], used with the rectangular work rule.
        terminal: Actual final state channels after the last solver interval.
        baseline_terminal: Matching actual final baseline state channels.
        evaluation: Separate deterministic evaluation and safety report.
        contact_start_s: Recorded nominal touchdown time [s].
        contact_threshold_n: Simulated contact detection threshold [N].

    Returns:
        Paired deviations, physical diagnostics and explicit invalidity. A
        nonfinite sample invalidates its aggregate; no finite-only selection or
        clock interpolation is used. Final deviation is never a pre-step sample.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    terminal, baseline_terminal = terminal or {}, baseline_terminal or {}
    time = np.asarray(trace.get("time_s", []), dtype=float)
    base_time = np.asarray(baseline_trace.get("time_s", []), dtype=float)
    clocks_match = bool(time.size and np.array_equal(time, base_time) and np.isfinite(time).all())
    terminal_clock_match = bool(
        "time_s" in terminal
        and "time_s" in baseline_terminal
        and np.isfinite(terminal["time_s"])
        and np.isfinite(baseline_terminal["time_s"])
        and terminal["time_s"] == baseline_terminal["time_s"]
    )
    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(value)))
        for name, value in trace.items()
        if not np.isfinite(value).all()
    }
    deviations = {}
    for name in _CHANNELS:
        delta = np.array([])
        if clocks_match and name in trace and name in baseline_trace:
            if np.shape(trace[name]) == time.shape and np.shape(baseline_trace[name]) == time.shape:
                delta = _difference(trace[name], baseline_trace[name], angle=name == "pitch_rad")
        final = None
        if terminal_clock_match and name in terminal and name in baseline_terminal:
            value = _difference(terminal[name], baseline_terminal[name], angle=name == "pitch_rad")
            final = float(value) if np.isfinite(value).all() else None
        deviations[name] = {
            "peak_abs_deviation": _finite_stat(delta, lambda value: np.max(np.abs(value))),
            "last_trace_deviation": _finite_stat(delta[-1:], lambda value: value[-1]),
            "final_deviation": final,
            "trace_status": "finite" if delta.size and np.isfinite(delta).all() else "invalid_or_unavailable",
            "final_status": "finite" if final is not None else "invalid_or_unavailable",
        }
    terminal_nonfinite = [name for name, value in terminal.items() if not np.isfinite(value).all()]
    result = {
        "paired_clock_match": clocks_match,
        "terminal_clock_match": terminal_clock_match,
        "terminal_nonfinite_fields": terminal_nonfinite,
        "deviations": deviations,
        "nonfinite_trace_counts": nonfinite,
        "tracking_loss": _integral(trace.get("tracking_error", []), dt),
        "peak_shoe_fz_n": _finite_stat(trace.get("shoe_fz_n", []), np.max),
        "peak_abs_shoe_fx_n": _finite_stat(trace.get("shoe_fx_n", []), lambda value: np.max(np.abs(value))),
        "shoe_vertical_impulse_n_s": _integral(trace.get("shoe_fz_n", []), dt),
        "minimum_last_clearance_m": _finite_stat(trace.get("last_clearance_m", []), np.min),
        "peak_compression_m": _finite_stat(trace.get("compression_m", []), np.max),
        "sample_count": int(time.size),
        "last_trace_time_s": float(time[-1]) if time.size and np.isfinite(time[-1]) else None,
        "final_state_time_s": terminal.get("time_s"),
        "peak_includes_terminal_state": False,
        "evaluation": evaluation or {},
        "safety_flags_or": None,
        "physical_validity": "unknown",
    }
    flags = np.asarray(trace.get("safety_flags", []))
    if flags.size and np.isfinite(flags).all():
        result["safety_flags_or"] = int(np.bitwise_or.reduce(flags.astype(np.int64)))
    if evaluation:
        safety = evaluation.get("safety_ok", [])
        numerical = evaluation.get("numerical_ok", [])
        valid = bool(
            safety and all(safety) and numerical and all(numerical) and not nonfinite and not terminal_nonfinite
        )
        valid = valid and result["safety_flags_or"] == 0
        result["physical_validity"] = "valid" if valid else "invalid"
    elif nonfinite or terminal_nonfinite or result["safety_flags_or"]:
        result["physical_validity"] = "invalid"
    touchdown = _touchdown(trace, contact_threshold_n)
    baseline_touchdown = _touchdown(baseline_trace, contact_threshold_n)
    result["contact"] = {
        "touchdown_s": touchdown,
        "baseline_touchdown_s": baseline_touchdown,
        "touchdown_delta_vs_baseline_s": (
            touchdown - baseline_touchdown if touchdown is not None and baseline_touchdown is not None else None
        ),
        "touchdown_delta_vs_reference_s": touchdown - contact_start_s if touchdown is not None else None,
        "threshold_n": contact_threshold_n,
        "detected": touchdown is not None,
    }
    fx = np.asarray(trace.get("push_force_x_n", []))
    fz = np.asarray(trace.get("push_force_z_n", []))
    power = np.asarray(trace.get("push_power_w", []))
    result["push"] = {
        "impulse_x_n_s": _integral(fx, dt),
        "impulse_z_n_s": _integral(fz, dt),
        "signed_work_j": _integral(power, dt),
        "positive_work_j": _integral(np.maximum(power, 0), dt),
        "force_weighted_contact_fraction": None,
    }
    contact_force = np.asarray(trace.get("shoe_fz_n", []))
    if fx.shape == fz.shape == contact_force.shape and fx.size:
        magnitude = np.hypot(fx, fz)
        if np.isfinite(magnitude).all() and np.isfinite(contact_force).all() and magnitude.sum() > 0:
            result["push"]["force_weighted_contact_fraction"] = float(
                np.sum(magnitude * (contact_force >= contact_threshold_n)) / magnitude.sum()
            )
    result["actuators"] = {}
    for actuator, quantity, unit in (("leg", "force", "n"), ("ankle", "torque", "n_m")):
        source = np.asarray(trace.get(f"{actuator}_source_power_w", []))
        limited = np.asarray(trace.get(f"{actuator}_{quantity}_limited", []))
        result["actuators"][actuator] = {
            "signed_source_work_j": _integral(source, dt),
            "positive_source_work_j": _integral(np.maximum(source, 0), dt),
            "negative_source_work_j": _integral(np.minimum(source, 0), dt),
            "signed_body_work_j": _integral(trace.get(f"{actuator}_body_power_w", []), dt),
            "positive_body_work_j": _integral(np.maximum(trace.get(f"{actuator}_body_power_w", []), 0), dt),
            "negative_body_work_j": _integral(np.minimum(trace.get(f"{actuator}_body_power_w", []), 0), dt),
            "damper_work_j": _integral(trace.get(f"{actuator}_damping_power_w", []), dt),
            "limit_work_j": _integral(trace.get(f"{actuator}_limit_power_w", []), dt),
            f"peak_abs_{quantity}_{unit}": _finite_stat(
                trace.get(f"{actuator}_{quantity}_{unit}", []), lambda value: np.max(np.abs(value))
            ),
            f"peak_abs_raw_{quantity}_{unit}": _finite_stat(
                trace.get(f"{actuator}_raw_{quantity}_{unit}", []), lambda value: np.max(np.abs(value))
            ),
            "limited_sample_fraction": _finite_stat(limited, lambda value: np.mean(value != 0)),
            "load_split": "unclamped nominal and feedback; no unique delivered split after total-load saturation",
        }
        energy = np.asarray(trace.get(f"{actuator}_spring_energy_j", []))
        result["actuators"][actuator].update(
            spring_energy_first_sample_j=_finite_stat(energy[:1], lambda value: value[0]),
            spring_energy_last_sample_j=_finite_stat(energy[-1:], lambda value: value[-1]),
            spring_energy_sampled_change_j=_finite_stat(energy, lambda value: value[-1] - value[0]),
            spring_energy_timing="pre-integration sample endpoints, not the terminal spring state",
        )
        for component in ("nominal", "feedback"):
            result["actuators"][actuator][f"peak_abs_{component}_{quantity}_{unit}"] = _finite_stat(
                trace.get(f"{actuator}_{component}_{quantity}_{unit}", []), lambda value: np.max(np.abs(value))
            )
        for component in ("nominal_body", "target_motion", "stiffness_source"):
            power = np.asarray(trace.get(f"{actuator}_{component}_power_w", []))
            result["actuators"][actuator][f"{component}_signed_work_j"] = _integral(power, dt)
            result["actuators"][actuator][f"{component}_positive_work_j"] = _integral(np.maximum(power, 0), dt)
    return _jsonable(result)


def _terminal_state(rig):
    q = rig.state_0.body_q.numpy()
    qd = rig.state_0.body_qd.numpy()
    foot, pelvis = q[0], q[1]
    result = {
        "time_s": float(rig.sim_time),
        "pelvis_z_m": float(pelvis[2]),
        "pitch_rad": float(2 * np.arctan2(foot[4], foot[6])),
        "leg_length_m": float(np.linalg.norm(pelvis[:3] - foot[:3])),
        "foot_x_m": float(foot[0]),
        "pelvis_x_m": float(pelvis[0]),
        "foot_z_m": float(foot[2]),
    }
    return result, {"terminal_body_q": q, "terminal_body_qd": qd}


def _write_json(path, record):
    path.write_text(json.dumps(_jsonable(record), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _panel(time, curves, title, unit):
    """Draw every finite raw sample and break the line at nonfinite gaps."""
    time = np.asarray(time, dtype=float)
    curves = [(label, np.asarray(values, dtype=float)) for label, values in curves if np.shape(values) == time.shape]
    if not time.size or not np.isfinite(time).all():
        return f"<p>{html.escape(title)}: invalid or unavailable clock.</p>"
    finite = [values[np.isfinite(values)] for _, values in curves]
    values = (
        np.concatenate([values for values in finite if values.size])
        if any(v.size for v in finite)
        else np.array([0, 1])
    )
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.05, abs(high) * 0.001, 1e-8)
    low, high = low - pad, high + pad
    sx = 70 + 710 * (time - time[0]) / max(float(time[-1] - time[0]), 1e-12)
    parts = [
        f'<section><h4>{html.escape(title)} [{html.escape(unit)}]</h4><svg viewBox="0 0 800 220" role="img" aria-label="{html.escape(title)}">'
    ]
    for value in np.linspace(low, high, 4):
        y = 185 - 160 * (value - low) / (high - low)
        parts.append(
            f'<path d="M70,{y:.3f}H780" stroke="#ddd"/><text x="65" y="{y:.3f}" text-anchor="end">{value:.4g}</text>'
        )
    for value in np.linspace(time[0], time[-1], 5):
        x = 70 + 710 * (value - time[0]) / max(float(time[-1] - time[0]), 1e-12)
        parts.append(f'<text x="{x:.3f}" y="210" text-anchor="middle">{value:.4g}</text>')
    colors = ("#0072b2", "#d55e00", "#009e73")
    for index, (label, values) in enumerate(curves):
        points, started = [], False
        for x, value in zip(sx, values, strict=True):
            if not np.isfinite(value):
                started = False
                continue
            y = 185 - 160 * (value - low) / (high - low)
            points.append(f"{'L' if started else 'M'}{x:.3f},{y:.3f}")
            started = True
        color = colors[index % len(colors)]
        parts.append(
            f'<path d="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.3"><title>{html.escape(label)}</title></path>'
        )
    parts.append("</svg><p>Time [s]. ")
    parts.extend(
        f'<span style="color:{colors[i % len(colors)]}">{html.escape(label)}</span> '
        for i, (label, _) in enumerate(curves)
    )
    return "".join(parts) + "</p></section>"


def _response_load_trace(destination, name, cache):
    """Read a saved trace without executing a rollout or changing saved metrics."""
    if not name:
        return {}, "trace file not recorded"
    if name not in cache:
        try:
            with np.load(destination / name, allow_pickle=False) as data:
                cache[name] = (dict(data), "")
        except (OSError, ValueError, EOFError) as error:
            cache[name] = ({}, f"trace unavailable ({type(error).__name__})")
    return cache[name]


def _response_terminal(trace, channel):
    """Decode a plotted terminal channel using the saved reduced-rig body layout."""
    time = np.asarray(trace.get("terminal_time_s", []))
    if time.size != 1 or not np.isfinite(time).all():
        return None
    q = np.asarray(trace.get("terminal_body_q", []))
    qd = np.asarray(trace.get("terminal_body_qd", []))
    if channel == "pelvis_vx_m_s":
        if qd.shape != (2, 6):
            return None
        value = qd[1, 0]
    else:
        if q.shape != (2, 7):
            return None
        if channel == "pelvis_z_m":
            value = q[1, 2]
        elif channel == "pelvis_x_m":
            value = q[1, 0]
        elif channel == "pitch_rad":
            value = 2.0 * np.arctan2(q[0, 4], q[0, 6])
        else:
            return None
    return (float(time.item()), float(value)) if np.isfinite(value) else None


def _response_group_key(case, *, sweep=False, compare_modes=False, pair=False):
    """Separate figures when saved controller, gains, damping or identities differ."""
    config = dict(case.get("response_config", {}))
    multiplier = case.get("stiffness_multiplier")
    mode = case.get("controller_mode")
    if compare_modes:
        config.pop("controller_mode", None)
        mode = None
    if pair:
        for name in ("ground_height_m", "push_force_x_n", "push_force_z_n"):
            config.pop(name, None)
    if sweep:
        for name in ("leg_stiffness_n_m", "ankle_stiffness_n_m_rad"):
            if name in config and isinstance(multiplier, (int, float)) and multiplier > 0:
                config[name] /= multiplier
        multiplier = None
    return json.dumps(
        _jsonable(
            {
                "mode": mode,
                "multiplier": multiplier,
                "config": config,
                "inputs": case.get("input_fingerprints"),
                "rig": case.get("physics_metadata", {}).get("config"),
                "device": case.get("device"),
            }
        ),
        sort_keys=True,
        allow_nan=False,
    )


def _response_overview(destination, record, *, export_svg=False):
    """Build figure-first comparisons from saved samples, not fresh simulation."""
    from .figures import BASELINE_COLOR, GAIN_COLORS, Curve, extract_svg, line_figure  # noqa: PLC0415

    cases = record.get("cases", [])
    by_id = {case["case_id"]: case for case in cases}
    cache = {}
    page = []
    mode_names = {"equilibrium": "Equilibrium", "intent": "Movement intent"}
    figure_index = 0

    def figure(curves, **options):
        nonlocal figure_index
        rendered = line_figure(curves, **options)
        if export_svg:
            (destination / "figures").mkdir(exist_ok=True)
            relative = f"figures/response_{figure_index:02d}.svg"
            (destination / relative).write_text(extract_svg(rendered), encoding="utf-8")
            rendered += f'<p><a href="{relative}" download>Download SVG: ' + html.escape(options["title"]) + "</a></p>"
        figure_index += 1
        return rendered

    def groups(selected, **kwargs):
        grouped = {}
        for case in selected:
            grouped.setdefault(_response_group_key(case, **kwargs), []).append(case)
        return list(grouped.values())

    def gain_label(case):
        value = case.get("stiffness_multiplier")
        return f"K x {value:g}" if isinstance(value, (float, int)) else "K multiplier unavailable"

    def gain_color(case):
        multiplier = case.get("stiffness_multiplier", 1)
        return BASELINE_COLOR if multiplier == 1 else GAIN_COLORS[0 if multiplier < 1 else 1]

    def curve(case, channel, *, color, paired=False, tracking=False, mode_label=False):
        trace, error = _response_load_trace(destination, case.get("trace_file"), cache)
        label = f"{mode_names.get(case.get('controller_mode'), 'Unknown controller')} · {gain_label(case)}"
        if not mode_label:
            label = gain_label(case)
        label += f" · {case['case_id']}"
        time = np.asarray(trace.get("time_s", []), dtype=float)
        value = np.asarray(trace.get(channel, []), dtype=float)
        terminal = _response_terminal(trace, channel)
        qualified = case.get("status") == "valid" and bool(case.get("pair_valid"))
        if time.ndim != 1 or not time.size or value.shape != time.shape:
            error = error or f"{channel} or clock unavailable"
        if paired and not error:
            baseline_case = by_id.get(case.get("baseline_case_id"))
            if (
                baseline_case is None
                or baseline_case.get("perturbation") != "unperturbed"
                or baseline_case.get("trace_file") != case.get("baseline_trace_file")
                or _response_group_key(case, pair=True) != _response_group_key(baseline_case, pair=True)
            ):
                error = "matching controller/gain baseline metadata unavailable or mismatched"
            else:
                baseline, error = _response_load_trace(destination, baseline_case.get("trace_file"), cache)
                base_time = np.asarray(baseline.get("time_s", []))
                base_value = np.asarray(baseline.get(channel, []))
                if not error and (not np.array_equal(time, base_time) or value.shape != base_value.shape):
                    error = "paired clock or channel mismatch; no interpolation"
                if not error:
                    value = _difference(value, base_value, angle=channel == "pitch_rad")
                    base_terminal = _response_terminal(baseline, channel)
                    if terminal and base_terminal and terminal[0] == base_terminal[0]:
                        terminal = (terminal[0], terminal[1] - base_terminal[1])
                    else:
                        terminal = None
                    qualified = qualified and baseline_case.get("status") == "valid"
        if tracking and not error:
            reference = np.asarray(trace.get("reference_" + channel, []))
            if reference.shape != value.shape:
                error = "saved reference channel unavailable"
            else:
                value = _difference(value, reference, angle=channel == "pitch_rad")
            # The NPZ stores no reference value at the actual terminal clock.
            terminal = None
        if error:
            label += f" — unavailable: {error}"
            time, value, terminal = np.array([]), np.array([]), None
            qualified = False
        elif terminal is None:
            label += " — pre-integration only"
        return Curve(
            label=label,
            time=time * 1000,
            value=value * 1000,
            color=color,
            qualified=qualified,
            terminal_time=terminal[0] * 1000 if terminal else None,
            terminal_value=terminal[1] * 1000 if terminal else None,
        )

    terminal_caption = (
        "Lines retain every saved pre-integration sample; gaps are not connected. "
        "Separate terminal markers use terminal_body_q / terminal_body_qd at terminal_time_s when available. "
        "Otherwise the legend says pre-integration only. Invalid cases remain unqualified diagnostics."
    )
    unperturbed = [case for case in cases if case.get("perturbation") == "unperturbed"]
    page.append("<h2>1. Nominal gains: compare the two controller laws</h2>")
    page.append(
        "<p>At nominal gains, the rewritten movement-intent law is intended to give nearly the same motion "
        "as the equilibrium law. Compare the saved height and pitch curves below; this is not an assumed "
        "benefit or a claim of equivalence for failed runs.</p>"
    )
    nominal = [case for case in unperturbed if case.get("stiffness_multiplier") == 1]
    nominal_groups = groups(nominal, compare_modes=True)
    if not nominal_groups:
        page.append("<p>Nominal comparison unavailable: no unperturbed K x 1 cases were saved.</p>")
    for index, group in enumerate(nominal_groups, 1):
        modes = {case.get("controller_mode") for case in group}
        if not {"equilibrium", "intent"}.issubset(modes):
            page.append(
                "<p>Two-law overlay unavailable in this metadata group: one controller is missing. "
                "The available saved motion is still shown.</p>"
            )
        for channel, label, unit in (("pelvis_z_m", "Height", "mm"), ("pitch_rad", "Foot pitch", "mrad")):
            page.append(
                figure(
                    [
                        curve(
                            case,
                            channel,
                            color=BASELINE_COLOR if case.get("controller_mode") == "equilibrium" else GAIN_COLORS[0],
                            mode_label=True,
                        )
                        for case in group
                    ],
                    title=f"Nominal unperturbed {label.lower()} · group {index}",
                    y_label=f"{label} [{unit}]",
                    caption=terminal_caption,
                )
            )
    page.append("<h2>2. Change stiffness: nominal tracking can change too</h2>")
    page.append(
        "<p>Changing stiffness changes the nominal force term only in the old equilibrium formulation: "
        "its equilibrium schedule stays frozen. Movement intent keeps its frozen inverse-dynamics "
        "nominal load separate from feedback. Both controllers can still change their achieved motion. "
        "These are unperturbed tracking errors relative to the saved reference, not perturbation response. "
        "Both leg and ankle stiffness change together in this sweep; this is not one-at-a-time sensitivity. "
        "Larger stiffness is not automatically better. Dark is K x 1, blue is below nominal, "
        "and orange is above nominal; legends give the actual multipliers.</p>"
    )
    if not unperturbed:
        page.append("<p>Stiffness overview unavailable: no unperturbed cases were saved.</p>")
    for index, group in enumerate(groups(unperturbed, sweep=True), 1):
        mode = mode_names.get(group[0].get("controller_mode"), "Unknown controller")
        for channel, label, unit in (("pelvis_z_m", "Height error", "mm"), ("pitch_rad", "Foot-pitch error", "mrad")):
            page.append(
                figure(
                    [curve(case, channel, color=gain_color(case), tracking=True) for case in group],
                    title=f"{mode}: unperturbed {label.lower()} · group {index}",
                    y_label=f"{label} [{unit}]",
                    zero=True,
                    caption="Deviation from the saved movement reference. Pre-integration only: no terminal "
                    "reference is inferred. The two controller modes and differing fixed metadata "
                    "are kept in separate figures. This is not a stiffness ranking.",
                )
            )
    page.append("<h2>3. Applied push: displacement and velocity response</h2>")
    page.append(
        "<p>Every curve subtracts the unperturbed rollout for the SAME controller, stiffness and damping. "
        "It is a deviation from that matching baseline, not from the movement reference. "
        "Zero means no change from that baseline, not necessarily good tracking. "
        "Time is the saved physical clock in ms, with no alignment or time warping.</p>"
    )
    pushes = [case for case in cases if case.get("perturbation") == "push"]
    if not pushes:
        page.append("<p>Paired push response unavailable: no push cases were saved.</p>")
    for index, group in enumerate(groups(pushes, sweep=True), 1):
        mode = mode_names.get(group[0].get("controller_mode"), "Unknown controller")
        config = group[0].get("response_config", {})
        start, duration = config.get("push_start_s"), config.get("push_duration_s")
        spans = ()
        if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
            spans = (
                {
                    "start": start * 1000,
                    "end": (start + duration) * 1000,
                    "label": "Scheduled push",
                    "color": "#eeeeee",
                },
            )
        for channel, label, unit in (
            ("pelvis_x_m", "Upper-body forward displacement deviation", "mm"),
            ("pelvis_vx_m_s", "Upper-body forward velocity deviation", "mm/s"),
        ):
            page.append(
                figure(
                    [curve(case, channel, color=gain_color(case), paired=True) for case in group],
                    title=f"{mode}: {label.lower()} · group {index}",
                    y_label=f"Forward {'velocity' if channel == 'pelvis_vx_m_s' else 'position'} change [{unit}]",
                    zero=True,
                    spans=spans,
                    caption="Same-controller/gain baseline subtraction, not reference tracking. "
                    "The shaded interval is the scheduled push, not proof of simulated contact. " + terminal_caption,
                )
            )
    return "".join(page)


def _response_write_details(destination, record, output_path):
    page = [
        '<!doctype html><meta charset="utf-8"><title>Paired impedance response</title>',
        "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}svg{width:100%;font:12px system-ui}td,th{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:1rem 0}section{border-top:1px solid #ddd}.invalid{color:#b00}</style>",
        "<h1>Paired fixed-gain impedance response</h1>",
        "<p>Each perturbation is paired with its own unperturbed controller and stiffness. "
        "Damping is identical across stiffness multipliers and modes. No training, time warping, "
        "contact-triggered schedule reset, target change, or initial body shift is used. "
        "The raised/lowered plane is static from the start; it is not a moving step.</p>",
        "<h2>Two distinct controller laws</h2>"
        "<p><b>Equilibrium baseline:</b> F = K(L0-L) + B(L0dot-Ldot), with the originally frozen "
        "equilibrium schedules. Changing K does not rebuild L0. The rotational analogue uses the frozen "
        "ankle equilibrium. <b>Movement intent:</b> F = F_ID + K(Lref-L) + B(Lrefdot-Ldot); "
        "τ = τ_ID + Ka·wrap(pitchref-pitch) + Ba(pitchrefdot-pitchdot). Here Lref is measured "
        "movement, not the equilibrium schedule.</p>"
        "<p><b>Intent mode applies frozen reduced-rig inverse-dynamics loads at runtime.</b> "
        "This nominal assistance is declared, not predictive anatomy. The same total force and torque "
        "limits apply to nominal plus feedback loads. Plotted nominal/feedback components are unclamped "
        "commands; a saturated total has no unique delivered split. Source work separates nominal body "
        "power, target-motion power, stiffness-change power (zero at fixed gains), and limit intervention. "
        "Source work is not motor electrical energy or metabolic cost. Different controller realizations "
        "use different spring storage, damping and moving-reference power ledgers. Even identical body "
        "forces and trajectories can have different signed source work. This is not an efficiency saving "
        "or a hardware-energy comparison. Body work and sampled spring-energy changes are reported "
        "separately to make that distinction visible.</p>"
        "<p>All plots show every raw pre-integration solver sample, with no smoothing or decimation. "
        "Invalid gaps break curves. Peaks below cover recorded pre-integration states; final values "
        "use the actual state after the last solver interval. NPZ files include those terminal states.</p>",
        "<p><b>Interpretation:</b> larger stiffness is not automatically better. Tracking, excursion, "
        "force limits and signed/positive actuator work are separate outcomes. A short nonstationary "
        "stride does not establish a fitted natural frequency, damping ratio, settling time, or passivity. "
        "The upper body is a pelvis-height surrogate, not true COM. Foot pitch is not anatomical ankle angle. "
        "This is not same-shoe or physiological validation.</p>",
        "<p>A push scheduled inside recorded contact can miss actual simulated contact. The report retains "
        "its force-weighted actual contact fraction; no time shift corrects it. Deterministic commands "
        "do not imply bitwise reproducibility of GPU reductions.</p>",
        "<details><summary>Numeric case summary</summary><table><tr><th>Case</th><th>Validity</th><th>Tracking loss</th><th>Pelvis peak / final [m]</th>"
        "<th>Pitch peak / final [rad]</th><th>Leg peak / final [m]</th></tr>",
    ]

    if record.get("replay", {}).get("changed_sources"):
        page.insert(
            3,
            '<p class="invalid"><b>PHYSICS/SOURCE UPDATE:</b> this replay used changed code. '
            "It is a new experiment; old results do not apply. See replay metadata.</p>",
        )

    def number(value):
        return "null" if value is None else f"{value:.5g}"

    for case in record["cases"]:
        metrics = case.get("metrics", {})
        validity = case["status"]
        pair_label = "valid pair" if case["pair_valid"] else "INVALID PAIR"
        row = [
            html.escape(case["case_id"]),
            html.escape(f"{validity}; {pair_label}"),
            number(metrics.get("tracking_loss")),
        ]
        for channel in ("pelvis_z_m", "pitch_rad", "leg_length_m"):
            deviation = metrics.get("deviations", {}).get(channel, {})
            row.append(number(deviation.get("peak_abs_deviation")) + " / " + number(deviation.get("final_deviation")))
        css = ' class="invalid"' if validity != "valid" or not case["pair_valid"] else ""
        page.append(f"<tr{css}>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>")
    page.append("</table></details>")
    for case in record["cases"]:
        page.append(f"<details><summary>{html.escape(case['case_id'])} — {html.escape(case['status'])}</summary>")
        if not case["pair_valid"]:
            page.append(
                '<p class="invalid"><b>INVALID PAIR:</b> this case or its baseline failed validity, '
                "or their physical clocks differ. Finite plotted deltas are retained diagnostics, "
                "not a qualified response comparison.</p>"
            )
        page.append("<pre>" + html.escape(json.dumps(_jsonable(case), indent=2, allow_nan=False)) + "</pre>")
        if case.get("trace_file") and case.get("baseline_trace_file"):
            cache = {}
            trace, trace_error = _response_load_trace(destination, case["trace_file"], cache)
            baseline, baseline_error = _response_load_trace(destination, case["baseline_trace_file"], cache)
            if trace_error or baseline_error:
                page.append("<p>Saved trace unavailable: " + html.escape(trace_error or baseline_error) + "</p>")
            time = trace.get("time_s", np.array([]))
            paired = np.array_equal(time, baseline.get("time_s", []))
            for channel, (label, unit) in _CHANNELS.items():
                if channel not in trace:
                    continue
                curves = [("This case", trace[channel])]
                if paired and channel in baseline:
                    curves.append(("Matching unperturbed", baseline[channel]))
                page.append(_panel(time, curves, label, unit))
                if paired and channel in baseline and case["perturbation"] != "unperturbed":
                    delta = _difference(trace[channel], baseline[channel], angle=channel == "pitch_rad")
                    page.append(_panel(time, [("Paired deviation", delta)], label + " deviation", unit))
            panels = (
                ("Push force (upper body)", "N", ("push_force_x_n", "push_force_z_n")),
                ("Push power", "W", ("push_power_w",)),
                ("Shoe ground force", "N", ("shoe_fx_n", "shoe_fz_n")),
                ("Leg applied and unclamped force", "N", ("leg_force_n", "leg_raw_force_n")),
                ("Ankle applied and unclamped torque", "N·m", ("ankle_torque_n_m", "ankle_raw_torque_n_m")),
                (
                    "Leg nominal and feedback commands (before total saturation)",
                    "N",
                    ("leg_nominal_force_n", "leg_feedback_force_n"),
                ),
                (
                    "Ankle nominal and feedback commands (before total saturation)",
                    "N·m",
                    ("ankle_nominal_torque_n_m", "ankle_feedback_torque_n_m"),
                ),
                (
                    "Leg nominal body and target-motion power",
                    "W",
                    ("leg_nominal_body_power_w", "leg_target_motion_power_w"),
                ),
                (
                    "Ankle nominal body and target-motion power",
                    "W",
                    ("ankle_nominal_body_power_w", "ankle_target_motion_power_w"),
                ),
                ("Actuator source power (includes limits)", "W", ("leg_source_power_w", "ankle_source_power_w")),
                ("Actuator mechanical body power", "W", ("leg_body_power_w", "ankle_body_power_w")),
                ("Stored spring energy (pre-integration)", "J", ("leg_spring_energy_j", "ankle_spring_energy_j")),
                ("Leg stiffness", "N/m", ("leg_stiffness_n_m",)),
                ("Ankle stiffness", "N·m/rad", ("ankle_stiffness_n_m_rad",)),
                ("Leg damping", "N·s/m", ("leg_damping_n_s_m",)),
                ("Ankle damping", "N·m·s/rad", ("ankle_damping_n_m_s_rad",)),
            )
            for label, unit, names in panels:
                page.append(_panel(time, [(name, trace[name]) for name in names if name in trace], label, unit))
        page.append("</details>")
    page.append("<details><summary>Suite configuration, commands and identities</summary><pre>")
    page.append(
        html.escape(
            json.dumps(
                _jsonable({key: value for key, value in record.items() if key != "cases"}), indent=2, allow_nan=False
            )
        )
    )
    page.append(
        "</pre></details><p>Full numeric traces: cases/*.npz. Strict JSON: summary.json. "
        "Inputs and replay.py are included. All plot data are inline; no CDN or server is needed.</p>"
    )
    output_path.write_text("".join(page), encoding="utf-8")
    return output_path


def _write_html(destination, record):
    """Render a compact visual index and linked raw-case pages from saved data."""
    destination = Path(destination)
    page = [
        '<!doctype html><meta charset="utf-8"><title>Paired impedance response</title>',
        "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}"
        "svg{width:100%;font:12px system-ui}td,th{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:1rem 0}"
        "figure{margin:1.5rem 0}figcaption{line-height:1.5}.invalid{color:#a00}</style>",
        "<h1>Paired fixed-gain impedance response</h1>",
        "<p>Read the figures first: nominal motion, stiffness effects, then paired push response. "
        "These are saved reduced-rig simulation traces, not measured experimental responses. "
        "This rendering adds no simulation, fitted metrics or new qualification to the saved results.</p>",
    ]
    if record.get("replay", {}).get("changed_sources"):
        page.append(
            '<p class="invalid"><b>PHYSICS/SOURCE UPDATE:</b> this replay used changed code. '
            "It is a new experiment; old results do not apply. See replay metadata.</p>"
        )
    failed = [
        (index, case)
        for index, case in enumerate(record.get("cases", []))
        if case.get("status") != "valid" or not case.get("pair_valid")
    ]
    if failed:
        page.append(
            '<p class="invalid"><b>Invalid or failed cases retained below:</b> '
            "finite plotted deltas are diagnostics, not a qualified response comparison.</p><ul>"
        )
        for index, case in failed:
            reason = case.get("error", {}).get("message", "") or case.get("trace_error", {}).get("message", "")
            if not reason:
                metrics = case.get("metrics", {})
                saved_reasons = metrics.get("evaluation", {}).get("safety_reasons", [])
                reasons = [
                    str(item) for world in saved_reasons for item in (world if isinstance(world, list) else [world])
                ]
                if reasons:
                    reason = "Saved safety reasons: " + ", ".join(dict.fromkeys(reasons))
                elif metrics.get("nonfinite_trace_counts"):
                    reason = "Saved nonfinite trace counts: " + json.dumps(metrics["nonfinite_trace_counts"])
                elif metrics.get("safety_flags_or"):
                    reason = f"Saved safety flags: {metrics['safety_flags_or']}"
            page.append(
                '<li class="invalid">'
                + html.escape(
                    f"{case['case_id']} — {case.get('status', 'unknown')}; "
                    f"{'valid pair' if case.get('pair_valid') else 'INVALID PAIR'} {reason}"
                )
                + f' <a href="cases/response_{index:03d}.html">Raw case details</a></li>'
            )
        page.append("</ul>")
    page.append(_response_overview(destination, record, export_svg=True))
    page.append(
        "<h2>How to read these figures</h2>"
        "<p>Height is a pelvis-height surrogate, not true COM. Foot pitch is not anatomical ankle angle. "
        "The figures show separate channels, not a full-state recovery test. A short nonstationary stride "
        "does not establish natural frequency, damping ratio, settling time or passivity. "
        "There is no score-ranking winner; larger stiffness is not automatically better.</p>"
        "<details><summary>Controller laws and interpretation limits</summary>"
        "<p>Equilibrium: F = K(L0-L) + B(L0dot-Ldot), with frozen equilibrium schedules. "
        "Movement intent: F = F_ID + K(Lref-L) + B(Lrefdot-Ldot), with the corresponding rotational law. "
        "Intent mode applies frozen reduced-rig inverse-dynamics loads at runtime. "
        "This is nominal assistance, not predictive anatomy. Identical total force and torque limits "
        "apply to nominal plus feedback commands.</p>"
        "<p>Each perturbation is paired with its own unperturbed controller and stiffness. "
        "Damping is fixed across the suite stiffness multipliers. No training, target change or clock "
        "warping is introduced. Raised/lowered planes are static from the start, not moving steps. "
        "A scheduled push can miss actual simulated contact. Deterministic commands do not guarantee "
        "bitwise reproducibility of GPU reductions.</p>"
        "<p>Detailed pages retain nominal and feedback commands before total saturation, "
        "nominal body and target-motion power, and separate body work and spring-energy changes. "
        "Controller realizations can have different signed source-work ledgers even for identical body "
        "motion. This is not an efficiency saving, electrical-energy comparison, metabolic cost, "
        "same-shoe validation or physiological validation.</p></details>"
        "<details><summary>Numeric case table (saved metrics, no ranking)</summary>"
        "<p>Peak and final values below are deviations from the same-controller/gain baseline, "
        "not the movement reference. Peaks use saved pre-integration states; final metrics use the "
        "actual terminal state when recorded. Original SI units are retained in this table.</p>"
        "<table><tr><th>Case</th><th>Validity</th><th>Tracking loss</th>"
        "<th>Pelvis peak / final [m]</th><th>Pitch peak / final [rad]</th>"
        "<th>Leg peak / final [m]</th></tr>"
    )

    def number(value):
        return "null" if value is None else f"{value:.5g}"

    for case in record.get("cases", []):
        metrics = case.get("metrics", {})
        validity = case.get("status", "unknown")
        pair_label = "valid pair" if case.get("pair_valid") else "INVALID PAIR"
        row = [
            html.escape(case["case_id"]),
            html.escape(f"{validity}; {pair_label}"),
            number(metrics.get("tracking_loss")),
        ]
        for channel in ("pelvis_z_m", "pitch_rad", "leg_length_m"):
            deviation = metrics.get("deviations", {}).get(channel, {})
            row.append(number(deviation.get("peak_abs_deviation")) + " / " + number(deviation.get("final_deviation")))
        css = ' class="invalid"' if validity != "valid" or not case.get("pair_valid") else ""
        page.append(f"<tr{css}>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>")
    page.append(
        "</table></details><details><summary>Raw case plots and metadata</summary>"
        "<p>Linked pages retain every raw case, including failed and static-plane cases. "
        "Channels include Upper-body horizontal position, Foot horizontal position, "
        "force, torque and work diagnostics in their original SI units. "
        "The large raw SVGs are kept outside this visual index.</p><ul>"
    )
    (destination / "cases").mkdir(exist_ok=True)
    for index, case in enumerate(record.get("cases", [])):
        relative = f"cases/response_{index:03d}.html"
        _response_write_details(destination, {**record, "cases": [case]}, destination / relative)
        page.append(
            f'<li><a href="{relative}">{html.escape(case["case_id"])}</a> — '
            f"{html.escape(case.get('status', 'unknown'))}</li>"
        )
    page.append("</ul></details><details><summary>Suite configuration, commands and identities</summary><pre>")
    page.append(
        html.escape(
            json.dumps(
                _jsonable({key: value for key, value in record.items() if key != "cases"}), indent=2, allow_nan=False
            )
        )
    )
    page.append(
        "</pre></details><p>Full numeric traces: cases/*.npz. Strict JSON: summary.json. "
        "Inputs and replay.py are included. Figures use inline SVG; no CDN or server is needed.</p>"
    )
    path = destination / "report.html"
    path.write_text("".join(page), encoding="utf-8")
    return path


_REPLAY = '''# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay the saved suite from the Newton checkout into a new empty output."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from projects.impedance_instron.simple.response import (
    _check_replay_sources, _write_html, _write_json, run_response_suite,
)
from projects.impedance_instron.simple.rig import RigConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--device", default=None)
parser.add_argument("--allow-physics-update", action="store_true")
args = parser.parse_args()
source = Path(__file__).resolve().parent
record = json.loads((source / "summary.json").read_text())
for name, key in (("reference.json", "reference_snapshot_sha256"), ("artifact.json", "artifact_sha256")):
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != record[key]:
        raise ValueError(f"Saved replay input changed: {name}; create a separate suite instead")
changed = _check_replay_sources(record.get("source_fingerprints", {}), allow_physics_update=args.allow_physics_update)
if changed:
    print("WARNING: sources changed; this is a new experiment. Old results do not apply.")
options = record["suite_config"].copy()
options["config"] = RigConfig.from_dict(record["rig_config"]) if record["rig_config"] is not None else None
options["device"] = args.device or record["requested_device"]
options["command"] = [sys.executable, *sys.argv]
report = run_response_suite(source / "reference.json", source / "artifact.json", args.output, **options)
new_record = json.loads(report.with_name("summary.json").read_text())
new_record["replay"] = {"source_report": str(source), "changed_sources": changed,
                        "allow_physics_update": args.allow_physics_update,
                        "old_results_inherited": False, "source_matches": not bool(changed),
                        "requested_device_override": args.device}
_write_json(report.with_name("summary.json"), new_record)
_write_html(report.parent, new_record)
'''


def run_response_suite(
    reference: Reference | str | Path,
    artifact_path: str | Path,
    output: str | Path,
    *,
    device: str | None = None,
    config: RigConfig | None = None,
    modes: tuple[str, ...] = ("equilibrium", "intent"),
    stiffness_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    leg_stiffness_n_m: float = 12000.0,
    ankle_stiffness_n_m_rad: float = 4000.0,
    leg_damping_n_s_m: float | None = None,
    ankle_damping_n_m_s_rad: float | None = None,
    push_start_s: float | None = None,
    push_duration_s: float = 0.04,
    push_force_x_n: float = 150.0,
    push_force_z_n: float = 0.0,
    ground_offset_m: float = 0.005,
    command: list[str] | None = None,
) -> Path:
    """Run 24 paired one-world rollouts by default and write an offline report.

    Args:
        reference: Sealed reference or path to its JSON file; never modified.
        artifact_path: Identified shoe JSON with the matching fixed geometry.
        output: New or empty directory. Nonempty paths are never overwritten.
        device: Warp device; GPU graph replay is recommended for the full suite.
        config: Frozen rig physics settings, not a response controller config.
        modes: Controller laws to test independently.
        stiffness_multipliers: Common multipliers for both stiffnesses. Damping
            does not vary with this multiplier.
        leg_stiffness_n_m: Base translational stiffness [N/m].
        ankle_stiffness_n_m_rad: Base rotational stiffness [N·m/rad].
        leg_damping_n_s_m: Fixed damping [N·s/m], or the frozen nominal value.
        ankle_damping_n_m_s_rad: Fixed damping [N·m·s/rad], or nominal value.
        push_start_s: Physical start time [s], default recorded contact phase 0.35.
        push_duration_s: Duration of one bounded raised-cosine push [s].
        push_force_x_n: Peak forward upper-body force [N].
        push_force_z_n: Peak upward upper-body force [N].
        ground_offset_m: Magnitude of separate static raised/lowered planes [m].
        command: Optional exact invocation tokens to record for provenance.

    Returns:
        Self-contained report.html path. summary.json, input snapshots,
        replay.py and every numeric trace are saved beside it. Invalid rollouts
        remain explicit cases; they never become valid through pairing.
    """
    destination = Path(output).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Response output must be new or empty: {destination}")
    reference_path = Path(reference).resolve() if isinstance(reference, (str, Path)) else None
    reference = Reference.load(reference_path) if reference_path is not None else reference
    artifact_path = Path(artifact_path).resolve()
    artifact_bytes = artifact_path.read_bytes()
    modes, stiffness_multipliers = tuple(modes), tuple(float(value) for value in stiffness_multipliers)
    if not modes or len(set(modes)) != len(modes) or any(mode not in ("equilibrium", "intent") for mode in modes):
        raise ValueError("modes must contain distinct equilibrium/intent controller names")
    if not stiffness_multipliers or len(set(stiffness_multipliers)) != len(stiffness_multipliers):
        raise ValueError("stiffness_multipliers must be nonempty and distinct")
    if any(not math.isfinite(value) or value <= 0 for value in stiffness_multipliers):
        raise ValueError("stiffness multipliers must be positive and finite")
    if not math.isfinite(ground_offset_m) or ground_offset_m <= 0:
        raise ValueError("ground_offset_m must be positive and finite")
    push_start_s = (
        reference.contact_start_s + 0.35 * reference.contact_duration_s if push_start_s is None else float(push_start_s)
    )
    if not math.isfinite(push_duration_s) or push_duration_s <= 0:
        raise ValueError("push_duration_s must be positive and finite")
    if not (
        reference.contact_start_s <= push_start_s
        and push_start_s + push_duration_s <= reference.contact_start_s + reference.contact_duration_s
    ):
        raise ValueError("The complete push must lie inside the recorded contact interval; schedules are not retimed")
    if not np.isfinite([push_force_x_n, push_force_z_n]).all() or not np.hypot(push_force_x_n, push_force_z_n) > 0:
        raise ValueError("Specify a finite nonzero push")
    rig_type, response_type = _response_types()
    base = response_type(
        leg_stiffness_n_m=leg_stiffness_n_m,
        ankle_stiffness_n_m_rad=ankle_stiffness_n_m_rad,
        leg_damping_n_s_m=leg_damping_n_s_m,
        ankle_damping_n_m_s_rad=ankle_damping_n_m_s_rad,
        push_start_s=push_start_s,
        push_duration_s=push_duration_s,
    ).resolved(reference)
    suite = {
        "modes": modes,
        "stiffness_multipliers": stiffness_multipliers,
        "leg_stiffness_n_m": leg_stiffness_n_m,
        "ankle_stiffness_n_m_rad": ankle_stiffness_n_m_rad,
        "leg_damping_n_s_m": base.leg_damping_n_s_m,
        "ankle_damping_n_m_s_rad": base.ankle_damping_n_m_s_rad,
        "push_start_s": push_start_s,
        "push_duration_s": push_duration_s,
        "push_force_x_n": push_force_x_n,
        "push_force_z_n": push_force_z_n,
        "ground_offset_m": ground_offset_m,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cases").mkdir()
    reference.save(destination / "reference.json")
    (destination / "artifact.json").write_bytes(artifact_bytes)
    (destination / "replay.py").write_text(_REPLAY, encoding="utf-8")
    record = {
        "schema_version": SCHEMA,
        "status": "running",
        "suite_config": suite,
        "rig_config": config.to_dict() if config is not None else None,
        "requested_device": device,
        "num_worlds_per_case": 1,
        "commands": {
            "invocation_argv": command,
            "replay": shlex.join(
                ["uv", "run", "--no-sync", "python", str(destination / "replay.py"), "NEW_EMPTY_OUTPUT"]
            ),
            "replay_device_override": "Append --device cpu or --device cuda:0 if needed; run from this Newton checkout",
        },
        "reference_identity": reference.identity,
        "reference_source_path": str(reference_path) if reference_path else None,
        "reference_source_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest() if reference_path else None,
        "reference_snapshot_sha256": hashlib.sha256((destination / "reference.json").read_bytes()).hexdigest(),
        "artifact_source_path": str(artifact_path),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "workflow_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_fingerprints": _source_fingerprints(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "contract": {
            "pairing": "same controller mode, stiffness, damping, reference, initial state and physical clock",
            "damping": "fixed identical coefficients across modes and stiffness multipliers",
            "planes": "static from t=0; reference and initial bodies are not shifted",
            "push": "one raised-cosine force pulse at the upper-body COM; fixed nominal contact time",
            "work_quadrature": "sum of pre-integration power times sim_dt, including first and last interval",
            "final_state": "actual terminal body state; no additional foundation evaluation",
            "inference_limits": "no stiffness ranking, fitted frequency, passivity or settling claims",
            "reproducibility": "deterministic fixed controls; GPU reduction order can differ between runs",
        },
        "cases": [],
    }
    _write_json(destination / "summary.json", record)
    perturbations = (
        ("unperturbed", 0.0, 0.0, 0.0),
        ("push", 0.0, push_force_x_n, push_force_z_n),
        ("ground_raised", ground_offset_m, 0.0, 0.0),
        ("ground_lowered", -ground_offset_m, 0.0, 0.0),
    )
    for mode in modes:
        for multiplier in stiffness_multipliers:
            baseline_trace, baseline_terminal, baseline_case = {}, {}, None
            for perturbation, ground, fx, fz in perturbations:
                case_id = f"{len(record['cases']):02d}_{mode}_k{multiplier:g}_{perturbation}"
                settings = response_type(
                    controller_mode=mode,
                    leg_stiffness_n_m=leg_stiffness_n_m * multiplier,
                    ankle_stiffness_n_m_rad=ankle_stiffness_n_m_rad * multiplier,
                    leg_damping_n_s_m=base.leg_damping_n_s_m,
                    ankle_damping_n_m_s_rad=base.ankle_damping_n_m_s_rad,
                    ground_height_m=ground,
                    push_start_s=push_start_s,
                    push_duration_s=push_duration_s,
                    push_force_x_n=fx,
                    push_force_z_n=fz,
                )
                case = {
                    "case_id": case_id,
                    "controller_mode": mode,
                    "stiffness_multiplier": multiplier,
                    "perturbation": perturbation,
                    "response_config": asdict(settings),
                    "baseline_case_id": baseline_case["case_id"] if baseline_case else case_id,
                    "status": "execution_error",
                    "trace_file": None,
                    "requested_push_impulse_n_s": [0.5 * fx * push_duration_s, 0.5 * fz * push_duration_s],
                }
                rig, trace, terminal, terminal_arrays, evaluation = None, {}, {}, {}, {}
                try:
                    rig = rig_type(
                        reference,
                        destination / "artifact.json",
                        response_config=settings,
                        config=config,
                        num_worlds=1,
                        device=device,
                    )
                    record["rig_config"] = rig.config.to_dict()
                    case["physics_metadata"] = rig.metadata
                    case["input_fingerprints"] = rig.input_fingerprints
                    case["device"] = str(rig.device)
                    case["limits"] = {
                        "leg_force_n": rig.config.force_limit_bw * reference.mass_kg * reference.gravity_m_s2,
                        "ankle_torque_n_m": rig.config.ankle_torque_limit_n_m,
                    }
                    evaluation = evaluate_policy(rig)
                except Exception as error:
                    case["error"] = {"type": type(error).__name__, "message": str(error)}
                if rig is not None:
                    try:
                        trace = rig.trace(0)
                        terminal, terminal_arrays = _terminal_state(rig)
                    except Exception as error:
                        case["trace_error"] = {"type": type(error).__name__, "message": str(error)}
                    case["graph_status"] = rig.graph_status
                if trace:
                    case["trace_file"] = f"cases/{case_id}.npz"
                    np.savez_compressed(
                        destination / case["trace_file"],
                        **trace,
                        **terminal_arrays,
                        terminal_time_s=np.asarray(terminal.get("time_s", np.nan)),
                    )
                    case["trace_sha256"] = hashlib.sha256((destination / case["trace_file"]).read_bytes()).hexdigest()
                if perturbation == "unperturbed":
                    baseline_trace, baseline_terminal, baseline_case = trace, terminal, case
                case["baseline_trace_file"] = baseline_case["trace_file"]
                if trace and rig is not None:
                    case["metrics"] = summarize_response(
                        trace,
                        baseline_trace,
                        rig.sim_dt,
                        terminal=terminal,
                        baseline_terminal=baseline_terminal,
                        evaluation=evaluation,
                        contact_start_s=reference.contact_start_s,
                        contact_threshold_n=rig.config.contact_threshold_n,
                    )
                    if "error" not in case and "trace_error" not in case:
                        case["status"] = case["metrics"]["physical_validity"]
                case["pair_valid"] = bool(
                    case["status"] == "valid"
                    and baseline_case["status"] == "valid"
                    and case.get("metrics", {}).get("paired_clock_match")
                    and case.get("metrics", {}).get("terminal_clock_match")
                )
                case["baseline_status"] = baseline_case["status"]
                record["cases"].append(case)
                _write_json(destination / "summary.json", record)
                del rig
    record["status"] = "complete"
    record["valid_case_count"] = sum(case["status"] == "valid" for case in record["cases"])
    record["invalid_or_failed_case_count"] = len(record["cases"]) - record["valid_case_count"]
    _write_json(destination / "summary.json", record)
    return _write_html(destination, record)
