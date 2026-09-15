# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explain saved material, gain and recovery comparisons with paired figures."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .figures import BASELINE_COLOR, GAIN_COLORS, PALETTE, Curve, extract_svg, line_figure, scatter_figure

_GAIN_LABELS = {
    "leg_stiffness_n_m": "leg stiffness",
    "ankle_stiffness_n_m_rad": "pitch stiffness",
    "leg_damping_n_s_m": "leg damping",
    "ankle_damping_n_m_s_rad": "pitch damping",
}
_DIRECTION_COLORS = {"forward": "#0072b2", "backward": "#d55e00", "upward": "#009e73", "downward": "#cc79a7"}


def material_label(material):
    """Describe the varied property instead of exposing an opaque case ID."""
    kind, factors = material.get("type"), material.get("factors", {})
    if material.get("baseline") or material.get("id") == "baseline":
        return "Original material"
    if kind == "modulus":
        return f"Modulus {factors.get('modulus_multiplier', 1):g}x"
    if kind == "relaxation":
        return f"Maxwell time {factors.get('relaxation_multiplier', 1):g}x"
    return "Imported material: " + str(material.get("id", "unknown"))


def controller_label(controller):
    """Describe which one of the four impedance coefficients changes."""
    gain = controller.get("varied_gain")
    if not gain:
        identifier = controller.get("controller_id", controller.get("id"))
        return (
            "Nominal gains"
            if identifier == "nominal" or "varied_gain" in controller
            else "Controller " + str(identifier or "unspecified")
        )
    return f"{_GAIN_LABELS.get(gain, gain).capitalize()} {controller.get('multiplier', 1):g}x"


def _good(case):
    return case.get("status") == "valid" and bool(case.get("pair_valid"))


def _delta(value, baseline, angle=False):
    with np.errstate(invalid="ignore", over="ignore"):
        difference = np.asarray(value, dtype=float) - np.asarray(baseline, dtype=float)
        return np.arctan2(np.sin(difference), np.cos(difference)) if angle else difference


def _mode(cases):
    modes = list(dict.fromkeys(case.get("controller_mode", "intent") for case in cases))
    return "intent" if "intent" in modes else (modes[0] if modes else "intent")


def _notice(cases):
    invalid = [str(case["case_id"]) for case in cases if not _good(case)]
    if not invalid:
        return ""
    return (
        '<p class="invalid">Unqualified cases remain visible with warning styles: '
        + html.escape(", ".join(invalid))
        + ". Do not treat them as valid comparisons.</p>"
    )


def _export(destination, name, content):
    directory = destination / "figures"
    directory.mkdir(exist_ok=True)
    svg = extract_svg(content)
    if svg:
        (directory / f"{name}.svg").write_text(svg.replace('href="pages/', 'href="../pages/'), encoding="utf-8")
        content += f'<p class="figure-download"><a href="figures/{name}.svg">Open this figure as SVG</a></p>'
    return f'<div id="{name}" class="figure-wrap">{content}</div>'


def overview_figures(destination: Path, record: dict, load_trace) -> str:
    """Build three figure-led explanations from unchanged saved solver samples.

    Args:
        destination: Existing suite output directory; only figure assets are written.
        record: Saved summary with controller/material identities and paired outcomes.
        load_trace: Reader accepting (destination, relative_npz_path); no simulation.

    Returns:
        HTML sections with figures, plain-language captions and explicit missing-data
        notices. Overview comparisons use one named controller mode at a time.
    """
    cases = record.get("cases", [])
    mode = _mode(cases)
    selected = [case for case in cases if case.get("controller_mode", "intent") == mode]
    controllers = {item.get("controller_id", item.get("id")): item for item in record.get("controllers", [])}
    materials = {item["id"]: item for item in record.get("materials", [])}
    nominal = next((key for key, item in controllers.items() if not item.get("varied_gain")), "nominal")
    original = next(
        (key for key, item in materials.items() if item.get("baseline") or key == "baseline"),
        next(iter(materials), "baseline"),
    )
    cache = {}
    case_lookup = {case["case_id"]: case for case in cases}

    def baseline_matches(case):
        identifier = case.get("baseline_case_id")
        if identifier is None:
            return None
        baseline_case = case_lookup.get(identifier)
        if baseline_case is None:
            return False
        same_controller = baseline_case.get("controller_id") == case.get("controller_id")
        same_mode = baseline_case.get("controller_mode", "intent") == case.get("controller_mode", "intent")
        same_file = baseline_case.get("trace_file") == case.get("baseline_trace_file")
        expected_material = case.get("material_id") if case.get("direction") else original
        return same_controller and same_mode and same_file and baseline_case.get("material_id") == expected_material

    def read(path):
        if path not in cache:
            cache[path] = load_trace(destination, path)
        return cache[path]

    def curve(case, channel, label, color, *, paired=False, scale=1.0):
        trace = read(case.get("trace_file"))
        time = np.asarray(trace.get("time_s", []), dtype=float)
        value = np.asarray(trace.get(channel, []), dtype=float)
        if not time.size or value.shape != time.shape or not np.isfinite(time).all():
            return None
        if paired:
            if baseline_matches(case) is False:
                return None
            baseline = read(case.get("baseline_trace_file"))
            if (
                not np.array_equal(time, baseline.get("time_s", []))
                or np.shape(baseline.get(channel, [])) != time.shape
            ):
                return None
            value = _delta(value, baseline[channel], angle=channel == "pitch_rad")
        terminal_time = terminal_value = None
        recovery = case.get("recovery", {})
        if paired and recovery.get("terminal_clock_match"):
            terminal_time = recovery.get("final_state_time_s")
            terminal_value = recovery.get("deviations", {}).get(channel, {}).get("final_deviation")
        elif not paired and channel in case.get("terminal", {}):
            terminal_time = case["terminal"].get("time_s")
            terminal_value = case["terminal"].get(channel)
        if terminal_time is None or terminal_value is None or not np.isfinite([terminal_time, terminal_value]).all():
            terminal_time = terminal_value = None
        return Curve(
            label,
            time * 1000,
            value * scale,
            color,
            terminal_time=None if terminal_time is None else terminal_time * 1000,
            terminal_value=None if terminal_value is None else terminal_value * scale,
            qualified=_good(case) and (not paired or baseline_matches(case) is True),
        )

    def plot(group, *, name, title, channel, y_label, caption, labels, paired=False, scale=1.0, band=None, spans=()):
        lines, missing = [], []
        for case, (label, color) in zip(group, labels, strict=True):
            line = curve(case, channel, label, color, paired=paired, scale=scale)
            if line is None:
                missing.append(case["case_id"])
            else:
                lines.append(line)
        figure = line_figure(
            lines,
            title=title,
            x_label="Time [ms]",
            y_label=y_label,
            caption=caption,
            zero=paired,
            band=band,
            spans=spans,
        )
        if missing:
            figure += (
                '<p class="invalid">Curve unavailable (missing channel, baseline identity or paired clock): '
                + html.escape(", ".join(missing))
                + ".</p>"
            )
        return _export(destination, name, figure)

    def push_spans(group, *, dwell=False):
        if not group:
            return ()
        pairs = [
            (
                item.get("response_config", {}).get("push_start_s"),
                item.get("response_config", {}).get("push_duration_s"),
            )
            for item in group
        ]
        spans = []
        if all(pair == pairs[0] for pair in pairs) and all(value is not None for value in pairs[0]):
            start, duration = pairs[0]
            spans.append(
                {"start": start * 1000, "end": (start + duration) * 1000, "label": "Push active", "color": "#f6c85f"}
            )
        if dwell:
            recovery = group[0].get("recovery", {})
            final = recovery.get("final_state_time_s")
            duration = recovery.get("window", {}).get("required_dwell_s")
            if final is not None and duration is not None:
                spans.append(
                    {
                        "start": (final - duration) * 1000,
                        "end": final * 1000,
                        "label": "Final dwell",
                        "color": "#c9dceb",
                    }
                )
        return tuple(spans)

    output = [
        f'<p class="muted">Overview mode: <b>{html.escape(mode)}</b>. All other saved cases remain in the detailed results below.</p>',
        '<section id="materials"><h2>1. What did the shoe material change?</h2>',
        "<p>Read compression and ground force together. More compression does not automatically mean a lower ground-force peak. "
        "These are closed-loop results: the same feedback controller reacts to each shoe.</p>",
        '<p class="note">Synthetic material variants are engineering sensitivity tests, not newly calibrated or validated shoes. The material stays changed for the full episode.</p>',
    ]
    quiet = [case for case in selected if case.get("controller_id") == nominal and case.get("direction") is None]
    base = next((case for case in quiet if case.get("material_id") == original), None)
    families = list(
        dict.fromkeys(
            materials.get(case.get("material_id"), {}).get("type", "imported")
            for case in quiet
            if case.get("material_id") != original
        )
    )
    if not base or not families:
        output.append(
            '<p class="note">No nominal-controller material family is available in this saved subset. See the individual case plots below.</p>'
        )
    for family in families:
        variants = [
            case
            for case in quiet
            if case.get("material_id") != original
            and materials.get(case.get("material_id"), {}).get("type", "imported") == family
        ]
        for offset in range(0, len(variants), 3):
            group = ([base] if base else []) + variants[offset : offset + 3]
            if not base:
                continue
            labels = [
                (
                    material_label(materials.get(case.get("material_id"), {"id": case.get("material_id")})),
                    BASELINE_COLOR if case is base else GAIN_COLORS[(i - 1) % len(GAIN_COLORS)],
                )
                for i, case in enumerate(group)
            ]
            family_label = {
                "modulus": "Foam modulus",
                "relaxation": "Material relaxation time",
                "imported": "Imported materials",
            }.get(family, str(family))
            family_key = (
                family if family in ("modulus", "relaxation", "imported") else f"other-{families.index(family)}"
            )
            prefix = f"material-{family_key}-{offset}"
            output.append(
                f"<h3>{html.escape(family_label)}: same nominal controller, no push</h3>"
                + _notice(group)
                + '<div class="figure-grid">'
            )
            output.append(
                plot(
                    group,
                    name=prefix + "-compression",
                    title="How much does the foam compress?",
                    channel="compression_m",
                    scale=1000,
                    y_label="Max column compression [mm]",
                    labels=labels,
                    caption="Absolute maximum compression among the shoe columns, not average shoe shortening. Material remains changed for the whole episode; this is not a recovery test.",
                )
            )
            output.append(
                plot(
                    group,
                    name=prefix + "-force",
                    title="What force reaches the ground?",
                    channel="shoe_fz_n",
                    y_label="Vertical ground force [N]",
                    labels=labels,
                    caption="Absolute vertical ground-reaction force. Compare the timing and peak with compression; softer material need not lower this closed-loop force.",
                )
            )
            output.append(
                plot(
                    group,
                    name=prefix + "-force-difference",
                    title="Where is ground force higher or lower?",
                    channel="shoe_fz_n",
                    y_label="Ground force change [N]",
                    labels=labels,
                    paired=True,
                    caption="Quiet changed material minus quiet original material at the SAME controller. Positive means more ground force at that instant; negative means less. This reveals differences hidden by nearly overlapping absolute force curves, without time alignment.",
                )
            )
            output.append(
                plot(
                    group,
                    name=prefix + "-pelvis",
                    title="How does the upper mass move differently?",
                    channel="pelvis_z_m",
                    scale=1000,
                    y_label="Pelvis-height difference [mm]",
                    labels=labels,
                    paired=True,
                    caption="Each quiet material variant minus the quiet original material at the SAME controller setting. Zero means no height change relative to the original shoe, not zero tracking error.",
                )
            )
            output.append("</div>")
    output.append(
        '</section><section id="gains"><h2>2. What did the controller change?</h2>'
        "<p>Each comparison changes one gain only. Every pushed curve subtracts its OWN unpushed controller baseline on the original material. "
        "A gain can improve one response while worsening another.</p>"
    )
    pushes = [case for case in selected if case.get("comparison_type") == "push_recovery"]
    for direction in _DIRECTION_COLORS:
        direction_cases = [
            case for case in pushes if case.get("direction") == direction and case.get("material_id") == original
        ]
        nominal_push = next((case for case in direction_cases if case.get("controller_id") == nominal), None)
        if not nominal_push:
            continue
        output.append(
            f'<details class="comparison-group" {"open" if direction == "forward" else ""}><summary>{html.escape(direction.capitalize())} push: independent gain comparisons</summary>'
        )
        for gain in ("leg_damping_n_s_m", "leg_stiffness_n_m", "ankle_stiffness_n_m_rad", "ankle_damping_n_m_s_rad"):
            variants = [
                case
                for case in direction_cases
                if controllers.get(case.get("controller_id"), {}).get("varied_gain") == gain
            ]
            if not variants:
                continue
            group = [nominal_push, *variants]
            labels = [("Nominal gains", BASELINE_COLOR)] + [
                (controller_label(controllers[case["controller_id"]]), GAIN_COLORS[i % len(GAIN_COLORS)])
                for i, case in enumerate(variants)
            ]
            title = _GAIN_LABELS[gain].capitalize()
            output.append(
                f'<details class="gain-group" {"open" if gain == "leg_damping_n_s_m" else ""}><summary>{html.escape(title)}: compare motion AND velocity</summary>'
                + _notice(group)
                + '<div class="figure-grid">'
            )
            caption = f"Same {direction} push and original material; only {title.lower()} changes. Each curve is pushed minus its OWN quiet controller baseline. Shading marks the force-pulse support, not a rectangular force waveform."
            tolerances = record.get("recovery_config", {})
            z_band = float(tolerances.get("position_tolerance_m", 0.001)) * 1000
            v_band = float(tolerances.get("velocity_tolerance_m_s", 0.01)) * 1000
            output.append(
                plot(
                    group,
                    name=f"gain-{direction}-{gain}-height",
                    title="Vertical displacement caused by the push",
                    channel="pelvis_z_m",
                    scale=1000,
                    y_label="Pelvis-height difference [mm]",
                    labels=labels,
                    paired=True,
                    band=(-z_band, z_band),
                    spans=push_spans(group),
                    caption=caption,
                )
            )
            output.append(
                plot(
                    group,
                    name=f"gain-{direction}-{gain}-velocity",
                    title="Forward velocity left by the push",
                    channel="pelvis_vx_m_s",
                    scale=1000,
                    y_label="Forward speed change [mm/s]",
                    labels=labels,
                    paired=True,
                    band=(-v_band, v_band),
                    spans=push_spans(group),
                    caption="Near zero means forward speed matches its own unpushed run. The engineering band is only one component of the full return screen; lower vertical motion alone is not recovery. "
                    + caption,
                )
            )
            output.append("</div></details>")
        output.append("</details>")
    if not pushes:
        output.append('<p class="note">This saved subset contains no directional push comparisons.</p>')
    output.append('</section><section id="recovery"><h2>3. Why is safety not the same as recovery?</h2>')
    returned = sum(case.get("recovery", {}).get("status") == "returned_within_window" for case in pushes)
    not_returned = sum(case.get("recovery", {}).get("status") == "not_returned_within_window" for case in pushes)
    output.append(
        f"<p><b>{returned} of {len(pushes)} saved push comparisons returned within the declared screen; {not_returned} did not return within the observed window.</b> "
        "Physical safety checks and the all-channel recovery screen answer different questions. This does not prove instability or rule out later recovery.</p>"
    )
    representative = next(
        (
            case
            for case in pushes
            if case.get("controller_id") == nominal
            and case.get("material_id") == original
            and case.get("direction") == "forward"
        ),
        pushes[0] if pushes else None,
    )
    if representative:
        output.append(
            "<h3>A single push, explained over time</h3><p>The terminal marker is the true post-integration endpoint (circle for a qualified pair, cross otherwise). Yellow shading is the push; blue shading is the final required dwell. "
            'The displacement AND velocity curves must stay in their bands through that dwell, along with every other required body channel.</p><div class="figure-grid">'
        )
        dv = representative.get("recovery", {}).get("deviations", {})
        for channel, label, unit, fallback in (
            ("pelvis_x_m", "Remaining forward displacement", "mm", 0.001),
            ("pelvis_vx_m_s", "Remaining forward velocity", "mm/s", 0.01),
        ):
            tolerance = dv.get(channel, {}).get("tolerance", fallback) * 1000
            final = dv.get(channel, {}).get("final_deviation")
            endpoint = "unavailable" if final is None or not np.isfinite(final) else f"{final * 1000:.3g} {unit}"
            output.append(
                plot(
                    [representative],
                    name="recovery-" + channel,
                    title=label,
                    channel=channel,
                    scale=1000,
                    y_label=f"Forward difference [{unit}]",
                    labels=[
                        (
                            "Pushed minus own quiet run",
                            _DIRECTION_COLORS.get(representative.get("direction"), PALETTE[0]),
                        )
                    ],
                    paired=True,
                    band=(-tolerance, tolerance),
                    spans=push_spans([representative], dwell=True),
                    caption=f"Selected case: {representative['case_id']}. At the true endpoint, the difference is {endpoint}. Returning height alone does not erase remaining position or speed error.",
                )
            )
        output.append("</div>")
    output.append(
        "<h3>Where did all pushed runs finish?</h3><p>Each point is one saved pushed case at the true terminal time; color gives push direction. "
        'Click a point for that case. The center box shows the two plotted engineering bands only.</p><div class="figure-grid">'
    )
    for channel, rate, name, label in (
        ("pelvis_x_m", "pelvis_vx_m_s", "forward", "Forward"),
        ("pelvis_z_m", "pelvis_vz_m_s", "vertical", "Vertical"),
    ):
        points = []
        for case in pushes:
            recovery = case.get("recovery", {})
            dv = recovery.get("deviations", {})
            x, y = dv.get(channel, {}).get("final_deviation"), dv.get(rate, {}).get("final_deviation")
            if not recovery.get("terminal_clock_match"):
                x = y = None
            index = next(i for i, item in enumerate(cases) if item is case)
            points.append(
                {
                    "label": f"{case.get('direction')}: {controller_label(controllers.get(case.get('controller_id'), {}))}; {material_label(materials.get(case.get('material_id'), {'id': case.get('material_id')}))}",
                    "x": None if x is None else x * 1000,
                    "y": None if y is None else y * 1000,
                    "color": _DIRECTION_COLORS.get(case.get("direction"), PALETTE[0]),
                    "legend_label": str(case.get("direction", "unknown")).capitalize(),
                    "qualified": _good(case),
                    "href": f"pages/case_{index:04d}.html",
                }
            )
        position_band = record.get("recovery_config", {}).get("position_tolerance_m", 0.001) * 1000
        velocity_band = record.get("recovery_config", {}).get("velocity_tolerance_m_s", 0.01) * 1000
        content = scatter_figure(
            points,
            title=f"{label} position AND velocity at episode end",
            x_label=f"{label} position change [mm]",
            y_label=f"{label} speed change [mm/s]",
            x_band=(-position_band, position_band),
            y_band=(-velocity_band, velocity_band),
            caption="This is only a two-component projection. A point inside this box is NOT sufficient for full recovery: every required position/rate and the final dwell must pass. Invalid or missing endpoints are identified, never plotted as zero.",
        )
        output.append(_export(destination, "recovery-terminal-" + name, content))
    output.append("</div></section>")
    return "".join(output)
