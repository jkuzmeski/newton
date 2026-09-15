# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a small offline sensitivity index and raw-sample per-case reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .figures import PALETTE, Curve, line_figure
from .report import _jsonable
from .response import _difference, _panel
from .sensitivity_figures import controller_label, material_label, overview_figures

_POSITION_CHANNELS = {
    "pelvis_x_m": ("Upper-body forward position", "m"),
    "pelvis_z_m": ("Pelvis height", "m"),
    "pitch_rad": ("Foot pitch", "rad"),
    "foot_x_m": ("Foot forward position", "m"),
    "foot_z_m": ("Foot height", "m"),
    "leg_length_m": ("Leg length", "m"),
}
_RATE_CHANNELS = {
    "pelvis_vx_m_s": ("Upper-body forward velocity", "m/s"),
    "pelvis_vz_m_s": ("Upper-body vertical velocity", "m/s"),
    "pitch_rate_rad_s": ("Foot pitch rate", "rad/s"),
    "foot_vx_m_s": ("Foot forward velocity", "m/s"),
    "foot_vz_m_s": ("Foot vertical velocity", "m/s"),
    "leg_rate_m_s": ("Leg length rate", "m/s"),
}
_STYLE = """body{font:16px system-ui;max-width:1450px;margin:1.5rem auto;padding:1rem;color:#19232d}
h1{font-size:2rem;line-height:1.2}h2{font-size:1.6rem;margin-top:1.5rem}h3{font-size:1.15rem}
.figure-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:1.2rem;align-items:start}
.figure-wrap{min-width:0}.figure{margin:0;padding:.8rem;border:1px solid #d8e2eb;border-radius:10px;background:#fff}
.figure-caption{font-size:.92rem;line-height:1.5;color:#344054}.figure-download{font-size:.8rem;margin:.3rem .8rem}
.plot-legend{display:flex;flex-wrap:wrap;gap:.5rem 1rem;font-size:.85rem}.comparison-group>summary,.gain-group>summary{cursor:pointer;font-weight:600;padding:.7rem;background:#f0f5fa;border-radius:6px}
.reading-guide{padding:1rem 1.3rem;background:#edf5fc;border-radius:10px}.report-nav{display:flex;gap:1rem;flex-wrap:wrap}
.technical>summary{cursor:pointer;font-size:1.1rem;font-weight:600;padding:1rem;background:#f2f4f7}
@media(max-width:650px){body{padding:.6rem;margin:.4rem}h1{font-size:1.6rem}.figure{padding:.3rem}}

a{color:#07579a}table{border-collapse:collapse;font-size:13px}th,td{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}
th{background:#eef3f8}section{border-top:1px solid #ddd;margin-top:1rem}svg{width:100%;max-width:1000px;font:12px system-ui}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.invalid{color:#a00000;font-weight:bold}.note{background:#fff4d9;padding:.8rem}
.scroll{overflow-x:auto}details{margin:1rem 0}.muted{color:#576774}"""


def _text(value):
    return html.escape(str(value))


def _number(value, scale=1.0):
    if value is None or not np.isfinite(value):
        return "unavailable"
    return f"{float(value) * scale:.5g}"


def _json(value):
    return _text(json.dumps(_jsonable(value), indent=2, allow_nan=False))


def _begin(title):
    return [
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{_text(title)}</title>',
        f"<style>{_STYLE}</style><body><h1>{_text(title)}</h1>",
    ]


def _safe_local(destination, relative):
    path = (destination / relative).resolve()
    if not path.is_relative_to(destination.resolve()):
        raise ValueError("Report artifact must stay inside the suite output")
    return path


def _load_trace(destination, relative):
    if not relative:
        return {}
    with np.load(_safe_local(destination, relative), allow_pickle=False) as archive:
        return dict(archive)


def _case_page(destination, page_path, case, *, controller=None, material=None):
    action = str(case.get("direction") or "Quiet material comparison").capitalize()
    if case.get("direction"):
        action += " push"
    title = (
        f"{action}: {controller_label(controller or {})}; {material_label(material or {'id': case.get('material_id')})}"
    )
    page = _begin(title)
    page.append('<p><a href="../report.html">Back to sensitivity overview</a></p>')
    qualified = case.get("status") == "valid" and case.get("pair_valid", False)
    page.append(
        f'<p class="{"" if qualified else "invalid"}">Physical check: {_text(case.get("status"))}. '
        f"Paired comparison: {'valid' if qualified else 'INVALID PAIR — not a qualified response'}.</p>"
    )
    page.append(
        f"<p>Controller: {_text(case.get('controller_id'))}. Material: {_text(case.get('material_id'))}. "
        f"Comparison: {_text(case.get('comparison_type'))}. Direction: {_text(case.get('direction'))}.</p>"
    )
    recovery = case.get("recovery", {})
    page.append(f"<h2>Finite-window screen: {_text(recovery.get('status', 'unavailable'))}</h2>")
    page.append(
        '<p class="note">Position AND velocity must return inside declared engineering tolerances '
        "through the final dwell and the actual terminal state. This is not an identified human acceptance "
        "band, asymptotic stability, or a fitted settling-time claim. The screen covers planar body motion, "
        "not recovery of internal foam, Maxwell or friction history. A permanent material change is not "
        "a removed pulse; its response is material sensitivity, not recovery to the original material.</p>"
    )
    window = recovery.get("window", {})
    page.append(
        "<p>Observed post-push window: "
        + _number(window.get("observed_post_pulse_s"), 1000)
        + " ms. Required final in-band dwell: "
        + _number(window.get("required_dwell_s"), 1000)
        + " ms. Position and velocity must both pass; the full stored screen checks every required body channel.</p>"
    )
    page.append("<details><summary>Exact observation-window values</summary><pre>" + _json(window) + "</pre></details>")
    page.append(
        "<p>Plots show every raw pre-integration sample. Table final values use the actual terminal state. "
        "No smoothing, resampling, time alignment, or invented post-stride motion is applied.</p>"
    )
    trace = _load_trace(destination, case.get("trace_file"))
    baseline = _load_trace(destination, case.get("baseline_trace_file"))
    time = np.asarray(trace.get("time_s", []))
    paired = bool(time.size and np.array_equal(time, baseline.get("time_s", [])))
    page.append("<details><summary>Absolute trajectories: pushed/material case and its baseline</summary>")
    for name in ("pelvis_x_m", "pelvis_z_m", "pitch_rad"):
        if name in trace:
            curves = [("Case", trace[name])]
            if paired and name in baseline:
                curves.append(("Matching baseline", baseline[name]))
            label, unit = _POSITION_CHANNELS[name]
            page.append(_panel(time, curves, label, unit))
    page.append("</details>")
    deviations = recovery.get("deviations", {})
    page.append(
        '<h2>Paired position and velocity response</h2><details><summary>Exact paired values</summary><div class="scroll"><table><tr>'
        "<th>Channel</th><th>Peak including terminal</th><th>True final deviation</th><th>Tolerance</th></tr>"
    )
    for name, (label, unit) in {**_POSITION_CHANNELS, **_RATE_CHANNELS}.items():
        metric = deviations.get(name, {})
        page.append(
            f"<tr><td>{_text(label)} [{unit}]</td><td>{_number(metric.get('peak_abs_including_terminal_deviation'))}</td>"
            f"<td>{_number(metric.get('final_deviation'))}</td><td>{_number(metric.get('tolerance'))}</td></tr>"
        )
    page.append("</table></div></details>")
    spans = []
    settings = case.get("response_config", {})
    if case.get("direction") and "push_start_s" in settings and "push_duration_s" in settings:
        start = settings["push_start_s"]
        spans.append(
            {
                "start": start * 1000,
                "end": (start + settings["push_duration_s"]) * 1000,
                "label": "Push active",
                "color": "#f6c85f",
            }
        )
    final_time = recovery.get("final_state_time_s")
    dwell = window.get("required_dwell_s")
    if case.get("direction") and final_time is not None and dwell is not None:
        spans.append(
            {"start": (final_time - dwell) * 1000, "end": final_time * 1000, "label": "Final dwell", "color": "#c9dceb"}
        )
    if paired:
        page.append('<div class="figure-grid">')
        for name, (label, unit) in {**_POSITION_CHANNELS, **_RATE_CHANNELS}.items():
            if name not in trace or name not in baseline:
                continue
            delta = _difference(trace[name], baseline[name], angle=name == "pitch_rad")
            metric = deviations.get(name, {})
            tolerance = metric.get("tolerance")
            terminal_value = metric.get("final_deviation") if recovery.get("terminal_clock_match") else None
            terminal_time = final_time if terminal_value is not None else None
            if (
                terminal_time is None
                or terminal_value is None
                or not np.isfinite([terminal_time, terminal_value]).all()
            ):
                terminal_time = terminal_value = None
            display_unit = {"m": "mm", "rad": "mrad", "m/s": "mm/s", "rad/s": "mrad/s"}[unit]
            band = (-tolerance * 1000, tolerance * 1000) if tolerance is not None and np.isfinite(tolerance) else None
            curves = [
                Curve(
                    "Case minus matching baseline",
                    time * 1000,
                    delta * 1000,
                    PALETTE[0],
                    qualified=qualified,
                    terminal_time=None if terminal_time is None else terminal_time * 1000,
                    terminal_value=None if terminal_value is None else terminal_value * 1000,
                )
            ]
            page.append(
                line_figure(
                    curves,
                    title=label + " deviation",
                    x_label="Time [ms]",
                    y_label=f"Difference [{display_unit}]",
                    zero=True,
                    band=band,
                    spans=tuple(spans),
                    caption="Zero means agreement with the matching baseline. Declared engineering band is shaded; the terminal marker (circle or cross) shows the true endpoint when available. One returning channel does not pass the full screen.",
                )
            )
        page.append("</div>")
    else:
        page.append(
            '<p class="invalid">Paired traces are missing or use different clocks. No paired plot is fabricated.</p>'
        )
    panels = (
        ("External push force", "N", ("push_force_x_n", "push_force_z_n")),
        ("Shoe ground reaction", "N", ("shoe_fx_n", "shoe_fz_n")),
        ("Shoe compression", "m", ("compression_m",)),
        (
            "Leg nominal, feedback and delivered force",
            "N",
            ("leg_nominal_force_n", "leg_feedback_force_n", "leg_force_n"),
        ),
        (
            "Pitch nominal, feedback and delivered torque",
            "N·m",
            ("ankle_nominal_torque_n_m", "ankle_feedback_torque_n_m", "ankle_torque_n_m"),
        ),
        ("Leg body and idealized source power", "W", ("leg_body_power_w", "leg_source_power_w")),
        ("Pitch body and idealized source power", "W", ("ankle_body_power_w", "ankle_source_power_w")),
    )
    for label, unit, names in panels:
        curves = [(name, trace[name]) for name in names if name in trace]
        if curves:
            display_scale = 1000 if unit == "m" else 1
            display_unit = "mm" if unit == "m" else unit
            lines = [
                Curve(
                    name.replace("_", " "),
                    time * 1000,
                    np.asarray(values) * display_scale,
                    PALETTE[index % len(PALETTE)],
                    qualified=qualified,
                )
                for index, (name, values) in enumerate(curves)
            ]
            page.append(
                line_figure(
                    lines,
                    title=label,
                    x_label="Time [ms]",
                    y_label=display_unit,
                    caption="Saved pre-integration samples; no smoothing or extra terminal force evaluation.",
                    spans=tuple(spans),
                )
            )
    page.append(
        "<p>Nominal/feedback split is before shared force limiting, not a unique delivered split. "
        "Source work depends on virtual storage and damping conventions; it is not motor, electrical, "
        "metabolic or hardware efficiency. The full foam energy balance is not identified here.</p>"
    )
    page.append(
        "<details><summary>All measured metrics and frozen case settings</summary><pre>"
        + _json(case)
        + "</pre></details></body></html>"
    )
    page_path.write_text("".join(page), encoding="utf-8")


def write_sensitivity_report(destination: str | Path, record: dict, *, write_case_pages: bool = True) -> Path:
    """Write an offline index and per-case raw response pages without rerunning physics.

    Args:
        destination: Suite directory containing immutable snapshots and numeric traces.
        record: Completed or partial sensitivity manifest with cases and metrics.
        write_case_pages: Also redraw linked per-case pages; false updates only the overview.

    Returns:
        The small report.html index. Each linked case page embeds every plotted
        sample; NPZ files remain the authoritative full-resolution numeric record.
    """
    destination = Path(destination)
    pages = destination / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    page = _begin("What changed when we changed the shoe or controller?")
    page.append(
        '<div class="reading-guide"><h2>How to read this report</h2>'
        "<p>Start with the figures. Gray is the baseline; colored lines show a changed material or gain. "
        "In a difference plot, zero means the two compared runs agree. Markers show true terminal values: "
        "a hollow circle means a physically qualified pair, NOT successful recovery; a cross marks an unqualified pair.</p>"
        '<nav class="report-nav"><a href="#materials">1. Shoe material</a><a href="#gains">2. Controller gains</a>'
        '<a href="#recovery">3. Recovery</a><a href="#full-results">Exact tables and all cases</a></nav>'
        '<p class="muted">These figures use the saved solver samples and outcomes. Redrawing does not rerun the physics or change the results.</p></div>'
    )
    all_cases = record.get("cases", [])
    unqualified = [
        (index, case)
        for index, case in enumerate(all_cases)
        if case.get("status") != "valid" or not case.get("pair_valid", False)
    ]
    physical_count = sum(case.get("status") == "valid" for case in all_cases)
    page.append(
        f"<p><b>Physical checks: {physical_count}/{len(all_cases)} passed. Qualified pairs: {len(all_cases) - len(unqualified)}/{len(all_cases)}.</b> Neither count is the recovery result.</p>"
    )
    if unqualified:
        page.append(
            '<div class="note invalid"><h3>Failed or unqualified comparisons: do not overlook these cases</h3><ul>'
        )
        for index, case in unqualified:
            reasons = case.get("metrics", {}).get("evaluation", {}).get("safety_reasons", [])
            detail = "; ".join(
                str(reason) for group in reasons for reason in (group if isinstance(group, list) else [group])
            )
            detail = detail or case.get("error", {}).get("message") or "matching pair not qualified"
            page.append(
                f'<li><a href="pages/case_{index:04d}.html">{_text(case["case_id"])}</a>: {_text(case.get("status"))}; {_text(detail)}</li>'
            )
        page.append("</ul></div>")
    page.append(overview_figures(destination, record, _load_trace))
    page.append(
        '<details class="technical" id="full-results"><summary>Supporting detail: setup, exact values and every case</summary>'
    )
    page.append(
        "<p>Movement intent and nominal inverse-dynamics assistance remain frozen. "
        "Each controller setting changes only ONE of leg stiffness, pitch stiffness, leg damping or pitch damping. "
        "The original two-stiffness RL policy is not retrained or expanded.</p>"
    )
    page.append(
        '<p class="note"><b>Material perturbations are constant for an entire episode.</b> '
        "Modulus variants scale both Ogden-Hill shear terms and their physically derived neighbor coupling. "
        "Relaxation variants change Maxwell relaxation time, not controller damping or numerical surround relaxation. "
        "Synthetic variants are sensitivity hypotheses, not newly calibrated or validated shoes.</p>"
    )
    page.append(
        "<h2>Two different paired questions</h2><ul><li><b>Material sensitivity:</b> compare an unpushed "
        "changed-material run with the original-material unpushed run at the SAME controller setting.</li>"
        "<li><b>Push recovery:</b> compare a pushed run with its OWN material and controller unpushed run. "
        "Do not confuse a permanent material-induced motion change with failure to recover from a push.</li></ul>"
    )
    page.append(
        '<h2>Declared controller gains</h2><div class="scroll"><table><tr><th>Controller</th><th>Varied gain</th>'
        "<th>Factor</th><th>Leg K [N/m]</th><th>Pitch K [N·m/rad]</th><th>Leg B [N·s/m]</th><th>Pitch B [N·m·s/rad]</th></tr>"
    )
    for controller in record.get("controllers", []):
        gains = controller.get("gains", {})
        cells = [
            _text(controller.get("controller_id", controller.get("id"))),
            _text(controller.get("varied_gain")),
            _number(controller.get("multiplier")),
        ]
        cells.extend(
            _number(gains.get(name))
            for name in ("leg_stiffness_n_m", "ankle_stiffness_n_m_rad", "leg_damping_n_s_m", "ankle_damping_n_m_s_rad")
        )
        page.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    page.append(
        '</table></div><h2>Declared material variants</h2><div class="scroll"><table><tr><th>Material</th><th>Kind</th>'
        "<th>Modulus factor</th><th>Relaxation factor</th><th>Instantaneous G1 [kPa]</th><th>Instantaneous G2 [kPa]</th><th>Maxwell time [ms]</th><th>Qualification</th></tr>"
    )
    for material in record.get("materials", []):
        parameters, factors = material.get("parameters", {}), material.get("factors", {})
        kind = material.get("type", "unavailable")
        qualification = (
            "synthetic; not validated" if kind in ("modulus", "relaxation") else "source only; no transfer validation"
        )
        cells = [
            _text(material.get("id")),
            _text(kind),
            _number(factors.get("modulus_multiplier", 1.0 if kind == "baseline" else None)),
            _number(factors.get("relaxation_multiplier", 1.0 if kind == "baseline" else None)),
            _number(parameters.get("instantaneous_shear_modulus_pa"), 0.001),
            _number(parameters.get("instantaneous_shear_modulus_2_pa"), 0.001),
            _number(parameters.get("maxwell_relaxation_time_s"), 1000),
            qualification,
        ]
        page.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    page.append("</table></div>")
    cases = record.get("cases", [])
    valid = sum(case.get("status") == "valid" for case in cases)
    paired = sum(bool(case.get("pair_valid")) for case in cases)
    page.append(
        f"<p>Run status: {_text(record.get('status'))}. {len(cases)} cases; {valid} pass existing physical checks; "
        f"{paired} have valid paired comparisons. Invalid results remain visible. No gain or material is automatically ranked best.</p>"
    )
    if not record.get("suite_config", {}).get("full_factorial", False):
        page.append(
            "<p><b>Staged coverage:</b> material-only tests cover all controller settings. "
            "Pushes cover all controllers on the original material and all materials at nominal controller settings. "
            "Non-nominal controller and changed-material push interactions are NOT tested. Use full-factorial mode for them.</p>"
        )
    else:
        page.append(
            "<p><b>Full coverage:</b> the requested directional pushes cover every selected controller/material combination.</p>"
        )
    page.append(
        "<p>Recovery is an engineering finite-window screen. The recorded reference is not padded or held artificially. "
        "A short window is reported as insufficient; remaining displacement and velocity are still reported. "
        "Permanent material changes are reported as persistent_material_change, not transient recovery.</p>"
    )
    page.append(
        '<div class="scroll"><table><tr><th>Case / raw plots</th><th>Controller</th><th>Material</th>'
        "<th>Comparison</th><th>Physical / pair</th><th>Recovery screen</th><th>Peak pelvis Z [mm]</th>"
        "<th>Final upper X [mm]</th><th>Final upper VX [mm/s]</th><th>Final upper VZ [mm/s]</th></tr>"
    )
    controller_lookup = {item.get("controller_id", item.get("id")): item for item in record.get("controllers", [])}
    material_lookup = {item["id"]: item for item in record.get("materials", [])}
    for index, case in enumerate(cases):
        filename = f"case_{index:04d}.html"
        if write_case_pages:
            _case_page(
                destination,
                pages / filename,
                case,
                controller=controller_lookup.get(case.get("controller_id")),
                material=material_lookup.get(case.get("material_id")),
            )
        recovery = case.get("recovery", {})
        deviations = recovery.get("deviations", {})

        def get(channel, metric, data=deviations):
            return data.get(channel, {}).get(metric)

        good = case.get("status") == "valid" and case.get("pair_valid", False)
        cells = [
            f'<a href="pages/{quote(filename)}">{_text(case["case_id"])}</a>',
            _text(case.get("controller_id")),
            _text(case.get("material_id")),
            _text(case.get("comparison_type")),
            _text(case.get("status")) + (" / valid" if good else " / INVALID PAIR"),
            _text(recovery.get("status", "unavailable")),
            _number(get("pelvis_z_m", "peak_abs_including_terminal_deviation"), 1000),
            _number(get("pelvis_x_m", "final_deviation"), 1000),
            _number(get("pelvis_vx_m_s", "final_deviation"), 1000),
            _number(get("pelvis_vz_m_s", "final_deviation"), 1000),
        ]
        page.append(
            f'<tr class="{"" if good else "invalid"}">' + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
        )
    page.append("</table></div>")
    page.append(
        "<h2>Controller and material settings</h2><p>Only declared material parameters change; geometry, "
        "fixture, initial state, measured reference and nominal ID loads are not refitted. "
        "Imported alternatives must pass the material-only identity checks.</p>"
    )
    for name in ("controllers", "materials", "suite_config", "replay", "commands"):
        if name in record:
            page.append(f"<details><summary>{_text(name)}</summary><pre>{_json(record[name])}</pre></details>")
    page.append(
        '<p>Strict JSON: <a href="summary.json">summary.json</a>. Numeric records: cases/*.npz. '
        "Replay: replay.py with input, setting and source guards. This is not a complete archived software environment.</p>"
    )
    page.append(
        '<p class="muted">The upper mass represents pelvis motion, not whole-body COM. Pitch torque reacts '
        "against the world, not an anatomical shank. The recorded and modeled shoes differ; upper backing/contact "
        "registration remains idealized. Passing mechanical checks does not validate these assumptions.</p></details></body></html>"
    )
    path = destination / "report.html"
    path.write_text("".join(page), encoding="utf-8")
    return path
