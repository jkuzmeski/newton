# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a local, offline friction-force comparison without plotting dependencies."""

import html
from pathlib import Path

import numpy as np


def write_friction_report(
    report: dict, time_s: np.ndarray, forces: dict, reference: dict, output: Path, *, force_times: dict | None = None
) -> None:
    """Write force curves and braking/propulsion metrics without modifying inputs.

    The HTML contains restricted-data derivatives when the supplied reference
    does. It is an application diagnostic, not a physical acceptance certificate.
    """
    colors = {"measured": "#222", "bristle": "#1976d2", "implicit_bristle": "#d97706", "regularized": "#b91c1c"}
    sign = report["forward_sign"]
    curves = {"measured": (reference["grf_time_s"], sign * reference["grf_target_n"][:, 0])}
    clocks = force_times or {}
    curves.update({name: (clocks.get(name, time_s), sign * force[:, 0]) for name, force in forces.items()})
    upward = {"measured": (reference["grf_time_s"], reference["grf_target_n"][:, 1])}
    upward.update({name: (clocks.get(name, time_s), force[:, 1]) for name, force in forces.items()})
    for name, (clock, values) in [*curves.items(), *upward.items()]:
        if len(clock) != len(values) or len(clock) < 2:
            raise ValueError(f"Curve {name!r} needs matching force and time support")
        if not np.isfinite(clock).all() or not np.isfinite(values).all() or np.any(np.diff(clock) <= 0):
            raise ValueError(f"Curve {name!r} needs finite forces and a strictly increasing clock")
    palette = ("#1976d2", "#d97706", "#15803d", "#b91c1c", "#7c3aed")
    for index, name in enumerate(forces):
        colors.setdefault(name, palette[index % len(palette)])
    description = html.escape(
        report.get("description", "Prescribed leg motion; normal contact and compression unchanged.")
    )
    qualification = html.escape(
        report.get(
            "qualification",
            "No controller refit. This is not a free-dynamics or independent physical-friction validation.",
        )
    )

    def figure(names: list[str], title: str, component: int = 0, window: list[float] | None = None) -> str:
        plotted = curves if component == 0 else upward
        axis_label = "Forward force on body [N]" if component == 0 else "Upward force on body [N]"
        x0, x1 = float(time_s[0]), float(reference["grf_time_s"][-1])
        if window is not None:
            x0, x1 = window
            if not np.isfinite([x0, x1]).all() or x1 <= x0:
                raise ValueError("Zoom interval must be finite and increasing")
            plotted = {
                name: (clock[(clock >= x0) & (clock <= x1)], values[(clock >= x0) & (clock <= x1)])
                for name, (clock, values) in plotted.items()
            }
            if any(len(plotted[name][0]) < 2 for name in names):
                raise ValueError("Zoom interval needs at least two stored samples per curve")
        low = min(0.0, *(float(np.min(plotted[name][1])) for name in names))
        high = max(0.0, *(float(np.max(plotted[name][1])) for name in names))
        pad = max((high - low) * 0.08, 1.0)
        low, high = low - pad, high + pad
        pieces = [
            f'<svg viewBox="0 0 1000 410" role="img" aria-label="{html.escape(title)}">',
            '<rect width="1000" height="410" fill="white"/>',
        ]
        for value in np.linspace(low, high, 5):
            y = 340 - 280 * (value - low) / (high - low)
            pieces.append(f'<path d="M80 {y:.2f}H960" stroke="#ddd"/>')
            pieces.append(f'<text x="72" y="{y + 5:.2f}" text-anchor="end">{value:.0f}</text>')
        zero_y = 340 - 280 * (0 - low) / (high - low)
        pieces.append(f'<path d="M80 {zero_y:.2f}H960" stroke="#555" stroke-dasharray="4 4"/>')
        for value in np.linspace(x0, x1, 5):
            x = 80 + 880 * (value - x0) / max(x1 - x0, 1e-12)
            pieces.append(f'<text x="{x:.2f}" y="363" text-anchor="middle">{value:.3f}</text>')
        for index, name in enumerate(names):
            time, force = plotted[name]
            x = 80 + 880 * (time - x0) / max(x1 - x0, 1e-12)
            y = 340 - 280 * (force - low) / (high - low)
            points = " ".join(f"{px:.2f},{py:.2f}" for px, py in zip(x, y, strict=True))
            dash = ' stroke-dasharray="7 4"' if name == "implicit_bristle" else ""
            pieces.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="2"{dash}/>')
            pieces.append(
                f'<text x="{80 + index * (880 / max(len(names), 1)):.0f}" y="30" fill="{colors[name]}">{html.escape(name)}</text>'
            )
        pieces.extend(
            [
                '<text x="500" y="398" text-anchor="middle">Time [s]</text>',
                f'<text x="20" y="220" transform="rotate(-90 20 220)">{axis_label}</text>',
                "</svg>",
            ]
        )
        return f"<h2>{html.escape(title)}</h2>" + "".join(pieces)

    entries = []
    labels = [
        ("braking_peak_magnitude_n", "Braking peak [N]"),
        ("propulsive_peak_magnitude_n", "Propulsive peak [N]"),
        ("braking_impulse_ns", "Braking impulse [N s]"),
        ("propulsive_impulse_ns", "Propulsive impulse [N s]"),
    ]
    scores = report["scores"]
    first = next(iter(scores.values()))
    for key, label in labels:
        values = [first.get("reference_metrics", {}).get(key)]
        values.extend(score.get("trace_metrics", {}).get(key) for score in scores.values())
        cells = "".join(f"<td>{value:.3f}</td>" if value is not None else "<td>Not scored</td>" for value in values)
        entries.append(f"<tr><th>{label}</th>{cells}</tr>")
    names = ["measured", *scores]
    heading = "".join(f"<th>{html.escape(name)}</th>" for name in names)
    charts = figure([name for name in names if name != "regularized"], "Bristles and measured braking/propulsion")
    if "regularized" in scores:
        charts += figure(names, "All models (expanded force scale)")
    if report.get("zoom_interval_s") is not None:
        charts += figure(
            names, "Contact-transition detail (stored samples, no smoothing)", window=report["zoom_interval_s"]
        )
    charts += figure(names, "Upward force (same observation policy as horizontal force)", component=1)
    contents = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Friction-only leg replay</title>
<style>body{{font:16px sans-serif;max-width:1100px;margin:35px auto;padding:0 20px;color:#222}}svg{{width:100%}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #ccc;text-align:right}}p{{line-height:1.5}}</style>
<h1>Friction-only leg comparison</h1><p>{description}</p><p><strong>{qualification}</strong></p>
<p>Steps: {report["steps_evaluated"]} / {report["total_source_steps"]}. Partial smoke: {report["is_partial_smoke"]}.
Negative force is braking; positive force is propulsion. Curves use the saved source clocks, without force extrapolation.
Phase metrics use the declared measured-normal stance mask; excluded endpoint counts are in report.json.</p>
<table><tr><th>Metric</th>{heading}</tr>{"".join(entries)}</table>{charts}
<p>See <a href="report.json">report.json</a> for errors, timings, source hashes and qualification metadata.
Data and figures are local restricted derivatives; see ASSET_PROVENANCE.md before sharing.</p></html>"""
    Path(output).write_text(contents, encoding="utf-8")
