# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Estimate conditional marathon effects from saved material/geometry work."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

WORK_SOURCE = "https://www.nature.com/articles/s41598-021-04215-6"
PACE_SOURCE = "https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2019.00079/full"
COUPLINGS = (0.50, 0.65, 0.80)


def weighted_work(positive_j: float, absorbed_j: float) -> float:
    """Return a work cost proxy [J], with absorbed work supplied as a magnitude."""
    if positive_j < 0 or absorbed_j < 0:
        raise ValueError("Work magnitudes must be nonnegative")
    return 4.0 * positive_j + (1.0 / 1.2) * absorbed_j


def oxygen_rate(speed: float, mass_kg: float = 66.5, frontal_area_m2: float = 0.45) -> float:
    """Evaluate Kipp et al. Eq. 2 [ml/kg/min] with explicit drag scaling."""
    return 3.54 * frontal_area_m2 / mass_kg * speed**3 + 1.5355 * speed**2 + 1.5354 * speed + 15.661


def marathon_seconds_saved(economy_saving: float, speed: float = 3.65, mass_kg: float = 66.5) -> float:
    """Solve an unchanged sustainable oxygen-rate scenario for time saved [s]."""
    if not -1.0 < economy_saving < 1.0 or speed <= 0:
        raise ValueError("Require positive speed and economy saving between -1 and 1")
    target = oxygen_rate(speed, mass_kg) / (1.0 - economy_saving)
    low, high = 0.0, max(speed, 1.0)
    while oxygen_rate(high, mass_kg) < target:
        high *= 2
    if target <= oxygen_rate(0.0, mass_kg):
        raise ValueError("Scenario lies below the oxygen-rate model intercept")
    for _ in range(80):
        mid = (low + high) / 2
        if oxygen_rate(mid, mass_kg) < target:
            low = mid
        else:
            high = mid
    return 42195.0 / speed - 42195.0 / ((low + high) / 2)


def marathon_section(run: Path, metrics: dict) -> dict:
    """Save reproducible extrapolations and return an HTML report section."""
    split = metrics["splits"]["eval"]
    conditions = split["conditions"]
    baseline = conditions["original__reference"]["common_means"]

    def proxy(means):
        return weighted_work(means["total_positive_work_j"], means["total_absorbed_work_j"])

    base_proxy = proxy(baseline)
    rows = []
    for condition, item in conditions.items():
        means = item["common_means"]
        reduction = 1.0 - proxy(means) / base_proxy
        economy = [coupling * reduction for coupling in COUPLINGS]
        times = [marathon_seconds_saved(value) for value in economy]
        rows.append(
            {
                "condition": condition,
                "positive_work_j": means["total_positive_work_j"],
                "absorbed_work_j": means["total_absorbed_work_j"],
                "weighted_work_proxy_j": proxy(means),
                "proxy_reduction_fraction": reduction,
                "economy_saving_fractions": economy,
                "marathon_seconds_saved": times,
            }
        )
    artifact = {
        "schema": "conditional-marathon-work-proxy-v1",
        "metrics_sha256": hashlib.sha256((run / "metrics.json").read_bytes()).hexdigest(),
        "split": "eval",
        "paired_window_ids": split["common_completed_ids"],
        "baseline_condition": "original__reference",
        "assumptions": {
            "distance_m": 42195,
            "baseline_speed_m_s": 3.65,
            "baseline_time_s": 42195 / 3.65,
            "body_mass_kg": 66.5,
            "assumed_frontal_area_m2": 0.45,
            "positive_metabolic_j_per_mechanical_j": 4.0,
            "absorbed_metabolic_j_per_mechanical_j": 1 / 1.2,
            "assumed_effective_coupling_fractions": COUPLINGS,
            "economy_rule": "saving = coupling * (1 - condition_proxy / baseline_proxy)",
            "pace_rule": "(1 - saving) * oxygen_rate(new_speed) = oxygen_rate(baseline_speed)",
            "interpretation": "Heuristic gross metabolic coupling; not an identified muscle-work fraction or a confidence interval",
        },
        "limitations": [
            "Independent rotational actuator joint work is not measured muscle fascicle work; other work channels are excluded.",
            "Finite single-leg transient windows have unequal positive/negative work; they are not periodic gait cycles.",
            "Assume relative work changes persist across both legs and the full race, and sustainable oxygen rate is unchanged.",
            "No fatigue, foam aging, activation, co-contraction, tendon partition or added shoe mass is modeled.",
            "Paper foam blocks exceed the measured compression range; ranking and real metabolic benefit may reverse.",
            "No absolute calorie saving is identified from these traces.",
        ],
        "sources": [WORK_SOURCE, PACE_SOURCE],
        "conditions": rows,
    }
    (run / "marathon-estimates.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    labels = {"original": "Current", "ff_leap": "LEAP", "ff_turbo_plus": "TURBO PLUS"}
    geometries = {"reference": "original", "long125": "+25% length", "block40": "40 mm block"}
    names = [f"{labels[r['condition'].split('__')[0]]} / {geometries[r['condition'].split('__')[1]]}" for r in rows]
    # Fixed shared scale leaves room for labels on a narrow-screen SVG.
    values = [value / 60 for row in rows for value in row["marathon_seconds_saved"]]
    lower, upper = min(-0.5, *values) - 0.4, max(0.5, *values) + 0.4

    def x(value):
        return 240 + (value - lower) / (upper - lower) * 420

    svg = [
        '<svg viewBox="0 0 720 390" role="img" aria-label="Conditional marathon minutes saved"><title>Conditional marathon minutes saved; assumption ranges, not confidence intervals</title>'
    ]
    svg.append('<text x="240" y="24" font-size="16">Minutes saved (+) / lost (-)</text>')
    svg.append(f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="40" y2="330" stroke="#999"/>')
    for i, (row, label) in enumerate(zip(rows, names, strict=True)):
        y = 53 + i * 31
        a, central, b = [value / 60 for value in row["marathon_seconds_saved"]]
        color = "#237d67" if central >= 0 else "#ad4660"
        svg.append(f'<text x="8" y="{y + 4}" font-size="13">{html.escape(label)}</text>')
        svg.append(f'<line x1="{x(a):.2f}" x2="{x(b):.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        svg.append(f'<circle cx="{x(central):.2f}" cy="{y}" r="5" fill="{color}"/>')
    for value in range(int(lower) - 1, int(upper) + 2):
        if lower <= value <= upper:
            svg.append(f'<text x="{x(value):.2f}" y="350" text-anchor="middle" font-size="13">{value:+d}</text>')
    svg.append('<text x="240" y="377" font-size="12">Dot: 65% coupling; line: 50-80% coupling</text></svg>')
    table_rows = []
    for row, label in zip(rows, names, strict=True):
        economy = row["economy_saving_fractions"][1] * 100
        central = row["marathon_seconds_saved"][1] / 60
        a, b = sorted([row["marathon_seconds_saved"][0] / 60, row["marathon_seconds_saved"][2] / 60])
        table_rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{economy:+.2f}%</td><td>{central:+.2f}</td><td>{a:+.2f} to {b:+.2f}</td></tr>"
        )
    body = (
        "<p>Conditional extrapolation from nine matched held-out windows, all relative to the current material with original geometry. Baseline pace 3.65 m/s gives 3:12:40 for 42.195 km. Positive values below mean a saving.</p>"
        + "".join(svg)
        + '<div class="table-scroll"><table><thead><tr><th>Condition</th><th>Assumed economy saving</th><th>Minutes saved, central</th><th>Assumption range [min]</th></tr></thead><tbody>'
        + "".join(table_rows)
        + "</tbody></table></div>"
        + f'<p><a href="{WORK_SOURCE}">Riddick and Kuo (2022)</a> motivate a work proxy C = 4 W+ + |W-|/1.2, charging for both generation and absorption. This is a simplified independent-joint proxy, not their complete model: tendon contributions, inter-joint transfers and soft-tissue dissipation are unresolved here. They estimated substantial work-related metabolic cost in steady human running; that does not establish the coupling for our controller.</p>'
        + "<p>We assume gross metabolic saving = k * (1 - C/Cbaseline), with k = 0.65 centrally and 0.50-0.80 for sensitivity. These are chosen effective couplings, not fitted parameters, published universal constants, or statistical bounds. They absorb the unknown relation between this actuator-work proxy and whole-body metabolism.</p>"
        + f'<p>Time uses <a href="{PACE_SOURCE}">Kipp et al. (2019), Eq. 2</a>: Q(v) = (3.54 * A/m)v³ + 1.5355v² + 1.5354v + 15.661 ml/kg/min. We set body mass m = 66.5 kg and assume frontal area A = 0.45 m², then solve (1 - saving)Q(vnew) = Q(3.65). Sustainable oxygen rate is held constant, and the saving is assumed to persist at the new speed. This population equation is not a subject-specific calibration.</p>'
        + "<p><strong>Exploratory scenario, not a demonstrated marathon improvement.</strong> The saved single-leg windows are transient and have unequal generated/absorbed work; neither full periodic gait nor active muscle energy is identified. We assume their relative changes persist in both legs across the race. Fatigue, activation, co-contraction, tendon partition, foam aging and added foam mass are absent. Both paper-foam blocks exceed measured compression support. Real benefit may be zero or reversed; the displayed range is not a positive lower bound. Absolute calories cannot be inferred reliably.</p>"
    )
    return {"title": "Conditional marathon performance estimate", "kind": "biomechanics", "html": body}
