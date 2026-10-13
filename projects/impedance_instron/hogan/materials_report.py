# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a paired joint-work report for a frozen-controller materials run."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from projects.impedance_instron.hogan.comparison_viewer_report import build_comparison_viewer
from projects.impedance_instron.hogan.materials import PAPER_FOAMS, PAPER_URL
from projects.impedance_instron.hogan.spring_viewer_report import update_viewer
from projects.impedance_instron.report_catalog import rebuild

CONDITIONS = ("original", "original_elastic", "ff_leap", "ff_turbo_plus")
LABELS = {
    "original": "Original calibrated shoe",
    "original_elastic": "Original, elasticity only",
    "ff_leap": "FF LEAP foam surrogate",
    "ff_turbo_plus": "FF TURBO PLUS foam surrogate",
}
JOINTS = ("Hip", "Knee", "Ankle")
CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
RENDERER = CODEX_HOME / "skills" / "instron-reports" / "scripts" / "render_report.py"


def read_json(path: Path):
    """Load one saved JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """Write deterministic, finite JSON evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mean(values):
    """Return the arithmetic mean or None for an empty sample."""
    return sum(values) / len(values) if values else None


def condition_summary(run: Path, condition: str) -> dict:
    """Read one condition's saved summary and index its trials."""
    summary = read_json(run / condition / "summary.json")
    rows = summary.get("trials", [])
    return {"summary": summary, "rows": {str(row["id"]): row for row in rows}}


def metric_value(row: dict, key: str, joint: int | None = None):
    value = row["metrics"].get(key)
    if joint is not None:
        return value[joint] if value is not None and len(value) > joint else None
    return value


def common_mean(conditions: dict, name: str, ids: list[str], metric: str):
    """Average one scalar or vector metric on a fixed common sample set."""
    values = [conditions[name]["rows"][i]["metrics"].get(metric) for i in ids]
    dimension = next((len(value) for value in values if isinstance(value, list)), None)
    if dimension is not None:
        return [
            mean([value[j] for value in values if isinstance(value, list) and value[j] is not None])
            for j in range(dimension)
        ]
    return mean([value for value in values if value is not None])


def aggregate(conditions: dict, split: str) -> dict:
    """Build all-row and paired completed-case metrics for a split."""
    ids = {
        name: sorted(row_id for row_id, row in item["rows"].items() if row.get("split") == split)
        for name, item in conditions.items()
    }
    completed = {
        name: {
            row_id
            for row_id in ids[name]
            if conditions[name]["rows"][row_id].get("metrics", {}).get("status") == "completed"
        }
        for name in conditions
    }
    common = sorted(set.intersection(*(completed[name] for name in CONDITIONS)))
    aggregates = {}
    metrics = (
        "joint_positive_work_j",
        "joint_negative_work_j",
        "joint_net_work_j",
        "joint_peak_positive_power_w",
        "joint_peak_absorbed_power_w",
        "total_positive_work_j",
        "total_absorbed_work_j",
        "peak_grf_n",
        "grf_impulse_ns",
        "contact_duration_s",
        "peak_compression_fraction",
        "loss",
    )
    for name in CONDITIONS:
        rows = [conditions[name]["rows"][row_id]["metrics"] for row_id in sorted(completed[name])]
        aggregates[name] = {
            "n_trials": len(ids[name]),
            "n_completed": len(rows),
            "n_failed": len(ids[name]) - len(rows),
            "completed_ids": sorted(completed[name]),
            "all_completed_means": {
                metric: (
                    [
                        mean([r[metric][j] for r in rows if r.get(metric) is not None and len(r[metric]) > j])
                        for j in range(next((len(r[metric]) for r in rows if isinstance(r.get(metric), list)), 0))
                    ]
                    if any(isinstance(r.get(metric), list) for r in rows)
                    else mean([r[metric] for r in rows if r.get(metric) is not None])
                )
                for metric in metrics
            },
        }
    paired = {}
    for name in CONDITIONS[1:]:
        paired[name] = {"n_common_completed": len(common), "common_completed_ids": common, "metrics": {}}
        for metric in metrics:
            base_rows = [conditions["original"]["rows"][i]["metrics"].get(metric) for i in common]
            candidate_rows = [conditions[name]["rows"][i]["metrics"].get(metric) for i in common]
            if any(isinstance(value, list) for value in base_rows + candidate_rows):
                dimension = next(len(value) for value in base_rows + candidate_rows if isinstance(value, list))
                baseline = [
                    [metric_value(conditions["original"]["rows"][i], metric, j) for i in common]
                    for j in range(dimension)
                ]
                candidate = [
                    [metric_value(conditions[name]["rows"][i], metric, j) for i in common] for j in range(dimension)
                ]
                paired[name]["metrics"][metric] = {
                    "original_mean": [mean([v for v in values if v is not None]) for values in baseline],
                    "condition_mean": [mean([v for v in values if v is not None]) for values in candidate],
                    "mean_paired_difference": [
                        mean([b - a for a, b in zip(base, cand, strict=True) if a is not None and b is not None])
                        for base, cand in zip(baseline, candidate, strict=True)
                    ],
                }
            else:
                base = [metric_value(conditions["original"]["rows"][i], metric) for i in common]
                cand = [metric_value(conditions[name]["rows"][i], metric) for i in common]
                differences = [b - a for a, b in zip(base, cand, strict=True) if a is not None and b is not None]
                baseline_mean = mean([v for v in base if v is not None])
                delta_mean = mean(differences)
                paired[name]["metrics"][metric] = {
                    "original_mean": baseline_mean,
                    "condition_mean": mean([v for v in cand if v is not None]),
                    "mean_paired_difference": delta_mean,
                    "paired_difference_sample_sd": (
                        math.sqrt(sum((value - delta_mean) ** 2 for value in differences) / (len(differences) - 1))
                        if len(differences) > 1
                        else None
                    ),
                    "paired_difference_min": min(differences, default=None),
                    "paired_difference_max": max(differences, default=None),
                    "mean_difference_percent_of_original": (
                        100.0 * delta_mean / baseline_mean if delta_mean is not None and baseline_mean else None
                    ),
                }
    paired_vs_elastic = {}
    for name in ("original", "ff_leap", "ff_turbo_plus"):
        matched_ids = sorted(completed["original_elastic"] & completed[name])
        rows = {"n_common_completed": len(matched_ids), "common_completed_ids": matched_ids, "metrics": {}}
        for metric in metrics:
            ref = [conditions["original_elastic"]["rows"][i]["metrics"].get(metric) for i in matched_ids]
            candidate = [conditions[name]["rows"][i]["metrics"].get(metric) for i in matched_ids]
            if any(isinstance(value, list) for value in ref + candidate):
                dimension = next(len(value) for value in ref + candidate if isinstance(value, list))
                deltas = [
                    [b[j] - a[j] for a, b in zip(ref, candidate, strict=True) if a is not None and b is not None]
                    for j in range(dimension)
                ]
                rows["metrics"][metric] = {
                    "mean_paired_difference": [mean(values) for values in deltas],
                    "original_elastic_mean": [
                        mean([v[j] for v in ref if isinstance(v, list)]) for j in range(dimension)
                    ],
                    "condition_mean": [
                        mean([v[j] for v in candidate if isinstance(v, list)]) for j in range(dimension)
                    ],
                }
            else:
                differences = [b - a for a, b in zip(ref, candidate, strict=True) if a is not None and b is not None]
                baseline_mean = mean([value for value in ref if value is not None])
                delta_mean = mean(differences)
                rows["metrics"][metric] = {
                    "original_elastic_mean": baseline_mean,
                    "condition_mean": mean([value for value in candidate if value is not None]),
                    "mean_paired_difference": delta_mean,
                    "mean_difference_percent_of_reference": (
                        100.0 * delta_mean / baseline_mean if delta_mean is not None and baseline_mean else None
                    ),
                    "paired_difference_sample_sd": (
                        math.sqrt(sum((value - delta_mean) ** 2 for value in differences) / (len(differences) - 1))
                        if len(differences) > 1
                        else None
                    ),
                    "paired_difference_min": min(differences, default=None),
                    "paired_difference_max": max(differences, default=None),
                }
        paired_vs_elastic[name] = rows
    failed = {name: [row_id for row_id in ids[name] if row_id not in completed[name]] for name in CONDITIONS}
    return {
        "split": split,
        "conditions": aggregates,
        "paired_vs_original": paired,
        "paired_vs_original_elastic": paired_vs_elastic,
        "common_completed_ids": common,
        "failed_trial_ids": failed,
    }


def svg_grouped(metrics: dict, key: str, title: str, unit: str, *, signed: bool = False) -> str:
    """Draw accessible grouped condition/joint bars in self-contained SVG."""
    colors = ("#365f91", "#62a58c", "#eea83b", "#cf6658")
    labels = CONDITIONS
    values = {condition: metrics[condition] for condition in labels}
    numeric = [v for condition in labels for v in values[condition] if v is not None and math.isfinite(v)]
    if not numeric:
        return "<p>No completed matched observations are available for this plot.</p>"
    low = min([0.0, *numeric]) if signed else 0.0
    high = max([0.0, *numeric])
    span = max(high - low, abs(high) * 0.1, 1e-9)
    width, height, left, top = 780, 370, 78, 45
    plot_w, plot_h = width - left - 20, height - top - 60

    def y(value):
        return top + (high - value) / span * plot_h

    baseline = y(0.0)
    out = [
        f'<div class="plot"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<text x="{left}" y="22" font-size="16">{html.escape(title)} [{html.escape(unit)}]</text>',
        f'<line x1="{left}" y1="{baseline:.1f}" x2="{width - 15}" y2="{baseline:.1f}" stroke="#536878"/>',
    ]
    for tick in range(5):
        value = low + span * tick / 4
        tick_y = y(value)
        out.append(f'<line x1="{left}" y1="{tick_y:.1f}" x2="{width - 15}" y2="{tick_y:.1f}" stroke="#d9e2e8"/>')
        out.append(f'<text x="{left - 8}" y="{tick_y + 4:.1f}" text-anchor="end" font-size="10">{value:.3g}</text>')
    group_w = plot_w / 3
    bar_w = min(38, group_w / 5)
    for joint in range(3):
        center = left + group_w * (joint + 0.5)
        for c, condition in enumerate(labels):
            v = values[condition][joint]
            if v is None or not math.isfinite(v):
                continue
            x = center + (c - 1.5) * (bar_w + 3) - bar_w / 2
            yy = y(v)
            out.append(
                f'<rect x="{x:.1f}" y="{min(yy, baseline):.1f}" width="{bar_w:.1f}" height="{max(abs(baseline - yy), 1):.1f}" fill="{colors[c]}"><title>{html.escape(LABELS[condition])}, {JOINTS[joint]}: {v:.5g} {html.escape(unit)}</title></rect>'
            )
        out.append(
            f'<text x="{center:.1f}" y="{height - 34}" text-anchor="middle" font-size="14">{JOINTS[joint]}</text>'
        )
    for c, condition in enumerate(labels):
        x = left + c * 170
        out.append(
            f'<rect x="{x}" y="{height - 18}" width="12" height="12" fill="{colors[c]}"/><text x="{x + 17}" y="{height - 8}" font-size="11">{html.escape(LABELS[condition])}</text>'
        )
    out.append("</svg></div>")
    return "".join(out)


def svg_bars(values: dict, title: str, unit: str) -> str:
    """Draw a four-condition scalar comparison."""
    colors = ("#365f91", "#62a58c", "#eea83b", "#cf6658")
    numeric = [value for value in values.values() if value is not None]
    if not numeric:
        return "<p>No completed matched observations are available for this plot.</p>"
    high = max(numeric) * 1.15 or 1.0
    left, top, width, height = 66, 48, 720, 340
    plot_h = height - top - 76
    names = ("Original", "Original elastic", "FF LEAP", "FF TURBO+")
    bars = []
    for index, condition in enumerate(CONDITIONS):
        value = values[condition]
        x = left + 35 + index * 155
        bar_h = value / high * plot_h
        bars.append(
            f'<rect x="{x}" y="{top + plot_h - bar_h:.1f}" width="74" height="{bar_h:.1f}" fill="{colors[index]}"><title>{html.escape(LABELS[condition])}: {value:.5g} {html.escape(unit)}</title></rect><text x="{x + 37}" y="{top + plot_h + 22}" text-anchor="middle" font-size="12">{names[index]}</text><text x="{x + 37}" y="{top + plot_h - bar_h - 5:.1f}" text-anchor="middle" font-size="11">{value:.4g}</text>'
        )
    return (
        f'<div class="plot"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} by material condition">'
        f'<title>{html.escape(title)} by material condition</title><text x="{left}" y="24" font-size="16">{html.escape(title)} [{html.escape(unit)}]</text>'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 18}" y2="{top + plot_h}" stroke="#536878"/>{"".join(bars)}</svg></div>'
    )


def contact_metrics_table(conditions: dict, ids: list[str]) -> str:
    """Summarize held-out GRF, impulse, contact duration and peak compression."""
    out = [
        f"<p>Condition means on the same {len(ids)} common completed held-out trial IDs as the work charts. GRF and impulse components are horizontal x / vertical z.</p>",
        '<div style="overflow-x:auto"><table><thead><tr><th>Condition</th><th>Peak GRF x / z [N]</th><th>GRF impulse x / z [N·s]</th><th>Contact [ms]</th><th>Peak compression [%]</th></tr></thead><tbody>',
    ]
    for name in CONDITIONS:
        peak = common_mean(conditions, name, ids, "peak_grf_n")
        impulse = common_mean(conditions, name, ids, "grf_impulse_ns")
        duration = common_mean(conditions, name, ids, "contact_duration_s")
        compression = common_mean(conditions, name, ids, "peak_compression_fraction")
        values = (
            LABELS[name],
            f"{fmt(peak[0], 1)} / {fmt(peak[1], 1)}",
            f"{fmt(impulse[0], 2)} / {fmt(impulse[1], 2)}",
            fmt(duration * 1000.0, 2),
            fmt(compression * 100.0, 2),
        )
        out.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def native_fz_svg(run: Path, conditions: dict) -> str:
    """Overlay each condition's saved native Fz trace for the same held-out trial."""
    sample_id = "FR3_1_eval_000"
    traces = {}
    for name in CONDITIONS:
        trial = conditions[name]["rows"].get(sample_id)
        if trial is None:
            return f"<p>Native trace {sample_id} is missing in condition {html.escape(name)}.</p>"
        with np.load(run / name / trial["trace"]) as archive:
            traces[name] = (archive["time_s"][:-1].copy(), archive["grf_n"][:, 1].copy())
    colors = ("#365f91", "#62a58c", "#eea83b", "#cf6658")
    max_t = max(float(time[-1]) for time, _ in traces.values())
    max_f = max(float(force.max()) for _, force in traces.values()) * 1.08
    width, height, left, top = 780, 390, 72, 45
    plot_w, plot_h = width - left - 22, height - top - 62
    lines = []
    for index, name in enumerate(CONDITIONS):
        time, force = traces[name]
        points = " ".join(
            f"{left + t / max_t * plot_w:.1f},{top + (1.0 - f / max_f) * plot_h:.1f}"
            for t, f in zip(time, force, strict=True)
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width="1.8"/>')
    axes = [
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 15}" y2="{top + plot_h}" stroke="#536878"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#536878"/>',
    ]
    for tick in range(4):
        t = max_t * tick / 3
        x = left + t / max_t * plot_w
        axes.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e2e8ec"/><text x="{x:.1f}" y="{top + plot_h + 17}" text-anchor="middle" font-size="10">{t:.2f}</text>'
        )
        force_tick = max_f * tick / 3
        y = top + (1.0 - force_tick / max_f) * plot_h
        axes.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - 15}" y2="{y:.1f}" stroke="#e2e8ec"/><text x="{left - 7}" y="{y + 3:.1f}" text-anchor="end" font-size="10">{force_tick:.0f}</text>'
        )
    legend = []
    for index, name in enumerate(CONDITIONS):
        x = left + 15 + index * 175
        legend.append(
            f'<line x1="{x}" y1="{height - 13}" x2="{x + 18}" y2="{height - 13}" stroke="{colors[index]}" stroke-width="3"/><text x="{x + 23}" y="{height - 9}" font-size="11">{html.escape(LABELS[name])}</text>'
        )
    return (
        f"<p>Native force output from matched held-out trial {sample_id}; same initial state, task and controller. Positive Fz is vertical ground reaction force.</p>"
        f'<div class="plot"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Vertical ground reaction force over time for all materials">'
        f'<title>Vertical GRF over time, {sample_id}</title><text x="{left}" y="22" font-size="15">Vertical GRF [N]</text>{"".join(axes)}{"".join(lines)}'
        f'<text x="{width / 2}" y="{height - 31}" text-anchor="middle" font-size="11">Time [s]</text>{"".join(legend)}</svg></div>'
    )


def chart_section(title: str, html_body: str) -> dict:
    return {"title": title, "kind": "biomechanics", "html": html_body}


def qualification(run: Path, conditions: dict) -> dict:
    """Collect replay, numerical half-step, and paper-support checks."""
    result = {
        "condition_checks": {},
        "paper_compression_support_strain_fraction": 0.6,
        "matched_input_checks": {},
        "material_geometry_checks": {},
    }
    reference_artifact = read_json(run / "materials" / "original.json")
    ignored_artifact_fields = {"shoe", "constitutive_model", "validation", "provenance"}
    result["material_geometry_checks"]["unchanged_artifact_fields"] = {
        name: [
            key
            for key, value in reference_artifact.items()
            if key not in ignored_artifact_fields and read_json(run / "materials" / f"{name}.json").get(key) != value
        ]
        for name in CONDITIONS[1:]
    }
    result["material_geometry_checks"]["column_bed_equal_original"] = {
        name: read_json(run / "materials" / f"{name}.json").get("column_bed") == reference_artifact.get("column_bed")
        for name in CONDITIONS
    }
    result["material_geometry_checks"]["shoe_geometry_equal_original"] = {}
    for name in CONDITIONS:
        shoe_ref = {k: v for k, v in reference_artifact["shoe"].items() if k not in {"id", "model_scope"}}
        shoe_test = {
            k: v
            for k, v in read_json(run / "materials" / f"{name}.json")["shoe"].items()
            if k not in {"id", "model_scope"}
        }
        result["material_geometry_checks"]["shoe_geometry_equal_original"][name] = shoe_ref == shoe_test
    hashes = {name: conditions[name]["summary"].get("input_model_sha256") for name in CONDITIONS}
    result["matched_input_checks"]["controller_hashes"] = hashes
    result["matched_input_checks"]["controller_hashes_identical"] = len(set(hashes.values())) == 1
    result["matched_input_checks"]["scenario_matches"] = {}
    for row_id in sorted(set.intersection(*(set(conditions[name]["rows"]) for name in CONDITIONS))):
        cases = {}
        for name in CONDITIONS:
            row = conditions[name]["rows"][row_id]
            case = read_json(run / name / row["scenario"])
            cases[name] = case
        ref = cases["original"]
        checks = {}
        for field in ("schema", "chain", "initial", "task", "duration_s", "config"):
            checks[field] = all(cases[name].get(field) == ref.get(field) for name in CONDITIONS)
        ref_shoe = {k: v for k, v in ref["shoe"].items() if k not in {"artifact", "sha256"}}
        checks["shoe_mount_pitch_friction"] = all(
            {k: v for k, v in cases[name]["shoe"].items() if k not in {"artifact", "sha256"}} == ref_shoe
            for name in CONDITIONS
        )
        result["matched_input_checks"]["scenario_matches"][row_id] = checks
    for name in CONDITIONS:
        directory = run / name
        replay = directory / "replay_evidence"
        validation_path = replay / "validation.json"
        deformation_path = replay / "deformation.npz"
        viewer_path = replay / "viewer.html"
        half_path = directory / "half-step.json"
        fits = conditions[name]["summary"].get("condition", {}).get("fit")
        fit_path = run / "materials" / f"{name}.json"
        if fits is None and fit_path.exists():
            artifact = read_json(fit_path)
            fits = artifact.get("provenance", {}).get("material_fit")
        result["condition_checks"][name] = {
            "controller_sha256": conditions[name]["summary"].get("input_model_sha256"),
            "matching_viewer_present": viewer_path.is_file(),
            "deformation_capture_present": deformation_path.is_file(),
            "deformation_validation": read_json(validation_path)
            if validation_path.is_file()
            else {"status": "missing", "reason": "Expected source-checked replay_evidence/validation.json."},
            "half_step": read_json(half_path) if half_path.is_file() else {"status": "missing"},
            "compression_fit": fits,
        }
        if name in PAPER_FOAMS and fits:
            reported = PAPER_FOAMS[name]["reported_compressive_modulus_pa"]
            tangent = fits.get("small_strain_compressive_tangent_pa")
            result["condition_checks"][name]["reported_modulus_comparison"] = {
                "reported_compressive_modulus_pa": reported,
                "fitted_small_strain_tangent_pa": tangent,
                "difference_pa": tangent - reported if tangent is not None else None,
                "difference_percent": 100.0 * (tangent - reported) / reported if tangent is not None else None,
            }
    return result


def report(run: Path, outputs: Path, renderer: Path = RENDERER) -> Path:
    """Create metrics, qualification, spec, rendered report, and Materials testing landing page."""
    top = read_json(run / "summary.json")
    if top.get("status") != "completed":
        raise ValueError(f"Run is not completed: {top.get('status')}")
    for condition in CONDITIONS:
        if (run / condition / "replay_evidence/viewer.html").is_file():
            update_viewer(run / condition)
    conditions = {name: condition_summary(run, name) for name in CONDITIONS}
    splits = {split: aggregate(conditions, split) for split in ("eval", "train")}
    qual = qualification(run, conditions)
    metrics = {
        "run_id": run.name,
        "primary_split": "eval",
        "controller_frozen": True,
        "split_metrics": splits,
        "condition_labels": LABELS,
        "out_of_paper_support_counts": {
            name: sum(
                int(row["metrics"].get("intervals_outside_paper_compression_support") or 0)
                for row in conditions[name]["rows"].values()
            )
            for name in CONDITIONS
        },
    }
    write_json(run / "metrics.json", metrics)
    write_json(run / "qualification.json", qual)
    common = splits["eval"]["common_completed_ids"]
    charts = []
    for metric, title, unit, signed in (
        ("joint_positive_work_j", "Positive actuator work by joint", "J", False),
        ("joint_negative_work_j", "Absorbed actuator work by joint", "J", True),
        ("joint_net_work_j", "Net actuator work by joint", "J", True),
        ("joint_peak_positive_power_w", "Peak positive actuator power by joint", "W", False),
        ("joint_peak_absorbed_power_w", "Peak absorbed actuator power by joint", "W", False),
    ):
        means = {name: common_mean(conditions, name, common, metric) for name in CONDITIONS}
        charts.append(
            chart_section(
                title,
                f"<p>Held-out common completed trials (n={len(common)}). Values are condition means; paired changes are listed below.</p>{svg_grouped(means, metric, title, unit, signed=signed)}",
            )
        )
    fz_means = {name: common_mean(conditions, name, common, "peak_grf_n")[1] for name in CONDITIONS}
    contact_means = {name: common_mean(conditions, name, common, "contact_duration_s") * 1000.0 for name in CONDITIONS}
    charts.append(
        chart_section(
            "Peak vertical ground reaction force",
            f"<p>Held-out common completed trials (n={len(common)}).</p>{svg_bars(fz_means, 'Peak vertical ground reaction force', 'N')}",
        )
    )
    charts.append(
        chart_section(
            "Contact duration",
            f"<p>Held-out common completed trials (n={len(common)}).</p>{svg_bars(contact_means, 'Contact duration', 'ms')}",
        )
    )
    charts.append(
        chart_section(
            "First held-out vertical ground reaction force trace",
            native_fz_svg(run, conditions),
        )
    )
    charts.append(
        {"title": "Held-out contact metrics", "kind": "details", "html": contact_metrics_table(conditions, common)}
    )
    common_positive = {
        name: mean([sum(conditions[name]["rows"][i]["metrics"].get("joint_positive_work_j", [])) for i in common])
        for name in CONDITIONS
    }
    leap_delta = (
        common_positive["ff_leap"] - common_positive["original"]
        if common and all(v is not None for v in common_positive.values())
        else None
    )
    leap_pct = (
        100.0 * leap_delta / common_positive["original"]
        if leap_delta is not None and common_positive["original"]
        else None
    )
    finding = (
        f"Across {len(common)} common completed held-out trials, LEAP changed mean total positive joint work by {fmt(leap_delta, 3)} J ({fmt(leap_pct, 2)}%) versus the original condition. This is a small mechanical-work difference in the frozen simulation and does not establish energy economy or a material winner."
        if leap_delta is not None
        else "There are no common completed held-out trials for a paired LEAP interpretation."
    )
    sections = [
        chart_section(
            "Paired changes from both references",
            f"<p>{html.escape(finding)}</p>"
            + total_work_paired_table(splits, conditions)
            + paired_table(splits, conditions),
        ),
        *charts,
        {
            "title": "Coverage, failures, and numerical sensitivity",
            "kind": "details",
            "html": qualification_html(splits, qual, conditions),
        },
        {"title": "Compression fit and interpretation limits", "kind": "details", "html": fit_html(qual)},
    ]
    first_viewer = run / "original" / "replay_evidence" / "viewer.html"
    if not first_viewer.is_file():
        first_viewer = run / "original_elastic" / "replay_evidence" / "viewer.html"
    comparison_viewer = (
        build_comparison_viewer(run, first_viewer.parent.parent.name) if first_viewer.is_file() else None
    )
    started = read_json(run / "provenance.json").get("run_started_at")
    spec = {
        "title": "Paper foam materials comparison",
        "domain": "impedance-instron",
        "run_id": run.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_started_at": started,
        "status": "Completed frozen-controller material comparison",
        "summary": "Matched held-out mechanical work and power comparison across the original shoe, an elasticity-only control, and two paper-derived foam surrogates. The controller stayed frozen; positive and absorbed joint work are reported from saved native actuator loads and velocities.",
        "motion_viewer": {
            "source": str(comparison_viewer),
            "title": "Shoe motion, overlays, and orbitable springs",
            "caption": "Select a shoe and an optional comparison overlay. Drag the spring close-up to rotate; all views use that condition's saved geometry and motion.",
        }
        if first_viewer.is_file()
        else None,
        "motion_missing_reason": "The matching original replay_evidence/viewer.html is not available; this comparison report is incomplete until each condition's native motion and deformation capture is completed.",
        "provenance": {
            "Run directory": str(run.resolve()),
            "Controller": "Frozen accepted Hogan controller; no optimization or parameter selection performed.",
            "Primary split": "Held-out eval trials; train split shown as secondary diagnostic.",
            "Paper": PAPER_URL,
            "Raw evidence": "summary.json, condition traces, half-step.json, and source-checked replay_evidence.",
        },
        "metrics": [
            {"label": "Common completed held-out trials", "value": str(len(common)), "unit": "paired trials"},
            {
                "label": "Held-out original positive joint work",
                "value": fmt(
                    splits["eval"]["conditions"]["original"]["all_completed_means"]["joint_positive_work_j"], 3
                ),
                "unit": "J (hip, knee, ankle)",
            },
            {
                "label": "Held-out original absorbed joint work",
                "value": fmt(
                    splits["eval"]["conditions"]["original"]["all_completed_means"]["joint_negative_work_j"], 3
                ),
                "unit": "J (hip, knee, ankle)",
            },
        ],
        "sections": sections,
        "attachments": [
            {
                "source": str(run / name / "replay_evidence" / "viewer.html"),
                "title": f"{LABELS[name]} motion and deformation viewer",
            }
            for name in CONDITIONS
            if (run / name / "replay_evidence" / "viewer.html").is_file()
        ],
    }
    spec["sections"] = [s for s in spec["sections"] if s]
    spec_path = run / "report-spec.json"
    write_json(spec_path, spec)
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "python",
            str(renderer),
            str(spec_path),
            "--outputs",
            str(outputs),
            "--overwrite",
        ],
        check=True,
    )
    slug = re.sub(r"[^a-z0-9]+", "-", run.name.lower()).strip("-")[:96].rstrip("-")
    report_path = outputs / "reports" / "impedance-instron" / slug / "report.html"
    # The initial motion view must load before a phone user scrolls.
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace('loading="lazy"', 'loading="eager"'), encoding="utf-8"
    )
    landing = outputs / "materials-testing" / "index.html"
    landing.parent.mkdir(parents=True, exist_ok=True)
    rel_report = "../reports/impedance-instron/" + slug + "/report.html"
    landing.write_text(landing_html(run.name, rel_report, splits), encoding="utf-8")
    add_catalog_link(outputs / "index.html", run.name, "materials-testing/index.html")
    return report_path


def fmt(value, places=3):
    """Format scalar or joint values for display."""
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return ", ".join("n/a" if v is None else f"{v:.{places}f}" for v in value)
    return f"{value:.{places}f}"


def paired_table(splits, conditions):
    """Render paired differences against the original and elasticity-only controls."""
    out = [
        "<p>Paired means use trial IDs completed by each condition/reference pair. Δ is condition minus the reference; absorbed work is signed negative. Held-out eval is primary and train is secondary. Joint values are hip / knee / ankle.</p>",
        '<div style="overflow-x:auto"><table><thead><tr><th>Split</th><th>Condition</th><th>vs reference</th><th>Paired n</th><th>Δ positive work [J]</th><th>Δ absorbed work [J]</th><th>Δ net work [J]</th><th>Δ peak positive power [W]</th><th>Δ native score</th></tr></thead><tbody>',
    ]
    for split_name in ("eval", "train"):
        split = splits[split_name]
        for ref in ("original", "original_elastic"):
            for name in CONDITIONS:
                if name == ref:
                    continue
                ids = sorted(
                    set(split["conditions"][ref]["completed_ids"]) & set(split["conditions"][name]["completed_ids"])
                )
                cols = []
                for metric in (
                    "joint_positive_work_j",
                    "joint_negative_work_j",
                    "joint_net_work_j",
                    "joint_peak_positive_power_w",
                    "loss",
                ):
                    vals_a, vals_b = [], []
                    for trial_id in ids:
                        va = conditions[ref]["rows"][trial_id]["metrics"].get(metric)
                        vb = conditions[name]["rows"][trial_id]["metrics"].get(metric)
                        if isinstance(va, list) and isinstance(vb, list):
                            vals_a.append(va)
                            vals_b.append(vb)
                        elif va is not None and vb is not None:
                            vals_a.append(va)
                            vals_b.append(vb)
                    if vals_a and isinstance(vals_a[0], list):
                        cols.append(
                            " / ".join(
                                fmt(mean([y[j] - x[j] for x, y in zip(vals_a, vals_b, strict=True)]), 3)
                                for j in range(3)
                            )
                        )
                    else:
                        cols.append(fmt(mean([y - x for x, y in zip(vals_a, vals_b, strict=True)]), 3))
                row = (split_name, LABELS[name], LABELS[ref], str(len(ids)), *cols)
                out.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def total_work_paired_table(splits, conditions):
    """Show total positive-work mean and paired spread for both references."""
    out = [
        "<h3>Total positive actuator work</h3><p>Each row is a paired comparison over completed matched trial windows. The standard deviation and observed min-max describe this small sample; they are not inferential confidence intervals.</p>",
        '<div style="overflow-x:auto"><table><thead><tr><th>Split</th><th>Condition</th><th>Reference</th><th>n</th><th>Condition mean [J]</th><th>Reference mean [J]</th><th>Mean paired Δ [J]</th><th>Δ / reference [%]</th><th>Paired SD [J]</th><th>Paired range [J]</th></tr></thead><tbody>',
    ]
    for split_name in ("eval", "train"):
        split = splits[split_name]
        for reference in ("original", "original_elastic"):
            for condition in CONDITIONS:
                if condition == reference:
                    continue
                ids = sorted(
                    set(split["conditions"][reference]["completed_ids"])
                    & set(split["conditions"][condition]["completed_ids"])
                )
                reference_values = [
                    conditions[reference]["rows"][trial_id]["metrics"].get("total_positive_work_j") for trial_id in ids
                ]
                condition_values = [
                    conditions[condition]["rows"][trial_id]["metrics"].get("total_positive_work_j") for trial_id in ids
                ]
                differences = [
                    b - a
                    for a, b in zip(reference_values, condition_values, strict=True)
                    if a is not None and b is not None
                ]
                ref_mean = mean([value for value in reference_values if value is not None])
                condition_mean = mean([value for value in condition_values if value is not None])
                delta = mean(differences)
                sample_sd = (
                    math.sqrt(sum((value - delta) ** 2 for value in differences) / (len(differences) - 1))
                    if len(differences) > 1
                    else None
                )
                row = (
                    split_name,
                    LABELS[condition],
                    LABELS[reference],
                    str(len(differences)),
                    fmt(condition_mean, 3),
                    fmt(ref_mean, 3),
                    fmt(delta, 3),
                    fmt(100.0 * delta / ref_mean if delta is not None and ref_mean else None, 2),
                    fmt(sample_sd, 3),
                    f"{fmt(min(differences), 3)} to {fmt(max(differences), 3)}" if differences else "n/a",
                )
                out.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>")
    out.append("</tbody></table></div><h3>Per-joint paired differences</h3>")
    return "".join(out)


def qualification_html(splits, qual, conditions):
    """Show failed cases and each condition's replay and timestep evidence."""
    out = [
        "<h3>Trial coverage</h3><p>Failures are retained below; paired work charts include only IDs completed by every material condition.</p>",
        '<div style="overflow-x:auto"><table><thead><tr><th>Split</th><th>Condition</th><th>Completed</th><th>Failed IDs</th><th>Outside 0.6 strain support</th></tr></thead><tbody>',
    ]
    for split, item in splits.items():
        for name in CONDITIONS:
            c = item["conditions"][name]
            out.append(
                f"<tr><td>{split}</td><td>{html.escape(LABELS[name])}</td><td>{c['n_completed']}/{c['n_trials']}</td><td>{html.escape(', '.join(item['failed_trial_ids'][name]) or 'None')}</td><td>{html.escape(str(sum(int(row['metrics'].get('intervals_outside_paper_compression_support') or 0) for row in conditions[name]['rows'].values() if row.get('split') == split)))}</td></tr>"
            )
    out.append("</tbody></table></div><h3>Replay and half-step checks</h3><ul>")
    for name, check in qual["condition_checks"].items():
        half = check["half_step"].get("metrics", {})
        validation = check["deformation_validation"]
        nominal = conditions[name]["rows"].get("FR3_1_eval_000", {}).get("metrics", {})
        fine_work = sum(half.get("joint_positive_work_j", [])) if half.get("joint_positive_work_j") else None
        nominal_work = sum(nominal.get("joint_positive_work_j", [])) if nominal.get("joint_positive_work_j") else None
        sensitivity = (
            100.0 * (fine_work - nominal_work) / nominal_work if fine_work is not None and nominal_work else None
        )
        validation_status = "passed" if validation.get("passed") else validation.get("status", "missing/failed")
        out.append(
            f"<li>{html.escape(LABELS[name])}: viewer={check['matching_viewer_present']}, deformation={check['deformation_capture_present']}, replay validation={html.escape(str(validation_status))}, max force error={html.escape(fmt(validation.get('max_force_error_n'), 6))} N, half-step dt={html.escape(str(check['half_step'].get('dt_s', 'missing')))} s, score={html.escape(fmt(half.get('loss'), 4))}, positive-work sensitivity={html.escape(fmt(sensitivity, 2))}%.</li>"
        )
    matched = qual["matched_input_checks"]
    geometry = qual["material_geometry_checks"]
    failed_geometry = {name: values for name, values in geometry["unchanged_artifact_fields"].items() if values}
    scenario_checks = list(matched["scenario_matches"].values())
    scenario_ok = all(all(row.values()) for row in scenario_checks)
    out.append(
        f"</ul><h3>Matched-condition qualification</h3><p>Controller hashes identical: {matched['controller_hashes_identical']}. Column-bed equality: {all(geometry['column_bed_equal_original'].values())}. Shoe geometry equality (excluding material-study labels): {all(geometry['shoe_geometry_equal_original'].values())}. All unchanged artifact fields equal: {not failed_geometry}. Matched scenario fields (chain, initial state/velocity/phase, task, duration, config, and shoe placement/friction) identical across {len(scenario_checks)} common trials: {scenario_ok}.</p>"
    )
    return "".join(out)


def fit_html(qual):
    """Describe compression-fit accuracy and explicit scientific claim boundaries."""
    out = [
        f'<p>The foam surrogates use the paper\'s published compression means and independent shear modulus. Fit diagnostics are copied from each material artifact; strains above 0.6 are outside the paper table support and are counted in the coverage section. Source: <a href="{PAPER_URL}">paper tables and methods</a>.</p>',
        '<div style="overflow-x:auto"><table><thead><tr><th>Material</th><th>Compression RMSE [Pa]</th><th>Max abs error [Pa]</th><th>R²</th><th>Fitted small-strain tangent [Pa]</th><th>Paper reported modulus [Pa]</th><th>Tangent mismatch</th><th>Energy return assumption</th></tr></thead><tbody>',
    ]
    for name in ("ff_leap", "ff_turbo_plus"):
        fit = qual["condition_checks"][name].get("compression_fit") or {}
        mismatch = qual["condition_checks"][name].get("reported_modulus_comparison", {})
        out.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(x))}</td>"
                for x in (
                    LABELS[name],
                    fmt(fit.get("rmse_pa")),
                    fmt(fit.get("maximum_absolute_error_pa")),
                    fmt(fit.get("r_squared"), 4),
                    fmt(fit.get("small_strain_compressive_tangent_pa")),
                    fmt(mismatch.get("reported_compressive_modulus_pa")),
                    f"{fmt(mismatch.get('difference_pa'))} Pa ({fmt(mismatch.get('difference_percent'), 1)}%)",
                    "Hyperelastic; measured loss return not reproduced",
                )
            )
            + "</tr>"
        )
        out.append(f'<tr><td colspan="8">{pressure_fit_svg(name, fit)}</td></tr>')
    out.append(
        "</tbody></table></div><p>These are reduced homogeneous foundation substitutions in the existing shoe geometry. They do not reproduce hysteresis energy return, the paper's full three-dimensional constitutive law, the complete ASICS shoe architecture, or metabolic benefit. Joint work and power are mechanical outputs of this frozen-controller simulation, not clinical or physiological validation.</p>"
    )
    return "".join(out)


def pressure_fit_svg(name: str, fit: dict) -> str:
    """Plot paper compression means and SD against the reduced fitted curve."""
    paper = PAPER_FOAMS[name]
    strain = fit.get("strain", [])
    target = fit.get("target_pressure_pa", [])
    predicted = fit.get("predicted_pressure_pa", [])
    sd = [v * 1000.0 for v in paper["compression_sd_kpa"]]
    if not strain or not target or not predicted:
        return "Compression fit curve unavailable."
    left, top, width, height = 70, 35, 600, 320
    max_y = max((p + e for p, e in zip(target, sd, strict=True)), default=1.0) * 1.08

    def x(value):
        return left + value / 0.6 * (width - left - 20)

    def y(value):
        return top + (height - top - 45) * (1.0 - value / max_y)

    paper_color, fit_color = "#365f91", "#cf6658"
    path = " ".join(f"{x(s):.1f},{y(p):.1f}" for s, p in zip(strain, predicted, strict=True))
    circles = []
    for s, p, e in zip(strain, target, sd, strict=True):
        px, py = x(s), y(p)
        circles.append(
            f'<line x1="{px:.1f}" y1="{y(p + e):.1f}" x2="{px:.1f}" y2="{y(max(p - e, 0)):.1f}" stroke="{paper_color}"/><circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{paper_color}"/>'
        )
    ticks = []
    for tick in range(4):
        strain_tick = tick * 0.2
        px = x(strain_tick)
        ticks.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{y(0):.1f}" stroke="#e2e8ec"/><text x="{px:.1f}" y="{y(0) + 16:.1f}" text-anchor="middle" font-size="10">{strain_tick:.1f}</text>'
        )
        pressure_tick = max_y * tick / 3
        py = y(pressure_tick)
        ticks.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - 15}" y2="{py:.1f}" stroke="#e2e8ec"/><text x="{left - 7}" y="{py + 3:.1f}" text-anchor="end" font-size="10">{pressure_tick / 1000:.0f}k</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(LABELS[name])} published and fitted compression response">'
        f"<title>{html.escape(LABELS[name])}: paper compression means and SD with fitted curve</title>"
        f'<text x="{left}" y="20" font-size="14">Compression pressure [Pa]</text>'
        f'<line x1="{left}" y1="{y(0):.1f}" x2="{width - 15}" y2="{y(0):.1f}" stroke="#536878"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{y(0):.1f}" stroke="#536878"/>'
        f"{''.join(ticks)}"
        f'<text x="{width / 2}" y="{height - 6}" text-anchor="middle" font-size="12">Compression strain [1]</text>'
        f'<polyline points="{path}" fill="none" stroke="{fit_color}" stroke-width="3"/>{"".join(circles)}'
        f'<line x1="{left + 10}" y1="{height - 25}" x2="{left + 35}" y2="{height - 25}" stroke="{fit_color}" stroke-width="3"/><text x="{left + 40}" y="{height - 21}" font-size="11">Two-term Hyperfoam fit</text>'
        f'<circle cx="{left + 235}" cy="{height - 25}" r="4" fill="{paper_color}"/><text x="{left + 245}" y="{height - 21}" font-size="11">Paper means ± SD</text></svg>'
    )


def landing_html(run_id, report_link, splits):
    """Render a small Materials testing category page for the generated comparison."""
    n = len(splits["eval"]["common_completed_ids"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Materials testing</title><style>body{{font:16px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#173040}}article{{border:1px solid #ccd7df;border-radius:12px;padding:1rem;margin:1rem 0}}a{{color:#075e84}}</style></head><body><p><a href="../index.html">All experiment reports</a></p><h1>Materials testing</h1><p>Frozen-controller material substitution studies with mechanical response metrics.</p><article><h2><a href="{html.escape(report_link)}">Paper foam materials comparison</a></h2><p>{html.escape(run_id)} · four material conditions · {n} common completed held-out trials. Compare joint positive, absorbed and net work plus peak power.</p></article></body></html>'''


def add_catalog_link(index: Path, run_id: str, link: str) -> None:
    """Refresh the shared dashboard and both experiment category pages."""
    rebuild(index.parent)


def main(argv=None):
    """Generate report presentation from an already completed materials run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed materials run directory")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"), help="Repository outputs directory")
    parser.add_argument("--renderer", type=Path, default=RENDERER, help="Installed Instron report renderer")
    args = parser.parse_args(argv)
    print(report(args.run.resolve(), args.outputs.resolve(), args.renderer.resolve()))


if __name__ == "__main__":
    main()
