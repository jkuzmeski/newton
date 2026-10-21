# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Report the frozen-controller material by spring-geometry experiment."""

from __future__ import annotations

import argparse
import html
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from projects.impedance_instron.cartesian.shoe import Shoe
from projects.impedance_instron.report_catalog import rebuild

from .comparison_viewer_report import build_comparison_viewer
from .identify import ForceObservation, load_trials
from .materials import sha256, write_json
from .materials_marathon_report import marathon_section
from .materials_report import RENDERER, read_json
from .mechanics import Chain
from .spring_viewer_report import update_viewer

MATERIALS = ("original", "ff_leap", "ff_turbo_plus")
GEOMETRIES = ("reference", "long125", "block40")
MATERIAL_LABELS = {
    "original": "Current material",
    "ff_leap": "FF LEAP elastic",
    "ff_turbo_plus": "FF TURBO PLUS elastic",
}
GEOMETRY_LABELS = {"reference": "Original geometry", "long125": "Spring length +25%", "block40": "Flat 40 mm block"}
COLORS = ("#365f91", "#d78315", "#ad4660")
SAMPLE = "FR3_1_eval_000"


def name(material, geometry):
    """Return a material and geometry condition identifier."""
    return f"{material}__{geometry}"


def table(headers, rows):
    """Build a scrollable escaped evidence table."""
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def fmt(value):
    """Format a physical metric or explicitly mark its absence."""
    return "Unavailable" if value is None else f"{value:.3f}"


def average(rows, key, index=None):
    """Average a physical channel across a declared completed sample."""
    values = [r[key] if index is None else r[key][index] for r in rows]
    return float(np.mean(values)) if values else None


def grouped(title, unit, groups, series, values):
    """Draw grouped comparisons with one shared physical scale."""
    finite = [v for row in values for v in row if v is not None]
    if not finite:
        return "<p>No common completed windows are available.</p>"
    low, high = min(0, *finite), max(0, *finite)
    span = max(high - low, 1e-6)
    high += span * 0.08
    low -= span * 0.04 if low < 0 else 0
    width, height, left, top, bottom = 720, 390, 74, 42, 290

    def y(value):
        return bottom - (value - low) / (high - low) * (bottom - top)

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}"><title>{html.escape(title)}</title><text x="{left}" y="24" font-size="17">{html.escape(title)} [{html.escape(unit)}]</text>'
    ]
    for value in np.linspace(low, high, 5):
        yy = y(value)
        out.append(
            f'<line x1="{left}" x2="700" y1="{yy}" y2="{yy}" stroke="#dce3e8"/><text x="65" y="{yy + 5}" text-anchor="end" font-size="14">{value:.3g}</text>'
        )
    group_width = (700 - left) / len(groups)
    bar_width = min(42, group_width / (len(series) + 2))
    for i, group in enumerate(groups):
        center = left + group_width * (i + 0.5)
        for j, _ in enumerate(series):
            value = values[j][i]
            if value is None:
                continue
            xx = center + (j - (len(series) - 1) / 2) * (bar_width + 3) - bar_width / 2
            out.append(
                f'<rect x="{xx}" y="{min(y(value), y(0))}" width="{bar_width}" height="{max(abs(y(value) - y(0)), 0.5)}" fill="{COLORS[j]}"><title>{html.escape(series[j])}: {value:.4f} {html.escape(unit)}</title></rect>'
            )
        out.append(f'<text x="{center}" y="315" text-anchor="middle" font-size="15">{html.escape(group)}</text>')
    for j, label in enumerate(series):
        xx = left + j * 210
        out.append(
            f'<rect x="{xx}" y="350" width="13" height="13" fill="{COLORS[j]}"/><text x="{xx + 18}" y="362" font-size="13">{html.escape(label)}</text>'
        )
    return "".join(out) + "</svg>"


def lines(title, unit, channels):
    """Plot saved time series without resampling their metric support."""
    all_y = np.concatenate([np.asarray(v) for _, _, v, _ in channels])
    low, high = float(all_y.min()), float(all_y.max())
    span = max(high - low, 1e-8)
    low -= span * 0.06
    high += span * 0.06
    tmax = max(float(np.max(t)) for _, t, _, _ in channels)

    def x(t):
        return 78 + 610 * t / tmax

    def y(v):
        return 290 - 235 * (v - low) / (high - low)

    out = [
        f'<svg viewBox="0 0 720 420" role="img" aria-label="{html.escape(title)}"><title>{html.escape(title)}</title><text x="78" y="24" font-size="16">{html.escape(title)} [{html.escape(unit)}]</text>'
    ]
    for tick in np.linspace(low, high, 5):
        yy = y(tick)
        out.append(
            f'<line x1="78" x2="688" y1="{yy}" y2="{yy}" stroke="#dce3e8"/><text x="68" y="{yy + 5}" text-anchor="end" font-size="13">{tick:.3g}</text>'
        )
    for tick in np.linspace(0, tmax, 5):
        out.append(f'<text x="{x(tick)}" y="312" text-anchor="middle" font-size="13">{tick:.2f}</text>')
    for j, (label, t, v, color) in enumerate(channels):
        indices = np.unique(np.linspace(0, len(t) - 1, min(len(t), 600), dtype=int))
        points = " ".join(f"{x(float(t[i])):.2f},{y(float(v[i])):.2f}" for i in indices)
        dash = ' stroke-dasharray="6 4"' if color == "#222222" else ""
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/><line x1="80" x2="100" y1="{350 + 17 * j}" y2="{350 + 17 * j}" stroke="{color}" stroke-width="3"/><text x="108" y="{355 + 17 * j}" font-size="13">{html.escape(label)}</text>'
        )
    return "".join(out) + '<text x="390" y="332" text-anchor="middle" font-size="15">Native time [s]</text></svg>'


def report(run, outputs, renderer):
    """Build a complete comparison from saved native evaluations and captures."""
    top = read_json(run / "summary.json")
    if top["status"] != "completed":
        raise ValueError("A completed evaluation is required; failed trials remain part of that evaluation.")
    condition_list = top["conditions"]
    condition_ids = [name(m, g) for m in MATERIALS for g in GEOMETRIES]
    if {c["id"] for c in condition_list} != set(condition_ids):
        raise ValueError("Expected all nine declared material/geometry conditions")
    summaries = {c: read_json(run / c / "summary.json") for c in condition_ids}
    rows = {c: {r["id"]: r["metrics"] for r in s["trials"]} for c, s in summaries.items()}
    evidence = {c: {r["id"]: r for r in s["trials"]} for c, s in summaries.items()}
    metrics = {
        "schema": "material-geometry-metrics-v1",
        "primary_split": "eval",
        "splits": {},
        "units": {
            "work": "J per saved window",
            "power": "W",
            "force": "N",
            "impulse": "N s",
            "time": "s",
            "compression": "fraction",
        },
    }
    summary_rows = []
    paired_rows = []
    for split in ("eval", "train"):
        completed = {
            c: {i for i, r in rows[c].items() if r["split"] == split and r["status"] == "completed"}
            for c in condition_ids
        }
        common = sorted(set.intersection(*completed.values()))
        split_result = {"common_completed_ids": common, "conditions": {}, "paired_geometry_deltas": {}}
        for c in condition_ids:
            all_rows = [r for r in rows[c].values() if r["split"] == split]
            matched = [rows[c][i] for i in common]
            means = {
                key: average(matched, key)
                for key in (
                    "total_positive_work_j",
                    "total_absorbed_work_j",
                    "contact_duration_s",
                    "peak_compression_fraction",
                    "loss",
                )
            }
            for key in (
                "joint_positive_work_j",
                "joint_negative_work_j",
                "joint_net_work_j",
                "joint_peak_positive_power_w",
                "joint_peak_absorbed_power_w",
                "peak_grf_n",
                "grf_impulse_ns",
            ):
                count = 2 if key in ("peak_grf_n", "grf_impulse_ns") else 3
                means[key] = [average(matched, key, j) for j in range(count)]
            split_result["conditions"][c] = {
                "n": len(all_rows),
                "completed": len(completed[c]),
                "failed_ids": [r["id"] for r in all_rows if r["status"] != "completed"],
                "common_means": means,
                "outside_support_intervals": sum(
                    r.get("intervals_outside_paper_compression_support") or 0 for r in all_rows
                ),
            }
            summary_rows.append(
                [
                    split,
                    c,
                    len(completed[c]),
                    len(all_rows) - len(completed[c]),
                    len(common),
                    fmt(means["total_positive_work_j"]),
                    fmt(means["total_absorbed_work_j"]),
                    fmt(means["peak_grf_n"][1]),
                    fmt(means["contact_duration_s"]),
                    fmt(means["peak_compression_fraction"]),
                ]
            )
        for m in MATERIALS:
            baseline = name(m, "reference")
            for g in GEOMETRIES[1:]:
                c = name(m, g)
                paired_ids = sorted(completed[c] & completed[baseline])
                deltas = np.array(
                    [
                        rows[c][i]["total_positive_work_j"] - rows[baseline][i]["total_positive_work_j"]
                        for i in paired_ids
                    ]
                )
                base = average([rows[baseline][i] for i in paired_ids], "total_positive_work_j")
                d = {
                    "ids": paired_ids,
                    "mean_delta_j": float(deltas.mean()) if len(deltas) else None,
                    "sd_delta_j": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
                    "range_delta_j": [float(deltas.min()), float(deltas.max())] if len(deltas) else [],
                    "percent_of_reference_mean": 100 * float(deltas.mean()) / base if len(deltas) and base else None,
                }
                split_result["paired_geometry_deltas"][c] = d
                paired_rows.append(
                    [
                        split,
                        MATERIAL_LABELS[m],
                        GEOMETRY_LABELS[g],
                        len(paired_ids),
                        fmt(d["mean_delta_j"]),
                        fmt(d["percent_of_reference_mean"]),
                        fmt(d["sd_delta_j"]),
                        str(d["range_delta_j"]),
                    ]
                )
        metrics["splits"][split] = split_result
    write_json(run / "metrics.json", metrics)
    qualification = qualify(run, condition_ids, summaries, evidence)
    write_json(run / "qualification.json", qualification)
    primary = metrics["splits"]["eval"]
    sections = []

    def section(title, body, kind="biomechanics"):
        sections.append({"title": title, "html": body, "kind": kind})

    findings = []
    for material in MATERIALS:
        reference_means = primary["conditions"][name(material, "reference")]["common_means"]
        long_means = primary["conditions"][name(material, "long125")]["common_means"]
        block_means = primary["conditions"][name(material, "block40")]["common_means"]
        if reference_means["total_positive_work_j"] is not None:
            work = reference_means["total_positive_work_j"]
            findings.append(
                f"{MATERIAL_LABELS[material]}: longer springs changed positive work by {100 * (long_means['total_positive_work_j'] / work - 1):+.2f}%; the block changed it by {100 * (block_means['total_positive_work_j'] / work - 1):+.2f}%, with absorbed work changing by {block_means['total_absorbed_work_j'] - reference_means['total_absorbed_work_j']:+.2f} J and peak vertical force by {block_means['peak_grf_n'][1] - reference_means['peak_grf_n'][1]:+.1f} N versus its original geometry."
            )
    section(
        "Observed geometry tradeoffs",
        "".join(f"<p>{html.escape(finding)}</p>" for finding in findings)
        + "<p>Lower positive actuator work alone does not establish a better shoe: assess absorbed work, force, motion, numerical sensitivity and material support together. Some block trials exceed the paper's measured compression range.</p>",
        "details",
    )
    geometry_rows = []
    for geometry in GEOMETRIES:
        artifact = read_json(run / "materials" / f"original__{geometry}.json")
        bed = artifact["column_bed"]
        lengths = np.asarray(bed["rest_length_m"]) * 1000
        geometry_rows.append(
            [
                GEOMETRY_LABELS[geometry],
                len(lengths),
                fmt(float(lengths.min())),
                fmt(float(lengths.mean())),
                fmt(float(lengths.max())),
            ]
        )
    section(
        "Actual spring geometry",
        table(
            ["Geometry", "Spring count", "Minimum length [mm]", "Mean length [mm]", "Maximum length [mm]"],
            geometry_rows,
        ),
        "details",
    )
    section(
        "Performance and coverage",
        table(
            [
                "Split",
                "Condition",
                "Completed",
                "Failed",
                "Common N",
                "Positive work [J]",
                "Absorbed work [J]",
                "Peak Fz [N]",
                "Contact [s]",
                "Peak compression [1]",
            ],
            summary_rows,
        ),
        "details",
    )
    section(
        "Paired geometry effects within each material",
        "<p>Positive work differences use matched completed windows within the same material. SD and range describe these windows; they are not population confidence intervals. Negative deltas indicate less positive actuator work.</p>"
        + table(
            ["Split", "Material", "Geometry", "Paired N", "Delta work [J]", "Delta [%]", "SD [J]", "Range [J]"],
            paired_rows,
        ),
        "details",
    )
    selected = primary["conditions"]
    groups = ["Original", "Length +25%", "Block 40 mm"]
    for key, title, unit in [
        ("total_positive_work_j", "Total positive joint work", "J"),
        ("total_absorbed_work_j", "Total absorbed joint work", "J"),
        ("contact_duration_s", "Contact duration", "s"),
        ("peak_compression_fraction", "Peak driven compression", "fraction"),
    ]:
        values = [[selected[name(m, g)]["common_means"][key] for g in GEOMETRIES] for m in MATERIALS]
        section(
            title,
            f"<p>Held-out means across {len(primary['common_completed_ids'])} windows completed by all nine conditions.</p>"
            + grouped(title, unit, groups, list(MATERIAL_LABELS.values()), values),
        )
    values = [[selected[name(m, g)]["common_means"]["peak_grf_n"][1] for g in GEOMETRIES] for m in MATERIALS]
    section(
        "Peak vertical ground reaction force",
        grouped("Peak vertical GRF", "N", groups, list(MATERIAL_LABELS.values()), values),
    )
    for m in MATERIALS:
        for key, title, unit in [
            ("joint_positive_work_j", "Positive work by joint", "J"),
            ("joint_negative_work_j", "Signed absorbed work by joint", "J"),
            ("joint_net_work_j", "Net work by joint", "J"),
            ("joint_peak_positive_power_w", "Peak positive power by joint", "W"),
        ]:
            values = [selected[name(m, g)]["common_means"][key] for g in GEOMETRIES]
            section(
                MATERIAL_LABELS[m] + " — " + title,
                grouped(title, unit, ["Hip", "Knee", "Ankle"], list(GEOMETRY_LABELS.values()), values),
            )
    add_native_sections(run, evidence, sections)
    check_rows = [[k, str(v)] for k, v in qualification["checks"].items()]
    section(
        "Geometry, initialization, and numerical qualification",
        table(["Check", "Result"], check_rows)
        + table(["Condition", "Native work [J]", "Half-step work [J]", "Change [%]"], qualification["half_step_rows"]),
        "details",
    )
    section(
        "Geometry protocol and interpretation",
        "<p>Length +25% preserves the original free-top surface and rigid last, extending each nominal spring bottom downward. The 40 mm block has flat bottom Z=0 and flat top Z=0.040 m at the existing spring XY coordinates. Areas, neighbor topology, fullfoot driven membership, mounting, rigid last, controller, and body inertial parameters are retained. The block changes the gap between the foam free top and the rigid carrier; this is part of its changed geometry.</p><p>Initial hip height is translated by the per-trial amount needed to preserve the original minimum nominal-bottom ground clearance at the original foot pitch. Angles, velocities, phase, actuator memory, and speed task are unchanged. Measured positional references receive the same vertical datum shift for diagnostic scoring; these are adjusted comparison references, not new measurements. Raw forces retain their original clock and values.</p><p>The shoe foundation is massless: added foam volume does not add shoe mass or inertia. The paper-derived foams retain their elastic pressure approximations and independent shear tangents; measured hysteresis is not reproduced. Compression beyond 60% is outside the paper table support. Mechanical joint work over finite saved windows does not establish metabolic energy, running economy, or a material winner. Half-step checks are representative sensitivity diagnostics, not a full convergence study.</p>",
        "details",
    )
    section(
        "Material support and source",
        '<p>Material constants are unchanged from the previous materials experiment. Paper: <a href="https://arxiv.org/html/2602.12694v2">Tables 1-2 and methods</a>. Each condition retains native traces, scenarios, material fit diagnostics, geometry identity, and replay verification.</p>'
        + table(
            ["Condition", "Train out-of-support intervals", "Eval out-of-support intervals"],
            [
                [
                    c,
                    metrics["splits"]["train"]["conditions"][c]["outside_support_intervals"],
                    primary["conditions"][c]["outside_support_intervals"],
                ]
                for c in condition_ids
            ],
        ),
        "details",
    )
    sections.insert(1, marathon_section(run, metrics))
    for condition in condition_ids:
        update_viewer(run / condition)
    reference = name("original", "reference")
    comparison_viewer = build_comparison_viewer(run, reference)
    spec = {
        "title": "Materials testing: spring length and 40 mm block",
        "domain": "impedance-instron",
        "run_id": run.name,
        "run_started_at": top["started_at"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "Completed: frozen-controller geometry diagnostic",
        "summary": f"Nine material/geometry combinations evaluated from fresh initial states. Primary comparisons use {len(primary['common_completed_ids'])} matched held-out windows; failed cases remain visible. Joint work, power and contact response are computed from native traces.",
        "provenance": {
            "Category": "Materials testing",
            "Run directory": str(run),
            "Controller SHA256": top["controller_sha256"],
            "Initialization": "Matched initial nominal-bottom ground clearance; declared per-trial vertical translation.",
            "Material scope": "Current calibrated material plus unchanged FF LEAP and FF TURBO PLUS elastic approximations.",
        },
        "motion_viewer": {
            "source": str(comparison_viewer),
            "title": "Shoe motion, overlays, and orbitable springs",
            "caption": "Select among all nine material/geometry conditions and optionally overlay another. Drag the spring close-up to orbit during compression. Same held-out example and shared 0-60 mm compression scale.",
        },
        "sections": sections,
        "attachments": [
            {"source": str(run / c / "replay_evidence/viewer.html"), "title": c + " — native motion and deformation"}
            for c in condition_ids
        ]
        + [
            {
                "source": str(run / "marathon-estimates.json"),
                "title": "Conditional marathon calculations and assumptions",
            }
        ],
    }
    write_json(run / "report-spec.json", spec)
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "python",
            str(renderer),
            str(run / "report-spec.json"),
            "--outputs",
            str(outputs),
            "--overwrite",
        ],
        check=True,
    )
    report_path = outputs / "reports/impedance-instron" / run.name.replace("_", "-") / "report.html"
    report_path.write_text(report_path.read_text().replace('loading="lazy"', 'loading="eager"'))
    category_catalog(outputs)
    return report_path


def qualify(run, ids, summaries, evidence):
    """Check saved material, geometry, initialization and capture identities."""
    checks = {}
    artifacts = {c: read_json(run / "materials" / f"{c}.json") for c in ids}
    original = artifacts[name("original", "reference")]
    bed = original["column_bed"]
    for c in ids:
        a = artifacts[c]
        m, g = c.split("__")
        b = a["column_bed"]
        bottom = np.asarray(b["anchor_bottom_m"])
        old_bottom = np.asarray(bed["anchor_bottom_m"])
        rest = np.asarray(b["rest_length_m"])
        old_rest = np.asarray(bed["rest_length_m"])
        checks[c + " footprint/topology"] = bool(
            np.array_equal(bottom[:, :2], old_bottom[:, :2])
            and all(b[k] == bed[k] for k in ("area_m2", "neighbors", "spacing_m"))
        )
        checks[c + " exact rest length"] = bool(
            np.array_equal(
                rest, old_rest if g == "reference" else old_rest * 1.25 if g == "long125" else np.full_like(rest, 0.04)
            )
        )
        checks[c + " vertical geometry"] = bool(
            np.allclose(bottom[:, 2] + rest, old_bottom[:, 2] + old_rest, rtol=0, atol=1e-15)
            if g == "long125"
            else np.all(bottom[:, 2] == 0)
            if g == "block40"
            else np.array_equal(bottom, old_bottom)
        )
        fixture = a["instron_fixtures"]["fullfoot_last"]
        original_fixture = original["instron_fixtures"]["fullfoot_last"]
        checks[c + " rigid carrier and driven footprint"] = (
            fixture["carrier_anchor_m"] == original_fixture["carrier_anchor_m"]
        )
        checks[c + " rigid last mesh"] = (
            a["visual_meshes"]["fullfoot_last"] == original["visual_meshes"]["fullfoot_last"]
        )
        checks[c + " fixture free top"] = (
            bool(np.all(np.asarray(fixture["foam_free_top_m"]) == 0.04))
            if g == "block40"
            else fixture["foam_free_top_m"] == original_fixture["foam_free_top_m"]
        )
        reference_parameters = artifacts[name(m, "reference")]["constitutive_model"]["parameters"]
        parameters = a["constitutive_model"]["parameters"]
        checks[c + " material constants"] = all(
            parameters[k] == v for k, v in reference_parameters.items() if k != "pasternak_n_per_m"
        )
        validation = read_json(run / c / "replay_evidence/validation.json")
        checks[c + " replay"] = (
            validation["passed"]
            and validation["model_sha256"] == sha256(run / "runner.json")
            and validation["shoe_sha256"] == sha256(run / "materials" / f"{c}.json")
        )
        checks[c + " controller"] = (
            sha256(run / c / "runner.json") == sha256(run / "runner.json") == summaries[c]["input_model_sha256"]
        )
    scenario_matches = True
    material_matches = True
    clearance_matches = True
    shoes = {}
    initial_clearance_records = {}
    for trial_id in evidence[ids[0]]:
        cases = {c: read_json(run / c / evidence[c][trial_id]["scenario"]) for c in ids}
        baseline = cases[name("original", "reference")]
        base_q = np.asarray(baseline["initial"]["q"])
        chain = Chain(**baseline["chain"])
        clearances = {}
        for condition_id, case in cases.items():
            q = np.asarray(case["initial"]["q"])
            if condition_id not in shoes:
                spec = case["shoe"]
                shoes[condition_id] = Shoe(
                    run / "materials" / f"{condition_id}.json",
                    spec["mount_m"],
                    spec["pitch_rad"],
                )
            ankle = chain.point(q, 3, np.zeros(2))[0]
            clearances[condition_id] = float(shoes[condition_id].outline(ankle, chain.angle(q, 3))[:, 1].min())
            initial = {k: v for k, v in case["initial"].items() if k != "q"}
            initial_base = {k: v for k, v in baseline["initial"].items() if k != "q"}
            scenario_matches &= (
                np.array_equal(q[[0, 2, 3, 4, 5]], base_q[[0, 2, 3, 4, 5]])
                and initial == initial_base
                and all(case[k] == baseline[k] for k in ("chain", "task", "duration_s", "config"))
            )
        base_clearance = clearances[name("original", "reference")]
        clearance_matches &= base_clearance > 0 and all(
            abs(value - base_clearance) < 1e-10 for value in clearances.values()
        )
        initial_clearance_records[trial_id] = clearances
        for g in GEOMETRIES:
            ref = cases[name("original", g)]
            material_matches &= all(cases[name(m, g)]["initial"] == ref["initial"] for m in MATERIALS)
    checks["All trial inputs differ only by declared initial height"] = bool(scenario_matches)
    checks["Same geometry has identical initialization across materials"] = bool(material_matches)
    checks["All 107 trials preserve positive initial ground clearance"] = bool(clearance_matches)
    old_run = run.parent / "materials_paper_20261010"
    reproduction = {}
    if old_run.is_dir():
        for m in MATERIALS:
            previous = read_json(old_run / m / "summary.json")
            reproduction[m] = {
                s: summaries[name(m, "reference")]["splits"][s]["mean_loss"] - previous["splits"][s]["mean_loss"]
                for s in ("train", "eval")
            }
        checks["Original geometry reproduces previous scores"] = all(
            abs(v) < 1e-12 for d in reproduction.values() for v in d.values()
        )
    fine_rows = []
    for c in ids:
        base = evidence[c][SAMPLE]["metrics"]["total_positive_work_j"]
        fine = read_json(run / c / "half-step.json")["metrics"]
        fine_rows.append(
            [c, fmt(base), fmt(fine["total_positive_work_j"]), fmt(100 * (fine["total_positive_work_j"] - base) / base)]
        )
    return {
        "checks": checks,
        "all_identity_checks_passed": all(checks.values()),
        "prior_score_deltas": reproduction,
        "initial_clearance_m": initial_clearance_records,
        "half_step_rows": fine_rows,
        "limitations": "Identity/replay checks do not establish physical validity. Half-step results are diagnostics without a universal threshold.",
    }


def add_native_sections(run, evidence, sections):
    """Show native GRF and representative limb channels with input-checked targets."""
    plan = read_json(run / "experiment-plan.json")
    dataset = Path(plan["command"]["args"]["dataset"])
    trials = load_trials(
        dataset,
        mount_m=[-0.03186147427106201, 0, 0.10943209684347802],
        speed_m_s=3.65,
        force_observation=ForceObservation(20, 1, 1),
    )
    trial = next(t for t in trials if t.id == SAMPLE)
    trace_map = {c: dict(np.load(run / c / evidence[c][SAMPLE]["trace"])) for c in evidence}
    for c in evidence:
        if sha256(trial.provenance["reference"]) != evidence[c][SAMPLE]["reference_sha256"]:
            raise ValueError("Reference source changed after execution")
    for m in MATERIALS:
        for axis, label in ((0, "Fore-aft GRF"), (1, "Vertical GRF")):
            channels = [
                (
                    GEOMETRY_LABELS[g],
                    trace_map[name(m, g)]["time_s"][:-1],
                    trace_map[name(m, g)]["grf_n"][:, axis],
                    COLORS[j],
                )
                for j, g in enumerate(GEOMETRIES)
            ]
            channels.append(("Measured reference", trial.force_time_s, trial.grf_n[:, axis], "#222222"))
            sections.append(
                {
                    "title": MATERIAL_LABELS[m] + " — " + label,
                    "kind": "biomechanics",
                    "html": "<p>Same first held-out window; native simulated contact forces and measured force target, original clocks. The target has the saved observation processing.</p>"
                    + lines(label, "N", channels),
                }
            )
    m = "original"
    channel_specs = [
        ("Hip X", "m", lambda t, ch: t["state"][:, 0], lambda q, ch: q[:, 0]),
        ("Hip height", "m", lambda t, ch: t["state"][:, 1] - t["state"][0, 1], lambda q, ch: q[:, 1] - q[0, 1]),
        (
            "Ankle X",
            "m",
            lambda t, ch: np.array([ch.point(q, 3, np.zeros(2))[0][0] for q in t["state"]]),
            lambda qs, ch: np.array([ch.point(q, 3, np.zeros(2))[0][0] for q in qs]),
        ),
        (
            "Ankle height change",
            "m",
            lambda t, ch: (
                np.array([ch.point(q, 3, np.zeros(2))[0][1] for q in t["state"]])
                - ch.point(t["state"][0], 3, np.zeros(2))[0][1]
            ),
            lambda qs, ch: (
                np.array([ch.point(q, 3, np.zeros(2))[0][1] for q in qs]) - ch.point(qs[0], 3, np.zeros(2))[0][1]
            ),
        ),
        ("Knee relative angle", "rad", lambda t, ch: t["state"][:, 4], lambda q, ch: q[:, 4]),
        ("Ankle relative angle", "rad", lambda t, ch: t["state"][:, 5], lambda q, ch: q[:, 5]),
        (
            "Shoe pitch",
            "rad",
            lambda t, ch: np.array([ch.angle(q, 3) for q in t["state"]]) - trial.shoe.static_pitch_rad,
            lambda qs, ch: np.array([ch.angle(q, 3) for q in qs]) - trial.shoe.static_pitch_rad,
        ),
    ]
    for label, unit, sim_func, target_func in channel_specs:
        channels = []
        for j, g in enumerate(GEOMETRIES):
            c = name(m, g)
            trace = trace_map[c]
            case = read_json(run / c / evidence[c][SAMPLE]["scenario"])
            chain = Chain(**case["chain"])
            channels.append((GEOMETRY_LABELS[g], trace["time_s"], sim_func(trace, chain), COLORS[j]))
        channels.append(
            ("Measured reference (prepared coordinates)", trial.time_s, target_func(trial.q, trial.chain), "#222222")
        )
        sections.append(
            {
                "title": "Current material — " + label,
                "kind": "biomechanics",
                "html": "<p>First held-out window; native saved states. Vertical channels show change from each initial height to remove the declared placement shift. Hip height is a proxy, not whole-body COM. Absolute pitch increases from world +X toward +Z and is shown relative to static shoe pitch.</p>"
                + lines(label, unit, channels),
            }
        )


def category_catalog(outputs):
    """Keep every material study in the Materials testing category landing page."""
    rebuild(outputs)


def main(argv=None):
    """Render the material by geometry report without evaluating physics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--renderer", type=Path, default=RENDERER)
    args = parser.parse_args(argv)
    print(report(args.run.resolve(), args.outputs.resolve(), args.renderer.resolve()))


if __name__ == "__main__":
    main()
