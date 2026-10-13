# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render all material sweep candidates and the independently selected finalists."""

from __future__ import annotations

import argparse
import csv
import html
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from projects.impedance_instron.report_catalog import rebuild

from .comparison_viewer_report import build_comparison_viewer
from .identify import ForceObservation, load_trials
from .materials import sha256, write_json
from .materials_geometry_report import lines, table
from .materials_marathon_report import PACE_SOURCE, WORK_SOURCE, marathon_seconds_saved, weighted_work
from .materials_report import RENDERER, read_json
from .materials_sweep import DATASET, MOUNT
from .mechanics import Chain

FACTORS = (
    ("equilibrium_compression_scale", "Equilibrium compression stiffness", "x baseline"),
    ("equilibrium_shear_scale", "Equilibrium shear stiffness", "x baseline"),
    ("equilibrium_fraction", "Equilibrium fraction", "fraction"),
    ("maxwell_relaxation_time_s", "Maxwell relaxation time", "ms"),
)
BASE_PARAMETERS = dict(zip((f[0] for f in FACTORS), (1.0, 1.0, 0.6952954850747634, 0.005000407970309165), strict=True))
COLORS = ("#365f91", "#237d67", "#d78315", "#ad4660")
SAMPLE = "FR3_1_eval_000"


def seconds(reduction: float, coupling: float = 0.65) -> float:
    """Convert a work reduction into the declared conditional time scenario [s]."""
    return marathon_seconds_saved(coupling * reduction)


def baseline_slice(row: dict, varied: str) -> bool:
    """Identify matched one-factor changes from the original material settings."""
    p = row["parameters"]
    for key, value in BASE_PARAMETERS.items():
        if key == varied or (key == "maxwell_relaxation_time_s" and p["equilibrium_fraction"] == 1.0):
            continue
        if not np.isclose(p[key], value, rtol=1e-10, atol=1e-12):
            return False
    return True


def sensitivity_chart(rows: list[dict], factor: tuple, base_work: float, span: float) -> str:
    """Show all candidates, including harm and rejected task-quality cases."""
    key, label, unit = factor
    levels = sorted({r["parameters"][key] for r in rows})
    low, high = -span, span

    def x(value):
        return 95 + 550 * (levels.index(value) + 0.5) / len(levels)

    def y(value):
        return 285 - 235 * (value - low) / (high - low)

    out = [
        f'<svg viewBox="0 0 720 410" role="img" aria-label="{html.escape(label)} sensitivity"><title>{html.escape(label)}: conditional marathon seconds saved or lost on training windows</title>'
    ]
    out.append('<text x="85" y="22" font-size="15">Conditional seconds saved (+) / lost (-), train</text>')
    for tick in np.linspace(low, high, 5):
        out.append(
            f'<line x1="80" x2="690" y1="{y(tick):.2f}" y2="{y(tick):.2f}" stroke="{"#72838e" if tick == 0 else "#dce3e8"}"/><text x="70" y="{y(tick) + 4:.2f}" text-anchor="end" font-size="13">{tick:+.0f}</text>'
        )
    groups = {v: [r for r in rows if r["parameters"][key] == v] for v in levels}
    for value, group in groups.items():
        for i, row in enumerate(group):
            xx = x(value) + 27 * (i / max(len(group) - 1, 1) - 0.5)
            complete = row["metrics"]["completed"] == row["metrics"]["n"]
            if not complete:
                out.append(
                    f'<text x="{xx:.2f}" y="302" fill="#aa3344" text-anchor="middle">x<title>{html.escape(row["id"])}: incomplete, no time estimate</title></text>'
                )
                continue
            value_s = seconds(1 - row["weighted_work_j"] / base_work)
            color = "#237d67" if value_s >= 0 else "#ad4660"
            fill = color if row["eligible"] else "white"
            out.append(
                f'<circle cx="{xx:.2f}" cy="{y(value_s):.2f}" r="4" fill="{fill}" stroke="{color}"><title>{html.escape(row["id"])}: {value_s:+.2f} s; eligible={row["eligible"]}</title></circle>'
            )
        text = f"{value * 1000:.3g}" if key.endswith("time_s") else f"{value:.3g}"
        out.append(f'<text x="{x(value):.2f}" y="325" text-anchor="middle" font-size="13">{text}</text>')
    matched = sorted(
        (r for r in rows if baseline_slice(r, key) and r["metrics"]["completed"] == r["metrics"]["n"]),
        key=lambda r: r["parameters"][key],
    )
    points = " ".join(
        f"{x(r['parameters'][key]):.2f},{y(seconds(1 - r['weighted_work_j'] / base_work)):.2f}" for r in matched
    )
    out.append(f'<polyline points="{points}" fill="none" stroke="#18334a" stroke-width="2"/>')
    for r in matched:
        out.append(
            f'<circle cx="{x(r["parameters"][key]):.2f}" cy="{y(seconds(1 - r["weighted_work_j"] / base_work)):.2f}" r="5" fill="#18334a"/>'
        )
    out.extend(
        (
            f'<text x="360" y="350" text-anchor="middle" font-size="14">{html.escape(label)} [{html.escape(unit)}]</text>',
            '<text x="80" y="375" font-size="12">Filled: eligible · hollow: rejected · x: incomplete</text>',
            '<text x="80" y="395" font-size="12">Dark line: change one factor from baseline; dots: all combinations</text></svg>',
        )
    )
    return "".join(out)


def add_native_sections(eval_root: Path, summaries: dict, labels: dict, sections: list) -> None:
    """Plot the finalists' native states and loads beside the measured sample."""
    trial = next(
        t
        for t in load_trials(DATASET, mount_m=MOUNT, speed_m_s=3.65, force_observation=ForceObservation(20, 1, 1))
        if t.id == SAMPLE
    )
    traces, cases = {}, {}
    for name, summary in summaries.items():
        row = next(r for r in summary["trials"] if r["id"] == SAMPLE)
        if sha256(trial.provenance["reference"]) != row["reference_sha256"]:
            raise ValueError("Measured reference identity changed")
        traces[name] = dict(np.load(eval_root / name / row["trace"]))
        cases[name] = read_json(eval_root / name / row["scenario"])

    def add(title, unit, simulated, reference, force=False):
        channels = []
        for j, name in enumerate(summaries):
            trace = traces[name]
            chain = Chain(**cases[name]["chain"])
            clock = trace["time_s"][:-1] if force else trace["time_s"]
            channels.append((labels[name], clock, simulated(trace, chain), COLORS[j]))
        clock = trial.force_time_s if force else trial.time_s
        channels.append(("Measured reference", clock, reference(trial), "#222222"))
        sections.append(
            {
                "title": title,
                "kind": "biomechanics",
                "html": f"<p>Native saved simulation and measured reference for {SAMPLE}; matched initial conditions. Hip height is a proxy for body motion.</p>"
                + lines(title, unit, channels).replace('viewBox="0 0 720 420"', 'viewBox="0 0 720 450"'),
            }
        )

    for axis, title in ((0, "Fore-aft GRF"), (1, "Vertical GRF")):
        add(title, "N", lambda t, c, a=axis: t["grf_n"][:, a], lambda t, a=axis: t.grf_n[:, a], True)
    for axis, title in ((0, "Hip X"), (1, "Hip height"), (4, "Knee relative angle"), (5, "Ankle relative angle")):
        add(title, "m" if axis < 2 else "rad", lambda t, c, a=axis: t["state"][:, a], lambda t, a=axis: t.q[:, a])
    for axis, title in ((0, "Ankle X"), (1, "Ankle height")):
        add(
            title,
            "m",
            lambda t, c, a=axis: np.array([c.point(q, 3, np.zeros(2))[0][a] for q in t["state"]]),
            lambda t, a=axis: np.array([t.chain.point(q, 3, np.zeros(2))[0][a] for q in t.q]),
        )
    add(
        "Shoe pitch relative to static mount",
        "rad",
        lambda t, c: np.array([c.angle(q, 3) for q in t["state"]]) - trial.shoe.static_pitch_rad,
        lambda t: np.array([t.chain.angle(q, 3) for q in t.q]) - t.shoe.static_pitch_rad,
    )
    for joint, title in enumerate(("Hip", "Knee", "Ankle")):
        for key, unit in (("load", "N·m"), ("power", "W"), ("equilibrium", "rad")):
            channels = []
            for j, name in enumerate(summaries):
                t = traces[name]
                values = t["load"][:, joint + 3]
                if key == "power":
                    values = values * t["velocity"][:-1, joint + 3]
                elif key == "equilibrium":
                    values = t["equilibrium_rad"][:, joint]
                channels.append((labels[name], t["time_s"][:-1], values, COLORS[j]))
            sections.append(
                {
                    "title": f"{title} actuator {key}",
                    "kind": "biomechanics",
                    "html": "<p>Native actuator output; no measured muscle torque, equilibrium schedule or power is available.</p>"
                    + lines(f"{title} actuator {key}", unit, channels),
                }
            )


def report(run: Path, outputs: Path) -> Path:
    """Publish the complete sweep with preserved adverse results and provenance."""
    search = read_json(run / "train_search.json")
    rows = search["results"] + search.get("refinements", [])
    base = search["baseline"]
    base_work = weighted_work(base["mean_positive_work_j"], base["mean_absorbed_work_j"])
    eval_root = run / "finalists"
    top = read_json(eval_root / "summary.json")
    ids = [c["id"] for c in top["conditions"]]
    summaries = {name: read_json(eval_root / name / "summary.json") for name in ids}
    labels = {name: "Baseline" if name == "original" else f"Finalist {i}" for i, name in enumerate(ids)}
    sections = []
    evaluated = {}
    for name, summary in summaries.items():
        evaluated[name] = {r["id"]: r["metrics"] for r in summary["trials"] if r["split"] == "eval"}
        for row in summary["trials"]:
            if row["split"] == "eval":
                with np.load(eval_root / name / row["trace"]) as trace:
                    evaluated[name][row["id"]]["peak_compression_fraction"] = float(
                        np.max(trace["compression_fraction"])
                    )
    common = sorted(
        set.intersection(*({i for i, r in group.items() if r["status"] == "completed"} for group in evaluated.values()))
    )
    if not common:
        raise ValueError("No common completed held-out windows")
    metrics = {
        "schema": "material-geometry-metrics-v1",
        "splits": {"eval": {"common_completed_ids": common, "conditions": {}}},
    }
    for name, group in evaluated.items():
        means = {
            key: np.mean([group[i][key] for i in common], axis=0).tolist()
            for key in (
                "total_positive_work_j",
                "total_absorbed_work_j",
                "contact_duration_s",
                "peak_compression_fraction",
                "peak_grf_n",
                "joint_positive_work_j",
                "joint_negative_work_j",
                "loss",
            )
        }
        metrics["splits"]["eval"]["conditions"][name] = {
            "n": len(group),
            "completed": sum(r["status"] == "completed" for r in group.values()),
            "common_means": means,
        }
    write_json(run / "metrics.json", metrics)
    means = metrics["splits"]["eval"]["conditions"]
    reference = means["original"]["common_means"]
    reference_cost = weighted_work(reference["total_positive_work_j"], reference["total_absorbed_work_j"])
    estimates, comparison = [], []
    for name, item in means.items():
        m = item["common_means"]
        reduction = 1 - weighted_work(m["total_positive_work_j"], m["total_absorbed_work_j"]) / reference_cost
        times = [seconds(reduction, c) for c in (0.5, 0.65, 0.8)]
        estimates.append({"condition": name, "proxy_reduction_fraction": reduction, "marathon_seconds_saved": times})
        comparison.append(
            [
                labels[name],
                f"{item['completed']}/{item['n']}",
                f"{m['total_positive_work_j']:.3f}",
                f"{m['total_absorbed_work_j']:.3f}",
                f"{100 * reduction:+.3f}",
                f"{times[1]:+.1f}",
                f"{times[0]:+.1f} to {times[2]:+.1f}",
            ]
        )
    write_json(
        run / "marathon-estimates.json",
        {
            "schema": "conditional-marathon-work-proxy-v1",
            "baseline_condition": "original",
            "split": "eval",
            "paired_window_ids": common,
            "conditions": estimates,
            "assumptions": {
                "couplings": [0.5, 0.65, 0.8],
                "speed_m_s": 3.65,
                "distance_m": 42195,
                "mass_kg": 66.5,
                "frontal_area_m2": 0.45,
                "proxy": "4 W+ + |W-|/1.2",
                "economy_rule": "saving = coupling * proxy_reduction_fraction",
                "interpretation": "Chosen effective coupling scenarios, not fitted parameters or confidence bounds",
            },
            "sources": [WORK_SOURCE, PACE_SOURCE],
        },
    )
    sections.append(
        {
            "title": "Held-out finalist performance",
            "kind": "biomechanics",
            "html": f"<p>Finalists were frozen before evaluating nine held-out windows. Paired means use {len(common)} windows completed by every condition. Baseline at 3.65 m/s corresponds to 3:12:40. Positive changes mean better performance.</p>"
            + table(
                [
                    "Condition",
                    "Completed",
                    "Positive work [J]",
                    "Absorbed work [J]",
                    "Work reduction [%]",
                    "Conditional saving [s]",
                    "50-80% coupling [s]",
                ],
                comparison,
            ),
        }
    )
    parameter_rows = []
    for condition in top["conditions"]:
        parameters = read_json(Path(condition["artifact"]))["constitutive_model"]["parameters"]
        eq = parameters["equilibrium_fraction"]
        total_mu = parameters["instantaneous_shear_modulus_pa"] + parameters["instantaneous_shear_modulus_2_pa"]
        shear = parameters.get("equilibrium_shear_modulus_override_pa") or total_mu * eq
        parameter_rows.append(
            [
                labels[condition["id"]],
                f"{2 * total_mu * eq / 1000:.3f}",
                f"{2 * total_mu / 1000:.3f}",
                f"{shear / 1000:.3f}",
                f"{eq:.6g}",
                f"{parameters['maxwell_relaxation_time_s'] * 1000:.6g}",
            ]
        )
    sections.append(
        {
            "title": "Finalist material properties",
            "kind": "details",
            "html": "<p>Small-strain normal tangents refer to the compression law; shear is independently specified. Equilibrium fraction is the ratio of relaxed to instantaneous stiffness, not a measured energy-return percentage. Both constitutive exponents and Poisson ratio remain at baseline.</p>"
            + table(
                [
                    "Condition",
                    "Equilibrium normal [kPa]",
                    "Instantaneous normal [kPa]",
                    "Shear [kPa]",
                    "Eq. fraction",
                    "Tau [ms]",
                ],
                parameter_rows,
            ),
        }
    )
    complete = [r for r in rows if r["metrics"]["completed"] == r["metrics"]["n"]]
    span = max(10, max(abs(seconds(1 - r["weighted_work_j"] / base_work)) for r in complete) * 1.12)
    sensitivity = []
    for factor in FACTORS:
        sections.append(
            {
                "title": factor[1] + " sensitivity",
                "kind": "biomechanics",
                "html": "<p>All completed training candidates, including worse results and task-quality rejections. The dark line changes only this factor from baseline; the scatter also shows interactions with other factors. Each candidate uses the same 98 training windows. Incomplete candidates receive no marathon estimate.</p>"
                + sensitivity_chart(rows, factor, base_work, span),
            }
        )
        sensitivity.extend(
            {
                "factor": factor[0],
                "candidate": r["id"],
                "level": r["parameters"][factor[0]],
                "one_factor_from_baseline": baseline_slice(r, factor[0]),
                "completed": r["metrics"]["completed"],
                "eligible": r["eligible"],
                "conditional_seconds_saved": seconds(1 - r["weighted_work_j"] / base_work) if r in complete else None,
            }
            for r in rows
        )
    write_json(run / "sensitivity.json", {"split": "train", "baseline_weighted_work_j": base_work, "rows": sensitivity})
    add_native_sections(eval_root, summaries, labels, sections)
    all_rows = []
    with (run / "all-candidates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "candidate",
                *[f[0] for f in FACTORS],
                "weighted_work_j",
                "reduction_percent",
                "eligible",
                "completed",
                "guardrail_failures",
            ]
        )
        for r in sorted(rows, key=lambda r: r["weighted_work_j"]):
            reduction = 100 * (1 - r["weighted_work_j"] / base_work)
            reasons = ", ".join(k for k, passed in r["guardrails"].items() if not passed)
            values = [
                r["id"],
                *[r["parameters"][f[0]] for f in FACTORS],
                r["weighted_work_j"],
                reduction,
                r["eligible"],
                r["metrics"]["completed"],
                reasons,
            ]
            writer.writerow(values)
            all_rows.append(
                [
                    values[0],
                    *[f"{v:.6g}" for v in values[1:5]],
                    f"{values[5]:.3f}",
                    f"{reduction:+.3f}" if r in complete else "Incomplete",
                    r["eligible"],
                    f"{r['metrics']['completed']}/{r['metrics']['n']}",
                    reasons or "None",
                ]
            )
    sections.append(
        {
            "title": "Every material candidate",
            "kind": "details",
            "html": f"<p>{len(rows)} candidates retained. No adverse or rejected experiment was discarded. Sorting is by training mechanical-work proxy; guardrails determine eligibility. A smaller incomplete-window work sum is not evidence of improvement.</p>"
            + table(
                [
                    "Candidate",
                    "Compression x",
                    "Shear x",
                    "Eq. fraction",
                    "Tau [s]",
                    "Work proxy [J]",
                    "Reduction [%]",
                    "Eligible",
                    "Completed",
                    "Failed guardrails",
                ],
                all_rows,
            ),
        }
    )
    qualification = (
        read_json(run / "qualification.json")
        if (run / "qualification.json").exists()
        else {"status": "See saved finalist half-step and replay validation records"}
    )
    sections.append(
        {
            "title": "Numerical qualification and scope",
            "kind": "details",
            "html": "<p>Geometry, controller, body, initial conditions and friction are fixed. Material exponents and Poisson ratio are fixed; the sweep varies four independent factors. These are synthetic uncalibrated material laws. A best tested boundary value does not establish a global optimum.</p><p>Finite single-leg transient windows do not identify periodic gait, muscle energetics, marathon fatigue, foam aging or added shoe mass. Time estimates may fail to predict real benefit.</p>"
            + table(["Check", "Result"], [(k, str(v)) for k, v in qualification.items()]),
        }
    )
    sections.append(
        {
            "title": "Conditional marathon assumptions",
            "kind": "details",
            "html": f'<p>The proxy C = 4 W+ + |W-|/1.2 uses work-cost coefficients discussed by <a href="{WORK_SOURCE}">Riddick and Kuo</a>, with unresolved tendon, joint-transfer and soft-tissue contributions. Gross economy saving is assumed to be 0.50, 0.65 or 0.80 times relative proxy reduction. These couplings are assumptions, not published universal constants or confidence limits.</p><p>Time uses <a href="{PACE_SOURCE}">Kipp et al., Eq. 2</a>, with oxygen rate Q(v) = (3.54 A/m)v³ + 1.5355v² + 1.5354v + 15.661, A = 0.45 m² and m = 66.5 kg, solving (1 - saving)Q(vnew) = Q(3.65). Sustainable oxygen rate and the fractional saving are assumed unchanged at the new speed.</p>',
        }
    )
    viewer = build_comparison_viewer(run, "finalists/original")
    provenance = read_json(run / "provenance.json")
    spec = {
        "title": "FR3 material sweep: performance and parameter effects",
        "domain": "impedance-instron",
        "run_id": run.name,
        "run_started_at": provenance["run_started_at"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "Completed bounded material sensitivity study; conditional performance",
        "summary": f"{len(rows)} material candidates, including {sum(r['weighted_work_j'] > base_work for r in rows)} worse results and {sum(not r['eligible'] for r in rows)} task-quality rejections, all retained. The training-selected winner reduces held-out work proxy by {100 * estimates[1]['proxy_reduction_fraction']:.2f}%, corresponding to {estimates[1]['marathon_seconds_saved'][1]:.0f} conditional seconds saved. Current FR3 geometry and controller frozen.",
        "provenance": {
            "Category": "Materials testing",
            "Run directory": str(run),
            "Controller SHA256": provenance["controller_sha256"],
            "Selection": "Training only; held-out data excluded",
            "Material scope": "Synthetic uncalibrated four-factor sweep",
        },
        "motion_viewer": {
            "source": str(viewer),
            "title": "Baseline and finalist motion with synchronized deformation",
            "caption": "Choose a condition and optionally overlay another; same predeclared held-out window, verified saved-pose contact, fixed 0-60 mm compression scale.",
        },
        "sections": sections,
        "attachments": [
            {"source": str(run / filename), "title": title}
            for filename, title in (
                ("train_search.json", "All training search results and per-window metrics"),
                ("all-candidates.csv", "All candidates CSV, including adverse results"),
                ("sensitivity.json", "Numerical data behind parameter graphs"),
                ("marathon-estimates.json", "Conditional held-out marathon estimates"),
                ("source-verification.json", "Primary-source verification"),
            )
        ],
    }
    write_json(run / "report-spec.json", spec)
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "python",
            str(RENDERER),
            str(run / "report-spec.json"),
            "--outputs",
            str(outputs),
            "--overwrite",
        ],
        check=True,
    )
    target = outputs / "reports/impedance-instron" / run.name.replace("_", "-") / "report.html"
    target.write_text(target.read_text().replace('loading="lazy"', 'loading="eager"'))
    rebuild(outputs)
    return target


def main() -> None:
    """Render saved sweep evidence without running new simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(report(args.run.resolve(), args.outputs.resolve()))


if __name__ == "__main__":
    main()
