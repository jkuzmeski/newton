# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a searchable catalog for saved Instron experiment reports."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return "—"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _classify(meta: dict, run_id: str) -> str:
    """Assign report categories from explicit metadata before name hints."""
    category = str(meta.get("provenance", {}).get("Category", "")).lower()
    if "controller" in category:
        return "controller"
    if meta.get("domain") == "digital-instron":
        return "materials"
    clues = " ".join((run_id, str(meta.get("title", "")), str(meta.get("provenance", {}).get("Category", "")))).lower()
    return "materials" if "material" in clues or "foam" in clues or "geometry" in clues else "controller"


def _controller_metrics(meta: dict, outputs: Path) -> dict:
    provenance = meta.get("provenance", {})
    selection = provenance.get("selection", {}) if isinstance(provenance, dict) else {}
    train = selection.get("training_loss")
    eval_loss = None
    native = selection.get("selected_native_summary")
    if native:
        native_path = Path(str(native))
        if not native_path.is_absolute():
            native_path = outputs / native_path
        summary = _read_json(native_path)
        eval_metrics = summary.get("eval_metrics", {})
        if isinstance(eval_metrics, dict):
            eval_loss = eval_metrics.get("mean_loss")
    # Current experiment summaries keep both split metrics on each candidate.
    if eval_loss is None:
        run_id = str(meta.get("run_id", ""))
        summary = _read_json(outputs / "impedance_instron" / run_id / "summary.json")
        best = summary.get("best", {})
        if isinstance(best, dict):
            if train is None:
                train = best.get("train_mean_loss")
            ev = best.get("eval_metrics", {})
            if isinstance(ev, dict):
                eval_loss = ev.get("mean_loss")
        if eval_loss is None:
            selected_id = selection.get("selected") if isinstance(selection, dict) else None
            for candidate in summary.get("candidates", []):
                if candidate.get("id") == selected_id:
                    train = train if train is not None else candidate.get("train_mean_loss")
                    ev = candidate.get("eval_metrics", {})
                    if isinstance(ev, dict):
                        eval_loss = ev.get("mean_loss")
                    break
    if train is None:
        run_id = meta.get("run_id", "")
        summary = _read_json(outputs / "impedance_instron" / str(run_id) / "summary.json")
        best = summary.get("best", {})
        if isinstance(best, dict):
            train = best.get("train_mean_loss")
            ev = best.get("eval_metrics", {})
            if isinstance(ev, dict):
                eval_loss = ev.get("mean_loss")
    if train is None:
        # This older sweep declared its representative before evaluating it.
        run_dir = outputs / "impedance_instron" / str(meta.get("run_id", ""))
        selected = _read_json(run_dir / "selected-model.json")
        model = selected.get("primary_model")
        if model:
            candidate_dir = Path(model).parent
            for candidate in _read_json(run_dir / "comparison.json").get("metrics", []):
                if candidate.get("condition") == "full w=1" and candidate_dir.name == "fit_vibration_1":
                    if candidate.get("split") == "train":
                        train = candidate.get("common_mean_loss")
                    elif candidate.get("split") == "eval":
                        eval_loss = candidate.get("common_mean_loss")
    return {"training_loss": train, "heldout_loss": eval_loss, "metric_name": "saved objective loss"}


def _material_rows(outputs: Path, run_id: str) -> list[dict]:
    metric_path = outputs / "impedance_instron" / run_id / "metrics.json"
    data = _read_json(metric_path)
    rows: list[dict] = []
    if data.get("schema") == "material-geometry-metrics-v1":
        split = data.get("splits", {}).get("eval", {})
        conditions = split.get("conditions", {})
        for condition, entry in conditions.items():
            means = entry.get("common_means", {})
            rows.append(
                {
                    "condition": condition,
                    "n": entry.get("n", 0),
                    "completed": entry.get("completed", 0),
                    "positive_work_j": means.get("total_positive_work_j"),
                    "absorbed_work_j": means.get("total_absorbed_work_j"),
                    "contact_duration_s": means.get("contact_duration_s"),
                    "peak_compression_fraction": means.get("peak_compression_fraction"),
                }
            )
    else:
        split = data.get("split_metrics", {}).get("eval", {})
        for condition, entry in split.get("conditions", {}).items():
            means = entry.get("all_completed_means", {})
            positive = means.get("total_positive_work_j")
            absorbed = means.get("total_absorbed_work_j")
            # Some early metrics exports contain only per-joint arrays.
            if positive is None and isinstance(means.get("joint_positive_work_j"), list):
                positive = sum(means["joint_positive_work_j"])
            if absorbed is None and isinstance(means.get("joint_negative_work_j"), list):
                absorbed = -sum(means["joint_negative_work_j"])
            rows.append(
                {
                    "condition": condition,
                    "n": entry.get("n_trials", 0),
                    "completed": entry.get("n_completed", 0),
                    "positive_work_j": positive,
                    "absorbed_work_j": absorbed,
                    "contact_duration_s": means.get("contact_duration_s"),
                    "peak_compression_fraction": means.get("peak_compression_fraction"),
                }
            )
    return rows


def _legacy_reports(outputs: Path) -> list[dict]:
    excluded = ("/replay_evidence/", "/assets/", "/source/", "/backups/", "/backup/")
    result = []
    for path in sorted((outputs / "impedance_instron").rglob("*.html")):
        rel = path.relative_to(outputs).as_posix()
        if not path.name.endswith("report.html") or any(part in f"/{rel}" for part in excluded):
            continue
        if not path.is_file():
            continue
        # Reports with canonical metadata have their own experiment card.
        parent = path.parent
        if (outputs / "reports" / "impedance-instron" / parent.name / "report-meta.json").exists():
            continue
        result.append({"title": parent.name.replace("_", " "), "href": rel})
    return result


def _discover(outputs: Path) -> tuple[list[dict], list[dict]]:
    retained = _read_json(
        outputs.parent / "projects/impedance_instron/hogan/baselines/generative_runner_score_20261010.provenance.json"
    )
    reports = []
    for meta_path in sorted((outputs / "reports").rglob("report-meta.json")):
        meta = _read_json(meta_path)
        if meta.get("domain") not in {"impedance-instron", "digital-instron"} or not meta.get("run_id"):
            continue
        run_id = str(meta["run_id"])
        category = _classify(meta, run_id)
        report = str(meta.get("report", "report.html"))
        report_path = (meta_path.parent / report).resolve()
        try:
            report_rel = report_path.relative_to(outputs.resolve()).as_posix()
        except ValueError:
            continue
        if not report_path.is_file():
            continue
        metrics = _material_rows(outputs, run_id) if category == "materials" else []
        run_dir = outputs / "impedance_instron" / run_id
        row = {
            "title": meta.get("title", run_id),
            "run_id": run_id,
            "category": category,
            "status": meta.get("status", "Status unavailable"),
            "summary": meta.get("summary", ""),
            "created_at": meta.get("run_started_at") or meta.get("created_at"),
            "date_kind": "Run started" if meta.get("run_started_at") else "Report created",
            "report": report_rel,
            "run_directory": run_dir.relative_to(outputs).as_posix(),
            "controller_metrics": _controller_metrics(meta, outputs) if category == "controller" else None,
            "conditions": metrics,
            "provenance": meta.get("provenance", {}),
            "retained_controller": category == "controller" and run_id == retained.get("run_id"),
        }
        reports.append(row)
    reports.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return reports, _legacy_reports(outputs)


def _relative_href(from_dir: Path, target: str, outputs: Path) -> str:
    start = from_dir.resolve()
    destination = (outputs / target).resolve()
    return Path(os.path.relpath(destination, start.as_posix())).as_posix()


def _date_label(value: str | None) -> str:
    """Show an artifact timestamp in the archive's Eastern time convention."""
    if not value:
        return "Date unavailable"
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            return value
        return instant.astimezone(ZoneInfo("America/New_York")).strftime("%b %d, %Y · %-I:%M %p %Z")
    except ValueError:
        return value


def _common_head(title: str, root_prefix: str) -> str:
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#102535"><title>{_esc(title)} · Instron</title><link rel="apple-touch-icon" sizes="180x180" href="{root_prefix}report-ui/apple-touch-icon.png"><link rel="icon" type="image/png" href="{root_prefix}report-ui/favicon.png"><link rel="manifest" href="{root_prefix}report-ui/site.webmanifest"><meta name="apple-mobile-web-app-title" content="Instron"><style>
:root{{--ink:#142c3b;--muted:#607586;--line:#d8e2e8;--paper:#f3f7f8;--white:#fff;--blue:#176b78;--teal:#bce8df;--amber:#fae6bc;font:16px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--paper)}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1200px;margin:auto;padding:clamp(16px,4vw,42px);min-width:0}}a{{color:#075d70;overflow-wrap:anywhere}}header{{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}}.brand{{font-size:.83rem;color:var(--muted);letter-spacing:.09em;text-transform:uppercase}}nav{{display:flex;gap:8px;flex-wrap:wrap}}nav a,.button{{display:inline-flex;align-items:center;justify-content:center;min-height:44px;padding:9px 14px;border:1px solid var(--line);border-radius:999px;background:white;text-decoration:none;font-weight:650}}h1{{font-size:clamp(2rem,5vw,3.4rem);letter-spacing:-.04em;line-height:1.08;margin:.4em 0}}h2{{font-size:1.22rem;margin:.2rem 0 .5rem}}p{{max-width:72ch}}.lede,.muted,.meta{{color:var(--muted)}}.hero{{padding:clamp(22px,5vw,48px);margin:24px 0;background:#e3f0ef;border-radius:24px}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,175px),1fr));gap:12px;margin:20px 0}}.stat,.panel,.card{{background:white;border:1px solid var(--line);border-radius:16px;padding:18px;min-width:0}}.stat strong{{display:block;font-size:1.8rem;line-height:1.2}}.toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}}input,select{{font:inherit;min-height:46px;padding:9px 12px;border:1px solid #9aafbd;border-radius:10px;background:#fff;max-width:100%}}input{{flex:1;min-width:min(100%,230px)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:14px}}.card{{display:flex;flex-direction:column;gap:8px}}.card h2 a{{text-decoration:none}}.pill{{display:inline-flex;padding:4px 10px;border-radius:999px;background:#e7eff2;color:#294958;font-size:.83rem;font-weight:650}}.row{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.chart{{display:flex;align-items:end;gap:12px;min-height:150px;padding:12px 0;overflow-x:auto}}.bar-group{{min-width:65px;text-align:center;font-size:.78rem;color:var(--muted)}}.bar-stack{{height:118px;display:flex;align-items:end;justify-content:center;gap:5px}}.bar{{width:18px;min-height:2px;border-radius:5px 5px 0 0;background:#277b86}}.bar.absorbed{{background:#e5aa52}}.table-wrap{{max-width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:12px}}table{{border-collapse:collapse;width:100%;min-width:690px;background:white}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{font-size:.83rem;color:var(--muted);background:#f8fafb}}.footnote{{font-size:.89rem;color:var(--muted)}}.archive{{margin-top:32px}}.archive li{{margin:7px 0}}[hidden]{{display:none!important}}@media(max-width:520px){{main{{padding:14px}}.hero{{border-radius:16px}}.card,.panel{{padding:15px}}nav{{width:100%}}nav a{{flex:1}}}}
</style></head><body>'''


def _nav(root_prefix: str) -> str:
    return f'<header><div class="brand">Newton · Instron studies</div><nav><a href="{root_prefix}index.html">Dashboard</a><a href="{root_prefix}controller-testing/index.html">Controller tests</a><a href="{root_prefix}materials-testing/index.html">Material tests</a></nav></header>'


def _experiment_card(item: dict, href: str) -> str:
    metric = item.get("controller_metrics") or {}
    extra = ""
    if item["category"] == "controller":
        extra = f'<div class="row"><span class="pill">Train loss: {_number(metric.get("training_loss"))}</span><span class="pill">Held-out loss: {_number(metric.get("heldout_loss"))}</span></div><p class="footnote">Saved objective loss; compare only within the same objective and dataset.</p>'
        if item.get("retained_controller"):
            extra = '<span class="pill" style="background:#bce8df">Retained controller baseline</span>' + extra
    stamp = item.get("date_kind", "Recorded") + " " + _date_label(item.get("created_at"))
    return f'<article class="card experiment" data-category="{_esc(item["category"])}" data-status="{_esc(item["status"])}" data-search="{_esc((item["title"] + " " + item["run_id"] + " " + item["summary"] + " " + item["status"]).lower())}"><div class="row"><span class="pill">{_esc(item["category"].title())}</span><span class="meta">{_esc(stamp)}</span></div><h2><a href="{_esc(href)}">{_esc(item["title"])}</a></h2><p>{_esc(item["summary"])}</p><div class="meta">{_esc(item["run_id"])}</div><div class="row"><span class="pill">{_esc(item["status"])}</span></div>{extra}<a class="button" href="{_esc(href)}">Open experiment report</a></article>'


def _svg_work_chart(rows: list[dict]) -> str:
    usable = [
        r
        for r in rows
        if isinstance(r.get("positive_work_j"), (float, int)) and isinstance(r.get("absorbed_work_j"), (float, int))
    ]
    if not usable:
        return '<p class="muted">No saved material work metrics were found.</p>'
    maximum = max(max(r["positive_work_j"], r["absorbed_work_j"]) for r in usable) or 1
    bars = []
    for row in usable:
        label = str(row["condition"]).replace("__", " · ").replace("_", " ")
        pos = max(2, round(104 * row["positive_work_j"] / maximum))
        neg = max(2, round(104 * row["absorbed_work_j"] / maximum))
        bars.append(
            f'<div class="bar-group" title="{_esc(label)}: positive {_number(row["positive_work_j"])} J; absorbed {_number(row["absorbed_work_j"])} J"><div class="bar-stack"><span class="bar" style="height:{pos}px"></span><span class="bar absorbed" style="height:{neg}px"></span></div>{_esc(label)}</div>'
        )
    return (
        '<div class="chart" role="img" aria-label="Positive and absorbed joint work in joules; teal is positive work and amber is absorbed work">'
        + "".join(bars)
        + '</div><div class="row"><span class="pill">■ Positive work (J)</span><span class="pill" style="background:#fae6bc">■ Absorbed work (J)</span></div>'
    )


def _condition_details(outputs: Path, item: dict, row: dict) -> str:
    run_id = item["run_id"]
    condition = str(row["condition"])
    run_dir = outputs / "impedance_instron" / run_id / condition
    viewer = run_dir / "replay_evidence" / "viewer.html"
    details_root = outputs / "experiment-details" / run_id / condition
    details_root.mkdir(parents=True, exist_ok=True)
    rel_viewer = (
        _relative_href(details_root, viewer.relative_to(outputs).as_posix(), outputs) if viewer.exists() else None
    )
    comparison_viewer = run_dir.parent / "comparison-viewer.html"
    if comparison_viewer.is_file():
        rel_viewer = _relative_href(details_root, comparison_viewer.relative_to(outputs).as_posix(), outputs)
        rel_viewer += "?shoe=" + quote(condition, safe="")
    report_href = _relative_href(details_root, item["report"], outputs)
    frame = (
        f'<iframe title="Saved replay for {_esc(condition)}" src="{_esc(rel_viewer)}" loading="eager" style="width:100%;height:min(85vh,1100px);min-height:620px;border:1px solid #d8e2e8;border-radius:14px;background:white"></iframe>'
        if rel_viewer
        else '<p class="muted">No saved condition replay is available.</p>'
    )
    body = f'''{_common_head(condition, "../../../")}<main>{_nav("../../../")}<section class="hero"><div class="brand">{_esc(item["run_id"])} · held-out results</div><h1>{_esc(condition.replace("__", " · ").replace("_", " "))}</h1><p>Saved condition metrics from {int(row.get("completed") or 0)} completed of {int(row.get("n") or 0)} matched evaluation windows.</p><a class="button" href="{_esc(report_href)}">Parent experiment report</a></section><section class="stats"><div class="stat"><span class="muted">Positive joint work</span><strong>{_number(row.get("positive_work_j"))} J</strong></div><div class="stat"><span class="muted">Absorbed joint work</span><strong>{_number(row.get("absorbed_work_j"))} J</strong></div><div class="stat"><span class="muted">Contact duration</span><strong>{_number(row.get("contact_duration_s"))} s</strong></div><div class="stat"><span class="muted">Peak compression fraction</span><strong>{_number(row.get("peak_compression_fraction"))}</strong></div></section><section class="panel"><h2>Saved native replay</h2><p class="footnote">Metrics above belong to this page's condition. The motion selector opens that shoe initially and lets you overlay other conditions from the same study. Drag the spring close-up to rotate.</p>{frame}</section></main></body></html>'''
    (details_root / "index.html").write_text(body, encoding="utf-8")
    return _relative_href(
        outputs / "materials-testing", (details_root / "index.html").relative_to(outputs).as_posix(), outputs
    )


def _material_table(outputs: Path, item: dict, page_dir: Path) -> str:
    rows = item["conditions"]
    if not rows:
        return '<p class="muted">Saved condition metrics are unavailable.</p>'
    cells = []
    for row in rows:
        detail_href = _condition_details(outputs, item, row)
        cells.append(
            f'<tr><td><a href="{_esc(detail_href)}">{_esc(row["condition"])}</a></td><td>{row.get("completed", 0)}/{row.get("n", 0)}</td><td>{_number(row.get("positive_work_j"))}</td><td>{_number(row.get("absorbed_work_j"))}</td><td>{_number(row.get("contact_duration_s"))}</td><td>{_number(row.get("peak_compression_fraction"))}</td></tr>'
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Material · geometry condition</th><th>Eval n</th><th>Positive joint work (J)</th><th>Absorbed joint work (J)</th><th>Contact (s)</th><th>Peak compression fraction</th></tr></thead><tbody>'
        + "".join(cells)
        + "</tbody></table></div>"
    )


def _build_pages(outputs: Path, reports: list[dict], legacy: list[dict]) -> None:
    out_data = {
        "schema": "instron-report-catalog-v1",
        "generated_from": "reports/**/report-meta.json and saved impedance_instron metrics.json",
        "canonical_experiment_count": len(reports),
        "experiments": reports,
        "legacy_reports": legacy,
    }
    (outputs / "catalog-data.json").write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (outputs / "controller-testing").mkdir(parents=True, exist_ok=True)
    (outputs / "materials-testing").mkdir(parents=True, exist_ok=True)

    dashboard = (
        _common_head("Experiment dashboard", "")
        + "<main>"
        + _nav("")
        + '<section class="hero"><div class="brand">Saved experiments · overview</div><h1>Experiment dashboard</h1><p class="lede">Browse controller development and material experiments, with each result linked to its saved report.</p><div class="stats"><div class="stat"><span class="muted">Study reports</span><strong>'
        + str(len(reports))
        + '</strong></div><div class="stat"><span class="muted">Controller tests</span><strong>'
        + str(sum(x["category"] == "controller" for x in reports))
        + '</strong></div><div class="stat"><span class="muted">Material studies</span><strong>'
        + str(sum(x["category"] == "materials" for x in reports))
        + '</strong></div></div><div class="row"><a class="button" href="controller-testing/index.html">Browse controller tests</a><a class="button" href="materials-testing/index.html">Browse material tests</a></div></section><section class="panel"><h2>Material work comparison</h2><p class="footnote">Held-out mean joint work from the saved material runs. Positive and absorbed work share one scale; teal shows generation and amber shows absorption. This is mechanical work, not a metabolic measurement.</p>'
    )
    all_material_rows = []
    for item in reports:
        if item["category"] == "materials":
            all_material_rows.extend([{**r, "study": item["title"]} for r in item["conditions"]])
    dashboard += (
        _svg_work_chart(all_material_rows)
        + '</section><section><div class="toolbar"><input id="search" type="search" placeholder="Search studies, status, or summary" aria-label="Search experiments"><select id="category" aria-label="Filter category"><option value="all">All categories</option><option value="controller">Controller</option><option value="materials">Materials</option></select><select id="status" aria-label="Filter status"><option value="all">All statuses</option><option value="completed">Completed</option><option value="stopped">Stopped / superseded</option></select><span id="count" class="muted"></span></div><div class="grid" id="experiments">'
    )
    dashboard += "".join(_experiment_card(item, item["report"]) for item in reports)
    dashboard += (
        '</div></section><section class="archive"><details><summary><strong>Original reports archive</strong> · '
        + str(len(legacy))
        + ' standalone reports</summary><p class="muted">Older reports remain available here. Counts above include only study reports with saved report metadata.</p><ul>'
        + "".join(f'<li><a href="{_esc(x["href"])}">{_esc(x["title"])}</a></li>' for x in legacy)
        + '</ul></details></section></main><script>const q=document.querySelector("#search"),c=document.querySelector("#category"),s=document.querySelector("#status"),cards=[...document.querySelectorAll(".experiment")];function apply(){let n=0;for(const x of cards){const ok=x.dataset.search.includes(q.value.toLowerCase())&&(c.value==="all"||c.value===x.dataset.category)&&(s.value==="all"||(s.value==="completed"?x.dataset.status.toLowerCase().includes("completed"):!x.dataset.status.toLowerCase().includes("completed")));x.hidden=!ok;if(ok)n++}document.querySelector("#count").textContent=n+" experiments"}q.addEventListener("input",apply);c.addEventListener("change",apply);s.addEventListener("change",apply);apply();</script></body></html>'
    )
    (outputs / "index.html").write_text(dashboard, encoding="utf-8")

    for category, dirname, title in (
        ("controller", "controller-testing", "Controller tests"),
        ("materials", "materials-testing", "Material tests"),
    ):
        page_dir = outputs / dirname
        selected = [r for r in reports if r["category"] == category]
        page = (
            _common_head(title, "../")
            + "<main>"
            + _nav("../")
            + f'<section class="hero"><div class="brand">Experiment collection</div><h1>{_esc(title)}</h1><p class="lede">{len(selected)} saved study reports. Open an experiment report for its full methods, plots, and artifacts.</p></section>'
        )
        if category == "materials":
            for item in selected:
                page += f'<section class="panel"><div class="row"><span class="pill">{_esc(item["status"])}</span><span class="meta">{_esc(_date_label(item.get("created_at")))}</span></div><h2><a href="{_esc(_relative_href(page_dir, item["report"], outputs))}">{_esc(item["title"])}</a></h2><p>{_esc(item["summary"])}</p><p class="footnote">Primary split: held-out evaluation. Positive and absorbed joint work are shown separately. Condition pages open the corresponding saved replay and geometry.</p>{_material_table(outputs, item, page_dir)}</section>'
            page += '<section class="panel"><h2>Conditional marathon estimates</h2><p class="footnote">Where present, these are heuristic projections from mechanical work with explicit coupling assumptions. They are not measured metabolic changes or a general performance ranking.</p>'
            for item in selected:
                estimate_path = outputs / "impedance_instron" / item["run_id"] / "marathon-estimates.json"
                if estimate_path.exists():
                    estimates = _read_json(estimate_path)
                    assumptions = estimates.get("assumptions", {})
                    page += f'<h3>{_esc(item["title"])}</h3><p class="footnote">{_esc(estimates.get("interpretation", "Conditional estimate"))}. Positive seconds mean faster; negative seconds mean slower. Ranges reflect assumptions, not confidence intervals; actual benefit may be absent or reversed. Baseline: {_number(assumptions.get("baseline_speed_m_s"))} m/s; distance: {_number(assumptions.get("distance_m"), 0)} m; work factors: +{_number(assumptions.get("positive_metabolic_j_per_mechanical_j"))} and absorbed {_number(assumptions.get("absorbed_metabolic_j_per_mechanical_j"))}. <a href="{_esc(_relative_href(page_dir, (outputs / "impedance_instron" / item["run_id"] / "marathon-estimates.json").relative_to(outputs).as_posix(), outputs))}">Open full assumptions and estimates JSON</a></p>'
                    for est in estimates.get("conditions", []):
                        seconds = est.get("marathon_seconds_saved", [])
                        couplings = assumptions.get("assumed_effective_coupling_fractions", [])
                        if not seconds:
                            continue
                        entries = "".join(f"<td>{_number(value, 0)} s</td>" for value in seconds)
                        headers = "".join(
                            f"<th>{round(float(value) * 100)}% coupling</th>" for value in couplings[: len(seconds)]
                        )
                        page += f'<h4>{_esc(est.get("condition", "Condition"))}</h4><div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody><tr>{entries}</tr></tbody></table></div>'
            page += "</section>"
        else:
            page += (
                '<div class="grid">'
                + "".join(
                    _experiment_card(item, _relative_href(page_dir, item["report"], outputs)) for item in selected
                )
                + "</div>"
            )
            if legacy:
                page += (
                    '<section class="archive"><details><summary><strong>Original standalone controller reports</strong> · '
                    + str(len(legacy))
                    + "</summary><ul>"
                    + "".join(
                        f'<li><a href="{_esc(_relative_href(page_dir, x["href"], outputs))}">{_esc(x["title"])}</a></li>'
                        for x in legacy
                    )
                    + "</ul></details></section>"
                )
        page += "</main></body></html>"
        (page_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    outputs = args.outputs.resolve()
    if not outputs.is_dir():
        parser.error(f"outputs directory does not exist: {outputs}")
    count = rebuild(outputs)
    print(f"Wrote dashboard for {count} canonical experiments to {outputs}")


def rebuild(outputs: Path) -> int:
    """Regenerate the experiment catalog and category pages under outputs."""
    outputs = Path(outputs).resolve()
    reports, legacy = _discover(outputs)
    _build_pages(outputs, reports, legacy)
    for report in reports:
        path = outputs / report["report"]
        source = path.read_text(encoding="utf-8")
        source = re.sub(r'<nav id="experiment-catalog-nav".*?</nav>', "", source, flags=re.S)
        category_path = (
            "materials-testing/index.html" if report["category"] == "materials" else "controller-testing/index.html"
        )
        category_label = "Material tests" if report["category"] == "materials" else "Controller tests"
        navigation = (
            '<nav id="experiment-catalog-nav" aria-label="Experiment collection" '
            'style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0">'
            f'<a href="{_esc(_relative_href(path.parent, "index.html", outputs))}" style="padding:10px">Dashboard</a>'
            f'<a href="{_esc(_relative_href(path.parent, category_path, outputs))}" style="padding:10px">{category_label}</a>'
            "</nav>"
        )
        path.write_text(source.replace("</header>", navigation + "</header>", 1), encoding="utf-8")
    return len(reports)


if __name__ == "__main__":
    main()
