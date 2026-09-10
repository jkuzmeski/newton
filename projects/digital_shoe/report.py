# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate a dependency-free HTML validation report for a Digital Shoe artifact."""

from __future__ import annotations

import argparse
import base64
import html
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .artifact import DigitalShoe, load_artifact


def _polyline(x: np.ndarray, y: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> str:
    dx = max(xlim[1] - xlim[0], 1.0e-12)
    dy = max(ylim[1] - ylim[0], 1.0e-12)
    px = 70.0 + (x - xlim[0]) / dx * 626.0
    py = 44.0 + (1.0 - (y - ylim[0]) / dy) * 226.0
    return " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px, py, strict=True))


def _svg(curve: dict, *, domain: str) -> str:
    measured = np.asarray(curve["measured_force_n"], dtype=np.float64)
    predicted = np.asarray(curve["predicted_force_n"], dtype=np.float64)
    if domain == "displacement":
        x = np.asarray(curve["displacement_m"], dtype=np.float64) * 1000.0
        xlabel = "Compression [mm]"
    else:
        x = np.asarray(curve["time_s"], dtype=np.float64)
        xlabel = "Time [s]"
    xlim = (float(x.min()), float(x.max()))
    ylim = (0.0, 1.05 * float(max(measured.max(), predicted.max(), 1.0)))
    measured_points = _polyline(x, measured, xlim, ylim)
    predicted_points = _polyline(x, predicted, xlim, ylim)
    ticks = []
    for fraction in np.linspace(0.0, 1.0, 5):
        px, py = 70.0 + 626.0 * fraction, 270.0 - 226.0 * fraction
        xvalue = xlim[0] + fraction * (xlim[1] - xlim[0])
        yvalue = fraction * ylim[1]
        ticks.append(
            f'<line x1="{px:.1f}" y1="44" x2="{px:.1f}" y2="270" stroke="#e5eaf0"/>'
            f'<line x1="70" y1="{py:.1f}" x2="696" y2="{py:.1f}" stroke="#e5eaf0"/>'
            f'<text x="{px:.1f}" y="290" text-anchor="middle">{xvalue:.3g}</text>'
            f'<text x="60" y="{py + 4:.1f}" text-anchor="end">{yvalue:.0f}</text>'
        )
    return f"""<svg class="response-plot" viewBox="0 0 720 324" role="img" aria-label="Measured and predicted force versus {xlabel}">
<g font-family="system-ui,sans-serif" font-size="13" fill="#526174">
<rect x="70" y="44" width="626" height="226" fill="#fff"/>
{"".join(ticks)}
<line x1="70" y1="270" x2="696" y2="270" stroke="#9aa8b8"/>
<line x1="70" y1="44" x2="70" y2="270" stroke="#9aa8b8"/>
<polyline points="{measured_points}" fill="none" stroke="#1261a0" stroke-width="2.5"/>
<polyline points="{predicted_points}" fill="none" stroke="#d94801" stroke-width="2.5" stroke-dasharray="7 4"/>
<text x="383" y="316" text-anchor="middle">{xlabel}</text>
<text x="18" y="157" transform="rotate(-90 18 157)" text-anchor="middle">Force [N]</text>
<line x1="70" y1="20" x2="98" y2="20" stroke="#1261a0" stroke-width="3"/>
<text x="106" y="24">Measured</text>
<line x1="210" y1="20" x2="238" y2="20" stroke="#d94801" stroke-width="3" stroke-dasharray="7 4"/>
<text x="246" y="24">Predicted</text>
</g></svg>"""


def _fixture_label(curve: dict) -> str:
    """Use readable fixture names while retaining unknown trial names."""
    return {
        "rearfoot_punch": "Rearfoot punch",
        "fullfoot_last": "Full-foot last",
    }.get(curve.get("fixture"), curve["name"])


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _metric_rows(curves: Iterable[dict]) -> str:
    rows = []
    for curve in curves:
        metric = curve["metrics"]
        passed = bool(metric["passed"])
        rows.append(
            "<tr>"
            f'<th scope="row">{html.escape(_fixture_label(curve))}</th>'
            f"<td>{_percent(metric['peak_force_error'])}</td>"
            f"<td>{_percent(metric['force_rmse_relative'])}</td>"
            f"<td>{_percent(metric['hysteresis_error'])}</td>"
            f"<td>{metric['measured_peak_force_n']:.0f} N</td>"
            f'<td><span class="{"pass" if passed else "fail"}">{"PASS" if passed else "NOT ALL GATES"}</span></td>'
            "</tr>"
        )
    return "".join(rows)


def _material_rows(shoe: DigitalShoe) -> str:
    labels = {
        "instantaneous_shear_modulus_pa": ("Instantaneous shear modulus", "Pa"),
        "hyperfoam_exponent": ("Hyperfoam exponent", "1"),
        "equilibrium_fraction": ("Equilibrium fraction", "1"),
        "pasternak_n_per_m": ("Pasternak coupling (derived bed mean of mu_eq x t)", "N/m"),
        "effective_poisson_ratio": ("Effective Poisson ratio (fixed)", "1"),
        "maxwell_relaxation_time_s": ("Maxwell relaxation time (fitted)", "s"),
    }
    values = shoe.raw["constitutive_model"]["parameters"]
    return "".join(
        f"<tr><td>{label}</td><td>{values[key]:.6g}</td><td>{unit}</td></tr>" for key, (label, unit) in labels.items()
    )


def _methods_section() -> str:
    """Render the model derivation, selection rationale, and Mermaid workflow."""
    directory = Path(__file__).resolve().parent
    diagram = (directory / "methods.svg").read_text()
    mermaid_source = html.escape((directory / "methods.mmd").read_text())
    return f"""<section id="methods"><h2>1. Methods</h2>
<p>One <strong>effective intact-shoe model</strong> connects the measured force&ndash;compression cycles to the Newton simulations. Geometry, fitted parameters, and validation curves travel together in a portable artifact.</p>
<ol class="workflow">
<li><strong>Measure</strong><span>Instron cycles and shoe geometry</span></li>
<li><strong>Identify</strong><span>One shared model across fixtures</span></li>
<li><strong>Export</strong><span>Portable digital_shoe.json</span></li>
<li><strong>Simulate</strong><span>Instron, drop, and rocker</span></li>
</ol>
<details class="workflow-details"><summary>Detailed workflow diagram</summary><div class="details-body">
<div class="method-diagram">{diagram}</div>
<details><summary>Mermaid source for the method diagram</summary><pre><code class="language-mermaid">{mermaid_source}</code></pre></details>
</div></details>
<details class="derivation"><summary>Model equations and assumptions</summary>
<div class="details-body">
<h3>1.1 Geometry and column kinematics</h3>
<p>The calibrated midsole mesh is sampled on a 5 mm grid. Each valid ray through the mesh creates a column with rest length &#8467;<sub>0,i</sub>, tributary area A<sub>i</sub>, and four-neighbor topology. The fixture or rigid shoe carrier determines the current top position. Ground is the horizontal z = 0 plane.</p>
<div class="equation">c<sub>i</sub> = max(z<sub>free,i</sub> &minus; z<sub>i</sub>(q), 0), &nbsp; &epsilon;<sub>i</sub> = c<sub>i</sub>/&#8467;<sub>0,i</sub>, &nbsp; &lambda;<sub>i</sub> = max(1 &minus; &epsilon;<sub>i</sub>, &lambda;<sub>min</sub>)</div>
<p>Here c is compression, &epsilon; is engineering compressive strain, &lambda; is the remaining thickness stretch, and &lambda;<sub>min</sub> = 0.05 prevents collapse to zero thickness. Released columns carry no tension.</p>

<h3>1.2 Smooth nonlinear equilibrium: first-order Hyperfoam</h3>
<p>A linear spring cannot reproduce the soft initial response and rapid densification of a running-shoe foam. The equilibrium network therefore uses a smooth first-order compressible Hyperfoam term.</p>
<div class="equation">G<sub>eq</sub> = f<sub>eq</sub>G<sub>inst</sub>, &nbsp; &beta; = &nu;/(1 &minus; 2&nu;), &nbsp; J<sub>i</sub> = &lambda;<sub>i</sub><sup>(1&minus;2&nu;)</sup></div>
<div class="equation">p<sub>eq,i</sub> = [2G<sub>eq</sub>/(&alpha;&lambda;<sub>i</sub>)] [J<sub>i</sub><sup>(&minus;&alpha;&beta;)</sup> &minus; &lambda;<sub>i</sub><sup>&alpha;</sup>]</div>
<p>G<sub>inst</sub> sets the instantaneous stiffness, &alpha; controls nonlinear stiffening, f<sub>eq</sub> is the long-term-to-instantaneous modulus fraction, and the effective Poisson ratio &nu; is fixed at 0.30 because the current tests do not identify it independently.</p>

<h3>1.3 Rate dependence and hysteresis: one Maxwell memory branch</h3>
<p>Hyperfoam alone is conservative and cannot open a load-unload loop. A generalized-Maxwell overstress q stores the minimal memory needed for rate-dependent hysteresis. The recurrence integrates exponential relaxation over each timestep.</p>
<div class="equation">d = exp(&minus;&Delta;t/&tau;), &nbsp; r = &tau;(1&minus;d)/&Delta;t, &nbsp; &gamma; = (1&minus;f<sub>eq</sub>)/f<sub>eq</sub></div>
<div class="equation">q<sub>i,n</sub> = d q<sub>i,n&minus;1</sub> + &gamma;r[p<sub>eq,i,n</sub> &minus; p<sub>eq,i,n&minus;1</sub>], &nbsp; p<sub>base,i</sub> = p<sub>eq,i</sub> + q<sub>i</sub></div>
<p>The relaxation time &tau; is fixed at 0.08 s. Additional free branches were not retained because the two current single-rate tests do not identify a unique relaxation spectrum.</p>

<h3>1.4 Lateral load spreading: Pasternak coupling</h3>
<p>Independent Winkler columns localize load too strongly under the rearfoot punch and curved last. A Pasternak shear layer couples neighboring compressions while keeping the solve GPU-local.</p>
<div class="equation">&nabla;<sup>2</sup>c<sub>i</sub> &approx; [&Sigma;<sub>j&isin;N(i)</sub> c<sub>j</sub> &minus; 4c<sub>i</sub>]/h<sup>2</sup>, &nbsp; p<sub>i</sub> = max(p<sub>base,i</sub> &minus; k<sub>p</sub>&nabla;<sup>2</sup>c<sub>i</sub>, 0)</div>
<p>The boundary uses a natural zero-gradient condition. The fitted k<sub>p</sub> has units N/m, so k<sub>p</sub>&nabla;<sup>2</sup>c has pressure units. This term represents effective intact-shoe confinement and load spreading, not an intrinsic foam shear modulus.</p>

<h3>1.5 Column force, wrench, COP, power, and work</h3>
<div class="equation">f<sub>n,i</sub> = max(p<sub>i</sub>A<sub>i</sub> &minus; c<sub>n</sub>v<sub>z,i</sub>, 0), &nbsp; F = &Sigma;<sub>i</sub> f<sub>i</sub>, &nbsp; M<sub>O</sub> = &Sigma;<sub>i</sub> r<sub>i</sub> &times; f<sub>i</sub></div>
<div class="equation">COP<sub>x,y</sub> = [&Sigma;<sub>i</sub>(x<sub>i</sub>,y<sub>i</sub>)f<sub>z,i</sub>]/&Sigma;<sub>i</sub>f<sub>z,i</sub>, &nbsp; P = &Sigma;<sub>i</sub> f<sub>i</sub>&middot;v<sub>i</sub>, &nbsp; W = &int;P dt</div>
<p>Optional normal damping and friction belong to a simulation scenario; they are not part of the four-parameter Instron fit. The Virtual Instron uses no added damping or friction. The free drop uses 5 N&middot;s/m per-column normal damping for impact stability.</p>

<h3>1.6 Parameter identification and held-out test</h3>
<div class="equation">&theta; = (G<sub>inst</sub>, &alpha;, f<sub>eq</sub>, k<sub>p</sub>), &nbsp; &theta;* = arg min<sub>&theta;</sub> &Sigma;<sub>trial,t</sub> [(F&#770;<sub>trial,t</sub>(&theta;) &minus; F<sub>trial,t</sub>)/F<sub>peak,trial</sub>]<sup>2</sup></div>
<p>One shared &theta; is fitted to every sample from rearfoot and full-foot cycles 90-98. Cycles 99-100 are held out. Peak force, active-region RMSE, and dissipated loop work are reported as validation metrics rather than extra fit weights. The authoritative fit uses bounded SciPy least squares; the exact-gradient Warp path is retained for future coupled design objectives.</p>

<h3>1.7 Why this foundation model</h3>
<ul>
<li><strong>Real geometry:</strong> column thickness and engagement come from the measured shoe and fixture meshes, not a uniform slab.</li>
<li><strong>Minimal nonlinear physics:</strong> Hyperfoam captures the J-shaped compression response without piecewise stiffness regions.</li>
<li><strong>Minimal memory:</strong> one Maxwell branch opens the hysteresis loop without claiming an unidentifiable relaxation spectrum.</li>
<li><strong>Spatial transfer:</strong> Pasternak coupling corrects the strongest failure of independent Winkler columns while remaining inexpensive.</li>
<li><strong>Runtime identity:</strong> the equations fitted in NumPy are the equations executed in Warp; no surrogate replaces the calibrated law.</li>
<li><strong>Practical speed:</strong> roughly 910 columns map naturally to one GPU thread per column and support real-time rigid-body experiments.</li>
</ul>
<p>A full three-dimensional finite-element foam model was not selected because the present data lack multi-rate, relaxation, shear, and lateral-strain measurements needed to identify it, while its cost conflicts with real-time and differentiable use. Linear and Kelvin-Voigt foundations were rejected because they do not transfer the observed nonlinear force envelope and loop work across both fixtures.</p>
</div></details>
</section>"""


def _experiment_media(media_dir: str | Path | None) -> str:
    """Return embedded experiment loops or a reproducible recording instruction."""
    labels = {
        "instron": (
            "Virtual Instron",
            "Held-out compression cycle after viscoelastic warm-up. Two endpoint nodes and their connecting springs remain visible beneath the fixture; the solid midsole surface is hidden.",
        ),
        "drop": (
            "Free six-DOF body-weight drop",
            "An 80 kg body-weight load carried by the calibrated shoe last above exposed springs; this impact extrapolates beyond the fitted amplitude.",
        ),
        "rocker": ("Rigid rocker", "Controlled heel-to-toe loading shown as springs and COP travel only."),
    }
    cards = []
    missing = []
    root = Path(media_dir) if media_dir is not None else None
    for mode, (title, description) in labels.items():
        path = root / f"{mode}.gif" if root is not None else None
        if path is None or not path.is_file():
            missing.append(mode)
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        cards.append(
            f"<figure><figcaption><strong>{html.escape(title)}</strong><p>{html.escape(description)}</p></figcaption>"
            f'<img class="experiment" src="data:image/gif;base64,{encoded}" '
            f'alt="{html.escape(title)} experiment loop" loading="lazy"></figure>'
        )
    if cards:
        note = ""
        if missing:
            note = f"<p>Missing loops: {html.escape(', '.join(missing))}.</p>"
        legend = (
            '<div class="heatmap-legend"><strong>Peak column compression within each displayed frame</strong>'
            '<div class="heatmap-bar"></div><div class="heatmap-labels">'
            "<span>Blue: 0 mm</span><span>Cyan: 6.7 mm</span><span>Yellow: 13.3 mm</span>"
            "<span>Red: 20+ mm</span></div></div>"
        )
        return f'<section id="examples"><h2>3. Examples</h2><p>The same exported shoe artifact drives all three scenes without refitting.</p>{legend}<div class="experiment-grid">{"".join(cards)}</div>{note}</section>'
    return (
        '<section id="examples"><h2>3. Examples</h2><p>No recordings are embedded yet. Generate all three loops with:</p>'
        "<pre><code>uv run --extra examples -m projects.digital_shoe.record_gifs "
        "--artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json</code></pre></section>"
    )


_REPORT_CSS = """
:root {
    color-scheme: light;
    --ink: #172b40;
    --muted: #526174;
    --line: #dce3eb;
    --panel: #f5f7fa;
    --blue: #1261a0;
}
* { box-sizing: border-box; }
html { scroll-padding-top: 1.5rem; }
body {
    margin: 0 auto;
    max-width: 1200px;
    padding: 3rem 2rem;
    color: var(--ink);
    background: #fff;
    font: 16px/1.65 system-ui, sans-serif;
}
a { color: var(--blue); text-underline-offset: .2em; }
a:focus-visible, summary:focus-visible { outline: 3px solid var(--blue); outline-offset: 4px; }
h1, h2, h3, h4 { line-height: 1.25; letter-spacing: -.025em; }
h1 { max-width: 850px; margin: .6rem 0 1rem; font-size: clamp(2rem, 4vw, 3.2rem); }
h2 { margin: 0 0 1.5rem; font-size: 1.75rem; }
h3 { margin: 2rem 0 1rem; font-size: 1.15rem; }
h4 { margin: 0 0 1rem; font-size: 1.1rem; }
p { margin: .75rem 0; }
.eyebrow { color: var(--blue); font-size: .8rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
.subtitle { max-width: 780px; color: var(--muted); font-size: 1.1rem; }
.shoe-id { margin: 1rem 0; color: var(--muted); }
.report-nav { display: flex; flex-wrap: wrap; gap: .65rem 1.75rem; padding: 1.25rem 0; border-bottom: 1px solid var(--line); }
.report-nav a { font-size: .9rem; font-weight: 650; text-decoration: none; }
.report-nav a:hover { text-decoration: underline; }
section { margin-top: 3rem; }
.status-panel { margin: 1.5rem 0 .5rem; padding: 1.1rem 1.3rem; border: 1px solid var(--line); border-radius: .65rem; }
.status-panel.fail { background: #fff8f4; border-color: #edc9b8; }
.status-panel.pass { background: #f0f9f3; border-color: #c0ddca; }
.status { display: block; font-size: .85rem; font-weight: 750; letter-spacing: .015em; }
.pass { color: #176b3a; }
.fail { color: #a3331d; }
.status-panel p { margin: .5rem 0 0; color: var(--ink); font-weight: 400; font-size: .95rem; }
.section-note, .plot-caption { color: var(--muted); font-size: .9rem; }
.table-scroll { max-width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: .55rem; }
table { width: 100%; border-collapse: collapse; font-size: .9rem; font-variant-numeric: tabular-nums; }
caption { padding: .8rem 1rem; text-align: left; color: var(--muted); }
th, td { padding: .85rem 1rem; border-bottom: 1px solid var(--line); text-align: left; }
thead th { background: var(--panel); font-weight: 650; white-space: nowrap; }
tbody th { font-weight: 600; }
tbody tr:last-child > * { border-bottom: 0; }
.metrics td { white-space: nowrap; }
.metrics .pass, .metrics .fail { font-size: .8rem; font-weight: 700; }
.plot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.25rem; }
.plot-grid > figure { min-width: 0; }
article { margin: 1.5rem 0; padding: 1.25rem; border: 1px solid var(--line); border-radius: .65rem; }
figure { margin: 0; }
.response-plot { display: block; width: 100%; height: auto; }
.plot-grid figcaption { margin-bottom: .5rem; color: var(--muted); font-size: .85rem; }
.experiment-grid { display: block; }
.experiment-grid figure { max-width: 1000px; margin: 0 auto 2rem; border: 1px solid var(--line); border-radius: .65rem; overflow: hidden; }
.experiment-grid figcaption { padding: 1rem 1.25rem; background: var(--panel); }
.experiment-grid figcaption strong { display: block; font-size: 1.05rem; }
.experiment-grid figcaption p { margin: .35rem 0 0; color: var(--muted); font-size: .9rem; }
img.experiment { display: block; width: 100%; height: auto; background: #fff; }
.heatmap-legend { max-width: 1000px; margin: 1.25rem auto 2rem; font-size: .85rem; }
.heatmap-bar { height: 14px; margin: .5rem 0; border: 1px solid #64748b; border-radius: .25rem; background: linear-gradient(90deg,#0000ff 0%,#00ffff 33.3%,#ffff00 66.7%,#ff0000 100%); }
.heatmap-labels { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .35rem .75rem; color: var(--muted); }
code { font-size: .85em; overflow-wrap: anywhere; }
pre { max-width: 100%; margin: .75rem 0; padding: 1rem; overflow-x: auto; border-radius: .4rem; background: var(--panel); font-size: .9rem; line-height: 1.65; }
pre code { overflow-wrap: normal; }
.method-diagram { margin: 1.5rem 0; padding: 1rem; border: 1px solid var(--line); border-radius: .6rem; overflow-x: auto; }
.method-diagram svg { display: block; width: 100%; height: auto; min-width: 740px; margin: auto; }
#mermaid-svg .nodeLabel { font: 14px/1.5 Arial, sans-serif; }
.workflow { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; list-style: none; padding: 0; margin: 1.25rem 0; counter-reset: stage; }
.workflow li { padding: 1rem; border: 1px solid var(--line); border-radius: .5rem; counter-increment: stage; }
.workflow strong { display: block; font-size: .95rem; }
.workflow strong::before { content: counter(stage) " / "; color: var(--blue); }
.workflow span { display: block; margin-top: .3rem; color: var(--muted); font-size: .85rem; overflow-wrap: anywhere; }
.equation { margin: .8rem 0; padding: .85rem 1rem; border-left: 3px solid var(--blue); background: #f1f6fb; font: .95rem/1.7 ui-monospace, monospace; overflow-x: auto; }
details { margin: 1rem 0; border: 1px solid var(--line); border-radius: .5rem; }
summary { padding: .85rem 1rem; cursor: pointer; font-size: .95rem; font-weight: 600; }
summary:hover { background: var(--panel); }
details > pre { margin: 0 1rem 1rem; white-space: pre-wrap; overflow-wrap: anywhere; }
.details-body { padding: 0 1.25rem 1.25rem; }
.details-body > h3:first-child { margin-top: 1rem; }
.sources { margin: 0; padding-left: 1.25rem; }
.sources li { padding: .75rem 0; border-bottom: 1px solid var(--line); }
.sources li:last-child { border-bottom: 0; }
.source-role { display: block; color: var(--muted); font-size: .85rem; }
.source-hash { display: block; margin-top: .25rem; color: var(--muted); }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .85rem; }
@media (max-width: 760px) {
    body { padding: 1.5rem 1rem; }
    section { margin-top: 2rem; }
    .plot-grid { grid-template-columns: minmax(0, 1fr); }
    .workflow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    article { padding: .75rem; }
    .status-panel { padding: 1rem; }
    th, td { padding: .7rem; }
}
@media print {
    body { max-width: none; padding: 0; font-size: 11pt; }
    .report-nav { display: none; }
    section { margin-top: 1.5rem; }
    article, figure, .status-panel { break-inside: avoid; }
    .table-scroll, .method-diagram { overflow: visible; }
    .method-diagram svg { min-width: 0; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; }
}
"""


def render_html(shoe: DigitalShoe, *, media_dir: str | Path | None = None) -> str:
    """Render a deterministic, self-contained validation report with optional GIF loops."""
    curves = shoe.validation["curves"]
    passed = bool(shoe.raw["identification"]["passed_all_declared_gates"])
    curve_figures = []
    for curve in curves:
        curve_figures.append(
            f"<article><h4>{html.escape(_fixture_label(curve))}</h4>"
            '<div class="plot-grid"><figure><figcaption>Force&ndash;compression loop</figcaption>'
            f"{_svg(curve, domain='displacement')}</figure>"
            "<figure><figcaption>Force history</figcaption>"
            f"{_svg(curve, domain='time')}</figure></div></article>"
        )
    status = "ALL DECLARED GATES PASSED" if passed else "RESEARCH BASELINE — SOME DECLARED GATES FAILED"
    status_class = "pass" if passed else "fail"
    claim = html.escape(shoe.validation["claim_boundary"])
    sources = "".join(
        f"<li><code>{html.escape(item['name'])}</code>"
        f'<span class="source-role">{html.escape(item["role"])}</span>'
        f'<code class="source-hash">SHA-256: {html.escape(item["sha256"])}</code></li>'
        for item in shoe.provenance["source_files"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Digital Instron showcase — {html.escape(shoe.shoe_id)}</title>
<style>{_REPORT_CSS}</style></head><body>
<header>
<p class="eyebrow">Digital Instron / Research showcase</p>
<h1>From Instron Data to a Digital Shoe</h1>
<p class="subtitle">Identify an effective shoe model from bench measurements. Export it once. Use the same model in three Newton simulations.</p>
<p class="shoe-id">Shoe artifact · <code>{html.escape(shoe.shoe_id)}</code></p>
<div class="status-panel {status_class}" role="note" aria-label="Validation status and limitations">
<strong class="status {status_class}">{status}</strong><p>{claim}</p>
</div>
<nav class="report-nav" aria-label="Report sections">
<a href="#methods">01 Methods</a><a href="#results">02 Results</a>
<a href="#examples">03 Examples</a><a href="#reproduce">04 Reproduce</a>
</nav>
</header>
<main>
{_methods_section()}
<section id="results"><h2>2. Results</h2>
<h3>2.1 Held-out validation</h3>
<p class="section-note">Lower errors are better. Each declared error must be below 10% to pass. These held-out cycles test local repeatability, not performance under new physical conditions.</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="Held-out validation metrics">
<table class="metrics"><caption>Measured versus predicted response · held-out cycles</caption>
<thead><tr><th scope="col">Fixture</th><th scope="col">Peak error</th><th scope="col">Active RMSE</th><th scope="col">Hysteresis error</th><th scope="col">Measured peak</th><th scope="col">Overall</th></tr></thead>
<tbody>{_metric_rows(curves)}</tbody></table></div>
<h3>2.2 Response curves</h3>
<p class="plot-caption">Blue solid: measured. Orange dashed: predicted. Both curves use the same axes within each plot.</p>
{"".join(curve_figures)}
<h3>2.3 Effective model parameters</h3>
<p class="section-note">These values describe the tested shoe assembly, not isolated foam. Fixed assumptions are marked below.</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="Effective model parameters">
<table><thead><tr><th scope="col">Parameter</th><th scope="col">Value</th><th scope="col">Unit</th></tr></thead><tbody>{_material_rows(shoe)}</tbody></table></div>
</section>
{_experiment_media(media_dir)}
<section id="reproduce"><h2>4. Reproduce and provenance</h2>
<p>Run these commands from the repository root. The report and its embedded animations work offline.</p>
<h3>Rebuild this report only</h3>
<p class="section-note">Reuse the existing artifact and GIFs. No fitting or simulation is needed.</p>
<pre><code>uv run -m projects.digital_shoe.report DigitalInstron/digital_shoe_showcase/digital_shoe.json --output DigitalInstron/digital_shoe_showcase/validation_report.html --media-dir DigitalInstron/digital_shoe_showcase</code></pre>
<details><summary>Full workflow: fit, simulate, and record</summary><div class="details-body">
<h3>Identify and export</h3>
<pre><code>uv run --extra examples -m projects.digital_instron_v2.export_digital_shoe --manifest DigitalInstron/manifest_v2.json --output DigitalInstron/digital_shoe_showcase</code></pre>
<h3>Open a mechanical example</h3>
<pre><code>uv run --extra examples -m projects.digital_shoe.showcase --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json --mode instron --viewer gl
uv run --extra examples -m projects.digital_shoe.showcase --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json --mode drop --viewer gl
uv run --extra examples -m projects.digital_shoe.showcase --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json --mode rocker --viewer gl</code></pre>
<h3>Record all three animations and rebuild the report</h3>
<pre><code>uv run --extra examples -m projects.digital_shoe.record_gifs --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json</code></pre>
</div></details>
<details><summary>Source files and SHA-256 hashes</summary><div class="details-body"><ul class="sources">{sources}</ul></div></details>
</section>
</main>
<footer>Digital Instron · Effective intact-shoe model · Standalone Newton showcase</footer>
</body></html>"""


def write_report(
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    media_dir: str | Path | None = None,
) -> Path:
    """Load an artifact and write its self-contained HTML report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(load_artifact(artifact_path), media_dir=media_dir))
    return output


def main() -> None:
    """Generate one report from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation_report.html"))
    parser.add_argument("--media-dir", type=Path, help="Directory containing instron.gif, drop.gif, and rocker.gif.")
    args = parser.parse_args()
    print(write_report(args.artifact, args.output, media_dir=args.media_dir))


if __name__ == "__main__":
    main()
