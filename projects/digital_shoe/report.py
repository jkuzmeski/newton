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
    width, height = 720.0, 300.0
    left, right, top, bottom = 60.0, 20.0, 20.0, 45.0
    plot_w, plot_h = width - left - right, height - top - bottom
    dx = max(xlim[1] - xlim[0], 1.0e-12)
    dy = max(ylim[1] - ylim[0], 1.0e-12)
    px = left + (x - xlim[0]) / dx * plot_w
    py = top + (1.0 - (y - ylim[0]) / dy) * plot_h
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
    return f"""<svg viewBox="0 0 720 300" role="img" aria-label="Measured and predicted force">
<rect x="60" y="20" width="640" height="235" fill="#fff" stroke="#ccd5df"/>
<line x1="60" y1="255" x2="700" y2="255" stroke="#334155"/><line x1="60" y1="20" x2="60" y2="255" stroke="#334155"/>
<polyline points="{measured_points}" fill="none" stroke="#1261a0" stroke-width="2.5"/>
<polyline points="{predicted_points}" fill="none" stroke="#d94801" stroke-width="2.5"/>
<text x="380" y="290" text-anchor="middle">{xlabel}</text><text x="15" y="140" transform="rotate(-90 15 140)" text-anchor="middle">Force [N]</text>
<text x="60" y="275">{xlim[0]:.3g}</text><text x="700" y="275" text-anchor="end">{xlim[1]:.3g}</text>
<text x="52" y="255" text-anchor="end">0</text><text x="52" y="28" text-anchor="end">{ylim[1]:.0f}</text>
<line x1="475" y1="35" x2="505" y2="35" stroke="#1261a0" stroke-width="3"/><text x="512" y="40">measured</text>
<line x1="585" y1="35" x2="615" y2="35" stroke="#d94801" stroke-width="3"/><text x="622" y="40">predicted</text>
</svg>"""


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _metric_rows(curves: Iterable[dict]) -> str:
    rows = []
    for curve in curves:
        metric = curve["metrics"]
        passed = bool(metric["passed"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(curve['name'])}</td>"
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
        "pasternak_n_per_m": ("Pasternak coupling", "N/m"),
        "effective_poisson_ratio": ("Effective Poisson ratio (fixed)", "1"),
        "maxwell_relaxation_time_s": ("Maxwell relaxation time (fixed)", "s"),
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
<p>The Digital Shoe is an <strong>effective intact-shoe model</strong>. It identifies one compact law from measured force-displacement cycles, bakes the measured geometry into a column bed, and deploys exactly that law in the runtime artifact. The fitted values therefore describe the tested shoe assembly—not isolated foam chemistry.</p>
<div class="method-diagram">{diagram}</div>
<details><summary>Mermaid source for the method diagram</summary><pre><code class="language-mermaid">{mermaid_source}</code></pre></details>

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
<p>Hyperfoam alone is conservative and cannot open a load-unload loop. A generalized-Maxwell overstress q stores the minimal memory needed for rate-dependent hysteresis. Its exact discrete update avoids timestep-dependent numerical damping.</p>
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
</section>"""


def _experiment_media(media_dir: str | Path | None) -> str:
    """Return embedded experiment loops or a reproducible recording instruction."""
    labels = {
        "instron": ("Virtual Instron", "Held-out compression cycle after viscoelastic warm-up."),
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
            f'<figure><img class="experiment" src="data:image/gif;base64,{encoded}" '
            f'alt="{html.escape(title)} experiment loop">'
            f"<figcaption><strong>{html.escape(title)}</strong><br>{html.escape(description)}</figcaption></figure>"
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
        "<section><h2>Mechanical experiment loops</h2><p>Generate and embed all three audited loops with:</p>"
        "<pre><code>uv run --extra examples -m projects.digital_shoe.record_gifs "
        "--artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json</code></pre></section>"
    )


def render_html(shoe: DigitalShoe, *, media_dir: str | Path | None = None) -> str:
    """Render a deterministic, self-contained validation report with optional GIF loops."""
    curves = shoe.validation["curves"]
    passed = bool(shoe.raw["identification"]["passed_all_declared_gates"])
    curve_figures = []
    for curve in curves:
        curve_figures.append(
            f"<article><h4>{html.escape(curve['name'])}: held-out cycles</h4>"
            '<div class="plot-grid"><figure>'
            f"{_svg(curve, domain='displacement')}<figcaption>Force-compression loop</figcaption></figure>"
            f"<figure>{_svg(curve, domain='time')}<figcaption>Force history</figcaption></figure></div></article>"
        )
    status = "ALL DECLARED GATES PASSED" if passed else "RESEARCH BASELINE — SOME DECLARED GATES FAILED"
    status_class = "pass" if passed else "fail"
    claim = html.escape(shoe.validation["claim_boundary"])
    sources = "".join(
        f"<li><code>{html.escape(item['name'])}</code> — {html.escape(item['role'])} — <code>{item['sha256']}</code></li>"
        for item in shoe.provenance["source_files"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Digital Shoe validation — {html.escape(shoe.shoe_id)}</title>
<style>
:root{{--ink:#172033;--muted:#5f6b7a;--panel:#f5f7fa;--blue:#1261a0;--orange:#d94801;--green:#137333;--red:#a61b1b}}body{{font:16px/1.5 system-ui,sans-serif;color:var(--ink);max-width:1200px;margin:auto;padding:2rem}}h1{{font-size:2.3rem;margin-bottom:.2rem}}h2{{margin-top:2.2rem}}.subtitle{{color:var(--muted);font-size:1.15rem}}.status{{display:inline-block;padding:.35rem .65rem;border-radius:.3rem;font-weight:700}}.pass{{color:var(--green);font-weight:700}}.fail{{color:var(--red);font-weight:700}}.status.pass{{background:#dff3e4}}.status.fail{{background:#fbe1e1}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin:1.5rem 0}}.card{{background:var(--panel);padding:1rem;border-radius:.5rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.65rem;border-bottom:1px solid #d7dee8;text-align:left}}th{{background:var(--panel)}}.plot-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:1rem}}.experiment-grid{{display:block}}.experiment-grid figure{{max-width:1000px;margin:0 auto 2.5rem}}.heatmap-legend{{max-width:1000px;margin:0 auto 1.5rem}}.heatmap-bar{{height:24px;margin-top:.5rem;border:1px solid #64748b;border-radius:.25rem;background:linear-gradient(90deg,#0000ff 0%,#00ffff 33.3%,#ffff00 66.7%,#ff0000 100%)}}.heatmap-labels{{display:flex;justify-content:space-between;gap:.5rem;font-size:.85rem;color:var(--muted)}}figure{{margin:0}}svg{{width:100%;height:auto;background:white}}img.experiment{{display:block;width:100%;height:auto;border:1px solid #ccd5df;border-radius:.4rem;background:#fff}}figcaption{{text-align:center;color:var(--muted);padding-top:.4rem}}code{{font-size:.85em;overflow-wrap:anywhere}}.boundary{{border-left:5px solid var(--orange);padding:1rem;background:#fff5ed}}.method-diagram{{margin:1.5rem 0;padding:1rem;border:1px solid #d7dee8;border-radius:.5rem;background:#fff;overflow-x:auto}}.method-diagram svg{{display:block;width:100%;height:auto;min-width:900px}}.equation{{margin:.8rem 0;padding:.85rem 1rem;border-left:4px solid var(--blue);background:#eef6ff;font:1.02rem/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-x:auto}}details{{margin:1rem 0}}details pre{{white-space:pre-wrap;background:var(--panel);padding:1rem;border-radius:.4rem}}article{{margin:1.5rem 0 2.5rem}}
</style></head><body>
<header><h1>From Instron Data to a Digital Shoe</h1><p class="subtitle">Portable effective shoe dynamics for <code>{html.escape(shoe.shoe_id)}</code></p></header>
{_methods_section()}
<section id="results"><h2>2. Results</h2><p><span class="status {status_class}">{status}</span></p>
<h3>2.1 Held-out validation summary</h3><table><thead><tr><th>Trial</th><th>Peak error</th><th>Active RMSE</th><th>Hysteresis error</th><th>Measured peak</th><th>Declared 10% gates</th></tr></thead><tbody>{_metric_rows(curves)}</tbody></table>
<h3>2.2 Held-out response curves</h3>{"".join(curve_figures)}
<h3>2.3 Identified effective model</h3><table><thead><tr><th>Parameter</th><th>Value</th><th>Unit</th></tr></thead><tbody>{_material_rows(shoe)}</tbody></table><p>The parameters describe the intact tested shoe system. They include geometry, outsole, plate, bonding, confinement, and foam response.</p>
<h3>2.4 Claim boundary</h3><p class="boundary">{claim}</p></section>
{_experiment_media(media_dir)}
<section id="reproduce"><h2>4. Reproduce and provenance</h2><h3>4.1 Commands</h3><pre><code>uv run -m projects.digital_instron_v2.export_digital_shoe --manifest DigitalInstron/manifest_v2.json --output DigitalInstron/digital_shoe_showcase
uv run --extra examples -m projects.digital_shoe.showcase --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json --mode instron --viewer gl
uv run --extra examples -m projects.digital_shoe.showcase --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json --mode drop --viewer gl
uv run --extra examples -m projects.digital_shoe.showcase --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json --mode rocker --viewer gl
uv run --extra examples -m projects.digital_shoe.record_gifs --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json</code></pre>
<h3>4.2 Source integrity</h3><ul>{sources}</ul></section>
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
