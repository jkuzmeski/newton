# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply the shared Instron phone/desktop presentation to saved generative reports."""

from __future__ import annotations

import base64
import html
import os
import re
from pathlib import Path

ASSETS = Path(__file__).with_name("report_assets")


def format_report(run: Path) -> Path:
    """Format saved report HTML without re-evaluating a controller.

    Args:
        run: Directory containing the native generative fit report and GIFs.

    Returns:
        Path of the self-contained formatted report.
    """
    path = run / "report.html"
    document = path.read_text(encoding="utf-8")
    if 'data-generative-layout="shared"' in document:
        return path
    sections = re.findall(r'<section id="([^"]+)">(.*?)</section>', document, flags=re.DOTALL)
    if not sections:
        raise ValueError(f"No generative report sections found in {path}")
    title = re.search(r"<h1>(.*?)</h1>", document, flags=re.DOTALL)
    subtitle = re.search(r'<p class="subtitle">(.*?)</p>', document, flags=re.DOTALL)
    status = re.search(r'<div class="status">(.*?)</div>', document, flags=re.DOTALL)
    gifs = []
    content = []
    for identity, raw_body in sections:
        body = raw_body
        heading = re.match(r"\s*<h2>(.*?)</h2>", body, flags=re.DOTALL)
        name = heading.group(1) if heading else identity
        if heading:
            body = body[heading.end() :]
        if identity == "motion":
            gif_pattern = r'<figure\b[^>]*class="[^"]*motion-gif[^"]*"[^>]*>.*?</figure>'
            gifs.extend(re.findall(gif_pattern, body, flags=re.DOTALL))
            body = re.sub(gif_pattern, "", body, flags=re.DOTALL)
        if identity == "method":
            convergence = re.search(r'<div class="grid">.*?</div>', body, flags=re.DOTALL)
            if convergence:
                content.append(
                    f'<section class="section report-section" data-kind="training" id="training"><h2>Fitting progress</h2><div class="html-content">{convergence.group()}</div></section>'
                )
                body = body[: convergence.start()] + body[convergence.end() :]
        kind = "biomechanics" if identity in ("results", "motion", "impedance") else "details"
        content.append(
            f'<section class="section report-section" data-kind="{kind}" id="{identity}"><h2>{name}</h2><div class="html-content">{body}</div></section>'
        )
    if not gifs:
        raise ValueError(f"Motion GIFs are missing from {path}; rebuild them from saved traces first")

    def embed_gif(match):
        source = (run / html.unescape(match.group(1))).resolve()
        if not source.is_relative_to(run.resolve()) or source.suffix.lower() != ".gif":
            raise ValueError(f"Unexpected motion GIF source: {source}")
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        return f'src="data:image/gif;base64,{encoded}"'

    motion = re.sub(r'src="([^"]+\.gif)"', embed_gif, "".join(gifs))
    css = (ASSETS / "report.css").read_text()
    css += "\nh2,h3{overflow-wrap:anywhere}.motion-gif{margin:12px 0}.motion-gif img{width:100%;max-width:720px;height:auto;margin:auto}.motion-gif figcaption{overflow-wrap:anywhere}.html-content div.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px}.motion-heading{flex-wrap:wrap}\n"
    script = (ASSETS / "report.js").read_text()
    output_root = next((p for p in run.resolve().parents if p.name == "outputs"), None)
    catalog = os.path.relpath(output_root / "index.html", run) if output_root else "#"
    heading = title.group(1) if title else "Generative runner fit"
    description = subtitle.group(1) if subtitle else "Saved generative fit results."
    summary = status.group(1) if status else "Diagnostic fit; physiological validity unqualified."
    tabs = '<button type="button" role="tab" data-reading-view="motion" aria-selected="true">Motion</button><button type="button" role="tab" data-reading-view="graphs" aria-selected="false">Biomech</button><button type="button" role="tab" data-reading-view="training" aria-selected="false">Training</button><button type="button" role="tab" data-reading-view="details" aria-selected="false">Details</button>'
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Generative runner fit</title><style>{css}</style></head>
<body data-generative-layout="shared" data-view="auto" data-initial-reading="motion"><header><div class="eyebrow">Impedance Instron · generative runner</div><h1>{heading}</h1><nav class="mode-nav" aria-label="Presentation mode"><button data-view-choice="auto" aria-pressed="true">Auto</button><button data-view-choice="phone" aria-pressed="false">Phone</button><button data-view-choice="desktop" aria-pressed="false">Desktop</button></nav><a href="{html.escape(catalog)}">← All experiment reports</a></header>
<main><nav class="reading-nav" aria-label="Report sections"><div class="reading-tabs" role="tablist">{tabs}</div><div class="detail-jump"><label for="section-jump">Go to section</label><select id="section-jump" aria-label="Go to details section"></select></div></nav><section class="summary"><p>{description}</p>{summary}</section><section class="motion-viewer-section report-section" id="motion-gifs"><div class="motion-heading"><h2>Saved simulated motion</h2></div>{motion}<p>Looping GIFs show the saved fitted rollout with its measured reference. Playback is slowed for inspection.</p></section>
<section class="graph-gallery" aria-label="Report graphs" hidden><div class="gallery-controls"><button type="button" data-chart-prev>&lsaquo; Previous</button><label for="chart-select">Graph</label><select id="chart-select" aria-label="Choose graph"></select><button type="button" data-chart-next>Next &rsaquo;</button></div><div class="gallery-frame"></div><div class="gallery-caption"><span data-chart-count></span><button type="button" data-chart-expand>Expand graph</button></div></section><dialog class="chart-dialog" aria-label="Expanded graph"><button type="button" data-chart-close class="dialog-close">Close</button><div class="dialog-frame"></div></dialog><div class="content-grid">{"".join(content)}</div><footer>Formatted from saved report data and simulated traces; no fitting or dynamics rerun.</footer></main><script>{script}</script></body></html>'''
    path.write_text(document, encoding="utf-8")
    return path
