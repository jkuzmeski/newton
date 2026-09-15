# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the fixed two-term mechanics narrative with six offline figures.

Run ``uv run --no-sync -m projects.digital_shoe.mechanics_report --help``.
The optional markdown-it-py renderer is imported only for actual rendering.
This command does not install dependencies, refit, or change source artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
from pathlib import Path

from ._provenance import DEFAULT_ARTIFACT, DEFAULT_OUTPUT, MANIFEST, ROOT, load_verified_artifact

SOURCE = Path(__file__).with_name("REPORT.md")
FIGURE_NAMES = (
    "bed_geometry",
    "material_equilibrium",
    "material_rate_dependence",
    "contact_bristle",
    "validation_rearfoot",
    "validation_fullfoot",
)


def render_report(
    artifact_path: Path,
    output_path: Path,
    *,
    source_path: Path = SOURCE,
    manifest_path: Path = MANIFEST,
    root: Path = ROOT,
) -> Path:
    """Verify provenance and embed six local figures in the fixed narrative.

    Args:
        artifact_path: Explicit two-term artifact covered by the frozen manifest.
        output_path: Destination HTML file; figures must be in its sibling ``figures/`` directory.
        source_path: Audited Markdown narrative to render.
        manifest_path: Frozen artifact and physics source checksums.
        root: Checkout root for source verification.

    Returns:
        Written HTML path. The supplied artifact and source files remain unchanged.
    """
    load_verified_artifact(artifact_path, manifest_path=manifest_path, root=root)
    from markdown_it import MarkdownIt  # noqa: PLC0415

    output_path = Path(output_path).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    artifact_resolved = Path(artifact_path)
    if not artifact_resolved.is_absolute():
        artifact_resolved = root / artifact_resolved
    protected = [artifact_resolved, source_path, manifest_path, *(root / path for path in manifest["files"])]
    if output_path in {Path(path).resolve() for path in protected}:
        raise ValueError("Report output must not overwrite an input or audited source")
    text = Path(source_path).read_text()
    if "{{" in text:
        raise ValueError("Unresolved report draft placeholder")
    rendered = MarkdownIt("commonmark", {"html": True}).enable("table").render(text)
    headings = []

    def heading(match: re.Match) -> str:
        level, title = match.group(1), match.group(2)
        plain = re.sub(r"<[^>]+>", "", title)
        anchor = "section-" + str(len(headings) + 1)
        if level == "2":
            headings.append((anchor, plain))
            return f'<h{level} id="{anchor}">{title}</h{level}>'
        return match.group(0)

    rendered = re.sub(r"<h([2])>(.*?)</h\1>", heading, rendered)
    figure_names = []
    metadata = json.loads((output_path.parent / "figures" / "metadata.json").read_text())
    if metadata["artifact_sha256"] != manifest["artifact_sha256"]:
        raise ValueError("Figure artifact differs from the audited two-term artifact")
    for source, expected in metadata["source_sha256"].items():
        source_path_actual = Path(source) if Path(source).is_absolute() else root / source
        if hashlib.sha256(source_path_actual.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Figure source changed: {source}; regenerate the figures")

    def image(match: re.Match) -> str:
        relative = match.group(1)
        name = Path(relative).stem
        if name not in FIGURE_NAMES or Path(relative).suffix != ".svg":
            raise ValueError(f"Unexpected report figure: {relative}")
        path = output_path.parent / "figures" / f"{name}.svg"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata["figure_sha256"][path.name]:
            raise ValueError(f"Figure changed: {path.name}; regenerate the figures")
        payload = base64.b64encode(raw).decode("ascii")
        figure_names.append(name)
        return f'src="data:image/svg+xml;base64,{payload}"'

    rendered = re.sub(r'src="([^"]+)"', image, rendered)
    if len(figure_names) != 6 or set(figure_names) != set(FIGURE_NAMES):
        raise ValueError("Expected exactly the six named two-term report figures")

    def local_link(match: re.Match) -> str:
        target = (Path(source_path).parent / match.group(1)).resolve()
        relative = Path(os.path.relpath(target, output_path.parent)).as_posix()
        return f'href="{html.escape(relative, quote=True)}"'

    rendered = re.sub(r'href="(\.\./[^"]+)"', local_link, rendered)
    navigation = "".join(f'<a href="#{anchor}">{html.escape(title)}</a>' for anchor, title in headings)
    style = """
:root {color-scheme:light; --ink:#182b3d; --blue:#146c94; --muted:#546474;}
* {box-sizing:border-box} body {margin:0;background:#edf2f6;color:var(--ink);font:17px/1.62 system-ui,-apple-system,Segoe UI,sans-serif}
header {background:#142f46;color:white;padding:24px 36px;font-size:15px;letter-spacing:.035em}
nav {position:fixed;left:0;top:75px;bottom:0;width:270px;overflow:auto;padding:20px 17px;background:#f6f8fa;border-right:1px solid #d2dce5}
nav a {display:block;color:#264d65;text-decoration:none;font-size:13px;line-height:1.4;padding:7px 5px;border-bottom:1px solid #e4e9ed}
main {max-width:1140px;margin:30px 28px 55px 300px;background:white;padding:44px 55px;box-shadow:0 5px 20px #172b3d12}
h1 {font-size:38px;line-height:1.2;color:#113e5b;margin:0 0 22px} h2 {font-size:27px;line-height:1.3;color:#164f70;margin-top:55px;border-bottom:2px solid #d9e7ef;padding-bottom:12px;scroll-margin-top:20px}
h3 {font-size:21px;color:#23556d;margin-top:30px} p {margin:15px 0} a {color:#096ba0}
strong {font-weight:700} table {width:100%;border-collapse:collapse;font-size:14px;line-height:1.45;margin:24px 0;display:table}
th {text-align:left;background:#e7f0f6;padding:11px 12px;border-bottom:2px solid #aac4d5} td {padding:10px 12px;border-bottom:1px solid #dfe6ec;vertical-align:top}
tr:nth-child(even) td {background:#f6f9fb} td code {overflow-wrap:anywhere}
pre {overflow:auto;border-left:4px solid #2590b6;background:#edf5f9;padding:18px 20px;font-size:13px;line-height:1.65;border-radius:0 5px 5px 0}
code {font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.87em} p code,li code {background:#edf3f7;padding:2px 4px;overflow-wrap:anywhere}
pre code {font-size:1em;background:none} img {display:block;width:100%;height:auto;margin:28px auto 10px} li {margin:9px 0} blockquote {border-left:4px solid #d4a848;padding:10px 20px;background:#fff9e9;margin:20px 0}
@media(min-width:1550px) {main {margin-left:auto;margin-right:auto;transform:translateX(110px)}}
@media(max-width:1000px) {nav {position:static;width:auto;max-height:260px} main {margin:15px;padding:25px} h1 {font-size:30px} table {font-size:12px} pre {font-size:12px}}
@media print {@page {size:A4;margin:15mm} body {background:white;font-size:10pt;line-height:1.5} header,nav {display:none} main {margin:0;padding:0;max-width:none;box-shadow:none;transform:none} h1 {font-size:25pt} h2 {font-size:17pt;break-after:avoid;margin-top:26px} h3 {font-size:13pt;break-after:avoid} table {font-size:8.5pt} tr,img,pre {break-inside:avoid} pre {white-space:pre-wrap;overflow-wrap:anywhere;font-size:8pt} a {color:inherit;text-decoration:none} img {max-height:220mm;object-fit:contain}}
"""
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Two-term footwear simulation — contact and material laws</title>"
        f"<style>{style}</style></head><body>"
        "<header>DIGITAL SHOE · DIGITAL INSTRON · TWO-TERM MECHANICS REPORT</header>"
        f'<nav aria-label="Contents">{navigation}</nav><main>{rendered}</main></body></html>'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    return output_path


def main(argv: list[str] | None = None) -> None:
    """Render the report beside previously generated local figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "report.html")
    args = parser.parse_args(argv)
    output = render_report(args.artifact, args.output)
    print(f"Wrote {output} with six embedded figures")


if __name__ == "__main__":
    main()
