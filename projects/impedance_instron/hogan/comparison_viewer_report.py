# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Package saved replay choices and synchronized shoe overlays."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def condition_label(condition: str) -> str:
    """Name a shoe condition without hiding its material or geometry."""
    materials = {
        "original": "Current material",
        "original_elastic": "Current material (elastic control)",
        "ff_leap": "FF LEAP",
        "ff_turbo_plus": "FF TURBO PLUS",
    }
    geometries = {"reference": "original geometry", "long125": "+25% spring length", "block40": "40 mm block"}
    parts = condition.split("__")
    if parts[0] in materials:
        return materials[parts[0]] + " · " + geometries.get(parts[-1], "original geometry")
    return condition.replace("/", " · ").replace("_", " ")


def build_comparison_viewer(run: Path, default: str | None = None) -> Path:
    """Write a portable selector for the run's qualified saved replay files."""
    run = run.resolve()
    outputs = next(parent.parent for parent in run.parents if parent.name == "impedance_instron")
    choices = []
    for evidence in sorted(run.glob("**/replay_evidence/spring-geometry.json")):
        condition_dir = evidence.parent.parent
        validation = json.loads((evidence.parent / "validation.json").read_text())
        if not validation.get("passed"):
            continue
        condition = condition_dir.relative_to(run).as_posix()
        choices.append(
            {
                "id": condition,
                "label": condition_label(condition),
                "href": (evidence.parent / "viewer.html").relative_to(outputs).as_posix(),
                "sample_id": validation["sample_id"],
                "split": validation.get("split"),
            }
        )
    if not choices:
        raise ValueError(f"No qualified spring replay choices under {run}")
    config_path = run / "comparison-viewer.json"
    previous = json.loads(config_path.read_text()) if config_path.exists() else {}
    default = default or previous.get("default") or choices[0]["id"]
    if default not in {choice["id"] for choice in choices}:
        raise ValueError(f"Default replay is not available: {default}")
    # Overlay only matching saved samples; unrelated examples cannot be synchronized.
    config = {"schema": "saved-shoe-comparison-v1", "run_id": run.name, "default": default, "choices": choices}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    payload = json.dumps(config).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shoe motion comparison</title><style>
*{box-sizing:border-box}body{font:16px/1.45 system-ui,sans-serif;color:#18334a;margin:0;background:#fff;overflow-wrap:anywhere}.choices{position:sticky;top:0;z-index:2;background:#f4f9fa;border-bottom:1px solid #ccdce0;padding:12px;display:flex;gap:12px;flex-wrap:wrap;align-items:end}label{display:flex;flex-direction:column;gap:4px;max-width:100%;font-weight:600}select,button{font:inherit;padding:8px;min-height:44px;max-width:100%;border:1px solid #9aafb7;border-radius:7px;background:white}.toggle{flex-direction:row;align-items:center;min-height:44px}.toggle input{width:20px;height:20px}.note{font-size:13px;padding:0 12px;color:#526979}#primary{display:block;width:100%;height:1800px;border:0}#secondary{display:none}#loading{padding:0 12px;color:#526979}@media(max-width:500px){.choices>label:not(.toggle){width:100%}select{width:100%}}
</style></head><body><div class="choices"><label>Shoe / run<select id="shoe" aria-label="Select shoe or run"></select></label><label class="toggle"><input id="overlay" type="checkbox">Overlay comparison</label><label>Compare with<select id="comparison" aria-label="Select overlay shoe" disabled></select></label></div><p class="note">Primary: compression colors. Overlay: purple dashed. Motion is compared at matching elapsed time from the same saved example; the spring close-ups are centered on each shoe. Drag the springs to orbit while paused or playing.</p><p id="loading" role="status">Loading saved replay…</p><iframe id="primary" title="Selected shoe motion and orbitable springs"></iframe><iframe id="secondary" title="Comparison data" aria-hidden="true" tabindex="-1"></iframe><script>
const config=CONFIG,shoe=document.getElementById('shoe'),comparison=document.getElementById('comparison'),toggle=document.getElementById('overlay'),primary=document.getElementById('primary'),secondary=document.getElementById('secondary'),status=document.getElementById('loading');let overlayPayload=null;
const pathname=location.pathname,canonical=pathname.lastIndexOf('/reports/impedance-instron/'),native=pathname.lastIndexOf('/impedance_instron/');const root=new URL(pathname.slice(0,(canonical>=0?canonical:native)+1),location.href);
for(const choice of config.choices){for(const select of [shoe,comparison]){const option=document.createElement('option');option.value=choice.id;option.textContent=choice.label;select.append(option)}}
const requested=new URLSearchParams(location.search).get('shoe');shoe.value=config.choices.some(c=>c.id===requested)?requested:config.default;
const choice=id=>config.choices.find(c=>c.id===id),send=(iframe,type,payload)=>iframe.contentWindow?.postMessage({type,payload},'*');
function updateSecondary(){overlayPayload=null;send(primary,'replay:overlay',null);comparison.disabled=!toggle.checked;secondary.removeAttribute('src');if(!toggle.checked)return;const a=choice(shoe.value),b=choice(comparison.value);if(a.sample_id!==b.sample_id||a.split!==b.split){status.textContent='These runs use different saved examples; choose a matching example to overlay.';return}status.textContent='Loading comparison…';secondary.src=new URL(b.href,root).href}
function loadPrimary(){status.textContent='Loading saved replay…';primary.src=new URL(choice(shoe.value).href,root).href;if(comparison.value===shoe.value||!comparison.value)comparison.value=config.choices.find(c=>c.id!==shoe.value)?.id||shoe.value;updateSecondary()}
shoe.onchange=loadPrimary;comparison.onchange=updateSecondary;toggle.onchange=updateSecondary;
primary.addEventListener('load',()=>{send(primary,'replay:overlay',toggle.checked?overlayPayload:null)});
secondary.addEventListener('load',()=>{if(toggle.checked)send(secondary,'replay:request')});
window.addEventListener('message',event=>{const message=event.data||{};if(event.source===secondary.contentWindow){if(message.type==='replay:ready'&&toggle.checked)send(secondary,'replay:request');if(message.type==='replay:data'&&toggle.checked&&message.payload?.replay_path===choice(comparison.value).href){overlayPayload=message.payload;overlayPayload.label=choice(comparison.value).label;send(primary,'replay:overlay',overlayPayload);status.textContent='Saved comparison ready.'}}else if(event.source===primary.contentWindow&&message.type==='replay:ready'){if(Number.isFinite(message.height))primary.style.height=Math.max(500,message.height+24)+'px';status.textContent='Saved replay ready.';send(primary,'replay:overlay',toggle.checked?overlayPayload:null)}});
loadPrimary();
</script></body></html>""".replace("CONFIG", payload)
    page = page.replace(
        "<title>Shoe motion comparison</title>", f"<title>{html.escape(run.name)} — shoe motion comparison</title>"
    )
    target = run / "comparison-viewer.html"
    target.write_text(page)
    return target


def main() -> None:
    """Package the available saved replays without simulating physics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--default")
    args = parser.parse_args()
    print(build_comparison_viewer(args.run, args.default))


if __name__ == "__main__":
    main()
