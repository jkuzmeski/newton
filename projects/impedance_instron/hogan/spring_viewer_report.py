# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Add spring-column geometry from saved poses to offline replay viewers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .mechanics import Chain


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _color(value: float, maximum: float) -> str:
    v = max(0.0, min(1.0, value / maximum))
    if v < 1 / 3:
        r, g, b = 0, round(v * 765), 255
    elif v < 2 / 3:
        r, g, b = round((v - 1 / 3) * 765), 255, round((2 / 3 - v) * 765)
    else:
        r, g, b = 255, round((1 - v) * 765), 0
    return f"rgb({r},{g},{b})"


def update_viewer(condition_dir: str | Path) -> Path:
    """Upgrade one qualified saved replay with captured, time-aligned columns."""
    condition_dir = Path(condition_dir)
    evidence = condition_dir / "replay_evidence"
    viewer = evidence / "viewer.html"
    deformation_path = evidence / "deformation.npz"
    validation_path = evidence / "validation.json"
    validation = json.loads(validation_path.read_text())
    if not validation.get("passed"):
        raise ValueError(f"Replay validation did not pass: {condition_dir}")

    # Bind the visualizer to exactly the capture that earned validation.
    run_root = condition_dir
    trial_rows = json.loads((run_root / "summary.json").read_text())["trials"]
    matches = [row for row in trial_rows if row["id"] == validation["sample_id"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one validated trial in {condition_dir}")
    trial_index = trial_rows.index(matches[0])
    if _sha256(run_root / f"trace_{trial_index:03d}.npz") != validation["trace_sha256"]:
        raise ValueError("Validated native trace hash changed")
    if _sha256(run_root / f"scenario_{trial_index:03d}.json") != validation["scenario_sha256"]:
        raise ValueError("Validated scenario hash changed")
    scenario = json.loads((run_root / f"scenario_{trial_index:03d}.json").read_text())
    shoe_path = Path(scenario["shoe"]["artifact"])
    if not shoe_path.is_absolute():
        shoe_path = condition_dir / shoe_path
    shoe_path = shoe_path.resolve()
    if _sha256(shoe_path) != validation["shoe_sha256"]:
        raise ValueError("Validated shoe hash changed")

    with np.load(deformation_path) as archive:
        times = archive["time_s"]
        frame_indices = archive["frame_index"]
        bottoms = archive["bottom_m"].astype(np.float64)
        tops = archive["top_m"].astype(np.float64)
        compression = archive["compression_m"]
        rest = archive["rest_length_m"]
        driven = archive["driven"]
        anchors = archive["anchor_local_m"]
    if bottoms.shape != tops.shape or bottoms.shape[0] != len(frame_indices):
        raise ValueError("Saved endpoint arrays are not frame aligned")
    backup = evidence / "viewer.before-springs.txt"
    text = (backup if backup.exists() else viewer).read_text()
    marker = "const d="
    if marker not in text:
        raise ValueError("Unknown replay viewer format")
    start = text.index(marker) + len(marker)
    end = text.index(";let frame=", start)
    payload = json.loads(text[start:end])
    if len(payload["times"]) != len(frame_indices):
        raise ValueError("Viewer pose and saved spring frame counts differ")
    if not np.allclose(payload["times"], times[frame_indices], atol=1e-9, rtol=0):
        raise ValueError("Viewer pose times differ from captured spring times")
    if len(payload["anchors"]) != len(rest):
        raise ValueError("Viewer column count differs from capture")
    if not np.allclose(np.asarray(payload["anchors"]), anchors[:, :2] * 1000, atol=0.002):
        raise ValueError("Viewer anchors differ from saved shoe geometry")
    if np.asarray(payload["compression"]).shape != compression[frame_indices].shape:
        raise ValueError("Viewer heatmap frame/column dimensions differ from capture")

    shoe_data = json.loads(shoe_path.read_text())
    bed = shoe_data["column_bed"]
    expected_anchors = np.asarray(bed["anchor_bottom_m"]) - np.asarray(scenario["shoe"]["mount_m"])
    np.testing.assert_allclose(anchors, expected_anchors, atol=1e-7, rtol=0)
    np.testing.assert_allclose(rest, bed["rest_length_m"], atol=1e-7, rtol=0)
    chain = Chain(**scenario["chain"])
    with np.load(run_root / f"trace_{trial_index:03d}.npz") as trace:
        states = trace["state"][frame_indices]
        np.testing.assert_allclose(times[frame_indices], trace["time_s"][frame_indices], atol=1e-9, rtol=0)
    captured_compression = np.asarray(payload["compression"]) / 1000
    # Old CPU captures retained views of reused buffers, repeating the last frame.
    # Reconstruct only geometry, following carried_column_segment, without contact integration.
    rebuilt_bottoms, rebuilt_tops = [], []
    for state, comp in zip(states, captured_compression, strict=True):
        ankle, _, _ = chain.point(state, 3, np.zeros(2))
        angle = chain.angle(state, 3) - scenario["shoe"]["pitch_rad"]
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        translation = np.array([ankle[0], 0, ankle[1]])
        bottom = anchors @ rotation.T + translation
        top_local = anchors.copy()
        top_local[:, 2] += rest
        top = top_local @ rotation.T + translation
        penetration = np.maximum(-bottom[:, 2], 0)
        retreat = np.where(driven, 0, penetration - np.clip(comp, 0, penetration))
        bottom[:, 2] = np.maximum(bottom[:, 2] + retreat, 0)
        top[:, 2] = np.maximum(top[:, 2] + retreat, 0)
        rebuilt_bottoms.append(bottom)
        rebuilt_tops.append(top)
    bottoms, tops = np.asarray(rebuilt_bottoms), np.asarray(rebuilt_tops)
    np.savez_compressed(
        evidence / "spring-geometry.npz",
        bottom_m=bottoms,
        top_m=tops,
        time_s=times[frame_indices],
        frame_index=frame_indices,
        rest_length_m=rest,
        compression_m=captured_compression,
        driven=driven,
    )
    geometry_validation = {
        "method": "Geometry-only reconstruction from saved native pose, artifact anchors/rest lengths, and captured per-frame compression; carried_column_segment convention",
        "physics_rerun": False,
        "known_capture_issue": "Original CPU deformation arrays alias reused buffers and repeat the last frame; original files retained",
        "compression_quantization_m": 1e-6,
        "display_endpoint_quantization_m": 1e-4,
        "trace_sha256": validation["trace_sha256"],
        "shoe_sha256": validation["shoe_sha256"],
        "scenario_sha256": validation["scenario_sha256"],
        "original_viewer_sha256": _sha256(backup if backup.exists() else viewer),
        "geometry_sha256": _sha256(evidence / "spring-geometry.npz"),
        "frame_count": len(frame_indices),
        "column_count": len(rest),
    }
    (evidence / "spring-geometry.json").write_text(json.dumps(geometry_validation, indent=2) + "\n")

    # Keep all three world axes relative to the foot-following frame center.
    # 0.1 mm quantization preserves more precision than the saved compression.
    origin = np.mean(np.concatenate((bottoms, tops), axis=1), axis=1, keepdims=True)
    b = np.rint((bottoms - origin) * 10000).astype(np.int16)
    t = np.rint((tops - origin) * 10000).astype(np.int16)
    payload["springs"] = {
        "bottom": b.reshape(len(b), -1).tolist(),
        "top": t.reshape(len(t), -1).tolist(),
        "rest_mm": np.round(rest * 1000, 2).tolist(),
        "driven": driven.astype(int).tolist(),
        "scale": 10000,
    }
    payload["heat_max"] = 40 if "red 40 mm" in text else 60
    outputs = next(parent.parent for parent in condition_dir.resolve().parents if parent.name == "impedance_instron")
    payload["replay_path"] = viewer.resolve().relative_to(outputs).as_posix()
    # Keep the original qualified viewer for audit and allow safe repeat runs.
    if not backup.exists():
        backup.write_text(text)

    text = text[:start] + json.dumps(payload, separators=(",", ":")) + text[end:]
    text = text.replace("const d=", "let d=", 1)
    insertion = (
        "</canvas><h3>Captured spring-column close-up</h3>"
        f"<p>Orbitable 3-D view of all {len(rest)} captured columns. Drag to rotate, use the presets and zoom controls, "
        "or use arrow keys while focused on the viewer. Solid columns are driven; dashed columns are passive. "
        "Compression colors use the same fixed scale as the plan view. Rest lengths: "
        f"{rest.min() * 1000:.1f}-{rest.max() * 1000:.1f} mm. Endpoints reconstructed from saved poses and shoe geometry; passive retreat uses captured compression rounded to 0.001 mm. No physics rerun.</p>"
        '<div class="spring-controls"><button data-view="side">Side</button><button data-view="top">Top</button><button data-view="end">End</button><button id="spring-zoom-out" aria-label="Zoom out">&#8722;</button><button id="spring-zoom-in" aria-label="Zoom in">+</button><button id="spring-reset">Reset view</button></div>'
        '<svg id="springs" viewBox="0 0 900 440" tabindex="0" role="img" aria-label="Interactive 3D view of captured shoe springs" style="width:100%;max-width:900px;border:1px solid #ccd;touch-action:none"></svg>'
        '<p id="comparison-key" hidden></p>'
        "<script>"
    )
    if 'id="springs"' not in text:
        text = text.replace("</canvas><script>", insertion, 1)
    text = text.replace(",ctx=heat.getContext('2d')", "")
    draw_code = (Path(__file__).with_name("spring_viewer.js")).read_text()
    draw_start = text.index("function draw(){")
    heat_start = text.index("ctx.clearRect", draw_start)
    scrub_start = text.index("scrub.oninput=", heat_start)
    text = text[:heat_start] + "drawHeat();drawSprings()}" + text[scrub_start:]
    text = text.replace("function draw(){", draw_code + "function draw(){", 1)
    text = text.replace("drawHeat();drawSprings()}", "drawHeat();drawSprings();drawPoseOverlay()}", 1)
    viewer.write_text(text)
    return viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditions", nargs="+", type=Path)
    args = parser.parse_args()
    for condition in args.conditions:
        print(update_viewer(condition))


if __name__ == "__main__":
    main()
