# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Redraw saved experiment figures without rerunning or relabeling the physics."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate saved JSON key: {key}")
            value[key] = item
        return value

    def reject(value):
        raise ValueError(f"Nonfinite saved JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=reject)


def _local(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"Saved input must remain inside report directory: {name}")
    return path


def render_saved_report(directory: str | Path, *, overview_only: bool = False) -> Path:
    """Redraw a saved response or sensitivity report in place from verified files.

    Args:
        directory: Saved suite with summary.json, snapshots and numeric traces.
        overview_only: Redraw only the sensitivity overview; retain its existing
            case pages. The response renderer always redraws its linked pages.

    Returns:
        Redrawn report.html. Only presentation files and render_metadata.json
        are written. Simulation source mismatches do not authorize a new score:
        recorded outcomes remain associated with their original implementation.
    """
    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    record = _read(summary_path)
    schema = record.get("schema_version")
    if schema not in ("impedance_material_sensitivity_1", "impedance_paired_response_1"):
        raise ValueError("Expected a saved impedance response or sensitivity suite")
    protected = {"summary.json": _sha(summary_path)}
    if schema == "impedance_material_sensitivity_1":
        module = importlib.import_module(".sensitivity", __package__)
        manifest_path = directory / "manifest.json"
        manifest = _read(manifest_path)
        module._verify_manifest(manifest)
        module._verify_manifest(record)
        if record["config_seal"] != manifest["config_seal"]:
            raise ValueError("Saved summary and manifest settings disagree")
        protected["manifest.json"] = _sha(manifest_path)
        protected.update(record["snapshot_manifest"])
    else:
        protected.update(
            {"reference.json": record["reference_snapshot_sha256"], "artifact.json": record["artifact_sha256"]}
        )
    for case in record.get("cases", []):
        if case.get("trace_file"):
            if not case.get("trace_sha256"):
                raise ValueError(f"Saved trace has no recorded hash: {case['trace_file']}")
            name, expected = case["trace_file"], case["trace_sha256"]
            if name in protected and protected[name] != expected:
                raise ValueError(f"Conflicting hashes for saved file: {name}")
            protected[name] = expected
    for name, expected in protected.items():
        if _sha(_local(directory, name)) != expected:
            raise ValueError(f"Saved data changed: {name}; report figures were not redrawn")
    local = Path(__file__).parent
    sources = ("render_report.py", "figures.py", "sensitivity_figures.py", "sensitivity_report.py", "response.py")
    renderer_hashes = {name: _sha(local / name) for name in sources}
    if schema == "impedance_material_sensitivity_1":
        renderer = importlib.import_module(".sensitivity_report", __package__)
        result = renderer.write_sensitivity_report(directory, record, write_case_pages=not overview_only)
    else:
        renderer = importlib.import_module(".response", __package__)
        result = renderer._write_html(directory, record)
    for name, expected in protected.items():
        if _sha(_local(directory, name)) != expected:
            raise RuntimeError(f"Saved data changed during report rendering: {name}")
    metadata = {
        "operation": "redraw_saved_figures_only",
        "physics_rerun": False,
        "saved_metrics_modified": False,
        "saved_source_fingerprints_modified": False,
        "original_summary_sha256": protected["summary.json"],
        "protected_file_count": len(protected),
        "case_pages_redrawn": schema != "impedance_material_sensitivity_1" or not overview_only,
        "renderer_source_sha256": renderer_hashes,
        "renderer_sources_changed_during_draw": {
            name: {"before": old, "after": _sha(local / name)}
            for name, old in renderer_hashes.items()
            if _sha(local / name) != old
        },
        "qualification": "Saved outcomes retain their original physical qualification; current source does not revalidate them.",
    }
    (directory / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
