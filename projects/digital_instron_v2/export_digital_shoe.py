# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit Instron data and export a path-independent Digital Shoe runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from projects.digital_shoe.artifact import MODEL_TYPE, SCHEMA_VERSION, validate_artifact
from projects.digital_shoe.report import write_report
from projects.digital_shoe.runtime import ShoeMaterial

from .core import EFFECTIVE_POISSON_RATIO, MAXWELL_RELAXATION_TIME_S, Material, predict
from .dynamics import _neighbor_indices, build_foundation_geometry
from .geometry import build_column_grid, load_mesh, transform_mesh
from .phase1 import evaluate
from .workflow import prepare_trials


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_records(manifest: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    base = manifest.parent
    sources: list[tuple[str, Path]] = [("manifest", manifest), ("midsole_geometry", base / config["midsole_mesh"])]
    for trial in config["trials"]:
        sources.append((f"{trial['name']}_raw_measurement", base / trial["raw_csv_path"]))
        sources.append((f"{trial['name']}_averaged_cycle", base / trial["averaged_cycle_path"]))
        if indenter := trial.get("indenter", {}).get("path"):
            sources.append((f"{trial['name']}_indenter_geometry", base / indenter))
    return [{"role": role, "name": path.name, "sha256": _sha256(path)} for role, path in sources]


def _whole_column_bed(manifest: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = manifest.parent
    mesh = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(mesh, config["grid"]["coarse_spacing_m"])
    z_shift = float(np.min(grid.bottom_m))
    anchor = np.column_stack([grid.uv_m, grid.bottom_m - z_shift])
    count = len(grid.slack_m)
    neighbors = _neighbor_indices(grid.uv_m, grid.uv_m, grid.spacing_m)
    return {
        "column_count": count,
        "anchor_bottom_m": anchor.tolist(),
        "rest_length_m": np.asarray(grid.slack_m, dtype=np.float64).tolist(),
        "area_m2": np.full(count, grid.area_m2, dtype=np.float64).tolist(),
        "neighbors": neighbors.tolist(),
        "spacing_m": float(grid.spacing_m),
        "neighbor_sentinels": {"-1": "natural_outer_boundary", "-2": "interior_inactive_gap"},
    }


def _mesh_json(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    """Encode a triangle mesh with compact runtime dtypes."""
    return {
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "vertices_m": np.asarray(vertices, dtype=np.float32).tolist(),
        "triangles": np.asarray(triangles, dtype=np.int32).tolist(),
    }


def _visual_meshes(manifest: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Bake the calibrated midsole and posed full-foot indenter into the artifact."""
    base = manifest.parent
    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    midsole_vertices = np.asarray(midsole.vertices, dtype=np.float64).copy()
    midsole_vertices[:, 2] -= float(np.min(midsole_vertices[:, 2]))

    source = next(item for item in config["trials"] if item["fixture"] == "fullfoot_last")
    indenter = source["indenter"]
    shoe_last = load_mesh(
        base / indenter["path"],
        0.001,
        indenter["rotation_deg"],
        indenter["crop_height_m"],
    )
    transform_mesh(
        shoe_last,
        indenter.get("pose_rotation_deg", [0.0, 0.0, 0.0]),
        indenter.get("pose_translation_m", [0.0, 0.0, 0.0]),
    )
    geometry = build_foundation_geometry(manifest, "fullfoot_last")
    last_vertices = np.asarray(shoe_last.vertices, dtype=np.float64).copy()
    last_vertices[:, geometry.thickness_axis] += geometry.indenter_shift_m
    last_vertices[:, 2] -= geometry.z_shift_m
    shoe_last.vertices = last_vertices
    shoe_last.merge_vertices()

    return {
        "midsole": _mesh_json(midsole_vertices, np.asarray(midsole.faces)),
        "fullfoot_last": _mesh_json(np.asarray(shoe_last.vertices), np.asarray(shoe_last.faces)),
    }


def _instron_fixture(manifest: Path, fixture: str) -> dict[str, Any]:
    geometry = build_foundation_geometry(manifest, fixture)
    config = json.loads(manifest.read_text())
    source = next(item for item in config["trials"] if item["fixture"] == fixture)
    if fixture == "rearfoot_punch":
        indenter = {"type": "circular_punch", "radius_m": float(source["indenter"]["radius_m"])}
    else:
        indenter = {"type": "baked_visual_mesh", "mesh": "fullfoot_last"}
    count = len(geometry.slack_m)
    carrier_anchor = np.column_stack([geometry.uv_m, geometry.surface_m])
    return {
        "column_count": count,
        "indenter": indenter,
        "carrier_anchor_m": carrier_anchor.tolist(),
        "foam_free_top_m": np.asarray(geometry.z_free_m, dtype=np.float64).tolist(),
        "foam_bottom_m": np.asarray(geometry.z_bottom_m, dtype=np.float64).tolist(),
        "rest_length_m": np.asarray(geometry.slack_m, dtype=np.float64).tolist(),
        "area_m2": np.full(count, geometry.area_m2, dtype=np.float64).tolist(),
        "neighbors": np.asarray(geometry.neighbors, dtype=np.int32).tolist(),
        "spacing_m": float(geometry.spacing_m),
    }


def _validation_curves(
    manifest: Path, config: dict[str, Any], report: dict[str, Any], material: Material
) -> list[dict[str, Any]]:
    base = manifest.parent
    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
    paths = {name: base / item["validate"]["csv"] for name, item in report["traces"].items()}
    trials, _, _ = prepare_trials(base, config, grid, midsole, trace_paths=paths)
    sources = {source["name"]: source for source in config["trials"]}
    curves = []
    for trial in trials:
        measured = np.asarray(trial.force_n, dtype=np.float64)
        measured -= float(measured.min())
        predicted = np.asarray(predict(trial, material), dtype=np.float64)
        predicted -= float(predicted.min())
        time = np.cumsum(np.asarray(trial.dt_s, dtype=np.float64))
        time -= time[0]
        curves.append(
            {
                "name": trial.name,
                "fixture": sources[trial.name]["fixture"],
                "split": "held_out_adjacent_cycles",
                "cycles": report["validate_cycles"],
                "time_s": time.tolist(),
                "displacement_m": np.asarray(trial.displacement_m, dtype=np.float64).tolist(),
                "measured_force_n": measured.tolist(),
                "predicted_force_n": predicted.tolist(),
                "metrics": report["validation_metrics"][trial.name],
            }
        )
    return curves


def build_artifact(manifest_path: str | Path, report: dict[str, Any], *, shoe_id: str) -> dict[str, Any]:
    """Bake mesh-derived columns, fitted parameters, validation, and hashes into JSON values."""
    manifest = Path(manifest_path).resolve()
    config = json.loads(manifest.read_text())
    fitted = Material(**report["material"])
    material = ShoeMaterial(
        fitted.instantaneous_shear_modulus_pa,
        fitted.hyperfoam_exponent,
        fitted.equilibrium_fraction,
        fitted.pasternak_n_per_m,
        EFFECTIVE_POISSON_RATIO,
        MAXWELL_RELAXATION_TIME_S,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "shoe": {
            "id": shoe_id,
            "model_scope": "effective intact-shoe response for the tested geometry and conditions",
        },
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "length_unit": "m",
            "force_unit": "N",
            "origin": "footprint_center_xy_and_lowest_outsole_z",
            "x_axis": "heel_to_toe_verified_for_this_asset",
        },
        "constitutive_model": {"type": MODEL_TYPE, "parameters": asdict(material)},
        "column_bed": _whole_column_bed(manifest, config),
        "visual_meshes": _visual_meshes(manifest, config),
        "instron_fixtures": {
            fixture: _instron_fixture(manifest, fixture) for fixture in ("rearfoot_punch", "fullfoot_last")
        },
        "identification": {
            "backend": report["backend"],
            "training_cycles": report["train_cycles"],
            "held_out_cycles": report["validate_cycles"],
            "metrics": report["validation_metrics"],
            "gates": report["gates"],
            "passed_all_declared_gates": report["passed"],
            "scenario_parameters_not_fitted": [
                "normal_damping",
                "tangential_bristle_stiffness",
                "tangential_damping",
                "friction_coefficient",
                "stretch_floor",
            ],
        },
        "validation": {
            "scope": "adjacent held-out cycles from the same approximately 0.5 s fixture protocols",
            "curves": _validation_curves(manifest, config, report, fitted),
            "claim_boundary": (
                "The artifact is not validated across new rates, temperatures, impact tests, or different shoes. "
                "Parameters are effective intact-shoe values, not intrinsic foam constants."
            ),
        },
        "provenance": {
            "generator": "projects.digital_instron_v2.export_digital_shoe",
            "source_files": _source_records(manifest, config),
        },
    }
    validate_artifact(artifact)
    return artifact


def identify_and_export(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    shoe_id: str = "puma_fast_r_nitro_elite_3_left",
    evaluations: int = 100,
) -> tuple[Path, Path]:
    """Fit training cycles, evaluate held-out cycles, and write the artifact and HTML report."""
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = evaluate(manifest, backend="scipy", evaluations=evaluations, write_report=False)
    artifact = build_artifact(manifest, report, shoe_id=shoe_id)
    artifact_path = output / "digital_shoe.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    report_path = write_report(artifact_path, output / "validation_report.html", media_dir=output)
    return artifact_path, report_path


def main() -> None:
    """Run the complete fitting and export workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("DigitalInstron/manifest_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("DigitalInstron/digital_shoe_showcase"))
    parser.add_argument("--shoe-id", default="puma_fast_r_nitro_elite_3_left")
    parser.add_argument("--evaluations", type=int, default=100)
    args = parser.parse_args()
    artifact, report = identify_and_export(
        args.manifest, args.output, shoe_id=args.shoe_id, evaluations=args.evaluations
    )
    print(f"artifact: {artifact}")
    print(f"validation report: {report}")


if __name__ == "__main__":
    main()
