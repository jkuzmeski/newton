# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export a shoe fitted by :mod:`sweep_fit` as a portable Digital Shoe artifact.

The artifact is built from the shoe's prepared column grid
(``DigitalInstron/<slug>/<side>/column_grid.npz``), the material in the
shoe's ``material_fit.json`` and the Instron last. Nothing is fitted here. The
last is registered again from the inputs the fit recorded and must land on the
recorded pose; the held-out sweep traces are replayed with the fitted material
through the two-surface column bed
(:class:`~projects.digital_shoe.column_bed.ColumnBed`) for the validation curves.

The shoe frame is the grid frame (+x heel to toe, +z up) shifted to the
footprint center and the lowest column bottom. The ``fullfoot_last`` fixture
lists the columns under the last tread with the last surface over each at first
contact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from projects.digital_shoe.artifact import MODEL_TYPE, SCHEMA_VERSION, validate_artifact
from projects.digital_shoe.report import write_report
from projects.digital_shoe.runtime import ShoeMaterial

from .core import EFFECTIVE_POISSON_RATIO, Material, metrics
from .geometry import ColumnGrid, neighbor_indices
from .last_replay import LastReplay, last_bed
from .registration import LastModel, LastRegistration, load_last, register_last
from .sweep_fit import DEFAULT_LAST, load_grid, load_trace
from .sweep_split import TRAIN_CYCLES, VALIDATION_CYCLES


def register_fitted_last(grid: ColumnGrid, last: LastModel, fit: dict[str, Any]) -> LastRegistration:
    """Register the last with the fit's recorded preload and check it lands on the fitted pose."""
    recorded = fit["registration"]
    preload = recorded["qc"]["preload"]
    registration = register_last(
        grid, last, modulus_pa=preload["modulus_pa"], preload_n=preload["preload_n"], expect_side=fit["side"]
    )
    if registration.qc["shoe_heel_at_input_plus_x"]:
        raise ValueError("the column grid must have its heel at -x")
    names = sorted(recorded["pose"])
    if not np.allclose(
        [registration.pose[k] for k in names], [recorded["pose"][k] for k in names], rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"last pose {registration.pose} differs from the fitted pose {recorded['pose']}")
    return registration


def _platen_to_grid(grid: ColumnGrid, registration: LastRegistration) -> tuple[np.ndarray, np.ndarray]:
    """Return ``R, t`` with ``platen = R grid + t`` from the registered column ends."""
    source = np.vstack((np.column_stack((grid.uv_m, grid.top_m)), np.column_stack((grid.uv_m, grid.bottom_m))))
    target = np.vstack((registration.plot["top_p"], registration.plot["bottom_p"]))
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ flip @ u.T
    return rotation, target_center - rotation @ source_center


def last_body_platen(path: Path, last: LastModel, registration: LastRegistration) -> tuple[np.ndarray, np.ndarray]:
    """Load the last body, without its attachment, in the platen frame at first contact.

    Returns:
        Vertices [m], shape (V, 3), and triangle indices, shape (T, 3).
    """
    import trimesh

    mesh = trimesh.load_mesh(path, process=True)
    local = mesh.vertices @ last.source_to_last[:3, :3].T + last.source_to_last[:3, 3]
    # The body lies below the empty gap between the last and its attachment plate.
    heights = np.sort(local[:, 2])
    body_top = heights[int(np.argmax(np.diff(heights)))]
    faces = mesh.faces[np.all(local[mesh.faces, 2] <= body_top, axis=1)]
    used, faces = np.unique(faces, return_inverse=True)
    local = local[used]
    pose = registration.pose
    c, s = np.cos(pose["yaw_rad"]), np.sin(pose["yaw_rad"])
    platen = np.column_stack(
        (local[:, :2] @ np.array([[c, s], [-s, c]]) + (pose["x_m"], pose["y_m"]), local[:, 2] + pose["lift_m"])
    )
    return platen, faces.reshape(-1, 3)


def _last_mesh(
    path: Path, last: LastModel, grid: ColumnGrid, registration: LastRegistration
) -> tuple[np.ndarray, np.ndarray]:
    """Place the last body, without its attachment, in the grid frame at first contact."""
    platen, faces = last_body_platen(path, last, registration)
    rotation, offset = _platen_to_grid(grid, registration)
    return (platen - offset) @ rotation, faces


def _mesh_json(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    """Encode a triangle mesh with compact runtime dtypes."""
    return {
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "vertices_m": np.asarray(vertices, dtype=np.float32).tolist(),
        "triangles": np.asarray(triangles, dtype=np.int32).tolist(),
    }


def _validation_curves(
    sweep: Path, grid: ColumnGrid, registration: LastRegistration, material: Material, substeps: int
) -> list[dict[str, Any]]:
    """Replay the held-out trace of every sweep condition with the fitted material.

    The stroke, preload indentation, frame interval and contact substeps follow :func:`sweep_fit.run`.
    """
    index = json.loads((sweep / "conditions.json").read_text())
    conditions = [c for c in index["conditions"] if c["role"] == "condition"]
    traces = [load_trace(sweep / c["validate_path"]) for c in conditions]
    indentation = registration.qc["preload"]["indentation_m"]
    depth = np.stack([np.maximum(t["displacement_m"], 0.0) + indentation for t in traces], axis=1)
    dt = float(np.mean(np.diff(traces[0]["time_s"])))
    force = LastReplay(last_bed(grid, registration), depth, dt, candidates=1, substeps=substeps).run([material])[0]
    return [
        {
            "name": f"{c['stance_ms']}ms_{c['target_n']}N",
            "fixture": "fullfoot_last",
            "split": "validate",
            "cycles": list(VALIDATION_CYCLES),
            "time_s": (t["time_s"] - t["time_s"][0]).tolist(),
            "displacement_m": t["displacement_m"].tolist(),
            "measured_force_n": t["force_n"].tolist(),
            "predicted_force_n": predicted.tolist(),
            "metrics": {
                **metrics(t["force_n"], predicted, t["displacement_m"]),
                "peak_force_n": float(np.max(predicted)),
                "measured_peak_force_n": float(np.max(t["force_n"])),
            },
        }
        for c, t, predicted in zip(conditions, traces, force, strict=True)
    ]


def build_artifact(
    shoe: Path,
    fit_path: Path,
    sweep: Path,
    last_path: Path = DEFAULT_LAST,
    *,
    mass_kg: float | None = None,
    mass_source: str | None = None,
) -> dict[str, Any]:
    """Bake the shoe's columns, fitted material, last placement, validation and hashes into JSON values.

    Args:
        shoe: Prepared shoe folder ``DigitalInstron/<slug>``.
        fit_path: The shoe's ``material_fit.json`` from :mod:`sweep_fit`.
        sweep: Processed sweep folder holding ``conditions.json``.
        last_path: Instron last STL with its merged attachment.
        mass_kg: Mass of one shoe [kg]; without it loaders use
            :data:`~projects.digital_shoe.artifact.DEFAULT_SHOE_MASS_KG`.
        mass_source: Where ``mass_kg`` was measured or published.
    """
    fit = json.loads(fit_path.read_text())
    if fit["shoe"] != shoe.name:
        raise ValueError(f"{fit_path} fits {fit['shoe']!r}, not {shoe.name!r}")
    grid_path = shoe / fit["side"] / "column_grid.npz"
    grid = load_grid(grid_path)
    last = load_last(last_path)
    registration = register_fitted_last(grid, last, fit)
    material = Material(**fit["material"])

    origin = np.array([*(grid.uv_m.min(axis=0) + grid.uv_m.max(axis=0)) / 2.0, grid.bottom_m.min()])
    rest = np.asarray(grid.slack_m, dtype=np.float64)
    count = len(rest)
    neighbors = neighbor_indices(grid.uv_m, grid.uv_m, grid.spacing_m)
    shear = material.equilibrium_shear_modulus_pa
    faces = np.argwhere(neighbors > np.arange(count)[:, None])
    face_coupling = shear * 0.5 * (rest[faces[:, 0]] + rest[neighbors[faces[:, 0], faces[:, 1]]])
    parameters = ShoeMaterial(
        material.instantaneous_shear_modulus_pa,
        material.hyperfoam_exponent,
        material.equilibrium_fraction,
        # Reported, not fitted: the face rule at the bed-mean rest thickness.
        float(shear * rest.mean()),
        EFFECTIVE_POISSON_RATIO,
        material.maxwell_relaxation_time_s,
        material.instantaneous_shear_modulus_2_pa,
        material.hyperfoam_exponent_2,
    )

    tread = np.isfinite(registration.surface_m)
    tread_uv = grid.uv_m[tread]
    last_vertices, last_triangles = _last_mesh(last_path, last, grid, registration)
    curves = _validation_curves(sweep, grid, registration, material, fit["substeps"])
    index = json.loads((sweep / "conditions.json").read_text())
    sources = [
        ("material_fit", fit_path),
        ("column_grid", grid_path),
        ("last_geometry", last_path),
        ("sweep_conditions", sweep / "conditions.json"),
    ]
    sources += [
        (f"{c['stance_ms']}ms_{c['target_n']}N_validate_trace", sweep / c["validate_path"])
        for c in index["conditions"]
        if c["role"] == "condition"
    ]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "shoe": {
            "id": f"{shoe.name.replace('-', '_')}_{fit['side']}",
            "model_scope": "effective intact-shoe response for the tested geometry and conditions",
            **({} if mass_kg is None else {"mass_kg": float(mass_kg), "mass_source": mass_source}),
        },
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Z",
            "length_unit": "m",
            "force_unit": "N",
            "origin": "footprint_center_xy_and_lowest_outsole_z",
            "x_axis": "heel_to_toe_verified_for_this_asset",
        },
        "constitutive_model": {
            "type": MODEL_TYPE,
            "parameters": asdict(parameters),
            "derived_quantities": {
                "hyperfoam_term_count": 2,
                "pasternak_rule": "k_ij = equilibrium_shear_modulus_pa * (rest_length_m[i] + rest_length_m[j]) / 2",
                "pasternak_n_per_m_is_fitted": False,
                "equilibrium_shear_modulus_pa": float(shear),
                "equilibrium_shear_modulus_term_1_pa": float(
                    material.instantaneous_shear_modulus_pa * material.equilibrium_fraction
                ),
                "equilibrium_shear_modulus_term_2_pa": float(
                    material.instantaneous_shear_modulus_2_pa * material.equilibrium_fraction
                ),
                "pasternak_n_per_m_min": float(face_coupling.min()),
                "pasternak_n_per_m_max": float(face_coupling.max()),
                "small_strain_compressive_modulus_pa": float(2.0 * shear),
            },
        },
        "column_bed": {
            "column_count": count,
            "anchor_bottom_m": (np.column_stack((grid.uv_m, grid.bottom_m)) - origin).tolist(),
            "rest_length_m": rest.tolist(),
            "area_m2": grid.area_m2.tolist(),
            "top_bonded": tread.tolist(),
            "neighbors": neighbors.tolist(),
            "spacing_m": float(grid.spacing_m),
            "neighbor_sentinels": {"-1": "natural_outer_boundary", "-2": "interior_inactive_gap"},
        },
        "visual_meshes": {"fullfoot_last": _mesh_json(last_vertices - origin, last_triangles)},
        "instron_fixtures": {
            "fullfoot_last": {
                "column_count": int(tread.sum()),
                "indenter": {"type": "baked_visual_mesh", "mesh": "fullfoot_last"},
                "carrier_anchor_m": (np.column_stack((tread_uv, registration.surface_m[tread])) - origin).tolist(),
                "foam_free_top_m": (grid.top_m[tread] - origin[2]).tolist(),
                "foam_bottom_m": (grid.bottom_m[tread] - origin[2]).tolist(),
                "rest_length_m": rest[tread].tolist(),
                "area_m2": grid.area_m2[tread].tolist(),
                "neighbors": neighbor_indices(tread_uv, grid.uv_m, grid.spacing_m).tolist(),
                "spacing_m": float(grid.spacing_m),
                "bed_bottom_above_platen_m": np.asarray(registration.bottom_above_platen_m, dtype=float).tolist(),
            }
        },
        "identification": {
            "method": "projects.digital_instron_v2.sweep_fit",
            "contact_law": "two_surface_column_bed",
            "sweep": fit["sweep"],
            "registration": fit["registration"],
            "training_cycles": list(TRAIN_CYCLES),
            "held_out_cycles": list(VALIDATION_CYCLES),
            "metrics": {
                name: {split: scores["fitted"] for split, scores in splits.items()}
                for name, splits in fit["conditions"].items()
            },
            "passed_all_declared_gates": all(curve["metrics"]["passed"] for curve in curves),
            "scenario_parameters_not_fitted": [
                "normal_damping",
                "tangential_bristle_stiffness",
                "tangential_damping",
                "friction_coefficient",
                "stretch_floor",
            ],
            "fixture_specific_parameters": [],
        },
        "validation": {
            "scope": "held-out cycles of every stance time and force target of the full-foot last sweep",
            "curves": curves,
            "claim_boundary": (
                "The artifact is not validated across new rates, temperatures, impact tests, or different shoes. "
                "Parameters are effective intact-shoe values, not intrinsic foam constants."
            ),
        },
        "provenance": {
            "generator": "projects.digital_instron_v2.export_digital_shoe",
            "source_files": [
                {"role": role, "name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for role, path in sources
            ],
        },
    }
    validate_artifact(artifact)
    return artifact


OPPOSITE_SIDE = {"left": "right", "right": "left"}


def mirror_artifact(data: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    """Return the opposite foot's shoe as the mirror image of ``data`` about the shoe's x-z plane.

    Every y coordinate changes sign, the -y and +y neighbor slots swap and
    triangle winding reverses so outward normals stay outward. The material,
    the bench pose heights and the validation are unchanged: the mirror is a
    geometric assumption, not a measured shoe of the other foot.
    """
    stem, _, side = data["shoe"]["id"].rpartition("_")
    mirrored = copy.deepcopy(data)
    reflect = np.array([1.0, -1.0, 1.0])
    slots = [0, 1, 3, 2]
    bed = mirrored["column_bed"]
    bed["anchor_bottom_m"] = (np.asarray(bed["anchor_bottom_m"]) * reflect).tolist()
    bed["neighbors"] = np.asarray(bed["neighbors"])[:, slots].tolist()
    for fixture in mirrored["instron_fixtures"].values():
        fixture["carrier_anchor_m"] = (np.asarray(fixture["carrier_anchor_m"]) * reflect).tolist()
        fixture["neighbors"] = np.asarray(fixture["neighbors"])[:, slots].tolist()
    for mesh in mirrored["visual_meshes"].values():
        mesh["vertices_m"] = (np.asarray(mesh["vertices_m"], dtype=np.float32) * reflect).tolist()
        mesh["triangles"] = np.asarray(mesh["triangles"])[:, [0, 2, 1]].tolist()
    pose = mirrored["identification"]["registration"]["pose"]
    pose["y_m"], pose["yaw_rad"] = -pose["y_m"], -pose["yaw_rad"]
    mirrored["shoe"]["id"] = f"{stem}_{OPPOSITE_SIDE[side]}"
    mirrored["shoe"]["mirrored_from"] = data["shoe"]["id"]
    mirrored["provenance"]["mirror"] = {
        "source_shoe_id": data["shoe"]["id"],
        "source_sha256": source_sha256,
        "operation": "reflect y -> -y; swap -y/+y neighbor slots; reverse triangle winding; negate last y and yaw",
        "claim": "Geometric mirror of the source shoe; not a measured opposite-foot shoe.",
    }
    validate_artifact(mirrored)
    return mirrored


def export(
    shoe: Path,
    fit_path: Path,
    sweep: Path,
    output: Path,
    last_path: Path = DEFAULT_LAST,
    *,
    mass_kg: float | None = None,
    mass_source: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write the fitted ``digital_shoe.json``, its mirror ``digital_shoe_<other side>.json`` and the HTML report."""
    artifact = build_artifact(shoe, fit_path, sweep, last_path, mass_kg=mass_kg, mass_source=mass_source)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "digital_shoe.json"
    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    artifact_path.write_text(text)
    mirrored = mirror_artifact(artifact, hashlib.sha256(text.encode()).hexdigest())
    mirror_path = output / f"digital_shoe_{mirrored['shoe']['id'].rpartition('_')[2]}.json"
    mirror_path.write_text(json.dumps(mirrored, indent=2, sort_keys=True) + "\n")
    report_path = write_report(artifact_path, output / "validation_report.html", media_dir=output)
    return artifact_path, mirror_path, report_path


def main() -> None:
    """Run the export command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoe", type=Path, required=True, help="Prepared shoe folder, e.g. DigitalInstron/<slug>.")
    parser.add_argument("--fit", type=Path, required=True, help="The shoe's material_fit.json from sweep_fit.")
    parser.add_argument("--sweep", type=Path, required=True, help="Processed sweep folder holding conditions.json.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mass-kg", type=float, help="Mass of one shoe [kg]; omit when unknown.")
    parser.add_argument("--mass-source", help="Where the shoe mass was measured or published.")
    parser.add_argument("--last", type=Path, default=DEFAULT_LAST)
    args = parser.parse_args()
    if (args.mass_kg is None) != (args.mass_source is None):
        parser.error("--mass-kg and --mass-source go together")
    paths = export(
        args.shoe, args.fit, args.sweep, args.output, args.last, mass_kg=args.mass_kg, mass_source=args.mass_source
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
