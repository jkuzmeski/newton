# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare foam materials across longer-column and flat-block shoe geometries."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.digital_shoe.artifact import validate_artifact
from projects.impedance_instron.cartesian.shoe import Shoe
from projects.impedance_instron.hogan import materials


def geometry_variant(raw: dict, variant: str) -> dict:
    """Return a validated copy with the requested column geometry."""
    data = copy.deepcopy(raw)
    bed = data["column_bed"]
    bottom = np.asarray(bed["anchor_bottom_m"], dtype=np.float64)
    rest = np.asarray(bed["rest_length_m"], dtype=np.float64)
    original_top = bottom[:, 2] + rest
    if variant == "reference":
        return data
    data.pop("identification", None)
    if variant == "long125":
        new_rest = rest * 1.25
        new_bottom = bottom.copy()
        new_bottom[:, 2] = original_top - new_rest
        bed["anchor_bottom_m"] = new_bottom.tolist()
    elif variant == "block40":
        new_rest = np.full_like(rest, 0.04)
        new_bottom = bottom.copy()
        new_bottom[:, 2] = 0.0
        bed["anchor_bottom_m"] = new_bottom.tolist()
    else:
        raise ValueError(f"Unknown geometry variant {variant!r}")
    bed["rest_length_m"] = new_rest.tolist()

    fixtures = data.get("instron_fixtures", {})
    for fixture in fixtures.values():
        all_xy = np.asarray(bed["anchor_bottom_m"], dtype=np.float64)[:, :2]
        xy_to_index = {tuple(np.round(xy, 8)): i for i, xy in enumerate(all_xy)}
        supported = np.asarray(
            [xy_to_index[tuple(np.round(xy, 8))] for xy in np.asarray(fixture["carrier_anchor_m"])[:, :2]],
            dtype=int,
        )
        fixture["foam_bottom_m"] = np.asarray(bed["anchor_bottom_m"])[supported, 2].tolist()
        fixture["rest_length_m"] = new_rest[supported].tolist()
        if variant == "block40":
            fixture["foam_free_top_m"] = np.full(len(supported), 0.04).tolist()
    if variant == "block40":
        fixtures.pop("rearfoot_punch", None)
    # These meshes encode the source shoe's old foam envelope. Keep the last
    # used by the kinematic carrier and drop visuals that would misstate it.
    meshes = data.get("visual_meshes", {})
    data["visual_meshes"] = {key: value for key, value in meshes.items() if key == "fullfoot_last"}
    data["shoe"]["id"] = data["shoe"]["id"].split("__")[0] + "__" + variant
    data["shoe"]["model_scope"] = (
        f"Virtual shoe geometry variant {variant}; original column XY coordinates, areas and neighborhood are preserved."
    )
    data["validation"] = {
        "status": "geometry/material experiment; original intact-shoe calibration does not apply",
        "claim_boundary": "Column-bed geometry variation only; no whole-shoe structural or experimental validation.",
    }
    data["provenance"] = {
        "geometry_variant": variant,
        "geometry_source": "Original Puma column-bed XY coordinates and topology",
    }
    parameters = data["constitutive_model"]["parameters"]
    shear = (
        parameters.get("equilibrium_shear_modulus_override_pa")
        or (parameters["instantaneous_shear_modulus_pa"] + parameters["instantaneous_shear_modulus_2_pa"])
        * parameters["equilibrium_fraction"]
    )
    mean_length = float(np.mean(new_rest))
    parameters["pasternak_n_per_m"] = float(shear * mean_length)
    derived = data["constitutive_model"].get("derived_quantities", {})
    derived["equilibrium_shear_modulus_pa"] = float(shear)
    derived["pasternak_n_per_m"] = float(shear * mean_length)
    derived["pasternak_rule"] = "Descriptive derived value: equilibrium shear modulus * mean column rest_length_m."
    data["constitutive_model"]["derived_quantities"] = derived
    validate_artifact(data)
    return data


def build_conditions(artifact: Path, output: Path) -> list[dict]:
    """Build three material definitions crossed with three geometry variants."""
    output.mkdir(parents=True)
    templates = materials.make_conditions(artifact, output / "material_templates")
    templates = {row["id"]: row for row in templates if row["id"] in ("original", "ff_leap", "ff_turbo_plus")}
    rows = []
    for material_id in ("original", "ff_leap", "ff_turbo_plus"):
        source = templates[material_id]
        source_raw = json.loads(Path(source["artifact"]).read_text())
        for geometry_id in ("reference", "long125", "block40"):
            raw = geometry_variant(source_raw, geometry_id)
            if material_id != "original" or geometry_id != "reference":
                raw.pop("identification", None)
                raw["shoe"]["id"] = raw["shoe"]["id"].split("__")[0] + f"__{material_id}__{geometry_id}"
                raw["provenance"].update(material_id=material_id, geometry_variant=geometry_id)
                raw["provenance"]["material_fit"] = source.get("fit")
            path = output / f"{material_id}__{geometry_id}.json"
            if material_id == "original" and geometry_id == "reference":
                shutil.copy2(artifact, path)
            else:
                path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            rows.append(
                {
                    "id": f"{material_id}__{geometry_id}",
                    "material_id": material_id,
                    "geometry_id": geometry_id,
                    "label": f"{material_id.replace('_', ' ').title()} / {geometry_id.replace('_', ' ')}",
                    "artifact": str(path.resolve()),
                    "artifact_sha256": materials.sha256(path),
                    "fit": source.get("fit"),
                    "geometry": {
                        "mean_rest_length_m": float(np.mean(raw["column_bed"]["rest_length_m"])),
                        "min_rest_length_m": float(np.min(raw["column_bed"]["rest_length_m"])),
                        "max_rest_length_m": float(np.max(raw["column_bed"]["rest_length_m"])),
                    },
                }
            )
    return rows


def rematerialize_matched(trials: list, condition: dict) -> list:
    """Match ankle-to-ground clearance while preserving all velocity and phase inputs."""
    artifact = Path(condition["artifact"])
    shoes: dict[tuple, Shoe] = {}
    result = []
    for trial in trials:
        key = (tuple(trial.shoe.mount_m), trial.shoe.static_pitch_rad)
        if key not in shoes:
            shoes[key] = Shoe(artifact, *key)
        shoe = shoes[key]
        old_ankle = trial.chain.point(trial.initial.q, 3, np.zeros(2))[0]
        angle = trial.chain.angle(trial.initial.q, 3)
        original_clearance = float(trial.shoe.outline(old_ankle, angle)[:, 1].min())
        new_clearance = float(shoe.outline(old_ankle, angle)[:, 1].min())
        if original_clearance <= 0.0:
            raise ValueError(f"Trial {trial.id} has nonpositive original clearance")
        shift = original_clearance - new_clearance
        q_initial = trial.initial.q.copy()
        q_initial[1] += shift
        adjusted_initial = replace(trial.initial, q=q_initial)
        q_targets = trial.q.copy()
        q_targets[:, 1] += shift
        shifted_ankle = trial.chain.point(adjusted_initial.q, 3, np.zeros(2))[0]
        matched_clearance = float(shoe.outline(shifted_ankle, angle)[:, 1].min())
        if not np.isfinite(matched_clearance) or abs(matched_clearance - original_clearance) > 1e-10:
            raise ValueError(f"Trial {trial.id} failed matched-clearance verification")
        updated_compatibility = dict(trial.provenance["compatibility"])
        updated_compatibility["source_initial_clearance_m"] = original_clearance
        updated_compatibility["initial_clearance_m"] = original_clearance
        updated_compatibility["passed"] = bool(updated_compatibility["passed"] and original_clearance > 0.0)
        provenance = trial.provenance | {
            "shoe_artifact": str(artifact.resolve()),
            "shoe_sha256": materials.sha256(artifact),
            "material_id": condition["material_id"],
            "geometry_id": condition["geometry_id"],
            "geometry_initialization": {
                "protocol": "Match minimum nominal shoe-outline clearance over the ground plane by vertical translation; preserve velocity, phase, load and torque.",
                "source_clearance_m": original_clearance,
                "new_geometry_clearance_before_shift_m": new_clearance,
                "vertical_translation_m": shift,
                "matched_clearance_m": matched_clearance,
            },
            "compatibility": updated_compatibility,
        }
        result.append(replace(trial, shoe=shoe, initial=adjusted_initial, q=q_targets, provenance=provenance))
    return result


def study_metadata(conditions: list[dict], trials: list, args) -> dict:
    """Declare the factorial geometry experiment and its matched initialization."""
    return {
        "question": "How do three foam material laws and two column-bed geometry changes affect joint mechanical work under one frozen controller?",
        "changed_variables": conditions,
        "material_controller": "Frozen accepted Hogan controller; no refitting or material-specific tuning.",
        "geometry_protocol": {
            "reference": "Original bed geometry.",
            "long125": "Increase each column's rest length by 25%; keep XY coordinates, area, topology and original uncompressed top elevation.",
            "block40": "Replace the bed with a flat 40 mm slab at z=0, retaining all original column XY coordinates, areas and topology.",
            "initialization": "For each trial and geometry, translate initial and scored vertical coordinates equally to match the reference initial ankle-to-ground clearance. Velocities, phase, load, torque, tasks and observations are otherwise preserved.",
            "mass_assumption": "Shoe geometry has no independently modeled mass; this compares foundation geometry under identical runner/controller dynamics.",
        },
        "selection_rule": "All 9 declared factorial conditions are reported; no winner selected. Primary comparison uses held-out matched completed windows.",
        "qualification_checks": [
            "Controller and friction settings fixed across all 9 conditions",
            "Bed XY coordinates, area and neighbor topology unchanged",
            "Long-column top surface preserved; block top is flat at 40 mm",
            "Initial ankle clearance matched by documented vertical translation with unchanged velocities",
            "Held-out representative stance replay and half-step numerical sensitivity for each condition",
        ],
        "replay_samples": [
            {
                "condition": row["id"],
                "sample_id": "FR3_1_eval_000",
                "split": "eval",
                "selection": "First compatible held-out sample",
            }
            for row in conditions
        ],
        "budget": {
            "evaluation_and_capture_minutes": 40,
            "reporting_minutes": 15,
            "execution": "Serial exact CUDA evaluations; native saved-pose CPU contact replay.",
        },
        "stopping_conditions": "One full evaluation per condition with unchanged numerical screens; retain failures and do not refit.",
        "capture_plan": {
            "native": "Save every trial trajectory, scenario, condition geometry and work metric.",
            "deformation": "Save source-checked per-column contact replay for the same held-out stance in each of 9 conditions.",
            "report": "Factorial materials-by-geometry report with per-joint positive, absorbed and net work and matched pair comparisons.",
        },
        "factorial_design": {
            "materials": ["original", "ff_leap", "ff_turbo_plus"],
            "geometries": ["reference", "long125", "block40"],
            "conditions": len(conditions),
        },
        "command": {
            "module": __name__,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    }


def main(argv=None):
    """Run the 3-by-3 material and geometry comparison from scratch."""
    return materials.main(
        argv,
        condition_builder=build_conditions,
        trial_builder=rematerialize_matched,
        study_metadata=study_metadata,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
