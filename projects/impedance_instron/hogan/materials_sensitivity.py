# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vary the fitted shoe material one property at a time with a frozen Hogan runner.

Every condition keeps the shoe's geometry and changes only constitutive
parameters, through :func:`scale_material`. The runner rollouts reuse
:func:`projects.impedance_instron.hogan.materials.main`. With ``--instron`` the
same conditions are also replayed on the held-out bench sweep, as the Digital
Shoe exporter computes its validation curves, so each variant has a physical
reading: peak force, secant stiffness and loop energy loss at a fixed stroke.

The column bed couples neighbouring columns with ``shear_layer_fraction *
mu_eq``, where ``mu_eq = (G_1 + G_2) * equilibrium_fraction``; the same
``mu_eq`` sets the tangential bristle stiffness. ``pasternak_n_per_m`` is only
reported, so the variants rewrite it as the exporter does (bed-mean layer
coupling) to keep the artifact self-consistent.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from dataclasses import fields
from functools import partial
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.core import Material
from projects.digital_instron_v2.export_digital_shoe import register_fitted_last, validation_curves
from projects.digital_instron_v2.registration import load_last
from projects.digital_instron_v2.sweep_fit import load_grid
from projects.digital_shoe.artifact import validate_artifact
from projects.impedance_instron.hogan import materials

KNOBS = ("stiffness", "maxwell_share", "slider_yield", "slider_modulus", "shear_layer", "exponent")
"""Multipliers understood by :func:`scale_material`; 1 leaves a property unchanged."""

VARIANTS: dict[str, dict[str, float]] = {
    **{f"stiffness_x{s:g}": {"stiffness": s} for s in (0.5, 0.7, 1.4, 2.0)},
    **{f"slider_yield_x{s:g}": {"slider_yield": s} for s in (0.0, 0.5, 2.0)},
    **{f"slider_modulus_x{s:g}": {"slider_modulus": s} for s in (0.5, 2.0)},
    "maxwell_off": {"maxwell_share": 0.0},
    "maxwell_share_x2": {"maxwell_share": 2.0},
    "no_loss": {"slider_yield": 0.0, "maxwell_share": 0.0},
    **{f"shear_layer_x{s:g}": {"shear_layer": s} for s in (0.5, 2.0)},
    **{f"exponent_x{s:g}": {"exponent": s} for s in (0.7, 1.4)},
}
"""One-at-a-time variants of the fitted material, by knob multiplier."""

INSTRON_CURVES = ("200ms_800N", "200ms_1600N")
"""Bench curves summarized per variant; every replayed curve is saved."""


def scale_material(parameters: dict, knobs: dict[str, float]) -> dict:
    """Return a copy of ``parameters`` with every knob multiplier applied.

    - ``stiffness`` scales both Ogden-Hill moduli: equilibrium, instantaneous
      and ``mu_eq`` (shear layer, bristles) together.
    - ``maxwell_share`` scales the Maxwell share ``1 - equilibrium_fraction`` of
      the instantaneous modulus and rescales the moduli so the equilibrium
      curve is unchanged; 0 turns the Maxwell branch off. The relaxation time
      is kept.
    - ``slider_yield`` and ``slider_modulus`` scale the slider branch; a zero
      yield turns it off.
    - ``shear_layer`` scales ``shear_layer_fraction``.
    - ``exponent`` scales both Ogden-Hill exponents. The small-strain tangent
      ``2 (G_1 + G_2) equilibrium_fraction`` does not depend on them, so the
      moduli need no rescaling.
    """
    if unknown := sorted(set(knobs) - set(KNOBS)):
        raise ValueError(f"unknown material knobs {unknown}")
    p = dict(parameters)
    moduli = ("instantaneous_shear_modulus_pa", "instantaneous_shear_modulus_2_pa")
    for key in moduli:
        p[key] *= knobs.get("stiffness", 1.0)
    if "maxwell_share" in knobs:
        fraction = p["equilibrium_fraction"]
        new_fraction = 1.0 - (1.0 - fraction) * knobs["maxwell_share"]
        if not 0.0 < new_fraction <= 1.0:
            raise ValueError(f"Maxwell share x{knobs['maxwell_share']:g} leaves no equilibrium network")
        for key in moduli:
            p[key] *= fraction / new_fraction
        p["equilibrium_fraction"] = new_fraction
    p["slider_yield_pa"] *= knobs.get("slider_yield", 1.0)
    p["slider_modulus_pa"] *= knobs.get("slider_modulus", 1.0)
    p["shear_layer_fraction"] *= knobs.get("shear_layer", 1.0)
    p["hyperfoam_exponent"] *= knobs.get("exponent", 1.0)
    p["hyperfoam_exponent_2"] *= knobs.get("exponent", 1.0)
    return p


def combine(variant_ids: list[str]) -> dict[str, float]:
    """Merge the knobs of several variants; one knob may come from one variant only."""
    knobs: dict[str, float] = {}
    for variant in variant_ids:
        for knob, value in VARIANTS[variant].items():
            if knob in knobs:
                raise ValueError(f"knob {knob!r} is set by more than one of {variant_ids}")
            knobs[knob] = value
    return knobs


def variant_artifact(raw: dict, condition: str, knobs: dict[str, float]) -> dict:
    """Return a validated artifact with the scaled material on unchanged geometry."""
    data = copy.deepcopy(raw)
    model = data["constitutive_model"]
    p = model["parameters"] = scale_material(model["parameters"], knobs)
    shear = (p["instantaneous_shear_modulus_pa"] + p["instantaneous_shear_modulus_2_pa"]) * p["equilibrium_fraction"]
    p["pasternak_n_per_m"] = float(p["shear_layer_fraction"] * shear * np.mean(data["column_bed"]["rest_length_m"]))
    model["derived_quantities"] = {
        "equilibrium_shear_modulus_pa": shear,
        "small_strain_compressive_modulus_pa": 2.0 * shear,
        "shear_layer_modulus_pa": p["shear_layer_fraction"] * shear,
        "pasternak_rule": "Reported only: shear_layer_fraction * equilibrium_shear_modulus_pa * mean rest_length_m.",
    }
    data.pop("identification", None)
    data["shoe"]["id"] += f"__{condition}"
    data["shoe"]["model_scope"] = "Scaled copy of the fitted shoe material on unchanged geometry; not calibrated."
    data["validation"] = {
        "status": "material sensitivity variant; the source calibration does not apply",
        "claim_boundary": "One-property scaling of a fitted effective material; no bench or whole-shoe validation.",
    }
    data["provenance"] = {"generator": __name__, "condition": condition, "knobs": knobs}
    validate_artifact(data)
    return data


def build_conditions(artifact: Path, output: Path, *, variants: dict[str, dict[str, float]]) -> list[dict]:
    """Write the fitted shoe and every variant as condition artifacts."""
    raw = json.loads(artifact.read_text())
    output.mkdir(parents=True)
    rows = []
    for name, knobs in {"original": {}, **variants}.items():
        path = output / f"{name}.json"
        if knobs:
            materials.write_json(path, variant_artifact(raw, name, knobs))
        else:
            path.write_bytes(artifact.read_bytes())
        rows.append(
            {
                "id": name,
                "label": name.replace("_", " "),
                "knobs": knobs,
                "artifact": str(path.resolve()),
                "artifact_sha256": materials.sha256(path),
                "parameters": json.loads(path.read_text())["constitutive_model"]["parameters"],
                "fit": None,
            }
        )
    return rows


def study_metadata(conditions: list[dict], trials: list, args) -> dict:
    """Declare the one-at-a-time material sensitivity study."""
    return {
        "question": "How much do the fitted shoe's own material properties change joint work under a frozen controller?",
        "changed_variables": conditions,
        "material_protocol": inspect.cleandoc(scale_material.__doc__),
        "selection_rule": "All declared variants are reported; no winner is selected. Held-out windows are primary.",
        "command": {
            "module": __name__,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    }


def instron_physics(depth_m: np.ndarray, force_n: np.ndarray) -> dict:
    """Return peak force, secant stiffness and loop energy loss of one bench cycle.

    The secant stiffness is the peak force over the depth at that peak. The
    energy loss is the loop area over the loading work, both integrated over
    the last's depth from first contact.
    """
    depth = np.asarray(depth_m, dtype=float)
    force = np.asarray(force_n, dtype=float)
    peak = int(np.argmax(depth))
    loading = float(np.trapezoid(force[: peak + 1], depth[: peak + 1]))
    unloading = float(-np.trapezoid(force[peak:], depth[peak:]))
    top = int(np.nanargmax(force))
    return {
        "peak_force_n": float(force[top]),
        "secant_stiffness_n_per_mm": float(force[top] / depth[top] / 1000.0),
        "loading_work_j": loading,
        "energy_loss_fraction": (loading - unloading) / loading,
    }


def instron_replay(conditions: list[dict], shoe: Path, fit_path: Path, sweep: Path, last: Path) -> dict:
    """Replay the held-out bench sweep for every condition in one batched launch."""
    fit = json.loads(fit_path.read_text())
    grid = load_grid(shoe / fit["side"] / "column_grid.npz")
    registration = register_fitted_last(grid, load_last(last), fit)
    names = [f.name for f in fields(Material)]
    batch = [Material(**{k: c["parameters"][k] for k in names}) for c in conditions]
    curves = validation_curves(sweep, grid, registration, batch, fit["substeps"], fit.get("start_offset_m", 0.0))
    result = {}
    for condition, condition_curves in zip(conditions, curves, strict=True):
        result[condition["id"]] = {
            curve["name"]: {
                "predicted": instron_physics(curve["indenter_depth_m"], curve["predicted_force_n"]),
                "measured": instron_physics(curve["indenter_depth_m"], curve["measured_force_n"]),
                "metrics": curve["metrics"],
            }
            for curve in condition_curves
        }
    return {
        "inputs": {
            key: str(path) for key, path in (("shoe", shoe), ("fit", fit_path), ("sweep", sweep), ("last", last))
        },
        "summary_curves": list(INSTRON_CURVES),
        "conditions": result,
    }


def main(argv=None):
    """Run the declared variants and optional combinations through the frozen-controller study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument(
        "--combine", action="append", default=[], metavar="NAME=VARIANT+VARIANT", help="Add a combined condition"
    )
    parser.add_argument(
        "--instron",
        nargs=4,
        type=Path,
        metavar=("SHOE", "FIT", "SWEEP", "LAST"),
        help="Also replay the bench sweep: prepared shoe folder, material_fit.json, sweep folder, last STL",
    )
    parser.add_argument("--output", type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    variants = {name: VARIANTS[name] for name in args.variants}
    for item in args.combine:
        name, members = item.split("=", 1)
        variants[name] = combine(members.split("+"))
    materials.main(
        [*rest, "--output", str(args.output)],
        condition_builder=partial(build_conditions, variants=variants),
        study_metadata=study_metadata,
    )
    summary = args.output / "summary.json"
    if args.instron and summary.exists():
        conditions = json.loads(summary.read_text())["conditions"]
        materials.write_json(args.output / "instron.json", instron_replay(conditions, *args.instron))


if __name__ == "__main__":
    main(sys.argv[1:])
