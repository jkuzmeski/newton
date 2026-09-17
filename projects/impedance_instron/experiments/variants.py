# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenario and variant builder for frozen-controller shoe sensitivity experiments."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from .paper_materials import (
    FF_LEAP_SURROGATE_PARAMS,
    FF_TURBO_PLUS_SURROGATE_PARAMS,
    get_paper_material_metadata,
)

# Primary factors and multipliers declared in EXPERIMENT_PLAN.md
MATERIAL_FACTORS: list[str] = ["mu1", "mu2", "alpha1", "alpha2", "branch", "tau"]
MATERIAL_SCALES: list[float] = [0.8, 0.9, 1.1, 1.2]
LENGTH_SCALES: list[float] = [0.8, 0.9, 1.1, 1.2]

FACTOR_DESCRIPTIONS: dict[str, str] = {
    "baseline": "Unchanged baseline shoe artifact",
    "mu1": "Ogden-Hill term 1 modulus (instantaneous_shear_modulus_pa)",
    "mu2": "Ogden-Hill term 2 modulus (instantaneous_shear_modulus_2_pa)",
    "alpha1": "Ogden-Hill term 1 exponent (hyperfoam_exponent)",
    "alpha2": "Ogden-Hill term 2 exponent (hyperfoam_exponent_2)",
    "branch": "Maxwell relaxable fraction b = 1 - equilibrium_fraction",
    "tau": "Maxwell relaxation time (maxwell_relaxation_time_s)",
    "length": "Raycast column rest length (thickness) with nominal top fixed",
    "rectangle": "Uniform rectangular spring bed spanning midsole mesh horizontal bounding box",
    "flat_footprint": "Flat top-plane/constant-height companion on original footprint",
    "paper_compression": "Paper-informed compression surrogate (McCulloch, Delp, Kuhl 2026)",
}


def _calculate_neighbors(uv: np.ndarray, grid_uv: np.ndarray, spacing: float) -> np.ndarray:
    """Return the Pasternak 4-neighbor index table (-u, +u, -v, +v).

    Each neighbor is a non-negative active-column index, ``-1`` for a footprint
    boundary, or ``-2`` for a midsole cell outside the active set.
    """
    cells = [tuple(np.rint(p / spacing).astype(int)) for p in uv]
    index = {c: i for i, c in enumerate(cells)}
    full = {tuple(np.rint(p / spacing).astype(int)) for p in grid_uv}
    out = np.full((len(cells), 4), -2, dtype=np.int32)
    for i, (u_val, v_val) in enumerate(cells):
        for side, (du, dv) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            c = (u_val + du, v_val + dv)
            if c in index:
                out[i, side] = index[c]
            elif c not in full:
                out[i, side] = -1
    return out


def conditions(suite: str = "sensitivity") -> list[dict[str, Any]]:
    """Return experiment conditions for the declared suite.

    Args:
        suite: Condition suite identifier. Supported suites:
            - ``"sensitivity"`` (default): The 31 primary OAT material and geometry
              sensitivity conditions.
            - ``"paper_compression"``: Baseline plus two paper-informed unconfined
              compression surrogate presets (FF LEAP and FF TURBO PLUS).

    Returns:
        List of condition dictionaries, each containing ``id``, ``family``,
        ``factor``, and ``scale`` (optional/float/None).
    """
    if suite == "sensitivity":
        conds: list[dict[str, Any]] = [
            {
                "id": "baseline",
                "family": "baseline",
                "factor": "baseline",
                "scale": 1.0,
                "description": FACTOR_DESCRIPTIONS["baseline"],
            }
        ]

        # 24 material variants
        for factor in MATERIAL_FACTORS:
            for scale in MATERIAL_SCALES:
                conds.append(
                    {
                        "id": f"material_{factor}_{scale}",
                        "family": "material",
                        "factor": factor,
                        "scale": scale,
                        "description": f"{FACTOR_DESCRIPTIONS[factor]} scaled by {scale}x",
                    }
                )

        # 4 length geometry variants
        for scale in LENGTH_SCALES:
            conds.append(
                {
                    "id": f"length_{scale}",
                    "family": "geometry",
                    "factor": "length",
                    "scale": scale,
                    "description": f"{FACTOR_DESCRIPTIONS['length']} scaled by {scale}x",
                }
            )

        # 2 geometry companions/endpoints
        conds.append(
            {
                "id": "rectangle",
                "family": "geometry",
                "factor": "rectangle",
                "scale": None,
                "description": FACTOR_DESCRIPTIONS["rectangle"],
            }
        )
        conds.append(
            {
                "id": "flat_footprint",
                "family": "geometry",
                "factor": "flat_footprint",
                "scale": None,
                "description": FACTOR_DESCRIPTIONS["flat_footprint"],
            }
        )

        return conds

    if suite == "paper_compression":
        return [
            {
                "id": "baseline",
                "family": "baseline",
                "factor": "baseline",
                "scale": 1.0,
                "description": FACTOR_DESCRIPTIONS["baseline"],
            },
            {
                "id": "material_ff_leap_compression",
                "family": "material",
                "factor": "paper_compression",
                "scale": None,
                "description": "FF LEAP paper unconfined compression surrogate (McCulloch et al. 2026)",
                "paper_material": get_paper_material_metadata("ff_leap_compression"),
            },
            {
                "id": "material_ff_turbo_plus_compression",
                "family": "material",
                "factor": "paper_compression",
                "scale": None,
                "description": "FF TURBO PLUS paper unconfined compression surrogate (McCulloch et al. 2026)",
                "paper_material": get_paper_material_metadata("ff_turbo_plus_compression"),
            },
        ]

    raise ValueError(f"Unknown condition suite: {suite!r}; expected 'sensitivity' or 'paper_compression'")


def condition_metadata(condition_or_id: str | dict[str, Any]) -> dict[str, Any]:
    """Return full normalized condition metadata for a condition dictionary or ID string.

    Supports condition IDs from both the primary ``"sensitivity"`` suite and the
    ``"paper_compression"`` preset suite.

    Args:
        condition_or_id: Condition identifier string (e.g. ``"material_mu1_0.8"`` or
            ``"material_ff_leap_compression"``) or partial condition dictionary.

    Returns:
        Complete metadata dictionary including ``id``, ``family``, ``factor``,
        ``scale``, and ``description`` (plus ``paper_material`` when applicable).
    """
    # Index conditions across both supported suites
    all_conds = {c["id"]: c for c in conditions("sensitivity")}
    for c in conditions("paper_compression"):
        all_conds[c["id"]] = c

    if isinstance(condition_or_id, str):
        cid = condition_or_id
        if cid in all_conds:
            return copy.deepcopy(all_conds[cid])
        for candidate_id, cdict in all_conds.items():
            if cid in (candidate_id, candidate_id.removeprefix("material_")):
                return copy.deepcopy(cdict)
        raise KeyError(f"Unknown condition ID: {cid!r}; available: {sorted(all_conds)}")

    if isinstance(condition_or_id, dict):
        cid = condition_or_id.get("id")
        if cid and cid in all_conds:
            merged = copy.deepcopy(all_conds[cid])
            merged.update(condition_or_id)
            return merged
        for candidate_id, cdict in all_conds.items():
            if cid in (candidate_id, candidate_id.removeprefix("material_")):
                merged = copy.deepcopy(cdict)
                merged.update(condition_or_id)
                return merged

        family = condition_or_id.get("family", "material")
        factor = condition_or_id.get("factor", "unknown")
        scale = condition_or_id.get("scale")
        desc = condition_or_id.get("description", FACTOR_DESCRIPTIONS.get(factor, f"{factor} variant"))
        res = {
            "id": cid or f"{family}_{factor}_{scale}",
            "family": family,
            "factor": factor,
            "scale": scale,
            "description": desc,
        }
        if cid in ("material_ff_leap_compression", "ff_leap_compression"):
            res["paper_material"] = get_paper_material_metadata("ff_leap_compression")
        elif cid in ("material_ff_turbo_plus_compression", "ff_turbo_plus_compression"):
            res["paper_material"] = get_paper_material_metadata("ff_turbo_plus_compression")
        return res

    raise TypeError(f"Expected str or dict for condition_or_id, got {type(condition_or_id).__name__}")


def _update_derived_pasternak(
    constitutive_model: dict[str, Any],
    rest_lengths_m: np.ndarray,
) -> None:
    """Update Pasternak coupling and derived constitutive quantities from current parameters and geometry."""
    params = constitutive_model["parameters"]
    derived = constitutive_model.setdefault("derived_quantities", {})

    g1 = float(params["instantaneous_shear_modulus_pa"])
    g2 = float(params.get("instantaneous_shear_modulus_2_pa", 0.0))
    feq = float(params["equilibrium_fraction"])

    mu_eq = (g1 + g2) * feq
    mu_eq_1 = g1 * feq
    mu_eq_2 = g2 * feq

    rest_lengths = np.asarray(rest_lengths_m, dtype=np.float64)
    coupling_n_per_m = mu_eq * rest_lengths
    mean_k = float(np.mean(coupling_n_per_m))

    params["pasternak_n_per_m"] = mean_k

    derived.update(
        {
            "hyperfoam_term_count": 2 if g2 > 0.0 else 1,
            "pasternak_rule": "k_i = equilibrium_shear_modulus_pa * rest_length_m[i]",
            "pasternak_n_per_m_is_fitted": False,
            "equilibrium_shear_modulus_pa": float(mu_eq),
            "equilibrium_shear_modulus_term_1_pa": float(mu_eq_1),
            "equilibrium_shear_modulus_term_2_pa": float(mu_eq_2),
            "pasternak_n_per_m_min": float(np.min(coupling_n_per_m)),
            "pasternak_n_per_m_max": float(np.max(coupling_n_per_m)),
            "small_strain_compressive_modulus_pa": float(2.0 * mu_eq),
        }
    )


def _mark_synthetic_and_strip_acceptance(
    artifact: dict[str, Any],
    source: dict[str, Any],
    condition: dict[str, Any],
) -> None:
    """Mark synthetic sensitivity provenance and replace inherited acceptance flags without breaking schema."""
    cid = condition["id"]
    base_shoe_id = source.get("shoe", {}).get("id", "shoe")
    artifact.setdefault("shoe", {})["id"] = f"{base_shoe_id}_{cid}"
    artifact["shoe"]["model_scope"] = (
        "Synthetic parameter sensitivity variant; not an independently fitted or accepted shoe product"
    )

    artifact["provenance"] = {
        "generator": "projects.impedance_instron.experiments.variants",
        "synthetic": True,
        "condition": {
            "id": cid,
            "family": condition.get("family"),
            "factor": condition.get("factor"),
            "scale": condition.get("scale"),
        },
        "parent_shoe_id": base_shoe_id,
        "parent_provenance": source.get("provenance", {}),
    }

    if "identification" in artifact:
        ident = artifact["identification"]
        ident["passed_all_declared_gates"] = False
        ident["acceptance_status"] = "synthetic_sensitivity_variant"
        ident["synthetic_variant"] = True
        ident["note"] = (
            "Synthetic variant derived from sealed baseline; does not inherit baseline identification acceptance."
        )

    if "validation" in artifact:
        val = artifact["validation"]
        val["passed_all_declared_gates"] = False
        val["claim_boundary"] = (
            "Synthetic sensitivity case only; do not cite as an accepted calibration or physical product."
        )
        for curve in val.get("curves", []):
            if "metrics" in curve and isinstance(curve["metrics"], dict):
                curve["metrics"]["passed"] = False


def _apply_material_condition(
    artifact: dict[str, Any],
    factor: str,
    scale: float,
) -> None:
    """Apply one-at-a-time (OAT) material parameter multiplier relative to the baseline."""
    params = artifact["constitutive_model"]["parameters"]

    if factor == "mu1":
        params["instantaneous_shear_modulus_pa"] = float(params["instantaneous_shear_modulus_pa"] * scale)
    elif factor == "mu2":
        params["instantaneous_shear_modulus_2_pa"] = float(params["instantaneous_shear_modulus_2_pa"] * scale)
    elif factor == "alpha1":
        params["hyperfoam_exponent"] = float(params["hyperfoam_exponent"] * scale)
    elif factor == "alpha2":
        params["hyperfoam_exponent_2"] = float(params["hyperfoam_exponent_2"] * scale)
    elif factor == "branch":
        b_base = 1.0 - float(params["equilibrium_fraction"])
        b_new = float(scale * b_base)
        params["equilibrium_fraction"] = float(1.0 - b_new)
    elif factor == "tau":
        params["maxwell_relaxation_time_s"] = float(params["maxwell_relaxation_time_s"] * scale)
    else:
        raise ValueError(f"Unknown material factor: {factor!r}; expected one of {MATERIAL_FACTORS}")

    rest_lengths = np.asarray(artifact["column_bed"]["rest_length_m"], dtype=np.float64)
    _update_derived_pasternak(artifact["constitutive_model"], rest_lengths)


def _apply_length_condition(
    artifact: dict[str, Any],
    scale: float,
) -> None:
    """Scale column rest lengths while holding physical nominal spring tops fixed."""
    bed = artifact["column_bed"]
    l_orig = np.asarray(bed["rest_length_m"], dtype=np.float64)
    bot_orig = np.asarray(bed["anchor_bottom_m"], dtype=np.float64)

    l_new = scale * l_orig
    delta_bottom = (1.0 - scale) * l_orig
    bot_new = bot_orig.copy()
    bot_new[:, 2] += delta_bottom

    bed["rest_length_m"] = l_new.tolist()
    bed["anchor_bottom_m"] = bot_new.tolist()

    bed_lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(bot_orig[:, :2])}
    for _fix_name, fixture in artifact.get("instron_fixtures", {}).items():
        carrier = np.asarray(fixture["carrier_anchor_m"], dtype=np.float64)
        supported = [bed_lookup[tuple(np.round(pt, 8))] for pt in carrier[:, :2]]
        fix_delta = delta_bottom[supported]

        fb_orig = np.asarray(fixture["foam_bottom_m"], dtype=np.float64)
        fb_new = fb_orig + fix_delta
        fixture["foam_bottom_m"] = fb_new.tolist()

        f_rest_orig = np.asarray(fixture["rest_length_m"], dtype=np.float64)
        f_rest_new = scale * f_rest_orig
        fixture["rest_length_m"] = f_rest_new.tolist()

        fixture["foam_free_top_m"] = (fb_new + f_rest_new).tolist()

    _update_derived_pasternak(artifact["constitutive_model"], l_new)


def _apply_flat_footprint_condition(
    artifact: dict[str, Any],
) -> None:
    """Create flat top-plane/constant-height companion on original footprint with original mask."""
    bed = artifact["column_bed"]
    l_orig = np.asarray(bed["rest_length_m"], dtype=np.float64)
    a_orig = np.asarray(bed["area_m2"], dtype=np.float64)
    bot_orig = np.asarray(bed["anchor_bottom_m"], dtype=np.float64)

    h0 = float(np.sum(a_orig * l_orig) / np.sum(a_orig))
    top_orig = bot_orig[:, 2] + l_orig
    top0 = float(np.sum(a_orig * top_orig) / np.sum(a_orig))
    bot0 = top0 - h0

    count = len(l_orig)
    l_flat = np.full(count, h0, dtype=np.float64)
    bot_flat = bot_orig.copy()
    bot_flat[:, 2] = bot0
    delta_bottom = bot_flat[:, 2] - bot_orig[:, 2]

    bed["rest_length_m"] = l_flat.tolist()
    bed["anchor_bottom_m"] = bot_flat.tolist()

    bed_lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(bot_orig[:, :2])}
    for _fix_name, fixture in artifact.get("instron_fixtures", {}).items():
        carrier = np.asarray(fixture["carrier_anchor_m"], dtype=np.float64)
        supported = [bed_lookup[tuple(np.round(pt, 8))] for pt in carrier[:, :2]]
        fix_delta = delta_bottom[supported]

        fb_orig = np.asarray(fixture["foam_bottom_m"], dtype=np.float64)
        fb_new = fb_orig + fix_delta
        fixture["foam_bottom_m"] = fb_new.tolist()

        f_rest_new = np.full(len(supported), h0, dtype=np.float64)
        fixture["rest_length_m"] = f_rest_new.tolist()
        fixture["foam_free_top_m"] = (fb_new + f_rest_new).tolist()

    _update_derived_pasternak(artifact["constitutive_model"], l_flat)


def _apply_rectangle_condition(
    artifact: dict[str, Any],
) -> None:
    """Build regular uniform rectangular spring bed over the horizontal midsole bounding box.

    For the stance simulation, sets top-level ``stance_attachment={"mode": "all_columns",
    "top_plane_z_m": top0}`` so the stance model drives all rectangle columns rigidly to
    the foot carrier.

    For bench fixtures (``fullfoot_last``, ``rearfoot_punch``), preserves the actual curved
    shoe last carrier surface and circular punch indenter geometry without flattening the last
    into a platen. The rectangle grid aligns with the original shoe bed lattice so that
    source fixture column (x, y) coordinates are preserved exactly.
    """
    spacing = float(artifact["column_bed"].get("spacing_m", 0.005))
    if "visual_meshes" in artifact and "midsole" in artifact["visual_meshes"]:
        verts = np.asarray(artifact["visual_meshes"]["midsole"]["vertices_m"], dtype=np.float64)
        lower = verts.min(axis=0)[:2]
        upper = verts.max(axis=0)[:2]
    else:
        orig_bot = np.asarray(artifact["column_bed"]["anchor_bottom_m"], dtype=np.float64)
        lower = orig_bot.min(axis=0)[:2]
        upper = orig_bot.max(axis=0)[:2]

    orig_bed = artifact["column_bed"]
    orig_l = np.asarray(orig_bed["rest_length_m"], dtype=np.float64)
    orig_a = np.asarray(orig_bed["area_m2"], dtype=np.float64)
    orig_bot = np.asarray(orig_bed["anchor_bottom_m"], dtype=np.float64)

    h0 = float(np.sum(orig_a * orig_l) / np.sum(orig_a))
    top_orig = orig_bot[:, 2] + orig_l
    top0 = float(np.sum(orig_a * top_orig) / np.sum(orig_a))
    bot0 = top0 - h0

    # Determine number of regular cells along x and y spanning [lower, upper]
    nx = int(np.ceil((upper[0] - lower[0]) / spacing - 1e-9))
    ny = int(np.ceil((upper[1] - lower[1]) / spacing - 1e-9))

    u_vals = lower[0] + np.arange(nx) * spacing
    v_vals = lower[1] + np.arange(ny) * spacing

    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
    xy = np.column_stack([uu.ravel(), vv.ravel()])
    count = len(xy)

    # Clipped edge-cell areas: interior cells receive spacing^2, boundary cells clip to box
    dx_vals = np.full(nx, spacing)
    dx_vals[-1] = upper[0] - u_vals[-1]
    dy_vals = np.full(ny, spacing)
    dy_vals[-1] = upper[1] - v_vals[-1]
    areas_2d = np.outer(dx_vals, dy_vals)
    areas = areas_2d.ravel()

    bottoms = np.column_stack([xy, np.full(count, bot0)])
    rest_lengths = np.full(count, h0)
    neighbors = _calculate_neighbors(xy, xy, spacing)

    artifact["column_bed"] = {
        "column_count": count,
        "anchor_bottom_m": bottoms.tolist(),
        "rest_length_m": rest_lengths.tolist(),
        "area_m2": areas.tolist(),
        "neighbors": neighbors.tolist(),
        "spacing_m": spacing,
    }

    # Stance-only all-columns driven override for rectangle (bench fixtures ignore this)
    artifact["stance_attachment"] = {
        "mode": "all_columns",
        "top_plane_z_m": float(top0),
    }

    # Map bench fixtures onto the rectangle lattice, preserving actual fixture carrier shapes
    rect_lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(xy)}
    orig_bed_lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(orig_bot[:, :2])}

    rect_fixtures: dict[str, Any] = {}
    for fix_name, fixture in artifact.get("instron_fixtures", {}).items():
        f_carrier = np.asarray(fixture["carrier_anchor_m"], dtype=np.float64)
        f_fb_orig = np.asarray(fixture["foam_bottom_m"], dtype=np.float64)
        f_xy = f_carrier[:, :2]

        # Match fixture columns to original bed to compute matched displacement
        supp_orig = [
            orig_bed_lookup[tuple(np.round(pt, 8))] for pt in f_xy if tuple(np.round(pt, 8)) in orig_bed_lookup
        ]
        if len(supp_orig) == len(f_xy):
            # Matched bottom displacement preserves actual fixture carrier surfaces
            delta_bot = bot0 - orig_bot[supp_orig, 2]
            fb_new = f_fb_orig + delta_bot
        else:
            fb_new = np.full(len(f_xy), bot0, dtype=np.float64)

        f_rest_new = np.full(len(f_xy), h0, dtype=np.float64)
        ft_new = fb_new + f_rest_new

        # Map fixture footprint onto rectangle lattice
        supp_rect = [rect_lookup[tuple(np.round(pt, 8))] for pt in f_xy if tuple(np.round(pt, 8)) in rect_lookup]
        if len(supp_rect) == len(f_xy):
            f_areas = areas[supp_rect]
        else:
            f_areas = np.asarray(fixture["area_m2"], dtype=np.float64)

        f_neighbors = _calculate_neighbors(f_xy, xy, spacing)

        rect_fixtures[fix_name] = {
            "column_count": len(f_xy),
            "carrier_anchor_m": f_carrier.tolist(),  # Invariant actual fixture carrier shape
            "foam_free_top_m": ft_new.tolist(),
            "foam_bottom_m": fb_new.tolist(),
            "rest_length_m": f_rest_new.tolist(),
            "area_m2": f_areas.tolist(),
            "neighbors": f_neighbors.tolist(),
            "spacing_m": spacing,
            "indenter": fixture.get("indenter", {}),
        }

    artifact["instron_fixtures"] = rect_fixtures
    _update_derived_pasternak(artifact["constitutive_model"], rest_lengths)


def _apply_paper_compression_condition(
    artifact: dict[str, Any],
    cid: str,
    source: dict[str, Any],
) -> None:
    """Apply literature unconfined compression surrogate parameters to the shoe constitutive model.

    Retains source equilibrium_fraction and maxwell_relaxation_time_s exactly.
    Converts paper surrogate equilibrium moduli to instantaneous moduli via:
        mu_inst = mu_eq / source.equilibrium_fraction
    Recomputes derived Pasternak foundation quantities using the existing helper.
    Enforces exact invariance of all geometry, fixtures, mass, friction, and rest lengths.
    """
    clean_id = cid.removeprefix("material_")
    if clean_id == "ff_leap_compression":
        paper_params = FF_LEAP_SURROGATE_PARAMS
    elif clean_id == "ff_turbo_plus_compression":
        paper_params = FF_TURBO_PLUS_SURROGATE_PARAMS
    else:
        raise ValueError(f"Unknown paper compression condition: {cid!r}")

    source_params = source["constitutive_model"]["parameters"]
    feq = float(source_params["equilibrium_fraction"])
    tau = float(source_params["maxwell_relaxation_time_s"])

    mu1_eq = float(paper_params["mu1_eq_pa"])
    alpha1 = float(paper_params["alpha1"])
    mu2_eq = float(paper_params["mu2_eq_pa"])
    alpha2 = float(paper_params["alpha2"])

    # Convert equilibrium moduli to instantaneous moduli holding feq fixed
    mu1_inst = mu1_eq / feq
    mu2_inst = mu2_eq / feq

    params = artifact["constitutive_model"]["parameters"]
    params["instantaneous_shear_modulus_pa"] = mu1_inst
    params["hyperfoam_exponent"] = alpha1
    params["instantaneous_shear_modulus_2_pa"] = mu2_inst
    params["hyperfoam_exponent_2"] = alpha2
    params["equilibrium_fraction"] = feq
    params["maxwell_relaxation_time_s"] = tau

    rest_lengths = np.asarray(artifact["column_bed"]["rest_length_m"], dtype=np.float64)
    _update_derived_pasternak(artifact["constitutive_model"], rest_lengths)

    # Explicit invariant guards (ValueError raised even under python -O)
    for key in (
        "column_bed",
        "instron_fixtures",
        "visual_meshes",
        "stance_attachment",
        "coordinate_system",
        "schema_version",
    ):
        if key in source:
            if json.dumps(artifact.get(key), sort_keys=True) != json.dumps(source.get(key), sort_keys=True):
                raise ValueError(f"Invariant violation: {key} altered in paper compression variant")

    # Verify all shoe metadata fields (mass, friction, mount, static_pitch, etc.) except synthetic id & model_scope
    source_shoe = source.get("shoe", {})
    art_shoe = artifact.get("shoe", {})
    for key, val in source_shoe.items():
        if key in ("id", "model_scope"):
            continue
        if art_shoe.get(key) != val:
            raise ValueError(f"Invariant violation: shoe[{key!r}] altered ({art_shoe.get(key)!r} != {val!r})")

    # Verify remaining constitutive parameters
    if params.get("effective_poisson_ratio") != source_params.get("effective_poisson_ratio"):
        raise ValueError("Invariant violation: effective_poisson_ratio altered in paper compression variant")
    if params["equilibrium_fraction"] != feq:
        raise ValueError("Invariant violation: equilibrium_fraction altered in paper compression variant")
    if params["maxwell_relaxation_time_s"] != tau:
        raise ValueError("Invariant violation: maxwell_relaxation_time_s altered in paper compression variant")


def build_variant(source: dict[str, Any], condition: dict[str, Any] | str) -> dict[str, Any]:
    """Build a complete digital shoe artifact dictionary for a specified experiment condition.

    Args:
        source: Base digital shoe artifact dictionary (e.g. loaded from baseline digital_shoe.json).
        condition: Condition dictionary or identifier string.

    Returns:
        Variant digital shoe artifact dictionary matching the ``digital_shoe_1`` schema.
        Identity baseline returns a byte-equivalent deepcopy. Non-baseline variants
        mark synthetic provenance and replace inherited acceptance flags.
    """
    cond = condition_metadata(condition) if isinstance(condition, str) else condition
    cid = cond.get("id", "")
    family = cond.get("family", "")
    factor = cond.get("factor", "")
    scale = cond.get("scale")

    # Identity baseline returns unchanged byte-equivalent deepcopy
    if cid == "baseline" or (family == "baseline" and (scale is None or scale == 1.0)):
        return copy.deepcopy(source)

    variant = copy.deepcopy(source)
    _mark_synthetic_and_strip_acceptance(variant, source, cond)

    # Normalize factor name
    clean_factor = factor.removeprefix("material_")

    if (
        factor == "paper_compression"
        or clean_factor in ("ff_leap_compression", "ff_turbo_plus_compression", "paper_compression")
        or cid in ("material_ff_leap_compression", "material_ff_turbo_plus_compression")
    ):
        _apply_paper_compression_condition(variant, cid, source)
    elif family == "material" or clean_factor in MATERIAL_FACTORS:
        if scale is None:
            raise ValueError(f"Material condition {cid!r} requires a numeric scale")
        _apply_material_condition(variant, clean_factor, float(scale))
    elif family == "geometry" or clean_factor in ("length", "rectangle", "flat_footprint"):
        if clean_factor == "length":
            if scale is None:
                raise ValueError(f"Length condition {cid!r} requires a numeric scale")
            _apply_length_condition(variant, float(scale))
        elif clean_factor == "rectangle":
            _apply_rectangle_condition(variant)
        elif clean_factor == "flat_footprint":
            _apply_flat_footprint_condition(variant)
        else:
            raise ValueError(f"Unknown geometry factor: {factor!r}")
    else:
        raise ValueError(f"Unknown condition family: {family!r} or factor: {factor!r}")

    return variant
