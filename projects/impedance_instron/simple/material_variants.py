# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Snapshot material-only sensitivity cases without changing the runtime law.

These are fixed, per-episode constants, not material randomization during a
rollout. Whole-law amplitude changes scale both Ogden-Hill shear moduli. The
native Pasternak layer is material-pinned: ``k_i = G_eq * rest_length_m[i]``.
Its neighbor conductance therefore changes too; it is not an independent fit
parameter. Relaxation cases change only the Maxwell time constant.

Synthetic cases are not calibrated shoes. Baseline validation stays in the
byte-identical baseline snapshot, not in synthetic case qualification records.
Only explicitly listed metadata fields are exempt from the nonmaterial hash.
Unknown fields, including fields under metadata containers, remain protected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from projects.digital_shoe.artifact import DigitalShoe, load_artifact, validate_artifact

WARNING = (
    "Synthetic material sensitivity only; not a newly calibrated or validated shoe. "
    "Baseline fit quality does not qualify a perturbed material. "
    "Parameters stay constant for each episode; geometry and the constitutive law stay fixed. "
    "Modulus scaling also scales the physically derived Pasternak neighbor coupling."
)
_PASTERNAK_RULE = "k_i = equilibrium_shear_modulus_pa * rest_length_m[i]"
_METADATA_KEYS = {
    "shoe": {"id", "name", "model_scope"},
    "identification": {
        "backend",
        "training_cycles",
        "held_out_cycles",
        "metrics",
        "gates",
        "passed_all_declared_gates",
        "status",
        "baseline_reference",
        "claim_boundary",
    },
    "validation": {"scope", "curves", "claim_boundary", "status", "baseline_reference"},
    "provenance": {"generator", "source_files", "material_sensitivity"},
}
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_MIN = float(np.nextafter(np.float32(0.0), np.float32(1.0)))


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _nonmaterial_payload(data: dict) -> dict:
    value = copy.deepcopy(data)
    model = value["constitutive_model"]
    model.pop("parameters", None)
    model.pop("derived_quantities", None)
    for container, keys in _METADATA_KEYS.items():
        if container in value:
            value[container] = {key: item for key, item in value[container].items() if key not in keys}
            if not value[container]:
                value.pop(container)
    return value


def _check_numbers(value: Any, location: str, *, float32: bool = False) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _check_numbers(item, f"{location}.{key}", float32=float32)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_numbers(item, f"{location}[{index}]", float32=float32)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{location} must be finite")
        if float32 and (abs(value) > _FLOAT32_MAX or (value != 0 and abs(value) < _FLOAT32_MIN)):
            raise ValueError(f"{location} is not representable as a finite nonzero float32")


def _numeric_array(value: Any, location: str, *, integral: bool = False) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a numeric array")
    for item in value:
        if isinstance(item, list):
            _numeric_array(item, location, integral=integral)
        elif isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{location} must contain numbers, not strings or booleans")
        elif integral and (not isinstance(item, int) or not -(2**31) <= item < 2**31):
            raise ValueError(f"{location} must contain int32 indices")


def _derived(shoe: DigitalShoe) -> dict:
    material = shoe.material
    term1 = material.instantaneous_shear_modulus_pa * material.equilibrium_fraction
    term2 = material.instantaneous_shear_modulus_2_pa * material.equilibrium_fraction
    total = term1 + term2
    with np.errstate(over="ignore", invalid="ignore"):
        coupling = total * shoe.column_bed.rest_length_m
        fixture_coupling = {
            name: (total * fixture.rest_length_m).tolist() for name, fixture in shoe.instron_fixtures.items()
        }
    result = {
        "hyperfoam_term_count": 2 if material.instantaneous_shear_modulus_2_pa > 0 else 1,
        "pasternak_rule": _PASTERNAK_RULE,
        "pasternak_n_per_m_is_fitted": False,
        "equilibrium_shear_modulus_pa": total,
        "equilibrium_shear_modulus_term_1_pa": term1,
        "equilibrium_shear_modulus_term_2_pa": term2,
        # The shared J = stretch**(1 - 2*nu) law has tangent 2*G_eq*(1+nu).
        # The calibration exporter fixes nu=0, recovering its reported 2*G_eq.
        "small_strain_compressive_modulus_pa": 2.0 * total * (1.0 + material.effective_poisson_ratio),
        "pasternak_n_per_m_min": float(np.min(coupling)),
        "pasternak_n_per_m_max": float(np.max(coupling)),
        "pasternak_n_per_m_mean": float(np.mean(coupling)),
        "pasternak_n_per_m_by_column": coupling.tolist(),
        "pasternak_n_per_m_by_fixture": fixture_coupling,
    }
    _check_numbers(result, "derived_quantities", float32=True)
    return result


def _validate_data(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("artifact must be a JSON object")
    _check_numbers(data, "artifact")
    for key in ("shoe", "coordinate_system", "constitutive_model", "column_bed", "validation", "provenance"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"artifact {key} must be an object")
    for key in ("identification", "visual_meshes", "instron_fixtures"):
        if key in data and not isinstance(data[key], dict):
            raise ValueError(f"artifact {key} must be an object")
    if not isinstance(data["shoe"].get("id"), str) or not data["shoe"]["id"]:
        raise ValueError("shoe.id must be a nonempty string")
    model = data["constitutive_model"]
    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("constitutive_model.parameters must be an object")
    for key, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"material parameter {key} must be numeric")
    if "derived_quantities" in model and not isinstance(model["derived_quantities"], dict):
        raise ValueError("constitutive_model.derived_quantities must be an object")
    _check_numbers(parameters, "parameters", float32=True)
    _check_numbers(model.get("derived_quantities", {}), "derived_quantities", float32=True)
    _check_numbers(_nonmaterial_payload(data), "nonmaterial", float32=True)
    geometry = [("column_bed", data["column_bed"]), *data.get("instron_fixtures", {}).items()]
    for name, bed in geometry:
        if not isinstance(bed, dict):
            raise ValueError(f"geometry {name} must be an object")
        for key in ("rest_length_m", "area_m2", "neighbors"):
            _numeric_array(bed.get(key), f"{name}.{key}", integral=key == "neighbors")
            if key != "neighbors" and np.asarray(bed[key]).ndim != 1:
                raise ValueError(f"{name}.{key} must be a one-dimensional array")
        for key in ("anchor_bottom_m", "carrier_anchor_m", "foam_free_top_m", "foam_bottom_m"):
            if key in bed:
                _numeric_array(bed[key], f"{name}.{key}")
        spacing = bed.get("spacing_m")
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or spacing <= 0:
            raise ValueError(f"{name}.spacing_m must be a finite positive number")
        _check_numbers(1.0 / spacing / spacing, f"{name}.inverse_spacing_squared", float32=True)
    for name, mesh in data.get("visual_meshes", {}).items():
        if not isinstance(mesh, dict):
            raise ValueError(f"visual mesh {name} must be an object")
        _numeric_array(mesh.get("vertices_m"), f"{name}.vertices_m")
        _numeric_array(mesh.get("triangles"), f"{name}.triangles", integral=True)
    try:
        validate_artifact(data)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Malformed DigitalShoe artifact: {exc}") from exc


def _report_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-12)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_report_matches(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_report_matches(actual[key], value) for key, value in expected.items())
        )
    return type(actual) is type(expected) and actual == expected


def _read(path: str | Path) -> tuple[Path, bytes, DigitalShoe, dict]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed DigitalShoe JSON: {path}") from exc
    _validate_data(data)
    # Use the native loader, not a replacement material or geometry contract.
    shoe = load_artifact(path)
    if shoe.raw != data:
        raise ValueError(f"Artifact changed while being read: {path}")
    material = shoe.material
    fraction = material.equilibrium_fraction
    poisson = material.effective_poisson_ratio
    constants = {
        "g_eq": material.instantaneous_shear_modulus_pa * fraction,
        "g_eq2": material.instantaneous_shear_modulus_2_pa * fraction,
        "overstress": (1.0 - fraction) / fraction,
        "beta": poisson / (1.0 - 2.0 * poisson),
        "one_minus_two_poisson": 1.0 - 2.0 * poisson,
    }
    _check_numbers(constants, "runtime_material", float32=True)
    derived = _derived(shoe)
    reported = data["constitutive_model"].get("derived_quantities", {})
    for key, expected in derived.items():
        if key not in reported or key in {"hyperfoam_term_count", "pasternak_n_per_m_is_fitted"}:
            continue
        if not _report_matches(reported[key], expected):
            raise ValueError(f"Inconsistent derived quantity {key}")
    if "pasternak_n_per_m_is_fitted" in reported and reported["pasternak_n_per_m_is_fitted"] is not False:
        raise ValueError("Native Pasternak coupling is derived, not fitted")
    if "hyperfoam_term_count" in reported:
        count = reported["hyperfoam_term_count"]
        if (
            type(count) is not int
            or count not in (1, 2)
            or (count == 1 and material.instantaneous_shear_modulus_2_pa > 0)
        ):
            raise ValueError("Inconsistent derived quantity hyperfoam_term_count")
    if "pasternak_rule" in reported or reported.get("pasternak_n_per_m_is_fitted") is False:
        if not math.isclose(material.pasternak_n_per_m, derived["pasternak_n_per_m_mean"], rel_tol=1.0e-6):
            raise ValueError("Reported pasternak_n_per_m must equal the bed mean of the declared derived rule")
    return path, raw, shoe, derived


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def material_artifact_identity(path: str | Path) -> dict:
    """Return validated hashes with a fail-closed nonmaterial comparison payload.

    The return keys match ``policy.artifact_identity``. ``geometry`` includes
    every nonmaterial field, not just the mesh. Only the exact metadata keys
    listed in this module are ignored. In particular, unknown shoe, fixture,
    mount, orientation, model, and metadata-container fields remain protected.
    """
    source, raw, shoe, _ = _read(path)
    geometry = _nonmaterial_payload(shoe.raw)
    material = shoe.raw["constitutive_model"]
    return {
        "path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "geometry_sha256": _json_hash(geometry),
        "geometry": geometry,
        "material_sha256": _json_hash(material),
        "material": material,
    }


def _factors(values: Any, name: str) -> tuple[float, ...]:
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of positive finite multipliers") from exc
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain numeric multipliers")
        _check_numbers(value, name, float32=True)
        if value <= 0.0:
            raise ValueError(f"{name} multipliers must be finite and positive")
        result.append(float(value))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate multipliers")
    return tuple(result)


def _clear_qualification(data: dict, baseline_sha256: str, variant_id: str, factors: dict) -> None:
    reference = {"sha256": baseline_sha256, "snapshot": "baseline.json", "claims_apply_to": "baseline_only"}
    for container in ("identification", "validation"):
        if container not in data:
            continue
        data[container] = {key: value for key, value in data[container].items() if key not in _METADATA_KEYS[container]}
        data[container].update(status="not_validated", baseline_reference=reference, claim_boundary=WARNING)
    data["shoe"]["model_scope"] = "synthetic material sensitivity; not a calibrated or validated shoe variant"
    data["provenance"]["generator"] = "projects.impedance_instron.simple.material_variants"
    data["provenance"]["material_sensitivity"] = {
        "generator": "projects.impedance_instron.simple.material_variants",
        "variant_id": variant_id,
        "baseline_reference": reference,
        "factors": factors,
        "qualification": "not_validated",
        "warning": WARNING,
    }


def build_material_variants(
    artifact_path: str | Path,
    output_dir: str | Path,
    *,
    modulus_multipliers: tuple[float, ...] = (0.75, 1.25),
    relaxation_multipliers: tuple[float, ...] = (0.5, 2.0),
    material_paths: tuple[str | Path, ...] = (),
) -> list[dict]:
    """Write independent, immutable-by-convention material snapshots for a suite.

    Args:
        artifact_path: Baseline self-contained DigitalShoe JSON. Never modified.
        output_dir: New or empty directory. Existing files are never overwritten.
        modulus_multipliers: Positive factors applied together to both shear
            moduli [Pa]. Shape exponents, fraction, and relaxation stay fixed.
        relaxation_multipliers: Positive factors applied only to relaxation [s].
        material_paths: User-supplied DigitalShoe JSON files with identical
            nonmaterial physical fields. Only declared metadata may differ.

    Returns:
        JSON-compatible records, baseline first, then modulus, relaxation, and
        imported cases. Paths are absolute snapshot paths; a suite can store
        them relative to its manifest. Each record contains source and snapshot
        hashes, both identities, factors, full resolved parameters, derived
        per-column coupling [N/m], and explicit qualification provenance.

    Raises:
        ValueError: If parameters, geometry, reporting conventions, or a
            material-only override are invalid, or the output is nonempty.
        FileExistsError: If a destination appears while writing snapshots.

    Notes:
        Baseline and imported snapshots preserve source bytes. Imported fit
        claims are not endorsed. Synthetic snapshots replace only known
        baseline qualification metadata. Legacy reported Pasternak scalars
        without a declared bed-mean convention are scaled as reports, but the
        native solver always derives coupling from both shear moduli.
        These numeric checks do not certify trajectory-dependent stability.
    """
    modulus = _factors(modulus_multipliers, "modulus_multipliers")
    relaxation = _factors(relaxation_multipliers, "relaxation_multipliers")
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Material snapshot output directory must be new or empty")
    baseline_path, baseline_bytes, baseline, baseline_derived = _read(artifact_path)
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    baseline_geometry = _nonmaterial_payload(baseline.raw)
    geometry_hash = _json_hash(baseline_geometry)
    cases = [("baseline", "baseline", {}, baseline_path, baseline_bytes)]
    for kind, values in (("modulus", modulus), ("relaxation", relaxation)):
        for index, factor in enumerate(values, 1):
            variant_id = f"{kind}_{index:02d}"
            factors = {
                "modulus_multiplier": factor if kind == "modulus" else 1.0,
                "relaxation_multiplier": factor if kind == "relaxation" else 1.0,
            }
            data = copy.deepcopy(baseline.raw)
            parameters = data["constitutive_model"]["parameters"]
            if kind == "modulus":
                parameters["instantaneous_shear_modulus_pa"] *= factor
                if "instantaneous_shear_modulus_2_pa" in parameters:
                    parameters["instantaneous_shear_modulus_2_pa"] *= factor
                parameters["pasternak_n_per_m"] *= factor
                derived = data["constitutive_model"].setdefault("derived_quantities", {})
                for key in (
                    "equilibrium_shear_modulus_pa",
                    "equilibrium_shear_modulus_term_1_pa",
                    "equilibrium_shear_modulus_term_2_pa",
                    "small_strain_compressive_modulus_pa",
                    "pasternak_n_per_m_min",
                    "pasternak_n_per_m_max",
                ):
                    derived[key] = baseline_derived[key] * factor
                # Optional detailed reports must not retain stale amplitudes.
                for key in ("pasternak_n_per_m_mean", "pasternak_n_per_m_by_column", "pasternak_n_per_m_by_fixture"):
                    if key in derived:
                        derived[key] = _scale_numbers(baseline_derived[key], factor)
            else:
                parameters["maxwell_relaxation_time_s"] = baseline.material.maxwell_relaxation_time_s * factor
            _clear_qualification(data, baseline_sha256, variant_id, factors)
            _validate_data(data)
            if _json_hash(_nonmaterial_payload(data)) != geometry_hash:
                raise ValueError("Synthetic variant unexpectedly changed nonmaterial fields")
            payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            cases.append((variant_id, kind, factors, baseline_path, payload))
    if isinstance(material_paths, (str, Path)):
        raise ValueError("material_paths must be a sequence of paths")
    for index, path in enumerate(material_paths, 1):
        source, raw, shoe, _ = _read(path)
        if _json_hash(_nonmaterial_payload(shoe.raw)) != geometry_hash:
            raise ValueError(
                f"Imported material must have identical geometry and all nonmaterial physical fields: {source}"
            )
        cases.append((f"imported_{index:02d}", "imported", {}, source, raw))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    written = []
    try:
        for variant_id, kind, factors, source, payload in cases:
            snapshot = output / f"{variant_id}.json"
            with snapshot.open("xb") as stream:
                written.append(snapshot)
                stream.write(payload)
            _, _, shoe, derived = _read(snapshot)
            material = shoe.raw["constitutive_model"]
            synthetic = kind in {"modulus", "relaxation"}
            source_hash = baseline_sha256 if synthetic else hashlib.sha256(payload).hexdigest()
            records.append(
                {
                    "id": variant_id,
                    "path": str(snapshot),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "material_identity": _json_hash(material),
                    "geometry_identity": geometry_hash,
                    "nonmaterial_identity": geometry_hash,
                    "baseline": kind == "baseline",
                    "synthetic": synthetic,
                    "type": kind,
                    "factors": factors,
                    "constitutive_type": material["type"],
                    "parameters": asdict(shoe.material),
                    "derived_quantities": derived,
                    "artifact_derived_quantities": copy.deepcopy(material.get("derived_quantities", {})),
                    "source_sha256": source_hash,
                    "qualification": "not_validated" if synthetic else "source_claims_not_requalified",
                    "provenance": {
                        "source_path": str(source),
                        "source_sha256": source_hash,
                        "baseline_sha256": baseline_sha256,
                        "baseline_claims_apply_to": "baseline_only",
                    },
                    "warning": WARNING if kind != "baseline" else "Baseline snapshot; no new validation is performed.",
                }
            )
    except Exception:
        # Leave caller-owned files alone, including a racing destination.
        for snapshot in written:
            snapshot.unlink(missing_ok=True)
        raise
    return records


def _scale_numbers(value: Any, factor: float) -> Any:
    if isinstance(value, dict):
        return {key: _scale_numbers(item, factor) for key, item in value.items()}
    if isinstance(value, list):
        return [_scale_numbers(item, factor) for item in value]
    return value * factor
