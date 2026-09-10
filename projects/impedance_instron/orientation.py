# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply an explicit shoe-side convention for the sagittal Impedance Instron.

This is an in-memory project adapter, not a change to Digital Instron calibration.
In the right-handed +X-forward/+Y-left/+Z-up fixture frame, a side change reflects
geometry through Y=0. It cannot establish anatomical side or marker registration.

The supplied ``*_left`` artifact is not a reliable side label. Its documented-left
STL passes through the exporter's odd ``[2, 1, 0]`` axis permutation without a
winding reversal. The resulting last has reversed surface orientation; its outline
and the midsole suggest a right-side interpretation, but lack certified anatomical
landmarks. ``source_side`` therefore must be supplied by the caller, never inferred.
See ``DigitalInstron/MODEL_FINDINGS.md`` (Geometry Findings),
``ASSET_PROVENANCE.md``, and ``projects/digital_instron_v2/geometry.py:load_mesh``.

Only the exact audited last mesh and recorded export provenance receive the known
winding repair. An open mesh's signed volume is not used to guess its orientation.
Other meshes retain their input surface orientation, even if it was inward.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from typing import Any

import numpy as np

from projects.digital_shoe import DigitalShoe, VisualMesh

# Pin the audited baked coordinates AND triangle order, not a filename or volume.
# Digest: shape as little-endian int64, then vertices as float64; triangle shape
# as int64, then indices as int32. All arrays are serialized in C order.
_AUDITED_LAST_MESH_SHA256 = "6fc77e1bdb51aea2d74d220559b9fac0ae07d8f8cf963b8c6b7ba51bfffd9c99"
_AUDITED_SOURCE_HASHES = {
    "manifest": "c3c6fc7b11dc352bdc3d72f28797d901796aab4cabb76a71dd725ee8a7cc8b9c",
    "midsole_geometry": "9347e6ad2bdeb4c7152cf5b7c50f784a5875d45df8656a23a567b9fb72c4753e",
    "fullfoot_185ms_indenter_geometry": "39b832c0011ea05b2b3e7afc65ad099e98d7ed2285b60607806be88a4bade508",
}
_AUDITED_GENERATOR = "projects.digital_instron_v2.export_digital_shoe"


def _mesh_sha256(mesh: VisualMesh) -> str:
    """Identify exact decoded geometry, including its original winding."""
    digest = hashlib.sha256()
    for array, dtype in ((mesh.vertices_m, "<f8"), (mesh.triangles, "<i4")):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(array, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def _audited_last_matches(shoe: DigitalShoe, mesh_digests: dict[str, str]) -> bool:
    """Match both audited baked geometry and recorded source-chain hashes."""
    if mesh_digests.get("fullfoot_last") != _AUDITED_LAST_MESH_SHA256:
        return False
    if shoe.provenance.get("generator") != _AUDITED_GENERATOR:
        return False
    records = shoe.provenance.get("source_files", [])
    for role, expected in _AUDITED_SOURCE_HASHES.items():
        matches = [record.get("sha256") for record in records if record.get("role") == role]
        if matches != [expected]:
            return False
    return True


def orient_shoe(
    shoe: DigitalShoe, target_side: str = "left", *, source_side: str
) -> tuple[DigitalShoe, dict[str, Any]]:
    """Copy a shoe into an explicitly chosen engineering side convention.

    The frame remains right-handed, +X forward, +Y left and +Z up. A side change
    reflects polar coordinates through Y=0, does not reorder columns, and swaps
    the ``-Y``/``+Y`` neighbor slots (slot order is ``-X,+X,-Y,+Y``). Scalar heights,
    areas, rest lengths and material parameters do not change. All visual meshes
    reverse triangle winding on reflection to preserve their surface orientation.
    Normals derived from the triangles therefore reflect as polar vectors.

    Independently of the side change, reverse the known inward ``fullfoot_last``
    triangles only when exact baked geometry and recorded source hashes match
    the pinned audit. This restores the source surface orientation, not a
    closed-solid certificate. Unknown input winding is never guessed or repaired.

    In a flat-ground, normal-only rig at Y=0 with pitch about +Y, reflection keeps
    every column's world X/Z, compression, Fz and COP X. It reverses COP Y and
    roll moment, not sagittal pitch. This is not a general 3D dynamics invariant.

    Args:
        shoe: Loaded source artifact. Its arrays and original file stay unchanged.
        target_side: Fixture-side convention, either ``"left"`` or ``"right"``.
        source_side: Required interpretation of the baked input geometry, either
            ``"left"`` or ``"right"``. For the audited supplied artifact, ``"right"``
            is an engineering interpretation informed by its outline and export
            history, not independently certified anatomical side.

    Returns:
        A detached :class:`DigitalShoe` copy and JSON-compatible orientation
        metadata. The source shoe ID and validation record remain unchanged.
        The copied ``raw`` geometry agrees with the decoded arrays; its provenance
        includes ``impedance_instron_orientation``. Source validation is retained
        as evidence, not extended to a new anatomical side or acquisition.

    Raises:
        ValueError: A side is unsupported, the declared frame is incompatible,
            or a previously oriented shoe has a conflicting source interpretation.
    """
    for name, side in (("source_side", source_side), ("target_side", target_side)):
        if side not in ("left", "right"):
            raise ValueError(f"{name} must be 'left' or 'right'; no side is inferred from labels")
    coordinate = shoe.raw.get("coordinate_system", {})
    if (
        coordinate.get("handedness") != "right"
        or coordinate.get("up_axis") != "+Z"
        or coordinate.get("length_unit") != "m"
    ):
        raise ValueError("orient_shoe requires a declared right-handed +Z-up frame in metres")
    prior = shoe.provenance.get("impedance_instron_orientation")
    if prior and prior.get("target_shoe_side") != source_side:
        raise ValueError("source_side conflicts with the previous orientation target_shoe_side")

    reflected = source_side != target_side
    multiplier = np.array([1.0, -1.0 if reflected else 1.0, 1.0])
    slots = [0, 1, 3, 2] if reflected else [0, 1, 2, 3]
    mesh_digests = {name: _mesh_sha256(mesh) for name, mesh in shoe.visual_meshes.items()}
    repair_last = _audited_last_matches(shoe, mesh_digests)
    repairs = ["fullfoot_last"] if repair_last else []
    mesh_winding = {
        name: {
            "input_geometry_sha256": mesh_digests[name],
            "audited_input_repair_applied": name in repairs,
            "reflection_winding_reversal_applied": reflected,
            "net_triangle_reversal": reflected != (name in repairs),
            "surface_orientation": (
                "audited_source_orientation_restored_open_mesh_not_solid_certified"
                if name in repairs
                else "input_orientation_preserved_not_certified"
            ),
        }
        for name in shoe.visual_meshes
    }
    metadata: dict[str, Any] = {
        "source_shoe_id": shoe.shoe_id,
        "source_shoe_side": source_side,
        "target_shoe_side": target_side,
        "side_interpretation": "explicit_engineering_convention",
        "source_side_inferred_from_label": False,
        "anatomical_side_validated": False,
        "anatomical_registration_validated": False,
        "reflection_applied": reflected,
        "local_transform_matrix": np.diag(multiplier).tolist(),
        "transform_determinant": -1 if reflected else 1,
        "neighbor_slot_order": ["-X", "+X", "-Y", "+Y"],
        "audited_last_winding_contract_matched": repair_last,
        "winding_repairs": repairs,
        "mesh_winding": mesh_winding,
        "claim_boundary": (
            "Side is an explicit engineering convention, not anatomical certification. "
            "Source labels do not establish baked chirality. Source validation and material "
            "parameters are retained, not extended to the chosen side or acquisition. "
            "A Y reflection does not change sagittal pitch or repair marker-to-shoe registration."
        ),
    }
    if prior:
        metadata["previous_orientation"] = copy.deepcopy(prior)

    result = copy.deepcopy(shoe)
    bed = replace(
        result.column_bed,
        anchor_bottom_m=result.column_bed.anchor_bottom_m * multiplier,
        neighbors=result.column_bed.neighbors[:, slots].copy(),
    )
    meshes = {
        name: replace(
            mesh,
            vertices_m=mesh.vertices_m * multiplier,
            triangles=mesh.triangles[:, [0, 2, 1]].copy()
            if mesh_winding[name]["net_triangle_reversal"]
            else mesh.triangles.copy(),
        )
        for name, mesh in result.visual_meshes.items()
    }
    fixtures = {
        name: replace(
            fixture,
            carrier_anchor_m=fixture.carrier_anchor_m * multiplier,
            neighbors=fixture.neighbors[:, slots].copy(),
        )
        for name, fixture in result.instron_fixtures.items()
    }
    # Keep portable raw data coherent, rather than leaving source coordinates in it.
    result.raw["column_bed"]["anchor_bottom_m"] = bed.anchor_bottom_m.tolist()
    result.raw["column_bed"]["neighbors"] = bed.neighbors.tolist()
    for name, mesh in meshes.items():
        result.raw["visual_meshes"][name]["vertices_m"] = mesh.vertices_m.tolist()
        result.raw["visual_meshes"][name]["triangles"] = mesh.triangles.tolist()
    for name, fixture in fixtures.items():
        result.raw["instron_fixtures"][name]["carrier_anchor_m"] = fixture.carrier_anchor_m.tolist()
        result.raw["instron_fixtures"][name]["neighbors"] = fixture.neighbors.tolist()
    result.provenance["impedance_instron_orientation"] = copy.deepcopy(metadata)
    result.raw["provenance"] = result.provenance
    return replace(result, column_bed=bed, visual_meshes=meshes, instron_fixtures=fixtures), metadata
