# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Project-local adapter between OpenSim foot contacts and shoe geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from newton.opensim import OsimFrameConverter, OsimImportResult

from .contracts import FootShoeAttachmentContract

_DIGITAL_SOLE_TO_OSIM_LOCAL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

OSIM_LOCAL_TO_Z_UP_JUMP_BASIS = OsimFrameConverter().matrix


@dataclass(frozen=True, slots=True)
class FootContactReference:
    """Validated OpenSim contact reference in the foot body frame [m]."""

    foot_body_name: str
    contact_geometry_names: tuple[str, ...]
    support_points_in_foot_m: np.ndarray
    origin_in_foot_m: np.ndarray


@dataclass(frozen=True, slots=True)
class ResolvedHumanShoeAttachment:
    """Resolved foot-to-shoe attachment in Newton body-index space."""

    foot_body_index: int
    shoe_carrier_body_index: int
    reference: FootContactReference
    shoe_to_foot: np.ndarray


@dataclass(frozen=True, slots=True)
class AttachedSoleGeometry:
    """Shoe sole geometry transformed into the OpenSim foot frame."""

    bottom_local: np.ndarray
    top_local: np.ndarray
    rest_len: np.ndarray
    alignment_rms_m: float
    alignment_max_m: float


def _as_vec3(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _as_points(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must have shape [N, 3]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _euler_xyz_matrix_deg(rotation_deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rotation_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rx_m @ ry_m @ rz_m


def _to_homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    xform = np.eye(4, dtype=np.float64)
    xform[:3, :3] = rotation
    xform[:3, 3] = translation
    return xform


def resolve_attachment(
    import_result: OsimImportResult,
    contract: FootShoeAttachmentContract,
) -> ResolvedHumanShoeAttachment:
    """Resolve the named OpenSim foot and shoe carrier bodies and contact points."""

    body_index = import_result.body_index
    if contract.foot_body_name not in body_index:
        raise KeyError(f"foot body '{contract.foot_body_name}' not found")
    if contract.shoe_carrier_body_name not in body_index:
        raise KeyError(f"shoe carrier body '{contract.shoe_carrier_body_name}' not found")

    foot_body_index = body_index[contract.foot_body_name]
    shoe_carrier_body_index = body_index[contract.shoe_carrier_body_name]
    if foot_body_index != shoe_carrier_body_index:
        raise ValueError(
            "human_shoe_experiment_1 requires foot_body_name and shoe_carrier_body_name to resolve to the same body"
        )

    contact_geometry_names: list[str] = []
    support_points: list[np.ndarray] = []
    for geom in import_result.model.contact_geometry:
        if geom.body != contract.foot_body_name:
            continue
        if geom.type == "ContactSphere":
            radius = float(geom.radius)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"contact sphere '{geom.name}' radius must be positive and finite")
            contact_geometry_names.append(geom.name)
            support_points.append(_as_vec3("location", geom.location) - np.array([0.0, radius, 0.0]))
        elif geom.type == "ContactMesh":
            contact_geometry_names.append(geom.name)
            support_points.append(_as_vec3("location", geom.location))
        else:
            raise ValueError(f"unsupported contact geometry type '{geom.type}' on foot body")

    if not contact_geometry_names:
        raise ValueError(f"no contact geometry attached to foot body '{contract.foot_body_name}'")

    support_points_in_foot_m = np.stack(support_points, axis=0)
    origin_in_foot_m = np.mean(support_points_in_foot_m, axis=0)
    reference = FootContactReference(
        foot_body_name=contract.foot_body_name,
        contact_geometry_names=tuple(contact_geometry_names),
        support_points_in_foot_m=support_points_in_foot_m,
        origin_in_foot_m=origin_in_foot_m,
    )

    basis = _DIGITAL_SOLE_TO_OSIM_LOCAL
    rotation = _euler_xyz_matrix_deg(_as_vec3("rotation_deg", contract.rotation_deg)) @ basis
    translation = origin_in_foot_m + _as_vec3("translation_m", contract.translation_m)
    shoe_to_foot = _to_homogeneous(rotation, translation)
    return ResolvedHumanShoeAttachment(
        foot_body_index=foot_body_index,
        shoe_carrier_body_index=shoe_carrier_body_index,
        reference=reference,
        shoe_to_foot=shoe_to_foot,
    )


def attach_sole_geometry(
    resolved: ResolvedHumanShoeAttachment,
    bottom_local: np.ndarray,
    top_local: np.ndarray,
    *,
    output_basis: np.ndarray | None = None,
) -> AttachedSoleGeometry:
    """Place sole geometry in the resolved OpenSim foot frame."""

    bottom = _as_points("bottom_local", bottom_local)
    top = _as_points("top_local", top_local)
    if bottom.shape != top.shape:
        raise ValueError("bottom_local and top_local must have matching shapes")

    top_centroid = np.mean(top, axis=0)
    rel_bottom = bottom - top_centroid
    rel_top = top - top_centroid

    rot = resolved.shoe_to_foot[:3, :3]
    trans = resolved.shoe_to_foot[:3, 3]
    bottom_out = rel_bottom @ rot.T + trans
    top_out = rel_top @ rot.T + trans

    if output_basis is not None:
        basis = np.asarray(output_basis, dtype=np.float64)
        if basis.shape != (3, 3):
            raise ValueError("output_basis must have shape (3, 3)")
        if not np.all(np.isfinite(basis)):
            raise ValueError("output_basis must contain finite values")
        bottom_out = bottom_out @ basis.T
        top_out = top_out @ basis.T

    rest_len = np.linalg.norm(bottom_out - top_out, axis=1)
    support_points = resolved.reference.support_points_in_foot_m
    if output_basis is not None:
        support_points = support_points @ basis.T
    alignment_distances = np.min(
        np.linalg.norm(support_points[:, None, :] - top_out[None, :, :], axis=2),
        axis=1,
    )
    return AttachedSoleGeometry(
        bottom_local=bottom_out,
        top_local=top_out,
        rest_len=rest_len,
        alignment_rms_m=float(np.sqrt(np.mean(alignment_distances**2))),
        alignment_max_m=float(np.max(alignment_distances)),
    )
