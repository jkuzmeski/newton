# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure host-side preparation shared by human-shoe replay and dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from newton.opensim import OsimImportResult
from projects.digital_instron_v2.dynamics import FoundationConfig, FoundationGeometry, build_foundation_geometry
from projects.digital_instron_v2.geometry import load_mesh

from .adapter import ResolvedHumanShoeAttachment, attach_sole_geometry, resolve_attachment
from .contracts import FootShoeAttachmentContract, load_manifest


@dataclass(frozen=True, slots=True)
class PreparedAttachedSole:
    """Host-side shoe mesh and foundation columns in the carrier body frame."""

    resolved: ResolvedHumanShoeAttachment
    foundation_geometry: FoundationGeometry
    midsole_vertices: np.ndarray
    midsole_indices: np.ndarray
    column_bottom_local: np.ndarray
    column_top_local: np.ndarray
    column_rest_len: np.ndarray
    column_area: np.ndarray
    alignment_rms_m: float
    alignment_max_m: float


def prepare_attached_sole(
    import_result: OsimImportResult,
    attachment: FootShoeAttachmentContract,
    manifest_path: str | Path,
) -> PreparedAttachedSole:
    """Load and attach a calibrated Digital Instron sole without device allocation."""
    manifest_path = Path(manifest_path).resolve()
    resolved = resolve_attachment(import_result, attachment)
    geometry = build_foundation_geometry(manifest_path)
    manifest = load_manifest(manifest_path)
    midsole = load_mesh(manifest_path.parent / manifest.midsole_mesh, 0.001)
    midsole_vertices = np.asarray(midsole.vertices, dtype=np.float64).copy()
    midsole_vertices[:, 2] -= geometry.z_shift_m
    top_interface = np.column_stack([geometry.uv_m, geometry.z_free_m])
    top_reference = np.broadcast_to(top_interface.mean(axis=0), midsole_vertices.shape)
    attached_mesh = attach_sole_geometry(resolved, midsole_vertices, top_reference)
    columns = attach_sole_geometry(
        resolved,
        np.column_stack([geometry.uv_m, geometry.z_bottom_m]),
        top_interface,
    )
    if columns.alignment_max_m > 0.5 * geometry.spacing_m + 1.0e-9:
        raise ValueError(
            f"shoe-top contact alignment residual {columns.alignment_max_m:.6f} m "
            f"exceeds half the {geometry.spacing_m:.6f} m column spacing"
        )
    count = len(columns.bottom_local)
    return PreparedAttachedSole(
        resolved=resolved,
        foundation_geometry=geometry,
        midsole_vertices=np.asarray(attached_mesh.bottom_local, dtype=np.float32),
        midsole_indices=np.asarray(midsole.faces, dtype=np.int32).reshape(-1),
        column_bottom_local=np.asarray(columns.bottom_local, dtype=np.float32),
        column_top_local=np.asarray(columns.top_local, dtype=np.float32),
        column_rest_len=np.asarray(columns.rest_len, dtype=np.float32),
        column_area=np.full(count, geometry.area_m2, dtype=np.float32),
        alignment_rms_m=columns.alignment_rms_m,
        alignment_max_m=columns.alignment_max_m,
    )


def make_human_shoe_foundation_config() -> FoundationConfig:
    """Return the versioned non-calibration contact settings used by human-shoe runs."""
    return FoundationConfig(
        stretch_floor=0.05,
        normal_damping=40.0,
        friction_stiffness=2.0e4,
        friction=20.0,
        mu=1.0,
    )
