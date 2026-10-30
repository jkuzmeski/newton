# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample vertical material columns from triangle meshes."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ColumnGrid:
    """Midsole columns and their mesh frame."""

    uv_m: np.ndarray
    bottom_m: np.ndarray
    top_m: np.ndarray
    area_m2: np.ndarray  # midsole footprint area of each column [m^2]
    spacing_m: float
    thickness_axis: int

    @property
    def slack_m(self) -> np.ndarray:
        """Return rest lengths."""

        return self.top_m - self.bottom_m


def _positive_rays(mesh, uv_m: np.ndarray, thickness_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Place rays [m] below the mesh and point them along the positive thickness axis."""
    plane_axes = [axis for axis in range(3) if axis != thickness_axis]
    origins = np.zeros((len(uv_m), 3))
    origins[:, plane_axes] = uv_m
    origins[:, thickness_axis] = mesh.bounds[0, thickness_axis] - 0.01
    directions = np.zeros_like(origins)
    directions[:, thickness_axis] = 1.0
    return origins, directions


def raycast(
    mesh: Any,
    uv_m: np.ndarray,
    thickness_axis: int,
    normal_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first and last mesh intersections at each footprint point."""

    origins, directions = _positive_rays(mesh, uv_m, thickness_axis)
    locations, rays, triangles = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    coordinate = locations[:, thickness_axis]
    bottom = np.full(len(uv_m), np.inf)
    top = np.full(len(uv_m), -np.inf)
    bottom_normal = np.zeros(len(uv_m))
    top_normal = np.zeros(len(uv_m))
    np.minimum.at(bottom, rays, coordinate)
    np.maximum.at(top, rays, coordinate)
    normals = mesh.face_normals[triangles, thickness_axis]
    for ray in np.unique(rays):
        hits = np.flatnonzero(rays == ray)
        hits = hits[np.argsort(coordinate[hits])]
        bottom_normal[ray] = normals[hits[0]]
        top_normal[ray] = normals[hits[-1]]
    valid = (
        np.isfinite(bottom)
        & np.isfinite(top)
        & (top > bottom)
        & (bottom_normal <= -normal_threshold)
        & (top_normal >= normal_threshold)
    )
    return np.where(valid, bottom, np.nan), np.where(valid, top, np.nan)


def build_column_grid(mesh: Any, spacing_m: float, min_thickness_m: float = 0.005) -> ColumnGrid:
    """Sample the midsole footprint at uniform spacing."""

    thickness_axis = int(np.argmin(np.ptp(mesh.vertices, axis=0)))
    plane_axes = [axis for axis in range(3) if axis != thickness_axis]
    lower, upper = mesh.bounds[:, plane_axes]
    u = np.arange(lower[0], upper[0] + spacing_m / 2.0, spacing_m)
    v = np.arange(lower[1], upper[1] + spacing_m / 2.0, spacing_m)
    uu, vv = np.meshgrid(u, v)
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    bottom, top = raycast(mesh, uv, thickness_axis, normal_threshold=0.1)
    valid = np.isfinite(bottom) & (top - bottom >= min_thickness_m)
    count = int(valid.sum())
    return ColumnGrid(uv[valid], bottom[valid], top[valid], np.full(count, spacing_m**2), spacing_m, thickness_axis)


def neighbor_indices(uv: np.ndarray, grid_uv: np.ndarray, spacing: float) -> np.ndarray:
    """Return the Pasternak 4-neighbour index table for the active columns.

    Each of the four in-plane neighbours (-u, +u, -v, +v) is a non-negative
    active-column index, ``-1`` for a footprint boundary, or ``-2`` for a midsole
    cell outside the active set. Both negative codes are free (zero-gradient)
    edges that carry no lateral coupling, so the shear layer redistributes load
    without creating it.
    """
    cells = [tuple(np.rint(p / spacing).astype(int)) for p in uv]
    index = {c: i for i, c in enumerate(cells)}
    full = {tuple(np.rint(p / spacing).astype(int)) for p in grid_uv}
    out = np.full((len(cells), 4), -2, dtype=np.int32)
    for i, (u, v) in enumerate(cells):
        for side, (du, dv) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            c = (u + du, v + dv)
            if c in index:
                out[i, side] = index[c]
            elif c not in full:
                out[i, side] = -1
    return out
