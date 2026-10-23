# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Turn RTINGS midsole ``.glb`` scans into canonical left/right column beds.

Pass any mix of ``.glb`` files and folders of ``.glb`` files::

    uv run --extra dev python -m projects.digital_instron_v2.shoe_prep a-3d-internal.glb b/source.glb folder/ --out-root DigitalInstron

The slug is the file stem without ``-3d-internal``, or the parent folder name
for ``source.glb``.

Canonical frame: +x heel to toe, +z up, +y = z cross x (left of the walking
direction, so +y is medial for a right foot and lateral for a left foot). The
footprint bounding box is centered in x/y and the lowest point sits at z = 0,
matching :func:`geometry.load_mesh`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import ColumnGrid, build_column_grid
from .registration import canonical_frame

__all__ = [
    "default_slug",
    "foot_side",
    "load_midsole",
    "mirror_grid",
    "prepare_shoe",
    "sample_columns",
]

SPACING_M = 0.005
MIN_THICKNESS_M = 0.005
# Sole-unit plausibility for the RTINGS test size (US men's 9).
LENGTH_RANGE_M = (0.255, 0.330)
WIDTH_RANGE_M = (0.080, 0.130)
COLUMN_COUNT_RANGE = (600, 1400)
MAX_THICKNESS_M = 0.070
MEDIAN_THICKNESS_RANGE_M = (0.015, 0.050)
MIN_TOE_SPRING_M = 0.005
MIN_ARCH_RATIO = 1.5
MIN_ARCH_DIFFERENCE_M = 0.003
TOE_OFFSET_CHECK_M = 0.005
MAX_DEBRIS_FACE_FRACTION = 0.001
NEIGHBOR_TOLERANCE_M = 0.005
MAX_REPAIRED_FRACTION = 0.02
CELL_RAYS = 9
"""Rays per cell side for the cell means; odd, so the center ray is one of them."""
_CODE_FILES = ("shoe_prep.py", "geometry.py", "registration.py")
_OWNED_OUTPUTS = (
    "shoe.json",
    "qc_top.png",
    *(
        f"{side}/{name}"
        for side in ("left", "right")
        for name in ("column_grid.npz", "preflight.json", "manifest_geometry_fragment.json")
    ),
)


class ShoePrepError(ValueError):
    """A shoe was rejected; the message states why."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for name in _CODE_FILES:
        digest.update((Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _boundary_stats(mesh: Any) -> dict[str, float]:
    """Count open edges [-] and their total length [m]."""
    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = unique[counts == 1]
    length = np.linalg.norm(mesh.vertices[open_edges[:, 0]] - mesh.vertices[open_edges[:, 1]], axis=1).sum()
    return {
        "open_edges": int(len(open_edges)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "open_edge_length_m": float(length),
    }


def load_midsole(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a one-body midsole scan in meters and record its repair statistics.

    UV seams are welded, single-triangle and single-quad holes are filled, and
    tiny loose fragments are dropped. The scan is rejected when it holds more
    than one substantial body or its winding is inconsistent.

    Args:
        path: ``.glb`` (or any trimesh-readable) midsole scan.

    Returns:
        The midsole mesh in source axes [m] and a QC dictionary.
    """
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    parts = []
    for node in sorted(scene.graph.nodes_geometry):
        name = scene.graph[node][1]
        material = getattr(getattr(scene.geometry[name], "visual", None), "material", None)
        parts.append(
            {
                "node": node,
                "geometry": name,
                "faces": len(scene.geometry[name].faces),
                "material": getattr(material, "name", None),
            }
        )
    if not parts:
        raise ShoePrepError("scene holds no triangle geometry")
    raw = scene.to_geometry()
    longest = float(np.ptp(raw.vertices, axis=0).max())
    # glTF declares meters, but RTINGS exports millimeters; accept either.
    if 0.15 <= longest <= 0.40:
        scale = 1.0
    elif 150.0 <= longest <= 400.0:
        scale = 0.001
    else:
        raise ShoePrepError(f"longest extent {longest:g} is neither a shoe in meters nor in millimeters")
    mesh = trimesh.Trimesh(raw.vertices * scale, raw.faces, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    bodies = sorted(mesh.split(only_watertight=False, repair=False), key=lambda body: -len(body.faces))
    debris_faces = sum(len(body.faces) for body in bodies[1:])
    if debris_faces > MAX_DEBRIS_FACE_FRACTION * len(mesh.faces):
        sizes = [len(body.faces) for body in bodies[:5]]
        raise ShoePrepError(f"scan holds {len(bodies)} bodies (largest face counts {sizes}); expected one midsole")
    mesh = bodies[0]
    before = _boundary_stats(mesh)
    mesh.fill_holes()
    after = _boundary_stats(mesh)
    if not mesh.is_winding_consistent:
        raise ShoePrepError("inconsistent face winding; column normal checks would be meaningless")
    if mesh.volume < 0.0:
        mesh.invert()
    qc = {
        "scene_parts": parts,
        "unit_scale_to_m": scale,
        "faces": int(len(mesh.faces)),
        "vertices_after_weld": int(len(mesh.vertices)),
        "dropped_fragments": len(bodies) - 1,
        "dropped_fragment_faces": int(debris_faces),
        "boundary_before_fill": before,
        "boundary_after_fill": after,
        "watertight": bool(mesh.is_watertight),
        "volume_m3": float(mesh.volume),
    }
    return mesh, qc


def _transformed(mesh: Any, rotation: np.ndarray) -> tuple[Any, np.ndarray]:
    """Rotate a mesh copy and center it like :func:`geometry.load_mesh`; return mesh and 4x4 map."""
    import trimesh

    points = mesh.vertices @ rotation.T
    offset = np.zeros(3)
    offset[:2] = -(points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2.0
    offset[2] = -points[:, 2].min()
    faces = mesh.faces if np.linalg.det(rotation) > 0.0 else mesh.faces[:, ::-1]
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = offset
    return trimesh.Trimesh(points + offset, faces, process=False), matrix


def _toe_spring(grid: ColumnGrid) -> float:
    """Rise [m] of the sole mid-surface at the toe above its lowest forefoot section."""
    x = grid.uv_m[:, 0]
    fraction = (x - x.min()) / np.ptp(x)
    middle = (grid.top_m + grid.bottom_m) / 2.0

    def band(lo: float, hi: float) -> float:
        return float(np.median(middle[(fraction >= lo) & (fraction < hi)]))

    return band(0.92, 0.99) - min(band(lo, lo + 0.05) for lo in np.arange(0.40, 0.85, 0.05))


def _canonical_mesh(mesh: Any) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Map a midsole into the canonical frame using only geometric evidence."""
    extents = np.ptp(mesh.vertices, axis=0)
    length_axis, up_axis = int(np.argmax(extents)), int(np.argmin(extents))
    side_axis = 3 - length_axis - up_axis
    if extents[up_axis] > 0.8 * extents[side_axis]:
        raise ShoePrepError(f"height {extents[up_axis]:.3f} m is not clearly below width {extents[side_axis]:.3f} m")
    x_axis, z_axis = np.eye(3)[length_axis], np.eye(3)[up_axis]
    rotation = np.vstack((x_axis, np.cross(z_axis, x_axis), z_axis))
    preliminary, _ = _transformed(mesh, rotation)
    grid = build_column_grid(preliminary, SPACING_M, MIN_THICKNESS_M)
    if len(grid.uv_m) < 50:
        raise ShoePrepError(f"only {len(grid.uv_m)} columns in the preliminary frame")
    try:
        frame = canonical_frame(
            preliminary.vertices, kind="shoe", shoe_uv_m=grid.uv_m, shoe_top_m=grid.top_m, shoe_bottom_m=grid.bottom_m
        )
    except ValueError as error:
        raise ShoePrepError(f"heel/toe not established: {error}") from error
    # The shoe frame keeps the preliminary up sign; it is a rotation about z here.
    rotation = frame.rotation @ rotation
    rotated = replace(grid, uv_m=grid.uv_m @ frame.rotation[:2, :2].T)
    spring = _toe_spring(rotated)
    if abs(spring) < MIN_TOE_SPRING_M:
        raise ShoePrepError(f"up direction not established: toe spring {spring * 1000:.1f} mm")
    flipped = spring < 0.0
    if flipped:
        rotation = np.diag([1.0, -1.0, -1.0]) @ rotation
        spring = _toe_spring(replace(rotated, top_m=-rotated.bottom_m, bottom_m=-rotated.top_m))
    canonical, matrix = _transformed(mesh, rotation)
    evidence = {
        "source_axes_length_side_up": [length_axis, side_axis, up_axis],
        "heel_toe_cues_m": frame.evidence,
        "toe_spring_m": spring,
        "up_flipped_from_source": flipped,
    }
    return canonical, matrix, evidence


def foot_side(vertices: np.ndarray) -> tuple[str, dict[str, float]]:
    """Classify a canonical midsole as a left or right shoe from its outline.

    The medial outline has the deeper arch indentation (largest gap to the
    convex hull between 30 % and 70 % of the length). When the toe tip is
    clearly off the heel centerline it must lean to the same side.

    Args:
        vertices: Canonical midsole vertices [m], shape [N, 3].

    Returns:
        ``"right"`` or ``"left"`` and the cue values [m].
    """
    from scipy.spatial import ConvexHull

    x, y = vertices[:, 0], vertices[:, 1]
    fraction = (x - x.min()) / np.ptp(x)
    bins = np.minimum((fraction * 100).astype(int), 99)
    centers, low, high = [], [], []
    for index in np.unique(bins):
        ys = y[bins == index]
        centers.append(x.min() + (index + 0.5) / 100 * np.ptp(x))
        low.append(ys.min())
        high.append(ys.max())
    centers, low, high = map(np.asarray, (centers, low, high))
    outline = np.vstack((np.column_stack((centers, low)), np.column_stack((centers, high))))
    hull = outline[ConvexHull(outline).vertices]
    edges = list(zip(hull, np.roll(hull, -1, axis=0), strict=True))

    def hull_span(position: float) -> tuple[float, float]:
        hits = [
            p[1] + (position - p[0]) / (q[0] - p[0]) * (q[1] - p[1])
            for p, q in edges
            if p[0] != q[0] and (p[0] - position) * (q[0] - position) <= 0.0
        ]
        return min(hits), max(hits)

    center_fraction = (centers - x.min()) / np.ptp(x)
    middle = (center_fraction > 0.3) & (center_fraction < 0.7)
    spans = [hull_span(c) for c in centers[middle]]
    arch_positive = float(max(span[1] - h for span, h in zip(spans, high[middle], strict=True)))
    arch_negative = float(max(l - span[0] for span, l in zip(spans, low[middle], strict=True)))
    heel = (center_fraction > 0.05) & (center_fraction < 0.25)
    heel_center = float(np.median((low + high)[heel]) / 2.0)
    toe_offset = float(np.mean(y[fraction > 0.98]) - heel_center)
    cues = {"arch_indent_pos_y_m": arch_positive, "arch_indent_neg_y_m": arch_negative, "toe_offset_y_m": toe_offset}
    deep, shallow = max(arch_positive, arch_negative), min(arch_positive, arch_negative)
    if deep - shallow < MIN_ARCH_DIFFERENCE_M or deep < MIN_ARCH_RATIO * shallow:
        raise ShoePrepError(
            f"medial side not established: arch indents {arch_positive * 1000:.1f} / {arch_negative * 1000:.1f} mm"
        )
    medial_sign = 1.0 if arch_positive > arch_negative else -1.0
    if abs(toe_offset) >= TOE_OFFSET_CHECK_M and np.sign(toe_offset) != medial_sign:
        raise ShoePrepError(f"arch and toe-tip side cues disagree (toe offset {toe_offset * 1000:.1f} mm)")
    return ("right" if medial_sign > 0.0 else "left"), cues


def _cell_means(mesh: Any, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean bottom and top over the rays of each cell that cross material, and that ray fraction."""
    offsets = ((np.arange(CELL_RAYS) + 0.5) / CELL_RAYS - 0.5) * SPACING_M
    du, dv = (a.ravel() for a in np.meshgrid(offsets, offsets))
    points = (uv[:, None, :] + np.column_stack((du, dv))[None]).reshape(-1, 2)
    origins = np.column_stack((points, np.full(len(points), mesh.bounds[0, 2] - 0.01)))
    directions = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    locations, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    low = np.full(len(points), np.inf)
    high = np.full(len(points), -np.inf)
    np.minimum.at(low, rays, locations[:, 2])
    np.maximum.at(high, rays, locations[:, 2])
    low, high = low.reshape(len(uv), -1), high.reshape(len(uv), -1)
    covered = high > low
    count = covered.sum(axis=1)
    bottom = np.where(covered, low, 0.0).sum(axis=1) / count
    top = np.where(covered, high, 0.0).sum(axis=1) / count
    return bottom, top, count / covered.shape[1]


def sample_columns(mesh: Any) -> tuple[ColumnGrid, dict[str, Any]]:
    """Sample columns with :func:`geometry.build_column_grid` and repair isolated defects.

    A lattice point inside the footprint (valid columns with enclosed gaps
    filled) whose ray yields no valid column although it crosses an open
    surface (odd hit count) or spans at least ``MIN_THICKNESS_M`` of material
    is a gap (open surface or a steep face failing the normal check); a ray
    with no hits there is a real through-hole and stays empty. A valid
    column that is an isolated thickness spike (thinner than all its neighbors
    or thicker than all of them by more than ``NEIGHBOR_TOLERANCE_M``) is
    suspect. Both are linearly interpolated from the trusted columns. More than ``MAX_REPAIRED_FRACTION`` repairs rejects the
    shoe because the footprint coverage is compromised.

    Each measured column is the mean over ``CELL_RAYS`` x ``CELL_RAYS`` rays
    across its cell: bottom and top are averaged over the rays that cross
    material, and the area is the cell area times that fraction. A cell on the
    outline whose center ray grazes the sloped sidewall thus keeps its covered
    area and mean thickness instead of the thin center value.

    Args:
        mesh: Canonical midsole mesh [m].

    Returns:
        The column grid and a QC dictionary.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import binary_fill_holes

    grid = build_column_grid(mesh, SPACING_M, MIN_THICKNESS_M)
    if grid.thickness_axis != 2:
        raise ShoePrepError("canonical mesh is not thinnest along z")
    lower, upper = mesh.bounds[:, :2]
    u = np.arange(lower[0], upper[0] + SPACING_M / 2.0, SPACING_M)
    v = np.arange(lower[1], upper[1] + SPACING_M / 2.0, SPACING_M)
    nx, ny = len(u), len(v)
    uu, vv = np.meshgrid(u, v)
    origins = np.column_stack((uu.ravel(), vv.ravel(), np.full(uu.size, mesh.bounds[0, 2] - 0.01)))
    directions = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
    locations, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    hits = np.bincount(rays, minlength=len(origins)).reshape(ny, nx)
    span = np.zeros(len(origins))
    np.maximum.at(span, rays, locations[:, 2] - mesh.bounds[0, 2])
    lowest = np.full(len(origins), np.inf)
    np.minimum.at(lowest, rays, locations[:, 2] - mesh.bounds[0, 2])
    span = (span - lowest).reshape(ny, nx)

    index = np.rint((grid.uv_m - lower) / SPACING_M).astype(int)
    bottom = np.full((ny, nx), np.nan)
    top = np.full((ny, nx), np.nan)
    bottom[index[:, 1], index[:, 0]] = grid.bottom_m
    top[index[:, 1], index[:, 0]] = grid.top_m
    valid = np.isfinite(bottom)

    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]

    def neighbors(a: np.ndarray, fill: float) -> np.ndarray:
        p = np.pad(a, 1, constant_values=fill)
        return np.stack([p[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx] for dy, dx in shifts])

    thickness = top - bottom
    near_thickness = neighbors(thickness, np.nan)
    near_count = np.sum(np.isfinite(near_thickness), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        near_low = np.nanmin(near_thickness, axis=0)
        near_high = np.nanmax(near_thickness, axis=0)
    footprint = binary_fill_holes(valid)
    # Thin columns with paired hits are real; an odd hit count means an open surface.
    gap = footprint & ~valid & (hits > 0) & ((hits % 2 == 1) | (span >= MIN_THICKNESS_M))
    spike = (thickness < near_low - NEIGHBOR_TOLERANCE_M) | (thickness > near_high + NEIGHBOR_TOLERANCE_M)
    outlier = valid & (near_count >= 5) & spike
    repair = gap | outlier
    if repair.sum() > MAX_REPAIRED_FRACTION * valid.sum():
        raise ShoePrepError(
            f"footprint coverage compromised: {int(gap.sum())} hole gaps and {int(outlier.sum())} suspect columns "
            f"out of {int(valid.sum())}"
        )
    trusted = valid & ~outlier
    if repair.any():
        fill = griddata(
            (uu[trusted], vv[trusted]), np.column_stack((bottom[trusted], top[trusted])), (uu[repair], vv[repair])
        )
        bottom[repair], top[repair] = fill[:, 0], fill[:, 1]
    keep = (valid | gap) & np.isfinite(bottom) & (top - bottom >= MIN_THICKNESS_M)
    order = np.flatnonzero(keep.ravel())
    uv = np.column_stack((uu.ravel(), vv.ravel()))[order]
    bottom, top = bottom.ravel()[order], top.ravel()[order]
    area = np.full(len(order), SPACING_M**2)
    measured = ~repair.ravel()[order]
    bottom[measured], top[measured], coverage = _cell_means(mesh, uv[measured])
    area[measured] *= coverage
    result = ColumnGrid(uv, bottom, top, area, SPACING_M, 2)
    odd = valid & (hits % 2 == 1)
    qc = {
        "lattice_points": int(hits.size),
        "rays_hitting_mesh": int(np.count_nonzero(hits)),
        "build_column_grid_columns": int(valid.sum()),
        "dropped_outside_or_thin": int(np.count_nonzero((hits > 0) & ~valid & ~gap)),
        "hole_gaps_filled": int(gap.sum()),
        "through_holes": int(np.count_nonzero(footprint & (hits == 0))),
        "suspect_columns_replaced": int(outlier.sum()),
        "odd_hit_count_columns": int(odd.sum()),
        "repaired_uv_m": np.column_stack((uu[repair], vv[repair])).round(9).tolist(),
        "partial_cells": int(np.count_nonzero(coverage < 1.0)),
        "covered_area_fraction": float(area.sum() / (len(order) * SPACING_M**2)),
        "columns": int(len(order)),
    }
    return result, qc


def mirror_grid(grid: ColumnGrid) -> ColumnGrid:
    """Mirror a canonical grid across the x-z plane (left <-> right shoe)."""
    uv = grid.uv_m * np.array([1.0, -1.0])
    order = np.lexsort((uv[:, 0], uv[:, 1]))
    return replace(
        grid, uv_m=uv[order], bottom_m=grid.bottom_m[order], top_m=grid.top_m[order], area_m2=grid.area_m2[order]
    )


def _grid_stats(grid: ColumnGrid) -> dict[str, float]:
    thickness = grid.slack_m * 1000.0
    x = grid.uv_m[:, 0]
    fraction = (x - x.min()) / np.ptp(x)
    heel = float(np.median(thickness[(fraction >= 0.08) & (fraction < 0.18)]))
    forefoot = float(np.median(thickness[(fraction >= 0.70) & (fraction < 0.80)]))
    return {
        "columns": len(x),
        "min_thickness_mm": float(thickness.min()),
        "median_thickness_mm": float(np.median(thickness)),
        "max_thickness_mm": float(thickness.max()),
        "heel_thickness_mm": heel,
        "forefoot_thickness_mm": forefoot,
        "drop_mm": heel - forefoot,
    }


def _gates(stats: dict[str, float], vertices: np.ndarray) -> dict[str, dict[str, Any]]:
    length, width = np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1])

    def gate(value: float, lo: float, hi: float) -> dict[str, Any]:
        return {"value": float(value), "range": [lo, hi], "passed": bool(lo <= value <= hi)}

    return {
        "column_count": gate(stats["columns"], *COLUMN_COUNT_RANGE),
        "max_thickness_mm": gate(stats["max_thickness_mm"], MIN_THICKNESS_M * 1000, MAX_THICKNESS_M * 1000),
        "median_thickness_mm": gate(stats["median_thickness_mm"], *(1000 * np.array(MEDIAN_THICKNESS_RANGE_M))),
        "footprint_length_mm": gate(length * 1000, *(1000 * np.array(LENGTH_RANGE_M))),
        "footprint_width_mm": gate(width * 1000, *(1000 * np.array(WIDTH_RANGE_M))),
        "drop_mm": gate(stats["drop_mm"], 0.0, 20.0),
    }


def _grid_sha256(grid: ColumnGrid) -> str:
    digest = hashlib.sha256()
    for array in (grid.uv_m, grid.bottom_m, grid.top_m, grid.area_m2):
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def _write_npz(path: Path, grid: ColumnGrid) -> None:
    """Write the column grid with fixed zip timestamps so bytes are reproducible."""
    arrays = {
        "uv_m": grid.uv_m,
        "bottom_m": grid.bottom_m,
        "top_m": grid.top_m,
        "area_m2": grid.area_m2,
        "spacing_m": np.float64(grid.spacing_m),
        "thickness_axis": np.int64(grid.thickness_axis),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), buffer.getvalue())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_figure(path: Path, grids: dict[str, ColumnGrid], repaired: dict[str, np.ndarray], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True)
    vmax = max(float(g.slack_m.max()) for g in grids.values()) * 1000
    for ax, side in zip(axes, ("left", "right"), strict=True):
        grid = grids[side]
        points = ax.scatter(
            grid.uv_m[:, 0] * 1000,
            grid.uv_m[:, 1] * 1000,
            c=grid.slack_m * 1000,
            marker="s",
            s=14,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        if len(repaired[side]):
            ax.scatter(repaired[side][:, 0] * 1000, repaired[side][:, 1] * 1000, marker="x", c="red", s=20)
        positive = "MEDIAL" if side == "right" else "LATERAL"
        negative = "LATERAL" if side == "right" else "MEDIAL"
        for text, xy, align in (
            ("HEEL", (0.005, 0.5), "left"),
            ("TOE", (0.995, 0.5), "right"),
            (positive, (0.5, 0.97), "center"),
            (negative, (0.5, 0.03), "center"),
        ):
            ax.text(*xy, text, transform=ax.transAxes, ha=align, va="center", fontweight="bold")
        stats = _grid_stats(grid)
        ax.set(
            title=f"{side} shoe: {stats['columns']} columns, drop {stats['drop_mm']:.1f} mm",
            xlabel="x heel → toe [mm]",
            ylabel="y [mm]",
            aspect="equal",
        )
        ax.margins(x=0.1, y=0.15)
    fig.colorbar(points, ax=axes, label="column rest thickness [mm]")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def default_slug(path: str | Path) -> str:
    """Return the shoe slug for ``<slug>-3d-internal.glb`` or ``<slug>/source.glb``."""
    path = Path(path)
    return path.resolve().parent.name if path.stem == "source" else path.stem.removesuffix("-3d-internal")


def prepare_shoe(
    glb_path: str | Path, out_root: str | Path, *, slug: str | None = None, product_name: str | None = None
) -> dict[str, Any]:
    """Prepare one shoe and write ``<out_root>/<slug>/``.

    Writes ``source.glb``, ``shoe.json``, ``qc_top.png`` and, per side,
    ``column_grid.npz``, ``preflight.json`` and ``manifest_geometry_fragment.json``.
    A rejected shoe gets ``shoe.json`` with ``geometry_status: "rejected"`` and
    a reason; stale side outputs are removed. All outputs are computed in a
    staging folder before any existing file is replaced.

    Args:
        glb_path: Midsole scan.
        out_root: Folder that holds one directory per shoe.
        slug: Shoe directory name; defaults to :func:`default_slug`.
        product_name: Display name stored in ``shoe.json``.

    Returns:
        The ``shoe.json`` payload.
    """
    glb_path = Path(glb_path)
    slug = slug or default_slug(glb_path)
    out = Path(out_root) / slug
    out.mkdir(parents=True, exist_ok=True)
    # Compute and write everything in a sibling staging folder first, so a
    # rejection or crash never leaves the existing outputs half-deleted.
    staging = Path(tempfile.mkdtemp(prefix=f".{slug}.", dir=out.parent))
    try:
        shoe, error = _stage_outputs(glb_path, staging, slug, product_name)
        source = out / "source.glb"
        if not source.exists() or (not source.samefile(glb_path) and _sha256_file(source) != shoe["source"]["sha256"]):
            shutil.copyfile(glb_path, staging / "source.glb")
        _publish(staging, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if error is not None:
        raise error
    return shoe


def _publish(staging: Path, out: Path) -> None:
    """Move staged files over the outputs with per-file atomic replaces; drop stale owned files."""
    staged = sorted(path.relative_to(staging) for path in staging.rglob("*") if path.is_file())
    # shoe.json goes last: it names the provenance every side file must match.
    staged.sort(key=lambda path: path == Path("shoe.json"))
    for relative in staged:
        (out / relative).parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / relative, out / relative)
    for relative in map(Path, _OWNED_OUTPUTS):
        if relative not in staged:
            (out / relative).unlink(missing_ok=True)


def _stage_outputs(
    glb_path: Path, staging: Path, slug: str, product_name: str | None
) -> tuple[dict[str, Any], ShoePrepError | None]:
    """Write all outputs of one shoe into ``staging``; return ``shoe.json`` and the rejection, if any."""
    source_sha = _sha256_file(glb_path)
    parameters = {
        "spacing_m": SPACING_M,
        "min_thickness_m": MIN_THICKNESS_M,
        "neighbor_tolerance_m": NEIGHBOR_TOLERANCE_M,
        "max_repaired_fraction": MAX_REPAIRED_FRACTION,
        "min_toe_spring_m": MIN_TOE_SPRING_M,
        "min_arch_ratio": MIN_ARCH_RATIO,
        "min_arch_difference_m": MIN_ARCH_DIFFERENCE_M,
    }
    code_sha = _code_sha256()
    provenance = hashlib.sha256(json.dumps([source_sha, code_sha, parameters], sort_keys=True).encode()).hexdigest()
    shoe: dict[str, Any] = {
        "slug": slug,
        "product_name": product_name or slug,
        "source": {"file": "source.glb", "input_name": glb_path.name, "sha256": source_sha},
        "code_sha256": code_sha,
        "parameters": parameters,
        "provenance_sha256": provenance,
        "frame": "+x heel to toe, +z up, +y = z cross x (medial on a right shoe, lateral on a left shoe)",
    }
    try:
        mesh, mesh_qc = load_midsole(glb_path)
        shoe["mesh_qc"] = mesh_qc
        canonical, matrix, frame_evidence = _canonical_mesh(mesh)
        shoe["frame_evidence"] = frame_evidence
        source_side, side_cues = foot_side(canonical.vertices)
        shoe["source_side"] = source_side
        shoe["side_cues_m"] = side_cues
        grid, column_qc = sample_columns(canonical)
        shoe["column_qc"] = column_qc
    except ShoePrepError as error:
        shoe.update(geometry_status="rejected", reason=str(error))
        _write_json(staging / "shoe.json", shoe)
        return shoe, error
    other = "left" if source_side == "right" else "right"
    grids = {source_side: grid, other: mirror_grid(grid)}
    repaired = np.asarray(column_qc["repaired_uv_m"]).reshape(-1, 2)
    repaired_by_side = {source_side: repaired, other: repaired * np.array([1.0, -1.0])}
    stats = _grid_stats(grid)
    gates = _gates(stats, canonical.vertices)
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    shoe["gates"] = gates
    to_canonical = matrix @ np.diag([mesh_qc["unit_scale_to_m"]] * 3 + [1.0])
    _write_figure(staging / "qc_top.png", grids, repaired_by_side, f"{slug} (source scan is a {source_side} shoe)")
    if failed:
        shoe.update(geometry_status="rejected", reason=f"gates failed: {', '.join(failed)}")
        _write_json(staging / "shoe.json", shoe)
        return shoe, ShoePrepError(shoe["reason"])
    shoe["geometry_status"] = "ready"
    for side, side_grid in grids.items():
        mirrored = side != source_side
        side_matrix = (np.diag([1.0, -1.0, 1.0, 1.0]) if mirrored else np.eye(4)) @ to_canonical
        side_stats = _grid_stats(side_grid)
        shoe[side] = {**side_stats, "mirrored_from_source": mirrored, "grid_sha256": _grid_sha256(side_grid)}
        folder = staging / side
        folder.mkdir()
        _write_npz(folder / "column_grid.npz", side_grid)
        _write_json(
            folder / "preflight.json",
            {
                "slug": slug,
                "side": side,
                "geometry_status": "ready",
                "mirrored_from_source": mirrored,
                "source_side": source_side,
                "provenance_sha256": provenance,
                "grid_sha256": shoe[side]["grid_sha256"],
                "source_to_canonical": side_matrix.round(12).tolist(),
                "stats": side_stats,
                "gates": gates,
                "column_qc": {k: v for k, v in column_qc.items() if k != "repaired_uv_m"},
            },
        )
        _write_json(
            folder / "manifest_geometry_fragment.json",
            {
                "midsole_mesh": "../source.glb",
                "midsole_to_canonical": side_matrix.round(12).tolist(),
                "grid": {"coarse_spacing_m": SPACING_M},
                "column_grid_cache": "column_grid.npz",
                "geometry_preflight": "preflight.json",
            },
        )
    _write_json(staging / "shoe.json", shoe)
    return shoe, None


def main(argv: list[str] | None = None) -> None:
    """Prepare each given ``.glb`` file and every ``.glb`` in each given folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", type=Path, nargs="+", help="Midsole .glb files and/or folders of .glb files.")
    parser.add_argument("--out-root", type=Path, default=Path("DigitalInstron"), help="Folder of shoe directories.")
    parser.add_argument("--slug", help="Shoe directory name (single file only).")
    parser.add_argument("--product-name", help="Display name (single file only).")
    args = parser.parse_args(argv)
    paths = [p for source in args.sources for p in (sorted(source.glob("*.glb")) if source.is_dir() else [source])]
    if not paths:
        parser.error("no .glb files found")
    if (args.slug or args.product_name) and len(paths) != 1:
        parser.error("--slug and --product-name need exactly one source file")
    rejected = 0
    for path in paths:
        slug = args.slug or default_slug(path)
        out = args.out_root / slug
        try:
            prepare_shoe(path, args.out_root, slug=slug, product_name=args.product_name)
            print(f"{slug}: ready -> {out}", flush=True)
        except ShoePrepError as error:
            rejected += 1
            print(f"{slug}: rejected ({error}) -> {out / 'shoe.json'}", flush=True)
    sys.exit(1 if rejected else 0)


if __name__ == "__main__":
    main()
