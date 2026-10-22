# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Register the Instron shoe last ("the missile") to a midsole column grid.

Physical model of the full-foot test:

* The midsole rests on a flat lower platen. A rocker sole tips until the
  preload contact under the last is balanced on its platen contact (least
  potential energy of the preload on linear columns).
* The last is bolted to the actuator through the square plate merged into the
  STL and moves straight down the load axis, so its up axis is the plate normal
  and its pitch/roll are fixed. Only x, y and yaw relative to the shoe are free;
  z follows from first contact.

Every heel/toe or up/down decision needs two agreeing geometric cues with a
margin, otherwise a :class:`ValueError` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CanonicalFrame",
    "LastModel",
    "LastRegistration",
    "canonical_frame",
    "load_last",
    "register_last",
    "write_qc_figure",
]

LAST_SPACING_M = 0.001
PLATE_PARALLEL_DEG = 0.5
MAX_PLATE_STUD_ANGLE_DEG = 1.0
MIN_ATTACHMENT_GAP_M = 0.020
# Plantar tread: last surface facing down within 45 deg of the load axis.
PLANTAR_NORMAL_Z = 0.7
MIN_LAST_WIDTH_CUE_M = 0.008
MIN_BALL_POSITION_CUE = 0.08
MAX_SUPPORT_TILT_DEG = 10.0
SUPPORT_BAND_M = 0.0005
MAX_OUTSIDE_FRACTION = 0.01
MAX_HEEL_OFFSET_M = 0.030


@dataclass(frozen=True)
class CanonicalFrame:
    """Rigid map from source coordinates to heel-to-toe, side, up."""

    rotation: np.ndarray
    translation: np.ndarray
    length_axis: int
    side_axis: int
    up_axis: int
    heel_at_positive_source: bool
    evidence: dict[str, float]

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map source points to canonical coordinates."""
        return (np.asarray(points, dtype=float) - self.translation) @ self.rotation.T


def canonical_frame(
    vertices: np.ndarray,
    *,
    kind: str = "shoe",
    shoe_top_m: np.ndarray,
    shoe_bottom_m: np.ndarray,
    shoe_uv_m: np.ndarray,
    min_cue_m: float = 0.003,
) -> CanonicalFrame:
    """Find the heel end of a midsole from its column grid.

    The heel stack is thicker and the heel is narrower than the forefoot; both
    cues must exceed ``min_cue_m`` and agree. The first UV axis must be the
    source length axis. The result maps +x heel to toe, keeps the source up
    axis and sets +y = z cross x.

    Args:
        vertices: Midsole points [m], shape [N, 3], used to pick the axes.
        kind: Must be ``"shoe"``.
        shoe_top_m: Column top heights [m].
        shoe_bottom_m: Column bottom heights [m].
        shoe_uv_m: Column footprint positions [m], shape [M, 2].
        min_cue_m: Smallest accepted cue magnitude [m].
    """
    if kind != "shoe":
        raise ValueError("canonical_frame only handles shoes; use load_last for the last")
    p = np.asarray(vertices, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 20 or not np.isfinite(p).all():
        raise ValueError("vertices must be a finite [N, 3] point array")
    extent = np.ptp(p, axis=0)
    length_axis = int(np.argmax(extent))
    up_axis = int(np.argmin(extent))
    side_axis = 3 - length_axis - up_axis
    if extent[length_axis] < 2.0 * extent[side_axis]:
        raise ValueError("low-confidence length axis")
    uv = np.asarray(shoe_uv_m, dtype=float)
    thickness = np.asarray(shoe_top_m, dtype=float) - np.asarray(shoe_bottom_m, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) != len(thickness):
        raise ValueError("column-grid arrays do not align")
    lo, span = uv[:, 0].min(), np.ptp(uv[:, 0])
    low = (uv[:, 0] >= lo + 0.08 * span) & (uv[:, 0] <= lo + 0.28 * span)
    high = (uv[:, 0] >= lo + 0.72 * span) & (uv[:, 0] <= lo + 0.92 * span)
    if low.sum() < 10 or high.sum() < 10:
        raise ValueError("insufficient shoe column samples for end cues")
    # Both cues are "low end minus high end" and positive when the toe is at low x.
    thick_cue = float(np.median(thickness[high]) - np.median(thickness[low]))
    width_cue = float(np.ptp(uv[low, 1]) - np.ptp(uv[high, 1]))
    cues = (thick_cue, width_cue)
    if min(abs(thick_cue), abs(width_cue)) < min_cue_m:
        raise ValueError(f"low-confidence heel/toe cues: {cues!r}")
    if np.sign(thick_cue) != np.sign(width_cue):
        raise ValueError(f"heel/toe cues disagree: {cues!r}")
    toe_is_low = thick_cue > 0.0
    up_vec = np.eye(3)[up_axis]
    length_vec = np.eye(3)[length_axis] * (-1.0 if toe_is_low else 1.0)
    rotation = np.vstack((length_vec, np.cross(up_vec, length_vec), up_vec))
    center = (p.min(axis=0) + p.max(axis=0)) * 0.5
    center[up_axis] = p[:, up_axis].min()
    evidence = {"low_end_thickness_or_elevation_m": thick_cue, "low_minus_high_width_m": width_cue}
    return CanonicalFrame(rotation, center, length_axis, side_axis, up_axis, toe_is_low, evidence)


@dataclass(frozen=True)
class LastModel:
    """Plantar height map of the last in its own frame.

    Frame: +z = attachment plate normal (load axis, up), +x heel to toe,
    +y = z cross x, x = 0 at the heel end, y = 0 at the footprint center,
    z = 0 at the lowest plantar point.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    plantar_z_m: np.ndarray
    source_to_last: np.ndarray
    evidence: dict[str, Any]

    def height(self, xy_m: np.ndarray) -> np.ndarray:
        """Plantar height [m] at last-frame points; NaN off the plantar surface."""
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator((self.y_m, self.x_m), self.plantar_z_m, bounds_error=False, fill_value=np.nan)
        return interp(np.asarray(xy_m)[..., ::-1])

    def outline(self) -> np.ndarray:
        """Last-frame xy [m] of the plantar map cells on the footprint boundary."""
        valid = np.pad(np.isfinite(self.plantar_z_m), 1)
        edge = valid[1:-1, 1:-1] & ~(valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:])
        rows, cols = np.nonzero(edge)
        return np.column_stack((self.x_m[cols], self.y_m[rows]))


def _parallel_face_direction(normals: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the direction with the most area of mutually parallel faces and that face mask."""
    cos = np.cos(np.deg2rad(PLATE_PARALLEL_DEG))
    candidates = np.argsort(areas)[-300:]
    scores = [areas[np.abs(normals @ normals[i]) > cos].sum() for i in candidates]
    mask = np.abs(normals @ normals[candidates[int(np.argmax(scores))]]) > cos
    weighted = (normals[mask] * areas[mask, None]).T @ normals[mask]
    return np.linalg.eigh(weighted)[1][:, -1], mask


def load_last(path: str | Path, *, spacing_m: float = LAST_SPACING_M) -> LastModel:
    """Load the last STL with its merged attachment and sample its plantar surface.

    The up axis is the attachment plate normal, cross-checked against the stud
    axis (faces spanning the empty gap between body and plate). The heel end is
    the narrower end and the ball tread (lowest plantar point) lies toward the
    toe. Pitch and roll are never fitted.

    Args:
        path: Last STL in millimeters or meters.
        spacing_m: Plantar map spacing [m].
    """
    import trimesh
    from scipy import ndimage

    mesh = trimesh.load_mesh(path, process=True)
    longest = float(np.ptp(mesh.vertices, axis=0).max())
    scale = 0.001 if 150.0 <= longest <= 600.0 else 1.0
    if not 0.15 <= longest * scale <= 0.6:
        raise ValueError(f"last extent {longest:g} is neither millimeters nor meters")
    vertices = np.asarray(mesh.vertices) * scale
    normals, areas = mesh.face_normals, mesh.area_faces * scale**2

    plate_normal, plate_faces = _parallel_face_direction(normals, areas)
    along = vertices @ plate_normal
    order = np.sort(along)
    gap_index = int(np.argmax(np.diff(order)))
    gap_lo, gap_hi = order[gap_index], order[gap_index + 1]
    if gap_hi - gap_lo < MIN_ATTACHMENT_GAP_M:
        raise ValueError(f"no attachment gap along the plate normal ({(gap_hi - gap_lo) * 1000:.1f} mm)")
    face_along = along[mesh.faces]
    stud = (face_along.min(axis=1) <= gap_lo) & (face_along.max(axis=1) >= gap_hi)
    if stud.sum() < 8:
        raise ValueError("no stud faces span the attachment gap")
    stud_axis = np.linalg.eigh((normals[stud] * areas[stud, None]).T @ normals[stud])[1][:, 0]
    stud_angle = float(np.degrees(np.arccos(min(1.0, abs(stud_axis @ plate_normal)))))
    if stud_angle > MAX_PLATE_STUD_ANGLE_DEG:
        raise ValueError(f"plate normal and stud axis differ by {stud_angle:.2f} deg")
    centers = mesh.triangles_center[plate_faces] * scale @ plate_normal
    plate_area_high = areas[plate_faces][centers > gap_hi].sum()
    plate_area_low = areas[plate_faces][centers < gap_lo].sum()
    body_low = float(np.ptp(order[: gap_index + 1])) > float(np.ptp(order[gap_index + 1 :]))
    # Up points from the body to the plate: the plate side holds the larger
    # parallel-face area, and the body is the side with the larger extent.
    plate_high = plate_area_high > plate_area_low
    if plate_high != body_low:
        raise ValueError("plate side and body side cues disagree")
    up = plate_normal if plate_high else -plate_normal
    body = (along <= gap_lo) if body_low else (along >= gap_hi)

    center = vertices[body].mean(axis=0)
    flat = (vertices[body] - center) - np.outer((vertices[body] - center) @ up, up)
    length = np.linalg.eigh(flat.T @ flat)[1][:, -1]

    def frame(length_vec: np.ndarray) -> np.ndarray:
        return np.vstack((length_vec, np.cross(up, length_vec), up))

    rotation = frame(length)
    local = (vertices - center) @ rotation.T
    body_mesh = trimesh.Trimesh(local, mesh.faces[np.all(body[mesh.faces], axis=1)], process=False)
    lo, hi = local[body, :2].min(axis=0), local[body, :2].max(axis=0)
    xs = np.arange(lo[0], hi[0] + spacing_m, spacing_m)
    ys = np.arange(lo[1], hi[1] + spacing_m, spacing_m)
    gx, gy = np.meshgrid(xs, ys)
    origins = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, local[:, 2].min() - 0.01)))
    hits, rays, triangles = body_mesh.ray.intersects_location(
        origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)), multiple_hits=False
    )
    z = np.full(gx.size, np.nan)
    facing_down = body_mesh.face_normals[triangles, 2] < -PLANTAR_NORMAL_Z
    z[rays[facing_down]] = hits[facing_down, 2]
    z = z.reshape(gx.shape)
    # Keep the main tread; down-facing patches on the side walls are not plantar.
    labels, count = ndimage.label(np.isfinite(z))
    if count > 1:
        z[labels != np.argmax(np.bincount(labels.ravel())[1:]) + 1] = np.nan

    fraction = (xs - xs[0]) / np.ptp(xs)
    valid = np.isfinite(z)
    width = np.array([np.ptp(ys[valid[:, i]]) if valid[:, i].any() else 0.0 for i in range(len(xs))])
    width_cue = float(width[fraction > 2 / 3].max() - width[fraction < 1 / 3].max())
    ball_fraction = float(fraction[np.nanargmin(np.nanmin(np.where(valid, z, np.inf), axis=0))])
    ball_cue = ball_fraction - 0.5
    cues = {"toe_side_minus_heel_side_width_m": width_cue, "ball_tread_fraction_minus_half": ball_cue}
    if abs(width_cue) < MIN_LAST_WIDTH_CUE_M or abs(ball_cue) < MIN_BALL_POSITION_CUE:
        raise ValueError(f"low-confidence last heel/toe cues: {cues!r}")
    if np.sign(width_cue) != np.sign(ball_cue):
        raise ValueError(f"last heel/toe cues disagree: {cues!r}")
    if width_cue < 0.0:
        rotation = frame(-length)
        xs, ys = -xs[::-1], -ys[::-1]
        z = z[::-1, ::-1]
    keep = np.flatnonzero(np.isfinite(z).any(axis=0))
    xs, z = xs[keep[0] : keep[-1] + 1], z[:, keep[0] : keep[-1] + 1]
    rows = np.flatnonzero(np.isfinite(z).any(axis=1))
    ys, z = ys[rows[0] : rows[-1] + 1], z[rows[0] : rows[-1] + 1]
    offset = np.array([xs[0], (ys[0] + ys[-1]) / 2.0, np.nanmin(z)])
    xs, ys, z = xs - offset[0], ys - offset[1], z - offset[2]

    source_to_last = np.eye(4)
    source_to_last[:3, :3] = rotation * scale
    source_to_last[:3, 3] = -center @ rotation.T - offset
    fraction = xs / xs[-1]
    centerline = z[np.argmin(np.abs(ys))]
    evidence = {
        "unit_scale_to_m": scale,
        "attachment_gap_m": float(gap_hi - gap_lo),
        "plate_stud_angle_deg": stud_angle,
        "up_in_source": up.tolist(),
        "heel_toe_cues": {
            "toe_side_minus_heel_side_width_m": abs(width_cue),
            "ball_tread_fraction_minus_half": abs(ball_cue),
        },
        "length_m": float(xs[-1]),
        "max_width_m": float(np.ptp(ys)),
        "heel_seat_height_m": float(np.nanmedian(centerline[(fraction >= 0.1) & (fraction < 0.2)])),
        "toe_tip_height_m": float(np.nanmax(centerline[fraction > 0.9])),
    }
    return LastModel(xs, ys, z, source_to_last, evidence)


def _inside_hull(points: np.ndarray, point: np.ndarray) -> bool:
    """Whether ``point`` lies in the 2D convex hull of ``points`` (False for fewer than 3)."""
    if len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
        return False
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    return bool(np.all(hull.equations[:, :2] @ point + hull.equations[:, 2] <= 1e-9))


def _tilt_rotation(pitch: float, roll: float) -> np.ndarray:
    """Rotation by ``roll`` about x, then ``pitch`` about y [rad]."""
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    )


def _rot2(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((np.stack((c, -s), -1), np.stack((s, c), -1)), -2)


def _spread(d: np.ndarray, scale_m: float = 0.001) -> np.ndarray:
    """Soft-L1 spread of the plantar clearance per pose [m]; NaN columns are ignored."""
    residual = d - np.nanmedian(d, axis=-1, keepdims=True)
    return np.nanmean(np.sqrt(1.0 + (residual / scale_m) ** 2) - 1.0, axis=-1) * scale_m


def _indent(gap: np.ndarray, stiffness: np.ndarray, force: float) -> tuple[float, np.ndarray]:
    """Indentation [m] at which linear columns carry ``force`` [N]; also the column forces."""
    order = np.argsort(gap)
    g, k = gap[order], stiffness[order]
    # Force carried when the indentation reaches g[i]: sum over j < i of k[j] (g[i] - g[j]).
    carried = g * np.concatenate(([0.0], np.cumsum(k)[:-1])) - np.concatenate(([0.0], np.cumsum(k * g)[:-1]))
    i = int(np.searchsorted(carried, force))
    depth = g[i - 1] + (force - carried[i - 1]) / k[:i].sum()
    return float(depth), stiffness * np.maximum(depth - gap, 0.0)


class _Platen:
    """Column tops and bottoms after tilting the midsole so it rests on the platen z = 0."""

    def __init__(self, uv: np.ndarray, bottom: np.ndarray, top: np.ndarray, rotation: np.ndarray):
        self.rotation = rotation
        self.bottom = np.column_stack((uv, bottom)) @ rotation.T
        self.top = np.column_stack((uv, top)) @ rotation.T
        lowest = self.bottom[:, 2].min()
        self.bottom[:, 2] -= lowest
        self.top[:, 2] -= lowest

    def clearance(self, last: LastModel, poses: np.ndarray) -> np.ndarray:
        """Midsole top minus last plantar height per pose and column [m] (NaN off the last)."""
        r = _rot2(poses[:, 2])
        local = np.einsum("pji,pnj->pni", r, self.top[None, :, :2] - poses[:, None, :2])
        return self.top[None, :, 2] - last.height(local)


def _fit_pose(platen: _Platen, last: LastModel, half_spacing: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Grid-search x, y, yaw for the least clearance spread with the tread inside the midsole."""
    from scipy.spatial import cKDTree

    outline = last.outline()
    xy = platen.top[:, :2]
    lo = xy.min(axis=0) - 0.01
    cells = np.ceil((xy.max(axis=0) + 0.01 - lo) / LAST_SPACING_M).astype(int)
    lattice = lo + LAST_SPACING_M * np.stack(np.meshgrid(*map(np.arange, cells), indexing="ij"), -1)
    inside = cKDTree(xy).query(lattice.reshape(-1, 2), p=np.inf)[0].reshape(cells) <= half_spacing

    def evaluate(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        placed = np.einsum("pij,nj->pni", _rot2(poses[:, 2]), outline) + poses[:, None, :2]
        index = np.floor((placed - lo) / LAST_SPACING_M).astype(int)
        valid = np.all((index >= 0) & (index < cells), axis=-1)
        index = np.clip(index, 0, cells - 1)
        outside = 1.0 - np.mean(valid & inside[index[..., 0], index[..., 1]], axis=1)
        cost = _spread(platen.clearance(last, poses))
        cost[(outside > MAX_OUTSIDE_FRACTION) | ~np.isfinite(cost)] = np.inf
        return cost, outside

    heel = xy[:, 0].min() - half_spacing
    center = np.array([heel + 0.020, float(np.median(xy[:, 1])), 0.0])
    steps, reach = np.array([0.002, 0.002, np.deg2rad(1.0)]), np.array([10, 12, 10])
    for _ in range(3):
        axes = [center[k] + steps[k] * np.arange(-reach[k], reach[k] + 1) for k in range(3)]
        poses = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        cost = np.concatenate([evaluate(poses[i : i + 512])[0] for i in range(0, len(poses), 512)])
        if not np.isfinite(cost).any():
            raise ValueError("no pose keeps the last tread inside the midsole outline")
        center, steps, reach = poses[int(np.argmin(cost))], steps / 4.0, np.array([8, 8, 8])
    cost, outside = evaluate(center[None])
    probes = {}
    for k, name, delta in ((0, "x_5mm", 0.005), (1, "y_5mm", 0.005), (2, "yaw_2deg", np.deg2rad(2.0))):
        probe = np.repeat(center[None], 2, axis=0)
        probe[:, k] += (-delta, delta)
        probes[name] = [float(c) if np.isfinite(c) else None for c in evaluate(probe)[0]]
    return center, {
        "spread_cost_m": float(cost[0]),
        "last_outside_fraction": float(outside[0]),
        "spread_cost_probes_m": probes,
        "heel_x_m": float(heel),
    }


@dataclass
class LastRegistration:
    """Last plantar surface over each column at zero displacement (first contact).

    ``surface_m`` and ``gap_m`` are NaN for columns outside the last tread.
    ``surface_m`` is in the input column-grid frame; ``gap_m`` is measured along
    the load axis. ``bottom_above_platen_m`` is each column's foam-bottom height
    over the platen in the rest pose; arch and rocker columns float above it.
    ``plot`` holds platen-frame arrays for :func:`write_qc_figure`.
    """

    surface_m: np.ndarray
    gap_m: np.ndarray
    bottom_above_platen_m: np.ndarray
    pose: dict[str, float]
    qc: dict[str, Any]
    plot: dict[str, Any] = field(repr=False)


def register_last(
    grid: Any,
    last: LastModel | str | Path,
    *,
    modulus_pa: float,
    preload_n: float = 20.0,
    expect_side: str = "left",
) -> LastRegistration:
    """Pose the last over a midsole column grid at zero displacement.

    The shoe heel end is proven with :func:`canonical_frame`; a grid with its
    heel at +x is handled by a 180 degree turn. The midsole pitch and roll on
    the platen minimize the potential energy of the preload on linear columns
    (a rocker sole tips until the last contact is balanced on its platen
    contact). The last keeps its load axis normal to the platen; x, y and yaw
    minimize the soft-L1 spread of the clearance to the midsole top with the
    last tread inside the midsole outline. z is set so the last just touches.
    The midsole pose and the last pose are refitted until both settle.

    Args:
        grid: :class:`geometry.ColumnGrid` with thickness axis 2.
        last: :class:`LastModel` or last STL path.
        modulus_pa: Linear column modulus [Pa] for the preload contact estimate.
        preload_n: Measured preload at the top of stroke [N].
        expect_side: Foot of the last; a midsole clearly of the other foot raises.

    Returns:
        The registration with QC values (lengths in meters, angles in degrees).
    """
    if grid.thickness_axis != 2:
        raise ValueError("column grid must be sampled along z")
    if not isinstance(last, LastModel):
        last = load_last(last)
    uv, bottom, top = np.asarray(grid.uv_m, float), np.asarray(grid.bottom_m, float), np.asarray(grid.top_m, float)
    frame = canonical_frame(np.column_stack((uv, top)), shoe_uv_m=uv, shoe_top_m=top, shoe_bottom_m=bottom)
    turn = -1.0 if frame.heel_at_positive_source else 1.0
    uv = uv * turn

    from .shoe_prep import ShoePrepError, foot_side  # noqa: PLC0415  # shoe_prep imports this module

    try:
        side, side_cues = foot_side(np.column_stack((uv, top)))
    except ShoePrepError as error:
        side, side_cues = "unverified", {"reason": str(error)}
    if side not in (expect_side, "unverified"):
        raise ValueError(f"midsole is a {side} shoe but the last is {expect_side} ({side_cues})")

    stiffness = modulus_pa * grid.area_m2 / (top - bottom)

    def potential(tilt: np.ndarray, pose: np.ndarray) -> float:
        """Preload times last height plus column strain energy once the preload is reached [J]."""
        clearance = _Platen(uv, bottom, top, _tilt_rotation(*tilt)).clearance(last, pose[None])[0]
        under = np.isfinite(clearance)
        gap = np.nanmax(clearance) - clearance[under]
        depth, force = _indent(gap, stiffness[under], preload_n)
        return preload_n * (np.nanmax(clearance) - depth) + 0.5 * float(np.sum(force**2 / stiffness[under]))

    # The midsole rocks on the platen until the preload potential is least; the
    # last x, y, yaw are refitted in each tilted frame until both settle.
    tilt = np.zeros(2)
    limit = np.deg2rad(MAX_SUPPORT_TILT_DEG)
    tolerance = np.deg2rad(0.05)
    fitted = []  # (tilt, last pose fitted in that tilt, its QC)
    for _ in range(10):
        platen = _Platen(uv, bottom, top, _tilt_rotation(*tilt))
        pose, fit_qc = _fit_pose(platen, last, grid.spacing_m / 2.0)
        fitted.append((tilt, pose, fit_qc))
        center, step = np.zeros(2), np.deg2rad(1.0)
        for _ in range(3):
            offsets = step * np.arange(-10, 11)
            tilts = np.stack(np.meshgrid(center[0] + offsets, center[1] + offsets, indexing="ij"), -1).reshape(-1, 2)
            tilts = tilts[np.all(np.abs(tilts) <= limit + 1e-12, axis=1)]
            center = tilts[int(np.argmin([potential(t, pose) for t in tilts]))]
            step /= 8.0
        if np.any(np.abs(center) >= limit - 1e-9):
            raise ValueError(f"midsole rest tilt reaches {MAX_SUPPORT_TILT_DEG} deg; the rocker pose is not bounded")
        if np.all(np.abs(center - tilt) < tolerance):
            tilt = center
            break
        if len(fitted) > 1 and np.all(np.abs(center - fitted[-2][0]) < tolerance):
            # Tilt and pose alternate between two states; keep the one of least potential.
            tilt, pose, fit_qc = min(fitted[-2:], key=lambda state: potential(state[0], state[1]))
            break
        tilt = center
    else:
        raise ValueError("rocker tilt and last pose did not settle")

    platen = _Platen(uv, bottom, top, _tilt_rotation(*tilt))
    clearance = platen.clearance(last, pose[None])[0]
    lift = float(np.nanmax(clearance))
    gap = lift - clearance
    under = np.isfinite(gap)
    depth, force = _indent(gap[under], stiffness[under], preload_n)
    surface = top + gap / float(platen.rotation[2, 2])
    support = platen.bottom[:, 2] <= SUPPORT_BAND_M
    centroid = force @ uv[under] / force.sum()
    heel_offset = float(pose[0] - fit_qc["heel_x_m"])
    if not 0.0 <= heel_offset <= MAX_HEEL_OFFSET_M:
        raise ValueError(f"last heel sits {heel_offset * 1000:.1f} mm from the midsole heel end")
    qc: dict[str, Any] = {
        "shoe_heel_at_input_plus_x": bool(frame.heel_at_positive_source),
        "shoe_heel_toe_cues_m": frame.evidence,
        "shoe_side": side,
        "shoe_side_cues_m": side_cues,
        "last_side": f"{expect_side} (from the STL name; not verified)",
        "last": last.evidence,
        "rest_pitch_deg": float(np.degrees(tilt[0])),
        "rest_roll_deg": float(np.degrees(tilt[1])),
        "platen_columns": int(support.sum()),
        "platen_columns_x_range_m": (np.sort(uv[support, 0] * turn)[[0, -1]]).tolist(),
        "preload_force_centroid_xy_m": (centroid * turn).tolist(),
        "preload_centroid_inside_platen_contact": bool(_inside_hull(uv[support], centroid)),
        "heel_offset_m": heel_offset,
        "yaw_deg": float(np.degrees(pose[2])),
        **{k: v for k, v in fit_qc.items() if k != "heel_x_m"},
        "columns": len(top),
        "overlap_columns": int(under.sum()),
        "gap_percentiles_mm": {f"p{q}": round(float(np.nanpercentile(gap, q)) * 1000, 2) for q in (5, 25, 50, 75, 95)},
        "columns_within_mm": {f"{t:g}": int(np.sum(gap <= t * 1e-3)) for t in (0.0, 0.5, 1.0, 2.0, 5.0)},
        "preload": {
            "preload_n": preload_n,
            "modulus_pa": modulus_pa,
            "indentation_m": depth,
            "compressed_columns": int(np.count_nonzero(force)),
        },
    }
    pose_out = {"x_m": float(pose[0]), "y_m": float(pose[1]), "yaw_rad": float(pose[2]), "lift_m": lift}
    plot = {
        "top_p": platen.top,
        "bottom_p": platen.bottom,
        "gap": gap,
        "pose": pose,
        "lift": lift,
        "support": support,
        "spacing": grid.spacing_m,
        "heel_x": fit_qc["heel_x_m"],
        "preload_depth": depth,
    }
    return LastRegistration(np.where(under, surface, np.nan), gap, platen.bottom[:, 2].copy(), pose_out, qc, plot)


def write_qc_figure(path: str | Path, registration: LastRegistration, last: LastModel, title: str) -> None:
    """Write the top-view gap map, centerline side profile and outline overlay.

    The platen frame is drawn with the heel on the left; +y is lateral for a
    left shoe.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = registration.plot
    top_p, bottom_p, gap = data["top_p"] * 1000, data["bottom_p"] * 1000, data["gap"] * 1000
    x0, y0, yaw = data["pose"]
    rotation = _rot2(np.array(yaw))
    outline = (last.outline() @ rotation.T + (x0, y0)) * 1000
    lateral, medial = ("LATERAL", "MEDIAL") if registration.qc["shoe_side"] != "right" else ("MEDIAL", "LATERAL")
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)

    def labels(ax: Any) -> None:
        for text, xy, align in (
            ("HEEL", (0.01, 0.5), "left"),
            ("TOE", (0.99, 0.5), "right"),
            (lateral, (0.5, 0.97), "center"),
            (medial, (0.5, 0.03), "center"),
        ):
            ax.text(*xy, text, transform=ax.transAxes, ha=align, va="center", fontweight="bold")

    ax = axes[0]
    ax.scatter(top_p[:, 0], top_p[:, 1], s=16, marker="s", c="0.85")
    shown = np.isfinite(gap)
    points = ax.scatter(
        top_p[shown, 0],
        top_p[shown, 1],
        c=gap[shown],
        s=16,
        marker="s",
        cmap="viridis_r",
        vmin=0.0,
        vmax=min(15.0, float(np.nanmax(gap))),
    )
    pressed = gap <= data["preload_depth"] * 1000
    ax.scatter(top_p[pressed, 0], top_p[pressed, 1], s=5, marker="x", c="red", lw=0.6)
    ax.plot(outline[:, 0], outline[:, 1], ",", color="red")
    fig.colorbar(points, ax=ax, label="last-to-midsole-top gap at first contact [mm]")
    preload = registration.qc["preload"]
    ax.set(
        title=f"{title}: gap at first contact; x = compressed at {preload['preload_n']:g} N "
        f"({preload['indentation_m'] * 1000:.1f} mm, {preload['compressed_columns']} columns)",
        xlabel="x heel → toe [mm]",
        ylabel="y [mm]",
        aspect="equal",
    )
    ax.margins(0.08)
    labels(ax)

    ax = axes[1]
    band = np.abs(top_p[:, 1] - y0 * 1000) <= data["spacing"] * 500 + 1e-6
    order = np.argsort(top_p[band, 0])
    ax.plot(top_p[band, 0][order], top_p[band, 2][order], "o-", ms=3, label="midsole top")
    ax.plot(bottom_p[band, 0][order], bottom_p[band, 2][order], "o-", ms=3, label="midsole bottom")
    ax.plot(top_p[band, 0][order], (top_p[band, 2] + gap[band])[order], "o-", ms=3, label="last plantar (columns)")
    xs = np.linspace(0.0, last.x_m[-1], 200)
    centre = last.height(np.column_stack((xs, np.zeros_like(xs))))
    world = np.column_stack((xs, np.zeros_like(xs))) @ rotation.T + (x0, y0)
    ax.plot(world[:, 0] * 1000, (centre + data["lift"]) * 1000, "k--", lw=1, label="last plantar centerline")
    contact = bottom_p[data["support"]]
    ax.plot(contact[:, 0], contact[:, 2], "rv", ms=4, label="columns on the platen (any y)")
    ax.axhline(0.0, color="0.3", lw=2, label="platen")
    ax.set(
        title=f"centerline profile (|y - last y| ≤ {data['spacing'] * 500:.1f} mm), midsole rests at "
        f"{registration.qc['rest_pitch_deg']:.2f} deg pitch, "
        f"{registration.qc['rest_roll_deg']:.2f} deg roll",
        xlabel="x heel → toe [mm]",
        ylabel="z [mm]",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small", ncol=3)

    ax = axes[2]
    half = data["spacing"] * 500
    for x, y in top_p[:, :2]:
        ax.add_patch(plt.Rectangle((x - half, y - half), 2 * half, 2 * half, color="0.8", lw=0))
    ax.plot(outline[:, 0], outline[:, 1], ".", ms=1.5, color="red", label="last plantar outline")
    ax.axvline(data["heel_x"] * 1000, color="blue", lw=1, label="midsole heel end")
    ax.axvline(x0 * 1000, color="red", lw=1, ls="--", label="last heel end")
    qc = registration.qc
    ax.set(
        title=f"outline: heel offset {qc['heel_offset_m'] * 1000:.1f} mm, yaw {qc['yaw_deg']:.1f} deg, "
        f"outside {qc['last_outside_fraction'] * 100:.1f} %",
        xlabel="x heel → toe [mm]",
        ylabel="y [mm]",
        aspect="equal",
    )
    ax.autoscale()
    ax.margins(0.08)
    ax.legend(fontsize="small", loc="upper right")
    labels(ax)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Register the last to the left grid of each shoe folder and write QC figures and JSON."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    from .geometry import ColumnGrid  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("shoes", nargs="+", type=Path, help="Shoe folders holding <side>/column_grid.npz.")
    parser.add_argument("--last", type=Path, required=True, help="Last STL with the merged attachment.")
    parser.add_argument("--modulus-pa", type=float, required=True, help="Linear column modulus for the preload.")
    parser.add_argument("--side", default="left", help="Shoe side folder; must match the last.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder.")
    args = parser.parse_args(argv)
    last = load_last(args.last)
    args.out.mkdir(parents=True, exist_ok=True)
    for shoe in args.shoes:
        with np.load(shoe / args.side / "column_grid.npz") as data:
            grid = ColumnGrid(
                *(data[k] for k in ("uv_m", "bottom_m", "top_m")),
                np.asarray(data["area_m2"], dtype=float),
                float(data["spacing_m"]),
                int(data["thickness_axis"]),
            )
        registration = register_last(grid, last, modulus_pa=args.modulus_pa, expect_side=args.side)
        stem = args.out / f"{shoe.name}_{args.side}"
        write_qc_figure(stem.with_suffix(".png"), registration, last, f"{shoe.name} ({args.side})")
        stem.with_suffix(".json").write_text(
            json.dumps({"pose": registration.pose, "qc": registration.qc}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{shoe.name}: {json.dumps(registration.pose)}")


if __name__ == "__main__":
    main()
