# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dynamic elastic-foundation midsole coupled into Newton rigid-body physics.

The calibrated column model from :mod:`projects.digital_instron_v2.core` is a
Winkler/Pasternak bed of nonlinear viscoelastic springs sampled from the shoe
midsole mesh. This module turns that bed into a live Warp force model: every
substep each column reads its carrier-body pose, computes its through-thickness
compression, evaluates the first-order Hyperfoam equilibrium pressure with a
real-time generalized-Maxwell overstress branch and Pasternak lateral coupling,
and accumulates the resulting wrench into :attr:`newton.State.body_f`.

The same foundation drives three scenarios:

* a displacement-controlled digital Instron that squishes the midsole between a
  shoe-last crosshead and the ground plane and records the force-displacement
  hysteresis loop,
* a free, massive midsole that rests in stable equilibrium on the foundation and
  resists lateral loads through Coulomb foam-shear friction, and
* a synthetic running stride whose heel-to-toe roll produces a ground-reaction
  force profile and a migrating center of pressure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_shoe.runtime import (
    FoundationConfig,
    FoundationParams,
    MidsoleFoundation,
    foundation_apply,
    foundation_pressure,
    foundation_reset,
)

from .core import CALIBRATED_MATERIAL, EFFECTIVE_POISSON_RATIO, MAXWELL_RELAXATION_TIME_S, Material
from .geometry import build_column_grid, load_mesh, raycast_surface, rearfoot_center, transform_mesh

POISSON = wp.constant(EFFECTIVE_POISSON_RATIO)
TAU_S = wp.constant(MAXWELL_RELAXATION_TIME_S)

__all__ = [
    "EFFECTIVE_POISSON_RATIO",
    "MAXWELL_RELAXATION_TIME_S",
    "POISSON",
    "TAU_S",
    "FoundationConfig",
    "FoundationParams",
    "MidsoleFoundation",
    "foundation_apply",
    "foundation_pressure",
    "foundation_reset",
]


# ---------------------------------------------------------------------------
# Geometry and calibration inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FoundationGeometry:
    """Column bed sampled from the calibrated midsole mesh (ground plane at z=0)."""

    uv_m: np.ndarray  # footprint coordinates [column_count, 2]
    slack_m: np.ndarray  # rest thickness [column_count]
    area_m2: float  # tributary area per column
    spacing_m: float  # grid spacing
    z_free_m: np.ndarray  # uncompressed foam-top height above ground [column_count]
    z_bottom_m: np.ndarray  # foam-bottom height above ground [column_count]
    surface_m: np.ndarray  # shoe-last underside height above ground at rest [column_count]
    gap0_m: np.ndarray  # initial foot-underside-to-foam-top clearance [column_count]
    neighbors: np.ndarray  # Pasternak 4-neighbour indices [column_count, 4]
    midsole_mesh_path: str
    z_shift_m: float  # ground offset applied to the raw mesh frame
    indenter_shift_m: float  # shift applied to the posed indenter so its contact face meets the foam top
    thickness_axis: int  # mesh axis along which columns compress (rendering/offset axis)


def build_foundation_geometry(manifest_path: str | Path, fixture: str = "fullfoot_last") -> FoundationGeometry:
    """Sample the calibrated midsole footprint into an elastic-foundation column bed.

    Reuses the calibration geometry pipeline: the midsole mesh is column-sampled
    on the manifest grid, and the shoe-last indenter is posed and raycast to give
    each column its initial clearance under the crosshead. Heights are shifted so
    the ground plane sits at ``z = 0``.
    """
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    base = path.parent
    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])

    source = next(t for t in config["trials"] if t["fixture"] == fixture)
    indenter = source["indenter"]
    if "path" not in indenter:
        return _build_rearfoot_geometry(config, base, grid, source, str(base / config["midsole_mesh"]))
    last = load_mesh(base / indenter["path"], 0.001, indenter["rotation_deg"], indenter["crop_height_m"])
    transform_mesh(
        last,
        indenter.get("pose_rotation_deg", [0.0, 0.0, 0.0]),
        indenter.get("pose_translation_m", [0.0, 0.0, 0.0]),
    )
    surface = raycast_surface(last, grid.uv_m, grid.thickness_axis, indenter["contact_side"])
    active = np.isfinite(surface)
    offset = np.percentile(grid.top_m[active] - surface[active], indenter["contact_percentile"])
    indenter_shift = float(offset + indenter["height_offset_m"])
    surface = surface + indenter_shift
    surface[active] = np.maximum(surface[active], grid.top_m[active])

    uv = grid.uv_m[active]
    slack = grid.slack_m[active]
    top = grid.top_m[active]
    bottom = grid.bottom_m[active]
    surf = surface[active]
    z_shift = float(np.min(bottom))
    return FoundationGeometry(
        uv_m=uv,
        slack_m=slack,
        area_m2=float(grid.area_m2),
        spacing_m=grid.spacing_m,
        z_free_m=top - z_shift,
        z_bottom_m=bottom - z_shift,
        surface_m=surf - z_shift,
        gap0_m=surf - top,
        neighbors=_neighbor_indices(uv, grid.uv_m, grid.spacing_m),
        midsole_mesh_path=str(base / config["midsole_mesh"]),
        z_shift_m=z_shift,
        indenter_shift_m=indenter_shift,
        thickness_axis=int(grid.thickness_axis),
    )


def _build_rearfoot_geometry(config: dict, base: Path, grid, source: dict, mesh_path: str) -> FoundationGeometry:
    """Sample a flat circular-punch column bed for the rearfoot fixture.

    The rearfoot test drives a rigid ``radius_m`` punch straight down onto the
    heel, so every column under the disc compresses uniformly. Anchoring each
    column top at its rest foam height (``z_free = slack``) reproduces the
    calibration's uniform-compression assumption: a carrier descent ``d`` gives
    every disc column the same compression ``d``.
    """
    radius = float(source["indenter"]["radius_m"])
    center = rearfoot_center(
        load_mesh(base / config["midsole_mesh"], 0.001), grid, config["grid"]["rearfoot_length_fraction"]
    )
    active = np.linalg.norm(grid.uv_m - center, axis=1) <= radius
    uv = grid.uv_m[active]
    slack = grid.slack_m[active]
    count = int(np.count_nonzero(active))
    area = float(np.pi * radius**2 / count)
    return FoundationGeometry(
        uv_m=uv,
        slack_m=slack,
        area_m2=area,
        spacing_m=grid.spacing_m,
        z_free_m=slack.copy(),
        z_bottom_m=np.zeros(count, dtype=np.float64),
        surface_m=slack.copy(),
        gap0_m=np.zeros(count, dtype=np.float64),
        neighbors=_neighbor_indices(uv, grid.uv_m, grid.spacing_m),
        midsole_mesh_path=mesh_path,
        z_shift_m=0.0,
        indenter_shift_m=0.0,
        thickness_axis=int(grid.thickness_axis),
    )


def _neighbor_indices(uv: np.ndarray, grid_uv: np.ndarray, spacing: float) -> np.ndarray:
    """Return the Pasternak 4-neighbour index table for the active columns.

    Each of the four in-plane neighbours (-u, +u, -v, +v) is a non-negative
    active-column index, ``-1`` for a footprint boundary (natural zero-gradient),
    or ``-2`` for an interior gap cell that contributes no lateral coupling.
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


def load_fitted_material(manifest_path: str | Path) -> Material:
    """Load the fitted artifact or the checked-in calibrated prediction baseline."""
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    artifact = path.parent / config["cache_dir"] / "digital_instron_material.json"
    if artifact.exists():
        values = json.loads(artifact.read_text())["material"]
        return Material(
            values["instantaneous_shear_modulus_pa"],
            values["hyperfoam_exponent"],
            values["equilibrium_fraction"],
            values["pasternak_n_per_m"],
        )
    return CALIBRATED_MATERIAL


def load_measured_cycle(
    manifest_path: str | Path, fixture: str = "fullfoot_last"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, displacement, and baseline-corrected force for one averaged trial."""
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    source = next(t for t in config["trials"] if t["fixture"] == fixture)
    data = np.genfromtxt(path.parent / source["averaged_cycle_path"], delimiter=",", names=True)
    time = np.asarray(data["time_s"])
    displacement = np.maximum(np.asarray(data["displacement_m"]), 0.0)
    force = np.asarray(data["force_n"])
    return time, displacement, force - np.min(force)


# ---------------------------------------------------------------------------
# Carrier trajectories
# ---------------------------------------------------------------------------
def cyclic_displacement(time_s: np.ndarray, displacement_m: np.ndarray):
    """Return a periodic descent function ``t -> depth`` from a measured cycle.

    The measured displacement trace is looped so the digital Instron can run
    several warm-up cycles before the reported hysteresis loop is recorded.
    """
    period = float(time_s[-1] - time_s[0])
    t0 = float(time_s[0])

    def depth(t: float) -> float:
        return float(np.interp((t - t0) % period, time_s - t0, displacement_m))

    return depth, period


def synthetic_stride(peak_depth_m: float, pitch_deg: float, roll_length_m: float, period_s: float):
    """Return a heel-to-toe running-stride pose function for the foot carrier.

    Args:
        peak_depth_m: Maximum vertical descent of the foot during stance [m].
        pitch_deg: Peak forefoot-down pitch amplitude during the roll [deg].
        roll_length_m: Horizontal travel of the foot across stance [m].
        period_s: Stride period [s].

    Returns:
        A callable ``t -> (wp.vec3 position, wp.quat orientation)`` describing a
        single stance phase (descent, heel-to-toe roll, push-off) followed by a
        clear swing phase where the foot lifts off the foundation.
    """

    def pose(t: float):
        phase = (t % period_s) / period_s
        stance = min(phase / 0.62, 1.0)  # ~62% duty factor for running
        if phase <= 0.62:
            depth = peak_depth_m * np.sin(np.pi * stance)
            pitch = np.radians(pitch_deg) * (2.0 * stance - 1.0)  # heel-down to toe-down
            x = roll_length_m * (stance - 0.5)
        else:
            swing = (phase - 0.62) / 0.38
            depth = -0.05 * np.sin(np.pi * swing)  # lift clear of the foundation
            pitch = np.radians(pitch_deg) * (1.0 - 2.0 * swing)
            x = roll_length_m * (0.5 - swing)
        pos = wp.vec3(float(x), 0.0, float(-depth))
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(pitch))
        return pos, rot

    return pose


@wp.kernel
def column_world_positions(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    out_points: wp.array[wp.vec3],
):
    """Write each column's foam-top contact point in world space for rendering."""
    i = wp.tid()
    world = wp.transform_point(body_q[carrier], anchor_local[i])
    top = z_free[i]
    if world[2] < top:
        top = world[2]
    out_points[i] = wp.vec3(world[0], world[1], top)


@wp.kernel
def column_colors(
    compression: wp.array[wp.float32],
    ref: wp.float32,
    out_colors: wp.array[wp.vec3],
):
    """Map per-column compression to a cool-to-hot contact colour for rendering."""
    i = wp.tid()
    t = wp.clamp(compression[i] / ref, 0.0, 1.0)
    cool = wp.vec3(0.13, 0.32, 0.92)  # uncompressed foam
    warm = wp.vec3(0.28, 0.86, 0.24)  # light contact
    hot = wp.vec3(0.96, 0.20, 0.10)  # firm contact
    if t < 0.5:
        s = t * 2.0
        out_colors[i] = cool * (1.0 - s) + warm * s
    else:
        s = (t - 0.5) * 2.0
        out_colors[i] = warm * (1.0 - s) + hot * s


@wp.kernel
def attach_coupling(
    body: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    target_traj: wp.array[wp.transform],
    target_vel_traj: wp.array[wp.spatial_vector],
    counter: wp.array[wp.int32],
    period: wp.int32,
    kp_lin: wp.float32,
    kd_lin: wp.float32,
    kp_ang: wp.float32,
    kd_ang: wp.float32,
    max_force: wp.float32,
    body_f: wp.array[wp.spatial_vector],
    out_force: wp.array[wp.float32],
):
    """Damped PD "shoe upper" holding one body to a moving target pose.

    Models the compliant but bilateral connection between the foot and the shoe:
    slack (tiny force) whenever the shoe can freely follow the foot in flight,
    and stiff (large force) when the ground blocks the shoe in stance. The COM
    is assumed to sit at the body origin (``body_com == 0``).

    The target pose/velocity for the current substep are read from a precomputed,
    periodic device trajectory indexed by ``counter[0] % period``; the counter is
    advanced on device so the whole substep loop stays host-free and CUDA-graph
    capturable.
    """
    idx = counter[0] % period
    target = target_traj[idx]
    target_vel = target_vel_traj[idx]
    pos = wp.transform_get_translation(body_q[body])
    rot = wp.transform_get_rotation(body_q[body])
    target_pos = wp.transform_get_translation(target)
    target_rot = wp.transform_get_rotation(target)

    e_p = target_pos - pos
    q_err = target_rot * wp.quat_inverse(rot)
    if q_err[3] < 0.0:
        q_err = wp.quat(-q_err[0], -q_err[1], -q_err[2], -q_err[3])
    e_r = 2.0 * wp.vec3(q_err[0], q_err[1], q_err[2])

    v = wp.spatial_top(body_qd[body])
    w = wp.spatial_bottom(body_qd[body])
    tv = wp.spatial_top(target_vel)
    tw = wp.spatial_bottom(target_vel)

    force = kp_lin * e_p + kd_lin * (tv - v)
    moment = kp_ang * e_r + kd_ang * (tw - w)

    mag = wp.length(force)
    if mag > max_force and mag > 1.0e-9:
        force = force * (max_force / mag)

    wp.atomic_add(body_f, body, wp.spatial_vector(force, moment))
    out_force[0] = wp.length(force)
    counter[0] = (idx + 1) % period


@wp.kernel
def attached_columns(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    anchor_bottom: wp.array[wp.vec3],
    rest_len: wp.array[wp.float32],
    bottom_out: wp.array[wp.vec3],
    top_out: wp.array[wp.vec3],
):
    """World foam-column endpoints for the attached shoe.

    ``bottom_out`` is the outsole contact point clamped to the ground plane and
    ``top_out`` is the sole-mounted foam top that rides rigidly with the shoe, so the
    bed lifts with the shoe in flight and the bars shorten as the foam penetrates the
    ground in stance.
    """
    i = wp.tid()
    q = body_q[carrier]
    b = wp.transform_point(q, anchor_bottom[i])
    t = wp.transform_point(q, anchor_bottom[i] + wp.vec3(0.0, 0.0, rest_len[i]))
    bottom_z = wp.max(b[2], 0.0)
    top_z = wp.max(t[2], 0.0)
    if top_z < bottom_z:
        bottom_z = top_z
    bottom_out[i] = wp.vec3(b[0], b[1], bottom_z)
    top_out[i] = wp.vec3(t[0], t[1], top_z)
