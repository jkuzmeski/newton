# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bench geometry, measured drive, and stride trajectories for the midsole example.

The calibrated column bed is sampled from the shoe midsole mesh and driven by
:mod:`projects.digital_instron_v2.example` through the shared two-surface
column bed (:class:`projects.digital_shoe.column_bed.ColumnBed`) as

* a displacement-controlled digital Instron that presses the midsole between a
  shoe-last crosshead and the platen and records the force-displacement
  hysteresis loop,
* a free, massive plate that rests in stable equilibrium on the foam and
  resists lateral loads through Coulomb foam-shear friction, and
* prescribed and foot-mounted running strides whose heel-to-toe roll produces a
  ground-reaction force profile and a migrating center of pressure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from .geometry import build_column_grid, load_mesh, neighbor_indices, raycast_surface, transform_mesh


# ---------------------------------------------------------------------------
# Geometry and calibration inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FoundationGeometry:
    """Whole-midsole column bed sampled from the calibrated mesh (ground plane at z=0).

    The bed spans every column of the manifest grid. ``driven`` marks the subset
    the indenter presses; the rest is free foam that deforms only through the
    lateral shear layer.
    """

    uv_m: np.ndarray  # footprint coordinates [column_count, 2]
    slack_m: np.ndarray  # rest thickness [column_count]
    area_m2: np.ndarray  # midsole footprint area [column_count]
    spacing_m: float  # grid spacing
    z_free_m: np.ndarray  # uncompressed foam-top height above ground [column_count]
    z_bottom_m: np.ndarray  # foam-bottom height above ground [column_count]
    surface_m: np.ndarray  # shoe-last underside height above ground at rest [column_count]
    gap0_m: np.ndarray  # initial foot-underside-to-foam-top clearance [column_count]
    neighbors: np.ndarray  # Pasternak 4-neighbour indices [column_count, 4]
    driven: np.ndarray  # bool: the indenter presses this column [column_count]
    midsole_mesh_path: str
    z_shift_m: float  # ground offset applied to the raw mesh frame
    indenter_shift_m: float  # shift applied to the posed indenter so its contact face meets the foam top
    thickness_axis: int  # mesh axis along which columns compress (rendering/offset axis)


def build_foundation_geometry(manifest_path: str | Path) -> FoundationGeometry:
    """Sample the whole calibrated midsole under the full-foot last into a column bed.

    Reuses the calibration geometry pipeline: the midsole mesh is column-sampled
    on the manifest grid, and the shoe-last indenter is posed and raycast to give
    each column its initial clearance under the crosshead. Heights are shifted so
    the ground plane sits at ``z = 0``.

    The bed keeps every midsole column, not only the ones the indenter touches.
    Dropping the rest left the shear layer with no material to act on, so the
    runtime lost the support the identification counts on. Columns the indenter
    misses are anchored on their own foam top (``gap0 = 0``) and reported as
    undriven.
    """
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    base = path.parent
    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])

    indenter = next(t for t in config["trials"] if t["fixture"] == "fullfoot_last")["indenter"]
    last = load_mesh(base / indenter["path"], 0.001, indenter["rotation_deg"], indenter["crop_height_m"])
    transform_mesh(
        last,
        indenter.get("pose_rotation_deg", [0.0, 0.0, 0.0]),
        indenter.get("pose_translation_m", [0.0, 0.0, 0.0]),
    )
    surface = raycast_surface(last, grid.uv_m, grid.thickness_axis, indenter["contact_side"])
    driven = np.isfinite(surface)
    offset = np.percentile(grid.top_m[driven] - surface[driven], indenter["contact_percentile"])
    indenter_shift = float(offset + indenter["height_offset_m"])
    surface = surface + indenter_shift
    surface[driven] = np.maximum(surface[driven], grid.top_m[driven])
    # The crosshead never reaches these columns, so their carrier anchor is their own
    # free foam top; the column balance, not the carrier pose, sets their compression.
    surface[~driven] = grid.top_m[~driven]

    top = np.asarray(grid.top_m, dtype=float)
    bottom = np.asarray(grid.bottom_m, dtype=float)
    z_shift = float(np.min(bottom))
    return FoundationGeometry(
        uv_m=np.asarray(grid.uv_m, dtype=float),
        slack_m=np.asarray(grid.slack_m, dtype=float),
        area_m2=np.asarray(grid.area_m2, dtype=float),
        spacing_m=grid.spacing_m,
        z_free_m=top - z_shift,
        z_bottom_m=bottom - z_shift,
        surface_m=surface - z_shift,
        gap0_m=surface - top,
        neighbors=neighbor_indices(grid.uv_m, grid.uv_m, grid.spacing_m),
        driven=driven,
        midsole_mesh_path=str(base / config["midsole_mesh"]),
        z_shift_m=z_shift,
        indenter_shift_m=indenter_shift,
        thickness_axis=int(grid.thickness_axis),
    )


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


def quat_multiply(a, b) -> np.ndarray:
    """Multiply two XYZW quaternions and retain the scenario's float32 output."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def quat_conjugate(a) -> np.ndarray:
    """Return the conjugate of an XYZW quaternion as float32."""
    return np.array([-a[0], -a[1], -a[2], a[3]], dtype=np.float32)


@wp.func
def attachment_pd_wrench(
    pose: wp.transform,
    velocity: wp.spatial_vector,
    target: wp.transform,
    target_velocity: wp.spatial_vector,
    kp_lin: float,
    kd_lin: float,
    kp_ang: float,
    kd_ang: float,
    max_force: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Return the scenarios' existing clamped attachment force and moment [N, N·m].

    Both poses use the same body-origin/COM convention as the original examples.
    The adapters own target indexing and accumulation, not a second PD law.
    """
    pos = wp.transform_get_translation(pose)
    rot = wp.transform_get_rotation(pose)
    target_pos = wp.transform_get_translation(target)
    target_rot = wp.transform_get_rotation(target)
    e_p = target_pos - pos
    q_err = target_rot * wp.quat_inverse(rot)
    if q_err[3] < 0.0:
        q_err = wp.quat(-q_err[0], -q_err[1], -q_err[2], -q_err[3])
    e_r = 2.0 * wp.vec3(q_err[0], q_err[1], q_err[2])
    v = wp.spatial_top(velocity)
    w = wp.spatial_bottom(velocity)
    tv = wp.spatial_top(target_velocity)
    tw = wp.spatial_bottom(target_velocity)
    force = kp_lin * e_p + kd_lin * (tv - v)
    moment = kp_ang * e_r + kd_ang * (tw - w)
    mag = wp.length(force)
    if mag > max_force and mag > 1.0e-9:
        force = force * (max_force / mag)
    return force, moment


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
    force, moment = attachment_pd_wrench(
        body_q[body],
        body_qd[body],
        target,
        target_vel,
        kp_lin,
        kd_lin,
        kp_ang,
        kd_ang,
        max_force,
    )

    wp.atomic_add(body_f, body, wp.spatial_vector(force, moment))
    out_force[0] = wp.length(force)
    counter[0] = (idx + 1) % period
