# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared setup helpers for the forward and differentiable Instron scenarios."""

import numpy as np
import warp as wp

from projects.digital_shoe.runtime import SurroundConfig

SURROUND_ATTACHMENT_N_M = 0.0
SURROUND_MAX_STRAIN = 0.9
SURROUND_SWEEPS = 32


def make_surround(
    driven: np.ndarray, *, carrier_bond: bool, attachment_n_m: float, max_strain: float, sweeps: int
) -> SurroundConfig:
    """Build one scenario surround without changing its explicit settings."""
    return SurroundConfig(
        driven=driven,
        attachment_n_m=attachment_n_m,
        max_strain=max_strain,
        sweeps=sweeps,
        carrier_bond=carrier_bond,
    )


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
