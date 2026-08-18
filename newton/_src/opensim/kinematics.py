# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""OpenSim-exact forward kinematics over the parsed model IR, in Warp kernels.

Ports the Simbody mobilizer math that OpenSim uses to place bodies given
generalized coordinates. The core is the :class:`CustomJoint` spatial transform
(three sequential axis rotations followed by summed axis translations, each
driven by a coordinate ``functions``), which is what the
standard gait models use. Every joint type (``PinJoint``, ``SliderJoint``,
``BallJoint``, ``FreeJoint``, ``PlanarJoint``, ``UniversalJoint``,
``WeldJoint``) is flattened to this uniform six-axis form so a single Warp
kernel evaluates the whole tree.

For a body B with parent P, joint frame F on the parent and M on the child, the
ground transform is composed as

.. math::

    X_{ground,B} = X_{ground,P}\, X_{P,F}\, X_{F,M}(q)\, X_{B,M}^{-1}

The function evaluation (including a faithful ``OpenSim::SimmSpline``), rotation
composition, transform chaining, and marker placement all run in ``float64``
Warp kernels (:func:`fk_kernel`, :func:`marker_kernel`) that evaluate a *batch*
of coordinate vectors at once, which the inverse-kinematics solver in
``ik`` uses for its batched finite-difference Jacobian.
Coordinates are in native units (radians for rotational, meters for
translational).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .functions import SimmSpline
from .types import OsimJoint, OsimModel

wp.set_module_options({"enable_backward": False})

__all__ = [
    "ForwardKinematics",
    "body_acceleration_kernel",
    "body_jacobian_kernel",
    "body_velocity_kernel",
    "com_kernel",
    "euler_xyz_to_matrix",
    "fk_kernel",
    "load_project_kernel",
    "make_transform",
    "marker_kernel",
    "momentum_kernel",
    "rotation_about_axis",
]

# Axis function type codes shared by the host flattener and the Warp kernel.
_FN_CONSTANT = 0
_FN_LINEAR = 1
_FN_SIMMSPLINE = 2
_FN_PIECEWISE = 3


# -----------------------------------------------------------------------------
# Host helpers (transform assembly, in NumPy)
# -----------------------------------------------------------------------------
def rotation_about_axis(axis, angle: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for ``angle`` [rad] about ``axis`` (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1.0e-12:
        return np.eye(3)
    x, y, z = a / n
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def euler_xyz_to_matrix(a: float, b: float, c: float) -> np.ndarray:
    """Return the rotation for OpenSim body-fixed XYZ Euler angles ``(a, b, c)`` [rad]."""
    return (
        rotation_about_axis((1.0, 0.0, 0.0), a)
        @ rotation_about_axis((0.0, 1.0, 0.0), b)
        @ rotation_about_axis((0.0, 0.0, 1.0), c)
    )


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation and a translation."""
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = np.asarray(translation, dtype=float)
    return m


# -----------------------------------------------------------------------------
# Warp kernels
# -----------------------------------------------------------------------------
_vec3d = wp.vec3d
_mat33d = wp.mat33d
_mat44d = wp.mat44d
_f64 = wp.float64


@wp.func
def _eval_baked(
    x: _f64,
    t: wp.int32,
    p0: _f64,
    p1: _f64,
    off: wp.int32,
    n: wp.int32,
    kx: wp.array[_f64],
    ky: wp.array[_f64],
    kb: wp.array[_f64],
    kc: wp.array[_f64],
    kd: wp.array[_f64],
) -> _f64:
    """Evaluate a baked scalar function (see :func:`_bake_function`) at ``x``.

    Shared by :func:`_eval_axis` (coordinate functions) and any curve of a single
    scalar input such as a ligament force-length curve.
    """
    if t == 0:  # constant
        return p0
    if t == 1:  # linear: slope * x + intercept
        return p0 * x + p1
    if t == 3:  # piecewise linear (flat extrapolation, matching numpy.interp)
        if x <= kx[off]:
            return ky[off]
        if x >= kx[off + n - 1]:
            return ky[off + n - 1]
        k = int(0)
        for i in range(n - 1):
            if x >= kx[off + i] and x <= kx[off + i + 1]:
                k = i
        frac = (x - kx[off + k]) / (kx[off + k + 1] - kx[off + k])
        return ky[off + k] + frac * (ky[off + k + 1] - ky[off + k])
    # SimmSpline (t == 2): Horner cubic with SIMM linear extrapolation.
    if x < kx[off]:
        return ky[off] + (x - kx[off]) * kb[off]
    if x > kx[off + n - 1]:
        return ky[off + n - 1] + (x - kx[off + n - 1]) * kb[off + n - 1]
    k = int(0)
    if n >= 3:
        lo = int(0)
        hi = int(n)
        for _ in range(64):
            k = (lo + hi) // 2
            if x < kx[off + k]:
                hi = k
            elif x > kx[off + k + 1]:
                lo = k
            else:
                break
    dx = x - kx[off + k]
    return ky[off + k] + dx * (kb[off + k] + dx * (kc[off + k] + dx * kd[off + k]))


@wp.func
def _eval_axis(
    a: int,
    s: int,
    q: wp.array2d[_f64],
    atype: wp.array[wp.int32],
    acoord: wp.array[wp.int32],
    ap0: wp.array[_f64],
    ap1: wp.array[_f64],
    koff: wp.array[wp.int32],
    kcnt: wp.array[wp.int32],
    kx: wp.array[_f64],
    ky: wp.array[_f64],
    kb: wp.array[_f64],
    kc: wp.array[_f64],
    kd: wp.array[_f64],
) -> _f64:
    """Evaluate the coordinate function driving flattened axis ``a`` at sample ``s``."""
    ci = acoord[a]
    x = _f64(0.0)
    if ci >= 0:
        x = q[s, ci]
    return _eval_baked(x, atype[a], ap0[a], ap1[a], koff[a], kcnt[a], kx, ky, kb, kc, kd)


@wp.func
def _rot_axis_mat(axis: _vec3d, angle: _f64) -> _mat33d:
    """Return the rotation matrix for ``angle`` about a unit ``axis`` (Rodrigues)."""
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = wp.cos(angle)
    s = wp.sin(angle)
    cc = _f64(1.0) - c
    return _mat33d(
        c + x * x * cc,
        x * y * cc - z * s,
        x * z * cc + y * s,
        y * x * cc + z * s,
        c + y * y * cc,
        y * z * cc - x * s,
        z * x * cc - y * s,
        z * y * cc + x * s,
        c + z * z * cc,
    )


@wp.func
def _mat44_from(rot: _mat33d, p: _vec3d) -> _mat44d:
    """Assemble a 4x4 transform from a 3x3 rotation and a translation."""
    return _mat44d(
        rot[0, 0],
        rot[0, 1],
        rot[0, 2],
        p[0],
        rot[1, 0],
        rot[1, 1],
        rot[1, 2],
        p[1],
        rot[2, 0],
        rot[2, 1],
        rot[2, 2],
        p[2],
        _f64(0.0),
        _f64(0.0),
        _f64(0.0),
        _f64(1.0),
    )


@wp.kernel
def fk_kernel(
    q: wp.array2d[_f64],
    njoint: int,
    nbody: int,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    xpf: wp.array[_mat44d],
    xbm_inv: wp.array[_mat44d],
    axis_dir: wp.array[_vec3d],
    atype: wp.array[wp.int32],
    acoord: wp.array[wp.int32],
    ap0: wp.array[_f64],
    ap1: wp.array[_f64],
    koff: wp.array[wp.int32],
    kcnt: wp.array[wp.int32],
    kx: wp.array[_f64],
    ky: wp.array[_f64],
    kb: wp.array[_f64],
    kc: wp.array[_f64],
    kd: wp.array[_f64],
    body_X: wp.array2d[_mat44d],
):
    """Compute every body's ground transform for one coordinate vector.

    One thread handles one batch sample. Joints are visited in the topological
    order established on the host, so a joint's parent transform is already
    written when the joint is processed.
    """
    s = wp.tid()
    ident = wp.identity(n=4, dtype=_f64)
    for b in range(nbody):
        body_X[s, b] = ident
    for j in range(njoint):
        rot = wp.identity(n=3, dtype=_f64)
        for k in range(3):
            a = j * 6 + k
            val = _eval_axis(a, s, q, atype, acoord, ap0, ap1, koff, kcnt, kx, ky, kb, kc, kd)
            rot = wp.mul(rot, _rot_axis_mat(axis_dir[a], val))
        pos = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
        for k in range(3, 6):
            a = j * 6 + k
            val = _eval_axis(a, s, q, atype, acoord, ap0, ap1, koff, kcnt, kx, ky, kb, kc, kd)
            pos = pos + axis_dir[a] * val
        xfm = _mat44_from(rot, pos)
        p = joint_parent[j]
        c = joint_child[j]
        body_X[s, c] = wp.mul(wp.mul(wp.mul(body_X[s, p], xpf[j]), xfm), xbm_inv[j])


@wp.kernel
def velocity_stencil_kernel(
    q: wp.array2d[_f64],
    qd: wp.array2d[_f64],
    h: _f64,
    qp: wp.array2d[_f64],
    qm: wp.array2d[_f64],
):
    """Build the ``q +/- h*qd`` velocity stencil on device."""
    b, c = wp.tid()
    delta = h * qd[b, c]
    qp[b, c] = q[b, c] + delta
    qm[b, c] = q[b, c] - delta


@wp.kernel
def acceleration_stencil_kernel(
    q: wp.array2d[_f64],
    qd: wp.array2d[_f64],
    qdd: wp.array2d[_f64],
    dt: _f64,
    qp: wp.array2d[_f64],
    qm: wp.array2d[_f64],
):
    """Build the Taylor-consistent acceleration stencil on device."""
    b, c = wp.tid()
    drift = _f64(0.5) * dt * dt * qdd[b, c]
    delta = dt * qd[b, c]
    qp[b, c] = q[b, c] + delta + drift
    qm[b, c] = q[b, c] - delta + drift


@wp.kernel
def jacobian_stencil_kernel(
    q: wp.array2d[_f64],
    eps: _f64,
    ncoord: int,
    qp: wp.array2d[_f64],
    qm: wp.array2d[_f64],
):
    """Build all ``q +/- eps*e_i`` Jacobian perturbations on device."""
    b, i, c = wp.tid()
    row = b * ncoord + i
    delta = eps if i == c else _f64(0.0)
    qp[row, c] = q[b, c] + delta
    qm[row, c] = q[b, c] - delta


@wp.kernel
def marker_kernel(
    body_X: wp.array2d[_mat44d],
    marker_body: wp.array[wp.int32],
    marker_loc: wp.array[_vec3d],
    out: wp.array2d[_vec3d],
):
    """Place each marker in ground from its body transform and local offset."""
    s, mk = wp.tid()
    x = body_X[s, marker_body[mk]]
    loc = marker_loc[mk]
    h = wp.mul(x, wp.vec4d(loc[0], loc[1], loc[2], _f64(1.0)))
    out[s, mk] = _vec3d(h[0], h[1], h[2])


@wp.kernel
def com_kernel(
    body_X: wp.array2d[_mat44d],
    body_mass: wp.array[_f64],
    body_com: wp.array[_vec3d],
    nbody: int,
    out: wp.array[_vec3d],
):
    """Mass-weighted sum of body center-of-mass positions in ground, per batch.

    ``out[s]`` holds ``sum_b m_b (X_b c_b)`` for configuration ``s``; the caller
    divides by the total mass to obtain the whole-body center of mass.
    """
    s = wp.tid()
    acc = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for b in range(nbody):
        x = body_X[s, b]
        c = body_com[b]
        h = wp.mul(x, wp.vec4d(c[0], c[1], c[2], _f64(1.0)))
        acc = acc + body_mass[b] * _vec3d(h[0], h[1], h[2])
    out[s] = acc


@wp.kernel
def com_velocity_kernel(
    body_x: wp.array2d[_mat44d],
    angular_velocity: wp.array2d[_vec3d],
    linear_velocity: wp.array2d[_vec3d],
    body_mass: wp.array[_f64],
    body_com: wp.array[_vec3d],
    total_mass: _f64,
    nbody: int,
    out: wp.array[_vec3d],
):
    """Compute whole-body center-of-mass velocity from device body kinematics."""
    s = wp.tid()
    value = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for body in range(nbody):
        x = body_x[s, body]
        rotation = _mat33d(x[0, 0], x[0, 1], x[0, 2], x[1, 0], x[1, 1], x[1, 2], x[2, 0], x[2, 1], x[2, 2])
        offset = rotation * body_com[body]
        velocity = linear_velocity[s, body] + wp.cross(angular_velocity[s, body], offset)
        value += body_mass[body] * velocity
    if total_mass > _f64(0.0):
        value /= total_mass
    out[s] = value


@wp.kernel
def com_acceleration_kernel(
    body_x: wp.array2d[_mat44d],
    angular_velocity: wp.array2d[_vec3d],
    angular_acceleration: wp.array2d[_vec3d],
    linear_acceleration: wp.array2d[_vec3d],
    body_mass: wp.array[_f64],
    body_com: wp.array[_vec3d],
    total_mass: _f64,
    nbody: int,
    out: wp.array[_vec3d],
):
    """Compute whole-body center-of-mass acceleration from device body kinematics."""
    s = wp.tid()
    value = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for body in range(nbody):
        x = body_x[s, body]
        rotation = _mat33d(x[0, 0], x[0, 1], x[0, 2], x[1, 0], x[1, 1], x[1, 2], x[2, 0], x[2, 1], x[2, 2])
        offset = rotation * body_com[body]
        omega = angular_velocity[s, body]
        acceleration = linear_acceleration[s, body]
        acceleration += wp.cross(angular_acceleration[s, body], offset)
        acceleration += wp.cross(omega, wp.cross(omega, offset))
        value += body_mass[body] * acceleration
    if total_mass > _f64(0.0):
        value /= total_mass
    out[s] = value


@wp.kernel
def pack_body_vec3_pair_kernel(first: wp.array2d[_vec3d], second: wp.array2d[_vec3d], out: wp.array3d[_f64]):
    """Pack two per-body vec3 fields for one final readback."""
    frame, body = wp.tid()
    a = first[frame, body]
    b = second[frame, body]
    out[frame, body, 0] = a[0]
    out[frame, body, 1] = a[1]
    out[frame, body, 2] = a[2]
    out[frame, body, 3] = b[0]
    out[frame, body, 4] = b[1]
    out[frame, body, 5] = b[2]


@wp.kernel
def pack_vec3_pair_kernel(first: wp.array[_vec3d], second: wp.array[_vec3d], out: wp.array2d[_f64]):
    """Pack two batched vec3 fields for one final readback."""
    frame = wp.tid()
    a = first[frame]
    b = second[frame]
    out[frame, 0] = a[0]
    out[frame, 1] = a[1]
    out[frame, 2] = a[2]
    out[frame, 3] = b[0]
    out[frame, 4] = b[1]
    out[frame, 5] = b[2]


@wp.kernel
def body_velocity_kernel(
    body_Xp: wp.array2d[_mat44d],
    body_Xm: wp.array2d[_mat44d],
    body_X0: wp.array2d[_mat44d],
    inv2h: _f64,
    ang_vel: wp.array2d[_vec3d],
    lin_vel: wp.array2d[_vec3d],
):
    """Body angular and origin-linear velocity in ground from perturbed poses.

    ``body_Xp`` / ``body_Xm`` are the forward-kinematics poses at ``q +/- h*qd``
    and ``body_X0`` at ``q``; ``inv2h`` is ``1 / (2 h)``. Angular velocity is the
    skew (vee) part of ``Rdot R0^T``; linear velocity is the central difference of
    the body-frame origin.
    """
    s, b = wp.tid()
    xp = body_Xp[s, b]
    xm = body_Xm[s, b]
    x0 = body_X0[s, b]
    rdot = wp.mat33d(
        (xp[0, 0] - xm[0, 0]) * inv2h,
        (xp[0, 1] - xm[0, 1]) * inv2h,
        (xp[0, 2] - xm[0, 2]) * inv2h,
        (xp[1, 0] - xm[1, 0]) * inv2h,
        (xp[1, 1] - xm[1, 1]) * inv2h,
        (xp[1, 2] - xm[1, 2]) * inv2h,
        (xp[2, 0] - xm[2, 0]) * inv2h,
        (xp[2, 1] - xm[2, 1]) * inv2h,
        (xp[2, 2] - xm[2, 2]) * inv2h,
    )
    r0 = wp.mat33d(
        x0[0, 0],
        x0[0, 1],
        x0[0, 2],
        x0[1, 0],
        x0[1, 1],
        x0[1, 2],
        x0[2, 0],
        x0[2, 1],
        x0[2, 2],
    )
    w = wp.mul(rdot, wp.transpose(r0))
    ang_vel[s, b] = _vec3d(
        _f64(0.5) * (w[2, 1] - w[1, 2]),
        _f64(0.5) * (w[0, 2] - w[2, 0]),
        _f64(0.5) * (w[1, 0] - w[0, 1]),
    )
    lin_vel[s, b] = _vec3d(
        (xp[0, 3] - xm[0, 3]) * inv2h,
        (xp[1, 3] - xm[1, 3]) * inv2h,
        (xp[2, 3] - xm[2, 3]) * inv2h,
    )


@wp.kernel
def body_jacobian_kernel(
    body_Xp: wp.array2d[_mat44d],
    body_Xm: wp.array2d[_mat44d],
    body_X0: wp.array2d[_mat44d],
    ncoord: int,
    inv2eps: _f64,
    jac: wp.array4d[_f64],
):
    """Spatial (angular-over-linear) Jacobian column of each body per coordinate.

    ``body_Xp`` / ``body_Xm`` hold the forward-kinematics poses at ``q +/- eps*e_i``
    stacked so row ``b*ncoord + i`` is the perturbation of coordinate ``i`` for
    configuration ``b``; ``body_X0`` are the unperturbed poses. Column ``i`` is
    the body-origin spatial velocity for a unit speed of coordinate ``i``: rows
    0-2 are the angular part (skew/vee of ``dR/dq_i R0^T``) and rows 3-5 the
    origin-linear part, both by central difference.
    """
    b, body, i = wp.tid()
    idx = b * ncoord + i
    xp = body_Xp[idx, body]
    xm = body_Xm[idx, body]
    x0 = body_X0[b, body]
    rdot = wp.mat33d(
        (xp[0, 0] - xm[0, 0]) * inv2eps,
        (xp[0, 1] - xm[0, 1]) * inv2eps,
        (xp[0, 2] - xm[0, 2]) * inv2eps,
        (xp[1, 0] - xm[1, 0]) * inv2eps,
        (xp[1, 1] - xm[1, 1]) * inv2eps,
        (xp[1, 2] - xm[1, 2]) * inv2eps,
        (xp[2, 0] - xm[2, 0]) * inv2eps,
        (xp[2, 1] - xm[2, 1]) * inv2eps,
        (xp[2, 2] - xm[2, 2]) * inv2eps,
    )
    r0 = wp.mat33d(
        x0[0, 0],
        x0[0, 1],
        x0[0, 2],
        x0[1, 0],
        x0[1, 1],
        x0[1, 2],
        x0[2, 0],
        x0[2, 1],
        x0[2, 2],
    )
    w = wp.mul(rdot, wp.transpose(r0))
    jac[b, body, 0, i] = _f64(0.5) * (w[2, 1] - w[1, 2])
    jac[b, body, 1, i] = _f64(0.5) * (w[0, 2] - w[2, 0])
    jac[b, body, 2, i] = _f64(0.5) * (w[1, 0] - w[0, 1])
    jac[b, body, 3, i] = (xp[0, 3] - xm[0, 3]) * inv2eps
    jac[b, body, 4, i] = (xp[1, 3] - xm[1, 3]) * inv2eps
    jac[b, body, 5, i] = (xp[2, 3] - xm[2, 3]) * inv2eps


@wp.kernel
def load_project_kernel(
    jac_body: wp.array3d[_f64],
    wrench: wp.array2d[_f64],
    tau: wp.array2d[_f64],
):
    """Project a per-configuration body wrench onto generalized forces.

    ``jac_body`` is one body's spatial Jacobian ``[batch, 6, num_coordinates]``
    and ``wrench`` the ground-frame ``[torque, force]`` referred to that body's
    origin ``[batch, 6]``. Column ``i`` of the generalized force is the Jacobian
    column dotted with the wrench (transposed-Jacobian projection).
    """
    b, i = wp.tid()
    acc = _f64(0.0)
    for r in range(6):
        acc += jac_body[b, r, i] * wrench[b, r]
    tau[b, i] = acc


@wp.kernel
def body_load_project_kernel(
    jac: wp.array4d[_f64],
    body_x: wp.array2d[_mat44d],
    body: int,
    point: _vec3d,
    force: wp.array[_vec3d],
    torque: wp.array[_vec3d],
    tau: wp.array2d[_f64],
):
    """Assemble a body-local point load and project it without host intermediates."""
    b, i = wp.tid()
    x = body_x[b, body]
    r = _vec3d(
        x[0, 0] * point[0] + x[0, 1] * point[1] + x[0, 2] * point[2],
        x[1, 0] * point[0] + x[1, 1] * point[1] + x[1, 2] * point[2],
        x[2, 0] * point[0] + x[2, 1] * point[1] + x[2, 2] * point[2],
    )
    moment = torque[b] + wp.cross(r, force[b])
    acc = jac[b, body, 0, i] * moment[0]
    acc += jac[b, body, 1, i] * moment[1]
    acc += jac[b, body, 2, i] * moment[2]
    acc += jac[b, body, 3, i] * force[b][0]
    acc += jac[b, body, 4, i] * force[b][1]
    acc += jac[b, body, 5, i] * force[b][2]
    tau[b, i] = acc


@wp.kernel
def momentum_kernel(
    body_X0: wp.array2d[_mat44d],
    ang_vel: wp.array2d[_vec3d],
    lin_vel: wp.array2d[_vec3d],
    body_mass: wp.array[_f64],
    body_com: wp.array[_vec3d],
    body_inertia: wp.array[wp.mat33d],
    total_mass: _f64,
    nbody: int,
    lin_mom: wp.array[_vec3d],
    ang_mom: wp.array[_vec3d],
):
    """Whole-body linear momentum and angular momentum about the whole-body COM.

    Assembles per body: COM position/velocity in ground (from origin velocity and
    ``omega x (R c)``), the ground inertia ``R I R^T``, then ``P = sum m v`` and
    ``H = sum (I omega + m (r - r_COM) x v)`` for configuration ``s``.
    """
    s = wp.tid()
    inv_m = _f64(1.0) / total_mass
    r_com = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for b in range(nbody):
        x = body_X0[s, b]
        c = body_com[b]
        h = wp.mul(x, wp.vec4d(c[0], c[1], c[2], _f64(1.0)))
        r_com = r_com + body_mass[b] * _vec3d(h[0], h[1], h[2])
    r_com = r_com * inv_m
    lin = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    ang = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for b in range(nbody):
        x = body_X0[s, b]
        r0 = wp.mat33d(
            x[0, 0],
            x[0, 1],
            x[0, 2],
            x[1, 0],
            x[1, 1],
            x[1, 2],
            x[2, 0],
            x[2, 1],
            x[2, 2],
        )
        c = body_com[b]
        off = wp.mul(r0, _vec3d(c[0], c[1], c[2]))
        r_b = _vec3d(x[0, 3], x[1, 3], x[2, 3]) + off
        w = ang_vel[s, b]
        v_b = lin_vel[s, b] + wp.cross(w, off)
        m = body_mass[b]
        lin = lin + m * v_b
        i_ground = wp.mul(wp.mul(r0, body_inertia[b]), wp.transpose(r0))
        ang = ang + wp.mul(i_ground, w) + m * wp.cross(r_b - r_com, v_b)
    lin_mom[s] = lin
    ang_mom[s] = ang


@wp.kernel
def body_acceleration_kernel(
    body_Xp: wp.array2d[_mat44d],
    body_Xm: wp.array2d[_mat44d],
    body_X0: wp.array2d[_mat44d],
    inv_dt2: _f64,
    ang_acc: wp.array2d[_vec3d],
    lin_acc: wp.array2d[_vec3d],
):
    """Body angular and origin-linear acceleration in ground from perturbed poses.

    ``body_Xp`` / ``body_Xm`` are the forward-kinematics poses at
    ``q +/- dt*qd + 0.5*dt^2*qdd`` and ``body_X0`` at ``q``; ``inv_dt2`` is
    ``1 / dt^2``. Angular acceleration is the skew (vee) part of ``Rddot R0^T``
    (the symmetric ``omega^2`` term drops out); linear acceleration is the second
    central difference of the body-frame origin.
    """
    s, b = wp.tid()
    xp = body_Xp[s, b]
    xm = body_Xm[s, b]
    x0 = body_X0[s, b]
    rddot = wp.mat33d(
        (xp[0, 0] + xm[0, 0] - _f64(2.0) * x0[0, 0]) * inv_dt2,
        (xp[0, 1] + xm[0, 1] - _f64(2.0) * x0[0, 1]) * inv_dt2,
        (xp[0, 2] + xm[0, 2] - _f64(2.0) * x0[0, 2]) * inv_dt2,
        (xp[1, 0] + xm[1, 0] - _f64(2.0) * x0[1, 0]) * inv_dt2,
        (xp[1, 1] + xm[1, 1] - _f64(2.0) * x0[1, 1]) * inv_dt2,
        (xp[1, 2] + xm[1, 2] - _f64(2.0) * x0[1, 2]) * inv_dt2,
        (xp[2, 0] + xm[2, 0] - _f64(2.0) * x0[2, 0]) * inv_dt2,
        (xp[2, 1] + xm[2, 1] - _f64(2.0) * x0[2, 1]) * inv_dt2,
        (xp[2, 2] + xm[2, 2] - _f64(2.0) * x0[2, 2]) * inv_dt2,
    )
    r0 = wp.mat33d(
        x0[0, 0],
        x0[0, 1],
        x0[0, 2],
        x0[1, 0],
        x0[1, 1],
        x0[1, 2],
        x0[2, 0],
        x0[2, 1],
        x0[2, 2],
    )
    a = wp.mul(rddot, wp.transpose(r0))
    ang_acc[s, b] = _vec3d(
        _f64(0.5) * (a[2, 1] - a[1, 2]),
        _f64(0.5) * (a[0, 2] - a[2, 0]),
        _f64(0.5) * (a[1, 0] - a[0, 1]),
    )
    lin_acc[s, b] = _vec3d(
        (xp[0, 3] + xm[0, 3] - _f64(2.0) * x0[0, 3]) * inv_dt2,
        (xp[1, 3] + xm[1, 3] - _f64(2.0) * x0[1, 3]) * inv_dt2,
        (xp[2, 3] + xm[2, 3] - _f64(2.0) * x0[2, 3]) * inv_dt2,
    )


# -----------------------------------------------------------------------------
# Host flattening of the joint tree into kernel-friendly arrays
# -----------------------------------------------------------------------------
def _unwrap_multiplier(ftype: str | None, params: dict) -> tuple[str | None, dict, float]:
    """Fold nested ``MultiplierFunction`` scales into a single scalar."""
    scale = 1.0
    while ftype == "MultiplierFunction":
        scale *= float(params.get("scale", 1.0))
        params = params.get("inner", {}) or {}
        ftype = params.get("type")
    return ftype, params, scale


def _synth_axes(joint: OsimJoint) -> list[tuple]:
    """Return six ``(axis, function_type, params, coord_name)`` tuples for a joint.

    Standard joints are expressed in the same six-axis ``CustomJoint`` form (three
    rotations then three translations) so the kernel handles a single joint type.
    """
    jt = joint.type
    coords = [c.name for c in joint.coordinates]
    lin = ("LinearFunction", {"coefficients": [1.0, 0.0]})
    con = ("Constant", {"value": 0.0})

    def c_axis() -> tuple:
        return ((0.0, 0.0, 1.0), con[0], con[1], None)

    def l_axis(axis, name) -> tuple:
        return (axis, lin[0], lin[1], name)

    if jt == "CustomJoint":
        out = []
        for ax in joint.spatial_transform:
            name = ax.coordinates[0] if ax.coordinates else None
            out.append((tuple(ax.axis), ax.function_type, ax.function, name))
        return out
    if jt in ("WeldJoint", "") or not coords:
        return [c_axis() for _ in range(6)]
    if jt == "PinJoint":
        return [l_axis((0.0, 0.0, 1.0), coords[0]), c_axis(), c_axis(), c_axis(), c_axis(), c_axis()]
    if jt == "SliderJoint":
        return [c_axis(), c_axis(), c_axis(), l_axis((1.0, 0.0, 0.0), coords[0]), c_axis(), c_axis()]
    if jt == "BallJoint":
        return [
            l_axis((1.0, 0.0, 0.0), coords[0]),
            l_axis((0.0, 1.0, 0.0), coords[1]),
            l_axis((0.0, 0.0, 1.0), coords[2]),
            c_axis(),
            c_axis(),
            c_axis(),
        ]
    if jt == "UniversalJoint":
        return [
            l_axis((1.0, 0.0, 0.0), coords[0]),
            l_axis((0.0, 1.0, 0.0), coords[1]),
            c_axis(),
            c_axis(),
            c_axis(),
            c_axis(),
        ]
    if jt == "PlanarJoint":
        return [
            l_axis((0.0, 0.0, 1.0), coords[0]),
            c_axis(),
            c_axis(),
            l_axis((1.0, 0.0, 0.0), coords[1]),
            l_axis((0.0, 1.0, 0.0), coords[2]),
            c_axis(),
        ]
    if jt == "FreeJoint":
        return [
            l_axis((1.0, 0.0, 0.0), coords[0]),
            l_axis((0.0, 1.0, 0.0), coords[1]),
            l_axis((0.0, 0.0, 1.0), coords[2]),
            l_axis((1.0, 0.0, 0.0), coords[3]),
            l_axis((0.0, 1.0, 0.0), coords[4]),
            l_axis((0.0, 0.0, 1.0), coords[5]),
        ]
    return [c_axis() for _ in range(6)]


def _bake_function(ftype: str | None, params: dict) -> tuple:
    """Return ``(type_code, p0, p1, x, y, b, c, d)`` for an axis function.

    Any ``MultiplierFunction`` scale is folded in; ``SimmSpline`` coefficients are
    precomputed with ``SimmSpline`` so the
    kernel only evaluates the cubic.
    """
    ftype, params, scale = _unwrap_multiplier(ftype, params)
    if ftype == "LinearFunction":
        coeffs = params.get("coefficients")
        if coeffs is not None:
            coeffs = list(coeffs)
            slope = coeffs[0] if coeffs else 1.0
            intercept = coeffs[1] if len(coeffs) > 1 else 0.0
        else:
            slope = params.get("slope", 1.0)
            intercept = params.get("intercept", 0.0)
        return (_FN_LINEAR, slope * scale, intercept * scale, [], [], [], [], [])
    if ftype == "Constant":
        return (_FN_CONSTANT, params.get("value", 0.0) * scale, 0.0, [], [], [], [], [])
    if ftype in ("SimmSpline", "NaturalCubicSpline", "GCVSpline"):
        x = np.asarray(params.get("x", []), float)
        y = np.asarray(params.get("y", []), float) * scale
        sp = SimmSpline(x, y)
        return (_FN_SIMMSPLINE, 0.0, 0.0, list(x), list(y), list(sp._b), list(sp._c), list(sp._d))
    if ftype == "PiecewiseLinearFunction":
        x = np.asarray(params.get("x", []), float)
        y = np.asarray(params.get("y", []), float) * scale
        return (_FN_PIECEWISE, 0.0, 0.0, list(x), list(y), [], [], [])
    return (_FN_CONSTANT, 0.0, 0.0, [], [], [], [], [])


def _inertia_matrix(inertia: tuple[float, float, float, float, float, float]) -> list[list[float]]:
    """Expand an OpenSim inertia 6-vector ``[Ixx Iyy Izz Ixy Ixz Iyz]`` to a symmetric 3x3."""
    ixx, iyy, izz, ixy, ixz, iyz = inertia
    return [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]


class ForwardKinematics:
    """Warp forward kinematics: body transforms and marker positions from ``q``.

    The model's joint tree is flattened once into device arrays; evaluating a
    pose (or a batch of poses) launches :func:`fk_kernel` and :func:`marker_kernel`.

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernels (``"cpu"``, ``"cuda"``, a
            :class:`warp.context.Device`, or ``None`` for the CPU). Batched
            solves benefit from a CUDA device.

    Attributes:
        coordinate_names: Generalized coordinate names in model order.
        coordinate_motion: Map of coordinate name to motion type.
        marker_names: Model marker names in output-column order.
        device: The Warp device the kernels run on.
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.device = wp.get_device(device) if device is not None else wp.get_device("cpu")
        self.coordinate_names: list[str] = [c.name for j in model.joints for c in j.coordinates]
        self.coordinate_motion: dict[str, str] = {c.name: c.motion_type for j in model.joints for c in j.coordinates}
        self._index = {name: i for i, name in enumerate(self.coordinate_names)}
        self.ncoord = len(self.coordinate_names)
        self.order = self._topological_order()
        self._flatten()
        self._upload()

    def _topological_order(self) -> list[OsimJoint]:
        placed = {"ground"}
        order: list[OsimJoint] = []
        remaining = list(self.model.joints)
        while remaining:
            progressed = False
            for j in list(remaining):
                if j.parent_body in placed:
                    order.append(j)
                    placed.add(j.child_body)
                    remaining.remove(j)
                    progressed = True
            if not progressed:
                raise ValueError("Joint tree has a cycle or a missing parent: " + ", ".join(j.name for j in remaining))
        return order

    def _flatten(self) -> None:
        model = self.model
        body_names = ["ground"] + [b.name for b in model.bodies]
        bidx = {n: i for i, n in enumerate(body_names)}
        self.body_names = body_names
        self.nbody = len(body_names)
        joints = self.order
        self.njoint = len(joints)

        self._joint_parent = np.array([bidx[j.parent_body] for j in joints], np.int32)
        self._joint_child = np.array([bidx[j.child_body] for j in joints], np.int32)
        xpf = []
        xbm_inv = []
        for j in joints:
            xpf.append(
                make_transform(euler_xyz_to_matrix(*j.parent_transform.orientation), j.parent_transform.translation)
            )
            x_bm = make_transform(euler_xyz_to_matrix(*j.child_transform.orientation), j.child_transform.translation)
            xbm_inv.append(np.linalg.inv(x_bm))
        self._xpf = np.stack(xpf).astype(np.float64) if joints else np.zeros((0, 4, 4))
        self._xbm_inv = np.stack(xbm_inv).astype(np.float64) if joints else np.zeros((0, 4, 4))

        adir, atype, acoord, ap0, ap1 = [], [], [], [], []
        koff, kcnt, kx, ky, kb, kc, kd = [], [], [], [], [], [], []
        for j in joints:
            for axis, ftype, params, name in _synth_axes(j):
                a = np.asarray(axis, float)
                n = float(np.linalg.norm(a))
                adir.append(a / n if n > 1.0e-12 else np.array([0.0, 0.0, 1.0]))
                code, p0, p1, x, y, b, c, d = _bake_function(ftype, params)
                atype.append(code)
                ap0.append(p0)
                ap1.append(p1)
                acoord.append(self._index.get(name, -1) if name else -1)
                if code in (_FN_SIMMSPLINE, _FN_PIECEWISE):
                    koff.append(len(kx))
                    kcnt.append(len(x))
                    kx.extend(x)
                    ky.extend(y)
                    if code == _FN_SIMMSPLINE:
                        kb.extend(b)
                        kc.extend(c)
                        kd.extend(d)
                    else:
                        kb.extend([0.0] * len(x))
                        kc.extend([0.0] * len(x))
                        kd.extend([0.0] * len(x))
                else:
                    koff.append(0)
                    kcnt.append(0)
        self._axis_dir = np.asarray(adir, np.float64)
        self._axis_type = np.asarray(atype, np.int32)
        self._axis_coord = np.asarray(acoord, np.int32)
        self._axis_p0 = np.asarray(ap0, np.float64)
        self._axis_p1 = np.asarray(ap1, np.float64)
        self._koff = np.asarray(koff, np.int32)
        self._kcnt = np.asarray(kcnt, np.int32)
        self._kx = np.asarray(kx or [0.0], np.float64)
        self._ky = np.asarray(ky or [0.0], np.float64)
        self._kb = np.asarray(kb or [0.0], np.float64)
        self._kc = np.asarray(kc or [0.0], np.float64)
        self._kd = np.asarray(kd or [0.0], np.float64)

        self.marker_names = [m.name for m in model.markers]
        self._marker_body = np.array([bidx.get(m.body, 0) for m in model.markers], np.int32)
        self._marker_loc = np.asarray([m.location for m in model.markers], np.float64).reshape(-1, 3)
        self.nmarker = len(self.marker_names)

        # Per-body inertial data for whole-body center of mass ("ground" body has zero mass).
        self._body_mass = np.array([0.0] + [b.mass for b in model.bodies], np.float64)
        self._body_com = np.asarray([(0.0, 0.0, 0.0)] + [b.mass_center for b in model.bodies], np.float64).reshape(
            -1, 3
        )
        self.total_mass = float(self._body_mass.sum())
        # Per-body inertia tensor about its COM (6-vector -> symmetric 3x3), ground body zero.
        self._body_inertia = np.array(
            [[[0.0] * 3] * 3] + [_inertia_matrix(b.inertia) for b in model.bodies], np.float64
        ).reshape(-1, 3, 3)

    def _upload(self) -> None:
        dev = self.device
        self.d_joint_parent = wp.array(self._joint_parent, dtype=wp.int32, device=dev)
        self.d_joint_child = wp.array(self._joint_child, dtype=wp.int32, device=dev)
        self.d_xpf = wp.array(self._xpf.reshape(-1, 16), dtype=_mat44d, device=dev)
        self.d_xbm_inv = wp.array(self._xbm_inv.reshape(-1, 16), dtype=_mat44d, device=dev)
        self.d_axis_dir = wp.array(self._axis_dir, dtype=_vec3d, device=dev)
        self.d_axis_type = wp.array(self._axis_type, dtype=wp.int32, device=dev)
        self.d_axis_coord = wp.array(self._axis_coord, dtype=wp.int32, device=dev)
        self.d_axis_p0 = wp.array(self._axis_p0, dtype=_f64, device=dev)
        self.d_axis_p1 = wp.array(self._axis_p1, dtype=_f64, device=dev)
        self.d_koff = wp.array(self._koff, dtype=wp.int32, device=dev)
        self.d_kcnt = wp.array(self._kcnt, dtype=wp.int32, device=dev)
        self.d_kx = wp.array(self._kx, dtype=_f64, device=dev)
        self.d_ky = wp.array(self._ky, dtype=_f64, device=dev)
        self.d_kb = wp.array(self._kb, dtype=_f64, device=dev)
        self.d_kc = wp.array(self._kc, dtype=_f64, device=dev)
        self.d_kd = wp.array(self._kd, dtype=_f64, device=dev)
        self.d_marker_body = wp.array(self._marker_body, dtype=wp.int32, device=dev)
        self.d_marker_loc = wp.array(self._marker_loc, dtype=_vec3d, device=dev)
        self.d_body_mass = wp.array(self._body_mass, dtype=_f64, device=dev)
        self.d_body_com = wp.array(self._body_com, dtype=_vec3d, device=dev)
        self.d_body_inertia = wp.array(self._body_inertia.reshape(-1, 9), dtype=wp.mat33d, device=dev)

    # ---- coordinate helpers -------------------------------------------------
    def _to_vector(self, q) -> np.ndarray:
        if isinstance(q, dict):
            arr = np.zeros(self.ncoord)
            for name, val in q.items():
                if name in self._index:
                    arr[self._index[name]] = val
            return arr
        return np.asarray(q, dtype=float).reshape(-1)

    # ---- device launches ----------------------------------------------------
    def _launch_body_transforms(self, q_wp: wp.array) -> wp.array:
        """Launch :func:`fk_kernel` for a device batch ``q_wp`` [B, ncoord]."""
        batch = q_wp.shape[0]
        body_x = wp.empty((batch, self.nbody), dtype=_mat44d, device=self.device)
        wp.launch(
            fk_kernel,
            dim=batch,
            inputs=[
                q_wp,
                self.njoint,
                self.nbody,
                self.d_joint_parent,
                self.d_joint_child,
                self.d_xpf,
                self.d_xbm_inv,
                self.d_axis_dir,
                self.d_axis_type,
                self.d_axis_coord,
                self.d_axis_p0,
                self.d_axis_p1,
                self.d_koff,
                self.d_kcnt,
                self.d_kx,
                self.d_ky,
                self.d_kb,
                self.d_kc,
                self.d_kd,
                body_x,
            ],
            device=self.device,
        )
        return body_x

    def _launch_markers(self, q_wp: wp.array) -> wp.array:
        """Return device marker positions [B, nmarker] for a device batch ``q_wp``."""
        body_x = self._launch_body_transforms(q_wp)
        batch = q_wp.shape[0]
        pos = wp.empty((batch, self.nmarker), dtype=_vec3d, device=self.device)
        wp.launch(
            marker_kernel,
            dim=(batch, self.nmarker),
            inputs=[body_x, self.d_marker_body, self.d_marker_loc, pos],
            device=self.device,
        )
        return pos

    def _launch_center_of_mass(self, body_x: wp.array[_mat44d]) -> wp.array[_vec3d]:
        """Return whole-body center-of-mass positions from device body transforms."""
        out = wp.empty(body_x.shape[0], dtype=_vec3d, device=self.device)
        wp.launch(
            com_kernel,
            dim=body_x.shape[0],
            inputs=[body_x, self.d_body_mass, self.d_body_com, self.nbody, out],
            device=self.device,
        )
        return out

    def _launch_com_velocity(
        self,
        body_x: wp.array[_mat44d],
        angular_velocity: wp.array[_vec3d],
        linear_velocity: wp.array[_vec3d],
    ) -> wp.array[_vec3d]:
        """Return whole-body COM velocity from device body kinematics."""
        out = wp.empty(body_x.shape[0], dtype=_vec3d, device=self.device)
        wp.launch(
            com_velocity_kernel,
            dim=body_x.shape[0],
            inputs=[
                body_x,
                angular_velocity,
                linear_velocity,
                self.d_body_mass,
                self.d_body_com,
                _f64(self.total_mass),
                self.nbody,
                out,
            ],
            device=self.device,
        )
        return out

    def _launch_com_acceleration(
        self,
        body_x: wp.array[_mat44d],
        angular_velocity: wp.array[_vec3d],
        angular_acceleration: wp.array[_vec3d],
        linear_acceleration: wp.array[_vec3d],
    ) -> wp.array[_vec3d]:
        """Return whole-body COM acceleration from device body kinematics."""
        out = wp.empty(body_x.shape[0], dtype=_vec3d, device=self.device)
        wp.launch(
            com_acceleration_kernel,
            dim=body_x.shape[0],
            inputs=[
                body_x,
                angular_velocity,
                angular_acceleration,
                linear_acceleration,
                self.d_body_mass,
                self.d_body_com,
                _f64(self.total_mass),
                self.nbody,
                out,
            ],
            device=self.device,
        )
        return out

    def _launch_body_velocities(
        self, q_wp: wp.array[_f64], speeds_wp: wp.array[_f64], h: float
    ) -> tuple[wp.array[_vec3d], wp.array[_vec3d], wp.array[_mat44d]]:
        """Return device body velocities and unperturbed poses from device state arrays."""
        batch = q_wp.shape[0]
        qp = wp.empty(q_wp.shape, dtype=_f64, device=self.device)
        qm = wp.empty(q_wp.shape, dtype=_f64, device=self.device)
        wp.launch(
            velocity_stencil_kernel,
            dim=q_wp.shape,
            inputs=[q_wp, speeds_wp, _f64(h), qp, qm],
            device=self.device,
        )
        xp = self._launch_body_transforms(qp)
        xm = self._launch_body_transforms(qm)
        x0 = self._launch_body_transforms(q_wp)
        ang = wp.empty((batch, self.nbody), dtype=_vec3d, device=self.device)
        lin = wp.empty((batch, self.nbody), dtype=_vec3d, device=self.device)
        wp.launch(
            body_velocity_kernel,
            dim=(batch, self.nbody),
            inputs=[xp, xm, x0, _f64(1.0 / (2.0 * h)), ang, lin],
            device=self.device,
        )
        return ang, lin, x0

    def _launch_body_accelerations(
        self,
        q_wp: wp.array[_f64],
        speeds_wp: wp.array[_f64],
        accels_wp: wp.array[_f64],
        dt: float,
        body_x: wp.array[_mat44d] | None = None,
    ) -> tuple[wp.array[_vec3d], wp.array[_vec3d], wp.array[_mat44d]]:
        """Return device body accelerations and unperturbed poses from device state arrays."""
        batch = q_wp.shape[0]
        qp = wp.empty(q_wp.shape, dtype=_f64, device=self.device)
        qm = wp.empty(q_wp.shape, dtype=_f64, device=self.device)
        wp.launch(
            acceleration_stencil_kernel,
            dim=q_wp.shape,
            inputs=[q_wp, speeds_wp, accels_wp, _f64(dt), qp, qm],
            device=self.device,
        )
        xp = self._launch_body_transforms(qp)
        xm = self._launch_body_transforms(qm)
        x0 = self._launch_body_transforms(q_wp) if body_x is None else body_x
        ang = wp.empty((batch, self.nbody), dtype=_vec3d, device=self.device)
        lin = wp.empty((batch, self.nbody), dtype=_vec3d, device=self.device)
        wp.launch(
            body_acceleration_kernel,
            dim=(batch, self.nbody),
            inputs=[xp, xm, x0, _f64(1.0 / (dt * dt)), ang, lin],
            device=self.device,
        )
        return ang, lin, x0

    def _launch_coordinate_perturbations(
        self, q_wp: wp.array[_f64], eps: float
    ) -> tuple[wp.array[_mat44d], wp.array[_mat44d]]:
        """Return body transforms for every ``q +/- eps*e_i`` perturbation."""
        batch, nc = q_wp.shape
        qp = wp.empty((batch * nc, nc), dtype=_f64, device=self.device)
        qm = wp.empty((batch * nc, nc), dtype=_f64, device=self.device)
        wp.launch(
            jacobian_stencil_kernel,
            dim=(batch, nc, nc),
            inputs=[q_wp, _f64(eps), nc, qp, qm],
            device=self.device,
        )
        return self._launch_body_transforms(qp), self._launch_body_transforms(qm)

    def _launch_body_jacobian_device(
        self, q_wp: wp.array[_f64], eps: float
    ) -> tuple[wp.array[_f64], wp.array[_mat44d]]:
        """Return device body Jacobians and unperturbed poses from device coordinates."""
        batch, nc = q_wp.shape
        xp, xm = self._launch_coordinate_perturbations(q_wp, eps)
        x0 = self._launch_body_transforms(q_wp)
        jac = wp.empty((batch, self.nbody, 6, nc), dtype=_f64, device=self.device)
        wp.launch(
            body_jacobian_kernel,
            dim=(batch, self.nbody, nc),
            inputs=[xp, xm, x0, nc, _f64(1.0 / (2.0 * eps)), jac],
            device=self.device,
        )
        return jac, x0

    # ---- public batched API -------------------------------------------------
    def marker_positions_batch(self, coords: np.ndarray) -> np.ndarray:
        """Return marker positions [m] for a batch of coordinate vectors.

        Args:
            coords: Coordinate values, shape ``[batch, num_coordinates]`` in
                native units (radians/meters).

        Returns:
            Ground marker positions, shape ``[batch, num_markers, 3]``, column
            order :attr:`marker_names`.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        return self._launch_markers(q_wp).numpy()

    def body_transforms_batch(self, coords: np.ndarray) -> np.ndarray:
        """Return body ground transforms [batch, num_bodies, 4, 4] for a batch."""
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        return self._launch_body_transforms(q_wp).numpy()

    def center_of_mass_batch(self, coords: np.ndarray) -> np.ndarray:
        """Return the whole-body center of mass [m] for a batch of coordinate vectors.

        Args:
            coords: Coordinate values, shape ``[batch, num_coordinates]`` in native
                units (radians/meters).

        Returns:
            Whole-body center of mass in ground, shape ``[batch, 3]``. Returns the
            (unweighted) origin when the model has no mass.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        body_x = self._launch_body_transforms(q_wp)
        out = self._launch_center_of_mass(body_x)
        return out.numpy() / (self.total_mass if self.total_mass > 0.0 else 1.0)

    def body_velocities_batch(self, coords: np.ndarray, speeds: np.ndarray, h: float = 1.0e-6) -> dict[str, np.ndarray]:
        """Return body angular and linear velocities in ground for a batch.

        Velocities are formed by a central difference of the forward-kinematics
        poses along the coordinate speeds (``q +/- h*qd``): angular velocity from
        the skew part of ``Rdot R^T`` and linear velocity of each body-frame
        origin. Exact in the limit of small ``h`` for the analytic body Jacobian.

        Args:
            coords: Coordinate values [batch, num_coordinates] (radians/meters).
            speeds: Coordinate speeds [batch, num_coordinates] (rad/s or m/s).
            h: Central-difference step in coordinate units.

        Returns:
            Dict with ``angular_velocity`` [rad/s] and ``linear_velocity`` [m/s],
            each shaped ``[batch, num_bodies, 3]`` in :attr:`body_names` order.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        speeds = np.ascontiguousarray(speeds, dtype=np.float64)
        if speeds.shape != coords.shape:
            raise ValueError("speeds must match coords shape [batch, num_coordinates]")
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        speeds_wp = wp.array(speeds, dtype=_f64, device=self.device)
        ang, lin, _ = self._launch_body_velocities(q_wp, speeds_wp, h)
        packed = wp.empty((coords.shape[0], self.nbody, 6), dtype=_f64, device=self.device)
        wp.launch(
            pack_body_vec3_pair_kernel,
            dim=(coords.shape[0], self.nbody),
            inputs=[ang, lin, packed],
            device=self.device,
        )
        data = packed.numpy()
        return {"angular_velocity": data[:, :, :3], "linear_velocity": data[:, :, 3:]}

    def body_jacobian_batch(self, coords: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
        """Return the spatial Jacobian of every body for a batch of configurations.

        Column ``i`` of a body's Jacobian is the ground-frame spatial velocity of
        the body-frame origin produced by a unit speed of coordinate ``i`` (all
        other speeds zero): rows 0-2 the angular part, rows 3-5 the linear part.
        It is formed by central-differencing the forward-kinematics poses along
        each coordinate (``q +/- eps*e_i``) and is exact for the analytic body
        Jacobian in the limit of small ``eps``. Satisfies ``J @ qd`` equal to the
        body spatial velocity from :meth:`body_velocities_batch`.

        Args:
            coords: Coordinate values [batch, num_coordinates] (radians/meters).
            eps: Central-difference step in coordinate units.

        Returns:
            Body spatial Jacobians ``[batch, num_bodies, 6, num_coordinates]`` in
            :attr:`body_names` order; the 6 rows are ``[wx, wy, wz, vx, vy, vz]``.
        """
        return self._launch_body_jacobian(coords, eps).numpy()

    def _launch_body_jacobian(self, coords: np.ndarray, eps: float = 1.0e-6):
        """Launch :func:`body_jacobian_kernel`; return the device Jacobian [batch, nbody, 6, ncoord]."""
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        if coords.shape[1] != self.ncoord:
            raise ValueError(f"coords has {coords.shape[1]} columns, expected {self.ncoord}")
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        jac, _ = self._launch_body_jacobian_device(q_wp, eps)
        return jac

    def whole_body_momentum_batch(
        self, coords: np.ndarray, speeds: np.ndarray, h: float = 1.0e-6
    ) -> dict[str, np.ndarray]:
        """Return whole-body linear and angular momentum for a batch.

        Linear momentum is ``sum_b m_b v_b`` and angular momentum is taken about the
        whole-body center of mass, ``sum_b (I_b omega_b + m_b (r_b - r_COM) x v_b)``,
        with body center-of-mass velocities, ground inertias, and the whole-body COM
        assembled on-device from the forward-kinematics poses and velocities.

        Args:
            coords: Coordinate values [batch, num_coordinates] (radians/meters).
            speeds: Coordinate speeds [batch, num_coordinates] (rad/s or m/s).
            h: Central-difference step used for the body velocities.

        Returns:
            Dict with ``linear_momentum`` [kg*m/s] and ``angular_momentum``
            [kg*m^2/s], each shaped ``[batch, 3]`` in ground.
        """
        if self.total_mass <= 0.0:
            raise ValueError("Model has no mass; whole-body momentum is undefined")
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        speeds = np.ascontiguousarray(speeds, dtype=np.float64)
        if speeds.shape != coords.shape:
            raise ValueError("speeds must match coords shape [batch, num_coordinates]")
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        speeds_wp = wp.array(speeds, dtype=_f64, device=self.device)
        ang, lin, x0 = self._launch_body_velocities(q_wp, speeds_wp, h)
        batch = coords.shape[0]
        lin_mom = wp.empty(batch, dtype=_vec3d, device=self.device)
        ang_mom = wp.empty(batch, dtype=_vec3d, device=self.device)
        wp.launch(
            momentum_kernel,
            dim=batch,
            inputs=[
                x0,
                ang,
                lin,
                self.d_body_mass,
                self.d_body_com,
                self.d_body_inertia,
                _f64(self.total_mass),
                self.nbody,
                lin_mom,
                ang_mom,
            ],
            device=self.device,
        )
        packed = wp.empty((batch, 6), dtype=_f64, device=self.device)
        wp.launch(pack_vec3_pair_kernel, dim=batch, inputs=[lin_mom, ang_mom, packed], device=self.device)
        data = packed.numpy()
        return {"linear_momentum": data[:, :3], "angular_momentum": data[:, 3:]}

    def body_accelerations_batch(
        self, coords: np.ndarray, speeds: np.ndarray, accels: np.ndarray, dt: float = 1.0e-4
    ) -> dict[str, np.ndarray]:
        """Return body angular and linear accelerations in ground for a batch.

        Accelerations are formed by a Taylor-consistent second central difference of
        the forward-kinematics poses along the trajectory (``q +/- dt*qd +
        0.5*dt^2*qdd``): angular acceleration from the skew part of ``Rddot R^T`` and
        linear acceleration of each body-frame origin. Exact in the small-``dt`` limit.

        Args:
            coords: Coordinate values [batch, num_coordinates] (radians/meters).
            speeds: Coordinate speeds [batch, num_coordinates] (rad/s or m/s).
            accels: Coordinate accelerations [batch, num_coordinates] (rad/s^2 or m/s^2).
            dt: Central-difference step in seconds.

        Returns:
            Dict with ``angular_acceleration`` [rad/s^2] and ``linear_acceleration``
            [m/s^2], each shaped ``[batch, num_bodies, 3]`` in :attr:`body_names` order.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        speeds = np.ascontiguousarray(speeds, dtype=np.float64)
        accels = np.ascontiguousarray(accels, dtype=np.float64)
        if speeds.shape != coords.shape or accels.shape != coords.shape:
            raise ValueError("speeds and accels must match coords shape [batch, num_coordinates]")
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        speeds_wp = wp.array(speeds, dtype=_f64, device=self.device)
        accels_wp = wp.array(accels, dtype=_f64, device=self.device)
        ang, lin, _ = self._launch_body_accelerations(q_wp, speeds_wp, accels_wp, dt)
        packed = wp.empty((coords.shape[0], self.nbody, 6), dtype=_f64, device=self.device)
        wp.launch(
            pack_body_vec3_pair_kernel,
            dim=(coords.shape[0], self.nbody),
            inputs=[ang, lin, packed],
            device=self.device,
        )
        data = packed.numpy()
        return {"angular_acceleration": data[:, :, :3], "linear_acceleration": data[:, :, 3:]}

    # ---- single-pose convenience API ---------------------------------------
    def body_transforms(self, q) -> dict[str, np.ndarray]:
        """Return each body's 4x4 ground transform for coordinates ``q``.

        Args:
            q: Coordinate values as a name->value dict or an array aligned to
                :attr:`coordinate_names` (radians/meters).

        Returns:
            Map of body name (including ``"ground"``) to a 4x4 homogeneous
            transform expressing the body frame in ground.
        """
        mats = self.body_transforms_batch(self._to_vector(q)[None, :])[0]
        return {name: mats[i] for i, name in enumerate(self.body_names)}

    def center_of_mass(self, q) -> np.ndarray:
        """Return the whole-body center of mass in ground [m] for coordinates ``q``.

        Args:
            q: Coordinate values as a name->value dict or an array aligned to
                :attr:`coordinate_names` (radians/meters).

        Returns:
            Length-3 whole-body center of mass in ground.
        """
        return self.center_of_mass_batch(self._to_vector(q)[None, :])[0]

    def body_velocities(self, q, qd) -> dict[str, dict[str, np.ndarray]]:
        """Return each body's angular and linear velocity in ground.

        Args:
            q: Coordinate values as a name->value dict or an array aligned to
                :attr:`coordinate_names` (radians/meters).
            qd: Coordinate speeds in the same layout as ``q`` (rad/s or m/s).

        Returns:
            Map of body name to ``{"angular": vec3 [rad/s], "linear": vec3 [m/s]}``.
        """
        out = self.body_velocities_batch(self._to_vector(q)[None, :], self._to_vector(qd)[None, :])
        ang, lin = out["angular_velocity"][0], out["linear_velocity"][0]
        return {name: {"angular": ang[i], "linear": lin[i]} for i, name in enumerate(self.body_names)}

    def body_jacobian(self, q, eps: float = 1.0e-6) -> dict[str, np.ndarray]:
        """Return the spatial Jacobian of every body for one configuration.

        Args:
            q: Coordinate values as a dict or array [num_coordinates].
            eps: Central-difference step in coordinate units.

        Returns:
            Dict mapping each name in :attr:`body_names` to its spatial Jacobian
            ``[6, num_coordinates]`` (rows ``[wx, wy, wz, vx, vy, vz]``).
        """
        coords = self._to_vector(q).reshape(1, -1)
        jac = self.body_jacobian_batch(coords, eps=eps)[0]
        return {name: jac[b] for b, name in enumerate(self.body_names)}

    def generalized_forces_from_body_load(
        self,
        coords: np.ndarray,
        body: str | int,
        point: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        force: np.ndarray | None = None,
        torque: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map an external load on one body to generalized forces [batch, num_coordinates].

        Applies a ground-frame ``force`` at a body-local ``point`` (and an optional
        ground-frame pure ``torque``) to ``body`` and projects the resulting wrench
        through that body's spatial Jacobian: :math:`\tau = J^{\top} w`, with the
        wrench referred to the body origin as ``torque + r x force`` where ``r`` is
        the ground-frame offset from the body origin to the load point. This is the
        OpenSim mechanism behind ``PointActuator``, ``TorqueActuator``, and applied
        external loads.

        Args:
            coords: Coordinate values [batch, num_coordinates] (radians/meters).
            body: Loaded body name or index in :attr:`body_names`.
            point: Load application point in the body frame [m].
            force: Ground-frame force [N], shape ``[batch, 3]`` or ``[3]``; defaults
                to zero.
            torque: Ground-frame pure torque [N·m], shape ``[batch, 3]`` or ``[3]``;
                defaults to zero.

        Returns:
            Generalized forces [batch, num_coordinates] in :attr:`coordinate_names`
            order (N for translational coordinates, N·m for rotational).
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        batch = coords.shape[0]
        b = body if isinstance(body, int) else self.body_names.index(body)
        f = np.zeros((batch, 3)) if force is None else np.broadcast_to(np.asarray(force, np.float64), (batch, 3))
        t = np.zeros((batch, 3)) if torque is None else np.broadcast_to(np.asarray(torque, np.float64), (batch, 3))
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        jac, body_x = self._launch_body_jacobian_device(q_wp, 1.0e-6)
        point_vec = _vec3d(*np.asarray(point, dtype=np.float64))
        f_wp = wp.array(np.ascontiguousarray(f), dtype=_vec3d, device=self.device)
        t_wp = wp.array(np.ascontiguousarray(t), dtype=_vec3d, device=self.device)
        tau = wp.empty((batch, self.ncoord), dtype=_f64, device=self.device)
        wp.launch(
            body_load_project_kernel,
            dim=(batch, self.ncoord),
            inputs=[jac, body_x, b, point_vec, f_wp, t_wp, tau],
            device=self.device,
        )
        return tau.numpy()

    def whole_body_momentum(self, q, qd) -> dict[str, np.ndarray]:
        """Return whole-body linear and angular momentum in ground for ``q``, ``qd``.

        Args:
            q: Coordinate values as a name->value dict or an array aligned to
                :attr:`coordinate_names` (radians/meters).
            qd: Coordinate speeds in the same layout as ``q`` (rad/s or m/s).

        Returns:
            Dict with ``linear`` [kg*m/s] and ``angular`` [kg*m^2/s] length-3 vectors.
        """
        out = self.whole_body_momentum_batch(self._to_vector(q)[None, :], self._to_vector(qd)[None, :])
        return {"linear": out["linear_momentum"][0], "angular": out["angular_momentum"][0]}

    def body_accelerations(self, q, qd, qdd) -> dict[str, dict[str, np.ndarray]]:
        """Return each body's angular and linear acceleration in ground.

        Args:
            q: Coordinate values as a name->value dict or an array aligned to
                :attr:`coordinate_names` (radians/meters).
            qd: Coordinate speeds in the same layout as ``q`` (rad/s or m/s).
            qdd: Coordinate accelerations in the same layout (rad/s^2 or m/s^2).

        Returns:
            Map of body name to ``{"angular": vec3 [rad/s^2], "linear": vec3 [m/s^2]}``.
        """
        out = self.body_accelerations_batch(
            self._to_vector(q)[None, :], self._to_vector(qd)[None, :], self._to_vector(qdd)[None, :]
        )
        ang, lin = out["angular_acceleration"][0], out["linear_acceleration"][0]
        return {name: {"angular": ang[i], "linear": lin[i]} for i, name in enumerate(self.body_names)}

    def marker_positions(self, q, transforms: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Return model marker positions in ground [m] for coordinates ``q``.

        Args:
            q: Coordinate values (see :meth:`body_transforms`).
            transforms: Optional precomputed body transforms to reuse instead of
                relaunching the kernels.

        Returns:
            Map of marker name to a length-3 ground position.
        """
        if transforms is not None:
            out: dict[str, np.ndarray] = {}
            for mk in self.model.markers:
                x_b = transforms.get(mk.body)
                if x_b is None:
                    continue
                loc = np.array([mk.location[0], mk.location[1], mk.location[2], 1.0])
                out[mk.name] = (x_b @ loc)[:3]
            return out
        pos = self.marker_positions_batch(self._to_vector(q)[None, :])[0]
        return {name: pos[i] for i, name in enumerate(self.marker_names)}
