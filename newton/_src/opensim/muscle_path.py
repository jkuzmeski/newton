# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Warp-native OpenSim muscle-tendon path geometry.

Computes muscle-tendon lengths and moment arms from a straight-line
``GeometryPath`` (ordered path points fixed to bodies), reproducing OpenSim's
``GeometryPath::getLength`` and moment-arm definition. Three path-point kinds are
supported:

- ``PathPoint`` — a point fixed in a body frame.
- ``ConditionalPathPoint`` — a fixed point that is only part of the path while a
  gating coordinate lies within a range (matching OpenSim's ``isActive``).
- ``MovingPathPoint`` — a point whose body-frame location is a function of a
  generalized coordinate (each axis an OpenSim ``SimmSpline``/``LinearFunction``/
  ``Constant``), matching OpenSim's ``MovingPathPoint``.

The path length is the sum of the Euclidean distances between consecutive active
points expressed in ground, and the moment arm about coordinate :math:`q_i` is
:math:`r_i = -\partial L_{MT}/\partial q_i` — exactly the definition OpenSim's
``testMomentArms`` uses (a central finite difference of the path length). The
per-configuration point transforms and segment-length summation run in a Warp
``float64`` kernel reusing the validated forward-kinematics body transforms;
the moving-point spline evaluation and the finite-difference assembly are host
preprocessing, mirroring the inverse-dynamics signal pipeline.

Path wrapping is modelled for all four OpenSim wrap surfaces -- ``WrapSphere``,
``WrapCylinder``, ``WrapEllipsoid``, and ``WrapTorus`` -- by inserting each
surface's tangent-arc-tangent detour on the segment it most penetrates (the
sphere/cylinder arcs are closed-form; the ellipsoid uses the affine "scaled
sphere" construction and the torus reduces locally to a sphere of the tube
radius on the nearest ring point). A non-penetrating surface leaves the
straight-line path length unchanged, so honoring the ``PathWrap`` ``range``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .kinematics import (
    _FN_CONSTANT,
    _FN_PIECEWISE,
    _FN_SIMMSPLINE,
    ForwardKinematics,
    _bake_function,
    _eval_axis,
    euler_xyz_to_matrix,
)
from .types import OsimModel

_f64 = wp.float64
_mat44d = wp.mat44d
_mat33d = wp.mat33d
_vec3d = wp.vec3d


@wp.func
def wrap_sphere_extra(p1: _vec3d, p2: _vec3d, c: _vec3d, r: _f64) -> _f64:
    """Extra path length from wrapping the segment ``p1``->``p2`` over a sphere.

    Returns the length the tangent-arc-tangent geodesic around the sphere
    (centre ``c``, radius ``r``) adds over the straight segment, or ``0`` when
    the straight segment does not penetrate the sphere (so a non-wrapping
    surface never changes the path). All points are in ground coordinates.
    """
    d1v = p1 - c
    d2v = p2 - c
    d1 = wp.length(d1v)
    d2 = wp.length(d2v)
    if d1 <= r or d2 <= r:
        return _f64(0.0)
    l1 = wp.sqrt(d1 * d1 - r * r)
    l2 = wp.sqrt(d2 * d2 - r * r)
    cphi = wp.clamp(wp.dot(d1v, d2v) / (d1 * d2), _f64(-1.0), _f64(1.0))
    beta = wp.acos(cphi) - wp.acos(r / d1) - wp.acos(r / d2)
    if beta <= _f64(0.0):
        return _f64(0.0)
    seg = p2 - p1
    return (l1 + l2 + r * beta) - wp.length(seg)


@wp.func
def wrap_cylinder_extra(p1: _vec3d, p2: _vec3d, o: _vec3d, axis: _vec3d, r: _f64) -> _f64:
    """Extra path length from wrapping the segment ``p1``->``p2`` over a cylinder.

    The cylinder has radius ``r`` and an infinite axis through ``o`` along the
    unit-normalizable direction ``axis`` (all in ground coordinates). The
    endpoints are split into their axial (``z``) and radial (in the plane normal
    to the axis) parts; the radial parts wrap the circular cross-section with the
    same tangent-arc-tangent construction as :func:`wrap_sphere_extra`, and the
    axial component is recombined by developing the cylinder surface
    (``length = sqrt(planar^2 + dz^2)``). Returns ``0`` when the straight segment
    does not penetrate the cylinder.
    """
    a = wp.normalize(axis)
    rel1 = p1 - o
    rel2 = p2 - o
    z1 = wp.dot(rel1, a)
    z2 = wp.dot(rel2, a)
    rad1 = rel1 - z1 * a
    rad2 = rel2 - z2 * a
    d1 = wp.length(rad1)
    d2 = wp.length(rad2)
    if d1 <= r or d2 <= r:
        return _f64(0.0)
    l1 = wp.sqrt(d1 * d1 - r * r)
    l2 = wp.sqrt(d2 * d2 - r * r)
    cphi = wp.clamp(wp.dot(rad1, rad2) / (d1 * d2), _f64(-1.0), _f64(1.0))
    beta = wp.acos(cphi) - wp.acos(r / d1) - wp.acos(r / d2)
    if beta <= _f64(0.0):
        return _f64(0.0)
    s_planar = l1 + l2 + r * beta
    d2d = wp.length(rad2 - rad1)
    dz = z2 - z1
    return wp.sqrt(s_planar * s_planar + dz * dz) - wp.sqrt(d2d * d2d + dz * dz)


@wp.func
def wrap_ellipsoid_extra(p1: _vec3d, p2: _vec3d, c: _vec3d, rg: _mat33d, a: _f64, b: _f64, cc: _f64) -> _f64:
    """Extra path length from wrapping the segment ``p1``->``p2`` over an ellipsoid.

    Uses the "scaled sphere" construction OpenSim adopts as a starting point: the
    segment endpoints are mapped into the ellipsoid frame (rotation ``rg``, origin
    ``c``, both in ground) and scaled by the reciprocal semi-axes ``(a, b, c_z)``
    so the ellipsoid becomes the unit sphere. The tangent-arc-tangent geodesic is
    built on that unit sphere; because an affine map takes straight lines to
    straight lines, the two tangent legs map back to exact straight legs in
    ground, while the wrapped arc is mapped back and integrated as a chord sum
    (the geodesic of the ellipsoid itself is not affine-invariant, so this is an
    approximation that is exact for the isotropic sphere). Returns ``0`` when the
    straight segment does not penetrate the ellipsoid, so a non-wrapping surface
    never changes the path.
    """
    rgt = wp.transpose(rg)
    l1 = rgt * (p1 - c)
    l2 = rgt * (p2 - c)
    s1 = wp.vec3d(l1[0] / a, l1[1] / b, l1[2] / cc)
    s2 = wp.vec3d(l2[0] / a, l2[1] / b, l2[2] / cc)
    d1 = wp.length(s1)
    d2 = wp.length(s2)
    if d1 <= _f64(1.0) or d2 <= _f64(1.0):
        return _f64(0.0)
    u1 = s1 / d1
    u2 = s2 / d2
    cphi = wp.clamp(wp.dot(u1, u2), _f64(-1.0), _f64(1.0))
    phi = wp.acos(cphi)
    a1 = wp.acos(_f64(1.0) / d1)
    a2 = wp.acos(_f64(1.0) / d2)
    beta = phi - a1 - a2
    if beta <= _f64(0.0):
        return _f64(0.0)
    n = wp.cross(u1, u2)
    nl = wp.length(n)
    if nl < _f64(1.0e-9):
        return _f64(0.0)
    n = n / nl
    w1 = wp.cross(n, u1)
    t1 = wp.cos(a1) * u1 + wp.sin(a1) * w1
    # Real-space image of the first tangent point (scaled point unscaled and posed).
    prev = c + rg * wp.vec3d(t1[0] * a, t1[1] * b, t1[2] * cc)
    t1_real = prev
    arc = _f64(0.0)
    # Integrate the wrapped arc as a 12-segment chord sum in real space (the
    # ellipsoid geodesic is not affine-invariant, so the mapped-back arc is only
    # approximate; 12 subdivisions keep the length error well below a millimetre).
    dtt = beta / _f64(12.0)
    for k in range(1, 13):
        tt = dtt * _f64(k)
        wt = wp.cross(n, t1)
        pt = wp.cos(tt) * t1 + wp.sin(tt) * wt
        cur = c + rg * wp.vec3d(pt[0] * a, pt[1] * b, pt[2] * cc)
        d = cur - prev
        arc = arc + wp.length(d)
        prev = cur
    t2_real = prev
    tangents = wp.length(p1 - t1_real) + wp.length(p2 - t2_real)
    return (arc + tangents) - wp.length(p2 - p1)


@wp.func
def wrap_torus_extra(p1: _vec3d, p2: _vec3d, c: _vec3d, rg: _mat33d, ring_r: _f64, tube_r: _f64) -> _f64:
    """Extra path length from wrapping the segment ``p1``->``p2`` over a torus.

    The torus lies in the wrap frame's local xy-plane (rotation ``rg``, origin
    ``c`` in ground), swept about the local z-axis with ring (centerline) radius
    ``ring_r`` and tube radius ``tube_r``. Near the tube the torus is locally a
    sphere of radius ``tube_r`` centered on the ring; the ring point closest to
    the segment (taken from the segment midpoint's radial direction) supplies that
    sphere center, and the tangent-arc-tangent detour is evaluated with
    :func:`wrap_sphere_extra`. Returns ``0`` when the straight segment does not
    penetrate the tube.
    """
    rgt = wp.transpose(rg)
    l1 = rgt * (p1 - c)
    l2 = rgt * (p2 - c)
    mid = _f64(0.5) * (l1 + l2)
    rho = wp.sqrt(mid[0] * mid[0] + mid[1] * mid[1])
    if rho < _f64(1.0e-9):
        rdir = wp.vec3d(_f64(1.0), _f64(0.0), _f64(0.0))
    else:
        rdir = wp.vec3d(mid[0] / rho, mid[1] / rho, _f64(0.0))
    ring_local = ring_r * rdir
    ring_ground = c + rg * ring_local
    return wrap_sphere_extra(p1, p2, ring_ground, tube_r)


@wp.kernel
def muscle_length_kernel(
    poses: wp.array2d[_mat44d],
    point_body: wp.array[wp.int32],
    point_loc: wp.array2d[_vec3d],
    point_active: wp.array2d[wp.int32],
    musc_off: wp.array[wp.int32],
    wrap_off: wp.array[wp.int32],
    wrap_body: wp.array[wp.int32],
    wrap_type: wp.array[wp.int32],
    wrap_center: wp.array[_vec3d],
    wrap_axis: wp.array[_vec3d],
    wrap_rot: wp.array[_mat33d],
    wrap_dims: wp.array[_vec3d],
    wrap_radius: wp.array[_f64],
    wrap_lo: wp.array[wp.int32],
    wrap_hi: wp.array[wp.int32],
    lengths: wp.array2d[_f64],
):
    """Sum active-segment lengths of every muscle path over a batch of poses.

    Launched with dim ``(batch, num_muscles)``. For muscle ``m`` in configuration
    ``b`` the active path points are transformed to ground with ``poses`` and the
    Euclidean distances between consecutive active points are accumulated. Each
    ``WrapSphere`` / ``WrapCylinder`` / ``WrapEllipsoid`` / ``WrapTorus`` associated
    with the muscle then adds its single most-penetrated segment's
    tangent-arc-tangent detour (:func:`wrap_sphere_extra`,
    :func:`wrap_cylinder_extra`, :func:`wrap_ellipsoid_extra`,
    :func:`wrap_torus_extra`), so a non-wrapping surface leaves the straight-line
    length unchanged.
    """
    b, m = wp.tid()
    start = musc_off[m]
    end = musc_off[m + 1]
    total = _f64(0.0)
    have_prev = wp.int32(0)
    prev = wp.vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for p in range(start, end):
        if point_active[b, p] == 1:
            x = poses[b, point_body[p]]
            loc = point_loc[b, p]
            g = wp.vec3d(
                x[0, 0] * loc[0] + x[0, 1] * loc[1] + x[0, 2] * loc[2] + x[0, 3],
                x[1, 0] * loc[0] + x[1, 1] * loc[1] + x[1, 2] * loc[2] + x[1, 3],
                x[2, 0] * loc[0] + x[2, 1] * loc[1] + x[2, 2] * loc[2] + x[2, 3],
            )
            if have_prev == 1:
                d = g - prev
                total = total + wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            prev = g
            have_prev = wp.int32(1)

    # Wrap surfaces: each sphere inserts once, on its most-penetrated segment.
    ws = wrap_off[m]
    we = wrap_off[m + 1]
    for w in range(ws, we):
        xw = poses[b, wrap_body[w]]
        cl = wrap_center[w]
        center = wp.vec3d(
            xw[0, 0] * cl[0] + xw[0, 1] * cl[1] + xw[0, 2] * cl[2] + xw[0, 3],
            xw[1, 0] * cl[0] + xw[1, 1] * cl[1] + xw[1, 2] * cl[2] + xw[1, 3],
            xw[2, 0] * cl[0] + xw[2, 1] * cl[1] + xw[2, 2] * cl[2] + xw[2, 3],
        )
        r = wrap_radius[w]
        # Cylinder axis direction rotated into ground (translation-free).
        al = wrap_axis[w]
        axis = wp.vec3d(
            xw[0, 0] * al[0] + xw[0, 1] * al[1] + xw[0, 2] * al[2],
            xw[1, 0] * al[0] + xw[1, 1] * al[1] + xw[1, 2] * al[2],
            xw[2, 0] * al[0] + xw[2, 1] * al[1] + xw[2, 2] * al[2],
        )
        # Ground rotation of the wrap frame (body rotation composed with the
        # wrap-object's body-fixed rotation), used by the ellipsoid/torus geodesics.
        body_rot = _mat33d(xw[0, 0], xw[0, 1], xw[0, 2], xw[1, 0], xw[1, 1], xw[1, 2], xw[2, 0], xw[2, 1], xw[2, 2])
        rg = body_rot * wrap_rot[w]
        dims = wrap_dims[w]
        wtype = wrap_type[w]
        lo = wrap_lo[w]
        hi = wrap_hi[w]
        best_extra = _f64(0.0)
        have_prev2 = wp.int32(0)
        prev2 = wp.vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
        prev_idx = wp.int32(0)
        for p in range(start, end):
            if point_active[b, p] == 1:
                x = poses[b, point_body[p]]
                loc = point_loc[b, p]
                g = wp.vec3d(
                    x[0, 0] * loc[0] + x[0, 1] * loc[1] + x[0, 2] * loc[2] + x[0, 3],
                    x[1, 0] * loc[0] + x[1, 1] * loc[1] + x[1, 2] * loc[2] + x[1, 3],
                    x[2, 0] * loc[0] + x[2, 1] * loc[1] + x[2, 2] * loc[2] + x[2, 3],
                )
                # Only wrap segments whose endpoints lie in the PathWrap range.
                if have_prev2 == 1 and prev_idx >= lo and p <= hi:
                    if wtype == wp.int32(1):
                        extra = wrap_cylinder_extra(prev2, g, center, axis, r)
                    elif wtype == wp.int32(2):
                        extra = wrap_ellipsoid_extra(prev2, g, center, rg, dims[0], dims[1], dims[2])
                    elif wtype == wp.int32(3):
                        extra = wrap_torus_extra(prev2, g, center, rg, dims[0], dims[1])
                    else:
                        extra = wrap_sphere_extra(prev2, g, center, r)
                    if extra > best_extra:
                        best_extra = extra
                prev2 = g
                prev_idx = p
                have_prev2 = wp.int32(1)
        total = total + best_extra

    lengths[b, m] = total


@wp.kernel
def point_sample_kernel(
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
    cond_coord: wp.array[wp.int32],
    cond_lo: wp.array[_f64],
    cond_hi: wp.array[_f64],
    point_loc: wp.array2d[_vec3d],
    point_active: wp.array2d[wp.int32],
):
    """Evaluate each path point's body-frame location and activity for a batch.

    Launched with dim ``(batch, num_points)``. Fixed and conditional points reuse
    their constant body-frame location; ``MovingPathPoint`` axes are evaluated
    from their baked coordinate functions (each axis stored at ``3 * p + ai``),
    reusing the forward-kinematics ``_eval_axis``.
    A conditional point is inactive when its gating coordinate leaves its range.
    """
    b, p = wp.tid()
    lx = _eval_axis(3 * p + 0, b, q, atype, acoord, ap0, ap1, koff, kcnt, kx, ky, kb, kc, kd)
    ly = _eval_axis(3 * p + 1, b, q, atype, acoord, ap0, ap1, koff, kcnt, kx, ky, kb, kc, kd)
    lz = _eval_axis(3 * p + 2, b, q, atype, acoord, ap0, ap1, koff, kcnt, kx, ky, kb, kc, kd)
    point_loc[b, p] = wp.vec3d(lx, ly, lz)
    act = wp.int32(1)
    cc = cond_coord[p]
    if cc >= 0:
        v = q[b, cc]
        if v < cond_lo[p] or v > cond_hi[p]:
            act = wp.int32(0)
    point_active[b, p] = act


@wp.kernel
def perturb_kernel(
    coords: wp.array2d[_f64],
    eps: _f64,
    batch: int,
    flat: wp.array2d[_f64],
):
    """Build the central-difference coordinate batch for moment arms on device.

    Launched with dim ``(2 * nc, batch, nc)``. Perturbation ``p`` perturbs
    coordinate ``p // 2`` by ``+eps`` (even ``p``) or ``-eps`` (odd ``p``); the
    perturbed sample ``b`` is written to row ``p * batch + b`` so the length
    kernel can evaluate all perturbations in one launch.
    """
    p, b, k = wp.tid()
    c = p // 2
    val = coords[b, k]
    if k == c:
        if p % 2 == 0:
            val = val + eps
        else:
            val = val - eps
    flat[p * batch + b, k] = val


@wp.kernel
def moment_arm_kernel(
    lens: wp.array2d[_f64],
    batch: int,
    eps: _f64,
    r: wp.array3d[_f64],
):
    r"""Assemble moment arms ``r = -dL/dq`` from the perturbed path lengths on device.

    Launched with dim ``(batch, num_muscles, nc)``. ``lens`` holds the muscle
    lengths for the ``perturb_kernel`` batch (row ``p * batch + b``), so the
    central difference for coordinate ``c`` reads rows ``2 c`` and ``2 c + 1``.
    """
    b, m, c = wp.tid()
    lp = lens[(2 * c) * batch + b, m]
    lm = lens[(2 * c + 1) * batch + b, m]
    r[b, m, c] = -(lp - lm) / (_f64(2.0) * eps)


@wp.kernel
def velocity_project_kernel(
    r: wp.array3d[_f64],
    speeds: wp.array2d[_f64],
    nc: int,
    vmt: wp.array2d[_f64],
):
    r"""Project coordinate speeds through the moment arms into path velocities on device.

    Launched with dim ``(batch, num_muscles)``; computes
    :math:`\dot L_{MT}=-\sum_c r_{m,c}\,\dot q_c`.
    """
    b, m = wp.tid()
    acc = _f64(0.0)
    for c in range(nc):
        acc += r[b, m, c] * speeds[b, c]
    vmt[b, m] = -acc


class MusclePaths:
    """Warp-native muscle-tendon path geometry for an :class:`OsimModel`.

    Builds a flattened description of every muscle's ``GeometryPath`` and
    evaluates muscle-tendon lengths and moment arms over batches of coordinate
    configurations. Reuses :class:`ForwardKinematics` for the body transforms.

    Args:
        model: Parsed OpenSim model.
        device: Warp device (defaults to CPU, matching the rest of the port).
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.muscle_names: list[str] = [m.name for m in model.muscles]
        self.coordinate_names = self.fk.coordinate_names
        self._build()

    # -- setup ---------------------------------------------------------------
    def _build(self) -> None:
        fk = self.fk
        bidx = {n: i for i, n in enumerate(fk.body_names)}
        cidx = fk._index

        offsets = [0]
        point_body: list[int] = []
        # Per point-axis coordinate-function tables (three axes per point, stored
        # at 3 * p + ai) so ``point_sample_kernel`` evaluates every path point
        # location on device with the shared forward-kinematics function eval.
        atype: list[int] = []
        acoord: list[int] = []
        ap0: list[float] = []
        ap1: list[float] = []
        koff: list[int] = []
        kcnt: list[int] = []
        kx: list[float] = []
        ky: list[float] = []
        kb: list[float] = []
        kc: list[float] = []
        kd: list[float] = []
        cond_coord: list[int] = []
        cond_lo: list[float] = []
        cond_hi: list[float] = []

        def _const_axis(value: float) -> None:
            atype.append(_FN_CONSTANT)
            acoord.append(-1)
            ap0.append(float(value))
            ap1.append(0.0)
            koff.append(0)
            kcnt.append(0)

        def _baked_axis(coord: int, ftype: str | None, params: dict) -> None:
            code, p0, p1, x, y, b, c, d = _bake_function(ftype, params)
            atype.append(code)
            acoord.append(coord)
            ap0.append(p0)
            ap1.append(p1)
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

        for mus in self.model.muscles:
            for pp in mus.path_points:
                point_body.append(bidx.get(pp.body, 0))
                loc = np.asarray(pp.location, float)
                moving = pp.moving if pp.type == "MovingPathPoint" else None
                for ai, axis in enumerate(("x", "y", "z")):
                    if moving and axis in moving:
                        coord_name, ftype, params = moving[axis]
                        _baked_axis(cidx.get(coord_name, -1), ftype, params)
                    else:
                        _const_axis(loc[ai])
                if pp.type == "ConditionalPathPoint" and pp.conditional_coordinate:
                    lo, hi = pp.conditional_range or (float("-inf"), float("inf"))
                    cond_coord.append(cidx.get(pp.conditional_coordinate, -1))
                    cond_lo.append(float(lo))
                    cond_hi.append(float(hi))
                else:
                    cond_coord.append(-1)
                    cond_lo.append(float("-inf"))
                    cond_hi.append(float("inf"))
            offsets.append(len(point_body))

        self.npoint = len(point_body)
        self._musc_off = np.asarray(offsets, np.int32)
        self._point_body = np.asarray(point_body, np.int32)
        self.d_point_body = wp.array(self._point_body, dtype=wp.int32, device=self.device)
        self.d_musc_off = wp.array(self._musc_off, dtype=wp.int32, device=self.device)

        dev = self.device
        self.d_atype = wp.array(np.asarray(atype, np.int32), dtype=wp.int32, device=dev)
        self.d_acoord = wp.array(np.asarray(acoord, np.int32), dtype=wp.int32, device=dev)
        self.d_ap0 = wp.array(np.asarray(ap0, np.float64), dtype=_f64, device=dev)
        self.d_ap1 = wp.array(np.asarray(ap1, np.float64), dtype=_f64, device=dev)
        self.d_koff = wp.array(np.asarray(koff, np.int32), dtype=wp.int32, device=dev)
        self.d_kcnt = wp.array(np.asarray(kcnt, np.int32), dtype=wp.int32, device=dev)
        self.d_kx = wp.array(np.asarray(kx or [0.0], np.float64), dtype=_f64, device=dev)
        self.d_ky = wp.array(np.asarray(ky or [0.0], np.float64), dtype=_f64, device=dev)
        self.d_kb = wp.array(np.asarray(kb or [0.0], np.float64), dtype=_f64, device=dev)
        self.d_kc = wp.array(np.asarray(kc or [0.0], np.float64), dtype=_f64, device=dev)
        self.d_kd = wp.array(np.asarray(kd or [0.0], np.float64), dtype=_f64, device=dev)
        self.d_cond_coord = wp.array(np.asarray(cond_coord, np.int32), dtype=wp.int32, device=dev)
        self.d_cond_lo = wp.array(np.asarray(cond_lo, np.float64), dtype=_f64, device=dev)
        self.d_cond_hi = wp.array(np.asarray(cond_hi, np.float64), dtype=_f64, device=dev)

        # Per-muscle wrap surfaces. All four OpenSim wrap types are modelled;
        # ``wrap_type`` selects the geodesic (0 sphere, 1 cylinder, 2 ellipsoid,
        # 3 torus). Each wrap carries its body-fixed ground-mappable rotation
        # ``wrap_rot`` (for the ellipsoid/torus frames) and ``wrap_dims`` (the
        # ellipsoid semi-axes, or the torus ring/tube radii).
        _WRAP_CODES = {"WrapSphere": 0, "WrapCylinder": 1, "WrapEllipsoid": 2, "WrapTorus": 3}
        wobj = {w.name: w for w in self.model.wrap_objects}
        wrap_off = [0]
        wrap_body: list[int] = []
        wrap_type: list[int] = []
        wrap_center: list[tuple[float, float, float]] = []
        wrap_axis: list[tuple[float, float, float]] = []
        wrap_rot: list[list[float]] = []
        wrap_dims: list[tuple[float, float, float]] = []
        wrap_radius: list[float] = []
        # Global point-index span a wrap may insert between (from PathWrap ``range``).
        wrap_lo: list[int] = []
        wrap_hi: list[int] = []
        for mi, mus in enumerate(self.model.muscles):
            base = offsets[mi]
            npts = offsets[mi + 1] - base
            for wr in mus.wraps:
                w = wobj.get(wr.wrap_object)
                if w is None or not w.active or w.type not in _WRAP_CODES:
                    continue
                wrap_body.append(bidx.get(w.body, 0))
                wrap_type.append(_WRAP_CODES[w.type])
                wrap_center.append((w.translation[0], w.translation[1], w.translation[2]))
                rot = euler_xyz_to_matrix(*w.rotation)
                # Cylinder axis is the wrap frame's local z; sphere axis is unused.
                axis = rot @ np.array([0.0, 0.0, 1.0])
                wrap_axis.append((float(axis[0]), float(axis[1]), float(axis[2])))
                wrap_rot.append([float(v) for v in np.asarray(rot).reshape(-1)])
                wrap_radius.append(float(w.radius))
                if w.type == "WrapEllipsoid":
                    wrap_dims.append((float(w.dimensions[0]), float(w.dimensions[1]), float(w.dimensions[2])))
                elif w.type == "WrapTorus":
                    ring_r = 0.5 * (w.inner_radius + w.outer_radius)
                    tube_r = 0.5 * (w.outer_radius - w.inner_radius)
                    wrap_dims.append((float(ring_r), float(tube_r), 0.0))
                else:
                    wrap_dims.append((float(w.radius), float(w.radius), float(w.radius)))
                # ``range`` is a 1-based inclusive PathPoint span in OpenSim; ``(-1, -1)``
                # (or a non-positive bound) means the whole path.
                r0, r1 = wr.range
                lo = base + (r0 - 1 if r0 >= 1 else 0)
                hi = base + (r1 - 1 if r1 >= 1 else npts - 1)
                wrap_lo.append(int(lo))
                wrap_hi.append(int(hi))
            wrap_off.append(len(wrap_body))
        self.nwrap = len(wrap_body)
        self._wrap_off = np.asarray(wrap_off, np.int32)
        self.d_wrap_off = wp.array(self._wrap_off, dtype=wp.int32, device=dev)
        self.d_wrap_body = wp.array(np.asarray(wrap_body or [0], np.int32), dtype=wp.int32, device=dev)
        self.d_wrap_type = wp.array(np.asarray(wrap_type or [0], np.int32), dtype=wp.int32, device=dev)
        self.d_wrap_center = wp.array(
            np.asarray(wrap_center or [(0.0, 0.0, 0.0)], np.float64), dtype=_vec3d, device=dev
        )
        self.d_wrap_axis = wp.array(np.asarray(wrap_axis or [(0.0, 0.0, 1.0)], np.float64), dtype=_vec3d, device=dev)
        self.d_wrap_rot = wp.array(
            np.asarray(wrap_rot or [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], np.float64),
            dtype=_mat33d,
            device=dev,
        )
        self.d_wrap_dims = wp.array(np.asarray(wrap_dims or [(0.0, 0.0, 0.0)], np.float64), dtype=_vec3d, device=dev)
        self.d_wrap_radius = wp.array(np.asarray(wrap_radius or [0.0], np.float64), dtype=_f64, device=dev)
        self.d_wrap_lo = wp.array(np.asarray(wrap_lo or [0], np.int32), dtype=wp.int32, device=dev)
        self.d_wrap_hi = wp.array(np.asarray(wrap_hi or [0], np.int32), dtype=wp.int32, device=dev)

    # -- device evaluation of per-config point locations / activity ----------
    def _sample_points(self, q_wp: wp.array) -> tuple[wp.array, wp.array]:
        """Return device path-point locations and activity for a coordinate batch."""
        batch = q_wp.shape[0]
        dev = self.device
        d_loc = wp.empty((batch, self.npoint), dtype=_vec3d, device=dev)
        d_active = wp.empty((batch, self.npoint), dtype=wp.int32, device=dev)
        wp.launch(
            point_sample_kernel,
            dim=(batch, self.npoint),
            inputs=[
                q_wp,
                self.d_atype,
                self.d_acoord,
                self.d_ap0,
                self.d_ap1,
                self.d_koff,
                self.d_kcnt,
                self.d_kx,
                self.d_ky,
                self.d_kb,
                self.d_kc,
                self.d_kd,
                self.d_cond_coord,
                self.d_cond_lo,
                self.d_cond_hi,
            ],
            outputs=[d_loc, d_active],
            device=dev,
        )
        return d_loc, d_active

    # -- public API ----------------------------------------------------------
    def _lengths_qwp(self, q_wp: wp.array2d) -> wp.array2d:
        """Device core of :meth:`lengths`: muscle lengths for a device coordinate batch.

        Args:
            q_wp: Coordinate configurations on device, shape [num_configs, num_coordinates].

        Returns:
            Muscle-tendon lengths [m] on device, shape [num_configs, num_muscles].
        """
        batch = q_wp.shape[0]
        nm = len(self.muscle_names)
        poses = self.fk._launch_body_transforms(q_wp)
        d_loc, d_active = self._sample_points(q_wp)
        d_len = wp.empty((batch, nm), dtype=_f64, device=self.device)
        wp.launch(
            muscle_length_kernel,
            dim=(batch, nm),
            inputs=[
                poses,
                self.d_point_body,
                d_loc,
                d_active,
                self.d_musc_off,
                self.d_wrap_off,
                self.d_wrap_body,
                self.d_wrap_type,
                self.d_wrap_center,
                self.d_wrap_axis,
                self.d_wrap_rot,
                self.d_wrap_dims,
                self.d_wrap_radius,
                self.d_wrap_lo,
                self.d_wrap_hi,
                d_len,
            ],
            device=self.device,
        )
        return d_len

    def lengths(self, coords: np.ndarray) -> np.ndarray:
        """Return muscle-tendon lengths [m], shape ``[batch, num_muscles]``.

        Args:
            coords: Coordinate configurations [batch, num_coordinates]
                (radians/metres, OpenSim order).
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        return self._lengths_qwp(q_wp).numpy()

    def _moment_arms_device(self, q_wp: wp.array2d, eps: float) -> wp.array3d:
        """Device core of :meth:`moment_arms`: central-difference moment arms on device.

        Builds the ``[2 nc, batch, nc]`` perturbation batch, evaluates all path
        lengths in one launch, and assembles ``r = -dL/dq`` without leaving the
        device.

        Args:
            q_wp: Coordinate configurations on device, shape [batch, num_coordinates].
            eps: Central-difference step [rad or m].

        Returns:
            Moment arms [m] on device, shape [batch, num_muscles, num_coordinates].
        """
        batch = q_wp.shape[0]
        nc = len(self.coordinate_names)
        nm = len(self.muscle_names)
        flat = wp.empty((2 * nc * batch, nc), dtype=_f64, device=self.device)
        wp.launch(
            perturb_kernel,
            dim=(2 * nc, batch, nc),
            inputs=[q_wp, _f64(eps), batch, flat],
            device=self.device,
        )
        d_lens = self._lengths_qwp(flat)
        d_r = wp.empty((batch, nm, nc), dtype=_f64, device=self.device)
        wp.launch(
            moment_arm_kernel,
            dim=(batch, nm, nc),
            inputs=[d_lens, batch, _f64(eps), d_r],
            device=self.device,
        )
        return d_r

    def _velocities_qwp(
        self,
        q_wp: wp.array[_f64],
        speeds_wp: wp.array[_f64],
        eps: float = 1.0e-5,
        moment_arms: wp.array[_f64] | None = None,
    ) -> wp.array[_f64]:
        """Return muscle-tendon velocities for device-resident state arrays.

        Args:
            q_wp: Coordinate configurations on device, shape [batch, num_coordinates].
            speeds_wp: Coordinate speeds on device, same shape as ``q_wp``.
            eps: Central-difference step [rad or m].
            moment_arms: Optional precomputed moment arms on device. Supplying this
                avoids reevaluating the path geometry.

        Returns:
            Muscle-tendon lengthening velocities [m/s] on device, shape
            [batch, num_muscles].
        """
        batch = q_wp.shape[0]
        nm = len(self.muscle_names)
        d_r = self._moment_arms_device(q_wp, eps) if moment_arms is None else moment_arms
        d_vmt = wp.empty((batch, nm), dtype=_f64, device=self.device)
        wp.launch(
            velocity_project_kernel,
            dim=(batch, nm),
            inputs=[d_r, speeds_wp, len(self.coordinate_names), d_vmt],
            device=self.device,
        )
        return d_vmt

    def velocities(self, coords: np.ndarray, speeds: np.ndarray, eps: float = 1.0e-5) -> np.ndarray:
        r"""Return muscle-tendon lengthening velocities [m/s], shape ``[batch, num_muscles]``.

        The path lengthening velocity is
        :math:`\dot L_{MT}=\sum_i (\partial L_{MT}/\partial q_i)\,\dot q_i
        =-\sum_i r_i\,\dot q_i`, formed from the moment arms and coordinate speeds.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates] (rad/s or m/s).
            eps: Central-difference step for the moment arms [rad or m].
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        speeds = np.ascontiguousarray(np.atleast_2d(speeds), dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        speeds_wp = wp.array(speeds, dtype=_f64, device=self.device)
        return self._velocities_qwp(q_wp, speeds_wp, eps).numpy()

    def moment_arms(self, coords: np.ndarray, eps: float = 1.0e-5) -> np.ndarray:
        r"""Return moment arms [m], shape ``[batch, num_muscles, num_coordinates]``.

        The moment arm of a muscle about coordinate :math:`q_i` is
        :math:`r_i = -\partial L_{MT}/\partial q_i`, evaluated with a central
        finite difference of step ``eps`` — OpenSim's moment-arm definition.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            eps: Central-difference step [rad or m].
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        q_wp = wp.array(coords, dtype=_f64, device=self.device)
        return self._moment_arms_device(q_wp, eps).numpy()


def compute_muscle_moment_arms(
    model: OsimModel,
    coords: np.ndarray,
    eps: float = 1.0e-5,
    device=None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute muscle-tendon lengths and moment arms for a coordinate batch.

    Args:
        model: Parsed OpenSim model.
        coords: Coordinate configurations [batch, num_coordinates].
        eps: Central-difference step for the moment arms [rad or m].
        device: Warp device (defaults to CPU).

    Returns:
        ``(lengths, moment_arms, muscle_names)`` where ``lengths`` is
        ``[batch, num_muscles]`` [m], ``moment_arms`` is
        ``[batch, num_muscles, num_coordinates]`` [m], and ``muscle_names`` lists
        the muscle order.
    """
    mp = MusclePaths(model, device=device)
    coords = np.atleast_2d(coords)
    return mp.lengths(coords), mp.moment_arms(coords, eps=eps), mp.muscle_names
