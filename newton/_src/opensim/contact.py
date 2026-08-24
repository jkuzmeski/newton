# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""Newton-native OpenSim compliant contact forces (Warp kernels).

OpenSim exposes three compliant (penalty) contact-force models, all built on
Simbody's Hunt-Crossley elastic-foundation theory:

* :class:`~newton.opensim.OsimContactForce` ``SmoothSphereHalfSpaceForce``
  -- the continuously differentiable sphere/half-space model used by Moco for
  gradient-based gait prediction (Serrancoli & Falisse; see
  ``SimTK::SmoothSphereHalfSpaceForceImpl::calcForce``). Every ``max``/``min`` of
  the underlying Hunt-Crossley law is replaced by a ``tanh`` (Hertz normal-force
  smoothing ``bd``) / rational (Hunt-Crossley velocity smoothing ``bv``)
  approximation so the force is :math:`C^\infty` in the state.
* ``HuntCrossleyForce`` -- the classic (non-smooth) Simbody point-contact model
  for sphere/half-space and sphere/sphere pairs
  (``SimTK::HuntCrossleyForceImpl::calcForce``): Hertz normal force
  :math:`f_H=\tfrac43 k\,\delta\sqrt{R_e k\,\delta}` with Hunt-Crossley
  dissipation :math:`f=f_H(1+\tfrac32 c\,\dot\delta)` (hard ``f>0`` cutoff) and
  Stribeck friction. Per-surface material properties are combined exactly as in
  Simbody (``k=k_1 s_1``, ``s_1=k_2/(k_1+k_2)``, harmonic-mean frictions).
* ``ElasticFoundationForce`` -- the mesh model
  (``SimTK::ElasticFoundationForceImpl::processContact``): each triangle face of
  a :class:`ContactMesh` carries an independent linear spring
  :math:`f=k\,A\,\delta(1+c\,\dot\delta)` (again with Stribeck friction) that
  presses against the other object's surface. Implemented for a mesh against a
  half-space or sphere (the common foot-on-floor case), one Warp thread per
  face-spring.

The kernels are ``float64`` throughout to match OpenSim/Simbody double precision
and the rest of the port (kinematics, dynamics). Body poses come from the
validated forward kinematics (:class:`~newton.opensim.ForwardKinematics`);
body point velocities are read from a three-pose central-difference stencil
(:math:`q`, :math:`q\pm h\dot q`), the same device-resident scheme the
inverse/forward dynamics use. Each force element accumulates a resultant spatial
wrench (force + couple about the body origin, ground frame) onto the two bodies
it couples; :meth:`OpenSimContact.body_wrenches` returns these in the
:class:`~newton.opensim.ExternalLoads` ``[F P T]`` layout so they
drop straight into :meth:`~newton.opensim.ForwardDynamics.accelerations`
for a contact-driven forward simulation, while :meth:`OpenSimContact.generalized_forces`
projects them onto the coordinates.

The single-pair sphere/half-space and sphere/sphere geometry matches
``collide_plane_sphere`` and
``collide_sphere_sphere``
(cross-checked in the unit tests); it is re-derived here in ``float64`` because
those helpers are ``float32``.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from .frame import OsimFrameConverter
from .kinematics import ForwardKinematics, euler_xyz_to_matrix
from .types import OsimModel

_f64 = wp.float64
_vec3d = wp.vec3d
_mat33d = wp.mat33d
_mat44d = wp.mat44d
_Z_UP_CONVERTER = OsimFrameConverter()


def _convert_world_vectors(values: np.ndarray, frame: Literal["newton", "opensim"]) -> np.ndarray:
    """Convert row-vector world quantities from OpenSim Y-up to the requested frame."""
    if frame == "opensim":
        return values
    if frame == "newton":
        return _Z_UP_CONVERTER.transform_vectors(values)
    raise ValueError("frame must be 'newton' or 'opensim'")


# --------------------------------------------------------------------------- #
# Small float64 transform / kinematics helpers (mirror opensim.dynamics).
# --------------------------------------------------------------------------- #
@wp.func
def _rot_of(X: _mat44d) -> _mat33d:
    return _mat33d(X[0, 0], X[0, 1], X[0, 2], X[1, 0], X[1, 1], X[1, 2], X[2, 0], X[2, 1], X[2, 2])


@wp.func
def _pos_of(X: _mat44d) -> _vec3d:
    return _vec3d(X[0, 3], X[1, 3], X[2, 3])


@wp.func
def _vee(M: _mat33d) -> _vec3d:
    return _vec3d(
        _f64(0.5) * (M[2, 1] - M[1, 2]),
        _f64(0.5) * (M[0, 2] - M[2, 0]),
        _f64(0.5) * (M[1, 0] - M[0, 1]),
    )


@wp.func
def _xform_point(X: _mat44d, p: _vec3d) -> _vec3d:
    h = wp.mul(X, wp.vec4d(p[0], p[1], p[2], _f64(1.0)))
    return _vec3d(h[0], h[1], h[2])


@wp.func
def _body_omega(x0: _mat44d, xp: _mat44d, xm: _mat44d, inv_2h: _f64) -> _vec3d:
    """Angular velocity from a central-difference pose stencil."""
    rdot = (_rot_of(xp) - _rot_of(xm)) * inv_2h
    return _vee(rdot * wp.transpose(_rot_of(x0)))


@wp.func
def _point_velocity(x0: _mat44d, xp: _mat44d, xm: _mat44d, inv_2h: _f64, point: _vec3d) -> _vec3d:
    """World velocity of a body-fixed ``point`` (ground) via the pose stencil."""
    v_origin = (_pos_of(xp) - _pos_of(xm)) * inv_2h
    omega = _body_omega(x0, xp, xm, inv_2h)
    return v_origin + wp.cross(omega, point - _pos_of(x0))


# --------------------------------------------------------------------------- #
# Stribeck friction shared by all three models: given the compressive normal
# force magnitude ``fn`` (>= 0) and the tangential slip velocity, return the
# tangential friction force vector. Matches every Simbody contact impl.
# --------------------------------------------------------------------------- #
@wp.func
def _stribeck_friction(fn: _f64, vtangent: _vec3d, vslip: _f64, us: _f64, ud: _f64, uv: _f64, vt: _f64) -> _vec3d:
    if vslip <= _f64(0.0):
        return _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    vrel = vslip / vt
    mu = wp.min(vrel, _f64(1.0)) * (ud + _f64(2.0) * (us - ud) / (_f64(1.0) + vrel * vrel))
    ff = fn * (mu + uv * vslip)
    return ff * vtangent / vslip


@wp.kernel
def contact_state_stencil_kernel(
    q: wp.array2d[_f64],
    qd: wp.array2d[_f64],
    h: _f64,
    stencil: wp.array2d[_f64],
):
    """Build interleaved ``q, q+h*qd, q-h*qd`` contact states on device."""
    b, row, c = wp.tid()
    delta = _f64(0.0)
    if row == 1:
        delta = h * qd[b, c]
    elif row == 2:
        delta = -h * qd[b, c]
    stencil[3 * b + row, c] = q[b, c] + delta


@wp.kernel
def contact_generalized_force_kernel(
    jac: wp.array4d[_f64],
    body_force: wp.array2d[_vec3d],
    body_torque: wp.array2d[_vec3d],
    nbody: int,
    tau: wp.array2d[_f64],
):
    """Project accumulated body contact wrenches through device Jacobians."""
    b, c = wp.tid()
    acc = _f64(0.0)
    for body in range(nbody):
        torque = body_torque[b, body]
        force = body_force[b, body]
        acc += jac[b, body, 0, c] * torque[0]
        acc += jac[b, body, 1, c] * torque[1]
        acc += jac[b, body, 2, c] * torque[2]
        acc += jac[b, body, 3, c] * force[0]
        acc += jac[b, body, 4, c] * force[1]
        acc += jac[b, body, 5, c] * force[2]
    tau[b, c] = acc


@wp.kernel
def reduce_contact_element_force_kernel(
    smooth_force: wp.array2d[_vec3d],
    smooth_owner: wp.array[wp.int32],
    n_smooth: int,
    hc_force: wp.array2d[_vec3d],
    hc_owner: wp.array[wp.int32],
    n_hc: int,
    ef_force: wp.array2d[_vec3d],
    ef_owner: wp.array[wp.int32],
    n_ef: int,
    out: wp.array2d[_vec3d],
):
    """Reduce contact sub-pairs/faces to one force per public element."""
    b, element = wp.tid()
    force = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for i in range(n_smooth):
        if smooth_owner[i] == element:
            force += smooth_force[b, i]
    for i in range(n_hc):
        if hc_owner[i] == element:
            force += hc_force[b, i]
    for i in range(n_ef):
        if ef_owner[i] == element:
            force += ef_force[b, i]
    out[b, element] = force


@wp.kernel
def pack_contact_body_wrench_kernel(
    poses: wp.array2d[_mat44d],
    body_force: wp.array2d[_vec3d],
    body_torque: wp.array2d[_vec3d],
    touched_bodies: wp.array[wp.int32],
    out: wp.array3d[_f64],
):
    """Pack touched-body contact loads directly into ExternalLoads layout."""
    b, column = wp.tid()
    body = touched_bodies[column]
    force = body_force[b, body]
    torque = body_torque[b, body]
    pose = poses[3 * b, body]
    out[b, column, 0] = force[0]
    out[b, column, 1] = force[1]
    out[b, column, 2] = force[2]
    out[b, column, 3] = pose[0, 3]
    out[b, column, 4] = pose[1, 3]
    out[b, column, 5] = pose[2, 3]
    out[b, column, 6] = torque[0]
    out[b, column, 7] = torque[1]
    out[b, column, 8] = torque[2]


# --------------------------------------------------------------------------- #
# Model 1: SmoothSphereHalfSpaceForce (Moco). One thread per (frame, element).
# --------------------------------------------------------------------------- #
@wp.kernel
def smooth_sphere_halfspace_kernel(
    poses: wp.array2d[_mat44d],
    s_body: wp.array[wp.int32],
    s_loc: wp.array[_vec3d],
    s_rad: wp.array[_f64],
    h_body: wp.array[wp.int32],
    h_loc: wp.array[_vec3d],
    h_normal: wp.array[_vec3d],
    stiffness: wp.array[_f64],
    dissipation: wp.array[_f64],
    us: wp.array[_f64],
    ud: wp.array[_f64],
    uv: wp.array[_f64],
    vt: wp.array[_f64],
    cf: wp.array[_f64],
    bd: wp.array[_f64],
    bv: wp.array[_f64],
    inv_2h: _f64,
    body_force: wp.array2d[_vec3d],
    body_torque: wp.array2d[_vec3d],
    elem_force: wp.array2d[_vec3d],
):
    """Smooth (differentiable) sphere / half-space Hunt-Crossley force.

    Faithful to ``SimTK::SmoothSphereHalfSpaceForceImpl::calcForce``: the normal
    is the half-space frame X axis pointing *into* the solid, the indentation is
    :math:`\\delta=R+(p_s-p_h)\\cdot\\hat n`, and both Hertz and Hunt-Crossley
    terms are ``tanh``/rational-smoothed. No penetration gate keeps it
    :math:`C^\\infty`.
    """
    f, e = wp.tid()
    base = 3 * f
    sb = s_body[e]
    hb = h_body[e]
    x0s = poses[base + 0, sb]
    xps = poses[base + 1, sb]
    xms = poses[base + 2, sb]
    x0h = poses[base + 0, hb]
    xph = poses[base + 1, hb]
    xmh = poses[base + 2, hb]

    r = s_rad[e]
    center = _xform_point(x0s, s_loc[e])
    hs_origin = _xform_point(x0h, h_loc[e])
    normal = wp.normalize(_rot_of(x0h) * h_normal[e])  # into the half-space

    indentation = r + wp.dot(center - hs_origin, normal)
    contact_pt = center + r * normal - _f64(0.5) * indentation * normal

    v = _point_velocity(x0s, xps, xms, inv_2h, contact_pt) - _point_velocity(x0h, xph, xmh, inv_2h, contact_pt)
    vnormal = wp.dot(v, normal)
    vtangent = v - vnormal * normal

    k = _f64(0.5) * wp.pow(stiffness[e], _f64(2.0) / _f64(3.0))
    cf_e = cf[e]
    fh_pos = (_f64(4.0) / _f64(3.0)) * k * wp.sqrt(r * k) * wp.pow(wp.sqrt(indentation * indentation + cf_e), _f64(1.5))
    fh_smooth = fh_pos * (_f64(0.5) + _f64(0.5) * wp.tanh(bd[e] * indentation))
    c = dissipation[e]
    fhc_pos = fh_smooth * (_f64(1.0) + _f64(1.5) * c * vnormal)
    if c != _f64(0.0):
        fhc = fhc_pos * (_f64(0.5) + _f64(0.5) * wp.tanh(bv[e] * (vnormal + _f64(2.0) / (_f64(3.0) * c))))
    else:
        fhc = fhc_pos
    force = fhc * normal  # Simbody "force" (on the half-space)

    vslip = wp.sqrt(wp.dot(vtangent, vtangent) + cf_e)  # smoothed slip speed
    force = force + _stribeck_friction(fhc, vtangent, vslip, us[e], ud[e], uv[e], vt[e])

    f_sphere = -force
    f_hs = force
    elem_force[f, e] = f_sphere
    o_s = _pos_of(x0s)
    o_h = _pos_of(x0h)
    wp.atomic_add(body_force, f, sb, f_sphere)
    wp.atomic_add(body_torque, f, sb, wp.cross(contact_pt - o_s, f_sphere))
    wp.atomic_add(body_force, f, hb, f_hs)
    wp.atomic_add(body_torque, f, hb, wp.cross(contact_pt - o_h, f_hs))


# --------------------------------------------------------------------------- #
# Model 2: classic HuntCrossleyForce point contact (sphere/half-space,
# sphere/sphere). One thread per (frame, element).
# --------------------------------------------------------------------------- #
@wp.kernel
def hunt_crossley_kernel(
    poses: wp.array2d[_mat44d],
    kind: wp.array[wp.int32],  # 0 = sphere/half-space, 1 = sphere/sphere
    a_body: wp.array[wp.int32],
    a_loc: wp.array[_vec3d],
    a_rad: wp.array[_f64],
    b_body: wp.array[wp.int32],
    b_loc: wp.array[_vec3d],
    b_rad: wp.array[_f64],
    b_normal: wp.array[_vec3d],
    a_stiff: wp.array[_f64],
    a_diss: wp.array[_f64],
    a_us: wp.array[_f64],
    a_ud: wp.array[_f64],
    a_uv: wp.array[_f64],
    b_stiff: wp.array[_f64],
    b_diss: wp.array[_f64],
    b_us: wp.array[_f64],
    b_ud: wp.array[_f64],
    b_uv: wp.array[_f64],
    vt: wp.array[_f64],
    inv_2h: _f64,
    body_force: wp.array2d[_vec3d],
    body_torque: wp.array2d[_vec3d],
    elem_force: wp.array2d[_vec3d],
):
    """Classic (non-smooth) Hunt-Crossley point contact.

    Faithful to ``SimTK::HuntCrossleyForceImpl::calcForce``: normal is
    surface1->surface2 (sphere A into surface B), depth
    :math:`\\delta`, effective radius (:math:`R` for a half-space,
    :math:`R_1R_2/(R_1+R_2)` for two spheres), combined stiffness ``k=k_1 s_1``
    (``s_1=k_2/(k_1+k_2)``), Hertz force
    :math:`f_H=\\tfrac43 k\\,\\delta\\sqrt{R_e k\\,\\delta}`, Hunt-Crossley
    dissipation with a hard ``f>0`` cutoff, and Stribeck friction.
    """
    f, e = wp.tid()
    base = 3 * f
    ab = a_body[e]
    bb = b_body[e]
    x0a = poses[base + 0, ab]
    xpa = poses[base + 1, ab]
    xma = poses[base + 2, ab]
    x0b = poses[base + 0, bb]
    xpb = poses[base + 1, bb]
    xmb = poses[base + 2, bb]

    ca = _xform_point(x0a, a_loc[e])
    ra = a_rad[e]

    normal = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    depth = _f64(0.0)
    eff_radius = _f64(0.0)
    base_loc = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    if kind[e] == 0:
        # sphere A vs half-space B: normal points into the half-space.
        hs_origin = _xform_point(x0b, b_loc[e])
        normal = wp.normalize(_rot_of(x0b) * b_normal[e])
        depth = ra + wp.dot(ca - hs_origin, normal)
        eff_radius = ra
        base_loc = ca + (ra - _f64(0.5) * depth) * normal
    else:
        # sphere A vs sphere B: normal points A -> B.
        cb = _xform_point(x0b, b_loc[e])
        rb = b_rad[e]
        d = cb - ca
        dist = wp.length(d)
        if dist > _f64(0.0):
            normal = d / dist
        else:
            normal = _vec3d(_f64(1.0), _f64(0.0), _f64(0.0))
        depth = ra + rb - dist
        eff_radius = ra * rb / (ra + rb)
        base_loc = ca + (ra - _f64(0.5) * depth) * normal

    elem_force[f, e] = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    if depth <= _f64(0.0):
        return

    s1 = b_stiff[e] / (a_stiff[e] + b_stiff[e])
    s2 = _f64(1.0) - s1
    location = base_loc + (depth * (_f64(0.5) - s1)) * normal
    k = a_stiff[e] * s1
    c = a_diss[e] * s1 + b_diss[e] * s2
    fH = (_f64(4.0) / _f64(3.0)) * k * depth * wp.sqrt(eff_radius * k * depth)

    v = _point_velocity(x0a, xpa, xma, inv_2h, location) - _point_velocity(x0b, xpb, xmb, inv_2h, location)
    vnormal = wp.dot(v, normal)
    vtangent = v - vnormal * normal
    fn = fH * (_f64(1.0) + _f64(1.5) * c * vnormal)
    if fn <= _f64(0.0):
        return

    # Combined (harmonic-mean) friction coefficients.
    us = _f64(0.0)
    if a_us[e] != _f64(0.0) or b_us[e] != _f64(0.0):
        us = _f64(2.0) * a_us[e] * b_us[e] / (a_us[e] + b_us[e])
    ud = _f64(0.0)
    if a_ud[e] != _f64(0.0) or b_ud[e] != _f64(0.0):
        ud = _f64(2.0) * a_ud[e] * b_ud[e] / (a_ud[e] + b_ud[e])
    uv = _f64(0.0)
    if a_uv[e] != _f64(0.0) or b_uv[e] != _f64(0.0):
        uv = _f64(2.0) * a_uv[e] * b_uv[e] / (a_uv[e] + b_uv[e])

    force = fn * normal
    vslip = wp.length(vtangent)
    force = force + _stribeck_friction(fn, vtangent, vslip, us, ud, uv, vt[e])

    f_a = -force
    f_b = force
    elem_force[f, e] = f_a
    o_a = _pos_of(x0a)
    o_b = _pos_of(x0b)
    wp.atomic_add(body_force, f, ab, f_a)
    wp.atomic_add(body_torque, f, ab, wp.cross(location - o_a, f_a))
    wp.atomic_add(body_force, f, bb, f_b)
    wp.atomic_add(body_torque, f, bb, wp.cross(location - o_b, f_b))


# --------------------------------------------------------------------------- #
# Model 3: ElasticFoundationForce (mesh springs vs half-space / sphere).
# One thread per (frame, face-spring).
# --------------------------------------------------------------------------- #
@wp.kernel
def elastic_foundation_kernel(
    poses: wp.array2d[_mat44d],
    mesh_body: wp.array[wp.int32],
    spring_pos: wp.array[_vec3d],
    spring_area: wp.array[_f64],
    other_kind: wp.array[wp.int32],  # 0 = half-space, 1 = sphere
    other_body: wp.array[wp.int32],
    other_loc: wp.array[_vec3d],
    other_normal: wp.array[_vec3d],
    other_rad: wp.array[_f64],
    stiffness: wp.array[_f64],
    dissipation: wp.array[_f64],
    us: wp.array[_f64],
    ud: wp.array[_f64],
    uv: wp.array[_f64],
    vt: wp.array[_f64],
    area_scale: wp.array[_f64],
    inv_2h: _f64,
    body_force: wp.array2d[_vec3d],
    body_torque: wp.array2d[_vec3d],
    elem_force: wp.array2d[_vec3d],
):
    """One elastic-foundation face-spring vs a half-space or sphere.

    Faithful to ``SimTK::ElasticFoundationForceImpl::processContact``: if the
    spring centroid is inside the other object, the linear spring force is
    :math:`f=k\\,A\\,\\delta(1+c\\,\\dot\\delta)` along the direction from the
    spring to its nearest surface point, plus Stribeck friction; the mesh body
    receives ``+force``, the other body ``-force``.
    """
    f, e = wp.tid()
    base = 3 * f
    mb = mesh_body[e]
    ob = other_body[e]
    x0m = poses[base + 0, mb]
    xpm = poses[base + 1, mb]
    xmm = poses[base + 2, mb]
    x0o = poses[base + 0, ob]
    xpo = poses[base + 1, ob]
    xmo = poses[base + 2, ob]

    sp = _xform_point(x0m, spring_pos[e])
    elem_force[f, e] = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))

    nearest = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    inside = False
    if other_kind[e] == 0:
        origin = _xform_point(x0o, other_loc[e])
        n_out = wp.normalize(_rot_of(x0o) * other_normal[e])  # out of the solid
        sd = wp.dot(sp - origin, n_out)
        if sd < _f64(0.0):
            inside = True
            nearest = sp - sd * n_out
    else:
        cen = _xform_point(x0o, other_loc[e])
        rad = other_rad[e]
        d = sp - cen
        dl = wp.length(d)
        if dl < rad and dl > _f64(0.0):
            inside = True
            nearest = cen + rad * d / dl

    if not inside:
        return

    displacement = nearest - sp
    distance = wp.length(displacement)
    if distance <= _f64(0.0):
        return
    force_dir = displacement / distance

    v = _point_velocity(x0o, xpo, xmo, inv_2h, nearest) - _point_velocity(x0m, xpm, xmm, inv_2h, nearest)
    vnormal = wp.dot(v, force_dir)
    vtangent = v - vnormal * force_dir

    area = area_scale[e] * spring_area[e]
    fn = stiffness[e] * area * distance * (_f64(1.0) + dissipation[e] * vnormal)
    if fn <= _f64(0.0):
        return
    force = fn * force_dir
    vslip = wp.length(vtangent)
    force = force + _stribeck_friction(fn, vtangent, vslip, us[e], ud[e], uv[e], vt[e])

    f_m = force
    f_o = -force
    elem_force[f, e] = f_m
    o_m = _pos_of(x0m)
    o_o = _pos_of(x0o)
    wp.atomic_add(body_force, f, mb, f_m)
    wp.atomic_add(body_torque, f, mb, wp.cross(nearest - o_m, f_m))
    wp.atomic_add(body_force, f, ob, f_o)
    wp.atomic_add(body_torque, f, ob, wp.cross(nearest - o_o, f_o))


# --------------------------------------------------------------------------- #
# Simple mesh loading for ContactMesh elastic-foundation springs.
# --------------------------------------------------------------------------- #
def _load_mesh(path: str):
    """Load an ``.obj`` or ASCII/binary ``.stl`` mesh as ``(vertices, faces)``.

    Returns ``(vertices[N,3] float64, faces[M,3] int)`` or ``None`` if the file
    cannot be read. VTP and other formats are not parsed (return ``None``).
    """
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".obj":
            verts: list[list[float]] = []
            faces: list[list[int]] = []
            with open(path) as fh:
                for line in fh:
                    if line.startswith("v "):
                        verts.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith("f "):
                        idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                        for k in range(1, len(idx) - 1):
                            faces.append([idx[0], idx[k], idx[k + 1]])
            if not verts or not faces:
                return None
            return np.asarray(verts, np.float64), np.asarray(faces, np.int64)
        if ext == ".stl":
            with open(path, "rb") as fh:
                head = fh.read(5)
                fh.seek(0)
                if head == b"solid":
                    text = fh.read().decode("ascii", "ignore")
                    if "facet" in text:
                        verts, faces = [], []
                        cur: list[list[float]] = []
                        for line in text.splitlines():
                            s = line.strip().split()
                            if len(s) >= 4 and s[0] == "vertex":
                                cur.append([float(s[1]), float(s[2]), float(s[3])])
                                if len(cur) == 3:
                                    n = len(verts)
                                    verts.extend(cur)
                                    faces.append([n, n + 1, n + 2])
                                    cur = []
                        if verts and faces:
                            return np.asarray(verts, np.float64), np.asarray(faces, np.int64)
                fh.seek(80)
                (ntri,) = struct.unpack("<I", fh.read(4))
                verts, faces = [], []
                for _ in range(ntri):
                    data = fh.read(50)
                    if len(data) < 50:
                        break
                    vals = struct.unpack("<12fH", data)
                    n = len(verts)
                    verts.append(list(vals[3:6]))
                    verts.append(list(vals[6:9]))
                    verts.append(list(vals[9:12]))
                    faces.append([n, n + 1, n + 2])
                if verts and faces:
                    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)
    except Exception:
        return None
    return None


# Default material properties (OpenSim ``constructProperties``).
_SMOOTH_DEFAULTS = {
    "stiffness": 1.0,
    "dissipation": 0.0,
    "static_friction": 0.0,
    "dynamic_friction": 0.0,
    "viscous_friction": 0.0,
    "transition_velocity": 0.01,
    "constant_contact_force": 1e-5,
    "hertz_smoothing": 300.0,
    "hunt_crossley_smoothing": 50.0,
}
_SURFACE_DEFAULTS = {
    "stiffness": 0.0,
    "dissipation": 0.0,
    "static_friction": 0.0,
    "dynamic_friction": 0.0,
    "viscous_friction": 0.0,
}


@dataclass
class _ContactWorkspace:
    """Fixed-shape device buffers for repeated contact evaluations."""

    batch: int
    stencil: wp.array[_f64]
    poses: wp.array[_mat44d]
    body_force: wp.array[_vec3d]
    body_torque: wp.array[_vec3d]
    smooth_force: wp.array[_vec3d]
    hc_force: wp.array[_vec3d]
    ef_force: wp.array[_vec3d]


class OpenSimContact:
    r"""Evaluate a model's OpenSim compliant contact forces with Warp kernels.

    All ``SmoothSphereHalfSpaceForce``, ``HuntCrossleyForce`` and
    ``ElasticFoundationForce`` elements in ``model`` are compiled to device
    element tables. Given coordinate values and speeds the class runs the
    validated forward kinematics, reads body point velocities from a
    central-difference pose stencil, and evaluates every contact element in
    parallel, returning the resultant body wrenches (ground frame).

    Args:
        model: Parsed model IR.
        device: Warp device (``"cpu"``, ``"cuda"``, a
            :class:`warp.context.Device`, or ``None`` for the CPU).
        meshes: Optional ``{contact_geometry_name: (vertices[N,3], faces[M,3])}``
            supplying triangle meshes for ``ContactMesh`` geometries whose files
            are absent or unparsable (used by ``ElasticFoundationForce``).

    Attributes:
        coordinate_names: Generalized coordinate names in model order.
        element_names: Contact-force element names in output-column order.
    """

    def __init__(self, model: OsimModel, device=None, meshes: dict | None = None):
        self.model = model
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.coordinate_names = self.fk.coordinate_names
        self.ncoord = self.fk.ncoord
        self._body_index = {n: i for i, n in enumerate(self.fk.body_names)}
        self._geom = {g.name: g for g in model.contact_geometry}
        self._user_meshes = meshes or {}
        self.element_names: list[str] = []
        self._build()

    # -- geometry helpers ---------------------------------------------------- #
    def _bidx(self, body: str) -> int:
        return self._body_index.get(body, 0)

    def _sphere(self, geom):
        return (self._bidx(geom.body), np.asarray(geom.location, np.float64), float(geom.radius))

    def _halfspace(self, geom):
        n_local = euler_xyz_to_matrix(*geom.orientation) @ np.array([1.0, 0.0, 0.0])
        return (self._bidx(geom.body), np.asarray(geom.location, np.float64), n_local)

    def _mesh_springs(self, geom):
        """Return ``(body, spring_pos[K,3], spring_area[K])`` in body frame for a ContactMesh."""
        mesh = self._user_meshes.get(geom.name)
        if mesh is None:
            mesh = _load_mesh(geom.mesh_file)
        if mesh is None:
            return None
        verts, faces = mesh
        verts = np.asarray(verts, np.float64)
        faces = np.asarray(faces, np.int64)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        x_off = np.eye(4)
        x_off[:3, :3] = euler_xyz_to_matrix(*geom.orientation)
        x_off[:3, 3] = np.asarray(geom.location, np.float64)
        pos_body = centroids @ x_off[:3, :3].T + x_off[:3, 3]
        return (self._bidx(geom.body), pos_body, areas)

    # -- build device element tables ---------------------------------------- #
    def _build(self):
        smooth: dict[str, list] = {
            k: []
            for k in (
                "s_body",
                "s_loc",
                "s_rad",
                "h_body",
                "h_loc",
                "h_normal",
                "stiffness",
                "dissipation",
                "us",
                "ud",
                "uv",
                "vt",
                "cf",
                "bd",
                "bv",
            )
        }
        hc: dict[str, list] = {
            k: []
            for k in (
                "kind",
                "a_body",
                "a_loc",
                "a_rad",
                "b_body",
                "b_loc",
                "b_rad",
                "b_normal",
                "a_stiff",
                "a_diss",
                "a_us",
                "a_ud",
                "a_uv",
                "b_stiff",
                "b_diss",
                "b_us",
                "b_ud",
                "b_uv",
                "vt",
            )
        }
        ef: dict[str, list] = {
            k: []
            for k in (
                "mesh_body",
                "spring_pos",
                "spring_area",
                "other_kind",
                "other_body",
                "other_loc",
                "other_normal",
                "other_rad",
                "stiffness",
                "dissipation",
                "us",
                "ud",
                "uv",
                "vt",
                "area_scale",
            )
        }
        self._smooth_owner: list[int] = []
        self._hc_owner: list[int] = []
        self._ef_owner: list[int] = []

        for ei, cf in enumerate(self.model.contact_forces):
            self.element_names.append(cf.name)
            if cf.type == "SmoothSphereHalfSpaceForce":
                self._add_smooth(cf, ei, smooth)
            elif cf.type == "HuntCrossleyForce":
                self._add_hunt_crossley(cf, ei, hc)
            elif cf.type == "ElasticFoundationForce":
                self._add_elastic_foundation(cf, ei, ef)

        self.num_elements = len(self.element_names)
        self._smooth = self._upload(smooth, _SMOOTH_SPEC)
        self._hc = self._upload(hc, _HC_SPEC)
        self._ef = self._upload(ef, _EF_SPEC)
        self.n_smooth = len(smooth["s_rad"])
        self.n_hc = len(hc["kind"])
        self.n_ef = len(ef["spring_area"])
        self.d_smooth_owner = wp.array(self._smooth_owner, dtype=wp.int32, device=self.device)
        self.d_hc_owner = wp.array(self._hc_owner, dtype=wp.int32, device=self.device)
        self.d_ef_owner = wp.array(self._ef_owner, dtype=wp.int32, device=self.device)
        touched = set(smooth["s_body"] + smooth["h_body"])
        touched.update(hc["a_body"] + hc["b_body"])
        touched.update(ef["mesh_body"] + ef["other_body"])
        self._all_touched_bodies = sorted(touched)
        self._touched_bodies = [body for body in self._all_touched_bodies if body != 0]
        self.d_touched_bodies = wp.array(self._touched_bodies, dtype=wp.int32, device=self.device)

    def _param(self, params: dict, key: str, defaults: dict) -> float:
        return float(params.get(key, defaults[key]))

    def _add_smooth(self, cf, ei, smooth):
        sph = self._geom.get(cf.sphere)
        hs = self._geom.get(cf.half_space)
        if sph is None or hs is None:
            return
        sb, sloc, srad = self._sphere(sph)
        hb, hloc, hn = self._halfspace(hs)
        p = cf.params
        smooth["s_body"].append(sb)
        smooth["s_loc"].append(sloc)
        smooth["s_rad"].append(srad)
        smooth["h_body"].append(hb)
        smooth["h_loc"].append(hloc)
        smooth["h_normal"].append(hn)
        smooth["stiffness"].append(self._param(p, "stiffness", _SMOOTH_DEFAULTS))
        smooth["dissipation"].append(self._param(p, "dissipation", _SMOOTH_DEFAULTS))
        smooth["us"].append(self._param(p, "static_friction", _SMOOTH_DEFAULTS))
        smooth["ud"].append(self._param(p, "dynamic_friction", _SMOOTH_DEFAULTS))
        smooth["uv"].append(self._param(p, "viscous_friction", _SMOOTH_DEFAULTS))
        smooth["vt"].append(self._param(p, "transition_velocity", _SMOOTH_DEFAULTS))
        smooth["cf"].append(self._param(p, "constant_contact_force", _SMOOTH_DEFAULTS))
        smooth["bd"].append(self._param(p, "hertz_smoothing", _SMOOTH_DEFAULTS))
        smooth["bv"].append(self._param(p, "hunt_crossley_smoothing", _SMOOTH_DEFAULTS))
        self._smooth_owner.append(ei)

    def _surf(self, cf, name, key):
        return self._param(cf.surface_params.get(name, {}), key, _SURFACE_DEFAULTS)

    def _add_hunt_crossley(self, cf, ei, hc):
        vt = float(cf.params.get("transition_velocity", 0.01))
        geoms = [self._geom[g] for g in cf.geometries if g in self._geom]
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                g1, g2 = geoms[i], geoms[j]
                pair = self._point_pair(g1, g2)
                if pair is None:
                    continue
                sphere_geom, other_geom, kind = pair
                if self._bidx(sphere_geom.body) == self._bidx(other_geom.body):
                    continue
                ab, a_center, arad = self._sphere(sphere_geom)
                if kind == 0:
                    bb, b_center, bn = self._halfspace(other_geom)
                    brad = 0.0
                else:
                    bb, b_center, brad = self._sphere(other_geom)
                    bn = np.zeros(3)
                hc["kind"].append(kind)
                hc["a_body"].append(ab)
                hc["a_loc"].append(a_center)
                hc["a_rad"].append(arad)
                hc["b_body"].append(bb)
                hc["b_loc"].append(b_center)
                hc["b_rad"].append(brad)
                hc["b_normal"].append(bn)
                hc["a_stiff"].append(self._surf(cf, sphere_geom.name, "stiffness"))
                hc["a_diss"].append(self._surf(cf, sphere_geom.name, "dissipation"))
                hc["a_us"].append(self._surf(cf, sphere_geom.name, "static_friction"))
                hc["a_ud"].append(self._surf(cf, sphere_geom.name, "dynamic_friction"))
                hc["a_uv"].append(self._surf(cf, sphere_geom.name, "viscous_friction"))
                hc["b_stiff"].append(self._surf(cf, other_geom.name, "stiffness"))
                hc["b_diss"].append(self._surf(cf, other_geom.name, "dissipation"))
                hc["b_us"].append(self._surf(cf, other_geom.name, "static_friction"))
                hc["b_ud"].append(self._surf(cf, other_geom.name, "dynamic_friction"))
                hc["b_uv"].append(self._surf(cf, other_geom.name, "viscous_friction"))
                hc["vt"].append(vt)
                self._hc_owner.append(ei)

    @staticmethod
    def _point_pair(g1, g2):
        """Return ``(sphere_geom, other_geom, kind)`` for a point-contact pair or ``None``."""
        t1, t2 = g1.type, g2.type
        if t1 == "ContactSphere" and t2 == "ContactHalfSpace":
            return g1, g2, 0
        if t2 == "ContactSphere" and t1 == "ContactHalfSpace":
            return g2, g1, 0
        if t1 == "ContactSphere" and t2 == "ContactSphere":
            return g1, g2, 1
        return None

    def _add_elastic_foundation(self, cf, ei, ef):
        vt = float(cf.params.get("transition_velocity", 0.01))
        geoms = [self._geom[g] for g in cf.geometries if g in self._geom]
        meshes = [g for g in geoms if g.type == "ContactMesh"]
        others = [g for g in geoms if g.type in ("ContactHalfSpace", "ContactSphere")]
        for mg in meshes:
            springs = self._mesh_springs(mg)
            if springs is None:
                continue
            mbody, pos_body, areas = springs
            for og in others:
                if self._bidx(mg.body) == self._bidx(og.body):
                    continue
                if og.type == "ContactHalfSpace":
                    okind = 0
                    ob, oloc, on = self._halfspace(og)
                    on = -on  # elastic_foundation_kernel wants the outward (free-space) normal
                    orad = 0.0
                else:
                    okind = 1
                    ob, oloc, orad = self._sphere(og)
                    on = np.zeros(3)
                area_scale = 1.0  # mesh vs primitive: only the mesh contributes springs
                for k in range(len(areas)):
                    ef["mesh_body"].append(mbody)
                    ef["spring_pos"].append(pos_body[k])
                    ef["spring_area"].append(float(areas[k]))
                    ef["other_kind"].append(okind)
                    ef["other_body"].append(ob)
                    ef["other_loc"].append(oloc)
                    ef["other_normal"].append(on)
                    ef["other_rad"].append(orad)
                    ef["stiffness"].append(self._surf(cf, mg.name, "stiffness"))
                    ef["dissipation"].append(self._surf(cf, mg.name, "dissipation"))
                    ef["us"].append(self._surf(cf, mg.name, "static_friction"))
                    ef["ud"].append(self._surf(cf, mg.name, "dynamic_friction"))
                    ef["uv"].append(self._surf(cf, mg.name, "viscous_friction"))
                    ef["vt"].append(vt)
                    ef["area_scale"].append(area_scale)
                    self._ef_owner.append(ei)

    def _upload(self, table: dict, spec: dict) -> dict:
        out = {}
        for key, dtype in spec.items():
            vals = table[key]
            if dtype is wp.int32:
                arr = np.asarray(vals, np.int32) if vals else np.zeros(0, np.int32)
            elif dtype is _f64:
                arr = np.asarray(vals, np.float64) if vals else np.zeros(0, np.float64)
            else:  # _vec3d
                arr = np.asarray(vals, np.float64).reshape(-1, 3) if vals else np.zeros((0, 3), np.float64)
            out[key] = wp.array(arr, dtype=dtype, device=self.device)
        return out

    # -- evaluation ---------------------------------------------------------- #
    def _create_device_workspace(self, batch: int) -> _ContactWorkspace:
        """Allocate reusable device buffers for a fixed contact batch."""
        return _ContactWorkspace(
            batch=batch,
            stencil=wp.empty((3 * batch, self.ncoord), dtype=_f64, device=self.device),
            poses=wp.empty((3 * batch, self.fk.nbody), dtype=_mat44d, device=self.device),
            body_force=wp.empty((batch, self.fk.nbody), dtype=_vec3d, device=self.device),
            body_torque=wp.empty((batch, self.fk.nbody), dtype=_vec3d, device=self.device),
            smooth_force=wp.empty((batch, max(self.n_smooth, 1)), dtype=_vec3d, device=self.device),
            hc_force=wp.empty((batch, max(self.n_hc, 1)), dtype=_vec3d, device=self.device),
            ef_force=wp.empty((batch, max(self.n_ef, 1)), dtype=_vec3d, device=self.device),
        )

    def _run(self, q: np.ndarray, qd: np.ndarray, h: float):
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float64)
        qd = np.ascontiguousarray(np.atleast_2d(qd), dtype=np.float64)
        q_wp = wp.array(q, dtype=_f64, device=self.device)
        qd_wp = wp.array(qd, dtype=_f64, device=self.device)
        return self._run_device(q_wp, qd_wp, h)

    def _run_device(
        self,
        q: wp.array[_f64],
        qd: wp.array[_f64],
        h: float,
        workspace: _ContactWorkspace | None = None,
    ):
        """Evaluate contact from device coordinate and speed batches."""
        n = q.shape[0]
        workspace = self._create_device_workspace(n) if workspace is None else workspace
        if workspace.batch != n:
            raise ValueError(f"contact workspace batch {workspace.batch} does not match input batch {n}")
        stencil = workspace.stencil
        wp.launch(
            contact_state_stencil_kernel,
            dim=(n, 3, self.ncoord),
            inputs=[q, qd, _f64(h), stencil],
            device=self.device,
        )
        self.fk._launch_body_transforms(stencil, out=workspace.poses)
        poses = workspace.poses
        body_force = workspace.body_force
        body_torque = workspace.body_torque
        body_force.zero_()
        body_torque.zero_()
        inv_2h = _f64(1.0 / (2.0 * h))
        # Element-force buffers are also reusable. Clear them because some
        # element families can accumulate more than one contact sub-pair.
        smooth_force = workspace.smooth_force
        hc_force = workspace.hc_force
        ef_force = workspace.ef_force
        smooth_force.zero_()
        hc_force.zero_()
        ef_force.zero_()

        if self.n_smooth:
            wp.launch(
                smooth_sphere_halfspace_kernel,
                dim=(n, self.n_smooth),
                inputs=[
                    poses,
                    self._smooth["s_body"],
                    self._smooth["s_loc"],
                    self._smooth["s_rad"],
                    self._smooth["h_body"],
                    self._smooth["h_loc"],
                    self._smooth["h_normal"],
                    self._smooth["stiffness"],
                    self._smooth["dissipation"],
                    self._smooth["us"],
                    self._smooth["ud"],
                    self._smooth["uv"],
                    self._smooth["vt"],
                    self._smooth["cf"],
                    self._smooth["bd"],
                    self._smooth["bv"],
                    inv_2h,
                    body_force,
                    body_torque,
                    smooth_force,
                ],
                device=self.device,
            )
        if self.n_hc:
            wp.launch(
                hunt_crossley_kernel,
                dim=(n, self.n_hc),
                inputs=[
                    poses,
                    self._hc["kind"],
                    self._hc["a_body"],
                    self._hc["a_loc"],
                    self._hc["a_rad"],
                    self._hc["b_body"],
                    self._hc["b_loc"],
                    self._hc["b_rad"],
                    self._hc["b_normal"],
                    self._hc["a_stiff"],
                    self._hc["a_diss"],
                    self._hc["a_us"],
                    self._hc["a_ud"],
                    self._hc["a_uv"],
                    self._hc["b_stiff"],
                    self._hc["b_diss"],
                    self._hc["b_us"],
                    self._hc["b_ud"],
                    self._hc["b_uv"],
                    self._hc["vt"],
                    inv_2h,
                    body_force,
                    body_torque,
                    hc_force,
                ],
                device=self.device,
            )
        if self.n_ef:
            wp.launch(
                elastic_foundation_kernel,
                dim=(n, self.n_ef),
                inputs=[
                    poses,
                    self._ef["mesh_body"],
                    self._ef["spring_pos"],
                    self._ef["spring_area"],
                    self._ef["other_kind"],
                    self._ef["other_body"],
                    self._ef["other_loc"],
                    self._ef["other_normal"],
                    self._ef["other_rad"],
                    self._ef["stiffness"],
                    self._ef["dissipation"],
                    self._ef["us"],
                    self._ef["ud"],
                    self._ef["uv"],
                    self._ef["vt"],
                    self._ef["area_scale"],
                    inv_2h,
                    body_force,
                    body_torque,
                    ef_force,
                ],
                device=self.device,
            )

        return poses, body_force, body_torque, smooth_force, hc_force, ef_force

    def _body_names(self) -> list[str]:
        """Return non-ground body names in device-wrench order."""
        return [self.fk.body_names[body] for body in self._touched_bodies]

    def _body_wrenches_device(
        self,
        q: wp.array[_f64],
        qd: wp.array[_f64],
        h: float,
        out: wp.array[_f64],
        workspace: _ContactWorkspace | None = None,
    ) -> None:
        """Write OpenSim-frame body wrenches without crossing the host boundary."""
        if not self._touched_bodies:
            return
        poses, body_force, body_torque, *_ = self._run_device(q, qd, h, workspace)
        wp.launch(
            pack_contact_body_wrench_kernel,
            dim=(q.shape[0], len(self._touched_bodies)),
            inputs=[poses, body_force, body_torque, self.d_touched_bodies, out],
            device=self.device,
        )

    def forces(
        self,
        q: np.ndarray,
        qd: np.ndarray | None = None,
        h: float = 1.0e-6,
        *,
        frame: Literal["newton", "opensim"] = "newton",
    ) -> np.ndarray:
        r"""World force per contact-force *element* (summed over sub-pairs).

        Args:
            q: Coordinate values, shape ``[num_frames, num_coordinates]`` or ``[num_coordinates]``.
            qd: Coordinate speeds, same shape (``None`` = at rest).
            h: Central-difference step for body point velocities [s].
            frame: Output world frame. ``"newton"`` returns Newton-standard
                Z-up vectors; ``"opensim"`` returns native OpenSim Y-up vectors.

        Returns:
            Force on the sphere/mesh body of each element [N], shape
            ``[num_frames, num_elements, 3]`` (single-frame collapsed if ``q`` is 1-D).
        """
        single = np.asarray(q).ndim == 1
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float64)
        if self.num_elements == 0:
            out = _convert_world_vectors(np.zeros((q.shape[0], 0, 3)), frame)
            return out[0] if single else out
        qd = np.zeros_like(q) if qd is None else np.ascontiguousarray(np.atleast_2d(qd), dtype=np.float64)
        _, _, _, smooth_force, hc_force, ef_force = self._run(q, qd, h)
        out_wp = wp.empty((q.shape[0], self.num_elements), dtype=_vec3d, device=self.device)
        wp.launch(
            reduce_contact_element_force_kernel,
            dim=(q.shape[0], self.num_elements),
            inputs=[
                smooth_force,
                self.d_smooth_owner,
                self.n_smooth,
                hc_force,
                self.d_hc_owner,
                self.n_hc,
                ef_force,
                self.d_ef_owner,
                self.n_ef,
                out_wp,
            ],
            device=self.device,
        )
        out = _convert_world_vectors(out_wp.numpy(), frame)
        return out[0] if single else out

    def body_wrenches(
        self,
        q: np.ndarray,
        qd: np.ndarray | None = None,
        h: float = 1.0e-6,
        *,
        frame: Literal["newton", "opensim"] = "newton",
    ):
        r"""Resultant contact wrench per body in the ``ExternalLoads`` ``[F P T]`` layout.

        Args:
            q: Coordinate values, shape ``[num_frames, num_coordinates]``.
            qd: Coordinate speeds, same shape (``None`` = at rest).
            h: Central-difference step for body point velocities [s].
            frame: Output world frame. ``"newton"`` returns Newton-standard
                Z-up vectors; ``"opensim"`` returns native OpenSim Y-up vectors.

        Returns:
            ``(bodies, wrenches)`` where ``bodies`` is the list of loaded body
            names and ``wrenches`` has shape ``[num_frames, num_bodies, 9]`` with
            columns ``[Fx Fy Fz Px Py Pz Tx Ty Tz]`` (force ``F`` applied at the
            body origin ``P`` plus resultant couple ``T``), ready for
            :meth:`~newton.opensim.ForwardDynamics.accelerations`.
            Returns ``([], zeros)`` if there are no contact forces.
        """
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float64)
        n = q.shape[0]
        if not self._touched_bodies:
            empty = np.zeros((n, 0, 3, 3))
            return [], _convert_world_vectors(empty, frame).reshape(n, 0, 9)
        qd = np.zeros_like(q) if qd is None else np.ascontiguousarray(np.atleast_2d(qd), dtype=np.float64)
        q_wp = wp.array(q, dtype=_f64, device=self.device)
        qd_wp = wp.array(qd, dtype=_f64, device=self.device)
        wrenches_wp = wp.empty((n, len(self._touched_bodies), 9), dtype=_f64, device=self.device)
        self._body_wrenches_device(q_wp, qd_wp, h, wrenches_wp)
        bodies = self._body_names()
        wrenches = wrenches_wp.numpy()
        vectors = wrenches.reshape(n, len(self._touched_bodies), 3, 3)
        wrenches = _convert_world_vectors(vectors, frame).reshape(wrenches.shape)
        return bodies, wrenches

    def _loaded_bodies(self) -> set[int]:
        """Return the cached body indices touched by any contact element."""
        return set(self._all_touched_bodies)

    def generalized_forces(
        self, q: np.ndarray, qd: np.ndarray | None = None, h: float = 1.0e-6, eps: float = 1.0e-6
    ) -> np.ndarray:
        r"""Project the contact body wrenches onto the generalized coordinates.

        Uses the geometric Jacobian transpose (central finite differences of the
        forward kinematics), matching the inverse/forward-dynamics convention, so
        the result adds directly to muscle/actuator generalized forces.

        Args:
            q: Coordinate values, shape ``[num_frames, num_coordinates]``.
            qd: Coordinate speeds, same shape (``None`` = at rest).
            h: Central-difference step for body point velocities [s].
            eps: Finite-difference step for the geometric Jacobian [rad or m].

        Returns:
            Generalized contact forces, shape ``[num_frames, num_coordinates]``
            (single-frame collapsed if ``q`` is 1-D).
        """
        single = np.asarray(q).ndim == 1
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float64)
        qd = np.zeros_like(q) if qd is None else np.ascontiguousarray(np.atleast_2d(qd), dtype=np.float64)
        q_wp = wp.array(q, dtype=_f64, device=self.device)
        qd_wp = wp.array(qd, dtype=_f64, device=self.device)
        _, body_force, body_torque, *_ = self._run_device(q_wp, qd_wp, h)
        jac, _ = self.fk._launch_body_jacobian_device(q_wp, eps)
        tau = wp.empty((q.shape[0], self.ncoord), dtype=_f64, device=self.device)
        wp.launch(
            contact_generalized_force_kernel,
            dim=(q.shape[0], self.ncoord),
            inputs=[jac, body_force, body_torque, self.fk.nbody, tau],
            device=self.device,
        )
        out = tau.numpy()
        return out[0] if single else out


# Upload specs: field name -> Warp dtype.
_SMOOTH_SPEC = {
    "s_body": wp.int32,
    "s_loc": _vec3d,
    "s_rad": _f64,
    "h_body": wp.int32,
    "h_loc": _vec3d,
    "h_normal": _vec3d,
    "stiffness": _f64,
    "dissipation": _f64,
    "us": _f64,
    "ud": _f64,
    "uv": _f64,
    "vt": _f64,
    "cf": _f64,
    "bd": _f64,
    "bv": _f64,
}
_HC_SPEC = {
    "kind": wp.int32,
    "a_body": wp.int32,
    "a_loc": _vec3d,
    "a_rad": _f64,
    "b_body": wp.int32,
    "b_loc": _vec3d,
    "b_rad": _f64,
    "b_normal": _vec3d,
    "a_stiff": _f64,
    "a_diss": _f64,
    "a_us": _f64,
    "a_ud": _f64,
    "a_uv": _f64,
    "b_stiff": _f64,
    "b_diss": _f64,
    "b_us": _f64,
    "b_ud": _f64,
    "b_uv": _f64,
    "vt": _f64,
}
_EF_SPEC = {
    "mesh_body": wp.int32,
    "spring_pos": _vec3d,
    "spring_area": _f64,
    "other_kind": wp.int32,
    "other_body": wp.int32,
    "other_loc": _vec3d,
    "other_normal": _vec3d,
    "other_rad": _f64,
    "stiffness": _f64,
    "dissipation": _f64,
    "us": _f64,
    "ud": _f64,
    "uv": _f64,
    "vt": _f64,
    "area_scale": _f64,
}
