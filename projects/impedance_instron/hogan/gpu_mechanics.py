# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Double-precision chain mechanics and shoe staging for the causal CUDA runner."""

from __future__ import annotations

import math

import warp as wp

from .mechanics import Chain

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

Vec6 = wp.types.vector(6, wp.float64)
Mat6 = wp.types.matrix((6, 6), wp.float64)
_PI = wp.constant(wp.float64(math.pi))
_HALF_PI = wp.constant(wp.float64(0.5 * math.pi))


@wp.struct
class ChainParams:
    lengths: wp.vec2d
    endpoint: wp.vec2d
    masses: wp.vec4d
    inertias: wp.vec4d
    com_x: wp.vec4d
    com_z: wp.vec4d


@wp.struct
class Settings:
    gravity: wp.float64
    threshold: wp.float64
    compression_limit: wp.float64
    max_force: wp.float64
    hip_floor: wp.float64
    max_speed: wp.float64
    pitch: wp.float64
    stance_count: int


@wp.func
def _rotate(angle: wp.float64, x: wp.float64, z: wp.float64):
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec2d(c * x - s * z, s * x + c * z)


@wp.func
def _angular(body: int):
    a = Vec6(wp.float64(0.0))
    for j in range(4):
        if j <= body:
            a[j + 2] = wp.float64(1.0)
    return a


@wp.func
def _angles(q: Vec6):
    a0 = q[2]
    a1 = a0 + q[3] + _PI
    a2 = a1 + q[4]
    a3 = a2 + q[5] + _HALF_PI
    return wp.vec4d(a0, a1, a2, a3)


@wp.func
def _ankle(q: Vec6, p: ChainParams):
    """Return ankle position [m], its x/z Jacobian rows, and the foot angle [rad]."""
    ang = _angles(q)
    r1 = _rotate(ang[1], p.lengths[0], wp.float64(0.0))
    r2 = _rotate(ang[2], p.lengths[1], wp.float64(0.0))
    jx = Vec6(wp.float64(0.0))
    jz = Vec6(wp.float64(0.0))
    jx[0] = wp.float64(1.0)
    jz[1] = wp.float64(1.0)
    a1 = _angular(1)
    a2 = _angular(2)
    jx = jx - r1[1] * a1 - r2[1] * a2
    jz = jz + r1[0] * a1 + r2[0] * a2
    position = wp.vec2d(q[0] + r1[0] + r2[0], q[1] + r1[1] + r2[1])
    return position, jx, jz, ang[3]


@wp.func
def _body_terms(
    mass: Mat6,
    bias: Vec6,
    m: wp.float64,
    inertia: wp.float64,
    a: Vec6,
    ox: Vec6,
    oz: Vec6,
    angle: wp.float64,
    cx: wp.float64,
    cz: wp.float64,
    omega: wp.float64,
    prior: wp.vec2d,
    gravity: wp.float64,
):
    c = _rotate(angle, cx, cz)
    jx = ox - c[1] * a
    jz = oz + c[0] * a
    w2 = omega * omega
    ax = prior[0] - c[0] * w2
    az = prior[1] - c[1] * w2 + gravity
    mass = mass + m * (wp.outer(jx, jx) + wp.outer(jz, jz)) + inertia * wp.outer(a, a)
    bias = bias + m * (ax * jx + az * jz)
    return mass, bias


@wp.func
def _dynamics(q: Vec6, v: Vec6, p: ChainParams, gravity: wp.float64):
    """Mirror :meth:`.mechanics.Chain.dynamics`: ``M @ acceleration + bias = load``."""
    ang = _angles(q)
    r1 = _rotate(ang[1], p.lengths[0], wp.float64(0.0))
    r2 = _rotate(ang[2], p.lengths[1], wp.float64(0.0))
    a0 = _angular(0)
    a1 = _angular(1)
    a2 = _angular(2)
    a3 = _angular(3)
    w1 = wp.dot(a1, v)
    w2 = wp.dot(a2, v)
    ox = Vec6(wp.float64(0.0))
    oz = Vec6(wp.float64(0.0))
    ox[0] = wp.float64(1.0)
    oz[1] = wp.float64(1.0)
    mass = Mat6(wp.float64(0.0))
    bias = Vec6(wp.float64(0.0))
    zero = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    mass, bias = _body_terms(
        mass, bias, p.masses[0], p.inertias[0], a0, ox, oz, ang[0], p.com_x[0], p.com_z[0], wp.dot(a0, v), zero, gravity
    )
    mass, bias = _body_terms(
        mass, bias, p.masses[1], p.inertias[1], a1, ox, oz, ang[1], p.com_x[1], p.com_z[1], w1, zero, gravity
    )
    ox2 = ox - r1[1] * a1
    oz2 = oz + r1[0] * a1
    prior2 = -r1 * (w1 * w1)
    mass, bias = _body_terms(
        mass, bias, p.masses[2], p.inertias[2], a2, ox2, oz2, ang[2], p.com_x[2], p.com_z[2], w2, prior2, gravity
    )
    ox3 = ox2 - r2[1] * a2
    oz3 = oz2 + r2[0] * a2
    prior3 = prior2 - r2 * (w2 * w2)
    mass, bias = _body_terms(
        mass,
        bias,
        p.masses[3],
        p.inertias[3],
        a3,
        ox3,
        oz3,
        ang[3],
        p.com_x[3],
        p.com_z[3],
        wp.dot(a3, v),
        prior3,
        gravity,
    )
    return mass, bias


@wp.func
def _cholesky_solve(m: Mat6, b: Vec6):
    """Solve the symmetric positive-definite system; report failure on a nonpositive pivot."""
    lower = Mat6(wp.float64(0.0))
    ok = True
    for j in range(6):
        d = m[j, j]
        for k in range(j):
            d -= lower[j, k] * lower[j, k]
        if not (d > wp.float64(0.0)):
            ok = False
            d = wp.float64(1.0)
        d = wp.sqrt(d)
        lower[j, j] = d
        for i in range(j + 1, 6):
            t = m[i, j]
            for k in range(j):
                t -= lower[i, k] * lower[j, k]
            lower[i, j] = t / d
    y = Vec6(wp.float64(0.0))
    for i in range(6):
        t = b[i]
        for k in range(i):
            t -= lower[i, k] * y[k]
        y[i] = t / lower[i, i]
    x = Vec6(wp.float64(0.0))
    for ii in range(6):
        i = 5 - ii
        t = y[i]
        for k in range(i + 1, 6):
            t -= lower[k, i] * x[k]
        x[i] = t / lower[i, i]
    return x, ok


@wp.func
def _finite6(x: Vec6):
    ok = True
    for i in range(6):
        ok = ok and wp.isfinite(x[i])
    return ok


@wp.kernel
def _stage(
    params: wp.array[ChainParams],
    cfg: Settings,
    status: wp.array[int],
    state: wp.array[Vec6],
    velocity: wp.array[Vec6],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    w = wp.tid()
    if status[w] != 0:
        return
    q = state[w]
    v = velocity[w]
    position, jx, jz, angle = _ankle(q, params[w % cfg.stance_count])
    half = (angle - cfg.pitch) / wp.float64(2.0)
    body_q[w] = wp.transform(
        wp.vec3(wp.float32(position[0]), 0.0, wp.float32(position[1])),
        wp.quat(0.0, wp.float32(-wp.sin(half)), 0.0, wp.float32(wp.cos(half))),
    )
    omega = v[2] + v[3] + v[4] + v[5]
    body_qd[w] = wp.spatial_vector(
        wp.float32(wp.dot(jx, v)), 0.0, wp.float32(wp.dot(jz, v)), 0.0, wp.float32(-omega), 0.0
    )


@wp.kernel
def _tick(clock: wp.array[int]):
    clock[0] = clock[0] + 1


def _chain_params(chain: Chain) -> ChainParams:
    p = ChainParams()
    p.lengths = wp.vec2d(*chain.lengths_m)
    p.endpoint = wp.vec2d(*chain.endpoint_local_m)
    p.masses = wp.vec4d(*chain.masses_kg)
    p.inertias = wp.vec4d(*chain.inertias_kg_m2)
    p.com_x = wp.vec4d(*chain.com_local_m[:, 0])
    p.com_z = wp.vec4d(*chain.com_local_m[:, 1])
    return p
