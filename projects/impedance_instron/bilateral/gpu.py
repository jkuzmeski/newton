# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Double-precision bilateral chain mechanics and impedance actuation for CUDA worlds.

Mirrors :class:`.mechanics.BilateralChain` term by term. Shoes use the lean
elastic-Coulomb kernels of :mod:`..hogan.gpu_shoe`, one foundation per foot.
"""

from __future__ import annotations

import math

import warp as wp

from .mechanics import BilateralChain

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

Vec6 = wp.types.vector(6, wp.float64)
Vec7 = wp.types.vector(7, wp.float64)
Vec9 = wp.types.vector(9, wp.float64)
Mat9 = wp.types.matrix((9, 9), wp.float64)
FEATURE_COUNT = 14
Features = wp.types.vector(FEATURE_COUNT, wp.float64)
_PI = wp.constant(wp.float64(math.pi))
_HALF_PI = wp.constant(wp.float64(0.5 * math.pi))
_TWO_PI = wp.constant(wp.float64(2.0 * math.pi))
_FLIGHT_HEIGHT = wp.constant(wp.float32(10.0))

RUNNING = 0
FAILURES = {
    2: "Hip height screen exceeded",
    3: "Numerical speed screen exceeded",
    4: "Nonfinite actuator/contact output",
    5: "Contact force screen exceeded",
    6: "Driven shoe compression screen exceeded",
    7: "Singular mass matrix",
    8: "Nonfinite integrated state",
    9: "Integrated state exceeded height/speed screen",
}
MODE_IMPEDANCE = 0
"""Equilibrium, stiffness, and damping are set per control step (neural policy)."""
MODE_LINEAR = 1
"""Shared linear Hogan feature schedule evaluated every substep."""


@wp.struct
class ChainParams:
    lengths: wp.vec4d
    """Thigh and shank lengths, right then left [m]."""
    masses: Vec7
    inertias: Vec7
    com_x: Vec7
    com_z: Vec7


@wp.struct
class Settings:
    mode: int
    gravity: wp.float64
    dt: wp.float64
    load_rate: wp.float64
    """Per-substep normal-load filter fraction."""
    phase_feedback: wp.float64
    reference_speed: wp.float64
    speed_scale: wp.float64
    cadence_gain: wp.float64
    immediate_damping: int
    lower: Vec6
    upper: Vec6
    stiffness_max: Vec6
    damping_max: Vec6
    torque_cap: Vec6
    torque_rate: Vec6
    pitch_r: wp.float64
    pitch_l: wp.float64
    max_force: wp.float64
    hip_floor: wp.float64
    max_speed: wp.float64
    compression_limit: wp.float64


@wp.struct
class Worlds:
    q: wp.array[Vec9]
    v: wp.array[Vec9]
    torque: wp.array[Vec6]
    load: wp.array[Vec6]
    phase: wp.array[wp.float64]
    sense: wp.array[wp.vec2d]
    """Filtered normal load per foot divided by body weight (right, left)."""
    status: wp.array[int]
    speed: wp.array[wp.float64]
    frequency: wp.array[wp.float64]
    """Autonomous oscillator frequency per world [Hz]."""
    response: wp.array[wp.float64]
    """Per-substep torque response fraction ``1 - exp(-dt / T)`` per world."""
    equilibrium: wp.array[Vec6]
    stiffness: wp.array[Vec6]
    damping: wp.array[Vec6]
    enabled_r: wp.array[int]
    enabled_l: wp.array[int]
    fraction_r: wp.array[wp.float64]
    fraction_l: wp.array[wp.float64]
    invalid_r: wp.array[int]
    invalid_l: wp.array[int]
    grf_sum: wp.array[wp.vec4d]
    """Summed fx, fz of the right then left foot over the control interval [N]."""
    fz_square: wp.array[wp.vec2d]
    effort: wp.array[Vec6]
    """Summed squared joint load over the control interval [N^2 m^2]."""
    power_positive: wp.array[Vec6]
    power_negative: wp.array[Vec6]
    compression_max: wp.array[wp.float64]
    substeps: wp.array[int]


@wp.func
def _rotate(angle: wp.float64, x: wp.float64, z: wp.float64):
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec2d(c * x - s * z, s * x + c * z)


@wp.func
def angular9(body: int):
    """Angular Jacobian of a body; pelvis 0, right thigh/shank/foot 1-3, left 4-6."""
    a = Vec9(wp.float64(0.0))
    a[2] = wp.float64(1.0)
    if body > 0:
        side = (body - 1) // 3
        segment = body - 1 - 3 * side
        for i in range(3):
            if i <= segment:
                a[3 + 3 * side + i] = wp.float64(1.0)
    return a


@wp.func
def _body_terms(
    mass: Mat9,
    bias: Vec9,
    m: wp.float64,
    inertia: wp.float64,
    a: Vec9,
    ox: Vec9,
    oz: Vec9,
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
def dynamics9(q: Vec9, v: Vec9, p: ChainParams, gravity: wp.float64):
    """Mirror :meth:`.mechanics.BilateralChain.dynamics`: ``M @ acceleration + bias = load``."""
    mass = Mat9(wp.float64(0.0))
    bias = Vec9(wp.float64(0.0))
    hx = Vec9(wp.float64(0.0))
    hz = Vec9(wp.float64(0.0))
    hx[0] = wp.float64(1.0)
    hz[1] = wp.float64(1.0)
    zero = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    a0 = angular9(0)
    mass, bias = _body_terms(
        mass, bias, p.masses[0], p.inertias[0], a0, hx, hz, q[2], p.com_x[0], p.com_z[0], wp.dot(a0, v), zero, gravity
    )
    for side in range(2):
        b = 1 + 3 * side
        a1 = angular9(b)
        a2 = angular9(b + 1)
        a3 = angular9(b + 2)
        angle1 = wp.dot(a1, q) + _PI
        angle2 = wp.dot(a2, q) + _PI
        angle3 = wp.dot(a3, q) + _PI + _HALF_PI
        w1 = wp.dot(a1, v)
        w2 = wp.dot(a2, v)
        mass, bias = _body_terms(
            mass, bias, p.masses[b], p.inertias[b], a1, hx, hz, angle1, p.com_x[b], p.com_z[b], w1, zero, gravity
        )
        r1 = _rotate(angle1, p.lengths[2 * side], wp.float64(0.0))
        ox2 = hx - r1[1] * a1
        oz2 = hz + r1[0] * a1
        prior2 = -r1 * (w1 * w1)
        mass, bias = _body_terms(
            mass,
            bias,
            p.masses[b + 1],
            p.inertias[b + 1],
            a2,
            ox2,
            oz2,
            angle2,
            p.com_x[b + 1],
            p.com_z[b + 1],
            w2,
            prior2,
            gravity,
        )
        r2 = _rotate(angle2, p.lengths[2 * side + 1], wp.float64(0.0))
        ox3 = ox2 - r2[1] * a2
        oz3 = oz2 + r2[0] * a2
        prior3 = prior2 - r2 * (w2 * w2)
        mass, bias = _body_terms(
            mass,
            bias,
            p.masses[b + 2],
            p.inertias[b + 2],
            a3,
            ox3,
            oz3,
            angle3,
            p.com_x[b + 2],
            p.com_z[b + 2],
            wp.dot(a3, v),
            prior3,
            gravity,
        )
    return mass, bias


@wp.func
def ankle9(q: Vec9, p: ChainParams, side: int):
    """Return an ankle's position [m], its x/z Jacobian rows, and the foot angle [rad]."""
    b = 1 + 3 * side
    a1 = angular9(b)
    a2 = angular9(b + 1)
    a3 = angular9(b + 2)
    r1 = _rotate(wp.dot(a1, q) + _PI, p.lengths[2 * side], wp.float64(0.0))
    r2 = _rotate(wp.dot(a2, q) + _PI, p.lengths[2 * side + 1], wp.float64(0.0))
    jx = Vec9(wp.float64(0.0))
    jz = Vec9(wp.float64(0.0))
    jx[0] = wp.float64(1.0)
    jz[1] = wp.float64(1.0)
    jx = jx - r1[1] * a1 - r2[1] * a2
    jz = jz + r1[0] * a1 + r2[0] * a2
    position = wp.vec2d(q[0] + r1[0] + r2[0], q[1] + r1[1] + r2[1])
    return position, jx, jz, wp.dot(a3, q) + _PI + _HALF_PI


@wp.func
def cholesky_solve9(m: Mat9, b: Vec9):
    """Solve the symmetric positive-definite system; report failure on a nonpositive pivot."""
    lower = Mat9(wp.float64(0.0))
    ok = True
    for j in range(9):
        d = m[j, j]
        for k in range(j):
            d -= lower[j, k] * lower[j, k]
        if not (d > wp.float64(0.0)):
            ok = False
            d = wp.float64(1.0)
        d = wp.sqrt(d)
        lower[j, j] = d
        for i in range(j + 1, 9):
            t = m[i, j]
            for k in range(j):
                t -= lower[i, k] * lower[j, k]
            lower[i, j] = t / d
    y = Vec9(wp.float64(0.0))
    for i in range(9):
        t = b[i]
        for k in range(i):
            t -= lower[i, k] * y[k]
        y[i] = t / lower[i, i]
    x = Vec9(wp.float64(0.0))
    for ii in range(9):
        i = 8 - ii
        t = y[i]
        for k in range(i + 1, 9):
            t -= lower[k, i] * x[k]
        x[i] = t / lower[i, i]
    return x, ok


@wp.func
def _finite9(x: Vec9):
    ok = True
    for i in range(9):
        ok = ok and wp.isfinite(x[i])
    return ok


@wp.func
def _finite6(x: Vec6):
    ok = True
    for i in range(6):
        ok = ok and wp.isfinite(x[i])
    return ok


@wp.func
def _too_fast(v: Vec9, maximum: wp.float64):
    fast = wp.length(wp.vec2d(v[0], v[1])) > maximum
    for j in range(2, 9):
        fast = fast or wp.abs(v[j]) > maximum
    return fast


@wp.func
def _carrier(q: Vec9, v: Vec9, p: ChainParams, side: int, pitch: wp.float64):
    position, jx, jz, angle = ankle9(q, p, side)
    half = (angle - pitch) / wp.float64(2.0)
    transform = wp.transform(
        wp.vec3(wp.float32(position[0]), 0.0, wp.float32(position[1])),
        wp.quat(0.0, wp.float32(-wp.sin(half)), 0.0, wp.float32(wp.cos(half))),
    )
    omega = wp.dot(angular9(3 + 3 * side), v)
    twist = wp.spatial_vector(wp.float32(wp.dot(jx, v)), 0.0, wp.float32(wp.dot(jz, v)), 0.0, wp.float32(-omega), 0.0)
    return transform, twist


@wp.func
def _features(
    q: Vec9, v: Vec9, p: ChainParams, cfg: Settings, phase: wp.float64, load: wp.float64, side: int, speed: wp.float64
):
    """Per-leg linear Hogan features; the left leg runs half a cycle behind the right."""
    ankle, _jx, _jz, _angle = ankle9(q, p, side)
    leg_phase = phase + wp.float64(side) * _PI
    load_bw = wp.clamp(load, wp.float64(0.0), wp.float64(4.0))
    return Features(
        wp.float64(1.0),
        wp.sin(leg_phase),
        wp.cos(leg_phase),
        wp.sin(wp.float64(2.0) * leg_phase),
        wp.cos(wp.float64(2.0) * leg_phase),
        wp.sin(wp.float64(3.0) * leg_phase),
        wp.cos(wp.float64(3.0) * leg_phase),
        load_bw,
        load_bw * wp.sin(leg_phase),
        load_bw * wp.cos(leg_phase),
        wp.clamp(
            (q[0] - ankle[0]) / (p.lengths[2 * side] + p.lengths[2 * side + 1]), wp.float64(-1.0), wp.float64(1.0)
        ),
        wp.clamp(q[2] - _HALF_PI, wp.float64(-1.0), wp.float64(1.0)),
        wp.clamp((speed - v[0]) / cfg.speed_scale, wp.float64(-2.0), wp.float64(2.0)),
        wp.clamp((speed - cfg.reference_speed) / cfg.speed_scale, wp.float64(-2.0), wp.float64(2.0)),
    )


@wp.kernel
def stage(
    cfg: Settings,
    params: wp.array[ChainParams],
    weights: wp.array3d[wp.float64],
    per_candidate: int,
    data: Worlds,
    body_q_r: wp.array[wp.transform],
    body_qd_r: wp.array[wp.spatial_vector],
    body_q_l: wp.array[wp.transform],
    body_qd_l: wp.array[wp.spatial_vector],
):
    """Command joint torque and stage both shoe carriers for one substep."""
    w = wp.tid()
    if data.status[w] != 0:
        return
    q = data.q[w]
    v = data.v[w]
    code = int(0)
    if q[1] < cfg.hip_floor:
        code = 2
    elif _too_fast(v, cfg.max_speed):
        code = 3
    if code != 0:
        data.status[w] = code
        data.enabled_r[w] = 0
        data.enabled_l[w] = 0
        return
    p = params[w]
    equilibrium = data.equilibrium[w]
    stiffness = data.stiffness[w]
    damping = data.damping[w]
    if cfg.mode == MODE_LINEAR:
        c = w // per_candidate
        sense = data.sense[w]
        for side in range(2):
            features = _features(q, v, p, cfg, data.phase[w], sense[side], side, data.speed[w])
            for output in range(3):
                for joint in range(3):
                    logit = wp.float64(0.0)
                    for feature in range(FEATURE_COUNT):
                        logit += weights[c, output * 3 + joint, feature] * features[feature]
                    logit = wp.clamp(logit, wp.float64(-40.0), wp.float64(40.0))
                    fraction = wp.float64(1.0) / (wp.float64(1.0) + wp.exp(-logit))
                    j = 3 * side + joint
                    if output == 0:
                        equilibrium[j] = cfg.lower[j] + (cfg.upper[j] - cfg.lower[j]) * fraction
                    elif output == 1:
                        stiffness[j] = cfg.stiffness_max[j] * fraction
                    else:
                        damping[j] = cfg.damping_max[j] * fraction
    load = Vec6(wp.float64(0.0))
    torque = data.torque[w]
    for j in range(6):
        spring = stiffness[j] * (equilibrium[j] - q[j + 3])
        command = spring - damping[j] * v[j + 3]
        immediate = wp.float64(0.0)
        if cfg.immediate_damping != 0:
            command = spring
            immediate = damping[j]
        cap = cfg.torque_cap[j]
        limit = cfg.torque_rate[j] * cfg.dt
        desired = wp.clamp(command, -cap, cap)
        change = wp.clamp(data.response[w] * (desired - torque[j]), -limit, limit)
        torque[j] = wp.clamp(torque[j] + change, -cap, cap)
        load[j] = wp.clamp(torque[j] - immediate * v[j + 3], -cap, cap)
    data.torque[w] = torque
    data.load[w] = load
    transform, twist = _carrier(q, v, p, 0, cfg.pitch_r)
    body_q_r[w] = transform
    body_qd_r[w] = twist
    transform, twist = _carrier(q, v, p, 1, cfg.pitch_l)
    body_q_l[w] = transform
    body_qd_l[w] = twist


@wp.kernel
def park(
    data: Worlds,
    body_q_r: wp.array[wp.transform],
    body_qd_r: wp.array[wp.spatial_vector],
    body_q_l: wp.array[wp.transform],
    body_qd_l: wp.array[wp.spatial_vector],
):
    """Lift both carriers of stopped worlds far above the ground so their shoes stay idle."""
    w = wp.tid()
    if data.status[w] == 0:
        return
    lifted = wp.transform(wp.vec3(0.0, 0.0, _FLIGHT_HEIGHT), wp.quat_identity())
    body_q_r[w] = lifted
    body_q_l[w] = lifted
    body_qd_r[w] = wp.spatial_vector()
    body_qd_l[w] = wp.spatial_vector()


@wp.kernel
def advance(
    cfg: Settings,
    params: wp.array[ChainParams],
    data: Worlds,
    body_f_r: wp.array[wp.spatial_vector],
    body_f_l: wp.array[wp.spatial_vector],
):
    """Apply both shoe wrenches, integrate one semi-implicit Euler substep, and update sensors."""
    w = wp.tid()
    if data.status[w] != 0:
        return
    p = params[w]
    q = data.q[w]
    v = data.v[w]
    load = data.load[w]
    fr = body_f_r[w]
    fl = body_f_l[w]
    fxr = wp.float64(fr[0])
    fzr = wp.float64(fr[2])
    mr = -wp.float64(fr[4])
    fxl = wp.float64(fl[0])
    fzl = wp.float64(fl[2])
    ml = -wp.float64(fl[4])
    compression = wp.max(data.fraction_r[w], data.fraction_l[w])
    code = int(0)
    finite = wp.isfinite(fxr) and wp.isfinite(fzr) and wp.isfinite(mr)
    finite = finite and wp.isfinite(fxl) and wp.isfinite(fzl) and wp.isfinite(ml)
    if not _finite6(load) or not finite or data.invalid_r[w] != 0 or data.invalid_l[w] != 0:
        code = 4
    elif (
        fzr < wp.float64(-1.0e-6)
        or fzl < wp.float64(-1.0e-6)
        or wp.length(wp.vec2d(fxr, fzr)) > cfg.max_force
        or wp.length(wp.vec2d(fxl, fzl)) > cfg.max_force
    ):
        code = 5
    elif compression > cfg.compression_limit + wp.float64(1.0e-6):
        code = 6
    if code != 0:
        data.status[w] = code
        data.enabled_r[w] = 0
        data.enabled_l[w] = 0
        return
    mass, bias = dynamics9(q, v, p, cfg.gravity)
    _pr, jxr, jzr, _ar = ankle9(q, p, 0)
    _pl, jxl, jzl, _al = ankle9(q, p, 1)
    force = fxr * jxr + fzr * jzr + mr * angular9(3) + fxl * jxl + fzl * jzl + ml * angular9(6)
    for j in range(6):
        force[j + 3] = force[j + 3] + load[j]
    acceleration, ok = cholesky_solve9(mass, force - bias)
    velocity = v + cfg.dt * acceleration
    position = q + cfg.dt * velocity
    if not ok:
        code = 7
    elif not _finite9(position) or not _finite9(velocity):
        code = 8
    elif position[1] < cfg.hip_floor or _too_fast(velocity, cfg.max_speed):
        code = 9
    if code != 0:
        data.status[w] = code
        data.enabled_r[w] = 0
        data.enabled_l[w] = 0
        return
    total = wp.float64(0.0)
    for b in range(7):
        total += p.masses[b]
    weight = total * cfg.gravity
    sense = data.sense[w]
    sense[0] = sense[0] + cfg.load_rate * (wp.max(fzr, wp.float64(0.0)) / weight - sense[0])
    sense[1] = sense[1] + cfg.load_rate * (wp.max(fzl, wp.float64(0.0)) / weight - sense[1])
    phase = data.phase[w]
    speed = wp.clamp((data.speed[w] - cfg.reference_speed) / cfg.speed_scale, wp.float64(-2.0), wp.float64(2.0))
    frequency = data.frequency[w] * wp.exp(wp.clamp(cfg.cadence_gain * speed, wp.float64(-2.0), wp.float64(2.0)))
    drive = wp.min(sense[0], wp.float64(1.0)) * wp.sin(phase) - wp.min(sense[1], wp.float64(1.0)) * wp.sin(phase)
    modulation = wp.float64(1.0) - cfg.phase_feedback * drive
    phase = phase + _TWO_PI * frequency * modulation * cfg.dt
    phase = phase - _TWO_PI * wp.floor(phase / _TWO_PI)
    data.sense[w] = sense
    data.phase[w] = phase
    data.q[w] = position
    data.v[w] = velocity
    data.grf_sum[w] = data.grf_sum[w] + wp.vec4d(fxr, fzr, fxl, fzl)
    data.fz_square[w] = data.fz_square[w] + wp.vec2d(fzr * fzr, fzl * fzl)
    effort = data.effort[w]
    positive = data.power_positive[w]
    negative = data.power_negative[w]
    for j in range(6):
        effort[j] = effort[j] + load[j] * load[j]
        power = load[j] * v[j + 3]
        positive[j] = positive[j] + wp.max(power, wp.float64(0.0))
        negative[j] = negative[j] + wp.min(power, wp.float64(0.0))
    data.effort[w] = effort
    data.power_positive[w] = positive
    data.power_negative[w] = negative
    data.compression_max[w] = wp.max(data.compression_max[w], compression)
    data.substeps[w] = data.substeps[w] + 1


@wp.kernel
def assign_params(
    source: wp.array[ChainParams], segment: wp.array[wp.int32], mask: wp.array[wp.int32], params: wp.array[ChainParams]
):
    """Give masked worlds the chain geometry of their measured segment."""
    w = wp.tid()
    if mask[w] != 0:
        params[w] = source[segment[w]]


def chain_params(chain: BilateralChain) -> ChainParams:
    """Pack a CPU chain for the CUDA kernels."""
    p = ChainParams()
    p.lengths = wp.vec4d(*chain.lengths_m.ravel())
    p.masses = Vec7(*chain.masses_kg)
    p.inertias = Vec7(*chain.inertias_kg_m2)
    p.com_x = Vec7(*chain.com_local_m[:, 0])
    p.com_z = Vec7(*chain.com_local_m[:, 1])
    return p
