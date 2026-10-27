# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared scalar contact mechanics of the column bed.

Callers choose boundary coordinates and own state storage. These functions own
the lateral shear coupling, the bristle friction law, contact kinematics, and
the two-surface column balance.
"""

import warp as wp


@wp.func
def pasternak_coupling(t_i: wp.float32, t_j: wp.float32, shear: wp.float32) -> wp.float32:
    """Return the symmetric shear conductance [N/m] between two columns.

    Each column contributes its own Pasternak coefficient ``shear * t`` from the
    shear modulus ``shear`` [Pa] of the layers and its rest thickness ``t`` [m].
    The two halves of the layer act in series, so the thinner column bounds the pair.
    """
    return 2.0 * shear * t_i * t_j / (t_i + t_j)


@wp.func
def cone_viscous_scale(f_elastic: wp.vec2, f_viscous: wp.vec2, f_max: wp.float32) -> wp.float32:
    """Largest fraction of a viscous force that keeps the total tangential force inside the cone.

    The elastic part is already on or inside the cone after the radial return, so scaling
    only the viscous part keeps the Coulomb bound strict while the retained fraction still
    opposes the slip velocity (the pair stays dissipative).
    """
    a = wp.dot(f_viscous, f_viscous)
    if a <= 1.0e-18:
        return 0.0
    b = 2.0 * wp.dot(f_elastic, f_viscous)
    c = wp.dot(f_elastic, f_elastic) - f_max * f_max
    if c > 0.0:
        c = 0.0
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 0.0
    return wp.clamp((-b + wp.sqrt(disc)) / (2.0 * a), 0.0, 1.0)


@wp.func
def bristle_step(
    position: wp.vec2,
    velocity: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    anchor: wp.vec2,
    stuck: int,
    dwell: float,
) -> tuple[wp.vec2, wp.vec2, int, float]:
    """Advance one anchored Coulomb bristle without mutating caller-owned state.

    Positions and anchor are [m], velocity [m/s], time and dwell [s], normal
    force [N], stiffness [N/m], damping [N·s/m]. Return tangential force [N],
    new anchor, grip flag and unloaded dwell. Float state can live in separate
    timestep buffers for autodiff; branches have the same meaning in both paths.
    """
    f_max = mu * normal
    force = wp.vec2(0.0, 0.0)
    next_anchor = anchor
    next_stuck = stuck
    next_dwell = dwell
    if normal <= 0.0 or kt <= 0.0:
        next_dwell = dwell + dt
        if kt <= 0.0 or stuck == 0 or next_dwell > release_dwell:
            next_anchor = position
            next_stuck = 0
            next_dwell = 0.0
    else:
        next_dwell = 0.0
        if stuck == 0:
            next_anchor = position
            next_stuck = 1
        # Implicit elastic trial avoids ringing in the stiff stick regime.
        p_next = position + velocity * dt
        elastic = -kt * (p_next - next_anchor)
        mag = wp.length(elastic)
        if mag > f_max:
            elastic = elastic * (f_max / wp.max(mag, 1.0e-12))
            next_anchor = p_next + elastic / kt
        force = elastic
        speed = wp.length(velocity)
        if kv > 0.0 and speed > 1.0e-12:
            viscous = -velocity * (wp.min(kv * speed, viscous_ratio * f_max) / speed)
            force = elastic + viscous * cone_viscous_scale(elastic, viscous, f_max)
    return force, next_anchor, next_stuck, next_dwell


@wp.func
def contact_kinematics(
    transform: wp.transform,
    velocity: wp.spatial_vector,
    com_local: wp.vec3,
    anchor_local: wp.vec3,
    ground_height: float,
    ground_plane: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
    """Return contact point, COM, point velocity and nominal plane clearance.

    Points and clearance are [m], velocity [m/s]. A bench top anchor remains a
    top anchor; a carried outsole uses its projection onto the declared plane.
    """
    world = wp.transform_point(transform, anchor_local)
    com = wp.transform_point(transform, com_local)
    point = world
    if ground_plane != 0:
        point = wp.vec3(world[0], world[1], ground_height)
    point_velocity = wp.spatial_top(velocity) + wp.cross(wp.spatial_bottom(velocity), point - com)
    return point, com, point_velocity, world[2] - ground_height


# Top connection of a two-surface column (see :func:`column_balance_pressures`).
TOP_BONDED = wp.constant(0)
TOP_PRESSED = wp.constant(1)
TOP_FREE = wp.constant(2)
# Support state bits of a balanced two-surface column.
TOP_HELD = wp.constant(1)
BOTTOM_LANDED = wp.constant(2)


@wp.func
def _hessian_product(foam: float, shear: float, v: wp.vec2) -> wp.vec2:
    """Return the local model's Hessian times ``v``.

    The Hessian is ``foam [[1, -1], [-1, 1]] + shear I``: the foam stiffens only
    the compression ``t - b`` and the face layers stiffen both ends alike.
    Keeping the two parts apart avoids the cancellation of ``foam + shear - foam``.
    """
    return wp.vec2(1.0, -1.0) * (foam * (v[0] - v[1])) + v * shear


@wp.func
def _model_value(grad: wp.vec2, foam: float, shear: float, d: wp.vec2) -> float:
    """Return the local quadratic model of the column energy for a step ``d``."""
    return wp.dot(grad, d) + 0.5 * wp.dot(d, _hessian_product(foam, shear, d))


@wp.func
def _line_minimum(y0: wp.vec2, grad: wp.vec2, foam: float, shear: float, n: wp.vec2, beta: float) -> wp.vec2:
    """Minimize the local model on the line ``n . y = beta``."""
    p = n * (beta / wp.dot(n, n))
    d = wp.vec2(-n[1], n[0])
    hd = _hessian_product(foam, shear, d)
    s = -(wp.dot(d, grad) + wp.dot(hd, p - y0)) / wp.dot(d, hd)
    return p + d * s


@wp.func
def _vertex(na: wp.vec2, ba: float, nb: wp.vec2, bb: float) -> tuple[wp.vec2, int]:
    """Intersect two constraint lines; the flag is zero for parallel lines."""
    det = na[0] * nb[1] - na[1] * nb[0]
    if wp.abs(det) < 1.0e-12:
        return wp.vec2(0.0, 0.0), 0
    return wp.vec2((ba * nb[1] - na[1] * bb) / det, (na[0] * bb - ba * nb[0]) / det), 1


@wp.func
def _keep_better(
    y: wp.vec2,
    valid: int,
    y0: wp.vec2,
    grad: wp.vec2,
    foam: float,
    shear: float,
    n0: wp.vec2,
    b0: float,
    n1: wp.vec2,
    b1: float,
    n2: wp.vec2,
    b2: float,
    tol: float,
    best: wp.vec2,
    best_value: float,
) -> tuple[wp.vec2, float]:
    """Keep a feasible candidate when it lowers the local model."""
    if valid == 0:
        return best, best_value
    if wp.dot(n0, y) > b0 + tol or wp.dot(n1, y) > b1 + tol or wp.dot(n2, y) > b2 + tol:
        return best, best_value
    value = _model_value(grad, foam, shear, y - y0)
    if value < best_value:
        return y, value
    return best, best_value


@wp.func
def column_balance_pressures(
    top: float,
    bottom: float,
    connection: int,
    top_bound: float,
    clearance: float,
    compression_limit: float,
    coupling_sum: float,
    top_pull: float,
    bottom_pull: float,
    thickness: float,
    overstress_base: float,
    overstress_gain: float,
    area: float,
    peq: float,
    peq_ahead: float,
    relaxation: float,
) -> tuple[float, float, int]:
    """Take one projected Newton step of a two-surface column with evaluated pressures.

    ``top`` and ``bottom`` are the downward displacements ``t`` and ``b`` [m] of
    the column ends from the rigidly carried rest shape; the compression is
    ``c = t - b``. The column energy is its foam energy ``W(c)`` plus the
    top- and bottom-face Pasternak layers ``1/2 sum_j k_ij (dt^2 + db^2)`` to
    neighbours held fixed for this step, with ``dt = t - t_j`` and
    ``db = b - b_j``. With every bottom fixed this is
    :func:`_surround_balance_pressures`' ``1/2 k dc^2``.

    Constraints: ``b <= clearance`` (the plane), ``c >= 0``, and by connection
    ``t = top_bound`` (:data:`TOP_BONDED`), ``t >= top_bound``
    (:data:`TOP_PRESSED`), or ``c <= compression_limit`` (:data:`TOP_FREE`).
    A :data:`TOP_BONDED` column keeps ``c <= compression_limit`` by pushing its
    bottom down, unless the plane is in the way.
    The local quadratic model is minimized exactly over these constraints by
    enumerating its active sets.

    Args:
        top_pull: ``sum_j k_ij (t - t_j)`` [N], summed as differences so that
            stiff layers keep their precision.
        bottom_pull: ``sum_j k_ij (b - b_j)`` [N].
        coupling_sum: ``sum_j k_ij`` [N/m].
        overstress_base: Maxwell overstress at zero equilibrium pressure [Pa].
        overstress_gain: Overstress per unit equilibrium pressure of this step.
        peq: Equilibrium pressure at ``c`` [Pa].
        peq_ahead: Equilibrium pressure at ``c + 1e-3 * thickness`` [Pa].
        relaxation: Fraction of the constrained step taken before the result is
            projected back onto the constraints; above one over-relaxes.

    Returns:
        Updated ``t`` and ``b`` [m] and the support bits :data:`TOP_HELD` and
        :data:`BOTTOM_LANDED`. Held and landed ends sit exactly on their bounds.
    """
    force = area * wp.max(peq + overstress_base + overstress_gain * peq, 0.0)
    step = 1.0e-3 * thickness
    ahead = area * wp.max(peq_ahead + overstress_base + overstress_gain * peq_ahead, 0.0)
    foam = wp.max((ahead - force) / step, 0.0)
    k = coupling_sum
    grad = wp.vec2(force + top_pull, -force + bottom_pull)
    shear = k + 1.0e-9
    y0 = wp.vec2(top, bottom)
    tol = 1.0e-6 * thickness
    t = top
    b = bottom
    if connection == TOP_BONDED:
        t = top_bound
        b = bottom - relaxation * (grad[1] - foam * (t - top)) / (foam + shear)
        # The compression stop pushes the bottom down, but the plane comes first.
        b = wp.min(wp.max(b, t - compression_limit), wp.min(clearance, t))
    else:
        # Constraints n . (t, b) <= beta: the plane, c >= 0, and the top bound or the strain limit.
        n0 = wp.vec2(0.0, 1.0)
        b0 = clearance
        n1 = wp.vec2(-1.0, 1.0)
        b1 = float(0.0)
        n2 = wp.vec2(-1.0, 0.0)
        b2 = -top_bound
        if connection == TOP_FREE:
            n2 = wp.vec2(1.0, -1.0)
            b2 = compression_limit
        # Newton step along the Hessian's eigenvectors: the rigid shift (1, 1)
        # with curvature shear and the squeeze (1, -1) with 2 foam + shear.
        rigid = (grad[0] + grad[1]) / (2.0 * shear)
        squeeze = (grad[0] - grad[1]) / (2.0 * (2.0 * foam + shear))
        free = y0 - wp.vec2(rigid + squeeze, rigid - squeeze)
        best = y0
        best_value = float(1.0e30)
        best, best_value = _keep_better(free, 1, y0, grad, foam, shear, n0, b0, n1, b1, n2, b2, tol, best, best_value)
        best, best_value = _keep_better(
            _line_minimum(y0, grad, foam, shear, n0, b0),
            1,
            y0,
            grad,
            foam,
            shear,
            n0,
            b0,
            n1,
            b1,
            n2,
            b2,
            tol,
            best,
            best_value,
        )
        best, best_value = _keep_better(
            _line_minimum(y0, grad, foam, shear, n1, b1),
            1,
            y0,
            grad,
            foam,
            shear,
            n0,
            b0,
            n1,
            b1,
            n2,
            b2,
            tol,
            best,
            best_value,
        )
        best, best_value = _keep_better(
            _line_minimum(y0, grad, foam, shear, n2, b2),
            1,
            y0,
            grad,
            foam,
            shear,
            n0,
            b0,
            n1,
            b1,
            n2,
            b2,
            tol,
            best,
            best_value,
        )
        y, valid = _vertex(n0, b0, n1, b1)
        best, best_value = _keep_better(y, valid, y0, grad, foam, shear, n0, b0, n1, b1, n2, b2, tol, best, best_value)
        y, valid = _vertex(n0, b0, n2, b2)
        best, best_value = _keep_better(y, valid, y0, grad, foam, shear, n0, b0, n1, b1, n2, b2, tol, best, best_value)
        y, valid = _vertex(n1, b1, n2, b2)
        best, best_value = _keep_better(y, valid, y0, grad, foam, shear, n0, b0, n1, b1, n2, b2, tol, best, best_value)
        t = top + relaxation * (best[0] - top)
        b = bottom + relaxation * (best[1] - bottom)
        if connection == TOP_PRESSED and t <= top_bound + tol:
            t = top_bound
    # Close at most half the remaining thickness per update, so a far-off step
    # cannot jump into the foam's lock-up; a converged state is unaffected.
    room = wp.max(0.5 * (thickness - (top - bottom)), 0.0)
    if (t - b) - (top - bottom) > room:
        if connection == TOP_BONDED:
            b = t - (top - bottom) - room
        else:
            s = room / ((t - b) - (top - bottom))
            t = top + s * (t - top)
            b = bottom + s * (b - bottom)
    if b >= clearance - tol:
        b = clearance
    if t < b:
        if connection == TOP_FREE:
            t = b
        else:
            b = t
    if connection == TOP_FREE and t - b > compression_limit:
        t = b + compression_limit
    support = int(0)
    if connection == TOP_BONDED or (connection == TOP_PRESSED and t == top_bound):
        support = support | TOP_HELD
    if b == clearance:
        support = support | BOTTOM_LANDED
    return t, b, support
