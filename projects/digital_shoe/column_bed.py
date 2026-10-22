# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-surface midsole column bed on the GPU: one kernel launch per step.

Each column has a top and a bottom displacement, positive down from the rest
shape the carrier carries rigidly; its compression is ``c = t - b``. The bottom
stands on the ground plane (``b <= h``, the bottom clearance), and the top is
connected to the carrier per column:

* :data:`TOP_BONDED`: the top rides with the carrier (a foot glued to the sole).
* :data:`TOP_PRESSED`: the carrier surface, ``top_gap_m`` above the rest top,
  pushes the top down but never pulls it (a last pressed into the sole).
* :data:`TOP_FREE`: nothing touches the top.

The bed minimises the Hyperfoam/Maxwell foam energy of every column plus two
Pasternak shear layers, ``1/2 sum k_ij (dt^2 + db^2)`` over faces, one at the
tops and one at the bottoms (:func:`~projects.digital_shoe.contact.column_balance_pressures`).
Each column's layers shear with :func:`~projects.digital_shoe.runtime.shear_layer_modulus`
at its current compression, so shear follows the axial tangent, and each
column carries the derivative of its layers' energy over its own compression
so that the layer forces stay conservative.
Held tops load the carrier with ``T``, landed bottoms load the plane with
``R``. Columns are one-dimensional: a column bottom moves sideways with the
carrier, so friction acts between the carrier and the plane. A landed bottom
sticks to the plane or slides at ``mu R`` (Coulomb).

One CUDA block of 256 lanes advances one world: it keeps the bed in shared
memory, runs the red-black over-relaxed sweeps, the Maxwell update, the column
reactions, Coulomb friction and the carrier wrench in one launch, so a
step costs one launch however many sweeps it takes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from .contact import (
    BOTTOM_LANDED,
    TOP_BONDED,
    TOP_FREE,
    TOP_HELD,
    TOP_PRESSED,
    column_balance_pressures,
    contact_kinematics,
    pasternak_coupling,
)
from .material import maxwell_coefficients, maxwell_step
from .runtime import (
    FoundationConfig,
    FoundationParams,
    _hyperfoam_pressure,
    material_shear_modulus,
    set_material_block,
    shear_layer_modulus,
    shear_layer_modulus_between,
)

__all__ = [
    "BOTTOM_LANDED",
    "MAX_COLUMNS",
    "TOP_BONDED",
    "TOP_FREE",
    "TOP_HELD",
    "TOP_PRESSED",
    "ColumnBed",
    "column_colors",
]

_CARRIER_PASSES = wp.constant(4)
_FRICTION_TURN = wp.constant(0.1)
"""Change of a sliding bottom's friction, as a fraction of its capacity, that repeats the carrier solve."""
_STICK_SPEED = wp.constant(1.0e-4)
"""Speed at which a sticking column's friction reaches its ground reaction [m/s].

A sticking bottom resists sliding as a damper of ``R / _STICK_SPEED``, so it
creeps at most ``mu * _STICK_SPEED`` before it slides. This is a solver
constant, not a material property."""
_STOP_MODULUS = wp.constant(1.0e13)
"""Modulus of a column at its compression stop [Pa].

A column at the stop is rigid to the implicit carrier: this makes it a hundred
times stiffer than any carrier it can meet. It is a solver constant, not a
material property."""
_BLOCK = wp.constant(256)
_SWEEP_TOLERANCE = wp.constant(1.0e-6)
"""Largest end move of a converged sweep, as a fraction of the column thickness."""
MAX_COLUMNS = 1024
"""Columns one block holds in shared memory."""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float column_top_buffer[1024];
if (write) column_top_buffer[index] = value;
return column_top_buffer[index];
#else
return value;
#endif
""")
def _top_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float column_bottom_buffer[1024];
if (write) column_bottom_buffer[index] = value;
return column_bottom_buffer[index];
#else
return value;
#endif
""")
def _bottom_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ int column_changed_buffer[3];
if (write) column_changed_buffer[index] = value;
return column_changed_buffer[index];
#else
return value;
#endif
""")
def _changed_shared(index: int, value: int, write: int) -> int: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float column_clearance_buffer[1024];
if (write) column_clearance_buffer[index] = value;
return column_clearance_buffer[index];
#else
return value;
#endif
""")
def _clearance_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float column_overstress_buffer[1024];
if (write) column_overstress_buffer[index] = value;
return column_overstress_buffer[index];
#else
return value;
#endif
""")
def _overstress_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ int column_support_buffer[1024];
if (write) column_support_buffer[index] = value;
return column_support_buffer[index];
#else
return value;
#endif
""")
def _support_shared(index: int, value: int, write: int) -> int: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float column_modulus_buffer[1024];
if (write) column_modulus_buffer[index] = value;
return column_modulus_buffer[index];
#else
return value;
#endif
""")
def _modulus_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float column_modulus_slope_buffer[1024];
if (write) column_modulus_slope_buffer[index] = value;
return column_modulus_slope_buffer[index];
#else
return value;
#endif
""")
def _modulus_slope_shared(index: int, value: float, write: int) -> float: ...


@wp.struct
class _Bed:
    column_count: int
    carrier: wp.array[wp.int32]
    body_com: wp.array[wp.vec3]
    implicit: int
    carrier_mobility: wp.array[wp.spatial_matrix]
    anchor_local: wp.array[wp.vec3]
    rest_len: wp.array[wp.float32]
    area: wp.array[wp.float32]
    neighbors: wp.array2d[wp.int32]
    order: wp.array[wp.int32]
    color_split: int
    connection: wp.array[wp.int32]
    top_gap: wp.array[wp.float32]
    ground_height: float
    max_strain: float
    over_relaxation: float
    sweeps: int
    params: wp.array[FoundationParams]
    dt: wp.array[wp.float32]
    top: wp.array[wp.float32]
    bottom: wp.array[wp.float32]
    q_state: wp.array[wp.float32]
    peq_prev: wp.array[wp.float32]
    stuck: wp.array[wp.int32]
    compression: wp.array[wp.float32]
    carried: wp.array[wp.float32]
    reaction: wp.array[wp.float32]
    support: wp.array[wp.int32]
    normal_force: wp.array[wp.float32]
    carried_force: wp.array[wp.float32]
    pressure_center: wp.array[wp.vec2]
    held_strain: wp.array[wp.float32]
    shear_modulus: wp.array[wp.float32]
    stiffness: wp.array[wp.float32]
    lever: wp.array[wp.vec3]
    tangential_stiffness: wp.array[wp.float32]
    tangential_load: wp.array[wp.vec2]
    stop_gap: wp.array[wp.float32]
    stop_stiffness: wp.array[wp.float32]


_STRAIN_STEP = wp.constant(1.0e-3)
"""Strain step of the finite differences of the foam and shear-layer laws."""


@wp.func
def _set_modulus(column: int, compression: float, thickness: float, rest: float, p: FoundationParams):
    """Store a column's shear-layer modulus and its slope over strain at ``compression``.

    ``rest`` is the equilibrium pressure one strain step from rest.
    """
    strain = wp.max(compression, 0.0) / thickness
    at = _hyperfoam_pressure(strain, p)
    modulus = material_shear_modulus(p)
    if strain >= _STRAIN_STEP:
        modulus = shear_layer_modulus_between(_hyperfoam_pressure(strain - _STRAIN_STEP, p), at, rest, p)
    ahead = shear_layer_modulus_between(at, _hyperfoam_pressure(strain + _STRAIN_STEP, p), rest, p)
    _modulus_shared(column, modulus, 1)
    _modulus_slope_shared(column, (ahead - modulus) / _STRAIN_STEP, 1)


@wp.func
def _face_coupling(bed: _Bed, column: int, j: int) -> float:
    """Return the shear conductance between two columns of one world [N/m]."""
    return pasternak_coupling(
        bed.rest_len[column], bed.rest_len[j], _modulus_shared(column, 0.0, 0), _modulus_shared(j, 0.0, 0)
    )


@wp.func
def _face_sums(bed: _Bed, column: int, top: float, bottom: float) -> tuple[float, float, float]:
    """Return ``sum k_ij``, ``sum k_ij (t - t_j)`` and ``sum k_ij (b - b_j)`` over a column's faces."""
    coupling_sum = float(0.0)
    top_pull = float(0.0)
    bottom_pull = float(0.0)
    for side in range(4):
        j = bed.neighbors[column, side]
        if j >= 0:
            k = _face_coupling(bed, column, j)
            coupling_sum += k
            top_pull += k * (top - _top_shared(j, 0.0, 0))
            bottom_pull += k * (bottom - _bottom_shared(j, 0.0, 0))
    return coupling_sum, top_pull, bottom_pull


@wp.func
def _layer_squeeze(bed: _Bed, column: int, top: float, bottom: float, modulus: float, slope: float) -> float:
    """Return the compression force of a column's shear layers [N].

    The layers store ``1/2 sum_j k_ij (dt^2 + db^2)`` with ``k_ij`` following
    both columns' compression. Its derivative over this column's compression,
    ``1/2 sum_j dk_ij/dc (dt^2 + db^2)``, is the force that makes the layer
    forces conservative: without it a load and unload cycle on the
    densification wall returns more work than it took. ``modulus`` and
    ``slope`` are the column's shear modulus [Pa] and its slope over strain [Pa].
    """
    own = modulus * bed.rest_len[column]
    total = float(0.0)
    for side in range(4):
        j = bed.neighbors[column, side]
        if j >= 0:
            other = _modulus_shared(j, 0.0, 0) * bed.rest_len[j]
            share = other / wp.max(own + other, 1.0e-30)
            d_top = top - _top_shared(j, 0.0, 0)
            d_bottom = bottom - _bottom_shared(j, 0.0, 0)
            total += share * share * slope * (d_top * d_top + d_bottom * d_bottom)
    return total


@wp.func
def _coulomb(velocity: wp.vec2, reaction: float, mu: float) -> tuple[wp.vec2, int]:
    """Return the friction on a bottom moving at ``velocity`` over the plane and whether it sticks."""
    if reaction <= 0.0 or mu <= 0.0:
        return wp.vec2(0.0), 0
    force = -(reaction / _STICK_SPEED) * velocity
    capacity = mu * reaction
    if wp.length(force) > capacity:
        return force * (capacity / wp.length(force)), 0
    return force, 1


@wp.func
def _block_sum(value: float) -> float:
    """Sum one value over the block in a fixed order."""
    return wp.tile_sum(wp.tile(value))[0]


_Packed = wp.types.vector(length=27, dtype=wp.float32)
_Mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_Vec6d = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def _cholesky6(a: _Mat66d) -> _Mat66d:
    """Return the lower Cholesky factor of a symmetric positive definite matrix."""
    low = _Mat66d()
    for j in range(6):
        d = a[j, j]
        for m in range(j):
            d -= low[j, m] * low[j, m]
        d = wp.sqrt(d)
        low[j, j] = d
        for i in range(j + 1, 6):
            v = a[i, j]
            for m in range(j):
                v -= low[i, m] * low[j, m]
            low[i, j] = v / d
    return low


@wp.func
def _cholesky6_solve(low: _Mat66d, b: _Vec6d) -> _Vec6d:
    """Solve ``L L^T x = b`` for the factor from :func:`_cholesky6`."""
    y = _Vec6d()
    for i in range(6):
        v = b[i]
        for m in range(i):
            v -= low[i, m] * y[m]
        y[i] = v / low[i, i]
    x = _Vec6d()
    for k in range(6):
        i = 5 - k
        v = y[i]
        for m in range(i + 1, 6):
            v -= low[m, i] * x[m]
        x[i] = v / low[i, i]
    return x


@wp.func
def _row(n: wp.vec3, r: wp.vec3) -> wp.spatial_vector:
    """Return the carrier twist row of a force direction ``n`` acting at lever ``r``."""
    return wp.spatial_vector(n, wp.cross(r, n))


@wp.func
def _row_speed(u: _Vec6d, n: wp.vec3, r: wp.vec3) -> float:
    """Return the speed along ``n`` of the carrier point at lever ``r`` [m/s]."""
    g = _row(n, r)
    s = wp.float64(0.0)
    for k in range(6):
        s += wp.float64(g[k]) * u[k]
    return wp.float32(s)


@wp.func
def _carrier_velocity(mass: _Mat66d, u0: _Vec6d, stiff: _Mat66d, load: _Vec6d, dt: float) -> _Vec6d:
    """Solve (M + dt A) u = M u0 + dt w for the end-of-step carrier twist.

    ``A`` is the bed stiffness times dt over the carrier's (v, w) at its center
    of mass. On the densification wall ``A`` spans many decades, so the solve
    runs in double precision on a unit diagonal.
    """
    step = wp.float64(dt)
    system = mass + step * stiff
    rhs = mass * u0 + step * load
    scale = _Vec6d()
    for row in range(6):
        scale[row] = wp.float64(1.0) / wp.sqrt(system[row, row])
    for row in range(6):
        for col in range(6):
            system[row, col] = scale[row] * system[row, col] * scale[col]
        rhs[row] = scale[row] * rhs[row]
    return wp.cw_mul(scale, _cholesky6_solve(_cholesky6(system), rhs))


def _column_bed_step(
    bed: _Bed,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    """Advance one world's bed by one step and write its carrier wrench."""
    w, lane = wp.tid()
    count = bed.column_count
    base = w * count
    p = bed.params[w]
    dt = bed.dt[w]
    decay, ramp = maxwell_coefficients(dt, p.tau_s)
    gain = p.overstress * ramp
    rest = _hyperfoam_pressure(_STRAIN_STEP, p)
    body = bed.carrier[w]
    pose = body_q[body]
    # The bed lives in shared memory; loop bounds stay runtime values so the
    # column solve is compiled once rather than unrolled per row and color.
    for slot in range(lane, count, _BLOCK):
        column = bed.order[slot]
        i = base + column
        _top_shared(column, bed.top[i], 1)
        _bottom_shared(column, bed.bottom[i], 1)
        _clearance_shared(column, wp.transform_point(pose, bed.anchor_local[column])[2] - bed.ground_height, 1)
        _overstress_shared(column, decay * bed.q_state[i] - gain * bed.peq_prev[i], 1)
        # The shear layers follow each column's compression, updated with it below.
        _set_modulus(column, bed.top[i] - bed.bottom[i], bed.rest_len[column], rest, p)
    _sync_threads()
    # Red-black Gauss-Seidel: lanes hold columns sorted by color and a color
    # reads only the other color, so each half sweep updates in place and then
    # synchronizes once. A sweep that moves no column end by more than
    # _SWEEP_TOLERANCE of its thickness ends the solve. The flag of
    # sweep s + 1 is cleared during sweep s; its last reader finished two
    # synchronizations earlier.
    if lane < 3:
        _changed_shared(lane, 0, 1)
    _sync_threads()
    for half in range(2 * bed.sweeps):
        sweep = half / 2
        start = 0
        end = bed.color_split
        if half % 2 == 0:
            if lane == 0:
                _changed_shared((sweep + 1) % 3, 0, 1)
        else:
            start = bed.color_split
            end = count
        for slot in range(start + lane, end, _BLOCK):
            column = bed.order[slot]
            thickness = bed.rest_len[column]
            top = _top_shared(column, 0.0, 0)
            bottom = _bottom_shared(column, 0.0, 0)
            coupling_sum, top_pull, bottom_pull = _face_sums(bed, column, top, bottom)
            c = top - bottom
            # The layer squeeze enters as foam pressure, at c and one strain step ahead.
            pressure = _hyperfoam_pressure(c / thickness, p)
            pressure_ahead = _hyperfoam_pressure((c + _STRAIN_STEP * thickness) / thickness, p)
            modulus = _modulus_shared(column, 0.0, 0)
            slope = _modulus_slope_shared(column, 0.0, 0)
            near = pressure_ahead
            if c < 0.0:
                near = rest
            far = _hyperfoam_pressure(wp.max(c, 0.0) / thickness + 2.0 * _STRAIN_STEP, p)
            slope_ahead = (shear_layer_modulus_between(near, far, rest, p) - modulus) / _STRAIN_STEP - slope
            per_pressure = 1.0 / (bed.area[column] * (1.0 + gain))
            squeeze = per_pressure * _layer_squeeze(bed, column, top, bottom, modulus, slope)
            squeeze_ahead = per_pressure * _layer_squeeze(
                bed, column, top, bottom, modulus + _STRAIN_STEP * slope, slope_ahead
            )
            t, b, state = column_balance_pressures(
                top,
                bottom,
                bed.connection[column],
                -bed.top_gap[column],
                _clearance_shared(column, 0.0, 0),
                bed.max_strain * thickness,
                coupling_sum,
                top_pull,
                bottom_pull,
                thickness,
                _overstress_shared(column, 0.0, 0),
                gain,
                bed.area[column],
                pressure + squeeze,
                pressure_ahead + squeeze_ahead,
                bed.over_relaxation,
            )
            if wp.abs(t - top) > _SWEEP_TOLERANCE * thickness or wp.abs(b - bottom) > _SWEEP_TOLERANCE * thickness:
                _changed_shared(sweep % 3, 1, 1)
                _set_modulus(column, t - b, thickness, rest, p)
            _top_shared(column, t, 1)
            _bottom_shared(column, b, 1)
            _support_shared(column, state, 1)
        _sync_threads()
        if half % 2 == 1 and _changed_shared(sweep % 3, 0, 0) == 0:
            break

    twist = body_qd[body]
    com_local = bed.body_com[body]
    force_sum = wp.vec3(0.0)
    torque_sum = wp.vec3(0.0)
    normal_sum = float(0.0)
    pressing_sum = float(0.0)
    carried_sum = float(0.0)
    moment_sum = wp.vec2(0.0)
    largest = float(0.0)
    # Load on the carrier from the columns outside its implicit rows.
    free_load = wp.spatial_vector()
    for slot in range(lane, count, _BLOCK):
        column = bed.order[slot]
        i = base + column
        thickness = bed.rest_len[column]
        area = bed.area[column]
        top = _top_shared(column, 0.0, 0)
        bottom = _bottom_shared(column, 0.0, 0)
        support = _support_shared(column, 0, 0)
        comp = wp.max(top - bottom, 0.0)
        peq = _hyperfoam_pressure(comp / thickness, p)
        q = maxwell_step(bed.q_state[i], peq, bed.peq_prev[i], p.overstress, decay, ramp)
        if comp == 0.0 or (support & BOTTOM_LANDED) == 0:
            # A column that is uncompressed or off the ground carries no axial
            # stress, so none is left over for when it is pressed again.
            q = 0.0
        bed.q_state[i] = q
        bed.peq_prev[i] = peq
        # Shear on each end, positive down, from the balanced neighbours.
        on_top = float(0.0)
        on_bottom = float(0.0)
        for side in range(4):
            j = bed.neighbors[column, side]
            if j >= 0:
                k = _face_coupling(bed, column, j)
                on_top += k * (top - _top_shared(j, 0.0, 0))
                on_bottom += k * (bottom - _bottom_shared(j, 0.0, 0))
        point, com, velocity, _gap = contact_kinematics(
            pose, twist, com_local, bed.anchor_local[column], bed.ground_height, 1
        )
        held = (support & TOP_HELD) != 0
        landed = (support & BOTTOM_LANDED) != 0
        # Axial column force. A column free at one end passes on exactly the shear
        # that balances that end, including the reaction of its compression bounds.
        modulus = _modulus_shared(column, 0.0, 0)
        slope = _modulus_slope_shared(column, 0.0, 0)
        bed.shear_modulus[i] = modulus
        squeeze = _layer_squeeze(bed, column, top, bottom, modulus, slope)
        # Clipped once, after the squeeze, as in the column balance.
        axial = wp.max(area * (peq + q) + squeeze, 0.0)
        if held and landed:
            if comp > 0.0:
                axial = wp.max(axial - p.normal_damping * velocity[2], 0.0)
        elif held:
            axial = on_bottom
        elif landed:
            axial = -on_top
        bed.stiffness[i] = 0.0
        bed.stop_gap[i] = 1.0e30
        bed.stop_stiffness[i] = 0.0
        if bed.implicit != 0 and held and landed:
            # Room left to the stop. A carrier already past it is held there, not
            # pushed back: the foam pushes it out, so the stop adds no energy.
            bed.stop_gap[i] = wp.max(bed.max_strain * thickness - comp, 0.0)
        if bed.implicit != 0 and held and comp > 0.0:
            # Axial tangent of the foam, by a backward difference so it never
            # steps onto the stretch floor.
            h = _STRAIN_STEP * thickness
            foam = area * (1.0 + gain) * (peq - _hyperfoam_pressure((comp - h) / thickness, p)) / h
            # Forward difference of the layer squeeze, as in the column balance.
            strain_ahead = comp / thickness + 2.0 * _STRAIN_STEP
            slope_ahead = (shear_layer_modulus(strain_ahead, p) - modulus) / _STRAIN_STEP - slope
            ahead = _layer_squeeze(bed, column, top, bottom, modulus + _STRAIN_STEP * slope, slope_ahead)
            foam += (ahead - squeeze) / h
            if not landed:
                # A floating bottom reaches the ground only through the shear
                # layer to its landed neighbours, in series with the foam.
                anchored = float(0.0)
                for side in range(4):
                    j = bed.neighbors[column, side]
                    if j >= 0 and (_support_shared(j, 0, 0) & BOTTOM_LANDED) != 0:
                        anchored += _face_coupling(bed, column, j)
                foam = foam * anchored / (foam + anchored + 1.0e-12)
            # The shear layer to a landed neighbour with a free top acts in series
            # with that neighbour's foam, or alone once it is pinned at its
            # compression limit.
            for side in range(4):
                j = bed.neighbors[column, side]
                if j >= 0 and (_support_shared(j, 0, 0) & (TOP_HELD | BOTTOM_LANDED)) == BOTTOM_LANDED:
                    layer = _face_coupling(bed, column, j)
                    t_j = bed.rest_len[j]
                    c_j = wp.max(_top_shared(j, 0.0, 0) - _bottom_shared(j, 0.0, 0), 0.0)
                    if c_j >= bed.max_strain * t_j * (1.0 - 1.0e-4):
                        foam += layer
                    else:
                        h_j = _STRAIN_STEP * t_j
                        under = (
                            bed.area[j]
                            * (1.0 + gain)
                            * (_hyperfoam_pressure(c_j / t_j, p) - _hyperfoam_pressure((c_j - h_j) / t_j, p))
                            / h_j
                        )
                        under = wp.max(under, 0.0)
                        foam += layer * under / (layer + under + 1.0e-12)
            bed.stiffness[i] = dt * wp.max(foam, 0.0)
        carried = float(0.0)
        if held:
            carried = axial + on_top
            largest = wp.max(largest, comp / thickness)
        reaction = float(0.0)
        if landed:
            reaction = axial - on_bottom
        tangential = wp.vec2(0.0)
        if bed.implicit != 0:
            # Friction is solved with the carrier below: a landed bottom starts
            # out sticking, as a damper of R / _STICK_SPEED.
            bed.lever[i] = point - com
            bed.tangential_stiffness[i] = 0.0
            bed.tangential_load[i] = wp.vec2(0.0)
            if reaction > 0.0 and p.mu > 0.0:
                bed.tangential_stiffness[i] = reaction / _STICK_SPEED
        else:
            tangential, sticking = _coulomb(wp.vec2(velocity[0], velocity[1]), reaction, p.mu)
            bed.stuck[i] = sticking
        bed.top[i] = top
        bed.bottom[i] = bottom
        bed.compression[i] = comp
        bed.carried[i] = carried
        bed.reaction[i] = reaction
        bed.support[i] = support
        force = wp.vec3(tangential[0], tangential[1], carried)
        force_sum += force
        torque_sum += wp.cross(point - com, force)
        if bed.stiffness[i] == 0.0:
            free_load += _row(wp.vec3(0.0, 0.0, 1.0), point - com) * carried
        normal_sum += reaction
        pressing_sum += wp.max(reaction, 0.0)
        carried_sum += carried
        moment_sum += wp.vec2(point[0], point[1]) * reaction
    force_total = wp.vec3(_block_sum(force_sum[0]), _block_sum(force_sum[1]), _block_sum(force_sum[2]))
    torque_total = wp.vec3(_block_sum(torque_sum[0]), _block_sum(torque_sum[1]), _block_sum(torque_sum[2]))
    normal_total = _block_sum(normal_sum)
    pressing_total = _block_sum(pressing_sum)
    carried_total = _block_sum(carried_sum)
    moment_total = wp.vec2(_block_sum(moment_sum[0]), _block_sum(moment_sum[1]))
    largest_total = wp.tile_max(wp.tile(largest))[0]
    if bed.implicit != 0:
        # Linearized backward Euler of the carrier on the bed: the column forces
        # are taken at the end-of-step twist u of the carrier with the effective
        # mass of its articulation. Every lane solves the same 6x6 system, so
        # the active sets below need no broadcast.
        mobility = bed.carrier_mobility[w]
        inverse_mass = _Mat66d()
        for row in range(6):
            for col in range(6):
                inverse_mass[row, col] = wp.float64(mobility[row, col])
        factor = _cholesky6(inverse_mass)
        mass = _Mat66d()
        for col in range(6):
            unit = _Vec6d()
            unit[col] = wp.float64(1.0)
            mass_col = _cholesky6_solve(factor, unit)
            for row in range(6):
                mass[row, col] = mass_col[row]
        u0 = _Vec6d()
        free = _Vec6d()
        for axis in range(6):
            u0[axis] = wp.float64(twist[axis])
            free[axis] = wp.float64(_block_sum(free_load[axis]))
        ex = wp.vec3(1.0, 0.0, 0.0)
        ey = wp.vec3(0.0, 1.0, 0.0)
        ez = wp.vec3(0.0, 0.0, 1.0)
        u = u0
        # The ground holds the bottoms with at most the friction of the load the
        # carrier presses down with; an unconverged balance can leave pushing and
        # pulling bottoms whose reactions are decades larger.
        share = float(1.0)
        if pressing_total > carried_total:
            share = wp.max(carried_total, 0.0) / pressing_total
            for slot in range(lane, count, _BLOCK):
                i = base + bed.order[slot]
                bed.tangential_stiffness[i] = share * bed.tangential_stiffness[i]
        # The bed pushes and never pulls, and a bottom sticks only up to mu R: a
        # column whose end-of-step axial force would pull leaves the system, and
        # a bottom whose friction would exceed its capacity slides at capacity.
        # Sums are rebuilt from the active columns rather than reduced, since on
        # the wall a column carries many decades more than the result.
        for _pass in range(_CARRIER_PASSES):
            stiff_lane = wp.spatial_matrix()
            load_lane = wp.spatial_vector()
            for slot in range(lane, count, _BLOCK):
                i = base + bed.order[slot]
                r = bed.lever[i]
                s_col = bed.stiffness[i]
                if s_col > 0.0:
                    gz = _row(ez, r)
                    stiff_lane += s_col * wp.outer(gz, gz)
                    load_lane += bed.carried[i] * gz
                s_stop = bed.stop_stiffness[i]
                if s_stop > 0.0:
                    gs = _row(ez, r)
                    stiff_lane += s_stop * wp.outer(gs, gs)
                    load_lane -= (s_stop / dt) * bed.stop_gap[i] * gs
                gx = _row(ex, r)
                gy = _row(ey, r)
                s_t = bed.tangential_stiffness[i]
                if s_t > 0.0:
                    stiff_lane += s_t * (wp.outer(gx, gx) + wp.outer(gy, gy))
                pending = bed.tangential_load[i]
                load_lane += pending[0] * gx + pending[1] * gy
            # One block reduction for the 21 stiffness entries and the 6 loads.
            packed = _Packed()
            n = int(0)
            for row in range(6):
                packed[21 + row] = load_lane[row]
                for col in range(row, 6):
                    packed[n] = stiff_lane[row, col]
                    n += 1
            total = wp.tile_sum(wp.tile(packed, preserve_type=True))[0]
            stiff = _Mat66d()
            load = free
            n = int(0)
            for row in range(6):
                load[row] += wp.float64(total[21 + row])
                for col in range(row, 6):
                    stiff[row, col] = wp.float64(total[n])
                    stiff[col, row] = wp.float64(total[n])
                    n += 1
            u = _carrier_velocity(mass, u0, stiff, load, dt)
            dropped = float(0.0)
            for slot in range(lane, count, _BLOCK):
                column = bed.order[slot]
                i = base + column
                r = bed.lever[i]
                s_col = bed.stiffness[i]
                if s_col > 0.0 and s_col * _row_speed(u, ez, r) > bed.carried[i]:
                    bed.stiffness[i] = 0.0
                    dropped += 1.0
                # A column pushes the carrier back to the stop once its end-of-step
                # compression would pass it, and lets go as soon as it would pull.
                closing = -_row_speed(u, ez, r) * dt - bed.stop_gap[i]
                if bed.stop_stiffness[i] > 0.0 and closing < 0.0:
                    bed.stop_stiffness[i] = 0.0
                    dropped += 1.0
                elif bed.stop_stiffness[i] == 0.0 and closing > 0.0:
                    bed.stop_stiffness[i] = dt * bed.area[column] * _STOP_MODULUS / bed.rest_len[column]
                    dropped += 1.0
                s_t = bed.tangential_stiffness[i]
                slip = wp.vec2(_row_speed(u, ex, r), _row_speed(u, ey, r))
                capacity = wp.max(p.mu, 0.0) * share * bed.reaction[i]
                if s_t > 0.0:
                    trial = bed.tangential_load[i] - s_t * slip
                    if wp.length(trial) > capacity:
                        bed.tangential_stiffness[i] = 0.0
                        bed.tangential_load[i] = trial * (capacity / wp.length(trial))
                        dropped += 1.0
                elif capacity > 0.0:
                    # A sliding bottom follows the slip of this pass, and the
                    # solve repeats while its friction still turns.
                    friction, _sticking = _coulomb(slip, bed.reaction[i], p.mu)
                    if wp.length(friction - bed.tangential_load[i]) > _FRICTION_TURN * capacity:
                        dropped += 1.0
                    bed.tangential_load[i] = friction
            if _block_sum(dropped) == 0.0:
                break
        # Friction at the end-of-step speed of each bottom, for the stick flag and its report.
        for slot in range(lane, count, _BLOCK):
            i = base + bed.order[slot]
            r = bed.lever[i]
            traction, sticking = _coulomb(wp.vec2(_row_speed(u, ex, r), _row_speed(u, ey, r)), bed.reaction[i], p.mu)
            bed.tangential_load[i] = traction
            bed.stuck[i] = sticking
        # The wrench is the carrier's momentum change M (u - u0) / dt, not the
        # difference w - A u of two terms that are large on the wall. Applying
        # all six components keeps the work the carrier gets equal to what the
        # solve took out of it: a force or torque rebuilt from the end-of-step
        # friction no longer matches the friction the solve used and adds work.
        change = mass * (u - u0) / wp.float64(dt)
        # A carrier that no column carries gets no wrench, not the round-off of the solve.
        if carried_total > 0.0:
            lift = wp.max(wp.float32(change[2]), 0.0)
            # The ground reaction and its center follow the carried load.
            ratio = lift / carried_total
            normal_total = normal_total * ratio
            moment_total = moment_total * ratio
            carried_total = lift
            force_total = wp.vec3(wp.float32(change[0]), wp.float32(change[1]), lift)
            torque_total = wp.vec3(wp.float32(change[3]), wp.float32(change[4]), wp.float32(change[5]))
        else:
            force_total = wp.vec3(0.0)
            torque_total = wp.vec3(0.0)
    if lane == 0:
        body_f[body] = wp.spatial_vector(force_total, torque_total)
        bed.normal_force[w] = normal_total
        bed.carried_force[w] = carried_total
        center = wp.vec2(0.0)
        if normal_total > 0.0:
            center = moment_total / normal_total
        bed.pressure_center[w] = center
        bed.held_strain[w] = largest_total


# Fast math changes forces by about 1e-6 of the peak against a converged solve,
# far below the error of the fixed sweep count, and makes the step 1.6x faster.
_STEP = wp.kernel(_column_bed_step, module="unique", launch_bounds=(256, 2), module_options={"fast_math": True})


@wp.kernel
def _reset_worlds(
    mask: wp.array[bool],
    column_count: int,
    top: wp.array[wp.float32],
    bottom: wp.array[wp.float32],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
):
    """Clear the columns of the masked worlds."""
    i = wp.tid()
    if mask[i / column_count]:
        top[i] = 0.0
        bottom[i] = 0.0
        q_state[i] = 0.0
        peq_prev[i] = 0.0


def column_colors(neighbors: np.ndarray) -> np.ndarray:
    """Two-color a column grid so that no face joins two columns of one color."""
    neighbors = np.asarray(neighbors)
    color = np.full(len(neighbors), -1, np.int32)
    for start in range(len(neighbors)):
        if color[start] >= 0:
            continue
        color[start] = 0
        stack = [start]
        while stack:
            i = stack.pop()
            for j in neighbors[i]:
                if j < 0:
                    continue
                if color[j] < 0:
                    color[j] = 1 - color[i]
                    stack.append(j)
                elif color[j] == color[i]:
                    raise ValueError("Column neighbours do not form a two-colorable grid")
    return color


@dataclass
class ColumnBed:
    """A batch of two-surface column beds, one per world, stepped on the GPU.

    Args:
        anchor_local: Nominal bottom of each column in the carrier frame [m],
            shape ``[column_count, 3]``. Its height over the plane is the bottom
            clearance.
        rest_len: Rest thickness per column [m].
        area: Plan area per column [m^2].
        neighbors: Four face neighbours per column, ``-1`` for none.
        spacing_m: Grid spacing [m].
        connection: :data:`TOP_BONDED`, :data:`TOP_PRESSED` or :data:`TOP_FREE`
            per column.
        materials: One material for every world, or one per world.
        config: Contact settings: the plane ``ground_height_m``, ``mu``,
            ``normal_damping`` and ``stretch_floor``. Friction is Coulomb:
            a landed bottom sticks to the plane or slides at ``mu R``.
        carrier: Carrier body index per world.
        body_com: Center of mass per body in its frame [m].
        top_gap_m: Rest clearance between the carrier surface and each top [m];
            zero by default.
        carrier_mobility: Inverse effective mass of each world's carrier in
            ``(v, w)`` at its center of mass, in world axes, one
            :class:`warp.spatial_matrix` per world, kept current by the caller.
            With it, each carrier takes the bed's normal and friction forces at
            its end-of-step twist (linearized backward Euler), which stays
            stable on the foam's densification wall. Without, carriers see the
            explicit forces, as kinematically driven fixtures do.
        max_strain: Hard compression stop as a fraction of rest thickness. A
            free top never closes past it, a bonded column's bottom is pushed
            down to keep it, and a landed column at the stop is rigid to the
            implicit carrier against further compression.
        sweeps: Red-black sweeps per step, warm started from the previous step.
        over_relaxation: Fraction of each column's constrained Newton step taken;
            above one is projected successive over-relaxation.
        device: CUDA device.
    """

    anchor_local: np.ndarray
    rest_len: np.ndarray
    area: np.ndarray
    neighbors: np.ndarray
    spacing_m: float
    connection: np.ndarray
    materials: object
    config: FoundationConfig
    carrier: np.ndarray
    body_com: wp.array
    top_gap_m: np.ndarray | None = None
    carrier_mobility: wp.array | None = None
    max_strain: float = 0.95
    sweeps: int = 16
    over_relaxation: float = 1.8
    device: object = None

    def __post_init__(self) -> None:
        self.device = wp.get_device(self.device)
        if not self.device.is_cuda:
            raise RuntimeError("ColumnBed runs only on a CUDA device")
        anchors = np.asarray(self.anchor_local, np.float32)
        n = len(anchors)
        if not 0 < n <= MAX_COLUMNS:
            raise ValueError(f"ColumnBed holds 1 to {MAX_COLUMNS} columns, got {n}")
        if self.config.ground_height_m is None:
            raise ValueError("ColumnBed needs a ground plane (config.ground_height_m)")
        connection = np.asarray(self.connection, np.int32)
        if connection.shape != (n,) or not np.isin(connection, (TOP_BONDED, TOP_PRESSED, TOP_FREE)).all():
            raise ValueError("connection must give TOP_BONDED, TOP_PRESSED or TOP_FREE per column")
        gap = np.zeros(n) if self.top_gap_m is None else np.asarray(self.top_gap_m)
        if gap.shape != (n,) or not np.isfinite(gap).all():
            raise ValueError("top_gap_m must give one finite clearance per column")
        if not 0.0 < self.max_strain < 1.0 or self.sweeps < 1 or not 0.0 < self.over_relaxation < 2.0:
            raise ValueError(
                "max_strain and over_relaxation must lie inside (0, 1) and (0, 2), sweeps must be positive"
            )
        carrier = np.atleast_1d(np.asarray(self.carrier, np.int32))
        self.world_count = len(carrier)
        self.column_count = n
        d = self.device
        total = self.world_count * n
        bed = _Bed()
        bed.column_count = n
        bed.carrier = wp.array(carrier, dtype=wp.int32, device=d)
        bed.body_com = self.body_com
        bed.implicit = int(self.carrier_mobility is not None)
        bed.carrier_mobility = (
            self.carrier_mobility if bed.implicit else wp.zeros(1, dtype=wp.spatial_matrix, device=self.device)
        )
        bed.anchor_local = wp.array(anchors, dtype=wp.vec3, device=d)
        bed.rest_len = wp.array(np.asarray(self.rest_len, np.float32), dtype=wp.float32, device=d)
        bed.area = wp.array(np.broadcast_to(np.asarray(self.area, np.float32), (n,)).copy(), dtype=wp.float32, device=d)
        bed.neighbors = wp.array(np.asarray(self.neighbors, np.int32), dtype=wp.int32, device=d)
        color = column_colors(self.neighbors)
        bed.order = wp.array(np.argsort(color, kind="stable").astype(np.int32), dtype=wp.int32, device=d)
        bed.color_split = int(np.count_nonzero(color == 0))
        bed.connection = wp.array(connection, dtype=wp.int32, device=d)
        bed.top_gap = wp.array(gap.astype(np.float32), dtype=wp.float32, device=d)
        bed.ground_height = float(self.config.ground_height_m)
        bed.max_strain = float(self.max_strain)
        bed.over_relaxation = float(self.over_relaxation)
        bed.sweeps = int(self.sweeps)
        bed.params = wp.zeros(self.world_count, dtype=FoundationParams, device=d)
        bed.dt = wp.zeros(self.world_count, dtype=wp.float32, device=d)
        for name in (
            "top",
            "bottom",
            "q_state",
            "peq_prev",
            "compression",
            "carried",
            "reaction",
            "shear_modulus",
            "stiffness",
            "stop_gap",
            "stop_stiffness",
        ):
            setattr(bed, name, wp.zeros(total, dtype=wp.float32, device=d))
        bed.lever = wp.zeros(total, dtype=wp.vec3, device=d)
        bed.tangential_stiffness = wp.zeros(total, dtype=wp.float32, device=d)
        bed.tangential_load = wp.zeros(total, dtype=wp.vec2, device=d)
        bed.stuck = wp.zeros(total, dtype=wp.int32, device=d)
        bed.support = wp.zeros(total, dtype=wp.int32, device=d)
        bed.normal_force = wp.zeros(self.world_count, dtype=wp.float32, device=d)
        bed.carried_force = wp.zeros(self.world_count, dtype=wp.float32, device=d)
        bed.pressure_center = wp.zeros(self.world_count, dtype=wp.vec2, device=d)
        bed.held_strain = wp.zeros(self.world_count, dtype=wp.float32, device=d)
        self.bed = bed
        self.set_world_materials(self.materials)

    def _params(self, material, mu: float) -> FoundationParams:
        params = FoundationParams()
        set_material_block(params, material)
        params.inv_h2 = 1.0 / self.spacing_m**2
        params.stretch_floor = self.config.stretch_floor
        params.normal_damping = self.config.normal_damping
        params.mu = mu
        return params

    def set_world_materials(self, materials, *, friction: Sequence[float] | None = None) -> None:
        """Set one material for every world, or one material per world.

        Args:
            materials: One material, or one per world.
            friction: Coulomb friction coefficient per world; ``None`` uses the configured ``mu``.
        """
        if not isinstance(materials, Sequence):
            materials = [materials] * self.world_count
        if friction is None:
            friction = [self.config.mu] * self.world_count
        if len(materials) != self.world_count or len(friction) != self.world_count:
            raise ValueError(f"Expected {self.world_count} materials and friction coefficients")
        self.bed.params.assign([self._params(m, float(mu)) for m, mu in zip(materials, friction, strict=True)])

    def set_timesteps(self, dt) -> None:
        """Set one step for every world, or one step per world [s]."""
        values = np.broadcast_to(np.asarray(dt, np.float32), (self.world_count,))
        if not np.all(values > 0.0):
            raise ValueError("Timesteps must be positive")
        self.bed.dt.assign(values.copy())

    def reset(self, mask: wp.array[bool] | None = None) -> None:
        """Return the worlds in ``mask``, or every world, to the unloaded rest shape with no history."""
        bed = self.bed
        if mask is None:
            for array in (bed.top, bed.bottom, bed.q_state, bed.peq_prev):
                array.zero_()
            return
        wp.launch(
            _reset_worlds,
            dim=self.world_count * self.column_count,
            inputs=[mask, self.column_count],
            outputs=[bed.top, bed.bottom, bed.q_state, bed.peq_prev],
            device=self.device,
        )

    def apply(self, state) -> None:
        """Advance every world one step and set its carrier's ``body_f``.

        ``state`` carries ``body_q``, ``body_qd`` and ``body_f``. The wrench acts
        about each carrier's center of mass; friction and ``T`` act at the plane
        point under each nominal bottom.
        """
        wp.launch_tiled(
            _STEP,
            dim=self.world_count,
            block_dim=256,
            inputs=[self.bed, state.body_q, state.body_qd, state.body_f],
            device=self.device,
        )

    def column_state(self, name: str) -> np.ndarray:
        """Return a per-column array as ``[world_count, column_count]``."""
        return getattr(self.bed, name).numpy().reshape(self.world_count, self.column_count, -1).squeeze(-1)
