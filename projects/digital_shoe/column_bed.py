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
at its compression at the start of the step, so shear follows the axial tangent.
Held tops load the carrier with ``T``, landed bottoms load the plane with
``R``; friction follows ``R``.

One CUDA block of 256 lanes advances one world: it keeps the bed in shared
memory, runs the red-black over-relaxed sweeps, the Maxwell update, the column
reactions, elastic-Coulomb friction and the carrier wrench in one launch, so a
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
from .friction_maxwell import bristle_maxwell_step, elastic_coulomb_stiffness
from .material import maxwell_coefficients, maxwell_step
from .runtime import (
    FoundationConfig,
    FoundationParams,
    _hyperfoam_pressure,
    material_shear_modulus,
    set_material_block,
    shear_layer_modulus,
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
_BLOCK = wp.constant(256)
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


@wp.struct
class _Bed:
    column_count: int
    carrier: wp.array[wp.int32]
    body_com: wp.array[wp.vec3]
    implicit: int
    carrier_mobility: wp.array[wp.mat44]
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
    deflection: wp.array[wp.vec2]
    tangential_q: wp.array[wp.vec2]
    stuck: wp.array[wp.int32]
    dwell: wp.array[wp.float32]
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
    lever: wp.array[wp.vec2]


@wp.func
def _face_coupling(bed: _Bed, base: int, column: int, j: int) -> float:
    """Return the shear conductance between two columns of one world [N/m]."""
    return pasternak_coupling(
        bed.rest_len[column], bed.rest_len[j], bed.shear_modulus[base + column], bed.shear_modulus[base + j]
    )


@wp.func
def _face_sums(bed: _Bed, base: int, column: int, top: float, bottom: float) -> tuple[float, float, float]:
    """Return ``sum k_ij``, ``sum k_ij (t - t_j)`` and ``sum k_ij (b - b_j)`` over a column's faces."""
    coupling_sum = float(0.0)
    top_pull = float(0.0)
    bottom_pull = float(0.0)
    for side in range(4):
        j = bed.neighbors[column, side]
        if j >= 0:
            k = _face_coupling(bed, base, column, j)
            coupling_sum += k
            top_pull += k * (top - _top_shared(j, 0.0, 0))
            bottom_pull += k * (bottom - _bottom_shared(j, 0.0, 0))
    return coupling_sum, top_pull, bottom_pull


@wp.func
def _block_sum(value: float) -> float:
    """Sum one value over the block in a fixed order."""
    return wp.tile_sum(wp.tile(value))[0]


@wp.func
def _carrier_velocity(
    mass: wp.mat44d, u0: wp.vec4d, stiff: wp.vec3, stiff_moments: wp.vec3, load: wp.vec4, dt: float
) -> wp.vec4d:
    """Solve (M + dt A) u = M u0 + dt w for the end-of-step carrier velocity.

    ``A`` is the bed stiffness times dt over (v_z, w_x, w_y, w_z) from the sums
    ``stiff`` = (s, s r_x, s r_y) and ``stiff_moments`` = (s r_x^2, s r_y^2, s r_x r_y).
    On the densification wall ``A`` spans many decades, so the solve runs in double
    precision on a unit diagonal.
    """
    s0 = wp.float64(stiff[0])
    sx = wp.float64(stiff[1])
    sy = wp.float64(stiff[2])
    sxx = wp.float64(stiff_moments[0])
    syy = wp.float64(stiff_moments[1])
    sxy = wp.float64(stiff_moments[2])
    zero = wp.float64(0.0)
    a = wp.mat44d(s0, sy, -sx, zero, sy, syy, -sxy, zero, -sx, -sxy, sxx, zero, zero, zero, zero, zero)
    step = wp.float64(dt)
    system = mass + step * a
    scale = wp.vec4d()
    for row in range(4):
        scale[row] = wp.float64(1.0) / wp.sqrt(system[row, row])
    scaled = wp.mat44d()
    for row in range(4):
        for col in range(4):
            scaled[row, col] = scale[row] * system[row, col] * scale[col]
    rhs = mass * u0 + step * wp.vec4d(
        wp.float64(load[0]), wp.float64(load[1]), wp.float64(load[2]), wp.float64(load[3])
    )
    return wp.cw_mul(scale, wp.inverse(scaled) * wp.cw_mul(scale, rhs))


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
    body = bed.carrier[w]
    pose = body_q[body]
    shear = material_shear_modulus(p)
    # The bed lives in shared memory; loop bounds stay runtime values so the
    # column solve is compiled once rather than unrolled per row and color.
    for slot in range(lane, count, _BLOCK):
        column = bed.order[slot]
        i = base + column
        _top_shared(column, bed.top[i], 1)
        _bottom_shared(column, bed.bottom[i], 1)
        _clearance_shared(column, wp.transform_point(pose, bed.anchor_local[column])[2] - bed.ground_height, 1)
        _overstress_shared(column, decay * bed.q_state[i] - gain * bed.peq_prev[i], 1)
        # The shear layers follow each column's compression at the start of the step.
        strain = wp.max(bed.top[i] - bed.bottom[i], 0.0) / bed.rest_len[column]
        bed.shear_modulus[i] = shear_layer_modulus(strain, p)
    _sync_threads()
    # Red-black Gauss-Seidel: lanes hold columns sorted by color and a color
    # reads only the other color, so each half sweep updates in place and then
    # synchronizes once. A sweep that moves no column leaves the next one
    # nothing to move, so the bed stops there with the same result. The flag of
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
            coupling_sum, top_pull, bottom_pull = _face_sums(bed, base, column, top, bottom)
            c = top - bottom
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
                _hyperfoam_pressure(c / thickness, p),
                _hyperfoam_pressure((c + 1.0e-3 * thickness) / thickness, p),
                bed.over_relaxation,
            )
            if t != top or b != bottom:
                _changed_shared(sweep % 3, 1, 1)
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
    carried_sum = float(0.0)
    moment_sum = wp.vec2(0.0)
    largest = float(0.0)
    # Axial stiffness of the held columns times dt, for the implicit carrier.
    stiff = float(0.0)
    reach = float(0.0)
    # Load on the carrier from everything but the implicit columns' axial force.
    free_z = float(0.0)
    free_torque = wp.vec3(0.0)
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
        bed.q_state[i] = q
        bed.peq_prev[i] = peq
        # Shear on each end, positive down, from the balanced neighbours.
        on_top = float(0.0)
        on_bottom = float(0.0)
        for side in range(4):
            j = bed.neighbors[column, side]
            if j >= 0:
                k = _face_coupling(bed, base, column, j)
                on_top += k * (top - _top_shared(j, 0.0, 0))
                on_bottom += k * (bottom - _bottom_shared(j, 0.0, 0))
        point, com, velocity, _gap = contact_kinematics(
            pose, twist, com_local, bed.anchor_local[column], bed.ground_height, 1
        )
        held = (support & TOP_HELD) != 0
        landed = (support & BOTTOM_LANDED) != 0
        # Axial column force. A column free at one end passes on exactly the shear
        # that balances that end, including the reaction of its compression bounds.
        axial = area * wp.max(peq + q, 0.0)
        if held and landed:
            if comp > 0.0:
                axial = wp.max(axial - p.normal_damping * velocity[2], 0.0)
        elif held:
            axial = on_bottom
        elif landed:
            axial = -on_top
        bed.stiffness[i] = 0.0
        if bed.implicit != 0 and held and comp > 0.0:
            # Axial tangent of the foam, by a backward difference so it never
            # steps onto the stretch floor.
            h = 1.0e-3 * thickness
            foam = area * (1.0 + gain) * (peq - _hyperfoam_pressure((comp - h) / thickness, p)) / h
            if not landed:
                # A floating bottom reaches the ground only through the shear
                # layer to its landed neighbours, in series with the foam.
                anchored = float(0.0)
                for side in range(4):
                    j = bed.neighbors[column, side]
                    if j >= 0 and (_support_shared(j, 0, 0) & BOTTOM_LANDED) != 0:
                        anchored += _face_coupling(bed, base, column, j)
                foam = foam * anchored / (foam + anchored + 1.0e-12)
            s_col = dt * wp.max(foam, 0.0)
            r = point - com
            bed.stiffness[i] = s_col
            bed.lever[i] = wp.vec2(r[0], r[1])
            stiff += s_col
            reach = wp.max(reach, wp.length(wp.vec2(r[0], r[1])))
        carried = float(0.0)
        if held:
            carried = axial + on_top
            largest = wp.max(largest, comp / thickness)
        reaction = float(0.0)
        if landed:
            reaction = axial - on_bottom
        kt = elastic_coulomb_stiffness(shear, area, thickness)
        # The bristle shears the column, so it relaxes with the foam's Maxwell branch.
        tangential, _jacobian, z, tq, s, elapsed = bristle_maxwell_step(
            wp.vec2(velocity[0], velocity[1]),
            dt,
            reaction,
            kt,
            p.overstress * kt * p.tau_s,
            p.tau_s,
            p.mu,
            p.friction_release_dwell_s,
            bed.deflection[i],
            bed.tangential_q[i],
            bed.stuck[i],
            bed.dwell[i],
        )
        bed.deflection[i] = z
        bed.tangential_q[i] = tq
        bed.stuck[i] = s
        bed.dwell[i] = elapsed
        bed.top[i] = top
        bed.bottom[i] = bottom
        bed.compression[i] = comp
        bed.carried[i] = carried
        bed.reaction[i] = reaction
        bed.support[i] = support
        force = wp.vec3(tangential[0], tangential[1], carried)
        force_sum += force
        torque_sum += wp.cross(point - com, force)
        free = force
        if bed.stiffness[i] > 0.0:
            free = wp.vec3(tangential[0], tangential[1], 0.0)
        free_z += free[2]
        free_torque += wp.cross(point - com, free)
        normal_sum += reaction
        carried_sum += carried
        moment_sum += wp.vec2(point[0], point[1]) * reaction
    force_total = wp.vec3(_block_sum(force_sum[0]), _block_sum(force_sum[1]), _block_sum(force_sum[2]))
    torque_total = wp.vec3(_block_sum(torque_sum[0]), _block_sum(torque_sum[1]), _block_sum(torque_sum[2]))
    normal_total = _block_sum(normal_sum)
    carried_total = _block_sum(carried_sum)
    moment_total = wp.vec2(_block_sum(moment_sum[0]), _block_sum(moment_sum[1]))
    largest_total = wp.tile_max(wp.tile(largest))[0]
    if bed.implicit != 0:
        s0 = _block_sum(stiff)
        reach_total = wp.tile_max(wp.tile(reach))[0]
        if s0 > 0.0:
            # Linearized backward Euler of the carrier on the bed: the vertical
            # force and its moments are taken at the end-of-step velocity u of
            # the carrier with the effective mass of its articulation. Every lane
            # solves the same 4x4 system, so the active set below needs no broadcast.
            mobility = bed.carrier_mobility[w]
            mass = wp.inverse(
                wp.mat44d(
                    wp.float64(mobility[0, 0]),
                    wp.float64(mobility[0, 1]),
                    wp.float64(mobility[0, 2]),
                    wp.float64(mobility[0, 3]),
                    wp.float64(mobility[1, 0]),
                    wp.float64(mobility[1, 1]),
                    wp.float64(mobility[1, 2]),
                    wp.float64(mobility[1, 3]),
                    wp.float64(mobility[2, 0]),
                    wp.float64(mobility[2, 1]),
                    wp.float64(mobility[2, 2]),
                    wp.float64(mobility[2, 3]),
                    wp.float64(mobility[3, 0]),
                    wp.float64(mobility[3, 1]),
                    wp.float64(mobility[3, 2]),
                    wp.float64(mobility[3, 3]),
                )
            )
            omega = wp.spatial_bottom(twist)
            u0 = wp.vec4d(
                wp.float64(wp.spatial_top(twist)[2]), wp.float64(omega[0]), wp.float64(omega[1]), wp.float64(omega[2])
            )
            free_load = wp.vec4(
                _block_sum(free_z), _block_sum(free_torque[0]), _block_sum(free_torque[1]), _block_sum(free_torque[2])
            )
            u = u0
            # The bed pushes and never pulls: a column whose end-of-step force
            # B - s v would be negative leaves the system, and the rest re-solve.
            # Sums are rebuilt from the active columns rather than reduced, since
            # on the wall a column carries many decades more than the result.
            for _pass in range(_CARRIER_PASSES):
                stiff_sums = wp.vec3(0.0)
                moment_sums = wp.vec3(0.0)
                held_load = wp.vec3(0.0)
                for slot in range(lane, count, _BLOCK):
                    i = base + bed.order[slot]
                    s_col = bed.stiffness[i]
                    if s_col > 0.0:
                        lever = bed.lever[i]
                        carried = bed.carried[i]
                        stiff_sums += s_col * wp.vec3(1.0, lever[0], lever[1])
                        moment_sums += s_col * wp.vec3(lever[0] * lever[0], lever[1] * lever[1], lever[0] * lever[1])
                        held_load += carried * wp.vec3(1.0, lever[1], -lever[0])
                stiff_sums = wp.vec3(_block_sum(stiff_sums[0]), _block_sum(stiff_sums[1]), _block_sum(stiff_sums[2]))
                moment_sums = wp.vec3(
                    _block_sum(moment_sums[0]), _block_sum(moment_sums[1]), _block_sum(moment_sums[2])
                )
                held_load = wp.vec3(_block_sum(held_load[0]), _block_sum(held_load[1]), _block_sum(held_load[2]))
                load = free_load + wp.vec4(held_load[0], held_load[1], held_load[2], 0.0)
                u = _carrier_velocity(mass, u0, stiff_sums, moment_sums, load, dt)
                dropped = float(0.0)
                for slot in range(lane, count, _BLOCK):
                    i = base + bed.order[slot]
                    s_col = bed.stiffness[i]
                    if s_col > 0.0:
                        lever = bed.lever[i]
                        speed = wp.float32(u[0] + u[1] * wp.float64(lever[1]) - u[2] * wp.float64(lever[0]))
                        if s_col * speed > bed.carried[i]:
                            bed.stiffness[i] = 0.0
                            dropped += 1.0
                if _block_sum(dropped) == 0.0:
                    break
            # The load is the carrier's momentum change M (u - u0) / dt, not the
            # difference w - A u of two terms that are large on the wall.
            change = mass * (u - u0) / wp.float64(dt)
            lift = wp.max(wp.float32(change[0]), 0.0)
            # Pushing columns cannot make a moment beyond their reach.
            moment = wp.vec2(wp.float32(change[1]), wp.float32(change[2]))
            if wp.length(moment) > lift * reach_total:
                moment = moment * (lift * reach_total / wp.length(moment))
            # The ground reaction and its center follow the carried load.
            if carried_total > 0.0:
                ratio = lift / carried_total
                normal_total = normal_total * ratio
                moment_total = moment_total * ratio
            carried_total = lift
            force_total = wp.vec3(force_total[0], force_total[1], lift)
            torque_total = wp.vec3(moment[0], moment[1], torque_total[2])
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
    deflection: wp.array[wp.vec2],
    tangential_q: wp.array[wp.vec2],
    stuck: wp.array[wp.int32],
    dwell: wp.array[wp.float32],
):
    """Clear the columns of the masked worlds."""
    i = wp.tid()
    if mask[i / column_count]:
        top[i] = 0.0
        bottom[i] = 0.0
        q_state[i] = 0.0
        peq_prev[i] = 0.0
        deflection[i] = wp.vec2(0.0)
        tangential_q[i] = wp.vec2(0.0)
        stuck[i] = 0
        dwell[i] = 0.0


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
            ``friction_release_dwell_s``, ``normal_damping`` and
            ``stretch_floor``. Friction is always elastic Coulomb.
        carrier: Carrier body index per world.
        body_com: Center of mass per body in its frame [m].
        top_gap_m: Rest clearance between the carrier surface and each top [m];
            zero by default.
        carrier_mobility: Inverse effective mass of each world's carrier in
            ``(v_z, w_x, w_y, w_z)`` at its center of mass, in world axes
            [1/kg, 1/(kg m), 1/(kg m^2)], one :class:`warp.mat44` per world, kept
            current by the caller. With it, each carrier takes the bed's normal
            force at its end-of-step velocity (linearized backward Euler), which
            stays stable on the foam's densification wall. Without, carriers see
            the explicit force, as kinematically driven fixtures do.
        max_strain: Compression limit of a free top as a fraction of rest
            thickness.
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
    max_strain: float = 0.9
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
            self.carrier_mobility if bed.implicit else wp.zeros(1, dtype=wp.mat44, device=self.device)
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
            "dwell",
            "compression",
            "carried",
            "reaction",
            "shear_modulus",
            "stiffness",
        ):
            setattr(bed, name, wp.zeros(total, dtype=wp.float32, device=d))
        bed.deflection = wp.zeros(total, dtype=wp.vec2, device=d)
        bed.tangential_q = wp.zeros(total, dtype=wp.vec2, device=d)
        bed.lever = wp.zeros(total, dtype=wp.vec2, device=d)
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
        params.friction_release_dwell_s = self.config.friction_release_dwell_s
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
            for array in (
                bed.top,
                bed.bottom,
                bed.q_state,
                bed.peq_prev,
                bed.deflection,
                bed.tangential_q,
                bed.stuck,
                bed.dwell,
            ):
                array.zero_()
            return
        wp.launch(
            _reset_worlds,
            dim=self.world_count * self.column_count,
            inputs=[mask, self.column_count],
            outputs=[
                bed.top,
                bed.bottom,
                bed.q_state,
                bed.peq_prev,
                bed.deflection,
                bed.tangential_q,
                bed.stuck,
                bed.dwell,
            ],
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
