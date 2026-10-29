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
Both layers have the constant modulus ``shear_layer_fraction`` of the material (default
:data:`~projects.digital_shoe.runtime.SHEAR_LAYER_FRACTION`) times the foam's rest shear modulus
:func:`~projects.digital_shoe.runtime.material_shear_modulus` at every compression, so the layers are
linear in the displacements and conserve energy.
Held tops load the carrier with ``T``, landed bottoms load the plane with
``R``. Columns are one-dimensional: a column bottom moves sideways with the
carrier, so friction acts between the carrier and the plane. Each landed bottom
grips the plane through an elastic-Coulomb bristle: a spring of constant
stiffness ``k_t = f G_eq A / L`` (``f`` is ``bristle_shear_fraction``) stores
the bottom's tangential deflection, its force is held to ``mu R`` and the
deflection is cleared when ``R`` is zero.

One CUDA block of 256 lanes advances one world: it keeps the bed in shared
memory, runs the red-black over-relaxed sweeps, the Maxwell update, the column
reactions, the bristles and the carrier wrench in one launch, so a step costs
one launch however many sweeps it takes. The bed is explicit: the carrier gets
the wrench of the bed at its start-of-step pose and twist.
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
    cone_viscous_scale,
    contact_kinematics,
    pasternak_coupling,
)
from .material import maxwell_coefficients, maxwell_step
from .runtime import (
    SHEAR_LAYER_FRACTION,
    FoundationConfig,
    FoundationParams,
    _hyperfoam_pressure,
    material_shear_modulus,
    set_material_block,
)

__all__ = [
    "BOTTOM_LANDED",
    "MAX_COLUMNS",
    "SHEAR_LAYER_FRACTION",
    "TOP_BONDED",
    "TOP_FREE",
    "TOP_HELD",
    "TOP_PRESSED",
    "ColumnBed",
    "column_colors",
]

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


@wp.struct
class _Bed:
    column_count: int
    carrier: wp.array[wp.int32]
    body_com: wp.array[wp.vec3]
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
    bristle_fraction: float
    bristle_time: float
    params: wp.array[FoundationParams]
    dt: wp.array[wp.float32]
    top: wp.array[wp.float32]
    bottom: wp.array[wp.float32]
    q_state: wp.array[wp.float32]
    peq_prev: wp.array[wp.float32]
    stuck: wp.array[wp.int32]
    deflection: wp.array[wp.vec2]
    tangential_load: wp.array[wp.vec2]
    compression: wp.array[wp.float32]
    carried: wp.array[wp.float32]
    reaction: wp.array[wp.float32]
    support: wp.array[wp.int32]
    normal_force: wp.array[wp.float32]
    carried_force: wp.array[wp.float32]
    pressure_center: wp.array[wp.vec2]
    held_strain: wp.array[wp.float32]


_STRAIN_STEP = wp.constant(1.0e-3)
"""Strain step of the finite differences of the foam law."""


@wp.func
def _face_coupling(bed: _Bed, column: int, j: int, shear: float) -> float:
    """Return the shear conductance between two columns of one world, whose layers have modulus ``shear`` [N/m]."""
    return pasternak_coupling(bed.rest_len[column], bed.rest_len[j], shear)


@wp.func
def _face_sums(bed: _Bed, column: int, top: float, bottom: float, shear: float) -> tuple[float, float, float]:
    """Return ``sum k_ij``, ``sum k_ij (t - t_j)`` and ``sum k_ij (b - b_j)`` over a column's faces."""
    coupling_sum = float(0.0)
    top_pull = float(0.0)
    bottom_pull = float(0.0)
    for side in range(4):
        j = bed.neighbors[column, side]
        if j >= 0:
            k = _face_coupling(bed, column, j, shear)
            coupling_sum += k
            top_pull += k * (top - _top_shared(j, 0.0, 0))
            bottom_pull += k * (bottom - _bottom_shared(j, 0.0, 0))
    return coupling_sum, top_pull, bottom_pull


@wp.func
def _bristle(
    deflection: wp.vec2, velocity: wp.vec2, reaction: float, mu: float, stiffness: float, damping_time: float, dt: float
) -> tuple[wp.vec2, wp.vec2, int]:
    """Return a landed bottom's friction, its new bristle deflection and whether it sticks.

    The bottom moves at ``velocity`` over the plane. The bristle spring of
    ``stiffness`` [N/m] takes the deflection the step adds, and a radial return
    holds its force to ``mu R``; a damper of ``stiffness * damping_time`` adds
    only as much force as still fits inside that disk. A bottom without a
    ground reaction loses its deflection.
    """
    if reaction <= 0.0 or mu <= 0.0:
        return wp.vec2(0.0), wp.vec2(0.0), 0
    capacity = mu * reaction
    stretch = deflection + dt * velocity
    elastic = -stiffness * stretch
    sticking = 1
    size = wp.length(elastic)
    if size > capacity:
        # The bottom slides: the spring keeps the stretch that carries mu R.
        stretch = stretch * (capacity / size)
        elastic = -stiffness * stretch
        sticking = 0
    viscous = -(stiffness * damping_time) * velocity
    return elastic + viscous * cone_viscous_scale(elastic, viscous, capacity), stretch, sticking


@wp.func
def _block_sum(value: float) -> float:
    """Sum one value over the block in a fixed order."""
    return wp.tile_sum(wp.tile(value))[0]


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
    shear = p.shear_fraction * material_shear_modulus(p)
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
            coupling_sum, top_pull, bottom_pull = _face_sums(bed, column, top, bottom, shear)
            c = top - bottom
            pressure = _hyperfoam_pressure(c / thickness, p)
            pressure_ahead = _hyperfoam_pressure((c + _STRAIN_STEP * thickness) / thickness, p)
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
                pressure,
                pressure_ahead,
                bed.over_relaxation,
            )
            if wp.abs(t - top) > _SWEEP_TOLERANCE * thickness or wp.abs(b - bottom) > _SWEEP_TOLERANCE * thickness:
                _changed_shared(sweep % 3, 1, 1)
            _top_shared(column, t, 1)
            _bottom_shared(column, b, 1)
            _support_shared(column, state, 1)
        _sync_threads()
        if half % 2 == 1 and _changed_shared(sweep % 3, 0, 0) == 0:
            break

    twist = body_qd[body]
    com_local = bed.body_com[body]
    bristle_modulus = bed.bristle_fraction * material_shear_modulus(p)
    force_sum = wp.vec3(0.0)
    torque_sum = wp.vec3(0.0)
    normal_sum = float(0.0)
    carried_sum = float(0.0)
    moment_sum = wp.vec2(0.0)
    largest = float(0.0)
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
                k = _face_coupling(bed, column, j, shear)
                on_top += k * (top - _top_shared(j, 0.0, 0))
                on_bottom += k * (bottom - _bottom_shared(j, 0.0, 0))
        point, com, velocity, _gap = contact_kinematics(
            pose, twist, com_local, bed.anchor_local[column], bed.ground_height, 1
        )
        held = (support & TOP_HELD) != 0
        landed = (support & BOTTOM_LANDED) != 0
        # Axial column force. A column free at one end passes on exactly the shear
        # that balances that end, including the reaction of its compression bounds.
        axial = wp.max(area * (peq + q), 0.0)
        if held and landed:
            if comp > 0.0:
                axial = wp.max(axial - p.normal_damping * velocity[2], 0.0)
        elif held:
            axial = on_bottom
        elif landed:
            axial = -on_top
        carried = float(0.0)
        if held:
            carried = axial + on_top
            largest = wp.max(largest, comp / thickness)
        reaction = float(0.0)
        if landed:
            reaction = axial - on_bottom
        tangential, deflection, sticking = _bristle(
            bed.deflection[i],
            wp.vec2(velocity[0], velocity[1]),
            reaction,
            p.mu,
            bristle_modulus * area / thickness,
            bed.bristle_time,
            dt,
        )
        bed.deflection[i] = deflection
        bed.tangential_load[i] = tangential
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
        normal_sum += reaction
        carried_sum += carried
        moment_sum += wp.vec2(point[0], point[1]) * reaction
    force_total = wp.vec3(_block_sum(force_sum[0]), _block_sum(force_sum[1]), _block_sum(force_sum[2]))
    torque_total = wp.vec3(_block_sum(torque_sum[0]), _block_sum(torque_sum[1]), _block_sum(torque_sum[2]))
    normal_total = _block_sum(normal_sum)
    carried_total = _block_sum(carried_sum)
    moment_total = wp.vec2(_block_sum(moment_sum[0]), _block_sum(moment_sum[1]))
    largest_total = wp.tile_max(wp.tile(largest))[0]
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
):
    """Clear the columns of the masked worlds."""
    i = wp.tid()
    if mask[i / column_count]:
        top[i] = 0.0
        bottom[i] = 0.0
        q_state[i] = 0.0
        peq_prev[i] = 0.0
        deflection[i] = wp.vec2(0.0)


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
            ``normal_damping``, ``stretch_floor`` and the bristle's
            ``bristle_shear_fraction`` and ``bristle_damping_time_s``. A landed
            bottom grips the plane through a bristle of stiffness
            ``bristle_shear_fraction * G_eq * area / rest_len`` whose force is
            held to ``mu R``.
        carrier: Carrier body index per world.
        body_com: Center of mass per body in its frame [m].
        top_gap_m: Rest clearance between the carrier surface and each top [m];
            zero by default.
        max_strain: Compression limit of the sweeps as a fraction of rest
            thickness, 0.95 by default; ``1 - max_strain`` must not be below
            ``config.stretch_floor``. The sweeps keep a free top from closing
            past it and push a bonded column's bottom down to keep it; a pressed
            top is not limited. It is a geometric bound only: a carrier pressed
            deeper meets the foam's 100 MPa tangent cap.
        sweeps: Red-black sweeps per step, warm started from the previous step;
            16 by default, which settles a running stance. Quasi-static replays
            that need the bed fully balanced every step pass 64.
        over_relaxation: Fraction of each column's constrained Newton step taken,
            1.0 by default; above one is projected successive over-relaxation.
        device: CUDA device.

    The bed is explicit: :meth:`apply` writes the wrench of the bed at the
    carrier's start-of-step pose and twist, and the caller integrates the
    carrier. On the capped foam law this is stable at running substeps up to
    about 5e-4 s; the bristle damper stays passive only while
    ``config.bristle_damping_time_s`` is at least 4 substeps.
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
    max_strain: float = 0.95
    sweeps: int = 16
    over_relaxation: float = 1.0
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
        if 1.0 - self.max_strain < self.config.stretch_floor - 1.0e-9:
            raise ValueError(
                f"max_strain {self.max_strain} leaves a stretch below config.stretch_floor {self.config.stretch_floor}"
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
        bed.bristle_fraction = float(self.config.bristle_shear_fraction)
        bed.bristle_time = float(self.config.bristle_damping_time_s)
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
        ):
            setattr(bed, name, wp.zeros(total, dtype=wp.float32, device=d))
        bed.deflection = wp.zeros(total, dtype=wp.vec2, device=d)
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
            for array in (bed.top, bed.bottom, bed.q_state, bed.peq_prev, bed.deflection):
                array.zero_()
            return
        wp.launch(
            _reset_worlds,
            dim=self.world_count * self.column_count,
            inputs=[mask, self.column_count],
            outputs=[bed.top, bed.bottom, bed.q_state, bed.peq_prev, bed.deflection],
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
        """Return a per-column array as ``[world_count, column_count]``, plus a last axis of 2 for ``deflection`` and ``tangential_load``."""
        values = getattr(self.bed, name).numpy()
        return values.reshape(self.world_count, self.column_count, *values.shape[1:])
