# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lean ground contact of the runner shoe: physics state only, one block per world.

The generic fused foundation also maintains pressure transfer, diagnostics, and
cached per-column forces for its many consumers. The runner only reads the
carrier wrench and the driven-compression screen, so these kernels evaluate the
same shared material, Maxwell, surround, and elastic-Coulomb laws for each
column and keep everything else in registers or shared memory. Per column they
read and write only surround, Maxwell, and bristle history; the wrench uses the
shared fixed-order reduction. Carrier forces and histories are bitwise identical
to :class:`FoundationFused`.

Each world's shoe is skipped while every column clears the ground by
:data:`FLIGHT_MARGIN_M`: before first contact its history is exactly zero, and
after lift-off the shared laws only decay that history at zero load. Skipped
lifted steps are counted and replayed column by column, with the same
functions, if the shoe comes back down.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from projects.digital_shoe.contact import _surround_balance_pressures, contact_kinematics, normal_reaction
from projects.digital_shoe.friction_maxwell import bristle_elastic_coulomb_step, elastic_coulomb_stiffness
from projects.digital_shoe.friction_parameter_adapter import FrictionParameterAdapter
from projects.digital_shoe.material import HYPERFOAM_ALPHA_FLOOR, maxwell_coefficients, ogden_hill_term
from projects.digital_shoe.runtime import FoundationParams, _pasternak_coupling

# Match the shared float32 shoe runtime, not the float64 leg module.
wp.set_module_options({"enable_backward": False, "fuse_fp": True})
_EXACT = {"enable_backward": False, "fuse_fp": True, "fast_math": False}
_FAST = {"enable_backward": False, "fuse_fp": True, "fast_math": True}

_BLOCK = wp.constant(256)
_ROWS = wp.constant(4)
_Rows = wp.types.vector(4, float)
_Wrench = wp.types.vector(6, float)
_SURROUND_SLOTS = wp.constant(1024)
_WARPS = wp.constant(8)
_MAX_SLOT = wp.constant(96)
_TOTAL_SLOT = wp.constant(104)
_SCREEN_BLOCK = wp.constant(64)
_ELASTIC_COULOMB = 9
_DRIVEN = wp.constant(-2)
_PRISTINE = wp.constant(0)
_ACTIVE = wp.constant(1)
_LIFTED = wp.constant(2)
FLIGHT_MARGIN_M = 5.0e-4
"""Rigid clearance above which a shoe's columns are treated as airborne [m]."""


# Raw block-shared buffers: register tiles synchronize the block on every
# cross-lane read, and tile assignment adds a barrier per call.
@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float surround_buffer[2 * 1024];
if (write) surround_buffer[index] = value;
return surround_buffer[index];
#else
return value;
#endif
""")
def _surround_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float wrench_buffer[24 * 256];
if (write) wrench_buffer[index] = value;
return wrench_buffer[index];
#else
return value;
#endif
""")
def _wrench_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ float reduce_buffer[128];
if (write) reduce_buffer[index] = value;
return reduce_buffer[index];
#else
return value;
#endif
""")
def _reduce_shared(index: int, value: float, write: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ double reduce_buffer64[32];
if (write) reduce_buffer64[index] = value;
return reduce_buffer64[index];
#else
return value;
#endif
""")
def _reduce_shared64(index: int, value: wp.float64, write: int) -> wp.float64: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
float result = value;
for (int offset = 16; offset > 0; offset >>= 1)
    result = fmaxf(result, __shfl_xor_sync(0xffffffffu, result, offset));
return result;
#else
return value;
#endif
""")
def _warp_max(value: float) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
double result = value;
for (int offset = 16; offset > 0; offset >>= 1)
    result = fmax(result, __shfl_xor_sync(0xffffffffu, result, offset));
return result;
#else
return value;
#endif
""")
def _warp_max64(value: wp.float64) -> wp.float64: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __fmaf_rn(decay, q, __fmul_rn(__fmul_rn(overstress, ramp), __fsub_rn(peq, peq_prev)));
#else
return decay * q + overstress * ramp * (peq - peq_prev);
#endif
""")
def _maxwell_update(q: float, peq: float, peq_prev: float, overstress: float, decay: float, ramp: float) -> float:
    """Return :func:`maxwell_step` with the rounding the shared CUDA kernels compile to.

    Fixing the fused multiply-add keeps contact and the lifted-step replay bitwise
    equal regardless of how either call site would otherwise be contracted.
    """
    ...


@wp.struct
class GroundShoe:
    """Persistent arrays the lean contact kernels read from a fused foundation."""

    enabled: wp.array[int]
    carrier: wp.array[wp.int32]
    column_count: int
    groups: int
    driven: wp.array[wp.int32]
    anchor_local: wp.array[wp.vec3]
    z_free_rigid: wp.array[wp.float32]
    rest_len: wp.array[wp.float32]
    area: wp.array[wp.float32]
    screen_rest: wp.array[wp.float64]
    exact_screen: int
    screen_limit: float
    world_params: wp.array[FoundationParams]
    world_dt: wp.array[wp.float32]
    q_state: wp.array[wp.float32]
    peq_prev: wp.array[wp.float32]
    surround: wp.array[wp.float32]
    has_surround: int
    body_com: wp.array[wp.vec3]
    settings: wp.array2d[float]
    deflection: wp.array[wp.vec2]
    stuck: wp.array[int]
    dwell: wp.array[float]
    ground_height: float
    fused_surround: int
    free_column: wp.array[int]
    column_slot: wp.array[int]
    neighbors: wp.array2d[wp.int32]
    zero_pressure: wp.array[wp.vec2]
    surround_decay: wp.array[wp.float32]
    surround_gain: wp.array[wp.float32]
    world_relaxation: wp.array[wp.float32]
    coupling_scale: float
    attachment: float
    max_strain: float
    carrier_bond: int
    sweeps: int
    surround_lanes: int
    phase: wp.array[int]
    pending: wp.array[int]
    dormancy: int
    flight_columns: wp.array[int]


@wp.func
def _ogden_hill(stretch: float, mu: float, alpha: float, beta: float, one_minus_two_poisson: float):
    """Evaluate :func:`ogden_hill_term` with its volumetric power skipped when it is exactly one.

    ``pow(J, -alpha * beta)`` with ``beta == 0`` is ``pow(J, +/-0)``, which IEEE and
    CUDA define as exactly one for every ``J``. Other materials and the small-alpha
    limit keep the shared expression unchanged.
    """
    if beta == 0.0 and wp.abs(alpha) >= HYPERFOAM_ALPHA_FLOOR:
        return 2.0 * mu / (alpha * stretch) * (1.0 - stretch**alpha)
    return ogden_hill_term(stretch, stretch**one_minus_two_poisson, mu, alpha, beta)


@wp.func
def _hyperfoam(strain: float, p: FoundationParams):
    """Return :func:`_hyperfoam_pressure` bitwise, avoiding redundant powers [Pa]."""
    stretch = wp.max(1.0 - strain, p.stretch_floor)
    return _ogden_hill(stretch, p.g_eq, p.alpha, p.beta, p.one_minus_two_poisson) + _ogden_hill(
        stretch, p.g_eq2, p.alpha2, p.beta, p.one_minus_two_poisson
    )


@wp.func
def _relax_free_column(shoe: GroundShoe, w: int, lane: int, pose: wp.transform):
    """Run the surround Jacobi sweeps of ``_surround_world`` for one undriven column per lane.

    Driven columns hold their old value in the first sweep and their rigid
    compression afterwards, so each lane keeps both for its driven neighbours and
    only undriven columns enter the shared exchange. Sums keep the original side
    order, unchanged compressions reuse their pressures, and a column without
    rigid penetration is clamped to zero by the carrier bond, so every
    compression stays bitwise unchanged.
    """
    count = shoe.column_count
    p = shoe.world_params[w]
    gain = shoe.surround_gain[w]
    relaxation = shoe.world_relaxation[w]
    c = float(0.0)
    rigid = float(0.0)
    thickness = float(1.0)
    column_area = float(0.0)
    overstress = float(0.0)
    coupling_sum = float(0.0)
    slots = wp.vec4i(-1)
    couplings = wp.vec4(0.0)
    old = wp.vec4(0.0)
    new = wp.vec4(0.0)
    column = shoe.free_column[lane]
    i = w * count + column
    if column >= 0:
        c = shoe.surround[i]
        world = wp.transform_point(pose, shoe.anchor_local[column])
        rigid = shoe.z_free_rigid[column] - world[2]
        thickness = shoe.rest_len[column]
        column_area = shoe.area[column]
        overstress = shoe.surround_decay[w] * shoe.q_state[i] - gain * shoe.peq_prev[i]
        for side in range(4):
            j = shoe.neighbors[column, side]
            if j >= 0:
                coupling = shoe.coupling_scale * _pasternak_coupling(thickness, shoe.rest_len[j], p)
                couplings[side] = coupling
                coupling_sum += coupling
                slot = shoe.column_slot[j]
                if slot >= 0:
                    slots[side] = slot
                else:
                    slots[side] = _DRIVEN
                    old[side] = shoe.surround[w * count + j]
                    neighbor = wp.transform_point(pose, shoe.anchor_local[j])
                    new[side] = wp.max(shoe.z_free_rigid[j] - neighbor[2], 0.0)
    upper = shoe.max_strain * thickness
    if shoe.carrier_bond != 0:
        upper = wp.clamp(rigid, 0.0, upper)
    pinned = shoe.carrier_bond != 0 and upper == 0.0
    step = 1.0e-3 * thickness
    cached_c = float(-1.0)
    cached_peq = float(0.0)
    cached_ahead = float(0.0)
    for sweep in range(shoe.sweeps):
        # Ping-pong buffers need one barrier per sweep: a buffer is rewritten only
        # after every lane has passed the following sweep's barrier.
        buffer = (sweep % 2) * _SURROUND_SLOTS
        _surround_shared(buffer + lane, c, 1)
        _sync_threads()
        next_c = c
        if column >= 0:
            if pinned:
                next_c = 0.0
            else:
                pull = float(0.0)
                for side in range(4):
                    slot = slots[side]
                    if slot != -1:
                        value = new[side]
                        if slot >= 0:
                            value = _surround_shared(buffer + slot, 0.0, 0)
                        elif sweep == 0:
                            value = old[side]
                        pull += couplings[side] * (value - c)
                peq = float(0.0)
                ahead = float(0.0)
                if c == 0.0:
                    cached = shoe.zero_pressure[i]
                    peq = cached[0]
                    ahead = cached[1]
                else:
                    # The Newton step of contact.surround_balance with the shared material law.
                    if c != cached_c:
                        cached_c = c
                        cached_peq = _hyperfoam(c / thickness, p)
                        cached_ahead = _hyperfoam((c + step) / thickness, p)
                    peq = cached_peq
                    ahead = cached_ahead
                next_c = _surround_balance_pressures(
                    c,
                    rigid,
                    pull,
                    coupling_sum,
                    thickness,
                    overstress,
                    gain,
                    column_area,
                    shoe.attachment,
                    shoe.max_strain,
                    relaxation,
                    shoe.carrier_bond,
                    peq,
                    ahead,
                )
        c = next_c
    if column >= 0:
        shoe.surround[i] = c


def _ground_surround(shoe: GroundShoe, body_q: wp.array[wp.transform]):
    """Relax one world's undriven columns before :func:`_ground_contact` reads them.

    Launch with one lane per undriven column, rounded up to whole warps.
    """
    w, lane = wp.tid()
    if shoe.enabled[w] == 0 or shoe.phase[w] != _ACTIVE:
        return
    _relax_free_column(shoe, w, lane, body_q[shoe.carrier[w]])


@wp.func
def _replay_lifted(shoe: GroundShoe, w: int, lane: int, steps: int):
    """Apply ``steps`` skipped lifted updates to every column, exactly as contact would.

    With all columns airborne, compression and normal reaction are exactly zero,
    the carrier bond clamps the surround to zero, and the bristle law takes its
    unloaded branch, which does not read velocity.
    """
    count = shoe.column_count
    p = shoe.world_params[w]
    dt = shoe.world_dt[w]
    decay, ramp = maxwell_coefficients(dt, p.tau_s)
    peq_zero = shoe.zero_pressure[w * count][0]
    shear = p.g_eq + p.g_eq2
    mu = shoe.settings[w, 1]
    kt_scale = shoe.settings[w, 2]
    release = shoe.settings[w, 5]
    for column in range(lane, count, _SCREEN_BLOCK):
        i = w * count + column
        q = shoe.q_state[i]
        peq_old = shoe.peq_prev[i]
        z = shoe.deflection[i]
        s = shoe.stuck[i]
        elapsed = shoe.dwell[i]
        kt = elastic_coulomb_stiffness(shear, shoe.area[column], shoe.rest_len[column]) * kt_scale
        for _step in range(steps):
            q = _maxwell_update(q, peq_zero, peq_old, p.overstress, decay, ramp)
            peq_old = peq_zero
            _force, _jacobian, z, s, elapsed = bristle_elastic_coulomb_step(
                wp.vec2(0.0), dt, 0.0, kt, mu, release, z, s, elapsed
            )
        shoe.q_state[i] = q
        shoe.peq_prev[i] = peq_old
        shoe.deflection[i] = z
        shoe.stuck[i] = s
        shoe.dwell[i] = elapsed
        if shoe.has_surround != 0:
            shoe.surround[i] = 0.0


def _flight_screen(shoe: GroundShoe, body_q: wp.array[wp.transform]):
    """Advance each world's pristine/active/lifted shoe phase from its carrier pose.

    A world is airborne when every flight-candidate column clears the ground by
    :data:`FLIGHT_MARGIN_M`, which bounds every column's clearance (see
    :func:`flight_columns`). Pristine worlds start contact on first approach;
    active worlds become lifted when airborne, if dormancy is enabled; lifted
    worlds count skipped steps and replay them before contact resumes.
    """
    w, lane = wp.tid()
    if shoe.enabled[w] == 0:
        return
    phase = shoe.phase[w]
    if phase == _ACTIVE and shoe.dormancy == 0:
        return
    pending = shoe.pending[w]
    pose = body_q[shoe.carrier[w]]
    near = int(0)
    for index in range(lane, shoe.flight_columns.shape[0], _SCREEN_BLOCK):
        column = shoe.flight_columns[index]
        world = wp.transform_point(pose, shoe.anchor_local[column])
        if shoe.z_free_rigid[column] - world[2] > -FLIGHT_MARGIN_M:
            near = 1
    # Every lane joins the reduction, which also orders the reads above before any write.
    near = wp.tile_max(wp.tile(near))[0]
    if phase == _PRISTINE:
        if near != 0 and lane == 0:
            shoe.phase[w] = _ACTIVE
    elif phase == _ACTIVE:
        if near == 0 and lane == 0:
            shoe.phase[w] = _LIFTED
            shoe.pending[w] = 1
    elif near == 0:
        if lane == 0:
            shoe.pending[w] = pending + 1
    else:
        _replay_lifted(shoe, w, lane, pending)
        if lane == 0:
            shoe.phase[w] = _ACTIVE
            shoe.pending[w] = 0


def _ground_contact(
    shoe: GroundShoe,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    fraction: wp.array[wp.float64],
    invalid: wp.array[int],
):
    """Advance one world's columns and write its carrier wrench and compression screen."""
    w, lane = wp.tid()
    if shoe.enabled[w] == 0:
        return
    if shoe.phase[w] != _ACTIVE:
        # Airborne columns carry zero compression and zero traction.
        if lane == 0:
            body_f[shoe.carrier[w]] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
            fraction[w] = wp.float64(0.0)
            invalid[w] = 0
        return
    count = shoe.column_count
    p = shoe.world_params[w]
    dt = shoe.world_dt[w]
    decay, ramp = maxwell_coefficients(dt, p.tau_s)
    body = shoe.carrier[w]
    pose = body_q[body]
    twist = body_qd[body]
    com_local = shoe.body_com[body]
    com_world = wp.transform_point(pose, com_local)
    mu = shoe.settings[w, 1]
    kt_scale = shoe.settings[w, 2]
    release = shoe.settings[w, 5]
    shear = p.g_eq + p.g_eq2
    # Zero strain means unit stretch for every thickness, so the generic path's cached
    # zero-compression pressure serves all columns.
    peq_zero = shoe.zero_pressure[w * count][0]
    driven_comp = _Rows(-1.0)
    largest32 = float(0.0)
    nonfinite = int(0)
    for row in range(_ROWS):
        column = row * _BLOCK + lane
        wrench = _Wrench(0.0)
        if column < count:
            i = w * count + column
            anchor = shoe.anchor_local[column]
            world = wp.transform_point(pose, anchor)
            # The relaxed free top of _surround_write_free_top, kept in a register.
            z_free = shoe.z_free_rigid[column]
            if shoe.has_surround != 0:
                if shoe.driven[column] == 0:
                    z_free = world[2] + shoe.surround[i]
                elif shoe.fused_surround != 0:
                    # Driven surround values are read as next step's first-sweep inputs.
                    shoe.surround[i] = wp.max(z_free - world[2], 0.0)
            comp = z_free - world[2]
            if comp < 0.0:
                comp = 0.0
            thickness = shoe.rest_len[column]
            peq = peq_zero
            if comp != 0.0:
                peq = _hyperfoam(comp / thickness, p)
            q_old = shoe.q_state[i]
            peq_old = shoe.peq_prev[i]
            qn = _maxwell_update(q_old, peq, peq_old, p.overstress, decay, ramp)
            # Idle histories stay +0.0, so skipping equal stores leaves every bit unchanged.
            if qn != q_old:
                shoe.q_state[i] = qn
            if peq != peq_old:
                shoe.peq_prev[i] = peq
            if shoe.driven[column] != 0:
                # A float32 estimate brackets the float64 screen ratio; see below.
                estimate = comp / thickness
                if not wp.isfinite(estimate):
                    nonfinite = 1
                else:
                    driven_comp[row] = comp
                    largest32 = wp.max(largest32, estimate)
            point, _com, point_velocity, gap = contact_kinematics(pose, twist, com_local, anchor, shoe.ground_height, 1)
            column_area = shoe.area[column]
            reaction = normal_reaction(comp, peq + qn, column_area, p.normal_damping, point_velocity[2], gap, 1)
            kt = elastic_coulomb_stiffness(shear, column_area, thickness) * kt_scale
            deflection = shoe.deflection[i]
            stuck = shoe.stuck[i]
            dwell = shoe.dwell[i]
            tangential, _jacobian, z, s, elapsed = bristle_elastic_coulomb_step(
                wp.vec2(point_velocity[0], point_velocity[1]),
                dt,
                reaction,
                kt,
                mu,
                release,
                deflection,
                stuck,
                dwell,
            )
            if z[0] != deflection[0] or z[1] != deflection[1]:
                shoe.deflection[i] = z
            if s != stuck:
                shoe.stuck[i] = s
            if elapsed != dwell:
                shoe.dwell[i] = elapsed
            force = wp.vec3(tangential[0], tangential[1], reaction)
            torque = wp.cross(point - com_world, force)
            for k in range(3):
                wrench[k] = force[k]
                wrench[k + 3] = torque[k]
        for k in range(6):
            _wrench_shared((6 * row + k) * _BLOCK + lane, wrench[k], 1)
    _sync_threads()
    # _foundation_partial sums stride group g over columns g, g + G, ... and
    # vec3 addition is per component, so one lane per (component, group) keeps
    # every rounding step of the shared reduction.
    groups = shoe.groups
    if lane < 6 * groups:
        partial = float(0.0)
        component = lane // groups
        for column in range(lane % groups, count, groups):
            partial += _wrench_shared((6 * (column // _BLOCK) + component) * _BLOCK + column % _BLOCK, 0.0, 0)
        _reduce_shared(lane, partial, 1)
    # The float32 estimate is within a few ulps of the float64 ratio, so only
    # columns near its maximum can hold the exact maximum; divide just those in
    # float64 (slow on this hardware), as the CPU does with the artifact
    # thickness. Without traces only the screen decision matters, which the
    # estimate settles unless it is near the limit. Maxima are order independent.
    key = largest32
    if nonfinite != 0:
        key = wp.inf
    key = _warp_max(key)
    if lane % 32 == 0:
        _reduce_shared(_MAX_SLOT + lane // 32, key, 1)
    _sync_threads()
    estimate_max = float(0.0)
    for warp in range(_WARPS):
        estimate_max = wp.max(estimate_max, _reduce_shared(_MAX_SLOT + warp, 0.0, 0))
    if lane < 6:
        value = float(0.0)
        for group in range(groups):
            value += _reduce_shared(lane * groups + group, 0.0, 0)
        _reduce_shared(_TOTAL_SLOT + lane, value, 1)
    exact = wp.float64(estimate_max)
    if estimate_max > 0.0 and estimate_max < wp.inf and (shoe.exact_screen != 0 or estimate_max >= shoe.screen_limit):
        cutoff = estimate_max * (1.0 - 1.0e-5)
        largest = wp.float64(0.0)
        for row in range(_ROWS):
            comp = driven_comp[row]
            if comp >= 0.0:
                column = row * _BLOCK + lane
                if comp / shoe.rest_len[column] >= cutoff:
                    largest = wp.max(largest, wp.float64(comp) / shoe.screen_rest[column])
        largest = _warp_max64(largest)
        if lane % 32 == 0:
            _reduce_shared64(lane // 32, largest, 1)
        _sync_threads()
        exact = wp.float64(0.0)
        for warp in range(_WARPS):
            exact = wp.max(exact, _reduce_shared64(warp, wp.float64(0.0), 0))
    _sync_threads()
    if lane == 0:
        total = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
        for k in range(6):
            total[k] = _reduce_shared(_TOTAL_SLOT + k, 0.0, 0)
        body_f[body] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0)) + total
        if estimate_max == wp.inf:
            exact = wp.float64(0.0)
        fraction[w] = exact
        invalid[w] = int(estimate_max == wp.inf)


def _kernels(options: dict) -> tuple:
    # One unique module per kernel keeps each launch's block size from recompiling
    # the others; module options, including fast math, are part of its hash.
    return (
        wp.kernel(_flight_screen, module="unique", module_options=options),
        wp.kernel(_ground_surround, module="unique", module_options=options),
        wp.kernel(_ground_contact, module="unique", module_options=options, launch_bounds=(256, 2)),
    )


_KERNELS = {False: _kernels(_EXACT), True: _kernels(_FAST)}
_FLIGHT_COLUMNS: dict[bytes, np.ndarray] = {}


def flight_columns(foundation, *, step_rad: float = 1.0e-3, band_m: float = 2.0e-3) -> np.ndarray:
    """Return columns that can carry the deepest rigid penetration at any carrier pitch.

    For each sampled pitch about the carrier y axis, every column within
    ``band_m`` of the deepest ``z_free_rigid - z_anchor`` is kept. Between
    samples a column within 1 m of the carrier moves at most ``step_rad / 2``
    metres, so an omitted column always stays at least ``band_m - step_rad``
    shallower than the deepest candidate. Results are cached per geometry.
    """
    anchors = foundation.anchor_local.numpy().astype(np.float64)
    rigid = (foundation.z_free_rigid if foundation.free_column_count else foundation.z_free).numpy()[
        : foundation.column_count
    ]
    if np.linalg.norm(anchors, axis=1).max() > 1.0 or step_rad >= band_m:
        raise ValueError("Flight screening assumes anchors within 1 m and a band wider than the pitch step")
    key = anchors.tobytes() + rigid.tobytes() + np.array([step_rad, band_m]).tobytes()
    if key not in _FLIGHT_COLUMNS:
        angles = np.arange(-np.pi, np.pi, step_rad)
        keep = np.zeros(len(anchors), dtype=bool)
        for start in range(0, len(angles), 1024):
            a = angles[start : start + 1024, None]
            # A y-axis rotation maps an anchor's world height to cos * z + sin * x, for either sign.
            depth = rigid[None, :] - (np.cos(a) * anchors[None, :, 2] + np.sin(a) * anchors[None, :, 0])
            keep |= (depth >= depth.max(axis=1, keepdims=True) - band_m).any(axis=0)
        _FLIGHT_COLUMNS[key] = np.flatnonzero(keep).astype(np.int32)
    return _FLIGHT_COLUMNS[key]


def ground_shoe(
    foundation, screen_rest, *, exact_screen: bool = True, compression_limit: float = 1.0
) -> GroundShoe | None:
    """Return lean-contact arrays, or ``None`` when the foundation needs the generic path.

    Eligible beds are fused-CUDA, ground-plane, at most 1,024 columns, with the
    default parameter adapter running elastic-Coulomb friction in every world.

    Args:
        foundation: Fused foundation whose state the lean kernel advances.
        screen_rest: Float64 rest thickness per column [m] for the compression screen.
        exact_screen: Always report the exact float64 driven-compression fraction.
            Otherwise it is exact only near ``compression_limit``, which keeps
            every screen decision unchanged.
        compression_limit: Screen limit on driven compression over rest thickness.
    """
    adapter = foundation.friction_solver
    if (
        not foundation._fused_eligible
        or type(adapter) is not FrictionParameterAdapter
        or not adapter.is_default
        or foundation.column_count > 1024
        or not np.all(adapter.settings.numpy()[:, 0] == _ELASTIC_COULOMB)
        # The float32 screen prefilter needs representable positive thicknesses.
        or np.any(foundation.rest_len.numpy() <= 0.0)
        # The block reduction reserves six components per stride group.
        or foundation.reduction_groups > 16
    ):
        return None
    d = foundation.device
    shoe = GroundShoe()
    shoe.enabled = foundation.enabled
    shoe.carrier = foundation.carrier
    shoe.column_count = foundation.column_count
    shoe.groups = foundation.reduction_groups
    shoe.driven = foundation.driven
    shoe.anchor_local = foundation.anchor_local
    shoe.has_surround = int(bool(foundation.free_column_count))
    shoe.z_free_rigid = foundation.z_free_rigid if foundation.free_column_count else foundation.z_free
    shoe.rest_len = foundation.rest_len
    shoe.area = foundation.area
    shoe.world_params = foundation.world_params
    shoe.world_dt = foundation.world_dt
    shoe.q_state = foundation.q_state
    shoe.peq_prev = foundation.peq_prev
    shoe.surround = foundation.surround_compression if foundation.free_column_count else foundation.z_free
    shoe.body_com = foundation.body_com
    shoe.settings = adapter.settings
    shoe.deflection = adapter.deflection
    shoe.stuck = foundation.tangent_stuck
    shoe.dwell = foundation.tangent_dwell
    shoe.ground_height = float(foundation.ground_height_m)
    shoe.screen_rest = wp.array(np.asarray(screen_rest, dtype=np.float64), dtype=wp.float64, device=d)
    shoe.exact_screen = int(bool(exact_screen))
    # A float32 estimate this far below the limit cannot round across it.
    shoe.screen_limit = float(compression_limit) - 1.0e-3
    driven = foundation.driven.numpy()
    free = np.flatnonzero(driven == 0)
    # One lane per undriven column, in whole warps; larger surrounds keep the
    # generic fused sweep kernel.
    shoe.fused_surround = int(0 < len(free) <= 1024)
    shoe.surround_lanes = 32 * max(1, -(-len(free) // 32))
    slots = np.full(shoe.surround_lanes, -1, dtype=np.int32)
    slot_of = np.full(foundation.column_count, -1, dtype=np.int32)
    if shoe.fused_surround:
        slots[: len(free)] = free
        slot_of[free] = np.arange(len(free))
    shoe.free_column = wp.array(slots, dtype=int, device=d)
    shoe.column_slot = wp.array(slot_of, dtype=int, device=d)
    shoe.neighbors = foundation.neighbors
    shoe.zero_pressure = foundation.zero_pressure
    shoe.surround_decay = foundation.surround_decay
    shoe.surround_gain = foundation.surround_gain
    shoe.world_relaxation = foundation.world_relaxation
    cfg = foundation.surround
    if cfg is not None:
        shoe.coupling_scale = float(cfg.coupling_scale)
        shoe.attachment = float(cfg.attachment_n_m)
        shoe.max_strain = float(cfg.max_strain)
        shoe.carrier_bond = int(bool(cfg.carrier_bond))
        shoe.sweeps = int(cfg.sweeps)
    shoe.phase = wp.zeros(foundation.world_count, dtype=int, device=d)
    shoe.pending = wp.zeros(foundation.world_count, dtype=int, device=d)
    # Lifted replay assumes the carrier bond zeroes an airborne surround, which the
    # generic sweep fallback would otherwise have to skip as well.
    shoe.dormancy = int(not foundation.free_column_count or (shoe.fused_surround and shoe.carrier_bond != 0))
    shoe.flight_columns = wp.array(flight_columns(foundation), dtype=int, device=d)
    return shoe


def reset_ground_shoe(shoe: GroundShoe) -> None:
    """Mark every world pristine after :meth:`FoundationFused.reset` cleared its history."""
    shoe.phase.zero_()
    shoe.pending.zero_()


def apply_ground_shoe(
    foundation, shoe: GroundShoe, state, fraction: wp.array, invalid: wp.array, *, fast: bool = False
) -> None:
    """Advance shoe phases, relax the passive surround, then advance lean contact.

    Args:
        foundation: Fused foundation that owns the shoe state.
        shoe: Lean-contact arrays from :func:`ground_shoe`.
        state: Carrier state with ``body_q``, ``body_qd`` and ``body_f``.
        fraction: Driven-compression screen per world, written here.
        invalid: Nonfinite-compression flag per world, written here.
        fast: Compile the shared laws with fast math. Results then agree with the
            exact kernels only to float32 rounding of the approximate intrinsics.
    """
    screen, surround, contact = _KERNELS[bool(fast)]
    wp.launch_tiled(
        screen,
        dim=foundation.world_count,
        block_dim=_SCREEN_BLOCK,
        inputs=[shoe, state.body_q],
        device=foundation.device,
    )
    if foundation.free_column_count and not shoe.fused_surround:
        foundation._relax_surround_block(state, None, mask_worlds=True)
    else:
        foundation._use_timesteps(None)
        foundation._refresh_surround_constants(None)
        if shoe.fused_surround:
            wp.launch_tiled(
                surround,
                dim=foundation.world_count,
                block_dim=shoe.surround_lanes,
                inputs=[shoe, state.body_q],
                device=foundation.device,
            )
    wp.launch_tiled(
        contact,
        dim=foundation.world_count,
        block_dim=_BLOCK,
        inputs=[shoe, state.body_q, state.body_qd, state.body_f, fraction, invalid],
        device=foundation.device,
    )
