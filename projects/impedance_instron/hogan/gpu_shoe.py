# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lean ground contact of the runner shoe: physics state only, one block per world.

The generic fused foundation also maintains pressure transfer, diagnostics, and
cached per-column forces for its many consumers. The runner only reads the
carrier wrench and the driven-compression screen, so these kernels evaluate the
same shared material, Maxwell, surround, and elastic-Coulomb laws for each
column and keep everything else in registers or shared memory. Per column they
read and write only surround, Maxwell, and bristle history; the wrench uses the
shared fixed-order reduction. A world whose history is still pristine and whose
columns all clear the ground is skipped, because the shared laws leave it
exactly at rest. Carrier forces and histories are bitwise identical to
:class:`FoundationFused`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from projects.digital_shoe.contact import _surround_balance_pressures, contact_kinematics, normal_reaction
from projects.digital_shoe.friction_maxwell import bristle_elastic_coulomb_step, elastic_coulomb_stiffness
from projects.digital_shoe.friction_parameter_adapter import FrictionParameterAdapter
from projects.digital_shoe.material import HYPERFOAM_ALPHA_FLOOR, maxwell_coefficients, maxwell_step, ogden_hill_term
from projects.digital_shoe.runtime import FoundationParams, _pasternak_coupling

# Match the shared float32 shoe runtime, not the float64 leg module.
wp.set_module_options({"enable_backward": False, "fuse_fp": True})

_BLOCK = wp.constant(256)
_ROWS = wp.constant(4)
_Rows = wp.types.vector(4, float)
_Wrench = wp.types.vector(6, float)
_WRENCH_ROWS = wp.constant(24)
_ELASTIC_COULOMB = 9
_DRIVEN = wp.constant(-2)
FLIGHT_MARGIN_M = 5.0e-4
"""Rigid clearance below which a pristine shoe is still integrated [m]."""


@wp.struct
class GroundShoe:
    """Persistent arrays the lean contact kernel reads from a fused foundation."""

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
    touched: wp.array[int]
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
    only undriven columns enter the shared tile. Sums keep the original side
    order, which leaves every compression bitwise unchanged.
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
    for sweep in range(shoe.sweeps):
        shared = wp.tile(c)
        next_c = c
        if column >= 0:
            pull = float(0.0)
            for side in range(4):
                slot = slots[side]
                if slot != -1:
                    value = new[side]
                    if slot >= 0:
                        value = shared[slot]
                    elif sweep == 0:
                        value = old[side]
                    pull += couplings[side] * (value - c)
            if c == 0.0:
                peq = shoe.zero_pressure[i]
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
                    peq[0],
                    peq[1],
                )
            else:
                # The Newton step of contact.surround_balance with the shared material law.
                step = 1.0e-3 * thickness
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
                    _hyperfoam(c / thickness, p),
                    _hyperfoam((c + step) / thickness, p),
                )
        c = next_c
    if column >= 0:
        shoe.surround[i] = c


@wp.kernel(module="unique")
def _ground_surround(shoe: GroundShoe, body_q: wp.array[wp.transform]):
    """Relax one world's undriven columns before :func:`_ground_contact` reads them.

    Launch with one lane per undriven column, rounded up to whole warps. A unique
    module keeps that bed-specific block size from recompiling the contact kernel.
    """
    w, lane = wp.tid()
    if shoe.enabled[w] == 0 or shoe.touched[w] == 0:
        return
    _relax_free_column(shoe, w, lane, body_q[shoe.carrier[w]])


@wp.kernel
def _flight_screen(shoe: GroundShoe, body_q: wp.array[wp.transform]):
    """Mark a world touched once any candidate column comes within the flight margin.

    Until then its shoe history is exactly zero and no column penetrates, so the
    shared laws return zero compression, pressure, traction and history updates;
    the contact kernels skip such worlds and write the zero wrench directly.
    """
    w = wp.tid()
    if shoe.enabled[w] == 0 or shoe.touched[w] != 0:
        return
    pose = body_q[shoe.carrier[w]]
    for index in range(shoe.flight_columns.shape[0]):
        column = shoe.flight_columns[index]
        world = wp.transform_point(pose, shoe.anchor_local[column])
        if shoe.z_free_rigid[column] - world[2] > -FLIGHT_MARGIN_M:
            shoe.touched[w] = 1
            return


@wp.kernel(launch_bounds=(256, 2))
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
    if shoe.touched[w] == 0:
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
    # Zero strain means unit stretch for every thickness, so one value serves all columns.
    peq_zero = _hyperfoam(0.0, p)
    # Force and torque components per column, reduced in the shared fixed order below.
    # Cross-lane reads must come from shared storage; register-tile extraction
    # synchronizes the block per element.
    columns = wp.tile_empty(shape=(_WRENCH_ROWS, _BLOCK), dtype=float, storage="shared")
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
            qn = maxwell_step(q_old, peq, peq_old, p.overstress, decay, ramp)
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
        wp.tile_assign(columns, wp.tile(wrench), offset=(6 * row, 0))
    # _foundation_partial sums stride group g over columns g, g + G, ... and
    # vec3 addition is per component, so one lane per (component, group) keeps
    # every rounding step of the shared reduction.
    groups = shoe.groups
    partial = float(0.0)
    if lane < 6 * groups:
        component = lane // groups
        for column in range(lane % groups, count, groups):
            partial += columns[6 * (column // _BLOCK) + component, column % _BLOCK]
    partials = wp.tile_empty(shape=_BLOCK, dtype=float, storage="shared")
    wp.tile_assign(partials, wp.tile(partial), offset=(0,))
    # The float32 estimate is within a few ulps of the float64 ratio, so only
    # columns near its maximum can hold the exact maximum; divide just those in
    # float64 (slow on this hardware), as the CPU does with the artifact
    # thickness. Every lane runs the register-tile extractions together.
    key = largest32
    if nonfinite != 0:
        key = wp.inf
    estimate_max = wp.tile_max(wp.tile(key))[0]
    exact = wp.float64(0.0)
    if estimate_max > 0.0 and estimate_max < wp.inf:
        cutoff = estimate_max * (1.0 - 1.0e-5)
        largest = wp.float64(0.0)
        for row in range(_ROWS):
            comp = driven_comp[row]
            if comp >= 0.0:
                column = row * _BLOCK + lane
                if comp / shoe.rest_len[column] >= cutoff:
                    largest = wp.max(largest, wp.float64(comp) / shoe.screen_rest[column])
        exact = wp.tile_max(wp.tile(largest))[0]
    if lane == 0:
        total = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
        for k in range(6):
            value = float(0.0)
            for group in range(groups):
                value += partials[k * groups + group]
            total[k] = value
        body_f[body] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0)) + total
        fraction[w] = exact
        invalid[w] = int(estimate_max == wp.inf)


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


def ground_shoe(foundation, screen_rest) -> GroundShoe | None:
    """Return lean-contact arrays, or ``None`` when the foundation needs the generic path.

    Eligible beds are fused-CUDA, ground-plane, at most 1,024 columns, with the
    default parameter adapter running elastic-Coulomb friction in every world.

    Args:
        foundation: Fused foundation whose state the lean kernel advances.
        screen_rest: Float64 rest thickness per column [m] for the compression screen.
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
    shoe.touched = wp.zeros(foundation.world_count, dtype=int, device=d)
    shoe.flight_columns = wp.array(flight_columns(foundation), dtype=int, device=d)
    return shoe


def reset_ground_shoe(shoe: GroundShoe) -> None:
    """Mark every world pristine after :meth:FoundationFused.reset cleared its history."""
    shoe.touched.zero_()


def apply_ground_shoe(foundation, shoe: GroundShoe, state, fraction: wp.array, invalid: wp.array) -> None:
    """Relax the passive surround, then advance lean contact for enabled worlds."""
    wp.launch(_flight_screen, dim=foundation.world_count, inputs=[shoe, state.body_q], device=foundation.device)
    if foundation.free_column_count and not shoe.fused_surround:
        foundation._relax_surround_block(state, None, mask_worlds=True)
    else:
        foundation._use_timesteps(None)
        foundation._refresh_surround_constants(None)
        if shoe.fused_surround:
            wp.launch_tiled(
                _ground_surround,
                dim=foundation.world_count,
                block_dim=shoe.surround_lanes,
                inputs=[shoe, state.body_q],
                device=foundation.device,
            )
    wp.launch_tiled(
        _ground_contact,
        dim=foundation.world_count,
        block_dim=_BLOCK,
        inputs=[shoe, state.body_q, state.body_qd, state.body_f, fraction, invalid],
        device=foundation.device,
    )
