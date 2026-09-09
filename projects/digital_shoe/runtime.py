# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable, GPU-native runtime for an identified digital shoe foundation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class ShoeMaterial:
    """Effective intact-shoe constitutive parameters used by the runtime."""

    instantaneous_shear_modulus_pa: float
    hyperfoam_exponent: float
    equilibrium_fraction: float
    pasternak_n_per_m: float
    effective_poisson_ratio: float = 0.30
    maxwell_relaxation_time_s: float = 0.08

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not np.all(np.isfinite(values)):
            raise ValueError("shoe material parameters must be finite")
        if self.instantaneous_shear_modulus_pa <= 0.0 or self.hyperfoam_exponent <= 0.0:
            raise ValueError("shear modulus and Hyperfoam exponent must be positive")
        if not 0.0 < self.equilibrium_fraction <= 1.0:
            raise ValueError("equilibrium fraction must be in (0, 1]")
        if self.pasternak_n_per_m < 0.0:
            raise ValueError("Pasternak coupling must be nonnegative")
        if not -1.0 < self.effective_poisson_ratio < 0.5:
            raise ValueError("effective Poisson ratio must be in (-1, 0.5)")
        if self.maxwell_relaxation_time_s <= 0.0:
            raise ValueError("Maxwell relaxation time must be positive")


@wp.struct
class FoundationParams:
    """Device-side constitutive and contact constants for the column bed."""

    g_eq: wp.float32  # equilibrium shear modulus G_inst * equilibrium_fraction [Pa]
    alpha: wp.float32  # Hyperfoam exponent
    beta: wp.float32  # poisson / (1 - 2 poisson)
    one_minus_two_poisson: wp.float32  # volumetric stretch exponent
    tau_s: wp.float32  # Maxwell relaxation time [s]
    overstress: wp.float32  # (1 - equilibrium_fraction) / equilibrium_fraction
    pasternak: wp.float32  # Pasternak lateral coupling [N/m]
    inv_h2: wp.float32  # 1 / spacing^2 [1/m^2]
    stretch_floor: wp.float32  # minimum stretch (foam densification limit)
    normal_damping: wp.float32  # per-column Kelvin-Voigt normal damping [N.s/m]
    friction_kt: wp.float32  # bristle tangential stiffness [N/m per column]
    friction_kv: wp.float32  # bristle tangential damping [N.s/m per column]
    mu: wp.float32  # Coulomb friction coefficient


@wp.func
def _hyperfoam_pressure(strain: wp.float32, p: FoundationParams) -> wp.float32:
    """Positive uniaxial compression pressure from the first-order Hyperfoam law."""
    stretch = 1.0 - strain
    if stretch < p.stretch_floor:
        stretch = p.stretch_floor
    volume_ratio = wp.pow(stretch, p.one_minus_two_poisson)
    return 2.0 * p.g_eq / (p.alpha * stretch) * (wp.pow(volume_ratio, -p.alpha * p.beta) - wp.pow(stretch, p.alpha))


@wp.kernel
def foundation_pressure(
    carrier: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    params: FoundationParams,
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
):
    """Compression, Hyperfoam equilibrium pressure, and real-time Maxwell overstress."""
    i = wp.tid()
    world = wp.transform_point(body_q[carrier], anchor_local[i])
    comp = z_free[i] - world[2]
    if comp < 0.0:
        comp = 0.0
    compression[i] = comp
    strain = comp / rest_len[i]
    peq = _hyperfoam_pressure(strain, params)
    decay = wp.exp(-dt / params.tau_s)
    ramp = params.tau_s * (1.0 - decay) / dt
    qn = decay * q_state[i] + params.overstress * ramp * (peq - peq_prev[i])
    q_state[i] = qn
    peq_prev[i] = peq
    base_pressure[i] = peq + qn


@wp.kernel
def foundation_apply(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    tangent_anchor: wp.array[wp.vec2],
    tangent_stuck: wp.array[wp.int32],
    params: FoundationParams,
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    column_force: wp.array[wp.vec3],
    pressed_force: wp.array[wp.float32],
):
    """Pasternak coupling, per-column wrench into ``body_f``, and force diagnostics."""
    i = wp.tid()
    ci = compression[i]
    lap = -4.0 * ci
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            lap += compression[j]
        elif j == -1:
            lap += ci  # natural (zero-gradient) footprint boundary
    lap *= params.inv_h2

    # Clamp only the unilateral ground reaction. Clamping the combined pressure
    # would also clip the shear-layer flux and invent support under columns that
    # carry no compression at all.
    ground = base_pressure[i]
    if ground < 0.0:
        ground = 0.0
    fn = (ground - params.pasternak * lap) * area[i]

    q_body = body_q[carrier]
    world = wp.transform_point(q_body, anchor_local[i])
    com_world = wp.transform_point(q_body, body_com[carrier])
    r = world - com_world
    vel = body_qd[carrier]
    point_vel = wp.spatial_top(vel) + wp.cross(wp.spatial_bottom(vel), r)

    if ci > 0.0:
        fn = fn - params.normal_damping * point_vel[2]
    # A column may now transmit a small pull where the shear layer lifts it.
    # Friction still needs a nonnegative cone, so it uses the pressed part only.
    pressed = fn
    if pressed < 0.0:
        pressed = 0.0

    # Anchored bristle (elastoplastic) Coulomb friction: a per-column tangential
    # spring pulls the contact patch back toward a world stick point, so a planted
    # patch holds (static regime, zero drift) and carries braking/propulsion shear
    # without needing a slip velocity. When the spring force would exceed the cone
    # mu*fn it saturates and the anchor slides forward onto the cone (kinetic regime).
    p_t = wp.vec2(world[0], world[1])
    f_max = params.mu * pressed
    f_tan = wp.vec2(0.0, 0.0)
    if pressed <= 0.0 or params.friction_kt <= 0.0:
        tangent_anchor[i] = p_t
        tangent_stuck[i] = 0
    else:
        if tangent_stuck[i] == 0:
            tangent_anchor[i] = p_t  # fresh contact: seat with no pre-stretch
            tangent_stuck[i] = 1
        v_tan = wp.vec2(point_vel[0], point_vel[1])
        f_tan = -params.friction_kt * (p_t - tangent_anchor[i]) - params.friction_kv * v_tan
        mag = wp.length(f_tan)
        if mag > f_max and mag > 1.0e-9:
            f_tan = f_tan * (f_max / mag)
            tangent_anchor[i] = p_t + f_tan / params.friction_kt  # slide the anchor onto the cone

    force = wp.vec3(f_tan[0], f_tan[1], fn)
    column_force[i] = force
    wp.atomic_add(body_f, carrier, wp.spatial_vector(force, wp.cross(r, force)))
    wp.atomic_add(normal_force, 0, fn)
    wp.atomic_add(cop_moment, 0, wp.vec3(world[0] * pressed, world[1] * pressed, 0.0))
    wp.atomic_add(pressed_force, 0, pressed)
    wp.atomic_add(resultant_force, 0, force)
    wp.atomic_add(resultant_moment_origin, 0, wp.cross(world, force))
    wp.atomic_add(contact_power, 0, wp.dot(force, point_vel))
    wp.atomic_max(max_compression, 0, ci)
    if ci > 0.0:
        wp.atomic_add(active_count, 0, 1)


@wp.kernel
def cycle_force(
    compression: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    dt_s: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
    fraction: wp.float32,
    tau_s: wp.float32,
    force_out: wp.array[wp.float32],
):
    """Sum one column's periodic ground reaction into every frame of a cycle.

    Each thread owns a column and walks the cycle twice: the first pass finds
    the periodic overstress state, the second accumulates force. The clamp keeps
    the reaction unilateral, matching the live foundation.
    """
    i = wp.tid()
    frames = compression.shape[0]
    thickness = slack[i]
    state = float(0.0)
    decay_product = float(1.0)
    previous = _hyperfoam_pressure(compression[frames - 1, i] / thickness, params)
    for frame in range(frames):
        equilibrium = _hyperfoam_pressure(compression[frame, i] / thickness, params)
        decay = wp.exp(-dt_s[frame] / tau_s)
        ramp = tau_s * (1.0 - decay) / dt_s[frame]
        state = decay * state + fraction * ramp * (equilibrium - previous)
        decay_product *= decay
        previous = equilibrium
    state = state / (1.0 - decay_product)
    previous = _hyperfoam_pressure(compression[frames - 1, i] / thickness, params)
    for frame in range(frames):
        equilibrium = _hyperfoam_pressure(compression[frame, i] / thickness, params)
        decay = wp.exp(-dt_s[frame] / tau_s)
        ramp = tau_s * (1.0 - decay) / dt_s[frame]
        state = decay * state + fraction * ramp * (equilibrium - previous)
        previous = equilibrium
        wp.atomic_add(force_out, frame, area * wp.max(equilibrium + state, 0.0))


@wp.kernel
def surround_sweep(
    compression_in: wp.array2d[wp.float32],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
    coupling: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    compression_out: wp.array2d[wp.float32],
):
    """Relax one untouched column toward its own quasi-static balance.

    Each thread owns one frame and one column. Driven columns pass straight
    through, so the indenter keeps its imposed compression while the surrounding
    foam settles against neighbour shear, its unilateral ground reaction and its
    bond to the shoe. Frames are independent, so a whole test relaxes at once.
    """
    frame, i = wp.tid()
    if driven[i] != 0:
        compression_out[frame, i] = compression_in[frame, i]
        return
    c = compression_in[frame, i]
    pull = float(0.0)
    links = float(0.0)
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            pull += coupling * (compression_in[frame, j] - c)
            links += 1.0
    thickness = slack[i]
    reaction = area * wp.max(_hyperfoam_pressure(c / thickness, params), 0.0)
    step = 1.0e-3 * thickness
    ahead = area * wp.max(_hyperfoam_pressure((c + step) / thickness, params), 0.0)
    stiffness = wp.max((ahead - reaction) / step + attachment + links * coupling, 1.0e-9)
    residual = reaction + attachment * c - pull
    compression_out[frame, i] = wp.clamp(c - residual / stiffness, 0.0, max_strain * thickness)


def relax_surround(
    driven_compression: np.ndarray,
    driven: np.ndarray,
    neighbors: np.ndarray,
    slack_m: np.ndarray,
    params: FoundationParams,
    *,
    area_m2: float,
    spacing_m: float,
    pasternak_n_per_m: float,
    attachment_n_m: float,
    max_strain: float,
    sweeps: int,
    device=None,
) -> np.ndarray:
    """Return whole-midsole compression for every frame of a test.

    The identification and the live runtime therefore share one contact model
    and one geometry: the indenter drives its columns and the rest relax.

    Args:
        driven_compression: Imposed compression of the driven columns [m],
            shape ``[frames, driven_count]``.
        driven: Nonzero for columns the indenter drives, shape ``[column_count]``.
        neighbors: Four in-plane neighbour indices; negative is a free edge.
        slack_m: Rest thickness per column [m].
        params: Device-side constitutive constants.
        area_m2: Tributary area per column [m^2].
        spacing_m: Column grid spacing [m].
        pasternak_n_per_m: Identified lateral shear coupling [N/m].
        attachment_n_m: Assumed vertical bond of untouched foam to the shoe [N/m].
        max_strain: Compression limit as a fraction of rest thickness.
        sweeps: Relaxation sweeps per solve.

    Returns:
        Device compression for every column and frame [m], shape
        ``[frames, column_count]``, ready for :func:`cycle_force`.
    """
    driven = np.ascontiguousarray(driven, np.int32)
    frames = len(driven_compression)
    count = len(slack_m)
    start = np.zeros((frames, count), np.float32)
    start[:, driven != 0] = driven_compression
    current = wp.array(start, dtype=wp.float32, device=device)
    scratch = wp.zeros_like(current)
    driven_device = wp.array(driven, dtype=wp.int32, device=device)
    neighbor_device = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)
    slack_device = wp.array(np.ascontiguousarray(slack_m, np.float32), dtype=wp.float32, device=device)
    coupling = pasternak_n_per_m * area_m2 / spacing_m**2
    for _ in range(max(sweeps, 0)):
        wp.launch(
            surround_sweep,
            dim=(frames, count),
            inputs=[
                current,
                driven_device,
                neighbor_device,
                slack_device,
                params,
                float(area_m2),
                float(coupling),
                float(attachment_n_m),
                float(max_strain),
                scratch,
            ],
            device=device,
        )
        current, scratch = scratch, current
    return current


@wp.kernel
def foundation_reset(
    carrier: wp.int32,
    clear_body_force: wp.int32,
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    pressed_force: wp.array[wp.float32],
):
    """Zero the per-substep foundation accumulators (and optionally the carrier wrench).

    Folds the reduction resets into a single one-thread kernel launch. Each of the
    accumulator memsets is a graph node that dwarfs the actual 611-column physics, so
    collapsing them to one node is the dominant cost saving for the captured attached
    loop. ``clear_body_force`` also zeros the carrier wrench so the attached loop needs
    no separate :meth:`newton.State.clear_forces` launch.
    """
    normal_force[0] = 0.0
    cop_moment[0] = wp.vec3(0.0, 0.0, 0.0)
    active_count[0] = 0
    resultant_force[0] = wp.vec3(0.0, 0.0, 0.0)
    resultant_moment_origin[0] = wp.vec3(0.0, 0.0, 0.0)
    contact_power[0] = 0.0
    max_compression[0] = 0.0
    pressed_force[0] = 0.0
    if clear_body_force != 0:
        body_f[carrier] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Foundation driver
# ---------------------------------------------------------------------------
@dataclass
class FoundationConfig:
    """Tunable dynamic parameters layered on the calibrated constitutive law.

    The Instron replay leaves ``normal_damping``, ``friction_stiffness`` and
    ``friction`` at zero and keeps ``stretch_floor`` below the calibration's peak
    strain so the collected loop reproduces the fitted force-displacement response
    exactly. The free-body scenarios add foam damping and an anchored bristle
    (elastoplastic) Coulomb friction: ``friction_stiffness`` is the per-column
    tangential spring that holds a planted contact patch (true stick), ``friction``
    is its stabilising damping, and ``mu`` bounds the tangential force at the cone
    ``mu * fn`` (slip).
    """

    stretch_floor: float = 0.05
    normal_damping: float = 0.0
    friction_stiffness: float = 0.0
    friction: float = 0.0
    mu: float = 0.0


class MidsoleFoundation:
    """Live Warp elastic-foundation force model attached to one carrier body.

    Args:
        anchor_local: Column attachment points in the carrier body frame [m],
            shape ``[column_count, 3]``.
        z_free: World height of each uncompressed foam column top [m], shape
            ``[column_count]``.
        rest_len: Column rest thickness [m], shape ``[column_count]``.
        area: Tributary area per column [m^2], shape ``[column_count]``.
        neighbors: Pasternak 4-neighbour indices, shape ``[column_count, 4]``.
        spacing_m: Column grid spacing [m].
        material: Calibrated :class:`ShoeMaterial`.
        carrier_body: Index of the rigid body carrying the foundation.
        body_com: Model center-of-mass array (``model.body_com``).
        config: Dynamic :class:`FoundationConfig`.
        device: Warp device.
    """

    def __init__(
        self,
        anchor_local: np.ndarray,
        z_free: np.ndarray,
        rest_len: np.ndarray,
        area: np.ndarray,
        neighbors: np.ndarray,
        spacing_m: float,
        material: ShoeMaterial,
        carrier_body: int,
        body_com,
        config: FoundationConfig | None = None,
        device=None,
    ) -> None:
        config = config or FoundationConfig()
        self.device = device
        self.carrier = int(carrier_body)
        self.body_com = body_com
        self.column_count = int(len(rest_len))

        params = FoundationParams()
        params.g_eq = material.instantaneous_shear_modulus_pa * material.equilibrium_fraction
        params.alpha = material.hyperfoam_exponent
        poisson = float(getattr(material, "effective_poisson_ratio", 0.30))
        params.beta = poisson / (1.0 - 2.0 * poisson)
        params.one_minus_two_poisson = 1.0 - 2.0 * poisson
        params.tau_s = float(getattr(material, "maxwell_relaxation_time_s", 0.08))
        params.overstress = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction
        params.pasternak = material.pasternak_n_per_m
        params.inv_h2 = 1.0 / spacing_m**2
        params.stretch_floor = config.stretch_floor
        params.normal_damping = config.normal_damping
        params.friction_kt = config.friction_stiffness
        params.friction_kv = config.friction
        params.mu = config.mu
        self.params = params

        m = self.column_count
        self.anchor_local = wp.array(np.ascontiguousarray(anchor_local, np.float32), dtype=wp.vec3, device=device)
        self.z_free = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=device)
        self.rest_len = wp.array(np.ascontiguousarray(rest_len, np.float32), dtype=wp.float32, device=device)
        self.area = wp.array(np.ascontiguousarray(area, np.float32), dtype=wp.float32, device=device)
        self.neighbors = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)
        self.q_state = wp.zeros(m, dtype=wp.float32, device=device)
        self.peq_prev = wp.zeros(m, dtype=wp.float32, device=device)
        self.compression = wp.zeros(m, dtype=wp.float32, device=device)
        self.base_pressure = wp.zeros(m, dtype=wp.float32, device=device)
        self.tangent_anchor = wp.zeros(m, dtype=wp.vec2, device=device)  # world XY stick point
        self.tangent_stuck = wp.zeros(m, dtype=wp.int32, device=device)  # 1 while the bristle grips
        self.normal_force = wp.zeros(1, dtype=wp.float32, device=device)
        self.cop_moment = wp.zeros(1, dtype=wp.vec3, device=device)
        self.active = wp.zeros(1, dtype=wp.int32, device=device)
        self.resultant_force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.resultant_moment_origin = wp.zeros(1, dtype=wp.vec3, device=device)
        self.contact_power = wp.zeros(1, dtype=wp.float32, device=device)
        self.max_compression = wp.zeros(1, dtype=wp.float32, device=device)
        self.column_force = wp.zeros(m, dtype=wp.vec3, device=device)
        self.pressed_force = wp.zeros(1, dtype=wp.float32, device=device)

    def reset(self) -> None:
        """Clear the viscoelastic overstress history and release the friction bristles."""
        self.q_state.zero_()
        self.peq_prev.zero_()
        self.tangent_stuck.zero_()

    def apply(self, state, dt: float, clear_body_force: bool = False) -> None:
        """Accumulate the foundation wrench into ``state.body_f`` for one substep.

        Args:
            state: Simulation state supplying the carrier pose/velocity and receiving the wrench.
            dt: Substep duration [s].
            clear_body_force: Also zero the carrier's ``body_f`` in the fused reset launch, so a
                caller that only loads the foundation wrench can skip a separate
                :meth:`newton.State.clear_forces`. Leave False when other forces are staged into
                ``body_f`` before this call (e.g. an external probe load).
        """
        wp.launch(
            foundation_reset,
            dim=1,
            inputs=[
                self.carrier,
                int(clear_body_force),
                state.body_f,
                self.normal_force,
                self.cop_moment,
                self.active,
                self.resultant_force,
                self.resultant_moment_origin,
                self.contact_power,
                self.max_compression,
                self.pressed_force,
            ],
            device=self.device,
        )
        wp.launch(
            foundation_pressure,
            dim=self.column_count,
            inputs=[
                self.carrier,
                dt,
                state.body_q,
                self.anchor_local,
                self.z_free,
                self.rest_len,
                self.params,
                self.q_state,
                self.peq_prev,
                self.compression,
                self.base_pressure,
            ],
            device=self.device,
        )
        wp.launch(
            foundation_apply,
            dim=self.column_count,
            inputs=[
                self.carrier,
                state.body_q,
                state.body_qd,
                self.body_com,
                self.anchor_local,
                self.area,
                self.neighbors,
                self.compression,
                self.base_pressure,
                self.tangent_anchor,
                self.tangent_stuck,
                self.params,
                state.body_f,
                self.normal_force,
                self.cop_moment,
                self.active,
                self.resultant_force,
                self.resultant_moment_origin,
                self.contact_power,
                self.max_compression,
                self.column_force,
                self.pressed_force,
            ],
            device=self.device,
        )

    def diagnostics(self) -> dict[str, float]:
        """Return the last substep's total normal force, center of pressure, and active count."""
        fz = float(self.normal_force.numpy()[0])
        pressed = float(self.pressed_force.numpy()[0])
        moment = self.cop_moment.numpy()[0]
        cop = (float(moment[0] / pressed), float(moment[1] / pressed)) if pressed > 1.0e-9 else (0.0, 0.0)
        return {
            "normal_force_n": fz,
            "pressed_force_n": pressed,
            "cop_x_m": cop[0],
            "cop_y_m": cop[1],
            "active_columns": int(self.active.numpy()[0]),
        }
