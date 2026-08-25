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

    pressure = base_pressure[i] - params.pasternak * lap
    if pressure < 0.0:
        pressure = 0.0
    fn = pressure * area[i]

    q_body = body_q[carrier]
    world = wp.transform_point(q_body, anchor_local[i])
    com_world = wp.transform_point(q_body, body_com[carrier])
    r = world - com_world
    vel = body_qd[carrier]
    point_vel = wp.spatial_top(vel) + wp.cross(wp.spatial_bottom(vel), r)

    if ci > 0.0:
        fn = fn - params.normal_damping * point_vel[2]
    if fn < 0.0:
        fn = 0.0

    # Anchored bristle (elastoplastic) Coulomb friction: a per-column tangential
    # spring pulls the contact patch back toward a world stick point, so a planted
    # patch holds (static regime, zero drift) and carries braking/propulsion shear
    # without needing a slip velocity. When the spring force would exceed the cone
    # mu*fn it saturates and the anchor slides forward onto the cone (kinetic regime).
    p_t = wp.vec2(world[0], world[1])
    f_max = params.mu * fn
    f_tan = wp.vec2(0.0, 0.0)
    if fn <= 0.0 or params.friction_kt <= 0.0:
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
    wp.atomic_add(cop_moment, 0, wp.vec3(world[0] * fn, world[1] * fn, 0.0))
    wp.atomic_add(resultant_force, 0, force)
    wp.atomic_add(resultant_moment_origin, 0, wp.cross(world, force))
    wp.atomic_add(contact_power, 0, wp.dot(force, point_vel))
    wp.atomic_max(max_compression, 0, ci)
    if ci > 0.0:
        wp.atomic_add(active_count, 0, 1)


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
            ],
            device=self.device,
        )

    def diagnostics(self) -> dict[str, float]:
        """Return the last substep's total normal force, center of pressure, and active count."""
        fz = float(self.normal_force.numpy()[0])
        moment = self.cop_moment.numpy()[0]
        cop = (float(moment[0] / fz), float(moment[1] / fz)) if fz > 1.0e-9 else (0.0, 0.0)
        return {
            "normal_force_n": fz,
            "cop_x_m": cop[0],
            "cop_y_m": cop[1],
            "active_columns": int(self.active.numpy()[0]),
        }


