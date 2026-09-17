# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse the shoe law while keeping a world's relaxation sweeps in one CUDA block."""

from __future__ import annotations

import numpy as np
import warp as wp

from projects.digital_shoe.contact import _surround_balance_pressures
from projects.digital_shoe.runtime import (
    FoundationParams,
    MidsoleFoundation,
    _foundation_column_forces,
    _foundation_finalize,
    _foundation_partial,
    _foundation_pressure,
    _hyperfoam_pressure,
    _pasternak_coupling,
    _surround_balance,
    _surround_write_free_top,
    surround_write_free_top,
)

# Match the shared float32 shoe runtime, not the float64 leg module.
wp.set_module_options({"enable_backward": False, "fuse_fp": True})
_Mat24i = wp.types.matrix(shape=(2, 4), dtype=int)
_Mat24 = wp.types.matrix(shape=(2, 4), dtype=float)


@wp.func
def _surround_world(
    world_index: int,
    lane: int,
    zero_pressure: wp.array[wp.vec2],
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    area: wp.array[wp.float32],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    decay: wp.array[wp.float32],
    overstress_gain: wp.array[wp.float32],
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
    sweeps: int,
    compression_in: wp.array[wp.float32],
    compression_out: wp.array[wp.float32],
):
    """Exchange old compression through a tile before each unchanged Jacobi sweep."""
    p = params[world_index]
    gain = overstress_gain[world_index]
    c = wp.vec2(0.0)
    rigid = wp.vec2(0.0)
    thickness = wp.vec2(1.0)
    column_area = wp.vec2(0.0)
    overstress = wp.vec2(0.0)
    driven_column = wp.vec2i(0)
    neighbor_ids = _Mat24i(-1)
    couplings = _Mat24(0.0)
    coupling_sum = wp.vec2(0.0)
    for row in range(2):
        column = row * 512 + lane
        i = world_index * column_count + column
        if column < column_count:
            c[row] = compression_in[i]
            world = wp.transform_point(body_q[carrier[world_index]], anchor_local[column])
            rigid[row] = z_free_rigid[column] - world[2]
            thickness[row] = rest_len[column]
            column_area[row] = area[column]
            driven_column[row] = driven[column]
            overstress[row] = decay[world_index] * q_state[i] - gain * peq_prev[i]
            for side in range(4):
                j = neighbors[column, side]
                neighbor_ids[row, side] = j
                if j >= 0:
                    coupling = coupling_scale * _pasternak_coupling(thickness[row], rest_len[j], p)
                    couplings[row, side] = coupling
                    coupling_sum[row] += coupling
    penultimate = c
    for _sweep in range(sweeps):
        # Every lane participates, including padding and driven columns. Each
        # sweep reads the old driven values, as in the separate-kernel path.
        shared = wp.tile(c)
        next_c = c
        for row in range(2):
            column = row * 512 + lane
            if column < column_count:
                if driven_column[row] != 0:
                    next_c[row] = wp.max(rigid[row], 0.0)
                else:
                    pull = float(0.0)
                    for side in range(4):
                        j = neighbor_ids[row, side]
                        if j >= 0:
                            pull += couplings[row, side] * (shared[j // 512, j % 512] - c[row])
                    if c[row] == 0.0:
                        peq = zero_pressure[world_index * column_count + column]
                        next_c[row] = _surround_balance_pressures(
                            c[row],
                            rigid[row],
                            pull,
                            coupling_sum[row],
                            thickness[row],
                            overstress[row],
                            gain,
                            column_area[row],
                            attachment,
                            max_strain,
                            relaxation,
                            carrier_bond,
                            peq[0],
                            peq[1],
                        )
                    else:
                        next_c[row] = _surround_balance(
                            c[row],
                            rigid[row],
                            pull,
                            coupling_sum[row],
                            thickness[row],
                            overstress[row],
                            gain,
                            p,
                            column_area[row],
                            attachment,
                            max_strain,
                            relaxation,
                            carrier_bond,
                        )
        penultimate = c
        c = next_c
    for row in range(2):
        column = row * 512 + lane
        i = world_index * column_count + column
        if column < column_count:
            compression_in[i] = c[row]
            if sweeps % 2 == 0:
                compression_out[i] = penultimate[row]
            else:
                compression_out[i] = c[row]


@wp.kernel(launch_bounds=512)
def _surround_fused(
    zero_pressure: wp.array[wp.vec2],
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    area: wp.array[wp.float32],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    decay: wp.array[wp.float32],
    overstress_gain: wp.array[wp.float32],
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
    sweeps: int,
    compression_in: wp.array[wp.float32],
    compression_out: wp.array[wp.float32],
):
    """Run the shared block-local surround sweeps."""
    world_index, lane = wp.tid()
    _surround_world(
        world_index,
        lane,
        zero_pressure,
        carrier,
        column_count,
        body_q,
        driven,
        neighbors,
        anchor_local,
        z_free_rigid,
        rest_len,
        area,
        q_state,
        peq_prev,
        params,
        decay,
        overstress_gain,
        coupling_scale,
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
        sweeps,
        compression_in,
        compression_out,
    )


class FoundationFused(MidsoleFoundation):
    """Keep up to 1,024 columns' surround sweeps within one CUDA block per world.

    Larger beds and CPU execution retain the shared runtime implementation.
    Constitutive functions, sweep order/count, and ping-pong buffer results are
    unchanged. Only launch scheduling and storage of sweep-local data differ.
    """

    @wp.struct
    class Data:
        """Keep persistent foundation arrays together for the block-local launch."""

        zero_pressure: wp.array[wp.vec2]
        record_diagnostics: int
        diagnostic_groups: int
        diagnostic_rest: wp.array[wp.float64]
        passive_cap: wp.float64
        diagnostic_maxima: wp.array2d[wp.vec2d]
        diagnostic_caps: wp.array2d[int]
        diagnostic_nonfinite: wp.array2d[int]
        clock: wp.array[int]
        tick: int
        ground_force: wp.array[wp.vec3]
        enabled: wp.array[int]
        free_column_count: int
        carrier: wp.array[wp.int32]
        column_count: wp.int32
        driven: wp.array[wp.int32]
        neighbors: wp.array2d[wp.int32]
        anchor_local: wp.array[wp.vec3]
        z_free_rigid: wp.array[wp.float32]
        rest_len: wp.array[wp.float32]
        area: wp.array[wp.float32]
        q_state: wp.array[wp.float32]
        peq_prev: wp.array[wp.float32]
        world_params: wp.array[FoundationParams]
        surround_decay: wp.array[wp.float32]
        surround_gain: wp.array[wp.float32]
        coupling_scale: wp.float32
        attachment: wp.float32
        max_strain: wp.float32
        relaxation: wp.float32
        carrier_bond: wp.int32
        sweeps: int
        surround_compression: wp.array[wp.float32]
        surround_scratch: wp.array[wp.float32]
        inv_dt: wp.float32
        compression: wp.array[wp.float32]
        surround_previous: wp.array[wp.float32]
        z_free: wp.array[wp.float32]
        surround_rate: wp.array[wp.float32]
        dt: wp.float32
        base_pressure: wp.array[wp.float32]
        body_com: wp.array[wp.vec3]
        tangent_anchor: wp.array[wp.vec2]
        tangent_stuck: wp.array[wp.int32]
        tangent_dwell: wp.array[wp.float32]
        friction_kt: wp.array[wp.float32]
        friction_kv: wp.array[wp.float32]
        column_force: wp.array[wp.vec3]
        column_pressed: wp.array[wp.float32]
        ground_height: wp.float32
        reduction_groups: wp.int32
        partial_force: wp.array[wp.vec3]
        partial_torque: wp.array[wp.vec3]
        partial_moment: wp.array[wp.vec3]
        partial_cop: wp.array[wp.vec3]
        partial_normal: wp.array[wp.float32]
        partial_pressed: wp.array[wp.float32]
        partial_power: wp.array[wp.float32]
        partial_max: wp.array[wp.float32]
        partial_active: wp.array[wp.int32]
        contact_point: wp.array[wp.vec3]
        normal_force: wp.array[wp.float32]
        cop_moment: wp.array[wp.vec3]
        active: wp.array[wp.int32]
        resultant_force: wp.array[wp.vec3]
        resultant_moment_origin: wp.array[wp.vec3]
        contact_power: wp.array[wp.float32]
        max_compression: wp.array[wp.float32]
        pressed_force: wp.array[wp.float32]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_pressure = wp.zeros(self.world_count * self.column_count, dtype=wp.vec2, device=self.device)
        self._refresh_zero_pressure()
        self.enabled = wp.ones(self.world_count, dtype=int, device=self.device)
        self.fused_apply = True
        self.diagnostics = None
        self._unique_carriers = len(np.unique(self.carrier.numpy())) == self.world_count

    def _refresh_surround_constants(self, dt: float) -> None:
        if self._materials_dirty:
            self._refresh_zero_pressure()
        super()._refresh_surround_constants(dt)

    def _refresh_zero_pressure(self):
        wp.launch(
            _cache_zero_pressure,
            dim=self.world_count * self.column_count,
            inputs=[self.column_count, self.rest_len, self.world_params, self.zero_pressure],
            device=self.device,
        )

    def reset(self):
        """Reset recurrent state and refresh constitutive constants for the next rollout."""
        super().reset()
        self._refresh_zero_pressure()

    @property
    def fused_diagnostics(self):
        """Report whether the CUDA block can also write the controller diagnostics."""
        return (
            self.fused_apply
            and self.friction_solver is None
            and self.device.is_cuda
            and self.column_count <= 1024
            and self.ground_height_m is not None
            and self._unique_carriers
            and self.diagnostics is not None
        )

    def apply(self, state, dt: float, clear_body_force: bool = False, *, tick=False) -> None:
        """Fuse unchanged shoe stages within each world's CUDA block."""
        if (
            not self.fused_apply
            or self.friction_solver is not None
            or not self.device.is_cuda
            or self.column_count > 1024
            or self.ground_height_m is None
            or not clear_body_force
            or not self._unique_carriers
        ):
            super().apply(state, dt, clear_body_force)
            return
        self._refresh_surround_constants(dt)
        cfg = self.surround
        data = self.Data()
        data.zero_pressure = self.zero_pressure
        if self.diagnostics is not None:
            rest, cap, maxima, caps, nonfinite, clock = self.diagnostics
            data.record_diagnostics = 1
            data.diagnostic_groups = maxima.shape[1]
            data.diagnostic_rest = rest
            data.passive_cap = cap
            data.diagnostic_maxima = maxima
            data.diagnostic_caps = caps
            data.diagnostic_nonfinite = nonfinite
            data.clock = clock
            data.tick = int(tick)
        data.ground_force = self.ground_force
        data.enabled = self.enabled
        data.free_column_count = self.free_column_count
        data.carrier = self.carrier
        data.column_count = self.column_count
        data.driven = self.driven
        data.neighbors = self.neighbors
        data.anchor_local = self.anchor_local
        data.z_free_rigid = self.z_free_rigid
        data.rest_len = self.rest_len
        data.area = self.area
        data.q_state = self.q_state
        data.peq_prev = self.peq_prev
        data.world_params = self.world_params
        data.surround_decay = self.surround_decay
        data.surround_gain = self.surround_gain
        data.coupling_scale = float(cfg.coupling_scale)
        data.attachment = float(cfg.attachment_n_m)
        data.max_strain = float(cfg.max_strain)
        data.relaxation = (
            1.0 if cfg.relaxation_time_s <= 0.0 else 1.0 - float(np.exp(-dt / cfg.sweeps / cfg.relaxation_time_s))
        )
        data.carrier_bond = int(bool(cfg.carrier_bond))
        data.sweeps = int(cfg.sweeps)
        data.surround_compression = self.surround_compression
        data.surround_scratch = self.surround_scratch
        data.inv_dt = float(1.0 / dt)
        data.compression = self.compression
        data.surround_previous = self.surround_previous
        data.z_free = self.z_free
        data.surround_rate = self.surround_rate
        data.dt = float(dt)
        data.base_pressure = self.base_pressure
        data.body_com = self.body_com
        data.tangent_anchor = self.tangent_anchor
        data.tangent_stuck = self.tangent_stuck
        data.tangent_dwell = self.tangent_dwell
        data.friction_kt = self.friction_kt
        data.friction_kv = self.friction_kv
        data.column_force = self.column_force
        data.column_pressed = self.column_pressed
        data.ground_height = float(self.ground_height_m)
        data.reduction_groups = self.reduction_groups
        data.partial_force = self.partial_force
        data.partial_torque = self.partial_torque
        data.partial_moment = self.partial_moment
        data.partial_cop = self.partial_cop
        data.partial_normal = self.partial_normal
        data.partial_pressed = self.partial_pressed
        data.partial_power = self.partial_power
        data.partial_max = self.partial_max
        data.partial_active = self.partial_active
        data.contact_point = self.contact_point
        data.normal_force = self.normal_force
        data.cop_moment = self.cop_moment
        data.active = self.active
        data.resultant_force = self.resultant_force
        data.resultant_moment_origin = self.resultant_moment_origin
        data.contact_power = self.contact_power
        data.max_compression = self.max_compression
        data.pressed_force = self.pressed_force
        wp.launch_tiled(
            _apply_world,
            dim=self.world_count,
            block_dim=512,
            inputs=[data, state.body_q, state.body_qd, state.body_f],
            device=self.device,
        )

    def relax_surround(self, state, dt: float) -> None:
        """Run the shared Jacobi balance with block-local exchanges on small CUDA beds."""
        if not self.device.is_cuda or self.column_count > 1024:
            super().relax_surround(state, dt)
            return
        cfg = self.surround
        sweeps = int(cfg.sweeps)
        sub_dt = dt / sweeps
        tau = float(cfg.relaxation_time_s)
        relaxation = 1.0 if tau <= 0.0 else 1.0 - float(np.exp(-sub_dt / tau))
        self._refresh_surround_constants(dt)
        wp.launch_tiled(
            _surround_fused,
            dim=self.world_count,
            block_dim=512,
            inputs=[
                self.zero_pressure,
                self.carrier,
                self.column_count,
                state.body_q,
                self.driven,
                self.neighbors,
                self.anchor_local,
                self.z_free_rigid,
                self.rest_len,
                self.area,
                self.q_state,
                self.peq_prev,
                self.world_params,
                self.surround_decay,
                self.surround_gain,
                float(cfg.coupling_scale),
                float(cfg.attachment_n_m),
                float(cfg.max_strain),
                relaxation,
                int(bool(cfg.carrier_bond)),
                sweeps,
                self.surround_compression,
                self.surround_scratch,
            ],
            device=self.device,
        )
        wp.launch(
            surround_write_free_top,
            dim=self.world_count * self.column_count,
            inputs=[
                self.carrier,
                self.column_count,
                float(1.0 / dt),
                state.body_q,
                self.driven,
                self.anchor_local,
                self.z_free_rigid,
                self.surround_compression,
                self.surround_previous,
                self.z_free,
                self.surround_rate,
            ],
            device=self.device,
        )


# Use the same explicit shared/global-memory fence as Kamino's block-local kernels.
@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
for (int offset = 16; offset > 0; offset >>= 1)
    v = fmax(v, __shfl_xor_sync(0xffffffffu, v, offset));
#endif
return v;
""")
def _warp_max_double(v: wp.float64) -> wp.float64: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, offset);
#endif
return v;
""")
def _warp_sum_int(v: int) -> int: ...


@wp.kernel(launch_bounds=512)
def _apply_world(
    data: FoundationFused.Data,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    """Keep each world's shared-law stages and ordered reductions in one block."""
    world_index, lane = wp.tid()
    # The shoe phases never read the clock, so world zero may tick it independently.
    if data.tick and world_index == 0 and lane == 0:
        data.clock[0] += 1
    if data.enabled[world_index] == 0:
        return
    if lane == 0:
        body_f[data.carrier[world_index]] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    if data.free_column_count:
        _surround_world(
            world_index,
            lane,
            data.zero_pressure,
            data.carrier,
            data.column_count,
            body_q,
            data.driven,
            data.neighbors,
            data.anchor_local,
            data.z_free_rigid,
            data.rest_len,
            data.area,
            data.q_state,
            data.peq_prev,
            data.world_params,
            data.surround_decay,
            data.surround_gain,
            data.coupling_scale,
            data.attachment,
            data.max_strain,
            data.relaxation,
            data.carrier_bond,
            data.sweeps,
            data.surround_compression,
            data.surround_scratch,
        )
    for row in range(2):
        column = row * 512 + lane
        if column < data.column_count:
            i = world_index * data.column_count + column
            if data.free_column_count:
                _surround_write_free_top(
                    i,
                    data.carrier,
                    data.column_count,
                    data.inv_dt,
                    body_q,
                    data.driven,
                    data.anchor_local,
                    data.z_free_rigid,
                    data.surround_compression,
                    data.surround_previous,
                    data.z_free,
                    data.surround_rate,
                )
            _foundation_pressure(
                i,
                data.carrier,
                data.column_count,
                data.dt,
                body_q,
                data.anchor_local,
                data.z_free,
                data.rest_len,
                data.world_params,
                data.q_state,
                data.peq_prev,
                data.compression,
                data.base_pressure,
            )
    _sync_threads()
    if data.record_diagnostics:
        for row in range(2):
            column = row * 512 + lane
            group = row * 16 + lane // 32
            dm = wp.float64(0.0)
            pm = wp.float64(0.0)
            cap = int(0)
            invalid = int(0)
            if column < data.column_count:
                fraction = (
                    wp.float64(data.compression[world_index * data.column_count + column])
                    / data.diagnostic_rest[column]
                )
                if not wp.isfinite(fraction):
                    invalid = 1
                elif data.driven[column] != 0:
                    dm = wp.max(dm, fraction)
                else:
                    pm = wp.max(pm, fraction)
                    if fraction >= data.passive_cap - wp.float64(1.0e-6):
                        cap = 1
            dm = _warp_max_double(dm)
            pm = _warp_max_double(pm)
            cap = _warp_sum_int(cap)
            invalid = _warp_sum_int(invalid)
            if lane % 32 == 0 and group < data.diagnostic_groups:
                data.diagnostic_maxima[world_index, group] = wp.vec2d(dm, pm)
                data.diagnostic_caps[world_index, group] = cap
                data.diagnostic_nonfinite[world_index, group] = int(invalid > 0)
    for row in range(2):
        column = row * 512 + lane
        if column < data.column_count:
            i = world_index * data.column_count + column
            force, point = _foundation_column_forces(
                i,
                data.carrier,
                data.column_count,
                data.dt,
                body_q,
                body_qd,
                data.body_com,
                data.anchor_local,
                data.area,
                data.rest_len,
                data.neighbors,
                data.compression,
                data.base_pressure,
                data.tangent_anchor,
                data.tangent_stuck,
                data.tangent_dwell,
                data.friction_kt,
                data.friction_kv,
                data.world_params,
                data.column_force,
                data.column_pressed,
                1,
                data.ground_height,
            )
            data.ground_force[i] = force
            data.contact_point[i] = point
    _sync_threads()
    if lane < data.reduction_groups:
        _foundation_partial(
            world_index * data.reduction_groups + lane,
            data.carrier,
            data.column_count,
            data.reduction_groups,
            body_q,
            body_qd,
            data.body_com,
            data.anchor_local,
            data.compression,
            data.ground_force,
            data.column_pressed,
            data.partial_force,
            data.partial_torque,
            data.partial_moment,
            data.partial_cop,
            data.partial_normal,
            data.partial_pressed,
            data.partial_power,
            data.partial_max,
            data.partial_active,
            1,
            data.contact_point,
        )
    _sync_threads()
    if lane == 0:
        _foundation_finalize(
            world_index,
            data.carrier,
            data.reduction_groups,
            data.partial_force,
            data.partial_torque,
            data.partial_moment,
            data.partial_cop,
            data.partial_normal,
            data.partial_pressed,
            data.partial_power,
            data.partial_max,
            data.partial_active,
            body_f,
            data.normal_force,
            data.cop_moment,
            data.active,
            data.resultant_force,
            data.resultant_moment_origin,
            data.contact_power,
            data.max_compression,
            data.pressed_force,
        )


@wp.kernel
def _cache_zero_pressure(
    column_count: int,
    rest_len: wp.array[float],
    params: wp.array[FoundationParams],
    pressure: wp.array[wp.vec2],
):
    i = wp.tid()
    thickness = rest_len[i % column_count]
    p = params[i // column_count]
    step = 1.0e-3 * thickness
    pressure[i] = wp.vec2(_hyperfoam_pressure(0.0 / thickness, p), _hyperfoam_pressure((0.0 + step) / thickness, p))
