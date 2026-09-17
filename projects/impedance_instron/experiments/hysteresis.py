# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hysteresis bench simulation for digital shoe conditions and fixtures.

Prescribes common cyclic compression-displacement protocols across variants,
evaluates warmup cycles with comprehensive state periodicity checks, preserves
full-substep raw histories and work calculations, monitors strain and force caps,
and outputs raw npz traces and plot-ready JSON without external plotting dependencies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.artifact import DigitalShoe, load_artifact
from projects.digital_shoe.runtime import (
    FoundationConfig,
    MidsoleFoundation,
    SurroundConfig,
)

INSTRON_SURROUND_SWEEPS = 5
DEFAULT_FIXTURES = ("rearfoot_punch", "fullfoot_last")


@wp.kernel
def _stage_hysteresis_pose(
    step_counter: wp.array[wp.int32],
    displacement_schedule: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Stage the next prescribed pose and zero velocity from device memory."""
    if wp.tid() == 0:
        step = step_counter[0]
        pose = body_q[0]
        pose[2] = -displacement_schedule[step]
        body_q[0] = pose
        body_qd[0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _advance_hysteresis_clock(step_counter: wp.array[wp.int32]):
    """Advance the device-side schedule clock after one completed substep."""
    if wp.tid() == 0:
        step_counter[0] += 1


@wp.kernel
def _record_hysteresis_diagnostics(
    step_counter: wp.array[wp.int32],
    normal_force: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    driven: wp.array[wp.int32],
    passive_compression: wp.array[wp.float32],
    passive_rate: wp.array[wp.float32],
    column_count: int,
    max_force_cap_n: float,
    max_strain_cap: float,
    displacement_schedule: wp.array[wp.float32],
    history_force: wp.array[wp.float32],
    history_max_compression: wp.array[wp.float32],
    history_max_strain: wp.array[wp.float32],
    history_passive_strain: wp.array[wp.float32],
    history_displacement: wp.array[wp.float32],
    history_reason: wp.array[wp.int32],
):
    """Record a device-side diagnostic for one substep.

    The kernel deliberately scans every column. This keeps the diagnostic source
    on the device and checks passive columns even when the resultant force is
    small or the indenter is at a zero-displacement endpoint.
    """
    if wp.tid() != 0:
        return
    step = step_counter[0]
    reason = 0
    force = normal_force[0]
    maximum = max_compression[0]
    if not wp.isfinite(force) or not wp.isfinite(maximum):
        reason = reason | 1
    elif force > max_force_cap_n or force < -100.0:
        reason = reason | 1

    max_strain = float(0.0)
    passive_strain = float(0.0)
    for column in range(column_count):
        rest = rest_len[column]
        comp = compression[column]
        if not wp.isfinite(rest) or rest <= 0.0 or not wp.isfinite(comp):
            reason = reason | 2
            continue
        strain = comp / rest
        if not wp.isfinite(strain):
            reason = reason | 4
        elif strain > max_strain:
            max_strain = strain
        if strain > max_strain_cap:
            reason = reason | 4
        if driven[column] == 0:
            passive = passive_compression[column]
            if not wp.isfinite(passive):
                reason = reason | 32
            elif passive / rest > passive_strain:
                passive_strain = passive / rest
            if not wp.isfinite(passive_rate[column]):
                reason = reason | 64
            if wp.isfinite(passive) and passive / rest > max_strain_cap:
                reason = reason | 32
        if not wp.isfinite(q_state[column]):
            reason = reason | 8
        if not wp.isfinite(peq_prev[column]):
            reason = reason | 16
    history_force[step] = force
    history_max_compression[step] = maximum
    history_max_strain[step] = max_strain
    history_passive_strain[step] = passive_strain
    history_displacement[step] = displacement_schedule[step]
    history_reason[step] = reason


def compute_loop_work(displacement_m: np.ndarray, force_n: np.ndarray) -> dict[str, float]:
    """Compute mechanical work input, returned, net, and loss ratio for a cycle [J].

    Evaluated on contiguous displacement and force arrays across a closed cycle.

    Args:
        displacement_m: 1D array of indenter kinematic displacement [m].
        force_n: 1D array of indenter normal force [N].

    Returns:
        Dictionary containing work_input_j, work_returned_j, work_net_j, and
        hysteresis_loss_ratio.
    """
    if len(displacement_m) < 2 or len(force_n) < 2:
        return {
            "work_input_j": 0.0,
            "work_returned_j": 0.0,
            "work_net_j": 0.0,
            "hysteresis_loss_ratio": 0.0,
        }
    dd = np.diff(displacement_m)
    f_mid = 0.5 * (force_n[:-1] + force_n[1:])
    dw = f_mid * dd

    # dd > 0: loading / compression phase (work input)
    # dd < 0: unloading / restitution phase (work returned)
    work_input = float(np.sum(dw[dd > 0]))
    work_returned = float(-np.sum(dw[dd < 0]))
    work_net = float(work_input - work_returned)
    ratio = float(work_net / work_input) if work_input > 1e-12 else 0.0

    return {
        "work_input_j": work_input,
        "work_returned_j": work_returned,
        "work_net_j": work_net,
        "hysteresis_loss_ratio": ratio,
    }


def _build_fixture_foundation(
    shoe: DigitalShoe,
    fixture_name: str,
    device: wp.Device,
    surround_sweeps: int = INSTRON_SURROUND_SWEEPS,
) -> tuple[MidsoleFoundation, newton.Model, newton.State, int]:
    """Build a MidsoleFoundation for the specified fixture using the full bed.

    Args:
        shoe: Loaded DigitalShoe artifact.
        fixture_name: Name of the instron fixture (e.g. 'rearfoot_punch').
        device: Target Warp device.
        surround_sweeps: Number of passive surround relaxation sweeps.

    Returns:
        Tuple of (foundation, model, state, carrier_index).
    """
    fixture = shoe.instron_fixture(fixture_name)
    bed = shoe.column_bed

    # Map fixture columns onto full bed
    lookup = {tuple(np.round(point, 8)): index for index, point in enumerate(bed.anchor_bottom_m[:, :2])}
    try:
        supported = np.array([lookup[tuple(np.round(point, 8))] for point in fixture.carrier_anchor_m[:, :2]])
    except KeyError as error:
        raise ValueError(f"The Instron fixture {fixture_name!r} must map onto the whole-shoe column bed") from error

    if len(np.unique(supported)) != len(supported):
        raise ValueError(f"The Instron fixture {fixture_name!r} maps more than once onto a bed column")

    bottom = bed.anchor_bottom_m[:, 2]
    free_top = bottom + bed.rest_length_m
    anchor = np.column_stack([bed.anchor_bottom_m[:, :2], free_top])

    datum_shift = bottom[supported] - fixture.foam_bottom_m
    anchor[supported] = fixture.carrier_anchor_m
    anchor[supported, 2] += datum_shift
    free_top[supported] = fixture.foam_free_top_m + datum_shift

    # Zero-displacement contact datum: ensure minimum initial clearance is zero
    initial_gap = anchor[supported, 2] - free_top[supported]
    min_gap = float(np.min(initial_gap))
    if min_gap > 0.0:
        anchor[supported, 2] -= min_gap
        free_top[supported] -= min_gap

    builder = newton.ModelBuilder()
    carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=device)
    state = model.state()

    driven = np.zeros(len(bed.rest_length_m), dtype=np.int32)
    driven[supported] = 1

    surround_config = SurroundConfig(
        driven=driven,
        attachment_n_m=0.0,
        sweeps=surround_sweeps,
        relaxation_time_s=0.0,
        carrier_bond=False,
    )
    foundation_config = FoundationConfig(stretch_floor=0.05)

    foundation = MidsoleFoundation(
        anchor,
        free_top,
        bed.rest_length_m,
        bed.area_m2,
        bed.neighbors,
        bed.spacing_m,
        shoe.material,
        carrier,
        model.body_com,
        foundation_config,
        device,
        surround_config,
    )

    return foundation, model, state, carrier


def _require_real(name: str, value: Any, *, positive: bool = False) -> float:
    """Validate a finite scalar protocol parameter and return it as ``float``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_count(name: str, value: Any, *, minimum: int = 0) -> int:
    """Validate an integer cycle/count parameter without accepting floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _diagnostic_reason(code: int) -> str:
    """Turn device diagnostic bits into a stable human-readable reason."""
    labels = (
        (1, "force_nonfinite_or_cap"),
        (2, "compression_nonfinite_or_invalid_rest_length"),
        (4, "column_strain_cap"),
        (8, "q_state_nonfinite"),
        (16, "peq_prev_nonfinite"),
        (32, "passive_compression_cap_or_nonfinite"),
        (64, "passive_rate_nonfinite"),
    )
    return "; ".join(label for bit, label in labels if code & bit) or "unknown_diagnostic"


def run_fixture_hysteresis(
    shoe: DigitalShoe,
    fixture_name: str,
    *,
    device: str | wp.Device = "cpu",
    dt_s: float = 6.25e-5,
    sample_dt_s: float = 1.0e-3,
    warmup_cycles: int = 3,
    total_cycles: int = 4,
    prescribed_depth_m: float = 0.010,
    period_s: float = 0.20,
    max_force_cap_n: float = 1.0e5,
    max_strain_cap: float = 0.90,
    periodicity_tol: float = 0.05,
    surround_sweeps: int = INSTRON_SURROUND_SWEEPS,
    use_measured_trace: bool = False,
) -> dict[str, Any]:
    """Run cyclic compression on one fixture and return a valid raw prefix.

    Diagnostics are recorded in device-resident histories on every substep. If a
    diagnostic fires, all samples from that sample onward are excluded from the
    returned evidence; this prevents later values after a divergence from being
    mistaken for a valid continuation.
    """
    if fixture_name not in shoe.raw.get("instron_fixtures", {}):
        return {
            "status": "blocked",
            "reason": f"Fixture {fixture_name!r} not present in artifact instron_fixtures",
            "protocol": None,
            "metrics": None,
            "raw": None,
        }

    dt_s = _require_real("dt_s", dt_s, positive=True)
    sample_dt_s = _require_real("sample_dt_s", sample_dt_s, positive=True)
    period_s = _require_real("period_s", period_s, positive=True)
    prescribed_depth_m = _require_real("prescribed_depth_m", prescribed_depth_m, positive=True)
    max_force_cap_n = _require_real("max_force_cap_n", max_force_cap_n, positive=True)
    max_strain_cap = _require_real("max_strain_cap", max_strain_cap, positive=True)
    periodicity_tol = _require_real("periodicity_tol", periodicity_tol)
    if periodicity_tol < 0.0:
        raise ValueError("periodicity_tol must be nonnegative")
    warmup_cycles = _require_count("warmup_cycles", warmup_cycles)
    total_cycles = _require_count("total_cycles", total_cycles, minimum=1)
    surround_sweeps = _require_count("surround_sweeps", surround_sweeps, minimum=1)
    if total_cycles <= warmup_cycles:
        raise ValueError(f"total_cycles ({total_cycles}) must be strictly greater than warmup_cycles ({warmup_cycles})")
    if not isinstance(use_measured_trace, (bool, np.bool_)):
        raise TypeError("use_measured_trace must be a bool")

    bed = shoe.column_bed
    rest_lengths = np.asarray(bed.rest_length_m, dtype=np.float64).reshape(-1)
    if not len(rest_lengths) or not np.all(np.isfinite(rest_lengths)) or np.any(rest_lengths <= 0.0):
        raise ValueError("artifact column rest lengths must be finite and positive")

    if use_measured_trace:
        curve = next((c for c in shoe.validation.get("curves", []) if c.get("fixture") == fixture_name), None)
        if curve is None or "time_s" not in curve or "displacement_m" not in curve:
            return {
                "status": "blocked",
                "reason": (
                    f"Requested measured trace for fixture {fixture_name!r} not found in artifact validation curves"
                ),
                "protocol": None,
                "metrics": None,
                "raw": None,
            }
        src_t = np.asarray(curve["time_s"], dtype=np.float64).reshape(-1)
        src_d = np.asarray(curve["displacement_m"], dtype=np.float64).reshape(-1)
        if (
            len(src_t) < 2
            or len(src_t) != len(src_d)
            or not np.all(np.isfinite(src_t))
            or not np.all(np.isfinite(src_d))
        ):
            raise ValueError("measured trace must contain at least two finite time/displacement pairs")
        if np.any(np.diff(src_t) <= 0.0):
            raise ValueError("measured trace times must be strictly increasing")
        cycle_period_s = float(src_t[-1] - src_t[0])
        peak_disp_m = float(np.max(src_d))
        if cycle_period_s <= 0.0 or not math.isfinite(cycle_period_s) or peak_disp_m <= 0.0:
            raise ValueError("measured trace must have a finite positive period and peak displacement")
        protocol_source = "calibration_trace"
    else:
        src_t = np.empty(0, dtype=np.float64)
        src_d = np.empty(0, dtype=np.float64)
        cycle_period_s = period_s
        peak_disp_m = prescribed_depth_m
        protocol_source = "synthetic_prescribed"

    wp_device = wp.get_device(device)
    foundation, _model, state, _carrier = _build_fixture_foundation(
        shoe, fixture_name, device=wp_device, surround_sweeps=surround_sweeps
    )
    column_count = len(rest_lengths)

    # Use full dt substeps and one exact remainder per cycle. Endpoints are therefore
    # at the requested period instead of at round(period / dt) * dt.
    full_steps = int(math.floor(cycle_period_s / dt_s))
    remainder = cycle_period_s - full_steps * dt_s
    if remainder <= max(1.0e-12, cycle_period_s * 1.0e-12):
        remainder = 0.0
    if full_steps == 0 and remainder == 0.0:
        raise ValueError("period_s must be at least one positive simulation substep")
    cycle_durations = np.full(full_steps, dt_s, dtype=np.float64)
    if remainder:
        cycle_durations = np.concatenate((cycle_durations, np.array([remainder], dtype=np.float64)))
    steps_per_cycle = len(cycle_durations)
    total_steps = total_cycles * steps_per_cycle
    step_durations = np.tile(cycle_durations, total_cycles)
    cycle_start_times = np.arange(total_cycles, dtype=np.float64) * cycle_period_s
    step_times = np.empty(total_steps + 1, dtype=np.float64)
    step_times[0] = 0.0
    for cycle in range(total_cycles):
        begin = cycle * steps_per_cycle
        end = begin + steps_per_cycle
        step_times[begin + 1 : end + 1] = cycle_start_times[cycle] + np.cumsum(cycle_durations)
    # The explicit assignment removes accumulated floating-point drift at endpoints.
    for cycle in range(total_cycles + 1):
        if cycle <= total_cycles:
            step_times[min(cycle * steps_per_cycle, total_steps)] = cycle * cycle_period_s
    cycle_end_indices = np.arange(1, total_cycles + 1, dtype=np.int64) * steps_per_cycle

    cycle_phase = np.mod(step_times, cycle_period_s)
    endpoint = np.isclose(cycle_phase, cycle_period_s, rtol=0.0, atol=max(1.0e-12, cycle_period_s * 1.0e-12))
    cycle_phase[endpoint] = 0.0
    if use_measured_trace:
        source_phase = src_t - src_t[0]
        substep_depths = np.interp(cycle_phase, source_phase, src_d)
    else:
        # Exact analytical haversine, not a 200-point interpolation of the waveform.
        substep_depths = peak_disp_m * 0.5 * (1.0 - np.cos(2.0 * np.pi * cycle_phase / cycle_period_s))
    if not np.all(np.isfinite(substep_depths)):
        raise ValueError("prescribed displacement waveform is not finite")

    hist_force = wp.zeros(total_steps + 1, dtype=wp.float32, device=wp_device)
    hist_max_comp = wp.zeros(total_steps + 1, dtype=wp.float32, device=wp_device)
    hist_max_strain = wp.zeros(total_steps + 1, dtype=wp.float32, device=wp_device)
    hist_passive_strain = wp.zeros(total_steps + 1, dtype=wp.float32, device=wp_device)
    hist_disp = wp.zeros(total_steps + 1, dtype=wp.float32, device=wp_device)
    hist_reason = wp.zeros(total_steps + 1, dtype=wp.int32, device=wp_device)
    if foundation.surround_compression is not None:
        passive_compression = foundation.surround_compression
        passive_rate = foundation.surround_rate
    else:
        passive_compression = wp.zeros(1, dtype=wp.float32, device=wp_device)
        passive_rate = wp.zeros(1, dtype=wp.float32, device=wp_device)

    pose = np.zeros(7, dtype=np.float32)
    pose[6] = 1.0
    displacement_schedule = wp.array(np.asarray(substep_depths, dtype=np.float32), dtype=wp.float32, device=wp_device)
    step_clock = wp.zeros(1, dtype=wp.int32, device=wp_device)
    cycle_boundary_snapshots: list[dict[str, np.ndarray]] = []

    def _snapshot_state() -> dict[str, np.ndarray]:
        snap = {
            "q_state_pa": foundation.q_state.numpy().copy(),
            "peq_prev_pa": foundation.peq_prev.numpy().copy(),
            "compression_m": foundation.compression.numpy().copy(),
        }
        if foundation.surround_compression is not None:
            snap["passive_compression_m"] = foundation.surround_compression.numpy().copy()
            snap["passive_rate_mps"] = foundation.surround_rate.numpy().copy()
        else:
            snap["passive_compression_m"] = np.empty(0, dtype=np.float32)
            snap["passive_rate_mps"] = np.empty(0, dtype=np.float32)
        return snap

    def _record() -> None:
        wp.launch(
            _record_hysteresis_diagnostics,
            dim=1,
            inputs=[
                step_clock,
                foundation.normal_force,
                foundation.max_compression,
                foundation.q_state,
                foundation.peq_prev,
                foundation.compression,
                foundation.rest_len,
                foundation.driven,
                passive_compression,
                passive_rate,
                column_count,
                max_force_cap_n,
                max_strain_cap,
                displacement_schedule,
                hist_force,
                hist_max_comp,
                hist_max_strain,
                hist_passive_strain,
                hist_disp,
                hist_reason,
            ],
            device=wp_device,
        )

    cycle_boundary_snapshots.append(_snapshot_state())
    _record()

    graph_enabled = bool(wp_device.is_cuda and remainder == 0.0)
    graph_chunk_steps = 256
    graph = None
    if graph_enabled:
        # Keep chunk boundaries on cycle boundaries so closure snapshots remain exact.
        while graph_chunk_steps > 1 and steps_per_cycle % graph_chunk_steps:
            graph_chunk_steps //= 2
        if steps_per_cycle % graph_chunk_steps:
            graph_enabled = False

    if graph_enabled:
        step_clock.assign(np.array([1], dtype=np.int32))
        with wp.ScopedCapture(device=wp_device) as capture:
            for _ in range(graph_chunk_steps):
                wp.launch(
                    _stage_hysteresis_pose,
                    dim=1,
                    inputs=[step_clock, displacement_schedule, state.body_q, state.body_qd],
                    device=wp_device,
                )
                foundation.apply(state, dt_s, clear_body_force=True)
                _record()
                wp.launch(_advance_hysteresis_clock, dim=1, inputs=[step_clock], device=wp_device)
        graph = capture.graph
        step_clock.assign(np.array([1], dtype=np.int32))
        chunk_count = total_steps // graph_chunk_steps
        for chunk in range(chunk_count):
            wp.capture_launch(graph)
            completed = (chunk + 1) * graph_chunk_steps
            if completed % steps_per_cycle == 0:
                cycle_boundary_snapshots.append(_snapshot_state())
    else:
        # Exact-remainder schedules use eager launches because foundation.apply takes
        # a scalar dt. The device schedule/clock is still used for diagnostics.
        for step in range(1, total_steps + 1):
            step_clock.assign(np.array([step], dtype=np.int32))
            depth = float(substep_depths[step])
            pose[2] = -depth
            state.body_q.assign(pose.reshape(1, 7))
            state.body_qd.zero_()
            state.clear_forces()
            foundation.apply(state, float(step_durations[step - 1]))
            _record()
            if step % steps_per_cycle == 0:
                cycle_boundary_snapshots.append(_snapshot_state())

    # One host copy after the complete device run. No per-step array transfer is used.
    force_all = hist_force.numpy().astype(np.float64)
    max_comp_all = hist_max_comp.numpy().astype(np.float64)
    max_strain_all = hist_max_strain.numpy().astype(np.float64)
    passive_strain_all = hist_passive_strain.numpy().astype(np.float64)
    disp_all = hist_disp.numpy().astype(np.float64)
    reason_all = hist_reason.numpy().astype(np.int32)
    first_failure = np.flatnonzero(reason_all != 0)
    first_failure_step = int(first_failure[0]) if len(first_failure) else None
    if first_failure_step is None:
        valid_count = total_steps + 1
        termination_reason = None
        termination_code = 0
        terminated = False
    else:
        # Exclude the offending sample and everything after it from saved evidence.
        valid_count = max(1, first_failure_step)
        termination_code = int(reason_all[first_failure_step])
        termination_reason = (
            f"{_diagnostic_reason(termination_code)} at step {first_failure_step} "
            f"(t={step_times[first_failure_step]:.9g} s)"
        )
        terminated = True
    force_all = force_all[:valid_count]
    max_comp_all = max_comp_all[:valid_count]
    max_strain_all = max_strain_all[:valid_count]
    passive_strain_all = passive_strain_all[:valid_count]
    disp_all = disp_all[:valid_count]
    time_all = step_times[:valid_count]
    cycle_index_all = np.searchsorted(cycle_end_indices, np.arange(valid_count), side="left").astype(np.int32)

    full_cycles = [cycle for cycle, end_index in enumerate(cycle_end_indices) if end_index < valid_count]
    first_end = min(int(cycle_end_indices[0]), valid_count - 1)
    d_first = disp_all[: first_end + 1]
    f_first = force_all[: first_end + 1]
    if full_cycles:
        eval_cycle = full_cycles[-1]
        final_start = int(eval_cycle * steps_per_cycle)
        final_end = int((eval_cycle + 1) * steps_per_cycle)
        final_is_full = True
    else:
        eval_cycle = 0
        final_start = 0
        final_end = valid_count - 1
        final_is_full = False
    d_final = disp_all[final_start : final_end + 1]
    f_final = force_all[final_start : final_end + 1]

    def _loop_work_or_none(disp: np.ndarray, force: np.ndarray) -> dict[str, float | None]:
        if not is_state_closed:
            return dict.fromkeys(("work_input_j", "work_returned_j", "work_net_j", "hysteresis_loss_ratio"))
        return compute_loop_work(disp, force)

    # Closure compares the complete physical state with independent units/scales.
    state_components: dict[str, float] = {}
    state_diff_norm = 0.0
    is_state_closed = False
    force_boundary_err = float("inf")
    loop_force_err = float("inf")
    loop_work_err = float("inf")
    if len(full_cycles) >= 2 and not terminated:
        prev_snap = cycle_boundary_snapshots[full_cycles[-1] - 1]
        curr_snap = cycle_boundary_snapshots[full_cycles[-1]]
        q_delta = np.max(np.abs(curr_snap["q_state_pa"] - prev_snap["q_state_pa"]))
        q_scale = max(
            1.0,
            float(np.max(np.abs(curr_snap["q_state_pa"]))),
            float(np.max(np.abs(prev_snap["q_state_pa"]))),
        )
        peq_delta = np.max(np.abs(curr_snap["peq_prev_pa"] - prev_snap["peq_prev_pa"]))
        peq_scale = max(
            1.0,
            float(np.max(np.abs(curr_snap["peq_prev_pa"]))),
            float(np.max(np.abs(prev_snap["peq_prev_pa"]))),
        )
        c_delta = np.max(np.abs(curr_snap["compression_m"] - prev_snap["compression_m"]) / rest_lengths)
        p_delta = 0.0
        rate_delta = 0.0
        if curr_snap["passive_compression_m"].size:
            p_delta = float(
                np.max(
                    np.abs(curr_snap["passive_compression_m"] - prev_snap["passive_compression_m"])
                    / np.tile(rest_lengths, 1)
                )
            )
            rate_scale = max(
                1.0e-6,
                float(np.max(np.abs(curr_snap["passive_rate_mps"]))),
                float(np.max(np.abs(prev_snap["passive_rate_mps"]))),
            )
            rate_delta = float(
                np.max(np.abs(curr_snap["passive_rate_mps"] - prev_snap["passive_rate_mps"])) / rate_scale
            )
        state_components = {
            "q_state_pa_normalized": float(q_delta / q_scale),
            "peq_prev_pa_normalized": float(peq_delta / peq_scale),
            "compression_m_normalized_by_rest_length": float(c_delta),
            "passive_compression_m_normalized_by_rest_length": float(p_delta),
            "passive_rate_mps_normalized": float(rate_delta),
        }
        state_diff_norm = max(state_components.values())
        f_start = f_final[0] if len(f_final) else 0.0
        f_end = f_final[-1] if len(f_final) else 0.0
        peak_f = max(float(np.max(np.abs(force_all))), 1.0e-6)
        force_boundary_err = abs(f_end - f_start) / peak_f
        penult_start = (full_cycles[-2]) * steps_per_cycle
        penult_end = (full_cycles[-2] + 1) * steps_per_cycle
        d_penult = disp_all[penult_start : penult_end + 1]
        f_penult = force_all[penult_start : penult_end + 1]
        loop_force_err = float(np.max(np.abs(f_final - f_penult)) / peak_f)
        penult_work = compute_loop_work(d_penult, f_penult)["work_net_j"]
        final_work = compute_loop_work(d_final, f_final)["work_net_j"]
        loop_work_err = abs(final_work - penult_work) / max(abs(final_work), abs(penult_work), 1.0e-12)
        is_state_closed = bool(
            state_diff_norm <= periodicity_tol
            and force_boundary_err <= periodicity_tol
            and loop_force_err <= periodicity_tol
            and loop_work_err <= periodicity_tol
        )
    elif len(f_final):
        force_boundary_err = float(abs(f_final[-1] - f_final[0]) / max(float(np.max(np.abs(force_all))), 1.0e-6))

    work_metrics = _loop_work_or_none(d_final, f_final)
    qualification = (
        "valid_closed_loop"
        if is_state_closed
        else (f"terminated_early: {termination_reason}" if terminated else "unclosed_residual_state")
    )
    metrics = {
        **work_metrics,
        "peak_force_n": float(np.max(force_all)) if len(force_all) else 0.0,
        "peak_displacement_m": float(np.max(disp_all)) if len(disp_all) else 0.0,
        "peak_column_strain": float(np.max(max_strain_all)) if len(max_strain_all) else 0.0,
        "peak_passive_strain": float(np.max(passive_strain_all)) if len(passive_strain_all) else 0.0,
        "periodicity_error": force_boundary_err,
        "state_difference_norm": state_diff_norm,
        "state_difference_components": state_components,
        "last_two_loop_force_error": loop_force_err,
        "last_two_loop_work_error": loop_work_err,
        "is_state_closed": is_state_closed,
        "dissipation_qualified": is_state_closed,
        "qualification": qualification,
        "first_failure_step": first_failure_step,
        "first_failure_code": termination_code,
    }
    protocol = {
        "source": protocol_source,
        "warmup_cycles": warmup_cycles,
        "total_cycles": total_cycles,
        "prescribed_depth_m": peak_disp_m,
        "period_s": cycle_period_s,
        "dt_s": dt_s,
    }
    stride = max(1, int(round(sample_dt_s / dt_s)))
    display = {
        "time_s": time_all[::stride].tolist(),
        "displacement_m": disp_all[::stride].tolist(),
        "force_n": force_all[::stride].tolist(),
        "cycle_index": cycle_index_all[::stride].tolist(),
    }
    raw_data = {
        "time_s": time_all,
        "displacement_m": disp_all,
        "force_n": force_all,
        "cycle_index": cycle_index_all,
        "max_compression_m": max_comp_all,
        "max_column_strain": max_strain_all,
        "max_passive_strain": passive_strain_all,
        "diagnostic_reason": reason_all[:valid_count],
        "first_loop_displacement_m": d_first,
        "first_loop_force_n": f_first,
        "final_loop_displacement_m": d_final,
        "final_loop_force_n": f_final,
    }
    return {
        "status": "terminated" if terminated else "completed",
        "reason": termination_reason,
        "protocol": protocol,
        "metrics": metrics,
        "raw": raw_data,
        "display": display,
        "final_loop_is_full": final_is_full,
    }


def run_hysteresis(
    artifact: Path | str,
    output: Path | str,
    *,
    fixtures: Sequence[str] = DEFAULT_FIXTURES,
    device: str | wp.Device = "cpu",
    dt_s: float = 6.25e-5,
    sample_dt_s: float = 1.0e-3,
    warmup_cycles: int = 3,
    total_cycles: int = 4,
    prescribed_depth_m: float = 0.010,
    period_s: float = 0.20,
    max_force_cap_n: float = 1.0e5,
    max_strain_cap: float = 0.90,
    periodicity_tol: float = 0.05,
    surround_sweeps: int = INSTRON_SURROUND_SWEEPS,
    use_measured_trace: bool = False,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Run hysteresis bench protocol for requested fixtures on a digital shoe artifact.

    Args:
        artifact: Path to digital_shoe.json artifact.
        output: Directory path where raw curves and summary json will be saved.
        fixtures: Sequence of fixture names to evaluate. Default: ('rearfoot_punch', 'fullfoot_last').
        device: Warp device. Default 'cpu'.
        dt_s: Simulation substep [s]. Default 6.25e-5.
        sample_dt_s: Output curve sampling stride [s] for display JSON. Default 1.0e-3.
        warmup_cycles: Number of preliminary warmup cycles. Default 3.
        total_cycles: Total cycles to run. Default 4.
        prescribed_depth_m: Common peak compression amplitude [m]. Default 0.010 (10 mm).
        period_s: Common cycle period [s]. Default 0.20 s (5 Hz).
        max_force_cap_n: Force limit to halt diverging tests. Default 100 kN.
        max_strain_cap: Strain limit to halt unphysical deformation. Default 0.90.
        periodicity_tol: Relative error threshold for certifying state closure. Default 0.05.
        surround_sweeps: Number of passive surround relaxation sweeps. Default 5.
        use_measured_trace: Whether to use artifact calibration trace. Default False.
        allow_overwrite: Whether to allow overwriting an existing output directory. Default False.

    Returns:
        Summary dictionary containing per-fixture status, file paths, and metrics.
    """
    artifact_path = Path(artifact).resolve()
    output_dir = Path(output).resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"Output directory {output_dir} already exists and is not empty. Refusing to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    shoe = load_artifact(artifact_path)

    summary_fixtures: dict[str, Any] = {}
    fixture_statuses: list[str] = []

    for fix_name in fixtures:
        res = run_fixture_hysteresis(
            shoe,
            fix_name,
            device=device,
            dt_s=dt_s,
            sample_dt_s=sample_dt_s,
            warmup_cycles=warmup_cycles,
            total_cycles=total_cycles,
            prescribed_depth_m=prescribed_depth_m,
            period_s=period_s,
            max_force_cap_n=max_force_cap_n,
            max_strain_cap=max_strain_cap,
            periodicity_tol=periodicity_tol,
            surround_sweeps=surround_sweeps,
            use_measured_trace=use_measured_trace,
        )

        status = res["status"]
        fixture_statuses.append(status)
        if status == "blocked":
            summary_fixtures[fix_name] = {
                "status": "blocked",
                "reason": res["reason"],
                "files": {},
                "protocol": None,
                "metrics": None,
            }
            continue

        raw = res["raw"]
        display = res["display"]
        npz_name = f"{fix_name}_raw.npz"
        json_name = f"{fix_name}_curve.json"
        npz_path = output_dir / npz_name
        json_path = output_dir / json_name

        # Save raw npz with full substep resolution
        np.savez_compressed(npz_path, **raw)

        # Save plot-ready JSON
        curve_data = {
            "fixture": fix_name,
            "first_loop": {
                "displacement_m": raw["first_loop_displacement_m"].tolist(),
                "force_n": raw["first_loop_force_n"].tolist(),
            },
            "final_loop": {
                "displacement_m": raw["final_loop_displacement_m"].tolist(),
                "force_n": raw["final_loop_force_n"].tolist(),
            },
            "all_cycles": display,
            "metrics": res["metrics"],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(curve_data, f, indent=2)

        summary_fixtures[fix_name] = {
            "status": status,
            "reason": res["reason"],
            "files": {
                "npz": str(npz_name),
                "curve_json": str(json_name),
            },
            "protocol": res["protocol"],
            "metrics": res["metrics"],
        }
    # Never report the condition as completed when any requested fixture stopped.
    if fixture_statuses and all(status == "completed" for status in fixture_statuses):
        cond_status = "completed"
    elif any(status == "terminated" for status in fixture_statuses):
        cond_status = "terminated" if not any(status == "blocked" for status in fixture_statuses) else "partial"
    elif any(status == "completed" for status in fixture_statuses):
        cond_status = "partial"
    elif fixture_statuses and all(status == "blocked" for status in fixture_statuses):
        cond_status = "blocked"
    else:
        cond_status = "failed"

    summary = {
        "artifact": str(artifact_path),
        "status": cond_status,
        "fixtures": summary_fixtures,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
