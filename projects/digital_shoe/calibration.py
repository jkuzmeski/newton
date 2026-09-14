# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable device-resident forward solves for the digital-shoe calibration.

This is an execution backend, not a new calibration algorithm. It retains the
legacy interval-ratio stopping estimate, which is not a remaining-error bound.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .runtime import (
    FoundationParams,
    _cycle_force_frame,
    _cycle_overstress_column,
    _surround_sweep_cell,
    surround_seed_driven,
    surround_update_max,
)


@wp.struct
class _SolveSettings:
    fraction: wp.float32
    tau_s: wp.float32
    blend: wp.float32
    over_relaxation: wp.float32


@wp.kernel
def _sweep(
    compression_in: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    slack: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    settings: wp.array[_SolveSettings],
    area: wp.float32,
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    compression_out: wp.array2d[wp.float32],
):
    """Read replaceable material constants for the shared surround sweep."""
    frame, i = wp.tid()
    _surround_sweep_cell(
        frame,
        i,
        compression_in,
        overstress,
        driven,
        neighbors,
        slack,
        params[0],
        area,
        coupling_scale,
        attachment,
        max_strain,
        settings[0].over_relaxation,
        compression_out,
    )


@wp.kernel
def _overstress(
    compression: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    dt_s: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    settings: wp.array[_SolveSettings],
    refreshed: wp.array2d[wp.float32],
):
    """Read replaceable material constants for the shared Maxwell recurrence."""
    _cycle_overstress_column(
        wp.tid(),
        compression,
        slack,
        dt_s,
        params[0],
        settings[0].fraction,
        settings[0].tau_s,
        refreshed,
    )


@wp.kernel
def _force(
    compression: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    area: wp.float32,
    force: wp.array[wp.float32],
):
    """Overwrite each frame's force using the shared fixed column order."""
    frame, lane = wp.tid()
    if lane == 0:
        force[frame] = _cycle_force_frame(frame, compression, overstress, slack, params[0], area)


# NumPy rounds subtract, multiply, and add separately in its float32 blend.
# Isolate this module option so the constitutive kernels retain their original
# floating-point compilation settings.
@wp.kernel(module="unique", module_options={"fuse_fp": False})
def _blend_and_measure(
    compression: wp.array2d[wp.float32],
    previous: wp.array2d[wp.float32],
    carried: wp.array2d[wp.float32],
    refreshed: wp.array2d[wp.float32],
    settings: wp.array[_SolveSettings],
    metrics: wp.array[wp.float32],
):
    """Blend Maxwell fields and reduce pass change and maximum compression."""
    frame, i = wp.tid()
    old = carried[frame, i]
    difference = refreshed[frame, i] - old
    increment = settings[0].blend * difference
    carried[frame, i] = old + increment
    c = compression[frame, i]
    wp.atomic_max(metrics, 0, wp.abs(c - previous[frame, i]))
    wp.atomic_max(metrics, 1, c)


class CalibrationWorkspace:
    """Own reusable geometry, fields, and optional 25-sweep CUDA graphs.

    The returned force and :attr:`compression` are workspace-owned arrays. They
    remain at stable addresses, but the next :meth:`solve` replaces their data.
    A caller that needs older results must explicitly copy them. The workspace
    is not reentrant and must be used on its construction stream.

    Args:
        driven_compression: Imposed compression [m], shape [frames, driven_count].
        driven: Nonzero for directly driven columns, shape [column_count].
        neighbors: Four neighboring column indices, with negative free edges.
        slack_m: Rest thickness [m], shape [column_count].
        dt_s: Frame duration [s], shape [frames].
        area_m2: Tributary area per column [m^2].
        spacing_m: Column grid spacing [m], retained for interface compatibility.
        attachment_n_m: Passive vertical attachment stiffness [N/m].
        max_strain: Maximum compression divided by column rest thickness.
        coupling_scale: Multiplier on the material-derived Pasternak coefficient.
        device: Warp device. None uses the current default device.
        use_graph: Capture CUDA chunks when available; otherwise use eager launches.
    """

    GRAPH_SWEEPS = 25

    def __init__(
        self,
        driven_compression: np.ndarray,
        driven: np.ndarray,
        neighbors: np.ndarray,
        slack_m: np.ndarray,
        dt_s: np.ndarray,
        *,
        area_m2: float,
        spacing_m: float,
        attachment_n_m: float,
        max_strain: float,
        coupling_scale: float = 1.0,
        device=None,
        use_graph: bool = True,
    ):
        self.device = wp.get_device(device)
        driven = np.ascontiguousarray(driven, np.int32)
        neighbors = np.ascontiguousarray(neighbors, np.int32)
        slack_m = np.ascontiguousarray(slack_m, np.float32)
        dt_s = np.ascontiguousarray(dt_s, np.float32)
        imposed = np.ascontiguousarray(driven_compression, np.float32)
        driven_index = np.ascontiguousarray(np.flatnonzero(driven != 0), np.int32)
        count = len(slack_m)
        frames = len(dt_s)
        if frames < 1 or count < 1 or len(driven_index) < 1:
            raise ValueError("calibration needs frames, columns, and driven columns")
        if driven.shape != (count,) or neighbors.shape != (count, 4):
            raise ValueError("calibration geometry must describe the same column count")
        if imposed.shape != (frames, len(driven_index)):
            raise ValueError("driven compression must have shape [frames, driven_count]")
        if dt_s.shape != (frames,) or slack_m.shape != (count,):
            raise ValueError("rest thickness and frame durations must be one-dimensional")
        if not np.all(np.isfinite(slack_m)) or np.any(slack_m <= 0.0):
            raise ValueError("rest thickness must be finite and positive")
        if not np.all(np.isfinite(dt_s)) or np.any(dt_s <= 0.0):
            raise ValueError("frame duration must be finite and positive")
        if np.any(neighbors >= count):
            raise ValueError("neighbor indices must refer to a column or a free edge")
        self.shape = (frames, count)
        self.area_m2 = float(area_m2)
        self.spacing_m = float(spacing_m)
        self.attachment_n_m = float(attachment_n_m)
        self.max_strain = float(max_strain)
        self.coupling_scale = float(coupling_scale)
        self._driven = wp.array(driven, dtype=wp.int32, device=self.device)
        self._neighbors = wp.array(neighbors, dtype=wp.int32, device=self.device)
        self._slack = wp.array(slack_m, dtype=wp.float32, device=self.device)
        self._dt = wp.array(dt_s, dtype=wp.float32, device=self.device)
        self._driven_index = wp.array(driven_index, dtype=wp.int32, device=self.device)
        self._imposed = wp.array(imposed, dtype=wp.float32, device=self.device)
        self._current = wp.zeros(self.shape, dtype=wp.float32, device=self.device)
        self._scratch = wp.zeros_like(self._current)
        self._previous = wp.zeros_like(self._current)
        self._carried = wp.zeros_like(self._current)
        self._refreshed = wp.zeros_like(self._current)
        self._force = wp.zeros(frames, dtype=wp.float32, device=self.device)
        self._update = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._metrics = wp.zeros(2, dtype=wp.float32, device=self.device)
        self._params = wp.zeros(1, dtype=FoundationParams, device=self.device)
        self._settings = wp.zeros(1, dtype=_SolveSettings, device=self.device)
        self._params_host = wp.zeros(1, dtype=FoundationParams, device="cpu")
        self._settings_host = wp.zeros(1, dtype=_SolveSettings, device="cpu")
        self._params_view = self._params_host.numpy()
        self._settings_view = self._settings_host.numpy()
        self.graph = None
        self.graph_fallback_reason: str | None = None
        self.stats: dict = {}
        if use_graph and self.device.is_cuda:
            try:
                # Compile before capture. Captured launches do not execute, so
                # graph preparation cannot advance the warm-start state.
                wp.load_module(module=__name__, device=self.device)
                with wp.ScopedCapture(device=self.device) as capture:
                    self._eager_chunk(self.GRAPH_SWEEPS, measure=True)
                self.graph = capture.graph
            except Exception as exc:
                self.graph_fallback_reason = f"{type(exc).__name__}: {exc}"
        elif use_graph:
            self.graph_fallback_reason = "CUDA graph capture is unavailable on the CPU"
        else:
            self.graph_fallback_reason = "CUDA graph capture was disabled"
        self.use_graph = self.graph is not None

    @property
    def compression(self) -> wp.array2d[wp.float32]:
        """Return the latest compression [m] at the workspace's stable address."""
        return self._current

    def _eager_chunk(self, sweeps: int, *, measure: bool) -> None:
        """Run a chunk and normalize odd parity without changing its sweep count."""
        current, scratch = self._current, self._scratch
        for _ in range(sweeps):
            wp.launch(
                _sweep,
                dim=self.shape,
                inputs=[
                    current,
                    self._carried,
                    self._driven,
                    self._neighbors,
                    self._slack,
                    self._params,
                    self._settings,
                    self.area_m2,
                    self.coupling_scale,
                    self.attachment_n_m,
                    self.max_strain,
                    scratch,
                ],
                device=self.device,
            )
            current, scratch = scratch, current
        if measure:
            self._update.zero_()
            wp.launch(
                surround_update_max,
                dim=self.shape,
                inputs=[scratch, current, self._update],
                device=self.device,
            )
        # The last update must be measured before this copy overwrites the
        # penultimate sweep. All captured pointers stay fixed across solves.
        if current is not self._current:
            wp.copy(self._current, current)

    def solve(
        self,
        params: FoundationParams,
        *,
        fraction: float,
        tau_s: float,
        blend: float,
        initial: wp.array2d[wp.float32] | None = None,
        passes: int,
        tolerance_m: float,
        sweeps: int,
        solve_tolerance_m: float,
        check_every: int,
        over_relaxation: float = 1.0,
    ) -> wp.array[wp.float32]:
        """Run the unchanged legacy nested solve using reusable device buffers.

        Maxwell carried state starts at zero on every call. Only ``initial``
        warms compression; omitting it requests a cold solve. The final force
        uses the refreshed, not blended, Maxwell field of the final compression.

        ``solve_tolerance_m`` is applied to the historical interval-ratio tail
        estimate. That estimator is intentionally preserved for execution A/B
        comparisons and must not be treated as an error bound. A nonpositive
        value disables inner early stopping. Outer stopping uses the largest
        compression change between passes and requires at least two passes.

        Args:
            params: Equilibrium constitutive constants, including both Ogden-Hill terms.
            fraction: Maxwell overstress fraction relative to equilibrium pressure.
            tau_s: Maxwell relaxation time [s].
            blend: Weight of the refreshed Maxwell field in the legacy float32 blend.
            initial: Optional device compression warm start [m].
            passes: Hard cap on Maxwell/surround self-consistency passes.
            tolerance_m: Outer pass-change stopping tolerance [m].
            sweeps: Hard cap on surround sweeps per pass.
            solve_tolerance_m: Legacy estimated remaining-travel threshold [m].
            check_every: Sweeps between inner stopping checks, clamped to at least one.
            over_relaxation: Fraction of the local Newton update taken per sweep.

        Returns:
            Workspace-owned device force history [N], shape [frames].
        """
        if passes < 1:
            raise ValueError("calibration solve needs at least one pass")
        if initial is not None and (
            tuple(initial.shape) != self.shape or initial.dtype != wp.float32 or initial.device != self.device
        ):
            raise ValueError("initial compression must match workspace shape, float32 dtype, and device")
        for name in FoundationParams.vars:
            self._params_view[name][0] = getattr(params, name)
        for name, value in (
            ("fraction", fraction),
            ("tau_s", tau_s),
            ("blend", blend),
            ("over_relaxation", over_relaxation),
        ):
            self._settings_view[name][0] = value
        wp.copy(self._params, self._params_host)
        wp.copy(self._settings, self._settings_host)
        self._carried.zero_()
        if initial is None:
            self._current.zero_()
        elif initial.ptr != self._current.ptr:
            wp.copy(self._current, initial)
        self._previous.zero_()
        interval = max(int(check_every), 1)
        cap = max(int(sweeps), 0)
        changes: list[float] = []
        sweeps_per_pass: list[int] = []
        updates_per_pass: list[float] = []
        remaining_per_pass: list[float] = []
        graph_chunks = 0
        eager_sweeps = 0
        scalar_checks = 0
        maximum = 0.0
        for outer in range(int(passes)):
            wp.launch(
                surround_seed_driven,
                dim=(self.shape[0], self._driven_index.size),
                inputs=[self._imposed, self._driven_index, self._current],
                device=self.device,
            )
            previous_update = float("inf")
            used = 0
            change = float("nan")
            remaining = float("nan")
            while used < cap:
                until_check = interval - used % interval if solve_tolerance_m > 0.0 else cap - used
                chunk = min(self.GRAPH_SWEEPS, cap - used, until_check)
                check = solve_tolerance_m > 0.0 and (used + chunk) % interval == 0
                if self.graph is not None and chunk == self.GRAPH_SWEEPS:
                    wp.capture_launch(self.graph)
                    graph_chunks += 1
                else:
                    self._eager_chunk(chunk, measure=check)
                    eager_sweeps += chunk
                used += chunk
                if check:
                    change = float(self._update.numpy()[0])
                    scalar_checks += 1
                    # Preserve the legacy interval-ratio estimator exactly.
                    # It is not a rigorous tail bound or a corrected per-sweep ratio.
                    decay = change / previous_update if 0.0 < previous_update < float("inf") else float("inf")
                    previous_update = change
                    if change == 0.0:
                        remaining = 0.0
                    elif decay >= 1.0:
                        remaining = float("inf")
                    else:
                        remaining = change * decay / (1.0 - decay)
                    if remaining < solve_tolerance_m:
                        break
            sweeps_per_pass.append(used)
            updates_per_pass.append(change)
            remaining_per_pass.append(remaining)
            wp.launch(
                _overstress,
                dim=self.shape[1],
                inputs=[self._current, self._slack, self._dt, self._params, self._settings, self._refreshed],
                device=self.device,
            )
            self._metrics.zero_()
            wp.launch(
                _blend_and_measure,
                dim=self.shape,
                inputs=[self._current, self._previous, self._carried, self._refreshed, self._settings, self._metrics],
                device=self.device,
            )
            metrics = self._metrics.numpy()
            scalar_checks += 1
            maximum = float(metrics[1])
            if outer:
                changes.append(float(metrics[0]))
            wp.copy(self._previous, self._current)
            if changes and changes[-1] < tolerance_m:
                break
        self.stats = {
            "pass_change_m": changes,
            "max_compression_m": maximum,
            "sweeps_per_pass": sweeps_per_pass,
            "solver_remaining_m": remaining_per_pass[-1],
            "solver_update_m": updates_per_pass[-1],
            "solver_remaining_per_pass_m": remaining_per_pass,
            "solver_update_per_pass_m": updates_per_pass,
            "stopping_estimator": "legacy_interval_ratio_estimate_not_bound",
            "graph_chunks": graph_chunks,
            "eager_sweeps": eager_sweeps,
            "scalar_checks": scalar_checks,
            "graph_enabled": self.use_graph,
            "graph_fallback_reason": self.graph_fallback_reason,
        }
        wp.launch(
            _force,
            dim=self.shape,
            inputs=[self._current, self._refreshed, self._slack, self._params, self.area_m2, self._force],
            device=self.device,
        )
        return self._force
