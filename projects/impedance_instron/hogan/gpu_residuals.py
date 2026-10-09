# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""On-device Levenberg-Marquardt residuals and normal equations for runner fits.

:class:`GpuResiduals` integrates candidate runners with the trace-free worlds of
:mod:`.gpu_runner` and writes the weighted residual vector of
:func:`.least_squares.residuals` directly into a resident row per candidate. The
Jacobian difference, ``J J^T`` and ``J r`` products then run on the device too, so
an LM iteration downloads only per-world screens, a few metrics, and one small
normal-equation matrix.

Measured targets are uploaded once and are read only by an observer kernel that
runs after each accepted integration step. They never enter actuation, contact,
or integration, so dynamics remain target-free exactly as in trace mode.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from .gpu_mechanics import Vec6
from .gpu_runner import _Buffers, _Group, _Inputs, _validate_models
from .runner import RolloutConfig, Runner

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

_HIP_SCALE_M = 0.02
_ANGLE_SCALE_RAD = 0.05
_FORCE_SCALE_N = 100.0
_TILE = 32
_DEPTH = 64
_CHUNK_TILES = 32
_COLUMN_BLOCK = _DEPTH * _CHUNK_TILES


@wp.struct
class _Targets:
    obs_count: wp.array[int]
    obs_trigger: wp.array2d[int]
    obs_node: wp.array2d[int]
    obs_offset: wp.array2d[wp.float64]
    obs_dx: wp.array2d[wp.float64]
    obs_q: wp.array2d[Vec6]
    grf: wp.array2d[wp.vec2d]
    offset: wp.array[int]
    hip_scale: wp.array[wp.float64]
    angle_scale: wp.array[wp.float64]
    force_scale: wp.array[wp.float64]
    weight: wp.float64
    # Per-trial force observation (identify.ForceObservation); observe == 0 keeps raw GRF.
    observe: wp.array[int]
    filter_b: wp.array[wp.vec3d]
    filter_a: wp.array[wp.vec2d]
    gate: wp.array[wp.float64]
    # Weight and first column of each trial's unobservable-vibration residuals (weight 0: no block).
    vibration: wp.array[wp.float64]
    vibration_offset: wp.array[int]


@wp.struct
class _Accumulators:
    seen: wp.array[int]
    cursor: wp.array[int]
    sumsq: wp.array[wp.float64]
    tracking: wp.array[Vec6]
    force: wp.array[wp.vec2d]
    peak: wp.array[wp.float64]
    rows: wp.array[int]
    residuals: wp.array2d[wp.float64]
    filter_z1: wp.array[wp.vec2d]
    filter_z2: wp.array[wp.vec2d]


@wp.kernel
def _clear(acc: _Accumulators):
    w = wp.tid()
    acc.seen[w] = 0
    acc.cursor[w] = 0
    acc.sumsq[w] = wp.float64(0.0)
    acc.tracking[w] = Vec6(wp.float64(0.0))
    acc.force[w] = wp.vec2d(wp.float64(0.0))
    acc.peak[w] = wp.float64(-1.0e300)
    acc.filter_z1[w] = wp.vec2d(wp.float64(0.0))
    acc.filter_z2[w] = wp.vec2d(wp.float64(0.0))


@wp.func
def _biquad_step(
    x: wp.vec2d, b: wp.vec3d, a: wp.vec2d, z1: wp.vec2d, z2: wp.vec2d
) -> tuple[wp.vec2d, wp.vec2d, wp.vec2d]:
    """Advance identify._biquad by one sample with its exact operation order."""
    y = wp.vec2d(b[0] * x[0] + z1[0], b[0] * x[1] + z1[1])
    n1 = wp.vec2d(b[1] * x[0] - a[0] * y[0] + z2[0], b[1] * x[1] - a[0] * y[1] + z2[1])
    n2 = wp.vec2d(b[2] * x[0] - a[1] * y[0], b[2] * x[1] - a[1] * y[1])
    return y, n1, n2


@wp.kernel
def _observe(trials: int, data: _Buffers, targets: _Targets, acc: _Accumulators):
    """Write residuals of the newly accepted interval, mirroring ``np.interp`` exactly."""
    w = wp.tid()
    k = acc.seen[w]
    # Idle candidate slots never record, so the graph needs no captured batch size.
    if data.recorded[w] <= k:
        return
    c = w // trials
    s = w % trials
    row = acc.rows[c]
    base = targets.offset[s]
    count = targets.obs_count[s]
    weight = targets.weight
    grf = data.grf[w]
    column = base + 6 * count + 2 * k
    total = acc.sumsq[w]
    if targets.observe[s] != 0:
        # Stash the forward filter pass (and the raw force for the vibration block);
        # _finish_observed forms these residuals after the rollout.
        y, z1, z2 = _biquad_step(grf, targets.filter_b[s], targets.filter_a[s], acc.filter_z1[w], acc.filter_z2[w])
        acc.filter_z1[w] = z1
        acc.filter_z2[w] = z2
        acc.residuals[row, column] = y[0]
        acc.residuals[row, column + 1] = y[1]
        if targets.vibration[s] > wp.float64(0.0):
            stash = targets.vibration_offset[s] + 2 * k
            acc.residuals[row, stash] = grf[0]
            acc.residuals[row, stash + 1] = grf[1]
    else:
        target = targets.grf[s, k]
        ex = grf[0] - target[0]
        ez = grf[1] - target[1]
        rx = weight * (ex / targets.force_scale[s])
        rz = weight * (ez / targets.force_scale[s])
        acc.residuals[row, column] = rx
        acc.residuals[row, column + 1] = rz
        total = acc.sumsq[w] + rx * rx + rz * rz
        acc.force[w] = acc.force[w] + wp.vec2d(ex * ex, ez * ez)
        acc.peak[w] = wp.max(acc.peak[w], grf[1])
    q0 = data.previous[w]
    q1 = data.state[w]
    tracking = acc.tracking[w]
    j = acc.cursor[w]
    while j < count and targets.obs_trigger[s, j] == k:
        simulated = q1
        if targets.obs_node[s, j] == 0:
            dx = targets.obs_dx[s, j]
            offset = targets.obs_offset[s, j]
            for m in range(6):
                simulated[m] = (q1[m] - q0[m]) / dx * offset + q0[m]
        error = simulated - targets.obs_q[s, j]
        for m in range(2):
            r = weight * (error[m] / targets.hip_scale[s])
            acc.residuals[row, base + 2 * j + m] = r
            total += r * r
        for m in range(4):
            r = weight * (error[m + 2] / targets.angle_scale[s])
            acc.residuals[row, base + 2 * count + 4 * j + m] = r
            total += r * r
        for m in range(6):
            tracking[m] = tracking[m] + error[m] * error[m]
        j += 1
    acc.tracking[w] = tracking
    acc.cursor[w] = j
    acc.sumsq[w] = total
    acc.seen[w] = k + 1


@wp.kernel
def _finish_observed(trials: int, data: _Buffers, targets: _Targets, acc: _Accumulators):
    """Run the backward filter pass and gate over each completed world, then write its GRF residuals.

    Mirrors ``identify.ForceObservation.apply`` followed by the GRF part of
    ``least_squares.residuals``; the forward pass was stored by :func:`_observe`.
    """
    w = wp.tid()
    s = w % trials
    # Only completed rollouts' residuals are consumed.
    if targets.observe[s] == 0 or data.status[w] != 1:
        return
    row = acc.rows[w // trials]
    base = targets.offset[s] + 6 * targets.obs_count[s]
    b = targets.filter_b[s]
    a = targets.filter_a[s]
    gate = targets.gate[s]
    scale = targets.force_scale[s]
    weight = targets.weight
    vibration = targets.vibration[s]
    stash = targets.vibration_offset[s]
    z1 = wp.vec2d(wp.float64(0.0))
    z2 = wp.vec2d(wp.float64(0.0))
    total = acc.sumsq[w]
    force = acc.force[w]
    peak = acc.peak[w]
    steps = data.recorded[w]
    for i in range(steps):
        k = steps - 1 - i
        column = base + 2 * k
        x = wp.vec2d(acc.residuals[row, column], acc.residuals[row, column + 1])
        y, z1, z2 = _biquad_step(x, b, a, z1, z2)
        if y[1] < gate:
            y = wp.vec2d(wp.float64(0.0))
        target = targets.grf[s, k]
        ex = y[0] - target[0]
        ez = y[1] - target[1]
        rx = weight * (ex / scale)
        rz = weight * (ez / scale)
        acc.residuals[row, column] = rx
        acc.residuals[row, column + 1] = rz
        total += rx * rx + rz * rz
        force = force + wp.vec2d(ex * ex, ez * ez)
        peak = wp.max(peak, y[1])
        if vibration > wp.float64(0.0):
            j = stash + 2 * k
            vx = weight * ((vibration * (acc.residuals[row, j] - y[0])) / scale)
            vz = weight * ((vibration * (acc.residuals[row, j + 1] - y[1])) / scale)
            acc.residuals[row, j] = vx
            acc.residuals[row, j + 1] = vz
            total += vx * vx + vz * vz
    acc.sumsq[w] = total
    acc.force[w] = force
    acc.peak[w] = peak


@wp.kernel
def _difference(
    store: wp.array2d[wp.float64],
    plus: wp.array[int],
    minus: wp.array[int],
    denominator: wp.array[wp.float64],
):
    """Form Jacobian rows in place; a zero denominator marks a failed column."""
    i, k = wp.tid()
    den = denominator[i]
    if den == wp.float64(0.0):
        store[i, k] = wp.float64(0.0)
    else:
        store[i, k] = (store[plus[i], k] - store[minus[i], k]) / den


@wp.kernel
def _copy_row(store: wp.array2d[wp.float64], source: int, target: int):
    k = wp.tid()
    store[target, k] = store[source, k]


@wp.kernel
def _gram_partial(store: wp.array2d[wp.float64], partial: wp.array3d[wp.float64]):
    """Accumulate one fixed column block of ``A A^T``; a second pass sums blocks in order."""
    chunk, bi, bj = wp.tid()
    total = wp.tile_zeros(shape=(_TILE, _TILE), dtype=wp.float64)
    for t in range(_CHUNK_TILES):
        k0 = chunk * _COLUMN_BLOCK + t * _DEPTH
        a = wp.tile_load(store, shape=(_TILE, _DEPTH), offset=(bi * _TILE, k0))
        b = wp.tile_load(store, shape=(_TILE, _DEPTH), offset=(bj * _TILE, k0))
        wp.tile_matmul(a, wp.tile_transpose(b), total)
    wp.tile_store(partial[chunk], total, offset=(bi * _TILE, bj * _TILE))


@wp.kernel
def _gram_sum(partial: wp.array3d[wp.float64], gram: wp.array2d[wp.float64]):
    i, j = wp.tid()
    total = wp.float64(0.0)
    for chunk in range(partial.shape[0]):
        total += partial[chunk, i, j]
    gram[i, j] = total


class _Observer:
    """Residual accumulation hooked into one persistent group's step graph."""

    def __init__(self, group: _Group, targets: _Targets, store: wp.array2d):
        d, w = group.device, group.world_count
        self.group = group
        self.targets = targets
        acc = _Accumulators()
        acc.seen = wp.zeros(w, dtype=int, device=d)
        acc.cursor = wp.zeros(w, dtype=int, device=d)
        acc.sumsq = wp.zeros(w, dtype=wp.float64, device=d)
        acc.tracking = wp.zeros(w, dtype=Vec6, device=d)
        acc.force = wp.zeros(w, dtype=wp.vec2d, device=d)
        acc.peak = wp.zeros(w, dtype=wp.float64, device=d)
        acc.rows = wp.zeros(group.candidates, dtype=int, device=d)
        acc.residuals = store
        acc.filter_z1 = wp.zeros(w, dtype=wp.vec2d, device=d)
        acc.filter_z2 = wp.zeros(w, dtype=wp.vec2d, device=d)
        self.acc = acc

    def reset(self):
        wp.launch(_clear, dim=self.group.world_count, inputs=[self.acc], device=self.group.device)

    def finish(self):
        """Form observed GRF residuals once every world has stopped integrating."""
        g = self.group
        wp.launch(
            _finish_observed,
            dim=g.world_count,
            inputs=[g.trial_count, g.data, self.targets, self.acc],
            device=g.device,
        )

    def step(self):
        g = self.group
        wp.launch(
            _observe,
            dim=g.world_count,
            inputs=[g.trial_count, g.data, self.targets, self.acc],
            device=g.device,
        )


def _interp_plan(times: np.ndarray, grid: np.ndarray):
    """Map observation times onto accepted steps exactly as ``np.interp`` would evaluate them."""
    last = len(grid) - 1
    trigger, node, offset, dx = [], [], [], []
    for t in times:
        j = int(np.searchsorted(grid, t, side="right")) - 1
        if j >= last or grid[j] == t:
            # Nodes (including the clamped horizon end) are the state after step n - 1.
            n = min(j, last)
            trigger.append(n - 1)
            node.append(1)
            offset.append(0.0)
            dx.append(1.0)
        else:
            trigger.append(j)
            node.append(0)
            offset.append(t - grid[j])
            dx.append(grid[j + 1] - grid[j])
    return trigger, node, offset, dx


class GpuResiduals:
    """Evaluate weighted LM residual rows and normal equations of candidate runners.

    Residual rows have exactly the layout of the concatenated
    :func:`.least_squares.residuals` vectors in trial order, scaled by
    ``1 / sqrt(len(trials))``. Every trial must have at least one observation
    after its prediction origin.

    Args:
        trials: Offline :class:`.identify.Trial` records; only predictive inputs
            reach the dynamics.
        config: Shared integration and numerical-screen settings.
        rows: Resident residual rows available to :meth:`evaluate`.
        device: CUDA device.
        chunk: Maximum candidates integrated concurrently per group.
        chunk_steps: Maximum steps per reusable captured graph.

    Attributes:
        length: Residual entries per row.
        rollouts: Candidate-trial rollouts integrated so far.
    """

    def __init__(
        self,
        trials: list,
        config: RolloutConfig,
        *,
        rows: int,
        device: str = "cuda:0",
        chunk: int = 128,
        chunk_steps: int = 64,
    ):
        if isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        self.trials = list(trials)
        shoes = [t.shoe for t in self.trials]
        self.inputs = _Inputs(
            [t.chain for t in self.trials],
            shoes,
            [t.initial for t in self.trials],
            [t.task for t in self.trials],
            [t.duration_s for t in self.trials],
            config,
            device,
            chunk_steps,
        )
        inputs = self.inputs
        self.device = inputs.device
        self.config = inputs.config
        self.chunk = int(chunk)
        self.rollouts = 0
        self.group_trial_indices = inputs.shoe_groups(shoes, len(self.trials))
        weight = 1.0 / math.sqrt(len(self.trials))
        plans, offsets, position = [], [], 0
        self.measured_peak = np.zeros(len(self.trials))
        for index, trial in enumerate(self.trials):
            steps, dt = int(inputs.steps[index]), float(inputs.dts[index])
            # Residuals are only consumed for completed rollouts, which end on this clock.
            grid = np.arange(steps + 1) * dt
            observed = (trial.time_s > 0) & (trial.time_s <= grid[-1] + 1e-12)
            count = int(observed.sum())
            if not count:
                raise ValueError("Each trial needs an observation after its prediction origin")
            grf = np.column_stack([np.interp(grid[:-1], trial.force_time_s, trial.grf_n[:, c]) for c in range(2)])
            observation = trial.force_observation
            vibration = 0.0 if observation is None else observation.vibration_weight
            observe = (
                (0, (0.0,) * 5, 0.0) if observation is None else (1, observation.coefficients(dt), observation.gate_n)
            )
            stash = position + 6 * count + 2 * steps
            plans.append(
                (
                    count,
                    _interp_plan(trial.time_s[observed], grid),
                    trial.q[observed],
                    grf,
                    (*observe, vibration, stash),
                )
            )
            offsets.append(position)
            position += 6 * count + 2 * steps * (2 if vibration > 0 else 1)
            interior = (trial.force_time_s > 0) & (trial.force_time_s < trial.duration_s)
            clock = np.concatenate(([0.0], trial.force_time_s[interior], [trial.duration_s]))
            self.measured_peak[index] = np.interp(clock, trial.force_time_s, trial.grf_n[:, 1]).max()
        self.length = position
        self.offsets = np.asarray(offsets)
        self.row_count = int(rows)
        if self.row_count < 1:
            raise ValueError("rows must be positive")
        padded_rows = _TILE * math.ceil(self.row_count / _TILE)
        padded_length = _COLUMN_BLOCK * math.ceil(position / _COLUMN_BLOCK)
        self.store = wp.zeros((padded_rows, padded_length), dtype=wp.float64, device=self.device)
        self._observed = False
        self._targets = [
            self._upload([plans[i] for i in indices], [offsets[i] for i in indices], weight)
            for indices in self.group_trial_indices
        ]
        self._groups = {}
        self._partial = None
        self._gram = None

    def _upload(self, plans, offsets, weight) -> _Targets:
        d = self.device
        width = max(count for count, *_ in plans)
        length = max(len(grf) for _, _, _, grf, _ in plans)
        trigger = np.full((len(plans), width), -1, dtype=np.int32)
        node = np.zeros((len(plans), width), dtype=np.int32)
        offset = np.zeros((len(plans), width))
        dx = np.ones((len(plans), width))
        q = np.zeros((len(plans), width, 6))
        grf = np.zeros((len(plans), length, 2))
        for s, (count, plan, observed, force, _) in enumerate(plans):
            trigger[s, :count], node[s, :count], offset[s, :count], dx[s, :count] = plan
            q[s, :count] = observed
            grf[s, : len(force)] = force
        counts = np.array([count for count, *_ in plans])
        steps = np.array([len(force) for _, _, _, force, _ in plans])
        observe = [plan[-1] for plan in plans]
        targets = _Targets()
        targets.obs_count = wp.array(counts, dtype=int, device=d)
        targets.obs_trigger = wp.array(trigger, dtype=int, device=d)
        targets.obs_node = wp.array(node, dtype=int, device=d)
        targets.obs_offset = wp.array(offset, dtype=wp.float64, device=d)
        targets.obs_dx = wp.array(dx, dtype=wp.float64, device=d)
        targets.obs_q = wp.array(q, dtype=Vec6, device=d)
        targets.grf = wp.array(grf, dtype=wp.vec2d, device=d)
        targets.offset = wp.array(np.asarray(offsets), dtype=int, device=d)
        targets.hip_scale = wp.array([_HIP_SCALE_M * math.sqrt(2 * n) for n in counts], dtype=wp.float64, device=d)
        targets.angle_scale = wp.array(
            [_ANGLE_SCALE_RAD * math.sqrt(4 * n) for n in counts], dtype=wp.float64, device=d
        )
        targets.force_scale = wp.array([_FORCE_SCALE_N * math.sqrt(2 * n) for n in steps], dtype=wp.float64, device=d)
        targets.weight = weight
        targets.observe = wp.array([flag for flag, *_ in observe], dtype=int, device=d)
        targets.filter_b = wp.array([c[:3] for _, c, *_ in observe], dtype=wp.vec3d, device=d)
        targets.filter_a = wp.array([c[3:] for _, c, *_ in observe], dtype=wp.vec2d, device=d)
        targets.gate = wp.array([gate for _, _, gate, *_ in observe], dtype=wp.float64, device=d)
        targets.vibration = wp.array([v for *_, v, _ in observe], dtype=wp.float64, device=d)
        targets.vibration_offset = wp.array([o for *_, o in observe], dtype=int, device=d)
        self._observed = self._observed or any(flag for flag, *_ in observe)
        return targets

    def _group(self, index: int, capacity: int) -> tuple[_Group, _Observer]:
        key = (index, capacity)
        if key not in self._groups:
            t = self.trials
            group = _Group(
                self.inputs,
                self.group_trial_indices[index],
                [x.chain for x in t],
                [x.shoe for x in t],
                self.inputs.initials,
                [x.task for x in t],
                candidates=capacity,
                record=False,
            )
            observer = _Observer(group, self._targets[index], self.store)
            group.observers.append(observer)
            self._groups[key] = (group, observer)
        return self._groups[key]

    def evaluate(self, models: list[Runner], rows, *, metrics: bool = False, fast: bool = False):
        """Integrate ``models`` and write each residual vector into its ``rows`` entry.

        With ``fast``, lean shoes use fast-math kernels, which agree with the exact
        rollouts only to float32 intrinsic rounding (see
        :func:`.gpu_shoe.apply_ground_shoe`).

        Returns:
            ``(completed, sumsq, motion)``: whether every trial completed, the
            residual sum of squares, and per-candidate lists of per-trial metric
            dictionaries (``None`` unless requested), each candidate in input order.
        """
        rows = np.asarray(rows, dtype=np.int32)
        if not models or rows.shape != (len(models),) or rows.min() < 0 or rows.max() >= self.row_count:
            raise ValueError("Each model needs one valid residual row")
        _validate_models(models, self.inputs.initials)
        n, trials = len(models), len(self.trials)
        status = np.zeros((n, trials), dtype=np.int32)
        sumsq = np.zeros((n, trials))
        extra = {name: np.zeros((n, trials, size)) for name, size in (("tracking", 6), ("force", 2))}
        peak = np.zeros((n, trials))
        counts = np.zeros((n, trials, 2), dtype=np.int64)
        for index, indices in enumerate(self.group_trial_indices):
            capacity = min(n, self.chunk)
            group, observer = self._group(index, capacity)
            group.fast_shoe = bool(fast)
            for start in range(0, n, capacity):
                block = models[start : start + capacity]
                chosen = np.zeros(capacity, dtype=np.int32)
                chosen[: len(block)] = rows[start : start + len(block)]
                observer.acc.rows.assign(chosen)
                group.launch(block)
                if self._observed:
                    observer.finish()
                self.rollouts += len(block) * len(indices)
                size = len(block) * len(indices)
                shape = (len(block), len(indices))
                lanes = slice(start, start + len(block))
                columns = list(indices)
                status[lanes, columns] = group.data.status.numpy()[:size].reshape(shape)
                sumsq[lanes, columns] = observer.acc.sumsq.numpy()[:size].reshape(shape)
                if metrics:
                    acc = observer.acc
                    extra["tracking"][lanes, columns] = acc.tracking.numpy()[:size].reshape((*shape, 6))
                    extra["force"][lanes, columns] = acc.force.numpy()[:size].reshape((*shape, 2))
                    peak[lanes, columns] = acc.peak.numpy()[:size].reshape(shape)
                    counts[lanes, columns, 0] = acc.cursor.numpy()[:size].reshape(shape)
                    counts[lanes, columns, 1] = group.data.recorded.numpy()[:size].reshape(shape)
        completed = np.all(status == 1, axis=1)
        motion = None
        if metrics:
            motion = []
            for c in range(n):
                scores = []
                for s in range(trials):
                    observed, steps = counts[c, s]
                    tracking = np.sqrt(extra["tracking"][c, s] / observed) if observed else np.zeros(6)
                    force = np.sqrt(extra["force"][c, s] / steps) if steps else np.zeros(2)
                    scores.append(
                        {
                            "tracking_rmse": tracking.tolist(),
                            "grf_rmse_n": force.tolist(),
                            "peak_fz_error_n": float((peak[c, s] if steps else 0.0) - self.measured_peak[s]),
                            "status": "completed" if status[c, s] == 1 else "failed",
                        }
                    )
                motion.append(scores)
        return completed, sumsq.sum(axis=1), motion

    def copy_row(self, source: int, target: int) -> None:
        """Copy one resident residual row to another without a host round trip."""
        wp.launch(_copy_row, dim=self.length, inputs=[self.store, source, target], device=self.device)

    def normal(self, plus, minus, denominator, reference: int) -> tuple[np.ndarray, np.ndarray]:
        """Form ``J`` in rows ``[0, n)`` and return ``J J^T`` and ``J r``.

        Column ``i`` of the Jacobian is ``(row[plus[i]] - row[minus[i]]) /
        denominator[i]``, or zero when the denominator is zero. ``reference``
        holds ``r``. The inputs of column ``i`` must not be another column's
        output row; LM perturbations use rows ``i`` and ``n + i`` and ``r``.
        """
        plus, minus = np.asarray(plus, dtype=np.int32), np.asarray(minus, dtype=np.int32)
        denominator = np.asarray(denominator, dtype=np.float64)
        n = len(plus)
        if n + 1 > self.row_count or reference < n:
            raise ValueError("Residual rows cannot hold the Jacobian and reference")
        d = self.device
        wp.launch(
            _difference,
            dim=(n, self.length),
            inputs=[
                self.store,
                wp.array(plus, dtype=int, device=d),
                wp.array(minus, dtype=int, device=d),
                wp.array(denominator, dtype=wp.float64, device=d),
            ],
            device=d,
        )
        if reference != n:
            self.copy_row(reference, n)
        tiles = math.ceil((n + 1) / _TILE)
        chunks = self.store.shape[1] // _COLUMN_BLOCK
        if self._partial is None or self._partial.shape[1] < tiles * _TILE:
            self._partial = wp.zeros((chunks, tiles * _TILE, tiles * _TILE), dtype=wp.float64, device=d)
            self._gram = wp.zeros((tiles * _TILE, tiles * _TILE), dtype=wp.float64, device=d)
        wp.launch_tiled(
            _gram_partial, dim=(chunks, tiles, tiles), inputs=[self.store, self._partial], block_dim=64, device=d
        )
        wp.launch(_gram_sum, dim=(tiles * _TILE, tiles * _TILE), inputs=[self._partial, self._gram], device=d)
        gram = self._gram.numpy()[: n + 1, : n + 1]
        return gram[:n, :n].copy(), gram[:n, n].copy()
