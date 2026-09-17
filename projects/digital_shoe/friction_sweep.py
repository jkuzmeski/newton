# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident friction-only parameter search against a frozen normal history.

Normal reactions and kinematics are immutable shared inputs. This is an effective
single-experiment identification, not independent outsole-friction calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from .contact import bristle_step
from .friction_deflection import bristle_deflection_step
from .friction_law import regularized_force_tangent
from .friction_metrics import compute_braking_propulsive_impulses, compute_force_peaks

wp.set_module_options({"enable_backward": False})

METHODS = {"legacy": 0, "deflection": 1, "regularized": 2, "anchor_nominal": 3}
PARAMETER_NAMES = ("method", "mu", "kt_scale", "kv_scale", "viscous_ratio", "release_dwell_s", "yield_width")
SCORE_NAMES = (
    "loss",
    "rmse_n",
    "braking_impulse_ns",
    "propulsive_impulse_ns",
    "braking_peak_n",
    "propulsive_peak_n",
    "positive_energy_residual_j",
    "max_deflection_m",
    "cone_excess_n",
    "work_j",
)


@wp.kernel
def _advance(
    count: int,
    dt: float,
    clock: wp.array[int],
    settings: wp.array2d[float],
    position: wp.array2d[wp.vec2],
    velocity: wp.array2d[wp.vec2],
    nominal_velocity: wp.array2d[wp.vec2],
    normal: wp.array2d[float],
    base_kt: wp.array[float],
    base_kv: wp.array[float],
    anchor: wp.array[wp.vec2],
    deflection: wp.array[wp.vec2],
    grip: wp.array[int],
    dwell: wp.array[float],
    stored_energy: wp.array[float],
    column_force: wp.array[wp.vec2],
    diagnostic: wp.array[wp.vec4],
):
    i = wp.tid()
    w = i // count
    c = i % count
    t = clock[0]
    method = int(settings[w, 0])
    mu = settings[w, 1]
    kt = base_kt[c] * settings[w, 2]
    kv = base_kv[c] * settings[w, 3]
    gamma = settings[w, 4]
    release = settings[w, 5]
    width = settings[w, 6]
    v = velocity[t, c]
    pos = position[t, c]
    n = normal[t, c]
    force = wp.vec2(0.0)
    z = deflection[i]
    a = anchor[i]
    s = grip[i]
    elapsed = dwell[i]
    old_energy = stored_energy[i]
    if method == 1:
        force, _jac, z, s, elapsed = bristle_deflection_step(v, dt, n, kt, kv, mu, gamma, release, width, z, s, elapsed)
    elif method == 2:
        force, _jac = regularized_force_tangent(v, dt, n, mu, 0.01)
        z = wp.vec2(0.0)
        s = 0
        elapsed = 0.0
    else:
        used_v = v
        if method == 3:
            used_v = nominal_velocity[t, c]
        force, a, s, elapsed = bristle_step(pos, used_v, dt, n, kt, kv, mu, gamma, release, a, s, elapsed)
        if n > 0.0 and kt > 0.0:
            z = pos + dt * used_v - a
        elif s == 0:
            z = wp.vec2(0.0)
    energy = 0.5 * kt * wp.dot(z, z)
    work = wp.dot(force, v) * dt
    # Use the accepted ground-point velocity for carrier work in every method.
    residual = energy - old_energy + work
    tolerance = 1.0e-6 * (energy + old_energy + wp.abs(work) + 1.0e-6)
    violation = wp.max(residual - tolerance, 0.0)
    cone_excess = wp.max(wp.length(force) - mu * wp.max(n, 0.0), 0.0)
    anchor[i] = a
    deflection[i] = z
    grip[i] = s
    dwell[i] = elapsed
    stored_energy[i] = energy
    column_force[i] = force
    diagnostic[i] = wp.vec4(violation, wp.length(z), cone_excess, work)


@wp.kernel
def _reduce_columns(
    count: int,
    groups: int,
    force: wp.array[wp.vec2],
    diagnostic: wp.array[wp.vec4],
    partial_force: wp.array[wp.vec2],
    partial_diagnostic: wp.array[wp.vec4],
):
    i = wp.tid()
    w = i // groups
    group = i % groups
    f = wp.vec2(0.0)
    d = wp.vec4(0.0)
    for c in range(group, count, groups):
        k = w * count + c
        f = f + force[k]
        value = diagnostic[k]
        d = wp.vec4(d[0] + value[0], wp.max(d[1], value[1]), wp.max(d[2], value[2]), d[3] + value[3])
    partial_force[i] = f
    partial_diagnostic[i] = d


@wp.kernel
def _finish_frame(
    groups: int,
    clock: wp.array[int],
    partial_force: wp.array[wp.vec2],
    partial_diagnostic: wp.array[wp.vec4],
    curves: wp.array2d[wp.vec2],
    totals: wp.array[wp.vec4],
):
    w = wp.tid()
    f = wp.vec2(0.0)
    d = totals[w]
    for group in range(groups):
        i = w * groups + group
        f = f + partial_force[i]
        value = partial_diagnostic[i]
        d = wp.vec4(d[0] + value[0], wp.max(d[1], value[1]), wp.max(d[2], value[2]), d[3] + value[3])
    curves[w, clock[0]] = f
    totals[w] = d


@wp.kernel
def _tick(clock: wp.array[int], condition: wp.array[int], limit: int):
    clock[0] = clock[0] + 1
    condition[0] = int(clock[0] < limit)


@wp.kernel
def _score(
    samples: int,
    target: wp.array[float],
    target_time: wp.array[float],
    active: wp.array[int],
    lower: wp.array[int],
    upper: wp.array[int],
    fraction: wp.array[float],
    targets: wp.vec4,
    force_scale: float,
    curves: wp.array2d[wp.vec2],
    totals: wp.array[wp.vec4],
    scores: wp.array2d[float],
):
    w = wp.tid()
    squared = float(0.0)
    brake = float(0.0)
    prop = float(0.0)
    brake_peak = float(0.0)
    prop_peak = float(0.0)
    previous = float(0.0)
    for j in range(samples):
        a = fraction[j]
        f = (1.0 - a) * curves[w, lower[j]][0] + a * curves[w, upper[j]][0]
        diff = f - target[j]
        squared += diff * diff
        if active[j] != 0:
            brake_peak = wp.max(brake_peak, -f)
            prop_peak = wp.max(prop_peak, f)
        if j > 0 and active[j] != 0 and active[j - 1] != 0:
            h = target_time[j] - target_time[j - 1]
            if previous >= 0.0 and f >= 0.0:
                prop += 0.5 * h * (previous + f)
            elif previous <= 0.0 and f <= 0.0:
                brake -= 0.5 * h * (previous + f)
            else:
                split = -previous / (f - previous)
                if previous > 0.0:
                    prop += 0.5 * h * split * previous
                    brake -= 0.5 * h * (1.0 - split) * f
                else:
                    brake -= 0.5 * h * split * previous
                    prop += 0.5 * h * (1.0 - split) * f
        previous = f
    mse = squared / float(samples)
    b_error = (brake - targets[0]) / wp.max(targets[0], 1.0)
    p_error = (prop - targets[1]) / wp.max(targets[1], 1.0)
    bp_error = (brake_peak - targets[2]) / wp.max(targets[2], 1.0)
    pp_error = (prop_peak - targets[3]) / wp.max(targets[3], 1.0)
    loss = mse / (force_scale * force_scale) + 0.25 * (b_error * b_error + p_error * p_error)
    loss += 0.1 * (bp_error * bp_error + pp_error * pp_error)
    scores[w, 0] = loss
    scores[w, 1] = wp.sqrt(mse)
    scores[w, 2] = brake
    scores[w, 3] = prop
    scores[w, 4] = brake_peak
    scores[w, 5] = prop_peak
    d = totals[w]
    scores[w, 6] = d[0]
    scores[w, 7] = d[1]
    scores[w, 8] = d[2]
    scores[w, 9] = d[3]


class FrictionSweep:
    """Evaluate independent parameter candidates without replaying normal mechanics.

    A device-side graph loop keeps all bristle history and reductions resident.
    Each world owns its state. Only the small score table is normally downloaded.
    The host retains unchanged input arrays for reproducibility, not per-step work.
    """

    def __init__(self, arrays: dict, batch_size: int = 128, *, device=None, use_graph: bool = True):
        self.arrays = arrays
        self.device = wp.get_device(device)
        self.batch_size = int(batch_size)
        normal = np.asarray(arrays["normal_n"], np.float32)
        self.steps, self.columns = normal.shape
        self.groups = min(self.columns, 16)
        times = np.asarray(arrays["time_s"], np.float64)
        increments = np.diff(times)
        if self.batch_size < 1 or self.steps < 2 or self.columns < 1:
            raise ValueError("Sweep requires positive batch/columns and at least two timesteps")
        if (
            not np.isfinite(times).all()
            or np.any(increments <= 0)
            or not np.allclose(increments, increments[0], rtol=1e-6, atol=1e-12)
        ):
            raise ValueError("Frozen history needs a strictly increasing uniform clock")
        self.dt = float(increments[0])
        for key, shape in (
            ("normal_n", normal.shape),
            ("position_xy", (*normal.shape, 2)),
            ("velocity_xy", (*normal.shape, 2)),
            ("nominal_velocity_xy", (*normal.shape, 2)),
            ("baseline_kt_n_m", (self.columns,)),
            ("baseline_kv_ns_m", (self.columns,)),
        ):
            values = np.asarray(arrays[key])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid frozen friction array {key}")
        if (
            np.any(normal < 0)
            or np.any(np.asarray(arrays["baseline_kt_n_m"]) <= 0)
            or np.any(np.asarray(arrays["baseline_kv_ns_m"]) < 0)
        ):
            raise ValueError("Normal loads, stiffness and damping must have physical signs")
        self.position = wp.array(arrays["position_xy"], dtype=wp.vec2, device=self.device)
        self.velocity = wp.array(arrays["velocity_xy"], dtype=wp.vec2, device=self.device)
        self.nominal_velocity = wp.array(arrays["nominal_velocity_xy"], dtype=wp.vec2, device=self.device)
        self.normal = wp.array(normal, dtype=float, device=self.device)
        self.kt = wp.array(arrays["baseline_kt_n_m"], dtype=float, device=self.device)
        self.kv = wp.array(arrays["baseline_kv_ns_m"], dtype=float, device=self.device)
        self.settings = wp.zeros((self.batch_size, 7), dtype=float, device=self.device)
        n = self.batch_size * self.columns
        self.anchor = wp.zeros(n, dtype=wp.vec2, device=self.device)
        self.deflection = wp.zeros(n, dtype=wp.vec2, device=self.device)
        self.grip = wp.zeros(n, dtype=int, device=self.device)
        self.dwell = wp.zeros(n, dtype=float, device=self.device)
        self.energy = wp.zeros(n, dtype=float, device=self.device)
        self.column_force = wp.zeros(n, dtype=wp.vec2, device=self.device)
        self.diagnostic = wp.zeros(n, dtype=wp.vec4, device=self.device)
        self.partial_force = wp.zeros(self.batch_size * self.groups, dtype=wp.vec2, device=self.device)
        self.partial_diagnostic = wp.zeros(self.batch_size * self.groups, dtype=wp.vec4, device=self.device)
        self.curves = wp.zeros((self.batch_size, self.steps), dtype=wp.vec2, device=self.device)
        self.totals = wp.zeros(self.batch_size, dtype=wp.vec4, device=self.device)
        self.clock = wp.zeros(1, dtype=int, device=self.device)
        self.condition = wp.ones(1, dtype=int, device=self.device)
        self.scores = wp.zeros((self.batch_size, len(SCORE_NAMES)), dtype=float, device=self.device)
        measured_t = np.asarray(arrays["measured_time_s"], float)
        measured_f = np.asarray(arrays["measured_force_n"], float)
        if (
            measured_t.ndim != 1
            or len(measured_t) < 2
            or measured_f.shape != (len(measured_t), 2)
            or not np.isfinite(measured_f).all()
            or not np.isfinite(measured_t).all()
            or np.any(np.diff(measured_t) <= 0)
        ):
            raise ValueError("Invalid measured force target")
        included = (measured_t >= times[0] - 1e-12) & (measured_t <= times[-1] + 1e-12)
        target_t = measured_t[included]
        target_f = measured_f[included, 0]
        active = measured_f[included, 1] >= 50.0
        if len(target_t) < 2 or not np.any(active):
            raise ValueError("No measured active force support in frozen history")
        intervals = []
        for start in np.flatnonzero(active & ~np.r_[False, active[:-1]]):
            end = start
            while end + 1 < len(active) and active[end + 1]:
                end += 1
            intervals.append((int(start), int(end)))
        braking, propulsive, _ = compute_braking_propulsive_impulses(target_t, target_f, intervals)
        peaks = compute_force_peaks(target_t, target_f, intervals)
        self.targets = wp.vec4(
            braking, propulsive, peaks["braking_peak_magnitude_n"] or 0.0, peaks["propulsive_peak_magnitude_n"] or 0.0
        )
        self.force_scale = max(float(np.max(np.abs(target_f))), 1.0)
        hi = np.clip(np.searchsorted(times, target_t, side="right"), 1, self.steps - 1)
        lo = hi - 1
        fraction = (target_t - times[lo]) / (times[hi] - times[lo])
        self.sample_count = len(target_t)
        self.target = wp.array(target_f, dtype=float, device=self.device)
        self.target_time = wp.array(target_t, dtype=float, device=self.device)
        self.active = wp.array(active.astype(np.int32), dtype=int, device=self.device)
        self.lower = wp.array(lo, dtype=int, device=self.device)
        self.upper = wp.array(hi, dtype=int, device=self.device)
        self.fraction = wp.array(fraction, dtype=float, device=self.device)
        self.graph = None
        self.graph_kind = "eager"
        if self.device.is_cuda and use_graph:
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._reset()
                wp.capture_while(self.condition, self._step)
                self._score()
            self.graph = capture.graph
            self.graph_kind = "device_while"

    def _reset(self):
        for array in (self.anchor, self.deflection, self.grip, self.dwell, self.energy, self.totals, self.clock):
            array.zero_()
        self.condition.fill_(1)

    def _step(self):
        wp.launch(
            _advance,
            dim=self.batch_size * self.columns,
            inputs=[
                self.columns,
                self.dt,
                self.clock,
                self.settings,
                self.position,
                self.velocity,
                self.nominal_velocity,
                self.normal,
                self.kt,
                self.kv,
                self.anchor,
                self.deflection,
                self.grip,
                self.dwell,
                self.energy,
                self.column_force,
                self.diagnostic,
            ],
            device=self.device,
        )
        wp.launch(
            _reduce_columns,
            dim=self.batch_size * self.groups,
            inputs=[
                self.columns,
                self.groups,
                self.column_force,
                self.diagnostic,
                self.partial_force,
                self.partial_diagnostic,
            ],
            device=self.device,
        )
        wp.launch(
            _finish_frame,
            dim=self.batch_size,
            inputs=[self.groups, self.clock, self.partial_force, self.partial_diagnostic, self.curves, self.totals],
            device=self.device,
        )
        wp.launch(_tick, dim=1, inputs=[self.clock, self.condition, self.steps], device=self.device)

    def _score(self):
        wp.launch(
            _score,
            dim=self.batch_size,
            inputs=[
                self.sample_count,
                self.target,
                self.target_time,
                self.active,
                self.lower,
                self.upper,
                self.fraction,
                self.targets,
                self.force_scale,
                self.curves,
                self.totals,
                self.scores,
            ],
            device=self.device,
        )

    def evaluate(self, parameters: np.ndarray, *, curves: bool = False):
        """Run independent cold-history candidates and return scores, optionally curves.

        Parameters are rows of method, mu, kt_scale, kv_scale, viscous_ratio,
        release_dwell_s and yield_width. Baseline stiffness/damping arrays are
        scaled, retaining tributary-area behavior. No parameter modifies normals.
        """
        values = np.asarray(parameters, np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != 7
            or not 1 <= len(values) <= self.batch_size
            or not np.isfinite(values).all()
        ):
            raise ValueError("Parameters must be finite candidate rows with seven fields")
        if np.any(values[:, 0] != values[:, 0].astype(int)) or np.any(values[:, 0] < 0) or np.any(values[:, 0] > 3):
            raise ValueError("Unknown sweep method")
        if np.any(values[:, 1:6] < 0) or np.any(values[:, 2] <= 0) or np.any(values[:, 6] != 0.0):
            raise ValueError(
                "Invalid physical parameter signs; yield_width must be zero to avoid numerical static creep"
            )
        padded = np.repeat(values[-1:], self.batch_size, axis=0)
        padded[: len(values)] = values
        self.settings.assign(padded)
        start = perf_counter()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._reset()
            for _ in range(self.steps):
                self._step()
            self._score()
        scores = self.scores.numpy()[: len(values)].copy()
        elapsed = perf_counter() - start
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite friction candidate score")
        self.last_seconds = elapsed
        return (scores, self.curves.numpy()[: len(values)].copy()) if curves else scores


# Exploratory bounds, not measured material limits. Stiffness/damping scale the
# cached area-normalized baseline; no coefficient changes the normal law.
_SEARCH_BOUNDS = np.array(
    [
        [np.log(0.05), np.log(1.2)],
        [np.log(0.001), np.log(10.0)],
        [0.0, np.log1p(10.0)],
        [0.0, 0.5],
        [0.0, 0.005],
        [0.0, 0.15],
    ]
)


def _decode(unit: np.ndarray, method: int) -> np.ndarray:
    values = _SEARCH_BOUNDS[:, 0] + unit * (_SEARCH_BOUNDS[:, 1] - _SEARCH_BOUNDS[:, 0])
    out = np.empty((len(unit), 7), np.float32)
    out[:, 0] = method
    out[:, 1] = np.exp(values[:, 0])
    out[:, 2] = np.exp(values[:, 1])
    out[:, 3] = np.expm1(values[:, 2])
    out[:, 4:] = values[:, 3:]
    # A repeatedly applied smooth projection relaxes static force at a rate
    # set by timestep. Do not fit that numerical artifact as a material property.
    out[:, 6] = 0.0
    return out


def _encode(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values[:, 1:], float).copy()
    transformed[:, :2] = np.log(transformed[:, :2])
    transformed[:, 2] = np.log1p(transformed[:, 2])
    return (transformed - _SEARCH_BOUNDS[:, 0]) / (_SEARCH_BOUNDS[:, 1] - _SEARCH_BOUNDS[:, 0])


def parameter_samples(count: int, method: int, rng: np.random.Generator) -> np.ndarray:
    """Stratify candidate coverage in bounded logarithmic/linear coordinates."""
    unit = (np.arange(count)[:, None] + rng.random((count, 6))) / count
    for column in range(6):
        rng.shuffle(unit[:, column])
    return _decode(unit, method)


def run_study(
    cache_path: Path,
    output: Path,
    *,
    candidates_per_method: int = 1024,
    batch_size: int = 128,
    generations: int = 6,
    seed: int = 17,
    methods=("legacy", "deflection", "anchor_nominal"),
    device="cuda:0",
) -> dict:
    """Sweep and locally refine effective friction candidates against one experiment.

    The selected result is not installed as a default. A fixed recorded trajectory
    is diagnostic: consistent friction may require different free leg motion.
    The search does not alter normal histories, controller parameters or inputs.
    """
    from .friction_history import load_history  # noqa: PLC0415
    from .friction_metrics import score_friction_trace  # noqa: PLC0415

    cache_path, output = Path(cache_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Study output already exists: {output}")
    if candidates_per_method < 1 or batch_size < 1 or generations < 0:
        raise ValueError("Study counts must be positive and generations nonnegative")
    if not methods or len(set(methods)) != len(methods) or any(name not in METHODS for name in methods):
        raise ValueError("Select a unique, nonempty set of supported methods")
    history = load_history(cache_path)
    arrays = vars(history)
    if not history.provenance.get("complete", False):
        raise ValueError("Parameter identification requires a complete frozen history")
    output.mkdir(parents=True)
    hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("friction_sweep.py", "friction_deflection.py", "friction_law.py", "contact.py")
    }
    cache_digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(seed)
    workspace = FrictionSweep(arrays, batch_size, device=device)
    baseline = np.array(
        [[0, history.settings["mu"], 1.0, 1.0, history.settings["viscous_ratio"], history.settings["dwell_s"], 0.0]],
        np.float32,
    )
    baseline_score, baseline_curve = workspace.evaluate(baseline, curves=True)
    cached_curve = np.asarray(history.baseline_force_xy, dtype=np.float64).sum(axis=1)
    replay_error = float(np.max(np.abs(cached_curve - baseline_curve[0])))
    if replay_error > 1e-3:
        raise RuntimeError(f"Cached baseline friction replay mismatch {replay_error:.6g} N")
    all_parameters = []
    all_scores = []
    all_stages = []
    timings = []

    def evaluate(parameters, stage):
        chunks = []
        for start in range(0, len(parameters), batch_size):
            part = parameters[start : start + batch_size]
            score = workspace.evaluate(part)
            chunks.append(score)
            timings.append({"candidates": len(part), "wall_s": workspace.last_seconds, "stage": stage})
        scores = np.concatenate(chunks)
        all_parameters.append(parameters.copy())
        all_scores.append(scores)
        all_stages.extend([stage] * len(parameters))
        return scores

    summaries = {}
    winners = {}
    for name in methods:
        method = METHODS[name]
        initial = parameter_samples(candidates_per_method, method, rng)
        explicit = baseline.copy()
        explicit[:, 0] = method
        initial = np.concatenate((explicit, initial))
        scores = evaluate(initial, f"{name}:coverage")
        pool = initial
        pool_scores = scores
        convergence = []
        best = float(scores[:, 0].min())
        for generation in range(generations):
            elite_indices = np.argsort(pool_scores[:, 0])[: min(12, len(pool_scores))]
            elite = _encode(pool[elite_indices])
            picks = rng.integers(0, len(elite), batch_size)
            radius = 0.18 * (0.55**generation)
            proposal = np.clip(elite[picks] + rng.normal(0.0, radius, (batch_size, 6)), 0.0, 1.0)
            new_parameters = _decode(proposal, method)
            new_score = evaluate(new_parameters, f"{name}:refine{generation}")
            pool = np.concatenate((pool, new_parameters))
            pool_scores = np.concatenate((pool_scores, new_score))
            current = float(pool_scores[:, 0].min())
            convergence.append(
                {"generation": generation, "radius": radius, "best_loss": current, "improvement": best - current}
            )
            best = current
        order = np.argsort(pool_scores[:, 0])
        chosen = int(order[0])
        near = pool_scores[:, 0] <= best * 1.05 + 1e-6
        physical = (
            (pool_scores[:, 6] <= 1e-3 * (np.abs(pool_scores[:, 9]) + 1.0))
            & (pool_scores[:, 7] <= 0.02)
            & (pool_scores[:, 8] <= 1e-3)
        )
        physical_indices = np.flatnonzero(physical)
        screened = None
        if len(physical_indices):
            screened = int(physical_indices[np.argmin(pool_scores[physical_indices, 0])])
        summaries[name] = {
            "best_loss": best,
            "parameters": dict(zip(PARAMETER_NAMES, pool[chosen].astype(float).tolist(), strict=True)),
            "score": dict(zip(SCORE_NAMES, pool_scores[chosen].astype(float).tolist(), strict=True)),
            "candidate_count": len(pool),
            "convergence": convergence,
            "near_best_count": int(near.sum()),
            "near_best_parameter_min": pool[near].min(axis=0).astype(float).tolist(),
            "near_best_parameter_max": pool[near].max(axis=0).astype(float).tolist(),
            "screened_candidate_count": int(physical.sum()),
            "best_screened_parameters": None
            if screened is None
            else dict(zip(PARAMETER_NAMES, pool[screened].astype(float).tolist(), strict=True)),
            "best_screened_score": None
            if screened is None
            else dict(zip(SCORE_NAMES, pool_scores[screened].astype(float).tolist(), strict=True)),
        }
        winners[name] = pool[chosen]
        print(name, json.dumps(summaries[name]), flush=True)

    parameters = np.concatenate(all_parameters)
    scores = np.concatenate(all_scores)
    np.savez_compressed(
        output / "candidates.npz",
        parameters=parameters,
        scores=scores,
        stage=np.asarray(all_stages),
        parameter_names=np.asarray(PARAMETER_NAMES),
        score_names=np.asarray(SCORE_NAMES),
    )
    reference = {"grf_time_s": history.measured_time_s, "grf_target_n": history.measured_force_n}
    summary = {"complete": True, "integrated_steps": len(history.time_s), "actual_dt_s": workspace.dt}
    curve_reports = {}
    for name, values in {"baseline": baseline[0], **winners}.items():
        _, curve = workspace.evaluate(values[None, :], curves=True)
        normal_total = history.normal_n.astype(np.float64).sum(axis=1)
        grf = np.column_stack((curve[0, :, 0], normal_total))
        np.savez_compressed(output / f"trace_{name}.npz", time_s=history.time_s, grf_n=grf)
        curve_reports[name] = score_friction_trace(
            reference, {"time_s": history.time_s, "grf_n": grf}, 1, summary=summary
        )
        (output / f"candidate_{name}.json").write_text(
            json.dumps(
                {
                    "parameters": dict(zip(PARAMETER_NAMES, values.astype(float).tolist(), strict=True)),
                    "qualification": "Effective fit to one frozen leg experiment; not independent friction calibration or an installed default.",
                    "cache_sha256": cache_digest,
                    "source_hashes": hashes,
                },
                indent=2,
            )
        )
    normal_error = float(np.max(np.abs(workspace.normal.numpy() - history.normal_n)))
    if normal_error != 0.0:
        raise RuntimeError("Sweep modified its read-only normal history")
    for name, digest in hashes.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Friction source changed during the study")
    total_seconds = sum(x["wall_s"] for x in timings)
    report = {
        "schema": "digital_shoe_friction_study_1",
        "source_hashes": hashes,
        "cache_sha256": cache_digest,
        "cache_provenance": history.provenance,
        "seed": seed,
        "bounds_transformed": _SEARCH_BOUNDS.tolist(),
        "parameter_names": PARAMETER_NAMES,
        "score_names": SCORE_NAMES,
        "objective": "Native-sample force MSE / peak_reference^2 + 0.25*(relative braking/propulsive impulse errors squared) + 0.1*(relative braking/propulsive peak errors squared).",
        "screening": "Positive tangential energy residual <=0.001*(abs(net tangential work)+1) J, max elastic deflection<=20mm, cone excess<=0.001N. Numerical diagnostics, not measured validation limits.",
        "baseline_replay_max_error_n": replay_error,
        "normal_history_max_change_n": normal_error,
        "baseline_score": dict(zip(SCORE_NAMES, baseline_score[0].astype(float).tolist(), strict=True)),
        "methods": summaries,
        "metrics": curve_reports,
        "timings": timings,
        "completed_candidates": len(parameters),
        "evaluation_wall_s": total_seconds,
        "completed_candidates_per_second": len(parameters) / max(total_seconds, 1e-12),
        "device": str(workspace.device),
        "graph_kind": workspace.graph_kind,
        "qualification": "Exploratory effective fit to the same frozen-motion stance. No independent held-out friction experiment. No normal/contact/controller changes. A small loss does not certify physical validity. Anchor_nominal is a diagnostic, not an automatically work-conjugate law. No default is overwritten.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    """Run a reproducible friction-only parameter study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-per-method", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHODS), default=["legacy", "deflection", "anchor_nominal"]
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_study(
        args.cache,
        args.output,
        candidates_per_method=args.candidates_per_method,
        batch_size=args.batch_size,
        generations=args.generations,
        seed=args.seed,
        methods=args.methods,
        device=args.device,
    )
    print(f"Completed {report['completed_candidates']} candidates; results in {args.output}")


if __name__ == "__main__":
    main()
