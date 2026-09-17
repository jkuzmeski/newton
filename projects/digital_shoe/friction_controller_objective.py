# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Raw-force controller objective with fixed-contact mechanics and foot-motion checks."""

from itertools import pairwise

import numpy as np
import warp as wp

from projects.impedance_instron.cartesian.gpu.mechanics import Params, Vec5, ankle, foot_angle
from projects.impedance_instron.cartesian.gpu.objective import MeasuredObjective
from projects.impedance_instron.cartesian.mechanics import Body

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.func
def _phase_segment(a: wp.float64, b: wp.float64, ta: wp.float64, tb: wp.float64) -> wp.vec4d:
    h = tb - ta
    ib = wp.float64(0.0)
    ip = wp.float64(0.0)
    mb = wp.float64(0.0)
    mp = wp.float64(0.0)
    if a <= wp.float64(0.0) and b <= wp.float64(0.0):
        ib = -h * (a + b) * wp.float64(0.5)
        mb = -h * ((wp.float64(2.0) * ta + tb) * a + (ta + wp.float64(2.0) * tb) * b) / wp.float64(6.0)
    elif a >= wp.float64(0.0) and b >= wp.float64(0.0):
        ip = h * (a + b) * wp.float64(0.5)
        mp = h * ((wp.float64(2.0) * ta + tb) * a + (ta + wp.float64(2.0) * tb) * b) / wp.float64(6.0)
    else:
        tc = ta + h * (-a / (b - a))
        if a < wp.float64(0.0):
            ib = -wp.float64(0.5) * (tc - ta) * a
            ip = wp.float64(0.5) * (tb - tc) * b
            mb = ib * (wp.float64(2.0) * ta + tc) / wp.float64(3.0)
            mp = ip * (tc + wp.float64(2.0) * tb) / wp.float64(3.0)
        else:
            ip = wp.float64(0.5) * (tc - ta) * a
            ib = -wp.float64(0.5) * (tb - tc) * b
            mp = ip * (wp.float64(2.0) * ta + tc) / wp.float64(3.0)
            mb = ib * (tc + wp.float64(2.0) * tb) / wp.float64(3.0)
    return wp.vec4d(ib, ip, mb, mp)


@wp.kernel
def _augment(
    p: Params,
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    forces: wp.array2d[wp.vec2d],
    actuator: wp.array2d[wp.vec4d],
    totals: wp.array[wp.vec4],
    time: wp.array[wp.float64],
    active: wp.array[int],
    ml: wp.array[int],
    mu: wp.array[int],
    mf: wp.array[wp.float64],
    foot_target: wp.array[wp.vec2d],
    foot_weights: wp.array[wp.vec2d],
    targets: wp.array[wp.float64],
    effort_bound: wp.vec4d,
    rate_bound: wp.vec4d,
    early_end: wp.float64,
    base_residual: wp.array2d[wp.float64],
    residual: wp.array2d[wp.float64],
    loss: wp.array[wp.float64],
    costs: wp.array2d[wp.float64],
    rmse: wp.array2d[wp.float64],
    maximum_error: wp.array2d[wp.float64],
    original_rmse: wp.array2d[wp.float64],
    tracking_limits: wp.array[wp.float64],
    tracking_penalty: wp.float64,
):
    w = wp.tid()
    base = base_residual.shape[0]
    count = foot_target.shape[0]
    complete = wp.isfinite(loss[w])
    for row in range(base):
        residual[row, w] = base_residual[row, w]
    extra = wp.float64(0.0)
    foot_loss = wp.float64(0.0)
    phase_loss = wp.float64(0.0)
    if complete:
        for j in range(count):
            q = states[ml[j], w] * (wp.float64(1.0) - mf[j]) + states[mu[j], w] * mf[j]
            v = velocities[ml[j], w] * (wp.float64(1.0) - mf[j]) + velocities[mu[j], w] * mf[j]
            point, jx, _jz = ankle(q, p)
            vx = wp.dot(jx, v) + (v[2] + v[3] + v[4]) * point[1]
            r0 = (foot_angle(q) - foot_target[j][0]) / wp.float64(0.05) * foot_weights[j][0]
            r1 = (vx - foot_target[j][1]) / wp.float64(0.5) * foot_weights[j][1]
            residual[base + 2 * j, w] = r0
            residual[base + 2 * j + 1, w] = r1
            foot_loss += r0 * r0 + r1 * r1
        phase = wp.vec4d(wp.float64(0.0))
        pb = wp.float64(0.0)
        pp = wp.float64(0.0)
        early_peak = wp.float64(0.0)
        early_ib = wp.float64(0.0)
        max_effort = wp.vec4d(wp.float64(0.0))
        max_rate = wp.vec4d(wp.float64(0.0))
        for j in range(forces.shape[0]):
            f = forces[j, w][0]
            if active[j] != 0:
                pb = wp.max(pb, -f)
                pp = wp.max(pp, f)
                if time[j] <= early_end:
                    early_peak = wp.max(early_peak, -f)
                if j > 0 and active[j - 1] != 0:
                    segment = _phase_segment(forces[j - 1, w][0], f, time[j - 1], time[j])
                    phase += segment
                    if time[j] <= early_end:
                        early_ib += segment[0]
            for k in range(4):
                max_effort[k] = wp.max(max_effort[k], wp.abs(actuator[j, w][k]))
                if j > 0:
                    max_rate[k] = wp.max(
                        max_rate[k], wp.abs(actuator[j, w][k] - actuator[j - 1, w][k]) / (time[j] - time[j - 1])
                    )
        offset = base + 2 * count
        for j in range(8):
            r = wp.float64(0.0)
            if j == 0:
                r = wp.float64(0.7) * (phase[0] - targets[0]) / wp.max(targets[0], wp.float64(1.0))
            elif j == 1:
                r = wp.float64(0.7) * (phase[1] - targets[1]) / wp.max(targets[1], wp.float64(1.0))
            elif j == 2:
                r = wp.float64(0.4) * (pb - targets[2]) / wp.max(targets[2], wp.float64(1.0))
            elif j == 3:
                r = wp.float64(0.4) * (pp - targets[3]) / wp.max(targets[3], wp.float64(1.0))
            elif j == 4:
                r = wp.float64(0.4) * (phase[2] / wp.max(phase[0], wp.float64(1.0e-8)) - targets[4]) / wp.float64(0.03)
            elif j == 5:
                r = wp.float64(0.4) * (phase[3] / wp.max(phase[1], wp.float64(1.0e-8)) - targets[5]) / wp.float64(0.03)
            elif j == 6:
                r = wp.float64(0.7) * wp.max(early_peak - targets[6], wp.float64(0.0)) / wp.float64(100.0)
            else:
                r = wp.float64(0.5) * (early_ib - targets[7]) / wp.max(targets[7], wp.float64(1.0))
            residual[offset + j, w] = r
            phase_loss += r * r
        for k in range(6):
            r = wp.sqrt(tracking_penalty) * wp.max(
                original_rmse[w, k] / tracking_limits[k] - wp.float64(1.0), wp.float64(0.0)
            )
            residual[offset + 8 + k, w] = r
            phase_loss += r * r
        for k in range(4):
            if max_effort[k] > effort_bound[k] or max_rate[k] > rate_bound[k]:
                complete = False
        diag = totals[w]
        if diag[0] > 0.001 * (wp.abs(diag[3]) + 1.0) or diag[1] > 0.02 or diag[2] > 0.001:
            complete = False
        for k in range(4):
            if not wp.isfinite(diag[k]):
                complete = False
        extra = foot_loss + phase_loss
        if not wp.isfinite(extra):
            complete = False
    if complete:
        loss[w] += extra
        costs[w, 1] += foot_loss
        costs[w, 2] += phase_loss
    else:
        loss[w] = wp.float64(wp.inf)
        for row in range(residual.shape[0]):
            residual[row, w] = wp.float64(wp.inf)
        for k in range(6):
            rmse[w, k] = wp.float64(wp.inf)
            maximum_error[w, k] = wp.float64(wp.inf)
        for k in range(3):
            costs[w, k] = wp.float64(wp.inf)


def phase_targets(time, force, active, early_end=0.08):
    """Compute exact piecewise-linear phase impulses and centroid targets on one clock."""
    values = np.zeros(4)
    early = 0.0
    for i in range(1, len(time)):
        if not (active[i - 1] and active[i]):
            continue
        a, b = force[i - 1], force[i]
        ta, tb = time[i - 1], time[i]
        cuts = [(ta, a), (tb, b)]
        if a * b < 0:
            cuts.insert(1, (ta + (tb - ta) * (-a / (b - a)), 0.0))
        for (x, fx), (y, fy) in pairwise(cuts):
            integral = 0.5 * (y - x) * (fx + fy)
            moment = (y - x) * ((2 * x + y) * fx + (x + 2 * y) * fy) / 6
            if integral < 0:
                values[0] -= integral
                values[2] -= moment
                if tb <= early_end:
                    early -= integral
            else:
                values[1] += integral
                values[3] += moment
    return np.array(
        [
            values[0],
            values[1],
            max(0.0, -float(force[active].min())),
            max(0.0, float(force[active].max())),
            values[2] / max(values[0], 1e-8),
            values[3] / max(values[1], 1e-8),
            max(0.0, -float(force[active & (time <= early_end)].min())),
            early,
        ]
    )


class ControllerObjective:
    """Compose raw measured residuals and fixed supplemental diagnostic guards.

    The simulator reference, physics and gain arrays are untouched. Additional
    effort guards are baseline-relative engineering bounds, not physiological data.
    """

    def __init__(self, engine, adapter, baseline_trace, *, early_end=0.08, tracking_penalty=100.0, tracking_margin=1.0):
        self.engine = engine
        self.adapter = adapter
        self.device = engine.device
        ref = engine.reference
        time = engine.time_s[:-1]
        target = np.column_stack(
            [np.interp(time, ref["grf_time_s"], ref["unfiltered_grf_target_n"][:, a]) for a in range(2)]
        )
        raw_reference = {**ref, "grf_time_s": time, "grf_target_n": target}
        self.base = MeasuredObjective(raw_reference, engine.settings, engine.time_s, engine.world_count, self.device)
        self.original = MeasuredObjective(ref, engine.settings, engine.time_s, engine.world_count, self.device)
        if not np.isfinite(tracking_margin) or not 0 < tracking_margin <= 1:
            raise ValueError("tracking_margin must be in (0,1]")
        limits = np.array(
            [engine.settings.hip_tolerance_m] * 2
            + [engine.settings.joint_tolerance_rad] * 2
            + [engine.settings.force_tolerance_n] * 2
        )
        self.tracking_limits = wp.array(limits * tracking_margin, dtype=wp.float64, device=self.device)
        self.tracking_penalty = float(tracking_penalty)
        if not np.isfinite(self.tracking_penalty) or self.tracking_penalty < 0:
            raise ValueError("tracking_penalty must be finite and nonnegative")
        self.loss = self.base.loss
        self.rmse = self.base.rmse
        self.maximum_error = self.base.maximum_error
        self.costs = self.base.costs
        motion_time = ref["time_s"]
        hi = np.clip(np.searchsorted(engine.time_s, motion_time, side="right"), 1, len(engine.time_s) - 1)
        lo = hi - 1
        fraction = (motion_time - engine.time_s[lo]) / (engine.time_s[hi] - engine.time_s[lo])
        body = Body(
            ref["lengths_m"],
            ref["endpoint_local_m"],
            engine.profile["masses_kg"],
            engine.profile["com_local_m"],
            engine.profile["inertias_kg_m2"],
        )
        foot = []
        for q, v in zip(ref["state"], ref["velocity"], strict=True):
            point, jac, _ = body.point(q, 2, [0.0, 0.0])
            foot.append([body.angle(q, 2), float((jac @ v)[0] + sum(v[2:]) * point[1])])
        foot = np.asarray(foot)
        weights = np.column_stack(
            (
                np.full(len(foot), np.sqrt(0.25 / len(foot))),
                (motion_time <= early_end) * np.sqrt(0.1 / max(1, np.count_nonzero(motion_time <= early_end))),
            )
        )
        self.residual_dim = self.base.residual_dim + 2 * len(foot) + 14
        self.residual = wp.empty((self.residual_dim, engine.world_count), dtype=wp.float64, device=self.device)
        self.time = wp.array(time, dtype=wp.float64, device=self.device)
        active = target[:, 1] >= 50.0
        self.active = wp.array(active.astype(np.int32), dtype=int, device=self.device)
        self.ml = wp.array(lo, dtype=int, device=self.device)
        self.mu = wp.array(hi, dtype=int, device=self.device)
        self.mf = wp.array(fraction, dtype=wp.float64, device=self.device)
        self.foot_target = wp.array(foot, dtype=wp.vec2d, device=self.device)
        self.foot_weights = wp.array(weights, dtype=wp.vec2d, device=self.device)
        self.targets = wp.array(
            phase_targets(time, target[:, 0], active, early_end), dtype=wp.float64, device=self.device
        )
        effort = np.column_stack((baseline_trace["hip_force_n"], baseline_trace["joint_torque_nm"]))
        effort_bound = 1.5 * np.max(abs(effort), axis=0)
        rate_bound = 2.0 * np.max(abs(np.diff(effort, axis=0) / np.diff(baseline_trace["time_s"])[:, None]), axis=0)
        self.effort_bound = wp.vec4d(*effort_bound)
        self.rate_bound = wp.vec4d(*rate_bound)
        self.early_end = early_end
        self.description = {
            **self.base.description,
            "policy": "raw simulation at every timestep, pre-20Hz cleaned reference only interpolated",
            "supplemental": "foot pitch(.05rad scale,.25weight), onset plane velocity(.5m/s scale,.1weight), phase impulses/peaks/centroids and early braking",
            "effort_peak_bound": effort_bound.tolist(),
            "effort_rate_bound": rate_bound.tolist(),
            "effort_guard_note": "1.5x peak and2x rate of old controller with this fixed contact model; additional engineering guards, not physiological limits",
            "gains": "unchanged",
            "early_end_s": early_end,
            "tracking_penalty_weight": self.tracking_penalty,
            "tracking_training_margin": tracking_margin,
            "tracking_gate_note": "Quadratic hinge on training_margin times the original six RMSE thresholds. Original validation limits remain unchanged and are evaluated separately.",
        }

    def launch(self, states, forces, integrated_steps, failure_code):
        """Evaluate original raw residual blocks then append the fixed diagnostic terms."""
        self.base.launch(states, forces, integrated_steps, failure_code)
        self.original.launch(states, forces, integrated_steps, failure_code)
        wp.launch(
            _augment,
            dim=self.engine.world_count,
            inputs=[
                self.engine.params,
                states,
                self.engine.velocities,
                forces,
                self.engine.actuator,
                self.adapter.totals,
                self.time,
                self.active,
                self.ml,
                self.mu,
                self.mf,
                self.foot_target,
                self.foot_weights,
                self.targets,
                self.effort_bound,
                self.rate_bound,
                self.early_end,
                self.base.residual,
                self.residual,
                self.loss,
                self.costs,
                self.rmse,
                self.maximum_error,
                self.original.rmse,
                self.tracking_limits,
                self.tracking_penalty,
            ],
            device=self.device,
        )

    def read(self):
        """Read standard objective arrays; keep snapshot compatibility without live extra buffers."""
        return {
            "loss": self.loss.numpy(),
            "rmse": self.rmse.numpy(),
            "maximum_error": self.maximum_error.numpy(),
            "costs": self.costs.numpy(),
            "residual": self.residual.numpy(),
        }
