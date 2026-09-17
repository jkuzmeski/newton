# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve tangential shoe traction with prescribed, read-only normal reactions.

The coupled paths solve v = v_free + dt M_inv sum(J.T f(J v)). Geometry,
normal reactions and world-space mobility are frozen during this friction step.
This is a friction-only split, not a new normal-contact or rigid-body integrator.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import warp as wp

from .friction_deflection import bristle_deflection_step
from .friction_law import bristle_force_tangent, regularized_force_tangent


@wp.struct
class FrictionParams:
    """Per-world tangential settings; normal force is a separate input."""

    mu: float
    viscous_ratio: float
    release_dwell: float
    smoothing_speed: float
    yield_width: float = 0.0


@wp.func
def _point_rows(arm: wp.vec3) -> tuple[wp.spatial_vector, wp.spatial_vector]:
    return wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, arm[2], -arm[1])), wp.spatial_vector(
        wp.vec3(0.0, 1.0, 0.0), wp.vec3(-arm[2], 0.0, arm[0])
    )


@wp.func
def _solve6(matrix: wp.spatial_matrix, rhs: wp.spatial_vector) -> wp.spatial_vector:
    # The mobility-form Jacobian need not be symmetric. Pivot rather than
    # assuming that a VBD positive-definite metric is the exact Jacobian.
    a = matrix
    b = rhs
    for k in range(6):
        pivot = k
        for row in range(k + 1, 6):
            if wp.abs(a[row, k]) > wp.abs(a[pivot, k]):
                pivot = row
        for col in range(6):
            value = a[k, col]
            a[k, col] = a[pivot, col]
            a[pivot, col] = value
        value = b[k]
        b[k] = b[pivot]
        b[pivot] = value
        diagonal = a[k, k]
        if wp.abs(diagonal) < 1.0e-10:
            diagonal = 1.0e-10
        for row in range(k + 1, 6):
            factor = a[row, k] / diagonal
            for col in range(k + 1, 6):
                a[row, col] = a[row, col] - factor * a[k, col]
            b[row] = b[row] - factor * b[k]
    x = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    for reverse in range(6):
        row = 5 - reverse
        value = b[row]
        for col in range(row + 1, 6):
            value = value - a[row, col] * x[col]
        diagonal = a[row, row]
        if wp.abs(diagonal) < 1.0e-10:
            diagonal = 1.0e-10
        x[row] = value / diagonal
    return x


@wp.func_grad(_solve6)
def _adj_solve6(matrix: wp.spatial_matrix, rhs: wp.spatial_vector, adj_x: wp.spatial_vector):
    x = _solve6(matrix, rhs)
    adj_rhs = _solve6(wp.transpose(matrix), adj_x)
    wp.adjoint[rhs] += adj_rhs
    wp.adjoint[matrix] -= wp.outer(adj_rhs, x)


@wp.kernel
def _evaluate(
    count: int,
    mode: int,
    dt: float,
    points: wp.array[wp.vec3],
    normal: wp.array[float],
    com: wp.array[wp.vec3],
    velocity: wp.array[wp.spatial_vector],
    kt: wp.array[float],
    kv: wp.array[float],
    params: wp.array[FrictionParams],
    anchor: wp.array[wp.vec2],
    stuck: wp.array[int],
    dwell: wp.array[float],
    in_deflection: wp.array[wp.vec2],
    has_deflection: int,
    force: wp.array[wp.vec2],
    tangent: wp.array[wp.mat22],
    next_anchor: wp.array[wp.vec2],
    next_stuck: wp.array[int],
    next_dwell: wp.array[float],
    next_deflection: wp.array[wp.vec2],
):
    i = wp.tid()
    world = i // count
    column = i % count
    p = params[world]
    row_x, row_y = _point_rows(points[i] - com[world])
    v = wp.vec2(wp.dot(row_x, velocity[world]), wp.dot(row_y, velocity[world]))
    pos = wp.vec2(points[i][0], points[i][1])
    if mode == 2:
        f, d = regularized_force_tangent(v, dt, normal[i], p.mu, p.smoothing_speed)
        a = pos
        s = 0
        t = 0.0
        z = wp.vec2(0.0, 0.0)
    elif mode == 3 or mode == 4:
        cur_z = wp.vec2(0.0, 0.0)
        if has_deflection != 0:
            cur_z = in_deflection[i]
        f, d, z, s, t = bristle_deflection_step(
            v,
            dt,
            normal[i],
            kt[column],
            kv[column],
            p.mu,
            p.viscous_ratio,
            p.release_dwell,
            p.yield_width,
            cur_z,
            stuck[i],
            dwell[i],
        )
        trialpoint = pos + v * dt
        a = trialpoint - z
    else:
        f, d, a, s, t = bristle_force_tangent(
            pos,
            v,
            dt,
            normal[i],
            kt[column],
            kv[column],
            p.mu,
            p.viscous_ratio,
            p.release_dwell,
            anchor[i],
            stuck[i],
            dwell[i],
        )
        z = pos - a
    force[i] = f
    tangent[i] = d
    next_anchor[i] = a
    next_stuck[i] = s
    next_dwell[i] = t
    next_deflection[i] = z


@wp.kernel
def _partial(
    count: int,
    groups: int,
    points: wp.array[wp.vec3],
    com: wp.array[wp.vec3],
    force: wp.array[wp.vec2],
    tangent: wp.array[wp.mat22],
    wrench: wp.array[wp.spatial_vector],
    derivative: wp.array[wp.spatial_matrix],
):
    index = wp.tid()
    world = index // groups
    group = index % groups
    f = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    d = wp.spatial_matrix(0.0)
    for column in range(group, count, groups):
        i = world * count + column
        x, y = _point_rows(points[i] - com[world])
        ft = force[i]
        j = tangent[i]
        f = f + x * ft[0] + y * ft[1]
        d = d + wp.outer(x, x) * j[0, 0] + wp.outer(x, y) * j[0, 1]
        d = d + wp.outer(y, x) * j[1, 0] + wp.outer(y, y) * j[1, 1]
    wrench[index] = f
    derivative[index] = d


@wp.kernel
def _newton(
    groups: int,
    dt: float,
    velocity: wp.array[wp.spatial_vector],
    free_velocity: wp.array[wp.spatial_vector],
    mobility: wp.array[wp.spatial_matrix],
    partial_wrench: wp.array[wp.spatial_vector],
    partial_derivative: wp.array[wp.spatial_matrix],
    direction: wp.array[wp.spatial_vector],
):
    world = wp.tid()
    f = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    j = wp.spatial_matrix(0.0)
    # Static reduction preserves the primal sum during Warp's backward replay.
    for group in range(16):
        if group < groups:
            f = f + partial_wrench[world * groups + group]
            j = j + partial_derivative[world * groups + group]
    residual = velocity[world] - free_velocity[world] - dt * (mobility[world] @ f)
    system = wp.identity(n=6, dtype=float) - dt * (mobility[world] @ j)
    direction[world] = _solve6(system, -residual)


@wp.kernel
def _trial_velocity(
    scale: float,
    velocity: wp.array[wp.spatial_vector],
    direction: wp.array[wp.spatial_vector],
    out: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    out[i] = velocity[i] + scale * direction[i]


@wp.kernel
def _choose(
    groups: int,
    dt: float,
    angular_scale: float,
    first: int,
    velocity: wp.array[wp.spatial_vector],
    free_velocity: wp.array[wp.spatial_vector],
    mobility: wp.array[wp.spatial_matrix],
    partial_wrench: wp.array[wp.spatial_vector],
    best_velocity: wp.array[wp.spatial_vector],
    best_error: wp.array[float],
    out_velocity: wp.array[wp.spatial_vector],
    out_error: wp.array[float],
):
    world = wp.tid()
    f = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    for group in range(16):
        if group < groups:
            f = f + partial_wrench[world * groups + group]
    residual = velocity[world] - free_velocity[world] - dt * (mobility[world] @ f)
    linear = wp.spatial_top(residual)
    angular = wp.spatial_bottom(residual) * angular_scale
    error = wp.dot(linear, linear) + wp.dot(angular, angular)
    if first != 0 or error < best_error[world]:
        out_velocity[world] = velocity[world]
        out_error[world] = error
    else:
        out_velocity[world] = best_velocity[world]
        out_error[world] = best_error[world]


@wp.kernel
def _residual(
    groups: int,
    dt: float,
    velocity: wp.array[wp.spatial_vector],
    free_velocity: wp.array[wp.spatial_vector],
    mobility: wp.array[wp.spatial_matrix],
    partial_wrench: wp.array[wp.spatial_vector],
    linear_error: wp.array[float],
    angular_error: wp.array[float],
    wrench: wp.array[wp.spatial_vector],
):
    world = wp.tid()
    f = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    for group in range(16):
        if group < groups:
            f = f + partial_wrench[world * groups + group]
    r = velocity[world] - free_velocity[world] - dt * (mobility[world] @ f)
    linear_error[world] = wp.length(wp.spatial_top(r))
    angular_error[world] = wp.length(wp.spatial_bottom(r))
    wrench[world] = f


class FrictionSolver:
    """Solve a frozen-geometry tangential step without evaluating normal contact.

    ``bristle`` is the existing explicit-velocity predictor. ``implicit_bristle``
    couples that same force law to all contacts on each carrier. ``regularized``
    couples a VBD-inspired regularized Coulomb law without elastic anchor memory.
    ``deflection`` is an explicit-velocity step with velocity-integrated deflection.
    ``implicit_deflection`` couples the velocity-integrated deflection law to all contacts.
    Normal inputs are never written. Modes are opt-in; the foundation default is
    unchanged. Convergence diagnostics must be checked before trusting a new step
    size. Fixed Newton/backtracking counts are GPU graph and tape compatible.

    Mobility maps world-space COM wrench [N, N*m] to acceleration [m/s^2,
    rad/s^2]. Supply the actual consumer mobility, including constraints; a
    display body's placeholder mass is not valid for an articulated limb.

    Args:
        column_count: Number of columns in each world.
        world_count: Number of isolated worlds sharing column stiffness/damping.
        mode: Tangential law/stepping selection.
        iterations: Newton iterations in coupled modes.
        max_steps: Preallocated independent evaluation slots for a tape rollout.
        angular_scale_m: Length converting angular residual to velocity for line search [m].
        device: Warp device.
        requires_grad: Allocate floating work buffers for reverse-mode derivatives.
    """

    @dataclass
    class Result:
        """Tangential force/history and coupling residuals for one evaluation slot."""

        force: wp.array[wp.vec2]
        anchor: wp.array[wp.vec2]
        stuck: wp.array[int]
        dwell: wp.array[float]
        velocity: wp.array[wp.spatial_vector]
        wrench: wp.array[wp.spatial_vector]
        linear_residual: wp.array[float]
        angular_residual: wp.array[float]
        deflection: wp.array[wp.vec2]

    MODES: ClassVar[dict[str, int]] = {
        "bristle": 0,
        "implicit_bristle": 1,
        "regularized": 2,
        "deflection": 3,
        "implicit_deflection": 4,
    }

    def __init__(
        self,
        column_count: int,
        world_count: int = 1,
        *,
        mode: str = "implicit_bristle",
        iterations: int = 8,
        max_steps: int = 1,
        angular_scale_m: float = 0.1,
        device=None,
        requires_grad: bool = False,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown friction mode {mode!r}; choose {tuple(self.MODES)}")
        if column_count < 1 or world_count < 1 or iterations < 1 or max_steps < 1:
            raise ValueError("Friction counts and iterations must be positive")
        if not np.isfinite(angular_scale_m) or angular_scale_m <= 0:
            raise ValueError("angular_scale_m must be finite and positive")
        self.column_count = int(column_count)
        self.world_count = int(world_count)
        self.mode = self.MODES[mode]
        self.iterations = 0 if (self.mode == 0 or self.mode == 3) else int(iterations)
        self.device = wp.get_device(device)
        self.angular_scale = float(angular_scale_m)
        self.requires_grad = requires_grad
        self.groups = min(16, self.column_count)
        self._slots = []
        n, w = self.column_count * self.world_count, self.world_count
        self._internal_zero_deflection = wp.zeros(n, dtype=wp.vec2, device=self.device)

        def zeros(count, dtype):
            return wp.zeros(count, dtype=dtype, device=self.device, requires_grad=requires_grad and dtype is not int)

        def evaluation():
            return [
                zeros(n, wp.vec2),
                zeros(n, wp.mat22),
                zeros(n, wp.vec2),
                zeros(n, int),
                zeros(n, float),
                zeros(w * self.groups, wp.spatial_vector),
                zeros(w * self.groups, wp.spatial_matrix),
                zeros(n, wp.vec2),
            ]

        for _ in range(max_steps):
            # Distinct iteration/candidate buffers preserve Warp tape history.
            iterations_work = []
            for _iteration in range(self.iterations):
                candidates = [
                    (zeros(w, wp.spatial_vector), evaluation(), zeros(w, wp.spatial_vector), zeros(w, float))
                    for _candidate in range(5)
                ]
                iterations_work.append((evaluation(), zeros(w, wp.spatial_vector), candidates))
            final = evaluation()
            result = self.Result(
                final[0],
                final[2],
                final[3],
                final[4],
                zeros(w, wp.spatial_vector),
                zeros(w, wp.spatial_vector),
                zeros(w, float),
                zeros(w, float),
                final[7],
            )
            self._slots.append((iterations_work, final, result))

    def solve(
        self,
        points,
        normal,
        com,
        velocity,
        mobility,
        kt,
        kv,
        params,
        anchor,
        stuck,
        dwell,
        dt: float,
        *,
        deflection=None,
        step: int = 0,
    ) -> Result:
        """Evaluate tangential forces using caller-owned frozen normal/contact data.

        Points, normal reactions and incoming history have length worlds*columns.
        COM, free velocity, mobility and settings have length worlds. Stiffness
        and damping have length columns and are shared across worlds. XY is the
        tangent plane. ``velocity`` is the friction-free velocity for this split;
        no normal or actuator force is added internally. ``deflection`` is an
        optional caller-owned tangential deflection buffer [m] for deflection modes.
        Result arrays belong to this solver and are overwritten only when their
        slot is reused. Use a distinct ``step`` for every call recorded on one Warp tape.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Friction dt must be finite and positive")
        if not 0 <= step < len(self._slots):
            raise ValueError("Friction step exceeds preallocated max_steps")
        n = self.column_count * self.world_count
        for value, length in (
            (points, n),
            (normal, n),
            (com, self.world_count),
            (velocity, self.world_count),
            (mobility, self.world_count),
            (kt, self.column_count),
            (kv, self.column_count),
            (params, self.world_count),
            (anchor, n),
            (stuck, n),
            (dwell, n),
        ):
            if value.shape != (length,) or value.device != self.device:
                raise ValueError("Friction inputs must match solver shapes and device")
        has_deflection = int(deflection is not None)
        in_deflection = deflection if deflection is not None else self._internal_zero_deflection
        if in_deflection.shape != (n,) or in_deflection.device != self.device:
            raise ValueError("deflection buffer must match solver shape and device")

        work, final, result = self._slots[step]

        def evaluate(v, buffers):
            wp.launch(
                _evaluate,
                dim=n,
                inputs=[
                    self.column_count,
                    self.mode,
                    dt,
                    points,
                    normal,
                    com,
                    v,
                    kt,
                    kv,
                    params,
                    anchor,
                    stuck,
                    dwell,
                    in_deflection,
                    has_deflection,
                    buffers[0],
                    buffers[1],
                    buffers[2],
                    buffers[3],
                    buffers[4],
                    buffers[7],
                ],
                device=self.device,
            )
            wp.launch(
                _partial,
                dim=self.world_count * self.groups,
                inputs=[self.column_count, self.groups, points, com, buffers[0], buffers[1], buffers[5], buffers[6]],
                device=self.device,
            )

        current = velocity
        for base, direction, candidates in work:
            evaluate(current, base)
            wp.launch(
                _newton,
                dim=self.world_count,
                inputs=[self.groups, dt, current, velocity, mobility, base[5], base[6], direction],
                device=self.device,
            )
            best_v = current
            best_error = candidates[0][3]
            for index, (trial, values, selected, error) in enumerate(candidates):
                if index == 0:
                    trial_v = current
                    partials = base[5]
                else:
                    wp.launch(
                        _trial_velocity,
                        dim=self.world_count,
                        inputs=[0.5 ** (index - 1), current, direction, trial],
                        device=self.device,
                    )
                    evaluate(trial, values)
                    trial_v = trial
                    partials = values[5]
                wp.launch(
                    _choose,
                    dim=self.world_count,
                    inputs=[
                        self.groups,
                        dt,
                        self.angular_scale,
                        int(index == 0),
                        trial_v,
                        velocity,
                        mobility,
                        partials,
                        best_v,
                        best_error,
                        selected,
                        error,
                    ],
                    device=self.device,
                )
                best_v, best_error = selected, error
            current = best_v
        evaluate(current, final)
        wp.copy(result.velocity, current)
        wp.launch(
            _residual,
            dim=self.world_count,
            inputs=[
                self.groups,
                dt,
                current,
                velocity,
                mobility,
                final[5],
                result.linear_residual,
                result.angular_residual,
                result.wrench,
            ],
            device=self.device,
        )
        return result
