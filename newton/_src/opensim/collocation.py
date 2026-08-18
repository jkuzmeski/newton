# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct-collocation trajectory optimization for OpenSim models.

This module ports the core of OpenSim Moco: it transcribes a continuous
optimal-control problem into a sparse nonlinear program with the separated
Hermite-Simpson scheme and solves it with a sequential-quadratic-programming
(SQP) method. The dynamics of a physics-based problem come from the Warp
inverse/forward-dynamics engine (:class:`ForwardDynamics`); the nonlinear
program itself -- the Hessian/Jacobian assembly and the Karush-Kuhn-Tucker
(KKT) linear solves -- runs on the host, mirroring the small dense solves of
the inverse-kinematics and forward-dynamics tools.

The transcription and defect equations match OpenSim Moco's CasOC
Hermite-Simpson transcription: for each mesh interval the interpolation defect
:math:`\\bar x_k - \\tfrac12(x_k+x_{k+1}) - \\tfrac h8(f_k-f_{k+1})=0` and the
Simpson defect :math:`x_{k+1}-x_k - \\tfrac h6(f_k+4\\bar f_k+f_{k+1})=0` are
enforced, and the running cost is integrated with Simpson's rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .dynamics import ForwardDynamics

__all__ = [
    "DirectCollocationSolver",
    "OptimalControlProblem",
    "OptimalControlSolution",
    "create_torque_driven_dynamics",
    "solve_optimal_control",
]

# A dynamics function maps stacked mesh samples to state derivatives:
# ``dynamics(t[M], x[M, num_states], u[M, num_controls]) -> xdot[M, num_states]``.
Dynamics = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
# Scalar cost integrand / endpoint terms.
Integrand = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
Endpoint = Callable[[np.ndarray], float]


@dataclass
class OptimalControlProblem:
    """Continuous optimal-control problem on a fixed time horizon.

    The problem minimizes ``endpoint_cost(x(t_final)) + int integral_cost dt``
    subject to ``xdot = dynamics(t, x, u)`` and the fixed boundary values in
    ``initial_state`` and ``final_state``. A ``None`` entry in a boundary list
    leaves that state component free at that end.

    Args:
        num_states: Number of state variables.
        num_controls: Number of control variables.
        dynamics: State-derivative function, vectorized over mesh samples.
        initial_state: Fixed initial values (``None`` leaves a state free).
        final_state: Fixed final values (``None`` leaves a state free).
        integral_cost: Running-cost integrand ``L(t, x, u)`` or ``None``.
        endpoint_cost: Terminal (Mayer) cost ``phi(x_final)`` or ``None``.
        time_initial: Initial time [s].
        time_final: Final time [s] (used as the fixed horizon, or as the initial
            guess when ``final_time_bounds`` makes the horizon free).
        control_bounds: Box bounds on the controls, either a single
            ``(low, high)`` pair applied to every control or a per-control list
            of ``(low, high)`` pairs (``None`` leaves a control unbounded).
        state_bounds: Per-state ``(low, high)`` box bounds (``None`` leaves a
            state unbounded); a single ``(low, high)`` pair applies to all states.
        final_time_bounds: ``(low, high)`` bounds that make the final time a free
            optimization variable (minimum-time and similar problems). ``None``
            keeps the horizon fixed at ``time_final``.
        minimize_final_time: Add the final time to the objective (a
            ``MocoFinalTimeGoal``); requires ``final_time_bounds``.
        path_constraints: Equality path constraints ``g(t, x, u) = 0`` enforced at
            every collocation point (a ``MocoPathConstraint``). The callable is
            vectorized over mesh samples and returns an array of shape
            ``[num_samples, num_path_constraints]``.
    """

    num_states: int
    num_controls: int
    dynamics: Dynamics
    initial_state: list[float | None]
    final_state: list[float | None]
    integral_cost: Integrand | None = None
    endpoint_cost: Endpoint | None = None
    time_initial: float = 0.0
    time_final: float = 1.0
    control_bounds: tuple[float, float] | list[tuple[float, float] | None] | None = None
    state_bounds: tuple[float, float] | list[tuple[float, float] | None] | None = None
    final_time_bounds: tuple[float, float] | None = None
    minimize_final_time: bool = False
    path_constraints: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None


@dataclass
class OptimalControlSolution:
    """Result of a direct-collocation solve.

    Args:
        time: Mesh time points [s], shape ``[num_mesh_intervals + 1]``.
        states: State trajectory, shape ``[num_mesh_intervals + 1, num_states]``.
        controls: Control trajectory, shape ``[num_mesh_intervals + 1, num_controls]``.
        objective: Optimal objective value.
        constraint_violation: Maximum absolute defect/boundary residual.
        num_iterations: Number of SQP iterations taken.
        converged: Whether the SQP reached the requested tolerance.
    """

    time: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    objective: float
    constraint_violation: float
    num_iterations: int
    converged: bool
    state_names: list[str] = field(default_factory=list)
    control_names: list[str] = field(default_factory=list)


def _expand_bounds(bounds, count):
    """Normalize a bounds specification to a length-``count`` list of pairs or ``None``."""
    if bounds is None:
        return [None] * count
    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]):
        return [(float(bounds[0]), float(bounds[1]))] * count
    out = []
    for b in bounds:
        out.append(None if b is None else (float(b[0]), float(b[1])))
    return out


def _point_jacobian(
    dynamics: Dynamics,
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    ns: int,
    nu: int,
    eps: float = 1.0e-6,
):
    """Central-difference dynamics Jacobians, batched over mesh samples.

    Returns ``fx`` with shape ``[M, ns, ns]`` and ``fu`` with shape ``[M, ns, nu]``.
    """
    samples = x.shape[0]
    variables = ns + nu
    x_batch = np.tile(x, (2 * variables, 1))
    u_batch = np.tile(u, (2 * variables, 1))
    for variable in range(variables):
        plus = 2 * variable * samples
        minus = plus + samples
        if variable < ns:
            x_batch[plus : plus + samples, variable] += eps
            x_batch[minus : minus + samples, variable] -= eps
        else:
            control = variable - ns
            u_batch[plus : plus + samples, control] += eps
            u_batch[minus : minus + samples, control] -= eps
    values = dynamics(np.tile(t, 2 * variables), x_batch, u_batch).reshape(2 * variables, samples, ns)
    derivatives = (values[0::2] - values[1::2]) / (2.0 * eps)
    fx = np.transpose(derivatives[:ns], (1, 2, 0))
    fu = np.transpose(derivatives[ns:], (1, 2, 0))
    return fx, fu


def _point_covector_hessian(
    dynamics: Dynamics,
    t: float,
    xrow: np.ndarray,
    urow: np.ndarray,
    covector: np.ndarray,
    ns: int,
    nu: int,
    eps: float = 1.0e-4,
):
    """Hessian of ``covector . dynamics`` over the local ``[x, u]`` block.

    Supplies the constraint-curvature term of the Lagrangian Hessian so the SQP
    behaves as an exact-Hessian Newton method for nonlinear dynamics.
    """
    nl = ns + nu
    v0 = np.concatenate([xrow, urow])
    pair_count = nl * (nl - 1) // 2
    probes = np.tile(v0, (1 + 2 * nl + 2 * pair_count, 1))
    for variable in range(nl):
        probes[1 + variable, variable] += eps
        probes[1 + nl + variable, variable] -= eps
    pairs: list[tuple[int, int]] = []
    row = 1 + 2 * nl
    for p in range(nl):
        for q in range(p + 1, nl):
            probes[row, p] += eps
            probes[row, q] += eps
            probes[row + 1, p] -= eps
            probes[row + 1, q] -= eps
            pairs.append((p, q))
            row += 2
    values = dynamics(
        np.full(probes.shape[0], t),
        probes[:, :ns],
        probes[:, ns:],
    )
    phi = values @ covector
    p0 = phi[0]
    fp = phi[1 : 1 + nl]
    fm = phi[1 + nl : 1 + 2 * nl]
    hl = np.zeros((nl, nl))
    hl[np.arange(nl), np.arange(nl)] = (fp - 2.0 * p0 + fm) / eps**2
    row = 1 + 2 * nl
    for p, q in pairs:
        value = (phi[row] - fp[p] - fp[q] + 2.0 * p0 - fm[p] - fm[q] + phi[row + 1]) / (2.0 * eps**2)
        hl[p, q] = value
        hl[q, p] = value
        row += 2
    return hl


class DirectCollocationSolver:
    """Separated Hermite-Simpson direct-collocation solver.

    The solver transcribes an :class:`OptimalControlProblem` onto a uniform mesh
    and solves the resulting nonlinear program with an SQP method: at each
    iteration it assembles the exact Lagrangian Hessian (cost curvature plus the
    per-sample dynamics curvature) and the constraint Jacobian, solves the KKT
    system for a step, and globalizes with an l1-merit Armijo line search.

    Args:
        num_mesh_intervals: Number of mesh intervals ``N``; the trajectory has
            ``N + 1`` mesh points.
        max_iterations: Maximum number of SQP iterations.
        tolerance: Convergence tolerance on the maximum constraint violation.
        exact_hessian: Include the dynamics curvature in the Lagrangian Hessian.
            Disable for linear dynamics to save curvature evaluations.
    """

    def __init__(
        self,
        num_mesh_intervals: int = 50,
        max_iterations: int = 100,
        tolerance: float = 1.0e-9,
        exact_hessian: bool = True,
    ):
        self.num_mesh_intervals = int(num_mesh_intervals)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.exact_hessian = bool(exact_hessian)

    def solve(
        self,
        problem: OptimalControlProblem,
        *,
        control_guess: np.ndarray | Callable[[float], np.ndarray] | None = None,
        state_guess: np.ndarray | None = None,
        verbose: bool = False,
    ) -> OptimalControlSolution:
        """Solve ``problem`` and return the discretized optimal trajectory.

        Args:
            problem: The optimal-control problem to solve.
            control_guess: Optional initial control trajectory, either an array
                of shape ``[N + 1, num_controls]`` or a callable ``u(t)``. When a
                callable is given the states are initialized by an RK4 rollout.
            state_guess: Optional initial state trajectory of shape
                ``[N + 1, num_states]`` (linear interpolation is used otherwise).
            verbose: Print per-iteration diagnostics.
        """
        bounded = (
            problem.final_time_bounds is not None
            or problem.control_bounds is not None
            or problem.state_bounds is not None
            or problem.path_constraints is not None
        )
        if bounded:
            return self._solve_interior_point(
                problem, control_guess=control_guess, state_guess=state_guess, verbose=verbose
            )
        ns = problem.num_states
        nu = problem.num_controls
        dyn = problem.dynamics
        n = self.num_mesh_intervals
        t0 = problem.time_initial
        tf = problem.time_final
        h = (tf - t0) / n
        tk = np.linspace(t0, tf, n + 1)
        tb = 0.5 * (tk[:-1] + tk[1:])

        n_x = (n + 1) * ns
        n_xb = n * ns
        n_u = (n + 1) * nu
        nz = n_x + n_xb + n_u + n * nu
        o_x, o_xb, o_u, o_ub = 0, n_x, n_x + n_xb, n_x + n_xb + n_u

        def idx_x(k):
            return o_x + k * ns

        def idx_xb(k):
            return o_xb + k * ns

        def idx_u(k):
            return o_u + k * nu

        def idx_ub(k):
            return o_ub + k * nu

        def unpack(z):
            return (
                z[o_x : o_x + n_x].reshape(n + 1, ns),
                z[o_xb : o_xb + n_xb].reshape(n, ns),
                z[o_u : o_u + n_u].reshape(n + 1, nu),
                z[o_ub : o_ub + n * nu].reshape(n, nu),
            )

        x0f = problem.initial_state
        xff = problem.final_state
        b0 = [i for i in range(ns) if x0f[i] is not None]
        bf = [i for i in range(ns) if xff[i] is not None]
        m = 2 * n * ns + len(b0) + len(bf)

        def residual(z):
            x, xb, u, ub = unpack(z)
            fk = dyn(tk, x, u)
            fb = dyn(tb, xb, ub)
            interp = xb - 0.5 * (x[:-1] + x[1:]) - (h / 8.0) * (fk[:-1] - fk[1:])
            simp = x[1:] - x[:-1] - (h / 6.0) * (fk[:-1] + 4.0 * fb + fk[1:])
            bc = [x[0, i] - x0f[i] for i in b0] + [x[-1, i] - xff[i] for i in bf]
            return np.concatenate([interp.ravel(), simp.ravel(), np.array(bc) if bc else np.zeros(0)])

        def cost(z):
            x, xb, u, ub = unpack(z)
            j = 0.0
            if problem.integral_cost is not None:
                lk = problem.integral_cost(tk, x, u)
                lb = problem.integral_cost(tb, xb, ub)
                j += float(np.sum((h / 6.0) * (lk[:-1] + 4.0 * lb + lk[1:])))
            if problem.endpoint_cost is not None:
                j += float(problem.endpoint_cost(x[-1]))
            return j

        def build(z, lam):
            x, xb, u, ub = unpack(z)
            fxk, fuk = _point_jacobian(dyn, tk, x, u, ns, nu)
            fxb, fub = _point_jacobian(dyn, tb, xb, ub, ns, nu)
            jac = np.zeros((m, nz))
            eye = np.eye(ns)
            r = 0
            for k in range(n):
                for i in range(ns):
                    row = jac[r]
                    row[idx_xb(k) + i] += 1.0
                    for j in range(ns):
                        row[idx_x(k) + j] += -0.5 * eye[i, j] - (h / 8.0) * fxk[k, i, j]
                        row[idx_x(k + 1) + j] += -0.5 * eye[i, j] + (h / 8.0) * fxk[k + 1, i, j]
                    for c in range(nu):
                        row[idx_u(k) + c] += -(h / 8.0) * fuk[k, i, c]
                        row[idx_u(k + 1) + c] += (h / 8.0) * fuk[k + 1, i, c]
                    r += 1
            for k in range(n):
                for i in range(ns):
                    row = jac[r]
                    for j in range(ns):
                        row[idx_x(k) + j] += -eye[i, j] - (h / 6.0) * fxk[k, i, j]
                        row[idx_x(k + 1) + j] += eye[i, j] - (h / 6.0) * fxk[k + 1, i, j]
                        row[idx_xb(k) + j] += -(h / 6.0) * 4.0 * fxb[k, i, j]
                    for c in range(nu):
                        row[idx_u(k) + c] += -(h / 6.0) * fuk[k, i, c]
                        row[idx_u(k + 1) + c] += -(h / 6.0) * fuk[k + 1, i, c]
                        row[idx_ub(k) + c] += -(h / 6.0) * 4.0 * fub[k, i, c]
                    r += 1
            for i in b0:
                jac[r, idx_x(0) + i] = 1.0
                r += 1
            for i in bf:
                jac[r, idx_x(n) + i] = 1.0
                r += 1

            grad = np.zeros(nz)
            hess = np.zeros((nz, nz))
            eps = 1.0e-6

            def add_cost(tt, xx, uu, w, xbase, ubase):
                def l1(xv, uv):
                    return float(problem.integral_cost(np.array([tt]), xv[None, :], uv[None, :])[0])

                l0 = l1(xx, uu)
                for a in range(ns):
                    xp = xx.copy()
                    xp[a] += eps
                    xm = xx.copy()
                    xm[a] -= eps
                    grad[xbase + a] += w * (l1(xp, uu) - l1(xm, uu)) / (2.0 * eps)
                    hess[xbase + a, xbase + a] += w * (l1(xp, uu) - 2.0 * l0 + l1(xm, uu)) / eps**2
                for a in range(nu):
                    up = uu.copy()
                    up[a] += eps
                    um = uu.copy()
                    um[a] -= eps
                    grad[ubase + a] += w * (l1(xx, up) - l1(xx, um)) / (2.0 * eps)
                    hess[ubase + a, ubase + a] += w * (l1(xx, up) - 2.0 * l0 + l1(xx, um)) / eps**2

            if problem.integral_cost is not None:
                for k in range(n + 1):
                    w = (h / 6.0) * (1.0 if (k == 0 or k == n) else 2.0)
                    add_cost(tk[k], x[k], u[k], w, idx_x(k), idx_u(k))
                for k in range(n):
                    add_cost(tb[k], xb[k], ub[k], (h / 6.0) * 4.0, idx_xb(k), idx_ub(k))
            if problem.endpoint_cost is not None:
                xn = x[-1]
                base = idx_x(n)
                e0 = float(problem.endpoint_cost(xn))
                for a in range(ns):
                    xp = xn.copy()
                    xp[a] += eps
                    xm = xn.copy()
                    xm[a] -= eps
                    grad[base + a] += (problem.endpoint_cost(xp) - problem.endpoint_cost(xm)) / (2.0 * eps)
                    hess[base + a, base + a] += (
                        problem.endpoint_cost(xp) - 2.0 * e0 + problem.endpoint_cost(xm)
                    ) / eps**2

            if self.exact_hessian:
                lint = lam[: n * ns].reshape(n, ns)
                lsimp = lam[n * ns : 2 * n * ns].reshape(n, ns)
                for j in range(n + 1):
                    a = np.zeros(ns)
                    if j < n:
                        a += -(h / 8.0) * lint[j] - (h / 6.0) * lsimp[j]
                    if j > 0:
                        a += (h / 8.0) * lint[j - 1] - (h / 6.0) * lsimp[j - 1]
                    if np.any(a):
                        hl = _point_covector_hessian(dyn, tk[j], x[j], u[j], a, ns, nu)
                        loc = list(range(idx_x(j), idx_x(j) + ns)) + list(range(idx_u(j), idx_u(j) + nu))
                        for p in range(ns + nu):
                            for q in range(ns + nu):
                                hess[loc[p], loc[q]] += hl[p, q]
                for j in range(n):
                    a = -(4.0 * h / 6.0) * lsimp[j]
                    if np.any(a):
                        hl = _point_covector_hessian(dyn, tb[j], xb[j], ub[j], a, ns, nu)
                        loc = list(range(idx_xb(j), idx_xb(j) + ns)) + list(range(idx_ub(j), idx_ub(j) + nu))
                        for p in range(ns + nu):
                            for q in range(ns + nu):
                                hess[loc[p], loc[q]] += hl[p, q]
            return jac, grad, hess

        # --- initial guess ---
        z = np.zeros(nz)
        x, xb, u, ub = unpack(z)
        if state_guess is not None:
            x[:] = state_guess
        else:
            for i in range(ns):
                a0 = x0f[i] if x0f[i] is not None else 0.0
                a1 = xff[i] if xff[i] is not None else a0
                x[:, i] = np.linspace(a0, a1, n + 1)
        if callable(control_guess):
            x[0] = np.array([x0f[i] if x0f[i] is not None else 0.0 for i in range(ns)])

            def f1(tt, xx, uu):
                return dyn(np.array([tt]), xx[None, :], np.atleast_1d(uu)[None, :])[0]

            for k in range(n):
                u0 = np.atleast_1d(np.asarray(control_guess(tk[k]), dtype=float))
                um = np.atleast_1d(np.asarray(control_guess(tk[k] + h / 2), dtype=float))
                u1 = np.atleast_1d(np.asarray(control_guess(tk[k] + h), dtype=float))
                k1 = f1(tk[k], x[k], u0)
                k2 = f1(tk[k] + h / 2, x[k] + h / 2 * k1, um)
                k3 = f1(tk[k] + h / 2, x[k] + h / 2 * k2, um)
                k4 = f1(tk[k] + h, x[k] + h * k3, u1)
                x[k + 1] = x[k] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            u[:] = np.array([np.atleast_1d(np.asarray(control_guess(t), dtype=float)) for t in tk])
            ub[:] = np.array([np.atleast_1d(np.asarray(control_guess(t), dtype=float)) for t in tb])
        elif control_guess is not None:
            u[:] = np.asarray(control_guess).reshape(n + 1, nu)
            ub[:] = 0.5 * (u[:-1] + u[1:])
        xb[:] = 0.5 * (x[:-1] + x[1:])
        z = np.concatenate([x.ravel(), xb.ravel(), u.ravel(), ub.ravel()])

        # --- SQP iterations ---
        lam = np.zeros(m)
        converged = False
        it = 0
        for it in range(self.max_iterations):
            c = residual(z)
            jac, grad, hess = build(z, lam)
            hess = 0.5 * (hess + hess.T)
            reg = 1.0e-8
            while True:
                kkt = np.block([[hess + reg * np.eye(nz), jac.T], [jac, np.zeros((m, m))]])
                try:
                    step = np.linalg.solve(kkt, -np.concatenate([grad, c]))
                    break
                except np.linalg.LinAlgError:
                    reg *= 10.0
                    if reg > 1.0e6:
                        step = np.linalg.lstsq(kkt, -np.concatenate([grad, c]), rcond=None)[0]
                        break
            dz = step[:nz]
            lam_new = step[nz:]
            dlam = lam_new - lam
            mu = np.max(np.abs(lam_new)) + 1.0
            c1 = np.sum(np.abs(c))
            dphi = grad @ dz - mu * c1
            phi0 = cost(z) + mu * c1
            al = 1.0
            for _ in range(40):
                zt = z + al * dz
                if cost(zt) + mu * np.sum(np.abs(residual(zt))) <= phi0 + 1.0e-4 * al * dphi:
                    break
                al *= 0.5
            z = z + al * dz
            lam = lam + al * dlam
            cn = float(np.max(np.abs(residual(z))))
            sn = float(np.max(np.abs(al * dz)))
            if verbose:
                print(f"  it{it:02d} ||c||={cn:.3e} step={sn:.3e} alpha={al:.3f} J={cost(z):.8f}")
            if cn < self.tolerance and sn < 1.0e-9:
                converged = True
                break

        x, xb, u, ub = unpack(z)
        return OptimalControlSolution(
            time=tk,
            states=x.copy(),
            controls=u.copy(),
            objective=cost(z),
            constraint_violation=float(np.max(np.abs(residual(z)))),
            num_iterations=it + 1,
            converged=converged,
        )

    def _solve_interior_point(self, problem, *, control_guess=None, state_guess=None, verbose=False):
        """Free-final-time / box-bounded solve via a primal-dual interior point.

        Handles minimum-time and bound-constrained problems: the mesh times scale
        with a free final time when ``final_time_bounds`` is set, and box bounds on
        the states and controls are enforced with a log-barrier whose primal-dual
        Newton step reuses the same Karush-Kuhn-Tucker structure as the equality
        solver (the barrier only adds a diagonal term and a modified gradient).
        The dynamics are assumed autonomous, as for mechanical systems.
        """
        ns = problem.num_states
        nu = problem.num_controls
        dyn = problem.dynamics
        n = self.num_mesh_intervals
        t0 = problem.time_initial
        free_time = problem.final_time_bounds is not None

        n_x = (n + 1) * ns
        n_xb = n * ns
        n_u = (n + 1) * nu
        n_ub = n * nu
        base = n_x + n_xb + n_u + n_ub
        nz = base + (1 if free_time else 0)
        i_tf = base
        o_x, o_xb, o_u, o_ub = 0, n_x, n_x + n_xb, n_x + n_xb + n_u
        tau = np.linspace(0.0, 1.0, n + 1)
        taub = 0.5 * (tau[:-1] + tau[1:])

        def idx_x(k):
            return o_x + k * ns

        def idx_xb(k):
            return o_xb + k * ns

        def idx_u(k):
            return o_u + k * nu

        def idx_ub(k):
            return o_ub + k * nu

        def unpack(z):
            tf = z[i_tf] if free_time else problem.time_final
            return (
                z[o_x : o_x + n_x].reshape(n + 1, ns),
                z[o_xb : o_xb + n_xb].reshape(n, ns),
                z[o_u : o_u + n_u].reshape(n + 1, nu),
                z[o_ub : o_ub + n_ub].reshape(n, nu),
                tf,
            )

        x0f = problem.initial_state
        xff = problem.final_state
        b0 = [i for i in range(ns) if x0f[i] is not None]
        bf = [i for i in range(ns) if xff[i] is not None]
        pcon = problem.path_constraints
        n_path = int(pcon(np.array([t0]), np.zeros((1, ns)), np.zeros((1, nu))).shape[1]) if pcon else 0
        o_path = 2 * n * ns + len(b0) + len(bf)
        m = o_path + (2 * n + 1) * n_path

        def residual(z):
            x, xb, u, ub, tf = unpack(z)
            h = (tf - t0) / n
            tk = t0 + (tf - t0) * tau
            tb = t0 + (tf - t0) * taub
            fk = dyn(tk, x, u)
            fb = dyn(tb, xb, ub)
            interp = xb - 0.5 * (x[:-1] + x[1:]) - (h / 8.0) * (fk[:-1] - fk[1:])
            simp = x[1:] - x[:-1] - (h / 6.0) * (fk[:-1] + 4.0 * fb + fk[1:])
            bc = [x[0, i] - x0f[i] for i in b0] + [x[-1, i] - xff[i] for i in bf]
            parts = [interp.ravel(), simp.ravel(), np.array(bc) if bc else np.zeros(0)]
            if n_path:
                parts.append(pcon(tk, x, u).ravel())
                parts.append(pcon(tb, xb, ub).ravel())
            return np.concatenate(parts)

        def cost(z):
            x, xb, u, ub, tf = unpack(z)
            h = (tf - t0) / n
            tk = t0 + (tf - t0) * tau
            tb = t0 + (tf - t0) * taub
            j = 0.0
            if problem.integral_cost is not None:
                lk = problem.integral_cost(tk, x, u)
                lb = problem.integral_cost(tb, xb, ub)
                j += float(np.sum((h / 6.0) * (lk[:-1] + 4.0 * lb + lk[1:])))
            if problem.endpoint_cost is not None:
                j += float(problem.endpoint_cost(x[-1]))
            if problem.minimize_final_time:
                j += float(tf)
            return j

        def cost_derivatives(z, with_hessian=True):
            """Return finite-difference cost derivatives without evaluating dynamics."""
            x, xb, u, ub, tf = unpack(z)
            h = (tf - t0) / n
            tk = t0 + (tf - t0) * tau
            tb = t0 + (tf - t0) * taub
            grad = np.zeros(nz)
            hess = np.zeros((nz, nz)) if with_hessian else None
            eps = 1.0e-6

            def add_cost(tt, xx, uu, w, xbase, ubase):
                def l1(xv, uv):
                    return float(problem.integral_cost(np.array([tt]), xv[None, :], uv[None, :])[0])

                l0 = l1(xx, uu) if with_hessian else 0.0
                for a in range(ns):
                    xp = xx.copy()
                    xp[a] += eps
                    xm = xx.copy()
                    xm[a] -= eps
                    lp = l1(xp, uu)
                    lm = l1(xm, uu)
                    grad[xbase + a] += w * (lp - lm) / (2.0 * eps)
                    if with_hessian:
                        hess[xbase + a, xbase + a] += w * (lp - 2.0 * l0 + lm) / eps**2
                for a in range(nu):
                    up = uu.copy()
                    up[a] += eps
                    um = uu.copy()
                    um[a] -= eps
                    lp = l1(xx, up)
                    lm = l1(xx, um)
                    grad[ubase + a] += w * (lp - lm) / (2.0 * eps)
                    if with_hessian:
                        hess[ubase + a, ubase + a] += w * (lp - 2.0 * l0 + lm) / eps**2

            if problem.integral_cost is not None:
                for k in range(n + 1):
                    w = (h / 6.0) * (1.0 if (k == 0 or k == n) else 2.0)
                    add_cost(tk[k], x[k], u[k], w, idx_x(k), idx_u(k))
                for k in range(n):
                    add_cost(tb[k], xb[k], ub[k], (h / 6.0) * 4.0, idx_xb(k), idx_ub(k))
            if problem.endpoint_cost is not None:
                xn = x[-1]
                bidx = idx_x(n)
                e0 = float(problem.endpoint_cost(xn)) if with_hessian else 0.0
                for a in range(ns):
                    xp = xn.copy()
                    xp[a] += eps
                    xm = xn.copy()
                    xm[a] -= eps
                    ep = problem.endpoint_cost(xp)
                    em = problem.endpoint_cost(xm)
                    grad[bidx + a] += (ep - em) / (2.0 * eps)
                    if with_hessian:
                        hess[bidx + a, bidx + a] += (ep - 2.0 * e0 + em) / eps**2
            if problem.minimize_final_time:
                grad[i_tf] += 1.0
            if free_time and (problem.integral_cost is not None):
                zp = z.copy()
                zp[i_tf] += eps
                zm = z.copy()
                zm[i_tf] -= eps
                grad[i_tf] += (cost(zp) - cost(zm)) / (2.0 * eps) - (1.0 if problem.minimize_final_time else 0.0)
            return grad, hess

        def build(z, lam):
            x, xb, u, ub, tf = unpack(z)
            h = (tf - t0) / n
            tk = t0 + (tf - t0) * tau
            tb = t0 + (tf - t0) * taub
            fxk, fuk = _point_jacobian(dyn, tk, x, u, ns, nu)
            fxb, fub = _point_jacobian(dyn, tb, xb, ub, ns, nu)
            fk = dyn(tk, x, u)
            fb = dyn(tb, xb, ub)
            jac = np.zeros((m, nz))
            eye = np.eye(ns)
            r = 0
            for k in range(n):
                for i in range(ns):
                    row = jac[r]
                    row[idx_xb(k) + i] += 1.0
                    for j in range(ns):
                        row[idx_x(k) + j] += -0.5 * eye[i, j] - (h / 8.0) * fxk[k, i, j]
                        row[idx_x(k + 1) + j] += -0.5 * eye[i, j] + (h / 8.0) * fxk[k + 1, i, j]
                    for c in range(nu):
                        row[idx_u(k) + c] += -(h / 8.0) * fuk[k, i, c]
                        row[idx_u(k + 1) + c] += (h / 8.0) * fuk[k + 1, i, c]
                    if free_time:
                        row[i_tf] += -(1.0 / (8.0 * n)) * (fk[k, i] - fk[k + 1, i])
                    r += 1
            for k in range(n):
                for i in range(ns):
                    row = jac[r]
                    for j in range(ns):
                        row[idx_x(k) + j] += -eye[i, j] - (h / 6.0) * fxk[k, i, j]
                        row[idx_x(k + 1) + j] += eye[i, j] - (h / 6.0) * fxk[k + 1, i, j]
                        row[idx_xb(k) + j] += -(h / 6.0) * 4.0 * fxb[k, i, j]
                    for c in range(nu):
                        row[idx_u(k) + c] += -(h / 6.0) * fuk[k, i, c]
                        row[idx_u(k + 1) + c] += -(h / 6.0) * fuk[k + 1, i, c]
                        row[idx_ub(k) + c] += -(h / 6.0) * 4.0 * fub[k, i, c]
                    if free_time:
                        row[i_tf] += -(1.0 / (6.0 * n)) * (fk[k, i] + 4.0 * fb[k, i] + fk[k + 1, i])
                    r += 1
            for i in b0:
                jac[r, idx_x(0) + i] = 1.0
                r += 1
            for i in bf:
                jac[r, idx_x(n) + i] = 1.0
                r += 1

            if n_path:
                geps = 1.0e-6

                def _gjac(tt, xx, uu):
                    gx = np.zeros((xx.shape[0], n_path, ns))
                    gu = np.zeros((xx.shape[0], n_path, nu))
                    for a in range(ns):
                        xp = xx.copy()
                        xp[:, a] += geps
                        xm = xx.copy()
                        xm[:, a] -= geps
                        gx[:, :, a] = (pcon(tt, xp, uu) - pcon(tt, xm, uu)) / (2.0 * geps)
                    for a in range(nu):
                        up = uu.copy()
                        up[:, a] += geps
                        um = uu.copy()
                        um[:, a] -= geps
                        gu[:, :, a] = (pcon(tt, xx, up) - pcon(tt, xx, um)) / (2.0 * geps)
                    return gx, gu

                gxk, guk = _gjac(tk, x, u)
                gxb, gub = _gjac(tb, xb, ub)
                for k in range(n + 1):
                    for ip in range(n_path):
                        row = jac[r]
                        for j in range(ns):
                            row[idx_x(k) + j] += gxk[k, ip, j]
                        for c in range(nu):
                            row[idx_u(k) + c] += guk[k, ip, c]
                        r += 1
                for k in range(n):
                    for ip in range(n_path):
                        row = jac[r]
                        for j in range(ns):
                            row[idx_xb(k) + j] += gxb[k, ip, j]
                        for c in range(nu):
                            row[idx_ub(k) + c] += gub[k, ip, c]
                        r += 1

            grad, hess = cost_derivatives(z)

            if self.exact_hessian:
                lint = lam[: n * ns].reshape(n, ns)
                lsimp = lam[n * ns : 2 * n * ns].reshape(n, ns)
                for j in range(n + 1):
                    a = np.zeros(ns)
                    if j < n:
                        a += -(h / 8.0) * lint[j] - (h / 6.0) * lsimp[j]
                    if j > 0:
                        a += (h / 8.0) * lint[j - 1] - (h / 6.0) * lsimp[j - 1]
                    if np.any(a):
                        hl = _point_covector_hessian(dyn, tk[j], x[j], u[j], a, ns, nu)
                        loc = list(range(idx_x(j), idx_x(j) + ns)) + list(range(idx_u(j), idx_u(j) + nu))
                        for p in range(ns + nu):
                            for q in range(ns + nu):
                                hess[loc[p], loc[q]] += hl[p, q]
                for j in range(n):
                    a = -(4.0 * h / 6.0) * lsimp[j]
                    if np.any(a):
                        hl = _point_covector_hessian(dyn, tb[j], xb[j], ub[j], a, ns, nu)
                        loc = list(range(idx_xb(j), idx_xb(j) + ns)) + list(range(idx_ub(j), idx_ub(j) + nu))
                        for p in range(ns + nu):
                            for q in range(ns + nu):
                                hess[loc[p], loc[q]] += hl[p, q]
                if n_path:
                    lpk = lam[o_path : o_path + (n + 1) * n_path].reshape(n + 1, n_path)
                    lpb = lam[o_path + (n + 1) * n_path : m].reshape(n, n_path)
                    for j in range(n + 1):
                        if np.any(lpk[j]):
                            hl = _point_covector_hessian(pcon, tk[j], x[j], u[j], lpk[j], ns, nu)
                            loc = list(range(idx_x(j), idx_x(j) + ns)) + list(range(idx_u(j), idx_u(j) + nu))
                            for p in range(ns + nu):
                                for q in range(ns + nu):
                                    hess[loc[p], loc[q]] += hl[p, q]
                    for j in range(n):
                        if np.any(lpb[j]):
                            hl = _point_covector_hessian(pcon, tb[j], xb[j], ub[j], lpb[j], ns, nu)
                            loc = list(range(idx_xb(j), idx_xb(j) + ns)) + list(range(idx_ub(j), idx_ub(j) + nu))
                            for p in range(ns + nu):
                                for q in range(ns + nu):
                                    hess[loc[p], loc[q]] += hl[p, q]
            return jac, grad, hess

        # --- box bounds (inf where unbounded); exclude boundary-fixed nodes ---
        lb = np.full(nz, -np.inf)
        ub_ = np.full(nz, np.inf)
        sb = _expand_bounds(problem.state_bounds, ns)
        cb = _expand_bounds(problem.control_bounds, nu)
        for k in range(n + 1):
            for i in range(ns):
                if sb[i] is not None:
                    lb[idx_x(k) + i], ub_[idx_x(k) + i] = sb[i]
        for k in range(n):
            for i in range(ns):
                if sb[i] is not None:
                    lb[idx_xb(k) + i], ub_[idx_xb(k) + i] = sb[i]
        for k in range(n + 1):
            for c in range(nu):
                if cb[c] is not None:
                    lb[idx_u(k) + c], ub_[idx_u(k) + c] = cb[c]
        for k in range(n):
            for c in range(nu):
                if cb[c] is not None:
                    lb[idx_ub(k) + c], ub_[idx_ub(k) + c] = cb[c]
        for i in b0:
            lb[idx_x(0) + i], ub_[idx_x(0) + i] = -np.inf, np.inf
        for i in bf:
            lb[idx_x(n) + i], ub_[idx_x(n) + i] = -np.inf, np.inf
        if free_time:
            lb[i_tf], ub_[i_tf] = problem.final_time_bounds

        # --- initial guess ---
        z = np.zeros(nz)
        if free_time:
            z[i_tf] = problem.time_final
        x, xb, u, ub, _ = unpack(z)
        tk_g = t0 + (problem.time_final - t0) * tau
        for i in range(ns):
            a0 = x0f[i] if x0f[i] is not None else 0.0
            a1 = xff[i] if xff[i] is not None else a0
            x[:, i] = np.linspace(a0, a1, n + 1)
        if state_guess is not None:
            if callable(state_guess):
                x[:] = np.array([np.atleast_1d(np.asarray(state_guess(tt), dtype=float)) for tt in tk_g])
            else:
                x[:] = np.asarray(state_guess).reshape(n + 1, ns)
            for i in b0:
                x[0, i] = x0f[i]
            for i in bf:
                x[-1, i] = xff[i]
        xb[:] = 0.5 * (x[:-1] + x[1:])
        if control_guess is not None:
            if callable(control_guess):
                u[:] = np.array([np.atleast_1d(np.asarray(control_guess(tt), dtype=float)) for tt in tk_g])
            else:
                u[:] = np.asarray(control_guess).reshape(n + 1, nu)
            ub[:] = 0.5 * (u[:-1] + u[1:])
        z = np.concatenate([x.ravel(), xb.ravel(), u.ravel(), ub.ravel(), ([z[i_tf]] if free_time else [])])
        fin_l = np.isfinite(lb)
        fin_u = np.isfinite(ub_)
        for i in range(nz):
            if fin_l[i] and fin_u[i]:
                z[i] = min(max(z[i], lb[i] + 0.01 * (ub_[i] - lb[i])), ub_[i] - 0.01 * (ub_[i] - lb[i]))
            elif fin_l[i]:
                z[i] = max(z[i], lb[i] + 1.0e-2)
            elif fin_u[i]:
                z[i] = min(z[i], ub_[i] - 1.0e-2)

        # --- primal-dual interior point with mu continuation ---
        lam = np.zeros(m)
        zl = np.where(fin_l, 1.0, 0.0)
        zu = np.where(fin_u, 1.0, 0.0)
        mu = 0.1
        tau_fb = 0.995
        converged = False
        total_it = 0
        for _ in range(80):
            for _ in range(40):
                total_it += 1
                c = residual(z)
                jac, grad, hess = build(z, lam)
                hess = 0.5 * (hess + hess.T)
                dl = np.where(fin_l, z - lb, 1.0)
                du = np.where(fin_u, ub_ - z, 1.0)
                sigma = np.where(fin_l, zl / dl, 0.0) + np.where(fin_u, zu / du, 0.0)
                rhs_g = grad + jac.T @ lam - np.where(fin_l, mu / dl, 0.0) + np.where(fin_u, mu / du, 0.0)
                kkt = np.block([[hess + np.diag(sigma) + 1.0e-10 * np.eye(nz), jac.T], [jac, np.zeros((m, m))]])
                try:
                    sol = np.linalg.solve(kkt, -np.concatenate([rhs_g, c]))
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(kkt, -np.concatenate([rhs_g, c]), rcond=None)[0]
                dz = sol[:nz]
                dlam = sol[nz:]
                dzl = np.where(fin_l, mu / dl - zl - (zl / dl) * dz, 0.0)
                dzu = np.where(fin_u, mu / du - zu + (zu / du) * dz, 0.0)
                a_p = 1.0
                for i in range(nz):
                    if fin_l[i] and dz[i] < 0:
                        a_p = min(a_p, -tau_fb * dl[i] / dz[i])
                    if fin_u[i] and dz[i] > 0:
                        a_p = min(a_p, tau_fb * du[i] / dz[i])
                a_d = 1.0
                for i in range(nz):
                    if fin_l[i] and dzl[i] < 0:
                        a_d = min(a_d, -tau_fb * zl[i] / dzl[i])
                    if fin_u[i] and dzu[i] < 0:
                        a_d = min(a_d, -tau_fb * zu[i] / dzu[i])
                z = z + a_p * dz
                lam = lam + a_d * dlam
                zl = np.maximum(zl + a_d * dzl, 1.0e-12)
                zu = np.maximum(zu + a_d * dzu, 1.0e-12)
                grad2 = cost_derivatives(z, with_hessian=False)[0]
                stat = np.max(np.abs(grad2 + jac.T @ lam - np.where(fin_l, zl, 0.0) + np.where(fin_u, zu, 0.0)))
                kkt_err = max(float(np.max(np.abs(residual(z)))), float(stat))
                if kkt_err < max(1.0e-9, 5.0 * mu):
                    break
            if verbose:
                print(f"  mu={mu:.2e} cviol={np.max(np.abs(residual(z))):.3e} J={cost(z):.8f}")
            if mu < 1.0e-9:
                converged = float(np.max(np.abs(residual(z)))) < max(1.0e-6, self.tolerance)
                break
            mu *= 0.2

        x, xb, u, ub, tf = unpack(z)
        return OptimalControlSolution(
            time=t0 + (tf - t0) * tau,
            states=x.copy(),
            controls=u.copy(),
            objective=cost(z),
            constraint_violation=float(np.max(np.abs(residual(z)))),
            num_iterations=total_it,
            converged=converged,
        )


def solve_optimal_control(
    problem: OptimalControlProblem,
    num_mesh_intervals: int = 50,
    *,
    max_iterations: int = 100,
    tolerance: float = 1.0e-9,
    exact_hessian: bool = True,
    control_guess=None,
    state_guess=None,
    verbose: bool = False,
) -> OptimalControlSolution:
    """Solve an :class:`OptimalControlProblem` with Hermite-Simpson collocation.

    Convenience wrapper around :class:`DirectCollocationSolver`.
    """
    solver = DirectCollocationSolver(
        num_mesh_intervals=num_mesh_intervals,
        max_iterations=max_iterations,
        tolerance=tolerance,
        exact_hessian=exact_hessian,
    )
    return solver.solve(problem, control_guess=control_guess, state_guess=state_guess, verbose=verbose)


def create_torque_driven_dynamics(
    model, coordinates: list[str] | None = None, gains: np.ndarray | None = None, device=None
) -> Dynamics:
    r"""Build a torque-driven state-derivative function backed by Warp forward dynamics.

    The returned callable maps a stacked batch of states :math:`x=[q,\dot q]` and
    controls :math:`u` to :math:`\dot x=[\dot q,\ddot q]`, where the generalized
    forces are :math:`\tau_i=\mathrm{gains}_i\,u_i` for the actuated coordinates and
    the accelerations come from :class:`ForwardDynamics`. This is the physics-based
    dynamics hook for :class:`DirectCollocationSolver`.

    Args:
        model: Parsed model IR.
        coordinates: Names of the actuated coordinates (defaults to all model
            coordinates, in model order).
        gains: Actuator gains mapping controls to generalized forces
            (defaults to ones), shape ``[num_controls]``.
        device: Warp device for the forward-dynamics kernels.

    Returns:
        A ``dynamics(t, x, u)`` function with ``num_states = 2 * num_coordinates``
        and ``num_controls = len(coordinates)``.
    """
    fd = ForwardDynamics(model, device=device)
    nc = fd.ncoord
    coord_names = list(fd.coordinate_names)
    if coordinates is None:
        coordinates = coord_names
    act_idx = np.array([coord_names.index(name) for name in coordinates], dtype=int)
    g = np.ones(len(coordinates)) if gains is None else np.asarray(gains, dtype=float)

    def dynamics(t, x, u):
        q = np.ascontiguousarray(x[:, :nc])
        qd = np.ascontiguousarray(x[:, nc:])
        tau = np.zeros((x.shape[0], nc))
        tau[:, act_idx] = u * g
        qdd = fd.accelerations(q, qd, tau)
        return np.concatenate([qd, qdd], axis=1)

    return dynamics
