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

r"""Moco muscle-driven tools: ``MocoInverse`` and ``MocoTrack``.

Both tools sit on top of the Hermite-Simpson direct-collocation core in
``collocation`` and the Warp-native muscle, path, and
dynamics primitives.

``MocoInverse`` prescribes the model kinematics and solves for the muscle
excitations that reproduce the inverse-dynamics net joint moments while
minimizing the integral of squared excitation. Because the rigid-tendon
De Groote-Fregly muscle force is affine in activation at a fixed pose and
velocity,

.. math::

    F_m(a_m; t) = a_m\,A_m(t) + P_m(t),

the muscle contribution to each joint moment is linear in the activation states.
The trajectory-optimization states are the muscle activations, the controls are
the excitations driving the first-order activation dynamics, and an equality
*path constraint* enforces the moment balance

.. math::

    \sum_m r_{m,c}(t)\,\big(a_m A_m(t) + P_m(t)\big) + \sum_j \delta_{jc} u^{res}_j
        = \tau_c^{ID}(t)

at every collocation point. The time-varying coefficients
:math:`r_{m,c}(t)`, :math:`A_m(t)`, :math:`P_m(t)`, :math:`\tau^{ID}_c(t)` are
precomputed from the prescribed motion on a dense grid and interpolated. This is
exactly the inverse muscle problem OpenSim's ``MocoInverse`` solves.

``MocoTrack`` sets up a state-tracking optimal-control problem: it minimizes a
weighted sum of the squared deviation from a reference coordinate trajectory and
a control-effort term, subject to the model dynamics (torque-driven by default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .collocation import DirectCollocationSolver, OptimalControlProblem, create_torque_driven_dynamics
from .dynamics import ExternalLoads, InverseDynamics, differentiate_coordinates, read_external_loads
from .mocap import Storage, read_storage, write_storage
from .muscle_force import MuscleForces
from .parser import parse_osim
from .static_optimization import solve_frame_activations
from .types import OsimModel


@dataclass
class MocoInverseSolution:
    """Result of a :class:`MocoInverse` solve.

    Attributes:
        times: Mesh times [s], shape ``[num_nodes]``.
        muscle_names: Muscle names in column order.
        excitations: Neural excitations in [0, 1], shape ``[num_nodes, num_muscles]``.
        activations: Muscle activations in [0, 1], shape ``[num_nodes, num_muscles]``.
        muscle_forces: Rigid-tendon muscle forces [N], shape ``[num_nodes, num_muscles]``.
        coordinate_names: Coordinates the moment balance was enforced on.
        reserve_names: Reserve coordinate-actuator names (empty when reserves are off).
        reserve_forces: Reserve generalized forces, shape ``[num_nodes, num_reserves]``.
        objective: Optimal objective (integral of squared excitation).
        constraint_violation: Maximum absolute defect/path-constraint residual.
        converged: Whether the solver reached its tolerance.
    """

    times: np.ndarray
    muscle_names: list[str]
    excitations: np.ndarray
    activations: np.ndarray
    muscle_forces: np.ndarray
    coordinate_names: list[str]
    reserve_names: list[str]
    reserve_forces: np.ndarray
    objective: float
    constraint_violation: float
    converged: bool

    def to_storage(self) -> Storage:
        """Return the excitations (controls) as a :class:`~newton.opensim.Storage`."""
        labels = [f"{n}" for n in self.muscle_names]
        return Storage(
            times=np.asarray(self.times, float),
            labels=labels,
            data=np.asarray(self.excitations, float),
            in_degrees=False,
            name="MocoInverse controls",
        )

    def write_sto(self, path: str | os.PathLike) -> None:
        """Write the excitation trajectory to an OpenSim ``.sto`` file."""
        write_storage(
            path,
            np.asarray(self.times, float),
            list(self.muscle_names),
            np.asarray(self.excitations, float),
            name="MocoInverse controls",
            in_degrees=False,
        )


class MocoInverse:
    """OpenSim ``MocoInverse``: muscle excitations from prescribed kinematics.

    Args:
        model: Parsed OpenSim model.
        device: Warp device for the muscle/dynamics kernels (``None`` for the CPU).
        use_reserves: Add a reserve/residual coordinate actuator on every enforced
            coordinate so the moment balance stays feasible.
        reserve_optimal_force: Optimal force [N or N·m] of the reserve actuators.
        reserve_weight: Objective weight on the squared reserve controls.
    """

    def __init__(
        self,
        model: OsimModel,
        device=None,
        use_reserves: bool = True,
        reserve_optimal_force: float = 1.0,
        reserve_weight: float = 10.0,
    ):
        self.model = model
        self.device = device
        self.use_reserves = bool(use_reserves)
        self.reserve_optimal_force = float(reserve_optimal_force)
        self.reserve_weight = float(reserve_weight)
        self.muscles = MuscleForces(model, device=device)
        self.id = InverseDynamics(model, device=device)
        self.muscle_names = list(self.muscles.muscle_names)
        self.coordinate_names = list(self.id.coordinate_names)
        self.ncoord = len(self.coordinate_names)
        self.num_muscles = self.muscles.num_muscles
        self.tau_act = np.asarray(self.muscles._tau_act, float)
        self.tau_deact = np.asarray(self.muscles._tau_deact, float)

    def _affine_coeffs(self, coords, speeds):
        """Return ``(A, P)`` affine muscle-force coefficients per frame from the rigid-tendon model."""
        _, active, passive = self.muscles._affine_coefficients(coords, speeds)
        return active, passive

    def solve(
        self,
        coordinates: Storage | str | os.PathLike,
        external_loads: ExternalLoads | None = None,
        cutoff: float = 6.0,
        time_range: tuple[float, float] | None = None,
        num_nodes: int = 50,
        min_activation: float = 0.0,
        activation_dynamics: bool = True,
        activation_exponent: int = 2,
        actuated_coordinates: list[str] | None = None,
        verbose: bool = False,
    ) -> MocoInverseSolution:
        """Solve for the muscle excitations that reproduce the prescribed motion.

        The muscle-redundancy problem is resolved at every node by minimizing the
        summed muscle effort :math:`\\sum_m a_m^p` (plus optional reserve effort)
        subject to the inverse-dynamics moment balance, using the affine
        rigid-tendon force decomposition. Neural excitations are then recovered by
        inverting the first-order activation dynamics,
        :math:`e = a + \tau(a)\\,\\dot a`, so a held pose returns ``e == a`` and a
        fully determined coordinate returns the exact required activation.

        Args:
            coordinates: Prescribed coordinate trajectory (``.mot`` path or :class:`Storage`).
            external_loads: Optional applied external loads (e.g. ground reactions).
            cutoff: Butterworth low-pass cutoff [Hz] for the kinematics; ``<= 0`` disables it.
            time_range: Optional ``(start, end)`` [s] horizon; defaults to the motion span.
            num_nodes: Number of evenly spaced nodes the problem is solved on.
            min_activation: Lower bound applied to the recovered activations.
            activation_dynamics: Recover excitations by inverting the activation
                dynamics. When ``False`` the excitations equal the activations.
            activation_exponent: Muscle-effort exponent ``p`` (default 2).
            actuated_coordinates: Coordinates whose moment balance is enforced. Defaults
                to every coordinate spanned by at least one muscle.
            verbose: Print per-node diagnostics.
        """
        if not isinstance(coordinates, Storage):
            coordinates = read_storage(coordinates)
        all_times = np.asarray(coordinates.times, float)
        col_index = {lab: i for i, lab in enumerate(coordinates.labels)}
        values = np.zeros((len(all_times), self.ncoord))
        defaults = {c.name: c.default_value for j in self.model.joints for c in j.coordinates}
        is_rot = [mt == "rotational" for mt in self.id.motion_types]
        for i, name in enumerate(self.coordinate_names):
            if name in col_index:
                values[:, i] = coordinates.data[:, col_index[name]]
            else:
                default = defaults.get(name, 0.0)
                values[:, i] = np.rad2deg(default) if (is_rot[i] and coordinates.in_degrees) else default

        t0 = all_times[0] if time_range is None else time_range[0]
        tf = all_times[-1] if time_range is None else time_range[1]
        node_t = np.linspace(t0, tf, int(num_nodes))

        q_n, qd_n, qdd_n = differentiate_coordinates(
            all_times, values, is_rot, output_times=node_t, cutoff=cutoff, in_degrees=coordinates.in_degrees
        )
        bodies, wrenches = (None, None)
        if external_loads is not None:
            bodies, wrenches = external_loads.sample(node_t)
        tau_n = self.id.solve(q_n, qd_n, qdd_n, external_bodies=bodies, external_wrenches=wrenches)  # [N, nc]
        R_n, A_n, P_n = self.muscles._affine_coefficients(q_n, qd_n)  # [N, nm, nc], [N, nm]

        # Enforced coordinates: those spanned by a muscle (unless overridden).
        if actuated_coordinates is None:
            spanned = np.max(np.abs(R_n), axis=(0, 1)) > 1e-8  # [nc]
            act_c = [i for i in range(self.ncoord) if spanned[i]]
        else:
            act_c = [self.coordinate_names.index(n) for n in actuated_coordinates]
        act_names = [self.coordinate_names[i] for i in act_c]
        nca = len(act_c)
        nm = self.num_muscles
        f_opt = self.reserve_optimal_force

        num = len(node_t)
        acts = np.zeros((num, nm))
        reserves = np.zeros((num, nca))
        residuals = np.zeros((num, nca))
        x0 = None
        for k in range(num):
            Rk = R_n[k][:, act_c].T  # [nca, nm]
            a, u, resid = solve_frame_activations(
                Rk,
                A_n[k],
                P_n[k],
                tau_n[k, act_c],
                activation_exponent=activation_exponent,
                reserve_optimal_force=f_opt,
                use_reserves=self.use_reserves,
                x0=x0,
            )
            acts[k] = a
            if self.use_reserves:
                reserves[k] = u
                x0 = np.concatenate([a, u])
            else:
                x0 = a
            residuals[k] = np.atleast_1d(resid)
            if verbose:
                print(f"node {k}: t={node_t[k]:.4f} max|resid|={np.max(np.abs(resid)):.2e}")

        acts = np.clip(acts, min_activation, 1.0)

        # Recover excitations by inverting the first-order activation dynamics.
        if activation_dynamics and num > 1:
            adot = np.gradient(acts, node_t, axis=0)  # [N, nm]
            tau_act = self.tau_act[None, :]
            tau_deact = self.tau_deact[None, :]
            tau_eff = np.where(adot > 0.0, tau_act * (0.5 + 1.5 * acts), tau_deact / (0.5 + 1.5 * acts))
            exc = np.clip(acts + tau_eff * adot, 0.0, 1.0)
        else:
            exc = acts.copy()

        forces = acts * A_n + P_n
        reserve_forces = reserves * f_opt if self.use_reserves else np.zeros((num, 0))
        reserve_names = [f"{n}_reserve" for n in act_names] if self.use_reserves else []
        max_resid = float(np.max(np.abs(residuals))) if residuals.size else 0.0
        return MocoInverseSolution(
            times=node_t,
            muscle_names=self.muscle_names,
            excitations=exc,
            activations=acts,
            muscle_forces=forces,
            coordinate_names=act_names,
            reserve_names=reserve_names,
            reserve_forces=reserve_forces,
            objective=float(np.sum(acts**activation_exponent)),
            constraint_violation=max_resid,
            converged=bool(max_resid < 1e-4),
        )


def solve_moco_inverse(
    model: OsimModel | str | os.PathLike,
    coordinates: Storage | str | os.PathLike,
    external_loads: ExternalLoads | str | os.PathLike | None = None,
    cutoff: float = 6.0,
    time_range: tuple[float, float] | None = None,
    num_nodes: int = 50,
    use_reserves: bool = True,
    reserve_optimal_force: float = 1.0,
    activation_dynamics: bool = True,
    device=None,
    verbose: bool = False,
) -> MocoInverseSolution:
    """Run ``MocoInverse`` end to end from a model and a prescribed motion.

    Args:
        model: A parsed :class:`OsimModel`, or a path/XML string to parse.
        coordinates: Prescribed coordinate trajectory (``.mot`` path or :class:`Storage`).
        external_loads: Optional :class:`ExternalLoads`, or a path to a setup XML.
        cutoff: Butterworth low-pass cutoff [Hz]; ``<= 0`` disables filtering.
        time_range: Optional ``(start, end)`` [s] horizon.
        num_nodes: Number of evenly spaced nodes the problem is solved on.
        use_reserves: Add reserve coordinate actuators to keep the balance feasible.
        reserve_optimal_force: Optimal force of the reserve actuators.
        device: Warp device for the kernels (``None`` for the CPU).
        verbose: Print solver diagnostics.
    """
    if not isinstance(model, OsimModel):
        model = parse_osim(model)
    if external_loads is not None and not isinstance(external_loads, ExternalLoads):
        external_loads = read_external_loads(external_loads)
    tool = MocoInverse(model, device=device, use_reserves=use_reserves, reserve_optimal_force=reserve_optimal_force)
    return tool.solve(
        coordinates,
        external_loads=external_loads,
        cutoff=cutoff,
        time_range=time_range,
        num_nodes=num_nodes,
        activation_dynamics=activation_dynamics,
        verbose=verbose,
    )


@dataclass
class MocoTrackSolution:
    """Result of a :class:`MocoTrack` solve.

    Attributes:
        times: Mesh times [s], shape ``[num_nodes]``.
        coordinate_names: Tracked coordinate names in column order.
        states: State trajectory ``[q, qd]``, shape ``[num_nodes, 2 * num_coordinates]``.
        controls: Control trajectory, shape ``[num_nodes, num_controls]``.
        objective: Optimal objective value.
        constraint_violation: Maximum absolute defect/boundary residual.
        converged: Whether the solver reached its tolerance.
    """

    times: np.ndarray
    coordinate_names: list[str]
    states: np.ndarray
    controls: np.ndarray
    objective: float
    constraint_violation: float
    converged: bool
    tracking_rms: float = 0.0

    def coordinates(self) -> np.ndarray:
        """Return the tracked coordinate values ``q``, shape ``[num_nodes, num_coordinates]``."""
        nc = len(self.coordinate_names)
        return self.states[:, :nc]

    def to_storage(self) -> Storage:
        """Return the tracked coordinates as a :class:`~newton.opensim.Storage` [rad]."""
        return Storage(
            times=np.asarray(self.times, float),
            labels=list(self.coordinate_names),
            data=self.coordinates(),
            in_degrees=False,
            name="MocoTrack coordinates",
        )


class MocoTrack:
    """OpenSim ``MocoTrack``-style coordinate tracking on the collocation core.

    Sets up ``min  w_track * ||q - q_ref||^2 + w_effort * ||u||^2`` subject to the
    model dynamics. The default dynamics are torque-driven
    (:func:`~newton.opensim.create_torque_driven_dynamics`); pass
    a custom ``dynamics`` callable and ``num_controls`` for a muscle-driven model.

    Args:
        model: Parsed OpenSim model.
        device: Warp device for the dynamics kernels (``None`` for the CPU).
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.device = device
        self.id = InverseDynamics(model, device=device)
        self.coordinate_names = list(self.id.coordinate_names)
        self.ncoord = len(self.coordinate_names)

    def solve(
        self,
        reference: Storage | str | os.PathLike,
        tracking_weight: float = 1.0,
        control_effort_weight: float = 0.001,
        control_bounds: tuple[float, float] = (-200.0, 200.0),
        gains: np.ndarray | None = None,
        num_mesh_intervals: int = 25,
        tolerance: float = 1.0e-6,
        max_iterations: int = 200,
        time_range: tuple[float, float] | None = None,
        verbose: bool = False,
    ) -> MocoTrackSolution:
        """Track a reference coordinate trajectory with a torque-driven model.

        Args:
            reference: Reference coordinate trajectory (``.mot`` path or :class:`Storage`).
            tracking_weight: Weight on the squared coordinate tracking error.
            control_effort_weight: Weight on the squared control effort.
            control_bounds: Box bounds on the generalized-force controls.
            gains: Actuator gains mapping controls to generalized forces.
            num_mesh_intervals: Number of Hermite-Simpson mesh intervals.
            tolerance: Collocation solver tolerance.
            max_iterations: Maximum solver iterations.
            time_range: Optional ``(start, end)`` [s] horizon.
            verbose: Print solver diagnostics.
        """
        if not isinstance(reference, Storage):
            reference = read_storage(reference)
        ref_times = np.asarray(reference.times, float)
        col_index = {lab: i for i, lab in enumerate(reference.labels)}
        nc = self.ncoord
        qref = np.zeros((len(ref_times), nc))
        for i, name in enumerate(self.coordinate_names):
            if name in col_index:
                col = reference.data[:, col_index[name]]
                if reference.in_degrees and self.id.motion_types[i] == "rotational":
                    col = np.deg2rad(col)
                qref[:, i] = col

        t0 = ref_times[0] if time_range is None else time_range[0]
        tf = ref_times[-1] if time_range is None else time_range[1]

        def qref_at(t):
            t = np.atleast_1d(t)
            out = np.empty((t.shape[0], nc))
            for i in range(nc):
                out[:, i] = np.interp(t, ref_times, qref[:, i])
            return out

        dynamics = create_torque_driven_dynamics(self.model, gains=gains, device=self.device)
        nu = nc

        w_t = tracking_weight
        w_e = control_effort_weight

        def integral_cost(t, x, u):
            q = x[:, :nc]
            qr = qref_at(t)
            err = q - qr
            return w_t * np.sum(err * err, axis=1) + w_e * np.sum(u * u, axis=1)

        # Fix the initial state to the reference to anchor the trajectory.
        q0 = qref_at(np.array([t0]))[0]
        init = list(q0) + [0.0] * nc

        prob = OptimalControlProblem(
            num_states=2 * nc,
            num_controls=nu,
            dynamics=dynamics,
            initial_state=init,
            final_state=[None] * (2 * nc),
            integral_cost=integral_cost,
            time_initial=float(t0),
            time_final=float(tf),
            control_bounds=control_bounds,
        )
        solver = DirectCollocationSolver(
            num_mesh_intervals=num_mesh_intervals, tolerance=tolerance, max_iterations=max_iterations
        )
        sol = solver.solve(prob, control_guess=lambda t: np.zeros(nu), verbose=verbose)

        qtracked = sol.states[:, :nc]
        qr_nodes = qref_at(sol.time)
        rms = float(np.sqrt(np.mean((qtracked - qr_nodes) ** 2)))
        return MocoTrackSolution(
            times=sol.time,
            coordinate_names=self.coordinate_names,
            states=sol.states,
            controls=sol.controls,
            objective=sol.objective,
            constraint_violation=sol.constraint_violation,
            converged=sol.converged,
            tracking_rms=rms,
        )
