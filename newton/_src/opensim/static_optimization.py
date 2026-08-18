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

r"""Static Optimization: OpenSim's per-frame muscle-force distribution.

Given a measured motion, Static Optimization resolves the muscle-redundancy
problem one time frame at a time: it distributes each frame's net joint moments
(from inverse dynamics) among the muscles that span those coordinates by
minimizing the sum of activations raised to a power, subject to the muscle
force-generating capacity and the moment-balance equations of motion. This
mirrors OpenSim's ``StaticOptimization`` analysis.

For the rigid-tendon De Groote-Fregly muscle the tendon force is *affine* in
activation at a fixed pose and velocity,

.. math::

    F_m(a_m) = a_m\,A_m + P_m,

with an active coefficient :math:`A_m` (the maximum active force scaled by the
force-length and force-velocity multipliers and the pennation cosine) and a
passive offset :math:`P_m`. The per-coordinate moment balance is therefore
linear in the activations,

.. math::

    \sum_m r_{m,c}\,A_m\,a_m + \sum_j \delta_{jc}\,f^{opt}_j\,u_j
        = \tau_c - \sum_m r_{m,c}\,P_m,

where :math:`r_{m,c}` are the muscle moment arms, :math:`\tau_c` the
inverse-dynamics net moments, and :math:`u_j` optional reserve/residual
coordinate-actuator controls (generalized force :math:`f^{opt}_j u_j`) that keep
the problem feasible when the muscles cannot supply a moment. Each frame is a
small bound-constrained quadratic program (for the default exponent 2) solved
with :func:`scipy.optimize.minimize` and warm-started from the previous frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .dynamics import ExternalLoads, InverseDynamics, differentiate_coordinates, read_external_loads
from .mocap import Storage, read_storage, write_storage
from .muscle_force import MuscleForces
from .parser import parse_osim
from .types import OsimModel


@dataclass
class SOResult:
    """Static-optimization result over a set of output frames.

    Attributes:
        times: Output frame times [s], shape ``[num_frames]``.
        muscle_names: Muscle names in column order.
        activations: Muscle activations in [0, 1], shape ``[num_frames, num_muscles]``.
        muscle_forces: Rigid-tendon muscle forces [N], shape ``[num_frames, num_muscles]``.
        coordinate_names: Actuated coordinate names the moment balance was enforced on.
        reserve_names: Reserve/residual coordinate-actuator names (one per actuated coordinate).
        reserve_forces: Reserve generalized forces [N or N·m], shape ``[num_frames, num_coordinates]``.
        moment_residuals: Residual of the moment balance per frame [N or N·m],
            shape ``[num_frames, num_coordinates]`` (near zero when reserves close the balance).
    """

    times: np.ndarray
    muscle_names: list[str]
    activations: np.ndarray
    muscle_forces: np.ndarray
    coordinate_names: list[str]
    reserve_names: list[str]
    reserve_forces: np.ndarray
    moment_residuals: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def to_storage(self) -> Storage:
        """Return the activations as a :class:`~newton.opensim.Storage` (``*_activation`` columns)."""
        labels = [f"{n}_activation" for n in self.muscle_names]
        return Storage(
            times=np.asarray(self.times, float),
            labels=labels,
            data=np.asarray(self.activations, float),
            in_degrees=False,
            name="Static Optimization Activations",
        )

    def forces_storage(self) -> Storage:
        """Return the muscle forces as a :class:`~newton.opensim.Storage` (``*_force`` columns)."""
        labels = [f"{n}_force" for n in self.muscle_names]
        return Storage(
            times=np.asarray(self.times, float),
            labels=labels,
            data=np.asarray(self.muscle_forces, float),
            in_degrees=False,
            name="Static Optimization Forces",
        )

    def write_sto(self, path: str | os.PathLike) -> None:
        """Write the activation trajectory to an OpenSim ``.sto`` file."""
        labels = [f"{n}_activation" for n in self.muscle_names]
        write_storage(
            path,
            np.asarray(self.times, float),
            labels,
            np.asarray(self.activations, float),
            name="Static Optimization Activations",
            in_degrees=False,
        )


def solve_frame_activations(
    R: np.ndarray,
    A: np.ndarray,
    P: np.ndarray,
    tau: np.ndarray,
    activation_exponent: int = 2,
    reserve_optimal_force: float = 1.0,
    reserve_bound: float = 1.0e4,
    use_reserves: bool = True,
    x0: np.ndarray | None = None,
):
    r"""Solve one static-optimization frame: distribute ``tau`` among the muscles.

    Minimizes :math:`\sum_m a_m^p + \sum_c |u_c|^p` subject to the linear
    moment balance :math:`R\,(A\odot a + P) + f^{opt} u = 	au` with
    ``0 <= a_m <= 1`` and reserve controls ``u_c`` bounded by ``reserve_bound``.
    For the rigid-tendon muscle the force is affine in activation
    (:math:`F_m = a_m A_m + P_m`), so with ``activation_exponent == 2`` this is a
    convex quadratic program.

    Args:
        R: Moment arms [m], shape ``[num_coordinates, num_muscles]`` (``R[c, m]``).
        A: Active force coefficient per muscle [N], shape ``[num_muscles]``.
        P: Passive force offset per muscle [N], shape ``[num_muscles]``.
        tau: Net joint moments [N·m or N], shape ``[num_coordinates]``.
        activation_exponent: Objective exponent ``p`` (default 2).
        reserve_optimal_force: Optimal force of the per-coordinate reserve actuators.
        reserve_bound: Bound on the reserve control magnitude.
        use_reserves: Add a reserve/residual actuator on every coordinate. When
            ``False`` the balance is enforced with muscles alone (the problem must
            be feasible), recovering the pure muscle least-effort solution.
        x0: Optional warm-start ``[a; u]`` of length ``num_muscles + num_coordinates``.

    Returns:
        ``(a, u, residual)`` with activations ``a`` in [0, 1], reserve controls
        ``u`` (generalized force ``u * reserve_optimal_force``; all zero when
        ``use_reserves`` is ``False``), and the moment-balance residual.
    """
    import scipy.optimize as _opt

    R = np.atleast_2d(np.asarray(R, float))
    A = np.asarray(A, float).ravel()
    P = np.asarray(P, float).ravel()
    tau = np.asarray(tau, float).ravel()
    nc, nm = R.shape
    nres = nc if use_reserves else 0
    nvar = nm + nres
    p = int(activation_exponent)
    f_opt = float(reserve_optimal_force)

    if x0 is None or len(x0) != nvar:
        x0 = np.concatenate([np.full(nm, 0.05), np.zeros(nres)])
    bounds = [(0.0, 1.0)] * nm + [(-reserve_bound, reserve_bound)] * nres

    def objective(x):
        a = x[:nm]
        u = x[nm:]
        val = float(np.sum(np.abs(a) ** p) + np.sum(np.abs(u) ** p))
        g = np.empty(nvar)
        g[:nm] = p * np.sign(a) * np.abs(a) ** (p - 1)
        g[nm:] = p * np.sign(u) * np.abs(u) ** (p - 1)
        return val, g

    Cmat = np.zeros((nc, nvar))
    Cmat[:, :nm] = R * A[None, :]
    if use_reserves:
        Cmat[np.arange(nc), nm + np.arange(nc)] = f_opt
    d = tau - R @ P
    res = _opt.minimize(
        objective,
        x0,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda x, C=Cmat, dd=d: C @ x - dd, "jac": lambda x, C=Cmat: C}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    x = res.x
    a = np.clip(x[:nm], 0.0, 1.0)
    u = x[nm:] if use_reserves else np.zeros(nc)
    reserve_force = f_opt * u if use_reserves else np.zeros(nc)
    residual = R @ (a * A + P) + reserve_force - tau
    return a, u, residual


class StaticOptimization:
    """OpenSim ``StaticOptimization`` as a per-frame linear muscle-force distribution.

    Args:
        model: Parsed OpenSim model.
        device: Warp device for the muscle/dynamics kernels (``None`` for the CPU).
        activation_exponent: Exponent ``p`` of the ``sum(a**p)`` objective (default 2,
            OpenSim's default).
        reserve_optimal_force: Optimal force [N or N·m] of the reserve/residual
            coordinate actuators added to every actuated coordinate. A reserve
            control ``u`` supplies a generalized force ``u * reserve_optimal_force``
            and is penalized as ``|u|**p``; a smaller optimal force makes the
            reserves a stronger last resort.
    """

    def __init__(
        self,
        model: OsimModel,
        device=None,
        activation_exponent: int = 2,
        reserve_optimal_force: float = 1.0,
        use_reserves: bool = True,
    ):
        self.model = model
        self.device = device
        self.activation_exponent = int(activation_exponent)
        self.reserve_optimal_force = float(reserve_optimal_force)
        self.use_reserves = bool(use_reserves)
        self.muscles = MuscleForces(model, device=device)
        self.id = InverseDynamics(model, device=device)
        self.muscle_names = list(self.muscles.muscle_names)
        self.coordinate_names = list(self.id.coordinate_names)
        self.ncoord = len(self.coordinate_names)
        self.num_muscles = self.muscles.num_muscles

    def _affine_coeffs(self, coords: np.ndarray, speeds: np.ndarray | None):
        """Return ``(A, P)`` affine muscle-force coefficients per frame.

        ``F_m(a) = a * A_m + P_m``; recovered as ``A = F(a=1) - F(a=0)`` and
        ``P = F(a=0)`` from the rigid-tendon muscle-force model at each pose.
        """
        _, active, passive = self.muscles._affine_coefficients(coords, speeds)
        return active, passive

    def solve(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        times: np.ndarray | None = None,
        external_bodies: list[str] | None = None,
        external_wrenches: np.ndarray | None = None,
    ) -> SOResult:
        """Resolve muscle activations frame by frame from kinematics and accelerations.

        Args:
            q: Coordinate values [rad or m], shape ``[num_frames, num_coordinates]``.
            qd: Coordinate speeds, shape ``[num_frames, num_coordinates]``.
            qdd: Coordinate accelerations, shape ``[num_frames, num_coordinates]``.
            times: Frame times [s]; defaults to ``0, 1, ...``.
            external_bodies: Optional body names for applied external wrenches (see
                :meth:`~newton.opensim.InverseDynamics.solve`).
            external_wrenches: Optional external wrenches ``[num_frames, num_bodies, 6]``.

        Returns:
            The per-frame activations, muscle forces, and reserve actuations.
        """
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float64)
        qd = np.ascontiguousarray(np.atleast_2d(qd), dtype=np.float64)
        qdd = np.ascontiguousarray(np.atleast_2d(qdd), dtype=np.float64)
        nframes, nc, nm = q.shape[0], self.ncoord, self.num_muscles
        if times is None:
            times = np.arange(nframes, dtype=float)

        tau = self.id.solve(q, qd, qdd, external_bodies=external_bodies, external_wrenches=external_wrenches)
        arms, A_all, P_all = self.muscles._affine_coefficients(q, qd)  # [nframes, nm, nc], [nframes, nm]

        f_opt = self.reserve_optimal_force
        activations = np.zeros((nframes, nm))
        forces = np.zeros((nframes, nm))
        reserves = np.zeros((nframes, nc))
        residuals = np.zeros((nframes, nc))

        x0 = None
        for k in range(nframes):
            R = arms[k].T  # [nc, nm]: R[c, m] = moment arm of muscle m about coordinate c
            A = A_all[k]
            P = P_all[k]
            a, u, residual = solve_frame_activations(
                R,
                A,
                P,
                tau[k],
                activation_exponent=self.activation_exponent,
                reserve_optimal_force=f_opt,
                use_reserves=self.use_reserves,
                x0=x0,
            )
            activations[k] = a
            forces[k] = a * A + P
            reserves[k] = u * f_opt if self.use_reserves else 0.0
            residuals[k] = residual
            x0 = np.concatenate([a, u]) if self.use_reserves else a  # warm start next frame

        reserve_names = [f"{c}_reserve" for c in self.coordinate_names]
        return SOResult(
            times=np.asarray(times, float),
            muscle_names=self.muscle_names,
            activations=activations,
            muscle_forces=forces,
            coordinate_names=self.coordinate_names,
            reserve_names=reserve_names,
            reserve_forces=reserves,
            moment_residuals=residuals,
        )

    def solve_from_motion(
        self,
        coordinates: Storage | str | os.PathLike,
        external_loads: ExternalLoads | None = None,
        cutoff: float = 6.0,
        time_range: tuple[float, float] | None = None,
        output_times: np.ndarray | None = None,
    ) -> SOResult:
        """Run static optimization from a coordinate motion (OpenSim ``StaticOptimization`` pipeline).

        Reproduces OpenSim's kinematics preprocessing (reflective padding, zero-lag
        Butterworth low-pass, quintic GCVSpline differentiation) shared with
        inverse dynamics, then resolves the muscle activations at each output frame.

        Args:
            coordinates: Coordinate trajectory (``.mot`` path or :class:`Storage`).
            external_loads: Optional applied external loads (e.g. ground reactions).
            cutoff: Butterworth low-pass cutoff [Hz]; ``<= 0`` disables filtering.
            time_range: Optional ``(start, end)`` [s] limiting the output frames.
            output_times: Explicit output times [s]; overrides ``time_range``.
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

        if output_times is None:
            output_times = (
                all_times
                if time_range is None
                else all_times[(all_times >= time_range[0]) & (all_times <= time_range[1])]
            )
        output_times = np.asarray(output_times, float)

        q, qd, qdd = differentiate_coordinates(
            all_times, values, is_rot, output_times=output_times, cutoff=cutoff, in_degrees=coordinates.in_degrees
        )
        bodies, wrenches = (None, None)
        if external_loads is not None:
            bodies, wrenches = external_loads.sample(output_times)
        return self.solve(q, qd, qdd, times=output_times, external_bodies=bodies, external_wrenches=wrenches)


def solve_static_optimization(
    model: OsimModel | str | os.PathLike,
    coordinates: Storage | str | os.PathLike,
    external_loads: ExternalLoads | str | os.PathLike | None = None,
    cutoff: float = 6.0,
    time_range: tuple[float, float] | None = None,
    activation_exponent: int = 2,
    reserve_optimal_force: float = 1.0,
    use_reserves: bool = True,
    device=None,
) -> SOResult:
    """Run static optimization end to end, matching OpenSim's ``StaticOptimization``.

    Args:
        model: A parsed :class:`OsimModel`, or a path/XML string to parse.
        coordinates: Coordinate trajectory (``.mot`` path or :class:`Storage`).
        external_loads: Optional :class:`ExternalLoads`, or a path to an
            ``ExternalLoads`` setup XML.
        cutoff: Butterworth low-pass cutoff [Hz]; ``<= 0`` disables filtering.
        time_range: Optional ``(start, end)`` [s] limiting the output frames.
        activation_exponent: Exponent of the ``sum(a**p)`` objective (default 2).
        reserve_optimal_force: Optimal force of the reserve coordinate actuators.
        device: Warp device for the kernels (``None`` for the CPU).

    Returns:
        The per-frame muscle activations and forces.
    """
    if not isinstance(model, OsimModel):
        model = parse_osim(model)
    if external_loads is not None and not isinstance(external_loads, ExternalLoads):
        external_loads = read_external_loads(external_loads)
    solver = StaticOptimization(
        model,
        device=device,
        activation_exponent=activation_exponent,
        reserve_optimal_force=reserve_optimal_force,
        use_reserves=use_reserves,
    )
    return solver.solve_from_motion(coordinates, external_loads=external_loads, cutoff=cutoff, time_range=time_range)
