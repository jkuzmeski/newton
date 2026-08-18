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

r"""Muscle-driven closed-loop forward simulation.

Integrates the equations of motion forward in time with the joint moments
supplied by Hill-type muscles. The augmented state is :math:`[q, \dot q, a]`
(coordinates, speeds, muscle activations). At each step the neural excitations
:math:`e(t)` come from a controller, the activations advance under first-order
activation dynamics, and the muscle generalized forces
:math:`\tau = r(q)\,F_m(a, q, \dot q)` drive the multibody forward dynamics
(:class:`~newton.opensim.ForwardDynamics`). This is the Newton-native
analogue of an OpenSim muscle-driven ``ForwardTool`` run.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

from .contact import OpenSimContact
from .controllers import ControlSet, PrescribedController
from .dynamics import (
    ExternalLoads,
    ForwardDynamics,
    read_external_loads,
    rk4_stage3_kernel,
    rk4_stage_kernel,
    rk4_update_kernel,
    semi_implicit_update_kernel,
)
from .mocap import Storage, write_storage
from .muscle_force import MuscleForces, activation_integrate_kernel, generalized_force_kernel
from .parser import parse_osim
from .types import OsimModel

_f64 = wp.float64
_f32 = wp.float32


@wp.kernel
def record_muscle_state_kernel(
    coords: wp.array2d[_f64],
    speeds: wp.array2d[_f64],
    activations: wp.array2d[_f32],
    excitations: wp.array2d[_f32],
    forces: wp.array2d[_f32],
    sample: wp.array[wp.int32],
    ncoord: int,
    nmuscle: int,
    out_mechanical: wp.array2d[_f64],
    out_muscle: wp.array2d[_f32],
):
    """Record one device-resident muscle simulation state."""
    i = wp.tid()
    s = sample[0]
    if i < ncoord:
        out_mechanical[s, i] = coords[0, i]
        out_mechanical[s, ncoord + i] = speeds[0, i]
    if i < nmuscle:
        out_muscle[s, i] = activations[0, i]
        out_muscle[s, nmuscle + i] = excitations[0, i]
        out_muscle[s, 2 * nmuscle + i] = forces[0, i]


@wp.kernel
def increment_sample_kernel(sample: wp.array[wp.int32]):
    """Advance the device trajectory sample index after recording."""
    sample[0] = sample[0] + 1


@wp.kernel
def load_excitation_kernel(
    excitation_history: wp.array2d[_f32], sample: wp.array[wp.int32], excitation: wp.array2d[_f32]
):
    """Load the current sample's presampled excitation without a host callback."""
    m = wp.tid()
    excitation[0, m] = excitation_history[sample[0], m]


@dataclass
class MuscleForwardResult:
    """Trajectory of a muscle-driven forward simulation.

    Attributes:
        times: Sample times [s], shape ``[num_steps + 1]``.
        coordinate_names: Coordinate names in column order.
        coordinates: Coordinate trajectory [m or rad], shape ``[num_steps + 1, num_coordinates]``.
        speeds: Coordinate speeds [m/s or rad/s], same shape as ``coordinates``.
        motion_types: ``"rotational"`` / ``"translational"`` per coordinate.
        muscle_names: Muscle names in column order.
        excitations: Applied neural excitations in [0, 1], shape ``[num_steps + 1, num_muscles]``.
        activations: Muscle activations in [0, 1], same shape as ``excitations``.
        muscle_forces: Rigid-tendon muscle forces [N], same shape as ``excitations``.
    """

    times: np.ndarray
    coordinate_names: list[str]
    coordinates: np.ndarray
    speeds: np.ndarray
    motion_types: list[str]
    muscle_names: list[str]
    excitations: np.ndarray
    activations: np.ndarray
    muscle_forces: np.ndarray

    def to_storage(self) -> Storage:
        """Return the coordinate trajectory as a :class:`~newton.opensim.Storage` [rad]."""
        return Storage(
            times=np.asarray(self.times, float),
            labels=list(self.coordinate_names),
            data=np.asarray(self.coordinates, float),
            in_degrees=False,
            name="Muscle-driven forward coordinates",
        )

    def write_sto(self, path: str | os.PathLike) -> None:
        """Write the coordinate trajectory to an OpenSim ``.sto`` file."""
        write_storage(
            path,
            np.asarray(self.times, float),
            list(self.coordinate_names),
            np.asarray(self.coordinates, float),
            name="Muscle-driven forward coordinates",
            in_degrees=False,
        )


class MuscleDrivenForward:
    """Muscle-driven forward dynamics on the augmented state :math:`[q, \\dot q, a]`.

    Args:
        model: Parsed OpenSim model with muscles.
        device: Warp device for the muscle/dynamics kernels (``None`` for the CPU).
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.device = device
        self.fd = ForwardDynamics(model, device=device)
        self.muscles = MuscleForces(model, device=device)
        self.coordinate_names = list(self.fd.coordinate_names)
        self.muscle_names = list(self.muscles.muscle_names)
        self.ncoord = len(self.coordinate_names)
        self.num_muscles = self.muscles.num_muscles
        self.motion_types = list(getattr(self.fd, "motion_types", ["rotational"] * self.ncoord))

    def _excitations_fn(self, excitations):
        """Return a callable ``t -> [num_muscles]`` from the supplied excitation source."""
        if isinstance(excitations, PrescribedController):
            return lambda t: np.clip(np.asarray(excitations(t), float).ravel(), 0.0, 1.0)
        if isinstance(excitations, ControlSet):
            ctrl = PrescribedController(self.muscle_names, excitations)
            return lambda t: np.clip(ctrl(t), 0.0, 1.0)
        if callable(excitations):
            return lambda t: np.clip(np.asarray(excitations(t), float).ravel(), 0.0, 1.0)
        arr = np.clip(np.asarray(excitations, float).ravel(), 0.0, 1.0)
        return lambda t: arr

    def _simulate_cuda_graph(
        self,
        initial_coordinates: np.ndarray,
        initial_speeds: np.ndarray,
        excitation_history: np.ndarray,
        initial_activations: np.ndarray,
        duration: float,
        dt: float,
        start_time: float,
        integrator: str,
        activation_substeps: int,
    ) -> MuscleForwardResult:
        """Run a presampled-excitation trajectory entirely in a CUDA graph."""
        dev = self.muscles.device
        nc = self.ncoord
        nm = self.num_muscles
        num_steps = int(round(duration / dt))
        q0 = np.asarray(initial_coordinates, dtype=np.float64).reshape(1, nc)
        v0 = np.asarray(initial_speeds, dtype=np.float64).reshape(1, nc)
        a0 = np.asarray(initial_activations, dtype=np.float32).reshape(1, nm)
        excitation_history = np.ascontiguousarray(excitation_history, dtype=np.float32).reshape(num_steps + 1, nm)
        e0 = excitation_history[:1]
        q = wp.array(q0, dtype=_f64, device=dev)
        v = wp.array(v0, dtype=_f64, device=dev)
        activation = wp.array(a0, dtype=_f32, device=dev)
        excitation_wp = wp.array(e0, dtype=_f32, device=dev)
        excitation_history_wp = wp.array(excitation_history, dtype=_f32, device=dev)
        stage_q = wp.empty((1, nc), dtype=_f64, device=dev)
        stage_v = wp.empty((1, nc), dtype=_f64, device=dev)
        stage_a = [wp.empty((1, nc), dtype=_f64, device=dev) for _ in range(4)]
        zero_a = wp.zeros((1, nc), dtype=_f64, device=dev)
        fd_workspace = self.fd._create_device_workspace(1)
        sample = wp.zeros(1, dtype=wp.int32, device=dev)
        out_mechanical = wp.empty((num_steps + 1, 2 * nc), dtype=_f64, device=dev)
        out_muscle = wp.empty((num_steps + 1, 3 * nm), dtype=_f32, device=dev)

        def muscle_acceleration(qc, vc, out_accel):
            moment_arms = self.muscles.paths._moment_arms_device(qc, 1.0e-5)
            muscle_force = self.muscles._forces_device(activation, qc, vc, 1.0e-5, moment_arms=moment_arms)
            tau = wp.empty((1, nc), dtype=_f64, device=dev)
            wp.launch(
                generalized_force_kernel,
                dim=(1, nc),
                inputs=[moment_arms, muscle_force, nm, tau],
                device=dev,
            )
            self.fd._accelerations_device(qc, vc, tau, out_accel, fd_workspace)

        def record(force):
            wp.launch(
                record_muscle_state_kernel,
                dim=max(nc, nm),
                inputs=[
                    q,
                    v,
                    activation,
                    excitation_wp,
                    force,
                    sample,
                    nc,
                    nm,
                    out_mechanical,
                    out_muscle,
                ],
                device=dev,
            )
            wp.launch(increment_sample_kernel, dim=1, inputs=[sample], device=dev)

        def step():
            if integrator == "semi_implicit":
                muscle_acceleration(q, v, stage_a[0])
                wp.launch(
                    semi_implicit_update_kernel,
                    dim=(1, nc),
                    inputs=[q, v, stage_a[0], _f64(dt)],
                    device=dev,
                )
            else:
                wp.launch(
                    rk4_stage_kernel,
                    dim=(1, nc),
                    inputs=[q, v, zero_a, _f64(0.0), _f64(0.0), stage_q, stage_v],
                    device=dev,
                )
                muscle_acceleration(stage_q, stage_v, stage_a[0])
                wp.launch(
                    rk4_stage_kernel,
                    dim=(1, nc),
                    inputs=[q, v, stage_a[0], _f64(0.5 * dt), _f64(0.5 * dt), stage_q, stage_v],
                    device=dev,
                )
                muscle_acceleration(stage_q, stage_v, stage_a[1])
                wp.launch(
                    rk4_stage3_kernel,
                    dim=(1, nc),
                    inputs=[
                        q,
                        v,
                        stage_a[0],
                        stage_a[1],
                        _f64(0.5 * dt),
                        _f64(0.5 * dt),
                        _f64(0.5 * dt),
                        stage_q,
                        stage_v,
                    ],
                    device=dev,
                )
                muscle_acceleration(stage_q, stage_v, stage_a[2])
                wp.launch(
                    rk4_stage3_kernel,
                    dim=(1, nc),
                    inputs=[
                        q,
                        v,
                        stage_a[1],
                        stage_a[2],
                        _f64(dt),
                        _f64(0.5 * dt),
                        _f64(dt),
                        stage_q,
                        stage_v,
                    ],
                    device=dev,
                )
                muscle_acceleration(stage_q, stage_v, stage_a[3])
                wp.launch(
                    rk4_update_kernel,
                    dim=(1, nc),
                    inputs=[q, v, stage_a[0], stage_a[1], stage_a[2], stage_a[3], _f64(dt)],
                    device=dev,
                )
            wp.launch(
                activation_integrate_kernel,
                dim=(1, nm),
                inputs=[
                    activation,
                    excitation_wp,
                    self.muscles.d_tau_act,
                    self.muscles.d_tau_deact,
                    _f32(dt),
                    int(activation_substeps),
                    activation,
                ],
                device=dev,
            )
            wp.launch(
                load_excitation_kernel,
                dim=nm,
                inputs=[excitation_history_wp, sample, excitation_wp],
                device=dev,
            )
            force = self.muscles._forces_device(activation, q, v)
            record(force)

        # Warm every kernel before capture, then restore the initial state.
        step()
        q.assign(q0)
        v.assign(v0)
        activation.assign(a0)
        excitation_wp.assign(e0)
        sample.zero_()
        initial_force = self.muscles._forces_device(activation, q, v)
        record(initial_force)
        with wp.ScopedCapture(device=dev) as capture:
            step()
        for _ in range(num_steps):
            wp.capture_launch(capture.graph)

        mechanical = out_mechanical.numpy()
        muscle = out_muscle.numpy().astype(np.float64)
        return MuscleForwardResult(
            times=start_time + np.arange(num_steps + 1) * dt,
            coordinate_names=self.coordinate_names,
            coordinates=mechanical[:, :nc],
            speeds=mechanical[:, nc:],
            motion_types=self.motion_types,
            muscle_names=self.muscle_names,
            excitations=muscle[:, nm : 2 * nm],
            activations=muscle[:, :nm],
            muscle_forces=muscle[:, 2 * nm :],
        )

    def simulate(
        self,
        initial_coordinates: np.ndarray,
        initial_speeds: np.ndarray,
        excitations,
        duration: float,
        dt: float,
        initial_activations: np.ndarray | None = None,
        start_time: float = 0.0,
        coordinate_controls: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
        external_loads: ExternalLoads | None = None,
        contact: OpenSimContact | None = None,
        integrator: str = "rk4",
        activation_substeps: int = 8,
        use_graph: bool = True,
    ) -> MuscleForwardResult:
        """Integrate the muscle-driven equations of motion forward in time.

        Args:
            initial_coordinates: Initial coordinate values, shape ``[num_coordinates]``.
            initial_speeds: Initial coordinate speeds, shape ``[num_coordinates]``.
            excitations: Neural excitations: a :class:`PrescribedController`, a
                :class:`ControlSet`, a ``callable(t) -> [num_muscles]``, or a constant
                array aligned to the model's muscle order.
            duration: Length of the simulation [s].
            dt: Fixed integration step [s].
            initial_activations: Initial muscle activations, shape ``[num_muscles]``
                (defaults to the excitation at ``start_time``).
            start_time: Time of the initial state [s].
            coordinate_controls: Optional extra generalized forces
                ``tau(t, q, qd) -> [num_coordinates]`` (e.g. non-muscle actuators).
            external_loads: Optional :class:`ExternalLoads` sampled at each step.
            contact: Optional :class:`~newton.opensim.OpenSimContact`. When given, its
                generalized contact forces are evaluated from the current state each
                substep and added to the muscle/actuator moments, closing the
                foot-ground contact loop inside the forward integration.
            integrator: ``"rk4"`` (Runge-Kutta 4 on the mechanical state, activation
                advanced by operator splitting) or ``"semi_implicit"`` (symplectic Euler).
            activation_substeps: RK4 substeps used to advance the activation dynamics.
            use_graph: Presample excitations, then capture and replay the device-resident
                step when no extra controls or loads are used on a CUDA device.
        """
        exc_fn = self._excitations_fn(excitations)
        nc = self.ncoord
        nm = self.num_muscles
        num_steps = int(round(duration / dt))

        q = np.asarray(initial_coordinates, float).ravel().copy()
        qd = np.asarray(initial_speeds, float).ravel().copy()
        e0 = exc_fn(start_time)
        a = e0.copy() if initial_activations is None else np.asarray(initial_activations, float).ravel().copy()
        if (
            use_graph
            and coordinate_controls is None
            and external_loads is None
            and contact is None
            and integrator in ("rk4", "semi_implicit")
            and self.muscles.device.is_cuda
        ):
            excitation_history = np.empty((num_steps + 1, nm))
            excitation_history[0] = e0
            for k in range(1, num_steps + 1):
                excitation_history[k] = exc_fn(start_time + k * dt)
            return self._simulate_cuda_graph(
                q,
                qd,
                excitation_history,
                a,
                duration,
                dt,
                start_time,
                integrator,
                activation_substeps,
            )

        times = np.empty(num_steps + 1)
        qs = np.empty((num_steps + 1, nc))
        qds = np.empty((num_steps + 1, nc))
        acts = np.empty((num_steps + 1, nm))
        excs = np.empty((num_steps + 1, nm))
        forces = np.empty((num_steps + 1, nm))

        def tau_of(t, qc, qv, av):
            gen = self.muscles.generalized_forces(av[None, :], qc[None, :], qv[None, :])[0]
            if coordinate_controls is not None:
                gen = gen + np.asarray(coordinate_controls(t, qc, qv), float).ravel()
            if contact is not None:
                gen = gen + np.asarray(contact.generalized_forces(qc, qv), float).ravel()
            return gen

        def accel(t, qc, qv, av):
            bodies, wrenches = (None, None)
            if external_loads is not None:
                b, w = external_loads.sample(np.array([t]))
                bodies, wrenches = b, w[0] if w is not None else None
            tau = tau_of(t, qc, qv, av)
            return self.fd.accelerations(
                qc[None, :], qv[None, :], tau[None, :], external_bodies=bodies, external_wrenches=wrenches
            )[0]

        t = start_time
        for k in range(num_steps + 1):
            e = exc_fn(t)
            f = self.muscles.forces(a[None, :], q[None, :], qd[None, :])[0]
            times[k] = t
            qs[k] = q
            qds[k] = qd
            acts[k] = a
            excs[k] = e
            forces[k] = f
            if k == num_steps:
                break
            # Advance activation over the step (operator splitting; excitation held).
            a_next = self.muscles.integrate_activation(a[None, :], e[None, :], dt, substeps=activation_substeps)[0]
            if integrator == "semi_implicit":
                qdd = accel(t, q, qd, a)
                qd = qd + dt * qdd
                q = q + dt * qd
            else:  # rk4 on the mechanical state, activation frozen at a within the step
                k1q, k1v = qd, accel(t, q, qd, a)
                k2q, k2v = qd + 0.5 * dt * k1v, accel(t + 0.5 * dt, q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v, a)
                k3q, k3v = qd + 0.5 * dt * k2v, accel(t + 0.5 * dt, q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v, a)
                k4q, k4v = qd + dt * k3v, accel(t + dt, q + dt * k3q, qd + dt * k3v, a)
                q = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
                qd = qd + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            a = a_next
            t = start_time + (k + 1) * dt

        return MuscleForwardResult(
            times=times,
            coordinate_names=self.coordinate_names,
            coordinates=qs,
            speeds=qds,
            motion_types=self.motion_types,
            muscle_names=self.muscle_names,
            excitations=excs,
            activations=acts,
            muscle_forces=forces,
        )


def simulate_muscle_driven(
    model: OsimModel | str | os.PathLike,
    excitations,
    initial_coordinates: np.ndarray,
    initial_speeds: np.ndarray,
    duration: float,
    dt: float,
    initial_activations: np.ndarray | None = None,
    start_time: float = 0.0,
    coordinate_controls: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
    external_loads: ExternalLoads | str | os.PathLike | None = None,
    contact: OpenSimContact | bool | None = None,
    integrator: str = "rk4",
    device=None,
    use_graph: bool = True,
) -> MuscleForwardResult:
    """Run a muscle-driven forward simulation end to end.

    Args:
        model: A parsed :class:`OsimModel`, or a path/XML string to parse.
        excitations: Neural excitations (:class:`PrescribedController`,
            :class:`ControlSet`, ``callable(t)``, or constant array).
        initial_coordinates: Initial coordinate values, shape ``[num_coordinates]``.
        initial_speeds: Initial coordinate speeds, shape ``[num_coordinates]``.
        duration: Length of the simulation [s].
        dt: Fixed integration step [s].
        initial_activations: Initial muscle activations (defaults to the excitation
            at ``start_time``).
        start_time: Time of the initial state [s].
        coordinate_controls: Optional extra generalized forces ``tau(t, q, qd)``.
        external_loads: Optional :class:`ExternalLoads`, or a path to a setup XML.
        contact: An :class:`~newton.opensim.OpenSimContact` to close the contact
            loop, or ``True`` to build one from ``model`` automatically.
        integrator: ``"rk4"`` or ``"semi_implicit"``.
        device: Warp device for the kernels (``None`` for the CPU).
        use_graph: Capture and replay a supported CUDA simulation step.
    """
    if not isinstance(model, OsimModel):
        model = parse_osim(model)
    if external_loads is not None and not isinstance(external_loads, ExternalLoads):
        external_loads = read_external_loads(external_loads)
    if contact is True:
        contact = OpenSimContact(model, device=device)
    elif contact is False:
        contact = None
    sim = MuscleDrivenForward(model, device=device)
    return sim.simulate(
        initial_coordinates,
        initial_speeds,
        excitations,
        duration,
        dt,
        initial_activations=initial_activations,
        start_time=start_time,
        coordinate_controls=coordinate_controls,
        external_loads=external_loads,
        contact=contact,
        integrator=integrator,
        use_graph=use_graph,
    )
