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

"""Warp-native non-muscle actuator generalized forces.

OpenSim ``CoordinateActuator`` (and ``ActivationCoordinateActuator``) apply a
generalized force directly to a single coordinate: :math:`\tau_i = f^{opt}\\,
\\mathrm{clamp}(u,\\,u_{min},\\,u_{max})` for control :math:`u` and optimal force
:math:`f^{opt}`. These torque/reserve actuators are common in torque-driven and
Moco models; combined with :class:`~newton.opensim.MuscleForces` generalized
forces they give the full generalized force applied to the skeleton.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .kinematics import ForwardKinematics
from .types import OsimModel

_f64 = wp.float64
_vec3d = wp.vec3d
_mat44d = wp.mat44d

# CoordinateActuator variants that apply an optimal-force-scaled control to one coordinate.
_COORDINATE_ACTUATOR_TYPES = ("CoordinateActuator", "ActivationCoordinateActuator")


@wp.kernel
def coordinate_actuator_kernel(
    controls: wp.array2d[_f64],
    coord: wp.array[wp.int32],
    gain: wp.array[_f64],
    lo: wp.array[_f64],
    hi: wp.array[_f64],
    tau: wp.array2d[_f64],
):
    """Scatter each coordinate actuator's ``optimal_force * clamp(control)`` onto its coordinate.

    Launched with dim ``(batch, num_actuators)``; actuators sharing a coordinate
    accumulate through an atomic add.
    """
    b, j = wp.tid()
    c = wp.clamp(controls[b, j], lo[j], hi[j])
    wp.atomic_add(tau, b, coord[j], gain[j] * c)


class CoordinateActuators:
    """Warp-native generalized forces from OpenSim ``CoordinateActuator`` s.

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernel (``"cpu"``, ``"cuda"``, a
            :class:`warp.context.Device`, or ``None`` for the CPU).

    Attributes:
        coordinate_names: Generalized coordinate names in model order.
        actuator_names: Coordinate-actuator names in control-column order.
        num_actuators: Number of coordinate actuators.
        device: The Warp device the kernel runs on.
    """

    def __init__(self, model: OsimModel, device=None):
        self.device = wp.get_device(device) if device is not None else wp.get_device("cpu")
        self.coordinate_names: list[str] = [c.name for j in model.joints for c in j.coordinates]
        self.ncoord = len(self.coordinate_names)
        index = {name: i for i, name in enumerate(self.coordinate_names)}
        acts = [a for a in model.actuators if a.type in _COORDINATE_ACTUATOR_TYPES and a.coordinate in index]
        self.actuator_names: list[str] = [a.name for a in acts]
        self.num_actuators = len(acts)
        coord = np.array([index[a.coordinate] for a in acts], np.int32)
        gain = np.array([a.optimal_force for a in acts], np.float64)
        # Unbounded controls (+/- inf) map to a wide finite clamp so the kernel is a no-op clamp.
        lo = np.array([max(a.min_control, -1.0e30) for a in acts], np.float64)
        hi = np.array([min(a.max_control, 1.0e30) for a in acts], np.float64)
        self._coord, self._gain, self._lo, self._hi = coord, gain, lo, hi
        self.d_coord = wp.array(coord, dtype=wp.int32, device=self.device)
        self.d_gain = wp.array(gain, dtype=_f64, device=self.device)
        self.d_lo = wp.array(lo, dtype=_f64, device=self.device)
        self.d_hi = wp.array(hi, dtype=_f64, device=self.device)

    def generalized_forces(self, controls: np.ndarray) -> np.ndarray:
        """Return coordinate-actuator generalized forces [N or N·m], shape ``[batch, num_coordinates]``.

        Args:
            controls: Actuator controls, shape ``[batch, num_actuators]`` or
                ``[num_actuators]`` (broadcast to a single-configuration batch).
                Column order is :attr:`actuator_names`.

        Returns:
            Generalized forces per coordinate [batch, num_coordinates], summed over
            actuators sharing a coordinate; column order :attr:`coordinate_names`.
        """
        controls = np.ascontiguousarray(np.atleast_2d(controls), dtype=np.float64)
        if controls.shape[1] != self.num_actuators:
            raise ValueError(f"controls has {controls.shape[1]} columns, expected {self.num_actuators}")
        batch = controls.shape[0]
        if not self.num_actuators:
            return np.zeros((batch, self.ncoord))
        tau = wp.zeros((batch, self.ncoord), dtype=_f64, device=self.device)
        if self.num_actuators:
            wp.launch(
                coordinate_actuator_kernel,
                dim=(batch, self.num_actuators),
                inputs=[
                    wp.array(controls, dtype=_f64, device=self.device),
                    self.d_coord,
                    self.d_gain,
                    self.d_lo,
                    self.d_hi,
                    tau,
                ],
                device=self.device,
            )
        return tau.numpy()


# PointActuator applies a force at a body point; TorqueActuator a pure torque between two bodies.
_SPATIAL_ACTUATOR_TYPES = ("PointActuator", "TorqueActuator")


@wp.kernel
def spatial_actuator_kernel(
    jac: wp.array4d[_f64],
    body_X: wp.array2d[_mat44d],
    controls: wp.array2d[_f64],
    body_idx: wp.array[wp.int32],
    act_col: wp.array[wp.int32],
    kind: wp.array[wp.int32],
    is_global: wp.array[wp.int32],
    point_global: wp.array[wp.int32],
    sign: wp.array[_f64],
    gain: wp.array[_f64],
    point: wp.array[_vec3d],
    direction: wp.array[_vec3d],
    n_app: int,
    tau: wp.array2d[_f64],
):
    """Accumulate spatial-actuator generalized forces by transposed-Jacobian projection.

    Each application ``app`` builds a ground-frame wrench about its body origin from
    the control and the (optionally body-frame) direction and point, then contracts
    it with that body's spatial Jacobian column ``i``: rows 0-2 with the torque and
    rows 3-5 with the force. ``kind`` 0 is a point force, 1 a pure torque.
    """
    b, i = wp.tid()
    acc = _f64(0.0)
    for app in range(n_app):
        body = body_idx[app]
        x = body_X[b, body]
        rot = wp.mat33d(
            x[0, 0],
            x[0, 1],
            x[0, 2],
            x[1, 0],
            x[1, 1],
            x[1, 2],
            x[2, 0],
            x[2, 1],
            x[2, 2],
        )
        d = direction[app]
        if is_global[app] == 1:
            dir_g = d
        else:
            dir_g = wp.mul(rot, d)
        mag = sign[app] * gain[app] * controls[b, act_col[app]]
        torque = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
        force = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
        if kind[app] == 0:
            force = mag * dir_g
            if point_global[app] == 1:
                r = point[app] - _vec3d(x[0, 3], x[1, 3], x[2, 3])
            else:
                r = wp.mul(rot, point[app])
            torque = wp.cross(r, force)
        else:
            torque = mag * dir_g
        acc += torque[0] * jac[b, body, 0, i] + torque[1] * jac[b, body, 1, i] + torque[2] * jac[b, body, 2, i]
        acc += force[0] * jac[b, body, 3, i] + force[1] * jac[b, body, 4, i] + force[2] * jac[b, body, 5, i]
    tau[b, i] = acc


class SpatialActuators:
    """Warp-native generalized forces from OpenSim ``PointActuator`` and ``TorqueActuator``.

    A ``PointActuator`` applies ``optimal_force * control`` along its direction at a
    point on a body; a ``TorqueActuator`` applies ``optimal_force * control`` about
    its axis to ``bodyA`` and the reaction to ``bodyB``. Each is reduced to a
    ground-frame wrench and projected through the body spatial Jacobian to
    generalized forces (see
    :meth:`~newton.opensim.ForwardKinematics.generalized_forces_from_body_load`).

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernels (``None`` selects the CPU).

    Attributes:
        fk: The :class:`~newton.opensim.ForwardKinematics` backing the Jacobians.
        coordinate_names: Generalized coordinate names in model order.
        actuator_names: Spatial-actuator names in control-column order.
        num_actuators: Number of spatial actuators.
    """

    def __init__(self, model: OsimModel, device=None):
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.coordinate_names = self.fk.coordinate_names
        self.ncoord = self.fk.ncoord
        acts = [a for a in model.actuators if a.type in _SPATIAL_ACTUATOR_TYPES]
        self.actuator_names = [a.name for a in acts]
        self.num_actuators = len(acts)
        names = self.fk.body_names

        def _bidx(name):
            return names.index(name) if name in names else 0

        body_idx, act_col, kind, is_global, point_global = [], [], [], [], []
        sign, gain, point, direction = [], [], [], []

        def _add(bi, col, kd, pg, s, g, pt, di):
            body_idx.append(bi)
            act_col.append(col)
            kind.append(kd)
            is_global.append(pg[0])
            point_global.append(pg[1])
            sign.append(s)
            gain.append(g)
            point.append(pt)
            direction.append(di)

        for col, a in enumerate(acts):
            fg = int(a.force_is_global)
            if a.type == "PointActuator":
                _add(
                    _bidx(a.body or "ground"),
                    col,
                    0,
                    (fg, int(a.point_is_global)),
                    1.0,
                    a.optimal_force,
                    a.point,
                    a.direction,
                )
            else:  # TorqueActuator: torque on bodyA, reaction on bodyB.
                _add(_bidx(a.body or "ground"), col, 1, (fg, 0), 1.0, a.optimal_force, (0.0, 0.0, 0.0), a.direction)
                _add(_bidx(a.body_b or "ground"), col, 1, (fg, 0), -1.0, a.optimal_force, (0.0, 0.0, 0.0), a.direction)

        self.num_apps = len(body_idx)
        self.d_body_idx = wp.array(np.array(body_idx, np.int32), dtype=wp.int32, device=self.device)
        self.d_act_col = wp.array(np.array(act_col, np.int32), dtype=wp.int32, device=self.device)
        self.d_kind = wp.array(np.array(kind, np.int32), dtype=wp.int32, device=self.device)
        self.d_is_global = wp.array(np.array(is_global, np.int32), dtype=wp.int32, device=self.device)
        self.d_point_global = wp.array(np.array(point_global, np.int32), dtype=wp.int32, device=self.device)
        self.d_sign = wp.array(np.array(sign, np.float64), dtype=_f64, device=self.device)
        self.d_gain = wp.array(np.array(gain, np.float64), dtype=_f64, device=self.device)
        self.d_point = wp.array(np.array(point, np.float64).reshape(-1, 3), dtype=_vec3d, device=self.device)
        self.d_direction = wp.array(np.array(direction, np.float64).reshape(-1, 3), dtype=_vec3d, device=self.device)

    def generalized_forces(self, coords: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Return spatial-actuator generalized forces [batch, num_coordinates].

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            controls: Actuator controls, shape ``[batch, num_actuators]`` or
                ``[num_actuators]`` (broadcast). Column order :attr:`actuator_names`.

        Returns:
            Generalized forces [batch, num_coordinates] in :attr:`coordinate_names`
            order, summed over all actuators.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        batch = coords.shape[0]
        if not self.num_apps or not self.num_actuators:
            return np.zeros((batch, self.ncoord))
        tau = wp.empty((batch, self.ncoord), dtype=_f64, device=self.device)
        if self.num_apps and self.num_actuators:
            controls = np.ascontiguousarray(np.atleast_2d(controls), dtype=np.float64)
            if controls.shape[1] != self.num_actuators:
                raise ValueError(f"controls has {controls.shape[1]} columns, expected {self.num_actuators}")
            if controls.shape[0] == 1 and batch > 1:
                controls = np.repeat(controls, batch, axis=0)
            q_wp = wp.array(coords, dtype=_f64, device=self.device)
            jac, body_x = self.fk._launch_body_jacobian_device(q_wp, 1.0e-6)
            wp.launch(
                spatial_actuator_kernel,
                dim=(batch, self.ncoord),
                inputs=[
                    jac,
                    body_x,
                    wp.array(controls, dtype=_f64, device=self.device),
                    self.d_body_idx,
                    self.d_act_col,
                    self.d_kind,
                    self.d_is_global,
                    self.d_point_global,
                    self.d_sign,
                    self.d_gain,
                    self.d_point,
                    self.d_direction,
                    self.num_apps,
                    tau,
                ],
                device=self.device,
            )
        return tau.numpy()


# BodyActuator applies a full 6-DOF spatial force (torque, force) to a body.
_BODY_ACTUATOR_TYPES = ("BodyActuator",)


@wp.kernel
def body_actuator_kernel(
    jac: wp.array4d[_f64],
    body_X: wp.array2d[_mat44d],
    controls: wp.array3d[_f64],
    body_idx: wp.array[wp.int32],
    is_global: wp.array[wp.int32],
    point_global: wp.array[wp.int32],
    gain: wp.array[_f64],
    point: wp.array[_vec3d],
    n_act: int,
    tau: wp.array2d[_f64],
):
    """Project each body actuator's spatial control ``[torque, force]`` to generalized forces.

    ``controls[b, a]`` is the 6-vector spatial force of actuator ``a``; scaled by
    ``gain`` and (unless ``is_global``) rotated from the body frame, it is applied
    at ``point`` on the body, reduced to a wrench about the body origin
    (``torque + r x force``), and contracted with the body Jacobian column ``i``.
    """
    b, i = wp.tid()
    acc = _f64(0.0)
    for a in range(n_act):
        body = body_idx[a]
        x = body_X[b, body]
        rot = wp.mat33d(
            x[0, 0],
            x[0, 1],
            x[0, 2],
            x[1, 0],
            x[1, 1],
            x[1, 2],
            x[2, 0],
            x[2, 1],
            x[2, 2],
        )
        scale = gain[a]
        tq_c = _vec3d(controls[b, a, 0], controls[b, a, 1], controls[b, a, 2])
        fc_c = _vec3d(controls[b, a, 3], controls[b, a, 4], controls[b, a, 5])
        if is_global[a] == 1:
            torque = scale * tq_c
            force = scale * fc_c
        else:
            torque = scale * wp.mul(rot, tq_c)
            force = scale * wp.mul(rot, fc_c)
        if point_global[a] == 1:
            r = point[a] - _vec3d(x[0, 3], x[1, 3], x[2, 3])
        else:
            r = wp.mul(rot, point[a])
        torque_o = torque + wp.cross(r, force)
        acc += torque_o[0] * jac[b, body, 0, i] + torque_o[1] * jac[b, body, 1, i] + torque_o[2] * jac[b, body, 2, i]
        acc += force[0] * jac[b, body, 3, i] + force[1] * jac[b, body, 4, i] + force[2] * jac[b, body, 5, i]
    tau[b, i] = acc


class BodyActuators:
    """Warp-native generalized forces from OpenSim ``BodyActuator`` s.

    A ``BodyActuator`` applies a 6-DOF spatial force ``optimal_force * control``
    (``control`` ordered ``[torque, force]``) at a point on a body. Each is reduced
    to a ground-frame wrench about the body origin and projected through the body
    spatial Jacobian to generalized forces.

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernels (``None`` selects the CPU).

    Attributes:
        fk: The :class:`~newton.opensim.ForwardKinematics` backing the Jacobians.
        coordinate_names: Generalized coordinate names in model order.
        actuator_names: Body-actuator names in control order.
        num_actuators: Number of body actuators.
    """

    def __init__(self, model: OsimModel, device=None):
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.coordinate_names = self.fk.coordinate_names
        self.ncoord = self.fk.ncoord
        acts = [a for a in model.actuators if a.type in _BODY_ACTUATOR_TYPES]
        self.actuator_names = [a.name for a in acts]
        self.num_actuators = len(acts)
        names = self.fk.body_names
        body_idx = [names.index(a.body) if a.body in names else 0 for a in acts]
        is_global = [int(a.force_is_global) for a in acts]
        point_global = [int(a.point_is_global) for a in acts]
        gain = [a.optimal_force for a in acts]
        point = [a.point for a in acts] if acts else []
        self.d_body_idx = wp.array(np.array(body_idx, np.int32), dtype=wp.int32, device=self.device)
        self.d_is_global = wp.array(np.array(is_global, np.int32), dtype=wp.int32, device=self.device)
        self.d_point_global = wp.array(np.array(point_global, np.int32), dtype=wp.int32, device=self.device)
        self.d_gain = wp.array(np.array(gain, np.float64), dtype=_f64, device=self.device)
        self.d_point = wp.array(np.array(point, np.float64).reshape(-1, 3), dtype=_vec3d, device=self.device)

    def generalized_forces(self, coords: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Return body-actuator generalized forces [batch, num_coordinates].

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            controls: Spatial controls ``[batch, num_actuators, 6]`` ordered
                ``[torque, force]`` per actuator; ``[num_actuators, 6]`` is
                broadcast over the batch.

        Returns:
            Generalized forces [batch, num_coordinates] in :attr:`coordinate_names`
            order, summed over all actuators.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        batch = coords.shape[0]
        if not self.num_actuators:
            return np.zeros((batch, self.ncoord))
        tau = wp.empty((batch, self.ncoord), dtype=_f64, device=self.device)
        if self.num_actuators:
            controls = np.ascontiguousarray(controls, dtype=np.float64)
            if controls.ndim == 2:
                controls = controls[None]
            if controls.shape[1:] != (self.num_actuators, 6):
                raise ValueError(f"controls must be [batch, {self.num_actuators}, 6]")
            if controls.shape[0] == 1 and batch > 1:
                controls = np.repeat(controls, batch, axis=0)
            q_wp = wp.array(coords, dtype=_f64, device=self.device)
            jac, body_x = self.fk._launch_body_jacobian_device(q_wp, 1.0e-6)
            wp.launch(
                body_actuator_kernel,
                dim=(batch, self.ncoord),
                inputs=[
                    jac,
                    body_x,
                    wp.array(np.ascontiguousarray(controls), dtype=_f64, device=self.device),
                    self.d_body_idx,
                    self.d_is_global,
                    self.d_point_global,
                    self.d_gain,
                    self.d_point,
                    self.num_actuators,
                    tau,
                ],
                device=self.device,
            )
        return tau.numpy()
