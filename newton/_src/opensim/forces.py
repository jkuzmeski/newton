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

r"""Warp-native passive force elements and the generalized forces they produce.

Covers the OpenSim ``PointToPointSpring`` (a linear spring between a point on each
of two bodies, projected through the body spatial Jacobians), the
``SpringGeneralizedForce`` (a passive spring-damper acting directly on a single
coordinate), and the elastic part of the ``BushingForce`` (a 6-DOF linear frame
bushing, whose conservative load equals the gradient of a quadratic deflection
potential and is obtained by central-differencing that potential on device). All
reuse :class:`~newton.opensim.ForwardKinematics` for coordinate ordering and body
transforms.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .kinematics import ForwardKinematics, euler_xyz_to_matrix, make_transform
from .types import OsimModel

_f64 = wp.float64
_vec3d = wp.vec3d
_mat44d = wp.mat44d


@wp.func
def _spring_geometry(x1: _mat44d, x2: _mat44d, p1: _vec3d, p2: _vec3d):
    """Ground-frame lever arms, separation, and unit direction of a two-point spring."""
    rot1 = wp.mat33d(x1[0, 0], x1[0, 1], x1[0, 2], x1[1, 0], x1[1, 1], x1[1, 2], x1[2, 0], x1[2, 1], x1[2, 2])
    rot2 = wp.mat33d(x2[0, 0], x2[0, 1], x2[0, 2], x2[1, 0], x2[1, 1], x2[1, 2], x2[2, 0], x2[2, 1], x2[2, 2])
    r1 = wp.mul(rot1, p1)
    r2 = wp.mul(rot2, p2)
    g1 = r1 + _vec3d(x1[0, 3], x1[1, 3], x1[2, 3])
    g2 = r2 + _vec3d(x2[0, 3], x2[1, 3], x2[2, 3])
    vec = g2 - g1
    d = wp.length(vec)
    unit = vec / wp.max(d, _f64(1.0e-12))
    return r1, r2, d, unit


@wp.kernel
def point_spring_tension_kernel(
    body_X: wp.array2d[_mat44d],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    point1: wp.array[_vec3d],
    point2: wp.array[_vec3d],
    stiffness: wp.array[_f64],
    rest_length: wp.array[_f64],
    tension: wp.array2d[_f64],
):
    """Two-point spring tension ``stiffness * (distance - rest_length)``.

    Launched with dim ``(batch, num_springs)``; positive tension pulls the
    attachment points together.
    """
    b, s = wp.tid()
    _r1, _r2, d, _unit = _spring_geometry(body_X[b, body1[s]], body_X[b, body2[s]], point1[s], point2[s])
    tension[b, s] = stiffness[s] * (d - rest_length[s])


@wp.kernel
def point_spring_kernel(
    jac: wp.array4d[_f64],
    body_X: wp.array2d[_mat44d],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    point1: wp.array[_vec3d],
    point2: wp.array[_vec3d],
    stiffness: wp.array[_f64],
    rest_length: wp.array[_f64],
    n_springs: int,
    tau: wp.array2d[_f64],
):
    """Accumulate two-point-spring generalized forces by transposed-Jacobian projection.

    Each spring applies ``+f * unit`` at its point on ``body1`` and ``-f * unit`` on
    ``body2`` (``f`` the tension, ``unit`` from body1 to body2), reduces each to a
    wrench about that body's origin, and contracts it with the body Jacobian column.
    """
    b, i = wp.tid()
    acc = _f64(0.0)
    for s in range(n_springs):
        b1 = body1[s]
        b2 = body2[s]
        r1, r2, d, unit = _spring_geometry(body_X[b, b1], body_X[b, b2], point1[s], point2[s])
        f = stiffness[s] * (d - rest_length[s])
        f1 = f * unit
        f2 = -f * unit
        t1 = wp.cross(r1, f1)
        t2 = wp.cross(r2, f2)
        acc += t1[0] * jac[b, b1, 0, i] + t1[1] * jac[b, b1, 1, i] + t1[2] * jac[b, b1, 2, i]
        acc += f1[0] * jac[b, b1, 3, i] + f1[1] * jac[b, b1, 4, i] + f1[2] * jac[b, b1, 5, i]
        acc += t2[0] * jac[b, b2, 0, i] + t2[1] * jac[b, b2, 1, i] + t2[2] * jac[b, b2, 2, i]
        acc += f2[0] * jac[b, b2, 3, i] + f2[1] * jac[b, b2, 4, i] + f2[2] * jac[b, b2, 5, i]
    tau[b, i] = acc


class PointToPointSprings:
    """Warp-native generalized forces from OpenSim ``PointToPointSpring`` elements.

    Each spring connects a point on one body to a point on another with a linear
    force ``stiffness * (distance - rest_length)`` along the line between them. The
    equal-and-opposite point forces are reduced to ground-frame wrenches and
    projected through the body spatial Jacobians to generalized forces (see
    :meth:`~newton.opensim.ForwardKinematics.generalized_forces_from_body_load`).

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernels (``None`` selects the CPU).

    Attributes:
        fk: The :class:`~newton.opensim.ForwardKinematics` backing the Jacobians.
        coordinate_names: Generalized coordinate names in model order.
        spring_names: Spring names in model order.
        num_springs: Number of two-point springs.
    """

    def __init__(self, model: OsimModel, device=None):
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.coordinate_names = self.fk.coordinate_names
        self.ncoord = self.fk.ncoord
        springs = model.point_to_point_springs
        self.spring_names = [s.name for s in springs]
        self.num_springs = len(springs)
        names = self.fk.body_names

        def _bidx(name: str) -> int:
            return names.index(name) if name in names else 0

        self.d_body1 = wp.array(
            np.array([_bidx(s.body1) for s in springs], np.int32), dtype=wp.int32, device=self.device
        )
        self.d_body2 = wp.array(
            np.array([_bidx(s.body2) for s in springs], np.int32), dtype=wp.int32, device=self.device
        )
        self.d_point1 = wp.array(
            np.array([s.point1 for s in springs], np.float64).reshape(-1, 3), dtype=_vec3d, device=self.device
        )
        self.d_point2 = wp.array(
            np.array([s.point2 for s in springs], np.float64).reshape(-1, 3), dtype=_vec3d, device=self.device
        )
        self.d_stiffness = wp.array(
            np.array([s.stiffness for s in springs], np.float64), dtype=_f64, device=self.device
        )
        self.d_rest = wp.array(np.array([s.rest_length for s in springs], np.float64), dtype=_f64, device=self.device)

    def forces(self, coords: np.ndarray) -> np.ndarray:
        """Return spring tensions [N], shape ``[batch, num_springs]`` (positive = stretched)."""
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch = coords.shape[0]
        if not self.num_springs:
            return np.zeros((batch, 0))
        tension = wp.empty((batch, self.num_springs), dtype=_f64, device=self.device)
        if self.num_springs:
            body_x = self.fk._launch_body_transforms(wp.array(coords, dtype=_f64, device=self.device))
            wp.launch(
                point_spring_tension_kernel,
                dim=(batch, self.num_springs),
                inputs=[
                    body_x,
                    self.d_body1,
                    self.d_body2,
                    self.d_point1,
                    self.d_point2,
                    self.d_stiffness,
                    self.d_rest,
                    tension,
                ],
                device=self.device,
            )
        return tension.numpy()

    def generalized_forces(self, coords: np.ndarray) -> np.ndarray:
        """Return spring generalized forces [N or N·m], shape ``[batch, num_coordinates]``.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].

        Returns:
            Generalized forces in :attr:`coordinate_names` order, summed over all springs.
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch = coords.shape[0]
        if not self.num_springs:
            return np.zeros((batch, self.ncoord))
        tau = wp.empty((batch, self.ncoord), dtype=_f64, device=self.device)
        if self.num_springs:
            q_wp = wp.array(coords, dtype=_f64, device=self.device)
            jac, body_x = self.fk._launch_body_jacobian_device(q_wp, 1.0e-6)
            wp.launch(
                point_spring_kernel,
                dim=(batch, self.ncoord),
                inputs=[
                    jac,
                    body_x,
                    self.d_body1,
                    self.d_body2,
                    self.d_point1,
                    self.d_point2,
                    self.d_stiffness,
                    self.d_rest,
                    self.num_springs,
                    tau,
                ],
                device=self.device,
            )
        return tau.numpy()


@wp.kernel
def spring_generalized_force_kernel(
    q: wp.array2d[_f64],
    qd: wp.array2d[_f64],
    coord: wp.array[wp.int32],
    stiffness: wp.array[_f64],
    rest_length: wp.array[_f64],
    viscosity: wp.array[_f64],
    tau: wp.array2d[_f64],
):
    """Scatter each coordinate spring-damper force onto its coordinate.

    Launched with dim ``(batch, num_springs)``; the generalized force is
    ``-stiffness * (q - rest_length) - viscosity * qd`` and springs sharing a
    coordinate accumulate through an atomic add.
    """
    b, s = wp.tid()
    c = coord[s]
    f = -stiffness[s] * (q[b, c] - rest_length[s]) - viscosity[s] * qd[b, c]
    wp.atomic_add(tau, b, c, f)


class SpringGeneralizedForces:
    """Warp-native generalized forces from OpenSim ``SpringGeneralizedForce`` elements.

    Each element applies ``-stiffness * (q - rest_length) - viscosity * qd`` directly
    to its coordinate, modelling passive joint stiffness and damping.

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernel (``None`` selects the CPU).

    Attributes:
        coordinate_names: Generalized coordinate names in model order.
        spring_names: Spring names in model order.
        num_springs: Number of coordinate spring-dampers.
    """

    def __init__(self, model: OsimModel, device=None):
        self.device = wp.get_device(device)
        self.coordinate_names: list[str] = [c.name for j in model.joints for c in j.coordinates]
        self.ncoord = len(self.coordinate_names)
        index = {name: i for i, name in enumerate(self.coordinate_names)}
        springs = [s for s in model.spring_generalized_forces if s.coordinate in index]
        self.spring_names = [s.name for s in springs]
        self.num_springs = len(springs)
        self.d_coord = wp.array(
            np.array([index[s.coordinate] for s in springs], np.int32), dtype=wp.int32, device=self.device
        )
        self.d_stiffness = wp.array(
            np.array([s.stiffness for s in springs], np.float64), dtype=_f64, device=self.device
        )
        self.d_rest = wp.array(np.array([s.rest_length for s in springs], np.float64), dtype=_f64, device=self.device)
        self.d_viscosity = wp.array(
            np.array([s.viscosity for s in springs], np.float64), dtype=_f64, device=self.device
        )

    def generalized_forces(self, coords: np.ndarray, speeds: np.ndarray | None = None) -> np.ndarray:
        """Return spring-damper generalized forces [N or N·m], shape ``[batch, num_coordinates]``.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            speeds: Coordinate speeds [batch, num_coordinates] or ``None`` (no damping).

        Returns:
            Generalized forces in :attr:`coordinate_names` order, summed over all springs.
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch = coords.shape[0]
        if not self.num_springs:
            return np.zeros((batch, self.ncoord))
        tau = wp.zeros((batch, self.ncoord), dtype=_f64, device=self.device)
        if self.num_springs:
            qd = (
                np.zeros_like(coords)
                if speeds is None
                else np.ascontiguousarray(np.atleast_2d(speeds), dtype=np.float64)
            )
            wp.launch(
                spring_generalized_force_kernel,
                dim=(batch, self.num_springs),
                inputs=[
                    wp.array(coords, dtype=_f64, device=self.device),
                    wp.array(qd, dtype=_f64, device=self.device),
                    self.d_coord,
                    self.d_stiffness,
                    self.d_rest,
                    self.d_viscosity,
                    tau,
                ],
                device=self.device,
            )
        return tau.numpy()


@wp.func
def _rigid_parts(x: _mat44d):
    """Rotation (mat33) and translation (vec3) of a homogeneous transform."""
    rot = wp.mat33d(x[0, 0], x[0, 1], x[0, 2], x[1, 0], x[1, 1], x[1, 2], x[2, 0], x[2, 1], x[2, 2])
    tr = _vec3d(x[0, 3], x[1, 3], x[2, 3])
    return rot, tr


@wp.func
def _euler_xyz_of(r: wp.mat33d):
    """Body-fixed XYZ Euler angles of a rotation ``r`` = Rx(a)·Ry(b)·Rz(c) [rad]."""
    b = wp.asin(wp.clamp(r[0, 2], _f64(-1.0), _f64(1.0)))
    a = wp.atan2(-r[1, 2], r[2, 2])
    c = wp.atan2(-r[0, 1], r[0, 0])
    return _vec3d(a, b, c)


@wp.func
def _bushing_energy(
    body_X: wp.array2d[_mat44d],
    cfg: int,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    t1: wp.array[_mat44d],
    t2: wp.array[_mat44d],
    krot: wp.array[_vec3d],
    ktrans: wp.array[_vec3d],
    n_bush: int,
):
    """Total elastic potential 0.5*(theta.Kr.theta + r.Kt.r) summed over bushings at ``cfg``."""
    u = _f64(0.0)
    for s in range(n_bush):
        xf1 = wp.mul(body_X[cfg, body1[s]], t1[s])
        xf2 = wp.mul(body_X[cfg, body2[s]], t2[s])
        r1, o1 = _rigid_parts(xf1)
        r2, o2 = _rigid_parts(xf2)
        r1t = wp.transpose(r1)
        rel = wp.mul(r1t, r2)
        d = wp.mul(r1t, o2 - o1)
        th = _euler_xyz_of(rel)
        kr = krot[s]
        kt = ktrans[s]
        u += _f64(0.5) * (kr[0] * th[0] * th[0] + kr[1] * th[1] * th[1] + kr[2] * th[2] * th[2])
        u += _f64(0.5) * (kt[0] * d[0] * d[0] + kt[1] * d[1] * d[1] + kt[2] * d[2] * d[2])
    return u


@wp.kernel
def bushing_energy_kernel(
    body_X: wp.array2d[_mat44d],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    t1: wp.array[_mat44d],
    t2: wp.array[_mat44d],
    krot: wp.array[_vec3d],
    ktrans: wp.array[_vec3d],
    energy: wp.array2d[_f64],
):
    """Per-bushing elastic potential energy 0.5*(theta.Kr.theta + r.Kt.r) [J].

    Launched with dim ``(batch, num_bushings)``.
    """
    b, s = wp.tid()
    xf1 = wp.mul(body_X[b, body1[s]], t1[s])
    xf2 = wp.mul(body_X[b, body2[s]], t2[s])
    r1, o1 = _rigid_parts(xf1)
    r2, o2 = _rigid_parts(xf2)
    r1t = wp.transpose(r1)
    rel = wp.mul(r1t, r2)
    d = wp.mul(r1t, o2 - o1)
    th = _euler_xyz_of(rel)
    kr = krot[s]
    kt = ktrans[s]
    energy[b, s] = _f64(0.5) * (
        kr[0] * th[0] * th[0]
        + kr[1] * th[1] * th[1]
        + kr[2] * th[2] * th[2]
        + kt[0] * d[0] * d[0]
        + kt[1] * d[1] * d[1]
        + kt[2] * d[2] * d[2]
    )


@wp.kernel
def bushing_kernel(
    xp: wp.array2d[_mat44d],
    xm: wp.array2d[_mat44d],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    t1: wp.array[_mat44d],
    t2: wp.array[_mat44d],
    krot: wp.array[_vec3d],
    ktrans: wp.array[_vec3d],
    n_bush: int,
    ncoord: int,
    inv2eps: _f64,
    tau: wp.array2d[_f64],
):
    """Elastic-bushing generalized forces by central-differencing the deflection potential.

    Launched with dim ``(batch, ncoord)``; ``tau[b, i] = -(U(q+e_i) - U(q-e_i)) / (2*eps)``
    with the perturbed body transforms supplied in ``xp``/``xm`` (row ``b*ncoord + i``).
    """
    b, i = wp.tid()
    cfg = b * ncoord + i
    up = _bushing_energy(xp, cfg, body1, body2, t1, t2, krot, ktrans, n_bush)
    um = _bushing_energy(xm, cfg, body1, body2, t1, t2, krot, ktrans, n_bush)
    tau[b, i] = -(up - um) * inv2eps


class BushingForces:
    """Warp-native elastic generalized forces from OpenSim ``BushingForce`` elements.

    A ``BushingForce`` resists the deflection of ``frame2`` relative to ``frame1`` with a
    linear 6-DOF load. Its conservative (elastic) part is the gradient of the quadratic
    potential ``U = 0.5 * (theta . Kr . theta + r . Kt . r)``, where ``theta`` is the
    body-fixed XYZ Euler deflection and ``r`` the translational deflection expressed in
    ``frame1``. The generalized forces ``-dU/dq`` are obtained by central-differencing
    ``U`` on device, so they match OpenSim's elastic bushing exactly. Damping is not
    included.

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernels (``None`` selects the CPU).

    Attributes:
        fk: The :class:`~newton.opensim.ForwardKinematics` backing the transforms.
        coordinate_names: Generalized coordinate names in model order.
        bushing_names: Bushing names in model order.
        num_bushings: Number of bushings.
    """

    def __init__(self, model: OsimModel, device=None):
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.coordinate_names = self.fk.coordinate_names
        self.ncoord = self.fk.ncoord
        bushings = model.bushing_forces
        self.bushing_names = [b.name for b in bushings]
        self.num_bushings = len(bushings)
        names = self.fk.body_names

        def _bidx(name: str) -> int:
            return names.index(name) if name in names else 0

        def _offset(tf) -> np.ndarray:
            return make_transform(euler_xyz_to_matrix(*tf.orientation), tf.translation)

        self.d_body1 = wp.array(
            np.array([_bidx(b.body1) for b in bushings], np.int32), dtype=wp.int32, device=self.device
        )
        self.d_body2 = wp.array(
            np.array([_bidx(b.body2) for b in bushings], np.int32), dtype=wp.int32, device=self.device
        )
        t1 = np.array([_offset(b.frame1_transform) for b in bushings], np.float64).reshape(-1, 4, 4)
        t2 = np.array([_offset(b.frame2_transform) for b in bushings], np.float64).reshape(-1, 4, 4)
        self.d_t1 = wp.array(t1, dtype=_mat44d, device=self.device)
        self.d_t2 = wp.array(t2, dtype=_mat44d, device=self.device)
        self.d_krot = wp.array(
            np.array([b.rotational_stiffness for b in bushings], np.float64).reshape(-1, 3),
            dtype=_vec3d,
            device=self.device,
        )
        self.d_ktrans = wp.array(
            np.array([b.translational_stiffness for b in bushings], np.float64).reshape(-1, 3),
            dtype=_vec3d,
            device=self.device,
        )

    def potential_energy(self, coords: np.ndarray) -> np.ndarray:
        """Return per-bushing elastic potential energy [J], shape ``[batch, num_bushings]``."""
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch = coords.shape[0]
        if not self.num_bushings:
            return np.zeros((batch, 0))
        energy = wp.empty((batch, self.num_bushings), dtype=_f64, device=self.device)
        if self.num_bushings:
            body_x = self.fk._launch_body_transforms(wp.array(coords, dtype=_f64, device=self.device))
            wp.launch(
                bushing_energy_kernel,
                dim=(batch, self.num_bushings),
                inputs=[body_x, self.d_body1, self.d_body2, self.d_t1, self.d_t2, self.d_krot, self.d_ktrans, energy],
                device=self.device,
            )
        return energy.numpy()

    def generalized_forces(self, coords: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
        """Return elastic-bushing generalized forces [N or N·m], shape ``[batch, num_coordinates]``.

        Args:
            coords: Coordinate configurations [batch, num_coordinates].
            eps: Central-difference step for the potential gradient.

        Returns:
            Generalized forces ``-dU/dq`` in :attr:`coordinate_names` order, summed over bushings.
        """
        coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        batch, nc = coords.shape
        if not self.num_bushings:
            return np.zeros((batch, self.ncoord))
        tau = wp.empty((batch, self.ncoord), dtype=_f64, device=self.device)
        if self.num_bushings:
            q_wp = wp.array(coords, dtype=_f64, device=self.device)
            xp, xm = self.fk._launch_coordinate_perturbations(q_wp, eps)
            wp.launch(
                bushing_kernel,
                dim=(batch, nc),
                inputs=[
                    xp,
                    xm,
                    self.d_body1,
                    self.d_body2,
                    self.d_t1,
                    self.d_t2,
                    self.d_krot,
                    self.d_ktrans,
                    self.num_bushings,
                    nc,
                    _f64(1.0 / (2.0 * eps)),
                    tau,
                ],
                device=self.device,
            )
        return tau.numpy()
