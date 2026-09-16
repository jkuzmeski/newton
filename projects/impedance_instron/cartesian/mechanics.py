# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic CPU mechanics for one floating planar thigh, shank, and foot.

Coordinates are ``[hip_x, hip_z, thigh_angle, knee_angle, ankle_angle]``.
Angles increase from world +x toward +z, about Newton's -Y axis. Knee flexion
is negative and ankle dorsiflexion is positive. No hip rotational motor or
additional body is present.
"""

from __future__ import annotations

import numpy as np


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert a finite value to the required floating-point array shape."""
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


class Body:
    """Describe the three integrated leg bodies, ordered thigh, shank, foot.

    Args:
        lengths_m: Thigh and shank lengths [m], shape (2,).
        endpoint_local_m: Foot endpoint offset from the ankle [m], shape (2,).
        masses_kg: The three segment masses [kg], shape (3,).
        com_local_m: COM offsets from each proximal joint [m], shape (3, 2).
        inertias_kg_m2: Planar inertias about each segment COM [kg m^2], shape (3,).

    Local +x points from proximal to distal along the long bones and from
    heel toward toe for the foot. The foot frame origin is the ankle.
    """

    def __init__(self, lengths_m, endpoint_local_m, masses_kg, com_local_m, inertias_kg_m2):
        self.lengths_m = _array(lengths_m, (2,), "lengths_m").copy()
        self.endpoint_local_m = _array(endpoint_local_m, (2,), "endpoint_local_m").copy()
        self.masses_kg = _array(masses_kg, (3,), "masses_kg").copy()
        self.com_local_m = _array(com_local_m, (3, 2), "com_local_m").copy()
        self.inertias_kg_m2 = _array(inertias_kg_m2, (3,), "inertias_kg_m2").copy()
        for name in ("lengths_m", "masses_kg", "inertias_kg_m2"):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f"{name} must be positive")
        self._angular = np.zeros((3, 5))
        self._angular[:, 2:] = np.tril(np.ones((3, 3)))
        self._rotational_mass = self._angular.T @ (self.inertias_kg_m2[:, None] * self._angular)
        self._mass_weights = np.repeat(self.masses_kg, 2)[:, None]
        self._offsets = np.array([[self.lengths_m[0], 0.0], [self.lengths_m[1], 0.0], self.endpoint_local_m])
        self._cached_q: np.ndarray | None = None

    @staticmethod
    def _body(body: int) -> int:
        """Check a segment index without accepting boolean values."""
        if isinstance(body, bool) or not isinstance(body, (int, np.integer)) or not 0 <= body < 3:
            raise ValueError("body must be 0 (thigh), 1 (shank), or 2 (foot)")
        return int(body)

    def _geometry(self, q) -> None:
        """Cache rotations and point Jacobians shared by contact and dynamics."""
        q = _array(q, (5,), "q")
        if self._cached_q is not None and np.array_equal(q, self._cached_q):
            return
        self._angles = np.cumsum(q[2:]) + np.array([0.0, 0.0, np.pi / 2])
        cosines, sines = np.cos(self._angles), np.sin(self._angles)
        self._rotations = np.empty((3, 2, 2))
        self._rotations[:, 0, 0] = cosines
        self._rotations[:, 0, 1] = -sines
        self._rotations[:, 1, 0] = sines
        self._rotations[:, 1, 1] = cosines
        self._radii = (self._rotations @ self._offsets[:, :, None])[:, :, 0]
        self._com_radii = (self._rotations @ self.com_local_m[:, :, None])[:, :, 0]
        self._joints = np.empty((4, 2))
        self._joints[0] = q[:2]
        self._joints[1:] = q[:2] + np.cumsum(self._radii, axis=0)
        perpendicular = self._radii[:, ::-1] * np.array([-1.0, 1.0])
        increments = perpendicular[:, :, None] * self._angular[:, None, :]
        self._origin_jacobians = np.zeros((3, 2, 5))
        self._origin_jacobians[:, :, :2] = np.eye(2)
        self._origin_jacobians[1:] += np.cumsum(increments[:2], axis=0)
        com_perpendicular = self._com_radii[:, ::-1] * np.array([-1.0, 1.0])
        self._com_jacobians = self._origin_jacobians + com_perpendicular[:, :, None] * self._angular[:, None, :]
        self._cached_q = q.copy()

    def angular_jacobian(self, body: int) -> np.ndarray:
        """Return the constant angular Jacobian, shape (5,)."""
        return self._angular[self._body(body)].copy()

    def angle(self, q, body: int) -> float:
        """Return the segment's absolute angle [rad]."""
        self._geometry(q)
        return float(self._angles[self._body(body)])

    def point(self, q, body: int, local, v=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a point's position, translational Jacobian, and ``Jdot @ v``.

        Args:
            q: Coordinates [m, m, rad, rad, rad], shape (5,).
            body: Segment index, from 0 (thigh) through 2 (foot).
            local: Point offset from the proximal joint [m], shape (2,).
            v: Velocities [m/s, m/s, rad/s, rad/s, rad/s], shape (5,).
                None uses zero velocity.

        Returns:
            Position [m], shape (2,); Jacobian [m/(m or rad)], shape (2, 5);
            and centripetal acceleration [m/s^2], shape (2,).
        """
        self._geometry(q)
        body = self._body(body)
        radius = self._rotations[body] @ _array(local, (2,), "local")
        position = self._joints[body] + radius
        jacobian = self._origin_jacobians[body] + np.outer(np.array([-radius[1], radius[0]]), self._angular[body])
        centripetal = np.zeros(2)
        if v is not None:
            omega_squared = np.square(self._angular @ _array(v, (5,), "v"))
            centripetal = -np.sum(self._radii[:body] * omega_squared[:body, None], axis=0)
            centripetal -= radius * omega_squared[body]
        return position, jacobian, centripetal

    def kinematics(self, q) -> np.ndarray:
        """Return hip, knee, ankle, and foot endpoint positions [m], shape (4, 2)."""
        self._geometry(q)
        return self._joints.copy()

    def dynamics(self, q, v, gravity: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
        """Return the mass matrix and bias in ``M @ acceleration + bias = load``.

        Args:
            q: Coordinates [m, m, rad, rad, rad], shape (5,).
            v: Velocities [m/s, m/s, rad/s, rad/s, rad/s], shape (5,).
            gravity: Nonnegative gravitational acceleration magnitude [m/s^2].
                Gravity acts along world -z on only the three declared masses.

        Returns:
            Mass matrix, shape (5, 5), and centripetal plus gravity bias
            [N, N, N m, N m, N m], shape (5,).
        """
        self._geometry(q)
        v = _array(v, (5,), "v")
        if not np.isfinite(gravity) or gravity < 0:
            raise ValueError("gravity must be finite and nonnegative")
        omega_squared = np.square(self._angular @ v)
        centripetal = -self._com_radii * omega_squared[:, None]
        centripetal[1:] -= np.cumsum(self._radii[:2] * omega_squared[:2, None], axis=0)
        centripetal[:, 1] += gravity
        jacobian = self._com_jacobians.reshape(6, 5)
        mass = jacobian.T @ (self._mass_weights * jacobian) + self._rotational_mass
        bias = jacobian.T @ (self.masses_kg[:, None] * centripetal).reshape(6)
        return mass, bias
