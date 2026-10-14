# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic planar mechanics for a floating pelvis carrying two legs.

Coordinates are ``[x, z, pelvis, hip_r, knee_r, ankle_r, hip_l, knee_l,
ankle_l]``. The pelvis origin is the mid-hip center; in the sagittal plane both
thighs attach there. Angle conventions match :mod:`..hogan.mechanics`: absolute
angles increase from world +x toward +z, the pelvis local +x axis points up the
trunk (``pi/2`` upright), a thigh's absolute angle is ``pelvis + hip + pi``,
knee flexion is negative, and ankle dorsiflexion is positive.

Unlike the single-leg chain, the pelvis lumps only the trunk, arms, and head
(HAT); each leg is modeled explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BODY_NAMES = ("pelvis", "thigh_r", "shank_r", "foot_r", "thigh_l", "shank_l", "foot_l")
COORDINATE_NAMES = ("x", "z", "pelvis", "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l")
JOINT_NAMES = COORDINATE_NAMES[3:]
SIDES = ("right", "left")
PARENTS = (-1, 0, 1, 2, 0, 4, 5)
FOOT_BODY = {"right": 3, "left": 6}
UPRIGHT_PELVIS_RAD = 0.5 * math.pi
_ANGLE_OFFSETS = np.array([0.0, math.pi, math.pi, 1.5 * math.pi, math.pi, math.pi, 1.5 * math.pi])


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def side_index(side: str) -> int:
    """Return 0 for the right leg and 1 for the left leg."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    return SIDES.index(side)


def leg_coordinates(side: str) -> slice:
    """Return the hip/knee/ankle coordinate slice of one leg."""
    start = 3 + 3 * side_index(side)
    return slice(start, start + 3)


def angular_matrix() -> np.ndarray:
    """Return the constant body angular Jacobians, shape (7, 9)."""
    angular = np.zeros((7, 9))
    angular[:, 2] = 1.0
    for s in range(2):
        for k in range(3):
            angular[1 + 3 * s + k, 3 + 3 * s : 4 + 3 * s + k] = 1.0
    return angular


def mirror_coordinates(q: np.ndarray) -> np.ndarray:
    """Swap the right and left leg blocks of coordinates, shape [..., 9]."""
    q = np.asarray(q, dtype=float)
    return np.concatenate((q[..., :3], q[..., 6:9], q[..., 3:6]), axis=-1)


@dataclass(frozen=True)
class HeadArmsTrunk:
    """Lumped trunk, arms, and head carried rigidly by the pelvis.

    Defaults follow Winter's HAT segment for a 0.52 m hip-to-shoulder length:
    COM 0.626 of that length above the hip center and a radius of gyration of
    0.496 of it. They are anthropometric estimates, not subject measurements.

    Attributes:
        com_local_m: COM offset from the mid-hip center in the pelvis frame [m].
        radius_of_gyration_m: Sagittal radius of gyration about the COM [m].
        mass_kg: Lumped mass [kg]. ``None`` uses subject mass minus both legs.
    """

    com_local_m: tuple[float, float] = (0.33, 0.0)
    radius_of_gyration_m: float = 0.26
    mass_kg: float | None = None


class BilateralChain:
    """Describe the seven integrated bodies of the bilateral planar runner.

    Args:
        lengths_m: Thigh and shank lengths per side [m], shape (2, 2), rows
            ordered right, left.
        endpoint_local_m: Foot endpoint offset from each ankle [m], shape (2, 2).
        masses_kg: Body masses in :data:`BODY_NAMES` order [kg], shape (7,).
        com_local_m: COM offsets from each body origin [m], shape (7, 2).
        inertias_kg_m2: Planar inertias about each body COM [kg m^2], shape (7,).
    """

    def __init__(self, lengths_m, endpoint_local_m, masses_kg, com_local_m, inertias_kg_m2):
        self.lengths_m = _array(lengths_m, (2, 2), "lengths_m").copy()
        self.endpoint_local_m = _array(endpoint_local_m, (2, 2), "endpoint_local_m").copy()
        self.masses_kg = _array(masses_kg, (7,), "masses_kg").copy()
        self.com_local_m = _array(com_local_m, (7, 2), "com_local_m").copy()
        self.inertias_kg_m2 = _array(inertias_kg_m2, (7,), "inertias_kg_m2").copy()
        for name in ("lengths_m", "masses_kg", "inertias_kg_m2"):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f"{name} must be positive")
        self._angular = angular_matrix()
        self._rotational_mass = self._angular.T @ (self.inertias_kg_m2[:, None] * self._angular)
        # Each body's origin in its parent's frame; both thighs attach at the mid-hip center.
        self._offsets = np.zeros((7, 2))
        for s in range(2):
            self._offsets[2 + 3 * s] = (self.lengths_m[s, 0], 0.0)
            self._offsets[3 + 3 * s] = (self.lengths_m[s, 1], 0.0)
        self._cached_q: np.ndarray | None = None

    @property
    def total_mass_kg(self) -> float:
        """Return the summed mass of all seven bodies [kg]."""
        return float(np.sum(self.masses_kg))

    def to_dict(self) -> dict:
        """Serialize the physical parameters."""
        return {
            name: getattr(self, name).tolist()
            for name in ("lengths_m", "endpoint_local_m", "masses_kg", "com_local_m", "inertias_kg_m2")
        }

    @classmethod
    def from_dict(cls, data: dict) -> BilateralChain:
        """Rebuild a chain from :meth:`to_dict`."""
        return cls(
            data["lengths_m"], data["endpoint_local_m"], data["masses_kg"], data["com_local_m"], data["inertias_kg_m2"]
        )

    @staticmethod
    def _body(body: int) -> int:
        if isinstance(body, bool) or not isinstance(body, (int, np.integer)) or not 0 <= body < 7:
            raise ValueError("body must index BODY_NAMES (0 to 6)")
        return int(body)

    def _geometry(self, q) -> None:
        q = _array(q, (9,), "q")
        if self._cached_q is not None and np.array_equal(q, self._cached_q):
            return
        self._angles = self._angular @ q + _ANGLE_OFFSETS
        c, s = np.cos(self._angles), np.sin(self._angles)
        self._rotations = np.empty((7, 2, 2))
        self._rotations[:, 0, 0] = c
        self._rotations[:, 0, 1] = -s
        self._rotations[:, 1, 0] = s
        self._rotations[:, 1, 1] = c
        self._origins = np.zeros((7, 2))
        self._origin_jacobians = np.zeros((7, 2, 9))
        self._links = np.zeros((7, 2))
        self._origins[0] = q[:2]
        self._origin_jacobians[0, :, :2] = np.eye(2)
        for body in range(1, 7):
            parent = PARENTS[body]
            link = self._rotations[parent] @ self._offsets[body]
            self._links[body] = link
            self._origins[body] = self._origins[parent] + link
            self._origin_jacobians[body] = self._origin_jacobians[parent] + np.outer(
                (-link[1], link[0]), self._angular[parent]
            )
        self._com_radii = (self._rotations @ self.com_local_m[:, :, None])[:, :, 0]
        perpendicular = self._com_radii[:, ::-1] * np.array([-1.0, 1.0])
        self._com_jacobians = self._origin_jacobians + perpendicular[:, :, None] * self._angular[:, None, :]
        self._cached_q = q.copy()

    def angular_jacobian(self, body: int) -> np.ndarray:
        """Return the constant angular Jacobian of a body, shape (9,)."""
        return self._angular[self._body(body)].copy()

    def angle(self, q, body: int) -> float:
        """Return a body's absolute angle [rad]."""
        self._geometry(q)
        return float(self._angles[self._body(body)])

    def _origin_centripetal(self, v) -> np.ndarray:
        omega_squared = np.square(self._angular @ _array(v, (9,), "v"))
        acceleration = np.zeros((7, 2))
        for body in range(1, 7):
            parent = PARENTS[body]
            acceleration[body] = acceleration[parent] - self._links[body] * omega_squared[parent]
        return acceleration, omega_squared

    def point(self, q, body: int, local, v=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a point's position [m], Jacobian, shape (2, 9), and ``Jdot @ v`` [m/s^2]."""
        self._geometry(q)
        body = self._body(body)
        radius = self._rotations[body] @ _array(local, (2,), "local")
        position = self._origins[body] + radius
        jacobian = self._origin_jacobians[body] + np.outer((-radius[1], radius[0]), self._angular[body])
        centripetal = np.zeros(2)
        if v is not None:
            origin, omega_squared = self._origin_centripetal(v)
            centripetal = origin[body] - radius * omega_squared[body]
        return position, jacobian, centripetal

    def ankle(self, q, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return one ankle's position [m] and Jacobian, shape (2, 9)."""
        position, jacobian, _ = self.point(q, FOOT_BODY[side], np.zeros(2))
        return position, jacobian

    def kinematics(self, q) -> np.ndarray:
        """Return hip, knee, ankle, and toe positions per side [m], shape (2, 4, 2)."""
        self._geometry(q)
        result = np.empty((2, 4, 2))
        for s in range(2):
            result[s, 0] = self._origins[1 + 3 * s]
            result[s, 1] = self._origins[2 + 3 * s]
            result[s, 2] = self._origins[3 + 3 * s]
            result[s, 3] = self._origins[3 + 3 * s] + self._rotations[3 + 3 * s] @ self.endpoint_local_m[s]
        return result

    def reach(self, q, ankle_m, side: str) -> np.ndarray:
        """Re-solve one leg's hip and knee so its ankle reaches ``ankle_m``, keeping the foot angle.

        Args:
            q: Coordinates, shape [N, 9].
            ankle_m: Ankle targets [m], shape [N, 2]. Out-of-reach targets straighten the knee.
            side: Leg to re-solve.

        Returns:
            Coordinates with that leg's hip, knee, and ankle angles replaced, shape [N, 9].
        """
        q = np.array(q, dtype=float)
        s = side_index(side)
        hip, knee, ankle = 3 + 3 * s, 4 + 3 * s, 5 + 3 * s
        delta = np.asarray(ankle_m, dtype=float) - q[:, :2]
        thigh, shank = self.lengths_m[s]
        distance = np.clip(np.linalg.norm(delta, axis=1), abs(thigh - shank) + 1e-9, (thigh + shank) * (1 - 1e-9))
        hip_angle = np.arccos((thigh**2 + distance**2 - shank**2) / (2 * thigh * distance))
        knee_angle = np.arccos((thigh**2 + shank**2 - distance**2) / (2 * thigh * shank)) - math.pi
        thigh_angle = np.arctan2(delta[:, 1], delta[:, 0]) + hip_angle
        old_thigh = q[:, 2] + q[:, hip] + math.pi
        total = q[:, 2] + q[:, hip] + q[:, knee] + q[:, ankle]
        q[:, hip] += np.angle(np.exp(1j * (thigh_angle - old_thigh)))
        q[:, knee] = knee_angle
        q[:, ankle] = total - q[:, 2] - q[:, hip] - q[:, knee]
        return q

    def fit_leg(
        self,
        q,
        ankle_m,
        side: str,
        *,
        position_scale_m: float = 0.005,
        angle_scale_rad: float = 0.2,
        iterations: int = 30,
    ) -> np.ndarray:
        """Regularized inverse kinematics of one leg, keeping the foot's absolute angle.

        Minimizes ``|FK - ankle|^2 / position_scale^2 + |angles - measured|^2 /
        angle_scale^2`` over hip and knee per frame. Where the measured ankle is
        reachable and consistent, it is matched to a fraction of
        ``position_scale_m``; near full knee extension, where the projected
        sagittal hip-to-ankle distance can exceed the fixed segment lengths, the
        measured angles keep the solution from snapping the knee straight.

        Args:
            q: Coordinates with measured joint angles, shape [N, 9].
            ankle_m: Ankle targets [m], shape [N, 2].
            side: Leg to fit.
            position_scale_m: Ankle position tolerance [m].
            angle_scale_rad: Joint angle tolerance [rad].
            iterations: Gauss-Newton iterations.
        """
        q = np.array(q, dtype=float)
        s = side_index(side)
        hip, knee, ankle = 3 + 3 * s, 4 + 3 * s, 5 + 3 * s
        target = np.asarray(ankle_m, dtype=float) - q[:, :2]
        thigh, shank = self.lengths_m[s]
        total = q[:, 2] + q[:, hip] + q[:, knee] + q[:, ankle]
        measured = q[:, [hip, knee]].copy()
        angles = measured.copy()
        for _ in range(iterations):
            a1 = q[:, 2] + angles[:, 0] + math.pi
            a2 = a1 + angles[:, 1]
            position = np.column_stack(
                (thigh * np.cos(a1) + shank * np.cos(a2), thigh * np.sin(a1) + shank * np.sin(a2))
            )
            d2 = np.column_stack((-shank * np.sin(a2), shank * np.cos(a2)))
            d1 = np.column_stack((-thigh * np.sin(a1), thigh * np.cos(a1))) + d2
            jacobian = np.zeros((len(q), 4, 2))
            jacobian[:, :2, 0] = d1 / position_scale_m
            jacobian[:, :2, 1] = d2 / position_scale_m
            jacobian[:, 2, 0] = 1.0 / angle_scale_rad
            jacobian[:, 3, 1] = 1.0 / angle_scale_rad
            residual = np.concatenate(
                ((position - target) / position_scale_m, (angles - measured) / angle_scale_rad), axis=1
            )
            normal = np.einsum("nij,nik->njk", jacobian, jacobian)
            gradient = np.einsum("nij,ni->nj", jacobian, residual)
            angles -= np.linalg.solve(normal, gradient[:, :, None])[:, :, 0]
            angles[:, 1] = np.minimum(angles[:, 1], 0.0)
        q[:, hip], q[:, knee] = angles[:, 0], angles[:, 1]
        q[:, ankle] = total - q[:, 2] - q[:, hip] - q[:, knee]
        return q

    def com(self, q) -> np.ndarray:
        """Return the whole-body center of mass [m], shape (2,)."""
        self._geometry(q)
        return self.masses_kg @ (self._origins + self._com_radii) / self.total_mass_kg

    def com_jacobian(self, q) -> np.ndarray:
        """Return the whole-body center-of-mass Jacobian, shape (2, 9)."""
        self._geometry(q)
        return np.tensordot(self.masses_kg, self._com_jacobians, axes=1) / self.total_mass_kg

    def dynamics(self, q, v, gravity: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
        """Return the mass matrix, shape (9, 9), and bias in ``M @ acceleration + bias = load``."""
        self._geometry(q)
        if not np.isfinite(gravity) or gravity < 0:
            raise ValueError("gravity must be finite and nonnegative")
        origin, omega_squared = self._origin_centripetal(v)
        acceleration = origin - self._com_radii * omega_squared[:, None]
        acceleration[:, 1] += gravity
        mass = np.einsum("b,bij,bik->jk", self.masses_kg, self._com_jacobians, self._com_jacobians)
        mass += self._rotational_mass
        bias = np.einsum("b,bij,bi->j", self.masses_kg, self._com_jacobians, acceleration)
        return mass, bias

    def energy(self, q, v, gravity: float = 9.81) -> float:
        """Return kinetic plus gravitational potential energy [J]."""
        mass, _ = self.dynamics(q, v, gravity)
        v = _array(v, (9,), "v")
        heights = (self._origins + self._com_radii)[:, 1]
        return float(0.5 * v @ mass @ v + gravity * self.masses_kg @ heights)


def chain_from_references(
    references: dict[str, dict], profile: dict, subject_mass_kg: float, hat: HeadArmsTrunk | None = None
) -> BilateralChain:
    """Build the bilateral chain from per-side measured geometry and one leg profile.

    Args:
        references: Per-side prepared references with ``lengths_m`` and
            ``endpoint_local_m``, keyed by ``"right"`` and ``"left"``.
        profile: Leg inertial profile (thigh, shank, foot) applied to both legs.
        subject_mass_kg: Whole-body mass [kg].
        hat: Trunk/arms/head assumptions.
    """
    hat = hat or HeadArmsTrunk()
    leg_masses = np.asarray(profile["masses_kg"], dtype=float)
    leg_com = np.asarray(profile["com_local_m"], dtype=float)
    leg_inertia = np.asarray(profile["inertias_kg_m2"], dtype=float)
    mass = float(subject_mass_kg) - 2.0 * float(leg_masses.sum()) if hat.mass_kg is None else float(hat.mass_kg)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("HAT mass must be positive; check subject mass and leg masses")
    if not np.isfinite(hat.radius_of_gyration_m) or hat.radius_of_gyration_m <= 0:
        raise ValueError("HAT radius of gyration must be positive")
    return BilateralChain(
        np.array([references[side]["lengths_m"] for side in SIDES]),
        np.array([references[side]["endpoint_local_m"] for side in SIDES]),
        np.concatenate(([mass], leg_masses, leg_masses)),
        np.vstack((np.asarray(hat.com_local_m, dtype=float), leg_com, leg_com)),
        np.concatenate(([mass * hat.radius_of_gyration_m**2], leg_inertia, leg_inertia)),
    )
