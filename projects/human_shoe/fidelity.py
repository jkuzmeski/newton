# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pose-fidelity diagnostics for approximate Newton OpenSim imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import newton
from newton import opensim


@dataclass(frozen=True, slots=True)
class BodyFidelityError:
    """Per-body error relative to exact OpenSim kinematics [SI]."""

    body_name: str
    position_m: float
    orientation_deg: float
    linear_velocity_m_s: float
    angular_velocity_rad_s: float


@dataclass(frozen=True, slots=True)
class PoseFidelityReport:
    """Root-aligned imported-model fidelity relative to exact OpenSim kinematics."""

    root_body_name: str
    body_errors: tuple[BodyFidelityError, ...]

    def _rms(self, attribute: str) -> float:
        values = np.asarray([getattr(error, attribute) for error in self.body_errors], dtype=np.float64)
        return float(np.sqrt(np.mean(values * values))) if len(values) else 0.0

    @property
    def rms_position_m(self) -> float:
        """Root-mean-square body-origin position error [m]."""
        return self._rms("position_m")

    @property
    def rms_orientation_deg(self) -> float:
        """Root-mean-square body orientation error [deg]."""
        return self._rms("orientation_deg")

    @property
    def rms_linear_velocity_m_s(self) -> float:
        """Root-mean-square body-origin linear-velocity error [m/s]."""
        return self._rms("linear_velocity_m_s")

    @property
    def rms_angular_velocity_rad_s(self) -> float:
        """Root-mean-square body angular-velocity error [rad/s]."""
        return self._rms("angular_velocity_rad_s")

    @property
    def max_position_m(self) -> float:
        """Maximum body-origin position error [m]."""
        return max((error.position_m for error in self.body_errors), default=0.0)

    @property
    def max_orientation_deg(self) -> float:
        """Maximum body orientation error [deg]."""
        return max((error.orientation_deg for error in self.body_errors), default=0.0)

    @property
    def max_linear_velocity_m_s(self) -> float:
        """Maximum body-origin linear-velocity error [m/s]."""
        return max((error.linear_velocity_m_s for error in self.body_errors), default=0.0)

    @property
    def max_angular_velocity_rad_s(self) -> float:
        """Maximum body angular-velocity error [rad/s]."""
        return max((error.angular_velocity_rad_s for error in self.body_errors), default=0.0)

    def within(
        self,
        *,
        position_m: float = 0.005,
        orientation_deg: float = 2.0,
        linear_velocity_m_s: float = 0.02,
        angular_velocity_rad_s: float = 0.05,
    ) -> bool:
        """Return whether all body errors satisfy the supplied acceptance limits."""
        return (
            self.max_position_m <= position_m
            and self.max_orientation_deg <= orientation_deg
            and self.max_linear_velocity_m_s <= linear_velocity_m_s
            and self.max_angular_velocity_rad_s <= angular_velocity_rad_s
        )


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a rotation matrix for a Warp-order ``[x, y, z, w]`` quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm <= 0.0:
        raise ValueError("quaternion must have nonzero norm")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compare_imported_state(
    model: newton.Model,
    state: newton.State,
    osim_model: opensim.OsimModel,
    coordinates: np.ndarray,
    speeds: np.ndarray,
    *,
    root_body_name: str = "pelvis",
    device=None,
) -> PoseFidelityReport:
    """Compare a Newton state with exact OpenSim FK for one coordinate state.

    Exact poses are converted to Newton Z-up and rigidly aligned to the imported
    root body. Relative body errors therefore measure articulation fidelity, not
    an arbitrary global placement difference.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    fk = opensim.ForwardKinematics(osim_model, device=device)
    if coordinates.shape != (fk.ncoord,) or speeds.shape != (fk.ncoord,):
        raise ValueError(f"coordinates and speeds must both have shape ({fk.ncoord},)")

    exact_native = fk.body_transforms_batch(coordinates[None, :])[0]
    exact_velocity_native = fk.body_velocities_batch(coordinates[None, :], speeds[None, :])
    basis = opensim.OsimFrameConverter().matrix
    exact_rotation = np.einsum("ij,bjk->bik", basis, exact_native[:, :3, :3])
    exact_position = exact_native[:, :3, 3] @ basis.T
    exact_angular = exact_velocity_native["angular_velocity"][0] @ basis.T
    exact_linear = exact_velocity_native["linear_velocity"][0] @ basis.T

    imported_pose = state.body_q.numpy()
    imported_twist = state.body_qd.numpy()
    imported_com = model.body_com.numpy()
    imported_names = list(model.body_label)
    exact_index = {name: index for index, name in enumerate(fk.body_names)}
    if root_body_name not in imported_names or root_body_name not in exact_index:
        raise KeyError(f"root body {root_body_name!r} is not present in both models")

    imported_rotation = np.stack([_quat_matrix(pose[3:]) for pose in imported_pose])
    imported_position = imported_pose[:, :3].astype(np.float64)
    root_imported = imported_names.index(root_body_name)
    root_exact = exact_index[root_body_name]
    alignment_rotation = imported_rotation[root_imported] @ exact_rotation[root_exact].T
    alignment_translation = imported_position[root_imported] - alignment_rotation @ exact_position[root_exact]

    body_errors = []
    for imported_index, body_name in enumerate(imported_names):
        source_index = exact_index.get(body_name)
        if source_index is None:
            continue
        target_rotation = alignment_rotation @ exact_rotation[source_index]
        target_position = alignment_rotation @ exact_position[source_index] + alignment_translation
        relative_rotation = imported_rotation[imported_index] @ target_rotation.T
        cosine = np.clip(0.5 * (np.trace(relative_rotation) - 1.0), -1.0, 1.0)

        angular = imported_twist[imported_index, 3:].astype(np.float64)
        com_offset = imported_rotation[imported_index] @ imported_com[imported_index]
        origin_linear = imported_twist[imported_index, :3].astype(np.float64) - np.cross(angular, com_offset)
        target_angular = alignment_rotation @ exact_angular[source_index]
        target_linear = alignment_rotation @ exact_linear[source_index]
        body_errors.append(
            BodyFidelityError(
                body_name=body_name,
                position_m=float(np.linalg.norm(imported_position[imported_index] - target_position)),
                orientation_deg=float(np.degrees(np.arccos(cosine))),
                linear_velocity_m_s=float(np.linalg.norm(origin_linear - target_linear)),
                angular_velocity_rad_s=float(np.linalg.norm(angular - target_angular)),
            )
        )

    return PoseFidelityReport(root_body_name=root_body_name, body_errors=tuple(body_errors))
