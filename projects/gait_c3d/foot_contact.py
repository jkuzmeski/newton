# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Anatomical foot contact spheres on a measured standing sole plane.

The scaled OpenSim foot geometry is a skeleton, so its lowest mesh vertex is
the bone surface, not the surface the subject stands on. Placing contact
spheres tangent to that mesh leaves the model floating 22 to 34 mm above the
ground through stance and tilts the modelled sole against the real one.

This module separates the two decisions that the old bounding-box rule mixed:

* where the ground is, which comes from the subject's own standing pose and
  yields a *sole plane* in each foot body frame, and
* where the spheres sit on that plane, which comes from a published anatomical
  layout expressed in fractions of the subject's own foot length.

The six landmarks per foot are the mean of two independent published
six-sphere sets, the Lin and Pandy lineage used by OpenCap and the OpenSim
``example3DWalking`` set. Both agree to within one percent of foot length on
the fifth metatarsal head, the hallux and the sphere radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FOOT_SPHERE_LAYOUT = (
    ("heel", 0.07, -0.03, False),
    ("rearfoot_lateral", 0.33, 0.11, False),
    ("metatarsal_5", 0.59, 0.21, False),
    ("metatarsal_1", 0.71, -0.13, False),
    ("toes_lateral", 0.83, 0.25, True),
    ("hallux", 0.96, -0.09, True),
)
"""Sphere landmarks as ``(name, length fraction, lateral fraction, on toes)``.

Length runs from the posterior foot border to the toe tip. The lateral
fraction is positive toward the lateral border of that foot.
"""

RADIUS_LENGTH_FRACTION = 0.13
"""Published sphere radius as a fraction of foot length."""

RADIUS_BOUNDS = (0.020, 0.035)
"""Accepted sphere radius range [m]."""

OVERLAP_MARGIN = 0.95
"""Fraction of the half spacing a radius may use before spheres overlap."""


@dataclass(frozen=True, slots=True)
class SolePlane:
    """The ground surface of one foot, expressed in that foot's body frame."""

    normal: np.ndarray
    """Unit ground normal in the foot body frame, shape [3]."""

    offset: float
    """Signed plane offset so the plane is ``normal . x == offset`` [m]."""

    samples: int
    """Number of standing samples the fit averaged."""

    residual: float
    """Root-mean-square distance of the fitted samples from the plane [m]."""

    def __post_init__(self) -> None:
        if self.normal.shape != (3,) or not np.isclose(np.linalg.norm(self.normal), 1.0, atol=1.0e-9):
            raise ValueError("sole plane normal must be a unit vector")
        if not math.isfinite(self.offset) or self.samples < 1:
            raise ValueError("sole plane needs a finite offset and at least one sample")
        if not math.isfinite(self.residual) or self.residual < 0.0:
            raise ValueError("sole plane residual must be finite and nonnegative")

    def height(self, point: np.ndarray) -> np.ndarray:
        """Return the height of foot-frame points above the sole plane [m].

        Args:
            point: Foot-frame positions, shape [3] or [point_count, 3].
        """
        return np.asarray(point, dtype=np.float64) @ self.normal - self.offset

    def project(self, point: np.ndarray) -> np.ndarray:
        """Return foot-frame points moved onto the sole plane along its normal.

        Args:
            point: Foot-frame positions, shape [3] or [point_count, 3].
        """
        point = np.asarray(point, dtype=np.float64)
        return point - np.asarray(self.height(point))[..., None] * self.normal

    def level_roll(self) -> SolePlane:
        """Return the same plane with its frontal-plane tilt removed.

        A quiet standing pose is toed out and rolled several degrees against
        the walking stance, and this model's ankle is a pure sagittal hinge, so
        all frontal-plane foot orientation comes from the leg chain. Measured
        across both subjects, standing roll differs from walking roll by 6.5 to
        8.4 degrees while standing pitch and plane height match walking within
        0.5 degrees and 0.5 mm. Levelling the roll therefore keeps the two
        well-measured terms and drops the one that does not transfer.
        """
        leveled = np.asarray((self.normal[0], 0.0, self.normal[2]))
        norm = np.linalg.norm(leveled)
        if norm < 1.0e-9:
            raise ValueError("a sole plane normal without a vertical component cannot be levelled")
        leveled = leveled / norm
        return SolePlane(leveled, self.offset * float(leveled @ self.normal), self.samples, self.residual)

    def tilt_degrees(self) -> tuple[float, float]:
        """Return the plane tilt against a level foot frame as (pitch, roll) [deg]."""
        pitch = math.degrees(math.atan2(-self.normal[0], self.normal[2]))
        roll = math.degrees(math.atan2(self.normal[1], self.normal[2]))
        return pitch, roll


@dataclass(frozen=True, slots=True)
class ContactSphere:
    """One placed foot contact sphere."""

    name: str
    """Landmark name from :data:`FOOT_SPHERE_LAYOUT`."""

    side: str
    """``"left"`` or ``"right"``."""

    body: str
    """Newton body the sphere belongs to."""

    center: tuple[float, float, float]
    """Sphere center in the body named by :attr:`body` [m]."""

    radius: float
    """Sphere radius [m]."""


def fit_sole_plane(poses: np.ndarray) -> SolePlane:
    """Fit the ground plane of one foot from standing foot poses.

    A world ground plane ``z = 0`` seen from a foot whose body frame sits at
    ``(rotation, translation)`` is the foot-frame plane
    ``(rotation^T z) . x == -translation_z``. Averaging that plane over the
    standing samples gives the surface the subject actually stands on,
    including the heel-pad and shoe thickness that the bone mesh omits.

    Args:
        poses: Standing foot body transforms, shape [sample_count, 4, 4], with
            rotation in ``[:3, :3]`` and translation in ``[:3, 3]``.

    Returns:
        The averaged sole plane of that foot.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 1:
        raise ValueError("standing foot poses must have shape [sample_count, 4, 4]")
    if not np.all(np.isfinite(poses)):
        raise ValueError("standing foot poses must be finite")
    normals = poses[:, 2, :3]
    offsets = -poses[:, 2, 3]
    mean_normal = normals.mean(axis=0)
    norm = np.linalg.norm(mean_normal)
    if norm < 1.0e-9:
        raise ValueError("standing foot poses do not define a consistent ground normal")
    mean_normal = mean_normal / norm
    # Re-project every sample onto the averaged normal so the offset stays the
    # distance to the same plane rather than to each sample's own tilt.
    projected = offsets * (normals @ mean_normal)
    offset = float(np.median(projected))
    residual = float(np.sqrt(np.mean((projected - offset) ** 2)))
    return SolePlane(mean_normal, offset, len(poses), residual)


def foot_sphere_radius(foot_length: float, *, spacing: float | None = None) -> float:
    """Return the contact sphere radius for one foot [m].

    The published radius is 13 percent of foot length. It is clamped to
    :data:`RADIUS_BOUNDS` and, when the landmark spacing is known, reduced so
    that neighboring spheres never overlap. Overlapping spheres would apply the
    same contact twice in one region.

    Args:
        foot_length: Foot length from the posterior border to the toe tip [m].
        spacing: Smallest center-to-center distance of the placed spheres [m].
    """
    if not math.isfinite(foot_length) or foot_length <= 0.0:
        raise ValueError("foot length must be finite and positive")
    radius = float(np.clip(RADIUS_LENGTH_FRACTION * foot_length, *RADIUS_BOUNDS))
    if spacing is not None:
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("sphere spacing must be finite and positive")
        # Non-overlap wins over the lower bound: two spheres that share a
        # region would apply the same contact twice.
        radius = min(radius, OVERLAP_MARGIN * 0.5 * spacing)
    return radius


def place_foot_spheres(
    side: str,
    sole: SolePlane,
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    lateral: np.ndarray,
    foot_length: float,
    toe_offset: np.ndarray,
    radius: float | None = None,
) -> tuple[ContactSphere, ...]:
    """Place the six anatomical contact spheres of one foot on its sole plane.

    Each landmark is laid out along the foot from ``origin``, projected onto
    the sole plane, then lifted one radius along the plane normal so the sphere
    touches the plane exactly.

    Args:
        side: ``"left"`` or ``"right"``.
        sole: Fitted sole plane of this foot.
        origin: Posterior foot border in the foot body frame, shape [3].
        forward: Unit foot-frame direction from heel to toe, shape [3].
        lateral: Unit foot-frame direction toward the lateral border, shape [3].
        foot_length: Foot length from the posterior border to the toe tip [m].
        toe_offset: Toes-body origin in the foot body frame [m], shape [3].
            This is required because toe sphere centers are returned in that
            body's local frame.
        radius: Explicit sphere radius [m], or ``None`` to derive it.
    """
    if side not in ("left", "right"):
        raise ValueError(f"unknown foot side {side!r}")
    if not math.isfinite(foot_length) or foot_length <= 0.0:
        raise ValueError("foot length must be finite and positive")
    origin = np.asarray(origin, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    lateral = np.asarray(lateral, dtype=np.float64)
    toe_offset = np.asarray(toe_offset, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("foot origin must be a finite three-component vector")
    if toe_offset.shape != (3,) or not np.all(np.isfinite(toe_offset)):
        raise ValueError("toe offset must be a finite three-component vector")
    for name, axis in (("forward", forward), ("lateral", lateral)):
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or not np.isclose(np.linalg.norm(axis), 1.0, atol=1.0e-9)
        ):
            raise ValueError(f"{name} axis must be a unit vector")
    if not np.isclose(forward @ lateral, 0.0, atol=1.0e-9):
        raise ValueError("forward and lateral axes must be orthogonal")
    points = np.stack(
        [
            origin + length * foot_length * forward + width * foot_length * lateral
            for _, length, width, _ in FOOT_SPHERE_LAYOUT
        ]
    )
    placed = sole.project(points)
    spacing = float(
        np.min(np.linalg.norm(placed[:, None, :] - placed[None, :, :], axis=-1)[np.triu_indices(len(placed), 1)])
    )
    radius = foot_sphere_radius(foot_length, spacing=spacing) if radius is None else float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("contact sphere radius must be finite and positive")
    centers = placed + radius * sole.normal
    return tuple(
        ContactSphere(
            name=name,
            side=side,
            body=f"{'toes' if on_toes else 'foot'}_{side}",
            center=tuple(float(value) for value in center - (toe_offset if on_toes else 0.0)),
            radius=radius,
        )
        for (name, _, _, on_toes), center in zip(FOOT_SPHERE_LAYOUT, centers, strict=True)
    )


def standing_foot_poses(
    site_positions: np.ndarray,
    measured: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Recover per-sample foot body poses from measured standing markers.

    The compiled subject stores each foot marker in its foot body frame, and a
    static standing capture measures the same markers in the laboratory frame.
    A per-sample rigid fit of one set onto the other therefore gives the pose
    of that foot body while the subject stands on the floor.

    Args:
        site_positions: Marker sites in the foot body frame [m], shape [marker_count, 3].
        measured: Measured marker positions [m], shape [sample_count, marker_count, 3].
        valid: Marker visibility mask, shape [sample_count, marker_count].

    Returns:
        Foot body transforms, shape [sample_count, 4, 4]. Samples with fewer
        than three visible markers are omitted.
    """
    site_positions = np.asarray(site_positions, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if site_positions.ndim != 2 or site_positions.shape[1] != 3:
        raise ValueError("marker sites must have shape [marker_count, 3]")
    if measured.ndim != 3 or measured.shape[1:] != site_positions.shape:
        raise ValueError("measured markers must have shape [sample_count, marker_count, 3]")
    valid = np.ones(measured.shape[:2], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != measured.shape[:2]:
        raise ValueError("marker visibility must have shape [sample_count, marker_count]")
    if not np.all(np.isfinite(site_positions)) or not np.all(np.isfinite(measured[valid])):
        raise ValueError("visible standing marker positions must be finite")
    poses = []
    for sample in range(len(measured)):
        columns = np.flatnonzero(valid[sample])
        if len(columns) < 3:
            continue
        local = site_positions[columns]
        world = measured[sample, columns]
        local_center = local.mean(axis=0)
        world_center = world.mean(axis=0)
        if np.linalg.matrix_rank(local - local_center) < 2 or np.linalg.matrix_rank(world - world_center) < 2:
            continue
        u, _, vt = np.linalg.svd((local - local_center).T @ (world - world_center))
        correction = np.diag((1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))))
        rotation = vt.T @ correction @ u.T
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = world_center - rotation @ local_center
        poses.append(pose)
    if not poses:
        raise ValueError("no standing sample has three visible foot markers")
    return np.stack(poses)


def foot_axes_from_bounds(
    side: str,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the foot layout axes taken from a foot mesh bounding box.

    Args:
        side: ``"left"`` or ``"right"``.
        minimum: Lowest mesh corner in the foot body frame [m], shape [3].
        maximum: Highest mesh corner in the foot body frame [m], shape [3].

    Returns:
        ``(origin, forward, lateral, length)`` where ``origin`` is the
        posterior foot border on the mesh center line, ``forward`` points to
        the toes, ``lateral`` points to that foot's lateral border, and
        ``length`` is the foot length [m].
    """
    if side not in ("left", "right"):
        raise ValueError(f"unknown foot side {side!r}")
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum <= minimum)
    ):
        raise ValueError("foot bounds must be a finite nondegenerate box")
    origin = np.asarray((minimum[0], 0.5 * (minimum[1] + maximum[1]), 0.5 * (minimum[2] + maximum[2])))
    forward = np.asarray((1.0, 0.0, 0.0))
    lateral = np.asarray((0.0, 1.0 if side == "left" else -1.0, 0.0))
    return origin, forward, lateral, float(maximum[0] - minimum[0])
