# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""OpenSim ``Analyze``-style report tools for the Newton OpenSim port.

Thin host orchestration around the validated on-device primitives in the port,
reproducing the reports OpenSim's ``AnalyzeTool`` produces from a coordinate
motion (and, where relevant, muscle activations and external loads):

- :class:`BodyKinematics` -- per-body origin position and body-fixed XYZ Euler
  orientation in ground, plus the whole-body center of mass, reusing
  :meth:`~newton.opensim.ForwardKinematics.body_transforms_batch`
  and :meth:`~newton.opensim.ForwardKinematics.center_of_mass_batch`
  (with optional velocities/accelerations from the matching batch methods).
- :class:`MuscleAnalysis` -- per-muscle muscle-tendon length, fiber/tendon
  length, moment arms per coordinate, and the active/passive/tendon force
  breakdown, reusing :class:`~newton.opensim.MuscleForces` and
  :class:`~newton.opensim.MusclePaths`.
- :class:`JointReaction` -- the constraint reaction wrench transmitted at each
  joint, formed by summing the required spatial force over the child subtree
  (Newton-Euler) with the per-body spatial forces of
  ``bodyforce_kernel``, so gravity, inertia,
  applied external loads, and (optionally) muscle body loads are all accounted
  for.

Every tool consumes a coordinate motion (a :class:`~newton.opensim.Storage`
or a ``.mot``/``.sto`` path) and returns OpenSim-style
:class:`~newton.opensim.Storage` tables. Velocities and accelerations,
when requested, reuse the same OpenSim signal pipeline as the inverse-dynamics
tool (Butterworth low-pass + GCVSpline, via
``differentiate_coordinates``).
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

from .dynamics import (
    ExternalLoads,
    InverseDynamics,
    bodyforce_kernel,
    differentiate_coordinates,
    id_stencil_kernel,
    read_external_loads,
)
from .kinematics import ForwardKinematics, euler_xyz_to_matrix, make_transform
from .mocap import Storage, read_storage, write_storage
from .muscle_force import MuscleForces
from .types import OsimModel

wp.set_module_options({"enable_backward": False})

_f64 = wp.float64
_f32 = wp.float32
_vec3d = wp.vec3d

__all__ = [
    "BodyKinematics",
    "JointReaction",
    "MuscleAnalysis",
    "euler_xyz_from_matrix",
    "write_sto",
]


@wp.kernel
def joint_reaction_kernel(
    poses: wp.array2d[wp.mat44d],
    force: wp.array2d[_vec3d],
    torque: wp.array2d[_vec3d],
    body_com: wp.array[_vec3d],
    joint_child: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    child_offset: wp.array[wp.mat44d],
    subtree_offset: wp.array[wp.int32],
    subtree_body: wp.array[wp.int32],
    stride: int,
    express_in: int,
    out: wp.array3d[_f64],
):
    """Reduce required body wrenches to one reaction wrench per joint."""
    frame, joint = wp.tid()
    base = frame * stride
    child = joint_child[joint]
    parent = joint_parent[joint]
    child_pose = poses[base, child]
    joint_pose = child_pose * child_offset[joint]
    joint_position = _vec3d(joint_pose[0, 3], joint_pose[1, 3], joint_pose[2, 3])
    reaction_force = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    reaction_moment = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for index in range(subtree_offset[joint], subtree_offset[joint + 1]):
        body = subtree_body[index]
        body_pose = poses[base, body]
        rotation = wp.mat33d(
            body_pose[0, 0],
            body_pose[0, 1],
            body_pose[0, 2],
            body_pose[1, 0],
            body_pose[1, 1],
            body_pose[1, 2],
            body_pose[2, 0],
            body_pose[2, 1],
            body_pose[2, 2],
        )
        com_position = rotation * body_com[body]
        com_position += _vec3d(body_pose[0, 3], body_pose[1, 3], body_pose[2, 3])
        body_force = force[frame, body]
        reaction_force += body_force
        reaction_moment += torque[frame, body] + wp.cross(com_position - joint_position, body_force)
    if express_in != 0:
        expressed_pose = child_pose
        if express_in == 2:
            expressed_pose = poses[base, parent]
        rotation = wp.mat33d(
            expressed_pose[0, 0],
            expressed_pose[0, 1],
            expressed_pose[0, 2],
            expressed_pose[1, 0],
            expressed_pose[1, 1],
            expressed_pose[1, 2],
            expressed_pose[2, 0],
            expressed_pose[2, 1],
            expressed_pose[2, 2],
        )
        reaction_force = wp.transpose(rotation) * reaction_force
        reaction_moment = wp.transpose(rotation) * reaction_moment
    out[frame, joint, 0] = reaction_force[0]
    out[frame, joint, 1] = reaction_force[1]
    out[frame, joint, 2] = reaction_force[2]
    out[frame, joint, 3] = reaction_moment[0]
    out[frame, joint, 4] = reaction_moment[1]
    out[frame, joint, 5] = reaction_moment[2]


@wp.kernel
def apply_muscle_body_loads_kernel(
    poses: wp.array2d[wp.mat44d],
    point_location: wp.array2d[_vec3d],
    point_active: wp.array2d[wp.int32],
    point_body: wp.array[wp.int32],
    muscle_offset: wp.array[wp.int32],
    muscle_force: wp.array2d[_f32],
    body_com: wp.array[_vec3d],
    nmuscle: int,
    stride: int,
    force_out: wp.array2d[_vec3d],
    torque_out: wp.array2d[_vec3d],
):
    """Subtract resolved muscle path-point loads from required body wrenches."""
    frame, body = wp.tid()
    base = frame * stride
    applied_force = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    applied_torque = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
    for muscle in range(nmuscle):
        begin = muscle_offset[muscle]
        end = muscle_offset[muscle + 1]
        for point in range(begin, end):
            if point_body[point] != body or point_active[frame, point] == 0:
                continue
            pose = poses[base, body]
            local = point_location[frame, point]
            hp = pose * wp.vec4d(local[0], local[1], local[2], _f64(1.0))
            position = _vec3d(hp[0], hp[1], hp[2])
            direction = _vec3d(_f64(0.0), _f64(0.0), _f64(0.0))
            previous = int(-1)
            for candidate in range(begin, point):
                if point_active[frame, candidate] != 0:
                    previous = candidate
            if previous >= 0:
                previous_body = point_body[previous]
                previous_pose = poses[base, previous_body]
                previous_local = point_location[frame, previous]
                previous_h = previous_pose * wp.vec4d(
                    previous_local[0], previous_local[1], previous_local[2], _f64(1.0)
                )
                delta = _vec3d(previous_h[0], previous_h[1], previous_h[2]) - position
                distance = wp.length(delta)
                if distance > _f64(1.0e-12):
                    direction += delta / distance
            next_point = int(-1)
            for candidate in range(point + 1, end):
                if next_point < 0 and point_active[frame, candidate] != 0:
                    next_point = candidate
            if next_point >= 0:
                next_body = point_body[next_point]
                next_pose = poses[base, next_body]
                next_local = point_location[frame, next_point]
                next_h = next_pose * wp.vec4d(next_local[0], next_local[1], next_local[2], _f64(1.0))
                delta = _vec3d(next_h[0], next_h[1], next_h[2]) - position
                distance = wp.length(delta)
                if distance > _f64(1.0e-12):
                    direction += delta / distance
            point_force = wp.float64(muscle_force[frame, muscle]) * direction
            rotation = wp.mat33d(
                pose[0, 0],
                pose[0, 1],
                pose[0, 2],
                pose[1, 0],
                pose[1, 1],
                pose[1, 2],
                pose[2, 0],
                pose[2, 1],
                pose[2, 2],
            )
            com_position = rotation * body_com[body]
            com_position += _vec3d(pose[0, 3], pose[1, 3], pose[2, 3])
            applied_force += point_force
            applied_torque += wp.cross(position - com_position, point_force)
    force_out[frame, body] = force_out[frame, body] - applied_force
    torque_out[frame, body] = torque_out[frame, body] - applied_torque


@wp.kernel
def pack_body_position_report_kernel(
    poses: wp.array2d[wp.mat44d],
    center_of_mass_sum: wp.array[_vec3d],
    body_index: wp.array[wp.int32],
    nbody: int,
    total_mass: _f64,
    in_degrees: int,
    out: wp.array2d[_f64],
):
    """Pack body origins, XYZ Euler angles, and whole-body COM into one report."""
    frame = wp.tid()
    angle_scale = _f64(1.0)
    if in_degrees != 0:
        angle_scale = _f64(57.29577951308232)
    for column in range(nbody):
        pose = poses[frame, body_index[column]]
        beta = wp.asin(wp.clamp(pose[0, 2], _f64(-1.0), _f64(1.0)))
        alpha = _f64(0.0)
        gamma = _f64(0.0)
        if wp.abs(wp.cos(beta)) > _f64(1.0e-9):
            alpha = wp.atan2(-pose[1, 2], pose[2, 2])
            gamma = wp.atan2(-pose[0, 1], pose[0, 0])
        else:
            alpha = wp.atan2(pose[2, 1], pose[1, 1])
        offset = 6 * column
        out[frame, offset + 0] = pose[0, 3]
        out[frame, offset + 1] = pose[1, 3]
        out[frame, offset + 2] = pose[2, 3]
        out[frame, offset + 3] = angle_scale * alpha
        out[frame, offset + 4] = angle_scale * beta
        out[frame, offset + 5] = angle_scale * gamma
    com = center_of_mass_sum[frame]
    if total_mass > _f64(0.0):
        com /= total_mass
    out[frame, 6 * nbody + 0] = com[0]
    out[frame, 6 * nbody + 1] = com[1]
    out[frame, 6 * nbody + 2] = com[2]


@wp.kernel
def pack_body_rate_report_kernel(
    linear: wp.array2d[_vec3d],
    angular: wp.array2d[_vec3d],
    center_of_mass: wp.array[_vec3d],
    body_index: wp.array[wp.int32],
    nbody: int,
    in_degrees: int,
    out: wp.array2d[_f64],
):
    """Pack body linear/angular rates and whole-body COM rate into one report."""
    frame = wp.tid()
    angle_scale = _f64(1.0)
    if in_degrees != 0:
        angle_scale = _f64(57.29577951308232)
    for column in range(nbody):
        body = body_index[column]
        offset = 6 * column
        out[frame, offset + 0] = linear[frame, body][0]
        out[frame, offset + 1] = linear[frame, body][1]
        out[frame, offset + 2] = linear[frame, body][2]
        out[frame, offset + 3] = angle_scale * angular[frame, body][0]
        out[frame, offset + 4] = angle_scale * angular[frame, body][1]
        out[frame, offset + 5] = angle_scale * angular[frame, body][2]
    com = center_of_mass[frame]
    out[frame, 6 * nbody + 0] = com[0]
    out[frame, 6 * nbody + 1] = com[1]
    out[frame, 6 * nbody + 2] = com[2]


# --------------------------------------------------------------------------- #
# Shared host helpers.
# --------------------------------------------------------------------------- #
def euler_xyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    r"""Return the body-fixed XYZ Euler angles [rad] of a rotation matrix.

    Inverts ``euler_xyz_to_matrix``
    (``R = Rx(a) Ry(b) Rz(c)``), matching OpenSim's ``BodyKinematics``
    orientation convention. Near the ``b = +/- pi/2`` gimbal lock the ``a``/``c``
    split is resolved by setting ``c = 0``.

    Args:
        rotation: A 3x3 rotation matrix (body frame expressed in ground).

    Returns:
        Length-3 array ``(a, b, c)`` [rad].
    """
    r = np.asarray(rotation, dtype=float)
    b = np.arcsin(np.clip(r[0, 2], -1.0, 1.0))
    if abs(np.cos(b)) > 1.0e-9:
        a = np.arctan2(-r[1, 2], r[2, 2])
        c = np.arctan2(-r[0, 1], r[0, 0])
    else:  # gimbal lock: fold the a/c degeneracy onto a
        a = np.arctan2(r[2, 1], r[1, 1])
        c = 0.0
    return np.array([a, b, c])


def write_sto(storage: Storage, path: str | os.PathLike) -> None:
    """Write a :class:`~newton.opensim.Storage` to an OpenSim ``.sto`` file."""
    write_storage(
        path,
        storage.times,
        storage.labels,
        storage.data,
        name=storage.name,
        in_degrees=storage.in_degrees,
    )


def _load_storage(motion: Storage | str | os.PathLike) -> Storage:
    """Return ``motion`` as a :class:`Storage` (reading a ``.mot``/``.sto`` path)."""
    return motion if isinstance(motion, Storage) else read_storage(motion)


def _coordinate_columns(
    model: OsimModel, coordinate_names: list[str], is_rotational: list[bool], storage: Storage
) -> np.ndarray:
    """Return the storage columns in model coordinate order, in storage units.

    Missing coordinates fall back to each coordinate's default value (converted to
    degrees when the storage is in degrees and the coordinate is rotational),
    matching :meth:`InverseDynamics.solve_from_motion`.
    """
    times = np.asarray(storage.times, float)
    col_index = {lab: i for i, lab in enumerate(storage.labels)}
    defaults = {c.name: c.default_value for j in model.joints for c in j.coordinates}
    values = np.zeros((len(times), len(coordinate_names)))
    for i, name in enumerate(coordinate_names):
        if name in col_index:
            values[:, i] = storage.data[:, col_index[name]]
        else:
            default = defaults.get(name, 0.0)
            values[:, i] = np.rad2deg(default) if (is_rotational[i] and storage.in_degrees) else default
    return values


def _direct_coordinates(
    model: OsimModel, coordinate_names: list[str], motion_types: list[str], storage: Storage
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, q)`` from a motion in native units (radians/meters), unfiltered."""
    is_rot = [mt == "rotational" for mt in motion_types]
    values = _coordinate_columns(model, coordinate_names, is_rot, storage)
    q = values.copy()
    if storage.in_degrees:
        for i, rot in enumerate(is_rot):
            if rot:
                q[:, i] = np.deg2rad(values[:, i])
    return np.asarray(storage.times, float), q


def _differentiated_coordinates(
    model: OsimModel,
    coordinate_names: list[str],
    motion_types: list[str],
    storage: Storage,
    cutoff: float,
    output_times: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(times, q, qd, qdd)`` in native units via the OpenSim signal pipeline."""
    is_rot = [mt == "rotational" for mt in motion_types]
    values = _coordinate_columns(model, coordinate_names, is_rot, storage)
    times = np.asarray(storage.times, float)
    if output_times is None:
        output_times = times
    output_times = np.asarray(output_times, float)
    q, qd, qdd = differentiate_coordinates(
        times, values, is_rot, output_times=output_times, cutoff=cutoff, in_degrees=storage.in_degrees
    )
    return output_times, q, qd, qdd


# --------------------------------------------------------------------------- #
# BodyKinematics.
# --------------------------------------------------------------------------- #
class BodyKinematics:
    """OpenSim ``BodyKinematics``-style body pose report from a coordinate motion.

    Reports, per body, the ground position of the body-frame origin
    (``<body>_X/_Y/_Z`` [m]) and its body-fixed XYZ Euler orientation
    (``<body>_Ox/_Oy/_Oz`` [rad or deg]), plus the whole-body center of mass
    (``center_of_mass_X/_Y/_Z`` [m]). Positions and orientations come straight
    from :meth:`~newton.opensim.ForwardKinematics.body_transforms_batch`
    and the center of mass from
    :meth:`~newton.opensim.ForwardKinematics.center_of_mass_batch`.

    Args:
        model: Parsed model IR (see :func:`~newton.opensim.parse_osim`).
        device: Warp device for the kernels (``None`` for the CPU).

    Attributes:
        body_names: Reported body names (model bodies, excluding ``ground``).
        coordinate_names: Generalized coordinate names in model order.
        device: The Warp device the kernels run on.
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.coordinate_names = list(self.fk.coordinate_names)
        self.motion_types = [self.fk.coordinate_motion[c] for c in self.coordinate_names]
        self.body_names = [b.name for b in model.bodies]  # reported bodies, excluding ground
        self._body_indices = [self.fk.body_names.index(n) for n in self.body_names]
        self.d_body_indices = wp.array(self._body_indices, dtype=wp.int32, device=self.device)

    @property
    def position_labels(self) -> list[str]:
        """Return the position/orientation column labels (``<body>_X.._Oz`` + COM)."""
        labels: list[str] = []
        for name in self.body_names:
            labels += [f"{name}_X", f"{name}_Y", f"{name}_Z", f"{name}_Ox", f"{name}_Oy", f"{name}_Oz"]
        labels += ["center_of_mass_X", "center_of_mass_Y", "center_of_mass_Z"]
        return labels

    def solve(self, motion: Storage | str | os.PathLike, in_degrees: bool = True) -> Storage:
        """Return body positions/orientations for a coordinate motion.

        Args:
            motion: Coordinate trajectory (``.mot``/``.sto`` path or :class:`Storage`).
            in_degrees: Emit orientations in degrees (OpenSim default) rather than
                radians. Position columns are always in meters.

        Returns:
            A :class:`Storage` with the columns of :attr:`position_labels`; the
            ``inDegrees`` flag mirrors ``in_degrees``.
        """
        storage = _load_storage(motion)
        times, coords = _direct_coordinates(self.model, self.coordinate_names, self.motion_types, storage)
        q_wp = wp.array(np.ascontiguousarray(coords, dtype=np.float64), dtype=_f64, device=self.device)
        transforms_wp = self.fk._launch_body_transforms(q_wp)
        com_wp = self.fk._launch_center_of_mass(transforms_wp)
        n_frames = coords.shape[0]
        data_wp = wp.empty((n_frames, len(self.position_labels)), dtype=_f64, device=self.device)
        wp.launch(
            pack_body_position_report_kernel,
            dim=n_frames,
            inputs=[
                transforms_wp,
                com_wp,
                self.d_body_indices,
                len(self.body_names),
                _f64(self.fk.total_mass),
                int(in_degrees),
                data_wp,
            ],
            device=self.device,
        )
        data = data_wp.numpy()

        return Storage(
            times=times,
            labels=self.position_labels,
            data=data,
            in_degrees=in_degrees,
            name="BodyKinematics Positions",
        )

    def solve_velocities(
        self,
        motion: Storage | str | os.PathLike,
        cutoff: float = 6.0,
        in_degrees: bool = True,
        output_times: np.ndarray | None = None,
    ) -> Storage:
        """Return body linear/angular velocities for a coordinate motion.

        Linear velocity of each body-frame origin fills ``<body>_X/_Y/_Z`` [m/s]
        and angular velocity fills ``<body>_Ox/_Oy/_Oz`` [rad/s or deg/s]; the
        whole-body center-of-mass velocity fills ``center_of_mass_X/_Y/_Z`` [m/s].
        Speeds come from the OpenSim signal pipeline (Butterworth + GCVSpline).

        Args:
            motion: Coordinate trajectory (``.mot``/``.sto`` path or :class:`Storage`).
            cutoff: Butterworth low-pass cutoff [Hz]; ``<= 0`` disables filtering.
            in_degrees: Emit angular velocities in deg/s rather than rad/s.
            output_times: Optional explicit output times [s].
        """
        storage = _load_storage(motion)
        times, q, qd, _ = _differentiated_coordinates(
            self.model, self.coordinate_names, self.motion_types, storage, cutoff, output_times
        )
        q_wp = wp.array(np.ascontiguousarray(q, dtype=np.float64), dtype=_f64, device=self.device)
        qd_wp = wp.array(np.ascontiguousarray(qd, dtype=np.float64), dtype=_f64, device=self.device)
        angular_wp, linear_wp, transforms_wp = self.fk._launch_body_velocities(q_wp, qd_wp, 1.0e-6)
        com_wp = self.fk._launch_com_velocity(transforms_wp, angular_wp, linear_wp)
        data_wp = wp.empty((q.shape[0], len(self.position_labels)), dtype=_f64, device=self.device)
        wp.launch(
            pack_body_rate_report_kernel,
            dim=q.shape[0],
            inputs=[
                linear_wp,
                angular_wp,
                com_wp,
                self.d_body_indices,
                len(self.body_names),
                int(in_degrees),
                data_wp,
            ],
            device=self.device,
        )
        data = data_wp.numpy()

        return Storage(
            times=times, labels=self.position_labels, data=data, in_degrees=in_degrees, name="BodyKinematics Velocities"
        )

    def solve_accelerations(
        self,
        motion: Storage | str | os.PathLike,
        cutoff: float = 6.0,
        in_degrees: bool = True,
        output_times: np.ndarray | None = None,
    ) -> Storage:
        """Return body linear/angular accelerations for a coordinate motion.

        Linear acceleration of each body-frame origin fills ``<body>_X/_Y/_Z``
        [m/s^2] and angular acceleration fills ``<body>_Ox/_Oy/_Oz`` [rad/s^2 or
        deg/s^2]; the whole-body center-of-mass acceleration fills
        ``center_of_mass_X/_Y/_Z`` [m/s^2].

        Args:
            motion: Coordinate trajectory (``.mot``/``.sto`` path or :class:`Storage`).
            cutoff: Butterworth low-pass cutoff [Hz]; ``<= 0`` disables filtering.
            in_degrees: Emit angular accelerations in deg/s^2 rather than rad/s^2.
            output_times: Optional explicit output times [s].
        """
        storage = _load_storage(motion)
        times, q, qd, qdd = _differentiated_coordinates(
            self.model, self.coordinate_names, self.motion_types, storage, cutoff, output_times
        )
        q_wp = wp.array(np.ascontiguousarray(q, dtype=np.float64), dtype=_f64, device=self.device)
        qd_wp = wp.array(np.ascontiguousarray(qd, dtype=np.float64), dtype=_f64, device=self.device)
        qdd_wp = wp.array(np.ascontiguousarray(qdd, dtype=np.float64), dtype=_f64, device=self.device)
        angular_v_wp, _, transforms_wp = self.fk._launch_body_velocities(q_wp, qd_wp, 1.0e-6)
        angular_a_wp, linear_a_wp, _ = self.fk._launch_body_accelerations(
            q_wp, qd_wp, qdd_wp, 1.0e-4, body_x=transforms_wp
        )
        com_wp = self.fk._launch_com_acceleration(transforms_wp, angular_v_wp, angular_a_wp, linear_a_wp)
        data_wp = wp.empty((q.shape[0], len(self.position_labels)), dtype=_f64, device=self.device)
        wp.launch(
            pack_body_rate_report_kernel,
            dim=q.shape[0],
            inputs=[
                linear_a_wp,
                angular_a_wp,
                com_wp,
                self.d_body_indices,
                len(self.body_names),
                int(in_degrees),
                data_wp,
            ],
            device=self.device,
        )
        data = data_wp.numpy()

        return Storage(
            times=times,
            labels=self.position_labels,
            data=data,
            in_degrees=in_degrees,
            name="BodyKinematics Accelerations",
        )

    def write_sto(self, storage: Storage, path: str | os.PathLike) -> None:
        """Write a report :class:`Storage` to an OpenSim ``.sto`` file."""
        write_sto(storage, path)


# --------------------------------------------------------------------------- #
# MuscleAnalysis.
# --------------------------------------------------------------------------- #
class MuscleAnalysis:
    """OpenSim ``MuscleAnalysis``-style report from a coordinate motion + activations.

    Reports, per muscle (columns are muscle names), the muscle-tendon
    ``Length`` [m], ``FiberLength``/``TendonLength`` [m], ``NormalizedFiberLength``,
    ``PennationAngle`` [rad or deg], ``ActiveFiberForce``/``PassiveFiberForce``/
    ``FiberForce``/``TendonForce`` [N], and a ``MomentArm_<coordinate>`` table [m]
    for every coordinate. All quantities reuse
    :class:`~newton.opensim.MuscleForces` and
    :class:`~newton.opensim.MusclePaths`.

    Args:
        model: Parsed model IR.
        device: Warp device for the kernels (``None`` for the CPU).

    Attributes:
        muscle_names: Muscle names in output-column order.
        coordinate_names: Generalized coordinate names in model order.
        device: The Warp device the kernels run on.
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.forces = MuscleForces(model, device=device)
        self.paths = self.forces.paths
        self.device = self.forces.device
        self.muscle_names = list(self.forces.muscle_names)
        self.coordinate_names = list(self.paths.coordinate_names)
        self.motion_types = [self.paths.fk.coordinate_motion[c] for c in self.coordinate_names]
        self._cos_penn = np.asarray(self.forces._cos_penn, dtype=float)

    @property
    def num_muscles(self) -> int:
        """Number of muscles."""
        return len(self.muscle_names)

    def _resolve_activations(self, activations, n_frames: int, times: np.ndarray) -> np.ndarray:
        """Return activations shaped ``[n_frames, num_muscles]``.

        Accepts a scalar, a per-muscle vector, a ``[n_frames, num_muscles]`` array,
        or a :class:`Storage` whose columns (matched by muscle name) are linearly
        resampled at ``times``.
        """
        nm = self.num_muscles
        if isinstance(activations, Storage):
            col_index = {lab: i for i, lab in enumerate(activations.labels)}
            out = np.zeros((n_frames, nm))
            for m, name in enumerate(self.muscle_names):
                j = col_index.get(name)
                if j is None:  # tolerate OpenSim's "<muscle>.activation" style labels
                    j = col_index.get(f"{name}.activation") or col_index.get(f"{name}/activation")
                if j is None:
                    continue
                out[:, m] = np.interp(times, np.asarray(activations.times, float), activations.data[:, j])
            return out
        arr = np.asarray(activations, dtype=float)
        if arr.ndim == 0:
            return np.full((n_frames, nm), float(arr))
        if arr.ndim == 1:
            if arr.shape[0] != nm:
                raise ValueError(f"activations vector must have {nm} muscles, got {arr.shape[0]}")
            return np.repeat(arr[None, :], n_frames, axis=0)
        if arr.shape != (n_frames, nm):
            raise ValueError(f"activations must be [{n_frames}, {nm}], got {arr.shape}")
        return arr

    def solve(
        self,
        motion: Storage | str | os.PathLike,
        activations: float | np.ndarray | Storage = 1.0,
        cutoff: float = 6.0,
        include_speeds: bool = True,
        in_degrees: bool = True,
        output_times: np.ndarray | None = None,
    ) -> dict[str, Storage]:
        """Return a dict of MuscleAnalysis :class:`Storage` tables.

        Args:
            motion: Coordinate trajectory (``.mot``/``.sto`` path or :class:`Storage`).
            activations: Muscle activations in [0, 1] as a scalar, per-muscle vector,
                ``[n_frames, num_muscles]`` array, or a :class:`Storage` resampled by
                muscle name.
            cutoff: Butterworth low-pass cutoff [Hz] for the coordinate speeds.
            include_speeds: Use filtered coordinate speeds for the velocity-dependent
                force and fiber velocity; ``False`` treats every pose as isometric.
            in_degrees: Emit ``PennationAngle`` in degrees rather than radians.
            output_times: Optional explicit output times [s].

        Returns:
            A dict keyed by quantity name (``Length``, ``FiberLength``,
            ``TendonLength``, ``NormalizedFiberLength``, ``PennationAngle``,
            ``ActiveFiberForce``, ``PassiveFiberForce``, ``FiberForce``,
            ``TendonForce``, and one ``MomentArm_<coordinate>`` per coordinate),
            each a :class:`Storage` whose columns are :attr:`muscle_names`.
        """
        storage = _load_storage(motion)
        times, q, qd, _ = _differentiated_coordinates(
            self.model, self.coordinate_names, self.motion_types, storage, cutoff, output_times
        )
        speeds = qd if include_speeds else None
        act = self._resolve_activations(activations, q.shape[0], times)

        length, moment_arms, quantities = self.forces._analysis_quantities(act, q, speeds)
        fiber = quantities
        force = quantities
        pennation = fiber["pennation_angle"]
        tendon_length = length - fiber["fiber_length"] * np.cos(pennation)

        def _table(data: np.ndarray, name: str, degrees: bool = False) -> Storage:
            return Storage(times=times, labels=list(self.muscle_names), data=data, in_degrees=degrees, name=name)

        results: dict[str, Storage] = {
            "Length": _table(length, "MuscleAnalysis Length"),
            "FiberLength": _table(fiber["fiber_length"], "MuscleAnalysis FiberLength"),
            "TendonLength": _table(tendon_length, "MuscleAnalysis TendonLength"),
            "NormalizedFiberLength": _table(fiber["normalized_fiber_length"], "MuscleAnalysis NormalizedFiberLength"),
            "PennationAngle": _table(
                np.rad2deg(pennation) if in_degrees else pennation, "MuscleAnalysis PennationAngle", degrees=in_degrees
            ),
            "ActiveFiberForce": _table(force["active_fiber_force"], "MuscleAnalysis ActiveFiberForce"),
            "PassiveFiberForce": _table(force["passive_fiber_force"], "MuscleAnalysis PassiveFiberForce"),
            "FiberForce": _table(force["fiber_force"], "MuscleAnalysis FiberForce"),
            "TendonForce": _table(force["tendon_force"], "MuscleAnalysis TendonForce"),
        }
        for c, coord in enumerate(self.coordinate_names):
            results[f"MomentArm_{coord}"] = _table(moment_arms[:, :, c], f"MuscleAnalysis MomentArm_{coord}")
        return results

    def write_sto(
        self, results: dict[str, Storage], directory: str | os.PathLike, prefix: str = "MuscleAnalysis"
    ) -> None:
        """Write every report table to ``{directory}/{prefix}_{quantity}.sto``."""
        os.makedirs(directory, exist_ok=True)
        for quantity, storage in results.items():
            write_sto(storage, os.path.join(directory, f"{prefix}_{quantity}.sto"))


# --------------------------------------------------------------------------- #
# JointReaction.
# --------------------------------------------------------------------------- #
class JointReaction:
    r"""OpenSim ``JointReaction``-style constraint reactions from a coordinate motion.

    For each joint the reaction transmitted from parent to child is the resultant
    spatial force the joint constraint must supply to move the child subtree,

    .. math::

        F_j = \sum_{b \in \mathrm{subtree}(c)} (m_b a_b - m_b g - F^{\mathrm{app}}_b),

    with the moment taken about the joint center. The per-body required spatial
    force ``m_b a_b - m_b g - F^{\mathrm{app}}_b`` is exactly what
    ``bodyforce_kernel`` forms (it already
    subtracts applied loads), so summing it over the child subtree cancels every
    internal joint/muscle wrench and leaves the reaction at the single joint that
    crosses the cut (Newton's third law). Applied loads are gravity (always),
    optional :class:`~newton.opensim.ExternalLoads`, and optional
    muscle body loads reconstructed from the muscle path geometry.

    Args:
        model: Parsed model IR.
        device: Warp device for the kernels (``None`` for the CPU).

    Attributes:
        joint_names: Reported joint names in model order.
        coordinate_names: Generalized coordinate names in model order.
        device: The Warp device the kernels run on.

    Note:
        The muscle body loads use the straight segment between consecutive active
        path points for each point's force direction; for muscles that wrap over a
        surface this direction is approximate (the muscle-tendon force magnitude
        and lengths remain exact). When no activations are supplied, muscle forces
        are treated as zero, so the reaction is the pure inertial + gravitational
        (and external) inter-segmental load.
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.id = InverseDynamics(model, device=device)
        self.fk = self.id.fk
        self.device = self.id.device
        self.coordinate_names = list(self.id.coordinate_names)
        self.motion_types = list(self.id.motion_types)
        self.ncoord = self.id.ncoord
        self._body_index = dict(self.id._body_index)
        self.muscles = MuscleForces(model, device=device) if model.muscles else None

        # Joint topology: reported joints, child/parent body indices, and the child
        # child-frame offset (X_BM) used to place the joint center in ground.
        self.joint_names = [j.name for j in model.joints]
        self._joint_child = [self._body_index[j.child_body] for j in model.joints]
        self._joint_parent = [self._body_index[j.parent_body] for j in model.joints]
        self._child_body_names = [j.child_body for j in model.joints]
        self._parent_body_names = [j.parent_body for j in model.joints]
        self._x_bm = [
            make_transform(euler_xyz_to_matrix(*j.child_transform.orientation), j.child_transform.translation)
            for j in model.joints
        ]
        # Child-subtree membership (body index sets) for each joint.
        children: dict[int, list[int]] = {}
        for j in model.joints:
            children.setdefault(self._body_index[j.parent_body], []).append(self._body_index[j.child_body])
        self._subtrees = [self._collect_subtree(self._body_index[j.child_body], children) for j in model.joints]
        subtree_offset = [0]
        subtree_body: list[int] = []
        for subtree in self._subtrees:
            subtree_body.extend(subtree)
            subtree_offset.append(len(subtree_body))
        self.d_joint_child = wp.array(self._joint_child, dtype=wp.int32, device=self.device)
        self.d_joint_parent = wp.array(self._joint_parent, dtype=wp.int32, device=self.device)
        self.d_x_bm = wp.array(
            np.asarray(self._x_bm, dtype=np.float64).reshape(-1, 4, 4), dtype=wp.mat44d, device=self.device
        )
        self.d_subtree_offset = wp.array(subtree_offset, dtype=wp.int32, device=self.device)
        self.d_subtree_body = wp.array(subtree_body, dtype=wp.int32, device=self.device)

    @staticmethod
    def _collect_subtree(root: int, children: dict[int, list[int]]) -> list[int]:
        """Return the body indices of the subtree rooted at ``root`` (inclusive)."""
        stack = [root]
        members: list[int] = []
        while stack:
            b = stack.pop()
            members.append(b)
            stack.extend(children.get(b, ()))
        return members

    def reaction_labels(self, express_in: str = "ground") -> list[str]:
        """Return the reaction column labels for a chosen expression frame."""
        labels: list[str] = []
        for name, child in zip(self.joint_names, self._child_body_names, strict=True):
            stem = f"{name}_on_{child}_in_{express_in}"
            labels += [f"{stem}_fx", f"{stem}_fy", f"{stem}_fz", f"{stem}_mx", f"{stem}_my", f"{stem}_mz"]
        return labels

    def _resolve_muscle_activations(self, activations, times: np.ndarray) -> np.ndarray:
        """Return joint-reaction muscle activations in model order."""
        n_frames = len(times)
        nm = self.muscles.num_muscles
        if isinstance(activations, Storage):
            columns = {label: index for index, label in enumerate(activations.labels)}
            out = np.zeros((n_frames, nm))
            for muscle, name in enumerate(self.muscles.muscle_names):
                index = columns.get(name)
                if index is None:
                    index = columns.get(f"{name}.activation")
                if index is None:
                    index = columns.get(f"{name}/activation")
                if index is not None:
                    out[:, muscle] = np.interp(times, np.asarray(activations.times), activations.data[:, index])
            return out
        values = np.asarray(activations, dtype=np.float32)
        if values.ndim == 0:
            return np.full((n_frames, nm), float(values), dtype=np.float32)
        if values.ndim == 1:
            if values.shape[0] != nm:
                raise ValueError(f"activations vector must have {nm} muscles, got {values.shape[0]}")
            return np.repeat(values[None], n_frames, axis=0)
        if values.shape != (n_frames, nm):
            raise ValueError(f"activations must be [{n_frames}, {nm}], got {values.shape}")
        return values

    def _per_body_spatial_forces(
        self, coords: np.ndarray, speeds: np.ndarray, accels: np.ndarray, activations, times, external_loads, h: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return device ``(force, torque, poses, stride)`` from :func:`bodyforce_kernel`.

        ``force[f, b]`` and ``torque[f, b]`` (about the body COM, in ground) are the
        required spatial force net of gravity and every applied load (external +
        muscle), i.e. the joint reaction wrench acting on body ``b``.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        n_frames, nc = coords.shape
        nb = self.fk.nbody
        stride = 3 + 2 * nc
        d_q = wp.array(coords, dtype=_f64, device=self.device)
        d_qd = wp.array(np.ascontiguousarray(speeds, dtype=np.float64), dtype=_f64, device=self.device)
        d_qdd = wp.array(np.ascontiguousarray(accels, dtype=np.float64), dtype=_f64, device=self.device)
        q_wp = wp.empty((n_frames * stride, nc), dtype=_f64, device=self.device)
        wp.launch(
            id_stencil_kernel,
            dim=(n_frames, stride, nc),
            inputs=[d_q, d_qd, d_qdd, _f64(h), _f64(1.0e-6), nc],
            outputs=[q_wp],
            device=self.device,
        )
        poses = self.fk._launch_body_transforms(q_wp)

        # Assemble applied external loads (ground reactions + muscle body loads).
        ext_bodies: list[int] = []
        wrench_blocks: list[np.ndarray] = []
        if external_loads is not None:
            bodies, ext_wrench = external_loads.sample(times)  # [frames, n_ext, 9]
            ext_bodies.extend(self._body_index[b] for b in bodies)
            if ext_wrench.shape[1]:
                wrench_blocks.append(np.ascontiguousarray(ext_wrench, dtype=float))
        if wrench_blocks:
            wrench = np.concatenate(wrench_blocks, axis=1)
            ext_idx = np.asarray(ext_bodies, dtype=np.int32)
        else:
            wrench = np.zeros((n_frames, 0, 9))
            ext_idx = np.zeros(0, dtype=np.int32)

        d_ext = wp.array(ext_idx, dtype=wp.int32, device=self.device)
        d_wrench = wp.array(np.ascontiguousarray(wrench, dtype=float), dtype=_f64, device=self.device)
        torque = wp.empty((n_frames, nb), dtype=_vec3d, device=self.device)
        force = wp.empty((n_frames, nb), dtype=_vec3d, device=self.device)
        wp.launch(
            bodyforce_kernel,
            dim=(n_frames, nb),
            inputs=[
                poses,
                self.id._mass,
                self.id._rcom,
                self.id._inertia,
                _vec3d(*self.id.gravity),
                stride,
                len(ext_idx),
                d_ext,
                d_wrench,
                _f64(h),
            ],
            outputs=[torque, force],
            device=self.device,
        )
        if self.muscles is not None and activations is not None:
            activation_values = self._resolve_muscle_activations(activations, times)
            d_activation = wp.array(
                np.ascontiguousarray(activation_values, dtype=np.float32), dtype=_f32, device=self.device
            )
            muscle_force = self.muscles._forces_device(d_activation, d_q, d_qd)
            point_location, point_active = self.muscles.paths._sample_points(d_q)
            wp.launch(
                apply_muscle_body_loads_kernel,
                dim=(n_frames, nb),
                inputs=[
                    poses,
                    point_location,
                    point_active,
                    self.muscles.paths.d_point_body,
                    self.muscles.paths.d_musc_off,
                    muscle_force,
                    self.id._rcom,
                    self.muscles.num_muscles,
                    stride,
                    force,
                    torque,
                ],
                device=self.device,
            )
        return force, torque, poses, stride

    def solve(
        self,
        motion: Storage | str | os.PathLike,
        activations: float | np.ndarray | Storage | None = None,
        external_loads: ExternalLoads | str | os.PathLike | None = None,
        express_in: str = "ground",
        cutoff: float = 6.0,
        h: float = 1.0e-4,
        output_times: np.ndarray | None = None,
    ) -> Storage:
        """Return joint reaction loads for a coordinate motion.

        Args:
            motion: Coordinate trajectory (``.mot``/``.sto`` path or :class:`Storage`).
            activations: Optional muscle activations (scalar, per-muscle vector,
                ``[n_frames, num_muscles]`` array, or a :class:`Storage`); ``None``
                treats muscle forces as zero.
            external_loads: Optional :class:`ExternalLoads` or path to an
                ``ExternalLoads`` setup XML (e.g. ground reactions).
            express_in: Frame the reaction is expressed in: ``"ground"``,
                ``"child"``, or ``"parent"``.
            cutoff: Butterworth low-pass cutoff [Hz] for the coordinate derivatives.
            h: Finite-difference step for velocities/accelerations [s].
            output_times: Optional explicit output times [s].

        Returns:
            A :class:`Storage` with, per joint, the reaction force [N]
            (``..._fx/_fy/_fz``) and moment [N·m] (``..._mx/_my/_mz``) applied to the
            child by the joint, about the joint center, in the requested frame.
        """
        if express_in not in ("ground", "child", "parent"):
            raise ValueError("express_in must be 'ground', 'child', or 'parent'")
        if external_loads is not None and not isinstance(external_loads, ExternalLoads):
            external_loads = read_external_loads(external_loads)

        storage = _load_storage(motion)
        times, q, qd, qdd = _differentiated_coordinates(
            self.model, self.coordinate_names, self.motion_types, storage, cutoff, output_times
        )
        force, torque, poses, stride = self._per_body_spatial_forces(q, qd, qdd, activations, times, external_loads, h)
        n_frames = q.shape[0]
        if self.joint_names:
            report = wp.empty((n_frames, len(self.joint_names), 6), dtype=_f64, device=self.device)
            wp.launch(
                joint_reaction_kernel,
                dim=(n_frames, len(self.joint_names)),
                inputs=[
                    poses,
                    force,
                    torque,
                    self.id._rcom,
                    self.d_joint_child,
                    self.d_joint_parent,
                    self.d_x_bm,
                    self.d_subtree_offset,
                    self.d_subtree_body,
                    stride,
                    {"ground": 0, "child": 1, "parent": 2}[express_in],
                    report,
                ],
                device=self.device,
            )
            data = report.numpy().reshape(n_frames, -1)
        else:
            data = np.zeros((n_frames, 0))

        return Storage(
            times=times,
            labels=self.reaction_labels(express_in),
            data=data,
            in_degrees=False,
            name="Joint Reaction Loads",
        )

    def write_sto(self, storage: Storage, path: str | os.PathLike) -> None:
        """Write a report :class:`Storage` to an OpenSim ``.sto`` file."""
        write_sto(storage, path)
