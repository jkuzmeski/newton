# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive an identified shoe with running motion and a vertical impedance leg."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.digital_shoe import FoundationConfig, MidsoleFoundation, load_artifact
from projects.digital_shoe.rendering import attached_column_endpoints, column_colors
from projects.digital_shoe.runtime import SurroundConfig

from .orientation import orient_shoe
from .profile import load_profile
from .report import write_report
from .trajectory import TrajectoryCubic

# Device reference columns: foot x/z, pitch, COM x/z, foot vx/vz, pitch rate,
# COM vx/vz, foot az, left Fz, other Fz, leg reference length/rate.


@wp.kernel
def _prescribe_axes(
    index: int,
    reference: wp.array2d[wp.float32],
    replay: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Prescribe track and pitch axes; leave vertical states free in impedance mode."""
    foot_z = wp.transform_get_translation(body_q[0])[2]
    com_z = wp.transform_get_translation(body_q[1])[2]
    foot_vz = wp.spatial_top(body_qd[0])[2]
    com_vz = wp.spatial_top(body_qd[1])[2]
    if replay != 0:
        foot_z = reference[index, 1]
        com_z = reference[index, 4]
        foot_vz = reference[index, 6]
        com_vz = reference[index, 9]
    body_q[0] = wp.transform(
        wp.vec3(reference[index, 0], 0.0, foot_z),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), reference[index, 2]),
    )
    body_q[1] = wp.transform(wp.vec3(reference[index, 3], 0.0, com_z), wp.quat_identity())
    body_qd[0] = wp.spatial_vector(wp.vec3(reference[index, 5], 0.0, foot_vz), wp.vec3(0.0, reference[index, 7], 0.0))
    body_qd[1] = wp.spatial_vector(wp.vec3(reference[index, 8], 0.0, com_vz), wp.vec3(0.0))


@wp.kernel
def _apply_leg(
    index: int,
    reference: wp.array2d[wp.float32],
    foot_mass: float,
    stiffness: float,
    damping: float,
    force_limit: float,
    gravity: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.float32],
):
    """Apply equal-and-opposite leg forces and explicit opposite-foot support."""
    length = wp.transform_get_translation(body_q[1])[2] - wp.transform_get_translation(body_q[0])[2]
    rate = wp.spatial_top(body_qd[1])[2] - wp.spatial_top(body_qd[0])[2]
    error = length - reference[index, 13]
    desired_rate = reference[index, 14]
    gain = float(1.0)
    gain_rate = float(0.0)
    retract_force = float(0.0)
    if reference.shape[1] >= 22:
        gain = reference[index, 19]
        gain_rate = reference[index, 20]
        retract_force = reference[index, 21]
    k = stiffness * gain
    b = damping * gain
    feedforward = reference[index, 11] - foot_mass * (gravity + reference[index, 10]) - retract_force
    raw_force = feedforward - k * error + b * (desired_rate - rate)
    force = wp.clamp(raw_force, -force_limit, force_limit)
    wp.atomic_add(body_f, 0, wp.spatial_vector(wp.vec3(0.0, 0.0, -force), wp.vec3(0.0)))
    wp.atomic_add(body_f, 1, wp.spatial_vector(wp.vec3(0.0, 0.0, force + reference[index, 12]), wp.vec3(0.0)))
    # Account separately for moving spring rest length, active feedforward,
    # physical damper dissipation, and any force-limit intervention.
    diagnostics[0] = force
    diagnostics[1] = (
        (feedforward + b * desired_rate + force - raw_force) * rate
        - k * error * desired_rate
        + 0.5 * stiffness * gain_rate * error * error
    )
    diagnostics[2] = -b * rate * rate
    diagnostics[3] = 0.5 * k * error * error
    diagnostics[4] = float(wp.abs(force - raw_force) > 1.0e-4)


@wp.kernel
def _record_sample(
    index: int,
    reference: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    diagnostics: wp.array[wp.float32],
    com_mass: float,
    pitch_inertia: float,
    foot_mass: float,
    replay: int,
    gravity: float,
    trace: wp.array2d[wp.float32],
):
    """Record pre-integration samples at their actual evaluation time."""
    foot_z = wp.transform_get_translation(body_q[0])[2]
    com_z = wp.transform_get_translation(body_q[1])[2]
    com_vz = wp.spatial_top(body_qd[1])[2]
    fz = force[0]
    trace[index, 0] = fz
    trace[index, 1] = 0.0
    if fz > 1.0:
        trace[index, 1] = cop_moment[0][0] / fz
    trace[index, 2] = foot_z
    trace[index, 3] = com_z
    trace[index, 4] = com_vz
    trace[index, 5] = diagnostics[0]
    trace[index, 6] = diagnostics[1]
    trace[index, 7] = diagnostics[2]
    # Isotropic fixture inertia: prescribed pitch drive balances contact torque.
    trace[index, 8] = (pitch_inertia * reference[index, 15] - wp.spatial_bottom(body_f[0])[1]) * reference[index, 7]
    trace[index, 18] = pitch_inertia * reference[index, 15] - wp.spatial_bottom(body_f[0])[1]
    trace[index, 9] = contact_power[0]
    trace[index, 10] = com_mass * (gravity * com_z + 0.5 * com_vz * com_vz)
    trace[index, 11] = compression[0]
    trace[index, 12] = diagnostics[4]
    trace[index, 13] = diagnostics[3]
    trace[index, 14] = reference[index, 12] * com_vz
    trace[index, 15] = wp.spatial_top(body_qd[0])[2]
    trace[index, 16] = 0.0
    if replay != 0:
        foot_drive = foot_mass * (reference[index, 10] + gravity) - fz + diagnostics[0]
        com_drive = com_mass * (reference[index, 16] + gravity) - diagnostics[0] - reference[index, 12]
        trace[index, 16] = foot_drive * reference[index, 6] + com_drive * reference[index, 9]
    trace[index, 17] = (
        foot_mass * reference[index, 17] * reference[index, 5] + com_mass * reference[index, 18] * reference[index, 8]
    )


@wp.kernel
def _free_column_metrics(
    driven: wp.array[wp.int32],
    compression: wp.array[wp.float32],
    rest: wp.array[wp.float32],
    forces: wp.array[wp.vec3],
    velocity: wp.array[wp.float32],
    metrics: wp.array[wp.float32],
):
    """Summarize the passive outer region without hiding its extremes."""
    i = wp.tid()
    if driven[i] != 0:
        wp.atomic_add(metrics, 0, forces[i][2])
        return
    wp.atomic_add(metrics, 1, forces[i][2])
    wp.atomic_max(metrics, 2, compression[i] / rest[i])
    wp.atomic_max(metrics, 3, wp.abs(velocity[i]))
    if forces[i][2] > 0.01:
        wp.atomic_add(metrics, 4, 1.0)


@wp.kernel
def _constrain_planar_axes(
    index: int,
    reference: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Leave both X/Z states dynamic; the external robot motor prescribes only pitch."""
    a = wp.transform_get_translation(body_q[0])
    c = wp.transform_get_translation(body_q[1])
    va = wp.spatial_top(body_qd[0])
    vc = wp.spatial_top(body_qd[1])
    body_q[0] = wp.transform(
        wp.vec3(a[0], 0.0, a[2]), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), reference[index, 2])
    )
    body_q[1] = wp.transform(wp.vec3(c[0], 0.0, c[2]), wp.quat_identity())
    body_qd[0] = wp.spatial_vector(wp.vec3(va[0], 0.0, va[2]), wp.vec3(0.0, reference[index, 7], 0.0))
    body_qd[1] = wp.spatial_vector(wp.vec3(vc[0], 0.0, vc[2]), wp.vec3(0.0))


@wp.kernel
def _apply_planar_leg(
    index: int,
    reference: wp.array2d[wp.float32],
    mass: float,
    foot_mass: float,
    stiffness: float,
    damping: float,
    force_limit: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.float32],
):
    """Couple the actual endpoints with a central axial force, not two unrelated Z sliders."""
    r = wp.transform_get_translation(body_q[1]) - wp.transform_get_translation(body_q[0])
    length = wp.max(wp.length(r), 1.0e-6)
    n = r / length
    relative_velocity = wp.spatial_top(body_qd[1]) - wp.spatial_top(body_qd[0])
    rate = wp.dot(n, relative_velocity)
    error = length - reference[index, 13]
    desired_rate = reference[index, 14]
    gain, gain_rate = reference[index, 19], reference[index, 20]
    k, b = stiffness * gain, damping * gain
    other = wp.vec3(reference[index, 23], 0.0, reference[index, 12])
    measured = wp.vec3(reference[index, 22], 0.0, reference[index, 11])
    target = (1.0 - foot_mass / mass) * measured - (foot_mass / mass) * other
    # A single axial actuator cannot independently impose both measured force components.
    feedforward = reference[index, 24] * wp.dot(target, n) - reference[index, 21] / wp.max(n[2], 0.25)
    raw = feedforward - k * error + b * (desired_rate - rate)
    force = wp.clamp(raw, -force_limit, force_limit)
    f = force * n
    wp.atomic_add(body_f, 0, wp.spatial_vector(-f, wp.vec3(0.0)))
    wp.atomic_add(body_f, 1, wp.spatial_vector(f + other, wp.vec3(0.0)))
    diagnostics[0] = force
    diagnostics[1] = (
        (feedforward + b * desired_rate + force - raw) * rate
        - k * error * desired_rate
        + 0.5 * stiffness * gain_rate * error * error
    )
    diagnostics[2] = -b * rate * rate
    diagnostics[3] = 0.5 * k * error * error
    diagnostics[4] = float(wp.abs(force - raw) > 1.0e-4)
    diagnostics[5] = length
    diagnostics[6] = rate
    diagnostics[7] = f[0]
    diagnostics[8] = f[2]


@wp.kernel
def _contact_motion_metrics(
    dt: float,
    mu: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    anchors: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    old_anchor: wp.array[wp.vec2],
    old_active: wp.array[wp.int32],
    ground_anchor: wp.array[wp.vec2],
    active: wp.array[wp.int32],
    metrics: wp.array[wp.float32],
):
    """Separate material-point speed, elastic shear, and plastic-anchor drift."""
    i = wp.tid()
    f = forces[i]
    if f[2] > 0.01:
        p = wp.transform_point(body_q[0], anchors[i])
        r = p - wp.transform_get_translation(body_q[0])
        v = wp.spatial_top(body_qd[0]) + wp.cross(wp.spatial_bottom(body_qd[0]), r)
        speed = wp.length(wp.vec2(v[0], v[1]))
        drift, extension = float(0.0), float(0.0)
        if active[i] != 0:
            extension = wp.length(wp.vec2(p[0], p[1]) - ground_anchor[i])
            if old_active[i] != 0:
                drift = wp.length(ground_anchor[i] - old_anchor[i]) / dt
        wp.atomic_add(metrics, 0, f[2])
        wp.atomic_add(metrics, 1, f[2] * speed)
        wp.atomic_add(metrics, 2, f[2] * drift)
        wp.atomic_add(metrics, 3, f[2] * extension)
        if drift > 0.001:
            wp.atomic_add(metrics, 4, f[2])
        wp.atomic_add(metrics, 5, f[0] * v[0] + f[1] * v[1])
        if mu > 0.0:
            wp.atomic_max(metrics, 6, wp.length(wp.vec2(f[0], f[1])) / (mu * f[2]))


@wp.kernel
def _draw_friction_columns(
    body_q: wp.array[wp.transform],
    anchors: wp.array[wp.vec3],
    rest: wp.array[wp.float32],
    forces: wp.array[wp.vec3],
    free_top: wp.array[wp.float32],
    ground_anchor: wp.array[wp.vec2],
    active: wp.array[wp.int32],
    bottom_out: wp.array[wp.vec3],
    top_out: wp.array[wp.vec3],
):
    """Draw each column where it actually is, including a lifted outer surface.

    The shoe carries its bed against a ground plane at ``z = 0``, so the rigid free
    top is zero and the relaxed one the foundation publishes is exactly minus the
    outer surface lift.
    """
    i = wp.tid()
    bottom = wp.transform_point(body_q[0], anchors[i])
    top = wp.transform_point(body_q[0], anchors[i] + wp.vec3(0.0, 0.0, rest[i]))
    lift = -free_top[i]
    bz = bottom[2] + lift
    tz = top[2] + lift
    # A loaded outsole cannot pass through the floor; an unloaded one leaves it.
    if bz < 0.0:
        bz = 0.0
    if tz < bz:
        tz = bz
    if forces[i][2] > 0.01 and active[i] != 0:
        bottom_out[i] = wp.vec3(ground_anchor[i][0], ground_anchor[i][1], bz)
    else:
        bottom_out[i] = wp.vec3(bottom[0], bottom[1], bz)
    top_out[i] = wp.vec3(top[0], top[1], tz)


@wp.kernel
def _record_planar_sample(
    index: int,
    reference: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    force: wp.array[wp.vec3],
    pressed: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    diagnostics: wp.array[wp.float32],
    contact_metrics: wp.array[wp.float32],
    free_metrics: wp.array[wp.float32],
    com_mass: float,
    pitch_inertia: float,
    gravity: float,
    trace: wp.array2d[wp.float32],
):
    """Record actual planar motion and friction, including all external motor work."""
    a = wp.transform_get_translation(body_q[0])
    c = wp.transform_get_translation(body_q[1])
    va = wp.spatial_top(body_qd[0])
    vc = wp.spatial_top(body_qd[1])
    f = force[0]
    torque = pitch_inertia * reference[index, 15] - wp.spatial_bottom(body_f[0])[1]
    trace[index, 0] = f[2]
    # The moment is weighted by the pressed load, so the centroid must divide by
    # the same quantity. Dividing by the net wrench, which also carries the pull
    # of lifted columns, drove the reported centre of pressure off the shoe.
    trace[index, 1] = 0.0
    if pressed[0] > 1.0:
        trace[index, 1] = cop_moment[0][0] / pressed[0]
    trace[index, 2] = a[2]
    trace[index, 3] = c[2]
    trace[index, 4] = vc[2]
    trace[index, 5] = diagnostics[0]
    trace[index, 6] = diagnostics[1]
    trace[index, 7] = diagnostics[2]
    trace[index, 8] = torque * reference[index, 7]
    trace[index, 9] = contact_power[0]
    trace[index, 10] = com_mass * (gravity * c[2] + 0.5 * vc[2] * vc[2])
    trace[index, 11] = compression[0]
    trace[index, 12] = diagnostics[4]
    trace[index, 13] = diagnostics[3]
    trace[index, 14] = reference[index, 12] * vc[2] + reference[index, 23] * vc[0]
    trace[index, 16] = 0.0
    trace[index, 17] = 0.0
    trace[index, 15] = va[2]
    trace[index, 18] = torque
    trace[index, 19] = a[0]
    trace[index, 20] = c[0]
    trace[index, 21] = va[0]
    trace[index, 22] = vc[0]
    trace[index, 23] = f[0]
    trace[index, 24] = f[1]
    trace[index, 25] = diagnostics[5]
    trace[index, 26] = diagnostics[6]
    trace[index, 27] = diagnostics[7]
    trace[index, 28] = diagnostics[8]
    normal = wp.max(contact_metrics[0], 1.0e-9)
    trace[index, 29] = contact_metrics[1] / normal
    trace[index, 30] = contact_metrics[2] / normal
    trace[index, 31] = contact_metrics[3] / normal
    trace[index, 32] = contact_metrics[4] / normal
    trace[index, 33] = contact_metrics[5]
    trace[index, 34] = contact_metrics[6]
    trace[index, 35] = free_metrics[0]
    trace[index, 36] = free_metrics[1]
    trace[index, 37] = free_metrics[2]
    trace[index, 38] = free_metrics[3]
    trace[index, 39] = free_metrics[4]
    trace[index, 40] = pressed[0]


def _curve(values: np.ndarray, source_time: np.ndarray, time: np.ndarray):
    """Interpolate optical knots with a C1 Hermite curve and analytic derivatives."""
    slopes = np.gradient(values, source_time, edge_order=2)
    index = np.clip(np.searchsorted(source_time, time, side="right") - 1, 0, len(source_time) - 2)
    interval = source_time[index + 1] - source_time[index]
    u = (time - source_time[index]) / interval
    y0, y1 = values[index], values[index + 1]
    d0, d1 = slopes[index] * interval, slopes[index + 1] * interval
    a = 2 * y0 - 2 * y1 + d0 + d1
    b = -3 * y0 + 3 * y1 - 2 * d0 - d1
    position = ((a * u + b) * u + d0) * u + y0
    velocity = ((3 * a * u + 2 * b) * u + d0) / interval
    acceleration = (6 * a * u + 2 * b) / interval**2
    return position, velocity, acceleration


def _camera(eye, target):
    direction = np.asarray(target) - np.asarray(eye)
    direction /= np.linalg.norm(direction)
    return (
        wp.vec3(*eye),
        float(np.degrees(np.arcsin(direction[2]))),
        float(np.degrees(np.arctan2(direction[1], direction[0]))),
    )


class Example:
    """Run a foot and body lump with a geometric leg, not a human skeleton.

    In planar mode both X/Z states respond to forces. An external ideal robot
    motor prescribes foot pitch, while the out-of-plane axes remain constrained.
    The old vertical-only setup is retained as an explicit comparison mode.
    """

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.args = args
        if (args.screenshot or args.record_gif) and not hasattr(viewer, "get_frame"):
            raise ValueError("Screenshots and GIF recording require --viewer gl")
        self.reference_mode = args.reference_mode
        self.planar = args.dynamics == "planar" and self.reference_mode == "pitch"
        if self.reference_mode == "pitch" and args.mode != "impedance":
            raise ValueError(
                "Pitch mode has no vertical replay trajectory; use --reference-mode markers for legacy replay"
            )
        self.profile = load_profile(args.profile)
        if self.reference_mode == "pitch" and self.profile["schema_version"] != "impedance_stance_2":
            raise ValueError(
                "Pitch mode requires a heel-cluster v2 profile; export stance_pitch.json or select --reference-mode markers"
            )
        self.gravity = float(self.profile["provenance"]["com_surrogate"]["gravity_m_s2"])
        source_rate = float(self.profile["provenance"]["kinematics"]["source_rate_hz"])
        if args.kinematic_rate_hz is not None and not math.isclose(args.kinematic_rate_hz, source_rate):
            raise ValueError("Kinematic rate must match the optical acquisition rate in the profile")
        args.kinematic_rate_hz = source_rate
        if not math.isfinite(args.initial_clearance) or args.initial_clearance < 0.0:
            raise ValueError("Initial clearance must be finite and nonnegative; preloaded history is not supported")
        self.shoe, self.shoe_orientation = orient_shoe(
            load_artifact(args.artifact), target_side=args.shoe_side, source_side=args.source_shoe_side
        )
        self.ankle_mount = np.asarray(
            args.ankle_mount if self.reference_mode == "pitch" else (0.0, 0.0, 0.0), dtype=float
        )
        if self.ankle_mount.shape != (3,) or not np.all(np.isfinite(self.ankle_mount)):
            raise ValueError("The mechanical ankle mount must be three finite shoe-local coordinates")
        self.device = wp.get_device()
        self.mode = args.mode
        self.mass = float(self.profile["mass_kg"])
        self.foot_mass = float(args.foot_mass)
        self.com_mass = self.mass - self.foot_mass
        params = [
            self.mass,
            self.foot_mass,
            self.com_mass,
            args.stiffness,
            args.damping,
            args.shoe_stiffness_scale,
            args.substeps,
            args.force_limit_bw,
            args.kinematic_rate_hz,
            self.gravity,
        ]
        if not np.all(np.isfinite(params)) or min(params) <= 0.0:
            raise ValueError("Masses, gains, scale, substeps, and force limit must be finite and positive")
        if not np.isfinite(args.ankle_x) or not np.isfinite(args.track_speed):
            raise ValueError("Track settings must be finite")
        if self.planar and args.track_speed != 0.0:
            raise ValueError(
                "Planar motion is dynamic; use --ankle-entry-vx for an initial velocity, not --track-speed"
            )
        if (
            not np.all(np.isfinite([args.com_offset_x, args.friction_mu, args.contact_kt, args.contact_kd]))
            or min(args.friction_mu, args.contact_kt, args.contact_kd) < 0
        ):
            raise ValueError(
                "Planar initial offset and contact parameters must be finite; friction parameters nonnegative"
            )
        self.frame_dt = 1.0 / 120.0
        self.sim_dt = self.frame_dt / args.substeps
        self.duration = float(self.profile["time_s"][-1])
        self.sample_count = math.ceil(self.duration / self.sim_dt) + 1
        # Reach the exact profile endpoint rather than extrapolating its loading.
        self.sim_dt = self.duration / (self.sample_count - 1)
        self.times = np.linspace(0.0, self.duration, self.sample_count)
        self.index = 0
        self.sim_time = 0.0
        self.history = []
        self.pitch_inertia = 0.025
        self._screenshot_saved = False
        self._gif_frames = []
        self._gif_times = []
        self._captured_index = -1
        self._make_reference()
        last_local = self.shoe.visual_mesh("fullfoot_last").vertices_m - self.ankle_mount
        self.minimum_last_offsets = np.empty(self.sample_count)
        for start in range(0, self.sample_count, 128):
            angle = self.reference[start : start + 128, 2].astype(float)
            self.minimum_last_offsets[start : start + 128] = np.min(
                -np.sin(angle[:, None]) * last_local[None, :, 0] + np.cos(angle[:, None]) * last_local[None, :, 2],
                axis=1,
            )

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -self.gravity))
        builder.add_ground_plane()
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        self.carrier = builder.add_body(
            mass=self.foot_mass,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3) * self.pitch_inertia),
            label="robot_foot_fixture",
        )
        mesh = self.shoe.visual_mesh("fullfoot_last")
        builder.add_shape_mesh(
            self.carrier,
            mesh=newton.Mesh(
                np.asarray(mesh.vertices_m - self.ankle_mount, np.float32), np.asarray(mesh.triangles, np.int32).ravel()
            ),
            cfg=cfg,
            color=(0.72, 0.77, 0.82),
            label="calibrated_last",
        )
        if self.reference_mode == "pitch":
            builder.add_shape_sphere(
                self.carrier, radius=0.017, cfg=cfg, color=(1.0, 0.5, 0.08), label="mechanical_ankle_pivot"
            )
        self.com_body = builder.add_body(
            mass=self.com_mass, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)), label="upper_inertial_slider"
        )
        # The overhead track is a visual boundary, not an extra body or force.
        lo = min(float(self.reference[:, 0].min()), float(self.reference[:, 3].min())) - 0.2
        hi = max(float(self.reference[:, 0].max()), float(self.reference[:, 3].max())) + 0.2
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3((lo + hi) / 2, 0.0, 1.25), wp.quat_identity()),
            hx=(hi - lo) / 2,
            hy=0.025,
            hz=0.018,
            cfg=cfg,
            color=(0.3, 0.34, 0.39),
        )
        builder.color()
        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, angular_damping=0.0, enable_tri_contact=False)
        initial = np.array(
            [
                [self.reference[0, 0], 0.0, self.reference[0, 1], 0, 0, 0, 1],
                [self.reference[0, 3], 0.0, self.reference[0, 4], 0, 0, 0, 1],
            ],
            np.float32,
        )
        if self.planar:
            initial[:, :3] = self.planar_initial_positions
        velocity = np.zeros((2, 6), np.float32)
        velocity[0, 2] = self.reference[0, 6]
        velocity[1, 2] = self.reference[0, 9]
        if self.planar:
            velocity = self.planar_initial_velocity.copy()
        self.state_0.body_q.assign(initial)
        self.state_0.body_qd.assign(velocity)
        self.state_1.body_q.assign(initial)
        self.state_1.body_qd.assign(velocity)
        self.reference_device = wp.array(self.reference, dtype=wp.float32, device=self.device)
        self.trace_device = wp.zeros((self.sample_count, 41), dtype=wp.float32, device=self.device)
        self.leg_diagnostics = wp.zeros(9, dtype=wp.float32, device=self.device)
        bed = self.shoe.column_bed
        scale = args.shoe_stiffness_scale
        material = replace(
            self.shoe.material,
            instantaneous_shear_modulus_pa=self.shoe.material.instantaneous_shear_modulus_pa * scale,
            # Reported only; the runtime rebuilds the per-column coupling from the
            # scaled equilibrium shear modulus, so keep the reported value in step.
            pasternak_n_per_m=self.shoe.material.pasternak_n_per_m * scale,
        )
        self.contact_config = FoundationConfig(
            friction_stiffness=args.contact_kt if self.planar else 0.0,
            friction=args.contact_kd if self.planar else 0.0,
            mu=args.friction_mu if self.planar else 0.0,
        )
        count = len(bed.rest_length_m)
        driven = np.ones(count, np.int32)
        if self.planar and args.passive_outer:
            lookup = {tuple(np.round(point, 8)): index for index, point in enumerate(bed.anchor_bottom_m[:, :2])}
            fixture = self.shoe.instron_fixture("fullfoot_last")
            supported = np.array([lookup[tuple(np.round(point, 8))] for point in fixture.carrier_anchor_m[:, :2]])
            driven[:] = 0
            driven[supported] = 1
        if int(count - driven.sum()) and (args.outer_relaxation <= 0.0 or not 0.0 < args.outer_max_strain < 1.0):
            raise ValueError("Passive outer relaxation time must be positive and its strain limit inside (0, 1)")
        # One relaxation for both projects: the same balance the Digital Instron
        # identification sweeps over its untouched foam runs here every substep.
        # ``carrier_bond`` is True because this shoe carries the whole bed, so an
        # outer column top is glued under the rigid last, not a free bench surface.
        # A massless outer surface avoids inventing an unidentified surface mass.
        self.surround_config = SurroundConfig(
            driven=driven,
            attachment_n_m=args.outer_attachment,
            max_strain=args.outer_max_strain,
            coupling_scale=args.outer_coupling_scale,
            sweeps=args.outer_substeps,
            relaxation_time_s=args.outer_relaxation,
            carrier_bond=True,
        )
        self.foundation = MidsoleFoundation(
            bed.anchor_bottom_m - self.ankle_mount,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            material,
            self.carrier,
            self.model.body_com,
            self.contact_config,
            self.device,
            self.surround_config,
        )
        self.free_columns = self.foundation.free_column_count
        self.driven = self.foundation.driven
        self.free_metrics = wp.zeros(5, dtype=wp.float32, device=self.device)
        self.old_tangent_anchor = wp.zeros_like(self.foundation.tangent_anchor)
        self.old_tangent_active = wp.zeros_like(self.foundation.tangent_stuck)
        self.contact_metrics = wp.zeros(7, dtype=wp.float32, device=self.device)
        self.points = wp.zeros(len(bed.rest_length_m), dtype=wp.vec3, device=self.device)
        self.tops = wp.zeros_like(self.points)
        self.colors = wp.zeros_like(self.points)
        self.leg_start = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.leg_end = wp.zeros_like(self.leg_start)
        self.com_point = wp.zeros_like(self.leg_start)
        self.com_color = wp.array([[0.12, 0.62, 0.95]], dtype=wp.vec3, device=self.device)
        self.viewer.set_model(self.model)
        center = 0.5 * (lo + hi)
        self.viewer.set_camera(*_camera((center + 0.7, -1.85, 0.95), (center, 0.0, 0.55)))
        self._prescribe(0)
        self.metadata = {
            "profile_hash": hashlib.sha256(Path(args.profile).read_bytes()).hexdigest(),
            "artifact_hash": hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest(),
            "mass_kg": self.mass,
            "foot_mass_kg": self.foot_mass,
            "stiffness_n_m": args.stiffness,
            "damping_n_s_m": args.damping,
            "dt_s": self.sim_dt,
            "mode": self.mode,
            "dynamics": "planar" if self.planar else "vertical",
            "reference_mode": self.reference_mode,
            "shoe_orientation": self.shoe_orientation,
            "ankle_mount_m": self.ankle_mount.tolist(),
            "expected_duration_s": self.duration,
            "shoe_stiffness_scale": scale,
            "scenario_changes": ["shoe_stiffness_scale"],
            "initial_state": {
                "ankle_x_m": float(initial[0, 0]),
                "upper_x_m": float(initial[1, 0]),
                "ankle_vx_m_s": float(velocity[0, 0]),
                "upper_vx_m_s": float(velocity[1, 0]),
                "foot_z_m": float(initial[0, 2]),
                "foot_vz_m_s": float(velocity[0, 2]),
                "com_z_m": float(initial[1, 2]),
                "com_vz_m_s": float(velocity[1, 2]),
            },
            "registration": self.registration,
            "kinematic_rate_hz": args.kinematic_rate_hz,
            "reference_processing": self.reference_processing,
            "processed_reference_hash": hashlib.sha256(self.reference.tobytes()).hexdigest(),
            "runtime_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "solver": "newton.solvers.SolverSemiImplicit; angular_damping=0; planar X/Z are dynamic, world pitch motor prescribed"
            if self.planar
            else "newton.solvers.SolverSemiImplicit; legacy prescribed horizontal axes",
            "pitch_inertia_kg_m2": self.pitch_inertia,
            "gravity_m_s2": self.gravity,
            "shoe_identification_passed": self.shoe.raw["identification"]["passed_all_declared_gates"],
            "initial_prescribed_state": self.reference[0].tolist(),
            "profile_provenance": self.profile.get("provenance", {}),
            "pitch_reconstruction": {
                key: self.profile.get("pitch_reference", {}).get(key)
                for key in ("method", "quality", "neutral_reference", "static_template")
            },
            "profile_limits": [
                "Representative running inputs from another shoe, NOT same-Puma validation; user confirmed cross-shoe example scope.",
                "Heel triangle tracks rearfoot rotation; LTOE is on the upper over the second metatarsal head, not the toe tip.",
                "Pitch uses a declared static neutral and fixed shoe-local mounting point; in planar mode the ankle position itself is free. This is not an anatomical fit.",
                "Opposite-foot vertical force is an explicit replay boundary, not a simulated second leg.",
                "The COM reference is force-integrated with assumed initial conditions, not measured anatomical COM.",
                "Planar mode integrates both X/Z states with a central geometric leg; nominal rolling geometry is not a prescribed trajectory.",
                "Friction parameters are assumed. Bristle reference anchors and their drift are not independently measured outsole slip.",
                "Pitch is an external ideal robotic motor with separately recorded work, not a complete anatomical ankle/shank actuator model.",
            ],
            "force_limit_n": args.force_limit_bw * self.mass * self.gravity,
            "normal_damping_n_s_m_per_column": 0.0,
            "friction": {
                "enabled": self.planar and self.args.friction_mu > 0 and self.args.contact_kt > 0,
                "mu": self.contact_config.mu,
                "stiffness_n_m_per_column": self.contact_config.friction_stiffness,
                "damping_n_s_m_per_column": self.contact_config.friction,
                "identified": False,
                "rendering": "Ground points are bristle reference anchors, not a no-slip constraint.",
                "slip_metric": "Material-point speed, elastic shear and plastic-anchor drift are distinct; contact initialization is not classified as drift.",
                "wrench_limit": "Existing effective foundation applies its tangential wrench at virtual bottom sites, sometimes below the plane; not a calibrated real-ground ankle moment.",
            },
            "power_definition": "Active source power includes feedforward/retraction, moving spring rest length, scheduled stiffness, desired damping rate, and saturation intervention; passive damping and guide drives are separate. Not metabolic cost.",
        }

    def _make_reference(self):
        if getattr(self.args, "reference_mode", "markers") == "pitch":
            self._make_pitch_reference()
            if self.planar:
                self._make_planar_reference()
            return
        self.reference_processing = "optical-clock C1 Hermite with analytic derivatives; legacy marker trajectory"
        source_t = np.asarray(self.profile["time_s"], dtype=float)

        def sample(key):
            return np.interp(self.times, source_t, np.asarray(self.profile[key], dtype=float))

        # Profiles retain native force sampling. Recover optical-rate knots
        # before differentiating, instead of differentiating upsampled corners.
        source_start = float(self.profile["source_time_s"][0])
        rate = self.args.kinematic_rate_hz
        start = math.ceil(source_start * rate) / rate
        optical = np.arange(start, source_start + self.duration, 1.0 / rate) - source_start
        knots = np.unique(np.clip(np.concatenate(([0.0], optical, [self.duration])), 0.0, self.duration))
        knots = knots[np.concatenate(([True], np.diff(knots) > 1.0e-7))]
        if len(knots) < 3:
            raise ValueError("The stance must contain at least three optical-rate knots")

        def optical_sample(key):
            return np.interp(knots, source_t, np.asarray(self.profile[key], dtype=float))

        pitch_knots = optical_sample("pitch_rad")
        # This fixed bench registration is not a calibrated anatomical fit.
        bed = self.shoe.column_bed.anchor_bottom_m
        offset = -float(bed[:, 0].min())
        x_knots = optical_sample("foot_x_m") + offset * np.cos(pitch_knots)
        z_knots = optical_sample("foot_z_m") - offset * np.sin(pitch_knots)
        touchdown = float(self.profile["provenance"]["running"]["selected_stance_source_s"][0]) - source_start
        touch_pitch = float(_curve(pitch_knots, knots, np.array([touchdown]))[0][0])
        touch_z = float(_curve(z_knots, knots, np.array([touchdown]))[0][0])
        minimum = float(np.min(-np.sin(touch_pitch) * bed[:, 0] + np.cos(touch_pitch) * bed[:, 2]))
        shift = -minimum + self.args.initial_clearance - touch_z
        z_knots += shift
        x, vx, ax = _curve(x_knots, knots, self.times)
        z, vz, az = _curve(z_knots, knots, self.times)
        pitch, omega, alpha = _curve(pitch_knots, knots, self.times)
        center_x, center_vx, center_ax = _curve(optical_sample("com_x_m"), knots, self.times)
        center_z = sample("com_z_m")
        center_vz = sample("reference_com_vz_m_s")
        self.centroid_reference = np.column_stack([center_x, center_z, center_vx, center_vz])
        # The force-integrated reference is the total centroid, not the upper
        # slider. Account for fixture inertia without counting its mass twice.
        cx = (self.mass * center_x - self.foot_mass * x) / self.com_mass
        cvx = (self.mass * center_vx - self.foot_mass * vx) / self.com_mass
        cax = (self.mass * center_ax - self.foot_mass * ax) / self.com_mass
        cz = (self.mass * center_z - self.foot_mass * z) / self.com_mass
        cvz = (self.mass * center_vz - self.foot_mass * vz) / self.com_mass
        caz = (
            sample("reference_fz_n") + sample("other_fz_n") - self.mass * self.gravity - self.foot_mass * az
        ) / self.com_mass
        self.reference = np.column_stack(
            [
                x,
                z,
                pitch,
                cx,
                cz,
                vx,
                vz,
                omega,
                cvx,
                cvz,
                az,
                sample("reference_fz_n"),
                sample("other_fz_n"),
                cz - z,
                cvz - vz,
                alpha,
                caz,
                ax,
                cax,
            ]
        ).astype(np.float32)
        self.registration = {
            "kind": "fixed bench heel-to-center offset; lowest outsole aligned at measured threshold touchdown",
            "touchdown_time_s": touchdown,
            "heel_to_center_m": offset,
            "vertical_shift_m": shift,
            "touchdown_clearance_m": self.args.initial_clearance,
            "anatomical_registration_validated": False,
        }
        self.raw_pitch = sample("pitch_rad")
        self.reference_fx = sample("reference_fx_n")
        cop = np.asarray([np.nan if v is None else v for v in self.profile["reference_cop_x_m"]])
        self.reference_cop = np.interp(self.times, source_t, cop)

    def _make_pitch_reference(self):
        """Drive only calibrated pitch; derive vertical intent from force and constant leg length."""
        source_t = np.asarray(self.profile["time_s"], dtype=float)

        def sample(key):
            return np.interp(self.times, source_t, np.asarray(self.profile[key], dtype=float))

        pitch_data = self.profile["pitch_reference"]
        trajectory = TrajectoryCubic.fit(
            pitch_data["knot_time_s"], pitch_data["pitch_rad"], cutoff_hz=self.args.pitch_cutoff
        )
        pitch, omega, alpha = trajectory.evaluate(self.times)
        self.raw_pitch = np.interp(self.times, pitch_data["knot_time_s"], pitch_data["pitch_rad"])
        self.reference_processing = trajectory.smoothing
        source_start = float(self.profile["source_time_s"][0])
        touchdown = float(self.profile["provenance"]["running"]["selected_stance_source_s"][0]) - source_start
        touch_pitch = float(trajectory.evaluate(np.array([touchdown]))[0][0])
        anchors = self.shoe.column_bed.anchor_bottom_m - self.ankle_mount
        touch_bottom = float(np.min(-np.sin(touch_pitch) * anchors[:, 0] + np.cos(touch_pitch) * anchors[:, 2]))
        center_z, center_vz = sample("com_z_m"), sample("reference_com_vz_m_s")
        # One initial height predicts threshold touchdown under free fall. It is
        # not a prescribed ankle trajectory or a calibrated anatomical contact.
        initial_ankle_z = (
            -touch_bottom - center_vz[0] * touchdown + 0.5 * self.gravity * touchdown**2 + self.args.initial_clearance
        )
        first_bottom = float(np.min(-np.sin(pitch[0]) * anchors[:, 0] + np.cos(pitch[0]) * anchors[:, 2]))
        if not self.planar and initial_ankle_z + first_bottom < -1.0e-6:
            raise ValueError(
                "The angle-only initialization would preload the shoe; choose a different declared initial condition"
            )
        x = self.args.ankle_x + self.args.track_speed * self.times
        vx, ax = np.full_like(self.times, self.args.track_speed), np.zeros_like(self.times)
        center_x = sample("com_x_m")
        center_vx = sample("reference_com_vx_m_s")
        center_ax = sample("total_measured_fx_n") / self.mass
        self.centroid_reference = np.column_stack([center_x, center_z, center_vx, center_vz])
        z = center_z - center_z[0] + initial_ankle_z
        vz = center_vz.copy()
        az = (sample("reference_fz_n") + sample("other_fz_n")) / self.mass - self.gravity
        cx = (self.mass * center_x - self.foot_mass * x) / self.com_mass
        cvx = (self.mass * center_vx - self.foot_mass * vx) / self.com_mass
        cax = (self.mass * center_ax - self.foot_mass * ax) / self.com_mass
        cz = (self.mass * center_z - self.foot_mass * z) / self.com_mass
        rest_length = float(cz[0] - z[0])
        if rest_length <= 0.0:
            raise ValueError("Virtual COM must begin above the mechanical ankle")
        toeoff = float(self.profile["provenance"]["running"]["selected_stance_source_s"][1]) - source_start
        duration = self.args.unload_duration
        if not np.isfinite(duration) or duration < 0.0 or duration >= toeoff - touchdown:
            raise ValueError("Unload duration must be zero or shorter than stance")
        if not np.isfinite(self.args.unload_acceleration) or self.args.unload_acceleration < 0.0:
            raise ValueError("Unload acceleration must be finite and nonnegative")
        gain, gain_rate = np.ones_like(z), np.zeros_like(z)
        if duration > 0.0:
            u = np.clip((self.times - (toeoff - duration)) / duration, 0.0, 1.0)
            gain = np.clip(1.0 - u**3 * (10.0 - 15.0 * u + 6.0 * u**2), 0.0, 1.0)
            gain_rate = -30.0 * u**2 * (1.0 - u) ** 2 / duration
        retract = (1.0 - gain) * self.foot_mass * (self.gravity + self.args.unload_acceleration)
        self.reference = np.column_stack(
            [
                x,
                z,
                pitch,
                cx,
                cz,
                vx,
                vz,
                omega,
                cvx,
                center_vz,
                az,
                sample("reference_fz_n"),
                sample("other_fz_n"),
                np.full_like(z, rest_length),
                np.zeros_like(z),
                alpha,
                az,
                ax,
                cax,
                gain,
                gain_rate,
                retract,
            ]
        ).astype(np.float32)
        self.reference_fx = sample("reference_fx_n")
        self.reference_cop = np.interp(
            self.times, source_t, [np.nan if v is None else v for v in self.profile["reference_cop_x_m"]]
        )
        self.registration = {
            "kind": "fixed mechanical ankle mount; no marker XYZ replay",
            "ankle_mount_in_oriented_shoe_m": self.ankle_mount.tolist(),
            "fixture_mass_location": "lumped at mechanical ankle; not anatomical foot COM",
            "ankle_x_m": self.args.ankle_x,
            "track_speed_m_s": self.args.track_speed,
            "initialization": "free-fall estimate to threshold touchdown using measured pitch and assumed COM entry velocity",
            "touchdown_time_s": touchdown,
            "touchdown_clearance_m": self.args.initial_clearance,
            "initial_ankle_z_m": initial_ankle_z,
            "initial_outsole_clearance_m": initial_ankle_z + first_bottom,
            "constant_leg_rest_length_m": rest_length,
            "unload_duration_s": self.args.unload_duration,
            "unload_acceleration_m_s2": self.args.unload_acceleration,
            "unload_policy": "quintic impedance fade to zero at measured toe-off plus bounded internal fixture lift; not a marker trajectory",
            "toeoff_time_s": toeoff,
            "vertical_reference": "force-integrated total centroid and constant leg length; NOT prescribed XYZ motion",
            "anatomical_registration_validated": False,
            "cop_comparison": "source measured COP kept as context; no marker-translation registration to the pitch-only rig",
        }

    def _make_planar_reference(self):
        """Anchor declared entry conditions at touchdown, independently of clip padding."""
        ref = self.reference.astype(np.float64)
        time = self.times
        angle, omega = ref[:, 2], ref[:, 7]
        anchors = self.shoe.column_bed.anchor_bottom_m - self.ankle_mount
        support_height = -np.min(
            -np.sin(angle[:, None]) * anchors[None, :, 0] + np.cos(angle[:, None]) * anchors[None, :, 2], axis=1
        )
        touchdown = self.registration["touchdown_time_s"]
        delta = time - touchdown
        support_entry = float(np.interp(touchdown, time, support_height)) + self.args.initial_clearance
        omega_entry = float(np.interp(touchdown, time, omega))
        ankle_entry_vx = omega_entry * support_entry if self.args.ankle_entry_vx is None else self.args.ankle_entry_vx
        entry_vx = float(self.profile["provenance"]["com_surrogate"]["initial_vx_m_s"])
        flight = float(self.profile["provenance"]["running"]["flight_before_s"])
        entry_vz = -0.5 * self.gravity * flight if self.args.com_entry_vz is None else self.args.com_entry_vz
        if (
            not np.all(np.isfinite([ankle_entry_vx, entry_vx, entry_vz, self.args.com_entry_height]))
            or self.args.com_entry_height <= support_entry
        ):
            raise ValueError("Entry state must be finite with COM above the ankle")
        entry_com_x = self.args.ankle_x + self.args.com_offset_x
        entry_com_z = self.args.com_entry_height
        centroid = self.centroid_reference.copy()
        # Re-anchor the force-integrated nominal path without changing sensor forces.
        for column, velocity_column, position_entry, velocity_entry in (
            (0, 2, entry_com_x, entry_vx),
            (1, 3, entry_com_z, entry_vz),
        ):
            old_position = float(np.interp(touchdown, time, centroid[:, column]))
            old_velocity = float(np.interp(touchdown, time, centroid[:, velocity_column]))
            adjustment = velocity_entry - old_velocity
            centroid[:, column] += adjustment * delta + position_entry - old_position
            centroid[:, velocity_column] += adjustment
        self.centroid_reference = centroid
        nominal_vx = omega * support_height
        distance = np.r_[0.0, np.cumsum(0.5 * (nominal_vx[1:] + nominal_vx[:-1]) * np.diff(time))]
        ankle_x = self.args.ankle_x + distance - float(np.interp(touchdown, time, distance))
        ankle_z = support_height + self.args.initial_clearance
        before = time < touchdown
        ankle_x[before] = self.args.ankle_x + ankle_entry_vx * delta[before]
        ankle_z[before] = support_entry + entry_vz * delta[before] - 0.5 * self.gravity * delta[before] ** 2
        nominal_vx[before] = ankle_entry_vx
        ankle_vz = np.gradient(ankle_z, time, edge_order=2)
        upper_x = (self.mass * centroid[:, 0] - self.foot_mass * ankle_x) / self.com_mass
        upper_z = (self.mass * centroid[:, 1] - self.foot_mass * ankle_z) / self.com_mass
        length = np.hypot(upper_x - ankle_x, upper_z - ankle_z)
        length_rate = np.gradient(length, time, edge_order=2)
        upper_vx = (self.mass * centroid[:, 2] - self.foot_mass * nominal_vx) / self.com_mass
        upper_vz = (self.mass * centroid[:, 3] - self.foot_mass * ankle_vz) / self.com_mass
        ref[:, 0], ref[:, 1], ref[:, 3], ref[:, 4] = ankle_x, ankle_z, upper_x, upper_z
        ref[:, 5], ref[:, 6], ref[:, 8], ref[:, 9] = nominal_vx, ankle_vz, upper_vx, upper_vz
        ref[:, 13], ref[:, 14] = length, length_rate
        engage_duration = self.args.engage_duration
        if not np.isfinite(engage_duration) or engage_duration <= 0.0:
            raise ValueError("Planar engagement duration must be finite and positive")
        w = np.clip((time - touchdown) / engage_duration, 0.0, 1.0)
        engage = w**3 * (10.0 - 15.0 * w + 6.0 * w**2)
        engage_rate = 30.0 * w**2 * (1.0 - w) ** 2 / engage_duration
        out_gain, out_rate = ref[:, 19].copy(), ref[:, 20].copy()
        ref[:, 19] = engage * out_gain
        ref[:, 20] = engage_rate * out_gain + engage * out_rate
        source_t = self.profile["time_s"]
        other_fx = np.interp(time, source_t, self.profile["other_fx_n"])
        self.reference = np.column_stack([ref, self.reference_fx, other_fx, engage * out_gain]).astype(np.float32)
        # Before contact both masses follow a ballistic approach. Adding frames
        # changes the starting state, not the declared landing position or speed.
        initial_ankle = np.array(
            [
                self.args.ankle_x - ankle_entry_vx * touchdown,
                0.0,
                support_entry - entry_vz * touchdown - 0.5 * self.gravity * touchdown**2,
            ]
        )
        initial_com = np.array(
            [
                entry_com_x - entry_vx * touchdown,
                0.0,
                entry_com_z - entry_vz * touchdown - 0.5 * self.gravity * touchdown**2,
            ]
        )
        initial_upper = (self.mass * initial_com - self.foot_mass * initial_ankle) / self.com_mass
        self.planar_initial_positions = np.array([initial_ankle, initial_upper], np.float32)
        velocity = np.zeros((2, 6), np.float32)
        velocity[0, 0], velocity[0, 2] = ankle_entry_vx, entry_vz + self.gravity * touchdown
        total_velocity = np.array([entry_vx, 0.0, entry_vz + self.gravity * touchdown])
        velocity[1, :3] = (self.mass * total_velocity - self.foot_mass * velocity[0, :3]) / self.com_mass
        self.planar_initial_velocity = velocity
        for key in (
            "constant_leg_rest_length_m",
            "vertical_reference",
            "initial_ankle_z_m",
            "ankle_x_m",
            "track_speed_m_s",
        ):
            self.registration.pop(key, None)
        self.registration.update(
            {
                "kind": "planar dynamic X/Z foot and body lump with a central axial leg",
                "touchdown_com_offset_x_m": self.args.com_offset_x,
                "touchdown_com_height_m": entry_com_z,
                "touchdown_com_vx_m_s": entry_vx,
                "touchdown_com_vz_m_s": entry_vz,
                "entry_vz_policy": "equal-height preceding flight assumption, not measured COM"
                if self.args.com_entry_vz is None
                else "explicit user scenario",
                "touchdown_ankle_x_m": self.args.ankle_x,
                "touchdown_ankle_z_m": support_entry,
                "touchdown_ankle_vx_m_s": float(ankle_entry_vx),
                "initial_com_xyz_m": initial_com.tolist(),
                "initial_ankle_xyz_m": initial_ankle.tolist(),
                "initial_geometric_leg_length_m": float(np.linalg.norm(initial_upper - initial_ankle)),
                "initial_outsole_clearance_m": float(initial_ankle[2] - support_height[0]),
                "predicted_precontact_min_clearance_m": float(np.min(ankle_z[before] - support_height[before]))
                if np.any(before)
                else 0.0,
                "initialization": "ballistic backward construction from declared touchdown state; no marker XYZ replay",
                "leg_reference": "nominal force-integrated COM and geometry-only rolling ankle define scalar reference length; no position targets are applied",
                "nominal_rolling": "ankle vx approximately omega*support height",
                "engage_duration_s": engage_duration,
                "dynamics": "Actual X/Z states are integrated; only pitch and out-of-plane axes are prescribed",
                "friction_parameters": {
                    "mu": self.args.friction_mu,
                    "stiffness_n_m_per_column": self.args.contact_kt,
                    "damping_n_s_m_per_column": self.args.contact_kd,
                    "identified": False,
                },
            }
        )

    def _prescribe(self, index):
        if self.planar:
            wp.launch(
                _constrain_planar_axes,
                dim=1,
                inputs=[index, self.reference_device, self.state_0.body_q, self.state_0.body_qd],
                device=self.device,
            )
            return
        wp.launch(
            _prescribe_axes,
            dim=1,
            inputs=[
                index,
                self.reference_device,
                int(self.mode == "replay"),
                self.state_0.body_q,
                self.state_0.body_qd,
            ],
            device=self.device,
        )

    def _sample(self, index):
        self._prescribe(index)
        self.state_0.clear_forces()
        if self.planar:
            wp.copy(self.old_tangent_anchor, self.foundation.tangent_anchor)
            wp.copy(self.old_tangent_active, self.foundation.tangent_stuck)
        # MidsoleFoundation now relaxes the passive outer columns itself, with the same
        # kernel the Digital Instron identification sweeps over its untouched foam.
        self.foundation.apply(self.state_0, self.sim_dt)
        if self.planar:
            wp.launch(
                _apply_planar_leg,
                dim=1,
                inputs=[
                    index,
                    self.reference_device,
                    self.mass,
                    self.foot_mass,
                    self.args.stiffness,
                    self.args.damping,
                    self.args.force_limit_bw * self.mass * self.gravity,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.leg_diagnostics,
                ],
                device=self.device,
            )
            self.free_metrics.zero_()
            if self.free_columns:
                wp.launch(
                    _free_column_metrics,
                    dim=self.foundation.column_count,
                    inputs=[
                        self.driven,
                        self.foundation.compression,
                        self.foundation.rest_len,
                        self.foundation.column_force,
                        self.foundation.surround_rate,
                        self.free_metrics,
                    ],
                    device=self.device,
                )
            self.contact_metrics.zero_()
            wp.launch(
                _contact_motion_metrics,
                dim=self.foundation.column_count,
                inputs=[
                    self.sim_dt,
                    self.args.friction_mu,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.foundation.anchor_local,
                    self.foundation.column_force,
                    self.old_tangent_anchor,
                    self.old_tangent_active,
                    self.foundation.tangent_anchor,
                    self.foundation.tangent_stuck,
                    self.contact_metrics,
                ],
                device=self.device,
            )
            wp.launch(
                _record_planar_sample,
                dim=1,
                inputs=[
                    index,
                    self.reference_device,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.foundation.resultant_force,
                    self.foundation.pressed_force,
                    self.foundation.cop_moment,
                    self.foundation.contact_power,
                    self.foundation.max_compression,
                    self.leg_diagnostics,
                    self.contact_metrics,
                    self.free_metrics,
                    self.com_mass,
                    self.pitch_inertia,
                    self.gravity,
                    self.trace_device,
                ],
                device=self.device,
            )
            return
        wp.launch(
            _apply_leg,
            dim=1,
            inputs=[
                index,
                self.reference_device,
                self.foot_mass,
                self.args.stiffness,
                self.args.damping,
                self.args.force_limit_bw * self.mass * self.gravity,
                self.gravity,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.body_f,
                self.leg_diagnostics,
            ],
            device=self.device,
        )
        wp.launch(
            _record_sample,
            dim=1,
            inputs=[
                index,
                self.reference_device,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.body_f,
                self.foundation.normal_force,
                self.foundation.cop_moment,
                self.foundation.contact_power,
                self.foundation.max_compression,
                self.leg_diagnostics,
                self.com_mass,
                self.pitch_inertia,
                self.foot_mass,
                int(self.mode == "replay"),
                self.gravity,
                self.trace_device,
            ],
            device=self.device,
        )

    def step(self):
        """Advance one frame and hold toe-off without inventing a swing."""
        if self.index >= self.sample_count:
            return
        if self.index == 0:
            self._sample(0)
            self.index = 1
        stop = min(self.index + self.args.substeps, self.sample_count)
        while self.index < stop:
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._sample(self.index)
            self.index += 1
        self.sim_time = float(self.times[self.index - 1])

    def render(self):
        """Show the measured track motion, dynamic COM, and actual shoe columns."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.planar:
            wp.launch(
                _draw_friction_columns,
                dim=self.foundation.column_count,
                inputs=[
                    self.state_0.body_q,
                    self.foundation.anchor_local,
                    self.foundation.rest_len,
                    self.foundation.column_force,
                    self.foundation.z_free,
                    self.foundation.tangent_anchor,
                    self.foundation.tangent_stuck,
                    self.points,
                    self.tops,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                attached_column_endpoints,
                dim=self.foundation.column_count,
                inputs=[
                    self.carrier,
                    self.state_0.body_q,
                    self.foundation.anchor_local,
                    self.foundation.rest_len,
                    self.points,
                    self.tops,
                ],
                device=self.device,
            )
        wp.launch(
            column_colors,
            dim=self.foundation.column_count,
            inputs=[self.foundation.compression, 0.020, self.colors],
            device=self.device,
        )
        self.viewer.log_lines("impedance/shoe_columns", self.points, self.tops, self.colors, width=0.002)
        self.viewer.log_points("impedance/shoe_bottom", self.points, radii=0.0015, colors=self.colors)
        positions = self.state_0.body_q.numpy()[:, :3]
        self.leg_start.assign(
            positions[0:1] + np.array([[0.0, 0.0, 0.0 if self.reference_mode == "pitch" else 0.09]], np.float32)
        )
        center = (self.foot_mass * positions[0] + self.com_mass * positions[1]) / self.mass
        self.leg_end.assign(positions[1:2] if self.planar else center.reshape(1, 3))
        self.com_point.assign(center.reshape(1, 3))
        self.viewer.log_points("impedance/virtual_COM", self.com_point, radii=0.065, colors=self.com_color)
        self.viewer.log_lines("impedance/virtual_leg", self.leg_start, self.leg_end, (0.2, 0.65, 0.9), width=0.012)
        self.viewer.log_scalar("/impedance/time_s", self.sim_time)
        self.viewer.log_scalar("/impedance/com_height_m", float(center[2]))
        self.viewer.log_scalar("/impedance/com_x_m", float(center[0]))
        self.viewer.log_scalar("/impedance/ankle_x_m", float(positions[0, 0]))
        if self.planar:
            self.viewer.log_scalar(
                "/impedance/geometric_leg_length_m", float(np.linalg.norm(positions[1] - positions[0]))
            )
            self.viewer.log_scalar("/impedance/shoe_Fx_n", float(self.foundation.resultant_force.numpy()[0, 0]))
        self.viewer.log_scalar("/impedance/reference_force_n", float(self.reference[max(0, self.index - 1), 11]))
        self.viewer.log_scalar("/impedance/shoe_force_n", float(self.foundation.normal_force.numpy()[0]))
        self.viewer.end_frame()
        if (self.args.screenshot or self.args.record_gif) and hasattr(self.viewer, "get_frame"):
            from PIL import Image, ImageOps

            image = Image.fromarray(self.viewer.get_frame().numpy())
            if self.args.record_gif and self.index != self._captured_index:
                self._gif_frames.append(image.resize((720, round(720 * image.height / image.width))))
                self._gif_times.append(self.sim_time)
                self._captured_index = self.index
            if self.args.screenshot and not self._screenshot_saved and self.sim_time >= self.duration * 0.45:
                path = Path(self.args.screenshot)
                path.parent.mkdir(parents=True, exist_ok=True)
                ImageOps.fit(image, (320, 320)).convert("RGB").save(path)
                self._screenshot_saved = True

    def rows(self):
        """Download the complete substep trace once, retaining force and power peaks."""
        trace = self.trace_device.numpy()[: self.index]
        result = []
        for i, values in enumerate(trace):
            ref = self.reference[i]
            center_z = (self.foot_mass * float(values[2]) + self.com_mass * float(values[3])) / self.mass
            center_vz = (self.foot_mass * float(values[15]) + self.com_mass * float(values[4])) / self.mass
            reference_x, reference_z, reference_vx, reference_vz = self.centroid_reference[i]
            ankle_x = float(values[19]) if self.planar else float(ref[0])
            upper_x = float(values[20]) if self.planar else float(ref[3])
            ankle_vx = float(values[21]) if self.planar else float(ref[5])
            upper_vx = float(values[22]) if self.planar else float(ref[8])
            center_x = (self.foot_mass * ankle_x + self.com_mass * upper_x) / self.mass
            center_vx = (self.foot_mass * ankle_vx + self.com_mass * upper_vx) / self.mass
            row = {
                "time_s": float(self.times[i]),
                "source_time_s": float(self.profile["source_time_s"][0] + self.times[i]),
                "reference_fz_n": float(ref[11]),
                "shoe_fz_n": float(values[0]),
                "reference_fx_n": float(self.reference_fx[i]),
                "other_fz_n": float(ref[12]),
                "reference_cop_x_m": float(self.reference_cop[i]) if self.reference_mode == "markers" else float("nan"),
                "source_cop_x_m": float(self.reference_cop[i]),
                "shoe_cop_x_m": float(values[1]) if values[0] > 1.0 else float("nan"),
                "foot_x_m": ankle_x,
                "foot_vx_m_s": ankle_vx,
                "foot_z_m": float(values[2]),
                "reference_foot_z_m": float(ref[1]),
                "pitch_rad": float(ref[2]),
                "raw_pitch_rad": float(self.raw_pitch[i]),
                "pitch_velocity_rad_s": float(ref[7]),
                "pitch_acceleration_rad_s2": float(ref[15]),
                "ankle_torque_nm": float(values[18]),
                "impedance_gain": float(ref[19]) if len(ref) >= 22 else 1.0,
                "retraction_force_n": float(ref[21]) if len(ref) >= 22 else 0.0,
                "last_min_height_m": float(values[2] + self.minimum_last_offsets[i]),
                "ankle_x_m": ankle_x if self.reference_mode == "pitch" else float("nan"),
                "ankle_z_m": float(values[2]) if self.reference_mode == "pitch" else float("nan"),
                "shoe_origin_x_m": float(
                    ankle_x - np.cos(ref[2]) * self.ankle_mount[0] - np.sin(ref[2]) * self.ankle_mount[2]
                ),
                "shoe_origin_z_m": float(
                    values[2] + np.sin(ref[2]) * self.ankle_mount[0] - np.cos(ref[2]) * self.ankle_mount[2]
                ),
                "com_x_m": float(center_x),
                "com_vx_m_s": float(center_vx),
                "reference_com_x_m": float(reference_x),
                "reference_com_vx_m_s": float(reference_vx),
                "upper_slider_x_m": upper_x,
                "upper_slider_vx_m_s": upper_vx,
                "com_z_m": center_z,
                "reference_com_z_m": float(reference_z),
                "com_vz_m_s": center_vz,
                "reference_com_vz_m_s": float(reference_vz),
                "upper_slider_z_m": float(values[3]),
                "upper_slider_vz_m_s": float(values[4]),
                "reference_upper_z_m": float(ref[4]),
                "reference_upper_vz_m_s": float(ref[9]),
                "leg_force_n": float(values[5]),
                "active_power_w": float(values[6]),
                "damping_power_w": float(values[7]),
                "pitch_power_w": float(values[8]),
                "shoe_contact_power_w": float(values[9]),
                "com_energy_j": self.mass * (self.gravity * center_z + 0.5 * (center_vz**2 + float(center_vx) ** 2)),
                "max_compression_m": float(values[11]),
                "controller_clipped": float(values[12]),
                "leg_spring_energy_j": float(values[13]),
                "other_support_power_w": float(values[14]),
                "foot_vz_m_s": float(values[15]),
                "replay_vertical_power_w": float(values[16]),
                "track_power_w": float(values[17]),
                "rig_energy_j": float(values[10] + values[13])
                + self.foot_mass * (self.gravity * float(values[2]) + 0.5 * float(values[15]) ** 2)
                + 0.5 * self.foot_mass * ankle_vx**2
                + 0.5 * self.com_mass * upper_vx**2
                + 0.5 * self.pitch_inertia * float(ref[7]) ** 2,
            }
            if self.planar:
                row.update(
                    {
                        "shoe_fx_n": float(values[23]),
                        "shoe_fy_n": float(values[24]),
                        "leg_length_m": float(values[25]),
                        "leg_length_rate_m_s": float(values[26]),
                        "reference_leg_length_m": float(ref[13]),
                        "leg_fx_n": float(values[27]),
                        "leg_fz_n": float(values[28]),
                        "contact_material_speed_m_s": float(values[29]),
                        "bristle_anchor_drift_m_s": float(values[30]),
                        "contact_shear_extension_m": float(values[31]),
                        "plastic_anchor_load_fraction": float(values[32]),
                        "tangential_contact_power_w": float(values[33]),
                        "max_coulomb_utilization": float(values[34]),
                        "pressed_force_n": float(values[40]),
                        "driven_column_force_n": float(values[35]),
                        "passive_column_force_n": float(values[36]),
                        "passive_max_strain": float(values[37]),
                        "passive_max_surface_speed_m_s": float(values[38]),
                        "passive_loaded_columns": float(values[39]),
                        "com_ankle_dx_m": float(center_x - ankle_x),
                        "com_ankle_distance_m": float(np.hypot(center_x - ankle_x, center_z - float(values[2]))),
                    }
                )
            result.append(row)
        return result

    def qualification(self):
        """Check engineering behavior without claiming human or material validation."""
        reasons = []
        if self.index != self.sample_count:
            reasons.append(
                f"Incomplete stance: use at least {math.ceil(self.sample_count / self.args.substeps)} frames"
            )
        trace = self.trace_device.numpy()[: self.index]
        if self.index < 2 or not np.all(np.isfinite(trace)) or not np.all(np.isfinite(self.state_0.body_q.numpy())):
            reasons.append("Missing or nonfinite rig state/trace")
        elif self.index:
            peak = float(trace[:, 0].max())
            compression = float(trace[:, 11].max())
            if not 0.25 * self.mass * self.gravity < peak < 6.0 * self.mass * self.gravity:
                reasons.append(f"Shoe peak outside engineering bounds: {peak:.1f} N")
            if not 0.0 < compression < 0.05:
                reasons.append(f"Compression outside engineering bounds: {compression:.6f} m")
            last_height = trace[:, 2] + self.minimum_last_offsets[: self.index]
            if np.min(last_height) < -0.001:
                reasons.append(f"Rigid last penetrated ground: {1000 * np.min(last_height):.2f} mm")
            if (
                self.reference_mode == "pitch"
                and self.index == self.sample_count
                and trace[-1, 0] > 0.1 * self.mass * self.gravity
            ):
                reasons.append(f"Fixture remains loaded after toe-off: {trace[-1, 0]:.1f} N")
            if np.any(trace[:, 12] != 0.0):
                reasons.append("Controller hit its force limit")
            if self.mode == "impedance" and np.max(np.abs(trace[:, 2] - self.reference[: self.index, 1])) < 1.0e-5:
                reasons.append("Impedance foot did not depart from prescribed motion")
        if self.model.body_count != 2 or self.model.joint_count != 2:
            reasons.append("Expected only a free fixture and an upper inertial slider")
        return {
            "passed": not reasons,
            "reasons": reasons,
            "scope": "runtime diagnostics for the mechanical example, not human validation or a new test suite",
        }

    def test_final(self):
        """Require a complete, finite stance with contact and free vertical response."""
        result = self.qualification()
        self.metadata["engineering_qualification"] = result
        if not result["passed"]:
            raise AssertionError("; ".join(result["reasons"]))
        trace = self.trace_device.numpy()
        print(
            f"[impedance Instron] {self.duration:.4f} s; peak={trace[:, 0].max():.1f} N; compression={trace[:, 11].max() * 1000:.2f} mm"
        )

    def save(self):
        """Write auditable traces and an offline report for the actually simulated interval."""
        self.metadata["engineering_qualification"] = self.qualification()
        report = write_report(Path(self.args.output), self.rows(), self.metadata, self.args.compare)
        if self.args.record_gif and self._gif_frames:
            path = Path(self.args.record_gif)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._gif_frames[0].save(path, save_all=True, append_images=self._gif_frames[1:], duration=50, loop=0)
            from PIL import Image, ImageDraw

            td = self.registration["touchdown_time_s"]
            to = self.registration.get("toeoff_time_s", self.duration)
            phases = [
                ("Approach", 0.0),
                ("Measured touchdown", td),
                ("Loading", td + 0.04),
                ("Midstance", 0.5 * (td + to)),
                ("Measured toe-off", to),
                ("Release context", self.duration),
            ]
            width = 480
            height = round(width * self._gif_frames[0].height / self._gif_frames[0].width)
            sheet = Image.new("RGB", (3 * width, 2 * (height + 34)), "white")
            draw = ImageDraw.Draw(sheet)
            for slot, (label, time_s) in enumerate(phases):
                index = int(np.argmin(np.abs(np.asarray(self._gif_times) - time_s)))
                x, y = (slot % 3) * width, (slot // 3) * (height + 34)
                sheet.paste(self._gif_frames[index].resize((width, height)), (x, y + 34))
                draw.text((x + 8, y + 8), f"{label} | t={self._gif_times[index]:.3f} s", fill="black")
            sequence = Path(self.args.output) / "contact_sequence.jpg"
            sheet.save(sequence)
            print(f"Contact sequence: {sequence}")
        if self.planar:
            rows = self.rows()
            td = self.registration["toeoff_time_s"]
            at_to = min(rows, key=lambda row: abs(row["time_s"] - td))
            print(
                f"Planar example: COM offset at toe-off {at_to['com_ankle_dx_m']:.3f} m; "
                f"ankle X {rows[0]['ankle_x_m']:.3f} -> {rows[-1]['ankle_x_m']:.3f} m; "
                f"peak Fz {max(row['shoe_fz_n'] for row in rows):.1f} N; final Fz {rows[-1]['shoe_fz_n']:.3f} N"
            )
        print(f"Report: {report}")
        return report


def create_parser():
    """Expose motion, material scenarios, and explicit controller settings."""
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=120)
    parser.add_argument("--profile", type=Path, default=Path("outputs/impedance_instron/stance_planar_context.json"))
    parser.add_argument("--artifact", type=Path, default=Path("DigitalInstron/digital_shoe_showcase/digital_shoe.json"))
    parser.add_argument("--mode", choices=["impedance", "replay"], default="impedance")
    parser.add_argument(
        "--dynamics",
        choices=["planar", "vertical"],
        default="planar",
        help="Force-driven X/Z foot and COM, or retained vertical-only comparison.",
    )
    parser.add_argument(
        "--com-offset-x", type=float, default=-0.4, help="Total COM X behind ankle at nominal touchdown [m]."
    )
    parser.add_argument(
        "--com-entry-height", type=float, default=1.0, help="Assumed total COM height at touchdown [m]."
    )
    parser.add_argument(
        "--com-entry-vz",
        type=float,
        default=None,
        help="Assumed COM vertical velocity at touchdown [m/s]; default uses half the preceding flight duration.",
    )
    parser.add_argument(
        "--ankle-entry-vx",
        type=float,
        default=None,
        help="Ankle X velocity at touchdown [m/s]; default is rolling-compatible and only initializes the state.",
    )
    parser.add_argument(
        "--engage-duration",
        type=float,
        default=0.02,
        help="Virtual impedance engagement interval after measured touchdown [s].",
    )
    parser.add_argument(
        "--passive-outer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let columns outside the last footprint deform through neighbors instead of following the rigid last.",
    )
    parser.add_argument(
        "--outer-relaxation",
        type=float,
        default=0.002,
        help="Passive outer surface relaxation time [s] toward its local balance; assumed, not identified.",
    )
    parser.add_argument(
        "--outer-coupling-scale",
        type=float,
        default=1.0,
        help="Assumed lateral drive on the passive outer surface as a fraction of the identified rigid-top coupling.",
    )
    parser.add_argument(
        "--outer-attachment",
        type=float,
        default=0.0,
        help=(
            "Vertical bond stiffness per outer column to the shoe above it [N/m]. Zero by default: its "
            "reaction never reaches the reported force or the carrier wrench, so a nonzero value is an "
            "undeclared rigid support, not a bond."
        ),
    )
    parser.add_argument(
        "--outer-max-strain", type=float, default=0.9, help="Passive outer compression limit as a fraction of rest."
    )
    parser.add_argument(
        "--outer-substeps", type=int, default=4, help="Free-surface relaxation substeps per solver substep."
    )
    parser.add_argument(
        "--friction-mu", type=float, default=0.8, help="Assumed Coulomb coefficient; not fitted from compression data."
    )
    parser.add_argument(
        "--contact-kt", type=float, default=10000.0, help="Bristle tangential stiffness per column [N/m]."
    )
    parser.add_argument("--contact-kd", type=float, default=10.0, help="Bristle tangential damping per column [N s/m].")
    parser.add_argument(
        "--reference-mode",
        choices=["pitch", "markers"],
        default="pitch",
        help="Pitch-only mechanical ankle, or legacy marker-trajectory experiment.",
    )
    parser.add_argument(
        "--ankle-mount",
        type=float,
        nargs=3,
        default=(-0.075, 0.0, 0.105),
        metavar=("X", "Y", "Z"),
        help="Fixed mechanical ankle in oriented shoe coordinates [m]; not an anatomical fit.",
    )
    parser.add_argument("--ankle-x", type=float, default=0.0, help="Initial world track coordinate [m].")
    parser.add_argument(
        "--track-speed",
        type=float,
        default=0.0,
        help="Prescribed ankle track speed [m/s], independent of marker translations.",
    )
    parser.add_argument(
        "--pitch-cutoff",
        type=float,
        default=12.0,
        help="Offline optical-angle smoothing cutoff [Hz], 0 disables smoothing (C2 interpolation remains).",
    )
    parser.add_argument(
        "--source-shoe-side",
        choices=["left", "right"],
        default="right",
        help="Explicit interpretation of baked geometry, NOT inferred from the artifact label; supplied artifact audit indicates right.",
    )
    parser.add_argument(
        "--shoe-side",
        choices=["left", "right"],
        default="left",
        help="Chosen mechanical fixture side; anatomy certification remains separate.",
    )
    parser.add_argument(
        "--unload-duration",
        type=float,
        default=0.04,
        help="Impedance fade duration before measured toe-off [s]; 0 disables release scheduling.",
    )
    parser.add_argument(
        "--unload-acceleration",
        type=float,
        default=0.0,
        help="Additional fixture lift acceleration during release [m/s^2]; default uses gravity compensation only.",
    )
    parser.add_argument("--stiffness", type=float, default=12000.0, help="Virtual leg stiffness [N/m].")
    parser.add_argument("--damping", type=float, default=500.0, help="Virtual leg damping [N s/m].")
    parser.add_argument(
        "--foot-mass", type=float, default=2.0, help="Fixture inertia mass [kg], included in total mass."
    )
    parser.add_argument(
        "--shoe-stiffness-scale",
        type=float,
        default=1.0,
        help="Synthetic G and Pasternak multiplier; not a fitted new shoe.",
    )
    parser.add_argument(
        "--force-limit-bw", type=float, default=5.0, help="Signed actuator force limit in total body weights."
    )
    parser.add_argument(
        "--touchdown-clearance",
        dest="initial_clearance",
        type=float,
        default=0.0,
        help="Lowest outsole height at measured threshold touchdown [m]; fixed across shoe comparisons.",
    )
    parser.add_argument("--substeps", type=int, default=64, help="Native solver substeps per 120 Hz display frame.")
    parser.add_argument(
        "--kinematic-rate-hz",
        type=float,
        default=None,
        help="Optical knot rate [Hz]; must match source marker sampling.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/impedance_instron/planar_baseline"))
    parser.add_argument("--compare", type=Path, help="Prior output directory for audited comparison.")
    parser.add_argument("--screenshot", type=Path, help="Save a 320x320 JPG near midstance with --viewer gl.")
    parser.add_argument("--record-gif", type=Path, help="Save a slowed OpenGL stance animation.")
    return parser


def main():
    """Run the reusable example through Newton's normal viewer interface."""
    viewer, args = newton.examples.init(create_parser())
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        if example.index > 1:
            example.save()


if __name__ == "__main__":
    main()
