# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton articulation of the three-dimensional bilateral runner.

Seven bodies per world: a floating pelvis carrying the trunk, arms, and head,
and per leg a thigh, shank, and shoe-aligned foot. Joints follow
:mod:`.kinematics3d`: three-axis hips with anatomical zero, and hinge knees and
ankles whose constant static offsets are in their child frames. Each foot body
origin is the ankle center with the intrinsic shoe axes, so the shoe foundation
attaches to it directly. There are no ground shapes; only the two shoes touch the
ground. The legs carry capsule collision shapes only when asked
(:func:`.leg_contact.add_capsules`), and those collide right leg against left only.

Segment masses, centers of mass, and sagittal inertias come from the leg profile
(the same as the planar model). Frontal inertias equal the sagittal ones, and
the axial inertias of the long segments are 15 % of them; those transverse and
axial values are engineering assumptions, not subject measurements.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from . import leg_contact
from .kinematics3d import ANKLE_AXIS, KNEE_AXIS, SIDES, Calibration, hip_axes
from .mechanics import HeadArmsTrunk

BODIES = ("pelvis", "thigh_r", "shank_r", "foot_r", "thigh_l", "shank_l", "foot_l")
FOOT_BODIES = {"right": 3, "left": 6}
COORDINATES_PER_WORLD = 17
DOFS_PER_WORLD = 16
ACTUATED = 10


def _quat(rotation: np.ndarray) -> wp.quat:
    return wp.quat_from_matrix(wp.mat33(*np.asarray(rotation, dtype=float).ravel()))


def leg_inertials(profile: dict) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return (mass, COM, inertia) of thigh, shank, and foot in their 3D body frames."""
    masses = np.asarray(profile["masses_kg"], dtype=float)
    com = np.asarray(profile["com_local_m"], dtype=float)
    sagittal = np.asarray(profile["inertias_kg_m2"], dtype=float)
    result = []
    for segment in range(2):
        inertia = np.diag([sagittal[segment], sagittal[segment], 0.15 * sagittal[segment]])
        result.append((masses[segment], np.array([0.0, 0.0, -com[segment, 0]]), inertia))
    # Planar foot COM is [forward, down] from the ankle in the shoe frame.
    foot_com = np.array([com[2, 0], 0.0, com[2, 1]])
    result.append((masses[2], foot_com, np.diag([0.3 * sagittal[2], sagittal[2], sagittal[2]])))
    return result


def build_runner(
    calibration: Calibration,
    profile: dict,
    subject_mass_kg: float,
    hat: HeadArmsTrunk | None = None,
    capsules: leg_contact.LegContactConfig | None = None,
) -> newton.ModelBuilder:
    """Build one world's articulation for a static calibration, with leg ``capsules`` if given."""
    hat = hat or HeadArmsTrunk()
    legs = leg_inertials(profile)
    leg_mass = sum(m for m, _, _ in legs)
    hat_mass = float(subject_mass_kg) - 2 * leg_mass if hat.mass_kg is None else float(hat.mass_kg)
    if hat_mass <= 0:
        raise ValueError("HAT mass must be positive")
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    rg = hat.radius_of_gyration_m
    pelvis = builder.add_link(
        mass=hat_mass,
        com=wp.vec3(0.0, 0.0, float(hat.com_local_m[0])),
        inertia=wp.mat33(*np.diag([hat_mass * rg**2, hat_mass * rg**2, hat_mass * 0.12**2]).ravel()),
        lock_inertia=True,
        label="pelvis",
    )
    articulation = [builder.add_joint_free(child=pelvis, label="root")]
    links = [pelvis]
    for side in SIDES:
        s = "r" if side == "right" else "l"
        bodies = []
        for name, (mass, com, inertia) in zip(("thigh", "shank", "foot"), legs, strict=True):
            bodies.append(
                builder.add_link(
                    mass=float(mass),
                    com=wp.vec3(*com),
                    inertia=wp.mat33(*inertia.ravel()),
                    lock_inertia=True,
                    label=f"{name}_{s}",
                )
            )
        thigh, shank, foot = bodies
        links.extend(bodies)
        dof = builder.JointDofConfig
        articulation.append(
            builder.add_joint_d6(
                pelvis,
                thigh,
                angular_axes=[dof(axis=axis, armature=0.0, damping=0.0) for axis in hip_axes(side)],
                parent_xform=wp.transform(wp.vec3(*calibration.hip_m[side]), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0), _quat(calibration.offsets[f"hip_{s}"].T)),
                label=f"hip_{s}",
            )
        )
        articulation.append(
            builder.add_joint_revolute(
                thigh,
                shank,
                axis=KNEE_AXIS,
                parent_xform=wp.transform(wp.vec3(*calibration.knee_m[side]), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0), _quat(calibration.offsets[f"knee_{s}"].T)),
                armature=0.0,
                damping=0.0,
                label=f"knee_{s}",
            )
        )
        articulation.append(
            builder.add_joint_revolute(
                shank,
                foot,
                axis=ANKLE_AXIS,
                parent_xform=wp.transform(wp.vec3(*calibration.ankle_m[side]), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0), _quat(calibration.offsets[f"ankle_{s}"].T)),
                armature=0.0,
                damping=0.0,
                label=f"ankle_{s}",
            )
        )
    builder.add_articulation(articulation)
    if capsules is not None:
        leg_contact.add_capsules(builder, links, calibration, capsules)
    return builder


def build_worlds(
    calibrations: list[Calibration],
    world_trials: np.ndarray,
    profile: dict,
    subject_mass_kg: float,
    hat: HeadArmsTrunk | None,
    device,
    capsules: leg_contact.LegContactConfig | None = None,
) -> newton.Model:
    """Finalize a model whose world ``w`` uses the calibration ``world_trials[w]``."""
    templates = [build_runner(c, profile, subject_mass_kg, hat, capsules) for c in calibrations]
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    for trial in world_trials:
        scene.add_world(templates[int(trial)])
    model = scene.finalize(device=device)
    if model.body_count != len(world_trials) * len(BODIES) or model.joint_coord_count != len(world_trials) * 17:
        raise RuntimeError("Unexpected per-world body or coordinate layout")
    return model
