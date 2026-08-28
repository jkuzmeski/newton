# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export a scaled simple gait subject as interoperable MJCF."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native_model import SimpleGaitConfig


@dataclass(frozen=True, slots=True)
class SubjectVisualMesh:
    """One body-local visual mesh referenced by a saved subject MJCF."""

    name: str
    """Stable mesh and geometry name."""

    body: str
    """Target simple-model body label."""

    file: str
    """Mesh path relative to the MJCF file."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Visual-only offset in the target body frame [m]."""


@dataclass(frozen=True, slots=True)
class SubjectInertial:
    """One subject body's inertial properties in the target body frame."""

    mass: float
    """Body mass [kg]."""

    position: tuple[float, float, float]
    """Center of mass in the target body frame [m]."""

    full_inertia: tuple[float, float, float, float, float, float]
    """Inertia about COM as Ixx, Iyy, Izz, Ixy, Ixz, Iyz [kg·m²]."""


@dataclass(frozen=True, slots=True)
class SubjectMarkerSite:
    """One motion-capture marker attached to a native subject body."""

    name: str
    """Motion-capture marker name."""

    body: str
    """Target native body label."""

    position: tuple[float, float, float]
    """Marker position in the target body frame [m]."""

    site_name: str
    """Stable MJCF site name."""


def _values(*items: float) -> str:
    return " ".join(f"{item:.9g}" for item in items)


def _box_inertia(mass: float, dimensions: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = dimensions
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def _trimmed_capsule_half_height(length: float, radius: float, clearance: float) -> float:
    """Return the capsule half-height after shortening both joint ends [m]."""
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("self_collision_joint_clearance must be finite and nonnegative")
    trimmed_length = length - 2.0 * clearance
    if trimmed_length <= 2.0 * radius:
        raise ValueError("self-collision joint clearance leaves no capsule body")
    return 0.5 * trimmed_length - radius


@dataclass(frozen=True, slots=True)
class _InertiaBox:
    """Inertia-equivalent box values in one body frame."""

    center: tuple[float, float, float]
    """Box center at the body-frame COM [m]."""

    half_extents: tuple[float, float, float]
    """Principal-frame box half-extents [m]."""

    quaternion_wxyz: tuple[float, float, float, float]
    """Principal-frame orientation as an MJCF ``wxyz`` quaternion."""

    long_axis: tuple[float, float, float]
    """Principal axis with the largest equivalent box extent."""

    long_half_extent: float
    """Largest equivalent box half-extent [m]."""

    capsule_radius: float
    """Smaller transverse equivalent box half-extent [m]."""


def _matrix_to_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to an MJCF ``wxyz`` quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif axis == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _inertia_box(
    mass: float,
    position: tuple[float, float, float],
    full_inertia: tuple[float, float, float, float, float, float],
) -> _InertiaBox:
    """Reproduce the viewer's inertia-box extents from body mass and inertia."""
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("inertia-box mass must be finite and positive")
    if len(position) != 3 or not np.all(np.isfinite(position)):
        raise ValueError("inertia-box position must contain three finite values")
    if len(full_inertia) != 6 or not np.all(np.isfinite(full_inertia)):
        raise ValueError("inertia-box tensor must contain six finite values")
    ixx, iyy, izz, ixy, ixz, iyz = full_inertia
    inertia = np.asarray(
        ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)),
        dtype=np.float64,
    )
    principal, axes = np.linalg.eigh(inertia)
    if np.any(principal <= 0.0) or not np.all(np.isfinite(principal)):
        raise ValueError("inertia-box tensor must be positive definite")
    box_inertia = principal * (12.0 / (8.0 * mass))
    squared_extents = np.asarray(
        (
            box_inertia[2] + box_inertia[1] - box_inertia[0],
            box_inertia[0] + box_inertia[2] - box_inertia[1],
            box_inertia[1] + box_inertia[0] - box_inertia[2],
        )
    )
    half_extents = np.sqrt(np.abs(squared_extents))
    if np.any(half_extents <= 0.0) or not np.all(np.isfinite(half_extents)):
        raise ValueError("inertia-box tensor produced invalid extents")
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    long_axis_index = int(np.argmax(half_extents))
    transverse = [index for index in range(3) if index != long_axis_index]
    return _InertiaBox(
        center=tuple(float(value) for value in position),
        half_extents=tuple(float(value) for value in half_extents),
        quaternion_wxyz=_matrix_to_wxyz(axes),
        long_axis=tuple(float(value) for value in axes[:, long_axis_index]),
        long_half_extent=float(half_extents[long_axis_index]),
        capsule_radius=float(np.min(half_extents[transverse])),
    )


def _capsule_fromto(proxy: _InertiaBox, clearance: float) -> tuple[str, float]:
    """Return body-frame capsule endpoints and radius from an inertia box."""
    half_length = proxy.long_half_extent - clearance
    if half_length <= proxy.capsule_radius:
        raise ValueError("self-collision clearance leaves no inertia-derived capsule body")
    center = np.asarray(proxy.center)
    axis = np.asarray(proxy.long_axis)
    start = center - axis * half_length
    end = center + axis * half_length
    return _values(*start, *end), proxy.capsule_radius


def _add_inertial(
    body: ET.Element,
    mass: float,
    dimensions: tuple[float, float, float],
    inertial: SubjectInertial | None = None,
) -> None:
    if inertial is None:
        ET.SubElement(
            body,
            "inertial",
            mass=f"{mass:.9g}",
            pos="0 0 0",
            diaginertia=_values(*_box_inertia(mass, dimensions)),
        )
    else:
        ET.SubElement(
            body,
            "inertial",
            mass=f"{inertial.mass:.9g}",
            pos=_values(*inertial.position),
            fullinertia=_values(*inertial.full_inertia),
        )


def _add_joint(
    body: ET.Element,
    *,
    name: str,
    position: tuple[float, float, float],
    axis: tuple[float, float, float],
    limits: tuple[float, float],
    damping: float,
    armature: float,
) -> None:
    ET.SubElement(
        body,
        "joint",
        name=name,
        type="hinge",
        pos=_values(*position),
        axis=_values(*axis),
        limited="true",
        range=_values(*limits),
        damping=f"{damping:.9g}",
        armature=f"{armature:.9g}",
    )


def _add_target_actuators(
    actuator: ET.Element,
    joint: str,
    limits: tuple[float, float],
) -> None:
    ET.SubElement(
        actuator,
        "position",
        name=f"{joint}_position",
        joint=joint,
        kp="100",
        ctrllimited="true",
        ctrlrange=_values(*limits),
    )
    ET.SubElement(
        actuator,
        "velocity",
        name=f"{joint}_velocity",
        joint=joint,
        kv="20",
        ctrllimited="true",
        ctrlrange="-20 20",
    )


def subject_mjcf_xml(
    config: SimpleGaitConfig,
    *,
    model_name: str = "simple_gait_subject",
    visual_meshes: Sequence[SubjectVisualMesh] = (),
    include_fallback_geometry: bool = True,
    contact_centers: dict[str, tuple[tuple[float, float, float], ...]] | None = None,
    contact_radius: float | None = None,
    inertial_data: dict[str, SubjectInertial] | None = None,
    joint_centers: dict[str, tuple[float, float, float]] | None = None,
    marker_sites: Sequence[SubjectMarkerSite] = (),
) -> str:
    """Create MJCF for one scaled simple-joint subject.

    Args:
        config: Subject-scaled dimensions, masses, and contact parameters.
        model_name: MJCF model label.
        visual_meshes: Body-local neutral mesh assets.
        include_fallback_geometry: Include box and capsule visuals when true.
            Collision-aware box and capsule proxies are always emitted separately.
        contact_centers: Optional mesh-derived sphere centers keyed by side.
        contact_radius: Optional mesh-derived contact radius [m].
        inertial_data: Optional OpenSim-derived inertial properties by target body.
            Proxy geometry always comes from mass and inertia. When provided,
            these values replace the scaled nominal fallback values and drive
            the body COMs, principal axes, and equivalent box extents.
        joint_centers: Optional official neutral joint centers in target child frames.
        marker_sites: Neutral motion-capture marker sites attached to target bodies.

    Returns:
        An MJCF XML document. It can be passed directly to
        :meth:`newton.ModelBuilder.add_mjcf`.
    """
    inertials = inertial_data or {}
    unknown_inertials = set(inertials) - {
        "pelvis",
        "torso",
        "femur_left",
        "femur_right",
        "tibia_left",
        "tibia_right",
        "foot_left",
        "foot_right",
    }
    if unknown_inertials:
        raise ValueError(f"inertial_data contains unknown bodies: {sorted(unknown_inertials)}")
    centers = joint_centers or {}
    expected_centers = {"hip_left", "hip_right", "knee_left", "knee_right", "ankle_left", "ankle_right"}
    if centers and set(centers) != expected_centers:
        raise ValueError(f"joint_centers must contain exactly {sorted(expected_centers)}")
    expected_bodies = {
        "pelvis",
        "torso",
        "femur_left",
        "femur_right",
        "tibia_left",
        "tibia_right",
        "foot_left",
        "foot_right",
    }
    marker_names: set[str] = set()
    marker_site_names: set[str] = set()
    for marker in marker_sites:
        if not marker.name or marker.name in marker_names:
            raise ValueError(f"empty or duplicate marker name: {marker.name!r}")
        if marker.body not in expected_bodies:
            raise ValueError(f"marker {marker.name!r} references unknown body {marker.body!r}")
        if (
            not marker.site_name
            or marker.site_name in marker_site_names
            or "/" in marker.site_name
            or any(character.isspace() for character in marker.site_name)
        ):
            raise ValueError(f"invalid or duplicate marker site name: {marker.site_name!r}")
        if len(marker.position) != 3 or not np.all(np.isfinite(marker.position)):
            raise ValueError(f"marker {marker.name!r} position must contain three finite values")
        marker_names.add(marker.name)
        marker_site_names.add(marker.site_name)
    nominal_bodies = {
        "pelvis": (config.pelvis_mass, config.pelvis_dimensions),
        "torso": (config.torso_mass, config.torso_dimensions),
        "femur_left": (config.thigh_mass, (2.0 * config.thigh_radius, 2.0 * config.thigh_radius, config.thigh_length)),
        "femur_right": (config.thigh_mass, (2.0 * config.thigh_radius, 2.0 * config.thigh_radius, config.thigh_length)),
        "tibia_left": (config.shank_mass, (2.0 * config.shank_radius, 2.0 * config.shank_radius, config.shank_length)),
        "tibia_right": (config.shank_mass, (2.0 * config.shank_radius, 2.0 * config.shank_radius, config.shank_length)),
    }

    def inertial_for(name: str) -> SubjectInertial:
        source = inertials.get(name)
        if source is not None:
            return source
        mass, dimensions = nominal_bodies[name]
        diagonal = _box_inertia(mass, dimensions)
        return SubjectInertial(mass, (0.0, 0.0, 0.0), (*diagonal, 0.0, 0.0, 0.0))

    inertia_boxes = {
        name: _inertia_box(source.mass, source.position, source.full_inertia)
        for name, source in ((name, inertial_for(name)) for name in nominal_bodies)
    }
    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 -9.80665", timestep="0.001")
    defaults = ET.SubElement(root, "default")
    visual = ET.SubElement(defaults, "default", attrib={"class": "visual"})
    ET.SubElement(visual, "geom", contype="0", conaffinity="0", group="2", rgba="0.72 0.72 0.78 1")
    collision = ET.SubElement(defaults, "default", attrib={"class": "collision"})
    ET.SubElement(
        collision,
        "geom",
        friction=_values(config.friction, 0.005, 0.0001),
        solref=_values(-config.ground_ke, -config.ground_kd),
        rgba="0.25 0.45 0.85 1",
    )
    self_collision = ET.SubElement(defaults, "default", attrib={"class": "self_collision"})
    ET.SubElement(
        self_collision,
        "geom",
        friction=_values(config.self_collision_mu, 0.005, 0.0001),
        solref=_values(-config.self_collision_ke, -config.self_collision_kd),
        rgba="0.95 0.45 0.18 0.35",
    )

    assets = ET.SubElement(root, "asset")
    mesh_names: set[str] = set()
    for mesh in visual_meshes:
        mesh_path = Path(mesh.file)
        if not mesh.name or mesh.name in mesh_names:
            raise ValueError(f"empty or duplicate visual mesh name: {mesh.name!r}")
        if mesh_path.is_absolute() or ".." in mesh_path.parts:
            raise ValueError(f"visual mesh path must stay relative to the MJCF: {mesh.file!r}")
        mesh_names.add(mesh.name)
        ET.SubElement(assets, "mesh", name=mesh.name, file=mesh_path.as_posix())

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="5 5 0.1",
        attrib={"class": "collision"},
        rgba="0.85 0.85 0.85 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="visual_ground",
        type="plane",
        size="5 5 0.1",
        pos="0 0 -0.002",
        attrib={"class": "visual"},
        rgba="0.78 0.78 0.78 1",
    )
    pelvis = ET.SubElement(world, "body", name="pelvis", pos=_values(0.0, 0.0, config.pelvis_height))
    ET.SubElement(pelvis, "freejoint", name="pelvis_free")
    _add_inertial(pelvis, config.pelvis_mass, config.pelvis_dimensions, inertials.get("pelvis"))
    if include_fallback_geometry:
        ET.SubElement(
            pelvis,
            "geom",
            name="geometry_pelvis",
            type="box",
            pos=_values(*inertia_boxes["pelvis"].center),
            quat=_values(*inertia_boxes["pelvis"].quaternion_wxyz),
            size=_values(*inertia_boxes["pelvis"].half_extents),
            attrib={"class": "visual"},
        )
    ET.SubElement(
        pelvis,
        "geom",
        name="collision_pelvis",
        type="box",
        pos=_values(*inertia_boxes["pelvis"].center),
        quat=_values(*inertia_boxes["pelvis"].quaternion_wxyz),
        size=_values(*inertia_boxes["pelvis"].half_extents),
        attrib={"class": "self_collision"},
    )

    torso = ET.SubElement(pelvis, "body", name="torso", pos=_values(0.0, 0.0, config.torso_center_offset))
    _add_inertial(torso, config.torso_mass, config.torso_dimensions, inertials.get("torso"))
    if include_fallback_geometry:
        ET.SubElement(
            torso,
            "geom",
            name="geometry_torso",
            type="box",
            pos=_values(*inertia_boxes["torso"].center),
            quat=_values(*inertia_boxes["torso"].quaternion_wxyz),
            size=_values(*inertia_boxes["torso"].half_extents),
            attrib={"class": "visual"},
        )
    ET.SubElement(
        torso,
        "geom",
        name="collision_torso",
        type="box",
        pos=_values(*inertia_boxes["torso"].center),
        quat=_values(*inertia_boxes["torso"].quaternion_wxyz),
        size=_values(*inertia_boxes["torso"].half_extents),
        attrib={"class": "self_collision"},
    )

    body_elements = {"pelvis": pelvis, "torso": torso}
    actuator = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="torso")
    for side in ("left", "right"):
        ET.SubElement(contact, "exclude", body1="pelvis", body2=f"femur_{side}")
        ET.SubElement(contact, "exclude", body1=f"femur_{side}", body2=f"tibia_{side}")
        ET.SubElement(contact, "exclude", body1=f"tibia_{side}", body2=f"foot_{side}")
    degrees = math.pi / 180.0
    radius = config.contact_radius if contact_radius is None else contact_radius
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("contact_radius must be finite and positive")
    if contact_centers is None:
        heel_x = -0.32 * config.foot_length
        forefoot_x = 0.48 * config.foot_length
        half_width = 0.35 * config.foot_width
        centers_by_side = dict.fromkeys(
            ("left", "right"),
            (
                (heel_x, -half_width, -config.contact_radius),
                (heel_x, half_width, -config.contact_radius),
                (forefoot_x, -half_width, -config.contact_radius),
                (forefoot_x, half_width, -config.contact_radius),
            ),
        )
    else:
        centers_by_side = contact_centers
        if set(centers_by_side) != {"left", "right"} or any(len(values) != 4 for values in centers_by_side.values()):
            raise ValueError("contact_centers must provide four centers for each foot")
    for side, lateral_sign in (("left", 1.0), ("right", -1.0)):
        femur_fromto, femur_radius = _capsule_fromto(
            inertia_boxes[f"femur_{side}"],
            config.self_collision_joint_clearance,
        )
        tibia_fromto, tibia_radius = _capsule_fromto(
            inertia_boxes[f"tibia_{side}"],
            config.self_collision_joint_clearance,
        )
        femur = ET.SubElement(
            pelvis,
            "body",
            name=f"femur_{side}",
            pos=_values(
                0.0,
                lateral_sign * config.hip_half_width,
                -config.pelvis_hip_drop - 0.5 * config.thigh_length,
            ),
        )
        body_elements[f"femur_{side}"] = femur
        _add_inertial(
            femur,
            config.thigh_mass,
            (2.0 * config.thigh_radius, 2.0 * config.thigh_radius, config.thigh_length),
            inertials.get(f"femur_{side}"),
        )
        hip_position = centers.get(f"hip_{side}", (0.0, 0.0, 0.5 * config.thigh_length))
        hip_specs = (
            ("flexion", (0.0, -1.0, 0.0), (-30.0 * degrees, 120.0 * degrees), 0.5),
            ("adduction", (lateral_sign, 0.0, 0.0), (-25.0 * degrees, 45.0 * degrees), 0.5),
            ("rotation", (0.0, 0.0, -lateral_sign), (-45.0 * degrees, 45.0 * degrees), 0.3),
        )
        for coordinate, axis, limits, damping in hip_specs:
            name = f"hip_{coordinate}_{side}"
            _add_joint(
                femur,
                name=name,
                position=hip_position,
                axis=axis,
                limits=limits,
                damping=damping,
                armature=0.01,
            )
            _add_target_actuators(actuator, name, limits)
        if include_fallback_geometry:
            ET.SubElement(
                femur,
                "geom",
                name=f"geometry_femur_{side}",
                type="capsule",
                size=f"{femur_radius:.9g}",
                fromto=femur_fromto,
                attrib={"class": "visual"},
            )
        ET.SubElement(
            femur,
            "geom",
            name=f"collision_femur_{side}",
            type="capsule",
            size=f"{femur_radius:.9g}",
            fromto=femur_fromto,
            attrib={"class": "self_collision"},
        )

        tibia = ET.SubElement(
            femur,
            "body",
            name=f"tibia_{side}",
            pos=_values(0.0, 0.0, -0.5 * (config.thigh_length + config.shank_length)),
        )
        body_elements[f"tibia_{side}"] = tibia
        _add_inertial(
            tibia,
            config.shank_mass,
            (2.0 * config.shank_radius, 2.0 * config.shank_radius, config.shank_length),
            inertials.get(f"tibia_{side}"),
        )
        knee_name = f"knee_{side}"
        knee_limits = (0.0, 150.0 * degrees)
        _add_joint(
            tibia,
            name=knee_name,
            position=centers.get(f"knee_{side}", (0.0, 0.0, 0.5 * config.shank_length)),
            axis=(0.0, 1.0, 0.0),
            limits=knee_limits,
            damping=0.3,
            armature=0.01,
        )
        _add_target_actuators(actuator, knee_name, knee_limits)
        if include_fallback_geometry:
            ET.SubElement(
                tibia,
                "geom",
                name=f"geometry_tibia_{side}",
                type="capsule",
                size=f"{tibia_radius:.9g}",
                fromto=tibia_fromto,
                attrib={"class": "visual"},
            )
        ET.SubElement(
            tibia,
            "geom",
            name=f"collision_tibia_{side}",
            type="capsule",
            size=f"{tibia_radius:.9g}",
            fromto=tibia_fromto,
            attrib={"class": "self_collision"},
        )

        foot = ET.SubElement(
            tibia,
            "body",
            name=f"foot_{side}",
            pos=_values(
                0.4 * config.foot_length,
                0.0,
                -0.5 * config.shank_length - config.contact_radius,
            ),
        )
        body_elements[f"foot_{side}"] = foot
        _add_inertial(
            foot,
            config.foot_mass,
            (config.foot_length, config.foot_width, 2.0 * config.contact_radius),
            inertials.get(f"foot_{side}"),
        )
        ankle_name = f"ankle_{side}"
        ankle_limits = (-50.0 * degrees, 30.0 * degrees)
        _add_joint(
            foot,
            name=ankle_name,
            position=centers.get(f"ankle_{side}", (-0.4 * config.foot_length, 0.0, config.contact_radius)),
            axis=(0.0, -1.0, 0.0),
            limits=ankle_limits,
            damping=0.2,
            armature=0.005,
        )
        _add_target_actuators(actuator, ankle_name, ankle_limits)
        for index, center in enumerate(centers_by_side[side]):
            ET.SubElement(
                foot,
                "geom",
                name=f"contact_{side}_{index}",
                type="sphere",
                size=f"{radius:.9g}",
                pos=_values(*center),
                attrib={"class": "collision"},
            )
            ET.SubElement(
                foot,
                "geom",
                name=f"visual_foot_{side}_{index}",
                type="sphere",
                size=f"{radius:.9g}",
                pos=_values(*center),
                attrib={"class": "visual"},
                rgba="0.18 0.32 0.58 0.35",
            )

    for marker in marker_sites:
        ET.SubElement(
            body_elements[marker.body],
            "site",
            name=marker.site_name,
            type="sphere",
            size="0.008",
            pos=_values(*marker.position),
            rgba="0.15 0.95 0.25 1",
        )

    for mesh in visual_meshes:
        body = body_elements.get(mesh.body)
        if body is None:
            raise ValueError(f"visual mesh {mesh.name!r} references unknown body {mesh.body!r}")
        ET.SubElement(
            body,
            "geom",
            name=mesh.name,
            type="mesh",
            mesh=mesh.name,
            pos=_values(*mesh.position),
            attrib={"class": "visual"},
        )
    if visual_meshes:
        connector_bottom = 0.40 * config.pelvis_dimensions[2]
        connector_top = max(connector_bottom + 0.02, config.torso_center_offset - 0.35 * config.torso_dimensions[2])
        ET.SubElement(
            pelvis,
            "geom",
            name="geometry_abdomen_connector",
            type="box",
            pos=_values(0.0, 0.0, 0.5 * (connector_bottom + connector_top)),
            size=_values(
                0.25 * min(config.pelvis_dimensions[0], config.torso_dimensions[0]),
                0.25 * min(config.pelvis_dimensions[1], config.torso_dimensions[1]),
                0.5 * (connector_top - connector_bottom),
            ),
            rgba="0.68 0.48 0.30 1",
            attrib={"class": "visual"},
        )

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="neutral",
        qpos=_values(0.0, 0.0, config.pelvis_height, 1.0, 0.0, 0.0, 0.0, *([0.0] * 10)),
    )
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def write_subject_mjcf(
    config: SimpleGaitConfig,
    output_path: str | os.PathLike,
    *,
    model_name: str = "simple_gait_subject",
    visual_meshes: Sequence[SubjectVisualMesh] = (),
    include_fallback_geometry: bool = True,
    contact_centers: dict[str, tuple[tuple[float, float, float], ...]] | None = None,
    contact_radius: float | None = None,
    inertial_data: dict[str, SubjectInertial] | None = None,
    joint_centers: dict[str, tuple[float, float, float]] | None = None,
    marker_sites: Sequence[SubjectMarkerSite] = (),
) -> Path:
    """Write a scaled subject MJCF that Newton can load in one builder call."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        subject_mjcf_xml(
            config,
            model_name=model_name,
            visual_meshes=visual_meshes,
            include_fallback_geometry=include_fallback_geometry,
            contact_centers=contact_centers,
            contact_radius=contact_radius,
            inertial_data=inertial_data,
            joint_centers=joint_centers,
            marker_sites=marker_sites,
        ),
        encoding="utf-8",
    )
    return path
