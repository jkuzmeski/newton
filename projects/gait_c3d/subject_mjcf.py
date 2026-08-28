# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export a scaled simple gait subject as interoperable MJCF."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .native_model import SimpleGaitConfig


def _canonical_json(value: dict) -> bytes:
    """Serialize manifest content deterministically for a SHA-256 seal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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
                (heel_x, -half_width, -radius),
                (heel_x, half_width, -radius),
                (forefoot_x, -half_width, -radius),
                (forefoot_x, half_width, -radius),
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
                -0.5 * config.shank_length - radius,
            ),
        )
        body_elements[f"foot_{side}"] = foot
        _add_inertial(
            foot,
            config.foot_mass,
            (config.foot_length, config.foot_width, 2.0 * radius),
            inertials.get(f"foot_{side}"),
        )
        ankle_name = f"ankle_{side}"
        ankle_limits = (-50.0 * degrees, 30.0 * degrees)
        _add_joint(
            foot,
            name=ankle_name,
            position=centers.get(f"ankle_{side}", (-0.4 * config.foot_length, 0.0, radius)),
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


@dataclass(frozen=True, slots=True)
class ScaledSubjectMJCF:
    """A native MJCF scaled from a compiled base subject."""

    path: Path
    """Scaled MJCF path."""

    base_subject: Path
    """Compiled subject bundle used as the scaling reference."""

    config: SimpleGaitConfig
    """Target native configuration derived from the base subject."""

    length_scale: float
    """Uniform length scale relative to the base subject."""

    mass_scale: float
    """Uniform mass scale relative to the base subject."""


def _format_scaled_values(text: str, scale: float, *, name: str) -> str:
    """Scale a finite whitespace-separated MJCF vector."""
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"MJCF {name} must contain finite numeric values")
    return " ".join(f"{float(value * scale):.9g}" for value in values)


def _scale_obj_vertices(source: Path, destination: Path, scale: float) -> None:
    """Copy an OBJ while applying a uniform scale to its vertex positions."""
    output = []
    for source_line in source.read_text(encoding="utf-8").splitlines():
        fields = source_line.split()
        output_line = source_line
        if fields and fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"OBJ vertex is incomplete: {source}")
            values = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"OBJ vertex is nonfinite: {source}")
            output_line = "v " + " ".join(f"{float(value * scale):.9g}" for value in values)
        output.append(output_line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def scale_subject_mjcf_from_base(
    base_subject_dir: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    body_height: float,
    body_mass: float,
    hip_width: float | None = None,
    model_name: str | None = None,
) -> ScaledSubjectMJCF:
    """Scale a complete native subject bundle from its neutral base geometry.

    The source MJCF, body-local OBJ meshes, marker site positions, joint frames,
    contacts, COMs, and inertia tensors are all scaled together. This keeps the
    S001 subject placement as the reference instead of mixing nominal geometry
    with subject-specific marker attachments.

    Args:
        base_subject_dir: Compiled base subject bundle, normally S001.
        output_path: Destination MJCF path. Its sibling ``Geometry`` directory
            and model manifest are created as part of the scaled bundle.
        body_height: Target subject standing height [m].
        body_mass: Target subject body mass [kg].
        hip_width: Optional target hip-joint center spacing [m].
        model_name: Optional MJCF model name.

    Returns:
        Metadata and path for the scaled native subject.

    Raises:
        FileNotFoundError: If the base bundle is incomplete.
        ValueError: If the base bundle or target scale is invalid.
    """
    base_root = Path(base_subject_dir).expanduser().resolve()
    base_xml = base_root / "model" / "subject.xml"
    base_model_manifest = base_root / "model" / "manifest.json"
    base_bundle_manifest = base_root / "subject.json"
    base_marker_layout = base_root / "model" / "marker_layout.json"
    for path in (base_xml, base_model_manifest, base_bundle_manifest, base_marker_layout):
        if not path.is_file():
            raise FileNotFoundError(f"base subject artifact is missing: {path}")
    output = Path(output_path).expanduser().resolve()
    if output == base_xml:
        raise ValueError("scaled subject output must not overwrite its base MJCF")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if (output.parent / "Geometry").exists() or (output.parent / "manifest.json").exists():
        raise FileExistsError(f"scaled subject output directory is not empty: {output.parent}")

    try:
        model_manifest = json.loads(base_model_manifest.read_text(encoding="utf-8"))
        bundle_manifest = json.loads(base_bundle_manifest.read_text(encoding="utf-8"))
        marker_layout_manifest = json.loads(base_marker_layout.read_text(encoding="utf-8"))
        if not isinstance(marker_layout_manifest, dict) or not isinstance(marker_layout_manifest.get("seal"), dict):
            raise ValueError("base subject marker layout must be a sealed JSON object")
        marker_layout_seal = marker_layout_manifest.pop("seal")
        expected_layout_hash = hashlib.sha256(_canonical_json(marker_layout_manifest)).hexdigest()
        if marker_layout_seal != {"algorithm": "sha256", "content_sha256": expected_layout_hash}:
            raise ValueError("base subject marker layout seal mismatch")
        native_layout_metadata = model_manifest.get("native_marker_layout")
        if isinstance(native_layout_metadata, dict):
            expected_layout_file_hash = native_layout_metadata.get("sha256")
            actual_layout_file_hash = hashlib.sha256(base_marker_layout.read_bytes()).hexdigest()
            if expected_layout_file_hash != actual_layout_file_hash:
                raise ValueError("base subject marker layout hash mismatch")
        native_metadata = model_manifest.get("native_subject")
        if isinstance(native_metadata, dict):
            expected_xml_hash = native_metadata.get("sha256")
            actual_xml_hash = hashlib.sha256(base_xml.read_bytes()).hexdigest()
            if expected_xml_hash != actual_xml_hash:
                raise ValueError("base subject MJCF hash mismatch")
        base_values = model_manifest.get("simple_model_config")
        base_metadata = bundle_manifest.get("subject")
        if not isinstance(base_values, dict) or not isinstance(base_metadata, dict):
            raise ValueError("base subject does not contain a simple model configuration")
        base_config = SimpleGaitConfig(**base_values)
        root = ET.parse(base_xml).getroot()
        expected_mesh_hashes = {
            record.get("output", {}).get("file"): record.get("output", {}).get("sha256")
            for record in model_manifest.get("meshes", [])
            if isinstance(record, dict)
        }
        base_pelvis = next(
            (
                element
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1] == "body" and element.get("name") == "pelvis"
            ),
            None,
        )
        if base_pelvis is None:
            raise ValueError("base MJCF is missing its pelvis body")
        base_pelvis_position = np.asarray(
            [float(value) for value in base_pelvis.get("pos", "").split()], dtype=np.float64
        )
        if base_pelvis_position.shape != (3,) or not np.all(np.isfinite(base_pelvis_position)):
            raise ValueError("base MJCF pelvis position must contain three finite values")
        # The compiler's visual-ground registration is already baked into the
        # final MJCF, so use that final root height rather than the pre-offset
        # value retained in the visual compiler manifest.
        contact_radii = [
            float(element.get("size", "nan"))
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "geom"
            and (element.get("name") or "").startswith(("contact_left_", "contact_right_"))
        ]
        if (
            not contact_radii
            or not np.all(np.isfinite(contact_radii))
            or not np.allclose(contact_radii, contact_radii[0])
        ):
            raise ValueError("base MJCF must contain finite, uniform foot contact radii")
        base_config = replace(
            base_config,
            pelvis_height=float(base_pelvis_position[2]),
            contact_radius=contact_radii[0],
        )
        base_height = float(base_metadata.get("height_m", "nan"))
        base_mass_metadata = float(base_metadata.get("mass_kg", "nan"))
        if not math.isfinite(base_height) or base_height <= 0.0:
            raise ValueError("base subject height must be finite and positive")
        base_mass = (
            base_config.pelvis_mass
            + base_config.torso_mass
            + 2.0 * (base_config.thigh_mass + base_config.shank_mass + base_config.foot_mass)
        )
        if not math.isfinite(base_mass) or base_mass <= 0.0:
            raise ValueError("base subject mass must be finite and positive")
        if math.isfinite(base_mass_metadata) and not math.isclose(base_mass, base_mass_metadata, rel_tol=1.0e-5):
            raise ValueError("base subject mass metadata does not match its native configuration")
        target_config = SimpleGaitConfig.for_subject_from_base(
            base_config,
            base_height=base_height,
            body_mass=body_mass,
            body_height=body_height,
            hip_width=hip_width,
        )
        length_scale = body_height / base_height
        mass_scale = body_mass / base_mass
        if model_name is not None:
            if not model_name or any(character.isspace() for character in model_name):
                raise ValueError("model_name must be nonempty and contain no whitespace")
            root.set("model", model_name)

        staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            staged_xml = staged / output.name
            staged_geometry = staged / "Geometry"
            copied_meshes: set[Path] = set()
            for element in root.iter():
                tag = element.tag.rsplit("}", 1)[-1]
                for attribute in ("pos", "fromto", "size"):
                    if attribute in element.attrib:
                        element.set(
                            attribute,
                            _format_scaled_values(element.attrib[attribute], length_scale, name=attribute),
                        )
                if tag == "inertial":
                    if "mass" not in element.attrib or "fullinertia" not in element.attrib:
                        raise ValueError("base MJCF inertial must contain mass and fullinertia")
                    element.set("mass", f"{float(element.attrib['mass']) * mass_scale:.9g}")
                    element.set(
                        "fullinertia",
                        _format_scaled_values(
                            element.attrib["fullinertia"],
                            mass_scale * length_scale * length_scale,
                            name="fullinertia",
                        ),
                    )
                elif tag == "key" and "qpos" in element.attrib:
                    values = np.asarray([float(value) for value in element.attrib["qpos"].split()], dtype=np.float64)
                    if values.size < 3 or not np.all(np.isfinite(values)):
                        raise ValueError("base MJCF keyframe qpos must contain finite values")
                    values[:3] *= length_scale
                    element.set("qpos", " ".join(f"{float(value):.9g}" for value in values))
                elif tag == "mesh":
                    mesh_file = element.get("file")
                    if not mesh_file:
                        raise ValueError("base MJCF mesh is missing its file")
                    relative = Path(mesh_file)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"base MJCF mesh path is unsafe: {mesh_file!r}")
                    source_mesh = (base_xml.parent / relative).resolve()
                    try:
                        source_mesh.relative_to(base_xml.parent.resolve())
                    except ValueError as error:
                        raise ValueError(f"base MJCF mesh path escapes its bundle: {mesh_file!r}") from error
                    if not source_mesh.is_file():
                        raise FileNotFoundError(f"base MJCF mesh is missing: {source_mesh}")
                    expected_mesh_hash = expected_mesh_hashes.get(relative.as_posix())
                    if expected_mesh_hash is not None:
                        actual_mesh_hash = hashlib.sha256(source_mesh.read_bytes()).hexdigest()
                        if expected_mesh_hash != actual_mesh_hash:
                            raise ValueError(f"base MJCF mesh hash mismatch: {relative.as_posix()}")
                    destination_mesh = staged / relative
                    if destination_mesh not in copied_meshes:
                        if source_mesh.suffix.lower() == ".obj":
                            _scale_obj_vertices(source_mesh, destination_mesh, length_scale)
                        else:
                            destination_mesh.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source_mesh, destination_mesh)
                        copied_meshes.add(destination_mesh)
            # The femur body translations encode the two hip centers in this
            # reduced model. Apply an explicit target width after uniform scaling.
            for body in root.iter():
                if body.tag.rsplit("}", 1)[-1] != "body" or body.get("name") not in {
                    "femur_left",
                    "femur_right",
                }:
                    continue
                position = np.asarray([float(value) for value in body.get("pos", "").split()], dtype=np.float64)
                if position.shape != (3,) or not np.all(np.isfinite(position)):
                    raise ValueError(f"base MJCF femur body {body.get('name')!r} has an invalid position")
                position[1] = (
                    target_config.hip_half_width if body.get("name") == "femur_left" else -target_config.hip_half_width
                )
                body.set("pos", " ".join(f"{float(value):.9g}" for value in position))
            staged_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
            base_marker_set = bundle_manifest.get("base_marker_set")
            if base_marker_set is None:
                base_marker_set = bundle_manifest.get("sources", {}).get("base_marker_set")
            scaled_manifest = {
                "schema_version": "gait_subject_mjcf_from_base_1",
                "coordinate_system": {"frame": "Newton world/body-local", "length_unit": "m", "up_axis": "Z"},
                "base_marker_set": base_marker_set,
                "source_subject": base_root.name,
                "source_model": {
                    "file": base_xml.name,
                    "sha256": hashlib.sha256(base_xml.read_bytes()).hexdigest(),
                },
                "source_marker_layout": {
                    "file": base_marker_layout.name,
                    "sha256": hashlib.sha256(base_marker_layout.read_bytes()).hexdigest(),
                },
                "meshes": [
                    {
                        "file": path.relative_to(staged).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in sorted(copied_meshes)
                ],
                "scale": {"length": length_scale, "mass": mass_scale},
                "simple_model_config": asdict(target_config),
            }
            scaled_manifest["seal"] = {
                "algorithm": "sha256",
                "content_sha256": hashlib.sha256(_canonical_json(scaled_manifest)).hexdigest(),
            }
            (staged / "manifest.json").write_text(
                json.dumps(scaled_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )
            shutil.copytree(staged_geometry, output.parent / "Geometry")
            os.replace(staged_xml, output)
            os.replace(staged / "manifest.json", output.parent / "manifest.json")
        finally:
            shutil.rmtree(staged, ignore_errors=True)
    except Exception:
        if output.exists():
            output.unlink()
        shutil.rmtree(output.parent / "Geometry", ignore_errors=True)
        manifest = output.parent / "manifest.json"
        if manifest.exists():
            manifest.unlink()
        raise
    return ScaledSubjectMJCF(output, base_root, target_config, length_scale, mass_scale)


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
