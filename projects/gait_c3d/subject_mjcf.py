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


def _values(*items: float) -> str:
    return " ".join(f"{item:.9g}" for item in items)


def _box_inertia(mass: float, dimensions: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = dimensions
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def _add_inertial(
    body: ET.Element,
    mass: float,
    dimensions: tuple[float, float, float],
) -> None:
    ET.SubElement(
        body,
        "inertial",
        mass=f"{mass:.9g}",
        pos="0 0 0",
        diaginertia=_values(*_box_inertia(mass, dimensions)),
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
) -> str:
    """Create MJCF for one scaled simple-joint subject.

    Args:
        config: Subject-scaled dimensions, masses, and contact parameters.
        model_name: MJCF model label.
        visual_meshes: Body-local neutral mesh assets.
        include_fallback_geometry: Include box and capsule visuals when true.

    Returns:
        An MJCF XML document. It can be passed directly to
        :meth:`newton.ModelBuilder.add_mjcf`.
    """
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
    pelvis = ET.SubElement(world, "body", name="pelvis", pos=_values(0.0, 0.0, config.pelvis_height))
    ET.SubElement(pelvis, "freejoint", name="pelvis_free")
    _add_inertial(pelvis, config.pelvis_mass, config.pelvis_dimensions)
    if include_fallback_geometry:
        ET.SubElement(
            pelvis,
            "geom",
            name="geometry_pelvis",
            type="box",
            size=_values(*(0.5 * value for value in config.pelvis_dimensions)),
            attrib={"class": "visual"},
        )

    torso = ET.SubElement(pelvis, "body", name="torso", pos=_values(0.0, 0.0, config.torso_center_offset))
    _add_inertial(torso, config.torso_mass, config.torso_dimensions)
    if include_fallback_geometry:
        ET.SubElement(
            torso,
            "geom",
            name="geometry_torso",
            type="box",
            size=_values(*(0.5 * value for value in config.torso_dimensions)),
            attrib={"class": "visual"},
        )

    body_elements = {"pelvis": pelvis, "torso": torso}
    actuator = ET.SubElement(root, "actuator")
    degrees = math.pi / 180.0
    for side, lateral_sign in (("left", 1.0), ("right", -1.0)):
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
        )
        hip_position = (0.0, 0.0, 0.5 * config.thigh_length)
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
                size=f"{config.thigh_radius:.9g}",
                fromto=_values(0.0, 0.0, -0.5 * config.thigh_length, 0.0, 0.0, 0.5 * config.thigh_length),
                attrib={"class": "visual"},
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
        )
        knee_name = f"knee_{side}"
        knee_limits = (0.0, 150.0 * degrees)
        _add_joint(
            tibia,
            name=knee_name,
            position=(0.0, 0.0, 0.5 * config.shank_length),
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
                size=f"{config.shank_radius:.9g}",
                fromto=_values(0.0, 0.0, -0.5 * config.shank_length, 0.0, 0.0, 0.5 * config.shank_length),
                attrib={"class": "visual"},
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
        )
        ankle_name = f"ankle_{side}"
        ankle_limits = (-50.0 * degrees, 30.0 * degrees)
        _add_joint(
            foot,
            name=ankle_name,
            position=(-0.4 * config.foot_length, 0.0, config.contact_radius),
            axis=(0.0, -1.0, 0.0),
            limits=ankle_limits,
            damping=0.2,
            armature=0.005,
        )
        _add_target_actuators(actuator, ankle_name, ankle_limits)
        heel_x = -0.32 * config.foot_length
        forefoot_x = 0.48 * config.foot_length
        half_width = 0.35 * config.foot_width
        for index, center in enumerate(
            (
                (heel_x, -half_width, -config.contact_radius),
                (heel_x, half_width, -config.contact_radius),
                (forefoot_x, -half_width, -config.contact_radius),
                (forefoot_x, half_width, -config.contact_radius),
            )
        ):
            ET.SubElement(
                foot,
                "geom",
                name=f"contact_{side}_{index}",
                type="sphere",
                size=f"{config.contact_radius:.9g}",
                pos=_values(*center),
                attrib={"class": "collision"},
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
        ),
        encoding="utf-8",
    )
    return path
