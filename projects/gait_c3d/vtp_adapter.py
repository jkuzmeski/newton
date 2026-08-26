# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile scaled OpenSim VTP display geometry into neutral MJCF assets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .native_model import SimpleGaitConfig
from .subject_mjcf import SubjectVisualMesh

_SCHEMA = "gait_c3d_scaled_vtp_visuals_1"
_SOURCE_TO_TARGET = {
    "pelvis": "pelvis",
    "torso": "torso",
    "femur_l": "femur_left",
    "femur_r": "femur_right",
    "tibia_l": "tibia_left",
    "tibia_r": "tibia_right",
    "talus_l": "foot_left",
    "talus_r": "foot_right",
    "calcn_l": "foot_left",
    "calcn_r": "foot_right",
    "toes_l": "foot_left",
    "toes_r": "foot_right",
}
_LEGACY_ALIASES = {
    ("femur_r", "femur.vtp"): ("femur_r.vtp", "r_femur.vtp"),
    ("tibia_r", "tibia.vtp"): ("r_tibia.vtp", "tibia_r.vtp"),
}
_OPENSIM_TO_NEWTON = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class DisplayGeometry:
    """One scaled OpenSim display mesh before neutral conversion."""

    source_body: str
    geometry_file: str
    transform: np.ndarray
    scale: np.ndarray
    mass_center: np.ndarray


@dataclass(frozen=True, slots=True)
class FootContactLayout:
    """Mesh-derived sphere contact layout for both merged feet."""

    radius: float
    centers: dict[str, tuple[tuple[float, float, float], ...]]
    visual_ground_offset_z: float


@dataclass(frozen=True, slots=True)
class CompiledVisuals:
    """Neutral visual assets and their sealed manifest."""

    root: Path
    manifest_path: Path
    meshes: tuple[SubjectVisualMesh, ...]
    contact_layout: FootContactLayout | None = None


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _floats(text: str | None) -> np.ndarray:
    return np.asarray([float(value) for value in (text or "").replace(",", " ").split()], dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _transform(values: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if len(values) == 0:
        return result
    if values.shape != (6,):
        raise ValueError("OpenSim display transform must contain six values")
    result[:3, :3] = _rotation_x(values[0]) @ _rotation_y(values[1]) @ _rotation_z(values[2])
    result[:3, 3] = values[3:]
    return result


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [child for child in element if child.tag.rsplit("}", 1)[-1] == name]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    children = _children(element, name)
    return children[0] if children else None


def _data_array(parent: ET.Element | None, *, name: str | None = None) -> np.ndarray:
    for array in _children(parent, "DataArray"):
        if name is not None and array.get("Name") != name:
            continue
        if (array.get("format") or "ascii").lower() != "ascii":
            raise NotImplementedError("only ASCII VTP DataArray values are supported")
        return _floats(array.text)
    raise ValueError(f"missing VTP DataArray {name or ''}".rstrip())


def _cells(
    piece: ET.Element,
    name: str,
    *,
    point_offset: int,
    point_count: int,
) -> list[tuple[int, int, int]]:
    parent = _child(piece, name)
    if parent is None:
        return []
    connectivity_values = _data_array(parent, name="connectivity")
    offset_values = _data_array(parent, name="offsets")
    if not np.all(connectivity_values == np.floor(connectivity_values)) or not np.all(
        offset_values == np.floor(offset_values)
    ):
        raise ValueError("VTP connectivity and offsets must be integers")
    connectivity = connectivity_values.astype(np.int64)
    offsets = offset_values.astype(np.int64)
    if np.any(connectivity < 0) or np.any(connectivity >= point_count):
        raise ValueError("VTP connectivity references a point outside its Piece")
    if np.any(offsets <= 0) or np.any(np.diff(offsets) <= 0) or (len(offsets) and offsets[-1] != len(connectivity)):
        raise ValueError("VTP cell offsets are inconsistent with connectivity")
    triangles: list[tuple[int, int, int]] = []
    start = 0
    for end in offsets:
        cell = connectivity[start:end]
        if len(cell) < 3:
            raise ValueError("VTP polygon and strip cells need at least three vertices")
        if name == "Polys":
            triangles.extend(
                (point_offset + int(cell[0]), point_offset + int(cell[index]), point_offset + int(cell[index + 1]))
                for index in range(1, len(cell) - 1)
            )
        else:
            for index in range(len(cell) - 2):
                a, b, c = (point_offset + int(value) for value in cell[index : index + 3])
                triangles.append((a, b, c) if index % 2 == 0 else (b, a, c))
        start = int(end)
    return triangles


def read_vtp(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read deterministic ASCII VTK PolyData and triangulate polygons/strips."""
    root = ET.parse(path).getroot()
    pieces = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Piece"]
    if not pieces:
        raise ValueError(f"{path}: not a VTK PolyData file")
    point_blocks: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    point_offset = 0
    for piece in pieces:
        points = _data_array(_child(piece, "Points")).reshape(-1, 3)
        if not np.all(np.isfinite(points)):
            raise ValueError("VTP points must be finite")
        point_blocks.append(points)
        triangles.extend(_cells(piece, "Polys", point_offset=point_offset, point_count=len(points)))
        triangles.extend(_cells(piece, "Strips", point_offset=point_offset, point_count=len(points)))
        point_offset += len(points)
    vertices = np.concatenate(point_blocks, axis=0).astype(np.float32)
    faces = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    if len(faces) == 0:
        raise ValueError("VTP mesh contains no triangles")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("VTP triangle index is outside the point array")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])):
        raise ValueError("VTP mesh contains a degenerate indexed triangle")
    return vertices, faces


def _owned_joint(body: ET.Element) -> ET.Element:
    for element in body.iter():
        if element.tag.rsplit("}", 1)[-1].endswith("Joint") and element.tag.rsplit("}", 1)[-1] != "Joint":
            return element
    raise ValueError(f"body {body.get('name')!r} has no owned joint")


def _function_value_at_zero(function: ET.Element) -> float:
    tag = function.tag.rsplit("}", 1)[-1]
    if tag == "Constant":
        value_element = _child(function, "value")
        return float(value_element.text if value_element is not None else "nan")
    if tag == "LinearFunction":
        coefficients_element = _child(function, "coefficients")
        coefficients = _floats(coefficients_element.text if coefficients_element is not None else None)
        if coefficients.shape != (2,):
            raise ValueError("LinearFunction must contain slope and intercept")
        return float(coefficients[1])
    if tag in {"NaturalCubicSpline", "SimmSpline"}:
        x_element = _child(function, "x")
        y_element = _child(function, "y")
        x = _floats(x_element.text if x_element is not None else None)
        y = _floats(y_element.text if y_element is not None else None)
        if len(x) != len(y) or len(x) < 2 or np.any(np.diff(x) <= 0.0):
            raise ValueError(f"{tag} knots are invalid")
        return float(np.interp(0.0, x, y))
    if tag == "MultiplierFunction":
        scale_element = _child(function, "scale")
        wrapper = _child(function, "function")
        if scale_element is None or wrapper is None or len(wrapper) != 1:
            raise ValueError("MultiplierFunction is missing its scale or nested function")
        return float(scale_element.text) * _function_value_at_zero(wrapper[0])
    raise ValueError(f"unsupported zero-pose joint function: {tag}")


def _joint_translation_at_zero(joint: ET.Element) -> np.ndarray:
    spatial = _child(joint, "SpatialTransform")
    translation = np.zeros(3, dtype=np.float64)
    for transform_axis in _children(spatial, "TransformAxis"):
        if not (transform_axis.get("name") or "").startswith("translation"):
            continue
        axis_element = _child(transform_axis, "axis")
        function_wrapper = _child(transform_axis, "function")
        if function_wrapper is not None and len(function_wrapper) == 1:
            function = function_wrapper[0]
        else:
            function_candidates = [
                child for child in transform_axis if child.tag.rsplit("}", 1)[-1] not in {"coordinates", "axis"}
            ]
            if len(function_candidates) != 1:
                raise ValueError("joint translation axis is missing its scalar function")
            function = function_candidates[0]
        if axis_element is None:
            raise ValueError("joint translation axis is missing its axis")
        axis = _floats(axis_element.text)
        value = _function_value_at_zero(function)
        if axis.shape != (3,) or not math.isfinite(value):
            raise ValueError("joint translation axis/value is invalid")
        translation += axis * value
    return translation


def simple_config_from_scaled_gait2354(
    path: str | os.PathLike,
    *,
    body_height: float,
) -> SimpleGaitConfig:
    """Derive the simple native subject parameters from one scaled gait2354 model."""
    root = ET.parse(path).getroot()
    bodies = {
        body.get("name", ""): body for body in root.iter() if body.tag.rsplit("}", 1)[-1] == "Body" and body.get("name")
    }
    required = {
        "pelvis",
        "torso",
        "femur_l",
        "femur_r",
        "tibia_l",
        "tibia_r",
        "talus_l",
        "talus_r",
        "calcn_l",
        "calcn_r",
        "toes_l",
        "toes_r",
    }
    if not required.issubset(bodies):
        raise ValueError(f"scaled gait2354 model is missing bodies: {sorted(required - bodies.keys())}")

    def mass(name: str) -> float:
        element = _child(bodies[name], "mass")
        value = float(element.text if element is not None else "nan")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"body {name!r} has an invalid mass")
        return value

    def mass_center(name: str) -> np.ndarray:
        element = _child(bodies[name], "mass_center")
        value = _floats(element.text if element is not None else None)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"body {name!r} has an invalid mass center")
        return value

    version = int(root.get("Version", "0"))
    joints = {
        joint.get("name", ""): joint
        for joint in root.iter()
        if joint.tag.rsplit("}", 1)[-1].endswith("Joint")
        and joint.tag.rsplit("}", 1)[-1] != "Joint"
        and joint.get("name")
    }

    def joint_frame_translation(joint: ET.Element, socket_tag: str) -> np.ndarray:
        frame_name = (_child(joint, socket_tag).text or "").strip()
        frame = next(
            (
                candidate
                for candidate in joint.iter()
                if candidate.tag.rsplit("}", 1)[-1] == "PhysicalOffsetFrame" and candidate.get("name") == frame_name
            ),
            None,
        )
        if frame is None:
            raise ValueError(f"joint {joint.get('name')!r} is missing frame {frame_name!r}")
        translation = _floats(_child(frame, "translation").text or "")
        if translation.shape != (3,):
            raise ValueError(f"joint frame {frame_name!r} has an invalid translation")
        return translation

    pelvis_com = mass_center("pelvis")
    hip_locations = []
    thigh_lengths = []
    shank_lengths = []
    for side in ("l", "r"):
        if version >= 30000:
            hip_joint = joints[f"hip_{side}"]
            hip_parent = joint_frame_translation(hip_joint, "socket_parent_frame")
            knee_joint = joints[f"knee_{side}"]
            ankle_joint = joints[f"ankle_{side}"]
            ankle_parent = joint_frame_translation(ankle_joint, "socket_parent_frame")
        else:
            hip_joint = _owned_joint(bodies[f"femur_{side}"])
            hip_parent_element = _child(hip_joint, "location_in_parent")
            hip_parent = _floats(hip_parent_element.text if hip_parent_element is not None else None)
            knee_joint = _owned_joint(bodies[f"tibia_{side}"])
            ankle_joint = _owned_joint(bodies[f"talus_{side}"])
            ankle_parent_element = _child(ankle_joint, "location_in_parent")
            ankle_parent = _floats(ankle_parent_element.text if ankle_parent_element is not None else None)
        if hip_parent.shape != (3,) or ankle_parent.shape != (3,):
            raise ValueError(f"scaled leg {side!r} has an invalid parent joint location")
        hip_locations.append(hip_parent - pelvis_com)
        thigh_lengths.append(float(np.linalg.norm(_joint_translation_at_zero(knee_joint))))
        shank_lengths.append(float(np.linalg.norm(ankle_parent)))

    if version >= 30000:
        back_joint = joints["back"]
        back_parent = joint_frame_translation(back_joint, "socket_parent_frame") - pelvis_com
        back_child = joint_frame_translation(back_joint, "socket_child_frame") - mass_center("torso")
    else:
        back_joint = _owned_joint(bodies["torso"])
        back_parent_element = _child(back_joint, "location_in_parent")
        back_child_element = _child(back_joint, "location")
        back_parent = _floats(back_parent_element.text if back_parent_element is not None else None) - pelvis_com
        back_child = _floats(back_child_element.text if back_child_element is not None else None) - mass_center("torso")
    torso_offset = float(back_parent[1] - back_child[1])
    hip_half_width = float(np.mean([abs(location[2]) for location in hip_locations]))
    hip_drop = float(-np.mean([location[1] for location in hip_locations]))
    foot_mass = np.mean([mass(f"talus_{side}") + mass(f"calcn_{side}") + mass(f"toes_{side}") for side in ("l", "r")])
    total_mass = sum(mass(name) for name in required)
    reference = SimpleGaitConfig.for_subject(
        body_mass=total_mass,
        body_height=body_height,
        hip_width=2.0 * hip_half_width,
    )
    thigh_length = float(np.mean(thigh_lengths))
    shank_length = float(np.mean(shank_lengths))
    pelvis_height = hip_drop + thigh_length + shank_length + 3.0 * reference.contact_radius
    return replace(
        reference,
        pelvis_height=pelvis_height,
        pelvis_mass=mass("pelvis"),
        torso_mass=mass("torso"),
        thigh_mass=0.5 * (mass("femur_l") + mass("femur_r")),
        shank_mass=0.5 * (mass("tibia_l") + mass("tibia_r")),
        foot_mass=float(foot_mass),
        hip_half_width=hip_half_width,
        thigh_length=thigh_length,
        shank_length=shank_length,
        pelvis_hip_drop=hip_drop,
        torso_center_offset=torso_offset,
    )


def read_scaled_display_geometry(path: str | os.PathLike) -> tuple[DisplayGeometry, ...]:
    """Read scaled legacy or modern display entries needed by the simple gait body map."""
    root = ET.parse(path).getroot()
    version = int(root.get("Version", "0"))
    output: list[DisplayGeometry] = []
    seen_bodies: set[str] = set()
    for body in (element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Body"):
        source_body = body.get("name", "")
        if source_body not in _SOURCE_TO_TARGET:
            continue
        if source_body in seen_bodies:
            raise ValueError(f"scaled model contains duplicate body {source_body!r}")
        seen_bodies.add(source_body)
        mass_center_element = _child(body, "mass_center")
        mass_center = _floats(mass_center_element.text if mass_center_element is not None else None)
        if mass_center.shape != (3,) or not np.all(np.isfinite(mass_center)):
            raise ValueError(f"body {source_body!r} needs a finite three-component mass center")
        if version >= 30000:
            for mesh in (element for element in body.iter() if element.tag.rsplit("}", 1)[-1] == "Mesh"):
                file_element = _child(mesh, "mesh_file")
                if file_element is None or not (file_element.text or "").strip():
                    continue
                socket_element = _child(mesh, "socket_frame")
                socket_frame = (socket_element.text or "..").strip() if socket_element is not None else ".."
                if socket_frame not in {"..", f"/bodyset/{source_body}"}:
                    raise ValueError(f"modern mesh {mesh.get('name')!r} uses unsupported offset frame {socket_frame!r}")
                scale_element = _child(mesh, "scale_factors")
                scale = _floats(scale_element.text if scale_element is not None else None)
                if len(scale) == 0:
                    scale = np.ones(3)
                if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
                    raise ValueError(f"modern mesh on {source_body!r} has an invalid scale")
                output.append(
                    DisplayGeometry(
                        source_body,
                        (file_element.text or "").strip(),
                        np.eye(4),
                        scale,
                        mass_center,
                    )
                )
            continue
        for visible in (element for element in body.iter() if element.tag.rsplit("}", 1)[-1] == "VisibleObject"):
            visible_transform_element = _child(visible, "transform")
            visible_scale_element = _child(visible, "scale_factors")
            visible_transform = _transform(
                _floats(visible_transform_element.text if visible_transform_element is not None else None)
            )
            visible_scale = _floats(visible_scale_element.text if visible_scale_element is not None else None)
            if len(visible_scale) == 0:
                visible_scale = np.ones(3)
            if visible_scale.shape != (3,) or not np.all(np.isfinite(visible_scale)):
                raise ValueError(f"visible object on {source_body!r} has an invalid scale")
            for display in (
                element for element in visible.iter() if element.tag.rsplit("}", 1)[-1] == "DisplayGeometry"
            ):
                file_element = _child(display, "geometry_file")
                if file_element is None or not (file_element.text or "").strip():
                    continue
                display_transform_element = _child(display, "transform")
                display_scale_element = _child(display, "scale_factors")
                display_scale = _floats(display_scale_element.text if display_scale_element is not None else None)
                if len(display_scale) == 0:
                    display_scale = np.ones(3)
                if display_scale.shape != (3,) or not np.all(np.isfinite(display_scale)):
                    raise ValueError(f"display geometry on {source_body!r} has an invalid scale")
                if not np.allclose(visible_scale, 1.0) and not np.allclose(display_scale, 1.0):
                    raise ValueError(
                        f"display geometry on {source_body!r} has two nonidentity legacy scale levels; "
                        "use a scale artifact that applies subject scaling exactly once"
                    )
                effective_scale = visible_scale * display_scale
                if (
                    effective_scale.shape != (3,)
                    or not np.all(np.isfinite(effective_scale))
                    or np.any(effective_scale <= 0.0)
                ):
                    raise ValueError(f"display geometry on {source_body!r} has an invalid scale")
                output.append(
                    DisplayGeometry(
                        source_body,
                        (file_element.text or "").strip(),
                        visible_transform
                        @ _transform(
                            _floats(display_transform_element.text if display_transform_element is not None else None)
                        ),
                        effective_scale,
                        mass_center,
                    )
                )
    return tuple(output)


def _resolve_geometry(entry: DisplayGeometry, geometry_dir: Path) -> Path:
    filename = Path(entry.geometry_file).name
    candidates = [filename, *_LEGACY_ALIASES.get((entry.source_body, filename), ())]
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for side in ("r", "l"):
        if entry.source_body.endswith(f"_{side}"):
            candidates.extend((f"{stem}_{side}{suffix}", f"{side}_{stem}{suffix}"))
    for candidate in candidates:
        path = geometry_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not resolve {filename!r} for body {entry.source_body!r}")


def _write_obj(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    lines = [*(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _target_body_transforms(config: SimpleGaitConfig) -> dict[str, np.ndarray]:
    """Return neutral simple-model body transforms in Newton world coordinates."""
    transforms = {}

    def translated(x: float, y: float, z: float) -> np.ndarray:
        result = np.eye(4)
        result[:3, 3] = (x, y, z)
        return result

    transforms["pelvis"] = translated(0.0, 0.0, config.pelvis_height)
    transforms["torso"] = translated(0.0, 0.0, config.pelvis_height + config.torso_center_offset)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        femur_z = config.pelvis_height - config.pelvis_hip_drop - 0.5 * config.thigh_length
        tibia_z = femur_z - 0.5 * (config.thigh_length + config.shank_length)
        transforms[f"femur_{side}"] = translated(0.0, sign * config.hip_half_width, femur_z)
        transforms[f"tibia_{side}"] = translated(0.0, sign * config.hip_half_width, tibia_z)
        transforms[f"foot_{side}"] = translated(
            0.4 * config.foot_length,
            sign * config.hip_half_width,
            tibia_z - 0.5 * config.shank_length - config.contact_radius,
        )
    return transforms


def _compile_scaled_vtp_visuals(
    scaled_osim: str | os.PathLike,
    geometry_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    config: SimpleGaitConfig,
    source_body_transforms: dict[str, np.ndarray] | None,
) -> CompiledVisuals:
    """Compile into a new staging directory."""
    model_path = Path(scaled_osim).resolve()
    source_geometry = Path(geometry_dir).resolve()
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    assets_dir = root / "Geometry"
    assets_dir.mkdir(parents=True)
    mesh_records = []
    meshes: list[SubjectVisualMesh] = []
    compiled_geometry: list[tuple[Path, str, np.ndarray, np.ndarray, dict]] = []
    exact_transforms = source_body_transforms is not None
    target_transforms = _target_body_transforms(config)
    source_transforms = source_body_transforms or {}
    for name, transform in source_transforms.items():
        array = np.asarray(transform, dtype=np.float64)
        if array.shape != (4, 4) or not np.all(np.isfinite(array)) or not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError(f"source body transform {name!r} is invalid")
        source_transforms[name] = array
    for entry_index, entry in enumerate(read_scaled_display_geometry(model_path)):
        source_path = _resolve_geometry(entry, source_geometry)
        vertices, triangles = read_vtp(source_path)
        target_body = _SOURCE_TO_TARGET[entry.source_body]
        if target_body.startswith("foot_") and not exact_transforms:
            continue
        scaled = vertices.astype(np.float64) * entry.scale
        transformed = scaled @ entry.transform[:3, :3].T + entry.transform[:3, 3]
        if exact_transforms:
            if entry.source_body not in source_transforms:
                raise ValueError(f"missing official neutral transform for body {entry.source_body!r}")
            source_transform = source_transforms[entry.source_body]
            target_transform = target_transforms[target_body]
            ground_opensim = transformed @ source_transform[:3, :3].T + source_transform[:3, 3]
            ground_newton = ground_opensim @ _OPENSIM_TO_NEWTON.T
            body_local = (ground_newton - target_transform[:3, 3]) @ target_transform[:3, :3]
            source_proximal = source_transform[:3, 3] @ _OPENSIM_TO_NEWTON.T
            target_proximal = target_transform[:3, 3]
            registration = target_proximal - source_proximal
        else:
            body_local = (transformed - entry.mass_center) @ _OPENSIM_TO_NEWTON.T
            source_proximal = (-entry.mass_center) @ _OPENSIM_TO_NEWTON.T
            if target_body == "torso":
                target_proximal = np.asarray((0.0, 0.0, -0.5 * config.torso_center_offset))
            elif target_body.startswith("femur_"):
                target_proximal = np.asarray((0.0, 0.0, 0.5 * config.thigh_length))
            elif target_body.startswith("tibia_"):
                target_proximal = np.asarray((0.0, 0.0, 0.5 * config.shank_length))
            else:
                target_proximal = source_proximal
            registration = target_proximal - source_proximal
            body_local += registration
        source_hash = _sha256(source_path)
        name = f"visual_{entry_index:02d}_{target_body}_{source_path.stem}_{source_hash[:8]}"
        output_path = assets_dir / f"{name}.obj"
        _write_obj(output_path, body_local, triangles)
        relative = output_path.relative_to(root).as_posix()
        meshes.append(SubjectVisualMesh(name, target_body, relative))
        record = {
            "mesh": asdict(meshes[-1]),
            "source": {
                "file": source_path.name,
                "sha256": source_hash,
                "scale": entry.scale.tolist(),
                "transform": entry.transform.tolist(),
                "mass_center": entry.mass_center.tolist(),
                "source_body": entry.source_body,
                "source_proximal_newton": source_proximal.tolist(),
                "target_proximal_newton": target_proximal.tolist(),
                "registration_translation": registration.tolist(),
                "registration_mode": "official_neutral_body_transform" if exact_transforms else "proximal_anchor",
            },
            "output": {
                "file": relative,
                "sha256": _sha256(output_path),
                "vertex_count": int(len(vertices)),
                "triangle_count": int(len(triangles)),
            },
        }
        mesh_records.append(record)
        compiled_geometry.append((output_path, target_body, body_local, triangles, record))
    contact_layout = None
    if exact_transforms:
        foot_geometry = {
            side: [vertices for _, body, vertices, _, _ in compiled_geometry if body == f"foot_{side}"]
            for side in ("left", "right")
        }
        if any(not values for values in foot_geometry.values()):
            raise ValueError("official visual conversion needs talus, calcaneus, and toe geometry for both feet")
        foot_global_min = min(
            float(np.min(vertices[:, 2] + target_transforms[f"foot_{side}"][2, 3]))
            for side, values in foot_geometry.items()
            for vertices in values
        )
        visual_ground_offset = -foot_global_min
        for output_path, _, vertices, triangles, record in compiled_geometry:
            vertices[:, 2] += visual_ground_offset
            _write_obj(output_path, vertices, triangles)
            record["output"]["sha256"] = _sha256(output_path)
        bounds = {
            side: (
                np.min(np.concatenate(values, axis=0), axis=0),
                np.max(np.concatenate(values, axis=0), axis=0),
            )
            for side, values in foot_geometry.items()
        }
        minimum_width = min(float(maximum[1] - minimum[1]) for minimum, maximum in bounds.values())
        radius = min(config.contact_radius, 0.30 * minimum_width)
        if not math.isfinite(radius) or radius <= 0.005:
            raise ValueError("converted foot geometry cannot support sphere contacts")
        centers = {}
        for side, (minimum, maximum) in bounds.items():
            heel_x = float(minimum[0] + radius)
            forefoot_x = float(maximum[0] - radius)
            lateral_y = float(minimum[1] + radius)
            medial_y = float(maximum[1] - radius)
            center_z = float(minimum[2] + radius)
            centers[side] = (
                (heel_x, lateral_y, center_z),
                (heel_x, medial_y, center_z),
                (forefoot_x, lateral_y, center_z),
                (forefoot_x, medial_y, center_z),
            )
        contact_layout = FootContactLayout(radius, centers, visual_ground_offset)

    expected = {"pelvis", "torso", "femur_left", "femur_right", "tibia_left", "tibia_right"}
    if exact_transforms:
        expected.update(("foot_left", "foot_right"))
    found = {mesh.body for mesh in meshes}
    if found != expected:
        raise ValueError(f"scaled display mapping is incomplete: expected {sorted(expected)}, got {sorted(found)}")
    manifest = {
        "schema_version": _SCHEMA,
        "coordinate_system": {
            "frame": "Newton body-local",
            "length_unit": "m",
            "up_axis": "Z",
            "visual_only": True,
        },
        "source_model": {"file": model_path.name, "sha256": _sha256(model_path)},
        "simple_model_config": asdict(config),
        "contact_layout": asdict(contact_layout) if contact_layout is not None else None,
        "meshes": mesh_records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return CompiledVisuals(root, manifest_path, tuple(meshes), contact_layout)


def compile_scaled_vtp_visuals(
    scaled_osim: str | os.PathLike,
    geometry_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    config: SimpleGaitConfig,
    *,
    source_body_transforms: str | os.PathLike | dict[str, np.ndarray] | None = None,
) -> CompiledVisuals:
    """Atomically bake scaled gait2354 VTP meshes into body-local OBJ assets.

    ``config`` must come from the same subject scaling result as ``scaled_osim``
    so proximal joint registration and segment lengths remain consistent.
    When official neutral body transforms are supplied, all talus, calcaneus,
    and toe meshes are baked into the merged Newton foot bodies as well.
    """
    if isinstance(source_body_transforms, (str, os.PathLike)):
        source_body_transforms = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in json.loads(Path(source_body_transforms).read_text()).items()
        }
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
        staged_root = Path(temporary) / "bundle"
        staged = _compile_scaled_vtp_visuals(
            scaled_osim,
            geometry_dir,
            staged_root,
            config,
            source_body_transforms,
        )
        os.rename(staged_root, root)
    return CompiledVisuals(root, root / staged.manifest_path.name, staged.meshes, staged.contact_layout)
