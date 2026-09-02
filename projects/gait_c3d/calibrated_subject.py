# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a personalized native subject from a static segment calibration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .marker_clusters import TRACKING_CLUSTER_C3D_SOURCES
from .segment_calibration import SegmentCalibration, load_static_segment_calibration

_SCHEMA = "gait_segment_calibration_to_mjcf_1"
_CANONICAL_TO_SOURCE = {
    "L.ASIS": "LASI",
    "R.ASIS": "RASI",
    "V.Sacral": "VSAC",
    "Top.Head": "TOPHEAD",
    "Sternum": "STRN",
    "L.Acromium": "LSHO",
    "R.Acromium": "RSHO",
    "L.Thigh.Upper": "LTH2",
    "L.Thigh.Front": "LTH3",
    "L.Thigh.Rear": "LTH4",
    "L.Knee.Lat": "LKNE",
    "L.Knee.Med": "LMKNE",
    "R.Thigh.Upper": "RTH2",
    "R.Thigh.Front": "RTH3",
    "R.Thigh.Rear": "RTH4",
    "R.Knee.Lat": "RKNE",
    "R.Knee.Med": "RMKNE",
    "L.Shank.Upper": "LTIB2",
    "L.Shank.Front": "LTIB3",
    "L.Shank.Rear": "LTIB4",
    "L.Ankle.Lat": "LANK",
    "L.Ankle.Med": "LMANK",
    "R.Shank.Upper": "RTIB2",
    "R.Shank.Front": "RTIB3",
    "R.Shank.Rear": "RTIB4",
    "R.Ankle.Lat": "RANK",
    "R.Ankle.Med": "RMANK",
    "L.Heel": "LHEE",
    "L.Toe.Lat": "LMTH5",
    "L.Toe.Med": "LMTH1",
    "L.Toe.Tip": "LHLX",
    "R.Heel": "RHEE",
    "R.Toe.Lat": "RMTH5",
    "R.Toe.Med": "RMTH1",
    "R.Toe.Tip": "RHLX",
    **{name: name for name in TRACKING_CLUSTER_C3D_SOURCES},
}


def _canonical_json(value: dict) -> bytes:
    """Serialize manifest content deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fmt(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    """Format finite MJCF numeric values."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("MJCF values must be finite")
    return " ".join(f"{float(value):.9g}" for value in values)


def _vector(element: ET.Element, name: str, default: str = "") -> np.ndarray:
    """Parse one numeric XML attribute."""
    return np.asarray([float(value) for value in element.get(name, default).split()], dtype=np.float64)


def _quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to an MJCF wxyz quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8)
    ):
        raise ValueError("body rotation must be proper and orthonormal")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]) * 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]) * 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert an MJCF wxyz quaternion to a proper rotation matrix."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise ValueError("MJCF quaternion must define a proper rotation")
    return rotation


def _element_rotation(element: ET.Element) -> np.ndarray:
    """Return an MJCF element rotation, defaulting to identity."""
    values = _vector(element, "quat", "1 0 0 0")
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid quaternion on {element.get('name', element.tag)!r}")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError(f"zero quaternion on {element.get('name', element.tag)!r}")
    return _rotation_wxyz(values / norm)


def _source_segment_length(body: ET.Element, child: ET.Element, proximal_prefix: str, distal_prefix: str) -> float:
    """Measure one source articulation span between its actual joint centers."""
    proximal = _source_joint_position(body, proximal_prefix)
    child_position = _vector(child, "pos", "0 0 0")
    distal = child_position + _element_rotation(child) @ _source_joint_position(child, distal_prefix)
    length = float(np.linalg.norm(distal - proximal))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"source segment {body.get('name')!r} has no positive joint span")
    return length


def _row_direction_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper row-vector rotation that maps one direction to another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1.0e-12:
        if cosine > 0.0:
            return np.eye(3)
        axis = np.asarray((1.0, 0.0, 0.0))
        if abs(float(source[0])) > 0.9:
            axis = np.asarray((0.0, 1.0, 0.0))
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        column_rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
        return column_rotation.T
    axis = cross / sine
    skew = np.asarray(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)))
    column_rotation = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
    return column_rotation.T


def _transform_fullinertia(values: np.ndarray, column_linear: np.ndarray, mass_scale: float) -> np.ndarray:
    """Transform a full inertia tensor under a linear coordinate map."""
    ixx, iyy, izz, ixy, ixz, iyz = np.asarray(values, dtype=np.float64)
    inertia = np.asarray(((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)))
    second_moment = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    transformed_moment = column_linear @ second_moment @ column_linear.T
    transformed = mass_scale * (np.trace(transformed_moment) * np.eye(3) - transformed_moment)
    rows, columns = (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)
    return transformed[rows, columns]


def _transform_box(geom: ET.Element, transform) -> None:
    """Transform an oriented box through a point map and save its body-frame bounds."""
    position = _vector(geom, "pos", "0 0 0")
    size = _vector(geom, "size")
    if position.shape != (3,) or size.shape != (3,):
        raise ValueError(f"invalid box geometry on {geom.get('name')!r}")
    signs = np.asarray([(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)], dtype=np.float64)
    corners = signs * size @ _element_rotation(geom).T + position
    transformed = transform(corners)
    centered = transformed - transformed.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    projected = transformed @ axes
    minimum, maximum = np.min(projected, axis=0), np.max(projected, axis=0)
    geom.set("pos", _fmt((0.5 * (minimum + maximum)) @ axes.T))
    geom.set("size", _fmt(0.5 * (maximum - minimum)))
    geom.set("quat", _fmt(_quat_wxyz(axes)))


def _read_obj(path: Path) -> np.ndarray:
    """Read OBJ vertex positions."""
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"OBJ vertex is incomplete: {path}")
            values = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"OBJ vertex is nonfinite: {path}")
            vertices.append(values)
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices)


def _write_obj(source: Path, destination: Path, vertices: np.ndarray, offset: np.ndarray) -> None:
    """Copy an OBJ with transformed vertex positions."""
    output = []
    vertex_index = 0
    for source_line in source.read_text(encoding="utf-8").splitlines():
        fields = source_line.split()
        output_line = source_line
        if fields and fields[0] == "v":
            value = vertices[vertex_index] + offset
            vertex_index += 1
            output_line = "v " + _fmt(value)
        output.append(output_line)
    if vertex_index != len(vertices):
        raise ValueError(f"OBJ vertex count changed while reading: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def _source_joint_position(body: ET.Element, joint_prefix: str) -> np.ndarray:
    """Find a source proximal joint offset in a body."""
    for joint in body.findall("joint"):
        if (joint.get("name") or "").startswith(joint_prefix):
            values = _vector(joint, "pos")
            if values.shape == (3,) and np.all(np.isfinite(values)):
                return values
    return np.zeros(3, dtype=np.float64)


def _segment_basis(calibration: SegmentCalibration, name: str) -> np.ndarray:
    """Return a segment basis with columns forward, left, up/longitudinal."""
    segment = calibration.segments[name]
    key = "basis_forward_left_up" if name.startswith("foot_") else "basis_forward_left_longitudinal"
    return np.asarray(segment[key], dtype=np.float64)


def _base_pelvis_marker_positions(base_root: Path) -> dict[str, np.ndarray]:
    """Load base ASIS/sacrum positions in the base pelvis body frame."""
    layout = json.loads((base_root / "model" / "marker_layout.json").read_text(encoding="utf-8"))
    return {
        entry["name"]: np.asarray(entry["position_m"], dtype=np.float64)
        for entry in layout["markers"]
        if entry["name"] in {"L.ASIS", "R.ASIS", "V.Sacral"}
    }


def _load_calibration(value: SegmentCalibration | str | os.PathLike) -> SegmentCalibration:
    """Load a calibration path, or return an already loaded calibration."""
    return value if isinstance(value, SegmentCalibration) else load_static_segment_calibration(value)


def _pelvis_basis(calibration: SegmentCalibration) -> np.ndarray:
    """Convert CODA right/anterior/up axes to Newton forward/left/up axes."""
    coda = np.asarray(calibration.pelvis["basis_right_anterior_up"], dtype=np.float64)
    return np.column_stack((coda[:, 1], -coda[:, 0], coda[:, 2]))


def _add_torso_joints(root: ET.Element, torso: ET.Element) -> None:
    """Add bounded rotational torso axes at the calibrated distal endpoint."""
    existing = {joint.get("name") for joint in torso.findall("joint")}
    if existing & {"torso_flexion", "torso_lateral", "torso_rotation"}:
        return
    degrees = math.pi / 180.0
    specifications = (
        ("torso_flexion", (0.0, 1.0, 0.0), (-40.0 * degrees, 40.0 * degrees)),
        ("torso_lateral", (1.0, 0.0, 0.0), (-30.0 * degrees, 30.0 * degrees)),
        ("torso_rotation", (0.0, 0.0, 1.0), (-30.0 * degrees, 30.0 * degrees)),
    )
    for name, axis, limits in specifications:
        ET.SubElement(
            torso,
            "joint",
            name=name,
            type="hinge",
            pos="0 0 0",
            axis=_fmt(axis),
            limited="true",
            range=_fmt(limits),
            damping="0.5",
            armature="0.01",
        )
    actuator = next((element for element in root if element.tag.rsplit("}", 1)[-1] == "actuator"), None)
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for name, _, limits in specifications:
        ET.SubElement(
            actuator,
            "position",
            name=f"{name}_position",
            joint=name,
            kp="100",
            ctrllimited="true",
            ctrlrange=_fmt(limits),
        )


def _fit_row_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a proper row-vector rigid map ``target = source @ rotation + translation``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError("rigid frame fit needs matching three-dimensional point sets")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    if not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise ValueError("marker frame fit produced an invalid rotation")
    return rotation, target_center - source_center @ rotation


@dataclass(frozen=True, slots=True)
class CalibratedSubjectMJCF:
    """Saved native MJCF produced from a static marker calibration."""

    path: Path
    """Saved MJCF path."""

    calibration: SegmentCalibration
    """Static calibration consumed by the builder."""

    base_subject: Path
    """S001 base bundle used for mesh and mass templates."""

    mass_scale: float
    """Uniform mass scale relative to the base subject."""


def write_calibrated_subject_mjcf(
    base_subject_dir: str | os.PathLike,
    calibration: SegmentCalibration | str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    body_mass: float,
    model_name: str = "calibrated_gait_subject",
    base_calibration: SegmentCalibration | str | os.PathLike | None = None,
) -> CalibratedSubjectMJCF:
    """Build a personalized MJCF using per-segment static marker geometry.

    Args:
        base_subject_dir: S001 native base bundle with actual OBJ meshes.
        calibration: Target static segment calibration.
        output_path: Destination MJCF path.
        body_mass: Target body mass [kg].
        model_name: MJCF model name.
        base_calibration: S001 calibration used for per-segment scale ratios.

    Returns:
        Metadata for the saved calibrated subject.
    """
    base_root = Path(base_subject_dir).expanduser().resolve()
    base_xml = base_root / "model" / "subject.xml"
    base_model_manifest_path = base_root / "model" / "manifest.json"
    base_bundle_path = base_root / "subject.json"
    base_calibration_path = base_root / "model" / "segment_calibration.json"
    for path in (base_xml, base_model_manifest_path, base_bundle_path):
        if not path.is_file():
            raise FileNotFoundError(f"base subject artifact is missing: {path}")
    target_calibration = _load_calibration(calibration)
    reference_calibration = _load_calibration(
        base_calibration if base_calibration is not None else base_calibration_path
    )
    if not math.isfinite(body_mass) or body_mass <= 0.0:
        raise ValueError("body_mass must be finite and positive")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if (output.parent / "Geometry").exists() or (output.parent / "manifest.json").exists():
        raise FileExistsError(f"calibrated output directory is not empty: {output.parent}")

    bundle_manifest = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    base_mass = float(bundle_manifest["subject"]["mass_kg"])
    if not math.isfinite(base_mass) or base_mass <= 0.0:
        raise ValueError("base subject mass must be finite and positive")
    mass_scale = body_mass / base_mass
    root = ET.parse(base_xml).getroot()
    bodies = {body.get("name", ""): body for body in root.iter("body") if body.get("name")}
    target_markers = target_calibration.marker_positions
    target_pelvis_origin = np.asarray(target_calibration.pelvis["origin_m"], dtype=np.float64)
    target_pelvis_rotation = _pelvis_basis(target_calibration)
    reference_pelvis = reference_calibration.pelvis
    reference_asis = float(reference_pelvis["asis_distance_m"])
    target_asis = float(target_calibration.pelvis["asis_distance_m"])
    if reference_asis <= 0.0 or target_asis <= 0.0:
        raise ValueError("ASIS distance must be positive")
    scale = {"pelvis": np.full(3, target_asis / reference_asis)}
    if "torso" in target_calibration.segments and "torso" in reference_calibration.segments:
        scale["torso"] = np.asarray(
            [
                float(target_calibration.segments["torso"][name]) / float(reference_calibration.segments["torso"][name])
                for name in ("depth_m", "width_m", "length_m")
            ]
        )
    for side in ("left", "right"):
        for segment, body, child, joints in (
            ("thigh", "femur", "tibia", ("hip_flexion", "knee")),
            ("shank", "tibia", "foot", ("knee", "ankle")),
            ("foot", "foot", None, ()),
        ):
            name = f"{segment}_{side}"
            target, reference = target_calibration.segments[name], reference_calibration.segments[name]
            values = tuple(float(record[key]) for record in (target, reference) for key in ("length_m", "width_m"))
            if min(values) <= 0.0:
                raise ValueError(f"invalid dimensions for {name}")
            target_length, target_width, reference_length, reference_width = values
            source_length = (
                reference_length
                if child is None
                else _source_segment_length(bodies[f"{body}_{side}"], bodies[f"{child}_{side}"], *joints)
            )
            length_scale, width_scale = target_length / source_length, target_width / reference_width
            scale[name] = np.asarray(
                (length_scale, width_scale, width_scale)
                if segment == "foot"
                else (width_scale, width_scale, length_scale)
            )
    if not model_name or any(character.isspace() for character in model_name):
        raise ValueError("model_name must be nonempty and contain no whitespace")
    root.set("model", model_name)
    base_pelvis_markers = _base_pelvis_marker_positions(base_root)
    base_pelvis_marker_origin = 0.5 * (base_pelvis_markers["L.ASIS"] + base_pelvis_markers["R.ASIS"])
    base_pelvis_source = (
        np.asarray([base_pelvis_markers[name] - base_pelvis_marker_origin for name in ("L.ASIS", "R.ASIS", "V.Sacral")])
        * scale["pelvis"]
    )
    target_pelvis_source = np.asarray(
        [target_markers[name] - target_pelvis_origin for name in ("LASI", "RASI", "VSAC")]
    )
    pelvis_map_rotation, pelvis_map_translation = _fit_row_rigid(base_pelvis_source, target_pelvis_source)

    body_scaling = {"pelvis": ("pelvis", np.zeros(3))}
    if "torso" in bodies:
        _add_torso_joints(root, bodies["torso"])
        torso_segment = "torso" if "torso" in scale else "pelvis"
        pelvis_position = _vector(bodies["pelvis"], "pos", "0 0 0")
        torso_position = _vector(bodies["torso"], "pos", "0 0 0")
        torso_proximal = pelvis_position + base_pelvis_markers["V.Sacral"] - (pelvis_position + torso_position)
        body_scaling["torso"] = (torso_segment, torso_proximal if torso_segment == "torso" else np.zeros(3))
    for side in ("left", "right"):
        for body, segment, joint in (
            ("femur", "thigh", "hip_flexion"),
            ("tibia", "shank", "knee"),
            ("foot", "foot", "ankle"),
        ):
            name = f"{body}_{side}"
            body_scaling[name] = (f"{segment}_{side}", _source_joint_position(bodies[name], joint))

    body_world = {"pelvis": (target_pelvis_origin, target_pelvis_rotation)}
    for side in ("left", "right"):
        for body, segment in (("femur", "thigh"), ("tibia", "shank"), ("foot", "foot")):
            name = f"{body}_{side}"
            origin = (
                target_calibration.pelvis["hip_centers_m"][side]
                if body == "femur"
                else target_calibration.segments[f"{segment}_{side}"]["proximal_m"]
            )
            body_world[name] = (
                np.asarray(origin, dtype=np.float64),
                _segment_basis(target_calibration, f"{segment}_{side}"),
            )
    if "torso" in bodies:
        if "torso" in target_calibration.segments:
            record = target_calibration.segments["torso"]
            body_world["torso"] = (
                np.asarray(record["distal_m"], dtype=np.float64),
                np.asarray(record["basis_forward_left_longitudinal"], dtype=np.float64),
            )
        else:
            offset = target_pelvis_rotation @ np.asarray((0.0, 0.0, 0.38004881829313497))
            body_world["torso"] = (target_pelvis_origin + offset, target_pelvis_rotation)

    source_hips = np.asarray(
        [
            _vector(bodies[name], "pos", "0 0 0") + _element_rotation(bodies[name]) @ body_scaling[name][1]
            for name in ("femur_left", "femur_right")
        ]
    )
    target_hips = np.asarray(
        [
            (np.asarray(target_calibration.pelvis["hip_centers_m"][side]) - target_pelvis_origin)
            @ target_pelvis_rotation
            for side in ("left", "right")
        ]
    )

    mapped_hips = (
        ((source_hips - base_pelvis_marker_origin) * scale["pelvis"]) @ pelvis_map_rotation + pelvis_map_translation
    ) @ target_pelvis_rotation
    mapped_vector = mapped_hips[0] - mapped_hips[1]
    target_vector = target_hips[0] - target_hips[1]
    hip_rotation = _row_direction_map(mapped_vector, target_vector)
    target_direction = target_vector / np.linalg.norm(target_vector)
    hip_width_scale = float(np.linalg.norm(target_vector) / np.linalg.norm(mapped_vector))
    hip_scale = np.eye(3) + (hip_width_scale - 1.0) * np.outer(target_direction, target_direction)
    pelvis_hip_linear = hip_rotation @ hip_scale
    pelvis_hip_translation = target_hips.mean(axis=0) - mapped_hips.mean(axis=0) @ pelvis_hip_linear
    pelvis_row_linear = np.diag(scale["pelvis"]) @ pelvis_map_rotation @ target_pelvis_rotation @ pelvis_hip_linear

    torso_mesh_min_z = 0.0
    torso_mesh_z_scale = 1.0

    def transform_points(body_name: str, points: np.ndarray) -> np.ndarray:
        """Map source points into a calibrated body frame."""
        if body_name == "pelvis":
            source = (points - base_pelvis_marker_origin) * scale["pelvis"]
            marker_mapped = (source @ pelvis_map_rotation + pelvis_map_translation) @ target_pelvis_rotation
            return marker_mapped @ pelvis_hip_linear + pelvis_hip_translation
        segment, proximal = body_scaling[body_name]
        transformed = (points - proximal) * scale[segment]
        if body_name == "torso" and segment == "torso":
            transformed[:, 2] = (transformed[:, 2] - torso_mesh_min_z) * torso_mesh_z_scale
        return transformed

    # Transform mesh vertices into calibrated body frames before computing the common floor.
    mesh_file = {mesh.get("name"): mesh.get("file") for mesh in root.iter("mesh") if mesh.get("name")}
    body_mesh_vertices: dict[str, list[tuple[ET.Element, Path, np.ndarray]]] = {}
    body_mesh_min_z: dict[str, float] = {}
    for body_name, body in bodies.items():
        if body_name not in body_scaling:
            continue
        records = []
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue
            relative = Path(mesh_file[mesh_name])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe base mesh path: {relative}")
            source = (base_xml.parent / relative).resolve()
            try:
                source.relative_to(base_xml.parent.resolve())
            except ValueError as error:
                raise ValueError(f"base mesh escapes subject bundle: {relative}") from error
            if not source.is_file():
                raise FileNotFoundError(source)
            geom_pos = np.asarray([float(value) for value in (geom.get("pos") or "0 0 0").split()])
            if geom_pos.shape != (3,):
                raise ValueError(f"mesh geometry position is invalid: {mesh_name}")
            transformed = transform_points(body_name, _read_obj(source) + geom_pos)
            records.append((geom, source, transformed))
            if body_name.startswith("foot_"):
                body_mesh_min_z[body_name] = min(
                    body_mesh_min_z.get(body_name, math.inf), float(np.min(transformed[:, 2]))
                )
        body_mesh_vertices[body_name] = records
    if "torso" in scale and body_mesh_vertices.get("torso"):
        torso_values = np.concatenate([vertices for _, _, vertices in body_mesh_vertices["torso"]], axis=0)
        torso_mesh_min_z = float(np.min(torso_values[:, 2]))
        torso_mesh_max_z = float(np.max(torso_values[:, 2]))
        torso_origin, torso_rotation = body_world["torso"]
        target_top_head_local = (target_markers["TOPHEAD"] - torso_origin) @ torso_rotation
        target_top_head_z = float(target_top_head_local[2])
        if (
            not math.isfinite(torso_mesh_max_z)
            or torso_mesh_max_z <= torso_mesh_min_z
            or not math.isfinite(target_top_head_z)
            or target_top_head_z <= 0.0
        ):
            raise ValueError("torso mesh and Top.Head must define a positive longitudinal extent")
        torso_mesh_z_scale = target_top_head_z / (torso_mesh_max_z - torso_mesh_min_z)
        for _, _, vertices in body_mesh_vertices["torso"]:
            vertices[:, 2] = (vertices[:, 2] - torso_mesh_min_z) * torso_mesh_z_scale
    foot_origins = {side: body_world[f"foot_{side}"][0][2] for side in ("left", "right")}
    foot_world_min = min(foot_origins[side] + body_mesh_min_z[f"foot_{side}"] for side in ("left", "right"))
    foot_offsets = {
        side: foot_world_min - foot_origins[side] - body_mesh_min_z[f"foot_{side}"] for side in ("left", "right")
    }
    global_offset = np.asarray((0.0, 0.0, -foot_world_min), dtype=np.float64)
    target_world = {name: (origin + global_offset, rotation) for name, (origin, rotation) in body_world.items()}
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent, ignore_cleanup_errors=True
    ) as temporary:
        staged = Path(temporary)
        staged_xml = staged / output.name
        staged_geometry = staged / "Geometry"
        output_meshes = []
        for body_name, records in body_mesh_vertices.items():
            for geom, source, vertices in records:
                offset = np.zeros(3)
                if body_name.startswith("foot_"):
                    offset[2] = foot_offsets[body_name.removeprefix("foot_")]
                destination = staged / mesh_file[geom.get("mesh")]
                _write_obj(source, destination, vertices, offset)
                output_meshes.append(destination)
                geom.set("pos", "0 0 0")

        # Author parent-relative body frames, inertials, primitives, joints, sites, and contacts.
        parent_by_body = {
            child.get("name"): parent.get("name") for parent in root.iter("body") for child in parent.findall("body")
        }
        parent_by_body["pelvis"] = None
        for body_name, body in bodies.items():
            if body_name not in target_world:
                continue
            origin, rotation = target_world[body_name]
            parent = parent_by_body[body_name]
            if parent is None:
                body.set("pos", _fmt(origin))
                body.set("quat", _fmt(_quat_wxyz(rotation)))
            else:
                parent_origin, parent_rotation = target_world[parent]
                body.set("pos", _fmt(parent_rotation.T @ (origin - parent_origin)))
                body.set("quat", _fmt(_quat_wxyz(parent_rotation.T @ rotation)))

            segment_key, _ = body_scaling[body_name]
            if body_name.startswith(("femur_", "tibia_", "foot_")):
                for joint in body.findall("joint"):
                    joint.set("pos", "0 0 0")
                    if (joint.get("name") or "").startswith("knee_"):
                        joint.set("range", _fmt((-0.5, 2.617993877991494)))
            body_scale = scale[segment_key]
            if body_name == "pelvis":
                column_linear = pelvis_row_linear.T
            else:
                inertia_scale = body_scale.copy()
                if body_name == "torso" and segment_key == "torso":
                    inertia_scale[2] *= torso_mesh_z_scale
                column_linear = np.diag(inertia_scale)
            for inertial in body.findall("inertial"):
                position = np.asarray([float(value) for value in (inertial.get("pos") or "0 0 0").split()])
                if position.shape != (3,):
                    raise ValueError(f"invalid inertial position on {body_name}")
                inertial.set("pos", _fmt(transform_points(body_name, position[None, :])[0]))
                inertial.set("mass", f"{float(inertial.get('mass', 'nan')) * mass_scale:.9g}")
                values = _vector(inertial, "fullinertia")
                if values.shape != (6,):
                    raise ValueError(f"invalid inertial tensor on {body_name}")
                inertial.set("fullinertia", _fmt(_transform_fullinertia(values, column_linear, mass_scale)))
            for geom in body.findall("geom"):
                name = geom.get("name", "")
                if name.startswith(("collision_femur_", "collision_tibia_")):
                    target = target_calibration.segments[segment_key]
                    reference = reference_calibration.segments[segment_key]
                    length = float(target["length_m"])
                    radius = max(0.5 * float(target["width_m"]), 0.005)
                    clearance = min(0.035 * length / float(reference["length_m"]), 0.25 * length)
                    proximal_center, distal_center = clearance + radius, length - clearance - radius
                    if distal_center <= proximal_center:
                        raise ValueError(f"collision capsule is too short for {body_name}")
                    geom.set("size", f"{radius:.9g}")
                    geom.set("fromto", _fmt((0.0, 0.0, -proximal_center, 0.0, 0.0, -distal_center)))
                elif body_name.startswith("foot_") and name.startswith(("contact_", "visual_foot_")):
                    pass
                elif geom.get("mesh") is None and geom.get("type") == "box":
                    _transform_box(geom, lambda points, name=body_name: transform_points(name, points))
                elif geom.get("mesh") is None:
                    if "pos" in geom.attrib:
                        position = _vector(geom, "pos")
                        geom.set("pos", _fmt(transform_points(body_name, position[None, :])[0]))
                    if "size" in geom.attrib:
                        values = _vector(geom, "size")
                        geom.set("size", _fmt(values * body_scale[: values.size]))
            if body_name.startswith("foot_"):
                side = body_name.removeprefix("foot_")
                all_vertices = np.concatenate(
                    [
                        vertices + np.asarray((0.0, 0.0, foot_offsets[side]))
                        for _, _, vertices in body_mesh_vertices[body_name]
                    ],
                    axis=0,
                )
                minimum, maximum = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)
                radius = max(
                    0.005,
                    min(
                        0.5 * float(target_calibration.segments[f"foot_{side}"]["width_m"]),
                        0.25 * float(maximum[0] - minimum[0]),
                        0.25 * float(maximum[1] - minimum[1]),
                        0.5 * float(reference_calibration.segments[f"foot_{side}"]["width_m"]),
                    ),
                )
                centers = [
                    (x, y, minimum[2] + radius)
                    for x in (minimum[0] + radius, maximum[0] - radius)
                    for y in (minimum[1] + radius, maximum[1] - radius)
                ]
                for geom in body.findall("geom"):
                    name = geom.get("name", "")
                    if name.startswith(("contact_", "visual_foot_")):
                        index = int(name.rsplit("_", 1)[-1])
                        geom.set("pos", _fmt(centers[index]))
                        geom.set("size", f"{radius:.9g}")
            for site in body.findall("site"):
                site_name = (site.get("name") or "").removeprefix("marker_")
                source_name = _CANONICAL_TO_SOURCE.get(site_name)
                if source_name is None or source_name not in target_markers:
                    raise ValueError(f"static calibration is missing marker for site {site_name!r}")
                site_world = target_markers[source_name] + global_offset
                site.set("pos", _fmt(rotation.T @ (site_world - origin)))
                site.set("size", f"{target_calibration.marker_radius:.9g}")

        pelvis_origin, pelvis_rotation = target_world["pelvis"]
        neutral = next(key for key in root.iter("key") if key.get("name") == "neutral")
        neutral.set(
            "qpos", _fmt((*pelvis_origin, *_quat_wxyz(pelvis_rotation), *([0.0] * len(list(root.iter("joint"))))))
        )
        staged_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
        output_manifest = {
            "schema_version": _SCHEMA,
            "coordinate_system": {
                "frame": "Newton world/body-local",
                "position_convention": "row_vectors",
                "length_unit": "m",
                "forward_axis": "X",
                "left_axis": "Y",
                "up_axis": "Z",
            },
            "base_marker_set": bundle_manifest.get("base_marker_set", "S001"),
            "source_subject": base_root.name,
            "source_model": {"file": base_xml.name, "sha256": hashlib.sha256(base_xml.read_bytes()).hexdigest()},
            "source_calibration": {
                "file": target_calibration.path.name,
                "sha256": hashlib.sha256(target_calibration.path.read_bytes()).hexdigest(),
            },
            "mass_scale": mass_scale,
            "segment_scales": {name: values.tolist() for name, values in scale.items()},
            "ground": {
                "normal_world": [0.0, 0.0, 1.0],
                "height_m": 0.0,
                "flat_foot": True,
                "global_offset_m": global_offset.tolist(),
            },
            "meshes": [
                {"file": path.relative_to(staged).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in output_meshes
            ],
        }
        output_manifest["seal"] = {
            "algorithm": "sha256",
            "content_sha256": hashlib.sha256(_canonical_json(output_manifest)).hexdigest(),
        }
        (staged / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        shutil.copytree(staged_geometry, output.parent / "Geometry")
        os.replace(staged_xml, output)
        os.replace(staged / "manifest.json", output.parent / "manifest.json")
    return CalibratedSubjectMJCF(output, target_calibration, base_root, mass_scale)
