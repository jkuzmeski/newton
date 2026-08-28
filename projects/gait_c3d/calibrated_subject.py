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
            values = np.asarray([float(value) for value in (joint.get("pos") or "").split()], dtype=np.float64)
            if values.shape == (3,) and np.all(np.isfinite(values)):
                return values
    return np.zeros(3, dtype=np.float64)


def _segment_basis(calibration: SegmentCalibration, name: str) -> np.ndarray:
    """Return a segment basis with columns forward, left, up/longitudinal."""
    segment = calibration.segments[name]
    key = "basis_forward_left_up" if name.startswith("foot_") else "basis_forward_left_longitudinal"
    return np.asarray(segment[key], dtype=np.float64)


def _base_torso_distal_local(base_root: Path) -> np.ndarray:
    """Return the S001 sacrum marker in the base torso body frame."""
    layout = json.loads((base_root / "model" / "marker_layout.json").read_text(encoding="utf-8"))
    marker = next(entry for entry in layout["markers"] if entry["name"] == "V.Sacral")
    pelvis = next(
        body
        for body in ET.parse(base_root / "model" / "subject.xml").getroot().iter("body")
        if body.get("name") == "pelvis"
    )
    torso = next(
        body
        for body in ET.parse(base_root / "model" / "subject.xml").getroot().iter("body")
        if body.get("name") == "torso"
    )
    pelvis_position = np.asarray([float(value) for value in pelvis.get("pos", "0 0 0").split()], dtype=np.float64)
    torso_position = np.asarray([float(value) for value in torso.get("pos", "0 0 0").split()], dtype=np.float64)
    marker_position = np.asarray(marker["position_m"], dtype=np.float64)
    return pelvis_position + marker_position - (pelvis_position + torso_position)


def _base_pelvis_marker_positions(base_root: Path) -> dict[str, np.ndarray]:
    """Load base ASIS/sacrum positions in the base pelvis body frame."""
    layout = json.loads((base_root / "model" / "marker_layout.json").read_text(encoding="utf-8"))
    return {
        entry["name"]: np.asarray(entry["position_m"], dtype=np.float64)
        for entry in layout["markers"]
        if entry["name"] in {"L.ASIS", "R.ASIS", "V.Sacral"}
    }


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
    if isinstance(calibration, SegmentCalibration):
        target_calibration = calibration
    else:
        target_calibration = load_static_segment_calibration(calibration)
    if base_calibration is None:
        base_calibration = base_calibration_path
    if isinstance(base_calibration, SegmentCalibration):
        reference_calibration = base_calibration
    else:
        reference_calibration = load_static_segment_calibration(base_calibration)
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
    target_markers = target_calibration.marker_positions
    target_pelvis_origin = np.asarray(target_calibration.pelvis["origin_m"], dtype=np.float64)
    target_pelvis_rotation = _pelvis_basis(target_calibration)
    reference_pelvis = reference_calibration.pelvis
    reference_asis = float(reference_pelvis["asis_distance_m"])
    target_asis = float(target_calibration.pelvis["asis_distance_m"])
    if reference_asis <= 0.0 or target_asis <= 0.0:
        raise ValueError("ASIS distance must be positive")
    scale = {
        "pelvis": np.full(3, target_asis / reference_asis),
    }
    if "torso" in target_calibration.segments and "torso" in reference_calibration.segments:
        target_torso = target_calibration.segments["torso"]
        reference_torso = reference_calibration.segments["torso"]
        scale["torso"] = np.asarray(
            (
                float(target_torso["depth_m"]) / float(reference_torso["depth_m"]),
                float(target_torso["width_m"]) / float(reference_torso["width_m"]),
                float(target_torso["length_m"]) / float(reference_torso["length_m"]),
            ),
            dtype=np.float64,
        )
    for side in ("left", "right"):
        for segment_name in (f"thigh_{side}", f"shank_{side}", f"foot_{side}"):
            target_segment = target_calibration.segments[segment_name]
            reference_segment = reference_calibration.segments[segment_name]
            target_length = float(target_segment["length_m"])
            reference_length = float(reference_segment["length_m"])
            target_width = float(target_segment["width_m"])
            reference_width = float(reference_segment["width_m"])
            if min(target_length, reference_length, target_width, reference_width) <= 0.0:
                raise ValueError(f"invalid dimensions for {segment_name}")
            scale[segment_name] = np.asarray(
                (
                    target_length / reference_length
                    if segment_name.startswith("foot_")
                    else target_width / reference_width,
                    target_width / reference_width,
                    target_width / reference_width
                    if segment_name.startswith("foot_")
                    else target_length / reference_length,
                ),
                dtype=np.float64,
            )
    root = ET.parse(base_xml).getroot()
    if not model_name or any(character.isspace() for character in model_name):
        raise ValueError("model_name must be nonempty and contain no whitespace")
    root.set("model", model_name)
    bodies = {body.get("name", ""): body for body in root.iter("body") if body.get("name")}
    if "torso" in bodies:
        _add_torso_joints(root, bodies["torso"])
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

    body_world = {
        "pelvis": (target_pelvis_origin, target_pelvis_rotation),
    }
    for side, _prefix in (("left", "L"), ("right", "R")):
        body_world[f"femur_{side}"] = (
            np.asarray(target_calibration.pelvis["hip_centers_m"][side], dtype=np.float64),
            _segment_basis(target_calibration, f"thigh_{side}"),
        )
        body_world[f"tibia_{side}"] = (
            np.asarray(target_calibration.segments[f"shank_{side}"]["proximal_m"], dtype=np.float64),
            _segment_basis(target_calibration, f"shank_{side}"),
        )
        body_world[f"foot_{side}"] = (
            np.asarray(target_calibration.segments[f"foot_{side}"]["proximal_m"], dtype=np.float64),
            _segment_basis(target_calibration, f"foot_{side}"),
        )
    torso_offset = np.asarray((0.0, 0.0, 0.38004881829313497), dtype=np.float64)
    if "torso" in bodies:
        if "torso" in target_calibration.segments:
            torso_record = target_calibration.segments["torso"]
            torso_origin = np.asarray(torso_record["distal_m"], dtype=np.float64)
            torso_rotation = np.asarray(torso_record["basis_forward_left_longitudinal"], dtype=np.float64)
        else:
            torso_rotation = target_pelvis_rotation
            torso_origin = target_pelvis_origin + target_pelvis_rotation @ torso_offset
        body_world["torso"] = (torso_origin, torso_rotation)
    # Transform mesh vertices into calibrated body frames before computing the common floor.
    mesh_file = {mesh.get("name"): mesh.get("file") for mesh in root.iter("mesh") if mesh.get("name")}
    body_mesh_vertices: dict[str, list[tuple[ET.Element, Path, np.ndarray]]] = {}
    body_mesh_min_z: dict[str, float] = {}
    for body_name, body in bodies.items():
        if body_name not in body_world:
            continue
        if body_name == "pelvis":
            segment_key = "pelvis"
            proximal = np.zeros(3)
        elif body_name == "torso":
            segment_key = "torso" if "torso" in scale else "pelvis"
            proximal = _base_torso_distal_local(base_root) if segment_key == "torso" else np.zeros(3)
        elif body_name.startswith("femur_"):
            segment_key = f"thigh_{body_name.removeprefix('femur_')}"
            proximal = _source_joint_position(body, "hip_flexion")
        elif body_name.startswith("tibia_"):
            segment_key = f"shank_{body_name.removeprefix('tibia_')}"
            proximal = _source_joint_position(body, "knee")
        elif body_name.startswith("foot_"):
            segment_key = f"foot_{body_name.removeprefix('foot_')}"
            proximal = _source_joint_position(body, "ankle")
        else:
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
            vertices = _read_obj(source)
            geom_pos = np.asarray([float(value) for value in (geom.get("pos") or "0 0 0").split()], dtype=np.float64)
            if geom_pos.shape != (3,):
                raise ValueError(f"mesh geometry position is invalid: {mesh_name}")
            if body_name == "pelvis":
                pelvis_source = (vertices + geom_pos - base_pelvis_marker_origin) * scale[segment_key]
                transformed = (pelvis_source @ pelvis_map_rotation + pelvis_map_translation) @ target_pelvis_rotation
            else:
                transformed = (vertices + geom_pos - proximal) * scale[segment_key]
            records.append((geom, source, transformed))
            if body_name.startswith("foot_"):
                body_mesh_min_z[body_name] = min(
                    body_mesh_min_z.get(body_name, math.inf), float(np.min(transformed[:, 2]))
                )
        body_mesh_vertices[body_name] = records
    torso_mesh_min_z = 0.0
    torso_mesh_z_scale = 1.0
    if "torso" in scale and body_mesh_vertices.get("torso"):
        torso_values = np.concatenate([vertices for _, _, vertices in body_mesh_vertices["torso"]], axis=0)
        torso_mesh_min_z = float(np.min(torso_values[:, 2]))
        torso_mesh_max_z = float(np.max(torso_values[:, 2]))
        if not math.isfinite(torso_mesh_max_z) or torso_mesh_max_z <= torso_mesh_min_z:
            raise ValueError("base torso mesh has no finite longitudinal extent")
        torso_mesh_z_scale = float(target_calibration.segments["torso"]["length_m"]) / (
            torso_mesh_max_z - torso_mesh_min_z
        )
        for _, _, vertices in body_mesh_vertices["torso"]:
            vertices[:, 2] = (vertices[:, 2] - torso_mesh_min_z) * torso_mesh_z_scale
    foot_origins = {side: body_world[f"foot_{side}"][0][2] for side in ("left", "right")}
    foot_world_min = min(foot_origins[side] + body_mesh_min_z[f"foot_{side}"] for side in ("left", "right"))
    foot_offsets = {
        side: foot_world_min - foot_origins[side] - body_mesh_min_z[f"foot_{side}"] for side in ("left", "right")
    }
    global_offset = np.asarray((0.0, 0.0, -foot_world_min), dtype=np.float64)
    target_world = {name: (origin + global_offset, rotation) for name, (origin, rotation) in body_world.items()}
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        staged_xml = staged / output.name
        staged_geometry = staged / "Geometry"
        output_meshes = []
        for body_name, records in body_mesh_vertices.items():
            for geom, source, vertices in records:
                offset = np.zeros(3)
                if body_name.startswith("foot_"):
                    offset[2] = foot_offsets[body_name.removeprefix("foot_")]
                destination_relative = Path(mesh_file[geom.get("mesh")])
                destination = staged / destination_relative
                _write_obj(source, destination, vertices, offset)
                output_meshes.append((destination, geom))
                geom.set("pos", "0 0 0")
        # Author parent-relative body frames, inertials, primitives, joints, sites, and contacts.
        parent_by_body = {
            "pelvis": None,
            "torso": "pelvis",
            "femur_left": "pelvis",
            "tibia_left": "femur_left",
            "foot_left": "tibia_left",
            "femur_right": "pelvis",
            "tibia_right": "femur_right",
            "foot_right": "tibia_right",
        }
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
            if body_name.startswith("femur_"):
                proximal = _source_joint_position(body, "hip_flexion")
                for joint in body.findall("joint"):
                    joint.set("pos", "0 0 0")
            elif body_name.startswith("tibia_"):
                proximal = _source_joint_position(body, "knee")
                for joint in body.findall("joint"):
                    joint.set("pos", "0 0 0")
            elif body_name.startswith("foot_"):
                proximal = _source_joint_position(body, "ankle")
                for joint in body.findall("joint"):
                    joint.set("pos", "0 0 0")
            elif body_name == "torso" and "torso" in scale:
                proximal = _base_torso_distal_local(base_root)
            else:
                proximal = np.zeros(3)
            segment_key = (
                "torso"
                if body_name == "torso" and "torso" in scale
                else "pelvis"
                if body_name == "pelvis" or body_name == "torso"
                else (
                    f"thigh_{body_name.removeprefix('femur_')}"
                    if body_name.startswith("femur_")
                    else f"shank_{body_name.removeprefix('tibia_')}"
                    if body_name.startswith("tibia_")
                    else f"foot_{body_name.removeprefix('foot_')}"
                    if body_name.startswith("foot_")
                    else "pelvis"
                )
            )
            body_scale = scale[segment_key]
            for inertial in body.findall("inertial"):
                position = np.asarray(
                    [float(value) for value in (inertial.get("pos") or "0 0 0").split()], dtype=np.float64
                )
                if position.shape != (3,):
                    raise ValueError(f"invalid inertial position on {body_name}")
                if body_name == "pelvis":
                    pelvis_source = (position - base_pelvis_marker_origin) * body_scale
                    inertial_position = (
                        pelvis_source @ pelvis_map_rotation + pelvis_map_translation
                    ) @ target_pelvis_rotation
                else:
                    inertial_position = (position - proximal) * body_scale
                    if body_name == "torso" and segment_key == "torso":
                        inertial_position[2] = (inertial_position[2] - torso_mesh_min_z) * torso_mesh_z_scale
                inertial.set("pos", _fmt(inertial_position))
                inertial.set("mass", f"{float(inertial.get('mass', 'nan')) * mass_scale:.9g}")
                inertia_scale = mass_scale * float(np.mean(body_scale**2))
                values = np.asarray(
                    [float(value) for value in (inertial.get("fullinertia") or "").split()], dtype=np.float64
                )
                if values.shape != (6,):
                    raise ValueError(f"invalid inertial tensor on {body_name}")
                inertial.set("fullinertia", _fmt(values * inertia_scale))
            for geom in body.findall("geom"):
                name = geom.get("name", "")
                if name.startswith("collision_femur_") or name.startswith("collision_tibia_"):
                    length = float(target_calibration.segments[segment_key]["length_m"])
                    radius = 0.5 * float(target_calibration.segments[segment_key]["width_m"])
                    clearance = min(0.035 * float(np.mean(body_scale)), 0.25 * length)
                    geom.set("size", f"{max(radius, 0.005):.9g}")
                    geom.set("fromto", _fmt(np.asarray((0.0, 0.0, -clearance, 0.0, 0.0, -(length - clearance)))))
                elif body_name.startswith("foot_") and name.startswith(("contact_", "visual_foot_")):
                    pass
                elif geom.get("mesh") is None:
                    for attribute in ("pos", "size"):
                        if attribute in geom.attrib:
                            values = np.asarray(
                                [float(value) for value in geom.get(attribute).split()], dtype=np.float64
                            )
                            if body_name == "pelvis" and attribute == "pos":
                                pelvis_source = (values - base_pelvis_marker_origin) * body_scale
                                values = (
                                    pelvis_source @ pelvis_map_rotation + pelvis_map_translation
                                ) @ target_pelvis_rotation
                            elif body_name == "torso" and attribute == "pos" and segment_key == "torso":
                                values = (values - proximal) * body_scale
                                values[2] = (values[2] - torso_mesh_min_z) * torso_mesh_z_scale
                            else:
                                values = values * body_scale[: values.size]
                            geom.set(attribute, _fmt(values))
            if body_name.startswith("foot_"):
                side = body_name.removeprefix("foot_")
                records = [
                    vertices + np.asarray((0.0, 0.0, foot_offsets[side]))
                    for _, _, vertices in body_mesh_vertices[body_name]
                ]
                all_vertices = np.concatenate(records, axis=0)
                minimum, maximum = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)
                radius = min(
                    0.5 * float(target_calibration.segments[f"foot_{side}"]["width_m"]),
                    0.25 * float(maximum[0] - minimum[0]),
                    0.25 * float(maximum[1] - minimum[1]),
                )
                radius = max(0.005, min(radius, 0.5 * float(reference_calibration.segments[f"foot_{side}"]["width_m"])))
                centers = (
                    (minimum[0] + radius, minimum[1] + radius, minimum[2] + radius),
                    (minimum[0] + radius, maximum[1] - radius, minimum[2] + radius),
                    (maximum[0] - radius, minimum[1] + radius, minimum[2] + radius),
                    (maximum[0] - radius, maximum[1] - radius, minimum[2] + radius),
                )
                for geom in body.findall("geom"):
                    name = geom.get("name", "")
                    if name.startswith("contact_") or name.startswith("visual_foot_"):
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
        # The ground plane remains at z=0; the foot body frames and meshes now share that plane.
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
                for path, _ in output_meshes
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
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return CalibratedSubjectMJCF(output, target_calibration, base_root, mass_scale)
