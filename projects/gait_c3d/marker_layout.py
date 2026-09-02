# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile offline marker placement into a sealed Newton marker layout."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .marker_clusters import TRACKING_CLUSTER_MARKERS
from .native_model import SimpleGaitConfig
from .subject_mjcf import SubjectMarkerSite
from .vtp_adapter import simple_gait_body_transforms

_SCHEMA = "gait_subject_marker_layout_1"
_OPENSIM_TO_NEWTON = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
_SOURCE_TO_TARGET = {
    "pelvis": "pelvis",
    "torso": "torso",
    "femur_l": "femur_left",
    "femur_r": "femur_right",
    "tibia_l": "tibia_left",
    "tibia_r": "tibia_right",
    "calcn_l": "foot_left",
    "calcn_r": "foot_right",
}


@dataclass(frozen=True, slots=True)
class MarkerLayoutEntry:
    """One converted marker with offline source provenance."""

    name: str
    """Motion-capture marker name."""

    source_body: str
    """Source OpenSim body name."""

    body: str
    """Target native body name."""

    position: tuple[float, float, float]
    """Marker position in the target body frame [m]."""

    site_name: str
    """Stable MJCF site name."""

    def as_site(self) -> SubjectMarkerSite:
        """Return the neutral site consumed by the subject MJCF exporter."""
        return SubjectMarkerSite(self.name, self.body, self.position, self.site_name)


@dataclass(frozen=True, slots=True)
class SubjectMarkerLayout:
    """A verified subject marker layout and its source metadata."""

    path: Path
    """Sealed layout JSON path."""

    markers: tuple[MarkerLayoutEntry, ...]
    """Converted markers in source order."""

    source_ground_offset_z: float
    """Shared source-to-target vertical registration [m]."""

    target_body_transforms: dict[str, np.ndarray]
    """Neutral native body transforms used during marker conversion."""

    @property
    def marker_sites(self) -> tuple[SubjectMarkerSite, ...]:
        """Return marker entries as neutral MJCF sites."""
        return tuple(marker.as_site() for marker in self.markers)


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if child.tag.rsplit("}", 1)[-1] == name), None)


def _is_safe_basename(value) -> bool:
    return isinstance(value, str) and bool(value) and value not in {".", ".."} and Path(value).name == value


def _site_name(marker_name: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", marker_name).strip("_")
    if not suffix:
        raise ValueError(f"marker name cannot form an MJCF site: {marker_name!r}")
    return f"marker_{suffix}"


def _validate_transform(name: str, value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"body transform {name!r} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
        raise ValueError(f"body transform {name!r} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-8) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8
    ):
        raise ValueError(f"body transform {name!r} rotation must be proper and orthonormal")
    return transform


def _load_source_transforms(source: str | os.PathLike | dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        source_file = path.name
        source_hash = _sha256(path)
    else:
        values = source
        source_file = None
        source_hash = None
    if not isinstance(values, dict):
        raise ValueError("source body transforms must be a JSON object")
    transforms = {name: _validate_transform(name, value) for name, value in values.items()}
    if source_hash is None:
        canonical_transforms = {name: value.tolist() for name, value in sorted(transforms.items())}
        source_hash = hashlib.sha256(_canonical_json(canonical_transforms)).hexdigest()
    return transforms, {"file": source_file, "sha256": source_hash}


def _parse_source_markers(path: Path) -> list[tuple[str, str, np.ndarray]]:
    markers = []
    names: set[str] = set()
    for element in ET.parse(path).getroot().iter():
        if element.tag.rsplit("}", 1)[-1] != "Marker":
            continue
        name = (element.get("name") or "").strip()
        if not name or name in names:
            raise ValueError(f"empty or duplicate source marker name: {name!r}")
        frame = _child(element, "socket_parent_frame")
        if frame is None:
            frame = _child(element, "body")
        location = _child(element, "location")
        source_body = ((frame.text if frame is not None else "") or "").strip().rsplit("/", 1)[-1]
        values = np.asarray([float(value) for value in ((location.text if location is not None else "") or "").split()])
        if source_body not in _SOURCE_TO_TARGET:
            raise ValueError(f"marker {name!r} references unsupported source body {source_body!r}")
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"marker {name!r} location must contain three finite values")
        names.add(name)
        markers.append((name, source_body, values))
    if not markers:
        raise ValueError(f"source marker set contains no markers: {path}")
    return markers


def _collapse_tracking_clusters(
    markers: list[tuple[str, str, np.ndarray]],
) -> list[tuple[str, str, np.ndarray]]:
    """Replace complete thigh and shank marker groups with centroids."""
    by_name = {name: (source_body, position) for name, source_body, position in markers}
    cluster_members = {member for members in TRACKING_CLUSTER_MARKERS.values() for member in members}
    for centroid, members in TRACKING_CLUSTER_MARKERS.items():
        present = [member for member in members if member in by_name]
        if present and len(present) != len(members):
            missing = [member for member in members if member not in by_name]
            raise ValueError(f"tracking cluster {centroid!r} is incomplete; missing {missing}")
        if centroid in by_name and present:
            raise ValueError(f"tracking cluster {centroid!r} has both source and centroid markers")
        if present:
            bodies = {by_name[member][0] for member in members}
            if len(bodies) != 1:
                raise ValueError(f"tracking cluster {centroid!r} spans multiple source bodies")

    collapsed = []
    emitted: set[str] = set()
    for name, source_body, source_local in markers:
        if name in cluster_members:
            cluster = next(
                (centroid for centroid, members in TRACKING_CLUSTER_MARKERS.items() if name in members),
                None,
            )
            if cluster is None or cluster in emitted:
                continue
            members = TRACKING_CLUSTER_MARKERS[cluster]
            cluster_source_body = by_name[members[0]][0]
            cluster_source_local = np.mean([by_name[member][1] for member in members], axis=0, dtype=np.float64)
            collapsed.append((cluster, cluster_source_body, cluster_source_local))
            emitted.add(cluster)
        else:
            collapsed.append((name, source_body, source_local))
    return collapsed


def compile_subject_marker_layout(
    marker_set_path: str | os.PathLike,
    source_body_transforms: str | os.PathLike | dict[str, np.ndarray],
    config: SimpleGaitConfig,
    output_path: str | os.PathLike,
    *,
    source_ground_offset_z: float,
) -> SubjectMarkerLayout:
    """Convert placed source markers into target-body coordinates and seal them.

    Args:
        marker_set_path: Offline placed OpenSim marker-set XML.
        source_body_transforms: Source body transforms in OpenSim ground.
        config: Final native subject configuration, including root height.
        output_path: Destination marker-layout JSON.
        source_ground_offset_z: Vertical registration applied to all source data [m].

    Returns:
        The verified neutral marker layout.
    """
    if not math.isfinite(source_ground_offset_z):
        raise ValueError("source_ground_offset_z must be finite")
    marker_path = Path(marker_set_path).resolve()
    source_markers = _parse_source_markers(marker_path)
    source_transforms, transforms_provenance = _load_source_transforms(source_body_transforms)
    target_transforms = {
        name: _validate_transform(name, value) for name, value in simple_gait_body_transforms(config).items()
    }

    source_markers = _collapse_tracking_clusters(source_markers)
    entries = []
    site_names: set[str] = set()
    for name, source_body, source_local in source_markers:
        source_transform = source_transforms.get(source_body)
        if source_transform is None:
            raise ValueError(f"marker {name!r} is missing source transform {source_body!r}")
        target_body = _SOURCE_TO_TARGET[source_body]
        target_transform = target_transforms[target_body]
        ground_opensim = source_local @ source_transform[:3, :3].T + source_transform[:3, 3]
        ground_newton = ground_opensim @ _OPENSIM_TO_NEWTON.T
        ground_newton[2] += source_ground_offset_z
        target_local = (ground_newton - target_transform[:3, 3]) @ target_transform[:3, :3]
        site_name = _site_name(name)
        if site_name in site_names:
            raise ValueError(f"marker site name collision: {site_name!r}")
        site_names.add(site_name)
        entries.append(
            MarkerLayoutEntry(
                name=name,
                source_body=source_body,
                body=target_body,
                position=tuple(float(value) for value in target_local),
                site_name=site_name,
            )
        )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": _SCHEMA,
        "coordinate_system": {
            "frame": "Newton target body-local",
            "length_unit": "m",
            "up_axis": "Z",
            "forward_axis": "X",
            "left_axis": "Y",
        },
        "source": {
            "marker_set_file": marker_path.name,
            "marker_set_sha256": _sha256(marker_path),
            "body_transforms_file": transforms_provenance["file"],
            "body_transforms_sha256": transforms_provenance["sha256"],
        },
        "target": {
            "neutral_body_transforms": {
                name: transform.tolist() for name, transform in sorted(target_transforms.items())
            },
        },
        "conversion": {
            "opensim_to_newton": _OPENSIM_TO_NEWTON.tolist(),
            "source_ground_offset_z_m": float(source_ground_offset_z),
            "position_convention": "row_vectors",
            "tracking_cluster_centroids": {
                centroid: list(members) for centroid, members in TRACKING_CLUSTER_MARKERS.items()
            },
        },
        "markers": [
            {
                "name": entry.name,
                "source_body": entry.source_body,
                "body": entry.body,
                "position_m": list(entry.position),
                "site_name": entry.site_name,
            }
            for entry in entries
        ],
    }
    manifest["seal"] = {
        "algorithm": "sha256",
        "content_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(staged, output)
    return load_subject_marker_layout(output)


def scale_subject_marker_layout_from_base(
    base_layout_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    length_scale: float,
    hip_width: float | None = None,
) -> SubjectMarkerLayout:
    """Scale a sealed marker layout from the canonical base placement.

    Marker positions, neutral body-frame translations, and the shared vertical
    registration are scaled together. Rotations and source provenance remain
    unchanged, and the output records the base-layout hash for auditability.

    Args:
        base_layout_path: Sealed marker layout used as the reference placement.
        output_path: Destination sealed marker-layout JSON.
        length_scale: Uniform physical scale relative to the base [1].
        hip_width: Optional target hip-joint center spacing [m]. When supplied,
            leg body-frame translations are updated to this spacing.

    Returns:
        The verified scaled marker layout.
    """
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    if hip_width is not None and (not math.isfinite(hip_width) or hip_width <= 0.0):
        raise ValueError("hip_width must be finite and positive")
    base_path = Path(base_layout_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if base_path == output:
        raise ValueError("scaled marker layout output must not overwrite its base layout")
    if output.exists():
        raise FileExistsError(output)
    load_subject_marker_layout(base_path)
    manifest = json.loads(base_path.read_text(encoding="utf-8"))
    manifest.pop("seal", None)
    target = manifest["target"]["neutral_body_transforms"]
    for name, transform in target.items():
        for row in transform[:3]:
            row[3] = float(row[3] * length_scale)
        if hip_width is not None and name in {"femur_left", "tibia_left", "foot_left"}:
            transform[1][3] = float(0.5 * hip_width)
        elif hip_width is not None and name in {"femur_right", "tibia_right", "foot_right"}:
            transform[1][3] = float(-0.5 * hip_width)
    conversion = manifest["conversion"]
    conversion["source_ground_offset_z_m"] = float(conversion["source_ground_offset_z_m"] * length_scale)
    for marker in manifest["markers"]:
        marker["position_m"] = [float(value * length_scale) for value in marker["position_m"]]
    manifest["derived_from"] = {
        "layout_file": base_path.name,
        "layout_sha256": _sha256(base_path),
        "length_scale": float(length_scale),
        "hip_width_m": float(hip_width) if hip_width is not None else None,
    }
    manifest["seal"] = {
        "algorithm": "sha256",
        "content_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(staged, output)
    return load_subject_marker_layout(output)


def load_subject_marker_layout(path: str | os.PathLike) -> SubjectMarkerLayout:
    """Verify and load a sealed neutral subject marker layout."""
    layout_path = Path(path).resolve()
    manifest = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("subject marker layout must be a JSON object")
    seal = manifest.pop("seal", None)
    expected = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    if seal != {"algorithm": "sha256", "content_sha256": expected}:
        raise ValueError("subject marker layout seal mismatch")
    if manifest.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported subject marker layout schema")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("subject marker layout source provenance is missing")
    marker_file = source.get("marker_set_file")
    marker_hash = source.get("marker_set_sha256")
    transforms_file = source.get("body_transforms_file")
    transforms_hash = source.get("body_transforms_sha256")
    if (
        not _is_safe_basename(marker_file)
        or not isinstance(marker_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", marker_hash) is None
        or (transforms_file is not None and not _is_safe_basename(transforms_file))
        or not isinstance(transforms_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", transforms_hash) is None
    ):
        raise ValueError("subject marker layout source provenance is invalid")
    target = manifest.get("target")
    raw_target_transforms = target.get("neutral_body_transforms") if isinstance(target, dict) else None
    expected_target_bodies = set(_SOURCE_TO_TARGET.values())
    if not isinstance(raw_target_transforms, dict) or set(raw_target_transforms) != expected_target_bodies:
        raise ValueError("subject marker layout target transforms are missing or incomplete")
    target_transforms = {name: _validate_transform(name, value) for name, value in raw_target_transforms.items()}
    expected_frame = {
        "frame": "Newton target body-local",
        "length_unit": "m",
        "up_axis": "Z",
        "forward_axis": "X",
        "left_axis": "Y",
    }
    if manifest.get("coordinate_system") != expected_frame:
        raise ValueError("subject marker layout coordinate system is invalid")
    conversion = manifest.get("conversion", {})
    if (
        conversion.get("opensim_to_newton") != _OPENSIM_TO_NEWTON.tolist()
        or conversion.get("position_convention") != "row_vectors"
        or (
            conversion.get("tracking_cluster_centroids") is not None
            and conversion.get("tracking_cluster_centroids")
            != {centroid: list(members) for centroid, members in TRACKING_CLUSTER_MARKERS.items()}
        )
    ):
        raise ValueError("subject marker layout conversion metadata is invalid")
    offset = conversion.get("source_ground_offset_z_m")
    if not isinstance(offset, (int, float)) or not math.isfinite(offset):
        raise ValueError("subject marker layout ground offset is invalid")

    markers = []
    names: set[str] = set()
    sites: set[str] = set()
    for item in manifest.get("markers", []):
        name = item.get("name")
        source_body = item.get("source_body")
        body = item.get("body")
        site_name = item.get("site_name")
        position = item.get("position_m")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"invalid or duplicate marker name in layout: {name!r}")
        if source_body not in _SOURCE_TO_TARGET or _SOURCE_TO_TARGET[source_body] != body:
            raise ValueError(f"marker {name!r} has an invalid body mapping")
        if not isinstance(site_name, str) or site_name != _site_name(name) or site_name in sites:
            raise ValueError(f"marker {name!r} has an invalid site name")
        try:
            position_array = np.asarray(position, dtype=np.float64)
        except (TypeError, ValueError):
            position_array = np.empty(0)
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            raise ValueError(f"marker {name!r} has an invalid target position")
        markers.append(
            MarkerLayoutEntry(name, source_body, body, tuple(float(value) for value in position_array), site_name)
        )
        names.add(name)
        sites.add(site_name)
    if not markers:
        raise ValueError("subject marker layout contains no markers")
    return SubjectMarkerLayout(layout_path, tuple(markers), float(offset), target_transforms)
