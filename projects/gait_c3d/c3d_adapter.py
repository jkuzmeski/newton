# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Decode C3D markers directly into sealed NumPy or Warp arrays."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

_SCHEMA = "gait_c3d_marker_artifact_1"
_AXIS = {
    "X": np.asarray((1.0, 0.0, 0.0)),
    "Y": np.asarray((0.0, 1.0, 0.0)),
    "Z": np.asarray((0.0, 0.0, 1.0)),
}
_UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_vector(value: str) -> np.ndarray:
    text = value.strip().upper()
    sign = 1.0
    if text.startswith(("+", "-")):
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    try:
        return sign * _AXIS[text]
    except KeyError as error:
        raise ValueError(f"axis must be X, Y, or Z with an optional sign, got {value!r}") from error


def lab_to_newton_rotation(up_axis: str, forward_axis: str) -> np.ndarray:
    """Return a lab-to-Newton rotation for X-forward, Y-left, Z-up Newton axes."""
    up = _axis_vector(up_axis)
    up = up / np.linalg.norm(up)
    forward = _axis_vector(forward_axis)
    forward = forward - np.dot(forward, up) * up
    norm = np.linalg.norm(forward)
    if norm < 1.0e-9:
        raise ValueError("forward_axis must not be parallel to up_axis")
    forward = forward / norm
    left = np.cross(up, forward)
    rotation = np.stack((forward, left, up))
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-12
    ):
        raise ValueError("lab axes do not define a proper orthonormal frame")
    return rotation


@dataclass(frozen=True, slots=True)
class WarpMarkerTrajectory:
    """Device arrays for one decoded C3D marker trajectory."""

    times: wp.array[float]
    """Frame times [s], shape ``[frame_count]``."""

    positions: wp.array2d[wp.vec3]
    """Newton-frame marker positions [m], shape ``[frame_count, marker_count]``."""

    valid: wp.array2d[wp.uint8]
    """Marker visibility mask, shape ``[frame_count, marker_count]``."""

    marker_names: tuple[str, ...]
    """Marker labels in array-column order."""


@dataclass(frozen=True, slots=True)
class C3DMarkerTrajectory:
    """Finite host arrays decoded directly from a C3D point stream."""

    times: np.ndarray
    """Frame times [s], shape ``[frame_count]``."""

    positions: np.ndarray
    """Newton-frame marker positions [m], shape ``[frame_count, marker_count, 3]``."""

    valid: np.ndarray
    """Marker visibility mask, shape ``[frame_count, marker_count]``."""

    marker_names: tuple[str, ...]
    """Marker labels in array-column order."""

    rate: float
    """Point sample rate [Hz]."""

    first_frame: int
    """First source C3D point-frame number."""

    lab_to_newton: np.ndarray
    """Source-lab to Newton rotation matrix."""

    source_file: str
    """Logical source C3D basename."""

    source_sha256: str
    """SHA-256 of the source C3D bytes."""

    def __post_init__(self) -> None:
        frame_count = len(self.times)
        marker_count = len(self.marker_names)
        if self.times.shape != (frame_count,):
            raise ValueError("times must be one-dimensional")
        if self.positions.shape != (frame_count, marker_count, 3):
            raise ValueError("positions have an invalid shape")
        if self.valid.shape != (frame_count, marker_count):
            raise ValueError("valid has an invalid shape")
        if self.positions.dtype != np.float32 or self.valid.dtype != np.bool_:
            raise ValueError("positions must be float32 and valid must be bool")
        if not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(self.positions)):
            raise ValueError("C3D marker artifact arrays must be finite")
        if len(self.times) > 1 and np.any(np.diff(self.times) <= 0.0):
            raise ValueError("C3D frame times must increase strictly")
        if not math.isfinite(self.rate) or self.rate <= 0.0:
            raise ValueError("C3D point rate must be finite and positive")
        if len(set(self.marker_names)) != marker_count or any(not name for name in self.marker_names):
            raise ValueError("C3D marker names must be nonempty and unique")
        if self.lab_to_newton.shape != (3, 3) or not np.allclose(
            self.lab_to_newton @ self.lab_to_newton.T, np.eye(3), atol=1.0e-9
        ):
            raise ValueError("lab_to_newton must be an orthonormal 3x3 matrix")

    def to_warp(self, device: str | wp.context.Device | None = None) -> WarpMarkerTrajectory:
        """Upload marker data directly to Warp arrays."""
        return WarpMarkerTrajectory(
            wp.array(self.times.astype(np.float32), dtype=float, device=device),
            wp.array(self.positions, dtype=wp.vec3, device=device),
            wp.array(self.valid.astype(np.uint8), dtype=wp.uint8, device=device),
            self.marker_names,
        )


def read_c3d_markers(
    path: str | os.PathLike,
    *,
    up_axis: str = "+Z",
    forward_axis: str = "-Y",
    strip_prefix: bool = True,
) -> C3DMarkerTrajectory:
    """Decode C3D point data directly into finite Newton-frame SI arrays.

    Args:
        path: Source C3D file.
        up_axis: Lab axis pointing upward.
        forward_axis: Lab axis pointing subject-forward.
        strip_prefix: Remove a ``subject:`` prefix from marker labels.

    Returns:
        Marker positions in meters. Missing samples are zero with ``valid=false``.

    Raises:
        ImportError: If the offline ``ezc3d`` decoder is unavailable.
    """
    try:
        import ezc3d  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - optional offline decoder
        raise ImportError("C3D decoding requires ezc3d; run with `uv run --with ezc3d ...`") from error

    source = Path(path).resolve()
    c3d = ezc3d.c3d(str(source))
    point_parameters = c3d["parameters"]["POINT"]
    labels = tuple(str(value).strip() for value in point_parameters["LABELS"]["value"])
    if strip_prefix:
        labels = tuple(label.split(":")[-1].strip() for label in labels)
    rate = float(point_parameters["RATE"]["value"][0])
    header = c3d.get("header", {}).get("points", {})
    first_frame = int(header.get("first_frame", 0))
    units_values = point_parameters.get("UNITS", {}).get("value", ())
    units = str(units_values[0]).strip().lower() if units_values and str(units_values[0]).strip() else "mm"
    try:
        unit_scale = _UNIT_TO_METERS[units]
    except KeyError as error:
        raise ValueError(f"unsupported C3D point units: {units!r}") from error

    raw_points = np.asarray(c3d["data"]["points"], dtype=np.float64)
    if raw_points.ndim != 3 or raw_points.shape[0] < 3 or raw_points.shape[1] != len(labels):
        raise ValueError("C3D point array has an invalid shape")
    positions = np.transpose(raw_points[:3], (2, 1, 0))
    residuals = np.asarray(c3d["data"]["meta_points"]["residuals"], dtype=np.float64)
    if residuals.shape != (1, len(labels), positions.shape[0]):
        raise ValueError("C3D point residual array has an invalid shape")
    valid = residuals[0].T >= 0.0
    valid &= np.all(np.isfinite(positions), axis=-1)
    rotation = lab_to_newton_rotation(up_axis, forward_axis)
    positions = positions @ rotation.T * unit_scale
    positions[~valid] = 0.0
    times = np.arange(len(positions), dtype=np.float64) / rate
    return C3DMarkerTrajectory(
        times,
        positions.astype(np.float32),
        valid.astype(bool),
        labels,
        rate,
        first_frame,
        rotation,
        source.name,
        _sha256(source),
    )


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_marker_artifact(markers: C3DMarkerTrajectory, output_dir: str | os.PathLike) -> Path:
    """Atomically publish a sealed numeric marker artifact and return its root."""
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
        staged = Path(temporary) / "artifact"
        staged.mkdir()
        payload = staged / "markers.npz"
        np.savez_compressed(
            payload,
            times=markers.times.astype(np.float64),
            positions=markers.positions,
            valid=markers.valid,
            lab_to_newton=markers.lab_to_newton.astype(np.float64),
        )
        manifest = {
            "schema_version": _SCHEMA,
            "coordinate_system": {
                "frame": "Newton world",
                "length_unit": "m",
                "up_axis": "Z",
                "forward_axis": "X",
                "left_axis": "Y",
            },
            "source": {
                "file": markers.source_file,
                "first_frame": markers.first_frame,
                "point_rate_hz": markers.rate,
                "sha256": markers.source_sha256,
            },
            "markers": {"names": list(markers.marker_names)},
            "payload": {
                "file": payload.name,
                "sha256": _sha256(payload),
                "arrays": {
                    "times": {"dtype": "float64", "shape": list(markers.times.shape)},
                    "positions": {"dtype": "float32", "shape": list(markers.positions.shape)},
                    "valid": {"dtype": "bool", "shape": list(markers.valid.shape)},
                    "lab_to_newton": {"dtype": "float64", "shape": [3, 3]},
                },
            },
        }
        manifest["seal"] = {
            "algorithm": "sha256",
            "content_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
        }
        _write_json(staged / "manifest.json", manifest)
        os.rename(staged, root)
    return root


def load_marker_artifact(path: str | os.PathLike) -> C3DMarkerTrajectory:
    """Verify and load a sealed marker artifact without a C3D dependency."""
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    seal = manifest.pop("seal", None)
    expected_seal = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    if seal != {"algorithm": "sha256", "content_sha256": expected_seal}:
        raise ValueError("C3D marker manifest seal mismatch")
    if manifest.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported C3D marker artifact schema")
    if manifest.get("coordinate_system") != {
        "frame": "Newton world",
        "length_unit": "m",
        "up_axis": "Z",
        "forward_axis": "X",
        "left_axis": "Y",
    }:
        raise ValueError("C3D marker coordinate-system metadata is invalid")
    payload_info = manifest["payload"]
    if payload_info.get("file") != "markers.npz":
        raise ValueError("C3D marker payload must be the in-root markers.npz file")
    payload = root / "markers.npz"
    if _sha256(payload) != payload_info["sha256"]:
        raise ValueError("C3D marker payload hash mismatch")
    with np.load(payload, allow_pickle=False) as archive:
        expected = {"times", "positions", "valid", "lab_to_newton"}
        if set(archive.files) != expected:
            raise ValueError("C3D marker payload arrays do not match the schema")
        times = np.asarray(archive["times"])
        positions = np.asarray(archive["positions"])
        valid = np.asarray(archive["valid"])
        rotation = np.asarray(archive["lab_to_newton"])
    arrays = payload_info["arrays"]
    for name, array in (("times", times), ("positions", positions), ("valid", valid), ("lab_to_newton", rotation)):
        expected_info = arrays[name]
        if str(array.dtype) != expected_info["dtype"] or list(array.shape) != expected_info["shape"]:
            raise ValueError(f"C3D marker array {name!r} does not match its declared dtype/shape")
    source = manifest["source"]
    return C3DMarkerTrajectory(
        times,
        positions,
        valid,
        tuple(manifest["markers"]["names"]),
        float(source["point_rate_hz"]),
        int(source["first_frame"]),
        rotation,
        str(source["file"]),
        str(source["sha256"]),
    )


def c3d_to_marker_artifact(
    c3d_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    up_axis: str = "+Z",
    forward_axis: str = "-Y",
    strip_prefix: bool = True,
) -> Path:
    """Decode one C3D directly into a sealed NPZ marker artifact."""
    markers = read_c3d_markers(
        c3d_path,
        up_axis=up_axis,
        forward_axis=forward_axis,
        strip_prefix=strip_prefix,
    )
    return write_marker_artifact(markers, output_dir)
