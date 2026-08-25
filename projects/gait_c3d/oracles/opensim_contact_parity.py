# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare official OpenSim and Newton SmoothSphereHalfSpace contact loads.

The official ``opensim`` package is optional. It is imported only while running
an official comparison. XML construction, frame selection, record ordering, and
comparison math remain usable without that package.

The comparison intentionally reads only archived coordinates and speeds.
Measured platform loads are not inputs to either contact implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import newton.opensim as newton_osim
from projects.gait_c3d.compatibility import predictive_contact

ARCHITECTURE_ROLE = "cross_runtime_oracle"

_SCHEMA = "gait_c3d_official_newton_contact_parity_1"
_FRAME = "opensim_x_forward_y_up_z_right"
_BODY_ORDER = ("calcn_l", "calcn_r")
_SIDE_ORDER = ("left", "right")
_FORCE_ATOL_N = 1.0e-3
_TORQUE_ATOL_NM = 1.0e-4
_RELATIVE_TOLERANCE = 1.0e-4
_VELOCITY_STENCIL_H_S = 1.0e-6
_DEFAULT_SIDECAR = Path(
    "/home/jo31399/newton-data/gait/processed/trial_101/"
    "stage2_prescribed_contact_calibrated_clean_v2/contact_sidecar.json"
)
_RECORD_SUFFIXES = (
    "Sphere.force.X",
    "Sphere.force.Y",
    "Sphere.force.Z",
    "Sphere.torque.X",
    "Sphere.torque.Y",
    "Sphere.torque.Z",
    "HalfSpace.force.X",
    "HalfSpace.force.Y",
    "HalfSpace.force.Z",
    "HalfSpace.torque.X",
    "HalfSpace.torque.Y",
    "HalfSpace.torque.Z",
)


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _float_text(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("OpenSim XML values must be finite")
    return format(value, ".17g")


def _vec_text(values: Sequence[float]) -> str:
    if len(values) != 3:
        raise ValueError("OpenSim vectors must contain three values")
    return " ".join(_float_text(value) for value in values)


def _element(parent: ET.Element, tag: str, text: str | float | None = None, **attributes: str) -> ET.Element:
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        child.text = _float_text(text) if isinstance(text, float) else text
    return child


def _xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True) + b"\n"


def build_contact_geometry_xml(sidecar: predictive_contact.PredictiveContactSidecar) -> ET.Element:
    """Build the deterministic official ContactGeometrySet document.

    The half-space is emitted first. Spheres retain the strict sidecar order.
    Socket paths are absolute paths in the scaled model.
    """
    document = ET.Element("OpenSimDocument", {"Version": "40000"})
    geometry_set = _element(document, "ContactGeometrySet", None, name="predictive_contact_geometry")
    objects = _element(geometry_set, "objects")
    half_space = _element(objects, "ContactHalfSpace", None, name=sidecar.ground.name)
    _element(half_space, "socket_frame", "/ground")
    _element(half_space, "location", _vec_text((0.0, sidecar.ground.height_m, 0.0)))
    _element(half_space, "orientation", _vec_text((0.0, 0.0, -0.5 * math.pi)))
    for sphere in sidecar.spheres:
        item = _element(objects, "ContactSphere", None, name=sphere.name)
        _element(item, "socket_frame", f"/bodyset/{sphere.body}")
        _element(item, "location", _vec_text(sphere.center_m))
        _element(item, "orientation", "0 0 0")
        _element(item, "radius", float(sphere.radius_m))
    _element(geometry_set, "groups")
    return document


def build_force_xml(sidecar: predictive_contact.PredictiveContactSidecar) -> ET.Element:
    """Build the deterministic official ForceSet document in sidecar order."""
    document = ET.Element("OpenSimDocument", {"Version": "40000"})
    force_set = _element(document, "ForceSet", None, name="predictive_contact_forces")
    objects = _element(force_set, "objects")
    parameters = sidecar.material.parameters()
    parameter_order = (
        "stiffness",
        "dissipation",
        "static_friction",
        "dynamic_friction",
        "viscous_friction",
        "transition_velocity",
        "constant_contact_force",
        "hertz_smoothing",
        "hunt_crossley_smoothing",
    )
    for sphere in sidecar.spheres:
        force = _element(objects, "SmoothSphereHalfSpaceForce", None, name=sphere.force_name)
        _element(force, "socket_sphere", f"/contactgeometryset/{sphere.name}")
        _element(force, "socket_half_space", f"/contactgeometryset/{sidecar.ground.name}")
        for name in parameter_order:
            _element(force, name, float(parameters[name]))
    _element(force_set, "groups")
    return document


def expected_record_labels(force_name: str) -> tuple[str, ...]:
    """Return official SmoothSphereHalfSpaceForce record-label order."""
    if not force_name:
        raise ValueError("force_name must not be empty")
    return tuple(f"{force_name}.{suffix}" for suffix in _RECORD_SUFFIXES)


def sphere_record_wrench(force_name: str, labels: Sequence[str], values: Sequence[float]) -> np.ndarray:
    """Extract official Sphere ``[force, torque]`` in the ground frame.

    Official OpenSim's Sphere torque is about the attached mobilized-body origin,
    which is the attached body origin for the generated geometry.
    """
    expected = expected_record_labels(force_name)
    actual = tuple(str(label) for label in labels)
    if actual != expected:
        raise ValueError(f"unexpected record labels for {force_name!r}: {actual}")
    array = np.asarray(values, dtype=float)
    if array.shape != (12,) or not np.all(np.isfinite(array)):
        raise ValueError("official record values must contain twelve finite values")
    return array[:6].copy()


def aggregate_element_wrenches(element_wrenches: np.ndarray, element_sides: Sequence[str]) -> np.ndarray:
    """Aggregate ordered element ``[F,T]`` records into left/right body loads."""
    values = np.asarray(element_wrenches, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (len(element_sides), 6):
        raise ValueError("element_wrenches must have shape [frame, element, 6]")
    if not np.all(np.isfinite(values)):
        raise ValueError("element_wrenches must be finite")
    if any(side not in _SIDE_ORDER for side in element_sides):
        raise ValueError("element sides must be left or right")
    result = np.zeros((len(values), len(_SIDE_ORDER), 6), dtype=float)
    for element, side in enumerate(element_sides):
        result[:, _SIDE_ORDER.index(side)] += values[:, element]
    return result


def select_frame_indices(
    frame_count: int, frame_indices: Sequence[int] | None = None, *, full_frames: bool = False
) -> np.ndarray:
    """Select full frames or deterministic endpoint/quartile audit frames."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if full_frames and frame_indices is not None:
        raise ValueError("frame_indices and full_frames are mutually exclusive")
    if full_frames:
        return np.arange(frame_count, dtype=np.int64)
    if frame_indices is None:
        return np.unique(np.rint(np.linspace(0, frame_count - 1, 5)).astype(np.int64))
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in frame_indices):
        raise TypeError("frame_indices must contain integers")
    result = np.asarray(frame_indices, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= frame_count):
        raise IndexError("frame index is outside the archived range")
    if len(result) > 1 and np.any(np.diff(result) <= 0):
        raise ValueError("frame_indices must be unique and strictly increasing")
    return result


def comparison_metrics(
    official_wrenches: np.ndarray,
    newton_wrenches: np.ndarray,
    *,
    force_atol_n: float = _FORCE_ATOL_N,
    torque_atol_nm: float = _TORQUE_ATOL_NM,
    relative_tolerance: float = _RELATIVE_TOLERANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Newton-minus-official differences and mixed absolute/relative gates."""
    official = np.asarray(official_wrenches, dtype=float)
    newton = np.asarray(newton_wrenches, dtype=float)
    if official.shape != newton.shape or official.ndim != 3 or official.shape[1:] != (2, 6):
        raise ValueError("wrenches must share shape [frame, 2, 6]")
    if not np.all(np.isfinite(official)) or not np.all(np.isfinite(newton)):
        raise ValueError("wrenches must be finite")
    for name, value in (
        ("force_atol_n", force_atol_n),
        ("torque_atol_nm", torque_atol_nm),
        ("relative_tolerance", relative_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    difference = newton - official

    def metrics(
        values: np.ndarray, reference: np.ndarray, candidate: np.ndarray, atol: float, unit: str
    ) -> dict[str, Any]:
        scale = np.maximum(np.linalg.norm(reference, axis=-1), np.linalg.norm(candidate, axis=-1))
        normalized = np.abs(values) / (atol + relative_tolerance * scale[..., None])
        return {
            "max_abs": float(np.max(np.abs(values))),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "atol": atol,
            "rtol": relative_tolerance,
            "max_normalized_error": float(np.max(normalized)),
            "unit": unit,
            "passed": bool(np.max(normalized) <= 1.0),
        }

    force = metrics(difference[..., :3], official[..., :3], newton[..., :3], force_atol_n, "N")
    torque = metrics(difference[..., 3:], official[..., 3:], newton[..., 3:], torque_atol_nm, "N*m")
    per_body: dict[str, Any] = {}
    for body_index, body in enumerate(_BODY_ORDER):
        per_body[body] = {
            "force": metrics(
                difference[:, body_index, :3], official[:, body_index, :3], newton[:, body_index, :3], force_atol_n, "N"
            ),
            "torque": metrics(
                difference[:, body_index, 3:],
                official[:, body_index, 3:],
                newton[:, body_index, 3:],
                torque_atol_nm,
                "N*m",
            ),
        }
    return difference, {
        "force": force,
        "torque": torque,
        "per_body": per_body,
        "passed": force["passed"] and torque["passed"],
    }


def _import_official_opensim() -> Any:
    try:
        return importlib.import_module("opensim")
    except ImportError as error:
        raise RuntimeError("official OpenSim Python bindings are required to run contact parity") from error


def _resolve_sidecar_source(raw_path: str, sidecar_path: Path) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (sidecar_path.parent / path).resolve()


def _load_official_augmented_model(opensim: Any, model_path: Path, geometry_path: Path, force_path: Path) -> Any:
    """Load the original model and append clones loaded from the generated sets."""
    model = opensim.Model(str(model_path))
    geometry_set = opensim.ContactGeometrySet(str(geometry_path))
    force_set = opensim.ForceSet(str(force_path))
    for index in range(geometry_set.getSize()):
        model.addContactGeometry(geometry_set.get(index).clone())
    for index in range(force_set.getSize()):
        model.addForce(force_set.get(index).clone())
    model.finalizeConnections()
    return model


def _official_contact_wrenches(
    opensim: Any,
    model: Any,
    sidecar: predictive_contact.PredictiveContactSidecar,
    coordinate_names: Sequence[str],
    times: np.ndarray,
    coordinates: np.ndarray,
    speeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    official_names = [
        model.getCoordinateSet().get(index).getName() for index in range(model.getCoordinateSet().getSize())
    ]
    if list(coordinate_names) != official_names:
        raise ValueError("archived coordinate order does not match official OpenSim")
    state = model.initSystem()
    force_objects = []
    label_order: list[list[str]] = []
    for sphere in sidecar.spheres:
        force = opensim.SmoothSphereHalfSpaceForce.safeDownCast(model.getForceSet().get(sphere.force_name))
        if force is None:
            raise TypeError(f"force {sphere.force_name!r} is not SmoothSphereHalfSpaceForce")
        labels = force.getRecordLabels()
        force_labels = [str(labels.get(index)) for index in range(labels.getSize())]
        if tuple(force_labels) != expected_record_labels(sphere.force_name):
            raise ValueError(f"official record ordering changed for {sphere.force_name!r}")
        force_objects.append(force)
        label_order.append(force_labels)

    element_values = np.empty((len(times), len(sidecar.spheres), 6), dtype=float)
    coordinate_set = model.getCoordinateSet()
    for frame in range(len(times)):
        state.setTime(float(times[frame]))
        for coordinate, name in enumerate(coordinate_names):
            component = coordinate_set.get(name)
            component.setValue(state, float(coordinates[frame, coordinate]), False)
            component.setSpeedValue(state, float(speeds[frame, coordinate]))
        model.realizeDynamics(state)
        for element, (sphere, force) in enumerate(zip(sidecar.spheres, force_objects, strict=True)):
            record = force.getRecordValues(state)
            values = [float(record.get(index)) for index in range(record.size())]
            element_values[frame, element] = sphere_record_wrench(sphere.force_name, label_order[element], values)
    aggregate = aggregate_element_wrenches(element_values, [sphere.side for sphere in sidecar.spheres])
    return aggregate, element_values, label_order


def _newton_contact_wrenches(
    model_path: Path,
    sidecar: predictive_contact.PredictiveContactSidecar,
    coordinate_names: Sequence[str],
    coordinates: np.ndarray,
    speeds: np.ndarray,
    *,
    device: str,
    velocity_stencil_h_s: float,
) -> tuple[np.ndarray, list[str]]:
    model = newton_osim.parse_osim(model_path)
    augmented = predictive_contact.augment_contact_model(model, sidecar)
    contact = newton_osim.OpenSimContact(augmented, device=device)
    if list(coordinate_names) != list(contact.coordinate_names):
        raise ValueError("archived coordinate order does not match Newton OpenSimContact")
    body_names, raw_wrenches = contact.body_wrenches(coordinates, speeds, h=velocity_stencil_h_s, frame="opensim")
    if len(set(body_names)) != len(body_names) or any(body not in body_names for body in _BODY_ORDER):
        raise ValueError("Newton OpenSimContact did not return unique bilateral contact bodies")
    indices = [body_names.index(body) for body in _BODY_ORDER]
    selected = np.asarray(raw_wrenches, dtype=float)[:, indices]
    if selected.shape != (len(coordinates), 2, 9) or not np.all(np.isfinite(selected)):
        raise ValueError("Newton OpenSimContact returned invalid body wrenches")
    # OpenSimContact is [force, point=body origin, torque about point].
    return np.concatenate((selected[..., :3], selected[..., 6:9]), axis=-1), list(body_names)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_runtime(repository_root: Path) -> dict[str, Any]:
    """Return code-identifying Git and source-file provenance."""
    try:
        commit = subprocess.check_output(["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(repository_root), "status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, status = "unknown", "unknown"
    contact_source = repository_root / "newton" / "_src" / "opensim" / "contact.py"
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "parity_source_sha256": _sha256(Path(__file__).resolve()),
        "predictive_contact_source_sha256": _sha256(Path(predictive_contact.__file__).resolve()),
        "newton_contact_source_sha256": _sha256(contact_source) if contact_source.is_file() else None,
    }


def run_contact_parity(
    sidecar_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    frame_indices: Sequence[int] | None = None,
    full_frames: bool = False,
    device: str = "cpu",
    velocity_stencil_h_s: float = _VELOCITY_STENCIL_H_S,
) -> Path:
    """Run official/Newton contact parity and publish a new atomic artifact.

    A failed numerical gate is a valid parity result and is published with
    ``passed: false``. Invalid inputs or an unavailable official runtime raise.
    Existing output paths are never overwritten.
    """
    if not math.isfinite(velocity_stencil_h_s) or velocity_stencil_h_s <= 0.0:
        raise ValueError("velocity_stencil_h_s must be positive and finite")
    sidecar_path = Path(sidecar_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = predictive_contact.load_contact_sidecar(sidecar_path)
    model_path = _resolve_sidecar_source(sidecar.source_model_path, sidecar_path)
    analysis_path = _resolve_sidecar_source(sidecar.source_analysis_path, sidecar_path)
    for path in (model_path, analysis_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(model_path) != sidecar.source_model_sha256:
        raise ValueError("sidecar source model SHA-256 does not match")
    if _sha256(analysis_path) != sidecar.source_analysis_sha256:
        raise ValueError("sidecar source analysis SHA-256 does not match")
    base_model = newton_osim.parse_osim(model_path)
    if base_model.contact_geometry or base_model.contact_forces:
        raise ValueError("contact parity requires a source model with no pre-existing contact elements")

    # Deliberately request only motion arrays. Measured GRF/COP/moments are not read.
    with np.load(analysis_path, allow_pickle=False) as archive:
        times = np.asarray(archive["times"], dtype=float)
        coordinates = np.asarray(archive["id_coordinates"], dtype=float)
        speeds = np.asarray(archive["id_speeds"], dtype=float)
        coordinate_names = [str(value) for value in np.asarray(archive["id_names"])]
    if times.ndim != 1 or coordinates.shape != speeds.shape or coordinates.shape != (len(times), len(coordinate_names)):
        raise ValueError("archived motion arrays have inconsistent shapes")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(speeds)):
        raise ValueError("archived time, q, and qd must be finite")
    indices = select_frame_indices(len(times), frame_indices, full_frames=full_frames)
    selected_times = times[indices]
    selected_coordinates = coordinates[indices]
    selected_speeds = speeds[indices]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        geometry_path = temporary / "contact_geometry.xml"
        force_path = temporary / "contact_forces.xml"
        geometry_path.write_bytes(_xml_bytes(build_contact_geometry_xml(sidecar)))
        force_path.write_bytes(_xml_bytes(build_force_xml(sidecar)))
        shutil.copyfile(sidecar_path, temporary / "contact_sidecar.json")

        opensim = _import_official_opensim()
        official_model = _load_official_augmented_model(opensim, model_path, geometry_path, force_path)
        official, official_elements, record_labels = _official_contact_wrenches(
            opensim,
            official_model,
            sidecar,
            coordinate_names,
            selected_times,
            selected_coordinates,
            selected_speeds,
        )
        newton, newton_returned_body_order = _newton_contact_wrenches(
            model_path,
            sidecar,
            coordinate_names,
            selected_coordinates,
            selected_speeds,
            device=device,
            velocity_stencil_h_s=velocity_stencil_h_s,
        )
        difference, metrics = comparison_metrics(official, newton)
        arrays_path = temporary / "contact_parity_arrays.npz"
        np.savez_compressed(
            arrays_path,
            frame_indices=indices,
            times=selected_times,
            coordinates=selected_coordinates,
            speeds=selected_speeds,
            official_element_sphere_wrenches=official_elements,
            official_body_wrenches=official,
            newton_body_wrenches=newton,
            newton_minus_official=difference,
        )
        artifact_hashes = {
            path.name: _sha256(path)
            for path in (geometry_path, force_path, temporary / "contact_sidecar.json", arrays_path)
        }
        manifest = {
            "schema_version": _SCHEMA,
            "scope": "official_opensim_vs_newton_native_smooth_sphere_half_space_contact",
            "source": {
                "model_path": str(model_path),
                "model_sha256": sidecar.source_model_sha256,
                "analysis_path": str(analysis_path),
                "analysis_sha256": sidecar.source_analysis_sha256,
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": _sha256(sidecar_path),
            },
            "runtime": {
                "official_opensim_version": str(opensim.GetVersion()),
                "newton_distribution_version": _package_version("newton"),
                "numpy_version": np.__version__,
                "newton_device": device,
                **_git_runtime(Path(__file__).resolve().parents[2]),
            },
            "frame": _FRAME,
            "units": {"force": "N", "torque": "N*m", "time": "s"},
            "selection": {
                "mode": "full"
                if full_frames
                else ("explicit" if frame_indices is not None else "deterministic_quartiles"),
                "source_frame_count": len(times),
                "frame_indices": indices.tolist(),
                "times_s": selected_times.tolist(),
            },
            "coordinate_order": coordinate_names,
            "anatomical_side_order": list(_SIDE_ORDER),
            "compared_body_order": list(_BODY_ORDER),
            "newton_returned_body_order": newton_returned_body_order,
            "element_order": [sphere.force_name for sphere in sidecar.spheres],
            "element_geometry_order": [sphere.name for sphere in sidecar.spheres],
            "element_side_order": [sphere.side for sphere in sidecar.spheres],
            "official_record_label_order": record_labels,
            "wrench_layout": ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"],
            "torque_reference": "attached_body_origin_ground_expressed",
            "velocity_stencil_h_s": velocity_stencil_h_s,
            "measured_loads_used": False,
            "comparison": metrics,
            "artifact_sha256": artifact_hashes,
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=_DEFAULT_SIDECAR)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frame-index", type=int, action="append", dest="frame_indices")
    selection.add_argument("--full-frames", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--velocity-stencil-h-s", type=float, default=_VELOCITY_STENCIL_H_S)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    result = run_contact_parity(
        args.sidecar,
        args.output_dir,
        frame_indices=args.frame_indices,
        full_frames=args.full_frames,
        device=args.device,
        velocity_stencil_h_s=args.velocity_stencil_h_s,
    )
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    print(result)
    print(json.dumps(manifest["comparison"], indent=2, sort_keys=True))
    return 0 if manifest["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
