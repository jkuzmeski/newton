# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert accepted RRA contact inputs into a neutral Newton contact artifact.

This is the OpenSim adapter boundary. The output contains only Newton Z-up body
poses/velocities, measured validation wrenches, and neutral sphere topology. The
native contact runtime does not import or call ``newton.opensim``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import newton.opensim as osim

ARCHITECTURE_ROLE = "source_adapter"

_SCHEMA = "gait_c3d_newton_contact_input_1"
_DEFAULT_INPUT = Path("/home/jo31399/newton-data/gait/processed/trial_101/rra_adjusted_contact_input")
_BODY_ORDER = ("calcn_l", "toes_l", "calcn_r", "toes_r")
_ROLE_ORDER = ("heel", "lateralRearfoot", "lateralMidfoot", "medialMidfoot", "lateralToe", "medialToe")
_ROLE_BODY = {
    "heel": "calcn",
    "lateralRearfoot": "calcn",
    "lateralMidfoot": "calcn",
    "medialMidfoot": "calcn",
    "lateralToe": "toes",
    "medialToe": "toes",
}
_FRAME_ROTATION = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True, slots=True)
class SourceContactInput:
    """Accepted source arrays used only on the conversion side of the boundary."""

    root: Path
    model_path: Path
    times: np.ndarray
    coordinate_names: tuple[str, ...]
    coordinates: np.ndarray
    speeds: np.ndarray
    measured_wrenches: np.ndarray
    measured_contact: np.ndarray


def _load_source_contact_input(root: str | os.PathLike) -> SourceContactInput:
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    qc_path = directory / "qc_summary.json"
    analysis_path = directory / "analysis.npz"
    model_path = directory / "S001_scaled.osim"
    for path in (manifest_path, qc_path, analysis_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    qc = json.loads(qc_path.read_text())
    if manifest.get("status") != "production_candidate" or qc.get("status") != "production_candidate":
        raise ValueError("conversion requires the accepted RRA production candidate")
    for name, expected in manifest["artifacts"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"source contact input is stale: {name}")
    archive = np.load(analysis_path, allow_pickle=False)
    times = np.asarray(archive["times"], dtype=float)
    coordinates = np.asarray(archive["id_coordinates"], dtype=float)
    speeds = np.asarray(archive["id_speeds"], dtype=float)
    names = tuple(str(value) for value in archive["id_names"])
    measured = np.zeros((len(times), 2, 9), dtype=float)
    measured[..., :3] = np.asarray(archive["grf"], dtype=float)
    measured[..., 3:6] = np.where(np.isfinite(archive["cop"]), archive["cop"], 0.0)
    measured[..., 6:9] = np.asarray(archive["free_torque"], dtype=float)
    contact = np.asarray(archive["contact"], dtype=bool)
    return SourceContactInput(directory, model_path, times, names, coordinates, speeds, measured, contact)


def _prepare_source_model(
    model: osim.OsimModel, coordinate_names: tuple[str, ...], coordinates: np.ndarray
) -> tuple[osim.OsimModel, dict[str, Any]]:
    """Repair the legacy zero-width toe ranges at the source adapter boundary."""
    prepared = copy.deepcopy(model)
    bound = np.deg2rad(30.0)
    repairs: dict[str, Any] = {}
    for joint in prepared.joints:
        for coordinate in joint.coordinates:
            if coordinate.name not in ("mtp_angle_l", "mtp_angle_r"):
                continue
            index = coordinate_names.index(coordinate.name)
            observed = [float(np.min(coordinates[:, index])), float(np.max(coordinates[:, index]))]
            old_range = coordinate.range
            if coordinate.locked or observed[0] < -bound or observed[1] > bound:
                raise ValueError(f"invalid accepted toe state for {coordinate.name}")
            if old_range is None or old_range[0] == old_range[1]:
                coordinate.range = (-bound, bound)
                coordinate.clamped = True
                repairs[coordinate.name] = {
                    "old_range_rad": old_range,
                    "new_range_rad": list(coordinate.range),
                    "observed_range_rad": observed,
                }
    if set(repairs) != {"mtp_angle_l", "mtp_angle_r"}:
        raise ValueError("expected explicit repairs for both legacy toe ranges")
    return prepared, repairs


def _subject_topology(model: osim.OsimModel) -> list[dict[str, Any]]:
    """Seed neutral spheres from scaled anatomical landmarks without force data."""
    markers = {marker.name: marker for marker in model.markers}
    kinematics = osim.ForwardKinematics(model, device="cpu")
    defaults = {coordinate.name: coordinate.default_value for joint in model.joints for coordinate in joint.coordinates}
    q0 = np.asarray([[defaults.get(name, 0.0) for name in kinematics.coordinate_names]], dtype=float)
    transforms = np.asarray(kinematics.body_transforms_batch(q0), dtype=float)[0]
    body_transform = {name: transforms[index] for index, name in enumerate(kinematics.body_names)}
    result: list[dict[str, Any]] = []
    radius = 0.035
    for side in ("left", "right"):
        suffix = "l" if side == "left" else "r"
        prefix = "L" if side == "left" else "R"
        marker_names = (f"{prefix}.Heel", f"{prefix}.Toe.Lat", f"{prefix}.Toe.Med")
        if not set(marker_names).issubset(markers):
            raise KeyError(f"scaled source model is missing {side} foot landmarks")
        heel, lateral, medial = (np.asarray(markers[name].location, dtype=float) for name in marker_names)
        if any(markers[name].body != f"calcn_{suffix}" for name in marker_names):
            raise ValueError("foot landmarks must be expressed in the calcaneus frame")
        points = {
            "heel": heel,
            "lateralRearfoot": heel + 0.44 * (lateral - heel),
            "lateralMidfoot": heel + 0.87 * (lateral - heel),
            "medialMidfoot": medial,
            "lateralToe": lateral,
            "medialToe": medial,
        }
        calcaneus = f"calcn_{suffix}"
        toes = f"toes_{suffix}"
        calc_to_world = body_transform[calcaneus]
        world_to_toes = np.linalg.inv(body_transform[toes])
        for role in _ROLE_ORDER:
            body = f"{_ROLE_BODY[role]}_{suffix}"
            point = points[role].copy()
            if body == toes:
                point = (world_to_toes @ calc_to_world @ np.asarray((*point, 1.0)))[:3]
            # The marker is a surface landmark. Move the sphere center inward by
            # one radius in the local vertical direction; force data are not used.
            point[1] -= radius
            result.append(
                {
                    "name": f"{role}_{suffix}",
                    "side": side,
                    "role": role,
                    "body": body,
                    "location_m": [float(value) for value in point],
                    "radius_m": radius,
                }
            )
    return result


def prepare_newton_contact_input(
    output_dir: str | os.PathLike,
    *,
    rra_input: str | os.PathLike = _DEFAULT_INPUT,
    device: str = "cuda:0",
) -> Path:
    """Publish a hash-sealed neutral Newton contact input artifact."""
    output = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[3]
    if output.exists():
        raise FileExistsError(output)
    if output == repository or output.is_relative_to(repository):
        raise ValueError("generated contact inputs must remain outside the repository")
    inputs = _load_source_contact_input(rra_input)
    if output == inputs.root or output.is_relative_to(inputs.root) or inputs.root.is_relative_to(output):
        raise ValueError("adapter input and output directories must not overlap")

    from scipy.spatial.transform import Rotation

    source_model = osim.parse_osim(inputs.model_path)
    prepared_model, mtp_repairs = _prepare_source_model(source_model, inputs.coordinate_names, inputs.coordinates)
    seed_spheres = _subject_topology(prepared_model)
    kinematics = osim.ForwardKinematics(prepared_model, device=device)
    if tuple(kinematics.coordinate_names) != inputs.coordinate_names:
        raise ValueError("accepted state order does not match the conversion model")
    transforms = np.asarray(kinematics.body_transforms_batch(inputs.coordinates), dtype=float)
    indices = [tuple(kinematics.body_names).index(name) for name in _BODY_ORDER]
    transforms = transforms[:, indices]
    rotation = np.einsum("ij,fbjk->fbik", _FRAME_ROTATION, transforms[..., :3, :3])
    position = np.einsum("ij,fbj->fbi", _FRAME_ROTATION, transforms[..., :3, 3])
    quaternion = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat().reshape(len(inputs.times), 4, 4)
    velocity = kinematics.body_velocities_batch(inputs.coordinates, inputs.speeds)
    linear_velocity = np.einsum(
        "ij,fbj->fbi", _FRAME_ROTATION, np.asarray(velocity["linear_velocity"], dtype=float)[:, indices]
    )
    angular_velocity = np.einsum(
        "ij,fbj->fbi", _FRAME_ROTATION, np.asarray(velocity["angular_velocity"], dtype=float)[:, indices]
    )
    body_pose = np.concatenate((position, quaternion), axis=-1)
    body_velocity = np.concatenate((linear_velocity, angular_velocity), axis=-1)

    measured = inputs.measured_wrenches.copy()
    for section in (slice(0, 3), slice(3, 6), slice(6, 9)):
        measured[..., section] = np.einsum("ij,fsj->fsi", _FRAME_ROTATION, measured[..., section])
    if not all(np.all(np.isfinite(value)) for value in (body_pose, body_velocity, measured)):
        raise ValueError("neutral Newton contact conversion produced non-finite arrays")

    topology = {
        "schema_version": _SCHEMA,
        "frame": "newton_x_forward_y_left_z_up",
        "body_order": list(_BODY_ORDER),
        "ground": {"type": "plane", "height_m": 0.0, "up_axis": "+Z"},
        "spheres": [dict(sphere) for sphere in seed_spheres],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        np.savez_compressed(
            temporary / "motion_and_targets.npz",
            times=inputs.times,
            body_pose=body_pose,
            body_velocity=body_velocity,
            measured_foot_wrenches=measured,
            measured_contact=inputs.measured_contact,
            body_names=np.asarray(_BODY_ORDER),
        )
        _write_json(temporary / "topology.json", topology)
        _write_json(temporary / "mtp_range_repairs.json", mtp_repairs)
        artifacts = ("motion_and_targets.npz", "topology.json", "mtp_range_repairs.json")
        manifest = {
            "schema_version": _SCHEMA,
            "status": "production_candidate",
            "scope": "source_conversion_only_neutral_newton_contact_input",
            "architecture": {
                "opensim_used_after_this_boundary": False,
                "runtime_types": ["newton.Model", "newton.State", "newton.Contacts"],
                "frame": "newton_x_forward_y_left_z_up",
            },
            "source": {
                "rra_input": str(inputs.root),
                "rra_manifest_sha256": _sha256(inputs.root / "manifest.json"),
                "model_sha256": _sha256(inputs.model_path),
                "analysis_sha256": _sha256(inputs.root / "analysis.npz"),
            },
            "conversion": {
                "body_order": list(_BODY_ORDER),
                "world_rotation": _FRAME_ROTATION.tolist(),
                "body_pose_representation": "position_xyz_plus_quaternion_xyzw",
                "body_velocity_representation": "world_linear_xyz_plus_world_angular_xyz",
                "sphere_local_coordinates_unchanged": True,
                "measured_loads_used_for_geometry": False,
            },
            "artifacts": {name: _sha256(temporary / name) for name in artifacts},
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rra-input", default=str(_DEFAULT_INPUT))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(prepare_newton_contact_input(args.output_dir, rra_input=args.rra_input, device=args.device))


if __name__ == "__main__":
    main()
