# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare three-dimensional bilateral references on the windows of a planar dataset.

Each segment of an existing ``bilateral_runner_dataset_1`` is re-prepared in 3D
with :mod:`.kinematics3d`: tracked pelvis pose, three-axis hips, and hinge knees
and ankles refined against the measured joint centers and shoe frames. The
single plate's 3D force, COP, and free moment are attributed to the contacting
foot with the planar dataset's contact rule. Splits and windows are unchanged,
so planar and 3D results compare on identical data. Each trial keeps its own
static calibration, stored in the manifest; the model geometry of a world must
use its segment's trial calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import data as planar
from .kinematics3d import COORDINATE_NAMES, Calibration, calibrate, coordinates, fit_legs, track
from .mechanics import SIDES

REFERENCE_SCHEMA = "bilateral_reference_3d_1"
DATASET_SCHEMA = "bilateral_runner_dataset_3d_1"


def calibration_to_dict(calibration: Calibration) -> dict:
    """Serialize a static calibration."""

    def arrays(values: dict) -> dict:
        return {key: np.asarray(value).tolist() for key, value in values.items()}

    return {
        "frames": arrays(calibration.frames),
        "origins": arrays(calibration.origins),
        "templates": arrays(calibration.templates),
        "offsets": arrays(calibration.offsets),
        "hip_m": arrays(calibration.hip_m),
        "knee_m": arrays(calibration.knee_m),
        "ankle_m": arrays(calibration.ankle_m),
    }


def calibration_from_dict(data: dict) -> Calibration:
    """Rebuild a serialized static calibration."""

    def arrays(values: dict) -> dict:
        return {key: np.asarray(value, dtype=float) for key, value in values.items()}

    return Calibration(**{name: arrays(data[name]) for name in Calibration.__dataclass_fields__})


def build_segment(
    trial_root: Path, member: dict, calibration: Calibration, rule: planar.ContactRule, belt_speed_m_s: float
) -> tuple[dict, dict]:
    """Prepare one 3D bilateral segment and its QC record."""
    tracked = track(trial_root, calibration, member["start_s"], member["end_s"], belt_speed_m_s)
    q, report = fit_legs(coordinates(tracked, calibration), tracked, calibration)
    trial, labels, _hip = planar._trial_context(trial_root, rule)
    mask = (trial.marker_time_s >= member["start_s"]) & (trial.marker_time_s <= member["end_s"])
    origin_s = float(trial.marker_time_s[mask][0])
    end_s = float(trial.marker_time_s[mask][-1])
    first = max(0, int(np.searchsorted(trial.analog_time_s, origin_s, side="left")) - 1)
    last = min(len(trial.analog_time_s) - 1, int(np.searchsorted(trial.analog_time_s, end_s, side="right")))
    analog = np.arange(first, last + 1)
    force_time = trial.analog_time_s[analog] - origin_s
    contact = np.stack([labels["masks"][side][analog] for side in SIDES])
    if np.any(contact[0] & contact[1]) or np.any(labels["ambiguous"][analog]):
        raise ValueError("Window contains double or unattributed support")
    force = trial.force_n[analog]
    cop = trial.cop_m[analog].copy()
    cop[:, 0] += force_time * belt_speed_m_s
    loaded = force[:, 2] > rule.threshold_n
    finite = loaded & np.isfinite(cop).all(axis=1)
    for axis in range(3):
        cop[:, axis] = np.interp(force_time, force_time[finite], cop[finite, axis]) if finite.any() else 0.0
    moment = trial.free_moment_nm[analog][:, 2] if trial.free_moment_nm.ndim == 2 else trial.free_moment_nm[analog]
    moment = np.nan_to_num(moment)
    reference = {
        "time_s": tracked["time_s"],
        "q": q,
        "joint_centers_m": np.stack(
            [tracked["centers"][f"{joint}_{s}"] for s in ("r", "l") for joint in ("hip", "knee", "ankle")]
        ),
        "grf_time_s": force_time,
        "grf_side_n": np.stack([np.where(contact[s][:, None], force, 0.0) for s in range(2)]),
        "cop_side_m": np.stack([np.where(contact[s][:, None], cop, 0.0) for s in range(2)]),
        "free_moment_side_nm": np.stack([np.where(contact[s], moment, 0.0) for s in range(2)]),
        "contact_side": contact.astype(np.int8),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema": REFERENCE_SCHEMA,
                    "coordinates": list(COORDINATE_NAMES),
                    "trial": trial_root.name,
                    "source_window_s": [member["start_s"], member["end_s"]],
                    "axes": "Newton: x forward (belt frame), y left, z up",
                }
            )
        ),
    }
    validate(reference)
    record = {"cluster_rms_max_m": tracked["cluster_rms_max_m"], "leg_fit": report}
    return reference, record


def validate(reference: dict) -> None:
    """Validate 3D bilateral reference shapes and metadata."""
    if json.loads(str(reference["metadata_json"])).get("schema") != REFERENCE_SCHEMA:
        raise ValueError("Unsupported 3D bilateral reference schema")
    n, m = len(reference["time_s"]), len(reference["grf_time_s"])
    shapes = {
        "q": (n, len(COORDINATE_NAMES)),
        "joint_centers_m": (6, n, 3),
        "grf_side_n": (2, m, 3),
        "cop_side_m": (2, m, 3),
        "free_moment_side_nm": (2, m),
        "contact_side": (2, m),
    }
    for name, shape in shapes.items():
        if reference[name].shape != shape or not np.isfinite(reference[name]).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
    if reference["time_s"][0] != 0 or np.any(np.diff(reference["time_s"]) <= 0):
        raise ValueError("time_s must start at zero and increase")
    if reference["grf_time_s"][0] > 0 or reference["grf_time_s"][-1] < reference["time_s"][-1]:
        raise ValueError("GRF must cover the motion window")


def load(path: str | Path) -> dict:
    """Load and validate a 3D bilateral reference NPZ without pickle."""
    with np.load(Path(path), allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    validate(reference)
    return reference


def build_dataset(planar_dataset: Path, output: Path, *, data_root: Path = planar.ROOT / "data/F01") -> Path:
    """Write the 3D dataset on the planar dataset's windows and splits."""
    planar_dataset = planar_dataset.resolve()
    source = json.loads((planar_dataset / "manifest.json").read_text(encoding="utf-8"))
    if source.get("schema") != planar.DATASET_SCHEMA:
        raise ValueError("Expected a planar bilateral dataset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    building = output.with_name(f".{output.name}.{os.getpid()}.building")
    building.mkdir(parents=True)
    rule = planar.ContactRule(**source["policy"]["contact_rule"])
    trials = sorted({member["trial"] for member in source["members"]})
    calibrations = {trial: calibrate(data_root / trial) for trial in trials}
    members, rejected = [], []
    # Copy the shared assets so the dataset is self-contained and portable between machines.
    (building / "assets").mkdir()
    shared_assets = {}
    for name, entry in source["shared_assets"].items():
        shutil.copyfile(planar_dataset / entry["file"], building / "assets" / name)
        shared_assets[name] = {"file": f"assets/{name}", "sha256": entry["sha256"]}
    for member in source["members"]:
        trial_root = data_root / member["trial"]
        try:
            reference, record = build_segment(
                trial_root, member, calibrations[member["trial"]], rule, float(member["speed_m_s"])
            )
        except ValueError as error:
            rejected.append({"id": member["id"], "reason": str(error)})
            continue
        path = building / member["split"] / f"{member['id']}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **reference)
        members.append(
            {key: member[key] for key in ("id", "trial", "split", "speed_m_s", "start_s", "end_s", "contacts")}
            | {"reference": path.relative_to(building).as_posix(), **record}
        )
        print(member["id"], member["split"], record["leg_fit"]["right"]["ankle_error_peak_m"], flush=True)
    manifest = {
        "schema": DATASET_SCHEMA,
        "planar_dataset": planar_dataset.name,
        "planar_manifest_sha256": hashlib.sha256((planar_dataset / "manifest.json").read_bytes()).hexdigest(),
        "subject_id": source.get("subject_id"),
        "subject_mass_kg": float(planar.load(planar_dataset / source["members"][0]["reference"])["subject_mass_kg"]),
        "head_arms_trunk": source["head_arms_trunk"],
        "shoe": source["shoe"],
        "shared_assets": shared_assets,
        "calibrations": {trial: calibration_to_dict(value) for trial, value in calibrations.items()},
        "contact_rule": asdict(rule),
        "coordinates": list(COORDINATE_NAMES),
        "members": members,
        "rejected": rejected,
    }
    (building / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    building.rename(output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planar-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=planar.ROOT / "data/F01")
    args = parser.parse_args(argv)
    print(build_dataset(args.planar_dataset, args.output, data_root=args.data_root))


if __name__ == "__main__":
    main()
