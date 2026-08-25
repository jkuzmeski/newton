# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate planned or completed multi-rate Digital Shoe acquisitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

ALLOWED_MODES = {"cyclic", "hold_relaxation", "drop_impact", "rocker"}
REQUIRED_STUDY_FIELDS = {
    "specimen_id",
    "shoe_model",
    "side",
    "size",
    "condition",
    "machine_id",
    "load_cell_id",
    "calibration_id",
    "operator",
    "temperature_c",
}


def _positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _portable_path(value: Any, name: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path without '..'")
    return str(path)


def validate_acquisition_manifest(data: dict[str, Any], *, base: str | Path | None = None) -> None:
    """Validate protocol metadata, acquisition identity, splits, and optional files."""
    if data.get("schema_version") != "digital_shoe_acquisition_1":
        raise ValueError("schema_version must be 'digital_shoe_acquisition_1'")
    study = data.get("study", {})
    if missing := sorted(REQUIRED_STUDY_FIELDS - study.keys()):
        raise ValueError(f"study is missing {missing}")
    if not math.isfinite(float(study["temperature_c"])):
        raise ValueError("study temperature_c must be finite")
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, dict) or not fixtures:
        raise ValueError("fixtures must be a nonempty object")

    protocols = data.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("protocols must be a nonempty list")
    protocol_by_id: dict[str, dict[str, Any]] = {}
    for protocol in protocols:
        identifier = str(protocol.get("id", ""))
        if not identifier or identifier in protocol_by_id:
            raise ValueError(f"invalid or duplicate protocol id {identifier!r}")
        if protocol.get("fixture_id") not in fixtures:
            raise ValueError(f"protocol {identifier!r} references an unknown fixture")
        mode = protocol.get("mode")
        if mode not in ALLOWED_MODES:
            raise ValueError(f"protocol {identifier!r} has unsupported mode {mode!r}")
        _positive(protocol.get("sample_rate_hz"), f"protocol {identifier!r} sample_rate_hz")
        if int(protocol.get("preconditioning_cycles", -1)) < 0 or float(protocol.get("recovery_s", -1)) < 0.0:
            raise ValueError(f"protocol {identifier!r} has invalid preconditioning or recovery")
        if mode == "cyclic":
            _positive(protocol.get("period_s"), f"protocol {identifier!r} period_s")
            _positive(protocol.get("target_displacement_m"), f"protocol {identifier!r} target_displacement_m")
            if int(protocol.get("cycles", 0)) < 1:
                raise ValueError(f"protocol {identifier!r} cycles must be positive")
        elif mode == "hold_relaxation":
            for field in ("ramp_duration_s", "hold_duration_s", "target_displacement_m"):
                _positive(protocol.get(field), f"protocol {identifier!r} {field}")
        protocol_by_id[identifier] = protocol

    acquisitions = data.get("acquisitions")
    if not isinstance(acquisitions, list) or not acquisitions:
        raise ValueError("acquisitions must be a nonempty list")
    acquisition_ids: set[str] = set()
    root = Path(base) if base is not None else None
    for acquisition in acquisitions:
        identifier = str(acquisition.get("id", ""))
        if not identifier or identifier in acquisition_ids:
            raise ValueError(f"invalid or duplicate acquisition id {identifier!r}")
        acquisition_ids.add(identifier)
        if acquisition.get("protocol_id") not in protocol_by_id:
            raise ValueError(f"acquisition {identifier!r} references an unknown protocol")
        raw_path = _portable_path(acquisition.get("raw_path"), f"acquisition {identifier!r} raw_path")
        if int(acquisition.get("repeat", 0)) < 1:
            raise ValueError(f"acquisition {identifier!r} repeat must be positive")
        channels = acquisition.get("channels", {})
        if not {"time", "displacement", "force"} <= channels.keys():
            raise ValueError(f"acquisition {identifier!r} requires time, displacement, and force channels")
        for channel_name in ("time", "displacement", "force"):
            channel = channels[channel_name]
            if not channel.get("column") or not channel.get("unit"):
                raise ValueError(f"acquisition {identifier!r} has incomplete {channel_name} channel metadata")
        digest = acquisition.get("sha256")
        if digest not in (None, "TO_BE_RECORDED") and (len(str(digest)) != 64 or any(c not in "0123456789abcdef" for c in str(digest))):
            raise ValueError(f"acquisition {identifier!r} sha256 must be lowercase hexadecimal")
        if root is not None and not (root / raw_path).is_file():
            raise ValueError(f"acquisition {identifier!r} raw_path does not exist: {raw_path}")

    splits = data.get("splits", {})
    train = set(splits.get("train_acquisition_ids", []))
    validate = set(splits.get("validate_acquisition_ids", []))
    if not train or not validate:
        raise ValueError("splits must contain nonempty train and validate acquisition IDs")
    if overlap := train & validate:
        raise ValueError(f"train and validate acquisitions overlap: {sorted(overlap)}")
    unknown = (train | validate) - acquisition_ids
    if unknown:
        raise ValueError(f"splits reference unknown acquisitions: {sorted(unknown)}")
    if not splits.get("policy"):
        raise ValueError("splits policy must explain the physical holdout")


def main() -> None:
    """Validate one acquisition manifest from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-files", action="store_true", help="Require every raw_path beside the manifest.")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    validate_acquisition_manifest(data, base=args.manifest.parent if args.check_files else None)
    print(f"valid acquisition manifest: {args.manifest} ({len(data['acquisitions'])} acquisitions)")


if __name__ == "__main__":
    main()
