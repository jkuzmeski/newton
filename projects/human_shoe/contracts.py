# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validated contract objects for human and shoe integration manifests."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _as_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_positive_float(name: str, value: Any) -> float:
    result = _as_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _as_nonnegative_float(name: str, value: Any) -> float:
    result = _as_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _as_optional_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _as_float(name, value)


def _as_vector3(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _as_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _strict_object(value: Any, *, context: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
    return value


@dataclass(frozen=True, slots=True)
class DigitalInstronGridContract:
    """Column-grid settings [m and dimensionless]."""

    coarse_spacing_m: float
    rearfoot_length_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "coarse_spacing_m", _as_positive_float("coarse_spacing_m", self.coarse_spacing_m))
        fraction = _as_float("rearfoot_length_fraction", self.rearfoot_length_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("rearfoot_length_fraction must be in the interval (0, 1]")
        object.__setattr__(self, "rearfoot_length_fraction", fraction)


@dataclass(frozen=True, slots=True)
class DigitalInstronFitContract:
    """Initial parameter guesses for the material fit."""

    initial_instantaneous_shear_modulus_pa: float
    initial_hyperfoam_exponent: float
    initial_equilibrium_fraction: float
    initial_pasternak_n_per_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_instantaneous_shear_modulus_pa",
            _as_positive_float("initial_instantaneous_shear_modulus_pa", self.initial_instantaneous_shear_modulus_pa),
        )
        object.__setattr__(
            self,
            "initial_hyperfoam_exponent",
            _as_positive_float("initial_hyperfoam_exponent", self.initial_hyperfoam_exponent),
        )
        equilibrium_fraction = _as_float("initial_equilibrium_fraction", self.initial_equilibrium_fraction)
        if not 0.0 <= equilibrium_fraction < 1.0:
            raise ValueError("initial_equilibrium_fraction must be in the interval [0, 1)")
        object.__setattr__(self, "initial_equilibrium_fraction", equilibrium_fraction)
        object.__setattr__(
            self,
            "initial_pasternak_n_per_m",
            _as_positive_float("initial_pasternak_n_per_m", self.initial_pasternak_n_per_m),
        )


@dataclass(frozen=True, slots=True)
class DigitalInstronIndenterContract:
    """Fixture indenter settings [m or deg]."""

    radius_m: float | None = None
    path: str | None = None
    rotation_deg: np.ndarray | None = None
    crop_height_m: float | None = None
    contact_side: str | None = None
    pose_rotation_deg: np.ndarray | None = None
    pose_translation_m: np.ndarray | None = None
    contact_percentile: float | None = None
    height_offset_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius_m", _as_optional_float("radius_m", self.radius_m))
        if self.radius_m is not None and self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if self.path is not None:
            object.__setattr__(self, "path", _as_str("path", self.path))
        if self.rotation_deg is not None:
            object.__setattr__(self, "rotation_deg", _as_vector3("rotation_deg", self.rotation_deg))
        object.__setattr__(self, "crop_height_m", _as_optional_float("crop_height_m", self.crop_height_m))
        if self.crop_height_m is not None and self.crop_height_m <= 0.0:
            raise ValueError("crop_height_m must be positive")
        if self.contact_side is not None:
            object.__setattr__(self, "contact_side", _as_str("contact_side", self.contact_side))
        if self.pose_rotation_deg is not None:
            object.__setattr__(self, "pose_rotation_deg", _as_vector3("pose_rotation_deg", self.pose_rotation_deg))
        if self.pose_translation_m is not None:
            object.__setattr__(self, "pose_translation_m", _as_vector3("pose_translation_m", self.pose_translation_m))
        object.__setattr__(
            self, "contact_percentile", _as_optional_float("contact_percentile", self.contact_percentile)
        )
        if self.contact_percentile is not None and not 0.0 <= self.contact_percentile <= 100.0:
            raise ValueError("contact_percentile must be in the interval [0, 100]")
        object.__setattr__(self, "height_offset_m", _as_optional_float("height_offset_m", self.height_offset_m))


@dataclass(frozen=True, slots=True)
class DigitalInstronTrialContract:
    """A single trial entry from the manifest."""

    name: str
    averaged_cycle_path: str
    fixture: str
    indenter: DigitalInstronIndenterContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_str("name", self.name))
        object.__setattr__(self, "averaged_cycle_path", _as_str("averaged_cycle_path", self.averaged_cycle_path))
        object.__setattr__(self, "fixture", _as_str("fixture", self.fixture))
        if not isinstance(self.indenter, DigitalInstronIndenterContract):
            raise TypeError("indenter must be a DigitalInstronIndenterContract")


@dataclass(frozen=True, slots=True)
class DigitalInstronManifestContract:
    """Top-level project manifest contract."""

    midsole_mesh: str
    cache_dir: str
    grid: DigitalInstronGridContract
    fit: DigitalInstronFitContract
    trials: tuple[DigitalInstronTrialContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "midsole_mesh", _as_str("midsole_mesh", self.midsole_mesh))
        object.__setattr__(self, "cache_dir", _as_str("cache_dir", self.cache_dir))
        if not isinstance(self.grid, DigitalInstronGridContract):
            raise TypeError("grid must be a DigitalInstronGridContract")
        if not isinstance(self.fit, DigitalInstronFitContract):
            raise TypeError("fit must be a DigitalInstronFitContract")
        trials = tuple(self.trials)
        if not trials:
            raise ValueError("trials must not be empty")
        for trial in trials:
            if not isinstance(trial, DigitalInstronTrialContract):
                raise TypeError("trials must contain DigitalInstronTrialContract values")
        object.__setattr__(self, "trials", trials)


@dataclass(frozen=True, slots=True)
class FootShoeAttachmentContract:
    """Shoe pose offset from the OpenSim foot-contact reference [m or deg]."""

    foot_body_name: str
    shoe_carrier_body_name: str
    translation_m: np.ndarray
    rotation_deg: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "foot_body_name", _as_str("foot_body_name", self.foot_body_name))
        object.__setattr__(
            self, "shoe_carrier_body_name", _as_str("shoe_carrier_body_name", self.shoe_carrier_body_name)
        )
        object.__setattr__(self, "translation_m", _as_vector3("translation_m", self.translation_m))
        object.__setattr__(self, "rotation_deg", _as_vector3("rotation_deg", self.rotation_deg))


@dataclass(frozen=True, slots=True)
class HumanShoeExperimentContract:
    """Reproducible human, shoe, and controller experiment definition."""

    schema_version: str
    human_model_path: str
    shoe_manifest_path: str
    controller_id: str
    attachment: FootShoeAttachmentContract
    time_step_s: float
    random_seed: int
    contact_sidecar_path: str | None = None
    motion_path: str | None = None
    initial_motion_frame: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != "human_shoe_experiment_1":
            raise ValueError("schema_version must be human_shoe_experiment_1")
        object.__setattr__(self, "human_model_path", _as_str("human_model_path", self.human_model_path))
        object.__setattr__(self, "shoe_manifest_path", _as_str("shoe_manifest_path", self.shoe_manifest_path))
        object.__setattr__(self, "controller_id", _as_str("controller_id", self.controller_id))
        if not isinstance(self.attachment, FootShoeAttachmentContract):
            raise TypeError("attachment must be a FootShoeAttachmentContract")
        object.__setattr__(self, "time_step_s", _as_positive_float("time_step_s", self.time_step_s))
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if self.contact_sidecar_path is not None:
            object.__setattr__(
                self,
                "contact_sidecar_path",
                _as_str("contact_sidecar_path", self.contact_sidecar_path),
            )
        if self.motion_path is not None:
            object.__setattr__(self, "motion_path", _as_str("motion_path", self.motion_path))
        if (
            not isinstance(self.initial_motion_frame, int)
            or isinstance(self.initial_motion_frame, bool)
            or self.initial_motion_frame < 0
        ):
            raise ValueError("initial_motion_frame must be a nonnegative integer")


def _parse_indenter(value: dict[str, Any]) -> DigitalInstronIndenterContract:
    _strict_object(
        value,
        context="indenter",
        allowed={
            "radius_m",
            "path",
            "rotation_deg",
            "crop_height_m",
            "contact_side",
            "pose_rotation_deg",
            "pose_translation_m",
            "contact_percentile",
            "height_offset_m",
        },
    )
    return DigitalInstronIndenterContract(
        radius_m=value.get("radius_m"),
        path=value.get("path"),
        rotation_deg=value.get("rotation_deg"),
        crop_height_m=value.get("crop_height_m"),
        contact_side=value.get("contact_side"),
        pose_rotation_deg=value.get("pose_rotation_deg"),
        pose_translation_m=value.get("pose_translation_m"),
        contact_percentile=value.get("contact_percentile"),
        height_offset_m=value.get("height_offset_m"),
    )


def load_manifest(path: str | Path) -> DigitalInstronManifestContract:
    """Load and validate a manifest from JSON."""

    data = json.loads(Path(path).read_text())
    _strict_object(
        data,
        context="Digital Instron manifest",
        allowed={"midsole_mesh", "cache_dir", "grid", "fit", "trials", "cycle_windows"},
    )
    _strict_object(
        data["grid"],
        context="grid",
        allowed={"coarse_spacing_m", "rearfoot_length_fraction"},
    )
    _strict_object(
        data["fit"],
        context="fit",
        allowed={
            "initial_instantaneous_shear_modulus_pa",
            "initial_hyperfoam_exponent",
            "initial_equilibrium_fraction",
            "initial_pasternak_n_per_m",
        },
    )
    grid = DigitalInstronGridContract(**data["grid"])
    fit = DigitalInstronFitContract(**data["fit"])
    trial_contracts: list[DigitalInstronTrialContract] = []
    for item in data["trials"]:
        _strict_object(
            item,
            context="trial",
            allowed={"name", "averaged_cycle_path", "fixture", "indenter", "raw_csv_path", "split_prefix"},
        )
        trial_contracts.append(
            DigitalInstronTrialContract(
                name=item["name"],
                averaged_cycle_path=item["averaged_cycle_path"],
                fixture=item["fixture"],
                indenter=_parse_indenter(item["indenter"]),
            )
        )
    return DigitalInstronManifestContract(
        midsole_mesh=data["midsole_mesh"],
        cache_dir=data["cache_dir"],
        grid=grid,
        fit=fit,
        trials=tuple(trial_contracts),
    )


def load_experiment(path: str | Path) -> HumanShoeExperimentContract:
    """Load and validate a human-shoe experiment definition from JSON."""

    data = json.loads(Path(path).read_text())
    _strict_object(
        data,
        context="human-shoe experiment",
        allowed={
            "schema_version",
            "human_model_path",
            "contact_sidecar_path",
            "shoe_manifest_path",
            "controller_id",
            "attachment",
            "time_step_s",
            "random_seed",
            "motion_path",
            "initial_motion_frame",
        },
    )
    _strict_object(
        data["attachment"],
        context="attachment",
        allowed={"foot_body_name", "shoe_carrier_body_name", "translation_m", "rotation_deg"},
    )
    return HumanShoeExperimentContract(
        schema_version=data["schema_version"],
        human_model_path=data["human_model_path"],
        shoe_manifest_path=data["shoe_manifest_path"],
        controller_id=data["controller_id"],
        attachment=FootShoeAttachmentContract(**data["attachment"]),
        time_step_s=data["time_step_s"],
        random_seed=data["random_seed"],
        contact_sidecar_path=data.get("contact_sidecar_path"),
        motion_path=data.get("motion_path"),
        initial_motion_frame=data.get("initial_motion_frame", 0),
    )
