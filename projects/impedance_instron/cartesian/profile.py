# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate plain-dictionary parameters for one Cartesian-actuated leg.

Body order is thigh, shank, foot. Only these three masses contribute gravity
or inertia. There is no trunk, pelvis body, opposite leg, or hidden upper-body
load. Local positive x points proximal-to-distal in the thigh and shank and
heel-to-toe in the foot; local coordinates are planar [x, z].

Actuator order is hip forward force, hip upward force, knee torque, ankle
torque. Equilibrium positions have units [m, m, rad, rad], their rates
[m/s, m/s, rad/s, rad/s], and accelerations [m/s^2, m/s^2, rad/s^2, rad/s^2].
Hip stiffness [N/m] and damping [N s/m] must be supplied explicitly, never
converted from a removed hip rotational actuator. Knee/ankle stiffness and
damping have units [N m/rad] and [N m s/rad]. Bounds describe an engineering
search envelope, not measured physiological limits or a force saturation law.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SCHEMA = "cartesian_single_leg_1"
BODY_NAMES = ("thigh", "shank", "foot")
JOINT_NAMES = ("knee", "ankle")
_SHAPES = {
    "masses_kg": (3,),
    "com_local_m": (3, 2),
    "inertias_kg_m2": (3,),
    "hip_stiffness_n_m": (2,),
    "hip_damping_ns_m": (2,),
    "joint_stiffness_nm_rad": (2,),
    "joint_damping_nms_rad": (2,),
    "equilibrium_lower": (4,),
    "equilibrium_upper": (4,),
    "equilibrium_rate_limit": (4,),
    "equilibrium_acceleration_limit": (4,),
    "joint_lower_rad": (2,),
    "joint_upper_rad": (2,),
}


def _array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Require finite numeric values without coercing strings or booleans."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must have numeric shape {shape}") from error
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite numeric shape {shape}")
    return array.astype(float, copy=False)


def validate(profile: dict) -> None:
    """Validate a profile without changing its lists or NumPy arrays.

    Args:
        profile: Plain mapping with three body inertial values, two Cartesian
            gains, two joint gains, and four-channel equilibrium limits.
            Required provenance entries ``inertial``, ``impedance``, and
            ``limits`` must be nonempty strings. Other provenance entries may
            contain JSON-compatible context. Unknown top-level fields are
            rejected to prevent ignored mechanical parameters.

    Raises:
        ValueError: If fields, shapes, numeric values, or bounds are invalid.
    """
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be a mapping")
    required = {*_SHAPES, "provenance"}
    missing = required - profile.keys()
    if missing:
        raise ValueError(f"Missing profile fields: {', '.join(sorted(missing))}")
    unknown = profile.keys() - required - {"schema"}
    if unknown:
        raise ValueError(f"Unsupported profile fields: {', '.join(sorted(unknown))}")
    if profile.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("Unsupported Cartesian profile schema")
    arrays = {name: _array(profile[name], name, shape) for name, shape in _SHAPES.items()}
    for name in (
        "masses_kg",
        "inertias_kg_m2",
        "hip_stiffness_n_m",
        "joint_stiffness_nm_rad",
        "equilibrium_rate_limit",
        "equilibrium_acceleration_limit",
    ):
        if np.any(arrays[name] <= 0):
            raise ValueError(f"{name} must be positive")
    for name in ("hip_damping_ns_m", "joint_damping_nms_rad"):
        if np.any(arrays[name] < 0):
            raise ValueError(f"{name} must be nonnegative")
    if np.any(arrays["joint_lower_rad"] >= arrays["joint_upper_rad"]):
        raise ValueError("joint_lower_rad must be strictly below joint_upper_rad")
    if arrays["joint_upper_rad"][0] > 0:
        raise ValueError("Knee flexion is negative; knee joint_upper_rad must not exceed zero")
    if np.any(arrays["equilibrium_lower"] > arrays["equilibrium_upper"]):
        raise ValueError("equilibrium_lower must not exceed equilibrium_upper")
    provenance = profile["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    for name in ("inertial", "impedance", "limits"):
        value = provenance.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"provenance.{name} must be a nonempty string")
    try:
        json.dumps(dict(provenance), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must contain finite JSON-compatible context") from error


def load(path: str | Path) -> dict:
    """Load a validated JSON profile as a plain dictionary.

    Args:
        path: Input JSON path.

    Returns:
        A plain dictionary whose numeric lists support ``np.asarray``.
    """
    with Path(path).open(encoding="utf-8") as stream:
        profile = json.load(stream)
    validate(profile)
    return profile
