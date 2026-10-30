# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load thigh, shank, and foot inertias for the Hogan chain."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SCHEMA = "cartesian_single_leg_1"
_SHAPES = {"masses_kg": (3,), "com_local_m": (3, 2), "inertias_kg_m2": (3,)}


def validate(profile: dict) -> None:
    """Require finite leg inertias and recorded inertial provenance."""
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
        raise ValueError("Unsupported leg profile schema")
    for name, shape in _SHAPES.items():
        array = np.asarray(profile[name])
        if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError(f"{name} must have finite numeric shape {shape}")
        if name != "com_local_m" and np.any(array <= 0):
            raise ValueError(f"{name} must be positive")
    provenance = profile["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    value = provenance.get("inertial")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("provenance.inertial must be a nonempty string")
    json.dumps(dict(provenance), allow_nan=False)


def load(path: str | Path) -> dict:
    """Read validated inertial inputs."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(profile)
    return {"schema": SCHEMA, **{name: profile[name] for name in (*_SHAPES, "provenance")}}
