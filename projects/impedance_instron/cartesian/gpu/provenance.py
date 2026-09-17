# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fail closed on source changes in newly qualified twelve-point pipelines.

Historical reports are evidence, not permission to run changed source. Rebuild
CPU/GPU qualification after a source change; no legacy hash exceptions remain.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_CARTESIAN = "projects/impedance_instron/cartesian/"
_RUNTIME_SOURCES = tuple(
    sorted(
        (
            *(
                _CARTESIAN + name
                for name in (
                    "__init__.py",
                    "data.py",
                    "fit.py",
                    "mechanics.py",
                    "profile.py",
                    "run.py",
                    "trajectory.py",
                    "spline.py",
                    "shoe.py",
                    "gpu/__init__.py",
                    "gpu/engine.py",
                    "gpu/foundation.py",
                    "gpu/mechanics.py",
                    "gpu/objective.py",
                    "gpu/provenance.py",
                )
            ),
            *(
                "projects/digital_shoe/" + name
                for name in (
                    "__init__.py",
                    "artifact.py",
                    "contact.py",
                    "material.py",
                    "provenance.py",
                    "rendering.py",
                    "runtime.py",
                    "friction_parameter_adapter.py",
                    "friction_maxwell.py",
                    "friction_deflection.py",
                    "friction_pressure.py",
                    "friction_slip_history.py",
                    "friction_stribeck.py",
                    "friction_adapter.py",
                    "friction_solver.py",
                    "friction_law.py",
                )
            ),
        )
    )
)


def source_snapshot() -> dict[str, str]:
    """Hash every retained CPU/GPU physics dependency or fail if one is missing."""
    return {path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest() for path in _RUNTIME_SOURCES}


def validate_sources(summary: dict) -> dict:
    """Require exact current source identities for all supplied runtime manifests.

    Args:
        summary: Newly qualified baseline or result with ``source_sha256``.

    Returns:
        Current source hashes and an explicit successful audit.

    Raises:
        ValueError: A manifest is missing, incomplete, or from different source.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("Frozen summary must be a mapping")
    current = source_snapshot()
    for label in ("source_sha256", "original_source_sha256", "physics_reference_source_sha256"):
        if label != "source_sha256" and label not in summary:
            continue
        expected = summary.get(label)
        if not isinstance(expected, Mapping) or set(expected) != set(current):
            raise ValueError(f"Incomplete or historical {label}; rebuild numerical qualification")
        changed = [path for path in current if expected[path] != current[path]]
        if changed:
            raise ValueError(f"Frozen runtime source changed in {label}: {', '.join(changed)}")
    return {
        "schema": "cartesian_gpu_runtime_source_audit_2",
        "validated": True,
        "source_sha256": current,
        "original_physics_reference_checked": "original_source_sha256" in summary,
    }
