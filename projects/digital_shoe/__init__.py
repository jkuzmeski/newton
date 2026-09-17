# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load and simulate portable Digital Shoe artifacts."""

from importlib import import_module

_EXPORT_MODULES = {
    "CalibrationWorkspace": "calibration",
    "ColumnBed": "artifact",
    "DigitalShoe": "artifact",
    "FoundationConfig": "runtime",
    "FrictionAdapter": "friction_adapter",
    "FrictionParams": "friction_solver",
    "FrictionSolver": "friction_solver",
    "InstronFixture": "artifact",
    "MidsoleFoundation": "runtime",
    "ShoeMaterial": "runtime",
    "SurroundConfig": "runtime",
    "VisualMesh": "artifact",
    "hyperfoam_pressure_numpy": "material",
    "load_artifact": "artifact",
    "maxwell_coefficients_numpy": "material",
    "maxwell_step_numpy": "material",
    "physics_source_identity": "provenance",
}

__all__ = [
    "CalibrationWorkspace",
    "ColumnBed",
    "DigitalShoe",
    "FoundationConfig",
    "FrictionAdapter",
    "FrictionParams",
    "FrictionSolver",
    "InstronFixture",
    "MidsoleFoundation",
    "ShoeMaterial",
    "SurroundConfig",
    "VisualMesh",
    "hyperfoam_pressure_numpy",
    "load_artifact",
    "maxwell_coefficients_numpy",
    "maxwell_step_numpy",
    "physics_source_identity",
]


def __getattr__(name: str) -> object:
    """Resolve a public export without importing unrelated project tools."""
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive package discovery."""
    return sorted(set(globals()) | set(__all__))
