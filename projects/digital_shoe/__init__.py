# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load and simulate portable Digital Shoe artifacts."""

from importlib import import_module

_EXPORT_MODULES = {
    "ColumnBed": "artifact",
    "DigitalShoe": "artifact",
    "FoundationConfig": "runtime",
    "InstronFixture": "artifact",
    "ShoeMaterial": "runtime",
    "VisualMesh": "artifact",
    "hyperfoam_pressure_numpy": "material",
    "load_artifact": "artifact",
    "maxwell_coefficients_numpy": "material",
    "maxwell_step_numpy": "material",
}

__all__ = list(_EXPORT_MODULES)


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
