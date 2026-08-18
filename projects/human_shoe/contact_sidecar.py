# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load and apply human-shoe contact sidecar manifests.

The sidecar format augments an existing OpenSim model with contact geometry
descriptions that can be injected into a parsed :class:`newton.opensim.OsimModel`
or written to a reproducible derived OpenSim document.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from newton import opensim


def _as_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _as_vec3(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _as_optional_vec3(name: str, value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return _as_vec3(name, value)


def _as_optional_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_optional_name(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _as_str(name, value)


def _strict_object(value: Any, *, context: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
    return value


@dataclass(frozen=True, slots=True)
class ContactGeometrySidecarContract:
    """A single sidecar contact geometry description [m or deg]."""

    name: str
    type: str
    body_name: str
    location_m: np.ndarray | None
    support_marker_name: str | None
    support_offset_m: np.ndarray
    orientation_deg: np.ndarray
    radius_m: float | None
    mesh_file: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_str("name", self.name))
        object.__setattr__(self, "type", _as_str("type", self.type))
        object.__setattr__(self, "body_name", _as_str("body_name", self.body_name))
        object.__setattr__(self, "location_m", _as_optional_vec3("location_m", self.location_m))
        object.__setattr__(
            self, "support_marker_name", _as_optional_name("support_marker_name", self.support_marker_name)
        )
        object.__setattr__(self, "support_offset_m", _as_vec3("support_offset_m", self.support_offset_m))
        object.__setattr__(self, "orientation_deg", _as_vec3("orientation_deg", self.orientation_deg))
        object.__setattr__(self, "radius_m", _as_optional_float("radius_m", self.radius_m))
        if self.radius_m is not None and self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        object.__setattr__(self, "mesh_file", _as_optional_name("mesh_file", self.mesh_file))
        if self.type == "ContactSphere":
            if self.mesh_file is not None:
                raise ValueError("ContactSphere must not define mesh_file")
            if (self.location_m is None) == (self.support_marker_name is None):
                raise ValueError("ContactSphere must define exactly one of location_m or support_marker_name")
            if self.radius_m is None:
                raise ValueError("ContactSphere must define radius_m")
        elif self.type == "ContactMesh":
            if self.location_m is None:
                raise ValueError("ContactMesh must define location_m")
            if self.radius_m is not None or self.support_marker_name is not None:
                raise ValueError("ContactMesh must not define radius_m or support_marker_name")
            if self.mesh_file is None:
                raise ValueError("ContactMesh must define mesh_file")
        else:
            raise ValueError(f"unsupported contact geometry type '{self.type}'")


@dataclass(frozen=True, slots=True)
class HumanShoeContactSidecarContract:
    """Top-level contact sidecar manifest contract."""

    schema_version: str
    source_model_path: str
    generated_model_path: str
    generated_model_name: str | None
    contacts: tuple[ContactGeometrySidecarContract, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "human_shoe_contact_sidecar_1":
            raise ValueError("schema_version must be human_shoe_contact_sidecar_1")
        object.__setattr__(self, "source_model_path", _as_str("source_model_path", self.source_model_path))
        object.__setattr__(self, "generated_model_path", _as_str("generated_model_path", self.generated_model_path))
        object.__setattr__(
            self, "generated_model_name", _as_optional_name("generated_model_name", self.generated_model_name)
        )
        contacts = tuple(self.contacts)
        if not contacts:
            raise ValueError("contacts must not be empty")
        for contact in contacts:
            if not isinstance(contact, ContactGeometrySidecarContract):
                raise TypeError("contacts must contain ContactGeometrySidecarContract values")
        names = [contact.name for contact in contacts]
        if len(names) != len(set(names)):
            raise ValueError("contacts must have unique names")
        object.__setattr__(self, "contacts", contacts)


def _parse_contact(contract: dict[str, Any]) -> ContactGeometrySidecarContract:
    _strict_object(
        contract,
        context="contact",
        allowed={
            "name",
            "type",
            "body_name",
            "location_m",
            "support_marker_name",
            "support_offset_m",
            "orientation_deg",
            "radius_m",
            "mesh_file",
        },
    )
    return ContactGeometrySidecarContract(
        name=contract["name"],
        type=contract["type"],
        body_name=contract["body_name"],
        location_m=contract.get("location_m"),
        support_marker_name=contract.get("support_marker_name"),
        support_offset_m=contract.get("support_offset_m", [0.0, 0.0, 0.0]),
        orientation_deg=contract.get("orientation_deg", [0.0, 0.0, 0.0]),
        radius_m=contract.get("radius_m"),
        mesh_file=contract.get("mesh_file"),
    )


def load_contact_sidecar(path: str | Path) -> HumanShoeContactSidecarContract:
    """Load a validated contact sidecar manifest from JSON."""

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    _strict_object(
        data,
        context="sidecar",
        allowed={"schema_version", "source_model_path", "generated_model_path", "generated_model_name", "contacts"},
    )
    if data["schema_version"] != "human_shoe_contact_sidecar_1":
        raise ValueError("schema_version must be human_shoe_contact_sidecar_1")
    contacts = tuple(_parse_contact(contact) for contact in data["contacts"])
    return HumanShoeContactSidecarContract(
        schema_version=data["schema_version"],
        source_model_path=data["source_model_path"],
        generated_model_path=data["generated_model_path"],
        generated_model_name=data.get("generated_model_name"),
        contacts=contacts,
    )


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sphere_location(contact: ContactGeometrySidecarContract, model: opensim.OsimModel) -> tuple[float, float, float]:
    if contact.location_m is not None:
        return tuple(float(x) for x in contact.location_m)
    assert contact.support_marker_name is not None
    marker = next((m for m in model.markers if m.name == contact.support_marker_name), None)
    if marker is None:
        raise KeyError(f"marker '{contact.support_marker_name}' not found")
    support = np.asarray(marker.location, dtype=np.float64) + contact.support_offset_m
    center = support + np.array([0.0, float(contact.radius_m), 0.0], dtype=np.float64)
    return tuple(float(x) for x in center)


def inject_contact_sidecar(
    model: opensim.OsimModel,
    sidecar: HumanShoeContactSidecarContract,
    *,
    replace_existing: bool = False,
) -> opensim.OsimModel:
    """Inject sidecar contact geometry into a deep-copied OpenSim model."""

    injected = copy.deepcopy(model)
    body_names = {body.name for body in injected.bodies}
    contact_by_name = {contact.name: contact for contact in injected.contact_geometry}

    for contact in sidecar.contacts:
        if contact.body_name not in body_names:
            raise KeyError(f"body '{contact.body_name}' not found")
        if contact.type == "ContactSphere":
            if contact.radius_m is None:
                raise ValueError("ContactSphere must define radius_m")
            if not np.isfinite(contact.radius_m) or contact.radius_m <= 0.0:
                raise ValueError("radius_m must be positive and finite")
            if contact.support_marker_name is not None:
                marker = next((m for m in injected.markers if m.name == contact.support_marker_name), None)
                if marker is None:
                    raise KeyError(f"marker '{contact.support_marker_name}' not found")
                if marker.body != contact.body_name:
                    raise ValueError(
                        f"marker '{contact.support_marker_name}' is attached to body '{marker.body}', not '{contact.body_name}'"
                    )
            location = _sphere_location(contact, injected)
            geometry = opensim.OsimContactGeometry(
                name=contact.name,
                type="ContactSphere",
                body=contact.body_name,
                location=location,
                orientation=tuple(np.deg2rad(contact.orientation_deg).tolist()),
                radius=float(contact.radius_m),
            )
        else:
            if contact.location_m is None or contact.mesh_file is None:
                raise ValueError("ContactMesh must define location_m and mesh_file")
            geometry = opensim.OsimContactGeometry(
                name=contact.name,
                type="ContactMesh",
                body=contact.body_name,
                location=tuple(float(x) for x in contact.location_m),
                orientation=tuple(np.deg2rad(contact.orientation_deg).tolist()),
                mesh_file=contact.mesh_file,
            )
        if contact.name in contact_by_name:
            if not replace_existing:
                raise ValueError(f"contact geometry '{contact.name}' already exists")
            injected.contact_geometry = [cg for cg in injected.contact_geometry if cg.name != contact.name]
        injected.contact_geometry.append(geometry)

    if sidecar.generated_model_name is not None:
        injected.name = sidecar.generated_model_name
    return injected


def write_contact_augmented_osim(
    sidecar_path: str | Path,
    output_path: str | Path | None = None,
    *,
    replace_existing: bool = False,
) -> Path:
    """Write an OpenSim model augmented with sidecar contact geometry."""

    sidecar_path = Path(sidecar_path)
    sidecar = load_contact_sidecar(sidecar_path)
    base_dir = sidecar_path.parent
    source_model_path = _resolve_path(base_dir, sidecar.source_model_path)
    generated_model_path = _resolve_path(base_dir, sidecar.generated_model_path)
    if output_path is None:
        output_path = generated_model_path
    else:
        output_path = Path(output_path)
    model = opensim.parse_osim(source_model_path)
    injected = inject_contact_sidecar(model, sidecar, replace_existing=replace_existing)
    # The public writer emits OpenSim 4.x sockets and offset frames even when
    # the source was a legacy document, so the derived document must advertise
    # the matching schema version.
    injected.version = 40000
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opensim.write_osim(injected, output_path)
    reparsed = opensim.parse_osim(output_path)
    expected = {contact.name for contact in sidecar.contacts}
    actual = {contact.name for contact in reparsed.contact_geometry}
    if not expected.issubset(actual):
        raise ValueError("written model is missing expected contact geometry")
    if len(reparsed.contact_forces) != len(model.contact_forces):
        raise ValueError("written model changed the contact-force count")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, help="Path to the contact sidecar JSON file")
    parser.add_argument("--output", help="Optional output .osim path")
    parser.add_argument("--replace-existing", action="store_true", help="Replace contact geometries by name")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the contact sidecar writer CLI."""

    args = _build_arg_parser().parse_args(argv)
    output = write_contact_augmented_osim(
        args.sidecar,
        output_path=args.output,
        replace_existing=args.replace_existing,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
