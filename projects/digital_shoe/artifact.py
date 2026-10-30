# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Data contract and loader for ``digital_shoe.json`` shoe artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import ShoeMaterial

SCHEMA_VERSION = "digital_shoe_2"
MODEL_TYPE = "effective_hyperfoam_maxwell_pasternak_foundation"
DEFAULT_SHOE_MASS_KG = 0.2
"""Mass of one shoe [kg] when the artifact omits ``shoe.mass_kg``."""

__all__ = [
    "DEFAULT_SHOE_MASS_KG",
    "MODEL_TYPE",
    "SCHEMA_VERSION",
    "ColumnBed",
    "DigitalShoe",
    "InstronFixture",
    "VisualMesh",
    "load_artifact",
    "validate_artifact",
]


@dataclass(frozen=True)
class ColumnBed:
    """Whole-shoe intrinsic column geometry in the declared shoe frame.

    ``top_bonded`` marks the columns under the bench last footprint, whose tops
    are bonded to the foot; the rim columns outside it are held by shear only.
    """

    anchor_bottom_m: np.ndarray
    rest_length_m: np.ndarray
    area_m2: np.ndarray
    neighbors: np.ndarray
    spacing_m: float
    top_bonded: np.ndarray

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ColumnBed:
        """Decode and validate the whole-shoe column bed."""
        required = {"anchor_bottom_m", "rest_length_m", "area_m2", "neighbors", "spacing_m", "top_bonded"}
        if missing := sorted(required - data.keys()):
            raise ValueError(f"column_bed is missing {missing}")
        bed = cls(
            anchor_bottom_m=np.asarray(data["anchor_bottom_m"], dtype=np.float64),
            rest_length_m=np.asarray(data["rest_length_m"], dtype=np.float64),
            area_m2=np.asarray(data["area_m2"], dtype=np.float64),
            neighbors=np.asarray(data["neighbors"], dtype=np.int32),
            spacing_m=float(data["spacing_m"]),
            top_bonded=np.asarray(data["top_bonded"], dtype=bool),
        )
        bed.validate()
        return bed

    def validate(self) -> None:
        """Reject malformed, nonfinite, or topologically invalid columns."""
        count = len(self.rest_length_m)
        expected = {
            "anchor_bottom_m": (count, 3),
            "area_m2": (count,),
            "neighbors": (count, 4),
            "top_bonded": (count,),
        }
        if count == 0:
            raise ValueError("column_bed has no columns")
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"column_bed {name} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"column_bed {name} contains nonfinite values")
        if not np.all(np.isfinite(self.rest_length_m)) or not np.all(self.rest_length_m > 0.0):
            raise ValueError("column_bed rest lengths must be finite and positive")
        if not np.all(self.area_m2 > 0.0) or not self.spacing_m > 0.0:
            raise ValueError("column_bed areas and spacing must be positive")
        if np.any(self.neighbors >= count) or np.any(self.neighbors < -2):
            raise ValueError("column_bed has invalid neighbor indices")
        row = np.arange(count, dtype=np.int32)[:, None]
        if np.any(self.neighbors == row):
            raise ValueError("column_bed contains a self-neighbor")


@dataclass(frozen=True)
class VisualMesh:
    """Path-independent triangle mesh in the declared shoe frame."""

    vertices_m: np.ndarray
    triangles: np.ndarray

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> VisualMesh:
        """Decode one visual mesh and reject malformed topology."""
        required = {"vertices_m", "triangles"}
        if missing := sorted(required - data.keys()):
            raise ValueError(f"visual mesh {name!r} is missing {missing}")
        mesh = cls(
            vertices_m=np.asarray(data["vertices_m"], dtype=np.float64),
            triangles=np.asarray(data["triangles"], dtype=np.int32),
        )
        mesh.validate(name)
        return mesh

    def validate(self, name: str = "mesh") -> None:
        """Require finite 3D vertices and in-range triangle indices."""
        if self.vertices_m.ndim != 2 or self.vertices_m.shape[1] != 3 or len(self.vertices_m) == 0:
            raise ValueError(f"visual mesh {name!r} has invalid vertices")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3 or len(self.triangles) == 0:
            raise ValueError(f"visual mesh {name!r} has invalid triangles")
        if not np.all(np.isfinite(self.vertices_m)):
            raise ValueError(f"visual mesh {name!r} contains nonfinite vertices")
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.vertices_m)):
            raise ValueError(f"visual mesh {name!r} has out-of-range triangle indices")


@dataclass(frozen=True)
class InstronFixture:
    """Fixture-specific kinematic mapping used only by the Virtual Instron.

    Older punch exports use a separate zero-bottom datum for each column.
    When placing them on the intrinsic shoe bed, translate both the carrier
    anchor and free top by ``bed_bottom - foam_bottom_m``. This preserves the
    fixture gap and vertical shortening without flattening or stretching foam.

    ``bed_bottom_above_platen_m`` is the shoe's pose on the bench: the rest
    bottom height of every whole-shoe column over the platen at first contact,
    in column-bed order. A rocker sole tilts on the platen, so these heights
    are not the shoe-frame bottoms.
    """

    fixture: str
    carrier_anchor_m: np.ndarray
    foam_free_top_m: np.ndarray
    foam_bottom_m: np.ndarray
    rest_length_m: np.ndarray
    area_m2: np.ndarray
    neighbors: np.ndarray
    spacing_m: float
    bed_bottom_above_platen_m: np.ndarray

    @classmethod
    def from_json(cls, fixture: str, data: dict[str, Any]) -> InstronFixture:
        """Decode one optional validation fixture."""
        required = {
            "carrier_anchor_m",
            "foam_free_top_m",
            "foam_bottom_m",
            "rest_length_m",
            "area_m2",
            "neighbors",
            "spacing_m",
            "bed_bottom_above_platen_m",
        }
        if missing := sorted(required - data.keys()):
            raise ValueError(f"fixture {fixture!r} is missing {missing}")
        value = cls(
            fixture=fixture,
            carrier_anchor_m=np.asarray(data["carrier_anchor_m"], dtype=np.float64),
            foam_free_top_m=np.asarray(data["foam_free_top_m"], dtype=np.float64),
            foam_bottom_m=np.asarray(data["foam_bottom_m"], dtype=np.float64),
            rest_length_m=np.asarray(data["rest_length_m"], dtype=np.float64),
            area_m2=np.asarray(data["area_m2"], dtype=np.float64),
            neighbors=np.asarray(data["neighbors"], dtype=np.int32),
            spacing_m=float(data["spacing_m"]),
            bed_bottom_above_platen_m=np.asarray(data["bed_bottom_above_platen_m"], dtype=np.float64),
        )
        value.validate()
        return value

    def validate(self) -> None:
        """Enforce equal fixture-array lengths and valid neighbors."""
        count = len(self.rest_length_m)
        expected = {
            "carrier_anchor_m": (count, 3),
            "foam_free_top_m": (count,),
            "foam_bottom_m": (count,),
            "area_m2": (count,),
            "neighbors": (count, 4),
        }
        if count == 0:
            raise ValueError(f"fixture {self.fixture!r} has no columns")
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"fixture {self.fixture!r} has invalid {name}")
        if not np.all(self.rest_length_m > 0.0) or not np.all(self.area_m2 > 0.0) or not self.spacing_m > 0.0:
            raise ValueError(f"fixture {self.fixture!r} has nonpositive geometry")
        if np.any(self.neighbors >= count) or np.any(self.neighbors < -2):
            raise ValueError(f"fixture {self.fixture!r} has invalid neighbor indices")
        heights = self.bed_bottom_above_platen_m
        if heights.ndim != 1 or not np.all(np.isfinite(heights)) or not np.all(heights >= 0.0):
            raise ValueError(f"fixture {self.fixture!r} has invalid bed_bottom_above_platen_m")


@dataclass(frozen=True)
class DigitalShoe:
    """Loaded runtime shoe, optional fixture mappings, and validation record."""

    shoe_id: str
    mass_kg: float
    """Mass of one shoe [kg], from ``shoe.mass_kg`` or :data:`DEFAULT_SHOE_MASS_KG`."""
    material: ShoeMaterial
    column_bed: ColumnBed
    visual_meshes: dict[str, VisualMesh]
    instron_fixtures: dict[str, InstronFixture]
    validation: dict[str, Any]
    provenance: dict[str, Any]
    raw: dict[str, Any]

    def visual_mesh(self, name: str) -> VisualMesh:
        """Return a baked visual mesh or list the available names in the error."""
        if name not in self.visual_meshes:
            choices = ", ".join(sorted(self.visual_meshes))
            raise KeyError(f"unknown visual mesh {name!r}; available meshes: {choices}")
        return self.visual_meshes[name]

    def instron_fixture(self, fixture: str = "fullfoot_last") -> InstronFixture:
        """Return a fixture or list the available names in the error."""
        if fixture not in self.instron_fixtures:
            choices = ", ".join(sorted(self.instron_fixtures))
            raise KeyError(f"unknown Instron fixture {fixture!r}; available fixtures: {choices}")
        return self.instron_fixtures[fixture]


def validate_artifact(data: dict[str, Any]) -> None:
    """Fail closed on an unknown or incomplete artifact contract."""
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {data.get('schema_version')!r}; expected {SCHEMA_VERSION!r}")
    required = {"shoe", "coordinate_system", "constitutive_model", "column_bed", "validation", "provenance"}
    if missing := sorted(required - data.keys()):
        raise ValueError(f"artifact is missing {missing}")
    model = data["constitutive_model"]
    if model.get("type") != MODEL_TYPE:
        raise ValueError(f"unsupported constitutive model {model.get('type')!r}")
    ShoeMaterial(**model["parameters"])
    mass_kg = data["shoe"].get("mass_kg", DEFAULT_SHOE_MASS_KG)
    if not np.isfinite(mass_kg) or not mass_kg > 0.0:
        raise ValueError(f"shoe mass_kg must be finite and positive, got {mass_kg!r}")
    bed = ColumnBed.from_json(data["column_bed"])
    for name, mesh_data in data.get("visual_meshes", {}).items():
        VisualMesh.from_json(name, mesh_data)
    for fixture, fixture_data in data.get("instron_fixtures", {}).items():
        if len(InstronFixture.from_json(fixture, fixture_data).bed_bottom_above_platen_m) != len(bed.rest_length_m):
            raise ValueError(f"fixture {fixture!r} needs one bed_bottom_above_platen_m per column_bed column")
    coordinate = data["coordinate_system"]
    if coordinate.get("length_unit") != "m" or coordinate.get("up_axis") != "+Z":
        raise ValueError("runtime currently requires metres and a +Z ground normal")


def load_artifact(path: str | Path) -> DigitalShoe:
    """Load one self-contained artifact independently of the current directory."""
    data = json.loads(Path(path).read_text())
    validate_artifact(data)
    return DigitalShoe(
        shoe_id=data["shoe"]["id"],
        mass_kg=float(data["shoe"].get("mass_kg", DEFAULT_SHOE_MASS_KG)),
        material=ShoeMaterial(**data["constitutive_model"]["parameters"]),
        column_bed=ColumnBed.from_json(data["column_bed"]),
        visual_meshes={
            name: VisualMesh.from_json(name, value) for name, value in data.get("visual_meshes", {}).items()
        },
        instron_fixtures={
            fixture: InstronFixture.from_json(fixture, value)
            for fixture, value in data.get("instron_fixtures", {}).items()
        },
        validation=data["validation"],
        provenance=data["provenance"],
        raw=data,
    )
