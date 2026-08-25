# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pinned OpenSim 3-D-walking contact reference for S001 Trial 101.

This module adapts the contact topology in OpenSim's ``example3DWalking`` to
S001's accepted RRA model and prescribed motion.  It does not run Moco or any
optimization.  The measured ``ExternalLoads`` file is a tracking and validation
reference only.  It is never added to the predictive model's dynamics.

All public array interfaces use the OpenSim ground frame (x forward, y up,
z right) and SI units.  A wrench is ``[force, point, torque_at_point]``.  Foot
resultants use the ground origin as their point so forces from the calcaneus and
toe bodies can be combined without losing the moment-arm term.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import newton.opensim as newton_osim

ARCHITECTURE_ROLE = "cross_runtime_oracle"

_SCHEMA = "gait_c3d_opensim_moco_contact_reference_1"
_PINNED_COMMIT = "11036b39ca7232c604685b37f483afafc056d92b"
_RAW_ROOT = f"https://raw.githubusercontent.com/opensim-org/opensim-core/{_PINNED_COMMIT}"
_SOURCE_FILES = {
    "OpenSim/Examples/Moco/example3DWalking/subject_walk_scaled_ContactGeometrySet.xml": "b64f1f1c7ebc9da5008c14896dc877f16b958505d8c4adbb29f86bdd3f1be47f",
    "OpenSim/Examples/Moco/example3DWalking/subject_walk_scaled_ContactForceSet.xml": "659d571d9f731c9e119d93371edcd46c7fc265796ef0681670d46bd875b1b54d",
    "OpenSim/Examples/Moco/example3DWalking/example3DWalking.cpp": "a6512ebe7ca4e54fb567f24c814a72803c6bb16c6ec72e2135b0ded47def283c",
    "Bindings/Python/examples/Moco/example3DWalking/example3DWalking.py": "5a4d7ee014c91ce0b09453f49b0ce33da6b1296e5a23801772d2a3b9fd1ca5e2",
    "OpenSim/Moco/MocoGoal/MocoContactTrackingGoal.cpp": "ed1e318529d54f2e4addaaa138c570eb9d893d2ae6b5e624096607859481fb69",
    "OpenSim/Moco/MocoGoal/MocoContactTrackingGoal.h": "1a41e08d1e758f53baba45b9cc00c78b2ffdd884b5c6773851036444c7f87a01",
}
_FRAME = "opensim_x_forward_y_up_z_right"
_UNITS = {"length": "m", "force": "N", "moment": "N*m", "time": "s"}
_SIDE_ORDER = ("left", "right")
_SUFFIX = {"left": "l", "right": "r"}
_BODY_ORDER = ("calcn_l", "toes_l", "calcn_r", "toes_r")
_ROLE_ORDER = (
    "heel",
    "lateralRearfoot",
    "lateralMidfoot",
    "medialMidfoot",
    "lateralToe",
    "medialToe",
)
_ROLE_BODY = {
    "heel": "calcn",
    "lateralRearfoot": "calcn",
    "lateralMidfoot": "calcn",
    "medialMidfoot": "calcn",
    "lateralToe": "toes",
    "medialToe": "toes",
}
# Right-foot x/z/radius values copied from the pinned geometry XML.  Left z is
# the exact mirror.  The upstream XML has local y == 0 for every sphere.
_RIGHT_TOPOLOGY = {
    "heel": (0.0146421, -0.0122799, 0.035),
    "lateralRearfoot": (0.0849291, 0.0338649, 0.035),
    "lateralMidfoot": (0.153362, 0.0594829, 0.035),
    "medialMidfoot": (0.203637, -0.0398688, 0.035),
    "lateralToe": (0.023014, 0.0670461, 0.035),
    "medialToe": (0.074363, -0.026039, 0.035),
}
_MATERIAL = {
    "stiffness": 1.0e6,
    "dissipation": 2.0,
    "static_friction": 0.8,
    "dynamic_friction": 0.8,
    "viscous_friction": 0.5,
    "transition_velocity": 0.2,
    "hertz_smoothing": 300.0,
    "hunt_crossley_smoothing": 50.0,
}
# This is the OpenSim default omitted by the pinned ForceSet XML.  State it in
# the Newton spec so the two implementations have an explicit common value.
_NEWTON_MATERIAL = {**_MATERIAL, "constant_contact_force": 1.0e-5}
_ALIGNMENT_BOUNDS_M = (-0.03, 0.03)
_OFFICIAL_EXAMPLE_OFFSET_M = 0.02
_DEFAULT_DATA = Path("/home/jo31399/newton-data/gait/processed/trial_101")
_DEFAULT_RRA_MODEL = (
    _DEFAULT_DATA / "opensim_rra_official_reference_fy4/results/trial101_official_opensim_rra_fy4_adjusted.osim"
)
_DEFAULT_RRA_Q = (
    _DEFAULT_DATA / "opensim_rra_official_reference_fy4/results/trial101_official_opensim_rra_fy4_Kinematics_q.sto"
)
_DEFAULT_RRA_U = (
    _DEFAULT_DATA / "opensim_rra_official_reference_fy4/results/trial101_official_opensim_rra_fy4_Kinematics_u.sto"
)
_DEFAULT_EXTERNAL_LOADS = _DEFAULT_DATA / "latest/trial_grf_context.xml"
_S001_ALIGNMENT_SOURCES = {
    "rra_adjusted_model": "7483c006cfc68c03d90bf6d35870f2b3cecf41dfa6a6145ff8a6b97e7a51b7b0",
    "rra_prescribed_coordinates": "2b5e6b50c5e2c2c1ca1402c34292f1420c2f8836b5dee09f7dfeddd229a84f4c",
    "rra_prescribed_speeds": "7020f1455e870a697cea766063365ffc490262299f520eb3c332d178115c035a",
    "measured_external_loads": "d63b1c6866426b380d0167375cfcaf7f91543f681187e58fc9e1923c02127078",
    "measured_external_loads_data": "0414071e51edd5fcf6c9e7db8a69a9267b4afdea02b61502eb4217520d175a65",
}
_RECORD_SUFFIXES = (
    "Sphere.force.X",
    "Sphere.force.Y",
    "Sphere.force.Z",
    "Sphere.torque.X",
    "Sphere.torque.Y",
    "Sphere.torque.Z",
    "HalfSpace.force.X",
    "HalfSpace.force.Y",
    "HalfSpace.force.Z",
    "HalfSpace.torque.X",
    "HalfSpace.torque.Y",
    "HalfSpace.torque.Z",
)


@dataclass(frozen=True, slots=True)
class VerticalAlignment:
    """Archived scalar local-y shift derived from prescribed stance."""

    offset_m: float
    unconstrained_offset_m: float
    bounds_m: tuple[float, float]
    stance_observation_count: int
    rms_clearance_before_m: float
    rms_clearance_after_m: float
    max_abs_clearance_after_m: float
    method: str = "least_squares_zero_lowest_sphere_clearance_per_loaded_foot_frame"
    measured_load_threshold_n: float = 50.0


# Each observation was the lowest of the six sphere clearances for one loaded
# foot/frame (measured vertical force >= 50 N).  The response to a shared local-y
# shift was evaluated in official OpenSim on the accepted RRA model.  This S001
# result is intentionally not the example's ad-hoc +2 cm shift.
S001_ALIGNMENT = VerticalAlignment(
    offset_m=-0.020713880614172762,
    unconstrained_offset_m=-0.020713880614172762,
    bounds_m=_ALIGNMENT_BOUNDS_M,
    stance_observation_count=1343,
    rms_clearance_before_m=0.023555019377173688,
    rms_clearance_after_m=0.012960839452307957,
    max_abs_clearance_after_m=0.04279222961413137,
)


@dataclass(frozen=True, slots=True)
class SphereSpec:
    """One pinned sphere after applying the S001 local-y alignment."""

    side: str
    role: str
    name: str
    force_name: str
    body: str
    location_m: tuple[float, float, float]
    radius_m: float


@dataclass(frozen=True, slots=True)
class MocoContactGroupSpec:
    """A force group consumed by ``MocoContactTrackingGoal``."""

    side: str
    contact_force_paths: tuple[str, ...]
    external_force_name: str
    applied_to_body: str
    alternative_frame_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContactEvaluation:
    """Prescribed-motion body and foot wrenches in the OpenSim ground frame."""

    body_names: tuple[str, ...]
    body_wrenches: np.ndarray
    foot_names: tuple[str, ...]
    foot_wrenches: np.ndarray


@dataclass(frozen=True, slots=True)
class NewtonAugmentationSpec:
    """Actual Newton IR objects to append to a parsed ``OsimModel``."""

    contact_geometry: tuple[Any, ...]
    contact_forces: tuple[Any, ...]


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_text(value: float) -> str:
    return format(_finite(value, "XML scalar"), ".17g")


def _vec_text(value: Sequence[float]) -> str:
    if len(value) != 3:
        raise ValueError("vector must contain three values")
    return " ".join(_float_text(item) for item in value)


def _element(parent: ET.Element, tag: str, text: str | float | None = None, **attributes: str) -> ET.Element:
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        child.text = _float_text(text) if isinstance(text, float) else text
    return child


def xml_bytes(root: ET.Element) -> bytes:
    """Serialize generated OpenSim XML deterministically."""
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True) + b"\n"


def sphere_specs(alignment: VerticalAlignment = S001_ALIGNMENT) -> tuple[SphereSpec, ...]:
    """Return six official-topology spheres per foot in fixed group order."""
    offset = _finite(alignment.offset_m, "alignment offset")
    low, high = alignment.bounds_m
    if not low <= offset <= high:
        raise ValueError("alignment offset is outside its archived bounds")
    result: list[SphereSpec] = []
    for side in _SIDE_ORDER:
        suffix = _SUFFIX[side]
        z_sign = -1.0 if side == "left" else 1.0
        for role in _ROLE_ORDER:
            x, right_z, radius = _RIGHT_TOPOLOGY[role]
            body = f"{_ROLE_BODY[role]}_{suffix}"
            result.append(
                SphereSpec(
                    side=side,
                    role=role,
                    name=f"{role}_{suffix}",
                    force_name=f"contact{role[0].upper()}{role[1:]}_{suffix}",
                    body=body,
                    location_m=(x, offset, z_sign * right_z),
                    radius_m=radius,
                )
            )
    return tuple(result)


def _validate_sphere_specs(spheres: Sequence[SphereSpec]) -> tuple[SphereSpec, ...]:
    """Require calibrated spheres to preserve the pinned topology contract."""
    values = tuple(spheres)
    defaults = sphere_specs()
    if len(values) != len(defaults):
        raise ValueError("contact topology must contain exactly 12 spheres")
    for value, default in zip(values, defaults, strict=True):
        if not isinstance(value, SphereSpec):
            raise TypeError("contact topology entries must be SphereSpec values")
        if (value.side, value.role, value.name, value.force_name, value.body) != (
            default.side,
            default.role,
            default.name,
            default.force_name,
            default.body,
        ):
            raise ValueError("calibrated contact changed pinned sphere identity, order, or body")
        if not np.all(np.isfinite(value.location_m)) or not math.isfinite(value.radius_m) or value.radius_m <= 0.0:
            raise ValueError("calibrated contact sphere geometry must be finite and positive")
    return values


def derive_vertical_alignment(
    world_center_y_m: np.ndarray,
    local_y_world_gain: np.ndarray,
    stance_mask: np.ndarray,
    *,
    radius_m: float = 0.035,
    ground_height_m: float = 0.0,
    bounds_m: tuple[float, float] = _ALIGNMENT_BOUNDS_M,
    measured_load_threshold_n: float = 50.0,
) -> VerticalAlignment:
    """Fit one bounded local-y shift from prescribed stance observations.

    Inputs have shape ``[frame, side, sphere]`` and ``stance_mask`` has shape
    ``[frame, side]``.  For each loaded foot/frame, the sphere with the lowest
    unshifted surface clearance is selected.  Official frame kinematics make
    ``local_y_world_gain`` the ground-y change per metre of body-local y shift.
    The returned shift minimizes squared selected clearances and is clipped to
    the declared bounds.
    """
    center = np.asarray(world_center_y_m, dtype=float)
    gain = np.asarray(local_y_world_gain, dtype=float)
    mask = np.asarray(stance_mask, dtype=bool)
    if center.ndim != 3 or center.shape != gain.shape or mask.shape != center.shape[:2]:
        raise ValueError("alignment arrays must have shapes [frame, side, sphere] and [frame, side]")
    if center.shape[1:] != (2, 6):
        raise ValueError("alignment arrays must describe two feet and six spheres per foot")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(gain)):
        raise ValueError("alignment observations must be finite")
    if not np.any(mask):
        raise ValueError("alignment requires at least one stance observation")
    low, high = (float(bounds_m[0]), float(bounds_m[1]))
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("alignment bounds must be a finite increasing pair")
    clearance = center - _finite(radius_m, "radius_m") - _finite(ground_height_m, "ground_height_m")
    selected_clearance: list[float] = []
    selected_gain: list[float] = []
    for frame, side in zip(*np.nonzero(mask), strict=True):
        sphere = int(np.argmin(clearance[frame, side]))
        selected_clearance.append(float(clearance[frame, side, sphere]))
        selected_gain.append(float(gain[frame, side, sphere]))
    values = np.asarray(selected_clearance)
    gains = np.asarray(selected_gain)
    denominator = float(np.dot(gains, gains))
    if denominator <= np.finfo(float).eps:
        raise ValueError("alignment observations have zero vertical sensitivity")
    unconstrained = -float(np.dot(gains, values)) / denominator
    offset = float(np.clip(unconstrained, low, high))
    residual = values + gains * offset
    return VerticalAlignment(
        offset_m=offset,
        unconstrained_offset_m=unconstrained,
        bounds_m=(low, high),
        stance_observation_count=len(values),
        rms_clearance_before_m=float(np.sqrt(np.mean(np.square(values)))),
        rms_clearance_after_m=float(np.sqrt(np.mean(np.square(residual)))),
        max_abs_clearance_after_m=float(np.max(np.abs(residual))),
        measured_load_threshold_n=_finite(measured_load_threshold_n, "measured_load_threshold_n"),
    )


def build_contact_geometry_xml(
    alignment: VerticalAlignment = S001_ALIGNMENT,
    *,
    spheres: Sequence[SphereSpec] | None = None,
) -> ET.Element:
    """Build an official ``ContactGeometrySet`` with the pinned topology.

    Args:
        alignment: Default S001 vertical alignment used when ``spheres`` is not supplied.
        spheres: Optional calibrated spheres that retain the pinned names, bodies, and order.
    """
    sphere_values = sphere_specs(alignment) if spheres is None else tuple(spheres)
    _validate_sphere_specs(sphere_values)
    document = ET.Element("OpenSimDocument", {"Version": "40600"})
    geometry_set = _element(document, "ContactGeometrySet", None, name="contactgeometryset")
    objects = _element(geometry_set, "objects")
    floor = _element(objects, "ContactHalfSpace", None, name="floor")
    _element(floor, "socket_frame", "/ground")
    _element(floor, "location", "0 0 0")
    _element(floor, "orientation", _vec_text((0.0, 0.0, -0.5 * math.pi)))
    for sphere in sphere_values:
        item = _element(objects, "ContactSphere", None, name=sphere.name)
        _element(item, "socket_frame", f"/bodyset/{sphere.body}")
        _element(item, "location", _vec_text(sphere.location_m))
        _element(item, "orientation", "0 0 0")
        _element(item, "radius", sphere.radius_m)
    _element(geometry_set, "groups")
    return document


def build_force_xml(
    alignment: VerticalAlignment = S001_ALIGNMENT,
    *,
    spheres: Sequence[SphereSpec] | None = None,
    material: Mapping[str, float] | None = None,
) -> ET.Element:
    """Build the official ``SmoothSphereHalfSpaceForce`` ForceSet.

    Args:
        alignment: Default S001 vertical alignment used when ``spheres`` is not supplied.
        spheres: Optional calibrated spheres that preserve the pinned topology.
        material: Optional complete calibrated material parameter mapping.
    """
    sphere_values = sphere_specs(alignment) if spheres is None else _validate_sphere_specs(spheres)
    material_values = dict(_MATERIAL if material is None else material)
    if set(material_values) != set(_MATERIAL) or not all(
        math.isfinite(float(value)) for value in material_values.values()
    ):
        raise ValueError("material must contain exactly the finite pinned SmoothSphereHalfSpace fields")
    document = ET.Element("OpenSimDocument", {"Version": "40600"})
    force_set = _element(document, "ForceSet", None, name="contact_force_set")
    objects = _element(force_set, "objects")
    for sphere in sphere_values:
        force = _element(objects, "SmoothSphereHalfSpaceForce", None, name=sphere.force_name)
        _element(force, "socket_sphere", f"/contactgeometryset/{sphere.name}")
        _element(force, "socket_half_space", "/contactgeometryset/floor")
        for name in _MATERIAL:
            _element(force, name, float(material_values[name]))
    _element(force_set, "groups")
    return document


def newton_augmentation_spec(
    alignment: VerticalAlignment = S001_ALIGNMENT,
    *,
    spheres: Sequence[SphereSpec] | None = None,
    material: Mapping[str, float] | None = None,
) -> NewtonAugmentationSpec:
    """Return the Newton ``OsimContactGeometry``/``OsimContactForce`` spec.

    Args:
        alignment: Default S001 vertical alignment used when ``spheres`` is not supplied.
        spheres: Optional calibrated spheres that preserve the pinned topology.
        material: Optional complete calibrated material parameter mapping.
    """
    sphere_values = sphere_specs(alignment) if spheres is None else _validate_sphere_specs(spheres)
    material_values = dict(_NEWTON_MATERIAL if material is None else material)
    if set(material_values) != set(_NEWTON_MATERIAL) or not all(
        math.isfinite(float(value)) for value in material_values.values()
    ):
        raise ValueError("material must contain exactly the finite pinned SmoothSphereHalfSpace fields")
    geometry: list[Any] = [
        newton_osim.OsimContactGeometry(
            name="floor",
            type="ContactHalfSpace",
            body="ground",
            location=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, -0.5 * math.pi),
        )
    ]
    forces: list[Any] = []
    for sphere in sphere_values:
        geometry.append(
            newton_osim.OsimContactGeometry(
                name=sphere.name,
                type="ContactSphere",
                body=sphere.body,
                location=sphere.location_m,
                radius=sphere.radius_m,
            )
        )
        forces.append(
            newton_osim.OsimContactForce(
                name=sphere.force_name,
                type="SmoothSphereHalfSpaceForce",
                sphere=sphere.name,
                half_space="floor",
                params=dict(material_values),
                geometries=[sphere.name, "floor"],
            )
        )
    return NewtonAugmentationSpec(tuple(geometry), tuple(forces))


def augment_opensim_compat_model(
    model: Any,
    alignment: VerticalAlignment = S001_ALIGNMENT,
    *,
    spheres: Sequence[SphereSpec] | None = None,
    material: Mapping[str, float] | None = None,
) -> Any:
    """Deep-copy and augment a Newton model; reject names and missing bodies."""
    augmented = copy.deepcopy(model)
    spec = newton_augmentation_spec(alignment, spheres=spheres, material=material)
    existing = {item.name for item in augmented.contact_geometry} | {item.name for item in augmented.contact_forces}
    requested = {item.name for item in spec.contact_geometry} | {item.name for item in spec.contact_forces}
    if existing & requested:
        raise ValueError(f"contact component names already exist: {sorted(existing & requested)}")
    bodies = {body.name for body in augmented.bodies}
    missing = {item.body for item in spec.contact_geometry if item.body != "ground" and item.body not in bodies}
    if missing:
        raise KeyError(f"contact bodies are missing: {sorted(missing)}")
    augmented.contact_geometry.extend(spec.contact_geometry)
    augmented.contact_forces.extend(spec.contact_forces)
    return augmented


def moco_contact_groups(alignment: VerticalAlignment = S001_ALIGNMENT) -> tuple[MocoContactGroupSpec, ...]:
    """Return left/right force groups, including each toe alternative frame."""
    spheres = sphere_specs(alignment)
    groups = []
    for side in _SIDE_ORDER:
        suffix = _SUFFIX[side]
        forces = tuple(f"/{sphere.force_name}" for sphere in spheres if sphere.side == side)
        groups.append(
            MocoContactGroupSpec(
                side=side,
                contact_force_paths=forces,
                external_force_name=side,
                applied_to_body=f"calcn_{suffix}",
                alternative_frame_paths=(f"/bodyset/toes_{suffix}",),
            )
        )
    return tuple(groups)


def validate_external_loads_reference(path: str | os.PathLike[str]) -> Path:
    """Validate the measured force names/frames used by the Moco goal only."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = ET.parse(source).getroot()
    external = {item.get("name", ""): item for item in root.iter("ExternalForce")}
    if set(external) != set(_SIDE_ORDER):
        raise ValueError("ExternalLoads must contain exactly the named left and right forces")
    for group in moco_contact_groups():
        item = external[group.external_force_name]
        if (item.findtext("applied_to_body") or "").strip() != group.applied_to_body:
            raise ValueError(f"ExternalForce {group.external_force_name!r} is applied to the wrong body")
        if (item.findtext("force_expressed_in_body") or "").strip() != "ground":
            raise ValueError("Moco contact reference forces must be expressed in ground")
    datafiles = [(item.text or "").strip() for item in root.iter("datafile") if (item.text or "").strip()]
    if len(datafiles) != 1:
        raise ValueError("ExternalLoads must reference exactly one measured data file")
    datafile = Path(datafiles[0])
    datafile = datafile if datafile.is_absolute() else source.parent / datafile
    if not datafile.resolve().is_file():
        raise FileNotFoundError(datafile.resolve())
    return datafile.resolve()


def configure_moco_contact_tracking_goal(
    opensim: Any,
    external_loads_path: str | os.PathLike[str],
    *,
    name: str = "grf_tracking",
    weight: float = 5.0e-3,
) -> Any:
    """Construct, but do not solve, the official force-vector tracking goal.

    ``setExternalLoadsFile`` makes the measured loads a reference.  This
    function deliberately accepts no model and has no code path that invokes
    ``ModOpAddExternalLoads`` or appends an ``ExternalForce``.
    """
    source = Path(external_loads_path).resolve()
    validate_external_loads_reference(source)
    goal = opensim.MocoContactTrackingGoal(name, _finite(weight, "weight"))
    goal.setExternalLoadsFile(str(source))
    for spec in moco_contact_groups():
        forces = opensim.StdVectorString()
        alternatives = opensim.StdVectorString()
        for path in spec.contact_force_paths:
            forces.append(path)
        for path in spec.alternative_frame_paths:
            alternatives.append(path)
        group = opensim.MocoContactTrackingGoalGroup(forces, spec.external_force_name, alternatives)
        goal.addContactGroup(group)
    return goal


def assert_model_has_no_external_loads(model_or_path: Any) -> None:
    """Reject predictive models containing model-applied measured loads."""
    if isinstance(model_or_path, (str, os.PathLike)):
        if any(item.tag == "ExternalForce" for item in ET.parse(model_or_path).getroot().iter()):
            raise ValueError("predictive model must not contain model-added ExternalForce loads")
        return
    # Inspect the full component tree, not only ForceSet. The pinned walking
    # example appends forces as root-level components, and ExternalForce can be
    # appended there too.
    components = model_or_path.getComponentsList()
    for component in components:
        if component.getConcreteClassName() == "ExternalForce":
            raise ValueError("predictive model must not contain model-added ExternalForce loads")


def expected_record_labels(force_name: str) -> tuple[str, ...]:
    if not force_name:
        raise ValueError("force_name must not be empty")
    return tuple(f"{force_name}.{suffix}" for suffix in _RECORD_SUFFIXES)


def aggregate_body_wrenches(body_names: Sequence[str], body_wrenches: np.ndarray) -> np.ndarray:
    """Combine calcaneus and toe body wrenches into left/right foot resultants.

    The input trailing layout is ``[F, P, T_at_P]``.  Output points are the
    ground origin and output torques are moments about that origin.
    """
    names = tuple(str(name) for name in body_names)
    if len(names) != len(set(names)) or set(names) != set(_BODY_ORDER):
        raise ValueError(f"body_names must contain exactly {_BODY_ORDER}")
    values = np.asarray(body_wrenches, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (4, 9) or not np.all(np.isfinite(values)):
        raise ValueError("body_wrenches must have finite shape [frame, 4, 9]")
    output = np.zeros((len(values), 2, 9), dtype=float)
    for side_index, side in enumerate(_SIDE_ORDER):
        suffix = _SUFFIX[side]
        for body in (f"calcn_{suffix}", f"toes_{suffix}"):
            wrench = values[:, names.index(body)]
            output[:, side_index, :3] += wrench[:, :3]
            output[:, side_index, 6:] += wrench[:, 6:] + np.cross(wrench[:, 3:6], wrench[:, :3])
    return output


def cop_and_free_moment(
    foot_wrenches: np.ndarray,
    *,
    ground_height_m: float = 0.0,
    load_threshold_n: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute independent COP and vertical free moment from foot resultants."""
    values = np.asarray(foot_wrenches, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 9:
        raise ValueError("foot_wrenches must have trailing shape 9")
    force = values[..., :3]
    point = values[..., 3:6]
    moment = np.cross(point, force) + values[..., 6:9]
    threshold = _finite(load_threshold_n, "load_threshold_n")
    if threshold <= 0.0:
        raise ValueError("load_threshold_n must be positive")
    loaded = np.all(np.isfinite(values), axis=-1) & (force[..., 1] >= threshold)
    cop = np.full(force.shape, np.nan)
    normal = np.array((0.0, 1.0, 0.0))
    numerator = np.cross(normal, moment) + _finite(ground_height_m, "ground_height_m") * force
    cop[loaded] = numerator[loaded] / force[..., 1][loaded, None]
    free = np.full(force.shape[:-1], np.nan)
    free[loaded] = (moment - np.cross(cop, force))[..., 1][loaded]
    return cop, free


def parity_metrics(official: ContactEvaluation, newton: ContactEvaluation) -> dict[str, Any]:
    """Report body-split and aggregate Newton-minus-official wrench parity."""
    if official.body_names != newton.body_names or official.foot_names != newton.foot_names:
        raise ValueError("official and Newton evaluation ordering differs")
    if official.body_wrenches.shape != newton.body_wrenches.shape:
        raise ValueError("official and Newton body wrench shapes differ")
    body_diff = np.asarray(newton.body_wrenches) - np.asarray(official.body_wrenches)
    foot_diff = np.asarray(newton.foot_wrenches) - np.asarray(official.foot_wrenches)

    def metrics(values: np.ndarray) -> dict[str, float]:
        return {
            "max_abs_force_N": float(np.max(np.abs(values[..., :3]))),
            "rms_force_N": float(np.sqrt(np.mean(np.square(values[..., :3])))),
            "max_abs_torque_Nm": float(np.max(np.abs(values[..., 6:9]))),
            "rms_torque_Nm": float(np.sqrt(np.mean(np.square(values[..., 6:9])))),
        }

    return {
        "body": metrics(body_diff),
        "foot": metrics(foot_diff),
        "per_body": {name: metrics(body_diff[:, index]) for index, name in enumerate(official.body_names)},
        "per_foot": {name: metrics(foot_diff[:, index]) for index, name in enumerate(official.foot_names)},
    }


def independent_load_validation(
    predicted_foot_wrenches: np.ndarray,
    measured_foot_wrenches: np.ndarray,
    *,
    ground_height_m: float = 0.0,
    load_threshold_n: float = 50.0,
) -> dict[str, Any]:
    """Compare COP/free moment outside ``MocoContactTrackingGoal``."""
    predicted = np.asarray(predicted_foot_wrenches, dtype=float)
    measured = np.asarray(measured_foot_wrenches, dtype=float)
    if predicted.shape != measured.shape or predicted.ndim != 3 or predicted.shape[1:] != (2, 9):
        raise ValueError("predicted and measured wrenches must share shape [frame, 2, 9]")
    predicted_cop, predicted_free = cop_and_free_moment(
        predicted, ground_height_m=ground_height_m, load_threshold_n=load_threshold_n
    )
    measured_cop, measured_free = cop_and_free_moment(
        measured, ground_height_m=ground_height_m, load_threshold_n=load_threshold_n
    )
    loaded = np.isfinite(predicted_free) & np.isfinite(measured_free)
    cop_error = predicted_cop - measured_cop
    free_error = predicted_free - measured_free
    return {
        "predicted_cop_m": predicted_cop,
        "measured_cop_m": measured_cop,
        "predicted_free_moment_Nm": predicted_free,
        "measured_free_moment_Nm": measured_free,
        "loaded_comparison_count": int(np.count_nonzero(loaded)),
        "cop_rms_m": float(np.sqrt(np.mean(np.square(cop_error[loaded])))) if np.any(loaded) else math.nan,
        "free_moment_rms_Nm": (float(np.sqrt(np.mean(np.square(free_error[loaded])))) if np.any(loaded) else math.nan),
    }


def _load_official_augmented_model(
    opensim: Any,
    model_path: str | os.PathLike[str],
    alignment: VerticalAlignment = S001_ALIGNMENT,
    *,
    spheres: Sequence[SphereSpec] | None = None,
    material: Mapping[str, float] | None = None,
) -> Any:
    model = opensim.Model(str(Path(model_path).resolve()))
    assert_model_has_no_external_loads(model)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        geometry_path = directory / "ContactGeometrySet.xml"
        force_path = directory / "ContactForceSet.xml"
        geometry_path.write_bytes(xml_bytes(build_contact_geometry_xml(alignment, spheres=spheres)))
        force_path.write_bytes(xml_bytes(build_force_xml(alignment, spheres=spheres, material=material)))
        geometry_set = opensim.ContactGeometrySet(str(geometry_path))
        force_set = opensim.ForceSet(str(force_path))
        for index in range(geometry_set.getSize()):
            model.addContactGeometry(geometry_set.get(index).clone())
        # Match the pinned example: root-level components and /contact* paths.
        for index in range(force_set.getSize()):
            model.addComponent(force_set.get(index).clone())
    model.finalizeConnections()
    return model


def _validate_motion_arrays(
    coordinate_names: Sequence[str], coordinates: np.ndarray, speeds: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    names = tuple(str(name) for name in coordinate_names)
    q = np.asarray(coordinates, dtype=float)
    qd = np.asarray(speeds, dtype=float)
    if len(names) != len(set(names)) or q.ndim != 2 or q.shape != qd.shape or q.shape[1] != len(names):
        raise ValueError("coordinates/speeds must share shape [frame, unique coordinate]")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
        raise ValueError("coordinates and speeds must be finite")
    return names, q, qd


def _read_storage(path: str | os.PathLike[str]) -> tuple[tuple[str, ...], np.ndarray, bool]:
    """Read one numeric OpenSim Storage table without importing OpenSim."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    try:
        header_end = next(index for index, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as error:
        raise ValueError(f"{source} has no endheader") from error
    labels = tuple(lines[header_end + 1].split())
    if not labels or labels[0] != "time" or len(labels) != len(set(labels)):
        raise ValueError(f"{source} has invalid or duplicate Storage labels")
    values = np.loadtxt(lines[header_end + 2 :], dtype=float, ndmin=2)
    if values.shape[1] != len(labels) or not np.all(np.isfinite(values)):
        raise ValueError(f"{source} has invalid numeric Storage data")
    if len(values) > 1 and np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{source} times must be strictly increasing")
    in_degrees = any(line.strip().lower() == "indegrees=yes" for line in lines[:header_end])
    return labels, values, in_degrees


def load_prescribed_q_qd(
    opensim: Any,
    model_path: str | os.PathLike[str],
    coordinate_path: str | os.PathLike[str],
    speed_path: str | os.PathLike[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Load synchronized RRA q/qd Storage tables and convert angular data to SI."""
    q_labels, q_values, q_degrees = _read_storage(coordinate_path)
    u_labels, u_values, u_degrees = _read_storage(speed_path)
    if q_labels != u_labels or q_values.shape != u_values.shape:
        raise ValueError("RRA coordinate and speed tables must have identical labels and shape")
    if not np.array_equal(q_values[:, 0], u_values[:, 0]):
        raise ValueError("RRA coordinate and speed times must match exactly")
    model = opensim.Model(str(Path(model_path).resolve()))
    names = q_labels[1:]
    official_names = tuple(
        model.getCoordinateSet().get(index).getName() for index in range(model.getCoordinateSet().getSize())
    )
    if names != official_names:
        raise ValueError("RRA Storage coordinate order does not match the model")
    q = q_values[:, 1:].copy()
    qd = u_values[:, 1:].copy()
    for index, name in enumerate(names):
        coordinate = model.getCoordinateSet().get(name)
        if coordinate.getMotionType() == opensim.Coordinate.Rotational:
            if q_degrees:
                q[:, index] = np.deg2rad(q[:, index])
            if u_degrees:
                qd[:, index] = np.deg2rad(qd[:, index])
    return q_values[:, 0].copy(), names, q, qd


def derive_alignment_from_official_prescribed(
    opensim: Any,
    model_path: str | os.PathLike[str] = _DEFAULT_RRA_MODEL,
    coordinate_path: str | os.PathLike[str] = _DEFAULT_RRA_Q,
    external_loads_path: str | os.PathLike[str] = _DEFAULT_EXTERNAL_LOADS,
    *,
    load_threshold_n: float = 50.0,
    bounds_m: tuple[float, float] = _ALIGNMENT_BOUNDS_M,
) -> VerticalAlignment:
    """Re-derive the archived S001 offset from official prescribed positions.

    Measured vertical force selects stance frames only.  It is not supplied to
    either contact evaluator or added to the model.
    """
    q_labels, q_values, q_degrees = _read_storage(coordinate_path)
    measured_path = validate_external_loads_reference(external_loads_path)
    measured_labels, measured_values, _ = _read_storage(measured_path)
    model = opensim.Model(str(Path(model_path).resolve()))
    names = q_labels[1:]
    official_names = tuple(
        model.getCoordinateSet().get(index).getName() for index in range(model.getCoordinateSet().getSize())
    )
    if names != official_names:
        raise ValueError("RRA Storage coordinate order does not match the model")
    state = model.initSystem()
    centers = np.empty((len(q_values), 2, 6), dtype=float)
    gains = np.empty_like(centers)
    stance = np.empty((len(q_values), 2), dtype=bool)
    coordinate_set = model.getCoordinateSet()
    body_set = model.getBodySet()
    for frame, row in enumerate(q_values):
        state.setTime(float(row[0]))
        for name, value in zip(names, row[1:], strict=True):
            coordinate = coordinate_set.get(name)
            scalar = float(value)
            if q_degrees and coordinate.getMotionType() == opensim.Coordinate.Rotational:
                scalar = math.radians(scalar)
            coordinate.setValue(state, scalar, False)
        model.realizePosition(state)
        for side_index, side in enumerate(_SIDE_ORDER):
            suffix = _SUFFIX[side]
            vertical_label = f"ground_force_{suffix}_vy"
            if vertical_label not in measured_labels:
                raise ValueError(f"measured reference has no {vertical_label!r} column")
            force_y = np.interp(
                row[0], measured_values[:, 0], measured_values[:, measured_labels.index(vertical_label)]
            )
            stance[frame, side_index] = force_y >= load_threshold_n
            for sphere_index, role in enumerate(_ROLE_ORDER):
                x, right_z, _radius = _RIGHT_TOPOLOGY[role]
                z = (-1.0 if side == "left" else 1.0) * right_z
                body = body_set.get(f"{_ROLE_BODY[role]}_{suffix}")
                base = body.findStationLocationInGround(state, opensim.Vec3(x, 0.0, z))
                unit = body.findStationLocationInGround(state, opensim.Vec3(x, 1.0, z))
                centers[frame, side_index, sphere_index] = float(base[1])
                gains[frame, side_index, sphere_index] = float(unit[1] - base[1])
    return derive_vertical_alignment(
        centers,
        gains,
        stance,
        bounds_m=bounds_m,
        measured_load_threshold_n=load_threshold_n,
    )


def evaluate_official_prescribed(
    opensim: Any,
    model_path: str | os.PathLike[str],
    coordinate_names: Sequence[str],
    coordinates: np.ndarray,
    speeds: np.ndarray,
    *,
    times_s: Sequence[float] | None = None,
    alignment: VerticalAlignment = S001_ALIGNMENT,
    spheres: Sequence[SphereSpec] | None = None,
    material: Mapping[str, float] | None = None,
) -> ContactEvaluation:
    """Evaluate official OpenSim contact on prescribed SI q/qd arrays."""
    names, q, qd = _validate_motion_arrays(coordinate_names, coordinates, speeds)
    model = _load_official_augmented_model(opensim, model_path, alignment, spheres=spheres, material=material)
    official_names = tuple(
        model.getCoordinateSet().get(index).getName() for index in range(model.getCoordinateSet().getSize())
    )
    if names != official_names:
        raise ValueError("prescribed coordinate order does not match official OpenSim")
    times = np.arange(len(q), dtype=float) if times_s is None else np.asarray(times_s, dtype=float)
    if times.shape != (len(q),) or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain one finite value per frame")
    specs = sphere_specs(alignment) if spheres is None else _validate_sphere_specs(spheres)
    force_objects = []
    for sphere in specs:
        force = opensim.SmoothSphereHalfSpaceForce.safeDownCast(model.getComponent(f"/{sphere.force_name}"))
        if force is None:
            raise TypeError(f"{sphere.force_name!r} is not SmoothSphereHalfSpaceForce")
        labels = force.getRecordLabels()
        actual = tuple(str(labels.get(index)) for index in range(labels.getSize()))
        if actual != expected_record_labels(sphere.force_name):
            raise ValueError(f"official record labels changed for {sphere.force_name}")
        force_objects.append(force)
    state = model.initSystem()
    body_values = np.zeros((len(q), 4, 9), dtype=float)
    coordinate_set = model.getCoordinateSet()
    body_set = model.getBodySet()
    for frame in range(len(q)):
        state.setTime(float(times[frame]))
        for coordinate, name in enumerate(names):
            component = coordinate_set.get(name)
            component.setValue(state, float(q[frame, coordinate]), False)
            component.setSpeedValue(state, float(qd[frame, coordinate]))
        model.realizeDynamics(state)
        for body_index, body_name in enumerate(_BODY_ORDER):
            point = body_set.get(body_name).findStationLocationInGround(state, opensim.Vec3(0.0))
            body_values[frame, body_index, 3:6] = [float(point[index]) for index in range(3)]
        for sphere, force in zip(specs, force_objects, strict=True):
            record = force.getRecordValues(state)
            values = np.asarray([float(record.get(index)) for index in range(record.size())])
            body_index = _BODY_ORDER.index(sphere.body)
            body_values[frame, body_index, :3] += values[:3]
            body_values[frame, body_index, 6:9] += values[3:6]
    foot = aggregate_body_wrenches(_BODY_ORDER, body_values)
    return ContactEvaluation(_BODY_ORDER, body_values, _SIDE_ORDER, foot)


def evaluate_newton_prescribed(
    model_path: str | os.PathLike[str],
    coordinate_names: Sequence[str],
    coordinates: np.ndarray,
    speeds: np.ndarray,
    *,
    device: str = "cpu",
    velocity_stencil_h_s: float = 1.0e-6,
    alignment: VerticalAlignment = S001_ALIGNMENT,
) -> ContactEvaluation:
    """Evaluate Newton contact on the same prescribed SI q/qd arrays."""
    names, q, qd = _validate_motion_arrays(coordinate_names, coordinates, speeds)
    model_path = Path(model_path).resolve()
    assert_model_has_no_external_loads(model_path)
    model = augment_opensim_compat_model(newton_osim.parse_osim(model_path), alignment)
    contact = newton_osim.OpenSimContact(model, device=device)
    if names != tuple(contact.coordinate_names):
        raise ValueError("prescribed coordinate order does not match Newton OpenSimContact")
    returned_names, values = contact.body_wrenches(q, qd, h=velocity_stencil_h_s, frame="opensim")
    returned_names = tuple(returned_names)
    missing = set(_BODY_ORDER) - set(returned_names)
    if missing:
        raise ValueError(f"Newton contact omitted bodies: {sorted(missing)}")
    selected = np.asarray(values, dtype=float)[:, [returned_names.index(name) for name in _BODY_ORDER]]
    if selected.shape != (len(q), 4, 9) or not np.all(np.isfinite(selected)):
        raise ValueError("Newton returned invalid body wrenches")
    foot = aggregate_body_wrenches(_BODY_ORDER, selected)
    return ContactEvaluation(_BODY_ORDER, selected, _SIDE_ORDER, foot)


def evaluate_prescribed_parity(
    opensim: Any,
    model_path: str | os.PathLike[str],
    coordinate_path: str | os.PathLike[str],
    speed_path: str | os.PathLike[str],
    *,
    frame_indices: Sequence[int] | None = None,
    device: str = "cpu",
    alignment: VerticalAlignment = S001_ALIGNMENT,
) -> tuple[ContactEvaluation, ContactEvaluation, dict[str, Any]]:
    """Evaluate the same archived q/qd in official OpenSim and Newton.

    This function reads no ExternalLoads data.  It performs no optimization or
    Moco solve.
    """
    times, names, q, qd = load_prescribed_q_qd(opensim, model_path, coordinate_path, speed_path)
    if frame_indices is not None:
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(times)):
            raise IndexError("frame_indices must be a nonempty in-range vector")
        if len(indices) > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError("frame_indices must be unique and strictly increasing")
        times, q, qd = times[indices], q[indices], qd[indices]
    official = evaluate_official_prescribed(opensim, model_path, names, q, qd, times_s=times, alignment=alignment)
    newton = evaluate_newton_prescribed(model_path, names, q, qd, device=device, alignment=alignment)
    return official, newton, parity_metrics(official, newton)


def provenance(alignment: VerticalAlignment = S001_ALIGNMENT) -> dict[str, Any]:
    """Return pinned source, frame, units, alignment, and scope metadata."""
    return {
        "schema_version": _SCHEMA,
        "frame": _FRAME,
        "units": dict(_UNITS),
        "scope": "prescribed_contact_reference_no_optimization_no_moco_solve",
        "pinned_upstream": {
            "commit": _PINNED_COMMIT,
            "files": {path: {"url": f"{_RAW_ROOT}/{path}", "sha256": digest} for path, digest in _SOURCE_FILES.items()},
        },
        "s001_vertical_alignment": {
            **asdict(alignment),
            "official_example_offset_m": _OFFICIAL_EXAMPLE_OFFSET_M,
            "source_sha256": dict(_S001_ALIGNMENT_SOURCES),
        },
        "contact_tracking_contract": {
            "tracked_quantity": "summed_ground_reaction_force_vector_only",
            "external_loads_usage": "MocoContactTrackingGoal_reference_only",
            "external_loads_added_to_predictive_model": False,
            "cop_and_free_moment": "independent_validation_only",
        },
    }


def write_reference_files(
    output_dir: str | os.PathLike[str],
    *,
    external_loads_path: str | os.PathLike[str] = _DEFAULT_EXTERNAL_LOADS,
    alignment: VerticalAlignment = S001_ALIGNMENT,
) -> Path:
    """Write deterministic XML/spec/manifest files without running a solve."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    measured_data = validate_external_loads_reference(external_loads_path)
    geometry_path = output / "S001_ContactGeometrySet.xml"
    force_path = output / "S001_ContactForceSet.xml"
    newton_path = output / "S001_newton_contact_augmentation.json"
    groups_path = output / "S001_MocoContactTrackingGoal_groups.json"
    manifest_path = output / "manifest.json"
    geometry_path.write_bytes(xml_bytes(build_contact_geometry_xml(alignment)))
    force_path.write_bytes(xml_bytes(build_force_xml(alignment)))
    spec = newton_augmentation_spec(alignment)
    newton_payload = {
        "contact_geometry": [asdict(item) for item in spec.contact_geometry],
        "contact_forces": [asdict(item) for item in spec.contact_forces],
    }
    newton_path.write_text(json.dumps(newton_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    groups_path.write_text(
        json.dumps(
            {
                "external_loads_reference": str(Path(external_loads_path).resolve()),
                "model_added_external_loads": False,
                "groups": [asdict(group) for group in moco_contact_groups(alignment)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = provenance(alignment)
    payload["measured_reference"] = {
        "external_loads_path": str(Path(external_loads_path).resolve()),
        "external_loads_sha256": _sha256(external_loads_path),
        "data_path": str(measured_data),
        "data_sha256": _sha256(measured_data),
    }
    payload["generated"] = {path.name: _sha256(path) for path in (geometry_path, force_path, newton_path, groups_path)}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
