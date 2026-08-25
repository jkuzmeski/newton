# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scale the pinned gait2354 reference directly from static C3D marker arrays.

This offline adapter translates OpenSim ModelScaler measurement semantics into a
saved scaled reference model. It is not part of the Newton simulation runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .c3d_adapter import C3DMarkerTrajectory

Vec3 = tuple[float, float, float]
ScaleFactorSet = dict[str, Vec3]
_REFERENCE_PATH = Path(__file__).with_name("assets") / "gait2354_scale_reference.json"
_NEWTON_TO_OPENSIM = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
_ALIASES = {
    "STRN": "Sternum",
    "RSHO": "R.Acromium",
    "LSHO": "L.Acromium",
    "RASI": "R.ASIS",
    "LASI": "L.ASIS",
    "RKNE": "R.Knee.Lat",
    "RMKNE": "R.Knee.Med",
    "LKNE": "L.Knee.Lat",
    "LMKNE": "L.Knee.Med",
    "RANK": "R.Ankle.Lat",
    "RMANK": "R.Ankle.Med",
    "LANK": "L.Ankle.Lat",
    "LMANK": "L.Ankle.Med",
    "RHEE": "R.Heel",
    "LHEE": "L.Heel",
    "RTOE": "R.Toe.Tip",
    "LTOE": "L.Toe.Tip",
    "RMTH1": "R.Toe.Med",
    "LMTH1": "L.Toe.Med",
    "RMTH5": "R.Toe.Lat",
    "LMTH5": "L.Toe.Lat",
    "RTH2": "R.Thigh.Upper",
    "RTH3": "R.Thigh.Front",
    "RTH4": "R.Thigh.Rear",
    "LTH2": "L.Thigh.Upper",
    "LTH3": "L.Thigh.Front",
    "LTH4": "L.Thigh.Rear",
    "RTIB2": "R.Shank.Upper",
    "RTIB3": "R.Shank.Front",
    "RTIB4": "R.Shank.Rear",
    "LTIB2": "L.Shank.Upper",
    "LTIB3": "L.Shank.Front",
    "LTIB4": "L.Shank.Rear",
}
_VIRTUAL = {"V.Sacral": ("LPSI", "RPSI"), "Top.Head": ("LFHD", "RFHD", "LBHD", "RBHD")}
_MEASUREMENTS = (
    ("pelvis", (("R.ASIS", "L.ASIS"),), ("pelvis",)),
    ("torso", (("R.Acromium", "L.Acromium"), ("Sternum", "R.ASIS"), ("Sternum", "L.ASIS")), ("torso",)),
    ("thigh", (("R.ASIS", "R.Knee.Lat"), ("L.ASIS", "L.Knee.Lat")), ("femur_r", "femur_l")),
    (
        "shank",
        (("R.Knee.Lat", "R.Ankle.Lat"), ("L.Knee.Lat", "L.Ankle.Lat")),
        ("tibia_r", "tibia_l", "talus_r", "talus_l"),
    ),
    ("foot", (("R.Heel", "R.Toe.Tip"), ("L.Heel", "L.Toe.Tip")), ("calcn_r", "calcn_l", "toes_r", "toes_l")),
)


@dataclass(frozen=True, slots=True)
class ScaledSubjectReference:
    """Scaled OpenSim reference model and measurement provenance."""

    root: Path
    model_path: Path
    manifest_path: Path
    scale_factors: ScaleFactorSet


@dataclass(frozen=True, slots=True)
class MarkerPlacementReference:
    """Official OpenSim MarkerPlacer oracle artifact."""

    root: Path
    model_path: Path
    motion_path: Path
    marker_set_path: Path
    manifest_path: Path
    marker_rms: float
    marker_max: float
    marker_rms_limit: float
    marker_max_limit: float


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _average_marker(
    markers: C3DMarkerTrajectory,
    indices: tuple[int, ...],
    frame_mask: np.ndarray,
) -> np.ndarray:
    values = []
    for index in indices:
        mask = frame_mask & markers.valid[:, index]
        if not np.any(mask):
            raise ValueError(f"marker {markers.marker_names[index]!r} has no valid static samples")
        values.append(np.mean(markers.positions[mask, index], axis=0))
    return np.mean(values, axis=0)


def _experimental_positions(
    markers: C3DMarkerTrajectory,
    time_range: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
    if time_range is None:
        frame_mask = np.ones(len(markers.times), dtype=bool)
    else:
        frame_mask = (markers.times >= time_range[0]) & (markers.times <= time_range[1])
    if not np.any(frame_mask):
        raise ValueError("static scaling time range contains no C3D frames")
    index = {name: column for column, name in enumerate(markers.marker_names)}
    output = {}
    for source, target in _ALIASES.items():
        if source in index:
            output[target] = _average_marker(markers, (index[source],), frame_mask) @ _NEWTON_TO_OPENSIM.T
    for target, sources in _VIRTUAL.items():
        missing = [source for source in sources if source not in index]
        if missing:
            raise ValueError(f"cannot build {target}: missing {missing}")
        output[target] = (
            _average_marker(markers, tuple(index[source] for source in sources), frame_mask) @ _NEWTON_TO_OPENSIM.T
        )
    return output


def _measurement_factors(
    markers: C3DMarkerTrajectory,
    time_range: tuple[float, float] | None,
) -> tuple[ScaleFactorSet, dict[str, dict]]:
    reference = json.loads(_REFERENCE_PATH.read_text())
    model_positions = {name: np.asarray(value) for name, value in reference["marker_positions"].items()}
    experimental = _experimental_positions(markers, time_range)
    factors: ScaleFactorSet = {}
    diagnostics = {}
    for name, pairs, bodies in _MEASUREMENTS:
        ratios = []
        for marker_a, marker_b in pairs:
            if marker_a not in experimental or marker_b not in experimental:
                raise ValueError(f"measurement {name!r} is missing {marker_a!r} or {marker_b!r}")
            source_distance = float(np.linalg.norm(experimental[marker_a] - experimental[marker_b]))
            model_distance = float(np.linalg.norm(model_positions[marker_a] - model_positions[marker_b]))
            if not math.isfinite(source_distance) or model_distance <= 1.0e-9:
                raise ValueError(f"measurement {name!r} has an invalid marker-pair distance")
            ratios.append(source_distance / model_distance)
        value = float(np.mean(ratios))
        for body in bodies:
            factors[body] = (value, value, value)
        diagnostics[name] = {"ratios": ratios, "scale": value}
    return factors, diagnostics


def _txt_vec(elem: ET.Element) -> list[float]:
    return [float(x) for x in (elem.text or "").split()]


def _set_vec(elem: ET.Element, vals) -> None:
    elem.text = " " + " ".join(f"{v:.8f}" for v in vals) + " "


def _scale_inertia(inertia6: tuple[float, ...], s: Vec3, mass_scale: float) -> tuple[float, ...]:
    """Scale an inertia tensor by ``s`` (solid-ellipsoid second-moment method).

    Reproduces OpenSim ``Body::scaleInertialProperties``: reconstruct the second
    moments ``A=sum(m x^2), B=sum(m y^2), C=sum(m z^2)`` from the diagonal,
    stretch them by ``s_i^2``, scale products by ``s_i s_j``, then optionally
    apply the mass scale (``mass_scale`` = ``sx sy sz`` when the geometry step
    also scales mass, else ``1``).
    """
    ixx, iyy, izz, ixy, ixz, iyz = inertia6
    A = 0.5 * (-ixx + iyy + izz)
    B = 0.5 * (ixx - iyy + izz)
    C = 0.5 * (ixx + iyy - izz)
    sx, sy, sz = s
    A, B, C = sx * sx * A, sy * sy * B, sz * sz * C
    nxx, nyy, nzz = B + C, A + C, A + B
    nxy, nxz, nyz = sx * sy * ixy, sx * sz * ixz, sy * sz * iyz
    return tuple(v * mass_scale for v in (nxx, nyy, nzz, nxy, nxz, nyz))


_WRAP_DIM_TAGS = ("radius", "length", "dimensions", "radii", "height")


def _scale_function_output(function: ET.Element | None, factor: float) -> None:
    """Scale the scalar output of a legacy OpenSim coordinate function."""
    if function is None or abs(factor - 1.0) < 1.0e-15:
        return
    child = next(iter(function), None)
    if child is None:
        return
    tag = child.tag.rsplit("}", 1)[-1]
    if tag == "LinearFunction":
        coefficients = child.find("coefficients")
        if coefficients is not None:
            values = _txt_vec(coefficients)
            if len(values) >= 2 and values[0] == 1.0 and values[1] == 0.0:
                return
    value_tag = {
        "Constant": "value",
        "LinearFunction": "coefficients",
        "SimmSpline": "y",
        "NaturalCubicSpline": "y",
        "GCVSpline": "y",
        "PiecewiseLinearFunction": "y",
    }.get(tag)
    if value_tag is not None:
        values = child.find(value_tag)
        if values is not None and (values.text or "").split():
            _set_vec(values, [factor * value for value in _txt_vec(values)])
        return
    if tag == "MultiplierFunction":
        scale = child.find("scale")
        if scale is None:
            raise ValueError("MultiplierFunction is missing its scale")
        scale.text = f" {factor * float(scale.text or 1.0):.8f} "
        return
    raise NotImplementedError(f"cannot scale the output of OpenSim function {tag}")


def _scale_spatial_transform_translations(joint: ET.Element, parent_scale: Vec3) -> None:
    """Scale CustomJoint translation axes/functions in the parent body frame."""
    spatial = joint.find("SpatialTransform")
    if spatial is None:
        return
    parent_scale_array = np.asarray(parent_scale, dtype=float)
    for transform_axis in spatial.findall("TransformAxis"):
        if not (transform_axis.get("name") or "").startswith("translation"):
            continue
        axis_element = transform_axis.find("axis")
        if axis_element is None or not (axis_element.text or "").split():
            continue
        axis = np.asarray(_txt_vec(axis_element), dtype=float)
        scale_factor = float(np.dot(np.abs(axis), parent_scale_array))
        if scale_factor < 1.0e-12:
            raise ValueError("scaled CustomJoint translation axis is degenerate")
        _scale_function_output(transform_axis.find("function"), scale_factor)


def _scale_display_geometry_once(body: ET.Element, body_scale: Vec3) -> None:
    """Apply body scaling once across nested legacy display-scale levels."""
    for visible in body.findall("VisibleObject"):
        visible_scale = visible.find("scale_factors")
        if visible_scale is not None and (visible_scale.text or "").split():
            _set_vec(
                visible_scale,
                [value * factor for value, factor in zip(_txt_vec(visible_scale), body_scale, strict=False)],
            )
            continue
        for geometry in visible.iter("DisplayGeometry"):
            geometry_scale = geometry.find("scale_factors")
            if geometry_scale is not None and (geometry_scale.text or "").split():
                _set_vec(
                    geometry_scale,
                    [value * factor for value, factor in zip(_txt_vec(geometry_scale), body_scale, strict=False)],
                )


def _scale_body_element(body: ET.Element, s: Vec3, parent_factors: ScaleFactorSet, scale_mass: bool) -> None:
    """Scale one ``<Body>`` element in place (geometry, inertia, joint, wrap)."""
    sx, sy, sz = s
    vol = sx * sy * sz

    mc = body.find("mass_center")
    if mc is not None and (mc.text or "").split():
        _set_vec(mc, [v * f for v, f in zip(_txt_vec(mc), s, strict=False)])

    inertia_tags = ("inertia_xx", "inertia_yy", "inertia_zz", "inertia_xy", "inertia_xz", "inertia_yz")
    elems = [body.find(t) for t in inertia_tags]
    if all(e is not None for e in elems):
        vals = tuple(float((e.text or "0").strip()) for e in elems)
        new = _scale_inertia(vals, s, vol if scale_mass else 1.0)
        for e, v in zip(elems, new, strict=False):
            e.text = f" {v:.8f} "

    if scale_mass:
        m = body.find("mass")
        if m is not None and (m.text or "").strip():
            m.text = f" {float(m.text) * vol:.8f} "

    # VisibleObject and DisplayGeometry scales are nested and multiplied at render time.
    _scale_display_geometry_once(body, s)

    # Joint owned by this body: location is in the child (this) frame; the
    # location_in_parent is in the parent frame.
    for joint in body.iter():
        if joint.tag.endswith("Joint") and joint.tag != "Joint":
            loc = joint.find("location")
            if loc is not None and (loc.text or "").split():
                _set_vec(loc, [v * f for v, f in zip(_txt_vec(loc), s, strict=False)])
            lip = joint.find("location_in_parent")
            pb = joint.find("parent_body")
            parent_scale = (
                parent_factors.get((pb.text or "").strip(), (1.0, 1.0, 1.0)) if pb is not None else (1.0, 1.0, 1.0)
            )
            if lip is not None and (lip.text or "").split():
                _set_vec(lip, [v * f for v, f in zip(_txt_vec(lip), parent_scale, strict=False)])
            _scale_spatial_transform_translations(joint, parent_scale)

    # Wrap objects on this body.
    for wrap in body.iter():
        if wrap.tag.startswith("Wrap") and wrap.tag != "WrapObjectSet":
            tr = wrap.find("translation")
            if tr is not None and (tr.text or "").split():
                _set_vec(tr, [v * f for v, f in zip(_txt_vec(tr), s, strict=False)])
            for dim in _WRAP_DIM_TAGS:
                de = wrap.find(dim)
                if de is not None and (de.text or "").split():
                    vals = _txt_vec(de)
                    iso = (sx + sy + sz) / 3.0
                    _set_vec(de, [v * iso for v in vals])


def _scale_osim_document(
    in_path: str,
    scale_factors: ScaleFactorSet,
    out_path: str,
    *,
    preserve_mass_distribution: bool,
    subject_mass: float | None,
) -> None:
    """Scale an OpenSim 3.x ``.osim`` document and write it to ``out_path``."""
    tree = ET.parse(in_path)
    root = tree.getroot()

    doc_ver = int(root.get("Version", "0"))
    if doc_ver >= 30000:
        raise NotImplementedError(
            f"The .osim scaler currently targets the OpenSim 3.x (Version < 30000) layout; got Version {doc_ver}."
        )

    scale_mass = not preserve_mass_distribution

    body_of = lambda name: scale_factors.get(name, (1.0, 1.0, 1.0))  # noqa: E731

    # Bodies: geometry, inertia, joint frames, wrap.
    for body in root.iter("Body"):
        name = body.get("name")
        if name is None or name == "ground":
            continue
        _scale_body_element(body, body_of(name), scale_factors, scale_mass)

    # Markers: location in their body frame.
    for marker in root.iter("Marker"):
        bd = marker.find("body")
        loc = marker.find("location")
        if bd is not None and loc is not None and (loc.text or "").split():
            s = body_of((bd.text or "").strip())
            _set_vec(loc, [v * f for v, f in zip(_txt_vec(loc), s, strict=False)])

    # Muscle / force path points: location in their body frame.
    for tag in ("PathPoint", "ConditionalPathPoint", "MovingPathPoint"):
        for pp in root.iter(tag):
            bd = pp.find("body")
            if bd is None:
                continue
            body_scale = body_of((bd.text or "").strip())
            loc = pp.find("location")
            if loc is not None and (loc.text or "").split():
                _set_vec(loc, [value * factor for value, factor in zip(_txt_vec(loc), body_scale, strict=False)])
            if tag == "MovingPathPoint":
                for axis, factor in zip("xyz", body_scale, strict=True):
                    _scale_function_output(pp.find(f"{axis}_location"), factor)

    # Mass normalization to the subject mass, preserving distribution.
    if subject_mass is not None:
        masses = [m for b in root.iter("Body") for m in [b.find("mass")] if m is not None and (m.text or "").strip()]
        total = sum(float(m.text) for m in masses)
        if total > 1e-9:
            ratio = subject_mass / total
            for b in root.iter("Body"):
                if b.get("name") == "ground":
                    continue
                m = b.find("mass")
                if m is not None and (m.text or "").strip():
                    m.text = f" {float(m.text) * ratio:.8f} "
                for t in ("inertia_xx", "inertia_yy", "inertia_zz", "inertia_xy", "inertia_xz", "inertia_yz"):
                    e = b.find(t)
                    if e is not None and (e.text or "").strip():
                        e.text = f" {float(e.text) * ratio:.8f} "

    tree.write(out_path, encoding="UTF-8", xml_declaration=True)


def scale_gait2354_from_markers(
    markers: C3DMarkerTrajectory,
    template_osim: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    subject_mass: float,
    time_range: tuple[float, float] | None = None,
) -> ScaledSubjectReference:
    """Apply OpenSim gait2354 measurement scaling and save a reference model.

    Marker placement remains a separate parity stage; this function preserves
    the accepted ModelScaler body, inertia, joint, muscle-path, and display-scale
    transformations without putting OpenSim into the Newton runtime.
    """
    template = Path(template_osim).resolve()
    reference = json.loads(_REFERENCE_PATH.read_text())
    if _sha256(template) != reference["template_sha256"]:
        raise ValueError("template model does not match the pinned gait2354 scaling reference")
    if not math.isfinite(subject_mass) or subject_mass <= 0.0:
        raise ValueError("subject_mass must be finite and positive")
    factors, diagnostics = _measurement_factors(markers, time_range)
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
        staged = Path(temporary) / "scaling"
        staged.mkdir()
        model_path = staged / "scaled_subject.osim"
        _scale_osim_document(
            str(template),
            factors,
            str(model_path),
            preserve_mass_distribution=True,
            subject_mass=subject_mass,
        )
        manifest = {
            "schema_version": "gait2354_c3d_model_scaling_1",
            "method_reference": {
                "name": "OpenSim ModelScaler-derived Trial 101 measurement scaling",
                "marker_placement": "not_applied_in_this_stage",
            },
            "source": {
                "c3d_file": markers.source_file,
                "c3d_sha256": markers.source_sha256,
                "template_file": template.name,
                "template_sha256": reference["template_sha256"],
            },
            "subject_mass_kg": subject_mass,
            "time_range_s": list(time_range) if time_range is not None else None,
            "measurements": diagnostics,
            "scale_factors": {name: list(value) for name, value in sorted(factors.items())},
            "output": {"file": model_path.name, "sha256": _sha256(model_path)},
        }
        manifest_path = staged / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.rename(staged, root)
    return ScaledSubjectReference(root, root / model_path.name, root / manifest_path.name, factors)


def _mapped_marker_trajectory(markers: C3DMarkerTrajectory):
    """Build an OpenSim-frame marker table for the official placement oracle."""
    from newton.opensim import OpenSimMarkerData  # noqa: PLC0415

    index = {name: column for column, name in enumerate(markers.marker_names)}
    names = []
    columns = []
    for source, target in _ALIASES.items():
        if source not in index:
            continue
        column = index[source]
        values = markers.positions[:, column].astype(np.float64) @ _NEWTON_TO_OPENSIM.T
        values[~markers.valid[:, column]] = np.nan
        names.append(target)
        columns.append(values)
    for target, sources in _VIRTUAL.items():
        available = [index[source] for source in sources if source in index]
        if len(available) < 2:
            raise ValueError(f"cannot build {target}: fewer than two source markers are available")
        values = markers.positions[:, available].astype(np.float64)
        valid = markers.valid[:, available]
        weighted = np.where(valid[..., None], values, 0.0)
        count = np.sum(valid, axis=1)
        mean = np.sum(weighted, axis=1) / np.maximum(count[:, None], 1)
        mean[count < 2] = np.nan
        names.append(target)
        columns.append(mean @ _NEWTON_TO_OPENSIM.T)
    return OpenSimMarkerData(
        times=markers.times,
        marker_names=names,
        data=np.stack(columns, axis=1),
        rate=markers.rate,
        units="m",
    )


_MARKER_PLACER_RUNNER = r"""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import opensim

root = Path.cwd()
config = json.loads((root / "runner_config.json").read_text())
tool = opensim.ScaleTool()
tool.setName("official_marker_placement")
tool.setPathToSubject(str(root) + os.sep)
tool.setSubjectMass(config["subject_mass"])
tool.setSubjectHeight(1000.0 * config["subject_height"])
tool.setPrintResultFiles(True)
tool.getGenericModelMaker().setModelFileName("scaled_subject.osim")
tool.getModelScaler().setApply(False)
placer = tool.getMarkerPlacer()
placer.setApply(True)
placer.setMarkerFileName("static_markers.trc")
interval = opensim.ArrayDouble()
interval.append(config["time_range"][0])
interval.append(config["time_range"][1])
placer.setTimeRange(interval)
placer.setMoveModelMarkers(True)
placer.setOutputModelFileName("placed_subject.osim")
placer.setOutputMotionFileName("static_pose.mot")
placer.setOutputMarkerFileName("adjusted_markers.xml")
setup_path = root / "marker_placer_setup.xml"
tool.printToXML(str(setup_path))

tree = ET.parse(setup_path)
placer_xml = next(tree.getroot().iter("MarkerPlacer"))
tasks = placer_xml.find("IKTaskSet")
if tasks is None:
    tasks = ET.SubElement(placer_xml, "IKTaskSet", name="gait2354_Scale")
objects = tasks.find("objects")
if objects is None:
    objects = ET.SubElement(tasks, "objects")
objects.clear()
for task in config["marker_tasks"]:
    element = ET.SubElement(objects, "IKMarkerTask", name=task["name"])
    ET.SubElement(element, "apply").text = str(task["apply"]).lower()
    ET.SubElement(element, "weight").text = str(task["weight"])
for task in config["coordinate_tasks"]:
    element = ET.SubElement(objects, "IKCoordinateTask", name=task["name"])
    ET.SubElement(element, "apply").text = str(task["apply"]).lower()
    ET.SubElement(element, "weight").text = str(task["weight"])
    ET.SubElement(element, "value_type").text = task["value_type"]
    ET.SubElement(element, "value").text = str(task["value"])
tree.write(setup_path, encoding="UTF-8", xml_declaration=True)
print("OPENSIM_VERSION=" + opensim.GetVersionAndDate())
if not opensim.ScaleTool(str(setup_path)).run():
    raise RuntimeError("official OpenSim MarkerPlacer failed")
"""


def place_markers_with_official_opensim(
    markers: C3DMarkerTrajectory,
    scaled_model: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    subject_mass: float,
    subject_height: float,
    time_range: tuple[float, float],
    max_marker_rms: float = 0.10,
    max_marker_error: float = 0.25,
) -> MarkerPlacementReference:
    """Run official OpenSim MarkerPlacer in a subprocess oracle sandbox."""
    from newton.opensim import write_trc  # noqa: PLC0415

    if max_marker_rms <= 0.0 or max_marker_error <= 0.0:
        raise ValueError("marker placement limits must be positive")
    reference = json.loads(_REFERENCE_PATH.read_text())
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
        staged = Path(temporary) / "placement"
        staged.mkdir()
        model_source = Path(scaled_model).resolve()
        model_copy = staged / "scaled_subject.osim"
        shutil.copy2(model_source, model_copy)
        marker_file = staged / "static_markers.trc"
        write_trc(marker_file, _mapped_marker_trajectory(markers), units="m")
        runner_config = {
            "subject_mass": subject_mass,
            "subject_height": subject_height,
            "time_range": list(time_range),
            "marker_tasks": reference["marker_tasks"],
            "coordinate_tasks": reference["coordinate_tasks"],
        }
        (staged / "runner_config.json").write_text(json.dumps(runner_config, indent=2, sort_keys=True) + "\n")
        runner_path = staged / "run_marker_placer.py"
        runner_path.write_text(_MARKER_PLACER_RUNNER)
        result = subprocess.run(
            [sys.executable, runner_path.name],
            cwd=staged,
            capture_output=True,
            text=True,
            check=False,
        )
        runner_log = staged / "runner_output.log"
        runner_log.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
        if result.returncode != 0:
            if "No module named 'opensim'" in result.stderr:
                raise ImportError("official marker placement requires `uv run --with opensim==4.6 ...`")
            raise RuntimeError(f"official OpenSim MarkerPlacer failed:\n{result.stderr[-2000:]}")
        opensim_log = staged / "opensim.log"
        log = result.stdout + result.stderr
        if opensim_log.is_file():
            log += opensim_log.read_text(errors="replace")
        matches = re.findall(r"marker error: RMS = ([^,]+), max = ([^ ]+)", log)
        if not matches:
            raise ValueError("official MarkerPlacer log did not report marker error")
        marker_rms, marker_max = (float(value) for value in matches[-1])
        placed_model = staged / "placed_subject.osim"
        motion = staged / "static_pose.mot"
        marker_set = staged / "adjusted_markers.xml"
        setup_path = staged / "marker_placer_setup.xml"
        if marker_rms > max_marker_rms or marker_max > max_marker_error:
            raise ValueError(
                f"official MarkerPlacer failed the engineering gate: RMS {marker_rms:.6f} > {max_marker_rms:.6f} "
                f"or max {marker_max:.6f} > {max_marker_error:.6f} m"
            )
        version_match = re.search(r"OPENSIM_VERSION=(.+)", result.stdout)
        manifest = {
            "schema_version": "gait2354_official_marker_placement_1",
            "method_reference": {
                "name": "official OpenSim MarkerPlacer",
                "version": version_match.group(1).strip() if version_match else "unknown",
                "setup": reference["opensim_scale_setup_reference"],
            },
            "source": {
                "c3d_file": markers.source_file,
                "c3d_sha256": markers.source_sha256,
                "scaled_model_file": model_source.name,
                "scaled_model_sha256": _sha256(model_source),
            },
            "time_range_s": list(time_range),
            "qc": {
                "passed": True,
                "marker_rms_m": marker_rms,
                "marker_rms_limit_m": max_marker_rms,
                "marker_max_m": marker_max,
                "marker_max_limit_m": max_marker_error,
            },
            "outputs": {
                path.name: _sha256(path)
                for path in (placed_model, motion, marker_set, setup_path, marker_file, runner_log)
            },
        }
        if opensim_log.is_file():
            manifest["outputs"][opensim_log.name] = _sha256(opensim_log)
        manifest_path = staged / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.rename(staged, root)
    return MarkerPlacementReference(
        root,
        root / placed_model.name,
        root / motion.name,
        root / marker_set.name,
        root / manifest_path.name,
        marker_rms,
        marker_max,
        max_marker_rms,
        max_marker_error,
    )
