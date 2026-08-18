# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Subject scaling for the OpenSim port.

Port of the OpenSim ``ScaleTool`` workflow (``GenericModelMaker`` ->
``ModelScaler`` -> ``MarkerPlacer``) so a generic ``.osim`` model can be fit to a
subject from a static motion-capture trial:

1. :func:`read_c3d` loads a raw ``.c3d`` static trial into a
   :class:`~newton.opensim.MarkerData` (lab frame -> OpenSim ground,
   mm -> m, occlusions -> ``NaN``).
2. :func:`apply_marker_assignment` maps the experimental marker labels onto the
   model's ``<Marker>`` names, discarding labels the model does not use.
3. :class:`ModelScaler` computes per-body scale factors from a
   :class:`MeasurementSet` of marker-pair distances (and/or manual overrides) and
   writes a geometrically scaled ``.osim`` (segments, joint frames, markers,
   wrap surfaces, muscle path points, mass, and inertia).
4. :class:`MarkerPlacer` runs a static-pose inverse-kinematics solve on the
   scaled model to move the model markers onto the subject and record the static
   pose.

:class:`ScaleTool` orchestrates the three stages. The inertia scaling reproduces
OpenSim's ``Body::scaleInertialProperties`` (a solid-ellipsoid second-moment
transform) and the mass handling reproduces ``ModelScaler`` (optional
``preserve_mass_distribution`` with a target subject mass).

.. note::

    ``read_c3d`` requires the optional ``ezc3d`` dependency (``newton[opensim]``).
    The ``.osim`` rewriter currently targets the OpenSim 3.x (``Version < 30000``)
    document layout used by the ``gait2354`` family.
"""

from __future__ import annotations

import difflib
import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .ik import solve_marker_ik
from .kinematics import ForwardKinematics
from .mocap import MarkerData, write_trc
from .parser import parse_osim
from .types import OsimModel

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Stage 1: raw C3D -> MarkerData
# ---------------------------------------------------------------------------
_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _axis_vector(spec: str) -> np.ndarray:
    """Parse a signed axis label such as ``"+Z"``, ``"z"``, or ``"-Y"`` into a unit vector."""
    text = spec.strip().upper()
    sign = 1.0
    if text and text[0] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    if text not in _AXES:
        raise ValueError(f"invalid axis '{spec}'; expected one of X, Y, Z with an optional sign")
    return sign * np.array(_AXES[text], dtype=float)


def lab_to_opensim_rotation(up_axis: str, forward_axis: str) -> np.ndarray:
    """Rotation mapping lab-frame coordinates to the OpenSim ground frame.

    The OpenSim ground frame is X anterior (forward), Y superior (up), Z to the
    subject's right. Given the lab-frame axes that point up and forward, this
    returns ``R`` such that ``p_opensim = p_lab @ R.T`` (and ``p_lab = p_opensim @ R``).

    Args:
        up_axis: Lab axis pointing up, e.g. ``"+Z"``.
        forward_axis: Lab axis pointing in the subject's forward direction, e.g. ``"-Y"``.

    Returns:
        A ``(3, 3)`` orthonormal rotation matrix.
    """
    up = _axis_vector(up_axis)
    up = up / np.linalg.norm(up)
    fwd = _axis_vector(forward_axis)
    fwd = fwd - np.dot(fwd, up) * up  # project onto the horizontal plane
    if np.linalg.norm(fwd) < 1e-9:
        raise ValueError("forward_axis must not be parallel to up_axis")
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    return np.array([fwd, up, right])


def read_c3d(
    path: str | os.PathLike,
    *,
    lab_frame: str = "vicon_zup",
    up_axis: str | None = None,
    forward_axis: str | None = None,
    strip_prefix: bool = True,
) -> MarkerData:
    """Read a ``.c3d`` motion-capture file into a :class:`MarkerData`.

    Args:
        path: Path to the ``.c3d`` file.
        lab_frame: Named lab-axis convention of the source data. ``"vicon_zup"``
            maps a Vicon Z-up lab (x forward, y left, z up) to the OpenSim Y-up
            ground; ``"opensim"`` leaves the axes unchanged. Ignored when both
            ``up_axis`` and ``forward_axis`` are given.
        up_axis: Lab axis pointing up (e.g. ``"+Z"``). Overrides ``lab_frame``.
        forward_axis: Lab axis pointing in the subject's forward direction
            (e.g. ``"-Y"``). Overrides ``lab_frame``.
        strip_prefix: Drop a ``Subject:MARKER`` label prefix so names match the
            model ``<Marker>`` names.

    Returns:
        Marker trajectories in the OpenSim ground frame [m], with occluded
        samples set to ``NaN``.

    Raises:
        ImportError: If the optional ``ezc3d`` dependency is not installed.
    """
    try:
        import ezc3d  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("read_c3d requires the optional 'ezc3d' dependency (install newton[opensim]).") from exc

    c = ezc3d.c3d(str(path))
    pt = c["parameters"]["POINT"]
    labels = list(pt["LABELS"]["value"])
    rate = float(pt["RATE"]["value"][0])
    first_frame = int(pt["FIRST_FRAME"]["value"][0]) if "FIRST_FRAME" in pt else 1

    pts = np.asarray(c["data"]["points"])[:3]  # (3, M, F)
    P = np.transpose(pts, (2, 1, 0)).astype(float)  # (F, M, 3), mm
    resid = np.asarray(c["data"]["meta_points"]["residuals"])[0].T  # (F, M)
    occ = (resid < 0) | np.all(P == 0.0, axis=-1)
    P[occ] = np.nan

    if up_axis is not None and forward_axis is not None:
        P = P @ lab_to_opensim_rotation(up_axis, forward_axis).T
    elif lab_frame == "vicon_zup":
        P = P[..., [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    elif lab_frame != "opensim":
        raise ValueError(f"unknown lab_frame '{lab_frame}'")
    P = P * 0.001  # mm -> m

    names = [(l.split(":")[-1].strip() if strip_prefix else l.strip()) for l in labels]
    times = (first_frame - 1 + np.arange(P.shape[0])) / rate
    return MarkerData(times=times, marker_names=names, data=P, rate=rate, units="mm")


# ---------------------------------------------------------------------------
# Stage 2: experimental labels -> model marker names
# ---------------------------------------------------------------------------
def synthesize_markers(
    markers: MarkerData,
    spec: dict[str, tuple[str, ...]] | dict[str, list[str]],
    *,
    min_present: int = 2,
) -> MarkerData:
    """Append virtual markers, each the per-frame centroid of existing labels.

    Use this to build derived landmarks a model expects but the capture only
    provides as a cluster, e.g. the sacrum from the PSIS pair or the head vertex
    from the head cluster (see :data:`GAIT2354_VIRTUAL_MARKERS`). The new label is
    the mean of its available source markers (ignoring occluded ``NaN`` samples),
    so naming it after the model ``<Marker>`` lets it match by name in
    :func:`suggest_marker_assignment`.

    Args:
        markers: Source marker trajectories.
        spec: Mapping ``new_label -> source_labels`` to average.
        min_present: Minimum number of source labels that must be present for the
            virtual marker to be created.

    Returns:
        A new :class:`MarkerData` with the virtual markers appended (or the input
        unchanged if none could be built).
    """
    names = list(markers.marker_names)
    index = {n: i for i, n in enumerate(names)}
    new_names: list[str] = []
    new_cols: list[np.ndarray] = []
    for label, sources in spec.items():
        if label in index:
            continue  # do not overwrite an existing marker
        present = [index[src] for src in sources if src in index]
        if len(present) < min(min_present, len(tuple(sources))):
            continue
        new_names.append(label)
        new_cols.append(np.nanmean(markers.data[:, present, :], axis=1))
    if not new_names:
        return markers
    data = np.concatenate([markers.data, np.stack(new_cols, axis=1)], axis=1)
    return MarkerData(
        times=markers.times,
        marker_names=names + new_names,
        data=data,
        rate=markers.rate,
        units=markers.units,
    )


def apply_marker_assignment(markers: MarkerData, mapping: dict[str, str]) -> MarkerData:
    """Rename experimental markers to model marker names, dropping unmapped ones.

    Args:
        markers: Experimental marker trajectories.
        mapping: Experimental-label -> model-marker-name. Labels absent from the
            mapping (or mapped to a falsy value) are discarded, so a full-body
            capture can drive a partial (e.g. single-leg) model.

    Returns:
        A new :class:`MarkerData` whose columns are the assigned model markers.
    """
    cols: list[int] = []
    out_names: list[str] = []
    for exp, model_name in mapping.items():
        if not model_name or exp not in markers.marker_names:
            continue
        cols.append(markers.marker_names.index(exp))
        out_names.append(model_name)
    data = markers.data[:, cols, :] if cols else markers.data[:, :0, :]
    return MarkerData(times=markers.times, marker_names=out_names, data=data, rate=markers.rate, units=markers.units)


def suggest_marker_assignment(
    exp_names: list[str], model_marker_names: list[str], *, aliases: dict[str, str] | None = None
) -> dict[str, str | None]:
    """Suggest an experimental-label -> model-marker mapping (alias + fuzzy match).

    Args:
        exp_names: Experimental marker labels (prefix stripped).
        model_marker_names: Model ``<Marker>`` names.
        aliases: Optional explicit ``exp -> model`` hints taking priority.

    Returns:
        A mapping with ``None`` where no confident match was found (edit in the GUI).
    """
    aliases = aliases or {}
    model_set = set(model_marker_names)
    out: dict[str, str | None] = {}
    for e in exp_names:
        if e in aliases and aliases[e] in model_set:
            out[e] = aliases[e]
            continue
        if e in model_set:
            out[e] = e
            continue
        m = difflib.get_close_matches(e, model_marker_names, n=1, cutoff=0.7)
        out[e] = m[0] if m else None
    return out


# ---------------------------------------------------------------------------
# Stage 3: measurement-based scale factors (OpenSim MeasurementSet)
# ---------------------------------------------------------------------------
@dataclass
class MarkerPair:
    """An ordered pair of model markers whose distance defines a measurement."""

    marker1: str
    marker2: str


@dataclass
class Measurement:
    """One OpenSim ``Measurement``: marker-pair distances scaling a set of bodies.

    Attributes:
        name: Measurement label.
        marker_pairs: Model marker pairs whose experimental/model distance ratios
            are averaged into this measurement's scale value.
        body_scales: Bodies scaled by this measurement, each with the axes
            (subset of ``"XYZ"``) the value applies to.
        apply: Whether the measurement is used.
    """

    name: str
    marker_pairs: list[MarkerPair]
    body_scales: dict[str, str]  # body -> axes, e.g. {"femur_r": "XYZ"}
    apply: bool = True


MeasurementSet = list[Measurement]
ScaleFactorSet = dict[str, Vec3]


def _averaged_positions(markers: MarkerData, time_range: tuple[float, float] | None) -> dict[str, np.ndarray]:
    """Average each marker over ``time_range`` (or the whole trial), ignoring NaN."""
    t = markers.times
    if time_range is None:
        sel = slice(None)
    else:
        lo, hi = time_range
        sel = (t >= lo) & (t <= hi)
    mean = np.nanmean(markers.data[sel], axis=0)  # (M, 3)
    return {n: mean[i] for i, n in enumerate(markers.marker_names)}


def _model_marker_positions(model: OsimModel) -> dict[str, np.ndarray]:
    """Model marker positions in ground at the default (zero) pose."""
    fk = ForwardKinematics(model)
    return {k: np.asarray(v) for k, v in fk.marker_positions(np.zeros(fk.ncoord)).items()}


def measurement_scale_value(
    meas: Measurement, exp_pos: dict[str, np.ndarray], model_pos: dict[str, np.ndarray]
) -> float | None:
    """Average experimental/model marker-pair distance ratio for one measurement."""
    ratios: list[float] = []
    for pair in meas.marker_pairs:
        a, b = pair.marker1, pair.marker2
        if a in exp_pos and b in exp_pos and a in model_pos and b in model_pos:
            d_exp = float(np.linalg.norm(exp_pos[a] - exp_pos[b]))
            d_mdl = float(np.linalg.norm(model_pos[a] - model_pos[b]))
            if np.isfinite(d_exp) and d_mdl > 1e-9:
                ratios.append(d_exp / d_mdl)
    if not ratios:
        return None
    return float(np.mean(ratios))


class ModelScaler:
    """Compute per-body scale factors and write a scaled ``.osim`` (OpenSim ``ModelScaler``).

    Args:
        model: The generic (parsed) model to scale.
        model_path: Path to the generic ``.osim`` document to rewrite.
    """

    def __init__(self, model: OsimModel, model_path: str | os.PathLike):
        self.model = model
        self.model_path = str(model_path)
        self.body_names = [b.name for b in model.bodies]

    def compute_scale_factors(
        self,
        static_markers: MarkerData,
        measurements: MeasurementSet,
        *,
        manual_scales: dict[str, Vec3] | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> ScaleFactorSet:
        """Return per-body ``(sx, sy, sz)`` factors from measurements and overrides.

        Args:
            static_markers: Assigned static-trial markers (model marker names).
            measurements: The measurement set defining marker-pair-based scaling.
            manual_scales: Optional ``body -> (sx, sy, sz)`` overrides applied last.
            time_range: Static-trial window to average, or the whole trial.

        Returns:
            ``body -> (sx, sy, sz)``; bodies with no measurement default to 1.0.
        """
        exp_pos = _averaged_positions(static_markers, time_range)
        model_pos = _model_marker_positions(self.model)
        factors: dict[str, list[float | None]] = {b: [None, None, None] for b in self.body_names}

        for meas in measurements:
            if not meas.apply:
                continue
            value = measurement_scale_value(meas, exp_pos, model_pos)
            if value is None:
                warnings.warn(f"Measurement '{meas.name}' has no usable marker pairs; skipped.", stacklevel=2)
                continue
            for body, axes in meas.body_scales.items():
                if body not in factors:
                    continue
                for ax in axes.upper():
                    factors[body]["XYZ".index(ax)] = value

        out: ScaleFactorSet = {}
        for b, comps in factors.items():
            out[b] = tuple(1.0 if c is None else float(c) for c in comps)  # type: ignore[assignment]
        for b, s in (manual_scales or {}).items():
            out[b] = (float(s[0]), float(s[1]), float(s[2]))
        return out

    def scale(
        self,
        scale_factors: ScaleFactorSet,
        out_path: str | os.PathLike,
        *,
        preserve_mass_distribution: bool = True,
        subject_mass: float | None = None,
    ) -> str:
        """Write a geometrically and inertially scaled ``.osim`` to ``out_path``.

        Args:
            scale_factors: Per-body ``(sx, sy, sz)`` factors.
            out_path: Destination ``.osim`` path.
            preserve_mass_distribution: Keep the generic mass distribution and
                scale the total to ``subject_mass`` (OpenSim default); otherwise
                scale each segment mass by its volume factor before normalizing.
            subject_mass: Measured subject mass [kg] used to normalize total mass.
                If ``None``, masses are left at their geometry-scaled values.

        Returns:
            ``out_path``.
        """
        _scale_osim_document(
            self.model_path,
            scale_factors,
            str(out_path),
            preserve_mass_distribution=preserve_mass_distribution,
            subject_mass=subject_mass,
        )
        return str(out_path)


# ---------------------------------------------------------------------------
# .osim document rewriting (OpenSim 3.x layout)
# ---------------------------------------------------------------------------
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

    # Display geometry scale factors (VisibleObject and each DisplayGeometry).
    for sf in body.iter("scale_factors"):
        if (sf.text or "").split():
            _set_vec(sf, [v * f for v, f in zip(_txt_vec(sf), s, strict=False)])

    # Joint owned by this body: location is in the child (this) frame; the
    # location_in_parent is in the parent frame.
    for joint in body.iter():
        if joint.tag.endswith("Joint") and joint.tag != "Joint":
            loc = joint.find("location")
            if loc is not None and (loc.text or "").split():
                _set_vec(loc, [v * f for v, f in zip(_txt_vec(loc), s, strict=False)])
            lip = joint.find("location_in_parent")
            pb = joint.find("parent_body")
            if lip is not None and pb is not None and (lip.text or "").split():
                pf = parent_factors.get((pb.text or "").strip(), (1.0, 1.0, 1.0))
                _set_vec(lip, [v * f for v, f in zip(_txt_vec(lip), pf, strict=False)])

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
            loc = pp.find("location")
            if bd is not None and loc is not None and (loc.text or "").split():
                s = body_of((bd.text or "").strip())
                _set_vec(loc, [v * f for v, f in zip(_txt_vec(loc), s, strict=False)])

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


# ---------------------------------------------------------------------------
# Stage 4: MarkerPlacer (static-pose IK)
# ---------------------------------------------------------------------------
@dataclass
class MarkerPlacementResult:
    """Result of the marker-placement stage.

    Attributes:
        model_path: Path to the placed subject model.
        coordinates: Static-pose coordinate values (name -> value, native units).
        marker_rms: Static-pose RMS marker error [m].
        marker_max: Static-pose maximum marker error [m].
    """

    model_path: str
    coordinates: dict[str, float]
    marker_rms: float
    marker_max: float


class MarkerPlacer:
    """Place model markers on the subject via a static-pose IK solve.

    Args:
        model_path: Path to the scaled subject ``.osim``.
        marker_weights: Optional per-marker IK weights (OpenSim ``IKTaskSet``).
    """

    def __init__(self, model_path: str | os.PathLike, marker_weights: dict[str, float] | None = None):
        self.model_path = str(model_path)
        self.marker_weights = marker_weights

    def run(
        self,
        static_markers: MarkerData,
        out_path: str | os.PathLike,
        *,
        time_range: tuple[float, float] | None = None,
        move_markers: bool = True,
    ) -> MarkerPlacementResult:
        """Solve the static pose, optionally move markers, and write the model.

        Args:
            static_markers: Assigned static-trial markers (model marker names).
            out_path: Destination ``.osim`` for the placed model.
            time_range: Static window to average into a single pose.
            move_markers: If True, relocate each model ``<Marker>`` to the
                subject's averaged marker position expressed in its body frame
                (OpenSim ``MarkerPlacer`` behavior).

        Returns:
            The :class:`MarkerPlacementResult`.
        """
        avg = _averaged_positions(static_markers, time_range)
        one = MarkerData(
            times=np.array([0.0]),
            marker_names=list(static_markers.marker_names),
            data=np.stack([np.array([avg[n] for n in static_markers.marker_names])], axis=0),
            rate=static_markers.rate,
            units=static_markers.units,
        )
        model = parse_osim(self.model_path)
        res = solve_marker_ik(model, one, marker_weights=self.marker_weights)
        q = {name: float(res.values[0, i]) for i, name in enumerate(res.coordinate_names)}

        tree = ET.parse(self.model_path)
        root = tree.getroot()
        if move_markers:
            fk = ForwardKinematics(model)
            qv = np.array([q.get(n, 0.0) for n in fk.coordinate_names])
            xf = fk.body_transforms(qv)  # body -> 4x4 in ground
            for marker in root.iter("Marker"):
                name = marker.get("name")
                bd = marker.find("body")
                loc = marker.find("location")
                if name in avg and bd is not None and loc is not None:
                    body = (bd.text or "").strip()
                    if body in xf and np.all(np.isfinite(avg[name])):
                        T = np.asarray(xf[body])
                        local = np.linalg.inv(T)[:3, :3] @ avg[name] + np.linalg.inv(T)[:3, 3]
                        _set_vec(loc, local)
        tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)

        return MarkerPlacementResult(
            model_path=str(out_path),
            coordinates=q,
            marker_rms=float(res.marker_rms[0]),
            marker_max=float(res.marker_max[0]),
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class ScaleResult:
    """Result of the full scale workflow.

    Attributes:
        scaled_model_path: The placed, subject-specific ``.osim`` path.
        scale_factors: Per-body ``(sx, sy, sz)`` factors that were applied.
        static_rms: Static-pose RMS marker error on the scaled model [m].
        static_max: Static-pose maximum marker error on the scaled model [m].
        static_trc: Path to the static-trial ``.trc`` written for reference.
    """

    scaled_model_path: str
    scale_factors: ScaleFactorSet
    static_rms: float
    static_max: float
    static_trc: str


class ScaleTool:
    """OpenSim ``ScaleTool``: fit a generic model to a subject static trial.

    Args:
        model_path: Path to the generic ``.osim`` model.
        measurements: Measurement set defining per-body marker-pair scaling.
        marker_weights: Optional IK weights for the marker-placement stage.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        measurements: MeasurementSet,
        *,
        marker_weights: dict[str, float] | None = None,
    ):
        self.model_path = str(model_path)
        self.measurements = measurements
        self.marker_weights = marker_weights

    def run(
        self,
        static_markers: MarkerData,
        out_path: str | os.PathLike,
        *,
        subject_mass: float | None = None,
        manual_scales: dict[str, Vec3] | None = None,
        preserve_mass_distribution: bool = True,
        time_range: tuple[float, float] | None = None,
        move_markers: bool = True,
    ) -> ScaleResult:
        """Run ModelScaler + MarkerPlacer and write the subject model.

        Args:
            static_markers: Assigned static-trial markers (model marker names).
            out_path: Destination ``.osim`` for the subject model.
            subject_mass: Measured subject mass [kg] for mass normalization.
            manual_scales: Optional ``body -> (sx, sy, sz)`` overrides.
            preserve_mass_distribution: See :meth:`ModelScaler.scale`.
            time_range: Static-trial window to average.
            move_markers: Relocate model markers onto the subject.

        Returns:
            The :class:`ScaleResult`.
        """
        model = parse_osim(self.model_path)
        scaler = ModelScaler(model, self.model_path)
        factors = scaler.compute_scale_factors(
            static_markers, self.measurements, manual_scales=manual_scales, time_range=time_range
        )
        scaled_tmp = str(out_path) + ".scaled.tmp.osim"
        scaler.scale(
            factors,
            scaled_tmp,
            preserve_mass_distribution=preserve_mass_distribution,
            subject_mass=subject_mass,
        )

        placer = MarkerPlacer(scaled_tmp, marker_weights=self.marker_weights)
        placed = placer.run(static_markers, out_path, time_range=time_range, move_markers=move_markers)

        trc_path = str(out_path).rsplit(".", 1)[0] + "_static.trc"
        write_trc(trc_path, static_markers)
        if os.path.exists(scaled_tmp):
            os.remove(scaled_tmp)

        return ScaleResult(
            scaled_model_path=placed.model_path,
            scale_factors=factors,
            static_rms=placed.marker_rms,
            static_max=placed.marker_max,
            static_trc=trc_path,
        )


# ---------------------------------------------------------------------------
# gait2354 defaults (OpenSim subject01 Scale setup)
# ---------------------------------------------------------------------------
#: Vicon Plug-in-Gait -> gait2354 model marker names (edit/extend in the GUI).
GAIT2354_VICON_ALIASES: dict[str, str] = {
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

#: gait2354 markers synthesized from raw C3D labels (derived landmark -> source labels
#: averaged per frame). ``V.Sacral`` is the PSIS midpoint; ``Top.Head`` the head-cluster
#: centroid. Feed to :func:`synthesize_markers` before assignment.
GAIT2354_VIRTUAL_MARKERS: dict[str, tuple[str, ...]] = {
    "V.Sacral": ("LPSI", "RPSI"),
    "Top.Head": ("LFHD", "RFHD", "LBHD", "RBHD"),
}


def gait2354_measurement_set() -> MeasurementSet:
    """Return a measurement set for the gait2354 model (OpenSim subject01 setup)."""

    def mp(a, b):
        return MarkerPair(a, b)

    return [
        Measurement("pelvis", [mp("R.ASIS", "L.ASIS")], {"pelvis": "XYZ"}),
        Measurement(
            "torso",
            [mp("R.Acromium", "L.Acromium"), mp("Sternum", "R.ASIS"), mp("Sternum", "L.ASIS")],
            {"torso": "XYZ"},
        ),
        Measurement("femur_r", [mp("R.ASIS", "R.Knee.Lat"), mp("R.Knee.Lat", "R.Knee.Med")], {"femur_r": "XYZ"}),
        Measurement("femur_l", [mp("L.ASIS", "L.Knee.Lat"), mp("L.Knee.Lat", "L.Knee.Med")], {"femur_l": "XYZ"}),
        Measurement("tibia_r", [mp("R.Knee.Lat", "R.Ankle.Lat"), mp("R.Ankle.Lat", "R.Ankle.Med")], {"tibia_r": "XYZ"}),
        Measurement("tibia_l", [mp("L.Knee.Lat", "L.Ankle.Lat"), mp("L.Ankle.Lat", "L.Ankle.Med")], {"tibia_l": "XYZ"}),
        Measurement("foot_r", [mp("R.Heel", "R.Toe.Tip")], {"talus_r": "XYZ", "calcn_r": "XYZ", "toes_r": "XYZ"}),
        Measurement("foot_l", [mp("L.Heel", "L.Toe.Tip")], {"talus_l": "XYZ", "calcn_l": "XYZ", "toes_l": "XYZ"}),
    ]
