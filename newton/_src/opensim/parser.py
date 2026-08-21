# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parser for OpenSim ``.osim`` (``OpenSimDocument``) XML into the Newton IR.

This targets the OpenSim 4.x component/socket schema (``Version`` >= 30000)
while remaining tolerant of common variations. The result is an
:class:`~newton.opensim.OsimModel` that downstream code (importer,
analyses) consumes without touching XML.

The parser is deliberately dependency-free (stdlib ``xml.etree`` + Python) so it
can run and be unit-tested without Warp or a GPU.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .types import (
    OsimActuator,
    OsimBody,
    OsimBushingForce,
    OsimContactForce,
    OsimContactGeometry,
    OsimCoordinate,
    OsimFrame,
    OsimGeometry,
    OsimJoint,
    OsimLigament,
    OsimMarker,
    OsimModel,
    OsimMuscle,
    OsimPathPoint,
    OsimPathSpring,
    OsimPointToPointSpring,
    OsimSpringGeneralizedForce,
    OsimTransform,
    OsimTransformAxis,
    OsimWrap,
    OsimWrapObject,
)

# OpenSim class names recognized as muscle-tendon actuators.
_MUSCLE_TYPES = {
    "Thelen2003Muscle",
    "Thelen2003Muscle_Deprecated",
    "Millard2012EquilibriumMuscle",
    "Millard2012AccelerationMuscle",
    "DeGrooteFregly2016Muscle",
    "RigidTendonMuscle",
    "Schutte1993Muscle",
    "Schutte1993Muscle_Deprecated",
    "McKibbenActuator",
}

# OpenSim class names recognized as (non-muscle) actuators.
_ACTUATOR_TYPES = {
    "CoordinateActuator",
    "PointActuator",
    "TorqueActuator",
    "BodyActuator",
    "ActivationCoordinateActuator",
}

_CONTACT_GEOM_TYPES = {"ContactSphere", "ContactHalfSpace", "ContactMesh"}
_CONTACT_FORCE_TYPES = {
    "SmoothSphereHalfSpaceForce",
    "HuntCrossleyForce",
    "ElasticFoundationForce",
}
_PATH_FORCE_TYPES = {"PathSpring"}
_LIGAMENT_TYPES = {"Ligament"}
_P2P_SPRING_TYPES = {"PointToPointSpring"}
_SPRING_GEN_FORCE_TYPES = {"SpringGeneralizedForce"}
_BUSHING_FORCE_TYPES = {"BushingForce"}

# Scalar parameter tags harvested verbatim for muscle-tendon models.
_MUSCLE_PARAM_TAGS = (
    "max_isometric_force",
    "optimal_fiber_length",
    "tendon_slack_length",
    "pennation_angle_at_optimal",
    "pennation_angle",
    "max_contraction_velocity",
    "Vmax",
    "Vmax0",
    "activation_time_constant",
    "deactivation_time_constant",
    "default_activation",
    "default_fiber_length",
    "fiber_damping",
    "tendon_strain_at_one_norm_force",
    "FmaxTendonStrain",
    "FmaxMuscleStrain",
    "KshapeActive",
    "KshapePassive",
    "Af",
    "Flen",
    "active_force_width_scale",
    "ignore_tendon_compliance",
    "ignore_activation_dynamics",
    "ignore_passive_fiber_force",
)


def _text(elem: ET.Element | None, default: str = "") -> str:
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def _floats(elem: ET.Element | None, n: int | None = None) -> list[float]:
    s = _text(elem)
    if not s:
        return [0.0] * n if n else []
    vals = [float(x) for x in s.replace(",", " ").split()]
    if n is not None and len(vals) < n:
        vals += [0.0] * (n - len(vals))
    return vals


def _float(elem: ET.Element | None, default: float = 0.0) -> float:
    s = _text(elem)
    try:
        return float(s)
    except ValueError:
        return default


def _bool(elem: ET.Element | None, default: bool = False) -> bool:
    s = _text(elem).lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return default


def _vec3(elem: ET.Element | None) -> tuple[float, float, float]:
    v = _floats(elem, 3)
    return (v[0], v[1], v[2])


def _leaf_name(path: str) -> str:
    """Return the final component of a socket/component path (``/bodyset/pelvis`` -> ``pelvis``)."""
    return path.strip().rstrip("/").split("/")[-1] if path else ""


def _find(elem: ET.Element, *tags: str) -> ET.Element | None:
    """Find the first direct child matching any of ``tags``."""
    for t in tags:
        c = elem.find(t)
        if c is not None:
            return c
    return None


def _objects(setelem: ET.Element | None) -> list[ET.Element]:
    """Return the ``<objects>`` children of an OpenSim ``*Set`` element."""
    if setelem is None:
        return []
    objs = setelem.find("objects")
    if objs is None:
        return list(setelem)
    return list(objs)


def _parse_transform(elem: ET.Element) -> OsimTransform:
    tr = _vec3(_find(elem, "translation"))
    orient = _vec3(_find(elem, "orientation"))
    return OsimTransform(translation=tr, orientation=orient)


def _parse_offset_frames(joint: ET.Element) -> dict[str, OsimFrame]:
    """Return owned ``PhysicalOffsetFrame`` components keyed by name."""
    frames: dict[str, OsimFrame] = {}
    frames_elem = joint.find("frames")
    if frames_elem is None:
        return frames
    for pof in frames_elem.findall("PhysicalOffsetFrame"):
        name = pof.get("name", "")
        parent = _leaf_name(_text(_find(pof, "socket_parent", "parent")))
        frames[name] = OsimFrame(name=name, parent=parent, transform=_parse_transform(pof))
    return frames


def _resolve_frame(socket_text: str, owned: dict[str, OsimFrame]) -> tuple[str, OsimTransform]:
    """Resolve a joint parent/child socket to ``(body_name, transform)``.

    ``socket_text`` may reference an owned ``PhysicalOffsetFrame`` (giving a
    body + offset) or reference a body directly (identity offset).
    """
    leaf = _leaf_name(socket_text)
    if leaf in owned:
        f = owned[leaf]
        return f.parent, f.transform
    return leaf, OsimTransform()


def _parse_geometry(frame_elem: ET.Element) -> list[OsimGeometry]:
    geoms: list[OsimGeometry] = []
    attached = frame_elem.find("attached_geometry")
    containers = [attached] if attached is not None else []
    for container in containers:
        for mesh in container.findall("Mesh"):
            app = mesh.find("Appearance")
            color = _vec3(_find(app, "color")) if app is not None else (1.0, 1.0, 1.0)
            opacity = _float(_find(app, "opacity"), 1.0) if app is not None else 1.0
            geoms.append(
                OsimGeometry(
                    name=mesh.get("name", ""),
                    mesh_file=_text(_find(mesh, "mesh_file")) or None,
                    scale_factors=_vec3(_find(mesh, "scale_factors")) or (1.0, 1.0, 1.0),
                    color=color,
                    opacity=opacity,
                    socket_frame=_text(_find(mesh, "socket_frame")) or None,
                )
            )
    return geoms


def _parse_body(elem: ET.Element) -> OsimBody:
    inertia_elem = _find(elem, "inertia")
    if inertia_elem is not None and _text(inertia_elem):
        inertia = _floats(inertia_elem, 6)
    else:
        # Legacy (<30000) models store the inertia as six scalar tags.
        inertia = [
            _float(_find(elem, "inertia_xx")),
            _float(_find(elem, "inertia_yy")),
            _float(_find(elem, "inertia_zz")),
            _float(_find(elem, "inertia_xy")),
            _float(_find(elem, "inertia_xz")),
            _float(_find(elem, "inertia_yz")),
        ]
    return OsimBody(
        name=elem.get("name", ""),
        mass=_float(_find(elem, "mass")),
        mass_center=_vec3(_find(elem, "mass_center")),
        inertia=(inertia[0], inertia[1], inertia[2], inertia[3], inertia[4], inertia[5]),
        geometry=_parse_geometry(elem),
    )


def _parse_coordinate(elem: ET.Element) -> OsimCoordinate:
    rng = _floats(_find(elem, "range"), 2) if _find(elem, "range") is not None else None
    return OsimCoordinate(
        name=elem.get("name", ""),
        motion_type=_text(_find(elem, "motion_type")) or "rotational",
        default_value=_float(_find(elem, "default_value")),
        default_speed=_float(_find(elem, "default_speed_value", "default_speed")),
        range=(rng[0], rng[1]) if rng else None,
        clamped=_bool(_find(elem, "clamped")),
        locked=_bool(_find(elem, "locked")),
    )


def _parse_function(fn: ET.Element) -> dict:
    """Structured parse of an OpenSim coordinate ``<function>`` body element.

    Returns a dict with a ``"type"`` key and the function\'s parameters. Nested
    functions (``MultiplierFunction``) are captured recursively under
    ``"inner"``. The result is consumed by
    ``build_function``.
    """
    tag = fn.tag
    if tag == "LinearFunction":
        return {"type": "LinearFunction", "coefficients": _floats(_find(fn, "coefficients"), 2)}
    if tag == "Constant":
        return {"type": "Constant", "value": _float(_find(fn, "value"))}
    if tag in ("SimmSpline", "NaturalCubicSpline", "GCVSpline"):
        return {"type": "SimmSpline", "x": _floats(_find(fn, "x")), "y": _floats(_find(fn, "y"))}
    if tag == "PiecewiseLinearFunction":
        return {"type": "PiecewiseLinearFunction", "x": _floats(_find(fn, "x")), "y": _floats(_find(fn, "y"))}
    if tag == "MultiplierFunction":
        inner_wrap = _find(fn, "function")
        inner: dict = {"type": "Constant", "value": 0.0}
        if inner_wrap is not None and len(list(inner_wrap)) > 0:
            inner = _parse_function(next(iter(inner_wrap)))
        return {"type": "MultiplierFunction", "scale": _float(_find(fn, "scale"), 1.0), "inner": inner}
    return {"type": tag}


def _parse_spatial_transform(elem: ET.Element) -> list[OsimTransformAxis]:
    st = elem.find("SpatialTransform")
    if st is None:
        return []
    axes: list[OsimTransformAxis] = []
    for ta in st.findall("TransformAxis"):
        coords_txt = _text(_find(ta, "coordinates"))
        coords = coords_txt.split() if coords_txt else []
        func_elem = _find(ta, "function")
        function_type = None
        function: dict = {}
        is_identity = True
        function_body = None
        if func_elem is not None and len(list(func_elem)) > 0:
            # Legacy serialization wraps the concrete function in <function>.
            function_body = next(iter(func_elem))
        else:
            # OpenSim 4.x writes the concrete Function directly and names the
            # component "function" (e.g. <SimmSpline name="function">).
            function_body = next(
                (child for child in ta if child.get("name") == "function" and child.tag != "function"),
                None,
            )
        if function_body is not None:
            function = _parse_function(function_body)
            function_type = function.get("type")
            constant_zero = function_type == "Constant" and function.get("value", 0.0) == 0.0
            is_identity = constant_zero and not coords
        elif coords:
            is_identity = False
        axes.append(
            OsimTransformAxis(
                axis=_vec3(_find(ta, "axis")),
                coordinates=coords,
                function_type=function_type,
                function=function,
                is_identity=is_identity,
            )
        )
    return axes


def _parse_joint(elem: ET.Element) -> OsimJoint:
    owned = _parse_offset_frames(elem)
    p_socket = _text(_find(elem, "socket_parent_frame"))
    c_socket = _text(_find(elem, "socket_child_frame"))
    parent_body, parent_xf = _resolve_frame(p_socket, owned)
    child_body, child_xf = _resolve_frame(c_socket, owned)

    coords_elem = elem.find("coordinates")
    coordinate_elements = _objects(coords_elem) if coords_elem is not None else []
    coordinates = [_parse_coordinate(coordinate) for coordinate in coordinate_elements]
    declared_motion = {
        coordinate.get("name", "") for coordinate in coordinate_elements if _find(coordinate, "motion_type") is not None
    }
    spatial_transform = _parse_spatial_transform(elem)

    # OpenSim 4.x Coordinate elements normally omit ``motion_type``. Infer it
    # from the joint's six-axis convention so translational states remain in
    # metres instead of being mistaken for degree-valued rotations. A rotational
    # coordinate may also drive coupled translation (e.g. the gait knee), so a
    # rotation-axis reference takes precedence.
    translational: set[str] = set()
    if elem.tag == "CustomJoint":
        rotational = {name for axis in spatial_transform[:3] for name in axis.coordinates}
        translational = {name for axis in spatial_transform[3:6] for name in axis.coordinates} - rotational
    elif elem.tag == "FreeJoint":
        translational.update(coordinate.name for coordinate in coordinates[3:6])
    elif elem.tag == "SliderJoint":
        translational.update(coordinate.name for coordinate in coordinates[:1])
    elif elem.tag == "PlanarJoint":
        translational.update(coordinate.name for coordinate in coordinates[1:3])
    for coordinate in coordinates:
        if coordinate.name not in declared_motion and coordinate.name in translational:
            coordinate.motion_type = "translational"

    return OsimJoint(
        name=elem.get("name", ""),
        type=elem.tag,
        parent_body=parent_body or "ground",
        child_body=child_body,
        parent_transform=parent_xf,
        child_transform=child_xf,
        coordinates=coordinates,
        spatial_transform=spatial_transform,
    )


def _parse_legacy_joint(je: ET.Element, child_body: str) -> OsimJoint:
    """Parse a legacy (<30000) inline ``<Joint>`` child owned by a ``Body``.

    Legacy joints store the parent by name (``<parent_body>``) and the joint
    frames as ``location_in_parent``/``orientation_in_parent`` (on the parent)
    and ``location``/``orientation`` (on the child), rather than 4.x sockets and
    ``PhysicalOffsetFrame`` components.
    """
    parent_body = _leaf_name(_text(_find(je, "parent_body"))) or "ground"
    parent_xf = OsimTransform(
        translation=_vec3(_find(je, "location_in_parent")),
        orientation=_vec3(_find(je, "orientation_in_parent")),
    )
    child_xf = OsimTransform(
        translation=_vec3(_find(je, "location")),
        orientation=_vec3(_find(je, "orientation")),
    )
    coords_elem = je.find("CoordinateSet")
    coordinates = [_parse_coordinate(c) for c in _objects(coords_elem)] if coords_elem is not None else []
    return OsimJoint(
        name=je.get("name", ""),
        type=je.tag,
        parent_body=parent_body,
        child_body=child_body,
        parent_transform=parent_xf,
        child_transform=child_xf,
        coordinates=coordinates,
        spatial_transform=_parse_spatial_transform(je),
    )


def _parse_path_point(elem: ET.Element) -> OsimPathPoint:
    body = _text(_find(elem, "body"))
    socket = _text(_find(elem, "socket_parent_frame"))
    if not body and socket:
        body = _leaf_name(socket)

    conditional_coordinate: str | None = None
    conditional_range: tuple[float, float] | None = None
    if elem.tag == "ConditionalPathPoint":
        conditional_coordinate = _leaf_name(_text(_find(elem, "coordinate"))) or None
        rng = _floats(_find(elem, "range"), 2)
        conditional_range = (rng[0], rng[1])

    moving: dict[str, tuple[str, str, dict]] | None = None
    if elem.tag == "MovingPathPoint":
        moving = {}
        for axis in ("x", "y", "z"):
            loc = _find(elem, f"{axis}_location")
            coord = _leaf_name(_text(_find(elem, f"{axis}_coordinate")))
            fn = next(iter(loc), None) if loc is not None else None
            if fn is None or not coord:
                continue
            spec = _parse_function(fn)
            moving[axis] = (coord, spec.get("type", fn.tag), spec)

    return OsimPathPoint(
        name=elem.get("name", ""),
        body=body,
        location=_vec3(_find(elem, "location")),
        type=elem.tag,
        socket_frame=socket or None,
        conditional_coordinate=conditional_coordinate,
        conditional_range=conditional_range,
        moving=moving,
    )


def _parse_geometry_path(elem: ET.Element) -> tuple[list[OsimPathPoint], list[OsimWrap]]:
    """Parse the ``GeometryPath`` of a path-based force into points and wraps."""
    gp = _find(elem, "GeometryPath")
    path_points: list[OsimPathPoint] = []
    wraps: list[OsimWrap] = []
    if gp is not None:
        pps = _find(gp, "PathPointSet")
        for pp in _objects(pps):
            path_points.append(_parse_path_point(pp))
        pws = _find(gp, "PathWrapSet")
        for w in _objects(pws):
            rng = _floats(_find(w, "range"), 2)
            wraps.append(
                OsimWrap(
                    wrap_object=_text(_find(w, "wrap_object")),
                    method=_text(_find(w, "method")) or "hybrid",
                    range=(int(rng[0]), int(rng[1])),
                )
            )
    return path_points, wraps


def _parse_path_spring(elem: ET.Element) -> OsimPathSpring:
    """Parse a ``PathSpring`` (path-based linear spring with dissipation)."""
    path_points, wraps = _parse_geometry_path(elem)
    return OsimPathSpring(
        name=elem.get("name", ""),
        path_points=path_points,
        wraps=wraps,
        resting_length=_float(_find(elem, "resting_length"), 0.0),
        stiffness=_float(_find(elem, "stiffness"), 0.0),
        dissipation=_float(_find(elem, "dissipation"), 0.0),
    )


def _parse_ligament(elem: ET.Element) -> OsimLigament:
    """Parse a ``Ligament`` (path-based force scaling a normalized force-length curve)."""
    path_points, wraps = _parse_geometry_path(elem)
    curve: dict = {"type": "Constant", "value": 0.0}
    flc = _find(elem, "force_length_curve")
    if flc is not None:
        children = list(flc)
        if children:
            curve = _parse_function(children[0])
    return OsimLigament(
        name=elem.get("name", ""),
        path_points=path_points,
        wraps=wraps,
        resting_length=_float(_find(elem, "resting_length"), 1.0),
        pcsa_force=_float(_find(elem, "pcsa_force"), 0.0),
        force_length_curve=curve,
    )


def _parse_point_to_point_spring(elem: ET.Element) -> OsimPointToPointSpring:
    """Parse a ``PointToPointSpring`` (linear spring between two body-fixed points)."""

    def _body(*tags: str) -> str:
        for tag in tags:
            txt = _text(_find(elem, tag))
            if txt:
                return _leaf_name(txt)
        return "ground"

    return OsimPointToPointSpring(
        name=elem.get("name", ""),
        body1=_body("body1", "socket_body1"),
        body2=_body("body2", "socket_body2"),
        point1=_vec3(_find(elem, "point1")),
        point2=_vec3(_find(elem, "point2")),
        stiffness=_float(_find(elem, "stiffness"), 0.0),
        rest_length=_float(_find(elem, "rest_length"), 0.0),
    )


def _parse_spring_generalized_force(elem: ET.Element) -> OsimSpringGeneralizedForce:
    """Parse a ``SpringGeneralizedForce`` (passive single-coordinate spring-damper)."""
    return OsimSpringGeneralizedForce(
        name=elem.get("name", ""),
        coordinate=_leaf_name(_text(_find(elem, "coordinate"))),
        stiffness=_float(_find(elem, "stiffness"), 0.0),
        rest_length=_float(_find(elem, "rest_length"), 0.0),
        viscosity=_float(_find(elem, "viscosity"), 0.0),
    )


def _parse_bushing_force(elem: ET.Element) -> OsimBushingForce:
    """Parse a ``BushingForce`` (6-DOF linear frame bushing).

    Supports the modern ``socket_frame1``/``socket_frame2`` + owned ``PhysicalOffsetFrame``
    layout and the legacy ``body_1``/``location_body_1``/``orientation_body_1`` layout.
    """
    owned = _parse_offset_frames(elem)

    def _frame(socket_tag: str, body_tag: str, loc_tag: str, orient_tag: str) -> tuple[str, OsimTransform]:
        socket = _text(_find(elem, socket_tag))
        if socket:
            return _resolve_frame(socket, owned)
        body = _leaf_name(_text(_find(elem, body_tag))) or "ground"
        return body, OsimTransform(translation=_vec3(_find(elem, loc_tag)), orientation=_vec3(_find(elem, orient_tag)))

    body1, tf1 = _frame("socket_frame1", "body_1", "location_body_1", "orientation_body_1")
    body2, tf2 = _frame("socket_frame2", "body_2", "location_body_2", "orientation_body_2")
    return OsimBushingForce(
        name=elem.get("name", ""),
        body1=body1,
        body2=body2,
        frame1_transform=tf1,
        frame2_transform=tf2,
        rotational_stiffness=_vec3(_find(elem, "rotational_stiffness")),
        translational_stiffness=_vec3(_find(elem, "translational_stiffness")),
        rotational_damping=_vec3(_find(elem, "rotational_damping")),
        translational_damping=_vec3(_find(elem, "translational_damping")),
    )


def _parse_muscle(elem: ET.Element) -> OsimMuscle:
    path_points, wraps = _parse_geometry_path(elem)

    params: dict[str, float] = {}
    for tag in _MUSCLE_PARAM_TAGS:
        e = elem.find(tag)
        if e is not None:
            txt = _text(e).lower()
            if txt in ("true", "false"):
                params[tag] = 1.0 if txt == "true" else 0.0
            else:
                params[tag] = _float(e)

    return OsimMuscle(
        name=elem.get("name", ""),
        type=elem.tag,
        path_points=path_points,
        wraps=wraps,
        params=params,
        min_control=_float(_find(elem, "min_control"), 0.0),
        max_control=_float(_find(elem, "max_control"), 1.0),
    )


def _parse_actuator(elem: ET.Element) -> OsimActuator:
    return OsimActuator(
        name=elem.get("name", ""),
        type=elem.tag,
        coordinate=_text(_find(elem, "coordinate")) or None,
        optimal_force=_float(_find(elem, "optimal_force"), 1.0),
        min_control=_float(_find(elem, "min_control"), -float("inf")),
        max_control=_float(_find(elem, "max_control"), float("inf")),
        body=_leaf_name(_text(_find(elem, "socket_frame", "body_name", "body", "bodyA"))) or None,
        body_b=_leaf_name(_text(_find(elem, "bodyB"))) or None,
        point=_vec3(_find(elem, "point")),
        point_is_global=_bool(_find(elem, "point_is_global"), False),
        direction=_vec3(_find(elem, "direction", "axis")),
        force_is_global=_bool(_find(elem, "force_is_global", "torque_is_global", "spatial_force_is_global"), True),
    )


def _parse_contact_geometry(elem: ET.Element) -> OsimContactGeometry:
    body = _leaf_name(_text(_find(elem, "socket_frame", "body_name", "body")))
    return OsimContactGeometry(
        name=elem.get("name", ""),
        type=elem.tag,
        body=body or "ground",
        location=_vec3(_find(elem, "location")),
        orientation=_vec3(_find(elem, "orientation")),
        radius=_float(_find(elem, "radius")),
        mesh_file=_text(_find(elem, "filename", "mesh_file")) or None,
    )


# Per-surface material property tags shared by every compliant contact force.
_CONTACT_MATERIAL_TAGS = (
    "stiffness",
    "dissipation",
    "static_friction",
    "dynamic_friction",
    "viscous_friction",
)


def _parse_contact_force(elem: ET.Element) -> OsimContactForce:
    """Parse a compliant contact force element into :class:`OsimContactForce`.

    ``SmoothSphereHalfSpaceForce`` stores every material property directly on the
    element and connects its geometry through ``socket_sphere`` /
    ``socket_half_space``. ``HuntCrossleyForce`` and ``ElasticFoundationForce``
    carry a ``<contact_parameters>`` set of per-geometry material properties plus
    an element-level ``<transition_velocity>``; the geometry names live in each
    block's ``<geometry>`` list.
    """
    force = OsimContactForce(name=elem.get("name", ""), type=elem.tag)

    # Element-level scalar properties (all material properties for the smooth
    # model; transition velocity / smoothing for the others).
    for tag in (
        *_CONTACT_MATERIAL_TAGS,
        "transition_velocity",
        "constant_contact_force",
        "hertz_smoothing",
        "hunt_crossley_smoothing",
    ):
        e = elem.find(tag)
        if e is not None:
            force.params[tag] = _float(e)

    force.sphere = _leaf_name(_text(_find(elem, "socket_sphere"))) or None
    force.half_space = _leaf_name(_text(_find(elem, "socket_half_space"))) or None
    if force.sphere:
        force.geometries.append(force.sphere)
    if force.half_space:
        force.geometries.append(force.half_space)

    # Per-surface parameters (Hunt-Crossley / elastic foundation).
    cps = elem.find("contact_parameters")
    for block in _objects(cps):
        geom_names = _text(_find(block, "geometry")).split()
        mat = {t: _float(_find(block, t)) for t in _CONTACT_MATERIAL_TAGS if _find(block, t) is not None}
        for name in geom_names:
            leaf = _leaf_name(name)
            if leaf and leaf not in force.geometries:
                force.geometries.append(leaf)
            if leaf:
                force.surface_params[leaf] = dict(mat)

    return force


def _iter_named(setelem: ET.Element | None, predicate) -> list[ET.Element]:
    return [e for e in _objects(setelem) if predicate(e.tag)]


_WRAP_TYPES = {"WrapSphere", "WrapCylinder", "WrapEllipsoid", "WrapTorus"}


def _parse_wrap_object(elem: ET.Element, body: str) -> OsimWrapObject:
    """Parse a ``WrapObjectSet`` entry (``WrapSphere``, ``WrapCylinder``, ...)."""
    active = _find(elem, "active")
    dims = _find(elem, "dimensions")
    return OsimWrapObject(
        name=elem.get("name", ""),
        type=elem.tag,
        body=body,
        translation=_vec3(_find(elem, "translation")),
        rotation=_vec3(_find(elem, "xyz_body_rotation")),
        radius=_float(_find(elem, "radius")),
        length=_float(_find(elem, "length")),
        dimensions=_vec3(dims) if dims is not None else (0.0, 0.0, 0.0),
        inner_radius=_float(_find(elem, "inner_radius")),
        outer_radius=_float(_find(elem, "outer_radius")),
        quadrant=_text(_find(elem, "quadrant")) or "all",
        active=_bool(active, True) if active is not None else True,
    )


def parse_osim(source: str | os.PathLike) -> OsimModel:
    """Parse an OpenSim ``.osim`` document into an :class:`OsimModel`.

    Args:
        source: Path to a ``.osim`` file, or a string containing the XML document.

    Returns:
        The parsed model IR.

    Raises:
        ValueError: If the document does not contain a ``<Model>`` element.
    """
    text: str
    if isinstance(source, (str, os.PathLike)) and os.path.exists(str(source)):
        with open(source, encoding="utf-8") as f:
            text = f.read()
    else:
        text = str(source)

    # OpenSim serializes nested concrete classes with a scoped tag such as
    # ``HuntCrossleyForce::ContactParameters``; the ``::`` is not valid in an XML
    # element name for the stdlib parser, so normalize it to ``__``.
    if "::" in text:
        text = text.replace("::", "__")

    doc = ET.fromstring(text)
    version = 0
    try:
        version = int(doc.get("Version", "0"))
    except ValueError:
        version = 0

    model_elem = doc.find("Model") if doc.tag != "Model" else doc
    if model_elem is None:
        raise ValueError("No <Model> element found in OpenSim document")

    model = OsimModel(name=model_elem.get("name", "model"), version=version)

    grav = _find(model_elem, "gravity")
    if grav is not None:
        model.gravity = _vec3(grav)

    # Bodies (4.x BodySet, or legacy BodySet under DynamicsEngine).
    bodyset = model_elem.find("BodySet")
    if bodyset is None:
        for de in model_elem.iter("BodySet"):
            bodyset = de
            break
    for b in _objects(bodyset):
        # ``ground`` appears as a Body in legacy (<30000) BodySets; 4.x keeps it
        # separate. Exclude it either way so ``bodies`` are true rigid bodies.
        if b.tag == "Body" and b.get("name") != "ground":
            model.bodies.append(_parse_body(b))

    # Joints (4.x JointSet).
    jointset = model_elem.find("JointSet")
    if jointset is None:
        for js in model_elem.iter("JointSet"):
            jointset = js
            break
    for j in _objects(jointset):
        if j.tag.endswith("Joint"):
            model.joints.append(_parse_joint(j))

    # Legacy (<30000): each Body owns the ``<Joint>`` connecting it to its parent.
    if not model.joints:
        for b in _objects(bodyset):
            if b.tag != "Body":
                continue
            jwrap = b.find("Joint")
            if jwrap is None:
                continue
            children = list(jwrap)
            if not children:
                continue
            model.joints.append(_parse_legacy_joint(children[0], b.get("name", "")))

    # Forces (muscles, actuators, contact forces) live in ForceSet + <components>.
    force_containers = []
    fs = model_elem.find("ForceSet")
    if fs is not None:
        force_containers.append(fs)
    comps = model_elem.find("components")
    if comps is not None:
        force_containers.append(comps)

    for container in force_containers:
        for e in _objects(container):
            tag = e.tag
            if tag in _MUSCLE_TYPES:
                model.muscles.append(_parse_muscle(e))
            elif tag in _ACTUATOR_TYPES:
                model.actuators.append(_parse_actuator(e))
            elif tag in _CONTACT_FORCE_TYPES:
                model.contact_forces.append(_parse_contact_force(e))
            elif tag in _PATH_FORCE_TYPES:
                model.path_springs.append(_parse_path_spring(e))
            elif tag in _LIGAMENT_TYPES:
                model.ligaments.append(_parse_ligament(e))
            elif tag in _P2P_SPRING_TYPES:
                model.point_to_point_springs.append(_parse_point_to_point_spring(e))
            elif tag in _SPRING_GEN_FORCE_TYPES:
                model.spring_generalized_forces.append(_parse_spring_generalized_force(e))
            elif tag in _BUSHING_FORCE_TYPES:
                model.bushing_forces.append(_parse_bushing_force(e))

    # Contact geometry.
    cgs = model_elem.find("ContactGeometrySet")
    for e in _objects(cgs):
        if e.tag in _CONTACT_GEOM_TYPES:
            model.contact_geometry.append(_parse_contact_geometry(e))

    # Markers.
    ms = model_elem.find("MarkerSet")
    if ms is None:
        ms = next(model_elem.iter("MarkerSet"), None)
    for e in _objects(ms):
        if e.tag == "Marker":
            body = _leaf_name(_text(_find(e, "socket_parent_frame", "body")))
            model.markers.append(OsimMarker(name=e.get("name", ""), body=body, location=_vec3(_find(e, "location"))))

    # Wrap surfaces (``WrapObjectSet`` owned by each body/ground).
    for b in _objects(bodyset):
        if b.tag not in ("Body", "Ground"):
            continue
        bname = b.get("name", "")
        for w in _objects(b.find("WrapObjectSet")):
            if w.tag in _WRAP_TYPES:
                model.wrap_objects.append(_parse_wrap_object(w, bname))

    return model
