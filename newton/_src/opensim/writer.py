# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialize an :class:`~newton.opensim.OsimModel` back to ``.osim`` XML.

:func:`osim_to_xml` / :func:`write_osim` round-trip the model IR produced by
:func:`~newton.opensim.parse_osim` into an OpenSim 4.x
(``Version="40000"``) document: bodies (mass, inertia, attached geometry, wrap
objects), joints (parent/child :class:`PhysicalOffsetFrame` offsets, coordinates,
``CustomJoint`` ``SpatialTransform`` functions), markers, muscles (``GeometryPath``
path points and ``PathWrapSet``, model parameters), coordinate/point/torque
actuators, contact geometry and forces, and the remaining path-based force
elements. Parsing the written document reproduces the original model.
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
    OsimJoint,
    OsimLigament,
    OsimModel,
    OsimMuscle,
    OsimPathPoint,
    OsimPathSpring,
    OsimPointToPointSpring,
    OsimSpringGeneralizedForce,
    OsimTransform,
    OsimWrapObject,
)


def _fmt(x: float) -> str:
    """Format a float compactly, preserving round-trip precision."""
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(float(x))


def _vec(parent: ET.Element, tag: str, vec) -> ET.Element:
    """Append ``<tag>v0 v1 ...</tag>`` to ``parent``."""
    e = ET.SubElement(parent, tag)
    e.text = " ".join(_fmt(v) for v in vec)
    return e


def _scalar(parent: ET.Element, tag: str, value) -> ET.Element:
    """Append ``<tag>value</tag>`` to ``parent``."""
    e = ET.SubElement(parent, tag)
    e.text = _fmt(value) if isinstance(value, (int, float)) else str(value)
    return e


def _objects(parent: ET.Element) -> ET.Element:
    """Append and return an ``<objects>`` container under ``parent``."""
    return ET.SubElement(parent, "objects")


def _add_transform(parent: ET.Element, xf: OsimTransform) -> None:
    """Write the translation/orientation of an offset frame."""
    _vec(parent, "translation", xf.translation)
    _vec(parent, "orientation", xf.orientation)


def _add_function(parent: ET.Element, spec: dict) -> None:
    """Write a coordinate ``<function>`` body element from a parsed function dict."""
    wrapper = ET.SubElement(parent, "function")
    ftype = spec.get("type")
    if ftype == "LinearFunction":
        fn = ET.SubElement(wrapper, "LinearFunction")
        _vec(fn, "coefficients", spec.get("coefficients", (1.0, 0.0)))
    elif ftype == "Constant":
        fn = ET.SubElement(wrapper, "Constant")
        _scalar(fn, "value", spec.get("value", 0.0))
    elif ftype in ("SimmSpline", "NaturalCubicSpline", "GCVSpline"):
        fn = ET.SubElement(wrapper, "SimmSpline")
        _vec(fn, "x", spec.get("x", ()))
        _vec(fn, "y", spec.get("y", ()))
    elif ftype == "PiecewiseLinearFunction":
        fn = ET.SubElement(wrapper, "PiecewiseLinearFunction")
        _vec(fn, "x", spec.get("x", ()))
        _vec(fn, "y", spec.get("y", ()))
    elif ftype == "MultiplierFunction":
        fn = ET.SubElement(wrapper, "MultiplierFunction")
        _scalar(fn, "scale", spec.get("scale", 1.0))
        _add_function(fn, spec.get("inner", {"type": "Constant", "value": 0.0}))
    elif ftype:
        ET.SubElement(wrapper, ftype)


def _add_wrap_object(parent: ET.Element, wrap: OsimWrapObject) -> None:
    """Write a wrap surface into a body's ``WrapObjectSet``."""
    e = ET.SubElement(parent, wrap.type)
    e.set("name", wrap.name)
    _scalar(e, "active", "true" if wrap.active else "false")
    _vec(e, "xyz_body_rotation", wrap.rotation)
    _vec(e, "translation", wrap.translation)
    _scalar(e, "quadrant", wrap.quadrant)
    if wrap.radius:
        _scalar(e, "radius", wrap.radius)
    if wrap.length:
        _scalar(e, "length", wrap.length)
    if any(wrap.dimensions):
        _vec(e, "dimensions", wrap.dimensions)
    if wrap.inner_radius:
        _scalar(e, "inner_radius", wrap.inner_radius)
    if wrap.outer_radius:
        _scalar(e, "outer_radius", wrap.outer_radius)


def _add_body(parent: ET.Element, body: OsimBody, wraps: list[OsimWrapObject]) -> None:
    """Write a ``Body`` (mass properties, attached geometry, wrap objects)."""
    e = ET.SubElement(parent, "Body")
    e.set("name", body.name)
    _scalar(e, "mass", body.mass)
    _vec(e, "mass_center", body.mass_center)
    _vec(e, "inertia", body.inertia)
    if body.geometry:
        ag = ET.SubElement(e, "attached_geometry")
        for geom in body.geometry:
            mesh = ET.SubElement(ag, "Mesh")
            mesh.set("name", geom.name)
            if geom.socket_frame:
                _scalar(mesh, "socket_frame", geom.socket_frame)
            if geom.mesh_file:
                _scalar(mesh, "mesh_file", geom.mesh_file)
            _vec(mesh, "scale_factors", geom.scale_factors)
            app = ET.SubElement(mesh, "Appearance")
            _vec(app, "color", geom.color)
            _scalar(app, "opacity", geom.opacity)
    if wraps:
        wset = ET.SubElement(e, "WrapObjectSet")
        wobjs = _objects(wset)
        for wrap in wraps:
            _add_wrap_object(wobjs, wrap)


def _add_offset_frame(parent: ET.Element, name: str, body: str, xf: OsimTransform) -> None:
    """Write a ``PhysicalOffsetFrame`` connected to ``body`` with offset ``xf``."""
    pof = ET.SubElement(parent, "PhysicalOffsetFrame")
    pof.set("name", name)
    _scalar(pof, "socket_parent", f"/bodyset/{body}" if body != "ground" else "/ground")
    _add_transform(pof, xf)


def _add_joint(parent: ET.Element, joint: OsimJoint) -> None:
    """Write a joint with parent/child offset frames, coordinates, and spatial transform."""
    e = ET.SubElement(parent, joint.type)
    e.set("name", joint.name)
    pframe = f"{joint.name}_parent_offset"
    cframe = f"{joint.name}_child_offset"
    _scalar(e, "socket_parent_frame", pframe)
    _scalar(e, "socket_child_frame", cframe)
    if joint.coordinates:
        coords = ET.SubElement(e, "coordinates")
        cobjs = _objects(coords)
        for c in joint.coordinates:
            ce = ET.SubElement(cobjs, "Coordinate")
            ce.set("name", c.name)
            _scalar(ce, "motion_type", c.motion_type)
            _scalar(ce, "default_value", c.default_value)
            _scalar(ce, "default_speed_value", c.default_speed)
            if c.range is not None:
                _vec(ce, "range", c.range)
            _scalar(ce, "clamped", "true" if c.clamped else "false")
            _scalar(ce, "locked", "true" if c.locked else "false")
    if joint.spatial_transform:
        st = ET.SubElement(e, "SpatialTransform")
        for i, axis in enumerate(joint.spatial_transform):
            ta = ET.SubElement(st, "TransformAxis")
            ta.set("name", f"axis{i}")
            if axis.coordinates:
                _scalar(ta, "coordinates", " ".join(axis.coordinates))
            _vec(ta, "axis", axis.axis)
            if axis.function:
                _add_function(ta, axis.function)
    frames = ET.SubElement(e, "frames")
    _add_offset_frame(frames, pframe, joint.parent_body, joint.parent_transform)
    _add_offset_frame(frames, cframe, joint.child_body, joint.child_transform)


def _add_path_point(parent: ET.Element, pp: OsimPathPoint) -> None:
    """Write a (conditional/moving) ``PathPoint`` into a ``PathPointSet``."""
    e = ET.SubElement(parent, pp.type or "PathPoint")
    e.set("name", pp.name)
    # Preserve how the source stored the attachment: a raw ``socket_parent_frame``
    # path (4.x) or a bare ``body`` tag (legacy models).
    if pp.socket_frame:
        _scalar(e, "socket_parent_frame", pp.socket_frame)
    else:
        _scalar(e, "body", pp.body)
    _vec(e, "location", pp.location)
    if pp.type == "ConditionalPathPoint":
        if pp.conditional_coordinate:
            _scalar(e, "coordinate", pp.conditional_coordinate)
        if pp.conditional_range is not None:
            _vec(e, "range", pp.conditional_range)
    if pp.type == "MovingPathPoint" and pp.moving:
        for axis, (coord, _ftype, spec) in pp.moving.items():
            loc = ET.SubElement(e, f"{axis}_location")
            _add_function(loc, spec)
            # rename the wrapper element from <function> to the axis-less body form
            fn_wrapper = loc.find("function")
            if fn_wrapper is not None:
                loc.remove(fn_wrapper)
                for child in list(fn_wrapper):
                    loc.append(child)
            _scalar(e, f"{axis}_coordinate", coord)


def _add_geometry_path(parent: ET.Element, path_points, wraps) -> None:
    """Write a ``GeometryPath`` (points + wrap set) for a path-based force."""
    gp = ET.SubElement(parent, "GeometryPath")
    pps = ET.SubElement(gp, "PathPointSet")
    pobjs = _objects(pps)
    for pp in path_points:
        _add_path_point(pobjs, pp)
    if wraps:
        pws = ET.SubElement(gp, "PathWrapSet")
        wobjs = _objects(pws)
        for w in wraps:
            pw = ET.SubElement(wobjs, "PathWrap")
            _scalar(pw, "wrap_object", w.wrap_object)
            _scalar(pw, "method", w.method)
            _vec(pw, "range", w.range)


def _add_muscle(parent: ET.Element, muscle: OsimMuscle) -> None:
    """Write a muscle element (geometry path + model parameters)."""
    e = ET.SubElement(parent, muscle.type)
    e.set("name", muscle.name)
    _add_geometry_path(e, muscle.path_points, muscle.wraps)
    _scalar(e, "min_control", muscle.min_control)
    _scalar(e, "max_control", muscle.max_control)
    for tag, value in muscle.params.items():
        if tag in ("ignore_tendon_compliance", "ignore_activation_dynamics", "ignore_passive_fiber_force"):
            _scalar(e, tag, "true" if value else "false")
        else:
            _scalar(e, tag, value)


def _add_actuator(parent: ET.Element, act: OsimActuator) -> None:
    """Write a coordinate/point/torque/body actuator."""
    e = ET.SubElement(parent, act.type)
    e.set("name", act.name)
    if act.coordinate:
        _scalar(e, "coordinate", act.coordinate)
    _scalar(e, "optimal_force", act.optimal_force)
    if act.min_control != -float("inf"):
        _scalar(e, "min_control", act.min_control)
    if act.max_control != float("inf"):
        _scalar(e, "max_control", act.max_control)
    if act.body:
        _scalar(e, "socket_frame", f"/bodyset/{act.body}" if act.body != "ground" else "/ground")
    if act.type in ("PointActuator", "TorqueActuator"):
        _vec(e, "point", act.point)
        _vec(e, "direction", act.direction)
        _scalar(e, "point_is_global", "true" if act.point_is_global else "false")
        _scalar(e, "force_is_global", "true" if act.force_is_global else "false")


def _add_contact_geometry(parent: ET.Element, cg: OsimContactGeometry) -> None:
    """Write a contact geometry (sphere/half-space/mesh)."""
    e = ET.SubElement(parent, cg.type)
    e.set("name", cg.name)
    _scalar(e, "socket_frame", f"/bodyset/{cg.body}" if cg.body != "ground" else "/ground")
    _vec(e, "location", cg.location)
    _vec(e, "orientation", cg.orientation)
    if cg.radius:
        _scalar(e, "radius", cg.radius)
    if cg.mesh_file:
        _scalar(e, "filename", cg.mesh_file)


def _add_contact_force(parent: ET.Element, cf: OsimContactForce) -> None:
    """Write a compliant contact force element."""
    e = ET.SubElement(parent, cf.type)
    e.set("name", cf.name)
    for tag, value in cf.params.items():
        _scalar(e, tag, value)
    if cf.sphere is not None:
        _scalar(e, "socket_sphere", cf.sphere)
    if cf.half_space is not None:
        _scalar(e, "socket_half_space", cf.half_space)
    if cf.surface_params:
        cps = ET.SubElement(e, "contact_parameters")
        cobjs = _objects(cps)
        for geom, mat in cf.surface_params.items():
            block = ET.SubElement(cobjs, "ContactParameters")
            for tag, value in mat.items():
                _scalar(block, tag, value)
            _scalar(block, "geometry", geom)


def _add_path_spring(parent: ET.Element, ps: OsimPathSpring) -> None:
    """Write a ``PathSpring`` element."""
    e = ET.SubElement(parent, "PathSpring")
    e.set("name", ps.name)
    _add_geometry_path(e, ps.path_points, ps.wraps)
    _scalar(e, "resting_length", ps.resting_length)
    _scalar(e, "stiffness", ps.stiffness)
    _scalar(e, "dissipation", ps.dissipation)


def _add_ligament(parent: ET.Element, lig: OsimLigament) -> None:
    """Write a ``Ligament`` element."""
    e = ET.SubElement(parent, "Ligament")
    e.set("name", lig.name)
    _add_geometry_path(e, lig.path_points, lig.wraps)
    _scalar(e, "resting_length", lig.resting_length)
    _scalar(e, "pcsa_force", lig.pcsa_force)
    fl = ET.SubElement(e, "force_length_curve")
    _add_function(fl, lig.force_length_curve)
    fn = fl.find("function")
    if fn is not None:
        fl.remove(fn)
        for child in list(fn):
            fl.append(child)


def _add_p2p_spring(parent: ET.Element, sp: OsimPointToPointSpring) -> None:
    """Write a ``PointToPointSpring`` element."""
    e = ET.SubElement(parent, "PointToPointSpring")
    e.set("name", sp.name)
    _scalar(e, "socket_body1", f"/bodyset/{sp.body1}" if sp.body1 != "ground" else "/ground")
    _scalar(e, "socket_body2", f"/bodyset/{sp.body2}" if sp.body2 != "ground" else "/ground")
    _vec(e, "point1", sp.point1)
    _vec(e, "point2", sp.point2)
    _scalar(e, "stiffness", sp.stiffness)
    _scalar(e, "rest_length", sp.rest_length)


def _add_spring_gen_force(parent: ET.Element, sg: OsimSpringGeneralizedForce) -> None:
    """Write a ``SpringGeneralizedForce`` element."""
    e = ET.SubElement(parent, "SpringGeneralizedForce")
    e.set("name", sg.name)
    _scalar(e, "coordinate", sg.coordinate)
    _scalar(e, "stiffness", sg.stiffness)
    _scalar(e, "rest_length", sg.rest_length)
    _scalar(e, "viscosity", sg.viscosity)


def _add_bushing_force(parent: ET.Element, bf: OsimBushingForce) -> None:
    """Write a ``BushingForce`` element.

    Uses the ``body_1``/``location_body_1``/``orientation_body_1`` layout so the
    per-body frame offsets round-trip without emitting owned offset frames.
    """
    e = ET.SubElement(parent, "BushingForce")
    e.set("name", bf.name)
    _scalar(e, "body_1", bf.body1)
    _scalar(e, "body_2", bf.body2)
    _vec(e, "location_body_1", bf.frame1_transform.translation)
    _vec(e, "orientation_body_1", bf.frame1_transform.orientation)
    _vec(e, "location_body_2", bf.frame2_transform.translation)
    _vec(e, "orientation_body_2", bf.frame2_transform.orientation)
    _vec(e, "rotational_stiffness", bf.rotational_stiffness)
    _vec(e, "translational_stiffness", bf.translational_stiffness)
    _vec(e, "rotational_damping", bf.rotational_damping)
    _vec(e, "translational_damping", bf.translational_damping)


def osim_to_xml(model: OsimModel) -> str:
    """Serialize ``model`` to an OpenSim ``.osim`` XML string.

    Args:
        model: The model IR to serialize.

    Returns:
        A pretty-printed ``OpenSimDocument`` XML string that
        :func:`~newton.opensim.parse_osim` round-trips to ``model``.
    """
    doc = ET.Element("OpenSimDocument")
    doc.set("Version", str(model.version or 40000))
    m = ET.SubElement(doc, "Model")
    m.set("name", model.name)
    _vec(m, "gravity", model.gravity)

    # Bodies (+ wrap objects owned by each body).
    wraps_by_body: dict[str, list[OsimWrapObject]] = {}
    for wrap in model.wrap_objects:
        wraps_by_body.setdefault(wrap.body, []).append(wrap)
    bodyset = ET.SubElement(m, "BodySet")
    bobjs = _objects(bodyset)
    for body in model.bodies:
        _add_body(bobjs, body, wraps_by_body.get(body.name, []))
    # Ground-owned wrap objects go on a Ground element in the BodySet.
    ground_wraps = wraps_by_body.get("ground", [])
    if ground_wraps:
        ground = ET.SubElement(bobjs, "Ground")
        ground.set("name", "ground")
        wset = ET.SubElement(ground, "WrapObjectSet")
        wobjs = _objects(wset)
        for wrap in ground_wraps:
            _add_wrap_object(wobjs, wrap)

    # Joints.
    jointset = ET.SubElement(m, "JointSet")
    jobjs = _objects(jointset)
    for joint in model.joints:
        _add_joint(jobjs, joint)

    # Markers.
    if model.markers:
        markerset = ET.SubElement(m, "MarkerSet")
        mobjs = _objects(markerset)
        for marker in model.markers:
            me = ET.SubElement(mobjs, "Marker")
            me.set("name", marker.name)
            _scalar(me, "socket_parent_frame", f"/bodyset/{marker.body}" if marker.body != "ground" else "/ground")
            _vec(me, "location", marker.location)

    # Forces.
    forceset = ET.SubElement(m, "ForceSet")
    fobjs = _objects(forceset)
    for muscle in model.muscles:
        _add_muscle(fobjs, muscle)
    for act in model.actuators:
        _add_actuator(fobjs, act)
    for cf in model.contact_forces:
        _add_contact_force(fobjs, cf)
    for ps in model.path_springs:
        _add_path_spring(fobjs, ps)
    for lig in model.ligaments:
        _add_ligament(fobjs, lig)
    for sp in model.point_to_point_springs:
        _add_p2p_spring(fobjs, sp)
    for sg in model.spring_generalized_forces:
        _add_spring_gen_force(fobjs, sg)
    for bf in model.bushing_forces:
        _add_bushing_force(fobjs, bf)

    # Contact geometry.
    if model.contact_geometry:
        cgs = ET.SubElement(m, "ContactGeometrySet")
        cobjs = _objects(cgs)
        for cg in model.contact_geometry:
            _add_contact_geometry(cobjs, cg)

    ET.indent(doc, space="\t")
    return '<?xml version="1.0" encoding="UTF-8" ?>\n' + ET.tostring(doc, encoding="unicode")


def write_osim(model: OsimModel, path: str | os.PathLike) -> None:
    """Write ``model`` to an OpenSim ``.osim`` file.

    Args:
        model: The model IR to serialize.
        path: Output ``.osim`` path.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(osim_to_xml(model))
