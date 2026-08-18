# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import an OpenSim model into a Newton :class:`~newton.ModelBuilder`.

This maps the parsed :class:`~newton.opensim.OsimModel` IR onto
Newton bodies, joints, and shapes, analogous to ``import_mjcf``.
Muscle-tendon actuators are recorded as :class:`OsimMuscleModel` metadata on the
returned :class:`OsimImportResult` so the Warp-native muscle solver
(``muscle``) can drive them after finalization.

OpenSim conventions: SI units, Y-up, body-fixed XYZ Euler orientations.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import warp as wp

from ..core.types import Axis
from ..sim import ModelBuilder
from .frame import OsimFrameConverter
from .parser import parse_osim
from .types import OsimJoint, OsimModel, OsimTransform

# OpenSim joint class -> default primary DOF axes (in the joint frame).
_PIN_AXIS = wp.vec3(0.0, 0.0, 1.0)
_SLIDER_AXIS = wp.vec3(1.0, 0.0, 0.0)


@dataclass
class OsimMuscleModel:
    """Finalized muscle metadata linking an OpenSim muscle to Newton bodies.

    Attributes:
        name: Muscle name.
        type: OpenSim muscle class name.
        body_indices: Newton body index for each path point (``-1`` for ground).
        local_points: Path-point locations in their body frames [m].
        fmax: Maximum isometric force [N].
        l_opt: Optimal fiber length [m].
        lt_slack: Tendon slack length [m].
        pennation_opt: Pennation angle at optimal fiber length [rad].
        vmax: Maximum contraction velocity [optimal fiber lengths / s].
        activation_tau: Activation time constant [s].
        deactivation_tau: Deactivation time constant [s].
    """

    name: str
    type: str
    body_indices: list[int]
    local_points: list[tuple[float, float, float]]
    fmax: float = 1.0
    l_opt: float = 0.1
    lt_slack: float = 0.1
    pennation_opt: float = 0.0
    vmax: float = 10.0
    activation_tau: float = 0.01
    deactivation_tau: float = 0.04


@dataclass
class OsimImportResult:
    """Result of importing an OpenSim model into a :class:`~newton.ModelBuilder`.

    Attributes:
        model: The parsed IR.
        body_index: Map from OpenSim body name to Newton body index (``ground`` -> -1).
        joint_index: Map from OpenSim joint name to Newton joint index.
        coordinate_dof: Map from coordinate name to the Newton ``joint_q`` index.
        muscles: Finalized muscle metadata.
        world_xform: Transform from the OpenSim Y-up world into the target
            builder's configured world frame.
    """

    model: OsimModel
    body_index: dict[str, int] = field(default_factory=dict)
    joint_index: dict[str, int] = field(default_factory=dict)
    coordinate_dof: dict[str, int] = field(default_factory=dict)
    muscles: list[OsimMuscleModel] = field(default_factory=list)
    world_xform: wp.transform | None = None


def _euler_xyz_to_quat(rx: float, ry: float, rz: float) -> wp.quat:
    """OpenSim body-fixed XYZ Euler angles [rad] to a Warp quaternion."""
    qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), rx)
    qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ry)
    qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rz)
    return wp.mul(wp.mul(qx, qy), qz)


def _to_transform(t: OsimTransform) -> wp.transform:
    q = _euler_xyz_to_quat(t.orientation[0], t.orientation[1], t.orientation[2])
    return wp.transform(wp.vec3(*t.translation), q)


def _inertia_matrix(inertia6: tuple[float, ...]) -> wp.mat33:
    ixx, iyy, izz, ixy, ixz, iyz = inertia6
    return wp.mat33(ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz)


def _topological_body_order(model: OsimModel) -> list[str]:
    """Order body names so every body appears after its joint parent.

    Bodies unreachable from ``ground`` (or whose parent is missing) are appended
    at the end in declaration order so nothing is dropped.
    """
    children: dict[str, list[str]] = {}
    parent_of: dict[str, str] = {}
    for j in model.joints:
        children.setdefault(j.parent_body, []).append(j.child_body)
        parent_of[j.child_body] = j.parent_body

    ordered: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        for c in children.get(name, []):
            if c not in seen:
                seen.add(c)
                ordered.append(c)
                visit(c)

    visit("ground")
    for b in model.bodies:
        if b.name not in seen:
            seen.add(b.name)
            ordered.append(b.name)
    return ordered


def _joint_of_child(model: OsimModel, child: str) -> OsimJoint | None:
    for j in model.joints:
        if j.child_body == child:
            return j
    return None


def _add_joint(
    builder: ModelBuilder,
    joint: OsimJoint,
    parent_idx: int,
    child_idx: int,
    root_xform: wp.transform,
) -> int:
    """Add a Newton joint for an OpenSim joint, returning the joint index."""
    parent_xform = _to_transform(joint.parent_transform)
    if parent_idx == -1:
        parent_xform = root_xform * parent_xform
    child_xform = _to_transform(joint.child_transform)
    jt = joint.type

    def dof(coord_name: str) -> tuple[float, float]:
        for c in joint.coordinates:
            if c.name == coord_name and c.range is not None:
                return c.range
        return (-1e6, 1e6)

    common = {"parent_xform": parent_xform, "child_xform": child_xform, "label": joint.name}

    if jt == "WeldJoint":
        return builder.add_joint_fixed(parent_idx, child_idx, **common)
    if jt == "PinJoint":
        lo, hi = dof(joint.coordinates[0].name) if joint.coordinates else (-1e6, 1e6)
        return builder.add_joint_revolute(
            parent_idx, child_idx, axis=_PIN_AXIS, limit_lower=lo, limit_upper=hi, **common
        )
    if jt == "SliderJoint":
        lo, hi = dof(joint.coordinates[0].name) if joint.coordinates else (-1e6, 1e6)
        return builder.add_joint_prismatic(
            parent_idx, child_idx, axis=_SLIDER_AXIS, limit_lower=lo, limit_upper=hi, **common
        )
    if jt == "BallJoint":
        return builder.add_joint_ball(parent_idx, child_idx, **common)
    if jt == "FreeJoint":
        return builder.add_joint_free(
            child_idx,
            parent=parent_idx,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label=joint.name,
        )
    if jt == "PlanarJoint":
        JDC = ModelBuilder.JointDofConfig
        linear = [JDC(axis=Axis.X), JDC(axis=Axis.Y)]
        angular = [JDC(axis=Axis.Z)]
        return builder.add_joint_d6(parent_idx, child_idx, linear_axes=linear, angular_axes=angular, **common)
    if jt == "UniversalJoint":
        JDC = ModelBuilder.JointDofConfig
        angular = [JDC(axis=Axis.X), JDC(axis=Axis.Y)]
        return builder.add_joint_d6(parent_idx, child_idx, angular_axes=angular, **common)
    if jt == "CustomJoint":
        return _add_custom_joint(builder, joint, parent_idx, child_idx, parent_xform, child_xform)

    warnings.warn(f"OpenSim joint type '{jt}' not yet supported; using FIXED for '{joint.name}'.", stacklevel=2)
    return builder.add_joint_fixed(parent_idx, child_idx, **common)


def _add_custom_joint(builder, joint, parent_idx, child_idx, parent_xform, child_xform) -> int:
    """Approximate an OpenSim ``CustomJoint`` as a Newton D6 joint.

    Each coordinate-driven ``TransformAxis`` becomes an independent DOF along its
    axis. Coupled (spline/multiplier) coordinate functions are treated as
    independent linear DOFs (a documented approximation).
    """
    JDC = ModelBuilder.JointDofConfig
    linear: list = []
    angular: list = []
    coord_ranges = {c.name: (c.range or (-1e6, 1e6)) for c in joint.coordinates}
    axes = joint.spatial_transform
    # Order: rotation1..3 (angular), translation1..3 (linear).
    for i, ta in enumerate(axes):
        if ta.is_identity or not ta.coordinates:
            continue
        lo, hi = coord_ranges.get(ta.coordinates[0], (-1e6, 1e6))
        cfg = JDC(axis=wp.vec3(*ta.axis), limit_lower=lo, limit_upper=hi)
        if i < 3:
            angular.append(cfg)
        else:
            linear.append(cfg)
    if not linear and not angular:
        return builder.add_joint_fixed(
            parent_idx, child_idx, parent_xform=parent_xform, child_xform=child_xform, label=joint.name
        )
    return builder.add_joint_d6(
        parent_idx,
        child_idx,
        linear_axes=linear,
        angular_axes=angular,
        parent_xform=parent_xform,
        child_xform=child_xform,
        label=joint.name,
    )


def _add_contact_geometry(
    builder: ModelBuilder,
    model: OsimModel,
    body_index: dict[str, int],
    root_xform: wp.transform,
) -> None:
    """Add ContactSphere / ContactHalfSpace geometry as Newton shapes."""
    for cg in model.contact_geometry:
        bidx = body_index.get(cg.body, -1)
        xf = wp.transform(wp.vec3(*cg.location), _euler_xyz_to_quat(*cg.orientation))
        if bidx == -1:
            xf = root_xform * xf
        if cg.type == "ContactSphere":
            builder.add_shape_sphere(body=bidx, xform=xf, radius=max(cg.radius, 1e-4))
        elif cg.type == "ContactHalfSpace":
            # OpenSim's outward half-space normal is local -X, while Newton's
            # plane normal is local +Z.
            plane_adapter = wp.transform(
                wp.vec3(0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.5 * wp.pi),
            )
            builder.add_shape_plane(body=bidx, xform=xf * plane_adapter)


def add_osim(
    builder: ModelBuilder,
    source: str | os.PathLike | OsimModel,
    *,
    xform: wp.transform | None = None,
    parse_muscles: bool = True,
    parse_contacts: bool = True,
    floating: bool | None = None,
    apply_gravity: bool = True,
    verbose: bool = False,
) -> OsimImportResult:
    """Parse an OpenSim model and add its bodies, joints, and shapes to ``builder``.

    Args:
        builder: Target model builder. OpenSim's Y-up world is rotated into
            the builder's configured up axis (Z-up by default).
        source: Path to a ``.osim`` file, XML string, or a parsed :class:`OsimModel`.
        xform: Optional mechanism-root placement applied in the target Newton
            world after the OpenSim-to-Newton up-axis conversion. Ground-attached
            geometry is basis-converted but is not moved by this placement.
        parse_muscles: If True, record muscle metadata on the result.
        parse_contacts: If True, add contact geometry shapes.
        floating: Force the root joint to FREE (True) or FIXED (False). ``None``
            keeps the joint declared in the model.
        apply_gravity: If True, rotate the OpenSim gravity vector into the
            builder's configured world frame and assign it to the builder.
        verbose: Print a short import summary.

    Returns:
        An :class:`OsimImportResult` with name/index maps and muscle metadata.
    """
    model = source if isinstance(source, OsimModel) else parse_osim(source)
    axis_xform = OsimFrameConverter(builder.up_axis).world_xform
    user_xform = xform if xform is not None else wp.transform()
    root_xform = user_xform * axis_xform
    result = OsimImportResult(model=model, world_xform=root_xform)
    if apply_gravity:
        builder.gravity = wp.quat_rotate(axis_xform.q, wp.vec3(*model.gravity))
    body_index: dict[str, int] = {"ground": -1}

    order = _topological_body_order(model)
    joints_added: list[int] = []

    for bname in order:
        b = model.body(bname)
        if b is None:
            continue
        joint = _joint_of_child(model, bname)
        # Inertia is specified about the COM in OpenSim.
        idx = builder.add_link(
            com=wp.vec3(*b.mass_center),
            inertia=_inertia_matrix(b.inertia),
            mass=b.mass,
            label=b.name,
        )
        body_index[bname] = idx

        parent_idx = body_index.get(joint.parent_body, -1) if joint else -1
        if joint is None:
            # No joint connects this body; attach with a free or fixed base.
            jt = "FreeJoint" if floating else "WeldJoint"
            j = _synthetic_root_joint(builder, jt, idx, root_xform)
        else:
            if floating is not None and parent_idx == -1:
                joint = _override_root(joint, floating)
            j = _add_joint(builder, joint, parent_idx, idx, root_xform)
            result.joint_index[joint.name] = j
        joints_added.append(j)

    if joints_added:
        builder.add_articulation(sorted(set(joints_added)))

    # Record coordinate -> joint_q dof indices (best-effort, joint order).
    _record_coordinate_dofs(builder, model, result)

    if parse_contacts:
        _add_contact_geometry(builder, model, body_index, axis_xform)

    if parse_muscles:
        _record_muscles(model, body_index, result)

    result.body_index = body_index
    if verbose:
        print(
            f"[opensim] '{model.name}': {len(model.bodies)} bodies, "
            f"{len(model.joints)} joints, {len(result.muscles)} muscles, "
            f"{len(model.contact_geometry)} contact geoms"
        )
    return result


def _synthetic_root_joint(builder, jt, child_idx, xform) -> int:
    root_xform = xform if xform is not None else wp.transform()
    if jt == "FreeJoint":
        return builder.add_joint_free(child_idx, parent_xform=root_xform)
    return builder.add_joint_fixed(-1, child_idx, parent_xform=root_xform)


def _override_root(joint: OsimJoint, floating: bool) -> OsimJoint:
    joint.type = "FreeJoint" if floating else "WeldJoint"
    return joint


def _record_coordinate_dofs(builder, model, result) -> None:
    """Map scalar OpenSim coordinates to their actual Newton ``joint_q`` indices."""
    for joint in model.joints:
        joint_index = result.joint_index.get(joint.name)
        if joint_index is None or not joint.coordinates:
            continue
        if joint.type in {"BallJoint", "FreeJoint"}:
            # Newton stores these orientations as quaternions, so there is no
            # one-to-one scalar coordinate index for the OpenSim Euler values.
            continue

        if joint.type == "CustomJoint":
            linear_names = [
                axis.coordinates[0] for axis in joint.spatial_transform[3:] if not axis.is_identity and axis.coordinates
            ]
            angular_names = [
                axis.coordinates[0] for axis in joint.spatial_transform[:3] if not axis.is_identity and axis.coordinates
            ]
            coordinate_names = linear_names + angular_names
        elif joint.type == "PlanarJoint":
            coordinate_names = [c.name for c in joint.coordinates if c.motion_type == "translational"]
            coordinate_names += [c.name for c in joint.coordinates if c.motion_type != "translational"]
        else:
            coordinate_names = [c.name for c in joint.coordinates]

        q_start = builder.joint_q_start[joint_index]
        q_end = (
            builder.joint_q_start[joint_index + 1]
            if joint_index + 1 < len(builder.joint_q_start)
            else builder.joint_coord_count
        )
        if len(coordinate_names) != q_end - q_start:
            warnings.warn(
                f"cannot map all OpenSim coordinates for joint {joint.name!r} onto its Newton coordinates",
                stacklevel=2,
            )
        for offset, coordinate_name in enumerate(coordinate_names[: q_end - q_start]):
            result.coordinate_dof.setdefault(coordinate_name, q_start + offset)


def _record_muscles(model, body_index, result) -> None:
    for mu in model.muscles:
        body_idx = [body_index.get(p.body, -1) for p in mu.path_points]
        local = [p.location for p in mu.path_points]
        p = mu.params
        pen = p.get("pennation_angle_at_optimal", p.get("pennation_angle", 0.0))
        result.muscles.append(
            OsimMuscleModel(
                name=mu.name,
                type=mu.type,
                body_indices=body_idx,
                local_points=local,
                fmax=p.get("max_isometric_force", 1.0),
                l_opt=p.get("optimal_fiber_length", 0.1),
                lt_slack=p.get("tendon_slack_length", 0.1),
                pennation_opt=pen,
                vmax=p.get("max_contraction_velocity", p.get("Vmax", 10.0)),
                activation_tau=p.get("activation_time_constant", 0.01),
                deactivation_tau=p.get("deactivation_time_constant", 0.04),
            )
        )
