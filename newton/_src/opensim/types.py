# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Intermediate representation (IR) for OpenSim models.

These dataclasses mirror the structure of an OpenSim ``.osim`` document
(``OpenSimDocument`` / ``Model``) in a solver-agnostic form. The parser in
``parser`` populates this IR from XML, and the importer
in ``importer`` maps it onto a
:class:`~newton.ModelBuilder`.

The IR intentionally keeps quantities in OpenSim conventions (SI units, body
frame Y-up meshes, XYZ body-fixed Euler orientations) so that a single parsing
pass is reusable across importers, analyses, and round-trip tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OsimTransform:
    """Rigid offset expressed as translation [m] and XYZ body-fixed Euler angles [rad]."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class OsimFrame:
    """A ``PhysicalOffsetFrame``: a fixed offset relative to a parent frame/body.

    Attributes:
        name: Component name.
        parent: Name of the parent physical frame (a body or ``ground``).
        transform: Offset from the parent frame.
    """

    name: str
    parent: str
    transform: OsimTransform = field(default_factory=OsimTransform)


@dataclass
class OsimGeometry:
    """Visual/attached geometry (currently ``Mesh`` primitives)."""

    name: str
    mesh_file: str | None = None
    scale_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    opacity: float = 1.0
    # Optional offset frame the geometry is attached to (socket_frame path).
    socket_frame: str | None = None


@dataclass
class OsimBody:
    """A rigid body with inertial properties and attached geometry.

    Attributes:
        name: Body name.
        mass: Mass [kg].
        mass_center: Center of mass in the body frame [m].
        inertia: Inertia tensor about the COM as ``[Ixx Iyy Izz Ixy Ixz Iyz]`` [kg·m^2].
        geometry: Attached display geometry.
    """

    name: str
    mass: float = 0.0
    mass_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    geometry: list[OsimGeometry] = field(default_factory=list)


@dataclass
class OsimCoordinate:
    """A joint generalized coordinate (q).

    Attributes:
        name: Coordinate name.
        motion_type: ``"rotational"``, ``"translational"``, or ``"coupled"``.
        default_value: Default coordinate value [m or rad].
        default_speed: Default coordinate speed [m/s or rad/s].
        range: ``(min, max)`` allowed values [m or rad].
        clamped: Whether the coordinate is clamped to ``range``.
        locked: Whether the coordinate is locked.
    """

    name: str
    motion_type: str = "rotational"
    default_value: float = 0.0
    default_speed: float = 0.0
    range: tuple[float, float] | None = None
    clamped: bool = False
    locked: bool = False


@dataclass
class OsimTransformAxis:
    """One axis of a ``CustomJoint`` ``SpatialTransform`` (rotation1..3, translation1..3).

    Attributes:
        axis: Unit axis in the joint frame.
        coordinates: Names of the coordinates driving this axis.
        function_type: Type of the coupling function (``"LinearFunction"``,
            ``"SimmSpline"``, ``"Constant"``, ``"MultiplierFunction"``, ...).
        function: Raw serialized parameters of the function (kept for fidelity).
        is_identity: True when the axis has no driving coordinate/function.
    """

    axis: tuple[float, float, float]
    coordinates: list[str] = field(default_factory=list)
    function_type: str | None = None
    function: dict = field(default_factory=dict)
    is_identity: bool = True


@dataclass
class OsimJoint:
    """A joint connecting a parent frame to a child frame.

    Attributes:
        name: Joint name.
        type: OpenSim class name (``PinJoint``, ``FreeJoint``, ``CustomJoint``, ...).
        parent_body: Resolved parent body name (``ground`` for the world).
        child_body: Resolved child body name.
        parent_transform: Offset from the parent body frame to the joint frame.
        child_transform: Offset from the child body frame to the joint frame.
        coordinates: Ordered generalized coordinates.
        spatial_transform: For ``CustomJoint``, the six ``TransformAxis`` entries
            (rot1, rot2, rot3, trans1, trans2, trans3).
    """

    name: str
    type: str
    parent_body: str = "ground"
    child_body: str = ""
    parent_transform: OsimTransform = field(default_factory=OsimTransform)
    child_transform: OsimTransform = field(default_factory=OsimTransform)
    coordinates: list[OsimCoordinate] = field(default_factory=list)
    spatial_transform: list[OsimTransformAxis] = field(default_factory=list)


@dataclass
class OsimPathPoint:
    """A point on a muscle/actuator ``GeometryPath``.

    Attributes:
        name: Point name.
        body: Body the point is fixed to.
        location: Location in the body frame [m]. For a ``MovingPathPoint`` this
            is the default location; the live location is given by ``moving``.
        type: ``"PathPoint"``, ``"ConditionalPathPoint"``, or ``"MovingPathPoint"``.
        socket_frame: For 4.x models, the physical frame the point attaches to.
        conditional_coordinate: For a ``ConditionalPathPoint``, the coordinate
            whose value gates whether the point is active.
        conditional_range: ``(min, max)`` range of ``conditional_coordinate``
            [m or rad] within which the point is active.
        moving: For a ``MovingPathPoint``, maps each axis (``"x"``/``"y"``/``"z"``)
            to ``(coordinate_name, function_type, params)`` describing the
            body-frame coordinate as a function of a generalized coordinate.
    """

    name: str
    body: str
    location: tuple[float, float, float]
    type: str = "PathPoint"
    socket_frame: str | None = None
    conditional_coordinate: str | None = None
    conditional_range: tuple[float, float] | None = None
    moving: dict[str, tuple[str, str, dict]] | None = None


@dataclass
class OsimWrap:
    """A ``PathWrap`` associating a muscle path with a wrap surface."""

    wrap_object: str
    method: str = "hybrid"
    range: tuple[int, int] = (-1, -1)


@dataclass
class OsimWrapObject:
    """A wrap surface (``WrapSphere``, ``WrapCylinder``, ...) fixed to a body.

    Attributes:
        name: Wrap-object name referenced by a muscle's :class:`OsimWrap`.
        type: OpenSim class name (``WrapSphere``, ``WrapCylinder``, ``WrapEllipsoid``,
            ``WrapTorus``).
        body: Body (or frame) the wrap surface is fixed to.
        translation: Wrap-surface origin in the body frame [m].
        rotation: Body-fixed XYZ Euler rotation of the wrap surface [rad].
        radius: Sphere/cylinder radius [m].
        length: Cylinder length [m] (unused for a sphere).
        dimensions: ``WrapEllipsoid`` semi-axis radii ``(a, b, c)`` [m].
        inner_radius: ``WrapTorus`` inner (hole) radius [m]; the ring radius is
            ``(inner_radius + outer_radius) / 2``.
        outer_radius: ``WrapTorus`` outer radius [m]; the tube radius is
            ``(outer_radius - inner_radius) / 2``.
        quadrant: Active quadrant restriction (``"all"``, ``"+x"``, ...).
        active: Whether the wrap surface is active.
    """

    name: str
    type: str
    body: str = "ground"
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    length: float = 0.0
    dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    quadrant: str = "all"
    active: bool = True


@dataclass
class OsimMuscle:
    """A muscle-tendon actuator with a geometry path.

    Attributes:
        name: Muscle name.
        type: OpenSim class name (``Thelen2003Muscle``, ``DeGrooteFregly2016Muscle``,
            ``Millard2012EquilibriumMuscle``, ``RigidTendonMuscle``, ...).
        path_points: Ordered points defining the muscle path.
        wraps: Wrap surfaces the path may wrap over.
        params: Model-specific scalar parameters (``max_isometric_force``,
            ``optimal_fiber_length``, ``tendon_slack_length``,
            ``pennation_angle_at_optimal``, ``max_contraction_velocity``, ...).
        min_control: Minimum excitation.
        max_control: Maximum excitation.
    """

    name: str
    type: str
    path_points: list[OsimPathPoint] = field(default_factory=list)
    wraps: list[OsimWrap] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
    min_control: float = 0.0
    max_control: float = 1.0


@dataclass
class OsimPathSpring:
    """A path-based linear spring with dissipation (OpenSim ``PathSpring``).

    Attributes:
        name: Element name.
        path_points: Ordered points defining the spring's geometry path.
        wraps: Wrap surfaces the path may wrap over.
        resting_length: Slack length below which the spring is unloaded [m].
        stiffness: Linear stiffness [N/m].
        dissipation: Dissipation coefficient [s/m] coupling tension to lengthening rate.
    """

    name: str = ""
    path_points: list[OsimPathPoint] = field(default_factory=list)
    wraps: list[OsimWrap] = field(default_factory=list)
    resting_length: float = 0.0
    stiffness: float = 0.0
    dissipation: float = 0.0


@dataclass
class OsimLigament:
    """A path-based ligament with a normalized force-length curve (OpenSim ``Ligament``).

    Attributes:
        name: Element name.
        path_points: Ordered points defining the ligament's geometry path.
        wraps: Wrap surfaces the path may wrap over.
        resting_length: Length at which the normalized force-length curve is sampled at 1 [m].
        pcsa_force: Scale (physiological cross-sectional area force) for the curve [N].
        force_length_curve: Parsed normalized force-length function (``{"type": ...}``),
            evaluated at ``length / resting_length``.
    """

    name: str = ""
    path_points: list[OsimPathPoint] = field(default_factory=list)
    wraps: list[OsimWrap] = field(default_factory=list)
    resting_length: float = 1.0
    pcsa_force: float = 0.0
    force_length_curve: dict = field(default_factory=lambda: {"type": "Constant", "value": 0.0})


@dataclass
class OsimPointToPointSpring:
    """A linear spring between a point on each of two bodies (OpenSim ``PointToPointSpring``).

    Attributes:
        name: Element name.
        body1: Name of the first attached body.
        body2: Name of the second attached body.
        point1: Attachment point in ``body1`` frame [m].
        point2: Attachment point in ``body2`` frame [m].
        stiffness: Linear stiffness [N/m].
        rest_length: Unstretched length [m].
    """

    name: str = ""
    body1: str = "ground"
    body2: str = "ground"
    point1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stiffness: float = 0.0
    rest_length: float = 0.0


@dataclass
class OsimSpringGeneralizedForce:
    """A passive linear spring-damper on a single coordinate (OpenSim ``SpringGeneralizedForce``).

    Applies ``-stiffness * (q - rest_length) - viscosity * qd`` to its coordinate.

    Attributes:
        name: Element name.
        coordinate: Name of the driven coordinate.
        stiffness: Linear stiffness [N/m or N·m/rad].
        rest_length: Coordinate value at which the spring force is zero [m or rad].
        viscosity: Linear damping coefficient [N·s/m or N·m·s/rad].
    """

    name: str = ""
    coordinate: str = ""
    stiffness: float = 0.0
    rest_length: float = 0.0
    viscosity: float = 0.0


@dataclass
class OsimBushingForce:
    """A 6-DOF linear bushing between a frame on each of two bodies (OpenSim ``BushingForce``).

    The bushing applies a linear elastic (and, in OpenSim, damping) load resisting the
    deflection of ``frame2`` relative to ``frame1``, where the rotational deflection is
    the body-fixed XYZ Euler angles of the relative orientation and the translational
    deflection is the relative position expressed in ``frame1``.

    Attributes:
        name: Element name.
        body1: Body carrying ``frame1``.
        body2: Body carrying ``frame2``.
        frame1_transform: Fixed offset of ``frame1`` in ``body1``.
        frame2_transform: Fixed offset of ``frame2`` in ``body2``.
        rotational_stiffness: Rotational stiffnesses about frame1 XYZ [N·m/rad].
        translational_stiffness: Translational stiffnesses along frame1 XYZ [N/m].
        rotational_damping: Rotational damping about frame1 XYZ [N·m·s/rad].
        translational_damping: Translational damping along frame1 XYZ [N·s/m].
    """

    name: str = ""
    body1: str = "ground"
    body2: str = "ground"
    frame1_transform: OsimTransform = field(default_factory=OsimTransform)
    frame2_transform: OsimTransform = field(default_factory=OsimTransform)
    rotational_stiffness: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translational_stiffness: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotational_damping: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translational_damping: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class OsimActuator:
    """A non-muscle actuator (``CoordinateActuator``, ``PointActuator``, ``TorqueActuator``).

    ``direction`` is the ``PointActuator`` force direction or the ``TorqueActuator``
    axis; ``force_is_global`` doubles as ``TorqueActuator`` ``torque_is_global``;
    ``body_b`` is the ``TorqueActuator`` reaction body.
    """

    name: str
    type: str
    coordinate: str | None = None
    optimal_force: float = 1.0
    min_control: float = -float("inf")
    max_control: float = float("inf")
    body: str | None = None
    body_b: str | None = None
    point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_is_global: bool = False
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_is_global: bool = True
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class OsimContactGeometry:
    """Contact geometry (``ContactSphere``, ``ContactHalfSpace``, ``ContactMesh``)."""

    name: str
    type: str
    body: str = "ground"
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    mesh_file: str | None = None


@dataclass
class OsimContactForce:
    """Compliant contact force.

    Covers ``SmoothSphereHalfSpaceForce`` (differentiable sphere/half-space, the
    socket-connected ``sphere``/``half_space`` with all material properties in
    ``params``), ``HuntCrossleyForce`` and ``ElasticFoundationForce`` (a
    ``contact_parameters`` set: ``geometries`` lists every named contact
    geometry, ``surface_params`` holds per-geometry material properties, and the
    element-level ``transition_velocity`` lives in ``params``).

    Attributes:
        name: Force element name.
        type: OpenSim class name.
        sphere: ``SmoothSphereHalfSpaceForce`` sphere geometry name.
        half_space: ``SmoothSphereHalfSpaceForce`` half-space geometry name.
        params: Element-level scalar properties (all material properties for the
            smooth model; ``transition_velocity`` for Hunt-Crossley / elastic
            foundation).
        surface_params: Per-geometry material properties, keyed by geometry name
            (Hunt-Crossley / elastic foundation).
        geometries: Names of all contact geometries the force couples.
    """

    name: str
    type: str
    sphere: str | None = None
    half_space: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    surface_params: dict[str, dict[str, float]] = field(default_factory=dict)
    geometries: list[str] = field(default_factory=list)


@dataclass
class OsimMarker:
    """A ``Marker`` fixed to a body (used for IK / experimental data)."""

    name: str
    body: str
    location: tuple[float, float, float]


@dataclass
class OsimModel:
    """Parsed OpenSim model IR.

    Attributes:
        name: Model name.
        version: ``OpenSimDocument`` version integer.
        gravity: Gravity vector [m/s^2].
        bodies: Rigid bodies (excluding ``ground``).
        joints: Joints connecting bodies.
        muscles: Muscle-tendon actuators.
        path_springs: Path-based linear springs.
        ligaments: Path-based ligaments.
        point_to_point_springs: Two-point linear springs.
        spring_generalized_forces: Passive single-coordinate spring-dampers.
        bushing_forces: Six-DOF linear frame bushings.
        actuators: Non-muscle actuators.
        contact_geometry: Named contact geometries.
        contact_forces: Compliant contact forces.
        markers: Body-fixed markers.
        frames: Model-level ``PhysicalOffsetFrame`` components keyed by name.
        wrap_objects: Wrap surfaces (``WrapSphere``, ...) muscle paths may wrap over.
    """

    name: str = "model"
    version: int = 40000
    gravity: tuple[float, float, float] = (0.0, -9.80665, 0.0)
    bodies: list[OsimBody] = field(default_factory=list)
    joints: list[OsimJoint] = field(default_factory=list)
    muscles: list[OsimMuscle] = field(default_factory=list)
    path_springs: list[OsimPathSpring] = field(default_factory=list)
    ligaments: list[OsimLigament] = field(default_factory=list)
    point_to_point_springs: list[OsimPointToPointSpring] = field(default_factory=list)
    spring_generalized_forces: list[OsimSpringGeneralizedForce] = field(default_factory=list)
    bushing_forces: list[OsimBushingForce] = field(default_factory=list)
    actuators: list[OsimActuator] = field(default_factory=list)
    contact_geometry: list[OsimContactGeometry] = field(default_factory=list)
    contact_forces: list[OsimContactForce] = field(default_factory=list)
    markers: list[OsimMarker] = field(default_factory=list)
    frames: dict[str, OsimFrame] = field(default_factory=dict)
    wrap_objects: list[OsimWrapObject] = field(default_factory=list)

    def body(self, name: str) -> OsimBody | None:
        """Return the body with ``name`` or ``None`` if absent."""
        for b in self.bodies:
            if b.name == name:
                return b
        return None
