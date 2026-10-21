# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision between the two legs of the three-dimensional runner.

The 3D runner has no body-to-body contact by default: :class:`.env3d.BilateralEnv3D`
integrates the articulation with :class:`newton.solvers.SolverFeatherstone` or with
:class:`newton.solvers.SolverMuJoCo` (``Env3DConfig.solver``, MuJoCo on the grid) with the
solver's own contacts disabled, and the only external force the environment adds each
substep is the hereditary shoe wrench on each foot (:meth:`.env3d.BilateralEnv3D._substep`).
The swing leg therefore passes through the stance leg freely, and a policy trained that
way can trade hip adduction for clearance.

This module gives the legs a capsule set (thigh, shank, and shoe-aligned foot per side)
and two ways to make it solid:

* A penalty kernel (every profile but ``"mujoco"``): one Warp kernel per substep tests the
  nine right-left capsule pairs and adds a critically damped, capped penalty with
  Coulomb-clamped tangential damping to ``body_f`` along the transverse direction of the
  closest pair of points. Limbs in contact slide past each other vertically, so a pair
  force that lifted one leg or drove a foot harder into the foam would model something
  that does not happen; :func:`profile` records the gains. One thread owns one world, so
  the six leg bodies need no atomics and the run stays bitwise reproducible. The cap makes
  crossing affordable: a policy can learn to push through it.
* MuJoCo's own self-collision (profile ``"mujoco"``, :attr:`LegContactConfig.engine`):
  :func:`add_capsules` puts the same capsules on the six leg bodies as collision shapes
  that only collide right leg against left leg, and the environment turns MuJoCo contacts
  on. The contact is a constraint, not a capped force, so the legs cannot be pushed through
  each other. The kernel still runs with zero stiffness to record penetration.

The ground stays the shoes' alone in both: no ground shape exists, and the pelvis carries
no capsule.

Geometry assumptions, in the body frames of :mod:`.model3d` (a thigh origin at the hip
center, a shank origin at the knee, and a foot origin at the ankle with the intrinsic
shoe axes):

* Each long segment is one capsule along its joint-centre line, ``span`` of the way
  from the proximal to the distal joint centre. Constant-radius capsules understate the
  proximal thigh, where a real thigh meets the pelvis.
* The foot is one capsule along the shoe's heel-to-toe direction at midsole height.
  :func:`shoe_envelope` measures that line and its half-width from the shoe's own column
  bed, so the shoe capsule follows the identified shoe rather than an assumption.

Radii and the penalty law are engineering choices, recorded by :func:`profile` and
measured in ``outputs/bilateral/grf_spike_2026_10_14/leg_collision/REPORT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

import newton

from .kinematics3d import SIDES
from .kinematics3d import model_frames as _model_frames

wp.set_module_options({"enable_backward": False})

BODY_LABELS = ("pelvis", "thigh_r", "shank_r", "foot_r", "thigh_l", "shank_l", "foot_l")
"""Body order within a world, as :func:`.model3d.build_runner` adds them."""
_SEGMENTS = ("thigh", "shank", "foot")
_PAIRS = tuple((right, left) for right in (1, 2, 3) for left in (4, 5, 6))
"""Index pairs (right segment, left segment) tested by :func:`_inter_leg`."""
EPS_SQUARED_M2 = 1.0e-12
"""Squared-length floor below which an axis or determinant is degenerate, as in the kernel."""
PAIR_LABELS = tuple(f"{_SEGMENTS[a - 1]}_r/{_SEGMENTS[b - 4]}_l" for a, b in _PAIRS)
"""Label of each pair, in the order the kernel tests them."""
_ORIGIN_KEY = ("hip_mid", "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l")
"""Key of each body's frame origin in :func:`.kinematics3d.model_frames`."""


@dataclass(frozen=True)
class LegContactConfig:
    """Capsule geometry and penalty law of one inter-leg contact set."""

    radius_m: tuple[float, float, float] = (0.060, 0.044, 0.035)
    """Effective collision radius of the thigh, shank, and foot [m].

    These are *below* the true half-widths: a capsule that thick would report penetration
    on the measured motion, which cannot contain any. Sweeping the set against that motion
    (:func:`shoe_envelope`) leaves 0.21 % of the training frames penetrating by at most
    12.6 mm, all at the right-foot/left-shank pair, so a real collision is reported well
    before the limbs would actually touch (:func:`shoe_config` measures the shoe's own
    half-width, 53 mm for the F01 shoe).
    """
    span: tuple[float, float] = (0.06, 1.0)
    """Capsule ends as a fraction of the hip-to-knee and knee-to-ankle line."""
    foot_span_m: tuple[float, float] = (-0.0949, 0.1331)
    """Foot capsule ends along the foot frame ``+x``, heel negative [m]."""
    foot_height_m: float = -0.0601
    """Foot capsule axis height in the foot frame [m]; negative is toward the sole."""
    stiffness_n_m: float = 4.0e4
    """Normal penalty stiffness [N/m]; zero measures penetration without any force."""
    damping_ratio: float = 1.0
    """Normal damping as a fraction of the pair's critical damping."""
    friction: float = 0.5
    """Coulomb coefficient of the tangential force."""
    maximum_force_n: float = 1500.0
    """Largest force one capsule pair may apply [N]."""
    ramp_steps: int = 0
    """Control steps over which the gains climb from ``ramp_from`` to this profile; 0 is fixed.

    A policy that has never felt its legs touch cannot absorb the target law on the first
    contact, so the fine-tune meets it gradually: one call of :meth:`LegContact.clear` is one
    control step, which makes the ramp a property of the contact rather than of the training
    loop. 14,400 steps is 30 % of a 1,500-iteration run at the 32-step PPO horizon.
    """
    ramp_from: tuple[float, float] = (1.0e3, 30.0)
    """Stiffness [N/m] and per-pair cap [N] the ramp starts from, below the target by design."""
    engine: str = "warp"
    """``"warp"`` applies the penalty law above; ``"mujoco"`` makes the capsules MuJoCo collision
    shapes (:func:`add_capsules`) and leaves the force to MuJoCo's contact solver, with
    ``friction`` as the contact's Coulomb coefficient."""
    solref: tuple[float, float] = (0.002, 1.0)
    """MuJoCo contact time constant [s] and damping ratio, for the ``"mujoco"`` engine.

    MuJoCo's default is 0.02 s. 2 ms is eight training substeps, four times the two-step floor
    MuJoCo recommends; 150 N m of adduction on both hips crosses the capsules by 2 mm at impact
    and well under 1 mm once pressed (5 mm and 1 mm at 5 ms).
    """

    @property
    def measures_only(self) -> bool:
        """Whether the profile leaves ``body_f`` untouched and only records penetration."""
        return self.stiffness_n_m <= 0.0


PROFILES: dict[str, LegContactConfig] = {
    "probe": LegContactConfig(stiffness_n_m=0.0, friction=0.0),
    "capsule_soft": LegContactConfig(stiffness_n_m=2.0e4, maximum_force_n=800.0),
    "capsule_light": LegContactConfig(stiffness_n_m=1.0e4, maximum_force_n=300.0),
    "capsule_bare": LegContactConfig(stiffness_n_m=4.0e3, maximum_force_n=120.0),
    "capsule_ramp": LegContactConfig(
        stiffness_n_m=4.0e3, maximum_force_n=120.0, ramp_steps=14_400, ramp_from=(1.0e3, 30.0)
    ),
    # The same climb to a law the policy cannot crush at its ceiling: 300 N per pair is two and a
    # half times the body weight, and the 120 N cap turned out to be affordable - the fine-tuned
    # policy kept its legs crossing and simply pushed at the ceiling instead.
    "capsule_ramp_light": LegContactConfig(
        stiffness_n_m=1.0e4, maximum_force_n=300.0, ramp_steps=14_400, ramp_from=(1.0e3, 30.0)
    ),
    "capsule_v1": LegContactConfig(),
    # The same capsules as MuJoCo collision shapes: MuJoCo applies the force, and the kernel only
    # records penetration, so a policy cannot buy its way through at a force cap.
    "mujoco": LegContactConfig(stiffness_n_m=0.0, engine="mujoco"),
}
"""Named profiles; ``"off"`` means no contact object at all and is handled by the caller."""


def profile(name: str) -> LegContactConfig:
    """Return the named contact profile or raise with the known names."""
    if name not in PROFILES:
        raise KeyError(f"unknown leg contact profile {name!r}; known: {', '.join(sorted(PROFILES))}, off")
    return PROFILES[name]


def shoe_envelope(anchor_bottom_m, rest_length_m, mount_m) -> tuple[tuple[float, float], float, float]:
    """Measure the shoe's heel-to-toe line, half-width, and midsole height in the foot frame.

    Args:
        anchor_bottom_m: Column bed anchors in the intrinsic shoe frame, shape [N, 3].
        rest_length_m: Undeformed column height of each anchor [m].
        mount_m: Ankle location in the intrinsic shoe frame [m], the shoe's mount.

    Returns:
        ``((heel_x, toe_x), half_width_m, height_m)`` in the foot frame: the capsule ends
        along ``+x`` at 5 % and 85 % of the bed's heel-to-toe extent, its maximum
        mediolateral half-width, and the mid-height of the bed.
    """
    points = np.asarray(anchor_bottom_m, dtype=float) - np.asarray(mount_m, dtype=float)
    tops = points.copy()
    tops[:, 2] += np.asarray(rest_length_m, dtype=float)
    low, high = np.minimum(points[:, 2], tops[:, 2]), np.maximum(points[:, 2], tops[:, 2])
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    span = x1 - x0
    near = (points[:, 0] > x0 + 0.05 * span) & (points[:, 0] < x0 + 0.85 * span)
    if not near.any():
        raise ValueError("The shoe bed is too short to measure a heel-to-toe capsule")
    half_width = float(np.abs(points[near, 1]).max())
    height = float(0.5 * (low[near].min() + high[near].max()))
    return (x0 + 0.05 * span, x0 + 0.85 * span), half_width, height


def shoe_config(anchor_bottom_m, rest_length_m, mount_m, base: LegContactConfig | None = None) -> LegContactConfig:
    """Return ``base`` with its foot capsule geometry replaced by the shoe's own envelope.

    The measured half-width caps the assumed foot radius instead of replacing it, so a
    narrow shoe does not silently make the foot thinner than the profile assumes.
    """
    cfg = base or LegContactConfig()
    (heel, toe), half_width, height = shoe_envelope(anchor_bottom_m, rest_length_m, mount_m)
    return replace(
        cfg,
        foot_span_m=(heel, toe),
        radius_m=(cfg.radius_m[0], cfg.radius_m[1], min(cfg.radius_m[2], half_width)),
        foot_height_m=height,
    )


def local_capsules(config: LegContactConfig, calibration) -> np.ndarray:
    """Capsule ends of every body of one world in its own body frame, shape [7, 2, 3].

    The thigh capsule lies on the hip-to-knee line and the shank capsule on the
    knee-to-ankle line, both given in the proximal body's frame by ``calibration``;
    the foot capsule lies on the shoe's heel-to-toe line in the foot frame.
    """
    ends = np.zeros((7, 2, 3))
    for side in SIDES:
        s = "r" if side == "right" else "l"
        axis = np.asarray(calibration.knee_m[side], dtype=float)
        ends[1 if s == "r" else 4] = np.outer(config.span, axis)
        axis = np.asarray(calibration.ankle_m[side], dtype=float)
        ends[2 if s == "r" else 5] = np.outer(config.span, axis)
    heel, toe = config.foot_span_m
    ends[3] = ends[6] = ((heel, 0.0, config.foot_height_m), (toe, 0.0, config.foot_height_m))
    return ends


def radii(config: LegContactConfig) -> np.ndarray:
    """Capsule radius of every body of one world, shape [7]; the pelvis has none."""
    out = np.zeros(7)
    out[1], out[2], out[3] = config.radius_m
    out[4], out[5], out[6] = config.radius_m
    return out


def add_capsules(builder, bodies, calibration, config: LegContactConfig) -> list[int]:
    """Add the leg capsules to one world's builder as MuJoCo collision shapes.

    Only right-leg against left-leg pairs collide: every same-leg pair is filtered, the pelvis
    carries no shape, and there is no ground shape. The shapes are massless, so the
    articulation's inertia is unchanged.

    Args:
        builder: The world's :class:`newton.ModelBuilder`.
        bodies: The world's seven body indices in :data:`BODY_LABELS` order.
        calibration: The world's static calibration, which places the joint centres.
        config: The capsule set; ``solref`` and ``friction`` set the contact.

    Returns:
        The six shape indices, right thigh first.
    """
    timeconst, ratio = config.solref
    # Newton passes a shape's ke/kd to MuJoCo as solref = (2 / kd, kd / (2 sqrt(ke))).
    shape_config = newton.ModelBuilder.ShapeConfig(
        density=0.0, mu=config.friction, ke=1.0 / (timeconst * ratio) ** 2, kd=2.0 / timeconst, gap=0.0
    )
    ends, radius = local_capsules(config, calibration), radii(config)
    shapes = []
    for index in range(1, 7):
        start, end = ends[index]
        length = float(np.linalg.norm(end - start))
        shapes.append(
            builder.add_shape_capsule(
                bodies[index],
                xform=wp.transform(
                    wp.vec3(*(0.5 * (start + end))),
                    wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*((end - start) / length))),
                ),
                radius=float(radius[index]),
                half_height=0.5 * length,
                cfg=shape_config,
                label=f"capsule_{BODY_LABELS[index]}",
            )
        )
    for leg in (shapes[:3], shapes[3:]):
        for k, a in enumerate(leg):
            for b in leg[k + 1 :]:
                builder.add_shape_collision_filter_pair(a, b)
    return shapes


def segment_gap(p0, p1, q0, q1) -> np.ndarray:
    """Distance between the axes of two line segments, endpoints ``[..., 3]``.

    The NumPy twin of the kernel's closest-point test; the test suite keeps them equal. The
    stationary point of the pair-distance quadratic is undefined for parallel axes, so the
    minimum is taken over it and over the four edges of the parameter square, which together
    contain the constrained minimum.
    """
    u, v, r = p1 - p0, q1 - q0, p0 - q0
    A = np.einsum("...i,...i->...", u, u)
    B = np.einsum("...i,...i->...", u, v)
    C = np.einsum("...i,...i->...", u, r)
    E = np.einsum("...i,...i->...", v, v)
    F = np.einsum("...i,...i->...", v, r)
    safe_a, safe_e = np.where(A > EPS_SQUARED_M2, A, 1.0), np.where(E > EPS_SQUARED_M2, E, 1.0)
    stationary = np.clip((B * F - C * E) / np.where(A * E - B * B > EPS_SQUARED_M2, A * E - B * B, 1.0), 0.0, 1.0)
    candidates = [
        (stationary, np.clip((B * stationary + F) / safe_e, 0.0, 1.0)),
        (np.clip(-C / safe_a, 0.0, 1.0), np.zeros_like(stationary)),
        (np.clip((B - C) / safe_a, 0.0, 1.0), np.ones_like(stationary)),
        (np.zeros_like(stationary), np.clip(F / safe_e, 0.0, 1.0)),
        (np.ones_like(stationary), np.clip((B + F) / safe_e, 0.0, 1.0)),
    ]
    return np.min(
        [np.linalg.norm((p0 + s[..., None] * u) - (q0 + t[..., None] * v), axis=-1) for s, t in candidates],
        axis=0,
    )


def capsule_axes(calibration, q, config: LegContactConfig) -> np.ndarray:
    """Capsule ends of one trial in the world frame, shape [frames, 7, 2, 3]."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    frames = _model_frames(q, calibration)
    ends = local_capsules(config, calibration)
    axes = np.empty((q.shape[0], 7, 2, 3))
    for i, label in enumerate(BODY_LABELS):
        axes[:, i] = frames[_ORIGIN_KEY[i]][:, None, :] + np.einsum(
            "fij,kj->fki", frames[label], ends[i], optimize=True
        )
    return axes


def capsule_gaps(axes: np.ndarray, config: LegContactConfig) -> np.ndarray:
    """Signed clearance of every pair, shape [frames, 9]; negative is penetration."""
    radius = radii(config)
    return np.stack(
        [
            segment_gap(axes[:, a, 0], axes[:, a, 1], axes[:, b, 0], axes[:, b, 1]) - radius[a] - radius[b]
            for a, b in _PAIRS
        ],
        axis=-1,
    )


@wp.func
def _axis_squared(a0: wp.vec3, ua: wp.vec3, b0: wp.vec3, ub: wp.vec3, s: float, t: float) -> float:
    """Squared distance between the points a fraction ``s`` and ``t`` along two axes."""
    delta = (a0 + s * ua) - (b0 + t * ub)
    return wp.dot(delta, delta)


@wp.kernel
def _inter_leg(
    gains: wp.array[float],
    damping: wp.array[float],
    status: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    ends: wp.array[wp.vec3],
    radius: wp.array[float],
    depth: wp.array[float],
    force: wp.array[float],
    pair: wp.array[int],
):
    """Add the two-leg capsule penalty forces to ``body_f`` and keep the deepest pair.

    One thread per world, so each leg body is written by that thread alone and the
    kernel needs no atomic. A zero ``gains[0]`` leaves ``body_f`` untouched and only
    records penetration.

    ``gains`` is ``[stiffness N/m, force cap N, friction]`` and ``damping`` the per-pair
    normal damping; both are arrays rather than constants so the environment can ramp them
    while its step stays captured in a CUDA graph.
    """
    w = wp.tid()
    if status[w] != 0:
        return
    stiffness = gains[0]
    cap = gains[1]
    mu = gains[2]
    base = w * 7
    worst = float(0.0)
    worst_pair = int(-1)
    peak = float(0.0)
    for i in range(3):
        gi = base + i + 1
        a0 = wp.transform_point(body_q[gi], ends[2 * gi])
        a1 = wp.transform_point(body_q[gi], ends[2 * gi + 1])
        ua = a1 - a0
        ca = wp.transform_point(body_q[gi], body_com[gi])
        wa = wp.spatial_bottom(body_qd[gi])
        for j in range(3):
            gj = base + j + 4
            b0 = wp.transform_point(body_q[gj], ends[2 * gj])
            b1 = wp.transform_point(body_q[gj], ends[2 * gj + 1])
            ub = b1 - b0
            wb = a0 - b0
            A = wp.dot(ua, ua)
            B = wp.dot(ua, ub)
            C = wp.dot(ua, wb)
            E = wp.dot(ub, ub)
            F = wp.dot(ub, wb)
            safe_a = wp.max(A, 1.0e-12)
            safe_e = wp.max(E, 1.0e-12)
            # The stationary point of the pair-distance quadratic plus the four edges of the
            # parameter square contain its constrained minimum. Testing all five keeps the
            # answer exact when the axes are parallel, where the stationary point is gone.
            s = wp.clamp((B * F - C * E) / wp.max(A * E - B * B, 1.0e-12), 0.0, 1.0)
            t = wp.clamp((B * s + F) / safe_e, 0.0, 1.0)
            best_s = s
            best_t = t
            best = _axis_squared(a0, ua, b0, ub, best_s, best_t)
            for edge in range(4):
                if edge == 0:
                    s = wp.clamp(-C / safe_a, 0.0, 1.0)
                    t = float(0.0)
                if edge == 1:
                    s = wp.clamp((B - C) / safe_a, 0.0, 1.0)
                    t = float(1.0)
                if edge == 2:
                    s = float(0.0)
                    t = wp.clamp(F / safe_e, 0.0, 1.0)
                if edge == 3:
                    s = float(1.0)
                    t = wp.clamp((B + F) / safe_e, 0.0, 1.0)
                closer = _axis_squared(a0, ua, b0, ub, s, t)
                if closer < best:
                    best = closer
                    best_s = s
                    best_t = t
            pa = a0 + best_s * ua
            pb = b0 + best_t * ub
            delta = pa - pb
            touch = 0.5 * (pa + pb)
            span = wp.sqrt(best)
            gap = span - radius[gi] - radius[gj]
            if gap >= 0.0:
                continue
            # The pair force acts in the transverse plane, because limbs in contact slide
            # past each other vertically rather than lifting each other or driving a foot
            # into the ground. Axes directly above one another give no transverse direction,
            # so those push along the coronal axis: right leg toward -y, left toward +y.
            across = wp.vec3(delta[0], delta[1], 0.0)
            across_sq = wp.dot(across, across)
            normal = wp.vec3(0.0, -1.0, 0.0)
            if across_sq > 1.0e-12:
                normal = across / wp.sqrt(across_sq)
            cb = wp.transform_point(body_q[gj], body_com[gj])
            va = wp.spatial_top(body_qd[gi]) + wp.cross(wa, touch - ca)
            vb = wp.spatial_top(body_qd[gj]) + wp.cross(wp.spatial_bottom(body_qd[gj]), touch - cb)
            slide = va - vb
            closing = wp.dot(slide, normal)
            cd = damping[3 * i + j]
            normal_force = stiffness * (-gap) - cd * closing
            normal_force = wp.min(cap, wp.max(0.0, normal_force))
            if stiffness > 0.0:
                total = normal_force * normal
                tangent = -cd * (slide - closing * normal)
                limit = mu * normal_force
                slip = wp.length(tangent)
                if slip > limit and slip > 0.0:
                    tangent = tangent * (limit / slip)
                total = total + tangent
                body_f[gi] = body_f[gi] + wp.spatial_vector(total, wp.cross(touch - ca, total))
                body_f[gj] = body_f[gj] - wp.spatial_vector(total, wp.cross(touch - cb, total))
                peak = wp.max(peak, wp.length(total))
            if -gap > worst:
                worst = -gap
                worst_pair = 3 * i + j
    if worst > depth[w]:
        depth[w] = worst
        pair[w] = worst_pair
    if peak > force[w]:
        force[w] = peak


class LegContact:
    """Capsules, penalty law, and penetration record of one environment's two legs.

    Built once from the world calibrations and the actuated body masses; :meth:`apply`
    answers the nine capsule pairs of one substep.
    """

    def __init__(
        self,
        config: LegContactConfig,
        calibrations: list,
        world_trials: np.ndarray,
        body_mass: np.ndarray,
        device,
    ):
        self.config = config
        self.worlds = len(world_trials)
        mass = np.asarray(body_mass, dtype=float).reshape(self.worlds, 7)
        ends = np.zeros((self.worlds * 7, 2, 3))
        radius = np.zeros(self.worlds * 7)
        for world, trial in enumerate(world_trials):
            calibration = calibrations[int(trial)]
            local = local_capsules(config, calibration)
            ends[world * 7 : world * 7 + 7] = local
            radius[world * 7 : world * 7 + 7] = radii(config)
        # Critical damping of a pair uses its reduced mass; the profile is shared, so the
        # first world's masses stand for all of them.
        reduced = np.array(
            [mass[0, a] * mass[0, b] / (mass[0, a] + mass[0, b]) for a, b in _PAIRS],
            dtype=np.float32,
        )
        self._sqrt_reduced = np.sqrt(reduced)
        self.gains = wp.zeros(3, dtype=float, device=device)
        self.gains_host = wp.array(np.zeros(3), dtype=float, device="cpu")
        self.damping = wp.zeros(len(_PAIRS), dtype=float, device=device)
        self.damping_host = wp.array(np.zeros(len(_PAIRS)), dtype=float, device="cpu")
        self.control_steps = 0
        self._write_gains(0.0 if config.ramp_steps > 0 else 1.0)
        self.ends = wp.array(np.ascontiguousarray(ends.reshape(-1, 3), dtype=np.float32), dtype=wp.vec3, device=device)
        self.radius = wp.array(radius.astype(np.float32), dtype=float, device=device)
        self.device = device
        self.depth = wp.zeros(self.worlds, dtype=float, device=device)
        self.force = wp.zeros(self.worlds, dtype=float, device=device)
        self.pair = wp.zeros(self.worlds, dtype=int, device=device)

    def _write_gains(self, span: float) -> None:
        """Set the live gains to ``span`` of the way from ``ramp_from`` to the target profile."""
        start = self.config.ramp_from
        stiffness = start[0] + span * (self.config.stiffness_n_m - start[0])
        self.gains_host.assign(
            [stiffness, start[1] + span * (self.config.maximum_force_n - start[1]), self.config.friction]
        )
        self.damping_host.assign((2.0 * self.config.damping_ratio * np.sqrt(stiffness) * self._sqrt_reduced).tolist())
        wp.copy(self.gains, self.gains_host)
        wp.copy(self.damping, self.damping_host)

    def apply(self, status, body_q, body_qd, body_com, body_f):
        """One substep: add the capsule forces to ``body_f`` and deepen the penetration record."""
        wp.launch(
            _inter_leg,
            dim=self.worlds,
            inputs=[
                self.gains,
                self.damping,
                status,
                body_q,
                body_qd,
                body_com,
                body_f,
                self.ends,
                self.radius,
                self.depth,
                self.force,
                self.pair,
            ],
            device=self.device,
        )

    def clear(self):
        """Empty the per-control-step maxima and move the gain ramp one control step on.

        Called once per control step by the environment, which is the clock the ramp needs;
        the gains are device arrays, so the ramp takes effect inside an already captured step.
        """
        for array in (self.depth, self.force, self.pair):
            array.zero_()
        steps = self.config.ramp_steps
        if steps <= 0 or self.control_steps >= steps:
            return
        self.control_steps += 1
        self._write_gains(min(1.0, self.control_steps / steps))

    def describe(self) -> dict:
        return {
            "capsules": dict(zip(BODY_LABELS[1:], self.config.radius_m * 2, strict=True)),
            "pairs": list(PAIR_LABELS),
            "law": {
                "stiffness_n_m": float(self.gains.numpy()[0]),
                "damping_ratio": self.config.damping_ratio,
                "damping_n_s_m": self.damping.numpy().tolist(),
                "friction": self.config.friction,
                "maximum_force_n": float(self.gains.numpy()[1]),
                "ramp": {
                    "steps": self.config.ramp_steps,
                    "from_n_m": self.config.ramp_from[0],
                    "from_cap_n": self.config.ramp_from[1],
                    "control_steps_done": self.control_steps,
                },
            },
            "foot_span_m": list(self.config.foot_span_m),
            "foot_height_m": self.config.foot_height_m,
            "engine": self.config.engine,
            **({"mujoco_solref": list(self.config.solref)} if self.config.engine == "mujoco" else {}),
        }
