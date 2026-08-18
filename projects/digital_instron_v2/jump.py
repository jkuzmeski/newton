# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Digital-Instron shoe on a jumping leg: drop -> settle -> jump -> land.

This module crosses the calibrated Digital-Instron midsole
(:mod:`projects.digital_instron_v2.core` /
:mod:`projects.digital_instron_v2.dynamics`) with an articulated Newton leg. The
fitted intact-shoe Hyperfoam-Maxwell-Pasternak column bed, sampled from the real
Puma midsole footprint, becomes the sole of the foot of a four-segment planar
leg (vertical pelvis slider + hip/knee/ankle hinges). A phase controller drives a
countermovement vertical jump; the calibrated foam is the only foot-ground
contact and its live wrench is integrated by :class:`newton.solvers.SolverFeatherstone`.
The sole's flat interface is aligned from heel and toe OpenSim contact geometry,
exercising the same project-local attachment adapter used by human models.

Two studies quantify how the shoe governs the movement:

* ``material_sweep`` perturbs each fitted parameter and measures the resulting
  jump-height, joint-work, and ground-reaction deviations.
* ``shape_comparison`` swaps the sole underside between the real shoe-last
  profile, a sphere, and an ellipsoid at fixed material.

Key finding: gross jump/landing mechanics are remarkably insensitive to the
fitted foam *modulus* and *nonlinearity* (a thin, locally stiff midsole is a
minor series compliance below the far more compliant musculoskeletal chain), but
strongly sensitive to the contact *geometry* and, secondarily, the Pasternak
lateral-coupling term. This is exactly why the shoe must be characterized on a
bench Digital Instron rather than inferred from whole-body motion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.opensim as osim
from projects.human_shoe import (
    OSIM_LOCAL_TO_Z_UP_JUMP_BASIS,
    FootShoeAttachmentContract,
    attach_sole_geometry,
    resolve_attachment,
)

from .core import CALIBRATED_MATERIAL, Material
from .dynamics import (
    FoundationConfig,
    MidsoleFoundation,
    attached_columns,
    build_foundation_geometry,
    column_colors,
)

MANIFEST = "DigitalInstron/manifest_v2.json"

# Single-leg hopper: segment masses [kg] (pelvis lumps the carried upper body).
MASS = (("pelvis", 12.0), ("thigh", 4.0), ("shank", 2.5), ("foot", 1.0))
GRAVITY = 9.81

_FOOT_BOX_HALF_X = 0.12
_FOOT_BOX_HALF_Y = 0.05
_FOOT_BOX_HALF_Z = 0.03
_FOOT_BOX_CENTER = np.array([0.06, 0.0, -0.02], dtype=np.float32)
_FOOT_BOX_BOTTOM_Z = float(_FOOT_BOX_CENTER[2] - _FOOT_BOX_HALF_Z)
_DEFAULT_DROP_HEIGHT_M = 0.012

# Joint targets [base_tz(m), hip, knee, ankle] (rad) for the jump phases.
_STAND = np.array([0.0, -0.05, 0.10, -0.05])
_CROUCH = np.array([0.0, -0.60, 1.20, -0.50])
_PUSH = np.array([0.0, 0.10, 0.10, 0.55])
_LAND = np.array([0.0, -0.50, 0.95, -0.40])


def _inertia(ixx: float, iyy: float, izz: float) -> wp.mat33:
    return wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)


@dataclass(frozen=True)
class SoleGeometry:
    """Foot-frame outsole geometry with a flat top interface."""

    bottom_local: np.ndarray
    top_local: np.ndarray
    rest_len: np.ndarray


_JUMP_FOOT_CONTACT_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="digital_instron_jump_foot">
    <BodySet name="bodyset"><objects>
      <Body name="foot"><mass>1.0</mass><mass_center>0 0 0</mass_center><inertia>1 1 1 0 0 0</inertia></Body>
    </objects></BodySet>
    <ContactGeometrySet name="contactgeometryset"><objects>
      <ContactSphere name="heel"><socket_frame>/bodyset/foot</socket_frame><location>-0.06 -0.02 0</location><radius>0.03</radius></ContactSphere>
      <ContactSphere name="toe"><socket_frame>/bodyset/foot</socket_frame><location>0.18 -0.02 0</location><radius>0.03</radius></ContactSphere>
    </objects></ContactGeometrySet>
  </Model>
</OpenSimDocument>"""


def _attach_sole_to_foot_contacts(sole: SoleGeometry, carrier: int) -> SoleGeometry:
    """Align a Digital Instron sole with the jump foot's OpenSim contact geometry."""
    osim_model = osim.parse_osim(_JUMP_FOOT_CONTACT_OSIM)
    imported = osim.OsimImportResult(
        model=osim_model,
        body_index={"ground": -1, "foot": carrier},
    )
    contract = FootShoeAttachmentContract(
        foot_body_name="foot",
        shoe_carrier_body_name="foot",
        translation_m=[0.0, 0.0, 0.0],
        rotation_deg=[0.0, 0.0, 0.0],
    )
    attached = attach_sole_geometry(
        resolve_attachment(imported, contract),
        sole.bottom_local,
        sole.top_local,
        output_basis=OSIM_LOCAL_TO_Z_UP_JUMP_BASIS,
    )
    return SoleGeometry(
        bottom_local=attached.bottom_local,
        top_local=attached.top_local,
        rest_len=attached.rest_len,
    )


def _sole_geometry(
    geo,
    shape: str = "last",
    sphere_radius: float = 0.28,
    ellipsoid_axes: tuple[float, float] = (0.9, 0.30),
) -> SoleGeometry:
    """Return a shared outsole profile under a flat foot-box interface."""
    cx, cy = geo.uv_m[:, 0].mean(), geo.uv_m[:, 1].mean()
    xl, yl = geo.uv_m[:, 0] - cx, geo.uv_m[:, 1] - cy
    if shape == "last":
        thickness = geo.slack_m
    elif shape == "sphere":
        r = sphere_radius
        prof = r - np.sqrt(np.maximum(r * r - (xl**2 + yl**2), 1.0e-6))
        thickness = geo.slack_m + (prof.max() - prof)
    elif shape == "ellipsoid":
        ax, ay = ellipsoid_axes
        u = np.clip((xl / ax) ** 2 + (yl / ay) ** 2, 0.0, 0.999)
        prof = 0.10 * (1.0 - np.sqrt(1.0 - u))
        thickness = geo.slack_m + (prof.max() - prof)
    else:
        raise ValueError(f"unknown sole shape {shape!r}")
    thickness = np.maximum(thickness, 0.0).astype(np.float32)
    top = np.column_stack(
        [xl + _FOOT_BOX_CENTER[0], yl + _FOOT_BOX_CENTER[1], np.full_like(thickness, _FOOT_BOX_BOTTOM_Z)]
    ).astype(np.float32)
    bottom = top.copy()
    bottom[:, 2] = top[:, 2] - thickness
    return SoleGeometry(bottom_local=bottom, top_local=top, rest_len=thickness)


def _standing_joint_q(drop_height: float, body_q: np.ndarray, carrier: int, sole: SoleGeometry) -> np.ndarray:
    """Return the standing joint pose with the lowest outsole point at ``drop_height`` [m]."""
    foot_q = np.asarray(body_q[carrier], dtype=np.float32)
    foot_xform = wp.transform(wp.vec3(*foot_q[:3]), wp.quat(*foot_q[3:]))
    world_bottom = min(
        float(wp.transform_point(foot_xform, wp.vec3(*sole.bottom_local[i]))[2]) for i in range(len(sole.bottom_local))
    )
    return np.array([drop_height - world_bottom, _STAND[1], _STAND[2], _STAND[3]], np.float32)


def build_leg() -> tuple[newton.Model, int]:
    """Build the planar leg (pelvis vertical slider, hip/knee/ankle hinges)."""
    b = newton.ModelBuilder()
    b.add_ground_plane()
    vis = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    z_hip, z_knee, z_ank = 0.84, 0.44, 0.04
    pelvis = b.add_link(
        xform=wp.transform(wp.vec3(0, 0, z_hip), wp.quat_identity()),
        com=wp.vec3(0, 0, 0.10),
        inertia=_inertia(0.15, 0.15, 0.10),
        mass=12.0,
        label="pelvis",
    )
    b.add_shape_box(pelvis, hx=0.10, hy=0.09, hz=0.12, cfg=vis, color=(0.35, 0.45, 0.75), label="pelvis_vis")
    thigh = b.add_link(
        xform=wp.transform(wp.vec3(0, 0, z_hip), wp.quat_identity()),
        com=wp.vec3(0, 0, -0.20),
        inertia=_inertia(0.053, 0.053, 0.004),
        mass=4.0,
        label="thigh",
    )
    b.add_shape_box(
        thigh,
        xform=wp.transform(wp.vec3(0, 0, -0.20), wp.quat_identity()),
        hx=0.06,
        hy=0.06,
        hz=0.22,
        cfg=vis,
        color=(0.55, 0.55, 0.75),
        label="thigh_vis",
    )
    shank = b.add_link(
        xform=wp.transform(wp.vec3(0, 0, z_knee), wp.quat_identity()),
        com=wp.vec3(0, 0, -0.20),
        inertia=_inertia(0.033, 0.033, 0.002),
        mass=2.5,
        label="shank",
    )
    b.add_shape_box(
        shank,
        xform=wp.transform(wp.vec3(0, 0, -0.20), wp.quat_identity()),
        hx=0.05,
        hy=0.05,
        hz=0.22,
        cfg=vis,
        color=(0.65, 0.55, 0.45),
        label="shank_vis",
    )
    foot = b.add_link(
        xform=wp.transform(wp.vec3(0, 0, z_ank), wp.quat_identity()),
        com=wp.vec3(0.06, 0, -0.02),
        inertia=_inertia(0.002, 0.005, 0.005),
        mass=1.0,
        label="foot",
    )
    b.add_shape_box(
        foot,
        xform=wp.transform(wp.vec3(*_FOOT_BOX_CENTER), wp.quat_identity()),
        hx=_FOOT_BOX_HALF_X,
        hy=_FOOT_BOX_HALF_Y,
        hz=_FOOT_BOX_HALF_Z,
        cfg=vis,
        color=(0.75, 0.55, 0.35),
        label="foot_vis",
    )
    jb = b.add_joint_prismatic(
        parent=-1,
        child=pelvis,
        axis=wp.vec3(0, 0, 1),
        parent_xform=wp.transform(wp.vec3(0, 0, z_hip), wp.quat_identity()),
        label="base_tz",
    )
    b.add_joint_revolute(parent=pelvis, child=thigh, axis=wp.vec3(0, 1, 0), label="hip")
    b.add_joint_revolute(
        parent=thigh,
        child=shank,
        axis=wp.vec3(0, 1, 0),
        parent_xform=wp.transform(wp.vec3(0, 0, -0.40), wp.quat_identity()),
        label="knee",
    )
    b.add_joint_revolute(
        parent=shank,
        child=foot,
        axis=wp.vec3(0, 1, 0),
        parent_xform=wp.transform(wp.vec3(0, 0, -0.40), wp.quat_identity()),
        label="ankle",
    )
    b.add_articulation([jb, jb + 1, jb + 2, jb + 3], label="leg")
    return b.finalize(), foot


@wp.kernel
def _control_k(
    jq: wp.array[wp.float32],
    jqd: wp.array[wp.float32],
    tgt: wp.array[wp.float32],
    kp: wp.array[wp.float32],
    kd: wp.array[wp.float32],
    dt: wp.float32,
    jf: wp.array[wp.float32],
    work: wp.array[wp.float32],
):
    i = wp.tid()
    tau = kp[i] * (tgt[i] - jq[i]) - kd[i] * jqd[i]
    if i == 0:
        tau = 0.0  # pelvis vertical slider is passive (gravity + foam only)
    jf[i] = tau
    work[i] += tau * jqd[i] * dt


@wp.kernel
def _maxcomp_k(comp: wp.array[wp.float32], out: wp.array[wp.float32]):
    wp.atomic_max(out, 0, comp[wp.tid()])


@wp.kernel
def _record_k(
    nf: wp.array[wp.float32],
    mc: wp.array[wp.float32],
    ctr: wp.array[wp.int32],
    grf: wp.array[wp.float32],
    cmp: wp.array[wp.float32],
):
    k = ctr[0]
    grf[k] = nf[0]
    cmp[k] = mc[0]
    mc[0] = 0.0
    ctr[0] = k + 1


def _phase(t: float, hold: bool):
    """Return (target, kp, kd) for the jump phase at time ``t`` [s]."""
    if hold:  # rigid-leg drop test: stiff hold of the standing posture
        return _STAND, np.array([0, 3000.0, 4200.0, 1600.0]), np.array([0, 80.0, 110.0, 45.0])
    if t < 0.40:  # drop + settle
        return _STAND, np.array([0, 700.0, 1000.0, 340.0]), np.array([0, 34.0, 50.0, 20.0])
    if t < 0.62:  # countermovement crouch
        a = (t - 0.40) / 0.22
        return _STAND + (_CROUCH - _STAND) * a, np.array([0, 500.0, 700.0, 240.0]), np.array([0, 28.0, 40.0, 16.0])
    if t < 0.76:  # explosive push-off
        a = (t - 0.62) / 0.14
        return _CROUCH + (_PUSH - _CROUCH) * a, np.array([0, 1300.0, 1900.0, 680.0]), np.array([0, 30.0, 44.0, 18.0])
    return _LAND, np.array([0, 520.0, 720.0, 250.0]), np.array([0, 30.0, 44.0, 18.0])  # compliant landing


def simulate_jump(
    material: Material,
    shape: str = "last",
    *,
    dt: float = 5.0e-5,
    duration: float = 1.6,
    drop_height: float = _DEFAULT_DROP_HEIGHT_M,
    hold: bool = False,
    use_graph: bool = True,
) -> dict:
    """Simulate one drop-settle-jump-land cycle and return traces + joint work.

    Args:
        material: Foam constitutive parameters.
        shape: Sole underside shape (see :func:`_sole_geometry`).
        dt: Substep [s]. duration: Simulated time [s].
        drop_height: Initial outsole clearance above ground [m].
        hold: If True, hold a stiff standing posture (rigid-leg drop test).
        use_graph: Capture the substep loop into a CUDA graph.
    """
    m, foot = build_leg()
    dev = wp.get_preferred_device()
    masses = np.array([v for _, v in MASS])
    mtot = float(masses.sum())
    body_com = m.body_com.numpy()
    geo = build_foundation_geometry(MANIFEST)
    n = len(geo.slack_m)
    sole = _attach_sole_to_foot_contacts(_sole_geometry(geo, shape), foot)
    cfg = FoundationConfig(stretch_floor=0.05, normal_damping=40.0, friction_stiffness=2.0e4, friction=20.0, mu=1.0)
    found = MidsoleFoundation(
        sole.bottom_local,
        np.zeros(n, np.float32),
        sole.rest_len,
        np.full(n, geo.area_m2, np.float32),
        geo.neighbors,
        geo.spacing_m,
        material,
        foot,
        m.body_com,
        cfg,
    )
    sol = newton.solvers.SolverFeatherstone(m)
    s0, s1, ctl = m.state(), m.state(), m.control()
    s0.joint_q.assign(np.array([0.0, _STAND[1], _STAND[2], _STAND[3]], np.float32))
    s0.joint_qd.zero_()
    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
    s0.joint_q.assign(_standing_joint_q(drop_height, s0.body_q.numpy(), foot, sole))
    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)

    tgt_d = wp.zeros(4, dtype=wp.float32, device=dev)
    kp_d = wp.zeros(4, dtype=wp.float32, device=dev)
    kd_d = wp.zeros(4, dtype=wp.float32, device=dev)
    work_d = wp.zeros(4, dtype=wp.float32, device=dev)
    frame_dt = 1.0 / 240.0
    sub = int(round(frame_dt / dt / 2)) * 2
    dt = frame_dt / sub
    nframes = int(duration / frame_dt)
    total = nframes * sub
    ctr = wp.zeros(1, dtype=wp.int32, device=dev)
    mc = wp.zeros(1, dtype=wp.float32, device=dev)
    grf = wp.zeros(total, dtype=wp.float32, device=dev)
    cmp = wp.zeros(total, dtype=wp.float32, device=dev)
    st = [s0, s1]

    def frame():
        for _ in range(sub):
            a, b = st[0], st[1]
            wp.launch(
                _control_k,
                dim=4,
                inputs=[a.joint_q, a.joint_qd, tgt_d, kp_d, kd_d, dt, ctl.joint_f, work_d],
                device=dev,
            )
            a.clear_forces()
            found.apply(a, dt)
            sol.step(a, b, ctl, None, dt)
            wp.launch(_maxcomp_k, dim=n, inputs=[found.compression, mc], device=dev)
            wp.launch(_record_k, dim=1, inputs=[found.normal_force, mc, ctr, grf, cmp], device=dev)
            st[0], st[1] = b, a

    graph = None
    ts, czs = [], []
    for fr in range(nframes):
        t = fr * frame_dt
        tgt, kp, kd = _phase(t, hold)
        tgt_d.assign(tgt.astype(np.float32))
        kp_d.assign(kp.astype(np.float32))
        kd_d.assign(kd.astype(np.float32))
        if use_graph and dev.is_cuda:
            if graph is None:
                with wp.ScopedCapture(device=dev) as cap:
                    frame()
                graph = cap.graph
            wp.capture_launch(graph)
        else:
            frame()
        bq = st[0].body_q.numpy()
        cz = 0.0
        for i in range(4):
            cz += masses[i] * wp.transform_point(wp.transform(*bq[i]), wp.vec3(*body_com[i]))[2]
        ts.append(t)
        czs.append(cz / mtot)
    work = work_d.numpy()
    return {
        "shape": shape,
        "ts": ts,
        "czs": czs,
        "dt": dt,
        "grf": grf.numpy().tolist(),
        "cmp": cmp.numpy().tolist(),
        "W_hip": float(work[1]),
        "W_knee": float(work[2]),
        "W_ankle": float(work[3]),
        "mtot": mtot,
    }


def compute_metrics(result: dict) -> dict:
    """Reduce one :func:`simulate_jump` trace to scalar jump / kinetic metrics."""
    bw = result["mtot"] * GRAVITY
    dt = result["dt"]
    g = np.array(result["grf"])
    c = np.array(result["cmp"])
    tt = np.arange(len(g)) * dt
    cz = np.array(result["czs"])
    ts = np.array(result["ts"])
    stand = cz[(ts > 0.30) & (ts < 0.39)].mean()
    fly = cz[ts > 0.6]
    apex = fly.max()
    apex_t = ts[ts > 0.6][int(np.argmax(fly))]
    drop_grf = float(g[tt < 0.40].max())
    push_grf = float(g[(tt >= 0.62) & (tt < 0.76)].max())
    post = tt > apex_t
    flight = np.where(g[post] < 0.1 * bw)[0]
    after = tt[post][flight[0]] if len(flight) else apex_t
    land = np.where((tt > after) & (g > 0.5 * bw))[0]
    if len(land):
        s = land[0]
        win = slice(s, min(s + int(0.15 / dt), len(g)))
        land_grf = float(g[win].max())
        loadrate = float(np.max(np.diff(g[win]) / dt)) / 1000.0 if win.stop > win.start + 1 else 0.0
        contact = float((g[s:] > 0.2 * bw).sum() * dt)
        land_comp = float(c[win].max())
    else:
        land_grf, loadrate, contact, land_comp = push_grf, 0.0, 0.0, float(c.max())
    return {
        "jump_mm": (apex - stand) * 1000.0,
        "drop_grf": drop_grf,
        "push_grf": push_grf,
        "land_grf": land_grf,
        "loadrate_kN_s": loadrate,
        "contact_s": contact,
        "land_comp_mm": land_comp * 1000.0,
        "bw": bw,
        "W_hip": result["W_hip"],
        "W_knee": result["W_knee"],
        "W_ankle": result["W_ankle"],
    }


class Example:
    """Live-viewer wrapper for the jump experiment."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.dev = wp.get_preferred_device()
        self.shape = getattr(args, "shape", "last")
        self.duration = getattr(args, "duration", 1.6)
        self.model, self.carrier = build_leg()
        self.geo = build_foundation_geometry(MANIFEST)
        self.sole = _attach_sole_to_foot_contacts(_sole_geometry(self.geo, self.shape), self.carrier)
        n = len(self.geo.slack_m)
        self.foundation = MidsoleFoundation(
            self.sole.bottom_local,
            np.zeros(n, np.float32),
            self.sole.rest_len,
            np.full(n, self.geo.area_m2, np.float32),
            self.geo.neighbors,
            self.geo.spacing_m,
            CALIBRATED_MATERIAL,
            self.carrier,
            self.model.body_com,
            FoundationConfig(stretch_floor=0.05, normal_damping=40.0, friction_stiffness=2.0e4, friction=20.0, mu=1.0),
        )
        self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.state_0, self.state_1, self.control = self.model.state(), self.model.state(), self.model.control()
        self.state_0.joint_q.assign(np.array([0.0, _STAND[1], _STAND[2], _STAND[3]], np.float32))
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_0.joint_q.assign(
            _standing_joint_q(_DEFAULT_DROP_HEIGHT_M, self.state_0.body_q.numpy(), self.carrier, self.sole)
        )
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_1.body_qd.assign(self.state_0.body_qd)
        self.frame_dt = 1.0 / 240.0
        self.sim_substeps = int(round(self.frame_dt / 5.0e-5 / 2)) * 2
        self.dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self._frame = 0
        self._points = wp.zeros(n, dtype=wp.vec3, device=self.dev)
        self._colors = wp.zeros(n, dtype=wp.vec3, device=self.dev)
        self._bottom_local = wp.array(
            np.ascontiguousarray(self.sole.bottom_local, np.float32), dtype=wp.vec3, device=self.dev
        )
        self._top_local = wp.array(
            np.ascontiguousarray(self.sole.top_local, np.float32), dtype=wp.vec3, device=self.dev
        )
        self._rest_len = wp.array(
            np.ascontiguousarray(self.sole.rest_len, np.float32), dtype=wp.float32, device=self.dev
        )
        self._z_free = wp.array(np.ascontiguousarray(self.geo.z_free_m, np.float32), dtype=wp.float32, device=self.dev)
        self._slack = wp.array(np.ascontiguousarray(self.geo.slack_m, np.float32), dtype=wp.float32, device=self.dev)
        self._foam_top = wp.zeros(n, dtype=wp.vec3, device=self.dev)
        self._carrier = self.carrier
        self.viewer.set_model(self.model)
        eye = wp.vec3(0.7, -0.9, 1.2)
        target = wp.vec3(0.15, 0.0, 0.35)
        self.viewer.set_camera(*_look_at(eye, target))

    def step(self):
        tgt, kp, kd = _phase(self.sim_time, False)
        tgt_d = wp.array(tgt.astype(np.float32), dtype=wp.float32, device=self.dev)
        kp_d = wp.array(kp.astype(np.float32), dtype=wp.float32, device=self.dev)
        kd_d = wp.array(kd.astype(np.float32), dtype=wp.float32, device=self.dev)
        work_d = wp.zeros(4, dtype=wp.float32, device=self.dev)
        for _ in range(self.sim_substeps):
            wp.launch(
                _control_k,
                dim=4,
                inputs=[
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                    tgt_d,
                    kp_d,
                    kd_d,
                    self.dt,
                    self.control.joint_f,
                    work_d,
                ],
                device=self.dev,
            )
            self.state_0.clear_forces()
            self.foundation.apply(self.state_0, self.dt)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.dt
        self._frame += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            column_colors,
            dim=len(self.geo.slack_m),
            inputs=[self.foundation.compression, 0.008, self._colors],
            device=self.dev,
        )
        wp.launch(
            attached_columns,
            dim=len(self.geo.slack_m),
            inputs=[
                self._carrier,
                self.state_0.body_q,
                self._bottom_local,
                self._rest_len,
                self._points,
                self._foam_top,
            ],
            device=self.dev,
        )
        self.viewer.log_lines("midsole_springs", self._points, self._foam_top, self._colors, width=0.0035)
        self.viewer.log_points("midsole_columns", self._points, radii=0.0028, colors=self._colors)
        self.viewer.end_frame()

    def test_final(self):
        assert self._frame > 0, "no frames simulated"
        assert np.all(np.isfinite(self.state_0.body_q.numpy())), "non-finite state"


def _scaled(base: Material, **factors) -> Material:
    d = {
        "instantaneous_shear_modulus_pa": base.instantaneous_shear_modulus_pa,
        "hyperfoam_exponent": base.hyperfoam_exponent,
        "equilibrium_fraction": base.equilibrium_fraction,
        "pasternak_n_per_m": base.pasternak_n_per_m,
    }
    for name, factor in factors.items():
        d[name] *= factor
    return Material(**d)


def material_sweep(base: Material = CALIBRATED_MATERIAL, shape: str = "last", fraction: float = 0.2, **kwargs) -> dict:
    """Perturb each fitted parameter by +/-``fraction`` and return metric deviations."""
    baseline = compute_metrics(simulate_jump(base, shape, **kwargs))
    out = {"baseline": baseline, "perturbations": {}}
    for name in ("instantaneous_shear_modulus_pa", "hyperfoam_exponent", "equilibrium_fraction", "pasternak_n_per_m"):
        for f in (1.0 - fraction, 1.0 + fraction):
            r = compute_metrics(simulate_jump(_scaled(base, **{name: f}), shape, **kwargs))
            out["perturbations"][f"{name}_{f:.2f}"] = r
    return out


def shape_comparison(base: Material = CALIBRATED_MATERIAL, **kwargs) -> dict:
    """Compare last / sphere / ellipsoid soles at fixed material."""
    return {shape: compute_metrics(simulate_jump(base, shape, **kwargs)) for shape in ("last", "sphere", "ellipsoid")}


def make_figure(shapes: dict, sweep: dict, path: str) -> None:
    """Render a 2x2 summary figure (COM + GRF traces, shape and material bars)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"last": "C0", "sphere": "C1", "ellipsoid": "C2"}
    for name in ("last", "sphere", "ellipsoid"):
        r = simulate_jump(CALIBRATED_MATERIAL, name)
        ax[0, 0].plot(r["ts"], r["czs"], color=colors[name], label=name)
        tt = np.arange(len(r["grf"])) * r["dt"]
        ax[0, 1].plot(tt, np.array(r["grf"]) / (r["mtot"] * GRAVITY), color=colors[name], lw=0.7, label=name)
    ax[0, 0].set(title="Center-of-mass height", xlabel="time [s]", ylabel="COM z [m]")
    ax[0, 0].legend()
    ax[0, 1].set(title="Ground reaction (body weights)", xlabel="time [s]", ylabel="GRF / BW")
    ax[0, 1].legend()

    names = list(shapes)
    jumps = [shapes[s]["jump_mm"] for s in names]
    lands = [shapes[s]["land_grf"] for s in names]
    x = np.arange(len(names))
    ax[1, 0].bar(x - 0.2, jumps, 0.4, label="jump [mm]", color="C0")
    twin = ax[1, 0].twinx()
    twin.bar(x + 0.2, lands, 0.4, label="landing GRF [N]", color="C3")
    ax[1, 0].set(title="Contact-shape effect", xticks=x, xticklabels=names, ylabel="jump height [mm]")
    twin.set_ylabel("landing GRF [N]")

    labels, djump, dgrf = [], [], []
    b = sweep["baseline"]
    short = {
        "instantaneous_shear_modulus_pa": "G_inst",
        "hyperfoam_exponent": "alpha",
        "equilibrium_fraction": "eq_frac",
        "pasternak_n_per_m": "Pasternak",
    }
    for key, r in sweep["perturbations"].items():
        name, f = key.rsplit("_", 1)
        labels.append(f"{short[name]}\n{'+' if float(f) > 1 else '-'}20%")
        djump.append(r["jump_mm"] - b["jump_mm"])
        dgrf.append(100.0 * (r["land_grf"] - b["land_grf"]) / b["land_grf"])
    x = np.arange(len(labels))
    ax[1, 1].bar(x - 0.2, djump, 0.4, label="d jump [mm]", color="C0")
    ax[1, 1].bar(x + 0.2, dgrf, 0.4, label="d land GRF [%]", color="C3")
    ax[1, 1].set(title="Material sweep (+/-20%)", xticks=x)
    ax[1, 1].set_xticklabels(labels, fontsize=7)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def _print_metrics(tag: str, r: dict) -> None:
    print(
        f"{tag:14s} jump={r['jump_mm']:6.1f}mm  drop={r['drop_grf']:6.0f}N  push={r['push_grf']:6.0f}N  "
        f"land={r['land_grf']:6.0f}N  contact={r['contact_s']:.2f}s  "
        f"W h/k/a={r['W_hip']:5.1f}/{r['W_knee']:5.1f}/{r['W_ankle']:5.1f} J"
    )


def _look_at(eye, target):
    """Return a camera pose tuple for a Z-up view."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    d /= np.linalg.norm(d)
    pitch = np.degrees(np.arcsin(d[2]))
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    return wp.vec3(*[float(v) for v in eye]), float(pitch), float(yaw)


def main() -> None:
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="null")
    parser.add_argument("--mode", choices=["jump", "sweep", "shapes", "drop"], default="jump")
    parser.add_argument("--shape", choices=["last", "sphere", "ellipsoid"], default="last")
    parser.add_argument("--duration", type=float, default=1.6)
    parser.add_argument("--figure", type=str, default="")
    parser.add_argument("--json", type=str, default="")
    viewer, args = newton.examples.init(parser)

    if args.mode == "jump" and (args.viewer != "null" or args.test):
        newton.examples.run(Example(viewer, args), args)
        return

    if args.mode == "jump":
        r = compute_metrics(simulate_jump(CALIBRATED_MATERIAL, args.shape, duration=args.duration))
        _print_metrics(f"jump[{args.shape}]", r)
        payload = r
    elif args.mode == "drop":
        base = compute_metrics(simulate_jump(CALIBRATED_MATERIAL, args.shape, hold=True, duration=0.7))
        print("rigid-leg drop test (material sensitivity of the isolated impact):")
        _print_metrics("calibrated", base)
        payload = {"baseline": base}
    elif args.mode == "shapes":
        payload = shape_comparison(duration=args.duration)
        print("Contact-shape comparison (calibrated material):")
        for name, r in payload.items():
            _print_metrics(name, r)
    else:  # sweep
        payload = material_sweep(shape=args.shape, duration=args.duration)
        print(f"Material sweep on '{args.shape}' sole (+/-20%):")
        b = payload["baseline"]
        _print_metrics("calibrated", b)
        for key, r in payload["perturbations"].items():
            name, f = key.rsplit("_", 1)
            sign = "+" if float(f) > 1 else "-"
            print(
                f"  {name:32s} {sign}20%: d_jump={r['jump_mm'] - b['jump_mm']:+5.1f}mm  "
                f"d_land_GRF={100 * (r['land_grf'] - b['land_grf']) / b['land_grf']:+5.1f}%  "
                f"d_W_knee={100 * (r['W_knee'] - b['W_knee']) / abs(b['W_knee']):+5.1f}%"
            )

    if args.figure:
        make_figure(shape_comparison(duration=args.duration), material_sweep(duration=args.duration), args.figure)
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
