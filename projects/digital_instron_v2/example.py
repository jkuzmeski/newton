# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Digital Instron Midsole
#
# A fully dynamic running-shoe midsole built from the calibrated column model
# in projects/digital_instron_v2. The midsole is a live two-surface column bed
# (projects/digital_shoe/column_bed.py) sampled from the shoe mesh and coupled
# into Newton rigid-body physics through state.body_f. Four modes reuse the
# same constitutive and contact implementation:
#
#   --mode instron  (default) Displacement-controlled digital Instron: a
#                   shoe-last crosshead presses the column tops under it while
#                   the rest of the shoe surface stays free and the outsole
#                   stands on the platen, and the force-displacement hysteresis
#                   loop is recorded.
#   --mode settle   A free, massive plate glued on the foam rests in stable
#                   equilibrium on the ground under gravity and resists a
#                   lateral load through Coulomb foam-shear friction.
#   --mode stride   A prescribed foot carries the shoe heel-to-toe over the
#                   ground, recording the ground reaction and its center of
#                   pressure.
#   --mode attached A fully dynamic, foot-mounted shoe with mass and inertia. A
#                   damped bilateral "upper" keeps the midsole coupled to the foot
#                   for the whole stride, so the shoe presses the foam into the
#                   ground in stance and the whole bed lifts clear with the foot in
#                   flight, with the stance/flight ground reaction recorded.
#
# --shoe DigitalInstron/<slug> (repeatable) replaces the scene with fitted shoes
# from the foam sweep: the last is registered at its fitted pose and replays a
# measured validate trace through the fitted material, shoes side by side.
#
# Command: uv run -m projects.digital_instron_v2.example --mode instron
#
###########################################################################

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.digital_shoe.column_bed import TOP_BONDED, TOP_FREE, TOP_PRESSED, ColumnBed
from projects.digital_shoe.rendering import camera_look_at as _look_at
from projects.digital_shoe.rendering import column_colors, column_endpoints
from projects.digital_shoe.runtime import FoundationConfig

from .core import CALIBRATED_MATERIAL, Material
from .dynamics import (
    attach_coupling,
    build_foundation_geometry,
    cyclic_displacement,
    load_measured_cycle,
    synthetic_stride,
)
from .dynamics import quat_conjugate as _quat_inv
from .dynamics import quat_multiply as _quat_mul
from .export_digital_shoe import last_body_platen, register_fitted_last
from .geometry import load_mesh, transform_mesh
from .last_replay import LastReplay, last_bed
from .registration import load_last
from .sweep_fit import DEFAULT_LAST, load_grid, load_trace

MANIFEST = "DigitalInstron/manifest_v2.json"
INSTRON_CYCLES = 6  # warm-up cycles before the reported hysteresis loop
# Foam damping per column for the free-body scenarios [N.s/m]. The whole midsole
# carries 910 columns, so a per-column dashpot that was merely stiff on the old
# fixture subset becomes a numerically explicit wall: 8.0 N.s/m per column is
# 7.3 kN.s/m under a 0.8 kg plate, about 65 times critical, and ``c * dt / m``
# reaches 4.7, so the settling plate rang instead of coming to rest. 0.5 N.s/m
# keeps real foam damping (about four times critical) with ``c * dt / m = 0.3``.
FREE_BODY_NORMAL_DAMPING_N_S_M = 0.5
SETTLE_PUSH_N = 4.0  # lateral load used to probe foam-shear friction [N]
SETTLE_MASS_KG = 0.8  # midsole + representative attachment mass [kg]
STRIDE_PERIOD_S = 0.6  # synthetic running-stride period [s]
ATTACHED_MASS_KG = 0.5  # dynamic shoe mass: midsole + effective foot/attachment [kg]
ATTACHED_PERIOD_S = 0.7  # attached running-stride period [s]
FOAM_SWEEP = Path("DigitalInstron/FoamSweep")  # fitted shoes and their processed sweeps
SHOE_SPACING_M = 0.06  # gap between fitted shoes standing side by side [m]
SHOE_COLOR_REF_M = 0.015  # column compression that saturates the --shoe colour [m]


def _loop_area(x: np.ndarray, y: np.ndarray) -> float:
    """Signed hysteresis-loop area (dissipated energy per cycle) via the shoelace formula [J]."""
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@wp.kernel
def _push(body: wp.int32, force: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    """Add a lateral probe force at the body's center of mass."""
    body_f[body] = body_f[body] + wp.spatial_vector(force, wp.vec3(0.0))


class Example:
    """Fully dynamic viscoelastic midsole driven as a digital Instron, a settling load, or a stride."""

    def __init__(self, viewer, args=None):
        newton.use_coord_layout_targets = True
        self.mode = getattr(args, "mode", "instron")
        manifest = getattr(args, "manifest", MANIFEST)
        fit = getattr(args, "fit", None)
        self.viewer = viewer
        self.device = wp.get_preferred_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.geo = build_foundation_geometry(manifest)
        self.material = Material(**json.loads(Path(fit).read_text())["material"]) if fit else CALIBRATED_MATERIAL
        self.column_count = len(self.geo.slack_m)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        anchor_local, connection, top_gap, config = self._build_mode(builder, manifest)

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._graph = None
        self._use_graph = self._attached and self.device.is_cuda and not getattr(args, "eager", False)
        if self._attached:
            q0 = self._target_pose(0.0)
            self.state_0.body_q.assign(q0.reshape(1, 7))
            self.state_1.body_q.assign(q0.reshape(1, 7))
            self._coupling_force = wp.zeros(1, dtype=wp.float32, device=self.device)

        self.bed = ColumnBed(
            anchor_local,
            self.geo.slack_m,
            self.geo.area_m2,
            self.geo.neighbors,
            self.geo.spacing_m,
            connection,
            self.material,
            config,
            np.array([self.carrier]),
            self.model.body_com,
            top_gap_m=top_gap,
            device=self.device,
        )
        self.bed.set_timesteps(self.sim_dt)
        # The Instron load cell weighs the crosshead; every other mode reports the ground reaction.
        self._force = self.bed.bed.carried_force if self.mode == "instron" else self.bed.bed.normal_force
        self._bottoms = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._tops = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._colors = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        # Compression that saturates the contact colour [m]; the attached stride penetrates
        # deeper than the bench Instron, so it needs a coarser scale.
        self._color_ref = 0.012 if self.mode == "attached" else 0.008

        # Diagnostics recorded once per frame.
        self.history: list[dict] = []

        self.viewer.set_model(self.model)
        span = float(np.ptp(self.geo.uv_m[:, 0]))
        eye = (self._cx + 0.9 * span, self._cy - 0.8 * span, 0.5 * span)
        target = (self._cx, self._cy, 0.02)
        self.viewer.set_camera(*_look_at(eye, target))

    # -- scene construction ------------------------------------------------
    def _build_mode(self, builder, manifest):
        """Configure the carrier body, column bottoms, top connection, and driver for the active mode.

        Every mode runs the same whole-midsole bed. In the Instron the crosshead
        presses the tops under its footprint and the rest of the shoe surface is
        free; in the other modes the carrier is the shoe or a plate glued on top
        of the whole bed.
        """
        geo = self.geo
        self._attached = False
        self._cx = float(geo.uv_m[:, 0].mean())
        self._cy = float(geo.uv_m[:, 1].mean())
        self._cz = float(geo.surface_m.mean())
        bonded = np.full(self.column_count, TOP_BONDED, dtype=np.int32)

        if self.mode == "settle":
            # A flat slab under the plate: every bottom starts on the ground.
            anchor_local = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], np.zeros(self.column_count)])
            self.carrier = builder.add_body(
                mass=SETTLE_MASS_KG,
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            )
            cfg = FoundationConfig(
                stretch_floor=0.05, normal_damping=FREE_BODY_NORMAL_DAMPING_N_S_M, mu=1.0, ground_height_m=0.0
            )
            self._add_plate_visual(builder)
            self._driven = False
            return anchor_local, bonded, None, cfg

        if self.mode in ("stride", "attached"):
            # The foam is the shoe's sole: anchor the outsole in the body frame so the whole
            # bed lifts with the shoe in flight and compresses only against the ground.
            self._com_z = float(geo.z_free_m.mean())
            self._cz = self._com_z
            anchor_local = np.column_stack(
                [geo.uv_m[:, 0] - self._cx, geo.uv_m[:, 1] - self._cy, geo.z_bottom_m - self._com_z]
            )
            span_x = float(np.ptp(geo.uv_m[:, 0]))
            if self.mode == "stride":
                self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
                self._add_last_visual(builder, manifest, at_com=True)
                self._stride = synthetic_stride(0.014, 5.0, 0.12 * span_x, STRIDE_PERIOD_S)
                self._driven = True
                return anchor_local, bonded, None, FoundationConfig(stretch_floor=0.05, ground_height_m=0.0)
            # Effective rotational inertia of the foot + shank that drives the pitch. A bare
            # thin-slab shoe inertia is far too small for the stiff, spatially distributed
            # foam and makes the explicit pitch mode diverge.
            self.carrier = builder.add_body(
                mass=ATTACHED_MASS_KG,
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.08),
            )
            self._add_last_visual(builder, manifest, at_com=True)
            # The stiff foam + light shoe is numerically stiff, so integrate the attached
            # stride with finer substeps than the driven/settle scenarios (converged at 128).
            self.sim_substeps = 128
            self.sim_dt = self.frame_dt / self.sim_substeps
            self._stride = synthetic_stride(0.024, 7.0, 0.08 * span_x, ATTACHED_PERIOD_S)
            # Compliant, bilateral shoe upper: near-silent in flight, stiff and damped in
            # stance so the foot presses the midsole down without separating.
            self._kp_lin, self._kd_lin = 1.0e5, 450.0
            self._kp_ang, self._kd_ang = 300.0, 40.0
            self._max_force = 20000.0
            self._build_target_trajectory()
            self._driven = False
            self._attached = True
            cfg = FoundationConfig(
                stretch_floor=0.05, normal_damping=FREE_BODY_NORMAL_DAMPING_N_S_M, mu=1.0, ground_height_m=0.0
            )
            return anchor_local, bonded, None, cfg

        self.mode = "instron"
        # The outsole stands on the platen; the crosshead presses the tops under it
        # across their rest clearance and leaves the rest of the shoe surface free.
        anchor_local = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.z_bottom_m])
        connection = np.where(geo.driven, TOP_PRESSED, TOP_FREE).astype(np.int32)
        top_gap = np.where(geo.driven, geo.gap0_m, 0.0)
        self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        time_s, disp_m, self._measured_force = load_measured_cycle(manifest)
        self._measured_disp = disp_m
        self._depth, self._period = cyclic_displacement(time_s, disp_m)
        self._add_last_visual(builder, manifest, at_com=False)
        self._driven = True
        return anchor_local, connection, top_gap, FoundationConfig(stretch_floor=0.05, ground_height_m=0.0)

    def _add_last_visual(self, builder, manifest, at_com):
        """Attach the shoe-last indenter mesh to the carrier body for rendering."""
        path = Path(manifest).resolve()
        cfg = json.loads(path.read_text())
        src = next(t for t in cfg["trials"] if t["fixture"] == "fullfoot_last")["indenter"]
        last = load_mesh(path.parent / src["path"], 0.001, src["rotation_deg"], src["crop_height_m"])
        transform_mesh(last, src.get("pose_rotation_deg", [0, 0, 0]), src.get("pose_translation_m", [0, 0, 0]))
        verts = np.asarray(last.vertices, np.float32).copy()
        # Match the contact-surface offset that build_foundation_geometry applied to the
        # physics so the rendered last rests on the foam top instead of inside the midsole.
        verts[:, self.geo.thickness_axis] += self.geo.indenter_shift_m
        verts[:, 2] -= self.geo.z_shift_m
        if at_com:
            verts[:, 0] -= self._cx
            verts[:, 1] -= self._cy
            verts[:, 2] -= self._cz
        mesh = newton.Mesh(verts, np.asarray(last.faces, np.int32).flatten())
        builder.add_shape_mesh(
            self.carrier,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.7, 0.72, 0.75),
            label="shoe_last",
        )

    def _add_plate_visual(self, builder):
        """Attach a thin load plate to the free midsole body for rendering."""
        hx = 0.5 * float(np.ptp(self.geo.uv_m[:, 0])) + 0.01
        hy = 0.5 * float(np.ptp(self.geo.uv_m[:, 1])) + 0.01
        builder.add_shape_box(
            self.carrier,
            xform=wp.transform(wp.vec3(self._cx, self._cy, float(self.geo.slack_m.mean()) + 0.006), wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=0.005,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.55, 0.45, 0.35),
            label="load_plate",
        )

    # -- simulation --------------------------------------------------------
    def _carrier_pose(self, t):
        """Return the prescribed carrier transform (7-vector) for a driven mode at time ``t``."""
        if self.mode == "stride":
            pos, rot = self._stride(t)
            return np.array(
                [self._cx + pos[0], self._cy + pos[1], self._cz + pos[2], rot[0], rot[1], rot[2], rot[3]],
                dtype=np.float32,
            )
        depth = self._depth(t)
        return np.array([0.0, 0.0, -depth, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _target_pose(self, t):
        """Return the prescribed foot/shoe target transform (7-vector) for the attached mode."""
        pos, rot = self._stride(t)
        return np.array(
            [self._cx + pos[0], self._cy + pos[1], self._com_z + pos[2], rot[0], rot[1], rot[2], rot[3]],
            dtype=np.float32,
        )

    def _pose_velocity(self, prev, cur, dt):
        """Return the (linear, angular) velocity carrying the target from ``prev`` to ``cur``."""
        v = (cur[:3] - prev[:3]) / dt
        q_rel = _quat_mul(cur[3:7], _quat_inv(prev[3:7]))
        if q_rel[3] < 0.0:
            q_rel = -q_rel
        w = 2.0 * q_rel[:3] / dt
        return v, w

    def _build_target_trajectory(self):
        """Precompute one stride period of per-substep target poses/velocities on the GPU.

        The prescribed foot stride is periodic, so evaluating it once into device arrays lets
        the attached substep loop run with no per-substep host work, which in turn lets the
        whole frame be captured into a single CUDA graph.
        """
        period = int(round(ATTACHED_PERIOD_S / self.sim_dt))
        poses = np.stack([self._target_pose(j * self.sim_dt) for j in range(period)]).astype(np.float32)
        vels = np.zeros((period, 6), dtype=np.float32)
        for j in range(period):
            v, w = self._pose_velocity(poses[(j - 1) % period], poses[j], self.sim_dt)
            vels[j] = np.concatenate([v, w])
        self._period_substeps = period
        self._target_traj = wp.array(np.ascontiguousarray(poses), dtype=wp.transform, device=self.device)
        self._target_vel_traj = wp.array(np.ascontiguousarray(vels), dtype=wp.spatial_vector, device=self.device)
        self._substep_counter = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _attached_substeps(self):
        """Advance the attached dynamic shoe by one frame using only device-side work."""
        for _ in range(self.sim_substeps):
            # The bed sets the carrier wrench, so the upper is added after it.
            self.bed.apply(self.state_0)
            wp.launch(
                attach_coupling,
                dim=1,
                inputs=[
                    self.carrier,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._target_traj,
                    self._target_vel_traj,
                    self._substep_counter,
                    self._period_substeps,
                    self._kp_lin,
                    self._kd_lin,
                    self._kp_ang,
                    self._kd_ang,
                    self._max_force,
                    self.state_0.body_f,
                    self._coupling_force,
                ],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        """Advance the foundation and, for the free-body modes, the rigid-body solver by one frame."""
        if self._attached:
            # The attached loop is fully device-side, so capture the whole frame into a CUDA
            # graph once and replay it -- this removes the per-substep kernel-launch overhead
            # that dominates the cost of this launch-bound scenario.
            if self._use_graph:
                if self._graph is None:
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._attached_substeps()
                    self._graph = capture.graph
                wp.capture_launch(self._graph)
            else:
                self._attached_substeps()
            self.sim_time += self.frame_dt
            return

        for _ in range(self.sim_substeps):
            if self._driven:
                self.state_0.body_q.assign(self._carrier_pose(self.sim_time).reshape(1, 7))
                self.state_0.body_qd.zero_()
                self.bed.apply(self.state_0)
            else:
                self.bed.apply(self.state_0)
                if self._pushing():
                    wp.launch(
                        _push,
                        dim=1,
                        inputs=[self.carrier, wp.vec3(SETTLE_PUSH_N, 0.0, 0.0), self.state_0.body_f],
                        device=self.device,
                    )
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def _pushing(self):
        """Return True while the settle-mode lateral friction probe is active."""
        return self.mode == "settle" and 1.5 <= self.sim_time < 1.9

    def step(self):
        self.simulate()
        self._record()

    def _record(self):
        """Record the crosshead load or the ground reaction and its pressure center."""
        center = self.bed.bed.pressure_center.numpy()[0]
        entry = {"t": self.sim_time, "normal_force_n": float(self._force.numpy()[0]), "cop_x_m": float(center[0])}
        if self.mode == "instron":
            entry["depth"] = self._depth(self.sim_time)
        elif self.mode == "stride":
            body_q = self.state_0.body_q.numpy()[self.carrier]
            entry["foot_x"] = float(body_q[0])
        elif self.mode == "attached":
            body_q = self.state_0.body_q.numpy()[self.carrier]
            entry["com_z"] = float(body_q[2])
            entry["com_x"] = float(body_q[0])
            entry["coupling_n"] = float(self._coupling_force.numpy()[0])
            entry["target_z"] = float(self._target_pose(self.sim_time)[2])
        else:
            body_q = self.state_0.body_q.numpy()[self.carrier]
            entry["com_z"] = float(body_q[2])
            entry["com_x"] = float(body_q[0])
        self.history.append(entry)

    # -- rendering ---------------------------------------------------------
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            column_colors,
            dim=self.column_count,
            inputs=[self.bed.bed.compression, self._color_ref, self._colors],
            device=self.device,
        )
        wp.launch(
            column_endpoints,
            dim=self.column_count,
            inputs=[
                self.carrier,
                self.state_0.body_q,
                self.bed.bed.anchor_local,
                self.bed.bed.rest_len,
                self.bed.bed.top,
                self.bed.bed.bottom,
                self._bottoms,
                self._tops,
            ],
            device=self.device,
        )
        self.viewer.log_lines("midsole_springs", self._bottoms, self._tops, self._colors, width=0.0035)
        self.viewer.log_points("midsole_columns", self._tops, radii=0.0028, colors=self._colors)
        self.viewer.end_frame()

    # -- validation --------------------------------------------------------
    def test_final(self):
        """Audit the recorded response for the active mode."""
        assert np.all(np.isfinite(self.state_0.body_q.numpy())), "non-finite carrier pose"
        assert len(self.history) > 0, "no frames simulated"
        forces = np.array([h["normal_force_n"] for h in self.history])
        assert np.all(np.isfinite(forces)), "non-finite foundation force"
        getattr(self, f"_test_{self.mode}")(forces)

    def _test_instron(self, forces):
        """Verify a dissipative, stiffening hysteresis loop.

        The peak is reported against the measured cycle, not gated: the default
        material was identified under the earlier single-surface bench law and
        the manifest last pose is not the fitted registration, so only
        ``--shoe`` replays a calibrated bed.
        """
        times = np.array([h["t"] for h in self.history])
        depth = np.array([h["depth"] for h in self.history])
        assert self.sim_time > (INSTRON_CYCLES - 1) * self._period, "run too short for a full hysteresis cycle"
        last = times >= (INSTRON_CYCLES - 1) * self._period
        loop_depth, loop_force = depth[last], forces[last]
        assert loop_depth.sum() > 0, "no compression cycle recorded"
        peak = float(loop_force.max())
        # A dissipative load-unload cycle (loading branch above unloading) traces the
        # (displacement, force) plane clockwise, so the dissipated energy is the negated
        # shoelace area; a positive signed area would mean unphysical energy generation.
        dissipated = -_loop_area(loop_depth * 1000.0, loop_force)  # mm * N = mJ
        loading = int(np.argmax(loop_depth)) + 1
        work = float(np.trapezoid(loop_force[:loading], loop_depth[:loading] * 1000.0))
        # The measured FR3 sweep loops lose 9-11% of the loading work; the energy is
        # gated as that fraction because the default material sets no absolute scale.
        assert dissipated > 0.05 * work, (
            f"hysteresis loop not dissipative: {dissipated:.1f} mJ of {work:.1f} mJ loading work"
        )
        assert forces.max() > 10.0 * forces[forces > 0].min(), "no nonlinear stiffening observed"
        print(
            f"[instron] peak {peak:.0f} N at {loop_depth.max() * 1000:.1f} mm "
            f"(measured {self._measured_force.max():.0f} N); hysteresis {dissipated / 1000.0:.3f} J/cycle, "
            f"{dissipated / work:.1%} of the loading work"
        )

    def _test_settle(self, forces):
        """Verify stable weight support and lateral grip from foam-shear friction."""
        com_z = np.array([h["com_z"] for h in self.history])
        com_x = np.array([h["com_x"] for h in self.history])
        settle = np.array([h["t"] for h in self.history]) < 1.5
        weight = SETTLE_MASS_KG * float(np.linalg.norm(self.model.gravity.numpy()[0]))
        pre = forces[settle]
        assert abs(pre[-1] - weight) < 0.25 * weight, f"foundation force {pre[-1]:.2f} N != weight {weight:.2f} N"
        assert com_z[settle][-1] > -0.006, f"midsole sank too far: {com_z[settle][-1] * 1000:.2f} mm"
        assert np.std(com_z[settle][-20:]) < 5.0e-4, "midsole did not reach a stable equilibrium"
        assert abs(com_x[-1]) < 0.02, (
            f"lateral grip failed: slid {com_x[-1] * 1000:.1f} mm under a {SETTLE_PUSH_N} N push"
        )
        print(
            f"[settle] {SETTLE_MASS_KG:.2f} kg held at {-com_z[settle][-1] * 1000:.2f} mm compression "
            f"(support {pre[-1]:.2f} N vs weight {weight:.2f} N); slid {abs(com_x).max() * 1000:.2f} mm "
            f"under a {SETTLE_PUSH_N:.0f} N push"
        )

    def _test_stride(self, forces):
        """Verify a rise-and-fall ground reaction and a heel-to-toe center of pressure."""
        times = np.array([h["t"] for h in self.history])
        cop_x = np.array([h["cop_x_m"] for h in self.history])
        foot_x = np.array([h["foot_x"] for h in self.history])
        assert forces.max() > 300.0, f"stride ground reaction too small: {forces.max():.0f} N"
        peak_frac = times[np.argmax(forces)] / STRIDE_PERIOD_S
        assert 0.1 < peak_frac < 0.6, f"ground reaction peak at {peak_frac:.2f} of stride, not mid-stance"
        assert forces.min() < 0.1 * forces.max(), "foot never lifted clear of the ground during swing"
        # Restrict the center-of-pressure roll to the first stance so the check is not
        # confused by the following stride's heel strike.
        stance = (times < 0.62 * STRIDE_PERIOD_S) & (forces > 0.15 * forces.max())
        rel = (cop_x - foot_x)[stance]
        assert rel.size > 4, "too few loaded stance frames to measure a roll"
        assert rel[-1] - rel[0] > 0.05, (
            f"center of pressure did not roll heel-to-toe: {(rel[-1] - rel[0]) * 1000:.1f} mm"
        )
        print(
            f"[stride] peak ground reaction {forces.max():.0f} N at {peak_frac * 100:.0f}% of stride; "
            f"center of pressure rolled {rel[0] * 1000:.0f} -> {rel[-1] * 1000:.0f} mm heel-to-toe"
        )

    def _test_attached(self, forces):
        """Verify a dynamic foot-mounted shoe: stance ground reaction, no flight blow-up, stays attached."""
        times = np.array([h["t"] for h in self.history])
        coupling = np.array([h["coupling_n"] for h in self.history])
        com_z = np.array([h["com_z"] for h in self.history])
        target_z = np.array([h["target_z"] for h in self.history])
        assert np.all(np.isfinite(coupling)), "non-finite shoe-upper force"
        assert np.all(np.isfinite(com_z)), "non-finite shoe pose"
        warm = times > ATTACHED_PERIOD_S  # skip the first stride's settling transient
        f, c = forces[warm], coupling[warm]
        stance = target_z[warm] < self._com_z - 0.001  # foot pressing the midsole below rest
        flight = target_z[warm] > self._com_z + 0.02  # foot lifted clear of the ground
        assert stance.any() and flight.any(), "stride never separated stance from flight"
        grf_stance, grf_flight = float(f[stance].max()), float(f[flight].max())
        assert grf_stance > 300.0, f"stance ground reaction too small: {grf_stance:.0f} N"
        # The foam is off the ground during swing, so no spurious flight contact should appear.
        assert grf_flight < 0.05 * grf_stance, f"spurious flight ground reaction: {grf_flight:.0f} N"
        c_stance, c_flight = float(c[stance].max()), float(c[flight].max())
        assert c_stance > 0.5 * grf_stance, "shoe upper did not transmit the stance load"
        assert c_flight < 0.2 * c_stance, f"shoe upper not unloaded in flight: {c_flight:.0f} N"
        # The bilateral upper keeps the shoe tracking the foot within a bounded lag all stride.
        lag = float(np.abs(com_z[warm] - target_z[warm]).max())
        assert lag < 0.03, f"shoe separated from foot by {lag * 1000:.1f} mm"
        print(
            f"[attached] {ATTACHED_MASS_KG:.1f} kg shoe: stance GRF {grf_stance:.0f} N, flight GRF {grf_flight:.1f} N; "
            f"shoe-upper force {c_flight:.0f} N (flight) -> {c_stance:.0f} N (stance); tracked foot within {lag * 1000:.1f} mm"
        )


@dataclass
class _FittedShoe:
    """One fitted shoe on the platen: its replayed bed, stroke, and render buffers."""

    label: str
    replay: LastReplay
    depth_m: np.ndarray
    rate_m_s: np.ndarray
    measured_force_n: np.ndarray
    replayed_force_n: np.ndarray
    anchors: wp.array
    bottoms: wp.array
    tops: wp.array
    colors: wp.array


class ShoeInstron:
    """Fitted shoes pressed by the Instron last through a measured validate trace.

    Each ``DigitalInstron/<slug>`` shoe is registered again from its
    ``material_fit.json`` and must land on the fitted pose
    (:func:`~projects.digital_instron_v2.export_digital_shoe.register_fitted_last`).
    The bed and stroke follow :func:`~projects.digital_instron_v2.sweep_fit.run`:
    :class:`~projects.digital_instron_v2.last_replay.LastReplay` presses the tops
    under the last tread, leaves the rest free, and starts the last at the
    registered preload indentation. The shoes stand side by side along x in the
    platen frame, and every frame advances the trace by ``frame_dt / slow_motion``.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_preferred_device()
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.trace_frame = 0
        last = load_last(DEFAULT_LAST)
        loaded = [self._load(Path(shoe), last, args.stance_ms, args.target_n) for shoe in args.shoe]
        self.trace_dt = loaded[0][2].dt_s
        if not np.allclose([replay.dt_s for _, _, replay, _, _ in loaded], self.trace_dt):
            raise ValueError("the shoes' traces have different frame intervals")
        self.frames_per_step = max(1, round(self.frame_dt / (args.slow_motion * self.trace_dt)))

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        self.shoes: list[_FittedShoe] = []
        x = 0.0
        for fit, registration, replay, trace, force in loaded:
            xy = registration.plot["top_p"][:, :2]
            shift = np.array([x - xy[:, 0].min(), -0.5 * (xy[:, 1].min() + xy[:, 1].max())])
            x += float(np.ptp(xy[:, 0])) + SHOE_SPACING_M
            vertices, faces = last_body_platen(DEFAULT_LAST, last, registration)
            vertices[:, :2] += shift
            body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), label=f"last_{fit['sweep']}")
            builder.add_shape_mesh(
                body,
                mesh=newton.Mesh(vertices.astype(np.float32), faces.astype(np.int32).flatten()),
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
                color=(0.7, 0.72, 0.75),
            )
            count = len(xy)
            self.shoes.append(
                _FittedShoe(
                    label=fit["sweep"],
                    replay=replay,
                    depth_m=replay.depth.numpy()[:, 0],
                    rate_m_s=replay.rate.numpy()[:, 0],
                    measured_force_n=trace["force_n"],
                    replayed_force_n=force,
                    anchors=wp.array(
                        np.column_stack((xy + shift, registration.bottom_above_platen_m)),
                        dtype=wp.vec3,
                        device=self.device,
                    ),
                    bottoms=wp.zeros(count, dtype=wp.vec3, device=self.device),
                    tops=wp.zeros(count, dtype=wp.vec3, device=self.device),
                    colors=wp.zeros(count, dtype=wp.vec3, device=self.device),
                )
            )
        self.model = builder.finalize()
        self.state = self.model.state()
        self.history: list[dict] = []

        self.viewer.set_model(self.model)
        span = x - SHOE_SPACING_M
        self.viewer.set_camera(*_look_at((0.5 * span, -0.78 * span, 0.24 * span), (0.5 * span, 0.0, 0.02)))

    def _load(self, shoe: Path, last, stance_ms: int, target_n: int):
        """Register a fitted shoe and settle its Maxwell history over the measured validate trace."""
        fit = json.loads((FOAM_SWEEP / "fit" / shoe.name / "material_fit.json").read_text())
        grid = load_grid(shoe / fit["side"] / "column_grid.npz")
        registration = register_fitted_last(grid, last, fit)
        sweep = FOAM_SWEEP / "processed" / fit["sweep"]
        condition = next(
            c
            for c in json.loads((sweep / "conditions.json").read_text())["conditions"]
            if c["role"] == "condition" and c["stance_ms"] == stance_ms and c["target_n"] == target_n
        )
        trace = load_trace(sweep / condition["validate_path"])
        depth = np.maximum(trace["displacement_m"], 0.0) + registration.qc["preload"]["indentation_m"]
        replay = LastReplay(
            last_bed(grid, registration), depth[:, None], float(np.mean(np.diff(trace["time_s"]))), candidates=1
        )
        force = replay.run([Material(**fit["material"])])[0, 0]
        if np.isnan(force).any():
            raise ValueError(f"{shoe.name}: the replayed validate cycle is not physical")
        return fit, registration, replay, trace, force

    def step(self):
        for _ in range(self.frames_per_step):
            for shoe in self.shoes:
                k = self.trace_frame % shoe.replay.frames
                carriers = shoe.replay.carriers
                carriers.body_q.assign(np.array([[0.0, 0.0, -shoe.depth_m[k], 0.0, 0.0, 0.0, 1.0]], np.float32))
                carriers.body_qd.assign(np.array([[0.0, 0.0, -shoe.rate_m_s[k], 0.0, 0.0, 0.0]], np.float32))
                shoe.replay.bed.apply(carriers)
            self.trace_frame += 1
            self.sim_time += self.trace_dt
        self.history.append(
            {
                "t": self.sim_time,
                "force_n": [float(shoe.replay.carriers.body_f.numpy()[0, 2]) for shoe in self.shoes],
                "frame": [(self.trace_frame - 1) % shoe.replay.frames for shoe in self.shoes],
            }
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.state.body_q.assign(
            np.array(
                [
                    [0.0, 0.0, -shoe.depth_m[(self.trace_frame - 1) % shoe.replay.frames], 0, 0, 0, 1]
                    for shoe in self.shoes
                ],
                np.float32,
            )
        )
        self.viewer.log_state(self.state)
        for index, shoe in enumerate(self.shoes):
            bed = shoe.replay.bed.bed
            wp.launch(
                column_colors,
                dim=len(shoe.anchors),
                inputs=[bed.compression, SHOE_COLOR_REF_M, shoe.colors],
                device=self.device,
            )
            wp.launch(
                column_endpoints,
                dim=len(shoe.anchors),
                inputs=[
                    0,
                    shoe.replay.carriers.body_q,
                    shoe.anchors,
                    bed.rest_len,
                    bed.top,
                    bed.bottom,
                    shoe.bottoms,
                    shoe.tops,
                ],
                device=self.device,
            )
            self.viewer.log_lines(f"columns_{index}", shoe.bottoms, shoe.tops, shoe.colors, width=0.0035)
            self.viewer.log_points(f"column_tops_{index}", shoe.tops, radii=0.0028, colors=shoe.colors)
        self.viewer.end_frame()

    def test_final(self):
        """Check the rendered strokes stay finite and the replayed peaks track the measured ones."""
        forces = np.array([h["force_n"] for h in self.history])
        assert len(forces) > 0 and np.all(np.isfinite(forces)), "non-finite last force"
        for shoe in self.shoes:
            peak, measured = shoe.replayed_force_n.max(), shoe.measured_force_n.max()
            print(f"[shoe instron] {shoe.label}: peak {peak:.0f} N (measured {measured:.0f} N)")
            assert abs(peak - measured) < 0.25 * measured, (
                f"{shoe.label}: peak {peak:.0f} N vs measured {measured:.0f} N"
            )


def main() -> None:
    """Run the source-backed mechanical example with its existing CLI."""
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["instron", "settle", "stride", "attached"],
        default="instron",
        help="Simulation scenario to run (instron | settle | stride | attached).",
    )
    parser.add_argument("--manifest", type=str, default=MANIFEST, help="Digital Instron manifest path.")
    parser.add_argument("--fit", type=str, default=None, help="sweep_fit material_fit.json to simulate.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Disable CUDA-graph capture in the attached mode (slower; for debugging).",
    )
    parser.add_argument(
        "--shoe",
        type=str,
        action="append",
        help="Fitted DigitalInstron/<slug> shoe for the Instron scene; repeat to stand shoes side by side.",
    )
    parser.add_argument("--stance-ms", type=int, default=200, help="Validate trace stance time for --shoe [ms].")
    parser.add_argument("--target-n", type=int, default=2800, help="Validate trace peak force for --shoe [N].")
    parser.add_argument("--slow-motion", type=float, default=1.0, help="Slow-down factor of the --shoe trace.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(ShoeInstron(viewer, args) if args.shoe else Example(viewer, args), args)


if __name__ == "__main__":
    main()
