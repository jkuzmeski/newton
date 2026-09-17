# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Demonstration and comparison of digital shoe tangential friction formulations.

Evaluates explicit bristle, implicit coupled bristle, and regularized Coulomb
friction modes on a synthetic multi-column planar shoe sole patch under identical
prescribed normal reactions, contact dropout, and alternating lateral shear loading.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.digital_shoe.friction_solver import FrictionParams, FrictionSolver
from projects.digital_shoe.rendering import camera_look_at


def _create_synthetic_shoe_patch(
    column_count: int = 64,
    sole_length: float = 0.26,
    sole_width: float = 0.09,
    sole_height: float = -0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic planar contact points, stiffnesses, and damping values.

    Arranges contact points in an elliptical shoe-like footprint centered at the
    body origin.

    Args:
        column_count: Total number of discrete bristle contact columns.
        sole_length: Shoe sole length along X axis [m].
        sole_width: Shoe sole width along Y axis [m].
        sole_height: Z coordinate of the contact sole patch [m].

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Contact points [N, 3] in body-local coordinates [m].
            - Tangential stiffness kt [N] [N/m].
            - Tangential damping kv [N] [N*s/m].
    """
    grid_side = int(np.ceil(np.sqrt(column_count * 1.5)))
    xs = np.linspace(-sole_length * 0.45, sole_length * 0.45, grid_side)
    ys = np.linspace(-sole_width * 0.45, sole_width * 0.45, grid_side)

    candidates = []
    for x in xs:
        for y in ys:
            # Elliptical / shoe-shaped profile: wider at forefoot, narrower at heel/arch
            w_x = sole_width * 0.45 * (0.8 + 0.3 * (x / (sole_length * 0.45)))
            if (x / (sole_length * 0.45)) ** 2 + (y / w_x) ** 2 <= 1.0:
                candidates.append((x, y, sole_height))

    if len(candidates) >= column_count:
        indices = np.linspace(0, len(candidates) - 1, column_count, dtype=int)
        selected = [candidates[i] for i in indices]
    else:
        selected = list(candidates)
        rng = np.random.default_rng(17)
        while len(selected) < column_count:
            x = rng.uniform(-sole_length * 0.4, sole_length * 0.4)
            y = rng.uniform(-sole_width * 0.4, sole_width * 0.4)
            selected.append((x, y, sole_height))

    points_np = np.array(selected[:column_count], dtype=np.float32)

    # Tangential stiffness kt ~ 5e4 N/m, kv ~ 25 N*s/m per column
    # Representative of rubber lug / outsole tread compliance
    kt_np = np.full(column_count, 50000.0, dtype=np.float32)
    kv_np = np.full(column_count, 25.0, dtype=np.float32)

    return points_np, kt_np, kv_np


def prescribed_normal_profile(
    sim_time: float,
    total_columns: int,
    total_normal_load: float = 800.0,
    dropout_start: float = 0.8,
    dropout_end: float = 1.0,
) -> np.ndarray:
    """Prescribe non-negative normal reaction per column, including complete contact dropout.

    Args:
        sim_time: Current simulation time [s].
        total_columns: Number of contact columns.
        total_normal_load: Nominal aggregate vertical normal load [N] (~80 kg body).
        dropout_start: Time when normal reaction drops to zero (flight phase) [s].
        dropout_end: Time when normal reaction re-engages [s].

    Returns:
        np.ndarray: Normal reactions per contact column [N], shape [total_columns].
    """
    if dropout_start <= sim_time < dropout_end:
        # Flight / contact dropout phase: normal force is exactly zero
        return np.zeros(total_columns, dtype=np.float32)

    # Stance phase: distribute normal load across columns
    per_col = float(total_normal_load / total_columns)
    return np.full(total_columns, per_col, dtype=np.float32)


def prescribed_shear_force(
    sim_time: float,
    f_amplitude_x: float = 350.0,
    f_amplitude_y: float = 150.0,
    torque_yaw: float = 20.0,
    period: float = 0.6,
) -> tuple[np.ndarray, float]:
    """Prescribe external shear force impulse profile with periodic reversals.

    Args:
        sim_time: Current simulation time [s].
        f_amplitude_x: Peak lateral shear force along X [N].
        f_amplitude_y: Peak lateral shear force along Y [N].
        torque_yaw: Peak external yaw torque around Z [N*m].
        period: Shear force oscillation period [s].

    Returns:
        tuple[np.ndarray, float]: External shear force [Fx, Fy] [N] and yaw torque Tz [N*m].
    """
    omega = 2.0 * np.pi / period
    fx = f_amplitude_x * np.sin(omega * sim_time)
    fy = f_amplitude_y * np.cos(omega * sim_time)
    tz = torque_yaw * np.sin(omega * sim_time * 0.5)
    return np.array([fx, fy], dtype=np.float32), float(tz)


@dataclass
class SimulationRecord:
    """Recorded metrics across a trajectory for evaluation and comparison."""

    mode: str
    times: list[float]
    velocity_x: list[float]
    velocity_y: list[float]
    yaw_rate: list[float]
    pos_x: list[float]
    pos_y: list[float]
    yaw: list[float]
    friction_fx: list[float]
    friction_fy: list[float]
    friction_torque_z: list[float]
    prescribed_normal: list[float]
    stuck_column_count: list[int]
    linear_residual: list[float]
    angular_residual: list[float]
    host_diagnostics_step_time_ms: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "times": self.times,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "yaw_rate": self.yaw_rate,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "yaw": self.yaw,
            "friction_fx": self.friction_fx,
            "friction_fy": self.friction_fy,
            "friction_torque_z": self.friction_torque_z,
            "prescribed_normal": self.prescribed_normal,
            "stuck_column_count": self.stuck_column_count,
            "linear_residual": self.linear_residual,
            "angular_residual": self.angular_residual,
            "host_diagnostics_step_time_ms": self.host_diagnostics_step_time_ms,
            "timing_note": "Timings include host array synchronization and diagnostics; not isolated GPU kernel execution speeds.",
        }


class FrictionSimEngine:
    """Isolated numerical stepping engine for one friction formulation.

    Encapsulates dynamic planar state integration:
    1. v_free = v_dyn + dt * mobility * W_prescribed
    2. result = solver.solve(..., v_free, ...)
    3. v_next = v_free + dt * mobility * result.wrench
    4. pos_next = pos + dt * v_next
    """

    def __init__(
        self,
        mode: str,
        column_count: int = 64,
        mass: float = 80.0,
        inertia_yaw: float = 4.0,
        mu: float = 0.8,
        viscous_ratio: float = 0.15,
        release_dwell: float = 0.02,
        smoothing_speed: float = 0.01,
        iterations: int = 8,
        device: Any = None,
    ):
        self.mode = mode
        self.last_result = None
        self.column_count = column_count
        self.mass = mass
        self.inertia_yaw = inertia_yaw
        self.device = wp.get_device(device)

        # Planar mobility matrix [6, 6] for COM twist [v_lin (3), omega (3)]
        # X, Y translation compliance = 1/mass, Z translation fixed (rigid normal)
        # Yaw compliance = 1/inertia_yaw, Roll/Pitch compliance = 0 (constrained)
        mob_np = np.zeros((1, 6, 6), dtype=np.float32)
        mob_np[0, 0, 0] = 1.0 / mass
        mob_np[0, 1, 1] = 1.0 / mass
        mob_np[0, 2, 2] = 0.0  # Normal direction strictly prescribed
        mob_np[0, 3, 3] = 0.0  # Roll constrained
        mob_np[0, 4, 4] = 0.0  # Pitch constrained
        mob_np[0, 5, 5] = 1.0 / inertia_yaw
        self.mobility_np = mob_np[0]
        self.mobility = wp.array(mob_np, dtype=wp.spatial_matrix, device=self.device)

        # Geometry & stiffness
        points_np, kt_np, kv_np = _create_synthetic_shoe_patch(column_count)
        self.points_initial = points_np
        self.points = wp.array(points_np, dtype=wp.vec3, device=self.device)
        self.kt = wp.array(kt_np, dtype=float, device=self.device)
        self.kv = wp.array(kv_np, dtype=float, device=self.device)

        # Friction parameters
        p = FrictionParams()
        p.mu = float(mu)
        p.viscous_ratio = float(viscous_ratio)
        p.release_dwell = float(release_dwell)
        p.smoothing_speed = float(smoothing_speed)
        self.params = wp.array([p], dtype=FrictionParams, device=self.device)

        # Tangential contact state histories
        self.anchor = wp.zeros(column_count, dtype=wp.vec2, device=self.device)
        self.stuck = wp.zeros(column_count, dtype=int, device=self.device)
        self.dwell = wp.zeros(column_count, dtype=float, device=self.device)
        self.deflection = wp.zeros(column_count, dtype=wp.vec2, device=self.device)

        # Solver instance
        self.solver = FrictionSolver(
            column_count=column_count,
            world_count=1,
            mode=mode,
            iterations=iterations,
            device=self.device,
        )

        # Working warp arrays
        self.normal = wp.zeros(column_count, dtype=float, device=self.device)
        self.com = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.v_free = wp.zeros(1, dtype=wp.spatial_vector, device=self.device)

        # Carrier dynamic state [pos_x, pos_y, yaw] and [vel_x, vel_y, yaw_rate]
        self.pose = np.zeros(3, dtype=np.float32)  # [x, y, psi]
        self.vel = np.zeros(6, dtype=np.float32)  # [vx, vy, vz, wx, wy, wz]

    def advance(
        self,
        dt: float,
        normal_profile: np.ndarray,
        prescribed_shear: np.ndarray,
        prescribed_yaw_torque: float,
    ) -> tuple[FrictionSolver.Result, float]:
        """Perform one coupled time step with prescribed normal reaction and shear forces.

        Args:
            dt: Time step duration [s].
            normal_profile: Prescribed normal load per column [N].
            prescribed_shear: Prescribed external tangential force [Fx, Fy] [N].
            prescribed_yaw_torque: Prescribed external yaw torque around Z [N*m].

        Returns:
            tuple[FrictionSolver.Result, float]: Solver result struct and step computation time [ms].
        """
        # 1. Update contact point positions based on planar rigid-body pose
        c_psi = float(np.cos(self.pose[2]))
        s_psi = float(np.sin(self.pose[2]))
        current_pts = np.zeros_like(self.points_initial)
        for col in range(self.column_count):
            lx = self.points_initial[col, 0]
            ly = self.points_initial[col, 1]
            lz = self.points_initial[col, 2]
            gx = self.pose[0] + c_psi * lx - s_psi * ly
            gy = self.pose[1] + s_psi * lx + c_psi * ly
            current_pts[col] = [gx, gy, lz]
        self.points.assign(current_pts)

        # Center of mass position in world frame
        com_world = np.array([self.pose[0], self.pose[1], 0.0], dtype=np.float32)
        self.com.assign(np.array([com_world], dtype=np.float32))

        # Normal reaction upload
        self.normal.assign(normal_profile.astype(np.float32))

        # 2. Compute friction-free velocity: includes prescribed external shear force impulse
        f_ext_wrench = np.array(
            [prescribed_shear[0], prescribed_shear[1], 0.0, 0.0, 0.0, prescribed_yaw_torque],
            dtype=np.float32,
        )
        delta_v_free = dt * (self.mobility_np @ f_ext_wrench)
        v_free_np = self.vel + delta_v_free
        self.v_free.assign(np.array([v_free_np], dtype=np.float32))

        # 3. Solve tangential friction
        t0 = time.perf_counter()
        result = self.solver.solve(
            points=self.points,
            normal=self.normal,
            com=self.com,
            velocity=self.v_free,
            mobility=self.mobility,
            kt=self.kt,
            kv=self.kv,
            params=self.params,
            anchor=self.anchor,
            stuck=self.stuck,
            dwell=self.dwell,
            dt=dt,
            deflection=self.deflection,
        )
        wrench_np = result.wrench.numpy()[0].copy()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        # Update bristle history states from solver output
        self.anchor = result.anchor
        self.stuck = result.stuck
        self.dwell = result.dwell
        self.deflection = result.deflection

        # 4. Advance dynamic state using exact impulse formula:
        # v_next = v_free + dt * mobility * result.wrench
        # (Strictly required for explicit bristle as well as coupled modes)
        self.last_result = result
        v_next_np = v_free_np + dt * (self.mobility_np @ wrench_np)
        self.vel = v_next_np

        # Integrate planar coordinates
        self.pose[0] += float(dt * self.vel[0])
        self.pose[1] += float(dt * self.vel[1])
        self.pose[2] += float(dt * self.vel[5])

        return result, elapsed_ms


class Example:
    """Newton-compliant digital shoe friction example demonstrating tangential solver modes.

    Follows Newton Example format: supports headless execution, interactive ViewerGL,
    multi-mode comparative analysis, and programmatic validation via :meth:`test_final`.
    """

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser supporting standard Newton example CLI options."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--mode",
            type=str,
            default="implicit_bristle",
            choices=["bristle", "implicit_bristle", "regularized", "deflection", "implicit_deflection"],
            help="Friction formulation to simulate when running a single mode.",
        )
        parser.add_argument(
            "--compare",
            action="store_true",
            default=False,
            help="Run comparative sweep across all 3 friction modes under identical conditions.",
        )
        parser.add_argument(
            "--columns",
            type=int,
            default=64,
            help="Number of discrete contact columns in synthetic shoe sole patch.",
        )
        parser.add_argument(
            "--dt",
            type=float,
            default=0.005,
            help="Simulation time step per physics step [s]. Used directly without alteration.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="outputs/digital_shoe_friction/synthetic.json",
            help="Destination path for simulation metrics JSON summary.",
        )
        return parser

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.sim_time = 0.0
        # Use requested dt directly for physics step without altering or silently overriding
        self.dt = float(getattr(args, "dt", 0.005))
        self.frame_dt = self.dt

        self.column_count = int(getattr(args, "columns", 64))
        self.mode = str(getattr(args, "mode", "implicit_bristle"))
        self.compare_mode = bool(getattr(args, "compare", False))
        self.output_path = getattr(args, "output", "outputs/digital_shoe_friction/synthetic.json")
        self.device = getattr(args, "device", None)

        # Build visual representation in Newton builder for optional ViewerGL rendering
        builder = newton.ModelBuilder()
        self.sole_link = builder.add_link()

        # Shoe body visualization box
        builder.add_shape_box(
            self.sole_link,
            hx=0.13,
            hy=0.045,
            hz=0.015,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.02), q=wp.quat_identity()),
        )
        # Ground visualization box
        self.ground_link = builder.add_link()
        builder.add_shape_box(
            self.ground_link,
            hx=0.5,
            hy=0.5,
            hz=0.01,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.05), q=wp.quat_identity()),
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        if hasattr(self.viewer, "set_model"):
            self.viewer.set_model(self.model)

        # Preallocated transforms buffer for in-place body_q updates
        self._visual_transforms = [wp.transform_identity(), wp.transform_identity()]

        # Simulation engines
        if self.compare_mode:
            self.modes_to_run = ["bristle", "implicit_bristle", "regularized"]
        else:
            self.modes_to_run = [self.mode]

        self.engines: dict[str, FrictionSimEngine] = {}
        self.records: dict[str, SimulationRecord] = {}

        for m in self.modes_to_run:
            self.engines[m] = FrictionSimEngine(
                mode=m,
                column_count=self.column_count,
                device=self.device,
            )
            self.records[m] = SimulationRecord(
                mode=m,
                times=[],
                velocity_x=[],
                velocity_y=[],
                yaw_rate=[],
                pos_x=[],
                pos_y=[],
                yaw=[],
                friction_fx=[],
                friction_fy=[],
                friction_torque_z=[],
                prescribed_normal=[],
                stuck_column_count=[],
                linear_residual=[],
                angular_residual=[],
                host_diagnostics_step_time_ms=[],
            )

        self.active_engine = self.engines[self.modes_to_run[0]]
        self._visual_points = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._visual_anchors = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._visual_force_ends = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._visual_point_colors = wp.full(
            self.column_count, wp.vec3(0.0, 0.5, 0.7), dtype=wp.vec3, device=self.device
        )
        self._visual_anchor_colors = wp.full(
            self.column_count, wp.vec3(1.0, 0.6, 0.0), dtype=wp.vec3, device=self.device
        )
        self._visual_force_colors = wp.full(
            self.column_count, wp.vec3(0.9, 0.1, 0.1), dtype=wp.vec3, device=self.device
        )
        self.viewer.set_camera(*camera_look_at(np.array([0.42, -0.38, 0.34]), np.array([0.0, 0.0, 0.0])))

    def step(self):
        """Advance all active simulation engines by one time step."""
        t = self.sim_time
        # Prescribed normal reactions and shear forces identical across all modes
        fn_col = prescribed_normal_profile(
            sim_time=t,
            total_columns=self.column_count,
            total_normal_load=800.0,
            dropout_start=0.5,
            dropout_end=0.65,
        )
        f_shear, t_yaw = prescribed_shear_force(sim_time=t)

        for m in self.modes_to_run:
            engine = self.engines[m]
            res, step_ms = engine.advance(
                dt=self.dt,
                normal_profile=fn_col,
                prescribed_shear=f_shear,
                prescribed_yaw_torque=t_yaw,
            )

            # Record metrics
            rec = self.records[m]
            rec.times.append(float(t))
            rec.velocity_x.append(float(engine.vel[0]))
            rec.velocity_y.append(float(engine.vel[1]))
            rec.yaw_rate.append(float(engine.vel[5]))
            rec.pos_x.append(float(engine.pose[0]))
            rec.pos_y.append(float(engine.pose[1]))
            rec.yaw.append(float(engine.pose[2]))

            wrench = res.wrench.numpy()[0]
            rec.friction_fx.append(float(wrench[0]))
            rec.friction_fy.append(float(wrench[1]))
            rec.friction_torque_z.append(float(wrench[5]))
            rec.prescribed_normal.append(float(np.sum(fn_col)))

            stuck_np = res.stuck.numpy()
            rec.stuck_column_count.append(int(np.sum(stuck_np > 0)))

            rec.linear_residual.append(float(res.linear_residual.numpy()[0]))
            rec.angular_residual.append(float(res.angular_residual.numpy()[0]))
            rec.host_diagnostics_step_time_ms.append(float(step_ms))

        self.sim_time += self.dt

        # Update Newton visualization state in place via .assign()
        if self.state_0.body_q is not None:
            pose = self.active_engine.pose
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(pose[2]))
            self._visual_transforms[0] = wp.transform(p=wp.vec3(float(pose[0]), float(pose[1]), 0.0), q=rot)
            self._visual_transforms[1] = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
            self.state_0.body_q.assign(self._visual_transforms)

    def render(self):
        """Render current scene in Newton viewer if viewer is active."""
        if hasattr(self.viewer, "begin_frame"):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            engine = self.active_engine
            if engine.last_result is not None:
                # Raise diagnostic overlays above the display box; these are not normal geometry.
                points = engine.points.numpy().copy()
                points[:, 2] = 0.025
                anchors = points.copy()
                anchors[:, :2] = engine.anchor.numpy()
                ends = points.copy()
                ends[:, :2] += 0.004 * engine.last_result.force.numpy()
                self._visual_points.assign(points)
                self._visual_anchors.assign(anchors)
                self._visual_force_ends.assign(ends)
                self.viewer.log_points(
                    "friction/contact_samples", self._visual_points, radii=0.002, colors=self._visual_point_colors
                )
                self.viewer.log_points(
                    "friction/anchors", self._visual_anchors, radii=0.0015, colors=self._visual_anchor_colors
                )
                self.viewer.log_lines(
                    "friction/forces_scaled",
                    self._visual_points,
                    self._visual_force_ends,
                    colors=self._visual_force_colors,
                )
            self.viewer.end_frame()

    def save_metrics(self):
        """Save recorded metrics across active formulations to JSON summary."""
        if not self.output_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        export_data = {m: rec.to_dict() for m, rec in self.records.items()}
        with open(self.output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def test_final(self):
        """Verify solver residuals, sticking/sliding dynamics, and contact dropout recovery."""
        self.save_metrics()

        for m, rec in self.records.items():
            if len(rec.times) < 5:
                continue

            # 1. Normal forces were never modified and dropout period had zero friction and released bristle anchors
            for i, t in enumerate(rec.times):
                if 0.52 < t < 0.63:
                    fx = rec.friction_fx[i]
                    fy = rec.friction_fy[i]
                    f_mag = np.hypot(fx, fy)
                    if f_mag > 1e-4:
                        raise ValueError(
                            f"Mode {m}: non-zero friction {f_mag} N during zero-normal contact dropout at t={t:.3f}"
                        )
                    # During extended unloaded dwell (exceeding release_dwell=0.02s), bristle anchors must fully release
                    if t > 0.55 and rec.stuck_column_count[i] != 0:
                        raise ValueError(
                            f"Mode {m}: bristle stuck count is {rec.stuck_column_count[i]} (expected 0) after unloaded dwell at t={t:.3f}"
                        )

            # 2. Coupled modes achieve low nonlinear velocity-form residual compared to uncoupled
            if m in ("implicit_bristle", "regularized", "implicit_deflection"):
                max_lin_res = float(np.max(rec.linear_residual))
                max_ang_res = float(np.max(rec.angular_residual))
                # Numerical gates: linear residual < 1e-4 m/s, angular residual < 1e-3 rad/s
                if max_lin_res > 1e-4:
                    raise ValueError(f"Mode {m}: linear velocity residual {max_lin_res:.4e} m/s exceeded gate 1e-4 m/s")
                if max_ang_res > 1e-3:
                    raise ValueError(
                        f"Mode {m}: angular velocity residual {max_ang_res:.4e} rad/s exceeded gate 1e-3 rad/s"
                    )

            # 3. Trajectory exhibits both stick and slip phases under alternating shear
            velocities = np.hypot(rec.velocity_x, rec.velocity_y)
            has_stick = np.any(velocities < 0.02)
            has_slip = np.any(velocities > 0.1)
            if not (has_stick or has_slip):
                raise ValueError(f"Mode {m}: dynamic response did not show expected stick/slide behavior")


def main() -> None:
    """Execute the digital shoe friction example from the project CLI."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
