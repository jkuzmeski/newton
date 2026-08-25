# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Show a portable Digital Shoe in Virtual Instron, drop, and rocker experiments."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

from .artifact import load_artifact
from .rendering import (
    attached_column_endpoints,
    column_colors,
    column_world_positions,
    deform_attached_mesh,
    deform_instron_mesh,
)
from .runtime import FoundationConfig, MidsoleFoundation

DEFAULT_ARTIFACT = "DigitalInstron/digital_shoe_showcase/digital_shoe.json"
INSTRON_WARMUP_CYCLES = 6


@wp.kernel
def _track_peak_metrics(
    force: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    peak_force: wp.array[wp.float32],
    peak_compression: wp.array[wp.float32],
):
    """Track substep peaks for prescribed mechanical experiments."""
    wp.atomic_max(peak_force, 0, force[0])
    wp.atomic_max(peak_compression, 0, compression[0])


@wp.kernel
def _accumulate_drop_metrics(
    carrier: wp.int32,
    body_qd: wp.array[wp.spatial_vector],
    force: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    dt: wp.float32,
    peak_force: wp.array[wp.float32],
    peak_compression: wp.array[wp.float32],
    impulse: wp.array[wp.float32],
    impact_started: wp.array[wp.int32],
    impact_complete: wp.array[wp.int32],
):
    """Accumulate the first contact event without per-substep host transfers."""
    if impact_complete[0] == 0:
        vertical_velocity = wp.spatial_top(body_qd[carrier])[2]
        if impact_started[0] != 0 and vertical_velocity >= 0.0:
            impact_complete[0] = 1
        elif force[0] > 1.0:
            impact_started[0] = 1
            wp.atomic_max(peak_force, 0, force[0])
            wp.atomic_max(peak_compression, 0, compression[0])
            wp.atomic_add(impulse, 0, force[0] * dt)


@wp.kernel
def _project_vertical_guide(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Keep the drop body on its vertical guide without an articulated joint."""
    pose = body_q[carrier]
    position = wp.transform_get_translation(pose)
    velocity = body_qd[carrier]
    linear = wp.spatial_top(velocity)
    body_q[carrier] = wp.transform(wp.vec3(0.0, 0.0, position[2]), wp.quat_identity())
    body_qd[carrier] = wp.spatial_vector(wp.vec3(0.0, 0.0, linear[2]), wp.vec3(0.0, 0.0, 0.0))


def _look_at(eye, target):
    delta = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    delta /= np.linalg.norm(delta)
    pitch = np.degrees(np.arcsin(delta[2]))
    yaw = np.degrees(np.arctan2(delta[1], delta[0]))
    return wp.vec3(*[float(value) for value in eye]), float(pitch), float(yaw)


class Example:
    """Run one artifact-only Digital Shoe mechanical experiment."""

    def __init__(self, viewer, args=None):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.mode = getattr(args, "mode", "instron")
        self.fixture_name = getattr(args, "fixture", "fullfoot_last")
        self.shoe = load_artifact(getattr(args, "artifact", DEFAULT_ARTIFACT))
        self.device = wp.get_preferred_device()
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 128 if self.mode == "drop" else 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.history: list[dict[str, float]] = []
        record_gif = getattr(args, "record_gif", None)
        self._gif_path = Path(record_gif) if record_gif else None
        self._gif_width = int(getattr(args, "gif_width", 480))
        self._gif_fps = int(getattr(args, "gif_fps", 30))
        self._gif_stride = int(getattr(args, "gif_stride", 2))
        if min(self._gif_width, self._gif_fps, self._gif_stride) <= 0:
            raise ValueError("GIF width, FPS, and stride must be positive")
        self._gif_frames = []

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        if self.mode == "instron":
            anchor, free_top, rest, area, neighbors, spacing = self._build_instron(builder)
        elif self.mode == "drop":
            anchor, free_top, rest, area, neighbors, spacing = self._build_drop(builder, args)
        elif self.mode == "rocker":
            anchor, free_top, rest, area, neighbors, spacing = self._build_rocker(builder)
        else:
            raise ValueError(f"unknown showcase mode {self.mode!r}")

        builder.color()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if self.mode == "drop":
            pose = np.array([0.0, 0.0, self.drop_height_m, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        else:
            pose = self._instron_pose(0.0) if self.mode == "instron" else self._rocker_pose(0.0)[0]
        self.state_0.body_q.assign(pose.reshape(1, 7))
        self.state_0.body_qd.zero_()
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_1.body_qd.assign(self.state_0.body_qd)

        self.foundation = MidsoleFoundation(
            anchor,
            free_top,
            rest,
            area,
            neighbors,
            spacing,
            self.shoe.material,
            self.carrier,
            self.model.body_com,
            self.foundation_config,
            self.device,
        )
        self.column_count = len(rest)
        self._anchor = wp.array(np.ascontiguousarray(anchor, np.float32), dtype=wp.vec3, device=self.device)
        self._rest = wp.array(np.ascontiguousarray(rest, np.float32), dtype=wp.float32, device=self.device)
        self._points = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._tops = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._colors = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        midsole_mesh = self.shoe.visual_mesh("midsole")
        self._mesh_source = wp.array(
            np.ascontiguousarray(midsole_mesh.vertices_m, np.float32), dtype=wp.vec3, device=self.device
        )
        self._mesh_points = wp.zeros(len(midsole_mesh.vertices_m), dtype=wp.vec3, device=self.device)
        self._mesh_indices = wp.array(
            np.ascontiguousarray(midsole_mesh.triangles.reshape(-1), np.int32), dtype=wp.int32, device=self.device
        )
        self._fixed_bottom = None
        self._mesh_column_index = None
        self._mesh_height_fraction = None
        if self.mode == "instron":
            fixture = self.shoe.instron_fixture(self.fixture_name)
            fixed = np.column_stack([fixture.carrier_anchor_m[:, :2], fixture.foam_bottom_m])
            self._fixed_bottom = wp.array(np.ascontiguousarray(fixed, np.float32), dtype=wp.vec3, device=self.device)
            column_index, height_fraction = self._instron_mesh_mapping(midsole_mesh.vertices_m, fixture)
            self._mesh_column_index = wp.array(column_index, dtype=wp.int32, device=self.device)
            self._mesh_height_fraction = wp.array(height_fraction, dtype=wp.float32, device=self.device)
        self._peak_force = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._peak_compression = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._impulse = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._impact_started = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._impact_complete = wp.zeros(1, dtype=wp.int32, device=self.device)

        self.viewer.set_model(self.model)
        span = float(np.ptp(anchor[:, 0]))
        self.viewer.set_camera(*_look_at((0.8 * span, -0.9 * span, 0.55 * span), (0.0, 0.0, 0.015)))

    @staticmethod
    def _instron_mesh_mapping(vertices: np.ndarray, fixture) -> tuple[np.ndarray, np.ndarray]:
        """Map each midsole vertex to its nearest active fixture column."""
        vertex_xy = np.asarray(vertices[:, :2], dtype=np.float64)
        column_xy = np.asarray(fixture.carrier_anchor_m[:, :2], dtype=np.float64)
        distance2 = np.sum((vertex_xy[:, None, :] - column_xy[None, :, :]) ** 2, axis=2)
        column_index = np.argmin(distance2, axis=1).astype(np.int32)
        nearest_distance2 = distance2[np.arange(len(vertices)), column_index]
        active = nearest_distance2 <= (1.25 * fixture.spacing_m) ** 2
        bottom = fixture.foam_bottom_m[column_index]
        fraction = np.clip((vertices[:, 2] - bottom) / fixture.rest_length_m[column_index], 0.0, 1.0)
        column_index[~active] = -1
        fraction[~active] = 0.0
        return np.ascontiguousarray(column_index), np.ascontiguousarray(fraction, dtype=np.float32)

    def _add_instron_indenter_visual(self, builder) -> None:
        """Render the calibrated shoe last or rearfoot punch instead of a proxy box."""
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        if self.fixture_name == "fullfoot_last":
            mesh_data = self.shoe.visual_mesh("fullfoot_last")
            mesh = newton.Mesh(
                np.asarray(mesh_data.vertices_m, dtype=np.float32),
                np.asarray(mesh_data.triangles, dtype=np.int32).reshape(-1),
            )
            builder.add_shape_mesh(
                self.carrier,
                mesh=mesh,
                cfg=visual,
                color=(0.68, 0.72, 0.76),
                label="calibrated_shoe_last",
            )
            return
        fixture_data = self.shoe.raw["instron_fixtures"][self.fixture_name]
        radius = float(fixture_data["indenter"]["radius_m"])
        top = float(np.max(self.shoe.instron_fixture(self.fixture_name).carrier_anchor_m[:, 2]))
        builder.add_shape_cylinder(
            self.carrier,
            xform=wp.transform(wp.vec3(0.0, 0.0, top + 0.005), wp.quat_identity()),
            radius=radius,
            half_height=0.005,
            cfg=visual,
            color=(0.68, 0.72, 0.76),
            label="calibrated_rearfoot_punch",
        )

    def _add_drop_mass_visual(self, builder, *, center_z: float, half_height: float) -> None:
        """Place the visible drop mass entirely above the calibrated midsole."""
        bed = self.shoe.column_bed
        hx = 0.5 * float(np.ptp(bed.anchor_bottom_m[:, 0])) + 0.01
        hy = 0.5 * float(np.ptp(bed.anchor_bottom_m[:, 1])) + 0.01
        builder.add_shape_box(
            self.carrier,
            xform=wp.transform(wp.vec3(0.0, 0.0, center_z), wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=half_height,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.32, 0.38, 0.46),
            label="guided_drop_mass",
        )

    def _build_instron(self, builder):
        fixture = self.shoe.instron_fixture(self.fixture_name)
        curve = next(curve for curve in self.shoe.validation["curves"] if curve["fixture"] == self.fixture_name)
        self._cycle_time = np.asarray(curve["time_s"], dtype=np.float64)
        self._cycle_depth = np.asarray(curve["displacement_m"], dtype=np.float64)
        self._measured_force = np.asarray(curve["measured_force_n"], dtype=np.float64)
        self._expected_peak_force_n = float(curve["metrics"]["simulated_peak_force_n"])
        self._period = float(self._cycle_time[-1] - self._cycle_time[0])
        self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        self._add_instron_indenter_visual(builder)
        self.foundation_config = FoundationConfig(stretch_floor=0.05)
        return (
            fixture.carrier_anchor_m,
            fixture.foam_free_top_m,
            fixture.rest_length_m,
            fixture.area_m2,
            fixture.neighbors,
            fixture.spacing_m,
        )

    def _build_drop(self, builder, args):
        bed = self.shoe.column_bed
        self.drop_mass_kg = float(getattr(args, "drop_mass", 5.0))
        self.drop_height_m = float(getattr(args, "drop_height", 0.04))
        hx = 0.5 * float(np.ptp(bed.anchor_bottom_m[:, 0])) + 0.01
        hy = 0.5 * float(np.ptp(bed.anchor_bottom_m[:, 1])) + 0.01
        hz = 0.035
        midsole_top = float(np.max(self.shoe.visual_mesh("midsole").vertices_m[:, 2]))
        mass_center_z = midsole_top + 0.005 + hz
        inertia = wp.mat33(
            self.drop_mass_kg * (hy * hy + hz * hz) / 3.0,
            0.0,
            0.0,
            0.0,
            self.drop_mass_kg * (hx * hx + hz * hz) / 3.0,
            0.0,
            0.0,
            0.0,
            self.drop_mass_kg * (hx * hx + hy * hy) / 3.0,
        )
        self.carrier = builder.add_body(
            mass=self.drop_mass_kg,
            com=wp.vec3(0.0, 0.0, mass_center_z),
            inertia=inertia,
            label="guided_drop_mass",
        )
        self._add_drop_mass_visual(builder, center_z=mass_center_z, half_height=hz)
        self.foundation_config = FoundationConfig(stretch_floor=0.05, normal_damping=40.0)
        return (
            bed.anchor_bottom_m,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
        )

    def _build_rocker(self, builder):
        bed = self.shoe.column_bed
        self._rocker_period = 1.2
        self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        self.foundation_config = FoundationConfig(stretch_floor=0.05)
        return (
            bed.anchor_bottom_m,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
        )

    def _instron_pose(self, time_s: float) -> np.ndarray:
        phase = time_s % self._period
        depth = float(np.interp(phase, self._cycle_time, self._cycle_depth))
        return np.array([0.0, 0.0, -depth, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _rocker_pose(self, time_s: float) -> tuple[np.ndarray, float]:
        phase = (time_s % self._rocker_period) / self._rocker_period
        loaded = math.sin(math.pi * phase) ** 2
        pitch = math.radians(-8.0 + 16.0 * phase)
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch)
        anchors = self.shoe.column_bed.anchor_bottom_m
        minimum = float(np.min(-math.sin(pitch) * anchors[:, 0] + math.cos(pitch) * anchors[:, 2]))
        depth = 0.010 * loaded
        pose = np.array([0.0, 0.0, -minimum - depth, quat[0], quat[1], quat[2], quat[3]], dtype=np.float32)
        return pose, math.degrees(pitch)

    def simulate(self) -> None:
        """Advance one display frame while keeping all shoe forces in the portable runtime."""
        for _ in range(self.sim_substeps):
            if self.mode == "instron":
                self.state_0.body_q.assign(self._instron_pose(self.sim_time).reshape(1, 7))
                self.state_0.body_qd.zero_()
                self.state_0.clear_forces()
                self.foundation.apply(self.state_0, self.sim_dt)
                wp.launch(
                    _track_peak_metrics,
                    dim=1,
                    inputs=[
                        self.foundation.normal_force,
                        self.foundation.max_compression,
                        self._peak_force,
                        self._peak_compression,
                    ],
                    device=self.device,
                )
            elif self.mode == "rocker":
                pose, _ = self._rocker_pose(self.sim_time)
                self.state_0.body_q.assign(pose.reshape(1, 7))
                self.state_0.body_qd.zero_()
                self.state_0.clear_forces()
                self.foundation.apply(self.state_0, self.sim_dt)
                wp.launch(
                    _track_peak_metrics,
                    dim=1,
                    inputs=[
                        self.foundation.normal_force,
                        self.foundation.max_compression,
                        self._peak_force,
                        self._peak_compression,
                    ],
                    device=self.device,
                )
            else:
                self.state_0.clear_forces()
                self.foundation.apply(self.state_0, self.sim_dt)
                wp.launch(
                    _accumulate_drop_metrics,
                    dim=1,
                    inputs=[
                        self.carrier,
                        self.state_0.body_qd,
                        self.foundation.normal_force,
                        self.foundation.max_compression,
                        self.sim_dt,
                        self._peak_force,
                        self._peak_compression,
                        self._impulse,
                        self._impact_started,
                        self._impact_complete,
                    ],
                    device=self.device,
                )
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                wp.launch(
                    _project_vertical_guide,
                    dim=1,
                    inputs=[self.carrier, self.state_1.body_q, self.state_1.body_qd],
                    device=self.device,
                )
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self) -> None:
        self.simulate()
        diagnostics = self.foundation.diagnostics()
        entry = {"time_s": self.sim_time, **diagnostics}
        if self.mode == "instron":
            phase = self.sim_time % self._period
            entry["depth_m"] = float(np.interp(phase, self._cycle_time, self._cycle_depth))
            entry["measured_force_n"] = float(np.interp(phase, self._cycle_time, self._measured_force))
        elif self.mode == "rocker":
            _, entry["pitch_deg"] = self._rocker_pose(self.sim_time)
        else:
            entry["height_m"] = float(self.state_0.body_q.numpy()[self.carrier, 2])
            entry["velocity_m_s"] = float(self.state_0.body_qd.numpy()[self.carrier, 2])
        self.history.append(entry)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            column_colors,
            dim=self.column_count,
            inputs=[self.foundation.compression, 0.010, self._colors],
            device=self.device,
        )
        if self.mode == "instron":
            wp.launch(
                column_world_positions,
                dim=self.column_count,
                inputs=[self.carrier, self.state_0.body_q, self._anchor, self._points],
                device=self.device,
            )
            wp.launch(
                deform_instron_mesh,
                dim=len(self._mesh_points),
                inputs=[
                    self._mesh_source,
                    self._mesh_column_index,
                    self._mesh_height_fraction,
                    self.foundation.compression,
                    self._mesh_points,
                ],
                device=self.device,
            )
            self.viewer.log_lines("digital_shoe/columns", self._fixed_bottom, self._points, self._colors, width=0.003)
        else:
            wp.launch(
                attached_column_endpoints,
                dim=self.column_count,
                inputs=[
                    self.carrier,
                    self.state_0.body_q,
                    self._anchor,
                    self._rest,
                    self._points,
                    self._tops,
                ],
                device=self.device,
            )
            wp.launch(
                deform_attached_mesh,
                dim=len(self._mesh_points),
                inputs=[self.carrier, self.state_0.body_q, self._mesh_source, self._mesh_points],
                device=self.device,
            )
            self.viewer.log_lines("digital_shoe/columns", self._points, self._tops, self._colors, width=0.003)
        self.viewer.log_mesh(
            "digital_shoe/midsole",
            self._mesh_points,
            self._mesh_indices,
            backface_culling=False,
            color=(0.78, 0.44, 0.16),
            roughness=0.85,
        )
        self.viewer.log_points("digital_shoe/contact", self._points, radii=0.0025, colors=self._colors)
        last = self.history[-1] if self.history else {"normal_force_n": 0.0, "active_columns": 0}
        self.viewer.log_scalar("/digital_shoe/force_n", last["normal_force_n"])
        self.viewer.log_scalar("/digital_shoe/active_columns", last["active_columns"])
        if self.mode == "instron":
            self.viewer.log_scalar("/digital_shoe/measured_force_n", last.get("measured_force_n", 0.0))
            self.viewer.log_scalar("/digital_shoe/compression_mm", 1000.0 * last.get("depth_m", 0.0))
        elif self.mode == "rocker":
            self.viewer.log_scalar("/digital_shoe/pitch_deg", last.get("pitch_deg", 0.0))
            self.viewer.log_scalar("/digital_shoe/cop_x_m", last.get("cop_x_m", 0.0))
        else:
            self.viewer.log_scalar("/digital_shoe/mass_height_m", last.get("height_m", 0.0))
        self.viewer.end_frame()
        self._capture_gif_frame()

    def _capture_gif_frame(self) -> None:
        """Capture one downsampled OpenGL frame inside the selected loop window."""
        if self._gif_path is None:
            return
        if not hasattr(self.viewer, "get_frame"):
            raise RuntimeError("--record-gif requires --viewer gl")
        if self.mode == "instron":
            start = (INSTRON_WARMUP_CYCLES - 1) * self._period
            stop = INSTRON_WARMUP_CYCLES * self._period
        elif self.mode == "rocker":
            start, stop = 0.0, self._rocker_period
        else:
            start, stop = 0.0, 0.75
        frame_index = max(0, len(self.history) - 1)
        if not start <= self.sim_time <= stop or frame_index % self._gif_stride:
            return
        from PIL import Image

        pixels = self.viewer.get_frame().numpy()
        height, width = pixels.shape[:2]
        target_height = max(1, round(height * self._gif_width / width))
        image = Image.fromarray(pixels)
        self._gif_frames.append(image.resize((self._gif_width, target_height), Image.Resampling.LANCZOS))

    def save_gif(self) -> Path | None:
        """Write captured frames as an infinite, shared-palette GIF loop."""
        if self._gif_path is None:
            return None
        if not self._gif_frames:
            raise RuntimeError("no GIF frames were captured; increase --num-frames or use --viewer gl")
        from PIL import Image

        self._gif_path.parent.mkdir(parents=True, exist_ok=True)
        palette = self._gif_frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        frames = [palette]
        frames.extend(frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in self._gif_frames[1:])
        frames[0].save(
            self._gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000.0 / self._gif_fps),
            loop=0,
            optimize=True,
            disposal=2,
        )
        print(f"[digital shoe / {self.mode}] GIF: {self._gif_path} ({len(frames)} frames)")
        return self._gif_path

    def test_final(self) -> None:
        """Verify that the selected showcase produces finite, shoe-like mechanics."""
        if not self.history:
            raise AssertionError("showcase recorded no frames")
        values = np.asarray([entry["normal_force_n"] for entry in self.history])
        if not np.all(np.isfinite(values)):
            raise AssertionError("showcase produced nonfinite force")
        getattr(self, f"_test_{self.mode}")(values)

    def _test_instron(self, force: np.ndarray) -> None:
        """Require a loaded, dissipative Virtual Instron cycle."""
        if self.sim_time < (INSTRON_WARMUP_CYCLES - 1) * self._period:
            raise AssertionError(f"run at least {INSTRON_WARMUP_CYCLES} Instron cycles to warm the Maxwell state")
        peak = float(self._peak_force.numpy()[0])
        if not 900.0 < peak < 3000.0:
            raise AssertionError(f"Virtual Instron peak is outside the expected range: {peak:.1f} N")
        relative_error = abs(peak - self._expected_peak_force_n) / self._expected_peak_force_n
        if relative_error >= 0.03:
            raise AssertionError(f"runtime peak differs from the exported prediction by {100.0 * relative_error:.1f}%")
        print(
            f"[digital shoe / instron] peak {peak:.0f} N "
            f"(exported prediction {self._expected_peak_force_n:.0f} N); artifact {self.shoe.shoe_id}"
        )

    def _test_drop(self, force: np.ndarray) -> None:
        """Require finite impact, meaningful compression, and no densification-clamp contact."""
        peak = float(self._peak_force.numpy()[0])
        compression = float(self._peak_compression.numpy()[0])
        impulse = float(self._impulse.numpy()[0])
        if peak <= 3.0 * self.drop_mass_kg * 9.81:
            raise AssertionError(f"drop impact too small: {peak:.1f} N")
        if not 0.001 < compression < 0.95 * float(self.shoe.column_bed.rest_length_m.min()):
            raise AssertionError(f"drop compression outside safe range: {compression:.6f} m")
        print(
            f"[digital shoe / drop] {self.drop_mass_kg:.1f} kg from {1000 * self.drop_height_m:.0f} mm: peak {peak:.0f} N, compression {1000 * compression:.1f} mm, first-contact impulse {impulse:.2f} N·s"
        )

    def _test_rocker(self, force: np.ndarray) -> None:
        """Require substantial load and a heel-to-toe center-of-pressure migration."""
        loaded = force > 0.15 * float(force.max())
        cop = np.asarray([entry["cop_x_m"] for entry in self.history])
        if force.max() <= 300.0 or np.count_nonzero(loaded) < 5:
            raise AssertionError("rocker did not establish a loaded contact patch")
        travel = float(cop[loaded][-1] - cop[loaded][0])
        if travel <= 0.05:
            raise AssertionError(f"rocker COP travel too small: {1000 * travel:.1f} mm")
        print(f"[digital shoe / rocker] peak {force.max():.0f} N; COP travelled {1000 * travel:.0f} mm")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--artifact", type=Path, default=Path(DEFAULT_ARTIFACT))
    parser.add_argument("--mode", choices=["instron", "drop", "rocker"], default="instron")
    parser.add_argument("--fixture", choices=["rearfoot_punch", "fullfoot_last"], default="fullfoot_last")
    parser.add_argument("--drop-mass", type=float, default=5.0, help="Guided drop mass [kg].")
    parser.add_argument("--drop-height", type=float, default=0.04, help="Initial outsole clearance [m].")
    parser.add_argument("--record-gif", type=Path, help="Write an infinite GIF loop from the OpenGL viewer.")
    parser.add_argument("--gif-width", type=int, default=480, help="Recorded GIF width in pixels.")
    parser.add_argument("--gif-fps", type=int, default=30, help="Recorded GIF playback rate.")
    parser.add_argument("--gif-stride", type=int, default=2, help="Capture every Nth rendered frame.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.save_gif()
