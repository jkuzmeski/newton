# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualize exact prescribed OpenSim shoe-load replay with no dynamics feedback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import opensim
from projects.human_shoe.preparation import prepare_attached_sole
from projects.human_shoe.replay import (
    PrescribedReplayConfig,
    _load_replay_inputs,
    _sample_motion_hermite,
    replay_prescribed_shoe_load,
)

BASE_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = BASE_DIR / "experiments/human_shoe/baseline_gait2354.json"


@wp.kernel
def _deform_replay_columns(
    frame: wp.int32,
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    bottom_local: wp.array[wp.vec3],
    top_local: wp.array[wp.vec3],
    compression_history: wp.array2d[wp.float32],
    color_history: wp.array2d[wp.vec3],
    ground_height: wp.float32,
    bottom_world: wp.array[wp.vec3],
    top_world: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    column = wp.tid()
    bottom = wp.transform_point(body_q[carrier], bottom_local[column])
    top = wp.transform_point(body_q[carrier], top_local[column])
    compression = compression_history[frame, column]
    bottom_z = wp.max(bottom[2], ground_height)
    if compression > 0.0:
        bottom_z = ground_height
    top_z = wp.max(top[2], ground_height)
    if top_z < bottom_z:
        bottom_z = top_z
    bottom_world[column] = wp.vec3(bottom[0], bottom[1], bottom_z)
    top_world[column] = wp.vec3(top[0], top[1], top_z)
    colors[column] = color_history[frame, column]


class Example:
    """Render exact prescribed kinematics and replayed Digital Instron loads."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()
        self.grf_scale = float(args.grf_scale)
        if not np.isfinite(self.grf_scale) or self.grf_scale <= 0.0:
            raise ValueError("grf_scale must be finite and positive")
        if not np.isfinite(args.playback_fps) or args.playback_fps <= 0.0:
            raise ValueError("playback_fps must be finite and positive")

        replay_config = PrescribedReplayConfig(
            dt_s=args.replay_dt,
            ground_height_m=args.ground_height,
            stance_index=args.stance_index,
            record_columns=True,
        )
        self.replay = replay_prescribed_shoe_load(args.experiment, replay_config, device=self.device)
        self.ground_height_m = replay_config.ground_height_m
        (
            resolved_experiment_path,
            self.experiment,
            self.osim_model,
            _,
            self.manifest_path,
            _,
            source_time,
            source_coordinates,
        ) = _load_replay_inputs(args.experiment)
        frame_dt = 1.0 / float(args.playback_fps)
        target_time = np.arange(
            self.replay.window.start_time_s,
            self.replay.window.end_time_s,
            frame_dt,
            dtype=np.float64,
        )
        right = np.searchsorted(self.replay.time_s, target_time, side="left")
        right = np.clip(right, 0, len(self.replay.time_s) - 1)
        left = np.maximum(right - 1, 0)
        replay_index = np.where(
            np.abs(self.replay.time_s[left] - target_time) <= np.abs(self.replay.time_s[right] - target_time),
            left,
            right,
        )
        self.replay_index = np.unique(replay_index)
        self.display_time = self.replay.time_s[self.replay_index]
        self.target_time_error_s = np.min(np.abs(self.display_time[:, None] - target_time[None, :]), axis=1)
        coordinates, _ = _sample_motion_hermite(source_time, source_coordinates, self.display_time)
        self.visualizer = opensim.MotionVisualizer(
            self.osim_model,
            coordinates,
            time=self.display_time,
            device=self.device,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        contact_shape_start = builder.shape_count
        import_result = opensim.add_osim(builder, self.osim_model, parse_contacts=True, parse_muscles=False)
        prepared = prepare_attached_sole(import_result, self.experiment.attachment, self.manifest_path)
        self.prepared_sole = prepared
        self.carrier = prepared.resolved.shoe_carrier_body_index
        imported_contacts = [
            contact
            for contact in import_result.model.contact_geometry
            if contact.type in {"ContactSphere", "ContactHalfSpace"}
        ]
        for shape, contact in zip(range(contact_shape_start, builder.shape_count), imported_contacts, strict=True):
            builder.shape_color[shape] = (0.25, 0.55, 0.95)
            builder.shape_label[shape] = contact.name
        mesh = newton.Mesh(
            prepared.midsole_vertices.copy(),
            prepared.midsole_indices.copy(),
            compute_inertia=False,
        )
        self.midsole_shape = builder.add_shape_mesh(
            self.carrier,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.22, 0.34, 0.72),
            label="digital_instron_midsole",
        )
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.body_q_frames = self.visualizer.body_transforms(self.model.body_label)

        compression = self.replay.column_compression_m[self.replay_index].astype(np.float32)
        column_force = self.replay.column_force_n[self.replay_index].astype(np.float32)
        normal_force = np.maximum(column_force[:, :, 2], 0.0)
        positive_force = normal_force[normal_force > 0.0]
        self.force_scale_n = float(np.percentile(positive_force, 99.0)) if len(positive_force) else 1.0
        normalized = np.clip(normal_force / self.force_scale_n, 0.0, 1.0)
        lower = normalized <= 0.5
        colors = np.empty((*normalized.shape, 3), dtype=np.float32)
        colors[:, :, 0] = np.where(lower, 0.0, 2.0 * normalized - 1.0)
        colors[:, :, 1] = np.where(lower, 2.0 * normalized, 2.0 * (1.0 - normalized))
        colors[:, :, 2] = np.where(lower, 1.0 - 2.0 * normalized, 0.0)
        self._compression_history = wp.array(compression, dtype=wp.float32, device=self.device)
        self._color_history = wp.array(colors, dtype=wp.vec3, device=self.device)
        self._bottom_local = wp.array(prepared.column_bottom_local, dtype=wp.vec3, device=self.device)
        self._top_local = wp.array(prepared.column_top_local, dtype=wp.vec3, device=self.device)
        column_count = len(prepared.column_bottom_local)
        self._bottom_world = wp.zeros(column_count, dtype=wp.vec3, device=self.device)
        self._top_world = wp.zeros(column_count, dtype=wp.vec3, device=self.device)
        self._column_colors = wp.zeros(column_count, dtype=wp.vec3, device=self.device)

        display_force = self.replay.grf_n[self.replay_index].astype(np.float32)
        display_cop = self.replay.cop_m[self.replay_index].astype(np.float32)
        display_valid = self.replay.cop_valid[self.replay_index]
        display_cop[~display_valid] = 0.0
        arrow_end = display_cop + self.grf_scale * display_force
        arrow_end[~display_valid] = display_cop[~display_valid]
        self._grf_start = wp.array(display_cop, dtype=wp.vec3, device=self.device)
        self._grf_end = wp.array(arrow_end, dtype=wp.vec3, device=self.device)
        self._cop_point_frame = np.flatnonzero(display_valid)
        self._cop_points = wp.array(display_cop[display_valid], dtype=wp.vec3, device=self.device)
        self._cop_radii = wp.array(
            np.where(display_valid, 0.008, 0.0).astype(np.float32), dtype=wp.float32, device=self.device
        )
        self._grf_colors = wp.array(
            np.tile(np.array([[0.1, 1.0, 0.2]], dtype=np.float32), (len(display_cop), 1)),
            dtype=wp.vec3,
            device=self.device,
        )
        self._cop_current_colors = wp.array(
            np.tile(np.array([[1.0, 0.85, 0.05]], dtype=np.float32), (len(display_cop), 1)),
            dtype=wp.vec3,
            device=self.device,
        )
        self._cop_path_colors = wp.array(
            np.tile(np.array([[1.0, 0.55, 0.05]], dtype=np.float32), (int(np.count_nonzero(display_valid)), 1)),
            dtype=wp.vec3,
            device=self.device,
        )
        valid_pairs = display_valid[:-1] & display_valid[1:]
        self._cop_segment_end_frame = np.flatnonzero(valid_pairs) + 1
        self._cop_line_start = wp.array(display_cop[:-1][valid_pairs], dtype=wp.vec3, device=self.device)
        self._cop_line_end = wp.array(display_cop[1:][valid_pairs], dtype=wp.vec3, device=self.device)
        self._cop_line_colors = wp.array(
            np.tile(np.array([[1.0, 0.55, 0.05]], dtype=np.float32), (int(np.count_nonzero(valid_pairs)), 1)),
            dtype=wp.vec3,
            device=self.device,
        )
        self._display_force = display_force
        self._display_cop = display_cop
        self._display_valid = display_valid

        self.frame = 0
        self.cycle = 0
        self.num_frames = len(self.display_time)
        self.sim_time = 0.0
        self.frame_dt = frame_dt
        wp.copy(self.state.body_q, self.body_q_frames[0])
        self.viewer.set_model(self.model)
        print(
            "[human_shoe.replay_viewer] PRESCRIBED EXACT KINEMATICS / NO FORCE FEEDBACK | "
            f"stance={args.stance_index}, peak_Fz={self.replay.peak_vertical_force_n:.1f} N, "
            f"impulse_z={self.replay.final_vertical_impulse_ns:.2f} N*s, "
            f"column_force_scale={self.force_scale_n:.1f} N"
        )
        if self.experiment.controller_id == "c3d_overground_exact_replay_v1":
            integration_path = resolved_experiment_path.parent / "human_shoe_integration_qc.json"
            if integration_path.is_file():
                integration_qc = json.loads(integration_path.read_text(encoding="utf-8"))
                print(f"[human_shoe.replay_viewer] INTEGRATION QC: {integration_qc.get('status', 'unknown').upper()}")
                for warning in integration_qc.get("warnings", []):
                    print(f"  - {warning}")
            else:
                print(
                    "[human_shoe.replay_viewer] WARNING: baseline shoe anchors are reused on scaled S001 "
                    "without independent subject/shoe registration."
                )

    def step(self) -> None:
        next_frame = self.frame + 1
        if next_frame >= self.num_frames:
            next_frame = 0
            self.cycle += 1
        self.frame = next_frame
        cycle_duration = self.display_time[-1] - self.display_time[0] + self.frame_dt
        self.sim_time = self.cycle * cycle_duration + (self.display_time[self.frame] - self.display_time[0])
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])

    def _update_columns(self) -> None:
        wp.launch(
            _deform_replay_columns,
            dim=len(self.prepared_sole.column_bottom_local),
            inputs=[
                self.frame,
                self.carrier,
                self.state.body_q,
                self._bottom_local,
                self._top_local,
                self._compression_history,
                self._color_history,
                self.ground_height_m,
                self._bottom_world,
                self._top_world,
                self._column_colors,
            ],
            device=self.device,
        )

    def render(self) -> None:
        self._update_columns()
        self.viewer.begin_frame(self.sim_time)
        cop_point_count = int(np.searchsorted(self._cop_point_frame, self.frame, side="right"))
        cop_segment_count = int(np.searchsorted(self._cop_segment_end_frame, self.frame, side="right"))
        self.viewer.log_state(self.state)
        self.visualizer.render_skeleton(self.viewer, self.frame)
        if not self.args.hide_muscles:
            self.visualizer.render_muscles(self.viewer, self.frame)
        self.viewer.log_points(
            "/human_shoe/columns/points",
            self._bottom_world,
            radii=0.0025,
            colors=self._column_colors,
        )
        self.viewer.log_lines(
            "/human_shoe/columns/lines",
            self._bottom_world,
            self._top_world,
            colors=self._column_colors,
            width=0.002,
        )
        self.viewer.log_arrows(
            "/human_shoe/grf",
            self._grf_start[self.frame : self.frame + 1],
            self._grf_end[self.frame : self.frame + 1],
            colors=self._grf_colors[self.frame : self.frame + 1],
            width=0.008,
            hidden=not bool(self._display_valid[self.frame]),
        )
        self.viewer.log_points(
            "/human_shoe/cop/current",
            self._grf_start[self.frame : self.frame + 1],
            radii=self._cop_radii[self.frame : self.frame + 1],
            colors=self._cop_current_colors[self.frame : self.frame + 1],
            hidden=not bool(self._display_valid[self.frame]),
        )
        self.viewer.log_points(
            "/human_shoe/cop/path_points",
            self._cop_points[:cop_point_count],
            radii=0.002,
            colors=self._cop_path_colors[:cop_point_count],
        )
        self.viewer.log_lines(
            "/human_shoe/cop/path_lines",
            self._cop_line_start[:cop_segment_count],
            self._cop_line_end[:cop_segment_count],
            colors=self._cop_line_colors[:cop_segment_count],
            width=0.002,
        )
        replay_sample = self.replay_index[self.frame]
        self.viewer.log_scalar("/human_shoe/source_time_s", self.display_time[self.frame])
        self.viewer.log_scalar("/human_shoe/grf_z_n", self._display_force[self.frame, 2])
        self.viewer.log_scalar("/human_shoe/active_columns", self.replay.active_columns[replay_sample])
        self.viewer.log_scalar("/human_shoe/max_compression_mm", 1000.0 * self.replay.max_compression_m[replay_sample])
        self.viewer.log_scalar("/human_shoe/contact_power_w", self.replay.contact_power_w[replay_sample])
        self.viewer.log_scalar("/human_shoe/contact_work_j", self.replay.contact_work_j[replay_sample])
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify visualization buffers stay synchronized with exact replay data."""
        frames = self.body_q_frames.numpy()
        if not np.all(np.isfinite(frames)):
            raise AssertionError("non-finite exact OpenSim body transforms")
        if not np.all(np.diff(self.display_time) > 0.0):
            raise AssertionError("display time is not strictly increasing")
        if not np.all(np.diff(self.replay_index) >= 0):
            raise AssertionError("display-to-replay indices are not monotone")
        if not np.all(np.isfinite(self._display_force)):
            raise AssertionError("non-finite displayed GRF")
        valid = self._display_valid
        arrow = self._grf_end.numpy() - self._grf_start.numpy()
        np.testing.assert_allclose(
            arrow[valid],
            self.grf_scale * self._display_force[valid],
            rtol=1.0e-5,
            atol=1.0e-7,
        )
        if self.replay.peak_vertical_force_n <= 0.0:
            raise AssertionError("exact replay did not produce a vertical shoe load")
        if self.force_scale_n <= 0.0:
            raise AssertionError("invalid column-force color scale")
        np.testing.assert_array_equal(self.display_time, self.replay.time_s[self.replay_index])
        if np.max(self.target_time_error_s, initial=0.0) > 0.5 * float(np.max(self.replay.dt_s)) + 1.0e-12:
            raise AssertionError("display frames are not nearest synchronized replay samples")
        self._update_columns()
        if float(np.min(self._bottom_world.numpy()[:, 2])) < self.ground_height_m - 1.0e-7:
            raise AssertionError("displayed columns penetrate below the ground plane")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--experiment", default=str(EXPERIMENT_PATH), help="Human-shoe experiment JSON.")
        parser.add_argument("--stance-index", type=int, default=0, help="Complete right stance to replay.")
        parser.add_argument(
            "--replay-dt",
            type=float,
            default=None,
            help="Foundation replay timestep [s]; defaults to the experiment manifest.",
        )
        parser.add_argument("--ground-height", type=float, default=0.0, help="Newton world ground height [m].")
        parser.add_argument("--playback-fps", type=float, default=120.0, help="Viewer playback rate [Hz].")
        parser.add_argument("--grf-scale", type=float, default=2.0e-4, help="GRF arrow length scale [m/N].")
        parser.add_argument("--hide-muscles", action="store_true", help="Hide exact OpenSim muscle paths.")
        return parser


def main() -> None:
    """Run the exact prescribed replay visualizer."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
