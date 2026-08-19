# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the baseline human-shoe attachment over Gait2354."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import opensim
from projects.digital_instron_v2.dynamics import column_colors
from projects.human_shoe import load_experiment
from projects.human_shoe.contact_sidecar import load_contact_sidecar
from projects.human_shoe.preparation import prepare_attached_sole

BASE_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = BASE_DIR / "experiments/human_shoe/baseline_gait2354.json"
MOTION_PATH = BASE_DIR / "newton/examples/assets/gait2354_subject01_walk.mot"


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


@wp.kernel
def _deform_attachment_columns(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    bottom_local: wp.array[wp.vec3],
    top_local: wp.array[wp.vec3],
    rest_len: wp.array[wp.float32],
    bottom_world: wp.array[wp.vec3],
    top_world: wp.array[wp.vec3],
    compression: wp.array[wp.float32],
):
    """Transform and deform Digital Instron columns in Newton's Z-up viewer frame."""
    i = wp.tid()
    b = wp.transform_point(body_q[carrier], bottom_local[i])
    t = wp.transform_point(body_q[carrier], top_local[i])
    bottom_z = wp.max(b[2], 0.0)
    top_z = wp.max(t[2], 0.0)
    if top_z < bottom_z:
        bottom_z = top_z
    deformed_bottom = wp.vec3(b[0], b[1], bottom_z)
    deformed_top = wp.vec3(t[0], t[1], top_z)
    bottom_world[i] = deformed_bottom
    top_world[i] = deformed_top
    compression[i] = wp.min(wp.max(-b[2], 0.0), rest_len[i])


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        experiment_path = _resolve_repo_path(getattr(args, "experiment", EXPERIMENT_PATH))
        self.experiment = load_experiment(experiment_path)
        self.human_model_path = _resolve_repo_path(self.experiment.human_model_path)
        self.sidecar_path = (
            _resolve_repo_path(self.experiment.contact_sidecar_path) if self.experiment.contact_sidecar_path else None
        )
        self.manifest_path = _resolve_repo_path(self.experiment.shoe_manifest_path)

        self.osim_model = opensim.parse_osim(self.human_model_path)
        motion_arg = getattr(args, "motion", None)
        motion_source = motion_arg or self.experiment.motion_path or MOTION_PATH
        self.motion_path = _resolve_repo_path(motion_source)
        self.time, coords = opensim.read_motion(self.osim_model, self.motion_path)
        self.viz = opensim.MotionVisualizer(self.osim_model, coords, time=self.time, device=self.device)
        self.num_frames = self.viz.num_frames

        geometry_dir = getattr(args, "geometry", None)
        if geometry_dir is None and getattr(args, "download_geometry", False):
            geometry_dir = opensim.fetch_opensim_geometry()
        self.use_bone_meshes = False
        if geometry_dir:
            # The derived OpenSim writer intentionally does not copy legacy
            # display metadata, so read bone references from the sidecar source.
            geometry_source = self.human_model_path
            if self.sidecar_path is not None:
                sidecar = load_contact_sidecar(self.sidecar_path)
                geometry_source = (self.sidecar_path.parent / sidecar.source_model_path).resolve()
            self.use_bone_meshes = self.viz.load_meshes(geometry_source, geometry_dir) > 0

        dt = float(np.mean(np.diff(self.time))) if len(self.time) > 1 else 1.0 / 60.0
        self.frame_dt = dt
        self.fps = 1.0 / dt
        self.sim_time = float(self.time[0]) if len(self.time) else 0.0
        self.frame = 0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        contact_shape_start = builder.shape_count
        import_result = opensim.add_osim(builder, self.osim_model, parse_contacts=True, parse_muscles=False)
        prepared = prepare_attached_sole(import_result, self.experiment.attachment, self.manifest_path)
        self.prepared_sole = prepared
        self.resolved = prepared.resolved

        imported_contact_geometry = [
            contact
            for contact in import_result.model.contact_geometry
            if contact.type in {"ContactSphere", "ContactHalfSpace"}
        ]
        imported_contact_shapes = range(contact_shape_start, builder.shape_count)
        for shape, contact in zip(imported_contact_shapes, imported_contact_geometry, strict=True):
            builder.shape_color[shape] = (0.25, 0.55, 0.95)
            builder.shape_label[shape] = contact.name

        for body in range(builder.body_count):
            builder.add_shape_sphere(body, radius=0.014, as_site=True, color=(0.9, 0.85, 0.55))

        mesh = newton.Mesh(
            prepared.midsole_vertices.copy(),
            prepared.midsole_indices.copy(),
            compute_inertia=False,
        )
        self.midsole_shape = builder.add_shape_mesh(
            self.resolved.shoe_carrier_body_index,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.25, 0.35, 0.75),
            label="digital_instron_midsole",
        )
        self.attachment_alignment_rms_m = prepared.alignment_rms_m
        self.attachment_alignment_max_m = prepared.alignment_max_m
        self.column_bottom_local = prepared.column_bottom_local
        self.column_top_local = prepared.column_top_local
        self.column_rest_len = prepared.column_rest_len
        self._column_bottom_local = wp.array(self.column_bottom_local, dtype=wp.vec3, device=self.device)
        self._column_top_local = wp.array(self.column_top_local, dtype=wp.vec3, device=self.device)
        self._column_rest_len = wp.array(self.column_rest_len, dtype=wp.float32, device=self.device)
        self._column_bottom_world = wp.zeros(len(self.column_bottom_local), dtype=wp.vec3, device=self.device)
        self._column_top_world = wp.zeros(len(self.column_top_local), dtype=wp.vec3, device=self.device)
        self._column_compression = wp.zeros(len(self.column_top_local), dtype=wp.float32, device=self.device)
        self._column_colors = wp.zeros(len(self.column_top_local), dtype=wp.vec3, device=self.device)

        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.body_q_frames = self.viz.body_transforms(self.model.body_label)
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])
        self.viewer.set_model(self.model)

    def step(self):
        self.frame = (self.frame + 1) % self.num_frames
        wp.copy(self.state.body_q, self.body_q_frames[self.frame])
        self.sim_time += self.frame_dt

    def _update_column_deformation(self) -> None:
        """Update prescribed Z-up ground contact for the visual column overlay."""
        wp.launch(
            _deform_attachment_columns,
            dim=len(self.column_top_local),
            inputs=[
                self.resolved.shoe_carrier_body_index,
                self.state.body_q,
                self._column_bottom_local,
                self._column_top_local,
                self._column_rest_len,
                self._column_bottom_world,
                self._column_top_world,
                self._column_compression,
            ],
            device=self.device,
        )
        wp.launch(
            column_colors,
            dim=len(self.column_top_local),
            inputs=[self._column_compression, 0.012, self._column_colors],
            device=self.device,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.use_bone_meshes:
            self.viz.render_meshes(self.viewer, self.frame)
        else:
            self.viz.render_skeleton(self.viewer, self.frame)
        self.viz.render_muscles(self.viewer, self.frame)
        show_columns = getattr(self.args, "show_columns", False)
        show_column_lines = getattr(self.args, "show_column_lines", False)
        if show_columns or show_column_lines:
            self._update_column_deformation()
        if show_columns:
            self.viewer.log_points(
                "/human_shoe/contact_points",
                self._column_bottom_world,
                radii=0.0025,
                colors=self._column_colors,
            )
        if show_column_lines:
            self.viewer.log_lines(
                "/human_shoe/columns",
                self._column_bottom_world,
                self._column_top_world,
                colors=self._column_colors,
                width=0.002,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify the imported gait playback and shoe attachment remain finite."""
        frames = self.body_q_frames.numpy()
        if not np.all(np.isfinite(frames)):
            raise AssertionError("non-finite body transforms in human-shoe playback")
        if self.resolved.foot_body_index < 0 or self.resolved.shoe_carrier_body_index < 0:
            raise AssertionError("attachment did not resolve valid body indices")
        if len(self.column_top_local) == 0 or len(self.column_bottom_local) == 0:
            raise AssertionError("foundation geometry did not produce any attachment columns")
        if self.model.shape_label[self.midsole_shape] != "digital_instron_midsole":
            raise AssertionError("midsole shape label mismatch")
        if int(self.model.shape_body.numpy()[self.midsole_shape]) != self.resolved.shoe_carrier_body_index:
            raise AssertionError("midsole mesh is not attached to calcn_r")
        if not np.allclose(
            self.column_rest_len, self.column_top_local[:, 1] - self.column_bottom_local[:, 1], atol=1e-6
        ):
            raise AssertionError("column rest lengths are inconsistent")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--experiment",
            type=str,
            default=str(EXPERIMENT_PATH),
            help="Human-shoe experiment JSON path.",
        )
        parser.add_argument(
            "--motion",
            type=str,
            default=None,
            help="OpenSim motion file. Defaults to the experiment manifest value.",
        )
        parser.add_argument(
            "--geometry",
            type=str,
            default=None,
            help="Optional OpenSim bone geometry directory, matching the gait viewer convention.",
        )
        parser.add_argument(
            "--download-geometry",
            action="store_true",
            help="Download standard OpenSim bone meshes and render solid bones.",
        )
        parser.add_argument("--show-columns", action="store_true", help="Render attached foundation column points.")
        parser.add_argument("--show-column-lines", action="store_true", help="Render attached foundation column lines.")
        return parser


def main() -> None:
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
