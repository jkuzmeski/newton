# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native marker IK and saved-motion replay example."""

from __future__ import annotations

import json
import re
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.gait_c3d.c3d_adapter import read_c3d_markers
from projects.gait_c3d.native_motion_fit import (
    fit_c3d_marker_motion,
    free_root_quaternion_slice,
    load_native_motion_artifact,
    map_c3d_markers_to_native,
    marker_attachments_from_model,
    marker_positions_from_joint_q,
    solve_marker_sequence,
    write_native_motion_artifact,
)


def _default_motion_output(subject: Path, c3d_path: str | Path) -> Path:
    """Return the subject-local output path for one dynamic trial."""
    trial_name = Path(c3d_path).stem
    if trial_name.lower().endswith(".v3d"):
        trial_name = trial_name[:-4]
    trial_slug = re.sub(r"[^a-zA-Z0-9]+", "_", trial_name).strip("_").lower()
    if not trial_slug:
        raise ValueError("C3D filename must contain an alphanumeric trial name")
    return subject / "motions" / f"{trial_slug}_native_motion"


class Example:
    """Solve native gait markers or replay a saved native motion."""

    def __init__(self, viewer, args):
        if args.motion and (args.c3d or args.synthetic):
            raise ValueError("--motion cannot be combined with --c3d or --synthetic")
        if args.motion:
            self._init_motion(viewer, args)
            return
        if args.c3d:
            self._init_real(viewer, args)
            return
        if not args.synthetic:
            raise ValueError("native_motion_fit requires --synthetic, --c3d, or --motion")
        self.viewer = viewer
        self.real_motion = False
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        subject = (
            Path(args.subject).expanduser().resolve()
            if args.subject
            else Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "assets" / "s001_calibrated"
        )
        subject_xml = subject / "model" / "subject.xml"
        if not subject_xml.is_file():
            raise FileNotFoundError(f"subject MJCF is missing: {subject_xml}")
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(subject_xml), floating=True, parse_sites=True, enable_self_collisions=True)
        self.model = builder.finalize(device=args.device)
        if self.model.joint_coord_count != 20:
            raise ValueError(
                "native_motion_fit requires the calibrated free-root subject with 20 coordinates; "
                "use projects/gait_c3d/assets/s001_calibrated"
            )
        self.state = self.model.state()
        self.attachments = marker_attachments_from_model(self.model)
        if args.occlude_every < 0:
            raise ValueError("--occlude-every must be nonnegative")
        if args.noise_mm < 0.0:
            raise ValueError("--noise-mm must be nonnegative")
        self.noise_mm = args.noise_mm
        visible = np.arange(len(self.attachments), dtype=np.int32)
        if args.occlude_every:
            visible = visible[(visible + 1) % args.occlude_every != 0]
        if len(visible) < 6:
            raise ValueError("occlusion leaves too few markers")
        self.visible_indices = visible
        self.visible_attachments = tuple(self.attachments[index] for index in visible)
        self.seed = self.model.joint_q.numpy().copy()
        self.target_coordinates = self._make_target_coordinates(args.frames)
        full_targets = np.asarray(
            [
                marker_positions_from_joint_q(self.model, self.attachments, coordinates)
                for coordinates in self.target_coordinates
            ]
        )
        rng = np.random.default_rng(args.seed)
        visible_targets = full_targets[:, visible]
        if args.noise_mm:
            visible_targets = visible_targets + rng.normal(0.0, args.noise_mm * 1.0e-3, size=visible_targets.shape)
        self.target_sequence = visible_targets
        neutral_visible = marker_positions_from_joint_q(self.model, self.visible_attachments, self.seed)
        neutral_errors = neutral_visible - visible_targets[0]
        self.neutral_rms = float(np.sqrt(np.mean(np.sum(neutral_errors * neutral_errors, axis=1))))
        iterations = 80 if args.iterations is None else args.iterations
        batch_size = 1 if args.batch_size is None else args.batch_size
        solve_start = time.perf_counter()
        self.frames = solve_marker_sequence(
            self.model,
            self.visible_attachments,
            visible_targets,
            self.seed,
            iterations=iterations,
            joint_limit_weight=args.joint_limit_weight,
            batch_size=batch_size,
        )
        self.solve_seconds = time.perf_counter() - solve_start
        self.frame_index = 0
        self.target_points = wp.array(
            self.frames[0].target_markers.astype(np.float32), dtype=wp.vec3, device=self.model.device
        )
        self.predicted_points = wp.array(
            self.frames[0].predicted_markers.astype(np.float32), dtype=wp.vec3, device=self.model.device
        )
        self.radii = wp.full(len(self.visible_attachments), 0.012, dtype=wp.float32, device=self.model.device)
        self.target_colors = wp.full(
            len(self.visible_attachments), wp.vec3(0.95, 0.15, 0.10), dtype=wp.vec3, device=self.model.device
        )
        self.predicted_colors = wp.full(
            len(self.visible_attachments), wp.vec3(0.10, 0.75, 0.98), dtype=wp.vec3, device=self.model.device
        )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -3.2, 1.7), pitch=-5.0, yaw=135.0)
        print(
            f"Synthetic native IK: {len(self.visible_attachments)}/{len(self.attachments)} markers, "
            f"{len(self.frames)} frames, neutral RMS {self.neutral_rms * 1000.0:.2f} mm"
        )
        print(f"Solved in {self.solve_seconds:.3f} s ({len(self.frames) / self.solve_seconds:.1f} frames/s)")
        print(
            f"Final marker RMS {max(frame.marker_rms for frame in self.frames) * 1000.0:.3f} mm, "
            f"max {max(frame.marker_max for frame in self.frames) * 1000.0:.3f} mm, "
            f"solver cost {max(frame.solver_cost for frame in self.frames):.3e}"
        )

    def _init_motion(self, viewer, args):
        """Load a saved native motion artifact for exact kinematic replay."""
        motion_path = Path(args.motion).expanduser().resolve()
        motion = load_native_motion_artifact(motion_path)
        subject = (
            Path(args.subject).expanduser().resolve()
            if args.subject
            else Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "assets" / "s001_calibrated"
        )
        subject_xml = subject / "model" / "subject.xml"
        if not subject_xml.is_file():
            raise FileNotFoundError(f"subject MJCF is missing: {subject_xml}")
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(subject_xml), floating=True, parse_sites=True, enable_self_collisions=True)
        self.model = builder.finalize(device=args.device)
        if self.model.joint_coord_count != motion.joint_q.shape[1]:
            raise ValueError("motion coordinates do not match the selected subject model")
        if self.model.joint_dof_count != motion.joint_qd.shape[1]:
            raise ValueError("motion velocities do not match the selected subject model")
        self.viewer = viewer
        self.real_motion = True
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 100.0
        self.state = self.model.state()
        self.attachments = marker_attachments_from_model(self.model)
        if tuple(attachment.name for attachment in self.attachments) != motion.marker_names:
            raise ValueError("motion markers do not match the selected subject model")
        self.motion = motion
        self.motion_output = motion_path if motion_path.is_dir() else motion_path.parent
        self.visible_indices = np.flatnonzero(np.all(self.motion.valid, axis=0))
        if len(self.visible_indices) == 0:
            raise ValueError("motion artifact contains no marker that is valid in every frame")
        self.frame_index = 0
        self.target_points = wp.array(
            self.motion.targets[0, self.visible_indices].astype(np.float32),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.predicted_points = wp.array(
            self.motion.predictions[0, self.visible_indices].astype(np.float32),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.radii = wp.full(len(self.visible_indices), 0.012, dtype=wp.float32, device=self.model.device)
        self.target_colors = wp.full(
            len(self.visible_indices), wp.vec3(0.95, 0.15, 0.10), dtype=wp.vec3, device=self.model.device
        )
        self.predicted_colors = wp.full(
            len(self.visible_indices), wp.vec3(0.10, 0.75, 0.98), dtype=wp.vec3, device=self.model.device
        )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -3.2, 1.7), pitch=-5.0, yaw=135.0)
        print(
            f"Loaded native motion: {len(self.visible_indices)}/{len(self.attachments)} always-valid markers, "
            f"{len(self.motion.times)} frames from {motion_path}"
        )
        print(
            f"Frame RMS median/p95/max: {np.median(self.motion.frame_rms) * 1000.0:.2f}/"
            f"{np.percentile(self.motion.frame_rms, 95) * 1000.0:.2f}/"
            f"{np.max(self.motion.frame_rms) * 1000.0:.2f} mm"
        )

    def _init_real(self, viewer, args):
        """Load, fit, and publish a real C3D trial."""
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 100.0
        subject = (
            Path(args.subject).expanduser().resolve()
            if args.subject
            else Path(__file__).resolve().parents[3] / "projects" / "gait_c3d" / "assets" / "s001_calibrated"
        )
        subject_xml = subject / "model" / "subject.xml"
        if not subject_xml.is_file():
            raise FileNotFoundError(f"subject MJCF is missing: {subject_xml}")
        motion_output = (
            Path(args.motion_output).expanduser().resolve()
            if args.motion_output
            else _default_motion_output(subject, args.c3d)
        )
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(subject_xml), floating=True, parse_sites=True, enable_self_collisions=True)
        self.model = builder.finalize(device=args.device)
        self.state = self.model.state()
        self.attachments = marker_attachments_from_model(self.model)
        source = read_c3d_markers(args.c3d, up_axis=args.c3d_up_axis, forward_axis=args.c3d_forward_axis)
        mapped = map_c3d_markers_to_native(source, self.attachments)
        model_manifest_path = subject / "model" / "manifest.json"
        if args.registration:
            registration = np.asarray(json.loads(Path(args.registration).read_text(encoding="utf-8")), dtype=np.float64)
            registration_mode = "explicit_matrix"
        else:
            registration = np.eye(4, dtype=np.float64)
            if model_manifest_path.is_file():
                model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
                registration[:3, 3] = np.asarray(
                    model_manifest.get("ground", {}).get("global_offset_m", (0.0, 0.0, 0.0)), dtype=np.float64
                )
            registration_mode = "saved_subject_ground_offset"
        iterations = 40 if args.iterations is None else args.iterations
        batch_size = 0 if args.batch_size is None else args.batch_size
        solve_start = time.perf_counter()
        self.motion = fit_c3d_marker_motion(
            self.model,
            self.attachments,
            mapped,
            self.model.joint_q.numpy(),
            registration=registration,
            iterations=iterations,
            joint_limit_weight=args.joint_limit_weight,
            batch_size=batch_size,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            max_frames=args.max_frames if args.max_frames > 0 else None,
        )
        self.solve_seconds = time.perf_counter() - solve_start
        calibration_path = subject / "model" / "segment_calibration.json"
        self.motion_output = write_native_motion_artifact(
            self.motion,
            motion_output,
            model_path=subject_xml,
            calibration_path=calibration_path if calibration_path.is_file() else None,
            overwrite=args.overwrite,
            settings={
                "iterations": iterations,
                "batch_size": batch_size,
                "start_frame": args.start_frame,
                "end_frame": args.end_frame,
                "stride": args.stride,
                "max_frames": args.max_frames,
                "registration_mode": registration_mode,
            },
        )
        self.real_motion = True
        self.visible_indices = np.flatnonzero(np.all(self.motion.valid, axis=0))
        self.frame_index = 0
        self.target_points = wp.array(
            self.motion.targets[0, self.visible_indices].astype(np.float32), dtype=wp.vec3, device=self.model.device
        )
        self.predicted_points = wp.array(
            self.motion.predictions[0, self.visible_indices].astype(np.float32), dtype=wp.vec3, device=self.model.device
        )
        self.radii = wp.full(len(self.visible_indices), 0.012, dtype=wp.float32, device=self.model.device)
        self.target_colors = wp.full(
            len(self.visible_indices), wp.vec3(0.95, 0.15, 0.10), dtype=wp.vec3, device=self.model.device
        )
        self.predicted_colors = wp.full(
            len(self.visible_indices), wp.vec3(0.10, 0.75, 0.98), dtype=wp.vec3, device=self.model.device
        )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -3.2, 1.7), pitch=-5.0, yaw=135.0)
        print(
            f"Real native IK: {len(self.visible_indices)}/{len(self.attachments)} always-valid markers, "
            f"{len(self.motion.times)} frames -> {self.motion_output}"
        )
        print(f"Solved in {self.solve_seconds:.3f} s ({len(self.motion.times) / self.solve_seconds:.1f} frames/s)")
        print(
            f"Frame RMS median/p95/max: {np.median(self.motion.frame_rms) * 1000.0:.2f}/"
            f"{np.percentile(self.motion.frame_rms, 95) * 1000.0:.2f}/"
            f"{np.max(self.motion.frame_rms) * 1000.0:.2f} mm"
        )

    def _make_target_coordinates(self, frame_count: int) -> np.ndarray:
        """Generate bounded free-root, torso, hip, knee, and ankle poses."""
        if frame_count <= 0:
            raise ValueError("--frames must be positive")
        base = self.seed.copy()
        coordinates = []
        amplitudes = np.asarray(
            (0.07, -0.05, 0.04, 0.08, -0.06, 0.05, 0.11, 0.09, 0.12, -0.08, 0.06, -0.10, 0.07),
            dtype=np.float32,
        )
        if len(amplitudes) != self.model.joint_coord_count - 7:
            raise ValueError("synthetic target amplitudes do not match the subject coordinate layout")
        for frame in range(frame_count):
            phase = 2.0 * np.pi * frame / max(frame_count, 1)
            coordinates_frame = base.copy()
            coordinates_frame[:3] = base[:3] + np.asarray(
                (0.025 * np.sin(phase), 0.018 * np.cos(phase), 0.012 * np.sin(phase + 0.4)), dtype=np.float32
            )
            angle = 0.08 * np.sin(phase)
            coordinates_frame[3:7] = np.asarray((0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)), dtype=np.float32)
            coordinates_frame[7:] = amplitudes * np.sin(phase + np.arange(len(amplitudes)) * 0.27)
            # Knee hinge coordinates are nonnegative in the native MJCF.
            coordinates_frame[13] = 0.18 + 0.08 * np.sin(phase)
            coordinates_frame[18] = 0.18 + 0.08 * np.cos(phase)
            coordinates.append(coordinates_frame)
        return np.asarray(coordinates, dtype=np.float32)

    def step(self):
        """Advance the displayed solved frame."""
        if self.real_motion:
            self.model.joint_q.assign(self.motion.joint_q[self.frame_index])
            self.model.joint_qd.assign(self.motion.joint_qd[self.frame_index])
            self.target_points.assign(self.motion.targets[self.frame_index, self.visible_indices].astype(np.float32))
            self.predicted_points.assign(
                self.motion.predictions[self.frame_index, self.visible_indices].astype(np.float32)
            )
            self.sim_time = float(self.motion.times[self.frame_index])
            self.frame_index = (self.frame_index + 1) % len(self.motion.times)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
            return
        frame = self.frames[self.frame_index]
        self.model.joint_q.assign(frame.joint_q)
        self.target_points.assign(frame.target_markers.astype(np.float32))
        self.predicted_points.assign(frame.predicted_markers.astype(np.float32))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.sim_time += self.frame_dt

    def test_final(self):
        """Verify finite warm-started IK and material marker improvement."""
        if self.real_motion:
            if (
                not np.all(np.isfinite(self.motion.joint_q))
                or not np.all(np.isfinite(self.motion.joint_qd))
                or not np.all(np.isfinite(self.motion.frame_rms))
                or not np.all(np.isfinite(self.motion.predictions))
                or not self.motion_output.is_dir()
            ):
                raise ValueError("real native IK produced a nonfinite result or missing artifact")
            if np.max(self.motion.joint_limit_violation) > 1.0e-4:
                print(
                    f"Warning: maximum real-trial joint-limit violation {np.max(self.motion.joint_limit_violation):.4f}"
                )
            return
        if not self.frames:
            raise ValueError("synthetic IK produced no frames")
        rms = np.asarray([frame.marker_rms for frame in self.frames])
        if not np.all(np.isfinite(rms)) or float(np.max(rms)) >= self.neutral_rms * 0.5:
            raise ValueError("synthetic IK did not materially improve the neutral seed")
        if self.noise_mm == 0.0 and max(frame.marker_max for frame in self.frames) >= 1.0e-4:
            raise ValueError("clean synthetic IK exceeded the 0.1 mm marker tolerance")
        for frame in self.frames:
            if (
                not np.all(np.isfinite(frame.joint_q))
                or not np.all(np.isfinite(frame.target_markers))
                or not np.all(np.isfinite(frame.predicted_markers))
                or not np.isfinite(frame.solver_cost)
                or not np.isfinite(frame.joint_limit_violation)
            ):
                raise ValueError("synthetic IK produced nonfinite coordinates, targets, or diagnostics")
            quaternion_slice = free_root_quaternion_slice(self.model)
            if quaternion_slice is None:
                raise ValueError("synthetic IK test model has no free root")
            quaternion_norm = np.linalg.norm(frame.joint_q[quaternion_slice])
            if not np.isclose(quaternion_norm, 1.0, atol=1.0e-5):
                raise ValueError("synthetic IK free-root quaternion is not normalized")
            if frame.joint_limit_violation > 1.0e-5:
                raise ValueError("synthetic IK violated a native joint limit")
        jumps = [
            np.linalg.norm(current.joint_q[7:] - previous.joint_q[7:]) for previous, current in pairwise(self.frames)
        ]
        if jumps and max(jumps) > 0.8:
            raise ValueError("warm-started synthetic IK produced a frame jump")

    def render(self):
        """Render target and predicted marker overlays."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        prefix = "motion" if self.real_motion else "synthetic"
        self.viewer.log_points(f"{prefix}/target_markers", self.target_points, self.radii, self.target_colors)
        self.viewer.log_points(f"{prefix}/predicted_markers", self.predicted_points, self.radii, self.predicted_colors)
        self.viewer.end_frame()


def create_parser():
    """Create the synthetic native marker-fit command line."""
    parser = newton.examples.create_parser()
    parser.add_argument("--synthetic", action="store_true", help="Run the synthetic native marker IK gate")
    parser.add_argument("--c3d", help="Fit a real dynamic C3D trial")
    parser.add_argument("--motion", help="Load a saved native motion artifact directory instead of solving")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing verified fitted-motion artifact")
    parser.add_argument("--subject", help="Subject bundle containing the native marker sites")
    parser.add_argument(
        "--motion-output",
        help="Override the default subject-local fitted-motion output directory",
    )
    parser.add_argument("--registration", help="Optional JSON 4x4 C3D-to-model registration matrix")
    parser.add_argument("--c3d-up-axis", default="+Z", help="Lab axis that points upward")
    parser.add_argument("--c3d-forward-axis", default="-Y", help="Lab axis that points subject-forward")
    parser.add_argument("--start-frame", type=int, default=0, help="First C3D frame index")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive C3D frame index")
    parser.add_argument("--stride", type=int, default=1, help="C3D frame stride")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum fitted frames; zero means all")
    parser.add_argument("--frames", type=int, default=12, help="Synthetic marker frames")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="LM iterations; default is 80 for synthetic IK and 40 for C3D",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Frames solved concurrently; default is sequential for synthetic IK and all-at-once for C3D",
    )
    parser.add_argument("--noise-mm", type=float, default=0.0, help="Deterministic marker noise [mm]")
    parser.add_argument("--occlude-every", type=int, default=0, help="Hide every Nth marker, or zero for none")
    parser.add_argument("--seed", type=int, default=12345, help="Synthetic noise seed")
    parser.add_argument("--joint-limit-weight", type=float, default=0.1, help="Joint-limit objective weight")
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(Example(viewer, args), args)
