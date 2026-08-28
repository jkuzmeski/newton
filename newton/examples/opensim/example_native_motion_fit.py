# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Synthetic native marker IK example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from projects.gait_c3d.native_motion_fit import (
    free_root_quaternion_slice,
    marker_attachments_from_model,
    marker_positions_from_joint_q,
    solve_marker_sequence,
)


class Example:
    """Recover a synthetic native gait motion through public Newton IK."""

    def __init__(self, viewer, args):
        if not args.synthetic:
            raise ValueError("native_motion_fit currently requires --synthetic")
        self.viewer = viewer
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
            [marker_positions_from_joint_q(self.model, self.attachments, coordinates) for coordinates in self.target_coordinates]
        )
        rng = np.random.default_rng(args.seed)
        visible_targets = full_targets[:, visible]
        if args.noise_mm:
            visible_targets = visible_targets + rng.normal(
                0.0, args.noise_mm * 1.0e-3, size=visible_targets.shape
            )
        self.target_sequence = visible_targets
        neutral_visible = marker_positions_from_joint_q(self.model, self.visible_attachments, self.seed)
        neutral_errors = neutral_visible - visible_targets[0]
        self.neutral_rms = float(np.sqrt(np.mean(np.sum(neutral_errors * neutral_errors, axis=1))))
        self.frames = solve_marker_sequence(
            self.model,
            self.visible_attachments,
            visible_targets,
            self.seed,
            iterations=args.iterations,
            joint_limit_weight=args.joint_limit_weight,
        )
        self.frame_index = 0
        self.target_points = wp.array(self.frames[0].target_markers.astype(np.float32), dtype=wp.vec3, device=self.model.device)
        self.predicted_points = wp.array(self.frames[0].predicted_markers.astype(np.float32), dtype=wp.vec3, device=self.model.device)
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
        print(
            f"Final marker RMS {max(frame.marker_rms for frame in self.frames) * 1000.0:.3f} mm, "
            f"max {max(frame.marker_max for frame in self.frames) * 1000.0:.3f} mm, "
            f"solver cost {max(frame.solver_cost for frame in self.frames):.3e}"
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
            coordinates_frame[3:7] = np.asarray(
                (0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)), dtype=np.float32
            )
            coordinates_frame[7:] = amplitudes * np.sin(phase + np.arange(len(amplitudes)) * 0.27)
            # Knee hinge coordinates are nonnegative in the native MJCF.
            coordinates_frame[13] = 0.18 + 0.08 * np.sin(phase)
            coordinates_frame[18] = 0.18 + 0.08 * np.cos(phase)
            coordinates.append(coordinates_frame)
        return np.asarray(coordinates, dtype=np.float32)

    def step(self):
        """Advance the displayed synthetic solved frame."""
        frame = self.frames[self.frame_index]
        self.model.joint_q.assign(frame.joint_q)
        self.target_points.assign(frame.target_markers.astype(np.float32))
        self.predicted_points.assign(frame.predicted_markers.astype(np.float32))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.sim_time += self.frame_dt

    def test_final(self):
        """Verify finite warm-started IK and material marker improvement."""
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
            np.linalg.norm(current.joint_q[7:] - previous.joint_q[7:])
            for previous, current in zip(self.frames, self.frames[1:])
        ]
        if jumps and max(jumps) > 0.8:
            raise ValueError("warm-started synthetic IK produced a frame jump")

    def render(self):
        """Render target and predicted marker overlays."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_points("synthetic/target_markers", self.target_points, self.radii, self.target_colors)
        self.viewer.log_points("synthetic/predicted_markers", self.predicted_points, self.radii, self.predicted_colors)
        self.viewer.end_frame()


def create_parser():
    """Create the synthetic native marker-fit command line."""
    parser = newton.examples.create_parser()
    parser.add_argument("--synthetic", action="store_true", help="Run the synthetic native marker IK gate")
    parser.add_argument("--subject", help="Subject bundle containing the native marker sites")
    parser.add_argument("--frames", type=int, default=12, help="Synthetic marker frames")
    parser.add_argument("--iterations", type=int, default=80, help="LM iterations per synthetic frame")
    parser.add_argument("--noise-mm", type=float, default=0.0, help="Deterministic marker noise [mm]")
    parser.add_argument("--occlude-every", type=int, default=0, help="Hide every Nth marker, or zero for none")
    parser.add_argument("--seed", type=int, default=12345, help="Synthetic noise seed")
    parser.add_argument("--joint-limit-weight", type=float, default=0.1, help="Joint-limit objective weight")
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(Example(viewer, args), args)
