# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay and plots of SuperMimic runner trajectories.

A trajectory is a dict of per-frame arrays in the treadmill frame: ``joint_q``
[T, 17] in the coordinate order of :data:`.reference.COORDINATE_NAMES` and
``dt`` [s], with optional ``grf`` and ``cop`` [T, 2 (right, left), 3] [N, m] and ``action``
[T, 10, 3] (equilibrium angle, stiffness, damping). :func:`reference_trajectory`
reads one F01 reference window into this form.

:class:`Replay` draws the body in a Newton viewer: each segment as the box with
its mass and principal inertia, the bones between joint centers and, with a
shoe, its columns under the feet. Replayed columns are rigid in the foot frame
and shortened where they would enter the ground, colored by that shortening.
Ground reaction forces are drawn as arrows from the center of pressure, or from
below the ankle when none is given.

Recording to ``.mp4`` needs ``imageio-ffmpeg``, for example
``uv run --extra examples --with imageio-ffmpeg -m projects.supermimic.viz ...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp
from matplotlib.figure import Figure
from PIL import Image

import newton
from projects.digital_shoe.rendering import camera_look_at, column_colors

from .body import BODY_NAMES, JOINT_NAMES, BodyParams, SegmentParams, add_body
from .reference import load_body_params
from .sim import Shoe, load_shoes

__all__ = ["Replay", "inertia_box", "main", "plot_trajectory", "record", "reference_trajectory"]

SIDES = ("right", "left")
SIDE_COLORS = ((0.85, 0.3, 0.25), (0.25, 0.45, 0.85))
"""Right and left segment colors."""
GRF_SCALE_M_PER_N = 5.0e-4
"""Arrow length per newton of ground reaction force [m/N]."""
COLOR_COMPRESSION_M = 0.02
"""Column shortening drawn in full red [m]."""
CAMERA_EYE = (1.2, -3.2, 0.6)
"""Camera position relative to the pelvis, on the runner's right [m]."""
CAMERA_TARGET_HEIGHT_M = 0.6


def inertia_box(segment: SegmentParams) -> np.ndarray:
    """Return the half extents [m] of the uniform box with the segment's mass and principal inertia."""
    i = np.asarray(segment.inertia)
    return 0.5 * np.sqrt(6.0 / segment.mass * (i.sum() - 2.0 * i))


def reference_trajectory(reference_dir: str | Path, window: int) -> tuple[dict, BodyParams]:
    """Return one window of a prepared F01 reference and its body parameters."""
    reference_dir = Path(reference_dir)
    data = np.load(reference_dir / "reference.npz")
    trajectory = {
        "joint_q": data["joint_q"][window],
        "grf": data["grf"][window],
        "cop": data["cop"][window],
        "dt": float(data["frame_dt"]),
    }
    return trajectory, load_body_params(reference_dir / "body_params.json")


class Replay:
    """Draws runner poses in a Newton viewer.

    Args:
        viewer: Newton viewer.
        body: Runner geometry and inertia.
        shoe: Shoe pair whose columns are drawn under the feet.
    """

    def __init__(self, viewer, body: BodyParams, shoe: Shoe | None = None):
        self.viewer = viewer
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        add_body(builder, body)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        z = wp.vec3(0.0, 0.0, 1.0)
        for index, name in enumerate(BODY_NAMES):
            segment, _, side = name.partition("_")
            params = getattr(body, segment)
            mirror = np.array((1.0, -1.0 if side == "l" else 1.0, 1.0))
            color = SIDE_COLORS[side == "l"] if side else (0.7, 0.7, 0.7)
            builder.add_shape_box(
                index,
                xform=wp.transform(wp.vec3(*(mirror * params.com)), wp.quat_identity()),
                hx=float(inertia_box(params)[0]),
                hy=float(inertia_box(params)[1]),
                hz=float(inertia_box(params)[2]),
                cfg=cfg,
                color=color,
                opacity=0.6,
            )
            ends = {
                "pelvis": (mirror * body.hip_offset, mirror * body.hip_offset * (1.0, -1.0, 1.0)),
                "thigh": (np.zeros(3), mirror * body.knee_offset),
                "shank": (np.zeros(3), mirror * body.ankle_offset),
            }.get(segment)
            if ends is not None:
                a, b = ends
                builder.add_shape_capsule(
                    index,
                    xform=wp.transform(wp.vec3(*(0.5 * (a + b))), wp.quat_between_vectors(z, wp.vec3(*(b - a)))),
                    radius=0.02,
                    half_height=0.5 * float(np.linalg.norm(b - a)),
                    cfg=cfg,
                    color=color,
                )
        builder.add_ground_plane(cfg=cfg)
        self.model = builder.finalize()
        self.state = self.model.state()
        self._joint_q = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        viewer.set_model(self.model)
        self.feet = [BODY_NAMES.index(f"foot_{s[0]}") for s in SIDES]
        self.columns = None
        if shoe is not None:
            anchors = [shoe.anchors_in_foot(side, body.sole_pitch) for side in SIDES]
            rest = [(shoe.right if side == "right" else shoe.left).column_bed.rest_length_m for side in SIDES]
            self.columns = anchors, rest

    def column_segments(self, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return column bottoms, tops [N, 3] [m] and shortening [N] [m] of both shoes at body poses ``body_q`` [bodies, 7]."""
        bottoms, tops = [], []
        for foot, anchors, rest in zip(self.feet, *self.columns, strict=True):
            pose = wp.transform(*body_q[foot])
            rotation = np.array(wp.quat_to_matrix(wp.transform_get_rotation(pose))).reshape(3, 3)
            origin = np.array(wp.transform_get_translation(pose))
            bottoms.append(origin + anchors @ rotation.T)
            tops.append(bottoms[-1] + np.outer(rest, rotation[:, 2]))
        bottoms, tops = np.concatenate(bottoms), np.concatenate(tops)
        shortening = np.maximum(-bottoms[:, 2], 0.0)
        bottoms[:, 2] += shortening
        tops[:, 2] = np.maximum(tops[:, 2], 0.0)
        return bottoms, tops, shortening

    def show(self, time: float, joint_q: np.ndarray, grf: np.ndarray | None = None, cop: np.ndarray | None = None):
        """Draw one frame.

        Args:
            time: Frame time [s].
            joint_q: Generalized coordinates, shape [17].
            grf: Ground reaction force per foot [N], shape [2, 3].
            cop: Center of pressure per foot [m], shape [2, 3]; NaN off contact.
        """
        viewer, device = self.viewer, self.model.device
        self._joint_q.assign(np.asarray(joint_q, dtype=np.float32))
        newton.eval_fk(self.model, self._joint_q, self.model.joint_qd, self.state)
        body_q = self.state.body_q.numpy()
        root = np.asarray(joint_q[:3])
        viewer.set_camera(*camera_look_at(root + CAMERA_EYE, (root[0], root[1], CAMERA_TARGET_HEIGHT_M)))
        viewer.begin_frame(time)
        viewer.log_state(self.state)
        if self.columns is not None:
            bottoms, tops, shortening = self.column_segments(body_q)
            colors = wp.empty(len(bottoms), dtype=wp.vec3, device=device)
            wp.launch(
                column_colors,
                dim=len(bottoms),
                inputs=[wp.array(shortening, dtype=wp.float32, device=device), COLOR_COMPRESSION_M],
                outputs=[colors],
                device=device,
            )
            viewer.log_lines(
                "shoe/columns",
                wp.array(bottoms, dtype=wp.vec3, device=device),
                wp.array(tops, dtype=wp.vec3, device=device),
                colors,
                width=0.003,
            )
        if grf is not None:
            force = np.asarray(grf, dtype=float)
            start = np.asarray(cop, dtype=float) if cop is not None else body_q[self.feet, :3] * (1.0, 1.0, 0.0)
            loaded = (np.linalg.norm(force, axis=1) > 0.0) & np.isfinite(start).all(axis=1)
            viewer.log_arrows(
                "grf",
                wp.array(start[loaded], dtype=wp.vec3, device=device),
                wp.array(start[loaded] + GRF_SCALE_M_PER_N * force[loaded], dtype=wp.vec3, device=device),
                (1.0, 0.85, 0.1),
            )
        viewer.end_frame()


def _frame_indices(trajectory: dict, speed: float, fps: float) -> np.ndarray:
    """Trajectory rows shown at ``fps`` frames per second, played at ``speed`` times real time."""
    duration = (len(trajectory["joint_q"]) - 1) * trajectory["dt"]
    return np.rint(np.arange(0.0, duration, speed / fps) / trajectory["dt"]).astype(int)


def _show_row(replay: Replay, trajectory: dict, row: int) -> None:
    replay.show(
        row * trajectory["dt"],
        trajectory["joint_q"][row],
        None if "grf" not in trajectory else trajectory["grf"][row],
        None if "cop" not in trajectory else trajectory["cop"][row],
    )


def record(
    replay: Replay,
    trajectory: dict,
    path: str | Path,
    *,
    speed: float = 0.25,
    fps: float = 25.0,
    screenshot_time: float | None = None,
) -> None:
    """Record a trajectory from a :class:`newton.viewer.ViewerGL` replay to ``.gif`` or ``.mp4``.

    Args:
        replay: Replay on a ViewerGL.
        trajectory: Trajectory to record.
        path: Output video path.
        speed: Playback speed relative to real time.
        fps: Video frame rate [1/s].
        screenshot_time: Trajectory time [s] whose frame is also written next to ``path`` as a PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _frame_indices(trajectory, speed, fps)
    frames = []
    for row in rows:
        _show_row(replay, trajectory, row)
        frames.append(replay.viewer.get_frame().numpy())
    if screenshot_time is not None:
        shot = int(np.argmin(np.abs(rows * trajectory["dt"] - screenshot_time)))
        Image.fromarray(frames[shot]).save(path.with_suffix(".png"))
    if path.suffix == ".gif":
        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000.0 / fps), loop=0)
    elif path.suffix == ".mp4":
        import imageio_ffmpeg  # noqa: PLC0415, optional; see the module docstring

        height, width = frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, macro_block_size=1)
        writer.send(None)
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
        writer.close()
    else:
        raise ValueError(f"Unsupported video format {path.suffix!r}; use .gif or .mp4")


def plot_trajectory(trajectory: dict, reference: dict | None, path: str | Path) -> None:
    """Plot joint angles, ground reaction forces and stiffness and damping commands over time.

    Args:
        trajectory: Trajectory to plot.
        reference: Trajectory drawn dashed for comparison, typically from :func:`reference_trajectory`.
        path: Output image path.
    """
    fig = Figure(figsize=(20, 13), layout="tight")
    axes = fig.subplots(4, 5)
    curves = [(trajectory, "-", "trajectory")] + ([(reference, "--", "reference")] if reference is not None else [])
    for data, style, label in curves:
        t = np.arange(len(data["joint_q"])) * data["dt"]
        for j, name in enumerate(JOINT_NAMES):
            ax = axes[j // 5, j % 5]
            ax.plot(t, np.degrees(data["joint_q"][:, 7 + j]), style, label=label)
            ax.set_title(f"{name} [deg]")
        if "grf" in data:
            for s, side in enumerate(SIDES):
                axes[2, s].plot(t, data["grf"][:, s, 2], style, color=SIDE_COLORS[s], label=label)
                axes[2, s].set_title(f"vertical GRF {side} [N]")
                axes[2, 2 + s].plot(t, data["grf"][:, s, 0], style, color=SIDE_COLORS[s], label=label)
                axes[2, 2 + s].set_title(f"forward GRF {side} [N]")
        if "action" in data:
            for k, (title, unit) in enumerate((("K", "N m/rad"), ("B", "N m s/rad"))):
                for s, side in enumerate(SIDES):
                    ax = axes[3, 2 * k + s]
                    for j in range(5):
                        ax.plot(t, data["action"][:, 5 * s + j, 1 + k], style, label=JOINT_NAMES[5 * s + j])
                    ax.set_title(f"{title} {side} [{unit}]")
                    ax.legend(fontsize=7)
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.set_xlabel("time [s]")
    for ax in axes.flat:
        ax.set_visible(ax.has_data())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a SuperMimic rollout or an F01 reference window.")
    parser.add_argument("--reference", type=Path, required=True, help="prepared F01 reference directory")
    parser.add_argument("--window", type=int, default=0, help="reference window")
    parser.add_argument("--rollout", type=Path, help="collected dataset folder; replays the reference when omitted")
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--shoes", type=Path, help="column-bed dataset with manifest.json")
    parser.add_argument("--shoe", default="puma-fast-r-nitro-elite-3", help="shoe of the reference replay")
    parser.add_argument("--record", type=Path, help="write a .gif or .mp4 instead of opening a window")
    parser.add_argument("--screenshot-time", type=float, help="also write the frame at this time [s] as a PNG")
    parser.add_argument("--plot", type=Path, help="write angle, GRF and command plots to this image")
    parser.add_argument("--speed", type=float, default=0.25, help="playback speed relative to real time")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args(argv)

    reference, body = reference_trajectory(args.reference, args.window)
    if args.rollout:
        from .collect import read_episode  # noqa: PLC0415, needs torch

        trajectory = read_episode(args.rollout, args.world, args.episode)
        shoe_name = trajectory["shoe"]
    else:
        trajectory, reference, shoe_name = reference, None, args.shoe
    if args.plot:
        plot_trajectory(trajectory, reference, args.plot)
    shoe = next(s for s in load_shoes(args.shoes) if s.name == shoe_name) if args.shoes else None
    viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=args.record is not None)
    replay = Replay(viewer, body, shoe)
    if args.record:
        record(replay, trajectory, args.record, speed=args.speed, fps=args.fps, screenshot_time=args.screenshot_time)
        viewer.close()
        return
    rows = _frame_indices(trajectory, args.speed, args.fps)
    frame = 0
    while viewer.is_running():
        _show_row(replay, trajectory, rows[frame])
        if not viewer.is_paused():
            frame = (frame + 1) % len(rows)


if __name__ == "__main__":
    main()
