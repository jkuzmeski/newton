# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render saved Hogan rollout traces as animated, labeled GIFs."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .report import _Scene


def write_motion_gif(trial, trace: dict[str, np.ndarray], path: Path, *, title: str) -> Path:
    """Render saved simulated and measured motion with the measured force trace.

    This function only draws the supplied saved trace. It does not run the model.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Rendering Hogan motion GIFs requires Pillow from the examples extra") from exc

    required = {"time_s", "state", "grf_n", "reference_state"}
    missing = required.difference(trace)
    if missing:
        raise ValueError(f"Saved motion trace is missing {', '.join(sorted(missing))}")
    time = np.asarray(trace["time_s"])
    state = np.asarray(trace["state"])
    measured_state = np.asarray(trace["reference_state"])
    simulated_force = np.asarray(trace["grf_n"])
    if not len(time) or state.shape != (len(time), 6) or measured_state.shape != state.shape:
        raise ValueError("Saved motion and measured reference must have matching non-empty stance samples")
    if simulated_force.shape != (len(time), 2) or not np.isfinite(time).all():
        raise ValueError("Saved motion force and time arrays are malformed")
    if trial.grf_n is None or not len(trial.force_time_s):
        raise ValueError(f"Measured GRF is missing for stance {trial.id}")

    width, height = 720, 610
    left, right = 55, 695
    ground_y = 410
    scale = 225.0
    scene = _Scene(trial.chain, trial.shoe, trial.provenance["rest_of_body"]["com_local_m"])
    count = min(64, len(time))
    indices = np.unique(np.linspace(0, len(time) - 1, count).round().astype(int))
    font = ImageFont.load_default(size=14)
    title_font = ImageFont.load_default(size=18)
    frames = []
    force_clock = np.asarray(trial.force_time_s)
    force_measured = np.asarray(trial.grf_n)
    for index in indices:
        q = state[index]
        ref = measured_state[index]
        _, _, ankle, _ = trial.chain.kinematics(q)
        _, _, ref_ankle, _ = trial.chain.kinematics(ref)
        center_x = 0.5 * (ankle[0] + ref_ankle[0])

        def xy(point, center_x=center_x):
            return (int(width / 2 + (point[0] - center_x) * scale), int(ground_y - point[1] * scale))

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((left, 18), title, fill="#172b40", font=title_font)
        draw.text((left, 39), "Blue: simulated    Grey: measured", fill="#526174", font=font)
        draw.line((left, ground_y, right, ground_y), fill="#7b9d86", width=2)

        def draw_pose(pose, simulated, draw=draw, xy=xy):
            hip, knee, foot, _ = trial.chain.kinematics(pose)
            rotation = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])
            com = hip + rotation @ scene.rest_com
            # The local shoe hulls are the registered digital shoe geometry.
            for outline, fill, color in (
                (scene.last, "#f1dfc8" if simulated else "white", "#b5803c" if simulated else "#8a97a6"),
                (scene.midsole, "#d2e8e4" if simulated else "white", "#3e7a6f" if simulated else "#8a97a6"),
            ):
                points = [xy(p) for p in scene._shoe_points(outline, foot, pose)]
                if len(points) >= 3:
                    draw.polygon(points, fill=fill, outline=color)
            color = "#1261a0" if simulated else "#8a97a6"
            draw.line([xy(com), xy(hip), xy(knee), xy(foot)], fill=color, width=5 if simulated else 3)
            for joint in (hip, knee, foot):
                x, y = xy(joint)
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="white", outline=color, width=2)

        draw_pose(ref, False)
        draw_pose(q, True)
        t = float(time[index])
        measured = np.array([np.interp(t, force_clock, force_measured[:, c]) for c in range(2)])
        simulated = simulated_force[index]
        draw.text(
            (left, 61),
            f"{t * 1000:.0f} ms    GRF measured Fx/Fz {measured[0]:.0f}/{measured[1]:.0f} N    simulated {simulated[0]:.0f}/{simulated[1]:.0f} N",
            fill="#172b40",
            font=font,
        )

        # A shared stance timeline makes each frame's motion and force readable.
        x0, x1, y0, y1 = left, right, 490, 585
        draw.text((left, 425), "Ground reaction force through the saved stance", fill="#172b40", font=font)
        max_force = max(100.0, float(np.nanmax(np.abs(np.concatenate((force_measured[:, 1], simulated_force[:, 1]))))))
        visible = (force_clock >= time[0]) & (force_clock <= time[-1])
        for axis, color, label in ((1, "#7a5230", "Fz"), (0, "#d18a00", "Fx")):
            vals = force_measured[visible, axis]
            clocks = force_clock[visible]
            if len(vals) > 1:
                pts = [
                    (
                        int(x0 + (tt - time[0]) / max(time[-1] - time[0], 1e-9) * (x1 - x0)),
                        int((y0 + y1) / 2 - val / max_force * 48),
                    )
                    for tt, val in zip(clocks, vals, strict=True)
                ]
                draw.line(pts, fill=color, width=2)
            simpts = [
                (
                    int(x0 + (tt - time[0]) / max(time[-1] - time[0], 1e-9) * (x1 - x0)),
                    int((y0 + y1) / 2 - val / max_force * 48),
                )
                for tt, val in zip(time, simulated_force[:, axis], strict=True)
            ]
            if len(simpts) > 1:
                draw.line(simpts, fill="#1261a0" if axis else "#6599bc", width=2)
            draw.text((left + (1 - axis) * 300, 448), label + " measured", fill=color, font=font)
            draw.text(
                (left + (1 - axis) * 300, 466), label + " simulated", fill="#1261a0" if axis else "#6599bc", font=font
            )
        cursor = int(x0 + (t - time[0]) / max(time[-1] - time[0], 1e-9) * (x1 - x0))
        draw.line((cursor, y0, cursor, y1), fill="#1261a0", width=2)
        draw.text((x0, y1 + 7), f"{time[0] * 1000:.0f} ms", fill="#526174", font=font)
        draw.text((x1 - 45, y1 + 7), f"{time[-1] * 1000:.0f} ms", fill="#526174", font=font)
        frames.append(image)

    if len(frames) < 2:
        raise ValueError("A motion GIF needs at least two saved samples")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=False)
    return path


def image_markup(path: Path, *, alt: str) -> str:
    """Return local HTML for a generated motion GIF."""
    return f'<figure class="motion-gif"><figcaption><strong>{html.escape(alt)}</strong></figcaption><img src="{html.escape(path.as_posix())}" alt="{html.escape(alt)}" style="display:block;width:100%;height:auto;border:1px solid #dce3eb;border-radius:.5rem"></figure>'
