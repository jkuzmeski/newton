# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render shared SVG and HTML helpers for the generative Hogan fit report."""

from __future__ import annotations

import html
import math

import numpy as np

from ..cartesian.shoe import Shoe
from ..cartesian.shoe import _hull as _shoe_hull

CONTACT_THRESHOLD_N = 50.0
MEASURED = "#3d4b5a"
RUN_COLORS = ("#1261a0", "#d94801", "#2b8a3e", "#6a3d9a")
BASELINE = "#c08a00"
COORDINATE_INFO = {
    "hip_x": ("Hip center, forward", "m", "N/m", "N·s/m"),
    "hip_z": ("Hip center, up", "m", "N/m", "N·s/m"),
    "pelvis": ("Lumped body tilt from upright (\u2212 = forward lean)", "rad", "N·m/rad", "N·m·s/rad"),
    "hip": ("Thigh relative to pelvis (+ = flexion)", "rad", "N·m/rad", "N·m·s/rad"),
    "knee": ("Shank relative to thigh (\u2212 = flexion)", "rad", "N·m/rad", "N·m·s/rad"),
    "ankle": ("Foot relative to shank (+ = dorsiflexion, 0 = foot ⟂ shank)", "rad", "N·m/rad", "N·m·s/rad"),
}


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    raw = (hi - lo) / max(count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw * 0.999)
    start = math.ceil(lo / step - 1e-9) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step + 1e-9) + 1)]


def _plot(series, *, xlabel: str, ylabel: str, band=None, xlim=None) -> str:
    """Return an SVG line plot; each series is ``(label, x, y, color, dash)``."""
    left, right, top, bottom = 78.0, 700.0, 66.0, 270.0
    xs = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
    ys = np.concatenate([np.asarray(s[2], dtype=float) for s in series])
    ys = ys[np.isfinite(ys)]
    xlim = xlim or (float(np.nanmin(xs)), float(np.nanmax(xs)))
    lo, hi = (float(ys.min()), float(ys.max())) if len(ys) else (0.0, 1.0)
    pad = max(0.06 * (hi - lo), 1e-6 + 1e-3 * max(abs(lo), abs(hi)))
    lo, hi = lo - pad, hi + pad

    def sx(x):
        return left + (np.asarray(x) - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)

    def sy(y):
        return bottom - (np.asarray(y) - lo) / (hi - lo) * (bottom - top)

    parts = []
    if band is not None:
        x0, x1 = sx(max(band[0], xlim[0])), sx(min(band[1], xlim[1]))
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{x1 - x0:.1f}" height="{bottom - top}" fill="#f1f6fb"/>')
    for value in _nice_ticks(*xlim):
        px = sx(value)
        parts.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}" stroke="#e5eaf0"/>'
            f'<text x="{px:.1f}" y="{bottom + 18}" text-anchor="middle">{value:g}</text>'
        )
    for value in _nice_ticks(lo, hi):
        py = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" stroke="#e5eaf0"/>'
            f'<text x="{left - 8}" y="{py + 4:.1f}" text-anchor="end">{value:g}</text>'
        )
    if lo < 0.0 < hi:
        parts.append(f'<line x1="{left}" y1="{sy(0.0):.1f}" x2="{right}" y2="{sy(0.0):.1f}" stroke="#9aa8b8"/>')
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9aa8b8"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9aa8b8"/>'
    )
    legend_x, legend_y = left, 22
    for label, x_raw, y_raw, color, dash in series:
        x, y = np.asarray(x_raw, dtype=float), np.asarray(y_raw, dtype=float)
        keep = np.isfinite(y) & (x >= xlim[0]) & (x <= xlim[1])
        stride = max(1, int(np.count_nonzero(keep) // 900))
        x, y = x[keep][::stride], y[keep][::stride]
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(sx(x), sy(y), strict=True))
        style = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{style}/>')
        entry = 44 + 7.6 * len(label)
        if legend_x + entry > right + 20 and legend_x > left:
            legend_x, legend_y = left, legend_y + 20
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" stroke="{color}" stroke-width="3"{style}/>'
            f'<text x="{legend_x + 32}" y="{legend_y + 5}">{html.escape(label)}</text>'
        )
        legend_x += entry
    return (
        f'<svg class="plot" viewBox="0 0 720 324" role="img" aria-label="{html.escape(ylabel)} versus {html.escape(xlabel)}">'
        '<g font-family="system-ui,sans-serif" font-size="15" fill="#526174">'
        + "".join(parts)
        + f'<text x="389" y="316" text-anchor="middle">{html.escape(xlabel)}</text>'
        f'<text x="18" y="{(top + bottom) / 2:.0f}" transform="rotate(-90 18 {(top + bottom) / 2:.0f})" '
        f'text-anchor="middle">{html.escape(ylabel)}</text></g></svg>'
    )


def _figure(title: str, svg: str, caption: str = "") -> str:
    note = f"<span>{caption}</span>" if caption else ""
    return f"<figure><figcaption><strong>{title}</strong>{note}</figcaption>{svg}</figure>"


def _hull(points: np.ndarray) -> np.ndarray:
    return _shoe_hull(np.round(points, 6))


def _table(header: list[str], rows: list[list[str]], caption: str = "") -> str:
    head = "".join(f'<th scope="col">{h}</th>' for h in header)
    body = "".join(
        "<tr>" + f'<th scope="row">{r[0]}</th>' + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>" for r in rows
    )
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="table-scroll"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class _Scene:
    """Draw sagittal stick figures with the registered shoe outline and ground forces."""

    def __init__(self, chain, shoe: Shoe, rest_com_local_m):
        self.chain = chain
        self.shoe = shoe
        self.last = _hull(shoe.last_vertices_local_m[:, (0, 2)])
        self.midsole = _hull(np.vstack((shoe.anchor_local_m, shoe.attachment_local_m))[:, (0, 2)])
        self.rest_com = np.asarray(rest_com_local_m, dtype=float)

    def _shoe_points(self, local: np.ndarray, ankle: np.ndarray, q) -> np.ndarray:
        angle = self.chain.angle(q, 3) - self.shoe.static_pitch_rad
        c, s = math.cos(angle), math.sin(angle)
        return local @ np.array([[c, s], [-s, c]]) + ankle

    def pose(self, q, to_px, *, simulated: bool, labels: bool = False) -> str:
        hip, knee, ankle, _ = self.chain.kinematics(q)
        pelvis = float(q[2])
        rotation = np.array([[math.cos(pelvis), -math.sin(pelvis)], [math.sin(pelvis), math.cos(pelvis)]])
        com = hip + rotation @ self.rest_com

        def poly(points):
            return " ".join(f"{a:.1f},{b:.1f}" for a, b in (to_px(p) for p in points))

        sole = poly(self._shoe_points(self.midsole, ankle, q))
        out = []
        if simulated:
            last = poly(self._shoe_points(self.last, ankle, q))
            out.append(f'<polygon points="{last}" fill="#e7b270" fill-opacity=".55" stroke="#b5803c"/>')
            out.append(f'<polygon points="{sole}" fill="#5c9f93" fill-opacity=".55" stroke="#3e7a6f"/>')
            stroke, width, dash = "#1261a0", 4.5, ""
        else:
            out.append(f'<polygon points="{sole}" fill="none" stroke="#8a97a6" stroke-dasharray="4 3"/>')
            stroke, width, dash = "#8a97a6", 2.2, ' stroke-dasharray="6 4"'
        out.append(
            f'<polyline points="{poly([com, hip])}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash}/>'
        )
        out.append(
            f'<polyline points="{poly([hip, knee, ankle])}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash}/>'
        )
        c = to_px(com)
        out.append(
            f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{9 if simulated else 7}" '
            f'fill="{"#1261a0" if simulated else "none"}" stroke="{stroke}"/>'
        )
        for joint in (hip, knee, ankle):
            p = to_px(joint)
            out.append(
                f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.5" fill="white" stroke="{stroke}" stroke-width="2"/>'
            )
        if labels:
            mass = self.chain.masses_kg
            for point, text, dx in (
                (com, f"Lumped rest of body {mass[0]:.1f} kg", 14),
                (hip, "Hip center (x, z free)", 10),
                (0.5 * (hip + knee), f"Thigh {mass[1]:.1f} kg", 10),
                (knee, "Knee", 10),
                (0.5 * (knee + ankle), f"Shank {mass[2]:.1f} kg", 10),
                (ankle, f"Ankle; foot + shoe carrier {mass[3]:.1f} kg", 14),
            ):
                p = to_px(point)
                out.append(f'<text x="{p[0] + dx:.1f}" y="{p[1] + 4:.1f}" fill="#172b40">{html.escape(text)}</text>')
        return "".join(out)

    @staticmethod
    def arrow(origin, force, to_px, color: str, scale_m_per_n: float = 2.0e-4) -> str:
        a = to_px(origin)
        b = to_px(np.asarray(origin) + scale_m_per_n * np.asarray(force))
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        head = [(b[0] - 9 * math.cos(angle + s), b[1] - 9 * math.sin(angle + s)) for s in (-0.45, 0.45)]
        return (
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}" stroke-width="2.5"/>'
            f'<polygon points="{b[0]:.1f},{b[1]:.1f} {head[0][0]:.1f},{head[0][1]:.1f} {head[1][0]:.1f},{head[1][1]:.1f}" fill="{color}"/>'
        )


def _simulated_cop(chain, trace) -> np.ndarray:
    """Return the simulated center of pressure on the ground [m]; NaN out of contact."""
    cop = np.full(len(trace["time_s"]), np.nan)
    for k, (q, force, moment) in enumerate(
        zip(trace["state"], trace["grf_n"], trace["ankle_contact_moment_nm"], strict=True)
    ):
        if force[1] > CONTACT_THRESHOLD_N:
            ankle = chain.kinematics(q)[2]
            cop[k] = ankle[0] + (moment - ankle[1] * force[0]) / force[1]
    return cop


def _snapshots(scene: _Scene, reference, trace, times, *, label: str) -> str:
    """Draw a whole-chain row and a zoomed foot row at each time."""
    width = 260.0
    # (row, panel height [px], scale [px/m], ground line [px], arrow scale [m/N])
    rows = (("body", 380.0, 215.0, 350.0, 2.0e-4), ("foot", 250.0, 700.0, 220.0, 1.0e-4))
    cop_sim = _simulated_cop(scene.chain, trace)
    panels = []
    top = 0.0
    for row, height, scale, ground, force_scale in rows:
        for i, t in enumerate(times):
            k = int(np.argmin(np.abs(trace["time_s"] - t)))
            q, q_ref = trace["state"][k], trace["reference_state"][k]
            hip_ref, _, ankle_ref, _ = scene.chain.kinematics(q_ref)
            cx = 0.5 * (hip_ref[0] + ankle_ref[0]) if row == "body" else ankle_ref[0] + 0.03

            def to_px(p, cx=cx, scale=scale, ground=ground):
                return (width / 2 + (p[0] - cx) * scale, ground - p[1] * scale)

            measured = np.array(
                [np.interp(t, reference["grf_time_s"], reference["grf_target_n"][:, j]) for j in range(2)]
            )
            measured_cop = float(np.interp(t, reference["grf_time_s"], reference["cop_target_m"]))
            parts = [
                f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" fill="none" stroke="#eef2f6"/>',
                f'<line x1="0" y1="{ground}" x2="{width}" y2="{ground}" stroke="#7b9d86" stroke-width="2"/>',
                scene.pose(q_ref, to_px, simulated=False),
                scene.pose(q, to_px, simulated=True),
            ]
            if measured[1] > CONTACT_THRESHOLD_N:
                parts.append(_Scene.arrow([measured_cop, 0.0], measured, to_px, "#7a5230", force_scale))
            if np.isfinite(cop_sim[k]):
                parts.append(_Scene.arrow([cop_sim[k], 0.0], trace["grf_n"][k], to_px, "#d94801", force_scale))
            if row == "body":
                parts.append(
                    f'<text x="{width / 2}" y="22" text-anchor="middle" font-weight="650" fill="#172b40">{t * 1000:.0f} ms</text>'
                    f'<text x="{width / 2}" y="42" text-anchor="middle">Fz meas. {measured[1]:.0f} N · sim. {trace["grf_n"][k][1]:.0f} N</text>'
                )
            else:
                pitch = math.degrees(scene.chain.angle(q, 3) - scene.chain.angle(q_ref, 3))
                parts.append(
                    f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle">Shoe pitch sim &minus; ref {pitch:+.1f}°</text>'
                )
            panels.append(
                f'<svg x="{i * width}" y="{top}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                + "".join(parts)
                + "</svg>"
            )
        top += height
    total = width * len(times)
    return (
        f'<svg class="scene" viewBox="0 0 {total:.0f} {top:.0f}" role="img" aria-label="{html.escape(label)}">'
        '<g font-family="system-ui,sans-serif" font-size="14" fill="#526174">' + "".join(panels) + "</g></svg>"
    )


_CSS = """
:root{--ink:#172b40;--muted:#526174;--line:#dce3eb;--panel:#f5f7fa;--blue:#1261a0}
*{box-sizing:border-box}body{margin:0 auto;max-width:1200px;padding:3rem 2rem;color:var(--ink);background:#fff;font:16px/1.65 system-ui,sans-serif}
a{color:var(--blue)}h1,h2,h3{line-height:1.25;letter-spacing:-.02em}h1{max-width:900px;margin:.6rem 0 1rem;font-size:clamp(1.9rem,3.6vw,2.8rem)}
h2{margin:0 0 1.25rem;font-size:1.7rem}h3{margin:2rem 0 .75rem;font-size:1.15rem}p{margin:.75rem 0}
.eyebrow{color:var(--blue);font-size:.8rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase}
.subtitle{max-width:820px;color:var(--muted);font-size:1.1rem}
nav{display:flex;flex-wrap:wrap;gap:.6rem 1.6rem;padding:1.1rem 0;border-bottom:1px solid var(--line)}nav a{font-size:.9rem;font-weight:650;text-decoration:none}
section{margin-top:3rem}.status{margin:1.5rem 0 .5rem;padding:1.1rem 1.3rem;border:1px solid #edc9b8;border-radius:.65rem;background:#fff8f4}
.status strong{display:block;color:#a3331d;font-size:.85rem;letter-spacing:.015em}.status p{margin:.4rem 0 0;font-size:.95rem}
.note{color:var(--muted);font-size:.9rem}.key{margin:1.25rem 0;padding:1rem 1.25rem;border-left:4px solid var(--blue);background:#f1f6fb;border-radius:.3rem}
.table-scroll{max-width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:.55rem;margin:1rem 0}
table{width:100%;border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums}caption{padding:.7rem 1rem;text-align:left;color:var(--muted)}
th,td{padding:.7rem 1rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}thead th{background:var(--panel);font-weight:650}
tbody th{font-weight:600}tbody tr:last-child>*{border-bottom:0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.25rem;margin:1rem 0}
figure{margin:0;padding:.9rem;border:1px solid var(--line);border-radius:.6rem;min-width:0}
figcaption{margin-bottom:.4rem;font-size:.9rem}figcaption strong{display:block}figcaption span{display:block;color:var(--muted);font-size:.85rem}
svg.plot,svg.scene,svg.model{display:block;width:100%;height:auto}svg.model{max-width:620px;margin:auto}
.workflow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.75rem;list-style:none;padding:0;margin:1.25rem 0;counter-reset:s}
.workflow li{padding:.9rem;border:1px solid var(--line);border-radius:.5rem;counter-increment:s}.workflow strong{display:block;font-size:.95rem}
.workflow strong::before{content:counter(s) " / ";color:var(--blue)}.workflow span{display:block;margin-top:.3rem;color:var(--muted);font-size:.85rem}
.equation{margin:.8rem 0;padding:.85rem 1rem;border-left:3px solid var(--blue);background:#f1f6fb;font:.95rem/1.7 ui-monospace,monospace;overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;color:var(--muted);font-size:.85rem;margin:.5rem 0}
.legend i{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;margin-right:.35rem}
pre{max-width:100%;padding:1rem;overflow-x:auto;border-radius:.4rem;background:var(--panel);font-size:.85rem}
ol.issues li{margin:.6rem 0}footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
@media(max-width:800px){body{padding:1.5rem 1rem}.grid{grid-template-columns:1fr}.workflow{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media print{body{max-width:none;padding:0;font-size:11pt}nav{display:none}figure{break-inside:avoid}}
"""
