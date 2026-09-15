# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render scientific SVG figures from raw samples with NumPy and the standard library.

All values, times, intervals, and tolerances use caller-supplied units. The
functions do not smooth, resample, infer terminal states, or qualify recovery.
"""

from __future__ import annotations

import html
import math
import re
import textwrap
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np

# Okabe-Ito colors, with a dark baseline and a darker yellow for white backgrounds.
BASELINE_COLOR = "#333333"
GAIN_COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#b58a00", "#e69f00")
PALETTE = (BASELINE_COLOR, *GAIN_COLORS)


@dataclass(frozen=True)
class Curve:
    """One unsmoothed trace and an optional, independently supplied terminal state.

    Args:
        label: Human-readable identity used in the legend.
        time: Raw sample times in the units of the figure's x label.
        value: Raw samples in the units of the figure's y label.
        color: SVG color shared by the trace, legend, and terminal marker.
        dashed: Draw both the trace and its legend key with a dashed line.
        terminal_time: Time of the true terminal state, not the last trace sample.
        terminal_value: Value of the true terminal state.
        qualified: Whether the caller qualifies the run; false adds an explicit
            unqualified label and a cross instead of a circular terminal marker.
    """

    label: str
    time: np.ndarray
    value: np.ndarray
    color: str
    dashed: bool = False
    terminal_time: float | None = None
    terminal_value: float | None = None
    qualified: bool = True


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value: float) -> str:
    return f"{value:.12g}"


def _tick(value: float, step: float | None = None) -> str:
    if value == 0:
        return "0"
    if step is None:
        return f"{value:.4g}"
    if abs(value) >= 1e5 or step < 1e-3:
        digits = max(2, math.ceil(math.log10(abs(value) / step)) + 1)
        return f"{value:.{digits}g}"
    decimals = max(0, 1 - math.floor(math.log10(step)))
    label = f"{value:.{decimals}f}"
    return label.rstrip("0").rstrip(".") if "." in label else label


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _band(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2 or not all(_finite(v) for v in value) or value[0] > value[1]:
        raise ValueError("Tolerance bounds must be finite, ordered (low, high) pairs")
    return float(value[0]), float(value[1])


def _limits(values: list[float], *, zero: bool = True) -> tuple[float, float]:
    bounds = [0.0, *values] if zero else values
    low, high = min(bounds), max(bounds)
    padding = (high - low) * 0.07 if high != low else max(abs(high) * 0.07, 1.0)
    return low - padding, high + padding


def _ticks(low: float, high: float) -> list[float]:
    raw = (high - low) / 5
    exponent = 10.0 ** math.floor(math.log10(raw))
    step = next(multiplier * exponent for multiplier in (1, 2, 2.5, 5, 10) if multiplier * exponent >= raw)
    first, last = math.ceil(low / step), math.floor(high / step)
    return [i * step for i in range(first, last + 1)]


def _safe_href(value) -> str | None:
    if value is None:
        return None
    href = str(value).strip()
    if not href or any(ord(c) < 32 for c in href) or "\\" in href:
        return None
    try:
        parsed = urlsplit(href)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("", "http", "https") or (not parsed.scheme and parsed.netloc):
        return None
    return _escape(href)


@dataclass
class _Plot:
    title: str
    x_label: str
    y_label: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    available: bool
    left: int = 108
    right: int = 730
    height: int = 230

    @property
    def title_lines(self) -> list[str]:
        return textwrap.wrap(self.title, 78) or [""]

    @property
    def top(self) -> int:
        return 38 + 21 * len(self.title_lines)

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def sx(self, value: float) -> float:
        return self.left + (float(value) - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.right - self.left)

    def sy(self, value: float) -> float:
        return self.bottom - (float(value) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.height

    def axes(self, *, zero: bool, origin: bool = False) -> list[str]:
        parts = []
        if self.available:
            x_ticks, y_ticks = _ticks(*self.xlim), _ticks(*self.ylim)
            x_step = x_ticks[1] - x_ticks[0] if len(x_ticks) > 1 else self.xlim[1] - self.xlim[0]
            y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else self.ylim[1] - self.ylim[0]
            for value in x_ticks:
                x = self.sx(value)
                parts.append(
                    f'<path class="grid" d="M{_number(x)},{self.top}V{self.bottom}" stroke="#e3e8ee"/>'
                    f'<text x="{_number(x)}" y="{self.bottom + 23}" text-anchor="middle">{_tick(value, x_step)}</text>'
                )
            for value in y_ticks:
                y = self.sy(value)
                parts.append(
                    f'<path class="grid" d="M{self.left},{_number(y)}H{self.right}" stroke="#e3e8ee"/>'
                    f'<text x="{self.left - 10}" y="{_number(y + 4)}" text-anchor="end">{_tick(value, y_step)}</text>'
                )
            if zero:
                parts.append(
                    f'<path class="zero-line zero-y" d="M{self.left},{_number(self.sy(0))}H{self.right}" '
                    'stroke="#718096" stroke-width="1.3"/>'
                )
            if origin:
                parts.append(
                    f'<path class="zero-line zero-x" d="M{_number(self.sx(0))},{self.top}V{self.bottom}" '
                    'stroke="#718096" stroke-width="1.3"/>'
                )
        parts.append(
            f'<path d="M{self.left},{self.top}V{self.bottom}H{self.right}" fill="none" stroke="#7d8793"/>'
            f'<text class="axis-label" x="{(self.left + self.right) / 2}" y="{self.bottom + 48}" '
            f'text-anchor="middle" font-size="14">{_escape(self.x_label)}</text>'
            f'<text class="axis-label" transform="translate(22 {(self.top + self.bottom) / 2}) rotate(-90)" '
            f'text-anchor="middle" font-size="14">{_escape(self.y_label)}</text>'
        )
        if not self.available:
            parts.append(
                f'<text x="{(self.left + self.right) / 2}" y="{(self.top + self.bottom) / 2}" '
                'text-anchor="middle" fill="#7a4032">Unavailable: no finite paired samples</text>'
            )
        return parts


def _marker(x: float, y: float, color: str, qualified: bool, *, label: str, kind: str) -> str:
    attributes = f'class="{kind} {"qualified" if qualified else "unqualified"}" stroke="{_escape(color)}"'
    title = f"<title>{_escape(label)}</title>"
    if qualified:
        return (
            f'<circle {attributes} cx="{_number(x)}" cy="{_number(y)}" r="4.5" '
            f'fill="white" stroke-width="2">{title}</circle>'
        )
    return (
        f'<path {attributes} d="M{_number(x - 5)},{_number(y - 5)}L{_number(x + 5)},{_number(y + 5)}'
        f'M{_number(x - 5)},{_number(y + 5)}L{_number(x + 5)},{_number(y - 5)}" '
        f'fill="none" stroke-width="2.4">{title}</path>'
    )


def _finish(plot: _Plot, body: list[str], legend: list[dict], caption: str, description: str) -> str:
    legend_svg = []
    row_y = plot.bottom + 77
    # Two columns keep short labels readable; long labels wrap without clipping.
    for offset in range(0, len(legend), 2):
        row = legend[offset : offset + 2]
        wrapped = [textwrap.wrap(str(entry["label"]), 43) or [""] for entry in row]
        for column, (entry, lines) in enumerate(zip(row, wrapped, strict=True)):
            x = 24 + column * 374
            y = row_y
            color = _escape(entry["color"])
            dash = ' stroke-dasharray="7 4"' if entry.get("dashed") else ""
            if entry.get("kind") == "span":
                symbol = f'<rect x="{x}" y="{y - 9}" width="23" height="12" fill="{color}" fill-opacity="0.22"/>'
            elif entry.get("kind") == "point":
                symbol = _marker(
                    x + 12,
                    y - 3,
                    entry["color"],
                    entry.get("qualified", True),
                    label=entry["label"],
                    kind="legend-marker",
                )
            else:
                symbol = f'<path d="M{x},{y - 3}h24" stroke="{color}" stroke-width="2"{dash}/>'
            text = "".join(
                f'<text x="{x + 32}" y="{y + i * 16}" font-size="12">{_escape(line)}</text>'
                for i, line in enumerate(lines)
            )
            item = f'<g class="legend-item">{symbol}{text}</g>'
            href = _safe_href(entry.get("href"))
            if href:
                item = f'<a href="{href}">{item}</a>'
            legend_svg.append(item)
        row_y += 16 * max(map(len, wrapped)) + 10
    svg_height = row_y + 10
    visible_title = "".join(
        f'<text x="24" y="{26 + i * 21}" font-size="17" font-weight="600">{_escape(line)}</text>'
        for i, line in enumerate(plot.title_lines)
    )
    # Presentation attributes keep exported SVGs independent of report CSS.
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {svg_height}" '
        f'role="img" aria-label="{_escape(plot.title)}" style="display:block;width:100%;height:auto" '
        'font-family="system-ui, sans-serif" font-size="12" fill="#263445">'
        f"<title>{_escape(plot.title)}</title><desc>{_escape(description)}</desc>"
        f'<rect width="760" height="{svg_height}" fill="white"/>{visible_title}'
        f'{"".join(body)}<g class="plot-legend">{"".join(legend_svg)}</g></svg>'
    )
    return (
        '<figure class="figure" style="margin:0 0 24px;min-width:0">'
        f'{svg}<figcaption class="figure-caption">{_escape(caption)}</figcaption></figure>'
    )


def line_figure(
    curves: list[Curve] | tuple[Curve, ...],
    *,
    title: str,
    x_label: str = "Time [ms]",
    y_label: str,
    caption: str = "",
    zero: bool = False,
    band: tuple[float, float] | None = None,
    spans: tuple[dict, ...] = (),
) -> str:
    """Render all raw samples, finite gaps, and separate true terminal markers.

    Args:
        curves: Traces; time and value must be equal-length one-dimensional arrays.
        title: Figure title.
        x_label: Horizontal quantity and units. Values are not converted.
        y_label: Vertical quantity and units. Values are not converted.
        caption: Plain text shown below the SVG and included in its description.
        zero: Include zero in the vertical limits and show its reference line.
            Otherwise fit the finite values, terminal states, and tolerance bounds.
        band: Exact lower and upper vertical tolerance bounds, if applicable.
        spans: Shaded intervals, each a dictionary with start, end, label, and
            color keys. For example, {"start": 10, "end": 30, "label": "Push",
            "color": "#e69f00"}. Bounds use the x-axis units.

    Returns:
        Accessible HTML figure containing a standalone inline SVG. Hollow circles
        mark qualified true terminal states; crosses mark unqualified states.
        No terminal marker is inferred from the pre-integration trace.
    """
    tolerance = _band(band)
    prepared, xs, ys, legend = [], [], [], []
    for curve in curves:
        time = np.asarray(curve.time, dtype=float)
        value = np.asarray(curve.value, dtype=float)
        if time.ndim != 1 or value.ndim != 1 or len(time) != len(value):
            raise ValueError("Curve time and value must be equal-length one-dimensional arrays")
        valid = np.isfinite(time) & np.isfinite(value)
        terminal = _finite(curve.terminal_time) and _finite(curve.terminal_value)
        xs.extend(time[valid].tolist())
        ys.extend(value[valid].tolist())
        if terminal:
            xs.append(float(curve.terminal_time))
            ys.append(float(curve.terminal_value))
        status = []
        if not curve.qualified:
            status.append("unqualified")
        if not valid.any():
            status.append("trace unavailable")
        elif not valid.all():
            status.append(f"{int((~valid).sum())} unavailable samples; gaps shown")
        if (curve.terminal_time is not None or curve.terminal_value is not None) and not terminal:
            status.append("terminal unavailable")
        label = curve.label + (" — " + "; ".join(status) if status else "")
        legend.append({"label": label, "color": curve.color, "dashed": curve.dashed})
        prepared.append((curve, time, value, valid, terminal))
    available = bool(xs)
    if tolerance is not None:
        ys.extend(tolerance)
    checked_spans = []
    for span in spans:
        start, end = span["start"], span["end"]
        if not _finite(start) or not _finite(end) or start > end:
            raise ValueError("Span bounds must be finite and ordered")
        checked_spans.append(span)
        xs.extend((float(start), float(end)))
    plot = _Plot(title, x_label, y_label, _limits(xs or [0.0]), _limits(ys or [0.0], zero=zero), available)
    body = []
    if available:
        for span in checked_spans:
            start, end = float(span["start"]), float(span["end"])
            body.append(
                f'<rect class="plot-span" data-start="{_number(start)}" data-end="{_number(end)}" '
                f'x="{_number(plot.sx(start))}" y="{plot.top}" width="{_number(plot.sx(end) - plot.sx(start))}" '
                f'height="{plot.height}" fill="{_escape(span["color"])}" fill-opacity="0.12">'
                f"<title>{_escape(span['label'])}: {_number(start)} to {_number(end)} ({_escape(x_label)})</title></rect>"
            )
            legend.append({"label": span["label"], "color": span["color"], "kind": "span"})
        if tolerance is not None:
            low, high = tolerance
            body.append(
                f'<rect class="tolerance-band" data-low="{_number(low)}" data-high="{_number(high)}" '
                f'x="{plot.left}" y="{_number(plot.sy(high))}" width="{plot.right - plot.left}" '
                f'height="{_number(plot.sy(low) - plot.sy(high))}" fill="#009e73" fill-opacity="0.10" '
                f'stroke="#009e73" stroke-width="0.8"><title>Tolerance: {_number(low)} to {_number(high)}</title></rect>'
            )
            legend.append({"label": f"Tolerance [{_tick(low)}, {_tick(high)}]", "color": "#009e73", "kind": "span"})
    body.extend(plot.axes(zero=zero))
    has_terminal = False
    for curve, time, value, valid, terminal in prepared:
        commands, previous_valid = [], False
        for t, v, finite in zip(time, value, valid, strict=True):
            if finite:
                commands.append(f"{'L' if previous_valid else 'M'}{_number(plot.sx(t))},{_number(plot.sy(v))}")
            previous_valid = bool(finite)
        dash = ' stroke-dasharray="7 4"' if curve.dashed else ""
        if commands:
            body.append(
                f'<path class="raw-trace" data-label="{_escape(curve.label)}" data-sample-count="{int(valid.sum())}" '
                f'd="{" ".join(commands)}" stroke="{_escape(curve.color)}" stroke-width="1.8" '
                f'fill="none" stroke-linejoin="round"{dash}><title>{_escape(curve.label)}</title></path>'
            )
            # Isolated samples have no drawable segment; keep them visible without bridging gaps.
            for index, finite in enumerate(valid):
                if (
                    finite
                    and (index == 0 or not valid[index - 1])
                    and (index == len(valid) - 1 or not valid[index + 1])
                ):
                    body.append(
                        f'<circle class="isolated-sample" cx="{_number(plot.sx(time[index]))}" '
                        f'cy="{_number(plot.sy(value[index]))}" r="2" fill="{_escape(curve.color)}"/>'
                    )
        if terminal:
            has_terminal = True
            body.append(
                _marker(
                    plot.sx(curve.terminal_time),
                    plot.sy(curve.terminal_value),
                    curve.color,
                    curve.qualified,
                    label=f"{curve.label}: true terminal state ({_number(curve.terminal_time)}, {_number(curve.terminal_value)})"
                    + (" — unqualified" if not curve.qualified else ""),
                    kind="terminal-marker",
                )
            )
    if has_terminal:
        legend.append(
            {"label": "True terminal state (not last trace sample)", "color": BASELINE_COLOR, "kind": "point"}
        )
    description = (
        f"{title}. {x_label}; {y_label}. Raw samples without smoothing or resampling. "
        "Missing or nonfinite samples break the trace. Hollow circles show qualified true terminal states; "
        "crosses show unqualified true terminal states. "
        + caption
        + " "
        + "; ".join(entry["label"] for entry in legend)
    )
    return _finish(plot, body, legend, caption, description)


def scatter_figure(
    points: list[dict],
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_band: tuple[float, float] | None = None,
    y_band: tuple[float, float] | None = None,
    caption: str = "",
) -> str:
    """Show paired terminal quantities with exact engineering acceptance bounds.

    Args:
        points: Dictionaries with label, x, y, and color keys; optional qualified
            (default true) and href keys. Unqualified finite points use crosses.
            Missing/nonfinite pairs are explicitly labeled unavailable, not plotted
            at a fabricated origin. Links allow relative paths and HTTP(S) only.
            Optional legend_label groups finite points with the same legend label,
            color, and qualification into one key, while retaining each full label
            and link on its marker. Unavailable cases remain individually labeled.
        title: Figure title.
        x_label: Horizontal quantity and units.
        y_label: Vertical quantity and units.
        x_band: Exact horizontal acceptance interval, if supplied.
        y_band: Exact vertical acceptance interval, if supplied. Together the two
            bands define an exact rectangle; neither is enlarged for visibility.
        caption: Plain text figure caption.

    Returns:
        Accessible HTML figure with a standalone SVG and labeled legend.
    """
    x_bounds, y_bounds = _band(x_band), _band(y_band)
    finite = [p for p in points if _finite(p.get("x")) and _finite(p.get("y"))]
    xs = [float(p["x"]) for p in finite] + list(x_bounds or ())
    ys = [float(p["y"]) for p in finite] + list(y_bounds or ())
    plot = _Plot(title, x_label, y_label, _limits(xs or [0.0]), _limits(ys or [0.0]), bool(finite))
    body, legend = [], []
    if finite and (x_bounds is not None or y_bounds is not None):
        low_x, high_x = x_bounds if x_bounds is not None else plot.xlim
        low_y, high_y = y_bounds if y_bounds is not None else plot.ylim
        body.append(
            f'<rect class="acceptance-box" data-x-low="{_number(low_x)}" data-x-high="{_number(high_x)}" '
            f'data-y-low="{_number(low_y)}" data-y-high="{_number(high_y)}" '
            f'x="{_number(plot.sx(low_x))}" y="{_number(plot.sy(high_y))}" '
            f'width="{_number(plot.sx(high_x) - plot.sx(low_x))}" '
            f'height="{_number(plot.sy(low_y) - plot.sy(high_y))}" '
            'fill="#009e73" fill-opacity="0.10" stroke="#009e73" stroke-width="0.8">'
            "<title>Exact engineering acceptance bounds; not enlarged for visibility</title></rect>"
        )
    body.extend(plot.axes(zero=True, origin=True))
    seen_groups = set()
    for point in points:
        qualified = bool(point.get("qualified", True))
        valid = _finite(point.get("x")) and _finite(point.get("y"))
        label = str(point["label"])
        if not qualified:
            label += " — unqualified"
        if not valid:
            label += " — unavailable (missing/nonfinite terminal pair)"
        grouped = valid and point.get("legend_label") is not None
        legend_label = str(point["legend_label"]) if grouped else label
        if grouped and not qualified:
            legend_label += " — unqualified"
        group_key = (legend_label, point["color"], qualified)
        if not grouped or group_key not in seen_groups:
            legend.append(
                {
                    "label": legend_label,
                    "color": point["color"],
                    "kind": "point",
                    "qualified": qualified and valid,
                    "href": None if grouped else point.get("href"),
                }
            )
            if grouped:
                seen_groups.add(group_key)
        if valid:
            tooltip = f"{label}: {x_label} = {_number(float(point['x']))}; {y_label} = {_number(float(point['y']))}"
            marker = _marker(
                plot.sx(point["x"]), plot.sy(point["y"]), point["color"], qualified, label=tooltip, kind="scatter-point"
            )
            href = _safe_href(point.get("href"))
            body.append(f'<a href="{href}">{marker}</a>' if href else marker)
    if x_bounds is not None or y_bounds is not None:
        label = "Acceptance: "
        label += f"x [{_tick(x_bounds[0])}, {_tick(x_bounds[1])}]" if x_bounds is not None else "any x"
        label += f"; y [{_tick(y_bounds[0])}, {_tick(y_bounds[1])}]" if y_bounds is not None else "; any y"
        legend.append({"label": label, "color": "#009e73", "kind": "span"})
    description = (
        f"{title}. {x_label}; {y_label}. True zero reference lines. Exact acceptance bounds, not enlarged. "
        "Hollow circles: qualified. Crosses: unqualified. Missing/nonfinite pairs are unavailable, not placed at zero. "
        + caption
        + " "
        + "; ".join(entry["label"] for entry in legend)
    )
    return _finish(plot, body, legend, caption, description)


def extract_svg(figure_html: str) -> str:
    """Extract one self-contained SVG for saving separately from its HTML figure."""
    match = re.search(r"<svg\b[^>]*>.*?</svg>", figure_html, flags=re.DOTALL)
    if match is None:
        raise ValueError("Figure does not contain an SVG")
    return match.group(0)
