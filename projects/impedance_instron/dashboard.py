# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serve a live, offline training dashboard for impedance reinforcement-learning runs.

The dashboard reads the plain-text logs written by
:mod:`projects.impedance_instron.train`, overlays every run on shared axes and
renders the page as inline SVG. It uses only the standard library and NumPy, so
it works without network access, JavaScript or a plotting package.
"""

from __future__ import annotations

import argparse
import html
import math
import re
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qsl, urlencode

import numpy as np

DEFAULT_DIRECTORY = "outputs/impedance_instron"
LOG_PATTERN = "train_*.log"
RUNNING_WINDOW_S = 60.0
MAX_POINTS = 600

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_ITERATION_RE = re.compile(
    rf"^iter\s+(?P<iteration>\d+)\s*\|\s*"
    rf"return\s+mean=\s*(?P<mean>{_NUMBER})\s+med=\s*(?P<median>{_NUMBER})\s+best=\s*(?P<best>{_NUMBER})\s*\|\s*"
    rf"pi=\s*(?P<pi>{_NUMBER})\s+v=\s*(?P<v>{_NUMBER})\s+ent=\s*(?P<ent>{_NUMBER})\s*\|\s*"
    rf"ev=\s*(?P<ev>{_NUMBER})\s+kl=\s*(?P<kl>{_NUMBER})\s+clip=\s*(?P<clip>{_NUMBER})\s*$"
)
_FROZEN_RE = re.compile(
    rf"^frozen(?P<source>\s+\S+)?:\s+episodes=(?P<episodes>\d+)\s+mean=(?P<mean>{_NUMBER})\s+"
    rf"median=(?P<median>{_NUMBER})\s+best=(?P<best>{_NUMBER})\s+worst=(?P<worst>{_NUMBER})\s*$"
)
_EVAL_REQUIRED = (
    "iteration",
    "objective_j",
    "feasible",
    "on_task",
    "excursion_duration",
    "excursion_impulse",
    "excursion_momentum",
    "violation_total",
    "eval_return",
)
_EVAL_PHYSICAL = (
    "peak_fz_n",
    "peak_fz_ref_n",
    "peak_time_pct",
    "peak_time_ref_pct",
    "fz_rms_n",
    "impulse_err_pct",
    "com_vz_rms",
    "com_z_rms_mm",
    "contact_ms",
    "peak_compression_mm",
)
_EVAL_FLAGS = ("feasible", "on_task")
_CONTACT_THRESHOLD_N = 20.0
UNTAGGED = "untagged"
NOTHING = "__none__"

# Share of body weight above which the shoe counts as loaded. This must stay equal to
# ``projects.impedance_instron.env.CONTACT_FORCE_FRACTION``, the one constant the environment's
# stance gate uses to write ``contact_start_s`` and ``contact_end_s`` into each evaluation archive.
# It is restated instead of imported because ``env`` pulls in Warp and this page stays pure NumPy.
CONTACT_FORCE_FRACTION = 0.02

# A trace is too coarse to imply a continuous line once the richest trace on the same axes carries
# more than this many samples for each of its own.
COARSE_SAMPLE_RATIO = 4

# Individual samples are drawn only while they stay countable; past this they merge into a band.
MARKER_LIMIT = 160

# Waveform panels carry every selected run at once, so they use a smaller point budget. At 548 SVG
# units of plot width this is still finer than one point per two units.
WAVEFORM_POINTS = 300

# Relative disagreement above which two runs are reported as carrying different measured references.
REFERENCE_TOLERANCE = 0.01

_CHANNELS = ("iteration", "mean", "median", "best", "pi", "v", "ent", "ev", "kl", "clip")
_EXCURSIONS = (
    ("duration", "excursion_duration", False),
    ("impulse", "excursion_impulse", "6 4"),
    ("momentum", "excursion_momentum", "2 3"),
)
_PHYSICAL_METRICS = (
    ("Peak vertical force", "N", "peak_fz_n", ("record", "peak_fz_ref_n", "measured peak force")),
    ("Time of peak force", "% of contact", "peak_time_pct", ("record", "peak_time_ref_pct", "measured peak timing")),
    ("Vertical force RMS error", "N", "fz_rms_n", None),
    ("Impulse error", "%", "impulse_err_pct", ("zero", "measured impulse")),
    ("COM vertical velocity RMS error", "m/s", "com_vz_rms", None),
    ("COM height RMS error", "mm", "com_z_rms_mm", None),
    ("Contact duration", "ms", "contact_ms", ("contact", "measured contact")),
    ("Peak shoe compression", "mm", "peak_compression_mm", None),
)
# One panel per waveform: title, unit, scale, simulated channel, and how the second curve is found.
# ``shared`` is one measured curve for every run; ``per_run`` is that run's own command.
_WAVEFORMS = (
    ("Vertical ground reaction force", "N", 1.0, "shoe_fz_n", ("shared", "reference_fz_n", "measured")),
    ("Fore-aft ground reaction force", "N", 1.0, "shoe_fx_n", ("shared", "reference_fx_n", "measured")),
    ("COM height", "mm", 1000.0, "com_z_m", ("shared", "reference_com_z_m", "reference")),
    ("COM vertical velocity", "m/s", 1.0, "com_vz_m_s", ("shared", "reference_com_vz_m_s", "reference")),
    ("Leg length", "mm", 1000.0, "leg_length_m", ("per_run", "commanded_length_m", "commanded L0")),
    ("Commanded stiffness", "N/m", 1.0, "stiffness_n_m", None),
    ("Commanded damping ratio", "0-1", 1.0, "damping_ratio", None),
)
_TRACE_ARRAYS = (
    "time_s",
    "shoe_fz_n",
    "reference_fz_n",
    "shoe_fx_n",
    "reference_fx_n",
    "com_z_m",
    "reference_com_z_m",
    "com_vz_m_s",
    "reference_com_vz_m_s",
    "leg_length_m",
    "commanded_length_m",
    "stiffness_n_m",
    "damping_ratio",
)
_TRACE_SCALARS = ("contact_start_s", "contact_end_s", "iteration")
_PALETTE = ("#1967b3", "#b55214", "#16806a", "#8054a0", "#b31b69", "#5c7a1f", "#0f7a8f", "#8a5a2b")
_METRICS = (
    ("Policy entropy", "nats", "ent", False),
    ("Explained variance", "0-1", "ev", False),
    ("Value loss", "log scale", "v", True),
    ("Approximate KL divergence", "nats", "kl", False),
    ("Clip fraction", "0-1", "clip", False),
    ("Policy surrogate loss", "dimensionless", "pi", False),
)


def _as_float(value) -> float:
    """Convert a parsed field to float, mapping anything unusable to ``nan``.

    Args:
        value: Parsed value, which may be ``None`` or a trace path string.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class Run:
    """One training log and the per-iteration series parsed out of it."""

    name: str
    path: Path
    samples: list[dict[str, float]] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    trace: dict[str, np.ndarray] | None = None
    frozen: str | None = None
    frozen_metrics: dict[str, float] | None = None
    modified_s: float = 0.0
    age_s: float = 0.0

    @property
    def iteration_count(self) -> int:
        """Number of parsed training iterations."""
        return len(self.samples)

    @property
    def status(self) -> str:
        """Report whether the run is finished, still writing, or stopped."""
        if self.frozen is not None:
            return "finished"
        if self.age_s <= RUNNING_WINDOW_S:
            return "running"
        return "stopped"

    def channel(self, key: str) -> np.ndarray:
        """Return one parsed channel as a float array.

        Args:
            key: Channel name, one of the parsed per-iteration fields.
        """
        return np.asarray([sample[key] for sample in self.samples], dtype=float)

    def final(self, key: str) -> float | None:
        """Return the last value of a channel, or ``None`` when the run is empty.

        Args:
            key: Channel name, one of the parsed per-iteration fields.
        """
        return float(self.samples[-1][key]) if self.samples else None

    def best_return(self) -> float | None:
        """Return the highest per-iteration best return seen so far."""
        return max((float(sample["best"]) for sample in self.samples), default=None)

    def eval_channel(self, key: str) -> np.ndarray:
        """Return one deterministic-evaluation channel as a float array.

        Missing fields, such as the physical block of an older record, read as ``nan``
        and are dropped at render time.

        Args:
            key: Channel name, one of the parsed evaluation fields.
        """
        return np.asarray([_as_float(record.get(key)) for record in self.evaluations], dtype=float)

    def eval_final(self, key: str) -> float | None:
        """Return the latest finite value of an evaluation channel, if any.

        Args:
            key: Channel name, one of the parsed evaluation fields.
        """
        for record in reversed(self.evaluations):
            value = _as_float(record.get(key))
            if math.isfinite(value):
                return value
        return None

    @property
    def latest_evaluation(self) -> dict | None:
        """Most recent deterministic-evaluation record, or ``None`` when absent."""
        return self.evaluations[-1] if self.evaluations else None

    def latest_excursion_total(self) -> float | None:
        """Sum of the three latest tolerance excursions; ``0`` means on task."""
        record = self.latest_evaluation
        if record is None:
            return None
        return float(sum(record[key] for _, key, _ in _EXCURSIONS))

    @property
    def materials(self) -> list[str | None]:
        """Material tokens seen in this log, in first-appearance order.

        More than one token means the log holds a material sweep under one policy.
        """
        seen: dict[str | None, None] = {}
        for record in self.evaluations:
            seen.setdefault(record.get("artifact"), None)
        return list(seen)

    @property
    def material(self) -> str | None:
        """Material token of the latest evaluation record, if the log carries one."""
        record = self.latest_evaluation
        return record.get("artifact") if record is not None else None

    @property
    def task_status(self) -> str:
        """Report ``on task``, ``off task`` or ``no eval`` from the latest record."""
        record = self.latest_evaluation
        if record is None:
            return "no eval"
        return "on task" if record["on_task"] >= 0.5 else "off task"


def parse_iteration(line: str) -> dict[str, float] | None:
    """Parse one training iteration line.

    Args:
        line: Raw log line, which may be noise or a partially written record.

    Returns:
        The parsed channels, or ``None`` when the line is not a complete record.
    """
    match = _ITERATION_RE.match(line.strip())
    if match is None:
        return None
    try:
        values = {key: float(match.group(key)) for key in _CHANNELS}
    except ValueError:  # pragma: no cover - the pattern already restricts the text
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def parse_frozen(line: str) -> dict[str, float] | None:
    """Parse the final frozen-evaluation line of a finished run.

    Args:
        line: Raw log line.

    Returns:
        The evaluation statistics, or ``None`` when the line does not match.
    """
    match = _FROZEN_RE.match(line.strip())
    if match is None:
        return None
    try:
        values = {key: float(match.group(key)) for key in ("mean", "median", "best", "worst")}
    except ValueError:  # pragma: no cover - the pattern already restricts the text
        return None
    values["episodes"] = float(match.group("episodes"))
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def parse_evaluation(line: str) -> dict | None:
    """Parse one periodic deterministic-evaluation record.

    The record is space-separated ``key=value`` text. An excursion of zero means
    the matching tolerance is satisfied, and ``objective_j`` may be ``nan``, which
    is kept as a nonfinite value and dropped at render time. The physical block and
    the ``trace`` path are optional as a whole, so an older short record still
    parses and a half-written long record loses only its truncated tail. The
    ``trace`` path and the ``artifact`` material token are optional text fields, so a
    log written before the material token existed still parses unchanged.

    Args:
        line: Raw log line, which may be noise or a partially written record.

    Returns:
        The parsed evaluation fields, or ``None`` when the line is not a usable record.
    """
    tokens = line.strip().split()
    if len(tokens) < 2 or tokens[0] != "eval":
        return None
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if not separator or not key or key in fields:
            return None
        fields[key] = value
    if not all(key in fields for key in _EVAL_REQUIRED):
        return None
    if not fields["iteration"].isdigit() or any(fields[key] not in ("0", "1") for key in _EVAL_FLAGS):
        return None
    record: dict = {}
    try:
        for key in _EVAL_REQUIRED:
            record[key] = float(fields[key])
    except ValueError:
        return None
    if not math.isfinite(record["iteration"]):  # pragma: no cover - isdigit already excludes this
        return None
    # The physical block is written as one unit, so a partial block means a truncated tail.
    if all(key in fields for key in _EVAL_PHYSICAL):
        try:
            physical = {key: float(fields[key]) for key in _EVAL_PHYSICAL}
        except ValueError:
            return record
        record.update(physical)
    for key in ("trace", "artifact"):
        if fields.get(key):
            record[key] = fields[key]
    return record


class ParsedLog(NamedTuple):
    """Everything read out of one training log."""

    samples: list[dict[str, float]]
    evaluations: list[dict]
    frozen: str | None
    frozen_metrics: dict[str, float] | None


def parse_log(text: str) -> ParsedLog:
    """Parse a whole log, ignoring Warp noise, tracebacks and truncated lines.

    Args:
        text: Full text of one training log.

    Returns:
        The per-iteration samples, evaluation records, raw frozen line and its statistics.
    """
    samples: list[dict[str, float]] = []
    evaluations: list[dict[str, float]] = []
    frozen: str | None = None
    frozen_metrics: dict[str, float] | None = None
    for line in text.splitlines():
        sample = parse_iteration(line)
        if sample is not None:
            samples.append(sample)
            continue
        record = parse_evaluation(line)
        if record is not None:
            evaluations.append(record)
            continue
        metrics = parse_frozen(line)
        if metrics is not None:
            frozen, frozen_metrics = line.strip(), metrics
    samples.sort(key=lambda sample: sample["iteration"])
    evaluations.sort(key=lambda record: record["iteration"])
    return ParsedLog(samples, evaluations, frozen, frozen_metrics)


def load_trace(reference: str | None, directory: Path | str) -> dict[str, np.ndarray] | None:
    """Load the latest evaluation waveform archive of a run.

    The trainer overwrites the archive while the dashboard reads it, so every
    failure, including a truncated or half-written file, degrades to ``None``.

    Args:
        reference: Path written in the ``trace=`` field, absolute or relative.
        directory: Log directory, searched when the written path does not exist.

    Returns:
        The trace arrays and scalars, or ``None`` when no usable archive was read.
    """
    if not reference:
        return None
    candidates = [Path(reference)]
    if not candidates[0].is_absolute():
        candidates.append(Path(directory) / Path(reference).name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    trace: dict[str, np.ndarray] = {}
    try:
        # Open the file here so a failed load never leaks a handle on a half-written archive.
        with path.open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
            names = set(archive.files)
            if "time_s" not in names:
                return None
            for key in (*_TRACE_ARRAYS, *_TRACE_SCALARS):
                if key not in names:
                    continue
                value = np.asarray(archive[key], dtype=float).reshape(-1)
                if value.size:
                    trace[key] = value
            # The archive also carries the material token, so a stale file can be spotted.
            if "artifact" in names:
                token = np.asarray(archive["artifact"]).reshape(-1)
                if token.size:
                    trace["artifact"] = str(token[0])
    except Exception:
        return None
    time_s = trace.get("time_s")
    if time_s is None or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        return None
    for key in list(trace):
        if key in _TRACE_ARRAYS and trace[key].size != time_s.size:
            del trace[key]
    return trace if "time_s" in trace else None


def trace_percent(trace: dict[str, np.ndarray]) -> tuple[np.ndarray, bool]:
    """Map trace time onto per cent of the contact window.

    Args:
        trace: Loaded trace arrays.

    Returns:
        The per-cent abscissa and whether a valid contact window was used.
    """
    time_s = trace["time_s"]
    start = float(trace["contact_start_s"][0]) if "contact_start_s" in trace else float("nan")
    end = float(trace["contact_end_s"][0]) if "contact_end_s" in trace else float("nan")
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        start, end = float(time_s[0]), float(time_s[-1])
        if end <= start:
            return np.linspace(0.0, 100.0, time_s.size), False
        return (time_s - start) / (end - start) * 100.0, False
    return (time_s - start) / (end - start) * 100.0, True


def reference_contact_ms(trace: dict[str, np.ndarray] | None) -> float | None:
    """Estimate the measured contact duration from the reference vertical force.

    The evaluation record has no measured contact-duration field, so the duration is
    measured from the reference force waveform above :data:`_CONTACT_THRESHOLD_N`.

    Args:
        trace: Loaded trace arrays, or ``None``.
    """
    if trace is None or "reference_fz_n" not in trace:
        return None
    time_s, force = trace["time_s"], trace["reference_fz_n"]
    loaded = np.isfinite(force) & (force > _CONTACT_THRESHOLD_N)
    if not np.any(loaded):
        return None
    index = np.flatnonzero(loaded)
    duration = float(time_s[index[-1]] - time_s[index[0]]) * 1000.0
    return duration if math.isfinite(duration) and duration > 0.0 else None


def load_runs(directory: Path | str, now: float | None = None) -> list[Run]:
    """Load every training log in a directory.

    Args:
        directory: Directory holding ``train_*.log`` files.
        now: Wall-clock time used to age the logs; defaults to the current time.

    Returns:
        One :class:`Run` per readable log, sorted by name.
    """
    directory = Path(directory)
    now = time.time() if now is None else now
    runs: list[Run] = []
    for path in sorted(directory.glob(LOG_PATTERN)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            modified = path.stat().st_mtime
        except OSError:
            continue
        parsed = parse_log(text)
        name = path.stem
        name = name[len("train_") :] if name.startswith("train_") and len(name) > len("train_") else name
        runs.append(
            Run(
                name=name,
                path=path,
                samples=parsed.samples,
                evaluations=parsed.evaluations,
                trace=load_trace(parsed.evaluations[-1].get("trace") if parsed.evaluations else None, directory),
                frozen=parsed.frozen,
                frozen_metrics=parsed.frozen_metrics,
                modified_s=modified,
                age_s=max(now - modified, 0.0),
            )
        )
    return runs


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value, digits: int = 6) -> str:
    if value is None:
        return "\u2014"
    value = float(value)
    if not math.isfinite(value):
        return "\u2014"
    return f"{value:.{digits}g}"


def _lighten(color: str, amount: float = 0.55) -> str:
    """Blend a hex colour toward white so overlaid context lines stay readable.

    Args:
        color: Colour as ``#rrggbb``.
        amount: Fraction of white to mix in, in ``[0, 1]``.
    """
    channels = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    blended = (round(value + (255 - value) * amount) for value in channels)
    return "#" + "".join(f"{value:02x}" for value in blended)


def downsample(x: np.ndarray, y: np.ndarray, limit: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Thin a long series by a fixed stride, always keeping the last sample.

    Args:
        x: Sample abscissae.
        y: Sample values.
        limit: Approximate number of points to keep.
    """
    count = len(x)
    if count <= limit or limit < 1:
        return x, y
    stride = -(-count // limit)
    index = np.arange(0, count, stride)
    if index[-1] != count - 1:
        index = np.append(index, count - 1)
    return x[index], y[index]


def _log_values(y: np.ndarray) -> np.ndarray:
    """Map values onto a base-10 log axis, dropping zero and negative samples.

    Args:
        y: Raw values, which may contain zero, negative or nonfinite entries.
    """
    y = np.asarray(y, dtype=float)
    safe = np.where(np.isfinite(y) & (y > 0.0), y, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(safe)


def _bounds(values: np.ndarray) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    if not math.isfinite(low) or not math.isfinite(high):
        return 0.0, 1.0
    margin = max((high - low) * 0.08, abs(high) * 0.005, 1.0e-6)
    return low - margin, high + margin


def _dash(dashed: bool | str) -> str:
    """Return the SVG dash attribute for a series style.

    Args:
        dashed: ``True`` for the default dash, an SVG dash array, or ``False`` for a solid line.
    """
    if not dashed:
        return ""
    pattern = dashed if isinstance(dashed, str) else "6 4"
    return f' stroke-dasharray="{pattern}"'


def plot(
    title: str,
    unit: str,
    series: list[tuple],
    log: bool = False,
    xlabel: str = "Iteration",
    reference: tuple[float, str] | None = None,
    markers: tuple[tuple[float, str], ...] = (),
    points: int = MAX_POINTS,
) -> str:
    """Draw one metric as an accessible inline SVG with labelled axes.

    Args:
        title: Chart heading.
        unit: Unit shown next to the heading.
        series: Tuples of ``(label, x, y, color, dashed)`` with an optional sixth
            ``marker`` flag, where ``dashed`` is a bool or an SVG dash array and
            ``marker`` draws the samples themselves.
        log: Draw the vertical axis on a base-10 log scale.
        xlabel: Label of the horizontal axis.
        reference: Value and label of a horizontal reference line, drawn on linear axes only.
        markers: Values and labels of vertical marker lines on the horizontal axis.
        points: Downsampling budget of each series.

    Returns:
        An HTML section, or an explanatory section when nothing is plottable.
    """
    prepared = []
    for label, raw_x, raw_y, color, dashed, *rest in series:
        values = _log_values(raw_y) if log else np.asarray(raw_y, dtype=float)
        abscissa = np.asarray(raw_x, dtype=float)
        if len(abscissa) == 0 or not np.any(np.isfinite(values)):
            continue
        prepared.append((label, *downsample(abscissa, values, points), color, dashed, bool(rest and rest[0])))
    if not prepared:
        reason = "No positive samples are available for this log-scale metric." if log else "No samples yet."
        return (
            f'<section class="plot"><h3>{_escape(title)} <small>[{_escape(unit)}]</small></h3>'
            f'<p class="subtitle">{_escape(reason)}</p></section>'
        )
    finite = np.concatenate([y[np.isfinite(y)] for _, _, y, _, _, _ in prepared])
    if not log and reference is not None and math.isfinite(reference[0]):
        # Keep the reference line inside the drawn range; comparing against it is the point.
        finite = np.append(finite, reference[0])
    low, high = _bounds(finite)
    xmin = min(float(np.min(x)) for _, x, _, _, _, _ in prepared)
    xmax = max(float(np.max(x)) for _, x, _, _, _, _ in prepared)
    if not math.isfinite(xmin) or not math.isfinite(xmax) or xmax <= xmin:
        xmin, xmax = xmin if math.isfinite(xmin) else 0.0, (xmin if math.isfinite(xmin) else 0.0) + 1.0
    span = high - low if high > low else 1.0
    width, height, left, top, plot_width, plot_height = 640, 285, 76, 18, 548, 217
    content = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}"><title>{_escape(title)}</title>'
    ]
    for fraction in np.linspace(0.0, 1.0, 5):
        sx, sy = left + fraction * plot_width, top + (1.0 - fraction) * plot_height
        tick = low + fraction * span
        label = f"1e{tick:+.1f}" if log else _number(tick, 4)
        content.append(f'<path d="M{left},{sy:.2f}h{plot_width}" class="grid"/>')
        content.append(f'<path d="M{sx:.2f},{top}v{plot_height}" class="grid"/>')
        content.append(f'<text x="{left - 8}" y="{sy + 4:.2f}" text-anchor="end">{_escape(label)}</text>')
        content.append(
            f'<text x="{sx:.2f}" y="{top + plot_height + 20}" text-anchor="middle">'
            f"{_number(xmin + fraction * (xmax - xmin), 4)}</text>"
        )
    for value, label in markers:
        if not math.isfinite(value) or not xmin <= value <= xmax:
            continue
        position = left + (value - xmin) / (xmax - xmin) * plot_width
        content.append(f'<path d="M{position:.2f},{top}v{plot_height}" stroke="#8a5a2b" stroke-dasharray="4 3"/>')
        content.append(f'<text x="{position + 3:.2f}" y="{top + 11}" fill="#8a5a2b">{_escape(label)}</text>')
    if not log and reference is not None and low <= reference[0] <= high:
        line = top + (high - reference[0]) / span * plot_height
        content.append(f'<path d="M{left},{line:.2f}h{plot_width}" stroke="#14663f" stroke-width="2"/>')
        content.append(
            f'<text x="{left + plot_width - 4}" y="{line - 6:.2f}" text-anchor="end" '
            f'fill="#14663f">{_escape(reference[1])}</text>'
        )
    elif not log and low <= 0.0 <= high:
        zero = top + high / span * plot_height
        content.append(f'<path d="M{left},{zero:.2f}h{plot_width}" stroke="#99a7b9" stroke-dasharray="3 4"/>')
    for label, x, y, color, dashed, marker in prepared:
        sx = left + (x - xmin) / (xmax - xmin) * plot_width
        sy = top + (high - y) / span * plot_height
        # Gaps mean the metric was unavailable there; keep every remaining sample.
        pieces = []
        active = False
        for px, py in zip(sx, sy, strict=True):
            if math.isfinite(px) and math.isfinite(py):
                pieces.append(f"{'L' if active else 'M'}{px:.1f},{py:.1f}")
                active = True
            else:
                active = False
        if not pieces:
            continue
        dash = _dash(dashed)
        content.append(
            f'<path d="{" ".join(pieces)}" fill="none" stroke="{color}" stroke-width="2"{dash}>'
            f"<title>{_escape(label)}</title></path>"
        )
        if marker and len(sx) <= MARKER_LIMIT:
            # The samples themselves, so a coarse series cannot be read as a resolved waveform.
            dots = "".join(
                f"M{px:.2f},{py:.2f}m-2.2,0a2.2,2.2 0 1,0 4.4,0a2.2,2.2 0 1,0 -4.4,0"
                for px, py in zip(sx, sy, strict=True)
                if math.isfinite(px) and math.isfinite(py)
            )
            content.append(
                f'<path d="{dots}" fill="{color}" stroke="none"><title>{_escape(label)} samples</title></path>'
            )
    content.append(f'<text x="{left + plot_width / 2}" y="277" text-anchor="middle">{_escape(xlabel)}</text></svg>')
    legend = " ".join(
        f'<span><i style="background:{_escape(color)}"></i>{_escape(label)}{" (dashed)" if dashed else ""}</span>'
        for label, _, _, color, dashed, _ in prepared
    )
    return (
        f'<section class="plot"><h3>{_escape(title)} <small>[{_escape(unit)}]</small></h3>'
        f'{"".join(content)}<div class="legend">{legend}</div></section>'
    )


class EvalGroup(NamedTuple):
    """One run's evaluation records for a single material token."""

    run: Run
    token: str | None
    records: list[dict]
    color: str
    label: str

    def channel(self, key: str) -> np.ndarray:
        """Return one evaluation channel of this group as a float array.

        Args:
            key: Channel name, one of the parsed evaluation fields.
        """
        return np.asarray([_as_float(record.get(key)) for record in self.records], dtype=float)


def eval_groups(runs: list[Run]) -> list[EvalGroup]:
    """Split every run into one group per material token and assign chart colours.

    A run's identity is the policy and the material it ran on, so a log holding several
    tokens is a material sweep and is never collapsed into one curve.

    Args:
        runs: Runs to group, in legend order.
    """
    groups: list[EvalGroup] = []
    for run in runs:
        tokens = run.materials or [None]
        for token in tokens:
            records = [record for record in run.evaluations if record.get("artifact") == token]
            color = _PALETTE[len(groups) % len(_PALETTE)]
            label = run.name if token is None else f"{run.name} [{token}]"
            groups.append(EvalGroup(run, token, records, color, label))
    return groups


def run_colors(runs: list[Run]) -> dict[str, str]:
    """Map each run name to the colour of its first material group.

    Args:
        runs: Runs to colour, in legend order.
    """
    colors: dict[str, str] = {}
    for group in eval_groups(runs):
        colors.setdefault(group.run.name, group.color)
    return colors


def _material_swatches(run: Run, group_colors: dict[str, str]) -> str:
    """Render the material tokens of a run with the colour used in the charts.

    Args:
        run: Run to describe.
        group_colors: Colour per group label from :func:`eval_groups`.
    """
    tokens = run.materials
    if not tokens or tokens == [None]:
        return "\u2014"
    items = []
    for token in tokens:
        label = run.name if token is None else f"{run.name} [{token}]"
        text = "no artifact token" if token is None else token
        items.append(
            f'<span class="material"><i class="swatch" style="background:{group_colors.get(label, "#737e88")}"></i>'
            f"<code>{_escape(text)}</code></span>"
        )
    sweep = f'<span class="sweep">sweep: {len(tokens)} materials</span> ' if len(tokens) > 1 else ""
    return sweep + " ".join(items)


def _waveform_label(run: Run) -> str:
    """Name a waveform curve by run, and by material when a token is known.

    Args:
        run: Run being drawn.
    """
    token = run.material or (str(run.trace["artifact"]) if run.trace and "artifact" in run.trace else None)
    return run.name if token is None else f"{run.name} [{token}]"


def _waveform_iteration(run: Run) -> int | None:
    """Report the evaluation iteration the archive of a run came from.

    Args:
        run: Run being drawn.
    """
    if run.trace is not None and "iteration" in run.trace:
        return int(run.trace["iteration"][0])
    record = run.latest_evaluation
    return int(record["iteration"]) if record is not None else None


def _reference_disagreement(drawn: list[tuple[Run, np.ndarray]], channel: str, coarse: dict[str, bool]) -> list[str]:
    """List runs whose measured reference differs from the first drawn run's.

    The measurement is one capture, so the curves should be identical; a difference means
    two runs were scored against different references and must not be compared silently.
    Coarse archives are skipped, because a sparse trace misses peaks and would report a
    resolution difference as a measurement difference.

    Args:
        drawn: Runs and their per-cent-of-contact abscissae.
        channel: Reference channel name.
        coarse: Whether each run's archive is too sparse to compare.
    """
    grid = np.linspace(0.0, 100.0, 200)
    baseline, names = None, []
    for run, percent in drawn:
        if channel not in run.trace or coarse.get(run.name):
            continue
        values = np.interp(grid, percent, run.trace[channel], left=np.nan, right=np.nan)
        if baseline is None:
            baseline = values
            continue
        both = np.isfinite(values) & np.isfinite(baseline)
        scale = max(float(np.max(np.abs(baseline[both]))) if np.any(both) else 0.0, 1.0e-9)
        if np.any(both) and float(np.max(np.abs(values[both] - baseline[both]))) > REFERENCE_TOLERANCE * scale:
            names.append(run.name)
    return names


def _waveform_section(runs: list[Run], colors: dict[str, str]) -> str:
    """Overlay the latest evaluation waveforms of every selected run on shared axes.

    Args:
        runs: Selected runs, in legend order.
        colors: Colour per run from :func:`run_colors`, so a colour means one run everywhere.
    """
    drawn = [(run, trace_percent(run.trace)) for run in runs if run.trace is not None]
    if not drawn:
        return (
            '<article><h2>Waveforms</h2><p class="subtitle">No waveforms for the current selection. '
            "A run joins this overlay once its evaluation writes a readable trace archive.</p></article>"
        )
    percents = [(run, percent) for run, (percent, _) in drawn]
    unwindowed = [run.name for run, (_, windowed) in drawn if not windowed]
    richest = max(run.trace["time_s"].size for run, _ in drawn)
    coarse = {run.name: run.trace["time_s"].size * COARSE_SAMPLE_RATIO < richest for run, _ in drawn}
    plots, disagreements = [], []
    for title, unit, scale, channel, reference in _WAVEFORMS:
        series = []
        for run, percent in percents:
            if channel not in run.trace:
                continue
            thin = coarse[run.name]
            series.append(
                (
                    _waveform_label(run),
                    percent,
                    run.trace[channel] * scale,
                    colors[run.name],
                    "2 3" if thin else False,
                    thin,
                )
            )
            if reference is not None and reference[0] == "per_run" and reference[1] in run.trace:
                series.append(
                    (
                        f"{_waveform_label(run)} {reference[2]}",
                        percent,
                        run.trace[reference[1]] * scale,
                        _lighten(colors[run.name]),
                        "6 4",
                        thin,
                    )
                )
        if reference is not None and reference[0] == "shared":
            # One measured curve for the whole overlay: the measurement is the same capture.
            source = next((item for item in percents if reference[1] in item[0].trace), None)
            if source is not None:
                series.append((reference[2], source[1], source[0].trace[reference[1]] * scale, "#737e88", "6 4"))
            differing = _reference_disagreement(percents, reference[1], coarse)
            if differing:
                disagreements.append(f"{title}: {', '.join(differing)}")
        plots.append(
            plot(
                title,
                unit,
                series,
                xlabel="Contact [%]",
                markers=((0.0, "contact start"), (100.0, "contact end")),
                points=WAVEFORM_POINTS,
            )
        )
    stamps = []
    for run, _ in drawn:
        iteration = _waveform_iteration(run)
        stamps.append(
            f"{_waveform_label(run)} at iteration {iteration}" if iteration is not None else _waveform_label(run)
        )
    caption = (
        f"Latest evaluation of each selected run, overlaid: {'; '.join(stamps)}. "
        f"The horizontal axis is per cent of the contact window the trainer recorded, which starts and ends "
        f"where the shoe carries CONTACT_FORCE_FRACTION = {CONTACT_FORCE_FRACTION} of body weight, so runs "
        f"with different contact timing stay comparable. The measured reference is drawn once for the whole "
        f"overlay. A run sampled more than {COARSE_SAMPLE_RATIO} times coarser than the richest archive here "
        f"is dashed with its samples marked. Each curve here keeps at most {WAVEFORM_POINTS} points."
    )
    if unwindowed:
        caption += f" No contact window in the archive of: {', '.join(unwindowed)}; those axes span the whole trace."
    warning = ""
    if disagreements:
        warning = (
            f'<p class="warning">Selected runs carry different measured references, so the drawn reference '
            f"describes only the first run. Differing: {_escape('; '.join(disagreements))}.</p>"
        )
    stale = [
        f"{run.name} logs {run.material} but its archive carries {run.trace['artifact']}"
        for run, _ in drawn
        if run.material is not None and "artifact" in run.trace and str(run.trace["artifact"]) != run.material
    ]
    if stale:
        warning += (
            f'<p class="warning">Archive material mismatch, waveforms may be stale: {_escape("; ".join(stale))}.</p>'
        )
    return (
        f"<article><h2>Waveforms: {len(drawn)} run{'s' if len(drawn) != 1 else ''} overlaid</h2>"
        f'<p class="subtitle">{_escape(caption)}</p>{warning}'
        f'<div class="panels">{"".join(plots)}</div></article>'
    )


def _physical_reference(runs: list[Run], reference) -> tuple[float, str] | None:
    """Resolve the horizontal reference line of one physical metric chart.

    Args:
        runs: Runs shown on the chart.
        reference: Reference specification from :data:`_PHYSICAL_METRICS`.
    """
    if reference is None:
        return None
    if reference[0] == "zero":
        return 0.0, reference[1]
    if reference[0] == "record":
        values = [run.eval_final(reference[1]) for run in runs]
    else:
        values = [reference_contact_ms(run.trace) for run in runs]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite)), reference[-1]


def _physical_plots(groups: list[EvalGroup], runs: list[Run]) -> str:
    """Overlay the physical evaluation metrics of every material group against iteration.

    Args:
        groups: Evaluation groups from :func:`eval_groups`; empty groups are absent.
        runs: Runs behind the groups, used to place the measured reference lines.
    """
    plots = []
    for title, unit, key, reference in _PHYSICAL_METRICS:
        series = [
            (group.label, group.channel("iteration"), group.channel(key), group.color, False)
            for group in groups
            if group.records
        ]
        plots.append(plot(title, unit, series, reference=_physical_reference(runs, reference)))
    return "".join(plots)


def _excursion_plot(groups: list[EvalGroup]) -> str:
    """Overlay the three tolerance excursions of every material group against iteration.

    Args:
        groups: Evaluation groups from :func:`eval_groups`; empty groups are absent.
    """
    series = []
    for group in groups:
        if not group.records:
            continue
        iterations = group.channel("iteration")
        for label, key, dashed in _EXCURSIONS:
            series.append((f"{group.label} {label}", iterations, group.channel(key), group.color, dashed))
    return plot(
        "Task excursions",
        "tolerance units, 0 is satisfied",
        series,
        reference=(0.0, "on task"),
    )


def _objective_plot(groups: list[EvalGroup]) -> str:
    """Overlay the deterministic work objective of every material group against iteration.

    Args:
        groups: Evaluation groups from :func:`eval_groups`; empty groups are absent.
    """
    series = [
        (group.label, group.channel("iteration"), group.channel("objective_j"), group.color, False)
        for group in groups
        if group.records
    ]
    return plot("Work objective", "J", series)


def _return_plot(runs: list[Run], colors: dict[str, str]) -> str:
    series = []
    for run in runs:
        if not run.samples:
            continue
        color = colors[run.name]
        iterations = run.channel("iteration")
        series.append((f"{run.name} mean", iterations, run.channel("mean"), color, False))
        series.append((f"{run.name} best", iterations, run.channel("best"), _lighten(color), True))
    return plot("Episode return", "return", series)


def _training_plots(runs: list[Run], colors: dict[str, str]) -> str:
    """Overlay the training-health metrics of every run against iteration.

    Training health is per policy, so these curves stay one per run even when the log
    holds several materials.

    Args:
        runs: Runs to overlay.
        colors: Colour per run from :func:`run_colors`.
    """
    plots = [_return_plot(runs, colors)]
    for title, unit, key, log in _METRICS:
        series = [
            (run.name, run.channel("iteration"), run.channel(key), colors[run.name], False)
            for run in runs
            if run.samples
        ]
        plots.append(plot(title, unit, series, log=log))
    return "".join(plots)


def _task_badge(run: Run) -> str:
    """Render the on-task badge of a run from its latest evaluation record.

    Args:
        run: Run to describe.
    """
    status = run.task_status
    css = {"on task": "ontask", "off task": "offtask"}.get(status, "noeval")
    text = {"on task": "ON TASK", "off task": "OFF TASK"}.get(status, "no eval")
    return f'<span class="task {css}">{text}</span>'


def _peak_force_cell(run: Run) -> str:
    """Format the latest peak vertical force and its per-cent error against the measurement.

    Args:
        run: Run to describe.
    """
    peak, measured = run.eval_final("peak_fz_n"), run.eval_final("peak_fz_ref_n")
    if peak is None:
        return "\u2014"
    if measured is None or measured == 0.0:
        return f"{_number(peak, 5)} N"
    return f"{_number(peak, 5)} N ({_number((peak - measured) / abs(measured) * 100.0, 3)} %)"


def _summary_table(runs: list[Run], colors: dict[str, str], group_colors: dict[str, str]) -> str:
    rows = []
    for run in runs:
        frozen = _escape(run.frozen) if run.frozen else "\u2014"
        record = run.latest_evaluation
        objective = _number(record["objective_j"], 4) if record is not None else "\u2014"
        excursion = _number(run.latest_excursion_total(), 4)
        peak_time, peak_time_ref = run.eval_final("peak_time_pct"), run.eval_final("peak_time_ref_pct")
        timing = _number(peak_time - peak_time_ref, 3) if None not in (peak_time, peak_time_ref) else "\u2014"
        contact, contact_ref = run.eval_final("contact_ms"), reference_contact_ms(run.trace)
        contact_error = _number(contact - contact_ref, 3) if None not in (contact, contact_ref) else "\u2014"
        rows.append(
            f'<tr><th scope="row"><i class="swatch" style="background:{colors[run.name]}"></i>{_escape(run.name)}</th>'
            f"<td>{_material_swatches(run, group_colors)}</td>"
            f"<td>{run.iteration_count}</td><td>{_task_badge(run)}</td>"
            f"<td>{_peak_force_cell(run)}</td><td>{timing}</td><td>{contact_error}</td>"
            f"<td>{objective}</td><td>{excursion}</td>"
            f"<td>{_number(run.final('mean'))}</td>"
            f"<td>{_number(run.best_return())}</td><td>{_number(run.final('ent'), 4)}</td>"
            f"<td>{_number(run.final('ev'), 4)}</td><td>{_number(run.final('v'), 4)}</td>"
            f'<td><span class="status {run.status}">{run.status}</span></td>'
            f"<td>{_number(run.age_s, 3)}</td><td><code>{frozen}</code></td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Run</th><th>Shoe material</th><th>Iterations</th><th>Task</th>'
        "<th>Peak force (error vs measured)</th><th>Peak-time error [pp]</th><th>Contact error [ms]</th>"
        "<th>Latest objective [J]</th><th>Latest total excursion</th><th>Final return</th>"
        "<th>Best return</th><th>Final entropy</th><th>Final explained variance</th><th>Final value loss</th>"
        "<th>Status</th><th>Log age [s]</th><th>Frozen evaluation</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _run_legend(runs: list[Run], colors: dict[str, str]) -> str:
    items = []
    for run in runs:
        material = "" if run.material is None else f" · material <code>{_escape(run.material)}</code>"
        sweep = " · material sweep" if len(run.materials) > 1 else ""
        items.append(
            f'<li><i class="swatch" style="background:{colors[run.name]}"></i><strong>{_escape(run.name)}</strong> · '
            f"{run.iteration_count} iters · latest {_number(run.final('mean'))} · "
            f"best {_number(run.best_return())} · "
            f'<span class="status {run.status}">{run.status}</span> {_task_badge(run)}{material}{sweep}</li>'
        )
    return f'<ul class="runs">{"".join(items)}</ul>'


class Selection(NamedTuple):
    """Requested run names and material tokens from the page query string."""

    runs: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        """Whether the query asked for a subset at all."""
        return bool(self.runs or self.materials)

    def query(self) -> str:
        """Rebuild the canonical query string of this selection."""
        pairs = [("run", name) for name in self.runs] + [("material", token) for token in self.materials]
        return urlencode(pairs)


def parse_selection(query: str) -> Selection:
    """Read the run and material filters out of a query string.

    Unknown keys and repeated values are tolerated: only ``run`` and ``material``
    are read, and each keeps its first-seen order.

    Args:
        query: Raw query string, with or without a leading ``?``.
    """
    try:
        pairs = parse_qsl(query.lstrip("?"), keep_blank_values=False)
    except ValueError:  # pragma: no cover - parse_qsl only raises in strict mode
        return Selection()
    chosen: dict[str, dict[str, None]] = {"run": {}, "material": {}}
    for key, value in pairs:
        if key in chosen and value:
            chosen[key].setdefault(value, None)
    return Selection(tuple(chosen["run"]), tuple(chosen["material"]))


def _token_key(token: str | None) -> str:
    return UNTAGGED if token is None else token


def apply_selection(runs: list[Run], selection: Selection) -> list[Run]:
    """Keep only the runs, and the evaluation records, that the selection asks for.

    Names that match nothing are ignored, so a stale bookmark renders an empty
    selection instead of an error.

    Args:
        runs: Every run discovered in the directory.
        selection: Parsed run and material filters.
    """
    visible: list[Run] = []
    for run in runs:
        if selection.runs and run.name not in selection.runs:
            continue
        shown = run
        if selection.materials:
            kept = [record for record in run.evaluations if _token_key(record.get("artifact")) in selection.materials]
            if not kept:
                continue
            if len(kept) != len(run.evaluations):
                # The archive belongs to the latest record, so drop it when that record is filtered out.
                trace = run.trace if kept[-1] is run.evaluations[-1] else None
                shown = replace(run, evaluations=kept, trace=trace)
        visible.append(shown)
    return visible


def refresh_url(selection: Selection) -> str:
    """Build the auto-refresh target that keeps the current selection.

    Args:
        selection: Parsed run and material filters.
    """
    query = selection.query()
    return f"/?{query}" if query else "/"


def _filter_form(runs: list[Run], selection: Selection) -> str:
    """Render the no-JavaScript selection form for runs and materials.

    Args:
        runs: Every run discovered in the directory.
        selection: Parsed run and material filters.
    """
    boxes = []
    for run in runs:
        checked = " checked" if run.name in selection.runs else ""
        boxes.append(
            f'<label><input type="checkbox" name="run" value="{_escape(run.name)}"{checked}>{_escape(run.name)}</label>'
        )
    tokens: dict[str, None] = {}
    for run in runs:
        for token in run.materials:
            tokens.setdefault(_token_key(token), None)
    materials = ""
    if tokens and tuple(tokens) != (UNTAGGED,):
        chips = []
        for token in tokens:
            checked = " checked" if token in selection.materials else ""
            text = "no material token" if token == UNTAGGED else token
            chips.append(
                f'<label><input type="checkbox" name="material" value="{_escape(token)}"{checked}>'
                f"{_escape(text)}</label>"
            )
        materials = f"<fieldset><legend>Materials</legend>{''.join(chips)}</fieldset>"
    return (
        '<form method="get" class="filter"><fieldset><legend>Runs</legend>'
        f"{''.join(boxes)}</fieldset>{materials}"
        '<p><button type="submit">Apply selection</button> '
        f'<a href="/">all runs</a> · <a href="/?run={NOTHING}">none</a>'
        " · An empty selection shows every run.</p></form>"
    )


def _selection_note(runs: list[Run], visible: list[Run], selection: Selection) -> str:
    """Say how much of the directory the page is showing.

    Args:
        runs: Every run discovered in the directory.
        visible: Runs that passed the selection.
        selection: Parsed run and material filters.
    """
    if not selection.active:
        return f'<p class="ok">Showing all {len(runs)} runs. Use the form above to compare a subset.</p>'
    hidden = [run.name for run in runs if run not in visible and all(run.name != shown.name for shown in visible)]
    materials = f" Material filter: {', '.join(selection.materials)}." if selection.materials else ""
    hidden_text = f" Hidden: {', '.join(hidden)}." if hidden else ""
    return (
        f'<p class="warning">Filtered view: showing {len(visible)} of {len(runs)} runs.{materials}'
        f"{hidden_text} This page is not the whole directory.</p>"
    )


def render(runs: list[Run], directory: Path | str, refresh: int = 10, now: float | None = None, query: str = "") -> str:
    """Render the whole dashboard page.

    Args:
        runs: Every run discovered in the directory, in legend order.
        directory: Directory the logs were read from, shown in the header.
        refresh: Auto-refresh period [s]; ``0`` disables the refresh tag.
        now: Wall-clock time used for the generated stamp.
        query: Query string selecting runs and materials; empty shows every run.

    Returns:
        A complete, self-contained HTML document.
    """
    now = time.time() if now is None else now
    selection = parse_selection(query)
    visible = apply_selection(runs, selection)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    # The refresh keeps the query string, or the selection would reset on every reload.
    refresh_tag = (
        f'<meta http-equiv="refresh" content="{int(refresh)}; url={_escape(refresh_url(selection))}">'
        if refresh and refresh > 0
        else ""
    )
    active = sum(1 for run in visible if run.status == "running")
    if visible:
        groups = eval_groups(visible)
        colors = run_colors(visible)
        group_colors = {group.label: group.color for group in groups}
        hidden = len(runs) - len(visible)
        note = f" {hidden} run rows are hidden by the current selection." if hidden else ""
        # Order: what the runs are, then what they did physically, then training health.
        body = (
            f"<article><h2>Runs</h2>{_summary_table(visible, colors, group_colors)}"
            f'<p class="subtitle">The table lists only the selected runs.{_escape(note)}</p>'
            f"{_run_legend(visible, colors)}</article>"
            f"{_waveform_section(visible, colors)}"
            f'<article><h2>Physical evaluation metrics</h2><div class="panels">'
            f"{_physical_plots(groups, visible)}</div></article>"
            f'<article><h2>Task excursions and work objective</h2><div class="panels">'
            f"{_excursion_plot(groups)}{_objective_plot(groups)}</div></article>"
            f'<article><h2>Training health</h2><div class="panels">{_training_plots(visible, colors)}</div></article>'
        )
    elif runs:
        body = (
            '<article><h2>Runs</h2><p class="warning">No runs match this selection, so nothing is drawn. '
            "Tick the runs you want in the form above, or follow the <b>all runs</b> link.</p></article>"
        )
    else:
        body = (
            f'<article><h2>Runs</h2><p class="warning">No <code>{_escape(LOG_PATTERN)}</code> logs were found in '
            f"<code>{_escape(directory)}</code>. Start a training run, or point the dashboard at another "
            f"directory with <code>--directory</code>. This page reloads by itself.</p></article>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh_tag}
<title>Impedance Instron training</title><style>
:root{{font-family:system-ui,sans-serif;color:#192d42;background:#f2f5f8;line-height:1.5}}
body{{max-width:1320px;margin:auto;padding:24px}}h1,h2,h3{{line-height:1.25}}h1{{margin-bottom:8px}}
a{{color:#125c9b}}header,article,.plot{{background:white;border:1px solid #dce4ec;border-radius:10px;padding:20px;margin-bottom:18px}}
.subtitle,small{{color:#536577}}.badge{{display:inline-block;padding:4px 10px;background:#e7edf4;border-radius:6px;margin-right:8px}}
.warning{{background:#fff0d5;border-left:4px solid #bd6b00;padding:12px}}.ok{{background:#e6f3eb;padding:12px}}
.panels{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.plot{{min-width:0;margin:0;padding:14px}}
.plot h3{{font-size:1rem;margin:0 0 8px}}svg{{width:100%;height:auto}}svg text{{font:11px system-ui;fill:#536577}}.grid{{stroke:#e5eaf0;fill:none}}
.legend{{font-size:.78rem;display:flex;flex-wrap:wrap;gap:4px 14px}}.legend i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
.runs{{list-style:none;padding:0;font-size:.88rem}}.runs li{{margin:6px 0}}
.swatch{{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;margin-right:7px}}
.status{{padding:2px 8px;border-radius:6px;font-size:.8rem}}.status.running{{background:#e6f3eb;color:#14663f}}
.status.finished{{background:#e7edf4;color:#2c4a68}}.status.stopped{{background:#fff0d5;color:#8a4d00}}
.task{{padding:3px 9px;border-radius:6px;font-size:.78rem;font-weight:700;letter-spacing:.04em;white-space:nowrap}}
.task.ontask{{background:#14663f;color:white}}.task.offtask{{background:#b31b1b;color:white}}
.task.noeval{{background:#e7edf4;color:#536577;font-weight:500}}
.material{{white-space:nowrap;margin-right:8px}}.sweep{{padding:2px 8px;border-radius:6px;background:#efe3f7;color:#5b2b7a;font-size:.78rem}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}th,td{{padding:8px 10px;border-bottom:1px solid #e5eaf0;text-align:left;vertical-align:top}}
td{{font-variant-numeric:tabular-nums}}tbody th{{font-weight:500}}thead{{background:#edf2f7}}code{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}
li{{margin:5px 0}}footer{{color:#536577;font-size:.85rem}}
.filter fieldset{{border:1px solid #dce4ec;border-radius:8px;margin:0 0 10px;padding:8px 12px}}
.filter legend{{font-size:.8rem;color:#536577}}.filter label{{display:inline-block;margin:3px 14px 3px 0;font-size:.88rem}}
.filter input{{margin-right:5px}}button{{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid #125c9b;background:#125c9b;color:white}}
@media(max-width:800px){{body{{padding:10px}}.panels{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Impedance Instron training</h1>
<p class="subtitle">Live overlay of every training log · inline SVG · no JavaScript, network or plotting package</p>
<span class="badge">Directory: {_escape(directory)}</span>
<span class="badge">Showing {len(visible)} of {len(runs)} runs</span>
<span class="badge">{active} running</span><span class="badge">Generated {_escape(stamp)}</span>
<span class="badge">Refresh: {int(refresh)} s</span></header>
<article><h2>Selection</h2>{_filter_form(runs, selection)}{_selection_note(runs, visible, selection)}</article>
{body}
<footer>The selection form is a plain GET form: it posts <code>?run=NAME</code> and
<code>?material=TOKEN</code>, repeated once per tick, and the two filters combine with AND. No query means
every run, so an old bookmark behaves as before. The auto-refresh target carries the same query string, so
a filtered view stays filtered. Names that match nothing are ignored. Logs are re-read on every request. A run counts as running when its log changed in the last
{int(RUNNING_WINDOW_S)} s, finished when it printed a <code>frozen:</code> line, and stopped otherwise.
Series longer than {MAX_POINTS} points are thinned by a fixed stride of ceil(n/{MAX_POINTS}), keeping the first and
last sample, so the SVG stays small; no smoothing or averaging is applied. Value loss uses a base-10 log
axis, and zero or negative samples are dropped from that chart because they have no log position.
Waveforms come from the compressed archive named by the <code>trace=</code> field of the latest
<code>eval</code> record, which the trainer overwrites on every evaluation; a missing, unreadable or
half-written archive simply shows no waveforms for that run. Waveform charts use per cent of the contact
window from <code>contact_start_s</code> to <code>contact_end_s</code>. The measured contact duration has no
field in the evaluation record, so it is measured from the reference force above
{int(_CONTACT_THRESHOLD_N)} N in that archive. Task excursions come from the <code>eval</code> records: each
excursion is 0 when that tolerance is satisfied, so the green line at 0 is the on-task limit and anything
above it is outside tolerance. Runs whose log holds no <code>eval</code> record are absent from the
evaluation charts and show a dash in the table. The <code>artifact</code> token names the material the
environment actually simulated, so a run is a policy and a material: evaluation curves are drawn one per
material token, never merged, and a log holding several tokens is marked as a sweep. Table cells outside
the material column come from the latest evaluation record, which belongs to one material. Training-health
curves stay one per run, because they describe the policy.
Nonfinite values, such as an undefined objective, are dropped from the charts and shown as a dash.
Lines that do not parse, such as Warp module-load messages and tracebacks, are ignored.</footer>
</body></html>"""


def render_directory(directory: Path | str, refresh: int = 10, query: str = "") -> str:
    """Read every log in a directory and render the dashboard page.

    Args:
        directory: Directory holding ``train_*.log`` files.
        refresh: Auto-refresh period [s].
        query: Query string selecting runs and materials; empty shows every run.
    """
    return render(load_runs(directory), directory, refresh=refresh, query=query)


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard, re-reading the logs on every request."""

    directory_path: Path = Path(DEFAULT_DIRECTORY)
    refresh_s: int = 10
    server_version = "ImpedanceDashboard/1.0"

    def do_GET(self) -> None:
        """Render the dashboard at ``/`` and report 404 for any other path."""
        path, _, query = self.path.partition("?")
        if path not in ("/", "/index.html"):
            self.send_error(404, "Only / is served")
            return
        try:
            page = render_directory(self.directory_path, refresh=self.refresh_s, query=query).encode("utf-8")
        except Exception as error:
            page = (
                "<!doctype html><html lang='en'><body><h1>Impedance Instron training</h1>"
                f"<p>The dashboard could not read {_escape(self.directory_path)}: {_escape(error)}</p>"
                "</body></html>"
            ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, format: str, *args) -> None:
        """Drop per-request logging so the training console stays readable.

        Args:
            format: Printf-style format string from :mod:`http.server`.
            *args: Format arguments.
        """


def create_server(
    directory: Path | str, host: str = "127.0.0.1", port: int = 8000, refresh: int = 10
) -> ThreadingHTTPServer:
    """Build a dashboard HTTP server without starting its request loop.

    Args:
        directory: Directory holding ``train_*.log`` files.
        host: Interface to bind.
        port: Port to bind; ``0`` picks a free port.
        refresh: Auto-refresh period [s] written into the page.
    """
    handler = type(
        "BoundDashboardHandler",
        (DashboardHandler,),
        {"directory_path": Path(directory), "refresh_s": int(refresh)},
    )
    return ThreadingHTTPServer((host, port), handler)


def serve(directory: Path | str, host: str = "127.0.0.1", port: int = 8000, refresh: int = 10) -> None:
    """Serve the dashboard until interrupted.

    Args:
        directory: Directory holding ``train_*.log`` files.
        host: Interface to bind.
        port: Port to bind; ``0`` picks a free port.
        refresh: Auto-refresh period [s] written into the page.
    """
    server = create_server(directory, host=host, port=port, refresh=refresh)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    print(f"dashboard: http://{bound_host}:{bound_port}/  (directory {Path(directory)}, refresh {int(refresh)} s)")
    print("dashboard: press Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard: stopped", flush=True)
    finally:
        server.shutdown()
        server.server_close()


def create_parser() -> argparse.ArgumentParser:
    """Build the dashboard command line."""
    parser = argparse.ArgumentParser(description="Serve a live training dashboard for impedance runs.")
    parser.add_argument("--directory", default=DEFAULT_DIRECTORY, help="directory holding train_*.log files")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="port to bind, 0 picks a free port")
    parser.add_argument("--refresh", type=int, default=10, help="page auto-refresh period in seconds")
    return parser


def main() -> None:
    """Serve the dashboard from the command line."""
    args = create_parser().parse_args()
    serve(args.directory, host=args.host, port=args.port, refresh=args.refresh)


if __name__ == "__main__":
    main()
