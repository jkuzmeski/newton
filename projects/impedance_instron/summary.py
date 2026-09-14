# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render ``projects/impedance_instron/LEGACY_REPORT.md`` as an illustrated offline page.

This renderer belongs to the retired momentum/work controller experiments, not
current two-stiffness results in ``REPORT.md``. Supply the historical output
directory with ``--data-directory``; ``outputs/impedance_instron/LEGACY_ARCHIVE.json``
may point to a local archive. Missing artifacts remain labelled placeholders.

The historical narrative is rendered in its own section order with inline SVG
figures. Other Markdown documents render without these legacy figures or numerical
tables. Everything uses the standard library and NumPy, so the page opens without
network access, JavaScript or a plotting package.

Run it with::

    uv run -m projects.impedance_instron.summary --output outputs/impedance_instron/report

which writes ``summary.html`` and ``summary.json``. The JSON holds every number the
figures plot, so a figure can be checked without re-reading the source artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from projects.impedance_instron.dashboard import parse_evaluation

DEFAULT_REPORT = "projects/impedance_instron/LEGACY_REPORT.md"
DEFAULT_DATA_DIRECTORY = "outputs/impedance_instron"
DEFAULT_OUTPUT = "outputs/impedance_instron/report"
DEFAULT_PROFILE_NAME = "stance_planar_context.json"
MAX_POINTS = 600

# A series is treated as too coarse to draw as a solid line once the richest series on the same
# axes carries more than this many samples for each of its own.
COARSE_SAMPLE_RATIO = 4

# Individual samples are only drawn while they stay countable; beyond this they would merge into a
# band and say nothing, so the dashes carry the message alone.
MARKER_LIMIT = 160

# Share of body weight above which the shoe counts as loaded. This must stay equal to
# ``projects.impedance_instron.env.CONTACT_FORCE_FRACTION``, which is the single constant the dense
# momentum reward's touchdown detector, :meth:`ImpedanceEnv._rollout`'s stance gate and
# :func:`projects.impedance_instron.optimize.simulate` share; two definitions of "stance began" is
# the defect LEGACY_REPORT.md section 4 documents. It is restated here instead of imported because ``env``
# pulls in Warp and this page stays pure NumPy, and ``newton/tests/test_impedance_summary.py`` pins
# this value against env.py so the two cannot drift.
CONTACT_FORCE_FRACTION = 0.02

# Fallback only, matching ``provenance.com_surrogate.gravity_m_s2`` of the stance profile. Body
# weight is taken from the profile's own ``mass_kg`` and gravity whenever the profile is readable.
FALLBACK_GRAVITY_M_S2 = 9.80665

# --- Historical report numbers; see LEGACY_REPORT.md ------------------------------------

# LEGACY_REPORT.md section 3.2, "Task tolerances": momentum excursion in multiples of the
# 0.044 m/s deadband. Zero means on task.
MOMENTUM_EXCURSION: tuple[tuple[str, float], ...] = (
    ("open-loop command J", 0.868),
    ("policy, prescribed pitch", 1.190),
    ("policy, prescribed, 3x budget", 1.330),
    ("policy, prescribed, broken reward (v1)", 1.951),
    ("policy with ankle (v2)", 0.315),
)
# LEGACY_REPORT.md section 3.2: three times the training budget moved the excursion the wrong
# way, from 1.190 to 1.330, so prescribed pitch has a floor rather than a best value.
PRESCRIBED_PITCH_FLOOR: tuple[float, float] = (1.190, 1.330)
# LEGACY_REPORT.md section 3.2: the deadband is the subject's own step-to-step variability.
MOMENTUM_DEADBAND_M_S = 0.044

# LEGACY_REPORT.md section 3.1, historical peak-timing table. Each entry is a run's own
# peak and the measured peak located in that SAME detected-contact window, in per cent of stance.
# Both are reported there on the shared window, shoe Fz above CONTACT_FORCE_FRACTION of body weight,
# and the page recomputes both so figure B can state whether they still agree.
REPORT_PEAK_TIMING_PCT: tuple[tuple[str, float, float], ...] = (
    ("legacy", 47.73, 44.22),
    ("command_j", 40.66, 40.28),
    ("ankle_v2", 43.95, 42.69),
)

# Agreement tolerance for that comparison [points of stance]. A coarse archive is allowed its own
# sample spacing on top, because one frame of quantisation is not a disagreement.
PEAK_TIMING_TOLERANCE_PCT = 0.5


# LEGACY_REPORT.md section 2.1, stiff-limit convergence table. One decade of ankle stiffness
# removes one decade of difference against the prescribed-pitch run.
STIFF_LIMIT_K_THETA: tuple[float, ...] = (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 3.0e6)
STIFF_LIMIT_SERIES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("pitch", (3.15e-01, 9.80e-02, 1.29e-02, 1.34e-03, 1.34e-04, 4.48e-05)),
    ("shoe Fz", (1.87e-01, 6.84e-02, 8.49e-03, 1.47e-03, 1.85e-04, 7.10e-05)),
    ("leg length", (2.76e-02, 3.38e-02, 2.94e-03, 3.01e-04, 3.00e-05, 9.69e-06)),
)

# LEGACY_REPORT.md section 5, "Infrastructure". Units differ per group; the caption says so.
INFRASTRUCTURE: tuple[tuple[str, str, float, float], ...] = (
    ("rollout", "s", 6.7, 0.517),
    ("foundation substep, 64 worlds", "ms", 0.170, 0.077),
    ("contact metric block", "us", 49.0, 10.0),
)

# LEGACY_REPORT.md section 3.2: v1 to v5 predate evaluation logging, so only their final
# values exist. They are drawn as horizontal reference lines in the training figure.
PRESCRIBED_FINALS: tuple[tuple[str, float, float], ...] = (
    ("policy v3, prescribed pitch", 1.190, 77.0),
    ("policy v5, prescribed, 3x budget", 1.330, 53.0),
)

# Strain window of the material comparison, matching the band quoted in
# projects/impedance_instron/mcclough.py and LEGACY_REPORT.md section 6.
MATERIAL_STRAIN_MIN = 0.02
MATERIAL_STRAIN_MAX = 0.60

COLOR_MEASURED = "#374151"
COLOR_LEGACY = "#b45309"
COLOR_COMMAND = "#2563eb"
COLOR_ANKLE = "#047857"
COLOR_ACCENT = "#7c3aed"
COLOR_GRID = "#d8dee7"
COLOR_RULE = "#99a7b9"
COLOR_BAND = "#f0d9a8"

# One colour per material, so a frozen-policy sweep across foams stays readable. The order is the
# order the materials first appear in a log, which keeps a rerun of the same log stable.
MATERIAL_COLORS: tuple[str, ...] = ("#047857", "#2563eb", "#b45309", "#7c3aed", "#be123c", "#0891b2")


# Canonical channel name to the (achieved, measured) column names of a trace. The force
# channels are recorded as ``shoe_*`` for the driven shoe and ``reference_*`` for the
# capture trial; the centre-of-mass channels keep their own names.
CHANNELS: dict[str, tuple[str, str]] = {
    "fz_n": ("shoe_fz_n", "reference_fz_n"),
    "fx_n": ("shoe_fx_n", "reference_fx_n"),
    "com_z_m": ("com_z_m", "reference_com_z_m"),
    "com_vz_m_s": ("com_vz_m_s", "reference_com_vz_m_s"),
    "leg_length_m": ("leg_length_m", "reference_leg_length_m"),
}


class Series(NamedTuple):
    """One polyline of a figure."""

    label: str
    x: np.ndarray
    y: np.ndarray
    color: str
    dash: str = ""
    width: float = 2.0
    markers: bool = False
    """Draw the samples themselves, for a series too coarse to imply a continuous line."""


@dataclass
class Run:
    """One controller trace mapped onto a common set of physical channels."""

    key: str
    label: str
    color: str
    source: str
    reference: bool = False
    dash: str = ""
    trace: dict[str, np.ndarray] | None = None
    contact: tuple[float, float] | None = None
    note: str = ""
    material: str | None = None
    """Token of the material the environment simulated, when the archive or its sidecar names one."""
    coarse: bool = False
    """The trace is sampled far more coarsely than the richest run on the same axes."""

    def column(self, base: str) -> np.ndarray | None:
        """Return one physical channel of this run.

        Args:
            base: Canonical channel name, such as ``"fz_n"`` or ``"com_z_m"``.

        Returns:
            The channel values, or ``None`` when the run or the channel is missing.
        """
        if self.trace is None:
            return None
        achieved, measured = CHANNELS.get(base, (base, f"reference_{base}"))
        values = self.trace.get(measured if self.reference else achieved)
        return None if values is None else np.asarray(values, dtype=float)

    @property
    def sample_count(self) -> int:
        """Number of samples in the underlying trace."""
        if self.trace is None:
            return 0
        time_s = self.trace.get("time_s")
        return 0 if time_s is None else int(np.size(time_s))

    @property
    def sample_interval_s(self) -> float | None:
        """Sampling interval of the trace [s], from the archive's own field or its time axis."""
        if self.trace is None:
            return None
        for key in ("substep_dt_s", "substep_dt", "sim_dt_s", "sim_dt", "dt_s"):
            value = self.trace.get(key)
            if value is not None and np.size(value):
                candidate = float(np.ravel(value)[0])
                if math.isfinite(candidate) and candidate > 0.0:
                    return candidate
        time_s = self.trace.get("time_s")
        if time_s is None or np.size(time_s) < 2:
            return None
        steps = np.diff(np.asarray(time_s, dtype=float))
        steps = steps[np.isfinite(steps) & (steps > 0.0)]
        return float(np.median(steps)) if steps.size else None

    def percent(self) -> np.ndarray | None:
        """Return the time axis as per cent of this run's own contact window."""
        if self.trace is None or self.contact is None:
            return None
        time_s = np.asarray(self.trace["time_s"], dtype=float)
        start, end = self.contact
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            return None
        return (time_s - start) / (end - start) * 100.0


@dataclass
class Sources:
    """Every artifact the figures read, with the runs already resolved."""

    directory: Path
    profile: dict[str, Any] | None = None
    runs: list[Run] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    logs: dict[str, str] = field(default_factory=dict)
    eval_counts: dict[str, int] = field(default_factory=dict)
    materials: dict[str | None, list[dict]] = field(default_factory=dict)
    archives: list[dict[str, Any]] = field(default_factory=list)
    superseded: list[dict[str, Any]] = field(default_factory=list)
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def threshold_n(self) -> float | None:
        """Loading threshold [N]: :data:`CONTACT_FORCE_FRACTION` of the profile's body weight."""
        return None if self.profile is None else float(self.profile["contact_threshold_n"])

    @property
    def stance_window_s(self) -> tuple[float, float] | None:
        """Measured stance window of the capture trial in trace time [s], when the profile has it."""
        if self.profile is None or self.profile.get("stance_window_s") is None:
            return None
        start, end = self.profile["stance_window_s"]
        return float(start), float(end)

    @property
    def ankle_keys(self) -> tuple[str, ...]:
        """Keys of the discovered policy runs, one per material, in drawing order."""
        return tuple(run.key for run in self.runs if run.key.startswith("ankle_v2"))

    def run(self, key: str) -> Run | None:
        """Return one run by key, or ``None`` when it was not loaded.

        Args:
            key: Run key such as ``"ankle_v2"``.
        """
        for item in self.runs:
            if item.key == key:
                return item
        return None

    def available(self, keys: tuple[str, ...], base: str) -> list[Run]:
        """Return the runs that carry a channel.

        Args:
            keys: Run keys to consider, in drawing order.
            base: Channel name without the ``reference_`` prefix.
        """
        found = []
        for key in keys:
            item = self.run(key)
            if item is not None and item.column(base) is not None and item.percent() is not None:
                found.append(item)
        return found


# --- Loading ---------------------------------------------------------------------------


def load_csv_trace(path: Path | str) -> dict[str, np.ndarray] | None:
    """Read a controller trace CSV into named columns.

    Args:
        path: Path of a ``trace.csv`` written by the evaluation or comparison tools.

    Returns:
        Mapping from column name to values, or ``None`` when the file is missing or unreadable.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        table = np.genfromtxt(path, delimiter=",", names=True)
    except (ValueError, OSError):
        return None
    if table.dtype.names is None or table.size == 0:
        return None
    return {name: np.atleast_1d(np.asarray(table[name], dtype=float)) for name in table.dtype.names}


def load_npz_trace(path: Path | str) -> dict[str, np.ndarray] | None:
    """Read the numeric channels of an evaluation ``.npz`` archive.

    Entries that are not numeric, such as the material token the trainer stores beside the
    channels, are skipped rather than failing the whole archive; read them with
    :func:`load_npz_text`.

    Args:
        path: Path of a ``policy_*.eval.npz`` archive.

    Returns:
        Mapping from key to values, or ``None`` when the file is missing or has no channels.
    """
    path = Path(path)
    if not path.is_file():
        return None
    columns: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                try:
                    columns[key] = np.atleast_1d(np.asarray(archive[key], dtype=float))
                except (ValueError, TypeError):
                    continue
    except (ValueError, OSError):
        return None
    return columns or None


def load_npz_text(path: Path | str) -> dict[str, str]:
    """Read the non-numeric entries of an evaluation archive as text.

    Args:
        path: Path of a ``policy_*.eval.npz`` archive.

    Returns:
        Mapping from key to text, empty when the archive is missing or carries no text.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    text: dict[str, str] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                if value.dtype.kind in "US" and value.size == 1:
                    text[key] = str(np.ravel(value)[0])
    except (ValueError, OSError):
        return {}
    return text


def load_profile_context(path: Path | str) -> dict[str, Any] | None:
    """Read the stance profile for body weight and the measured stance window.

    Args:
        path: Path of the stance profile JSON the rig is driven from.

    Returns:
        Mass, gravity, body weight, the derived contact threshold and the measured stance
        window in trace time, or ``None`` when the profile is missing or unusable.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
        mass_kg = float(profile["mass_kg"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    provenance = profile.get("provenance", {}) if isinstance(profile.get("provenance"), dict) else {}
    surrogate = provenance.get("com_surrogate", {}) if isinstance(provenance.get("com_surrogate"), dict) else {}
    try:
        gravity = float(surrogate.get("gravity_m_s2", FALLBACK_GRAVITY_M_S2))
    except (TypeError, ValueError):
        gravity = FALLBACK_GRAVITY_M_S2
    if not (math.isfinite(mass_kg) and mass_kg > 0.0 and math.isfinite(gravity) and gravity > 0.0):
        return None
    weight = mass_kg * gravity
    context: dict[str, Any] = {
        "source": str(path),
        "mass_kg": mass_kg,
        "gravity_m_s2": gravity,
        "body_weight_n": weight,
        "contact_force_fraction": CONTACT_FORCE_FRACTION,
        "contact_threshold_n": CONTACT_FORCE_FRACTION * weight,
        "stance_window_s": None,
    }
    # The measured stance window of the capture trial, expressed in trace time. explain.py uses it
    # as a fixed clock for every run; the figures use the detected contact window instead, and the
    # payload carries both so the two conventions can be compared.
    running = provenance.get("running", {}) if isinstance(provenance.get("running"), dict) else {}
    selected = running.get("selected_stance_source_s")
    source_time = profile.get("source_time_s")
    if isinstance(selected, list) and len(selected) == 2 and isinstance(source_time, list) and source_time:
        try:
            origin = float(source_time[0])
            window = (float(selected[0]) - origin, float(selected[1]) - origin)
        except (TypeError, ValueError):
            window = None
        if window is not None and math.isfinite(window[0]) and window[1] > window[0]:
            context["stance_window_s"] = list(window)
    return context


def contact_window(
    trace: dict[str, np.ndarray] | None, force_key: str, threshold_n: float | None
) -> tuple[float, float] | None:
    """Find the contact interval of a trace [s].

    The archived traces carry ``contact_start_s`` and ``contact_end_s``, which the runtime
    already derived with :data:`CONTACT_FORCE_FRACTION`. The CSV traces do not, so the window
    is measured from the force waveform above the same fraction of body weight.

    Args:
        trace: Loaded trace columns, or ``None``.
        force_key: Vertical force column that defines loading for this run.
        threshold_n: Loading threshold [N], normally ``CONTACT_FORCE_FRACTION`` of body weight.

    Returns:
        Start and end times, or ``None`` when no loaded interval exists or no threshold is known.
    """
    if trace is None or "time_s" not in trace:
        return None
    time_s = np.asarray(trace["time_s"], dtype=float)
    start, end = trace.get("contact_start_s"), trace.get("contact_end_s")
    if start is not None and end is not None and np.size(start) and np.size(end):
        low, high = float(np.ravel(start)[0]), float(np.ravel(end)[0])
        if math.isfinite(low) and math.isfinite(high) and high > low:
            return low, high
    force = trace.get(force_key)
    if force is None or threshold_n is None or not math.isfinite(threshold_n):
        return None
    loaded = np.flatnonzero(np.isfinite(force) & (np.asarray(force, dtype=float) > threshold_n))
    if loaded.size < 2:
        return None
    low, high = float(time_s[loaded[0]]), float(time_s[loaded[-1]])
    return (low, high) if high > low else None


# Keys under which the trainer may record the material token: inside the evaluation archive, in its
# JSON sidecar, or as the optional ``artifact`` field of an evaluation log record. The token hashes
# the material the environment actually simulated, so two runs of one artifact path with different
# stiffness scaling carry different tokens and must not be merged.
MATERIAL_TOKEN_KEYS: tuple[str, ...] = ("artifact", "artifact_token", "material_token", "material", "shoe_artifact")
MATERIAL_LOG_KEY = "artifact"


def optional_token(line: str, key: str = MATERIAL_LOG_KEY) -> str | None:
    """Read one optional whitespace-free ``key=value`` token out of a log line.

    Absent, empty and duplicated keys all return ``None``, so a log written before the key
    existed behaves exactly as it did before.

    Args:
        line: Raw log line.
        key: Field name to read.

    Returns:
        The token, or ``None`` when the line does not carry exactly one usable value.
    """
    found: str | None = None
    for token in line.split():
        name, separator, value = token.partition("=")
        if not separator or name != key or not value:
            continue
        if found is not None:
            return None
        found = value
    return found


def parse_evaluation_record(line: str) -> dict | None:
    """Parse one evaluation record, including the optional material token.

    The numeric fields are parsed by :func:`projects.impedance_instron.dashboard.parse_evaluation`,
    so the two pages cannot drift. The optional ``artifact`` token is added on top, and is simply
    absent for a log written before the trainer emitted it.

    Args:
        line: Raw log line, which may be noise or a partially written record.

    Returns:
        The parsed record, or ``None`` when the line is not a usable evaluation record.
    """
    record = parse_evaluation(line)
    if record is None:
        return None
    if not record.get(MATERIAL_LOG_KEY):
        token = optional_token(line)
        if token:
            record[MATERIAL_LOG_KEY] = token
    return record


def material_token(text: dict[str, str], path: Path | str) -> str | None:
    """Find the material token of one evaluation archive.

    The archive itself is checked first, then its ``.json`` sidecar, because either may carry
    the token depending on when the run was written.

    Args:
        text: Text entries of the archive, from :func:`load_npz_text`.
        path: Path of the archive.

    Returns:
        The token, or ``None`` when neither source names one.
    """
    for key in MATERIAL_TOKEN_KEYS:
        value = text.get(key)
        if value:
            return value
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in MATERIAL_TOKEN_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def group_by_material(records: list[dict]) -> dict[str | None, list[dict]]:
    """Split evaluation records by the material that produced them.

    Records without a token are kept under ``None`` rather than merged into a named material,
    so a sweep is never silently collapsed onto one curve.

    Args:
        records: Parsed evaluation records.

    Returns:
        Mapping from token to its records, ordered by first appearance.
    """
    grouped: dict[str | None, list[dict]] = {}
    for record in records:
        token = record.get(MATERIAL_LOG_KEY)
        grouped.setdefault(token if isinstance(token, str) and token else None, []).append(record)
    return grouped


POLICY_ARCHIVE_STEM = "policy_ankle_v2"
POLICY_ARCHIVE_SUFFIX = ".eval.npz"


def discover_archives(directory: Path | str, stem: str = POLICY_ARCHIVE_STEM) -> list[dict[str, Any]]:
    """Find the evaluation archives of one policy, newest first within each material.

    The trainer names an archive after the policy AND the material, so a fixed file name goes
    stale as soon as a run is regenerated. The rule here is:

    1. glob ``<stem>*.eval.npz`` in the directory;
    2. read each archive's material token, from the archive or its JSON sidecar, falling back to
       the name suffix between the stem and ``.eval.npz``;
    3. keep the newest file per material, because a rerun of one material supersedes its own
       earlier archive and nothing else;
    4. order tokenised archives before an untokenised one, then newest first, so the primary run
       is the most recently written identified material.

    Several surviving entries mean a material sweep, and each becomes its own run rather than a
    choice between them.

    Args:
        directory: Directory holding the impedance-instron outputs.
        stem: Archive stem of the policy.

    Returns:
        One entry per material with ``path``, ``token``, ``modified`` and ``tokenised``.
    """
    directory = Path(directory)
    found: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"{stem}*{POLICY_ARCHIVE_SUFFIX}")):
        name = path.name[: -len(POLICY_ARCHIVE_SUFFIX)]
        if name != stem and not name.startswith(f"{stem}_"):
            continue
        token = material_token(load_npz_text(path), path) or (name[len(stem) + 1 :] or None)
        try:
            modified = path.stat().st_mtime
        except OSError:  # pragma: no cover - the file was listed a moment ago
            continue
        found.append({"path": path, "token": token, "modified": modified, "tokenised": name != stem})
    newest: dict[str | None, dict[str, Any]] = {}
    for entry in found:
        current = newest.get(entry["token"])
        if current is None or entry["modified"] > current["modified"]:
            newest[entry["token"]] = entry
    return sorted(newest.values(), key=lambda entry: (not entry["tokenised"], -entry["modified"]))


def select_archives(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split discovered archives into the ones to draw and the ones superseded by them.

    A tokenised archive names the material it was run on, so once one exists an untokenised
    archive of the same policy is both older and unattributable and is not drawn. Several
    tokenised archives are a material sweep and are all kept.

    Args:
        entries: Discovered archives from :func:`discover_archives`.

    Returns:
        The archives to draw and the ones left out, each in the discovered order.
    """
    tokenised = [entry for entry in entries if entry["tokenised"]]
    if not tokenised:
        return entries, []
    return tokenised, [entry for entry in entries if not entry["tokenised"]]


def load_sources(directory: Path | str = DEFAULT_DATA_DIRECTORY, profile: Path | str | None = None) -> Sources:
    """Load every artifact the figures need, recording what is missing.

    Nothing here raises on a missing file: an absent artifact leaves its run without a
    trace, and the figures that need it render a labelled placeholder instead. Without the
    stance profile there is no body weight, so there is no contact threshold and the
    waveform figures become placeholders rather than inventing a window. The policy archive is
    DISCOVERED, never assumed: see :func:`discover_archives` and :func:`select_archives`.

    Args:
        directory: Directory holding the impedance-instron outputs.
        profile: Stance profile JSON, defaulting to :data:`DEFAULT_PROFILE_NAME` in ``directory``.

    Returns:
        The resolved sources.
    """
    directory = Path(directory)
    profile_path = Path(profile) if profile is not None else directory / DEFAULT_PROFILE_NAME
    legacy_path = directory / "legacy_compare" / "trace.csv"
    command_path = directory / "eval_j" / "trace.csv"
    legacy = load_csv_trace(legacy_path)
    command = load_csv_trace(command_path)
    sources = Sources(directory=directory, profile=load_profile_context(profile_path))
    if sources.profile is None:
        sources.missing["stance profile"] = str(profile_path)
    threshold = sources.threshold_n
    if legacy is None:
        sources.missing["legacy_compare"] = str(legacy_path)
    if command is None:
        sources.missing["eval_j"] = str(command_path)
    sources.runs = [
        Run(
            key="measured",
            label="measured (force plate)",
            color=COLOR_MEASURED,
            source=str(legacy_path),
            reference=True,
            dash="5 4",
            trace=legacy,
            contact=contact_window(legacy, "reference_fz_n", threshold),
            note="reference columns of the legacy comparison trace",
        ),
        Run(
            key="legacy",
            label="legacy schedule (fed measured GRF)",
            color=COLOR_LEGACY,
            source=str(legacy_path),
            trace=legacy,
            contact=contact_window(legacy, "shoe_fz_n", threshold),
        ),
        Run(
            key="command_j",
            label="open-loop command J (CMA-ES)",
            color=COLOR_COMMAND,
            source=str(command_path),
            trace=command,
            contact=contact_window(command, "shoe_fz_n", threshold),
        ),
    ]
    discovered = discover_archives(directory)
    chosen, superseded = select_archives(discovered)
    sources.archives = discovered
    sources.superseded = superseded
    if not chosen:
        sources.missing[POLICY_ARCHIVE_STEM] = str(directory / f"{POLICY_ARCHIVE_STEM}*{POLICY_ARCHIVE_SUFFIX}")
    for index, entry in enumerate(chosen):
        trace = load_npz_trace(entry["path"])
        if trace is None:
            sources.missing[entry["path"].name] = str(entry["path"])
            continue
        # The first archive keeps the plain key, so a single-material page and every reference to
        # it are unchanged; a sweep adds one keyed run per further material.
        key = "ankle_v2" if index == 0 else f"ankle_v2_{entry['token']}"
        sources.runs.append(
            Run(
                key=key,
                label="closed-loop policy with ankle (v2)",
                color=COLOR_ANKLE if index == 0 else MATERIAL_COLORS[index % len(MATERIAL_COLORS)],
                source=str(entry["path"]),
                trace=trace,
                contact=contact_window(trace, "shoe_fz_n", threshold),
                material=entry["token"],
            )
        )
    # A run's identity is (policy, material) once the token exists, so name it on the curve.
    for run in sources.runs:
        if run.material:
            run.label = f"{run.label} on material {run.material}"

    # A run sampled far below the others cannot be drawn as a solid line without implying a
    # resolution it does not have, so it is marked here and rendered as dashes with visible samples.
    richest = max((run.sample_count for run in sources.runs), default=0)
    for run in sources.runs:
        if run.trace is None or run.sample_count == 0:
            continue
        run.coarse = run.sample_count * COARSE_SAMPLE_RATIO < richest
        if run.coarse:
            interval = run.sample_interval_s
            spacing = f", {interval * 1.0e3:.1f} ms apart" if interval is not None else ""
            run.note = f"{run.sample_count} samples against {richest}{spacing}; drawn as dashes with its samples marked"
    for name in ("train_ankle_v2.log", "train_v3.log", "train_v5.log"):
        path = directory / name
        if path.is_file():
            sources.logs[name] = path.read_text(encoding="utf-8", errors="replace")
        else:
            sources.missing[name] = str(path)
    for name, text in sources.logs.items():
        records = [record for line in text.splitlines() if (record := parse_evaluation_record(line)) is not None]
        records.sort(key=lambda record: record["iteration"])
        sources.eval_counts[name] = len(records)
        if name == "train_ankle_v2.log":
            sources.evaluations = records
            sources.materials = group_by_material(records)
    return sources


# --- Drawing primitives ----------------------------------------------------------------


def _escape(value: Any) -> str:
    """Escape a value for both HTML and XML text.

    Args:
        value: Any value; it is rendered with ``str``.
    """
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 4) -> str:
    """Format a number for a label, using an em dash for anything non-finite.

    Args:
        value: Value to format.
        digits: Significant digits.
    """
    if value is None:
        return "\u2014"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "\u2014"
    return f"{value:.{digits}g}" if math.isfinite(value) else "\u2014"


def downsample(x: np.ndarray, y: np.ndarray, limit: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Thin a long series by a fixed stride, always keeping the last sample.

    Args:
        x: Sample abscissae.
        y: Sample values.
        limit: Approximate number of points to keep.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    count = min(x.size, y.size)
    x, y = x[:count], y[:count]
    if count <= limit or limit < 1:
        return x, y
    stride = -(-count // limit)
    index = np.arange(0, count, stride)
    if index[-1] != count - 1:
        index = np.append(index, count - 1)
    return x[index], y[index]


def axis_values(values: np.ndarray, log: bool) -> np.ndarray:
    """Map values onto an axis, dropping what a log axis cannot show.

    Zero, negative and non-finite samples become ``nan`` on a log axis and are then
    skipped when the polyline is built, so no coordinate is ever written as ``nan``.

    Args:
        values: Raw values.
        log: Whether the axis is base-10 logarithmic.
    """
    values = np.asarray(values, dtype=float)
    if not log:
        return values
    safe = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(safe)


def _bounds(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    """Return padded axis bounds over the finite entries of ``values``.

    Args:
        values: Candidate axis values, already mapped onto the axis.
        pad: Fractional padding added at both ends.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        margin = max(abs(high) * 0.05, 0.5)
        return low - margin, high + margin
    margin = (high - low) * pad
    return low - margin, high + margin


def _nice_step(raw: float) -> float:
    """Round a raw tick spacing up to 1, 2, 2.5 or 5 times a power of ten.

    Args:
        raw: Ideal spacing between ticks, in axis units.
    """
    if not math.isfinite(raw) or raw <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10.0 * magnitude


def _tick_values(low: float, high: float, log: bool, count: int = 5) -> list[float]:
    """Choose tick positions for one axis.

    Args:
        low: Axis minimum, in axis units.
        high: Axis maximum, in axis units.
        log: Whether the axis is base-10 logarithmic, so units are decades.
        count: Preferred number of ticks on a linear axis.
    """
    if high <= low:
        high = low + 1.0
    if log:
        first, last = math.ceil(low - 1.0e-9), math.floor(high + 1.0e-9)
        decades = [float(value) for value in range(int(first), int(last) + 1)]
        if len(decades) >= 2:
            return decades[:: max(1, len(decades) // 6)]
        return [low + (high - low) * fraction for fraction in np.linspace(0.0, 1.0, count)]
    # Prefer round numbers, halving the spacing while a round step leaves too few ticks.
    step = _nice_step((high - low) / max(count - 1, 1))
    for _ in range(3):
        ticks: list[float] = []
        value = math.ceil(low / step) * step
        while value <= high + step * 1.0e-6 and len(ticks) <= 3 * count:
            ticks.append(round(value, 12))
            value += step
        if len(ticks) >= 3:
            return ticks
        step = _nice_step(step / 2.0)
    return [low + (high - low) * fraction for fraction in np.linspace(0.0, 1.0, count)]


def _tick_label(value: float, log: bool, digits: int = 4) -> str:
    """Format one axis tick.

    Args:
        value: Tick position in axis units.
        log: Whether the axis is logarithmic, so ``value`` is a decade exponent.
        digits: Significant digits on a linear axis.
    """
    if not math.isfinite(value):
        return ""
    if log:
        return f"1e{value:g}"
    if value != 0.0 and abs(value) >= 1000.0 and float(value).is_integer():
        return f"{value:.0f}"
    return _number(value, digits)


def _polyline(sx: np.ndarray, sy: np.ndarray) -> str:
    """Build an SVG path, breaking the line wherever a sample is not drawable.

    Args:
        sx: Screen abscissae.
        sy: Screen ordinates.
    """
    pieces: list[str] = []
    active = False
    for px, py in zip(np.asarray(sx, dtype=float), np.asarray(sy, dtype=float), strict=False):
        if math.isfinite(px) and math.isfinite(py):
            pieces.append(f"{'L' if active else 'M'}{px:.2f},{py:.2f}")
            active = True
        else:
            active = False
    return " ".join(pieces)


def _legend(entries: list[tuple[str, str, str, bool]]) -> str:
    """Render a colour legend.

    Args:
        entries: Tuples of ``(label, colour, dash, markers)``. A dashed series says so, and a
            series whose own samples are drawn says that instead, because that is the stronger
            statement about what the line does and does not resolve.
    """
    if not entries:
        return ""

    def suffix(dash: str, markers: bool) -> str:
        if markers:
            return " (samples drawn)"
        return " (dashed)" if dash else ""

    items = "".join(
        f'<span class="key"><i style="background:{_escape(color)}"></i>{_escape(label)}{suffix(dash, markers)}</span>'
        for label, color, dash, markers in entries
    )
    return f'<div class="legend">{items}</div>'


def plot(
    title: str,
    series: list[Series],
    *,
    xlabel: str,
    ylabel: str,
    xlog: bool = False,
    ylog: bool = False,
    hlines: tuple[tuple[float, str, str], ...] = (),
    vlines: tuple[tuple[float, str, str], ...] = (),
    yband: tuple[float, float, str] | None = None,
    note: str = "",
    width: int = 680,
    height: int = 320,
) -> str:
    """Draw one panel as inline SVG with labelled, optionally logarithmic axes.

    Args:
        title: Panel heading.
        series: Polylines to draw.
        xlabel: Horizontal axis label, including its unit.
        ylabel: Vertical axis label, including its unit.
        xlog: Draw the horizontal axis on a base-10 log scale.
        ylog: Draw the vertical axis on a base-10 log scale.
        hlines: Horizontal reference lines as ``(value, label, colour)`` in data units.
        vlines: Vertical reference lines as ``(value, label, colour)`` in data units.
        yband: Shaded vertical band as ``(low, high, label)`` in data units.
        note: Extra line shown under the panel heading.
        width: SVG viewport width.
        height: SVG viewport height.

    Returns:
        A panel element, or an explanatory panel when nothing is drawable.
    """
    prepared: list[tuple[Series, np.ndarray, np.ndarray, bool]] = []
    for item in series:
        x = axis_values(item.x, xlog)
        y = axis_values(item.y, ylog)
        x, y = downsample(x, y)
        drawable = int(np.count_nonzero(np.isfinite(x) & np.isfinite(y)))
        if x.size and y.size and drawable:
            prepared.append((item, x, y, item.markers and drawable <= MARKER_LIMIT))
    if not prepared:
        reason = "No drawable samples for this panel."
        if xlog or ylog:
            reason = "No positive samples for this logarithmic panel."
        return f'<div class="panel"><h4>{_escape(title)}</h4><p class="miss">{_escape(reason)}</p></div>'

    xs = np.concatenate([x for _, x, _, _ in prepared])
    ys = np.concatenate([y for _, _, y, _ in prepared])
    extra_y = [axis_values(np.array([value]), ylog)[0] for value, _, _ in hlines]
    if yband is not None:
        extra_y.extend(axis_values(np.array([yband[0], yband[1]]), ylog).tolist())
    if extra_y:
        ys = np.concatenate([ys, np.asarray(extra_y, dtype=float)])
    extra_x = [axis_values(np.array([value]), xlog)[0] for value, _, _ in vlines]
    if extra_x:
        xs = np.concatenate([xs, np.asarray(extra_x, dtype=float)])
    xmin, xmax = _bounds(xs, pad=0.02)
    ymin, ymax = _bounds(ys)
    left, top, plot_width, plot_height = 78, 20, width - 100, height - 66

    def px(values: np.ndarray) -> np.ndarray:
        return left + (np.asarray(values, dtype=float) - xmin) / (xmax - xmin) * plot_width

    def py(values: np.ndarray) -> np.ndarray:
        return top + (ymax - np.asarray(values, dtype=float)) / (ymax - ymin) * plot_height

    content = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        f"<title>{_escape(title)}</title>",
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="{COLOR_GRID}"></rect>',
    ]
    if yband is not None:
        low, high = sorted(axis_values(np.array([yband[0], yband[1]]), ylog).tolist())
        if math.isfinite(low) and math.isfinite(high):
            y_high, y_low = float(py(np.array([high]))[0]), float(py(np.array([low]))[0])
            y_high, y_low = max(y_high, top), min(y_low, top + plot_height)
            if y_low > y_high:
                content.append(
                    f'<rect x="{left}" y="{y_high:.2f}" width="{plot_width}" '
                    f'height="{y_low - y_high:.2f}" fill="{COLOR_BAND}" opacity="0.55"></rect>'
                )
                content.append(
                    f'<text x="{left + 6}" y="{max(y_high + 13, top + 13):.2f}" class="band">{_escape(yband[2])}</text>'
                )
    for tick in _tick_values(ymin, ymax, ylog):
        y = float(py(np.array([tick]))[0])
        if not math.isfinite(y):
            continue
        content.append(f'<path d="M{left},{y:.2f}h{plot_width}" class="grid"></path>')
        content.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{_escape(_tick_label(tick, ylog))}</text>'
        )
    for tick in _tick_values(xmin, xmax, xlog):
        x = float(px(np.array([tick]))[0])
        if not math.isfinite(x):
            continue
        content.append(f'<path d="M{x:.2f},{top}v{plot_height}" class="grid"></path>')
        content.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 18}" text-anchor="middle">'
            f"{_escape(_tick_label(tick, xlog))}</text>"
        )
    for value, label, color in hlines:
        y = float(py(axis_values(np.array([value]), ylog))[0])
        if not math.isfinite(y) or not top - 1 <= y <= top + plot_height + 1:
            continue
        content.append(
            f'<path d="M{left},{y:.2f}h{plot_width}" stroke="{_escape(color)}" stroke-width="1.5" stroke-dasharray="7 4"></path>'
        )
        content.append(
            f'<text x="{left + plot_width - 4}" y="{max(y - 5, top + 11):.2f}" text-anchor="end" '
            f'fill="{_escape(color)}">{_escape(label)}</text>'
        )
    for value, label, color in vlines:
        x = float(px(axis_values(np.array([value]), xlog))[0])
        if not math.isfinite(x) or not left - 1 <= x <= left + plot_width + 1:
            continue
        content.append(
            f'<path d="M{x:.2f},{top}v{plot_height}" stroke="{_escape(color)}" stroke-dasharray="4 3"></path>'
        )
        content.append(f'<text x="{x + 4:.2f}" y="{top + 12}" fill="{_escape(color)}">{_escape(label)}</text>')
    for item, x, y, markers in prepared:
        path = _polyline(px(x), py(y))
        if not path:
            continue
        dash = f' stroke-dasharray="{_escape(item.dash)}"' if item.dash else ""
        content.append(
            f'<path d="{path}" fill="none" stroke="{_escape(item.color)}" '
            f'stroke-width="{item.width:g}" stroke-linejoin="round"{dash}>'
            f"<title>{_escape(item.label)}</title></path>"
        )
        if markers:
            # The samples themselves, so a coarse series cannot be read as a resolved waveform.
            for mx, my in zip(px(x), py(y), strict=True):
                if math.isfinite(mx) and math.isfinite(my):
                    content.append(
                        f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="2.1" fill="{_escape(item.color)}"></circle>'
                    )
    content.append(
        f'<text x="{left + plot_width / 2:.0f}" y="{height - 8}" text-anchor="middle" class="axis">{_escape(xlabel)}</text>'
    )
    content.append(
        f'<text x="14" y="{top + plot_height / 2:.0f}" text-anchor="middle" class="axis" '
        f'transform="rotate(-90 14 {top + plot_height / 2:.0f})">{_escape(ylabel)}</text>'
    )
    content.append("</svg>")
    subtitle = f'<p class="note">{_escape(note)}</p>' if note else ""
    legend = _legend([(item.label, item.color, item.dash, markers) for item, _, _, markers in prepared])
    return f'<div class="panel"><h4>{_escape(title)}</h4>{subtitle}{"".join(content)}{legend}</div>'


def bars(
    title: str,
    rows: list[tuple[str, float, str, str]],
    *,
    xlabel: str,
    log: bool = False,
    band: tuple[float, float, str] | None = None,
    marker: tuple[float, str] | None = None,
    note: str = "",
    width: int = 680,
) -> str:
    """Draw labelled horizontal bars on a linear or logarithmic value axis.

    Args:
        title: Panel heading.
        rows: Tuples of ``(label, value, colour, value text)``.
        xlabel: Value axis label, including its unit.
        log: Draw the value axis on a base-10 log scale.
        band: Shaded vertical band as ``(low, high, label)`` in data units.
        marker: Vertical marker line as ``(value, label)`` in data units.
        note: Extra line shown under the panel heading.
        width: SVG viewport width.

    Returns:
        A panel element, or an explanatory panel when no bar is drawable.
    """
    drawable = [(label, float(value), color, text) for label, value, color, text in rows if math.isfinite(float(value))]
    if log:
        drawable = [row for row in drawable if row[1] > 0.0]
    if not drawable:
        return f'<div class="panel"><h4>{_escape(title)}</h4><p class="miss">No values to draw.</p></div>'
    values = axis_values(np.array([row[1] for row in drawable]), log)
    candidates = [values]
    if band is not None:
        candidates.append(axis_values(np.array(band[:2], dtype=float), log))
    if marker is not None:
        candidates.append(axis_values(np.array([marker[0]]), log))
    span = np.concatenate(candidates)
    if log:
        low, high = _bounds(span, pad=0.12)
    else:
        low = min(0.0, float(np.nanmin(span)))
        high = float(np.nanmax(span))
        high = high + max(abs(high), 1.0e-9) * 0.28
    if high <= low:
        high = low + 1.0
    row_height, top, left = 34, 24, 250
    height = top + row_height * len(drawable) + 46
    plot_width = width - left - 30

    def px(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    content = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        f"<title>{_escape(title)}</title>",
    ]
    if band is not None:
        edges = sorted(axis_values(np.array(band[:2], dtype=float), log).tolist())
        if all(math.isfinite(edge) for edge in edges):
            x0, x1 = px(edges[0]), px(edges[1])
            content.append(
                f'<rect x="{x0:.2f}" y="{top - 6}" width="{max(x1 - x0, 1.5):.2f}" '
                f'height="{row_height * len(drawable) + 6}" fill="{COLOR_BAND}" opacity="0.6"></rect>'
            )
            content.append(f'<text x="{x1 + 5:.2f}" y="{top + 6}" class="band">{_escape(band[2])}</text>')
    for tick in _tick_values(low, high, log):
        x = px(tick)
        content.append(f'<path d="M{x:.2f},{top - 6}v{row_height * len(drawable) + 6}" class="grid"></path>')
        content.append(
            f'<text x="{x:.2f}" y="{top + row_height * len(drawable) + 18}" text-anchor="middle">'
            f"{_escape(_tick_label(tick, log))}</text>"
        )
    for index, (label, value, color, text) in enumerate(drawable):
        y = top + index * row_height
        mapped = float(axis_values(np.array([value]), log)[0])
        x = px(mapped) if math.isfinite(mapped) else px(low)
        start = px(max(low, 0.0)) if not log else px(low)
        content.append(
            f'<rect x="{min(start, x):.2f}" y="{y + 4:.2f}" width="{max(abs(x - start), 1.5):.2f}" '
            f'height="{row_height - 14}" fill="{_escape(color)}" opacity="0.85"><title>{_escape(label)}</title></rect>'
        )
        content.append(f'<text x="{left - 10}" y="{y + row_height / 2:.2f}" text-anchor="end">{_escape(label)}</text>')
        content.append(f'<text x="{x + 6:.2f}" y="{y + row_height / 2:.2f}" class="value">{_escape(text)}</text>')
    if marker is not None:
        mapped = float(axis_values(np.array([marker[0]]), log)[0])
        if math.isfinite(mapped):
            x = px(mapped)
            content.append(
                f'<path d="M{x:.2f},{top - 8}v{row_height * len(drawable) + 8}" stroke="{COLOR_ANKLE}" stroke-width="2"></path>'
            )
            content.append(f'<text x="{x + 5:.2f}" y="{top - 10}" fill="{COLOR_ANKLE}">{_escape(marker[1])}</text>')
    content.append(
        f'<text x="{left + plot_width / 2:.0f}" y="{height - 8}" text-anchor="middle" class="axis">{_escape(xlabel)}</text>'
    )
    content.append("</svg>")
    subtitle = f'<p class="note">{_escape(note)}</p>' if note else ""
    return f'<div class="panel"><h4>{_escape(title)}</h4>{subtitle}{"".join(content)}</div>'


def figure(identifier: str, title: str, panels: str, caption: str) -> str:
    """Wrap panels as a numbered figure with its caption.

    Args:
        identifier: Figure letter, matching the report text.
        title: Short figure title.
        panels: Rendered panel markup.
        caption: Sentence stating what the figure shows and what to conclude from it.

    Returns:
        A ``figure`` element.
    """
    return (
        f'<figure id="figure-{_escape(identifier.lower())}" class="figure">'
        f'<div class="panels">{panels}</div>'
        f"<figcaption><b>Figure {_escape(identifier)}. {_escape(title)}.</b> {caption}</figcaption>"
        f"</figure>"
    )


def placeholder(identifier: str, title: str, reason: str) -> str:
    """Render a figure whose data is missing, without failing the page.

    Args:
        identifier: Figure letter.
        title: Short figure title.
        reason: Why the figure could not be drawn, naming the missing artifact.

    Returns:
        A ``figure`` element carrying the explanation in place of the panels.
    """
    panel = (
        f'<div class="panel missing"><h4>{_escape(title)}</h4><p class="miss">Not drawn: {_escape(reason)}</p></div>'
    )
    return figure(identifier, title, panel, f"Placeholder. {_escape(reason)}")


# --- Series helpers --------------------------------------------------------------------

PERCENT_CLIP = (-12.0, 112.0)


def percent_series(
    run: Run, base: str, *, scale: float = 1.0, clip: tuple[float, float] = PERCENT_CLIP
) -> Series | None:
    """Build one series against per cent of that run's own contact window.

    Args:
        run: Run to read.
        base: Channel name without the ``reference_`` prefix.
        scale: Multiplier applied to the channel, for unit changes.
        clip: Percent window kept around contact.

    Returns:
        The series, or ``None`` when the run lacks the channel or a contact window.
    """
    percent, values = run.percent(), run.column(base)
    if percent is None or values is None or percent.size != values.size or percent.size == 0:
        return None
    inside = (percent >= clip[0]) & (percent <= clip[1])
    if not np.any(inside):
        return None
    dash = run.dash or ("2 3" if run.coarse else "")
    return Series(run.label, percent[inside], values[inside] * scale, run.color, dash, markers=run.coarse)


def cumulative_impulse(run: Run) -> Series | None:
    """Integrate vertical force over the run's contact window.

    Args:
        run: Run to read.

    Returns:
        Cumulative impulse [N s] against per cent of contact, or ``None`` when unavailable.
    """
    percent, force = run.percent(), run.column("fz_n")
    if percent is None or force is None or percent.size != force.size or percent.size < 2:
        return None
    if run.contact is None:
        return None
    time_s = np.asarray(run.trace["time_s"], dtype=float)
    inside = (percent >= 0.0) & (percent <= 100.0)
    if np.count_nonzero(inside) < 2:
        return None
    window_time, window_force = time_s[inside], np.nan_to_num(force[inside], nan=0.0)
    steps = np.diff(window_time)
    increments = 0.5 * (window_force[1:] + window_force[:-1]) * steps
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    dash = run.dash or ("2 3" if run.coarse else "")
    return Series(run.label, percent[inside], cumulative, run.color, dash, markers=run.coarse)


def peak_percent(
    time_s: np.ndarray | None, values: np.ndarray | None, window: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Locate the maximum of a channel inside a time window.

    Args:
        time_s: Sample times [s].
        values: Channel values.
        window: Start and end of the window [s].

    Returns:
        The peak value and its position as per cent of the window, or ``None`` when the
        window holds fewer than two samples.
    """
    if time_s is None or values is None or window is None:
        return None
    time_s, values = np.asarray(time_s, dtype=float), np.asarray(values, dtype=float)
    if time_s.size != values.size or time_s.size == 0:
        return None
    inside = (time_s >= window[0]) & (time_s <= window[1]) & np.isfinite(values)
    if np.count_nonzero(inside) < 2:
        return None
    times, window_values = time_s[inside], values[inside]
    # Measure against the declared window, not the first and last sample inside it: an archive
    # states its own contact interval, and the figures put per cent of contact on that same axis.
    start, end = float(window[0]), float(window[1])
    span = end - start
    if not span > 0.0:
        return None
    index = int(np.argmax(window_values))
    return float(window_values[index]), float(times[index] - start) / span * 100.0


def sample_step_statistics(run: Run) -> dict[str, Any] | None:
    """Measure how finely one run resolves its own vertical force inside contact.

    A coarse archive shows large jumps between neighbouring samples simply because the
    samples are far apart. Reporting the spacing beside the jump keeps a sampling artefact
    from being read as contact chatter.

    Args:
        run: Run to read.

    Returns:
        Sample spacing, the largest and median change between neighbouring samples, the share
        of sign reversals and the number of contact transitions, or ``None`` when unavailable.
    """
    if run.trace is None or run.contact is None:
        return None
    time_s, force = run.trace.get("time_s"), run.column("fz_n")
    if time_s is None or force is None or np.size(time_s) != np.size(force) or np.size(force) < 3:
        return None
    time_s, force = np.asarray(time_s, dtype=float), np.asarray(force, dtype=float)
    inside = (time_s >= run.contact[0]) & (time_s <= run.contact[1]) & np.isfinite(force)
    if np.count_nonzero(inside) < 3:
        return None
    steps = np.diff(force[inside])
    signs = np.sign(steps)
    reversals = float(np.mean(signs[1:] * signs[:-1] < 0.0)) * 100.0 if signs.size > 1 else 0.0
    # One per cent of the peak, so the count is scale free and immune to float noise at lift-off.
    loaded = np.isfinite(force) & (force > 0.01 * float(np.max(force[inside])))
    transitions = int(np.count_nonzero(np.diff(loaded.astype(int)) != 0))
    return {
        "sample_interval_ms": (run.sample_interval_s or float("nan")) * 1.0e3,
        "max_step_n": float(np.max(np.abs(steps))),
        "median_step_n": float(np.median(np.abs(steps))),
        "sign_reversal_pct": reversals,
        "contact_transitions": transitions,
        "samples_in_contact": int(np.count_nonzero(inside)),
    }


def run_statistics(run: Run, stance_window: tuple[float, float] | None = None) -> dict[str, Any]:
    """Summarize one run's vertical force under both stance-clock conventions.

    Args:
        run: Run to read.
        stance_window: Measured stance window of the capture trial in trace time [s], the
            fixed clock :mod:`projects.impedance_instron.explain` uses for every run.

    Returns:
        Peak force, timing, impulse, contact duration and provenance, with ``None`` where
        the channel is missing. ``peak_at_pct_contact`` uses each run's own detected contact
        window; ``peak_at_pct_stance_window`` uses the fixed measured stance window, so the
        two conventions can be compared instead of being silently mixed.
    """
    stats: dict[str, Any] = {
        "label": run.label,
        "source": run.source,
        "samples": run.sample_count,
        "contact_s": list(run.contact) if run.contact is not None else None,
        "contact_ms": None,
        "peak_fz_n": None,
        "peak_at_pct_contact": None,
        "reference_peak_at_pct_contact": None,
        "peak_at_pct_stance_window": None,
        "reference_peak_at_pct_stance_window": None,
        "impulse_n_s": None,
    }
    if run.contact is not None:
        stats["contact_ms"] = (run.contact[1] - run.contact[0]) * 1000.0
    impulse = cumulative_impulse(run)
    if impulse is not None and impulse.y.size:
        stats["impulse_n_s"] = float(impulse.y[-1])
    if run.trace is None:
        return stats
    time_s = run.trace.get("time_s")
    force, measured = run.column("fz_n"), run.trace.get("reference_fz_n")
    own = peak_percent(time_s, force, run.contact)
    if own is not None:
        stats["peak_fz_n"], stats["peak_at_pct_contact"] = own
    reference = peak_percent(time_s, measured, run.contact)
    if reference is not None:
        stats["reference_peak_at_pct_contact"] = reference[1]
    fixed = peak_percent(time_s, force, stance_window)
    if fixed is not None:
        stats["peak_at_pct_stance_window"] = fixed[1]
    fixed_reference = peak_percent(time_s, measured, stance_window)
    if fixed_reference is not None:
        stats["reference_peak_at_pct_stance_window"] = fixed_reference[1]
    return stats


# --- Figures ---------------------------------------------------------------------------

# Fixed comparison runs; the policy runs are discovered, so they are appended per figure.
GRF_RUNS = ("measured", "legacy", "command_j")
COM_RUNS = ("measured", "legacy")


def grf_keys(sources: Sources) -> tuple[str, ...]:
    """Return the force-overlay runs: the fixed comparison set plus every discovered policy run."""
    return GRF_RUNS + sources.ankle_keys


def com_keys(sources: Sources) -> tuple[str, ...]:
    """Return the centre-of-mass runs: measured, the legacy schedule and every policy run."""
    return COM_RUNS + sources.ankle_keys


def figure_momentum(_: Sources | None = None) -> str:
    """Figure A: momentum excursion by controller, against the on-task line."""
    colors = {
        "open-loop command J": COLOR_COMMAND,
        "policy, prescribed pitch": COLOR_ACCENT,
        "policy, prescribed, 3x budget": COLOR_ACCENT,
        "policy, prescribed, broken reward (v1)": "#9ca3af",
        "policy with ankle (v2)": COLOR_ANKLE,
    }
    rows = [(label, value, colors.get(label, COLOR_COMMAND), f"{value:.3f}") for label, value in MOMENTUM_EXCURSION]
    panel = bars(
        "Momentum excursion, multiples of the tolerance deadband",
        rows,
        xlabel="excursion [multiples of the 0.044 m/s deadband]",
        band=(PRESCRIBED_PITCH_FLOOR[0], PRESCRIBED_PITCH_FLOOR[1], "prescribed-pitch floor"),
        marker=(0.0, "on task"),
        note="Zero means the tolerance is satisfied; every bar shown is outside the deadband.",
    )
    caption = (
        "Momentum excursion of every controller, in multiples of the 0.044 m/s deadband derived from the "
        "subject's own step-to-step variability (LEGACY_REPORT.md section 3.2). The shaded band is the floor that "
        "prescribed pitch could not cross: three times the training budget moved the excursion the wrong way, "
        f"from {PRESCRIBED_PITCH_FLOOR[0]:.3f} to {PRESCRIBED_PITCH_FLOOR[1]:.3f}. Making pitch a compliant "
        "decision variable reached 0.315, a factor of 4.2 below that floor. Conclude that the compliant ankle "
        "removed a structural limit, and that nothing here is on task, because no bar reaches the line at zero."
    )
    return figure("A", "Momentum excursion by controller", panel, caption)


def peak_timing_comparison(sources: Sources) -> dict[str, dict[str, Any]]:
    """Recompute the peak timings LEGACY_REPORT.md section 3.1 publishes, and check the agreement.

    Both quantities use the shared detected-contact window: a run's own peak, and the measured
    peak located in that same window. A coarse archive is allowed one sample of quantisation on
    top of :data:`PEAK_TIMING_TOLERANCE_PCT`, because one frame is not a disagreement.

    Args:
        sources: Loaded artifacts.

    Returns:
        Mapping from run key to the published pair, the recomputed pair, the tolerance used and
        whether the two agree.
    """
    comparison: dict[str, dict[str, Any]] = {}
    for key, published_run, published_measured in REPORT_PEAK_TIMING_PCT:
        run = sources.run(key)
        if run is None:
            continue
        stats = run_statistics(run)
        own, measured = stats["peak_at_pct_contact"], stats["reference_peak_at_pct_contact"]
        tolerance = PEAK_TIMING_TOLERANCE_PCT
        interval, contact = run.sample_interval_s, stats["contact_ms"]
        if run.coarse and interval is not None and contact:
            tolerance += interval * 1.0e3 / contact * 100.0
        agrees = (
            own is not None
            and measured is not None
            and abs(own - published_run) <= tolerance
            and abs(measured - published_measured) <= tolerance
        )
        comparison[key] = {
            "label": run.label,
            "published_run_peak_pct": published_run,
            "published_measured_peak_pct": published_measured,
            "published_error_points": round(published_run - published_measured, 2),
            "run_peak_pct": own,
            "measured_peak_pct": measured,
            "error_points": None if own is None or measured is None else own - measured,
            "tolerance_pct": tolerance,
            "coarse": bool(run.coarse),
            "agrees": bool(agrees),
        }
    return comparison


def _timing_sentence(sources: Sources) -> str:
    """State the window convention and confirm it against LEGACY_REPORT.md section 3.1.

    Args:
        sources: Loaded artifacts.

    Returns:
        A sentence naming the shared threshold, and a second sentence reporting whether the
        recomputed peak timings still match the published ones.
    """
    threshold = sources.threshold_n
    if threshold is None:
        return (
            "No stance profile was found, so no body weight and no contact threshold could be derived; "
            "the archived contact windows are used where they exist."
        )
    sentence = (
        f"Contact is defined once, as shoe force above CONTACT_FORCE_FRACTION = {CONTACT_FORCE_FRACTION} of "
        f"body weight ({threshold:.2f} N from the profile's {sources.profile['mass_kg']:.2f} kg and "
        f"{sources.profile['gravity_m_s2']} m/s squared), the same constant the dense momentum reward's "
        "touchdown detector, the rollout's stance gate and the open-loop solver share, and the window "
        "LEGACY_REPORT.md section 3.1 reports peak timing on."
    )
    comparison = peak_timing_comparison(sources)
    if not comparison:
        return sentence
    agreeing, differing = [], []
    for entry in comparison.values():
        if entry["run_peak_pct"] is None:
            continue
        name = entry["label"].split(" (")[0]
        quantised = (
            f" ({entry['run_peak_pct']:.2f} from this coarse archive, within its own "
            f"{entry['tolerance_pct']:.1f} point sample spacing)"
            if entry["coarse"]
            else ""
        )
        text = f"{name} {entry['published_run_peak_pct']:.2f} against {entry['published_measured_peak_pct']:.2f}{quantised}"
        (agreeing if entry["agrees"] else differing).append(text)
    parts = [sentence]
    if agreeing:
        parts.append(
            "Recomputed here, the published pairs hold: " + "; ".join(agreeing) + ". The open-loop command is "
            "closest on peak timing, not the policy."
        )
    if differing:
        parts.append("These do NOT reproduce and should be checked: " + "; ".join(differing) + ".")
    return " ".join(parts)


def _resolution_sentence(runs: list[Run]) -> str:
    """State how finely each series is sampled, and what that does and does not show.

    Args:
        runs: Runs drawn on one set of axes, in drawing order.

    Returns:
        A sentence naming the sample counts, the rendering used for a coarse series and, at
        full resolution, the measured smoothness of the force it resolves.
    """
    counted = [run for run in runs if run.sample_count]
    if not counted:
        return ""
    richest = max(run.sample_count for run in counted)
    coarse = [run for run in counted if run.coarse]
    if not coarse:
        parts = [
            f"Every series is drawn at its native resolution ({richest} substeps), and nothing was resampled or "
            "interpolated."
        ]
        # Prefer the archived policy run: it is the series a coarse archive used to distort.
        ordered = sorted(counted, key=lambda run: (run.reference, not run.source.endswith(".npz")))
        for run in ordered:
            steps = sample_step_statistics(run)
            if steps is None or run.reference:
                continue
            parts.append(
                f"Inside contact the {run.label.split(' (')[0]} force changes by at most {steps['max_step_n']:.1f} N "
                f"between neighbouring samples {steps['sample_interval_ms']:.3f} ms apart, median "
                f"{steps['median_step_n']:.2f} N, with {steps['sign_reversal_pct']:.1f} % sign reversals and "
                f"{steps['contact_transitions']} contact transitions, so the waveform is resolved and smooth "
                "rather than chattering."
            )
            break
        return " ".join(parts)
    run = coarse[0]
    steps = sample_step_statistics(run)
    spacing = f"{steps['sample_interval_ms']:.1f} ms" if steps else "its own frame spacing"
    jump = f"{steps['max_step_n']:.0f} N" if steps else "a large value"
    return (
        f"The {run.label.split(' (')[0]} archive holds {run.sample_count} samples against {richest} substeps for "
        f"the other series, so it is drawn DASHED WITH ITS SAMPLES MARKED rather than as a solid line, and nothing "
        f"was resampled or interpolated. Neighbouring archive samples are {spacing} apart and differ by up to "
        f"{jump}: that step is a sampling artefact of the spacing, not evidence about contact, because at this "
        "spacing the archive cannot resolve contact behaviour either way."
    )


def figure_vertical_grf(sources: Sources) -> str:
    """Figure B: vertical ground reaction force over contact, four controllers."""
    runs = sources.available(grf_keys(sources), "fz_n")
    if not runs:
        return placeholder("B", "Vertical ground reaction force", "no trace carries a vertical force channel")
    series = [item for item in (percent_series(run, "fz_n") for run in runs) if item is not None]
    panel = plot(
        "Vertical ground reaction force",
        series,
        xlabel="per cent of contact [%]",
        ylabel="vertical force [N]",
        note=(
            "Each series uses its own contact window, detected with the shared "
            f"{CONTACT_FORCE_FRACTION:g} of body weight, so the curves align on touchdown and toe-off."
            + (
                " A dashed series with drawn samples is coarser than the others; the dashes mean the line "
                "between its samples is not measured."
                if any(run.coarse for run in runs)
                else ""
            )
        ),
    )
    caption = (
        "Vertical force through contact, with each curve placed on per cent of its own contact window. "
        f"{_timing_sentence(sources)} {_resolution_sentence(runs)} The legacy schedule overshoots the measured "
        "peak; the open-loop command and the closed-loop policy both land near it. Conclude that the policy "
        "matches the measured waveform without ever being given the measured force, which the legacy controller "
        "receives as feedforward."
    )
    return figure("B", "Vertical ground reaction force", panel, caption)


def figure_foreaft_grf(sources: Sources) -> str:
    """Figure C: fore-aft ground reaction force over contact."""
    runs = sources.available(grf_keys(sources), "fx_n")
    if not runs:
        return placeholder("C", "Fore-aft ground reaction force", "no trace carries a fore-aft force channel")
    series = [item for item in (percent_series(run, "fx_n") for run in runs) if item is not None]
    missing = [key for key in grf_keys(sources) if key not in {run.key for run in runs}]
    note = "Braking is negative, propulsion positive."
    if missing:
        note += f" Missing fore-aft channel: {', '.join(missing)}."
    panel = plot(
        "Fore-aft ground reaction force",
        series,
        xlabel="per cent of contact [%]",
        ylabel="fore-aft force [N]",
        note=note,
    )
    caption = (
        "Fore-aft force over the same contact windows, for every run that records the channel. The braking and "
        "propulsion phases are where the controllers differ most, because the fore-aft axis is not part of any "
        "tolerance. Conclude that agreement on the vertical axis does not imply agreement here, and that the "
        "fore-aft response should be reported separately in any material comparison."
    )
    return figure("C", "Fore-aft ground reaction force", panel, caption)


def figure_com(sources: Sources) -> str:
    """Figure D: centre-of-mass height and vertical velocity."""
    height_runs = sources.available(com_keys(sources), "com_z_m")
    velocity_runs = sources.available(com_keys(sources), "com_vz_m_s")
    if not height_runs and not velocity_runs:
        return placeholder("D", "Centre-of-mass trajectory", "no trace carries the centre-of-mass channels")
    panels = plot(
        "COM height",
        [item for item in (percent_series(run, "com_z_m", scale=1000.0) for run in height_runs) if item is not None],
        xlabel="per cent of contact [%]",
        ylabel="COM height [mm]",
        width=520,
    ) + plot(
        "COM vertical velocity",
        [item for item in (percent_series(run, "com_vz_m_s") for run in velocity_runs) if item is not None],
        xlabel="per cent of contact [%]",
        ylabel="COM vertical velocity [m/s]",
        width=520,
        hlines=((0.0, "zero crossing", COLOR_RULE),),
    )
    caption = (
        "Centre-of-mass height and vertical velocity, measured against the legacy schedule and the closed-loop "
        "policy with the compliant ankle. Height errors are tens of millimetres for both controllers, but the "
        "velocity panel is the one the task tolerance reads: the policy holds 0.141 m/s RMS against the legacy "
        "controller's 0.160 m/s (LEGACY_REPORT.md section 3.1). Conclude that the remaining task gap lives in the "
        "velocity history, not in the height."
    )
    return figure("D", "Centre-of-mass height and vertical velocity", panels, caption)


def figure_impulse(sources: Sources) -> str:
    """Figure E: cumulative vertical impulse over contact."""
    runs = sources.available(grf_keys(sources), "fz_n")
    series = [item for item in (cumulative_impulse(run) for run in runs) if item is not None]
    if not series:
        return placeholder("E", "Cumulative vertical impulse", "no trace has both a contact window and a force channel")
    endpoints = ", ".join(f"{item.label.split(' (')[0]} {item.y[-1]:.1f} N s" for item in series)
    panel = plot(
        "Cumulative vertical impulse",
        series,
        xlabel="per cent of contact [%]",
        ylabel="cumulative impulse [N s]",
        note="Trapezoidal integration of vertical force from touchdown to toe-off.",
    )
    caption = (
        "Vertical force integrated from touchdown. The endpoints agree to within a few per cent while the paths "
        f"through stance differ ({endpoints}). That is the whole difficulty of the task specification: impulse "
        "and duration are integrals a controller can satisfy by scaling force magnitude, but momentum is a "
        "history, and two controllers with the same endpoint can take visibly different routes to it. Conclude "
        "that an endpoint check cannot replace the momentum-history check."
    )
    return figure("E", "Cumulative vertical impulse", panel, caption)


def _material_series(grouped: dict[str | None, list[dict]], field_name: str, suffix: str = "") -> list[Series]:
    """Build one series per material out of grouped evaluation records.

    Args:
        grouped: Records keyed by material token, from :func:`group_by_material`.
        field_name: Evaluation field to plot against iteration.
        suffix: Text appended to each label, for a panel that needs a qualifier.

    Returns:
        One series per material, in the order the materials first appear.
    """
    series: list[Series] = []
    single = len(grouped) == 1
    for index, (token, records) in enumerate(grouped.items()):
        iterations = np.array([record["iteration"] for record in records], dtype=float)
        values = np.array([record.get(field_name, float("nan")) for record in records], dtype=float)
        if token is None:
            label = "policy with ankle (v2)" if single else "records without a material token"
        else:
            label = f"material {token}"
        series.append(Series(f"{label}{suffix}", iterations, values, MATERIAL_COLORS[index % len(MATERIAL_COLORS)]))
    return series


def figure_training(sources: Sources) -> str:
    """Figure F: evaluated momentum excursion and work objective against iteration."""
    records = sources.evaluations
    if not records:
        return placeholder(
            "F",
            "Training curves",
            "outputs/impedance_instron/train_ankle_v2.log has no parsable evaluation records",
        )
    grouped = sources.materials or group_by_material(records)
    momentum_lines = tuple(
        (value, f"{label} (final {value:.3f})", COLOR_ACCENT) for label, value, _ in PRESCRIBED_FINALS
    )
    work_lines = tuple(
        (work, f"{label} (final {work:.0f} J, leg only)", COLOR_ACCENT) for label, _, work in PRESCRIBED_FINALS
    )
    tokens = [token for token in grouped if token is not None]
    material_note = (
        f" One curve per material, {len(grouped)} in this log; records are grouped by their own token, never merged."
        if len(grouped) > 1
        else ""
    )
    panels = plot(
        "Momentum excursion during training",
        _material_series(grouped, "excursion_momentum"),
        xlabel="iteration",
        ylabel="momentum excursion [multiples of deadband]",
        hlines=momentum_lines,
        width=520,
        note="Dashed lines are final values of the prescribed-pitch runs, which predate evaluation logging."
        + material_note,
    ) + plot(
        "Work objective during training",
        _material_series(grouped, "objective_j", suffix=", both actuators"),
        xlabel="iteration",
        ylabel="work proxy [J]",
        hlines=work_lines,
        width=520,
        note="The v2 objective charges both actuators; the reference lines charge the leg only." + material_note,
    )
    counted = ", ".join(
        f"{name} {sources.eval_counts[name]}"
        for name in ("train_v3.log", "train_v5.log")
        if name in sources.eval_counts
    )
    absent = f" Evaluation records found in those logs: {counted}." if counted else ""
    if len(grouped) > 1:
        named = ", ".join(sorted(token for token in tokens))
        material = (
            f" The log carries {len(grouped)} materials ({named}), so each is drawn as its own curve: a frozen "
            "policy run on different foams is a comparison BETWEEN materials, and averaging them would destroy "
            "exactly the difference the experiment measures."
        )
    elif tokens:
        material = (
            f" Every record carries the material token {tokens[0]}, which hashes the material the environment "
            "actually simulated rather than the artifact path, so a stiffness-scaled rerun of the same file "
            "would be a different curve here."
        )
    else:
        material = (
            " These records predate the material token, so the curve is attributable to this policy but not, from "
            "the log alone, to a named material."
        )
    caption = (
        f"Deterministic evaluations of the compliant-ankle run, {len(records)} records parsed from "
        "train_ankle_v2.log with the same defensive key=value parser the live dashboard uses. The v1 to v5 "
        "prescribed-pitch runs predate evaluation logging and have no eval lines at all, so their FINAL values "
        f"are drawn as horizontal reference lines instead of curves.{absent}{material} The work reference lines "
        "are leg-only and are therefore not comparable term by term with the v2 curve, which charges the ankle as "
        "well. Conclude that the excursion fell below the prescribed-pitch floor early and stayed there, and that "
        "the later budget bought efficiency, not task accuracy."
    )
    return figure("F", "Training curves", panels, caption)


def figure_stiff_limit(_: Sources | None = None) -> str:
    """Figure G: convergence of the compliant ankle onto the prescribed-pitch run."""
    k_theta = np.array(STIFF_LIMIT_K_THETA, dtype=float)
    colors = (COLOR_ANKLE, COLOR_COMMAND, COLOR_LEGACY)
    series = [
        Series(label, k_theta, np.array(values, dtype=float), color)
        for (label, values), color in zip(STIFF_LIMIT_SERIES, colors, strict=False)
    ]
    # A slope of -1 through the first pitch point is "one decade per decade" as a straight line.
    anchor = STIFF_LIMIT_SERIES[0][1][0] * k_theta[0]
    series.append(Series("reference slope -1", k_theta, anchor / k_theta, COLOR_RULE, dash="6 4", width=1.5))
    panel = plot(
        "Stiff-limit convergence",
        series,
        xlabel="ankle stiffness k_theta [N m/rad]",
        ylabel="relative difference from the prescribed run [-]",
        xlog=True,
        ylog=True,
        note="Both axes are base-10 logarithmic, so a slope of -1 is one decade removed per decade added.",
    )
    caption = (
        "Relative difference between the compliant-ankle formulation and the prescribed-pitch one, as ankle "
        "stiffness is raised (LEGACY_REPORT.md section 2.1). All three channels track the dashed slope of -1, so each "
        "decade of stiffness removes a decade of difference, down to float32 noise near 3e6 N m/rad. Conclude "
        "that the new formulation provably CONTAINS the old one as its stiff limit, so the comparison between "
        "them is a change of freedom, not a change of model."
    )
    return figure("G", "Stiff-limit convergence", panel, caption)


def figure_impedance(sources: Sources) -> str:
    """Figure H: the impedance the closed-loop policy actually commanded."""
    run = sources.run("ankle_v2")
    if run is None or run.trace is None:
        return placeholder("H", "Commanded impedance", "outputs/impedance_instron/policy_ankle_v2.eval.npz is missing")
    stiffness = percent_series(run, "stiffness_n_m")
    damping = percent_series(run, "damping_ratio")
    commanded = percent_series(run, "commanded_length_m", scale=1000.0)
    achieved = percent_series(run, "leg_length_m", scale=1000.0)
    if stiffness is None and damping is None and commanded is None:
        return placeholder("H", "Commanded impedance", "the evaluation archive carries no impedance channels")
    panels = (
        plot(
            "Commanded leg stiffness K(t)",
            [stiffness] if stiffness is not None else [],
            xlabel="per cent of contact [%]",
            ylabel="stiffness [N/m]",
            width=440,
            height=300,
        )
        + plot(
            "Commanded damping ratio zeta(t)",
            [damping] if damping is not None else [],
            xlabel="per cent of contact [%]",
            ylabel="damping ratio [-]",
            width=440,
            height=300,
        )
        + plot(
            "Commanded L0(t) against achieved leg length",
            [
                item
                for item in (
                    commanded._replace(label="commanded L0(t)", color=COLOR_ACCENT) if commanded is not None else None,
                    achieved._replace(label="achieved leg length", color=COLOR_ANKLE) if achieved is not None else None,
                )
                if item is not None
            ],
            xlabel="per cent of contact [%]",
            ylabel="length [mm]",
            width=440,
            height=300,
        )
    )
    caption = (
        "The impedance the frozen policy chose, read back out of its own evaluation archive. Stiffness and "
        "damping ratio are on separate panels because they cannot share a linear axis: K runs in the thousands "
        "of N/m while zeta is a fraction. The third panel shows the commanded equilibrium length held ABOVE the "
        "achieved leg length through loading, which is what makes the unilateral leg push, and closing on it "
        "near toe-off, which is what ends stance without any release schedule. Conclude that the residual "
        "action left the command interpretable: these are the quantities that would be compared across "
        "materials."
    )
    return figure("H", "Commanded impedance", panels, caption)


def figure_materials(_: Sources | None = None) -> str:
    """Figure I: equilibrium pressure of both published foams against this shoe."""
    try:
        # Deferred: mcclough reaches the shoe runtime, which pulls in Warp, and this page is
        # pure NumPy so that it renders on a machine with no GPU stack.
        from projects.impedance_instron import mcclough  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - exercised only without the project package
        return placeholder("I", "Material comparison", f"projects.impedance_instron.mcclough is unavailable ({error})")
    strain = np.linspace(MATERIAL_STRAIN_MIN, MATERIAL_STRAIN_MAX, 240)
    shoe = np.asarray(mcclough.equilibrium_pressure_pa(mcclough.REFERENCE_SHOE, strain), dtype=float)
    ratios: list[Series] = []
    for (name, material), color in zip(mcclough.materials().items(), (COLOR_COMMAND, COLOR_LEGACY), strict=False):
        pressure = np.asarray(mcclough.equilibrium_pressure_pa(material, strain), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(shoe) & (shoe > 0.0), pressure / shoe, np.nan)
        ratios.append(Series(name.replace("_", " "), strain * 100.0, ratio, color))
    panel = plot(
        "Equilibrium pressure ratio, foam over this shoe",
        ratios,
        xlabel="compressive strain [%]",
        ylabel="pressure ratio, foam / our shoe [-]",
        hlines=((1.0, "equal response", COLOR_RULE),),
        note="Equilibrium Ogden-Hill pressure evaluated by projects.impedance_instron.mcclough.",
    )
    caption = (
        "Both published McClough foams divided by this project's shoe, over the strains a stance actually "
        "visits. Our shoe is 10-20 % STIFFER than both below 5 % strain and 1.9-2.4x SOFTER above 30 % strain, "
        "so the two curves cross the line of equal response. That is a change of curve SHAPE, driven by the "
        "first Ogden-Hill exponent (18.08 here against 7.75 and 4.48). Conclude that a modulus-only "
        "randomisation band, at any width, cannot reproduce this shape change: the band must vary the "
        "EXPONENTS, or a frozen policy will extrapolate on both foams through midstance."
    )
    return figure("I", "Material comparison", panel, caption)


def figure_infrastructure(_: Sources | None = None) -> str:
    """Figure J: the three measured infrastructure speedups."""
    rows: list[tuple[str, float, str, str]] = []
    for label, unit, before, after in INFRASTRUCTURE:
        rows.append((f"{label}, before", before, "#9ca3af", f"{before:g} {unit}"))
        rows.append((f"{label}, after", after, COLOR_ANKLE, f"{after:g} {unit} ({before / after:.1f}x)"))
    panel = bars(
        "Runtime cost before and after, logarithmic axis",
        rows,
        xlabel="cost [logarithmic, units differ per pair; see the labels]",
        log=True,
        note="Each pair shares one unit; the pairs do not. The axis compares magnitudes, not units.",
    )
    caption = (
        "The three measured infrastructure changes of LEGACY_REPORT.md section 5: CUDA graph replay took a rollout "
        "from 6.7 s to 0.517 s, the world-batched foundation substep went from 0.170 ms to 0.077 ms at 64 "
        "worlds, and the deterministic contact-metric block went from 49 us to 10 us. The axis is logarithmic "
        "and each pair carries its own unit, so compare within a pair, not across pairs. Conclude that the "
        "workload was launch-bound rather than compute-bound, and that the reduction change mattered more for "
        "reproducibility than for the 2.3x it also bought."
    )
    return figure("J", "Infrastructure cost", panel, caption)


FIGURE_BUILDERS: tuple[tuple[str, str, Any], ...] = (
    ("A", "Momentum excursion by controller", figure_momentum),
    ("B", "Vertical ground reaction force", figure_vertical_grf),
    ("C", "Fore-aft ground reaction force", figure_foreaft_grf),
    ("D", "Centre-of-mass height and vertical velocity", figure_com),
    ("E", "Cumulative vertical impulse", figure_impulse),
    ("F", "Training curves", figure_training),
    ("G", "Stiff-limit convergence", figure_stiff_limit),
    ("H", "Commanded impedance", figure_impedance),
    ("I", "Material comparison", figure_materials),
    ("J", "Infrastructure cost", figure_infrastructure),
)

# Which section of LEGACY_REPORT.md each figure belongs under, by heading text.
FIGURE_PLACEMENT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Impedance Instron: controller rebuild and closed-loop policy", ("A",)),
    ("2.1 Equilibrium-point impedance, both axes", ("G",)),
    ("2.3 Closed-loop policy", ("H",)),
    ("3.1 Against the measured stance", ("B", "C", "D", "E")),
    ("3.2 Task tolerances", ("F",)),
    ("5. Infrastructure", ("J",)),
    ("6. Material mapping, and what it cannot do", ("I",)),
)


def build_figures(sources: Sources) -> dict[str, str]:
    """Render every figure, substituting a placeholder for any that cannot be drawn.

    Args:
        sources: Loaded artifacts.

    Returns:
        Mapping from figure letter to its rendered markup.
    """
    rendered: dict[str, str] = {}
    for identifier, title, builder in FIGURE_BUILDERS:
        try:
            rendered[identifier] = builder(sources)
        except Exception as error:  # pragma: no cover - defensive; a figure must not fail the page
            rendered[identifier] = placeholder(identifier, title, f"rendering failed: {error}")
    return rendered


# --- The Markdown subset LEGACY_REPORT.md uses ------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_RULE_RE = re.compile(r"^-{3,}\s*$")
_UNORDERED_RE = re.compile(r"^[*-]\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\d+)\.\s+(.*)$")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
_ALIGNMENT_CELL_RE = re.compile(r"^:?-{2,}:?$")


def render_inline(text: str) -> str:
    """Render the inline Markdown LEGACY_REPORT.md uses: bold and code spans.

    Args:
        text: Raw inline Markdown.

    Returns:
        Escaped HTML with ``b`` and ``code`` elements.
    """
    rendered = _escape(text)
    rendered = _CODE_SPAN_RE.sub(lambda match: f"<code>{match.group(1)}</code>", rendered)
    return _BOLD_RE.sub(lambda match: f"<b>{match.group(1)}</b>", rendered)


def _split_row(line: str) -> list[str]:
    """Split one Markdown table row into its cells.

    Args:
        line: Row text, with or without the outer pipes.
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _alignments(cells: list[str]) -> list[str] | None:
    """Read a Markdown alignment row.

    Args:
        cells: Cells of the candidate alignment row.

    Returns:
        One CSS alignment per column, or ``None`` when the row is not an alignment row.
    """
    if not cells or not all(_ALIGNMENT_CELL_RE.match(cell) for cell in cells):
        return None
    alignments = []
    for cell in cells:
        left, right = cell.startswith(":"), cell.endswith(":")
        alignments.append("center" if left and right else "right" if right else "left")
    return alignments


def parse_blocks(text: str) -> list[dict]:
    """Parse the Markdown subset LEGACY_REPORT.md uses into blocks.

    Supported: ATX headings, paragraphs, bullet and numbered lists, pipe tables with an
    alignment row, four-space indented code blocks and horizontal rules. Anything else is
    kept as paragraph text rather than dropped.

    Args:
        text: Full Markdown document.

    Returns:
        Blocks, each a dictionary carrying a ``type`` and its payload.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[dict] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            blocks.append({"type": "heading", "level": len(heading.group(1)), "text": heading.group(2)})
            index += 1
            continue
        if _RULE_RE.match(line):
            blocks.append({"type": "rule"})
            index += 1
            continue
        if line.startswith("    "):
            code: list[str] = []
            while index < len(lines) and (lines[index].startswith("    ") or not lines[index].strip()):
                if not lines[index].strip() and not (index + 1 < len(lines) and lines[index + 1].startswith("    ")):
                    break
                code.append(lines[index][4:] if lines[index].startswith("    ") else "")
                index += 1
            blocks.append({"type": "code", "text": "\n".join(code).strip("\n")})
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                rows.append(_split_row(lines[index]))
                index += 1
            alignments = _alignments(rows[1]) if len(rows) > 1 else None
            if alignments is not None:
                blocks.append({"type": "table", "header": rows[0], "align": alignments, "rows": rows[2:]})
            else:
                blocks.append({"type": "paragraph", "text": " ".join(" | ".join(row) for row in rows)})
            continue
        unordered, ordered = _UNORDERED_RE.match(line), _ORDERED_RE.match(line)
        if unordered or ordered:
            items: list[str] = []
            is_ordered = ordered is not None
            while index < len(lines):
                current = lines[index]
                if not current.strip():
                    index += 1
                    if index < len(lines) and not (
                        _UNORDERED_RE.match(lines[index])
                        or _ORDERED_RE.match(lines[index])
                        or lines[index].startswith(" ")
                    ):
                        break
                    continue
                item = _ORDERED_RE.match(current) if is_ordered else _UNORDERED_RE.match(current)
                if item:
                    items.append(item.group(2) if is_ordered else item.group(1))
                elif current.startswith(" ") and items:
                    items[-1] = f"{items[-1]} {current.strip()}"
                else:
                    break
                index += 1
            blocks.append({"type": "list", "ordered": is_ordered, "items": items})
            continue
        paragraph: list[str] = []
        while index < len(lines) and lines[index].strip():
            current = lines[index]
            if (
                _HEADING_RE.match(current)
                or _RULE_RE.match(current)
                or current.lstrip().startswith("|")
                or _UNORDERED_RE.match(current)
                or _ORDERED_RE.match(current)
                or current.startswith("    ")
            ):
                break
            paragraph.append(current.strip())
            index += 1
        if paragraph:
            blocks.append({"type": "paragraph", "text": " ".join(paragraph)})
        else:  # pragma: no cover - only reachable if a guard above changes
            index += 1
    return blocks


def slug(text: str) -> str:
    """Make a stable anchor from a heading.

    Args:
        text: Heading text.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "section"


def render_block(block: dict) -> str:
    """Render one parsed Markdown block as HTML.

    Args:
        block: Block from :func:`parse_blocks`.
    """
    kind = block["type"]
    if kind == "heading":
        level = min(max(int(block["level"]), 1), 6)
        return f'<h{level} id="{_escape(slug(block["text"]))}">{render_inline(block["text"])}</h{level}>'
    if kind == "rule":
        return "<hr></hr>"
    if kind == "code":
        return f"<pre><code>{_escape(block['text'])}</code></pre>"
    if kind == "list":
        tag = "ol" if block["ordered"] else "ul"
        items = "".join(f"<li>{render_inline(item)}</li>" for item in block["items"])
        return f"<{tag}>{items}</{tag}>"
    if kind == "table":
        align = block["align"]

        def cells(row: list[str], tag: str) -> str:
            rendered = []
            for column, cell in enumerate(row):
                style = align[column] if column < len(align) else "left"
                rendered.append(f'<{tag} style="text-align:{style}">{render_inline(cell)}</{tag}>')
            return "".join(rendered)

        head = f"<thead><tr>{cells(block['header'], 'th')}</tr></thead>"
        body = "".join(f"<tr>{cells(row, 'td')}</tr>" for row in block["rows"])
        return f'<div class="scroll"><table>{head}<tbody>{body}</tbody></table></div>'
    return f"<p>{render_inline(block['text'])}</p>"


def render_report(blocks: list[dict], figures: dict[str, str]) -> tuple[str, list[str]]:
    """Render the report with each figure placed inside the section it illustrates.

    A figure is emitted at the end of its anchor section, that is just before the next
    heading or horizontal rule. A figure whose anchor heading is absent is appended to the
    end of the page rather than dropped.

    Args:
        blocks: Parsed Markdown blocks.
        figures: Rendered figures keyed by letter.

    Returns:
        The page body and the letters of any figure that could not be placed in a section.
    """
    placement = dict(FIGURE_PLACEMENT)
    headings = {block["text"] for block in blocks if block["type"] == "heading"}
    unplaced = [letter for heading, letters in FIGURE_PLACEMENT for letter in letters if heading not in headings]
    parts: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        while pending:
            letter = pending.pop(0)
            if letter in figures:
                parts.append(figures[letter])

    for block in blocks:
        if block["type"] in ("heading", "rule"):
            flush()
        parts.append(render_block(block))
        if block["type"] == "heading" and block["text"] in placement:
            pending = [letter for letter in placement[block["text"]] if letter in figures]
    flush()
    if unplaced:
        parts.append('<h2 id="unplaced-figures">Figures without a matching section</h2>')
        parts.extend(figures[letter] for letter in unplaced if letter in figures)
    return "".join(parts), unplaced


STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 32px 24px 64px; max-width: 1180px; background: #f7f8fa; color: #111827;
  font: 16px/1.62 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size: 30px; margin: 8px 0 6px; }
h2 { font-size: 23px; margin: 34px 0 10px; border-bottom: 2px solid #dbe1ea; padding-bottom: 6px; }
h3 { font-size: 19px; margin: 26px 0 8px; }
h4 { font-size: 17px; margin: 22px 0 8px; }
.panel h4 { font-size: 14px; margin: 0 0 6px; font-weight: 600; color: #1f2937; }
p { margin: 10px 0; }
hr { border: 0; border-top: 1px solid #dbe1ea; margin: 26px 0; }
code { background: #eef1f6; border-radius: 4px; padding: 1px 4px; font-size: 0.92em; }
pre { background: #111827; color: #e5e7eb; padding: 12px 16px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; color: inherit; padding: 0; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }
th, td { border: 1px solid #dbe1ea; padding: 6px 10px; }
th { background: #eef1f6; }
.scroll { overflow-x: auto; margin: 12px 0; }
figure { margin: 22px 0; padding: 16px; background: #fff; border: 1px solid #dbe1ea; border-radius: 10px; }
figcaption { margin-top: 10px; font-size: 14px; color: #374151; }
.panels { display: flex; flex-wrap: wrap; gap: 14px; }
.panel { flex: 1 1 420px; min-width: 320px; }
.panel.missing { border: 1px dashed #b91c1c; border-radius: 8px; padding: 14px; }
.miss { color: #b91c1c; font-size: 14px; }
.note { margin: 0 0 6px; font-size: 12.5px; color: #6b7280; }
svg { width: 100%; height: auto; font: 11px -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  fill: #4b5563; }
svg .grid { stroke: #e5e9f0; fill: none; }
svg .axis { font-size: 12px; fill: #374151; }
svg .band { font-size: 11px; fill: #92400e; }
svg .value { font-size: 11px; fill: #1f2937; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px; font-size: 12.5px; color: #374151; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 5px;
  vertical-align: -1px; }
.provenance { font-size: 13px; color: #4b5563; background: #fff; border: 1px solid #dbe1ea;
  border-radius: 10px; padding: 12px 16px; }
.provenance ul { margin: 6px 0 0; padding-left: 20px; }
"""


def _archive_provenance(sources: Sources) -> str:
    """List which evaluation archive each policy curve was read from.

    Args:
        sources: Loaded artifacts.

    Returns:
        An HTML list, or an empty string when no policy archive was found.
    """
    items = []
    for run in sources.runs:
        if not run.key.startswith("ankle_v2"):
            continue
        material = f", material {run.material}" if run.material else ", no material token"
        items.append(
            f"<li><b>{_escape(run.key)}</b> read <code>{_escape(run.source)}</code> "
            f"({run.sample_count} samples{_escape(material)})</li>"
        )
    for entry in sources.superseded:
        items.append(
            f"<li>not drawn: <code>{_escape(entry['path'])}</code>, superseded by a tokenised archive "
            "that names the material it was run on</li>"
        )
    if not items:
        return ""
    return f"<p>Evaluation archives, discovered rather than assumed:</p><ul>{''.join(items)}</ul>"


def _material_provenance(sources: Sources) -> str:
    """Describe which material produced the evaluation records, for the provenance block.

    Args:
        sources: Loaded artifacts.
    """
    grouped = sources.materials or {}
    tokens = sorted(token for token in grouped if token)
    if len(tokens) > 1:
        return f"Evaluation records cover {len(tokens)} materials ({', '.join(tokens)}), drawn as separate curves."
    if tokens:
        return f"All evaluation records carry the material token {tokens[0]}."
    return "No evaluation record names a material, so these runs predate the material token."


def _is_legacy_report(blocks: list[dict]) -> bool:
    """Recognize the historical title before attaching its scientific figures."""
    title = next((block["text"] for block in blocks if block["type"] == "heading" and block["level"] == 1), None)
    return title == FIGURE_PLACEMENT[0][0]


def render_page(report_text: str, figures: dict[str, str], sources: Sources) -> str:
    """Render Markdown, illustrating only the historical momentum/work report.

    A document without the historical report title keeps its text but receives no
    legacy figures. This also applies when the current ``REPORT.md`` is supplied.

    Args:
        report_text: Raw LEGACY_REPORT.md text.
        figures: Rendered figures keyed by letter.
        sources: Loaded artifacts, used for the provenance block.

    Returns:
        A complete, self-contained HTML document.
    """
    blocks = parse_blocks(report_text)
    legacy = _is_legacy_report(blocks)
    if legacy:
        body, unplaced = render_report(blocks, figures)
    else:
        body, unplaced = "".join(render_block(block) for block in blocks), []
    title = next((block["text"] for block in blocks if block["type"] == "heading"), "Impedance Instron report")
    listed = "".join(
        f"<li><b>Figure {_escape(letter)}</b>: {_escape(name)}</li>" for letter, name, _ in FIGURE_BUILDERS
    )
    missing = "".join(f"<li>{_escape(name)}: {_escape(path)}</li>" for name, path in sorted(sources.missing.items()))
    provenance = (
        '<section class="provenance"><b>Figures and their data</b>'
        f"<ul>{listed}</ul>"
        f"<p>Data directory: <code>{_escape(sources.directory)}</code>. {_escape(_material_provenance(sources))} "
        "Every number the figures plot is also written to <code>summary.json</code>.</p>"
        + _archive_provenance(sources)
        + (f"<p>Missing artifacts, drawn as placeholders:</p><ul>{missing}</ul>" if missing else "")
        + (f"<p>Figures with no matching section: {_escape(', '.join(unplaced))}.</p>" if unplaced else "")
        + "</section>"
    )
    if legacy:
        notice = (
            '<p class="provenance"><b>Historical momentum/work experiments.</b> '
            "These figures do not describe the current two-stiffness workflow. "
            "The fixed tables are historical claims, not a new validation of the supplied artifacts.</p>"
        )
    else:
        notice = (
            '<p class="provenance"><b>Legacy figures omitted.</b> '
            "This document is not the historical momentum/work report. "
            "Use LEGACY_REPORT.md and its matching archived artifacts for those figures.</p>"
        )
        provenance = ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title)}</title><style>{STYLE}</style></head><body>"
        f"{notice}{body}{provenance}</body></html>\n"
    )


# --- JSON payload ----------------------------------------------------------------------


def _finite_list(values: np.ndarray, digits: int = 6) -> list[float | None]:
    """Convert an array to JSON numbers, mapping anything non-finite to ``null``.

    Args:
        values: Array to convert.
        digits: Decimal places kept.
    """
    return [round(float(value), digits) if math.isfinite(float(value)) else None for value in np.ravel(values)]


def _series_payload(series: Series | None) -> dict[str, Any] | None:
    """Serialize one plotted series after the same thinning the figure uses.

    Args:
        series: Series to serialize, or ``None``.
    """
    if series is None:
        return None
    x, y = downsample(series.x, series.y)
    return {"label": series.label, "x": _finite_list(x), "y": _finite_list(y)}


def summary_payload(sources: Sources, figures: dict[str, str]) -> dict[str, Any]:
    """Collect every number the figures plot.

    Args:
        sources: Loaded artifacts.
        figures: Rendered figures keyed by letter, used to flag placeholders.

    Returns:
        A JSON-serializable dictionary.
    """
    stance_window = sources.stance_window_s
    runs = {run.key: run_statistics(run, stance_window) for run in sources.runs}
    waveforms: dict[str, Any] = {}
    for run in sources.runs:
        waveforms[run.key] = {
            "fz_n": _series_payload(percent_series(run, "fz_n")),
            "fx_n": _series_payload(percent_series(run, "fx_n")),
            "com_z_mm": _series_payload(percent_series(run, "com_z_m", scale=1000.0)),
            "com_vz_m_s": _series_payload(percent_series(run, "com_vz_m_s")),
            "cumulative_impulse_n_s": _series_payload(cumulative_impulse(run)),
        }
    ankle = sources.run("ankle_v2")
    impedance = {}
    if ankle is not None:
        impedance = {
            "stiffness_n_m": _series_payload(percent_series(ankle, "stiffness_n_m")),
            "damping_ratio": _series_payload(percent_series(ankle, "damping_ratio")),
            "commanded_length_mm": _series_payload(percent_series(ankle, "commanded_length_m", scale=1000.0)),
            "leg_length_mm": _series_payload(percent_series(ankle, "leg_length_m", scale=1000.0)),
        }
    materials: dict[str, Any] = {}
    try:
        from projects.impedance_instron import mcclough  # noqa: PLC0415  # see figure_materials

        strain = np.linspace(MATERIAL_STRAIN_MIN, MATERIAL_STRAIN_MAX, 240)
        shoe = np.asarray(mcclough.equilibrium_pressure_pa(mcclough.REFERENCE_SHOE, strain), dtype=float)
        materials["strain_pct"] = _finite_list(strain * 100.0)
        materials["reference_shoe_pressure_pa"] = _finite_list(shoe, digits=3)
        for name, material in mcclough.materials().items():
            pressure = np.asarray(mcclough.equilibrium_pressure_pa(material, strain), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                materials[f"{name}_pressure_ratio"] = _finite_list(np.where(shoe > 0.0, pressure / shoe, np.nan))
    except ImportError as error:  # pragma: no cover - exercised only without the project package
        materials["unavailable"] = str(error)
    records = sources.evaluations
    return {
        "figures": {
            letter: {"title": name, "placeholder": "Not drawn:" in figures.get(letter, "")}
            for letter, name, _ in FIGURE_BUILDERS
        },
        "momentum_excursion": dict(MOMENTUM_EXCURSION),
        "prescribed_pitch_floor": list(PRESCRIBED_PITCH_FLOOR),
        "momentum_deadband_m_s": MOMENTUM_DEADBAND_M_S,
        "stiff_limit": {
            "k_theta_n_m_per_rad": list(STIFF_LIMIT_K_THETA),
            **{label: list(values) for label, values in STIFF_LIMIT_SERIES},
            "reference_slope": -1.0,
        },
        "infrastructure": [
            {"change": label, "unit": unit, "before": before, "after": after, "speedup": before / after}
            for label, unit, before, after in INFRASTRUCTURE
        ],
        "contact_definition": {
            "contact_force_fraction": CONTACT_FORCE_FRACTION,
            "source_of_truth": "projects.impedance_instron.env.CONTACT_FORCE_FRACTION",
            "profile": sources.profile,
            "threshold_n": sources.threshold_n,
            "stance_window_s": list(stance_window) if stance_window is not None else None,
        },
        "peak_timing_pct": {
            "detected_contact": {key: stats["peak_at_pct_contact"] for key, stats in runs.items()},
            "detected_contact_measured": {key: stats["reference_peak_at_pct_contact"] for key, stats in runs.items()},
            "fixed_stance_window": {key: stats["peak_at_pct_stance_window"] for key, stats in runs.items()},
            "report_section_3_1": peak_timing_comparison(sources),
        },
        "archives": {
            "rule": (
                "glob policy_ankle_v2*.eval.npz, read each material token, keep the newest file per "
                "material, prefer tokenised names over the bare one, and draw every surviving material"
            ),
            "discovered": [
                {
                    "path": str(entry["path"]),
                    "token": entry["token"],
                    "tokenised": entry["tokenised"],
                    "modified": datetime.fromtimestamp(entry["modified"], tz=timezone.utc).isoformat(),
                    "used": entry not in sources.superseded,
                }
                for entry in sources.archives
            ],
            "used_by_run": {run.key: run.source for run in sources.runs if run.key.startswith("ankle_v2")},
        },
        "material_tokens": {
            "source_of_truth": "hash of the material the environment simulated, not of the artifact path",
            "runs": {run.key: run.material for run in sources.runs},
            "evaluation_records": {
                (token or "without a token"): len(group) for token, group in (sources.materials or {}).items()
            },
        },
        "resolution": {
            run.key: {
                "samples": run.sample_count,
                "sample_interval_s": run.sample_interval_s,
                "coarse": bool(run.coarse),
                "steps": sample_step_statistics(run),
            }
            for run in sources.runs
        },
        "runs": runs,
        "waveforms": waveforms,
        "impedance": impedance,
        "training": {
            "ankle_v2": {
                "iteration": _finite_list(np.array([record["iteration"] for record in records], dtype=float)),
                "excursion_momentum": _finite_list(
                    np.array([record.get("excursion_momentum", float("nan")) for record in records], dtype=float)
                ),
                "objective_j": _finite_list(
                    np.array([record.get("objective_j", float("nan")) for record in records], dtype=float)
                ),
            },
            "eval_record_counts": dict(sorted(sources.eval_counts.items())),
            "prescribed_finals": [
                {"run": label, "excursion_momentum": momentum, "work_j_leg_only": work}
                for label, momentum, work in PRESCRIBED_FINALS
            ],
        },
        "materials": materials,
        "missing_sources": dict(sorted(sources.missing.items())),
    }


def write_summary(
    output: Path | str = DEFAULT_OUTPUT,
    directory: Path | str = DEFAULT_DATA_DIRECTORY,
    report: Path | str = DEFAULT_REPORT,
) -> dict[str, Path]:
    """Build the illustrated report and write both artifacts.

    Args:
        output: Directory receiving ``summary.html`` and ``summary.json``.
        directory: Directory holding the impedance-instron outputs.
        report: Path of the Markdown report to illustrate.

    Returns:
        Mapping with the ``html`` and ``json`` paths written.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report)
    report_text = (
        report_path.read_text(encoding="utf-8")
        if report_path.is_file()
        else f"# {FIGURE_PLACEMENT[0][0]}\n\nThe report source `{report_path}` is missing.\n"
    )
    sources = load_sources(directory)
    legacy = _is_legacy_report(parse_blocks(report_text))
    figures = build_figures(sources) if legacy else {}
    payload = summary_payload(sources, figures) if legacy else {"figures": {}}
    payload["report"] = {
        "source": str(report_path),
        "kind": "legacy-momentum-work" if legacy else "unillustrated",
        "figures_included": bool(figures),
    }
    html_path, json_path = output / "summary.html", output / "summary.json"
    html_path.write_text(render_page(report_text, figures, sources), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"html": html_path, "json": json_path}


def create_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Render LEGACY_REPORT.md with inline SVG figures.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Directory receiving summary.html and summary.json.")
    parser.add_argument("--data-directory", default=DEFAULT_DATA_DIRECTORY, help="Directory of run artifacts.")
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="Historical Markdown report; other documents render without legacy figures.",
    )
    return parser


def main() -> None:
    """Render the illustrated report from the command line."""
    arguments = create_parser().parse_args()
    written = write_summary(arguments.output, arguments.data_directory, arguments.report)
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
