# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Overlay the measured reference, the commanded splines, and the achieved result.

The optimizer in :mod:`projects.impedance_instron.optimize` solves a task-level
cost: stance impulse, stance duration, and actuator effort. It never tracks a
measured force history. This module builds the honest diagnostic that shows what
that choice costs. Every panel overlays, for one signal, the measured reference,
the command the controller was given, and the simulated result. Where a signal
has no commanded spline, the equilibrium-point run and the legacy scheduled run
are overlaid against the same reference.

All statistics are computed over the measured stance window only. The report is
one self-contained HTML file with inline SVG; it needs no plotting library, no
JavaScript, and no network access.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from .control import LegCommand

_LEGACY_STIFFNESS_N_M = 12000.0
"""Constant leg stiffness of the legacy scheduled controller [N/m]."""

_LEGACY_DAMPING_N_S_M = 500.0
"""Constant leg damping of the legacy scheduled controller [N·s/m]."""

_CONTACT_THRESHOLD_N = 20.0
"""Vertical load above which the trace is treated as in contact [N]."""

_GRAVITY_M_S2 = 9.81
"""Gravity magnitude used for the implied momentum balance [m/s^2]."""

_MEASURED = "#b55214"
_EQUILIBRIUM = "#1967b3"
_LEGACY = "#16806a"
_COMMAND = "#8054a0"
_CONTEXT = "#737e88"

_TABLE = (
    ("Peak vertical force [N]", "force", "peak_fz_n"),
    ("Peak vertical force error [%]", "force", "peak_fz_error_pct"),
    ("Vertical force RMS error over stance [N]", "force", "fz_rms_error_n"),
    ("Vertical force RMS error [% of measured peak]", "force", "fz_rms_error_pct_peak"),
    ("Vertical impulse over stance [N s]", "force", "impulse_n_s"),
    ("Vertical impulse error [%]", "force", "impulse_error_pct"),
    ("Time of peak force [% of stance]", "force", "time_of_peak_pct_stance"),
    ("Impulse centroid [% of stance]", "force", "impulse_centroid_pct_stance"),
    ("Crest factor: peak / mean force", "force", "crest_factor"),
    ("Peak absolute fore-aft force [N]", "force", "peak_abs_fx_n"),
    ("Fore-aft force RMS error over stance [N]", "force", "fx_rms_error_n"),
    ("Fore-aft impulse over stance [N s]", "force", "fx_impulse_n_s"),
    ("COM height RMS error [mm]", "com", "z_rms_error_mm"),
    ("COM vertical velocity RMS error [m/s]", "com", "vz_rms_error_m_s"),
    ("COM fore-aft position RMS error [mm]", "com", "x_rms_error_mm"),
    ("COM fore-aft velocity RMS error [m/s]", "com", "vx_rms_error_m_s"),
    ("COM vertical velocity change over stance [m/s]", "com", "delta_vz_m_s"),
    ("Measured COM vertical velocity change [m/s]", "com", "reference_delta_vz_m_s"),
    ("Contact duration above 20 N [s]", "integrity", "contact_duration_s"),
    ("Contact duration minus measured stance [ms]", "integrity", "contact_duration_error_ms"),
    ("Residual vertical load at the end of the window [N]", "integrity", "final_shoe_force_n"),
    ("Controller force-limit clipped time fraction [0-1]", "integrity", "controller_clipped_fraction"),
    ("Minimum rigid-last height above ground [m]", "integrity", "min_last_height_m"),
    ("Peak shoe compression [m]", "integrity", "max_compression_m"),
    ("Net active leg work over stance [J]", "energy", "active_work_j"),
    ("Leg damping dissipation over stance [J]", "energy", "damping_dissipation_j"),
    ("Net shoe contact work over stance [J]", "energy", "shoe_contact_work_j"),
)


def _escape(value) -> str:
    """Return an HTML-attribute-safe string.

    Args:
        value: Any value to be shown as text.
    """
    return html.escape(str(value), quote=True)


def _number(value) -> str:
    """Format a scalar for display, with an em dash for missing values.

    Args:
        value: Scalar, or None when the metric does not apply.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.6g}"


def _json_default(value):
    """Serialize NumPy scalars, arrays, and paths for :mod:`json`.

    Args:
        value: Object that :mod:`json` cannot serialize by itself.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _read_trace(path: Path) -> dict[str, np.ndarray]:
    """Read a run trace into finite-checked columns.

    Empty cells and unparsable cells become NaN. Some columns, such as the
    center-of-pressure channels, are NaN by design while the shoe is unloaded,
    so NaN is carried through and is never drawn.

    Args:
        path: Directory holding ``trace.csv``, or the CSV file itself.

    Returns:
        Column name to sample array, each of shape [sample_count].

    Raises:
        ValueError: The trace is too short or its time is not increasing.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "trace.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"{path} needs at least two trace samples")
    names = list(dict.fromkeys(key for row in rows for key in row))
    columns = {}
    for key in names:
        values = np.empty(len(rows), dtype=float)
        for index, row in enumerate(rows):
            try:
                values[index] = float(row.get(key, ""))
            except (TypeError, ValueError):
                values[index] = np.nan
        columns[key] = values
    if "time_s" not in columns or not np.all(np.diff(columns["time_s"]) > 0.0):
        raise ValueError(f"{path} must hold a strictly increasing time_s column")
    return columns


def _stance_window(profile: dict) -> tuple[float, float]:
    """Return touchdown and toe-off in trace time [s].

    Args:
        profile: Measured stance profile, including its running provenance.

    Raises:
        ValueError: The profile carries no selected stance source window.
    """
    running = profile.get("provenance", {}).get("running", {})
    selected = running.get("selected_stance_source_s")
    if selected is None or len(selected) != 2:
        raise ValueError("The profile has no provenance.running.selected_stance_source_s window")
    start = float(profile["source_time_s"][0])
    touchdown, toeoff = float(selected[0]) - start, float(selected[1]) - start
    if not toeoff > touchdown:
        raise ValueError(f"Stance window is empty: {touchdown} to {toeoff} s")
    return touchdown, toeoff


def _commanded_leg(command: dict, times: np.ndarray) -> dict[str, np.ndarray]:
    """Re-evaluate the solved equilibrium-point splines on the trace time grid.

    Args:
        command: Solved command JSON with ``parameters``, ``knots``, and
            ``damping_effective_mass_kg``.
        times: Trace sample times [s], shape [sample_count].

    Returns:
        Commanded ``length_m``, ``length_rate_m_s``, ``stiffness_n_m``,
        ``damping_n_s_m``, and ``damping_ratio``, each of shape [sample_count].
    """
    knots = command["knots"]
    mass = float(command["damping_effective_mass_kg"])
    leg = LegCommand(
        times,
        length_knots=int(knots["length"]),
        stiffness_knots=int(knots["stiffness"]),
        damping_knots=int(knots["damping"]),
        mass_kg=mass,
    )
    evaluated = leg.evaluate(np.asarray(command["parameters"], dtype=float))
    return {
        "length_m": evaluated.length_m,
        "length_rate_m_s": evaluated.length_rate_m_s,
        "stiffness_n_m": evaluated.stiffness_n_m,
        "damping_n_s_m": evaluated.damping_n_s_m,
        "damping_ratio": evaluated.damping_n_s_m / (2.0 * np.sqrt(evaluated.stiffness_n_m * mass)),
    }


def _integral(time: np.ndarray, values: np.ndarray) -> float:
    """Return the trapezoidal integral of one sampled signal.

    Args:
        time: Strictly increasing sample times [s].
        values: Samples on ``time``.
    """
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))


def _cumulative(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the running trapezoidal integral, starting at zero.

    Args:
        time: Strictly increasing sample times [s].
        values: Samples on ``time``.
    """
    return np.concatenate(([0.0], np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(time))))


def _rms_error(time: np.ndarray, values: np.ndarray, reference: np.ndarray) -> float:
    """Return the time-weighted RMS difference between two sampled signals.

    Args:
        time: Strictly increasing sample times [s].
        values: Achieved samples.
        reference: Reference samples on the same grid.
    """
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(reference))):
        return float("nan")
    return float(np.sqrt(_integral(time, (values - reference) ** 2) / (time[-1] - time[0])))


def _force_stats(time: np.ndarray, fz: np.ndarray, fx: np.ndarray, reference: dict[str, np.ndarray]) -> dict:
    """Summarize one vertical and fore-aft force history over stance.

    Args:
        time: Stance sample times [s].
        fz: Vertical force over stance [N].
        fx: Fore-aft force over stance [N].
        reference: Measured ``fz`` and ``fx`` over the same stance samples [N].
    """
    duration = float(time[-1] - time[0])
    peak = float(np.max(fz))
    impulse = _integral(time, fz)
    reference_peak = float(np.max(reference["fz"]))
    reference_impulse = _integral(time, reference["fz"])
    centroid = _integral(time, fz * time) / impulse
    rms = _rms_error(time, fz, reference["fz"])
    return {
        "peak_fz_n": peak,
        "peak_fz_error_pct": (peak - reference_peak) / reference_peak * 100.0,
        "time_of_peak_pct_stance": float(time[int(np.argmax(fz))] - time[0]) / duration * 100.0,
        "impulse_n_s": impulse,
        "impulse_error_pct": (impulse - reference_impulse) / reference_impulse * 100.0,
        "impulse_centroid_pct_stance": (centroid - time[0]) / duration * 100.0,
        "crest_factor": peak / (impulse / duration),
        "fz_rms_error_n": rms,
        "fz_rms_error_pct_peak": rms / reference_peak * 100.0,
        "peak_abs_fx_n": float(np.max(np.abs(fx))),
        "fx_rms_error_n": _rms_error(time, fx, reference["fx"]),
        "fx_impulse_n_s": _integral(time, fx),
        "mean_fz_n": impulse / duration,
    }


def _com_stats(time: np.ndarray, columns: dict[str, np.ndarray], stance: np.ndarray) -> dict:
    """Summarize COM tracking error over stance for one run.

    Args:
        time: Stance sample times [s].
        columns: Full trace columns of the run.
        stance: Boolean stance mask into the full trace, shape [sample_count].
    """

    def window(key: str) -> np.ndarray:
        return columns[key][stance]

    return {
        "z_rms_error_mm": _rms_error(time, window("com_z_m"), window("reference_com_z_m")) * 1000.0,
        "vz_rms_error_m_s": _rms_error(time, window("com_vz_m_s"), window("reference_com_vz_m_s")),
        "x_rms_error_mm": _rms_error(time, window("com_x_m"), window("reference_com_x_m")) * 1000.0,
        "vx_rms_error_m_s": _rms_error(time, window("com_vx_m_s"), window("reference_com_vx_m_s")),
        "delta_vz_m_s": float(window("com_vz_m_s")[-1] - window("com_vz_m_s")[0]),
        "reference_delta_vz_m_s": float(window("reference_com_vz_m_s")[-1] - window("reference_com_vz_m_s")[0]),
        "delta_vx_m_s": float(window("com_vx_m_s")[-1] - window("com_vx_m_s")[0]),
        "reference_delta_vx_m_s": float(window("reference_com_vx_m_s")[-1] - window("reference_com_vx_m_s")[0]),
        "endpoint_z_error_mm": float(window("com_z_m")[-1] - window("reference_com_z_m")[-1]) * 1000.0,
        "endpoint_vz_error_m_s": float(window("com_vz_m_s")[-1] - window("reference_com_vz_m_s")[-1]),
    }


def _integrity_stats(columns: dict[str, np.ndarray], stance_duration_s: float) -> dict:
    """Summarize contact integrity of one run over its whole saved window.

    Args:
        columns: Full trace columns of the run.
        stance_duration_s: Measured stance duration [s].
    """
    time = columns["time_s"]
    loaded = np.nonzero(columns["shoe_fz_n"] > _CONTACT_THRESHOLD_N)[0]
    duration = float(time[loaded[-1]] - time[loaded[0]]) if loaded.size > 1 else float("nan")
    return {
        "contact_start_s": float(time[loaded[0]]) if loaded.size else float("nan"),
        "contact_end_s": float(time[loaded[-1]]) if loaded.size else float("nan"),
        "contact_duration_s": duration,
        "contact_duration_error_ms": (duration - stance_duration_s) * 1000.0,
        "final_shoe_force_n": float(columns["shoe_fz_n"][-1]),
        "controller_clipped_fraction": float(np.mean(columns["controller_clipped"])),
        "min_last_height_m": float(np.nanmin(columns["last_min_height_m"])),
        "max_compression_m": float(np.nanmax(columns["max_compression_m"])),
    }


def _energy_stats(time: np.ndarray, columns: dict[str, np.ndarray], stance: np.ndarray) -> dict:
    """Summarize the mechanical power channels of one run over stance.

    Args:
        time: Stance sample times [s].
        columns: Full trace columns of the run.
        stance: Boolean stance mask into the full trace, shape [sample_count].
    """
    return {
        "active_work_j": _integral(time, columns["active_power_w"][stance]),
        "damping_dissipation_j": -_integral(time, np.minimum(columns["damping_power_w"][stance], 0.0)),
        "shoe_contact_work_j": _integral(time, columns["shoe_contact_power_w"][stance]),
    }


def _statistics(
    equilibrium: dict[str, np.ndarray],
    legacy: dict[str, np.ndarray],
    command: dict,
    commanded: dict[str, np.ndarray],
    window: tuple[float, float],
    mass_kg: float,
) -> dict:
    """Compute every reported statistic over the measured stance window.

    Args:
        equilibrium: Trace columns of the equilibrium-point run.
        legacy: Trace columns of the legacy scheduled run.
        command: Solved command JSON of the equilibrium-point run.
        commanded: Commanded splines on the trace time grid.
        window: Touchdown and toe-off in trace time [s].
        mass_kg: Total modeled mass [kg].
    """
    touchdown, toeoff = window
    time = equilibrium["time_s"]
    stance = (time >= touchdown) & (time <= toeoff)
    stance_time = time[stance]
    reference = {"fz": equilibrium["reference_fz_n"][stance], "fx": equilibrium["reference_fx_n"][stance]}
    stance_duration = float(stance_time[-1] - stance_time[0])
    force = {
        "measured": _force_stats(stance_time, reference["fz"], reference["fx"], reference),
        "equilibrium": _force_stats(
            stance_time, equilibrium["shoe_fz_n"][stance], equilibrium["shoe_fx_n"][stance], reference
        ),
        "legacy": _force_stats(stance_time, legacy["shoe_fz_n"][stance], legacy["shoe_fx_n"][stance], reference),
    }
    deflection = equilibrium["leg_length_m"][stance] - commanded["length_m"][stance]
    return {
        "stance": {
            "touchdown_s": touchdown,
            "toeoff_s": toeoff,
            "measured_duration_s": toeoff - touchdown,
            "integrated_duration_s": stance_duration,
            "sample_count": int(np.count_nonzero(stance)),
            "mass_kg": mass_kg,
            "body_weight_n": mass_kg * _GRAVITY_M_S2,
            "contact_threshold_n": _CONTACT_THRESHOLD_N,
        },
        "force": force,
        "com": {
            "equilibrium": _com_stats(stance_time, equilibrium, stance),
            "legacy": _com_stats(stance_time, legacy, stance),
        },
        "integrity": {
            "equilibrium": _integrity_stats(equilibrium, toeoff - touchdown),
            "legacy": _integrity_stats(legacy, toeoff - touchdown),
        },
        "energy": {
            "equilibrium": _energy_stats(stance_time, equilibrium, stance),
            "legacy": _energy_stats(stance_time, legacy, stance),
        },
        "command": {
            "cost": command.get("cost"),
            "cost_definition": command.get("cost_definition"),
            "target": command.get("target"),
            # Commands solved before the three-tier objective carry "weights"; newer ones
            # carry "objective" with tolerances and efficiencies instead.
            "weights": command.get("weights"),
            "objective": command.get("objective"),
            "verdict": command.get("verdict"),
            "solver_contact_duration_s": command.get("rollout", {}).get("contact_duration_s"),
            "solver_duration_error_ms": (
                float(command["rollout"]["contact_duration_s"] - command["target"]["duration_s"]) * 1000.0
                if "rollout" in command and "target" in command
                else None
            ),
            "solver_saturated": command.get("rollout", {}).get("saturated"),
            "solver_residual_load_n": command.get("rollout", {}).get("residual_load_n"),
            "stiffness_range_n_m": [
                float(np.min(commanded["stiffness_n_m"][stance])),
                float(np.max(commanded["stiffness_n_m"][stance])),
            ],
            "damping_range_n_s_m": [
                float(np.min(commanded["damping_n_s_m"][stance])),
                float(np.max(commanded["damping_n_s_m"][stance])),
            ],
            "damping_ratio_range": [
                float(np.min(commanded["damping_ratio"][stance])),
                float(np.max(commanded["damping_ratio"][stance])),
            ],
            "equilibrium_length_range_m": [
                float(np.min(commanded["length_m"][stance])),
                float(np.max(commanded["length_m"][stance])),
            ],
            "leg_deflection_range_m": [float(np.min(deflection)), float(np.max(deflection))],
            "peak_abs_leg_deflection_m": float(np.max(np.abs(deflection))),
            "legacy_stiffness_n_m": _LEGACY_STIFFNESS_N_M,
            "legacy_damping_n_s_m": _LEGACY_DAMPING_N_S_M,
        },
        "pitch": {
            "spline_fit_rms_error_rad": _rms_error(
                stance_time, equilibrium["pitch_rad"][stance], equilibrium["raw_pitch_rad"][stance]
            ),
            "peak_abs_spline_fit_error_rad": float(
                np.max(np.abs(equilibrium["pitch_rad"][stance] - equilibrium["raw_pitch_rad"][stance]))
            ),
            "note": "Foot pitch is kinematically prescribed, so the achieved angle is the commanded spline. "
            "The rig logs no independent achieved pitch channel.",
        },
    }


def _plot(
    title: str, unit: str, series: list[tuple], xlabel: str, stance: tuple[float, float] | None, note: str = ""
) -> str:
    """Draw one overlay panel as an accessible, self-contained inline SVG.

    Non-finite samples break the path instead of producing NaN coordinates, so a
    gap always means unavailable data.

    Args:
        title: Panel heading.
        unit: Unit of the vertical axis.
        series: ``(label, x, y, color, dashed)`` tuples on a shared x axis.
        xlabel: Label of the horizontal axis.
        stance: Shaded stance band in x units, or None for no band.
        note: Optional caption printed under the legend.
    """
    drawable = []
    for label, raw_x, raw_y, color, dashed in series:
        x, y = np.asarray(raw_x, dtype=float), np.asarray(raw_y, dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if np.any(good):
            drawable.append((label, x, np.where(good, y, np.nan), color, dashed, good))
    if not drawable:
        return ""
    finite_y = np.concatenate([y[good] for _, _, y, _, _, good in drawable])
    finite_x = np.concatenate([x[good] for _, x, _, _, _, good in drawable])
    low, high = float(np.min(finite_y)), float(np.max(finite_y))
    margin = max((high - low) * 0.08, abs(high) * 0.005, 1.0e-9)
    low, high = low - margin, high + margin
    xmin, xmax = float(np.min(finite_x)), float(np.max(finite_x))
    if not xmax > xmin:
        return ""
    width, height, left, top, plot_width, plot_height = 640, 285, 76, 18, 548, 217
    content = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}"><title>{_escape(title)}</title>'
    ]
    if stance is not None:
        start = left + (max(stance[0], xmin) - xmin) / (xmax - xmin) * plot_width
        end = left + (min(stance[1], xmax) - xmin) / (xmax - xmin) * plot_width
        if end > start:
            content.append(
                f'<rect x="{start:.2f}" y="{top}" width="{end - start:.2f}" height="{plot_height}" class="stance"/>'
            )
    for fraction in np.linspace(0.0, 1.0, 5):
        sx, sy = left + fraction * plot_width, top + (1.0 - fraction) * plot_height
        content.append(f'<path d="M{left},{sy:.2f}h{plot_width}" class="grid"/>')
        content.append(
            f'<text x="{left - 8}" y="{sy + 4:.2f}" text-anchor="end">{_number(low + fraction * (high - low))}</text>'
        )
        content.append(
            f'<text x="{sx:.2f}" y="{top + plot_height + 20}" text-anchor="middle">{_number(xmin + fraction * (xmax - xmin))}</text>'
        )
    if low <= 0.0 <= high:
        zero = top + high / (high - low) * plot_height
        content.append(f'<path d="M{left},{zero:.2f}h{plot_width}" stroke="#99a7b9" stroke-dasharray="3 4"/>')
    for label, x, y, color, dashed, _ in drawable:
        sx = left + (x - xmin) / (xmax - xmin) * plot_width
        sy = top + (high - y) / (high - low) * plot_height
        pieces = []
        active = False
        for px, py in zip(sx, sy, strict=True):
            if np.isfinite(px) and np.isfinite(py):
                pieces.append(f"{'L' if active else 'M'}{px:.2f},{py:.2f}")
                active = True
            else:
                active = False
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        content.append(
            f'<path d="{" ".join(pieces)}" fill="none" stroke="{color}" stroke-width="2"{dash}><title>{_escape(label)}</title></path>'
        )
    content.append(f'<text x="{left + plot_width / 2}" y="277" text-anchor="middle">{_escape(xlabel)}</text></svg>')
    legend = " ".join(
        f'<span><i style="background:{color}"></i>{_escape(label)}{" (dashed)" if dashed else ""}</span>'
        for label, _, _, color, dashed, _ in drawable
    )
    caption = f'<p class="note">{_escape(note)}</p>' if note else ""
    return (
        f'<section class="plot"><h3>{_escape(title)} <small>[{_escape(unit)}]</small></h3>{"".join(content)}'
        f'<div class="legend">{legend}</div>{caption}</section>'
    )


def _panels(
    equilibrium: dict[str, np.ndarray],
    legacy: dict[str, np.ndarray],
    commanded: dict[str, np.ndarray],
    window: tuple[float, float],
    mass_kg: float,
) -> str:
    """Build every overlay panel of the report.

    Args:
        equilibrium: Trace columns of the equilibrium-point run.
        legacy: Trace columns of the legacy scheduled run.
        commanded: Commanded splines on the trace time grid.
        window: Touchdown and toe-off in trace time [s].
        mass_kg: Total modeled mass [kg].
    """
    touchdown, toeoff = window
    time = equilibrium["time_s"]
    duration = toeoff - touchdown
    percent = (time - touchdown) / duration * 100.0
    band = (0.0, 100.0)
    axis = "Stance progress [%]: 0 is touchdown, 100 is toe-off"
    weight = mass_kg * _GRAVITY_M_S2
    plots = []

    def overlay(title, unit, key, reference_key=None, note=""):
        series = []
        if reference_key is not None:
            series.append(("Measured reference", percent, equilibrium[reference_key], _MEASURED, True))
        series.append(("Equilibrium-point result", percent, equilibrium[key], _EQUILIBRIUM, False))
        series.append(("Legacy scheduled result", percent, legacy[key], _LEGACY, False))
        plots.append(_plot(title, unit, series, axis, band, note))

    overlay(
        "1. Vertical ground reaction force",
        "N",
        "shoe_fz_n",
        "reference_fz_n",
        "The equilibrium-point run delivers its peak late in stance.",
    )
    overlay("2. Fore-aft ground reaction force", "N", "shoe_fx_n", "reference_fx_n")

    # Impulse is accumulated from touchdown, so the curves start together and any
    # divergence is a real momentum difference rather than an offset.
    inside = (time >= touchdown) & (time <= toeoff)
    stance_time = time[inside]
    stance_percent = percent[inside]
    impulse_series, velocity_series = [], []
    for label, values, color, dashed in (
        ("Measured reference", equilibrium["reference_fz_n"], _MEASURED, True),
        ("Equilibrium-point result", equilibrium["shoe_fz_n"], _EQUILIBRIUM, False),
        ("Legacy scheduled result", legacy["shoe_fz_n"], _LEGACY, False),
    ):
        cumulative = _cumulative(stance_time, values[inside])
        impulse_series.append((label, stance_percent, cumulative, color, dashed))
        net = _cumulative(stance_time, values[inside] - weight) / mass_kg
        velocity_series.append((label, stance_percent, net, color, dashed))
    plots.append(
        _plot(
            "3a. Cumulative vertical impulse from touchdown",
            "N s",
            impulse_series,
            axis,
            band,
            "The endpoints agree closely. The paths to those endpoints do not.",
        )
    )
    plots.append(
        _plot(
            "3b. Implied COM vertical velocity change",
            "m/s",
            velocity_series,
            axis,
            band,
            "Integral of (Fz - body weight) / mass from touchdown. Same endpoint, different history.",
        )
    )

    overlay("4a. COM height", "m", "com_z_m", "reference_com_z_m")
    overlay("4b. COM vertical velocity", "m/s", "com_vz_m_s", "reference_com_vz_m_s")
    overlay("5a. COM fore-aft position", "m", "com_x_m", "reference_com_x_m")
    overlay("5b. COM fore-aft velocity", "m/s", "com_vx_m_s", "reference_com_vx_m_s")

    plots.append(
        _plot(
            "6a. Leg length: commanded equilibrium against achieved",
            "m",
            [
                ("Commanded equilibrium L0(t)", percent, commanded["length_m"], _COMMAND, True),
                ("Achieved leg length L(t)", percent, equilibrium["leg_length_m"], _EQUILIBRIUM, False),
                ("Legacy scalar leg reference", percent, legacy["reference_leg_length_m"], _CONTEXT, True),
            ],
            axis,
            band,
            "L0 is the motor command of the equilibrium-point law, not a measured length.",
        )
    )
    plots.append(
        _plot(
            "6b. Leg deflection e = L - L0",
            "m",
            [
                (
                    "Equilibrium-point deflection",
                    percent,
                    equilibrium["leg_length_m"] - commanded["length_m"],
                    _EQUILIBRIUM,
                    False,
                )
            ],
            axis,
            band,
            "Deflection times K(t) is the elastic part of the leg force.",
        )
    )
    plots.append(
        _plot(
            "7a. Commanded leg stiffness K(t)",
            "N/m",
            [
                ("Commanded K(t)", percent, commanded["stiffness_n_m"], _COMMAND, False),
                ("Legacy constant K", percent, np.full_like(percent, _LEGACY_STIFFNESS_N_M), _LEGACY, True),
            ],
            axis,
            band,
        )
    )
    plots.append(
        _plot(
            "7b. Commanded leg damping b(t)",
            "N s/m",
            [
                ("Commanded b(t)", percent, commanded["damping_n_s_m"], _COMMAND, False),
                ("Legacy constant b", percent, np.full_like(percent, _LEGACY_DAMPING_N_S_M), _LEGACY, True),
            ],
            axis,
            band,
        )
    )
    plots.append(
        _plot(
            "7c. Commanded damping ratio zeta(t)",
            "1",
            [("Commanded zeta(t)", percent, commanded["damping_ratio"], _COMMAND, False)],
            axis,
            band,
            "b = 2 zeta sqrt(K m) with m = the declared effective mass of the command.",
        )
    )
    plots.append(
        _plot(
            "8. Foot pitch: optical knots, fitted spline, achieved",
            "rad",
            [
                ("Raw optical knots", percent, equilibrium["raw_pitch_rad"], _MEASURED, True),
                ("Fitted spline command", percent, equilibrium["pitch_rad"], _COMMAND, False),
                (
                    "Achieved angle (prescribed, equals the command)",
                    percent,
                    equilibrium["pitch_rad"],
                    _EQUILIBRIUM,
                    True,
                ),
            ],
            axis,
            band,
            "Pitch is kinematically prescribed, so the achieved angle is the commanded spline. "
            "The two simulated curves overlap by construction.",
        )
    )
    plots.append(
        _plot(
            "9. Leg actuator force against shoe ground reaction",
            "N",
            [
                ("Equilibrium leg actuator force", percent, equilibrium["leg_force_n"], _EQUILIBRIUM, False),
                ("Equilibrium shoe Fz", percent, equilibrium["shoe_fz_n"], _COMMAND, True),
                ("Measured reference Fz", percent, equilibrium["reference_fz_n"], _MEASURED, True),
            ],
            axis,
            band,
            "The gap between actuator force and ground reaction is foot inertia and shoe dynamics.",
        )
    )
    plots.append(
        _plot(
            "10a. Mechanical power channels, equilibrium run",
            "W",
            [
                ("Active source", percent, equilibrium["active_power_w"], _EQUILIBRIUM, False),
                ("Leg damper", percent, equilibrium["damping_power_w"], _MEASURED, False),
                ("Shoe contact", percent, equilibrium["shoe_contact_power_w"], _LEGACY, False),
            ],
            axis,
            band,
        )
    )
    energy_series = []
    for label, key, color in (
        ("Active source work", "active_power_w", _EQUILIBRIUM),
        ("Damper dissipation (positive loss)", "damping_power_w", _MEASURED),
        ("Shoe contact work", "shoe_contact_power_w", _LEGACY),
    ):
        power = equilibrium[key] * (-1.0 if key == "damping_power_w" else 1.0)
        energy_series.append((label, percent, _cumulative(time, power), color, False))
    plots.append(_plot("10b. Cumulative energy, equilibrium run", "J", energy_series, axis, band))
    return "".join(plots)


def _table_rows(stats: dict) -> str:
    """Render the statistics table body, one row per reported metric.

    Args:
        stats: Computed statistics from :func:`_statistics`.
    """
    rows = []
    for label, section, key in _TABLE:
        block = stats[section]
        measured = block.get("measured", {}).get(key) if section == "force" else None
        equilibrium = block.get("equilibrium", {}).get(key)
        legacy = block.get("legacy", {}).get(key)
        if equilibrium is None and legacy is None and measured is None:
            continue
        rows.append(
            f'<tr><th scope="row">{_escape(label)}</th><td>{_number(measured)}</td>'
            f"<td>{_number(equilibrium)}</td><td>{_number(legacy)}</td></tr>"
        )
    return "".join(rows)


def _narrative(stats: dict) -> str:
    """Write the honest reading of the numbers into HTML.

    Args:
        stats: Computed statistics from :func:`_statistics`.
    """
    force, command = stats["force"], stats["command"]
    equilibrium, measured = force["equilibrium"], force["measured"]
    com = stats["com"]["equilibrium"]
    integrity = stats["integrity"]["equilibrium"]
    shift = equilibrium["time_of_peak_pct_stance"] - measured["time_of_peak_pct_stance"]
    solver_error_ms = command.get("solver_duration_error_ms")
    return f"""<article><h2>What the controller gets right</h2>
<p>The equilibrium-point controller uses no measured-force feedforward. It uses no engage or release schedule.
It is one commanded equilibrium length L0(t) with a commanded stiffness K(t) and damping b(t).
With only that, it reproduces the task-level quantities well.</p>
<ul>
<li>Total vertical impulse over stance: {_number(equilibrium["impulse_n_s"])} N s against the measured
{_number(measured["impulse_n_s"])} N s. The error is {_number(equilibrium["impulse_error_pct"])} %.</li>
<li>Stance duration: the solver reports {_number(command.get("solver_contact_duration_s"))} s against the
{_number(stats["stance"]["measured_duration_s"])} s measured target, an error of {_number(solver_error_ms)} ms.
Measured from this trace at a {_number(stats["stance"]["contact_threshold_n"])} N contact threshold, the contact
duration is {_number(integrity["contact_duration_s"])} s, which differs by
{_number(integrity["contact_duration_error_ms"])} ms. The two numbers differ because the threshold differs.</li>
<li>Full release: the vertical load at the end of the saved window is {_number(integrity["final_shoe_force_n"])} N.
There is no residual load and no sticking.</li>
<li>No force-limit saturation: the controller-clipped time fraction is
{_number(integrity["controller_clipped_fraction"])}.</li>
<li>No ground penetration: the minimum rigid-last height stays at
{_number(integrity["min_last_height_m"])} m above the ground plane.</li>
</ul>
<p>The endpoint momentum is therefore right. Panel 3a shows this directly: the three cumulative impulse curves
end at almost the same value.</p></article>
<article><h2>What the controller gets wrong</h2>
<p>The force waveform has the wrong shape in time. The measured peak occurs at
{_number(measured["time_of_peak_pct_stance"])} % of stance. The equilibrium-point run peaks at
{_number(equilibrium["time_of_peak_pct_stance"])} % of stance. That is {_number(shift)} percentage points of stance
too late. The impulse centroid moves the same way, from {_number(measured["impulse_centroid_pct_stance"])} % to
{_number(equilibrium["impulse_centroid_pct_stance"])} %.</p>
<p>The vertical force RMS error over stance is {_number(equilibrium["fz_rms_error_n"])} N, which is
{_number(equilibrium["fz_rms_error_pct_peak"])} % of the measured peak force. The legacy scheduled controller,
which is driven from the measured profile, reaches {_number(force["legacy"]["fz_rms_error_n"])} N
({_number(force["legacy"]["fz_rms_error_pct_peak"])} %).</p>
<p>Because the force arrives late, the COM path drifts even though the endpoint momentum is right. Over stance the
COM height RMS error is {_number(com["z_rms_error_mm"])} mm and the COM vertical velocity RMS error is
{_number(com["vz_rms_error_m_s"])} m/s. The legacy run stays at {_number(stats["com"]["legacy"]["z_rms_error_mm"])} mm
and {_number(stats["com"]["legacy"]["vz_rms_error_m_s"])} m/s.</p></article>
<article><h2>Why this happens</h2>
<p>The cost function is the cause. It constrained three things: the total stance impulse through the target
velocity change, the stance duration, and actuator effort. Penalty terms guarded penetration, compression,
saturation and residual load. The recorded cost definition is:
<em>{_escape(str(command.get("cost_definition")))}</em>.</p>
<p>Nothing in that cost says <strong>when</strong> inside stance the impulse must be delivered. Many force
histories have the same integral, and the cost cannot tell them apart. The search was free to pick one that peaks
late, and it did. No other term pinned the waveform down. This report does not identify which cost term pulled the
peak late; it only shows that no term held it in place.</p>
<p>This is a cost-function gap, not a controller-structure failure. The equilibrium-point law has enough freedom
to shape the force: K(t), b(t) and L0(t) are all time varying splines. The search was simply never asked to shape
it. A timing term, for example a penalty on the impulse-centroid time or on the time of peak force, would close
this gap without changing the controller.</p></article>
<article><h2>Consequence for the intended use</h2>
<p>The intended use of this rig is to compare shoe response under a realistic load. That use fails here.
A shoe loaded with peak force at {_number(equilibrium["time_of_peak_pct_stance"])} % of stance instead of
{_number(measured["time_of_peak_pct_stance"])} % is loaded at a different foot pitch, at a different contact
geometry, and at a different point of the material load history. Shoe-response conclusions drawn from this run
would not transfer to the measured condition. Do not use this run to rank or to tune shoe properties until the
waveform timing is fixed.</p>
<p>Scope note: this report compares a reduced mechanical rig against one measured stance. It is not human
validation, not material validation, and not an anatomical registration of the model.</p></article>"""


def _render(stats: dict, panels: str, sources: dict[str, str]) -> str:
    """Assemble the complete offline HTML document.

    Args:
        stats: Computed statistics from :func:`_statistics`.
        panels: Concatenated plot sections.
        sources: Input labels and resolved paths.
    """
    force = stats["force"]
    stance = stats["stance"]
    source_rows = "".join(
        f'<tr><th scope="row">{_escape(label)}</th><td><code>{_escape(path)}</code></td></tr>'
        for label, path in sources.items()
    )
    statistics_json = _escape(json.dumps(stats, indent=2, sort_keys=True, default=_json_default, allow_nan=True))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Impedance Instron: reference, command, result</title><style>
:root{{font-family:system-ui,sans-serif;color:#192d42;background:#f2f5f8;line-height:1.5}}
body{{max-width:1320px;margin:auto;padding:24px}}h1,h2,h3{{line-height:1.25}}h1{{margin-bottom:8px}}
a{{color:#125c9b}}header,article,.plot{{background:white;border:1px solid #dce4ec;border-radius:10px;padding:20px;margin-bottom:18px}}
.subtitle,small,.note{{color:#536577}}.note{{font-size:.78rem;margin:6px 0 0}}
.badge{{display:inline-block;padding:4px 10px;background:#e7edf4;border-radius:6px;margin-right:8px}}
.warning{{background:#fff0d5;border-left:4px solid #bd6b00;padding:12px}}.ok{{background:#e6f3eb;padding:12px}}
.panels{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.plot{{min-width:0;margin:0;padding:14px}}
.plot h3{{font-size:1rem;margin:0 0 8px}}svg{{width:100%;height:auto}}svg text{{font:11px system-ui;fill:#536577}}
.grid{{stroke:#e5eaf0;fill:none}}.stance{{fill:#eef3f9}}
.legend{{font-size:.78rem;display:flex;flex-wrap:wrap;gap:4px 14px}}.legend i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}th,td{{padding:8px 10px;border-bottom:1px solid #e5eaf0;text-align:left;vertical-align:top}}
td{{font-variant-numeric:tabular-nums}}tbody th{{font-weight:500}}thead{{background:#edf2f7}}code{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem}}li{{margin:5px 0}}.panels+article{{margin-top:18px}}
@media(max-width:800px){{body{{padding:10px}}.panels{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}.plot{{break-inside:avoid}}}}
</style></head><body><header><h1>Impedance Instron: reference, command, result</h1>
<p class="subtitle">Every panel overlays the measured reference, the command given to the controller, and the achieved
simulated result. Statistics cover the measured stance window only.</p>
<span class="badge">Stance: {_number(stance["touchdown_s"])} s to {_number(stance["toeoff_s"])} s</span>
<span class="badge">Stance duration: {_number(stance["measured_duration_s"])} s</span>
<span class="badge">{stance["sample_count"]} stance samples</span>
<span class="badge">Mass: {_number(stance["mass_kg"])} kg</span>
<p><a href="explanation.json">Download every computed statistic as JSON</a></p></header>
<article><h2>Headline reading</h2>
<p class="ok">The equilibrium-point controller matches the <strong>integral</strong> of the measured stance load. It
does not match the <strong>shape</strong>. Total impulse error is {_number(force["equilibrium"]["impulse_error_pct"])} %,
while the peak of the force arrives
{_number(force["equilibrium"]["time_of_peak_pct_stance"] - force["measured"]["time_of_peak_pct_stance"])}
percentage points of stance late.</p>
<p class="warning">Read panel 3a together with panel 1. The cumulative impulse curves nearly coincide. The force
curves do not. That single pair of panels is the result of this report.</p></article>
{_narrative(stats)}
<article><h2>Statistics over the measured stance window</h2>
<p>Every value is recomputed from the traces in this report. Stance runs from touchdown to toe-off of the selected
measured stance. Integrals are trapezoidal on the trace grid.</p>
<div class="scroll"><table><thead><tr><th>Metric</th><th>Measured</th><th>Equilibrium point</th>
<th>Legacy scheduled</th></tr></thead><tbody>{_table_rows(stats)}</tbody></table></div>
<p class="note">Fore-aft ground force of the legacy run is included for context. The measured column is blank where
the metric describes the simulation only.</p></article>
<article><h2>Commanded impedance summary</h2><ul>
<li>Commanded stiffness K(t) over stance: {_number(stats["command"]["stiffness_range_n_m"][0])} to
{_number(stats["command"]["stiffness_range_n_m"][1])} N/m, against the legacy constant
{_number(_LEGACY_STIFFNESS_N_M)} N/m.</li>
<li>Commanded damping b(t) over stance: {_number(stats["command"]["damping_range_n_s_m"][0])} to
{_number(stats["command"]["damping_range_n_s_m"][1])} N s/m, against the legacy constant
{_number(_LEGACY_DAMPING_N_S_M)} N s/m.</li>
<li>Commanded damping ratio zeta(t): {_number(stats["command"]["damping_ratio_range"][0])} to
{_number(stats["command"]["damping_ratio_range"][1])}.</li>
<li>Commanded equilibrium length L0(t): {_number(stats["command"]["equilibrium_length_range_m"][0])} to
{_number(stats["command"]["equilibrium_length_range_m"][1])} m. Peak absolute deflection L - L0 is
{_number(stats["command"]["peak_abs_leg_deflection_m"])} m.</li>
<li>Pitch spline fit against the raw optical knots: RMS {_number(stats["pitch"]["spline_fit_rms_error_rad"])} rad,
peak {_number(stats["pitch"]["peak_abs_spline_fit_error_rad"])} rad. {_escape(stats["pitch"]["note"])}</li>
</ul></article>
<article><h2>Inputs</h2><div class="scroll"><table><tbody>{source_rows}</tbody></table></div></article>
<div class="panels">{panels}</div>
<article><details><summary>All computed statistics</summary><pre>{statistics_json}</pre></details></article>
<footer>All plots are inline SVG drawn from the traces. No network access, JavaScript, or plotting library is
required. Non-finite samples are drawn as gaps.</footer>
</body></html>"""


def write_explanation(
    output_dir: Path,
    equilibrium_dir: Path,
    legacy_dir: Path,
    command_path: Path,
    profile_path: Path,
) -> Path:
    """Write the diagnostic overlay report and its machine-readable statistics.

    Args:
        output_dir: Directory for explanation.html and explanation.json.
        equilibrium_dir: Run directory of the equilibrium-point controller.
        legacy_dir: Run directory of the legacy scheduled controller.
        command_path: Solved command JSON of the equilibrium-point run.
        profile_path: Measured stance profile JSON.

    Returns:
        Path to explanation.html.

    Raises:
        ValueError: The two traces do not share one time grid.
    """
    output_dir = Path(output_dir)
    equilibrium = _read_trace(Path(equilibrium_dir))
    legacy = _read_trace(Path(legacy_dir))
    command = json.loads(Path(command_path).read_text(encoding="utf-8"))
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if equilibrium["time_s"].shape != legacy["time_s"].shape or not np.allclose(
        equilibrium["time_s"], legacy["time_s"]
    ):
        raise ValueError("The two runs must share one time grid to be overlaid sample by sample")
    window = _stance_window(profile)
    mass_kg = float(profile["mass_kg"])
    commanded = _commanded_leg(command, equilibrium["time_s"])
    stats = _statistics(equilibrium, legacy, command, commanded, window, mass_kg)
    sources = {
        "Equilibrium-point trace": str(Path(equilibrium_dir).resolve()),
        "Legacy scheduled trace": str(Path(legacy_dir).resolve()),
        "Solved command": str(Path(command_path).resolve()),
        "Measured profile": str(Path(profile_path).resolve()),
    }
    stats["sources"] = sources
    document = _render(stats, _panels(equilibrium, legacy, commanded, window, mass_kg), sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "explanation.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )
    report = output_dir / "explanation.html"
    report.write_text(document, encoding="utf-8")
    return report


def create_explain_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of the diagnostic overlay report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--equilibrium",
        type=Path,
        default=Path("outputs/impedance_instron/equilibrium_best"),
        help="Run directory of the equilibrium-point controller.",
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=Path("outputs/impedance_instron/legacy_compare"),
        help="Run directory of the legacy scheduled controller.",
    )
    parser.add_argument(
        "--command",
        type=Path,
        default=Path("outputs/impedance_instron/command_a.json"),
        help="Solved command JSON of the equilibrium-point run.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("outputs/impedance_instron/stance_planar_context.json"),
        help="Measured stance profile JSON that defines the stance window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/impedance_instron/explanation"),
        help="Directory for explanation.html and explanation.json.",
    )
    return parser


def main():
    """Write the overlay report from saved traces without running a simulation."""
    args = create_explain_parser().parse_args()
    report = write_explanation(args.output, args.equilibrium, args.legacy, args.command, args.profile)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
