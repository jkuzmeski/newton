# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Redraw the two-term footwear report from one hash-checked saved artifact.

Run from the Newton repository root::

    uv run --no-sync -m projects.digital_shoe.mechanics_report.figures \
        --artifact outputs/impedance_instron/inputs/digital_shoe.json \
        --output outputs/footwear_contact_material_report/figures

Saved held-out predictions are plotted without fitting or replaying their model.
Material-point and bristle examples execute the current shared project laws.
They are illustrative calculations, not additional experimental validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
import warp as wp

from projects.digital_instron_v2.validation import validate_trace_metrics
from projects.digital_shoe.contact import bristle_step
from projects.digital_shoe.material import (
    hyperfoam_pressure_numpy,
    maxwell_coefficients_numpy,
    maxwell_step_numpy,
    ogden_hill_term_numpy,
)

from ._provenance import DEFAULT_ARTIFACT, DEFAULT_OUTPUT, MANIFEST, ROOT, load_verified_artifact

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
INK = "#243746"


@wp.kernel
def bristle_history(
    position: wp.array[float],
    velocity: wp.array[float],
    dt: float,
    normal: float,
    kt: float,
    kv: float,
    mu: float,
    viscous_ratio: float,
    release_dwell: float,
    force_out: wp.array[float],
    anchor_out: wp.array[float],
):
    """Execute the shared bristle recurrence serially for one material contact."""
    anchor = wp.vec2(0.0, 0.0)
    stuck = int(0)
    dwell = float(0.0)
    for i in range(position.shape[0]):
        force, anchor, stuck, dwell = bristle_step(
            wp.vec2(position[i], 0.0),
            wp.vec2(velocity[i], 0.0),
            dt,
            normal,
            kt,
            kv,
            mu,
            viscous_ratio,
            release_dwell,
            anchor,
            stuck,
            dwell,
        )
        force_out[i] = force[0]
        anchor_out[i] = anchor[0]


def sha256(path: Path) -> str:
    """Hash source bytes for exact provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    """Set accessible colors, readable labels and editable vector text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": "#788792",
            "grid.color": "#DFE5E9",
            "grid.alpha": 0.8,
            "lines.linewidth": 2.1,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "two-term-footwear-report",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def decorate(ax, xlabel: str, ylabel: str) -> None:
    """Apply axis labels and a quiet background grid."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, directory: Path, stem: str, title: str, note: str) -> None:
    """Write matching PNG and SVG figures with embedded qualification text."""
    import matplotlib.pyplot as plt

    fig.suptitle(title, x=0.07, y=0.975, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.022, note, ha="left", va="bottom", fontsize=9, color="#516574")
    fig.tight_layout(rect=(0.015, 0.09, 0.99, 0.91), w_pad=2.6)
    for suffix in ("svg", "png"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=210)
    plt.close(fig)


def cells(ax, xy, spacing, values=None, **kwargs):
    """Render each actual square column footprint in geometric units."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    patches = [Rectangle((x - spacing / 2, y - spacing / 2), spacing, spacing) for x, y in xy]
    collection = PatchCollection(patches, linewidth=0.22, **kwargs)
    if values is not None:
        collection.set_array(values)
    ax.add_collection(collection)
    return collection


def bed_figure(artifact, directory):
    """Show thickness and stored driven fixture coordinates without mesh interpolation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bed = artifact["column_bed"]
    xy = np.asarray(bed["anchor_bottom_m"])[:, :2] * 1000
    thickness = np.asarray(bed["rest_length_m"]) * 1000
    spacing = bed["spacing_m"] * 1000
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.75))
    image = cells(axes[0], xy, spacing, thickness, cmap="viridis", edgecolor="white")
    fig.colorbar(image, ax=axes[0], label="Rest thickness [mm]", fraction=0.035, pad=0.025)
    axes[0].set_title(f"Whole bed: {len(xy)} columns, {spacing:g} mm spacing", loc="left")
    cells(axes[1], xy, spacing, facecolor="#E2E6EA", edgecolor="white")
    handles = [Patch(facecolor="#E2E6EA", label="Whole bed")]
    for name, color, label in (
        ("fullfoot_last", BLUE, "Full-foot driven"),
        ("rearfoot_punch", ORANGE, "Rearfoot driven"),
    ):
        fixture = artifact["instron_fixtures"][name]
        coords = np.asarray(fixture["carrier_anchor_m"])[:, :2] * 1000
        cells(axes[1], coords, fixture["spacing_m"] * 1000, facecolor=color, edgecolor="white", alpha=0.88)
        handles.append(Patch(facecolor=color, label=f"{label}: {len(coords)}"))
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.35), ncol=2, fontsize=9)
    axes[1].set_title("Stored fixture footprints (rearfoot drawn on top)", loc="left")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(xy[:, 0].min() - 8, xy[:, 0].max() + 8)
        ax.set_ylim(xy[:, 1].min() - 8, xy[:, 1].max() + 8)
        decorate(ax, "Heel → toe, x [mm]", "Lateral coordinate, y [mm]")
    save(
        fig,
        directory,
        "bed_geometry",
        "Artifact geometry: thickness and driven columns",
        "Stored two-term artifact geometry. Footprints identify driven columns, not an observed pressure or contact-area map.",
    )
    return {
        "column_count": len(xy),
        "spacing_mm": spacing,
        "thickness_range_mm": [float(thickness.min()), float(thickness.max())],
        "fixture_counts": {k: v["column_count"] for k, v in artifact["instron_fixtures"].items()},
        "sources": ["column_bed", "instron_fixtures.*.carrier_anchor_m"],
        "rendering": "Exact stored square-column footprints; rearfoot overlay is drawn last.",
    }


def pressure_arguments(parameters):
    """Derive equilibrium inputs from the active two-term material parameters."""
    p = parameters
    fraction = p["equilibrium_fraction"]
    nu = p["effective_poisson_ratio"]
    return (
        fraction * p["instantaneous_shear_modulus_pa"],
        p["hyperfoam_exponent"],
        fraction * p["instantaneous_shear_modulus_2_pa"],
        p["hyperfoam_exponent_2"],
        nu / (1 - 2 * nu),
        1 - 2 * nu,
        0.05,
    )


def pressure_figure(parameters, directory):
    """Plot both active equilibrium terms and the ideal fast Maxwell limit."""
    import matplotlib.pyplot as plt

    args = pressure_arguments(parameters)
    strain = np.linspace(0, 0.70, 701)
    stretch = np.maximum(1 - strain, args[-1])
    volume = stretch ** args[-2]
    first = ogden_hill_term_numpy(stretch, volume, args[0], args[1], args[4])
    second = ogden_hill_term_numpy(stretch, volume, args[2], args[3], args[4])
    total = hyperfoam_pressure_numpy(strain, *args)
    np.testing.assert_allclose(first + second, total, rtol=1e-13, atol=1e-9)
    fast = total / parameters["equilibrium_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3))
    axes[0].plot(strain * 100, total / 1000, color=INK, label="Two-term equilibrium total")
    axes[0].plot(strain * 100, first / 1000, color=BLUE, linestyle="--", label=f"Term 1: α₁ = {args[1]:.3f}")
    axes[0].plot(strain * 100, second / 1000, color=ORANGE, linestyle="-.", label=f"Term 2: α₂ = {args[3]:.3f}")
    axes[0].set_title("Contributions within the current two-term law", loc="left")
    axes[1].plot(strain * 100, fast / 1000, color=ORANGE, label="Ideal rapid limit: p_eq / g∞")
    axes[1].plot(strain * 100, total / 1000, color=INK, label="Fully relaxed: p_eq")
    axes[1].set_title("Ideal limits from the same fitted law", loc="left")
    axes[1].text(
        0.03,
        0.60,
        f"g∞ = {parameters['equilibrium_fraction']:.4f}\nτ = {1000 * parameters['maxwell_relaxation_time_s']:.3f} ms",
        transform=axes[1].transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    for ax in axes:
        decorate(ax, "Compression strain [%]", "Pressure [kPa]")
        ax.set_xlim(0, 70)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
    save(
        fig,
        directory,
        "material_equilibrium",
        "Nonlinear pressure: two active terms and rate limits",
        "Computed material-point curves, not measurements. Rapid limit assumes zero prior history and loading time ≪ τ.\n"
        "Effective intact-shoe law; the plotted strain range is illustrative, not a validated material domain.",
    )
    return {
        "functions": ["ogden_hill_term_numpy", "hyperfoam_pressure_numpy"],
        "module": "projects.digital_shoe.material",
        "strain_range": [0, 0.70],
        "samples": len(strain),
        "stretch_floor_assumed": 0.05,
        "stretch_floor_active": False,
        "equilibrium_shear_terms_pa": [args[0], args[2]],
        "rapid_limit": "p_eq / equilibrium_fraction; zero-history ideal instantaneous loading, not tested data",
        "pressure_at_70_percent_pa": float(total[-1]),
        "term_sum_max_error_pa": float(np.max(np.abs(first + second - total))),
    }


def material_history(strain, dt, parameters, initial_q=0.0):
    """Execute the exact shared Maxwell recurrence along an imposed strain history."""
    peq = hyperfoam_pressure_numpy(strain, *pressure_arguments(parameters))
    ratio = (1 - parameters["equilibrium_fraction"]) / parameters["equilibrium_fraction"]
    decay, ramp = maxwell_coefficients_numpy(dt, parameters["maxwell_relaxation_time_s"])
    q = np.zeros_like(peq)
    q[0] = initial_q
    for i in range(1, len(q)):
        q[i] = maxwell_step_numpy(q[i - 1], peq[i], peq[i - 1], ratio, decay, ramp)
    return peq, q


def maxwell_figure(parameters, directory):
    """Illustrate duration-dependent cycles and ideal step relaxation without experiments."""
    import matplotlib.pyplot as plt

    peak_strain = 0.35
    durations = [0.1, 0.5, 2.0]
    tau = parameters["maxwell_relaxation_time_s"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3))
    records = []
    for duration, color in zip(durations, [ORANGE, BLUE, GREEN], strict=True):
        time = np.linspace(0, duration, 4001)
        dt = float(time[1] - time[0])
        strain = peak_strain * np.sin(np.pi * time / duration) ** 2
        peq, q = material_history(strain, dt, parameters)
        support = np.maximum(peq + q, 0)
        axes[0].plot(strain * 100, support / 1000, color=color, label=f"Cycle duration {duration:g} s")
        records.append(
            {
                "duration_s": duration,
                "dt_s": dt,
                "samples": len(time),
                "peak_support_pressure_pa": float(support.max()),
                "minimum_unclipped_pressure_pa": float((peq + q).min()),
            }
        )
    eq_strain = np.linspace(0, peak_strain, 400)
    axes[0].plot(
        eq_strain * 100,
        hyperfoam_pressure_numpy(eq_strain, *pressure_arguments(parameters)) / 1000,
        color=INK,
        linestyle="--",
        linewidth=1.4,
        label="Equilibrium",
    )
    axes[0].set_title("Same 35% strain cycle, three chosen durations", loc="left")
    decorate(axes[0], "Compression strain [%]", "Unilateral support pressure [kPa]")
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(0, 35)
    axes[0].set_ylim(bottom=0)
    hold_time = np.linspace(0, 8 * tau, 1601)
    held_strain = np.full_like(hold_time, peak_strain)
    pressure_eq = float(hyperfoam_pressure_numpy(peak_strain, *pressure_arguments(parameters)))
    initial_q = pressure_eq * (1 - parameters["equilibrium_fraction"]) / parameters["equilibrium_fraction"]
    peq, q = material_history(held_strain, float(hold_time[1]), parameters, initial_q)
    expected = initial_q * np.exp(-hold_time / tau)
    np.testing.assert_allclose(q, expected, rtol=2e-12, atol=1e-8)
    axes[1].plot(hold_time * 1000, (peq + q) / 1000, color=BLUE, label="Shared recurrence after ideal step")
    axes[1].axhline(pressure_eq / 1000, color=INK, linestyle="--", linewidth=1.4, label="Equilibrium at 35% strain")
    axes[1].axvline(tau * 1000, color="#7A8993", linestyle=":", linewidth=1.4)
    axes[1].text(tau * 1000 + 0.7, (peq[0] + 0.62 * q[0]) / 1000, f"τ = {tau * 1000:.3f} ms", fontsize=10)
    axes[1].set_title("Hold after an ideal instantaneous 35% step", loc="left")
    decorate(axes[1], "Time after step [ms]", "Pressure [kPa]")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(0, hold_time[-1] * 1000)
    axes[1].set_ylim(pressure_eq * 0.93 / 1000, (peq[0] + q[0]) * 1.06 / 1000)
    save(
        fig,
        directory,
        "material_rate_dependence",
        "Illustrative Maxwell response — not experimental validation",
        "Two-term p_eq plus the shared Maxwell recurrence. Cycles: ε = 0.35 sin²(πt/T), initially relaxed; support = max(p_eq + q, 0).\n"
        "Single imposed material point; no fixture geometry, neighbor coupling or normal damping. The ideal step is not a measured test.",
    )
    return {
        "provenance": "Fresh illustrative material-point calculations, no experimental comparison",
        "functions": ["hyperfoam_pressure_numpy", "maxwell_coefficients_numpy", "maxwell_step_numpy"],
        "cycle_strain": "0.35*sin(pi*t/T)**2",
        "peak_strain": peak_strain,
        "cycle_initial_overstress_pa": 0,
        "cycle_support": "maximum(p_eq + q, 0)",
        "cycles": records,
        "omitted": ["geometry", "Pasternak coupling", "normal damping"],
        "step_hold": {
            "duration_s": float(hold_time[-1]),
            "dt_s": float(hold_time[1]),
            "initial_q_pa": initial_q,
            "held_strain": peak_strain,
            "equilibrium_pressure_pa": pressure_eq,
            "analytic_decay_max_error_pa": float(np.max(np.abs(q - expected))),
        },
    }


def validation_figure(curve, directory, stem):
    """Redraw the artifact's measured and predicted held-out arrays unchanged."""
    import matplotlib.pyplot as plt

    title = {"rearfoot_punch": "Rearfoot punch", "fullfoot_last": "Full-foot last"}[curve["fixture"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    measured = np.asarray(curve["measured_force_n"])
    predicted = np.asarray(curve["predicted_force_n"])
    for ax, values, xlabel in zip(
        axes,
        [curve["time_s"], np.asarray(curve["displacement_m"]) * 1000],
        ["Stored cycle time [s]", "Stored displacement [mm]"],
        strict=True,
    ):
        ax.plot(values, measured, color=INK, label="Stored measurement")
        ax.plot(values, predicted, color=ORANGE, linestyle="--", label="Stored two-term prediction")
        decorate(ax, xlabel, "Force [N]")
        ax.legend(loc="upper left")
        ax.set_ylim(bottom=0)
    axes[0].set_title("Force–time", loc="left")  # noqa: RUF001
    axes[1].set_title("Force–displacement", loc="left")  # noqa: RUF001
    metrics = curve["metrics"]
    recomputed = validate_trace_metrics(measured, predicted, np.asarray(curve["displacement_m"])).as_dict()
    for name, value in metrics.items():
        if isinstance(value, (bool, int)):
            if recomputed[name] != value:
                raise AssertionError(f"Stored {curve['name']} {name} does not agree with official recomputation")
        else:
            np.testing.assert_allclose(
                recomputed[name], value, rtol=1e-5, atol=1e-9, err_msg=f"Stored {curve['name']} {name}"
            )
    note = (
        f"Stored held-out cycles {', '.join(map(str, curve['cycles']))}; 501 samples over 0.5 s. Predictions were not rerun. "
        f"Stored relative force RMSE: {metrics['force_rmse_relative']:.2%}.\n"
        "Adjacent cycles from the same fixture protocol; not validation at new rates, temperatures, impacts or shoes."
    )
    save(fig, directory, stem, f"{title}: saved held-out response (not rerun)", note)
    return {
        "provenance": "Exact saved artifact arrays, only displacement unit converted m to mm",
        "artifact_key": f"validation.curves[name={curve['name']}]",
        "fixture": curve["fixture"],
        "cycles": curve["cycles"],
        "samples": len(measured),
        "time_range_s": [curve["time_s"][0], curve["time_s"][-1]],
        "metrics_stored": metrics,
        "metrics_official_recomputed": recomputed,
        "metric_function": "projects.digital_instron_v2.validation.validate_trace_metrics",
        "metric_recomputation": {
            "agrees_with_stored_within_tolerance": True,
            "rtol": 1e-5,
            "atol": 1e-9,
            "active_fraction": 0.05,
            "top_count": 5,
            "pass_threshold": 0.1,
            "note": "Stored summary metrics and saved trace recomputation differ slightly; they are not bitwise equal. Original arrays and metrics are preserved.",
            "recomputed_minus_stored": {k: recomputed[k] - v for k, v in metrics.items() if not isinstance(v, bool)},
        },
        "prediction_rerun": False,
        "measured_arrays": ["time_s", "displacement_m", "measured_force_n"],
        "predicted_arrays": ["predicted_force_n"],
    }


def bristle_figure(directory):
    """Plot cyclic friction from the actual shared Warp bristle implementation."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    wp.init()
    settings = {
        "normal_n": 10.0,
        "kt_n_per_m": 10000.0,
        "kv_n_s_per_m": 10.0,
        "mu": 0.8,
        "viscous_ratio": 0.2,
        "release_dwell_s": 0.0005,
        "amplitude_m": 0.0024,
        "period_s": 0.8,
        "dt_s": 0.0002,
        "cycles": 2,
    }
    dt = settings["dt_s"]
    time = np.arange(round(settings["cycles"] * settings["period_s"] / dt) + 1) * dt
    # Supply positions at the start of each increment: bristle_step evaluates p_next = p + v*dt.
    end_position = settings["amplitude_m"] * np.sin(2 * np.pi * time / settings["period_s"])
    start_position = np.r_[0, end_position[:-1]]
    velocity = (end_position - start_position) / dt
    force_array = wp.empty(len(time), dtype=float, device="cpu")
    anchor_array = wp.empty(len(time), dtype=float, device="cpu")
    wp.launch(
        bristle_history,
        dim=1,
        inputs=[
            wp.array(start_position, dtype=float, device="cpu"),
            wp.array(velocity, dtype=float, device="cpu"),
            dt,
            settings["normal_n"],
            settings["kt_n_per_m"],
            settings["kv_n_s_per_m"],
            settings["mu"],
            settings["viscous_ratio"],
            settings["release_dwell_s"],
            force_array,
            anchor_array,
        ],
        device="cpu",
    )
    force = force_array.numpy()
    anchor = anchor_array.numpy()
    cap = settings["normal_n"] * settings["mu"]
    if np.max(np.abs(force)) > cap + 2e-4:
        raise AssertionError("Shared bristle output exceeded the Coulomb cap")
    first = time < settings["period_s"]
    final = time >= settings["period_s"]
    x = end_position[final] * 1000
    y = force[final]
    moved = np.abs(np.diff(anchor[final])) > 1e-9
    segments = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.plot(end_position[first] * 1000, force[first], color="#B8C2C9", linestyle="--", linewidth=1.2)
    ax.add_collection(LineCollection(segments, colors=np.where(moved, ORANGE, BLUE), linewidth=2.5))
    ax.axhline(cap, color="#8B979F", linestyle=":", linewidth=1)
    ax.axhline(-cap, color="#8B979F", linestyle=":", linewidth=1)
    amplitude_mm = settings["amplitude_m"] * 1000
    ax.set(xlim=(-1.12 * amplitude_mm, 1.12 * amplitude_mm), ylim=(-1.34 * cap, 1.34 * cap))
    decorate(ax, "Tangential position [mm]", "Tangential contact force [N]")
    ax.set_title("Illustrative N = 10 N; kₜ = 10,000 N/m; μ = 0.8; kᵥ = 10 N·s/m", loc="left", fontsize=11, pad=38)
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, label="Elastic stick (anchor fixed)"),
            Line2D([0], [0], color=ORANGE, label="Slip (anchor return)"),
            Line2D([0], [0], color="#B8C2C9", linestyle="--", label="First cycle"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=9,
    )
    ax.text(1.08 * amplitude_mm, 1.05 * cap, "+μN", ha="right", fontsize=9)
    ax.text(1.08 * amplitude_mm, -1.12 * cap, "−μN", ha="right", fontsize=9)  # noqa: RUF001
    for phase in (0.07, 0.35, 0.59, 0.86):
        index = int(phase * (len(x) - 1))
        ahead = min(index + 60, len(x) - 1)
        ax.annotate(
            "",
            xy=(x[ahead], y[ahead]),
            xytext=(x[index], y[index]),
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.25},
        )
    save(
        fig,
        directory,
        "contact_bristle",
        "Illustrative bristle cycle — shared Warp contact law",
        "Declared illustrative contact settings with fixed N; not Instron-fitted or experimental validation.\n"
        "Second sinusoidal cycle shown in color: ±2.4 mm, period 0.8 s, Δt = 0.2 ms. N stays fixed; no lift-off.\n"
        "Viscous ratio = 0.2; release dwell = 0.5 ms (inactive). Force is the reaction on the moving contact.",
    )
    work = float(np.sum(0.5 * (force[1:] + force[:-1]) * np.diff(end_position)))
    return {
        "provenance": "Fresh Warp CPU execution of projects.digital_shoe.contact.bristle_step",
        "settings_assumed_not_identified": settings,
        "contact_defaults_source": "Illustrative settings declared in projects/digital_shoe/mechanics_report/figures.py",
        "illustrative_inputs": ["normal_n", "amplitude_m", "period_s", "dt_s", "cycles"],
        "elastic_slip_limit_m": cap / settings["kt_n_per_m"],
        "device": "cpu",
        "warp_version": wp.__version__,
        "position_history": "amplitude*sin(2*pi*t/period)",
        "discretization": "position supplied at increment start, velocity=(end-start)/dt, plotted at p_next",
        "classification": "Slip if returned anchor changes by >1e-9 m; grip flag is not a slip indicator",
        "colored_cycle": 2,
        "coulomb_cap_n": cap,
        "maximum_abs_force_n": float(np.max(np.abs(force))),
        "trapezoid_contact_work_over_two_cycles_j": work,
        "release_dwell_exercised": False,
        "artifact_parameters_used": False,
    }


def _display_path(path: Path) -> str:
    """Record checkout-relative paths when possible, else absolute paths."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def generate_figures(artifact_path: Path, output_dir: Path) -> dict:
    """Redraw six fixed two-term figure pairs and write their provenance.

    The approved artifact hash, two active terms, zero effective Poisson ratio,
    and audited source hashes are verified before any output is written.
    Saved held-out predictions are not rerun. Material and contact calculations
    use the shared current laws and are illustrative, not validation data.
    Matplotlib is optional until this function starts plotting.

    Args:
        artifact_path: Approved saved artifact JSON. Relative paths resolve from
            the repository root. A byte-identical copy outside it is supported.
        output_dir: Destination directory for six PNG/SVG pairs and
            ``metadata.json``. Relative paths resolve from the repository root.

    Returns:
        The same provenance and numerical checks written to ``metadata.json``.

    Raises:
        ValueError: Artifact or source provenance does not match the audit.
        ImportError: Optional Matplotlib is unavailable for figure generation.
        AssertionError: A material, contact, or official metric check fails.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    artifact_path = artifact_path.resolve()
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    artifact = load_verified_artifact(artifact_path, manifest_path=MANIFEST)
    parameters = artifact["constitutive_model"]["parameters"]
    if artifact["constitutive_model"]["derived_quantities"]["hyperfoam_term_count"] != 2:
        raise ValueError("This report requires the active two-term model")
    digest = sha256(artifact_path)
    style()
    import matplotlib

    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "bed_geometry": bed_figure(artifact, directory),
        "material_equilibrium": pressure_figure(parameters, directory),
        "material_rate_dependence": maxwell_figure(parameters, directory),
    }
    stems = {"rearfoot_punch": "validation_rearfoot", "fullfoot_last": "validation_fullfoot"}
    for curve in artifact["validation"]["curves"]:
        stem = stems[curve["fixture"]]
        figures[stem] = validation_figure(curve, directory, stem)
    figures["contact_bristle"] = bristle_figure(directory)
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("_provenance.py").resolve(),
        MANIFEST,
        ROOT / "projects/digital_shoe/material.py",
        ROOT / "projects/digital_shoe/contact.py",
        ROOT / "projects/digital_instron_v2/validation.py",
    ]
    command = shlex.join(
        [
            "uv",
            "run",
            "--no-sync",
            "-m",
            "projects.digital_shoe.mechanics_report.figures",
            "--artifact",
            _display_path(artifact_path),
            "--output",
            _display_path(directory),
        ]
    )
    metadata = {
        "artifact_path": _display_path(artifact_path),
        "artifact_sha256": digest,
        "model_scope": artifact["shoe"]["model_scope"],
        "hyperfoam_term_count": 2,
        "material_parameters": parameters,
        "artifact_source_provenance": artifact["provenance"],
        "validation_scope": artifact["validation"]["scope"],
        "claim_boundary": artifact["validation"]["claim_boundary"],
        "source_sha256": {_display_path(p): sha256(p) for p in source_paths},
        "command": command,
        "figure_sha256": {
            f"{stem}.{suffix}": sha256(directory / f"{stem}.{suffix}") for stem in figures for suffix in ("svg", "png")
        },
        "libraries": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "warp": wp.__version__},
        "output_formats": ["svg", "png"],
        "png_dpi": 210,
        "figures": figures,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    """Regenerate figures and write exact provenance plus numerical self-checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="Approved two-term artifact JSON (relative paths start at the repository root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT / "figures",
        help="Output directory (default: outputs/footwear_contact_material_report/figures)",
    )
    args = parser.parse_args()
    metadata = generate_figures(args.artifact, args.output)
    directory = args.output if args.output.is_absolute() else ROOT / args.output
    print(f"Wrote {len(metadata['figures'])} SVG/PNG pairs and metadata to {directory.resolve()}")
    print(f"Verified artifact SHA256: {metadata['artifact_sha256']}")


if __name__ == "__main__":
    main()
