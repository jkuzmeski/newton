# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build one self-contained HTML report of the foam-sweep material fit.

The report reads the ``material_fit.json`` of each shoe, the processed sweep,
the registration QC and the optional old-law fits and kernel profile from disk,
replays every trace once with the fitted material through
:class:`~projects.digital_instron_v2.last_replay.LastReplay` (one short GPU
run per shoe), and writes one HTML file with every figure inlined::

    uv run --extra dev python -m projects.digital_instron_v2.sweep_report \
        --out DigitalInstron/FoamSweep/fit/report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import time
from pathlib import Path

import numpy as np

from projects.digital_shoe.column_bed import BOTTOM_LANDED, TOP_HELD
from projects.digital_shoe.material import ogden_hill_term_numpy
from projects.digital_shoe.mechanics_report.figures import BLUE, GREEN, INK, ORANGE, cells, decorate, style

from .core import FIT_LOWER_BOUNDS, FIT_UPPER_BOUNDS, Material, metrics
from .sweep_fit import DEFAULT_LAST, identifiability, load_grid, load_trace, near_bounds

SWEEP = Path("DigitalInstron/FoamSweep")
SHOES = {"puma-fast-r-nitro-elite-3": "FR3", "puma-deviate-nitro-elite-3": "Dev3Elite"}
NAMES = {"FR3": "Puma Fast-R Nitro Elite 3 (FR3)", "Dev3Elite": "Puma Deviate Nitro Elite 3 (Dev3Elite)"}
BACKUPS = SWEEP / "fit" / "history"
"""Earlier fits shown for comparison and used as fit starts."""
STANCES = (160, 200, 240, 280)
FORCES = (800, 1200, 1600, 2000, 2400, 2800)
MAP_FORCES = (800, 2000, 2800)
NEAR_BEST = 0.25
"""Starts within this fraction of the best training loss count as near-best."""

FIELDS = {
    "instantaneous_shear_modulus_pa": ("G1, term 1 shear modulus", "kPa", 1.0e-3),
    "hyperfoam_exponent": ("alpha1, term 1 exponent", "-", 1.0),
    "instantaneous_shear_modulus_2_pa": ("G2, term 2 shear modulus", "kPa", 1.0e-3),
    "hyperfoam_exponent_2": ("alpha2, term 2 exponent", "-", 1.0),
    "equilibrium_fraction": ("f_eq, equilibrium fraction", "-", 1.0),
    "maxwell_relaxation_time_s": ("tau, Maxwell relaxation time", "ms", 1.0e3),
}
STATES = (
    ("held, landed", "#1b9e77"),
    ("held, floating", "#d95f02"),
    ("open, landed", "#7570b3"),
    ("open, floating", "#d9d9d9"),
    ("outside, landed", "#e7298a"),
    ("outside, floating", "#f2f2f2"),
)

CSS = """
body{margin:0;background:#eef2f5;color:#1d2b38;font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 32px 60px;background:#fff}
h1{font-size:28px;margin:8px 0 4px}h2{margin-top:42px;border-bottom:2px solid #d6dee5;padding-bottom:4px}
h3{margin-top:26px}nav a{margin-right:14px;font-size:14px}
table{border-collapse:collapse;margin:10px 0 16px;font-size:13.5px}
th,td{border:1px solid #d6dee5;padding:4px 8px;text-align:right}th{background:#f3f6f8}
td:first-child,th:first-child{text-align:left}td.bad{background:#fde3dc}td.flag{background:#fff2c6}
img{max-width:100%;display:block;margin:10px 0}figure{margin:16px 0}figcaption{color:#4f6070;font-size:13.5px}
pre{background:#f5f7f9;padding:10px 14px;overflow-x:auto;font-size:13.5px}code{font-size:13.5px}
.note{color:#4f6070;font-size:13.5px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
"""


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def table(headers: list[str], rows: list[list], classes: list[list[str]] | None = None) -> str:
    """Return an HTML table; cells are inserted as given, so escape text first."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for r, row in enumerate(rows):
        cls = classes[r] if classes else [""] * len(row)
        body.append("<tr>" + "".join(f'<td class="{c}">{v}</td>' for v, c in zip(row, cls, strict=True)) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def image(data: bytes, mime: str, caption: str) -> str:
    payload = base64.b64encode(data).decode("ascii")
    return f'<figure><img src="data:{mime};base64,{payload}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def figure(fig, caption: str) -> str:
    """Inline a matplotlib figure as PNG."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return image(buffer.getvalue(), "image/png", caption)


def label(stance_ms: int, force_n: int, repeat: bool = False) -> str:
    return f"{stance_ms}ms_{force_n}N" + ("_repeat" if repeat else "")


def fitted(fit: dict, key: str, split: str) -> dict:
    return fit["conditions"][key][split]["fitted"]


# ---------------------------------------------------------------- model replay


def replay(slug: str, fit: dict) -> dict:
    """Replay every sweep trace once with the fitted material, set up as in :func:`sweep_fit.run`."""
    from .last_replay import LastReplay, last_bed  # noqa: PLC0415
    from .registration import load_last, register_last  # noqa: PLC0415

    sweep = SWEEP / "processed" / SHOES[slug]
    grid = load_grid(Path("DigitalInstron") / slug / fit["side"] / "column_grid.npz")
    index = json.loads((sweep / "conditions.json").read_text())
    preload = float(np.median([c["preload_n"] for c in index["conditions"]]))
    registration = register_last(
        grid,
        load_last(DEFAULT_LAST),
        modulus_pa=3.0 * Material(**fit["initial_material"]).equilibrium_shear_modulus_pa,
        preload_n=preload,
        expect_side=fit["side"],
    )
    bed = last_bed(grid, registration)
    keys, traces = [], []
    for c in index["conditions"]:
        repeat = c["role"] == "repeat"
        for split in ("repeat",) if repeat else ("train", "validate"):
            keys.append((label(c["stance_ms"], c["target_n"], repeat), split))
            traces.append(load_trace(sweep / c[f"{split}_path"]))
    indentation = registration.qc["preload"]["indentation_m"]
    depth = np.stack([np.maximum(t["displacement_m"], 0.0) + indentation for t in traces], axis=1)
    dt = float(np.mean(np.diff(traces[0]["time_s"])))
    peaks = np.argmax(depth, axis=0)
    model = LastReplay(bed, depth, dt, candidates=1, substeps=fit["substeps"])
    force, columns = model.run(
        [Material(**fit["material"])], offsets_m=[fit.get("start_offset_m", 0.0)], snapshot_frames=peaks
    )
    reaction = model.reaction.numpy().T.reshape(force.shape)
    return {
        "index": index,
        "registration": registration,
        "bed": bed,
        "keys": keys,
        "traces": traces,
        "force": force[0],
        "reaction": reaction[0],
        "columns": columns,
    }


def states(columns: dict, world: int, bed) -> np.ndarray:
    """Return each column's index into :data:`STATES` at the snapshot frame."""
    support = columns["support"][world]
    held = (support & TOP_HELD) != 0
    landed = (support & BOTTOM_LANDED) != 0
    return np.select(
        [bed.tread & held & landed, bed.tread & held, bed.tread & landed, bed.tread, landed],
        [0, 1, 2, 3, 4],
        5,
    )


def flagged(fit: dict, r: dict) -> dict[tuple[str, str], dict]:
    """Return the fresh metrics of traces whose stored score the fresh replay does not reproduce.

    A trace is flagged when its stored peak error differs from the fresh replay
    by more than one percentage point or its stored column strain exceeds one
    (compression beyond the rest thickness).
    """
    out = {}
    for k, (key, split) in enumerate(r["keys"]):
        stored = fit["conditions"][key][split]["fitted"]
        fresh = metrics(r["traces"][k]["force_n"], r["force"][k], r["traces"][k]["displacement_m"])
        if abs(fresh["peak_force_error"] - stored["peak_force_error"]) > 0.01 or stored["contact"]["max_strain"] > 1.0:
            out[(key, split)] = {**fresh, "stored": stored}
    return out


def flag_note(shoes: dict) -> str:
    items = [
        f"{SHOES[slug]} {key.replace('_', ' ')} {split}: stored peak error {pct(v['stored']['peak_force_error'])}%, "
        f"stored max strain {v['stored']['contact']['max_strain']:.2f}; fresh replay peak error {pct(v['peak_force_error'])}%, "
        f"RMSE {pct(v['force_rmse_relative'])}%, loop {pct(v['hysteresis_error'])}%"
        for slug, s in shoes.items()
        for (key, split), v in s["flagged"].items()
    ]
    if not items:
        return ""
    return (
        '<p class="note"><b>Flagged traces.</b> The stored fit score of these traces is a solver blow-up that a fresh '
        "replay of the same material does not reproduce. The last load and platen reaction blew up together, so the "
        "balance check did not catch it. Ranges above leave them out; tables below show the stored value.</p><ul>"
        + "".join(f"<li>{html.escape(item)}</li>" for item in items)
        + "</ul>"
    )


# --------------------------------------------------------------------- sections


def summary_section(shoes: dict, old: dict) -> str:
    rows = []
    for slug, s in shoes.items():
        fit = s["fit"]
        val = [
            fitted(fit, label(t, f), "validate")
            for t in STANCES
            for f in FORCES
            if (label(t, f), "validate") not in s["flagged"]
        ]
        rep = [fitted(fit, label(200, f, True), "repeat") for f in FORCES]
        m = fit["material"]
        rows.append(
            [
                html.escape(NAMES[SHOES[slug]]),
                f"{pct(min(v['peak_force_error'] for v in val))}-{pct(max(v['peak_force_error'] for v in val))}",
                f"{pct(min(v['force_rmse_relative'] for v in val))}-{pct(max(v['force_rmse_relative'] for v in val))}",
                f"{pct(min(v['hysteresis_error'] for v in val))}-{pct(max(v['hysteresis_error'] for v in val))}",
                f"{pct(max(v['peak_force_error'] for v in rep))} / {pct(max(v['force_rmse_relative'] for v in rep))}",
                f"{sum(v['passed'] for v in val)} / 24",
                f"G1 {m['instantaneous_shear_modulus_pa'] / 1e3:.0f} kPa, alpha1 {m['hyperfoam_exponent']:.2f}, "
                f"f_eq {m['equilibrium_fraction']:.3f}, tau {m['maxwell_relaxation_time_s'] * 1e3:.1f} ms",
            ]
        )
    change = ""
    if old:
        change = (
            " With the old single-face law the same fit pipeline reached the errors in "
            '<a href="#old">section 6</a>; the new law removes the floating-column problem described there.'
        )
    gif = SWEEP / "fit" / "instron_fr3_dev3.gif"
    animation = (
        image(gif.read_bytes(), "image/gif", "Replay of both shoes in the digital Instron.") if gif.exists() else ""
    )
    return f"""<h2 id="summary">1. Summary</h2>
<p>One material is fitted per shoe to its Instron foam sweep: a two-term Hyperfoam (Ogden-Hill) equilibrium network
with one Maxwell branch, 6 parameters. Every sweep trace is replayed through the GPU column bed with the
<b>new two-face column contact law</b> (<code>ColumnBed</code>): each midsole column has a top pressed by the last and a
bottom that lands on the platen. The fit uses train cycles only; validate and repeat traces are scored, not fitted.</p>
{table(["Shoe", "Validate peak error %", "Validate RMSE %", "Validate loop-area error %", "Repeat block max peak / RMSE %", "Validate traces within all 10% gates", "Fitted material"], rows)}
<p class="note">Errors are relative to the measured peak force (peak, RMSE) and to the measured loop area (hysteresis).
Ranges are over the 24 validate traces. The repeat block is 6 traces.{change}</p>
{flag_note(shoes)}
{animation}"""


def data_section(shoes: dict) -> str:
    out = [
        """<h2 id="data">2. Data</h2>
<p>Each shoe was cycled on the Instron with a full-foot last at a 0.5 s period. The programmed grid is
4 stance times (160, 200, 240, 280 ms) x 6 peak forces (800 to 2800 N in 400 N steps), 100 cycles per condition.
Each condition gives two phase-averaged traces (501 samples per cycle):</p>
<ul><li><b>train</b>: mean of cycles 90-98, used by the fit;</li>
<li><b>validate</b>: mean of cycles 99-100, only scored.</li></ul>
<p>The programmed 320 ms block is in fact a second copy of the 200 ms block (same loaded duration). It is not a new
stance time; it is used as a <b>repeatability check</b> (the "repeat" traces).</p>"""
    ]
    for slug, s in shoes.items():
        conditions = s["replay"]["index"]["conditions"]
        by = {(c["stance_ms"], c["target_n"], c["role"]): c for c in conditions}
        rows = [
            [f"{t} ms"]
            + [
                f"{by[(t, f, 'condition')]['measured_peak_n']:.0f} N, {by[(t, f, 'condition')]['loaded_duration_s'] * 1e3:.0f} ms"
                for f in FORCES
            ]
            for t in STANCES
        ]
        repeat = [
            ["repeat - original"]
            + [
                f"{by[(200, f, 'repeat')]['repeat_peak_difference_n']:+.1f} N peak, {by[(200, f, 'repeat')]['repeat_force_rmse_n']:.1f} N RMSE"
                for f in FORCES
            ]
        ]
        out.append(f"<h3>{html.escape(NAMES[SHOES[slug]])}</h3>")
        out.append(table(["Stance"] + [f"{f} N" for f in FORCES], rows + repeat))
        out.append(
            '<p class="note">Cells: measured peak force and loaded duration (force above the preload). '
            "Last row: the 320 ms (repeat) block minus the 200 ms block. Median preload "
            f"{np.median([c['preload_n'] for c in conditions]):.1f} N.</p>"
        )
    return "\n".join(out)


REGISTRATION_NOTES = """<p><b>Caveats</b></p>
<ul>
<li>FR3: the last tread just fits the narrow FR3 waist. The outline sets x, y and yaw, not the surface fit.</li>
<li>First contact is at the heel cup rim and heel. The forefoot footbed sits 5-9 mm below the last plantar surface.</li>
<li>The last side (left) comes from the file name. Mirror check: the left last does not fit inside the right FR3
outline at all, and fits the right Deviate worse (spread 0.88 vs 0.76 mm). The Deviate grid side cue is weak
('unverified'). shoe_prep says both scans are right shoes, so left = mirror.</li>
<li>The midsole is rigid while it rocks. The preload uses linear columns.</li>
</ul>
<p><b>Legacy manifest pose</b> (<code>manifest_v2.json</code> fullfoot_last), for comparison with older results</p>
<ul>
<li><code>geometry.load_mesh</code> reorders the STL axes to (z, y, x). This is a mirror (det -1): the left last became
a right last. The legacy obj is a right shoe, so the pair still had matching sides.</li>
<li><code>rotation_deg [90, 0, 0]</code> makes up = STL +y. That is 3.3 deg off the plate normal.
<code>crop_height_m 0.08</code> removes the stud and plate. Then <code>pose_rotation_deg [-1.0, -1.8, 0]</code> makes
the error 5.3 deg. The load axis in the shoe frame was about 4.7 deg pitch, toe down, plus 2.5 deg roll. That pitch
matches the FR3 rocker rest pitch in the table above.</li>
<li>Heel ends were on the same side. The last heel sat about 27 mm in from the midsole heel edge, at y = -6 mm. This
came from bounding-box centering, not from heel alignment.</li>
<li>The 95th-percentile contact offset started 31 of 611 columns compressed by up to 1.4 mm.</li>
</ul>"""


def registration_section(shoes: dict) -> str:
    import matplotlib.pyplot as plt

    out = [
        """<h2 id="shoe">3. Shoe and last</h2>
<p>The last pose over the midsole column grid comes from <code>registration.register_last</code>, run inside
the fit. Inputs are the shoe_prep <code>left/column_grid.npz</code> files (canonical frame, heel at -x). The preload
contact uses a linear column modulus of 3 x the initial equilibrium shear modulus; it only sets the 20 N preload
estimate, and the rest pitch changes by less than 0.3 deg when it is halved or doubled.</p>
<p><b>Method</b></p>
<ul>
<li>Last up = normal of the square attachment plate. It agrees with the stud axis to 0.002 deg. It is 3.3 deg off the
STL +y axis, so last pitch and roll are fixed by the attachment and never fitted.</li>
<li>Last heel = the narrow end (23 mm narrower). The ball tread (lowest plantar point) sits at 70% of the length.
The heel seat is 5.1 mm above the ball tread.</li>
<li>Shoe heel = the thicker and narrower end (<code>canonical_frame</code>).</li>
<li>The midsole rocks on the platen. Its pitch and roll minimize the potential energy of the 20 N preload on linear
columns. A rocker sole tips until the last contact is balanced over its platen contact.</li>
<li>Last x, y and yaw minimize the soft-L1 spread of the last-to-top clearance. The last tread (surface within 45 deg
of horizontal) must stay inside the midsole outline. z is set at first contact.</li>
</ul>
<p>Current pose of each fit (from <code>material_fit.json</code>, left shoe, left last):</p>"""
    ]
    rows = []
    for slug, s in shoes.items():
        qc = s["fit"]["registration"]["qc"]
        pose = s["fit"]["registration"]["pose"]
        g = qc["gap_percentiles_mm"]
        rows.append(
            [
                SHOES[slug],
                f"{qc['rest_pitch_deg']:.2f} / {qc['rest_roll_deg']:.2f}",
                f"{qc['yaw_deg']:.2f}",
                f"{pose['x_m'] * 1e3:.1f}, {pose['y_m'] * 1e3:.1f}",
                f"{qc['overlap_columns']} / {qc['columns']}",
                f"{qc['platen_columns']}",
                f"{g['p5']:.1f} / {g['p50']:.1f} / {g['p95']:.1f}",
                f"{qc['preload']['indentation_m'] * 1e3:.1f} mm, {qc['preload']['compressed_columns']}",
                f"{100 * qc['last_outside_fraction']:.1f}",
                f"{qc['heel_offset_m'] * 1e3:.1f}",
            ]
        )
    out.append(
        table(
            [
                "Shoe",
                "Midsole rest pitch / roll [deg]",
                "Last yaw [deg]",
                "Last x, y [mm]",
                "Columns under the last",
                "Columns on the platen at rest",
                "Gap p5 / p50 / p95 [mm]",
                "Preload indentation, columns",
                "Tread outside outline %",
                "Last heel behind midsole heel end [mm]",
            ],
            rows,
        )
    )
    out.append(REGISTRATION_NOTES)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for (slug, s), color in zip(shoes.items(), (BLUE, ORANGE), strict=False):
        bed = s["replay"]["bed"]
        axes[0].hist(bed.gap_m[bed.tread] * 1e3, bins=40, alpha=0.6, color=color, label=SHOES[slug])
        axes[1].hist(bed.bottom_m * 1e3, bins=40, alpha=0.6, color=color, label=SHOES[slug])
    decorate(axes[0], "last-to-top gap g at first contact [mm]", "columns under the last")
    decorate(axes[1], "bottom clearance h over the platen at rest [mm]", "columns")
    axes[0].legend()
    out.append(
        figure(
            fig,
            "Gap distribution. Left: gap g between the last and each column top under the last tread. "
            "Right: clearance h between each column bottom and the platen (most columns float at rest; "
            "only the columns near 0 stand on the platen).",
        )
    )
    for slug in shoes:
        png = SWEEP / "fit" / slug / "registration.png"
        if png.exists():
            out.append(image(png.read_bytes(), "image/png", f"{SHOES[slug]}: registration QC written by the fit."))
    return "\n".join(out)


def contact_law_section() -> str:
    return """<h2 id="law">4. Contact law: two-face columns</h2>
<p>The midsole is a grid of vertical columns (about 6 mm spacing). Each column has rest thickness L, footprint area A,
and <b>two unknowns</b>: the downward displacement of its top, t, and of its bottom, b. The compression is
c = t - b. The last presses the tops; the platen supports the bottoms. Both contacts are one-way.</p>
<pre>
compression            c = t - b,     0 &lt;= c &lt; L
last on the top        t &gt;= d - g     (d = last stroke, g = rest gap; the last pushes, never pulls)
platen on the bottom   b &lt;= h         (h = rest clearance of the bottom over the platen)
outside the last       top free

energy   E = sum_i A_i W(c_i)  +  1/2 sum_faces k_ij [ (t_i - t_j)^2 + (b_i - b_j)^2 ]
         W'(c) = p(c/L): two-term Hyperfoam equilibrium pressure + Maxwell overstress
         k_ij  = mu_eq (L_i + L_j) / 2,    mu_eq = (G1 + G2) f_eq     (Pasternak shear layer, no new parameter)

reactions   T_i = last on top i (held tops only),   R_i = platen on bottom i (landed bottoms only)
balance     sum_i T_i = sum_i R_i   (the shear terms cancel over the bed)
</pre>
<p><b>No cross term.</b> The energy has one shear layer at the tops and one at the bottoms, and nothing that couples a
top difference to a bottom difference. An earlier variant had such a cross term. It lifted thick edge columns off the
platen, so it was dropped.</p>
<p><b>Four column states</b> (for columns under the last tread):</p>
<ul>
<li><b>held, landed</b>: the last presses the top and the bottom stands on the platen; the foam carries the load straight through.</li>
<li><b>held, floating</b>: the last presses the top, the bottom is in the air; the column passes its load sideways to
neighbours through the bottom shear layer.</li>
<li><b>open, landed</b>: the bottom stands on the platen, the last does not touch the top; it carries only the shear its
neighbours put on it.</li>
<li><b>open, floating</b>: neither face touches; the column only follows its neighbours.</li>
</ul>
<p>Columns outside the last tread have a free top and are only <i>landed</i> or <i>floating</i>.</p>
<p><b>Range of shear support.</b> A held column with a floating bottom is supported only by shear from landed
neighbours. In the continuum limit the bottom-layer stiffness per area is mu_eq L and the foam stiffness per area is
E/L with E = 2 mu_eq at small strain, so the support decays over a length
lambda = sqrt(mu_eq L^2 / E) = L/sqrt(2), about 14-25 mm for 20-35 mm thick columns, and less once the foam stiffens
at large strain. A floating region much wider than that gets almost no support.</p>
<p><b>Solver.</b> One CUDA block per world runs red-black Gauss-Seidel sweeps. Each column update solves its local
2x2 problem exactly over the active constraints. A step limiter (fraction-to-boundary, factor 0.5) lets one update
close at most half of the remaining open thickness L - c, so a far-off step cannot jump into the foam lock-up near
c = L. Converged states are not changed by it.</p>
<p><b>Physical checks.</b> A replayed cycle counts as physical only if T and R are finite, the platen only pushes,
and |sum T - sum R| stays within 2% of the peak load every frame (<code>last_replay.physical</code>). A cycle that
fails returns NaN. In the fit, a start whose own replay is not physical stops with a message, a Jacobian column is
differenced from the other side, and the final scoring raises instead of writing a result. So a broken solve fails
loudly instead of giving a smooth wrong force.</p>
<p><b>Old single-face law.</b> The old bed had one unknown per column: its compression, set by the last's position,
with the bottom fixed. A column whose bottom floats above the platen (arch, rocker toe and heel) cannot be described.
It either carried nothing (old law, option A) or it was treated as standing on the platen and was crushed by the
stroke plus its clearance (the "grounded columns" stopgap). The new law lets the bottom land when it reaches the
platen, so floating columns carry load only once they really touch.</p>"""


def material_rows(fit: dict) -> str:
    near = set(near_bounds(fit["material"]))
    rows, classes = [], []
    names = list(Material.__dataclass_fields__)
    lower, upper = dict(zip(names, FIT_LOWER_BOUNDS, strict=True)), dict(zip(names, FIT_UPPER_BOUNDS, strict=True))
    for key, (name, unit, scale) in FIELDS.items():
        low, high = lower[key], upper[key]
        flag = key in near
        rows.append(
            [
                html.escape(name),
                unit,
                f"{fit['initial_material'][key] * scale:.4g}",
                f"<b>{fit['material'][key] * scale:.4g}</b>",
                f"{low * scale:.4g}",
                f"{high * scale:.4g}",
                "at bound" if flag else "",
            ]
        )
        classes.append(["", "", "", "", "", "", "flag" if flag else ""])
    mu = Material(**fit["material"]).equilibrium_shear_modulus_pa
    mu0 = Material(**fit["initial_material"]).equilibrium_shear_modulus_pa
    rows.append(["mu_eq = (G1 + G2) f_eq (derived)", "kPa", f"{mu0 / 1e3:.4g}", f"<b>{mu / 1e3:.4g}</b>", "", "", ""])
    classes.append([""] * 7)
    return table(["Parameter", "Unit", "Start (calibrated)", "Fitted", "Lower bound", "Upper bound", ""], rows, classes)


def multistart_tables(fit: dict) -> str:
    starts = fit["optimization"]["starts"]
    best = min(s["loss"] for s in starts)
    rows = []
    for k, s in enumerate(starts):
        m = s.get("material")
        rows.append(
            [
                f"{k}" + (" (calibrated)" if k == 0 else ""),
                f"{s['loss']:.5f}",
                f"{s['loss'] / best:.2f}",
                str(s.get("residual_evaluations", "-")),
                ", ".join(s.get("near_bounds", [])) or "-",
            ]
            + ([f"{m[key] * scale:.4g}" for key, (_, _, scale) in FIELDS.items()] if m else ["-"] * len(FIELDS))
        )
    out = table(
        ["Start", "Train loss", "x best", "Evaluations", "Near bounds"] + [v[0].split(",")[0] for v in FIELDS.values()],
        rows,
    )
    spread = identifiability(starts, tolerance=NEAR_BEST)
    near = [s["material"] for s in starts if s["loss"] <= (1 + NEAR_BEST) * best]
    mu = [Material(**m).equilibrium_shear_modulus_pa / 1e3 for m in near]
    rows = [
        [html.escape(name), unit, f"{lo * scale:.4g}", f"{hi * scale:.4g}"]
        for key, (name, unit, scale) in FIELDS.items()
        for lo, hi in [spread["range"][key]]
    ]
    rows.append(["mu_eq (derived)", "kPa", f"{min(mu):.4g}", f"{max(mu):.4g}"])
    out += f"<p>Parameter range over the {spread['starts']} starts within {100 * NEAR_BEST:.0f}% of the best loss:</p>"
    out += table(["Parameter", "Unit", "Min", "Max"], rows)
    return out


def error_grid(fit: dict, flags: dict, split: str = "validate") -> str:
    rows, classes = [], []
    for t in STANCES:
        row, cls = [f"{t} ms"], [""]
        for f in FORCES:
            v = fitted(fit, label(t, f), split)
            row.append(f"{pct(v['peak_force_error'])} / {pct(v['force_rmse_relative'])} / {pct(v['hysteresis_error'])}")
            cls.append("flag" if (label(t, f), split) in flags else "" if v["passed"] else "bad")
        rows.append(row)
        classes.append(cls)
    return table(["Stance"] + [f"{f} N" for f in FORCES], rows, classes)


def repeat_table(fit: dict, conditions: list[dict]) -> str:
    by = {c["target_n"]: c for c in conditions if c["role"] == "repeat"}
    rows = []
    for f in FORCES:
        v = fitted(fit, label(200, f, True), "repeat")
        o = fitted(fit, label(200, f), "validate")
        rows.append(
            [
                f"{f} N",
                f"{100 * by[f]['repeat_force_rmse_n'] / by[f]['measured_peak_n']:.2f}",
                f"{100 * by[f]['repeat_loop_energy_difference_j'] / by[f]['original_loop_energy_j']:+.1f}",
                f"{pct(v['peak_force_error'])} / {pct(v['force_rmse_relative'])} / {pct(v['hysteresis_error'])}",
                f"{pct(o['peak_force_error'])} / {pct(o['force_rmse_relative'])} / {pct(o['hysteresis_error'])}",
            ]
        )
    return table(
        [
            "Force",
            "Measured repeat vs 200 ms: RMSE % of peak",
            "Measured loop-energy change %",
            "Model on repeat: peak / RMSE / loop %",
            "Model on 200 ms validate",
        ],
        rows,
    )


def overlay_figure(slug: str, r: dict) -> str:
    import matplotlib.pyplot as plt

    index = {key: k for k, key in enumerate(r["keys"])}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(FORCES)))
    for f, color in zip(FORCES, colors, strict=True):
        k = index[(label(200, f), "validate")]
        trace, model = r["traces"][k], r["force"][k]
        axes[0, 0].plot(trace["time_s"] * 1e3, trace["force_n"], color=color, label=f"{f} N")
        axes[0, 0].plot(trace["time_s"] * 1e3, model, color=color, ls="--", lw=1.6)
        axes[0, 1].plot(trace["displacement_m"] * 1e3, trace["force_n"], color=color)
        axes[0, 1].plot(trace["displacement_m"] * 1e3, model, color=color, ls="--", lw=1.6)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(STANCES)))
    for t, color in zip(STANCES, colors, strict=True):
        k = index[(label(t, 2800), "validate")]
        trace, model = r["traces"][k], r["force"][k]
        axes[1, 0].plot(trace["time_s"] * 1e3, trace["force_n"], color=color, label=f"{t} ms")
        axes[1, 0].plot(trace["time_s"] * 1e3, model, color=color, ls="--", lw=1.6)
        axes[1, 1].plot(trace["displacement_m"] * 1e3, trace["force_n"], color=color)
        axes[1, 1].plot(trace["displacement_m"] * 1e3, model, color=color, ls="--", lw=1.6)
    loaded = [
        r["traces"][k]["time_s"][r["traces"][k]["force_n"] > 0.05 * r["traces"][k]["force_n"].max() + 30.0] * 1e3
        for k in range(len(r["traces"]))
    ]
    window = (min(t.min() for t in loaded) - 20.0, max(t.max() for t in loaded) + 20.0)
    for row, title in ((0, "200 ms stance, all forces"), (1, "2800 N, all stance times")):
        decorate(axes[row, 0], "time in cycle [ms]", "force [N]")
        decorate(axes[row, 1], "crosshead displacement [mm]", "force [N]")
        axes[row, 0].set_title(f"{title}: force vs time", loc="left")
        axes[row, 1].set_title(f"{title}: force vs displacement", loc="left")
        axes[row, 0].legend(ncol=2)
        axes[row, 0].set_xlim(*window)
    return figure(fig, f"{SHOES[slug]}: measured (solid) and model (dashed) force on the validate traces.")


def column_map_figure(slug: str, r: dict) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    bed, columns = r["bed"], r["columns"]
    xy = bed.uv_m * 1e3
    spacing = bed.spacing_m * 1e3
    index = {key: k for k, key in enumerate(r["keys"])}
    fig, axes = plt.subplots(2, len(MAP_FORCES), figsize=(13, 5.0), layout="constrained")
    for j, f in enumerate(MAP_FORCES):
        k = index[(label(200, f), "validate")]
        strain = columns["compression"][k] / bed.rest_m
        image = cells(axes[0, j], xy, spacing, strain, cmap="magma_r", edgecolor="none")
        image.set_clim(0.0, 0.8)
        cells(
            axes[1, j],
            xy,
            spacing,
            states(columns, k, bed),
            cmap=ListedColormap([c for _, c in STATES]),
            edgecolor="none",
        ).set_clim(-0.5, 5.5)
        share = outside_share(r, k)
        axes[0, j].set_title(f"{f} N: max strain {strain.max():.2f}", loc="left")
        axes[1, j].set_title(f"{f} N: {100 * share:.0f}% of the load outside the last", loc="left")
        for ax in axes[:, j]:
            ax.set_aspect("equal")
            ax.autoscale_view()
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
    fig.colorbar(image, ax=axes[0, :], label="compression c / L at peak", fraction=0.02)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#999", label=n) for n, c in STATES], loc="lower center", ncol=6)
    return figure(
        fig,
        f"{SHOES[slug]}: top view of the column bed at peak stroke, 200 ms stance, validate traces (heel on the left). "
        "Top row: compression strain. Bottom row: column state. 'Outside' columns are not under the last tread.",
    )


def outside_share(r: dict, k: int) -> float:
    """Return the share of the platen reaction under columns outside the last tread at peak stroke."""
    reaction = r["columns"]["reaction"][k]
    return float(reaction[~r["bed"].tread].sum() / reaction.sum())


def outside_share_figure(shoes: dict) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(shoes), figsize=(11, 3.6), sharey=True)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(STANCES)))
    for ax, (slug, s) in zip(axes, shoes.items(), strict=True):
        for t, color in zip(STANCES, colors, strict=True):
            r = s["replay"]
            share = [outside_share(r, r["keys"].index((label(t, f), "validate"))) for f in FORCES]
            ax.plot(FORCES, 100 * np.array(share), "o-", color=color, label=f"{t} ms")
        decorate(ax, "target force [N]", "platen load outside the last [%]")
        ax.set_title(SHOES[slug], loc="left")
    axes[0].legend()
    return figure(
        fig,
        "Share of the platen reaction carried by columns that are not under the last tread, at peak stroke. "
        "These columns are loaded only through shear from their neighbours.",
    )


def conservation_block(shoes: dict) -> str:
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(2, len(shoes), figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": (2, 1)})
    for ax, low, (slug, s) in zip(axes[0], axes[1], shoes.items(), strict=True):
        r = s["replay"]
        peak = np.max(np.abs(r["force"]), axis=1)
        imbalance = np.max(np.abs(r["force"] - r["reaction"]), axis=1) / peak
        worst = int(np.argmax(imbalance))
        rows.append(
            [
                SHOES[slug],
                f"{100 * np.max(imbalance):.3f}",
                html.escape(" ".join(r["keys"][worst])),
                f"{100 * np.median(imbalance):.4f}",
            ]
        )
        k = r["keys"].index((label(200, 2800), "validate"))
        time_ms = r["traces"][k]["time_s"] * 1e3
        ax.plot(time_ms, r["force"][k], color=BLUE, label="sum T (last)")
        ax.plot(time_ms, r["reaction"][k], color=ORANGE, ls="--", label="sum R (platen)")
        low.plot(time_ms, r["force"][k] - r["reaction"][k], color=GREEN, lw=1.0)
        decorate(ax, "", "force [N]")
        decorate(low, "time in cycle [ms]", "sum T - sum R [N]")
        ax.set_title(f"{SHOES[slug]}, 200 ms, 2800 N", loc="left")
        ax.legend(loc="upper right")
    out = figure(fig, "Load balance over one replayed cycle: the last's load sum T and the platen reaction sum R.")
    out += table(["Shoe", "Max |sum T - sum R| / peak, all traces [%]", "Worst trace", "Median over traces [%]"], rows)
    return out


WALL_ONSET = 0.1
"""The densification wall starts where the second term reaches this fraction of the first."""


def term_pressures(strain: np.ndarray, material: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the equilibrium pressure of the two Hyperfoam terms [Pa] at the model's zero Poisson ratio."""
    m = Material(**material)
    stretch = 1.0 - strain
    one = np.ones_like(stretch)
    return (
        ogden_hill_term_numpy(
            stretch, one, m.equilibrium_fraction * m.instantaneous_shear_modulus_pa, m.hyperfoam_exponent, 0.0
        ),
        ogden_hill_term_numpy(
            stretch, one, m.equilibrium_fraction * m.instantaneous_shear_modulus_2_pa, m.hyperfoam_exponent_2, 0.0
        ),
    )


def wall_onset(material: dict) -> float:
    """Return the strain where the second term first reaches :data:`WALL_ONSET` of the first; NaN if it never does."""
    strain = np.linspace(1.0e-3, 0.95, 1900)
    first, second = term_pressures(strain, material)
    above = np.nonzero(second >= WALL_ONSET * first)[0]
    return float(strain[above[0]]) if len(above) else float("nan")


def max_strain(r: dict) -> float:
    """Return the largest column compression strain of the fresh replay at peak stroke."""
    return float(np.max(r["columns"]["compression"] / r["bed"].rest_m))


def wall_onsets(fit: dict) -> list[float]:
    """Return the wall onset of every near-best start with a densifying (negative alpha2) second term."""
    starts = fit["optimization"]["starts"]
    best = min(s["loss"] for s in starts)
    return [
        wall_onset(s["material"])
        for s in starts
        if s["loss"] <= (1 + NEAR_BEST) * best and s["material"]["hyperfoam_exponent_2"] < 0.0
    ]


def second_term_block(shoes: dict) -> str:
    """Explain the densification wall of the second Hyperfoam term with the fitted numbers."""
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(1, len(shoes), figsize=(11, 3.8), sharey=True)
    for ax, (slug, s) in zip(axes, shoes.items(), strict=True):
        material = s["fit"]["material"]
        onset, reached = wall_onset(material), max_strain(s["replay"])
        grid = np.arange(np.floor(onset / 0.05) * 0.05, reached + 1e-9, 0.05)
        first, second = term_pressures(grid, material)
        near = wall_onsets(s["fit"])
        bound20 = s["bound20"]
        rows.append(
            [
                SHOES[slug],
                f"{material['instantaneous_shear_modulus_2_pa']:.3g}",
                f"{material['hyperfoam_exponent_2']:.1f}"
                + (" (at bound)" if "hyperfoam_exponent_2" in near_bounds(material) else ""),
                ", ".join(f"{q:.2f} at {e:.2f}" for e, q in zip(grid, second / first, strict=True)),
                f"{onset:.2f}",
                f"{reached:.2f}",
                f"{wall_onset(bound20['material']):.2f}" if bound20 else "-",
                f"{min(near):.2f}-{max(near):.2f} (n = {len(near)})",
            ]
        )
        strain = np.linspace(1.0e-3, min(0.95, reached + 0.12), 300)
        first, second = term_pressures(strain, material)
        ax.plot(strain, first / 1e6, color=BLUE, label="term 1")
        ax.plot(strain, np.maximum(second, 1e-12) / 1e6, color=ORANGE, label="term 2 (wall)")
        ax.plot(strain, (first + second) / 1e6, color=INK, ls="--", label="sum")
        ax.axvline(onset, color=GREEN, ls=":", label=f"onset (term 2 = {100 * WALL_ONSET:.0f}% of term 1) {onset:.2f}")
        ax.axvline(reached, color="#c0392b", ls="-.", label=f"max column strain in sweep {reached:.2f}")
        decorate(ax, "compression strain c / L", "equilibrium pressure [MPa]")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e2)
        ax.set_title(SHOES[slug], loc="left")
        ax.legend(loc="upper left", fontsize=8.5)
    table_html = table(
        [
            "Shoe",
            "G2 [Pa]",
            "alpha2",
            "term 2 / term 1 at strain",
            "Wall onset strain",
            "Max column strain",
            "Onset, bound-20 fit",
            "Onset, near-best wall starts",
        ],
        rows,
    )
    return (
        """<h4>Second Hyperfoam term: a densification wall</h4>
<p>The second Hyperfoam term is <b>not</b> switched off. With alpha2 strongly negative,
term2 = 2 G2 / (|alpha2| lam) (lam^-|alpha2| - 1) explodes as the stretch lam = 1 - c/L goes to 0, so a tiny G2 still
gives a densification wall. The table gives term2 / term1 (the ratio is the same for the instantaneous and the
equilibrium network). The wall is active when the most compressed column passes the onset strain.</p>"""
        + table_html
        + """<p>The onset strain (term 2 = 10% of term 1) is the same in the bound-20 and bound-40 fits and across the
near-best starts with a wall. So the wall <b>onset</b> is identified by this sweep; its <b>steepness</b> (the split
between G2 and alpha2) is not. Dev3Elite pushes alpha2 to its -20 bound: it wants an even steeper wall, close to a
hard stop near 0.5 strain.</p>"""
        + figure(
            fig,
            "Equilibrium pressure of the two Hyperfoam terms and their sum for the best fits (log scale), with the wall "
            "onset and the largest column strain the sweep reaches.",
        )
    )


def bound20_block(shoes: dict) -> str:
    """Compare the alpha1 <= 20 fit (backup) with the current alpha1 <= 40 fit."""
    if not any(s["bound20"] for s in shoes.values()):
        return ""
    out = [
        f"""<h4>Bound 20 vs bound 40</h4>
<p>The first fit capped the term-1 exponent alpha1 at 20, and FR3 ended on that cap. The current fit allows
alpha1 up to {FIT_UPPER_BOUNDS[1]:g}. The bound-20 results are from a backup and predate the kernel step-limiter fix;
their Dev3Elite 280 ms / 2800 N validate score was a solver blow-up. Validate numbers: median / max over 24 traces.</p>"""
    ]
    for slug, s in shoes.items():
        if not s["bound20"]:
            continue
        runs = {"alpha1 &lt;= 20 (backup)": s["bound20"], f"alpha1 &lt;= {FIT_UPPER_BOUNDS[1]:g} (current)": s["fit"]}
        rows = []
        for name, fit in runs.items():
            m = fit["material"]
            vals = [fit["conditions"][label(t, f)]["validate"]["fitted"] for t in STANCES for f in FORCES]
            peak = [v["peak_force_error"] for v in vals]
            rmse = [v["force_rmse_relative"] for v in vals]
            loop = [v["hysteresis_error"] for v in vals]
            rows.append(
                [
                    name,
                    f"{m['hyperfoam_exponent']:.2f}",
                    f"{m['instantaneous_shear_modulus_pa'] / 1e3:.0f}",
                    f"{m['instantaneous_shear_modulus_2_pa']:.3g} / {m['hyperfoam_exponent_2']:.1f}",
                    f"{Material(**m).equilibrium_shear_modulus_pa / 1e3:.0f}",
                    f"{m['equilibrium_fraction']:.3f} / {m['maxwell_relaxation_time_s'] * 1e3:.1f}",
                    f"{wall_onset(m):.2f}",
                    f"{min(st['loss'] for st in fit['optimization']['starts']):.5f}",
                    f"{pct(np.median(peak))} / {pct(max(peak))}",
                    f"{pct(np.median(rmse))} / {pct(max(rmse))}",
                    f"{pct(np.median(loop))} / {pct(max(loop))}",
                    f"{sum(v['passed'] for v in vals)} / 24",
                ]
            )
        out.append(f"<p><b>{html.escape(NAMES[SHOES[slug]])}</b></p>")
        out.append(
            table(
                [
                    "Fit",
                    "alpha1",
                    "G1 [kPa]",
                    "G2 [Pa] / alpha2",
                    "mu_eq [kPa]",
                    "f_eq / tau [ms]",
                    "Wall onset",
                    "Best train loss",
                    "Peak %",
                    "RMSE %",
                    "Loop %",
                    "Within all gates",
                ],
                rows,
            )
        )
    return "\n".join(out)


def results_section(shoes: dict) -> str:
    out = [
        """<h2 id="results">5. Results per shoe</h2>
<p>Train loss is the mean squared residual of <code>core.force_residual</code> (force over the cycle, loop area x5,
peak x6, all relative). Each start is a bounded least-squares solve with up to 80 residual evaluations; all starts run
together on the GPU. Cells marked red miss at least one 10% gate (peak, RMSE, loading or unloading RMSE,
loop area).</p>
<p>G1, G2 and alpha2 trade off between near-best starts, but the derived mu_eq, f_eq and tau stay tight. The second
Hyperfoam term acts as a densification wall whose onset is identified and whose steepness is not; see
<a href="#wall">the second-term section</a> below.</p>"""
    ]
    for slug, s in shoes.items():
        fit, r = s["fit"], s["replay"]
        o = fit["optimization"]
        out.append(f"<h3>{html.escape(NAMES[SHOES[slug]])}</h3>")
        out.append("<h4>Fitted material</h4>" + material_rows(fit))
        out.append(
            f"<h4>Multi-start</h4><p>{len(o['starts'])} starts, best = start {o['best_start']}, "
            f"fit time {o['seconds'] / 60:.1f} min.</p>" + multistart_tables(fit)
        )
        out.append("<h4>Validate traces: peak / RMSE / loop-area error [%]</h4>" + error_grid(fit, s["flagged"]))
        if s["flagged"]:
            out.append('<p class="note">Yellow: flagged trace (stored score not reproduced, see the summary).</p>')
        out.append(
            "<h4>Repeat block (320 ms label = second 200 ms block)</h4>" + repeat_table(fit, r["index"]["conditions"])
        )
        out.append("<h4>Measured vs model</h4>" + overlay_figure(slug, r))
        out.append("<h4>Column state at peak</h4>" + column_map_figure(slug, r))
        out.append(
            f'<p class="note">The fresh replay for these figures reproduces the stored peak error of '
            f"{len(r['keys']) - len(s['flagged'])} of {len(r['keys'])} traces within one percentage point.</p>"
        )
    out.append('<h3 id="wall">Both shoes</h3>' + second_term_block(shoes))
    out.append(bound20_block(shoes))
    out.append("<h4>Load outside the last</h4>" + outside_share_figure(shoes))
    out.append("<h4>Conservation: sum T vs sum R</h4>" + conservation_block(shoes))
    return "\n".join(out)


def load_old(backups: Path, slug: str) -> dict[str, dict]:
    """Return the old-law and stopgap fits that exist on disk, by label."""
    paths = {
        "Stopgap (grounded columns)": SWEEP / "fit" / f"{slug}_grounded_columns" / "material_fit.json",
        "Old law A (2 evaluations)": backups / f"{slug}_A2.json",
        "Old law A (30 evaluations)": backups / f"{slug}_30" / "material_fit.json",
    }
    return {name: json.loads(path.read_text()) for name, path in paths.items() if path.exists()}


def old_law_section(shoes: dict, old: dict) -> str:
    if not old:
        return ""
    out = [
        """<h2 id="old">6. Old vs new law</h2>
<p>Same sweep, same split, same objective. Old law A is the single-face bed where floating columns carry nothing;
the stopgap treats every column as standing on the platen. New law C is this report's fit. Numbers are over the 24
validate traces: median and worst. A worst value in the thousands of percent is a solver blow-up of that run.</p>"""
    ]
    for slug, s in shoes.items():
        runs = {**old.get(slug, {}), "New law C (this fit)": s["fit"]}
        rows = []
        for name, fit in runs.items():
            vals = [fit["conditions"][label(t, f)]["validate"]["fitted"] for t in STANCES for f in FORCES]
            row = [html.escape(name)]
            for metric in ("peak_force_error", "force_rmse_relative", "hysteresis_error"):
                v = [x[metric] for x in vals]
                row.append(f"{pct(np.median(v))} / {pct(max(v))}")
            row.append(f"{sum(x['passed'] for x in vals)} / {len(vals)}")
            rows.append(row)
        out.append(f"<h3>{html.escape(NAMES[SHOES[slug]])}</h3>")
        out.append(
            table(["Fit", "Peak error % (median / max)", "RMSE %", "Loop-area error %", "Within all gates"], rows)
        )
        per = []
        for f in FORCES:
            row = [f"200 ms, {f} N"]
            for fit in runs.values():
                v = fit["conditions"][label(200, f)]["validate"]["fitted"]
                row.append(f"{pct(v['peak_force_error'])} / {pct(v['force_rmse_relative'])}")
            per.append(row)
        out.append(
            "<details><summary>Per force at 200 ms (peak / RMSE %)</summary>"
            + table(["Condition"] + [html.escape(n) for n in runs], per)
            + "</details>"
        )
    return "\n".join(out)


def flatten(value, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [item for key, v in value.items() for item in flatten(v, f"{prefix}{key}.")]
    if isinstance(value, float):
        return [(prefix[:-1], f"{value:.4g}")]
    return [(prefix[:-1], html.escape(str(value)))]


def before_after_rows(value, name: str = "") -> list[list[str]]:
    """Collect the ``before``/``after`` and ``before_x``/``after_x`` pairs of the kernel profile at any depth."""
    if not isinstance(value, dict):
        return []

    def row(label: str, before, after) -> list[str]:
        numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before, after))
        speedup = f"{before / after:.2f}x" if numbers and after else ""
        text = [
            f"{v:.4g}" if isinstance(v, float) else "-" if v is None else html.escape(str(v)) for v in (before, after)
        ]
        return [html.escape(label), *text, speedup]

    if "before" in value and "after" in value:
        before, after = value["before"], value["after"]
        if isinstance(after, dict):
            return [row(f"{name}: {key}", before.get(key), v) for key, v in after.items()]
        return [row(name, before, after)]
    rows = [
        row(f"{name} ({key[len('after_') :]})", value[f"before_{key[len('after_') :]}"], value[key])
        for key in value
        if key.startswith("after_") and f"before_{key[len('after_') :]}" in value
    ]
    for key, v in value.items():
        rows += before_after_rows(v, f"{name} / {key}" if name else key)
    return rows


def kernel_section(path: Path) -> str:
    head = '<h2 id="kernel">7. Kernel performance</h2>'
    if not path.exists():
        return (
            head + f"<p>Pending: the profiling run has not written <code>{html.escape(path.as_posix())}</code> yet.</p>"
        )
    profile = json.loads(path.read_text())
    setups = [(key, v) for key, v in flatten(profile) if key == "gpu" or key.endswith("setup")]
    other = [
        [html.escape(key), v]
        for key, v in flatten(profile)
        if not any(part.startswith(("before", "after")) for part in key.split("."))
        and key != "gpu"
        and not key.endswith("setup")
        and not key.startswith("notes")
    ]
    return (
        head + "<p>The column-bed kernel was restructured for speed with the same arithmetic per column and "
        "compiled with <code>fast_math</code>. "
        f"Numbers from <code>{html.escape(path.as_posix())}</code>.</p><ul>"
        + "".join(f"<li><b>{html.escape(key)}</b>: {v}</li>" for key, v in setups)
        + "</ul><h4>Before and after</h4>"
        + table(["Quantity", "Before", "After", "Before / after"], before_after_rows(profile))
        + "<h4>Other measurements</h4>"
        + table(["Quantity", "Value"], other)
        + "<h4>Where the time goes and what was tried</h4><ul>"
        + "".join(f"<li>{html.escape(note)}</li>" for note in profile.get("notes", []))
        + "</ul><p><code>fast_math</code> is on: against a converged solve it changes forces by about 3e-6 of the "
        "peak, far below the 16-sweep truncation error. A convergence-based early exit from the sweeps was studied and "
        "not adopted (<code>result_changing_study</code>).</p>"
    )


def open_issues_section(shoes: dict) -> str:
    loops, shares = [], []
    for slug, s in shoes.items():
        loop = [fitted(s["fit"], label(t, 800), "validate")["hysteresis_error"] for t in STANCES]
        r = s["replay"]
        share = [outside_share(r, k) for k in range(len(r["keys"]))]
        loops.append(f"{SHOES[slug]} {pct(min(loop))}-{pct(max(loop))}%")
        shares.append(f"{SHOES[slug]} {pct(min(share))}-{pct(max(share))}%")

    count = sum(len(s["flagged"]) for s in shoes.values())
    flag_issue = (
        f"<li><b>Unreproduced blow-up in the stored scores.</b> {count} stored trace score(s) are solver blow-ups "
        "(column strain above 1) that a fresh replay does not reproduce, and the T = R balance check did not catch "
        "them because both blew up together. A check on column strain (c &lt; L) would catch this case.</li>"
        if count
        else ""
    )
    bounds = []
    for slug, s in shoes.items():
        m = s["fit"]["material"]
        at = [FIELDS[key][0].split(",")[0] for key in near_bounds(m) if key != "instantaneous_shear_modulus_2_pa"]
        bounds.append(
            f"{SHOES[slug]}: alpha1 = {m['hyperfoam_exponent']:.2f} (upper bound {FIT_UPPER_BOUNDS[1]:g}); "
            + (f"at a bound: {', '.join(at)}" if at else "no other parameter at a bound")
        )
    return f"""<h2 id="issues">8. Open issues</h2>
<ul>
<li><b>Hysteresis at low force.</b> The loop area is the largest error, worst at 800 N ({", ".join(loops)}). One Maxwell
branch with one relaxation time does not shape the small loops well.</li>
<li><b>Load outside the last.</b> Columns that the last never touches still carry {", ".join(shares)} of the platen load
at peak through shear. This share is not measured; a pressure-mat test would check it.</li>
<li><b>Parameters at a bound.</b> {"; ".join(bounds)}. Dev3Elite drives alpha2 to its lower bound: it wants a
steeper densification wall, close to a hard stop near 0.5 strain. G2 is tiny on purpose (it sets the wall steepness
together with alpha2) and is not listed.</li>
{flag_issue}<li><b>Shoe and last side.</b> The left last is assumed from the STL file name; the side is not confirmed.</li>
<li><b>Runner migration.</b> The GPU runner already steps the same <code>ColumnBed</code> kernel with the fitted shoes,
but its gait results are not yet validated against measured references.</li>
</ul>"""


def reproduce_section() -> str:
    starts = " ".join(f"--start {BACKUPS.as_posix()}/{slug}_C2nx/material_fit.json" for slug in SHOES)
    lines = [
        "git lfs pull",
        "uv run --extra dev python -m projects.digital_instron_v2.sweep_split",
        *(
            f"uv run --extra dev -m projects.digital_instron_v2 fit --shoe DigitalInstron/{slug} "
            f"--sweep {SWEEP.as_posix()}/processed/{name} --out {SWEEP.as_posix()}/fit/{slug} {starts}"
            for slug, name in SHOES.items()
        ),
        "uv run --extra dev python -m projects.digital_instron_v2.sweep_report --out DigitalInstron/FoamSweep/fit/report.html",
        "uv run --extra dev -m projects.digital_instron_v2.example --shoe DigitalInstron/puma-fast-r-nitro-elite-3 --shoe DigitalInstron/puma-deviate-nitro-elite-3 --slow-motion 6.7",
    ]
    return f"""<h2 id="reproduce">9. How to reproduce</h2>
<p>PowerShell, from the repository root. Split the raw sweeps, fit each shoe, build this report, open the
interactive viewer. Each fit starts from the calibrated material, the two earlier fits in
<code>{BACKUPS.as_posix()}</code>, and six Latin-hypercube starts (seed 0). See
<code>{SWEEP.as_posix()}/README.md</code> for the data files.</p>
<pre>{html.escape(chr(10).join(lines))}</pre>"""


def build(out: Path, backups: Path = BACKUPS) -> Path:
    """Replay both shoes, render every section, and write the report."""
    start = time.perf_counter()
    shoes = {}
    for slug in SHOES:
        fit = json.loads((SWEEP / "fit" / slug / "material_fit.json").read_text())
        r = replay(slug, fit)
        bound20 = backups / f"{slug}_MS_bound20.json"
        shoes[slug] = {
            "fit": fit,
            "replay": r,
            "flagged": flagged(fit, r),
            "bound20": json.loads(bound20.read_text()) if bound20.exists() else None,
        }
    old = {slug: runs for slug in SHOES if (runs := load_old(backups, slug))}
    style()
    sections = [
        summary_section(shoes, old),
        data_section(shoes),
        registration_section(shoes),
        contact_law_section(),
        results_section(shoes),
        old_law_section(shoes, old),
        kernel_section(SWEEP / "fit" / "kernel_profile.json"),
        open_issues_section(shoes),
        reproduce_section(),
    ]
    nav = "".join(
        f'<a href="#{a}">{t}</a>'
        for a, t in (
            ("summary", "Summary"),
            ("data", "Data"),
            ("shoe", "Shoe and last"),
            ("law", "Contact law"),
            ("results", "Results"),
            ("old", "Old vs new"),
            ("kernel", "Kernel"),
            ("issues", "Open issues"),
            ("reproduce", "Reproduce"),
        )
        if a != "old" or old
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Foam sweep fit: two-face column contact law</title>
<style>{CSS}</style></head><body><main>
<h1>Foam sweep fit with the two-face column contact law</h1>
<p class="note">Built {time.strftime("%Y-%m-%d %H:%M")} from files on disk by
<code>projects/digital_instron_v2/sweep_report.py</code>.</p>
<nav>{nav}</nav>
{"".join(sections)}
</main></body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.perf_counter() - start:.0f} s")
    return out


def main() -> None:
    """Run the report command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=SWEEP / "fit" / "report.html")
    parser.add_argument(
        "--backups", type=Path, default=BACKUPS, help="Folder with the earlier fits; missing ones are skipped."
    )
    args = parser.parse_args()
    build(args.out, args.backups)


if __name__ == "__main__":
    main()
