# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit one shoe's material to its Instron foam sweep.

Each sweep condition (stance time x force target) is one full-foot last trace
on the shoe's own column grid. The last pose comes from
:func:`registration.register_last`; the traces come from :mod:`sweep_split`.
Every trace is replayed through the runtime's two-surface column bed
(:mod:`last_replay`), one world per trace and material. Train traces drive a
multi-start least-squares fit whose starts run together on the GPU; validate
and repeat traces are only scored.
"""

import argparse
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from projects.digital_shoe.column_bed import BOTTOM_LANDED, TOP_HELD

from .core import CALIBRATED_MATERIAL, FIT_LOWER_BOUNDS, FIT_UPPER_BOUNDS, Material, force_residual, metrics
from .geometry import ColumnGrid
from .last_replay import LastBed, LastReplay, last_bed
from .registration import load_last, register_last, write_qc_figure

DEFAULT_LAST = Path("DigitalInstron/Instron Shoe Last Size 9 6drop merged attachment 1 left.STL")
TOUCHING_M = 5.0e-4
"""Rest bottom clearance up to which a column counts as standing on the platen [m]."""


def load_grid(path: Path) -> ColumnGrid:
    """Load a prepared ``column_grid.npz``."""
    with np.load(path) as data:
        return ColumnGrid(
            data["uv_m"],
            data["bottom_m"],
            data["top_m"],
            np.asarray(data["area_m2"], dtype=float),
            float(data["spacing_m"]),
            int(data["thickness_axis"]),
        )


def load_trace(path: Path) -> dict[str, np.ndarray]:
    """Load one processed trace: time [s], crosshead displacement [m], force [N]."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {"time_s": data["time_s"], "displacement_m": data["displacement_m"], "force_n": data["force_n"]}


class _Batcher:
    """Run the residual requests of several optimizer threads as one launch.

    Each start runs its own least-squares solve in a thread. A thread asking for
    residuals blocks until every live thread has asked or finished; the serving
    thread then replays all requested materials together and hands the rows back.
    """

    def __init__(self, evaluate, threads: int):
        self.evaluate = evaluate
        self.live = threads
        self.pending: list[dict] = []
        self.condition = threading.Condition()

    def request(self, points: list[np.ndarray]) -> list[np.ndarray]:
        with self.condition:
            slot = {"points": points, "rows": None}
            self.pending.append(slot)
            self.condition.notify_all()
            while slot["rows"] is None:
                self.condition.wait()
            return slot["rows"]

    def finish(self) -> None:
        with self.condition:
            self.live -= 1
            self.condition.notify_all()

    def serve(self) -> None:
        while True:
            with self.condition:
                while self.live and len(self.pending) < self.live:
                    self.condition.wait()
                if not self.live:
                    return
                batch, self.pending = self.pending, []
            rows = self.evaluate([point for slot in batch for point in slot["points"]])
            with self.condition:
                for slot in batch:
                    slot["rows"], rows = rows[: len(slot["points"])], rows[len(slot["points"]) :]
                self.condition.notify_all()


MAX_START_OFFSET_M = 0.005
"""Deepest start below the registered preload indentation the fit may choose [m]."""

_LOG_PARAMETERS = np.array([True, False, False, True, True, False, True, True, True])
"""Material fields searched in log space: the moduli, the relaxation time, and the slider yield."""


def _search_space() -> tuple[np.ndarray, np.ndarray]:
    """Return the fit bounds in search space; the zero lower bound of the second modulus becomes 1e2 Pa."""
    lower, upper = np.asarray(FIT_LOWER_BOUNDS, float), np.asarray(FIT_UPPER_BOUNDS, float)
    log = _LOG_PARAMETERS
    lower = np.where(log, np.log10(np.where(lower > 0.0, lower, 1.0e2)), lower)
    return lower, np.where(log, np.log10(upper), upper)


def near_bounds(material: dict[str, float], fraction: float = 0.01) -> list[str]:
    """Return the fields within ``fraction`` of the search-space width of a fit bound."""
    lower, upper = _search_space()
    material = asdict(Material(**material))
    x = np.array(list(material.values()), dtype=float)
    x = np.where(_LOG_PARAMETERS, np.log10(np.maximum(x, 1.0e-30)), x)
    margin = fraction * (upper - lower)
    return [name for name, near in zip(material, (x <= lower + margin) | (x >= upper - margin), strict=True) if near]


def latin_hypercube_starts(count: int, seed: int) -> list[Material]:
    """Return ``count`` Latin-hypercube materials inside the fit bounds, sampled in search space."""
    lower, upper = _search_space()
    log = _LOG_PARAMETERS
    rng = np.random.default_rng(seed)
    unit = (
        rng.permuted(np.tile(np.arange(count), (len(lower), 1)), axis=1).T + rng.random((count, len(lower)))
    ) / count
    points = lower + unit * (upper - lower)
    return [Material(*np.where(log, 10.0**p, p)) for p in points]


def fit_material(
    bed: LastBed,
    depth_m: np.ndarray,
    dt_s: float,
    traces: list[dict[str, np.ndarray]],
    starts: list[Material],
    evaluations: int,
    substeps: int,
    offset_bounds_m: tuple[float, float],
) -> tuple[Material, float, list[dict]]:
    """Fit the replayed traces from every start at once and keep the lowest loss.

    Each start is a bounded least-squares solve on :func:`core.force_residual`
    with up to ``evaluations`` residual evaluations. A Jacobian is a forward
    difference; the materials every start asks for in a round, Jacobian columns
    included, run as candidate worlds of one launch. Each trace frame is
    replayed in ``substeps`` contact steps. A start offset within
    ``offset_bounds_m`` deepens every stroke so the model's start force can meet
    the measured preload; it is fitted with the material.

    Returns:
        The best material and start offset [m] by training loss, and one record per start.
    """
    from scipy.optimize import least_squares

    fields = tuple(Material.__dataclass_fields__)
    lower = np.array([*FIT_LOWER_BOUNDS, offset_bounds_m[0]])
    upper = np.array([*FIT_UPPER_BOUNDS, offset_bounds_m[1]])
    width = len(lower) + 1
    replays: dict[int, LastReplay] = {}

    def material_of(point: np.ndarray) -> Material:
        """Return the material represented by one search point."""
        return Material(**dict(zip(fields, point[:-1], strict=True)))

    def evaluate(points: list[np.ndarray]) -> list[np.ndarray]:
        # Launch sizes are rounded up to whole Jacobians so few replays are built.
        size = width * -(-len(points) // width)
        if size not in replays:
            replays[size] = LastReplay(bed, depth_m, dt_s, candidates=size, substeps=substeps)
        pad = size - len(points)
        materials = [material_of(p) for p in points]
        offsets = [float(p[-1]) for p in points]
        force = replays[size].run(materials + [materials[0]] * pad, offsets_m=offsets + [offsets[0]] * pad)
        return [
            np.concatenate(
                [force_residual(t["force_n"], t["displacement_m"], f) for t, f in zip(traces, force[c], strict=True)]
            )
            for c in range(len(points))
        ]

    batcher = _Batcher(evaluate, len(starts))
    records: list[dict] = [{} for _ in starts]

    def solve(index: int) -> None:
        cache: dict[bytes, np.ndarray] = {}
        nonphysical = [0]

        def residuals(points: list[np.ndarray]) -> list[np.ndarray]:
            missing = [p for p in points if p.tobytes() not in cache]
            for point, row in zip(missing, batcher.request(missing) if missing else [], strict=True):
                cache[point.tobytes()] = row
                nonphysical[0] += int(not np.all(np.isfinite(row)))
            return [cache[p.tobytes()] for p in points]

        def jac(x: np.ndarray) -> np.ndarray:
            step = 1.0e-3 * np.maximum(np.abs(x), 1.0e-2)
            step = np.where(x + step > upper, -step, step)
            rows = residuals([x] + [x + step[j] * np.eye(len(x))[j] for j in range(len(x))])
            # A non-physical perturbed replay is differenced from the other side instead.
            flip = [j for j in range(len(x)) if not np.all(np.isfinite(rows[j + 1]))]
            if flip:
                step[flip] = -step[flip]
                for j, row in zip(flip, residuals([x + step[j] * np.eye(len(x))[j] for j in flip]), strict=True):
                    rows[j + 1] = row
            return np.column_stack([(rows[j + 1] - rows[0]) / step[j] for j in range(len(x))])

        try:
            start = starts[index]
            x0 = np.clip(np.array([*asdict(start).values(), 0.0], dtype=float), lower, upper)
            try:
                result = least_squares(
                    lambda x: residuals([x])[0],
                    x0,
                    jac=jac,
                    bounds=(lower, upper),
                    x_scale="jac",
                    max_nfev=evaluations,
                )
            except ValueError as error:
                # A start whose own replay is not physical has no fit.
                records[index] = {
                    "start": asdict(starts[index]),
                    "loss": float("inf"),
                    "message": str(error),
                    "nonphysical_evaluations": nonphysical[0],
                }
                return
            records[index] = {
                "start": asdict(start),
                "material": asdict(material_of(result.x)),
                "start_offset_m": float(result.x[-1]),
                "loss": float(np.mean(result.fun**2)),
                "residual_evaluations": int(result.nfev),
                "jacobians": int(result.njev),
                "status": int(result.status),
                "message": str(result.message),
                "near_bounds": near_bounds(asdict(material_of(result.x))),
                "nonphysical_evaluations": nonphysical[0],
            }
        finally:
            batcher.finish()

    threads = [threading.Thread(target=solve, args=(k,)) for k in range(len(starts))]
    for thread in threads:
        thread.start()
    batcher.serve()
    for thread in threads:
        thread.join()
    if not all(records):
        raise RuntimeError("A multi-start solve failed; see the raised thread exception above")
    best = min(records, key=lambda record: record["loss"])
    return Material(**best["material"]), best["start_offset_m"], records


def identifiability(records: list[dict], tolerance: float = 0.1) -> dict:
    """Return the parameter range over starts whose loss is within ``tolerance`` of the best."""
    best = min(record["loss"] for record in records)
    near = [record["material"] for record in records if record["loss"] <= (1.0 + tolerance) * best]
    return {
        "loss_tolerance": tolerance,
        "starts": len(near),
        "range": {name: [min(m[name] for m in near), max(m[name] for m in near)] for name in near[0]},
    }


def contact_diagnostics(columns: dict[str, np.ndarray], world: int, bed) -> dict:
    """Summarize one world's column state at its peak stroke.

    Tread columns are ``held`` when the last presses their top, ``open``
    otherwise; every column is ``landed`` when its bottom stands on the platen,
    ``floating`` otherwise. Columns standing on the platen at rest (bottom
    clearance up to :data:`TOUCHING_M`) are counted among the landed ones. The
    summed last reaction ``T`` and platen reaction ``R`` agree up to the
    unbalanced shear of columns free at both ends; the carrier force is the
    last's vertical wrench.
    """
    support = columns["support"][world]
    held = (support & TOP_HELD) != 0
    landed = (support & BOTTOM_LANDED) != 0
    tread = bed.tread
    reaction = columns["reaction"][world]
    return {
        "max_strain": float(np.max(columns["compression"][world] / bed.rest_m)),
        "held_landed": int(np.sum(held & landed)),
        "held_floating": int(np.sum(held & ~landed)),
        "open_landed": int(np.sum(tread & ~held & landed)),
        "open_floating": int(np.sum(tread & ~held & ~landed)),
        "outside_landed": int(np.sum(~tread & landed)),
        "outside_floating": int(np.sum(~tread & ~landed)),
        "landed_touching_at_rest": int(np.sum(landed & (bed.bottom_m <= TOUCHING_M))),
        "outside_force_n": float(reaction[~tread].sum()),
        "platen_reaction_n": float(reaction.sum()),
        "last_reaction_n": float(columns["carried"][world].sum()),
        "carrier_force_n": float(columns["carrier"][world]),
    }


def run(
    shoe: Path,
    sweep: Path,
    out: Path,
    *,
    side: str,
    last: Path,
    evaluations: int,
    random_starts: int,
    seed: int,
    substeps: int,
) -> dict:
    """Register the last, fit the material to the train traces from many starts, and score every trace.

    Every trace frame is replayed in ``substeps`` contact steps.

    The starts are :data:`CALIBRATED_MATERIAL` and ``random_starts``
    Latin-hypercube materials drawn with ``seed``.
    """
    grid = load_grid(shoe / side / "column_grid.npz")
    index = json.loads((sweep / "conditions.json").read_text())
    preload = float(np.median([c["preload_n"] for c in index["conditions"]]))
    initial = CALIBRATED_MATERIAL
    last_model = load_last(last)
    registration = register_last(
        grid, last_model, modulus_pa=3.0 * initial.equilibrium_shear_modulus_pa, preload_n=preload, expect_side=side
    )
    out.mkdir(parents=True, exist_ok=True)
    write_qc_figure(out / "registration.png", registration, last_model, f"{shoe.name} ({side})")
    bed = last_bed(grid, registration)

    def label(c: dict) -> str:
        return f"{c['stance_ms']}ms_{c['target_n']}N" + ("_repeat" if c["role"] == "repeat" else "")

    # The trace zero is the top of stroke at the preload, so the last starts at the
    # registered preload indentation and the measured preload force is kept.
    names, traces = [], []
    for c in index["conditions"]:
        for split in ("train", "validate") if c["role"] == "condition" else ("repeat",):
            names.append((label(c), split))
            traces.append(load_trace(sweep / c[f"{split}_path"]))
    indentation = registration.qc["preload"]["indentation_m"]
    depth = np.stack([np.maximum(t["displacement_m"], 0.0) + indentation for t in traces], axis=1)
    dt = float(np.mean(np.diff(traces[0]["time_s"])))
    train = [k for k, (_, split) in enumerate(names) if split == "train"]

    starts = [initial, *latin_hypercube_starts(random_starts, seed)]
    start = time.perf_counter()
    # The start may move from first contact (the whole indentation undone) to
    # MAX_START_OFFSET_M deeper than the registered preload indentation.
    material, offset, records = fit_material(
        bed,
        depth[:, train],
        dt,
        [traces[k] for k in train],
        starts,
        evaluations,
        substeps,
        (-indentation, MAX_START_OFFSET_M),
    )
    fit_seconds = time.perf_counter() - start

    peaks = np.argmax(depth, axis=0)
    force, columns = LastReplay(bed, depth, dt, candidates=2, substeps=substeps).run(
        [initial, material], offsets_m=[0.0, offset], snapshot_frames=peaks
    )
    broken = [
        f"{key} {name} {split}"
        for c, key in enumerate(("initial", "fitted"))
        for k, (name, split) in enumerate(names)
        if np.isnan(force[c, k]).any()
    ]
    if broken:
        raise RuntimeError(f"Non-physical replay, nothing written: {', '.join(broken)}")
    scores = {}
    for k, ((name, split), trace) in enumerate(zip(names, traces, strict=True)):
        scores.setdefault(name, {})[split] = {
            key: {
                **metrics(trace["force_n"], force[c, k], trace["displacement_m"]),
                "peak_force_n": float(np.max(force[c, k])),
                "measured_peak_force_n": float(np.max(trace["force_n"])),
                "contact": contact_diagnostics(columns, c * len(traces) + k, bed),
            }
            for c, key in enumerate(("initial", "fitted"))
        }
    report = {
        "shoe": shoe.name,
        "side": side,
        "sweep": index["shoe"],
        "evaluations_per_start": evaluations,
        "substeps": substeps,
        "registration": {"pose": registration.pose, "qc": registration.qc},
        "initial_material": asdict(initial),
        "material": asdict(material),
        "start_offset_m": offset,
        "optimization": {
            "seconds": fit_seconds,
            "best_start": int(np.argmin([record["loss"] for record in records])),
            "identifiability": identifiability(records),
            "starts": records,
        },
        "conditions": scores,
    }
    (out / "material_fit.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    return report


def main() -> None:
    """Run the sweep fit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoe", type=Path, required=True, help="Prepared shoe folder, e.g. DigitalInstron/<slug>.")
    parser.add_argument("--sweep", type=Path, required=True, help="Processed sweep folder holding conditions.json.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", default="left")
    parser.add_argument("--last", type=Path, default=DEFAULT_LAST)
    parser.add_argument("--evaluations", type=int, default=80, help="Residual evaluations per start.")
    parser.add_argument("--random-starts", type=int, default=6, help="Latin-hypercube starts inside the bounds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--substeps",
        type=int,
        default=8,
        help="Contact steps per 1 ms trace frame; 8 is the runner's 1.25e-4 s contact step.",
    )
    args = parser.parse_args()
    report = run(
        args.shoe,
        args.sweep,
        args.out,
        side=args.side,
        last=args.last,
        evaluations=args.evaluations,
        random_starts=args.random_starts,
        seed=args.seed,
        substeps=args.substeps,
    )
    print(json.dumps({k: report[k] for k in ("shoe", "material", "registration")}, indent=2, default=float))


if __name__ == "__main__":
    main()
