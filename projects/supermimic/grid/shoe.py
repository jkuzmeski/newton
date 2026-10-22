# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Add a shoe to a SuperMimic column-bed dataset: fit its foam on the grid, export it, enter it in the manifest.

From the repository root, after ``shoe_prep`` has written
``DigitalInstron/<slug>/{left,right}/column_grid.npz``::

    uv run --no-sync python -m projects.supermimic.grid.shoe submit <slug> --sweep <tracking.csv> --label <Label>
    uv run --no-sync python -m projects.supermimic.grid.shoe add <run folder> --dataset <dataset folder> \
        --mass-kg 0.18 --mass-source "<scale or URL>"

``submit`` splits the Instron foam sweep into train and validate traces under
``DigitalInstron/FoamSweep/processed/<Label>`` and queues one fit per seed on
the grid; every fit starts from the earlier fits in ``DigitalInstron/FoamSweep/fit``
and from Latin-hypercube materials. ``add`` runs once the job has finished: it
exports the lowest-loss fit as a left artifact and its mirror (the export
replays the validate traces), writes them to ``assets/`` and the validation
report to ``reports/``, and enters the shoe in ``manifest.json``. The mount and
static pitch come from the shoe worn in the motion trials through the shared
Instron last. Without ``--mass-kg`` the artifacts carry no mass and loaders use
:data:`~projects.digital_shoe.artifact.DEFAULT_SHOE_MASS_KG`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.export_digital_shoe import export
from projects.digital_instron_v2.sweep_fit import DEFAULT_LAST
from projects.digital_instron_v2.sweep_split import process_sweep
from projects.digital_shoe.artifact import load_artifact

from .launch import LOCAL_ROOT, REMOTE_ROOT, submit

DIGITAL_INSTRON = Path("DigitalInstron")
SWEEPS = DIGITAL_INSTRON / "FoamSweep" / "processed"
FITS = DIGITAL_INSTRON / "FoamSweep" / "fit"
"""Earlier ``material_fit.json`` files; every new fit starts from each of them."""
FIT_ARGS = ("--substeps", "8", "--evaluations", "80", "--random-starts", "6")


def _digest(folder: Path) -> str:
    sha = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        sha.update(path.relative_to(folder).as_posix().encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def submit_fit(slug: str, tracking_csv: Path, label: str, commit: str, seeds: int, time_limit: str) -> dict:
    """Split the sweep, stage the fit inputs and queue one fit per seed; return the submission record."""
    process_sweep(tracking_csv, SWEEPS / label, shoe=label)
    starts = sorted(FITS.rglob("material_fit.json"))
    with tempfile.TemporaryDirectory() as directory:
        inputs = Path(directory)
        shutil.copytree(DIGITAL_INSTRON / slug, inputs / "shoe" / slug, ignore=shutil.ignore_patterns("*.glb"))
        shutil.copytree(SWEEPS / label, inputs / "sweep")
        shutil.copy(DEFAULT_LAST, inputs / "last.stl")
        for k, path in enumerate(starts):
            (inputs / "starts").mkdir(exist_ok=True)
            shutil.copy(path, inputs / "starts" / f"{k}_{path.parent.name}.json")
        name = f"shoe_{slug}_{_digest(inputs)[:8]}"
        root = f"{{data}}/{name}"
        start_args = [
            a for p in sorted((inputs / "starts").glob("*.json")) for a in ("--start", f"{root}/starts/{p.name}")
        ]
        tasks = [
            {
                "name": f"seed{seed}",
                "args": [
                    "projects.digital_instron_v2", "fit", "--shoe", f"{root}/shoe/{slug}", "--sweep", f"{root}/sweep",
                    "--out", "{out}", "--last", f"{root}/last.stl", "--seed", str(seed), *FIT_ARGS, *start_args,
                ],
            }
            for seed in range(seeds)
        ]  # fmt: skip
        return submit(f"shoe_fit_{slug}", commit, tasks, {name: inputs}, time_limit)


def register_to_measured(measured: Path, other: Path, mount_m, pitch_rad: float) -> tuple[list, float, dict]:
    """Carry the measured shoe's mount and static pitch to another shoe through the shared last.

    Both artifacts carry the same full-foot last, each in its own shoe frame.
    The rigid transform that maps the measured shoe's last onto the other's
    (least squares over corresponding vertices) moves the ankle with the foot;
    the static pitch changes by the rise of the shoe's long axis under it.

    Returns:
        The other shoe's mount [m], its static pitch [rad] and the registration record.
    """
    source = load_artifact(measured).visual_mesh("fullfoot_last").vertices_m
    target = load_artifact(other).visual_mesh("fullfoot_last").vertices_m
    if source.shape != target.shape:
        raise ValueError("both shoes must carry the same fullfoot last mesh")
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    translation = target_mean - rotation @ source_mean
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    delta = float(np.arcsin(rotation[2, 0]))
    record = {
        "method": "rigid least-squares fit of the shared fullfoot last",
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "pitch_change_rad": delta,
        "residual_rms_m": float(np.sqrt(np.mean(residual**2))),
        "residual_max_m": float(residual.max()),
    }
    return (rotation @ np.asarray(mount_m, dtype=float) + translation).tolist(), float(pitch_rad) + delta, record


def add_shoe(run: Path, dataset: Path, mass_kg: float | None, mass_source: str | None) -> dict:
    """Export the run's lowest-loss fit into ``dataset`` and enter the shoe in its manifest; return the entry."""
    fits = sorted(run.glob("*/material_fit.json"))
    if not fits:
        raise SystemExit(f"no material_fit.json under {run}; has the job finished?")
    loss = [min(s["loss"] for s in json.loads(p.read_text())["optimization"]["starts"]) for p in fits]
    best = fits[int(np.argmin(loss))]
    fit = json.loads(best.read_text())
    slug = fit["shoe"]
    reports = dataset / "reports" / slug
    left, right, report = export(
        DIGITAL_INSTRON / slug, best, SWEEPS / fit["sweep"], reports, mass_kg=mass_kg, mass_source=mass_source
    )
    assets = {side: f"assets/{slug}_{side}.json" for side in ("left", "right")}
    for side, path in (("left", left), ("right", right)):
        shutil.move(path, dataset / assets[side])
    shutil.copy(best, reports / "material_fit.json")

    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    measured = next(s for s in manifest["shoes"] if "registration" not in s)
    mount, pitch, registration = register_to_measured(
        dataset / measured["left"], dataset / assets["left"], measured["mount_m"], measured["pitch_rad"]
    )
    artifact = json.loads((dataset / assets["left"]).read_text())
    entry = {
        "name": slug,
        **assets,
        "fit_passed_all_declared_gates": bool(artifact["identification"]["passed_all_declared_gates"]),
        "mount_m": mount,
        "pitch_rad": pitch,
        "registration": {**registration, "measured_shoe": measured["name"]},
        "fit": {"run": str(run), "file": best.relative_to(run).as_posix(), "loss": min(loss)},
    }
    manifest["shoes"] = [s for s in manifest["shoes"] if s["name"] != slug] + [entry]
    for path in assets.values():
        manifest["shared_assets"][Path(path).name] = {
            "file": path,
            "sha256": hashlib.sha256((dataset / path).read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    curves = artifact["validation"]["curves"]
    rmse = [c["metrics"]["force_rmse_relative"] for c in curves]
    print(f"{slug}: {len(curves)} validate traces, force RMSE mean {np.mean(rmse):.2%} max {np.max(rmse):.2%}")
    print(f"report: {report}")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("submit", help="Split the foam sweep and queue the fits on the grid.")
    sub.add_argument("slug", help="Shoe folder under DigitalInstron from shoe_prep.")
    sub.add_argument("--sweep", type=Path, required=True, help="Instron *.steps.tracking.csv of the foam sweep.")
    sub.add_argument("--label", required=True, help="Sweep name, e.g. FR3; traces land in FoamSweep/processed/<label>.")
    sub.add_argument("--commit", default="HEAD", help="Commit the grid runs; the working tree is never shipped.")
    sub.add_argument("--seeds", type=int, default=2, help="Fits with different random starts, one GPU each.")
    sub.add_argument("--time", default="02:00:00")
    add = commands.add_parser("add", help="Export the finished fit and enter the shoe in a dataset manifest.")
    add.add_argument("run", type=Path, help=f"Run folder of the fit, {LOCAL_ROOT}/runs/shoe_fit_<slug>/<run>.")
    add.add_argument("--dataset", type=Path, required=True, help="Dataset folder holding manifest.json.")
    add.add_argument("--mass-kg", type=float, help="Mass of one shoe [kg]; omit when unknown.")
    add.add_argument("--mass-source", help="Where the shoe mass was measured or published.")
    args = parser.parse_args()
    if args.command == "submit":
        record = submit_fit(args.slug, args.sweep, args.label, args.commit, args.seeds, args.time)
        record["local_run_dir"] = record["run_dir"].replace(REMOTE_ROOT, LOCAL_ROOT.as_posix())
        print(json.dumps(record, indent=1))
    else:
        if (args.mass_kg is None) != (args.mass_source is None):
            parser.error("--mass-kg and --mass-source go together")
        print(json.dumps(add_shoe(args.run, args.dataset, args.mass_kg, args.mass_source), indent=1))


if __name__ == "__main__":
    main()
