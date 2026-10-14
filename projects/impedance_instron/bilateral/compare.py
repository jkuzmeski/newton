# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare bilateral models with the single-leg runner on stances outside bilateral training.

Every single-leg held-out stance lies inside a bilateral training segment, so
scoring there would leak. The common set is the single-leg stances that no
bilateral training segment overlaps in source time: held-out for the bilateral
models and in-sample for the single-leg runner, which favors the latter. The
single-leg loss of a bilateral model is that runner's loss restricted to the
right leg and foot (:func:`.rollout.score_world`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def clean_stances(bilateral_dataset: Path, stance_dataset: Path) -> list[str]:
    """Return single-leg stance IDs outside every bilateral training segment."""
    train = [
        (m["trial"], m["start_s"], m["end_s"])
        for m in json.loads((bilateral_dataset / "manifest.json").read_text(encoding="utf-8"))["members"]
        if m["split"] == "train"
    ]
    stances = json.loads((stance_dataset / "manifest.json").read_text(encoding="utf-8"))["members"]
    return sorted(
        m["id"]
        for m in stances
        if not any(trial == m["trial"] and m["start_s"] < end and m["end_s"] > start for trial, start, end in train)
    )


def _single_leg_terms(row: dict) -> dict:
    t = np.asarray(row["tracking_rmse"])
    return {
        "loss": row["loss"],
        "root": float(np.mean((t[:2] / 0.02) ** 2)),
        "angles": float(np.mean((t[2:] / 0.05) ** 2)),
        "grf": float(np.mean((np.asarray(row["grf_rmse_n"]) / 100) ** 2)),
        "peak": (row["peak_fz_error_n"] / 100) ** 2,
        "angle_rmse_deg": np.degrees(t[2:]).tolist(),
        "fz_rmse_n": row["grf_rmse_n"][1],
        "failed": row.get("status") != "completed",
    }


def _bilateral_terms(row: dict) -> dict:
    t = np.asarray(row["tracking_rmse"])
    peaks = list(row["peak_fz_error_n"])
    return {
        "loss": row["single_leg_loss"],
        "root": float(np.mean((t[:2] / 0.02) ** 2)),
        "angles": float(np.mean((t[2:6] / 0.05) ** 2)),
        "grf": float(np.mean((np.asarray(row["grf_rmse_n"][:2]) / 100) ** 2)),
        "peak": float(np.mean(np.square(peaks)) / 1e4) if peaks else 0.0,
        "angle_rmse_deg": np.degrees(t[2:6]).tolist(),
        "fz_rmse_n": row["grf_rmse_n"][1],
        "failed": row["status"] != "completed",
    }


def _rows(summary: Path, key: str) -> dict[str, dict]:
    splits = json.loads(summary.read_text(encoding="utf-8"))["splits"]
    return {row["id"]: row for split in splits.values() for row in split[key]}


def compare(bilateral_dataset: Path, stance_dataset: Path, single_leg: Path, models: dict[str, Path]) -> dict:
    """Return mean terms over the clean stances for the single-leg runner and each bilateral model."""
    ids = clean_stances(bilateral_dataset, stance_dataset)
    result = {"stances": ids}
    entries = {"single_leg": [_single_leg_terms(r) for i, r in _rows(single_leg, "trials").items() if i in ids]}
    for name, path in models.items():
        entries[name] = [_bilateral_terms(r) for i, r in _rows(path, "segments").items() if i in ids]
    for name, rows in entries.items():
        if len(rows) != len(ids):
            raise ValueError(f"{name} does not score every clean stance")
        result[name] = {
            key: (
                np.mean([r[key] for r in rows], axis=0).tolist() if key != "failed" else int(sum(r[key] for r in rows))
            )
            for key in rows[0]
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bilateral-dataset", type=Path, required=True)
    parser.add_argument("--stance-dataset", type=Path, required=True)
    parser.add_argument("--single-leg", type=Path, required=True, help="summary.json of the single-leg evaluate")
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "SUMMARY"), default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.bilateral_dataset, args.stance_dataset, args.single_leg, {n: Path(p) for n, p in args.model})
    print(f"{len(result['stances'])} clean stances")
    print(f"{'model':<16}{'loss':>8}{'root':>8}{'angles':>8}{'grf':>8}{'peak':>8}  angle RMSE [deg]  Fz RMSE")
    for name, terms in result.items():
        if name == "stances":
            continue
        angles = " ".join(f"{a:5.1f}" for a in terms["angle_rmse_deg"])
        print(
            f"{name:<16}{terms['loss']:8.2f}{terms['root']:8.2f}{terms['angles']:8.2f}{terms['grf']:8.2f}"
            f"{terms['peak']:8.2f}  {angles}  {terms['fz_rmse_n']:6.1f}  failed {terms['failed']}"
        )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
