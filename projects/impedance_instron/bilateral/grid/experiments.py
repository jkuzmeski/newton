# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment definitions: each experiment is a flat list of runs, one array index per run.

A run names its dataset (a directory under the grid data root), its training
entry point, and its command-line arguments. Add new experiments here and
commit or snapshot them with the code, so the run list is pinned to the code
that executes it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Run:
    """One array index of an experiment."""

    name: str
    dataset: str
    args: tuple[str, ...]
    entry: str = "projects.impedance_instron.bilateral.ppo"
    smoke_args: tuple[str, ...] = field(default=("--iterations", "5", "--worlds", "1024", "--save-every", "5"))
    """Arguments appended for a smoke test; later flags override earlier ones."""


PLANAR = "dataset_6step"
SPATIAL = "dataset_3d"
_BASE = ("--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "0")
_RUN4 = ("--reward-kernels", "pooled", "--force-penalty", "0")
_BODY = ("--body-position-weight", "0.3", "--body-velocity-weight", "0.1")
_REGULARIZERS = ("--contact-match-penalty", "0.1", "--action-smoothness-penalty", "0.02", "--previous-action-obs")

EXPERIMENTS: dict[str, tuple[Run, ...]] = {
    # From-scratch ablation of the ProtoMotions-style terms on the planar model
    # (no symmetry loss, fitted 5.3 ms torque response). Index 3 is the run 5 recipe.
    "pm_ablation": (
        Run("baseline", PLANAR, _BASE + _RUN4),
        Run("body_terms", PLANAR, _BASE + _RUN4 + _BODY),
        Run("regularizers", PLANAR, _BASE + _RUN4 + _REGULARIZERS),
        Run("all_terms", PLANAR, _BASE + _RUN4 + _BODY + _REGULARIZERS),
    ),
    # Seed replicates of the full recipe, to size the run-to-run spread.
    "pm_seeds": tuple(
        Run(f"all_terms_seed{seed}", PLANAR, (*_BASE[:-1], str(seed), *_RUN4, *_BODY, *_REGULARIZERS))
        for seed in (1, 2, 3)
    ),
    # Three-dimensional articulation, engineering-seed start.
    "spatial_baseline": (
        Run(
            "spatial_baseline",
            SPATIAL,
            ("--kind", "3d", "--worlds", "1024", "--iterations", "1500", "--save-every", "100"),
            smoke_args=("--iterations", "3", "--worlds", "256", "--save-every", "3"),
        ),
    ),
}


def get(experiment: str) -> tuple[Run, ...]:
    """Return an experiment's run list or raise with the known names."""
    if experiment not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {experiment!r}; known: {', '.join(sorted(EXPERIMENTS))}")
    return EXPERIMENTS[experiment]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List grid experiments and their runs.")
    parser.add_argument("experiment", nargs="?")
    args = parser.parse_args(argv)
    if args.experiment is None:
        for name, runs in EXPERIMENTS.items():
            print(f"{name}: {len(runs)} runs ({', '.join(run.name for run in runs)})")
        return
    print(json.dumps([asdict(run) for run in get(args.experiment)], indent=2))


if __name__ == "__main__":
    main()
