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
PLANAR_STATIC = "dataset_6step_static"
SPATIAL_STATIC = "dataset_3d_static"
_BASE = ("--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "0")
_RUN4 = ("--reward-kernels", "pooled", "--force-penalty", "0")
_BODY = ("--body-position-weight", "0.3", "--body-velocity-weight", "0.1")
_REGULARIZERS = ("--contact-match-penalty", "0.1", "--action-smoothness-penalty", "0.02", "--previous-action-obs")
_FITTED_ACTUATION = ("--response-time-s", "0.005334445782959161", "--slew-limit")
"""Actuation of the runs before the no-lag, no-slew decision; the environment default is now bare."""
_SPATIAL = ("--kind", "3d", "--worlds", "1024", "--iterations", "1500", "--save-every", "100", "--seed", "0")
_SPATIAL_SMOKE = ("--iterations", "3", "--worlds", "256", "--save-every", "3")

EXPERIMENTS: dict[str, tuple[Run, ...]] = {
    # From-scratch ablation of the ProtoMotions-style terms on the planar model
    # (no symmetry loss, fitted 5.3 ms torque response). Index 3 is the run 5 recipe.
    "pm_ablation": (
        Run("baseline", PLANAR, _BASE + _RUN4 + _FITTED_ACTUATION),
        Run("body_terms", PLANAR, _BASE + _RUN4 + _BODY + _FITTED_ACTUATION),
        Run("regularizers", PLANAR, _BASE + _RUN4 + _REGULARIZERS + _FITTED_ACTUATION),
        Run("all_terms", PLANAR, _BASE + _RUN4 + _BODY + _REGULARIZERS + _FITTED_ACTUATION),
    ),
    # Seed replicates of the full recipe, to size the run-to-run spread.
    "pm_seeds": tuple(
        Run(
            f"all_terms_seed{seed}",
            PLANAR,
            (*_BASE[:-1], str(seed), *_RUN4, *_BODY, *_REGULARIZERS, *_FITTED_ACTUATION),
        )
        for seed in (1, 2, 3)
    ),
    # Torque lag and slew limit on the best recipe so far (pm_ablation index 2).
    # Neither is fitted to the data: the single-leg fit pushed the lag to its 5 ms floor.
    "actuation": (
        Run("lag5ms_slew", PLANAR, _BASE + _RUN4 + _REGULARIZERS + _FITTED_ACTUATION),
        Run("nolag_slew", PLANAR, (*_BASE, *_RUN4, *_REGULARIZERS, "--response-time-s", "0", "--slew-limit")),
        Run("nolag_noslew", PLANAR, (*_BASE, *_RUN4, *_REGULARIZERS, "--response-time-s", "0", "--no-slew-limit")),
        Run("lag25ms_slew", PLANAR, (*_BASE, *_RUN4, *_REGULARIZERS, "--response-time-s", "0.025", "--slew-limit")),
    ),
    # Static-trial shoe mount (z 89.5 mm instead of 109.4 mm), no torque lag or
    # slew. Index 0 repeats the actuation study's nolag_noslew run on the new
    # mount; indices 1 and 2 separate the mount's effect in 3D.
    "static_mount": (
        Run("planar_static", PLANAR_STATIC, _BASE + _RUN4 + _REGULARIZERS),
        Run("spatial_static", SPATIAL_STATIC, _SPATIAL, smoke_args=_SPATIAL_SMOKE),
        Run("spatial_old_mount", SPATIAL, _SPATIAL, smoke_args=_SPATIAL_SMOKE),
    ),
    # 3D runner for open-ended running: 10 s episodes that continue the reference
    # across segments and follow a pacer, with 4096 worlds. The 0.25 ms substep
    # doubles throughput (0.8% GRF impulse change); the 0.125 ms run is its control.
    # gamma 0.99 at 500 Hz looks 0.2 s ahead; the g998 run looks about 1 s ahead.
    "spatial_pacer": (
        Run(
            "pacer_dt250us",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "6000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", "8"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
        Run(
            "pacer_dt125us",
            SPATIAL_STATIC,
            ("--kind", "3d", "--worlds", "4096", "--iterations", "6000", "--save-every", "100", "--seed", "0"),
            smoke_args=_SPATIAL_SMOKE,
        ),
        Run(
            "pacer_dt250us_g998",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "3000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", "8", "--gamma", "0.998", "--horizon", "64"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
    ),
    # Lower policy rates over the same 0.25 ms substep. The impedance law still runs
    # every substep; gamma 0.99 then looks about 1 s (100 Hz) or 2 s (50 Hz) ahead.
    "spatial_rate": tuple(
        Run(
            f"pacer_{hz}hz",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "3000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", str(substeps)),
            ),
            smoke_args=_SPATIAL_SMOKE,
        )
        for hz, substeps in ((100, 40), (50, 80))
    ),
    # 200 Hz with the same 1 s discount horizon as 100 Hz at gamma 0.99: half the
    # boundary torque steps of 100 Hz.
    "spatial_rate_c": (
        Run(
            "pacer_200hz_g995",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "4000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", "20", "--gamma", "0.995"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
    ),
    # 100 Hz with a 2 s discount horizon, and with the larger network.
    "spatial_rate_b": (
        Run(
            "pacer_100hz_g995",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "3000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--gamma", "0.995"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
        Run(
            "pacer_100hz_512x512x256",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "3000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--hidden", "512", "512", "256"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
    ),
    # Larger actor and critic at the 500 Hz policy rate.
    "spatial_capacity": (
        Run(
            "pacer_512x512x256",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "6000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", "8", "--hidden", "512", "512", "256"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
    ),
    # One kilometre of deterministic inference at the acceptance substep (0.125 ms).
    # Policies are copied to <grid root>/policies before submission.
    "endurance_1km": tuple(
        Run(
            policy,
            SPATIAL_STATIC,
            (
                *("--policy", f"/home/gridsan/JO31399/newton_grid/policies/{policy}.pt", "--distance-m", "1000"),
                *("--worlds", "64", "--dt-s", "1.25e-4", "--quiet"),
            ),
            entry="projects.impedance_instron.bilateral.endurance3d",
            smoke_args=("--distance-m", "5"),
        )
        for policy in ("pacer_100hz_it344",)
    ),
    # Three-dimensional articulation, engineering-seed start.
    "spatial_baseline": (
        Run(
            "spatial_baseline",
            SPATIAL,
            _SPATIAL[:-2],
            smoke_args=_SPATIAL_SMOKE,
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
