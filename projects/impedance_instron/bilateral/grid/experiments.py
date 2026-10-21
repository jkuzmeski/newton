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
    tensorboard: bool = True
    """Pass the entry a ``--tensorboard`` log directory; only training entries take one."""


PLANAR = "dataset_6step"
SPATIAL = "dataset_3d"
PLANAR_STATIC = "dataset_6step_static"
SPATIAL_STATIC = "dataset_3d_static"
SPATIAL_STATIC_FIXED = "dataset_3d_static_sidefix"
"""The static dataset with each scanned shoe asset mounted on the foot it was scanned for."""
_BASE = ("--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "0")
_RUN4 = ("--reward-kernels", "pooled", "--force-penalty", "0")
_BODY = ("--body-position-weight", "0.3", "--body-velocity-weight", "0.1")
_REGULARIZERS = ("--contact-match-penalty", "0.1", "--action-smoothness-penalty", "0.02", "--previous-action-obs")
_FITTED_ACTUATION = ("--response-time-s", "0.005334445782959161", "--slew-limit")
"""Actuation of the runs before the no-lag, no-slew decision; the environment default is now bare."""
_SPATIAL = ("--kind", "3d", "--worlds", "1024", "--iterations", "1500", "--save-every", "100", "--seed", "0")
_SPATIAL_SMOKE = ("--iterations", "3", "--worlds", "256", "--save-every", "3")
_FT100 = "/home/gridsan/JO31399/newton_grid/runs/spatial_mujoco/16685657/00_ft100_mujoco/checkpoint.pt"
"""The finished MuJoCo fine-tune every shoe-adaptation arm starts from."""
LADDER = "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746"
"""The shoe-adaptation ladder job: teacher and robust arms, 1000 iterations each."""
TEACHER = "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746/02_teacher_asym/policy_teacher.pt"
TEACHER_CKPT = "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746/02_teacher_asym/checkpoint.pt"
"""The same teacher as a training checkpoint: actor, critic, both normalizers and the exploration
width, which is what a fine-tune should start from - a student with a fresh critic spends its
first iterations fitting a value function instead of learning to read its estimator."""
"""The teacher a student is distilled from; re-point it when a better teacher finishes.

The weights are run 16701746's final teacher, rebuilt from that arm's ``checkpoint.pt`` so the
file also carries the exploration std the teacher converged to - a student fine-tune that
restarts the noise at the library default is a wider policy than the weights describe, and it
spends its first hundred iterations learning not to fall over instead of learning to adapt.
``policy.pt`` beside it holds the same weights without the std."""
LONG = "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt_long/16701815"
"""The matched-compute ladder: every arm of 16701746 continued for 1000 more iterations."""
LONG_IDX = {
    "robust_asym_long": "00_robust_asym_long",
    "robust_sym_long": "01_robust_sym_long",
    "teacher_asym_long": "02_teacher_asym_long",
    "teacher_sym_long": "03_teacher_sym_long",
}
STUDENTS = "/home/gridsan/JO31399/newton_grid/runs/shoe_student_v2/16701881"
"""The RMA students: interface surgery from the headline teacher's training checkpoint, estimator
trained beside it. Job 16701841 is the same three arms after a first update above the rate cap
and with a critic that had to be fitted from scratch; its pose term settled about 40 per cent
under its teacher's and it is reported as that variant, not scored here."""
STUDENT_IDX = {
    "student_asym": "00_student_asym",
    "student_sym": "01_student_sym",
    "student_asym_frozen": "02_student_asym_frozen",
}
NOMINAL = "/home/gridsan/JO31399/newton_grid/runs/nominal_sameplant/16701854/00_nominal_sameplant/policy.pt"
"""The contrast arm: fine-tuned on the ladder's plant with the shoe fixed at its measured values."""
_SPATIAL_ACTUATION = (
    *("--kind", "3d", "--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "2"),
    *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco", "--leg-contact", "capsule_ramp"),
    *(
        "--resume",
        "/home/gridsan/JO31399/newton_grid/runs/spatial_mujoco/16685657/00_ft100_mujoco/checkpoint.pt",
    ),
)

FROZEN_POLICY = "/home/gridsan/JO31399/newton_grid/runs/spatial_mujoco/16685657/00_ft100_mujoco/policy.pt"
"""The published tracker: the frozen arm of the evaluation and the starting checkpoint of the ladder."""


_HELD_OUT = (
    "--replicates",
    "32",
    "--seconds",
    "10",
    "--raw-seconds",
    "1.5",
    "--shoes",
    "nominal,interp,extrap,archetype,swap",
    "--shoe-dr",
    "wide",
    "--compression-limit",
    "0.999",
    "--seed",
    "3",
)
"""The held-out shoe protocol: 32 start states per shoe, and one seed, so that worlds pair
across jobs. The compression screen is raised above the training value so that a very soft
shoe is scored rather than aborted."""


def student_args(teacher: str, name: str) -> tuple[str, ...]:
    """The RMA student's flags: interface surgery from ``teacher``, the shoe parameters out of
    the actor's inputs, an estimator over the contact history putting them back, and the
    asymmetric reward for the arms whose name ends in ``asym``. Index 2 is A-RMA's third phase
    (arXiv 2205.15299): the estimator is frozen and only the policy adapts to its errors, the
    arm that closes the gap between the estimate used to collect and the estimate used to
    deploy."""
    return (
        *("--kind", "3d", "--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "2"),
        *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
        *("--leg-contact", "capsule_ramp", "--actuation", "foh"),
        *("--teacher", teacher),
        *("--shoe-dr", "wide", "--contact-obs", "--estimator"),
        # Fine-tuning from a converged teacher: the KL controller climbs to its cap inside one
        # iteration, and at the default cap that overwrites the teacher behaviour this arm is
        # meant to keep.
        *("--learning-rate-cap", "5.0e-5"),
        *(
            (
                "--stance-pose-relief",
                "0.5",
                "--pose-group-weights",
                "1.0",
                "0.5",
                "0.5",
                "--cadence-weight",
                "0.05",
            )
            if name.endswith("asym")
            else ()
        ),
        *(("--freeze-estimator",) if "frozen" in name else ()),
    )


def arms(*pairs: tuple[str, str]) -> tuple[str, ...]:
    """``--arm name=checkpoint`` flags for a sequence of (name, checkpoint) pairs."""
    return tuple(flag for name, checkpoint in pairs for flag in ("--arm", f"{name}={checkpoint}"))


def estimators(*pairs: tuple[str, str]) -> tuple[str, ...]:
    """``--encoder name=checkpoint`` flags, for the student arms that read an embedding."""
    return tuple(flag for name, checkpoint in pairs for flag in ("--encoder", f"{name}={checkpoint}"))


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
            tensorboard=False,
        )
        for policy in (
            "pacer_100hz_it344",
            "pacer_100hz_it800",
            "pacer_100g995_it600",
            "pacer_100big_it500",
            "pacer_200g995_it1000",
        )
    ),
    # Solver throughput: Featherstone against MuJoCo Warp over world counts.
    "solver_bench": (
        Run(
            "bench_dt250us",
            SPATIAL_STATIC,
            (
                *("--policy", "/home/gridsan/JO31399/newton_grid/policies/pacer_100hz_it1400.pt"),
                *("--worlds", "64", "1024", "4096", "16384", "32768", "--dt-s", "2.5e-4"),
            ),
            entry="projects.impedance_instron.bilateral.benchmark3d",
            smoke_args=("--worlds", "64"),
            tensorboard=False,
        ),
    ),
    # The Featherstone-trained 100 Hz policy run on each solver for 100 m.
    "endurance_solver": tuple(
        Run(
            f"it1400_{solver}",
            SPATIAL_STATIC,
            (
                *("--policy", "/home/gridsan/JO31399/newton_grid/policies/pacer_100hz_it1400.pt"),
                *("--distance-m", "100", "--worlds", "64", "--dt-s", "1.25e-4", "--solver", solver, "--quiet"),
            ),
            entry="projects.impedance_instron.bilateral.endurance3d",
            smoke_args=("--distance-m", "5"),
            tensorboard=False,
        )
        for solver in ("featherstone", "mujoco")
    ),
    # Training on MuJoCo Warp: fine-tuned from the Featherstone 100 Hz run, and from scratch.
    "spatial_mujoco": (
        Run(
            "ft100_mujoco",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "2"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
                *("--resume", "/home/gridsan/JO31399/newton_grid/policies/pacer_100hz_checkpoint_mj.pt"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
        Run(
            "pacer_100hz_mujoco",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "3000", "--save-every", "100", "--seed", "0"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
    ),
    # Full gait recordings (kinematics, kinetics, body transforms) for the biomechanics report.
    "biomech_gait": tuple(
        Run(
            f"gait_{name}",
            SPATIAL_STATIC,
            (*("--policy", f"/home/gridsan/JO31399/newton_grid/{policy}"), "--worlds", "64", "--seconds", "12"),
            entry="projects.impedance_instron.bilateral.gait3d",
            smoke_args=("--seconds", "3"),
            tensorboard=False,
        )
        for name, policy in (
            ("base_it1400", "policies/pacer_100hz_it1400.pt"),
            ("pose", "runs/spatial_finetune/16684704/00_ft100_pose/policy_01500.pt"),
            ("mujoco_ft", "runs/spatial_mujoco/16685657/00_ft100_mujoco/policy_01500.pt"),
            ("mujoco_scratch", "runs/spatial_mujoco/16685657/01_pacer_100hz_mujoco/policy_03000.pt"),
        )
    ),
    # Gait scores and a 1 km run for the finished fine-tunes and MuJoCo runs against the 100 Hz base.
    "finetune_eval": tuple(
        Run(
            f"{test}_{name}",
            SPATIAL_STATIC,
            (*("--policy", f"/home/gridsan/JO31399/newton_grid/{policy}"), *args),
            entry=f"projects.impedance_instron.bilateral.{test}",
            smoke_args=("--seconds", "3") if test == "gait3d" else ("--distance-m", "5"),
            tensorboard=False,
        )
        for name, policy in (
            ("base_it1400", "policies/pacer_100hz_it1400.pt"),
            *(
                (tag, f"runs/spatial_finetune/16684704/{index}_ft100_{tag}/policy_01500.pt")
                for index, tag in (("00", "pose"), ("01", "pose_smooth"), ("02", "pose_noeffort"))
            ),
            *(
                (tag, f"runs/spatial_mujoco/16685657/{index}_{run}/policy_01500.pt")
                for index, tag, run in (
                    ("00", "mujoco_ft", "ft100_mujoco"),
                    ("01", "mujoco_scratch", "pacer_100hz_mujoco"),
                )
            ),
        )
        for test, args in (
            ("gait3d", ("--worlds", "64", "--seconds", "10")),
            ("endurance3d", ("--distance-m", "1000", "--worlds", "64", "--dt-s", "1.25e-4", "--quiet")),
        )
    ),
    # Fine-tune the 100 Hz run's iteration 1200 toward the measured motion. The base run
    # continuing to 3000 iterations is the control for "more training".
    "spatial_finetune": tuple(
        Run(
            f"ft100_{name}",
            SPATIAL_STATIC,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "1"),
                *("--dt-s", "2.5e-4", "--substeps", "40"),
                *("--resume", "/home/gridsan/JO31399/newton_grid/policies/pacer_100hz_checkpoint.pt"),
                *("--reward-weights", *weights),
            ),
            smoke_args=_SPATIAL_SMOKE,
        )
        for name, weights in (
            ("pose", ("0.5", "0.1", "0.15", "0.15", "0.05", "0.05")),
            ("pose_smooth", ("0.45", "0.1", "0.15", "0.1", "0.05", "0.15")),
            ("pose_noeffort", ("0.55", "0.1", "0.15", "0.15", "0", "0.05")),
        )
    ),
    # Fine-tune the MuJoCo fine-tune with the legs allowed to collide, on the dataset whose
    # shoe assets sit on the feet they were scanned for. The frozen policy puts its legs
    # inside its own capsules on 21 % of control steps (median depth 37 mm) against 0.9 %
    # (3.7 mm) for the motion it imitates, so contact has to arrive by fine-tuning rather
    # than by decision: 18 of 64 worlds keep running under ``capsule_bare`` (4 kN/m, 120 N
    # per pair) and none survive ``capsule_light`` (10 kN/m, 300 N). Index 1 is the same run
    # without contact, the control for the dataset fix alone. Index 2 eases the same law in:
    # a policy that has never felt its legs touch cannot absorb 4 kN/m on the first contact,
    # so the stiffness and the per-pair cap climb from 1 kN/m and 30 N to the target over the
    # first 14,400 control steps, 30 % of the run, and hold there. Numbers and the capsule
    # set are in outputs/bilateral/grf_spike_2026_10_14/leg_collision/REPORT.md.
    "leg_contact": tuple(
        Run(
            f"ft100_{name}",
            SPATIAL_STATIC_FIXED,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "2"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
                *(
                    "--resume",
                    "/home/gridsan/JO31399/newton_grid/runs/spatial_mujoco/16685657/00_ft100_mujoco/checkpoint.pt",
                ),
                *("--leg-contact", profile),
                *extra,
            ),
            smoke_args=_SPATIAL_SMOKE,
        )
        for name, profile, extra in (
            ("contact", "capsule_bare", ()),
            ("fixed_data", "off", ()),
            ("contact_ramp", "capsule_ramp", ()),
            ("ramp_light", "capsule_ramp_light", ()),
            # The physics alone cannot fix the adduction - a 120 N cap is affordable, so the policy
            # pushes through instead of moving differently - so this run prices the crossing in the
            # reward: 0.2 per control step at full crossing, saturating at 30 mm of penetration.
            ("ramp_penalty", "capsule_ramp", ("--leg-contact-penalty", "0.2")),
            # The adduction offset is worth 1.3 % of the reward under the pose term as written - one
            # coordinate of ten behind a shared 8.6 deg scale - so these two runs reweight hip
            # adduction inside that term instead of adding another penalty to the sum.
            (
                "ramp_penalty_add4",
                "capsule_ramp",
                (
                    "--leg-contact-penalty",
                    "0.2",
                    "--pose-dof-weights",
                    "1",
                    "4",
                    "1",
                    "1",
                    "1",
                    "1",
                    "4",
                    "1",
                    "1",
                    "1",
                ),
            ),
            (
                "ramp_penalty_add8",
                "capsule_ramp",
                (
                    "--leg-contact-penalty",
                    "0.2",
                    "--pose-dof-weights",
                    "1",
                    "8",
                    "1",
                    "1",
                    "1",
                    "1",
                    "8",
                    "1",
                    "1",
                    "1",
                ),
            ),
        )
    ),
    # Shoe adaptation. The midsole parameters (foam moduli, equilibrium fraction, Maxwell
    # time, unfitted normal damping, Coulomb coefficient, stack height) become a per-world
    # input drawn from a prior, resampled at reset and mid-episode, so a novel shoe is a
    # testable input rather than a retraining event. Prior, archetypes and the held-out
    # split are in projects/impedance_instron/bilateral/shoe_randomization.py; the write
    # path is checked against the shared runtime bit for bit in
    # newton/tests/test_impedance_bilateral_shoe_randomization.py. This is a 2x2: the
    # information the policy gets (nothing privileged versus the shoe parameters itself,
    # which is the teacher an RMA student is later distilled from) by the asymmetry of the
    # reward (strict everywhere versus strict pelvis, belt speed and cadence with the
    # stance knee and ankle relaxed, which is what lets a leg adapt to a different sole).
    # All four fine-tune the same finished MuJoCo policy, so index 2 doubles as the
    # control for what randomizing the shoe costs the tracker. Leg contact is on from the
    # first step on the ramp profile, as for the leg-contact study. The impedance setpoints
    # are interpolated across the substeps ("foh"): with the frozen 100 Hz policy it halves
    # the mid-stance ground-force roughness above 30 Hz, from 32.6 N to 14.4 N, and it adds
    # no delay, which the filtered and rate-limited profiles do (they stop the frozen runner
    # walking at all, so those are fine-tune-only arms).
    # See outputs/bilateral/grf_spike_2026_10_14/smooth_force/ for the roughness numbers.
    "shoe_adapt": tuple(
        Run(
            f"{information}_{reward}",
            SPATIAL_STATIC_FIXED,
            (
                *(
                    "--kind",
                    "3d",
                    "--worlds",
                    "4096",
                    "--iterations",
                    "1000",
                    "--save-every",
                    "100",
                    "--seed",
                    "2",
                    "--dt-s",
                    "2.5e-4",
                    "--substeps",
                    "40",
                    "--solver",
                    "mujoco",
                    "--leg-contact",
                    "capsule_ramp",
                    "--actuation",
                    "foh",
                ),
                *(
                    "--resume",
                    "/home/gridsan/JO31399/newton_grid/runs/spatial_mujoco/16685657/00_ft100_mujoco/checkpoint.pt",
                ),
                *("--shoe-dr", "wide"),
                *(("--contact-obs", "--theta-obs") if information == "teacher" else ()),
                *(
                    (
                        "--stance-pose-relief",
                        "0.5",
                        "--pose-group-weights",
                        "1.0",
                        "0.5",
                        "0.5",
                        "--cadence-weight",
                        "0.05",
                    )
                    if reward == "asym"
                    else ()
                ),
            ),
            smoke_args=_SPATIAL_SMOKE,
        )
        for information in ("robust", "teacher")
        for reward in ("asym", "sym")
    ),
    # Fine-tune the finished MuJoCo policy once per actuation model of the impedance
    # setpoints, so each arm is trained against the plant it will be scored with. None of
    # these can be judged by replay: with the frozen 100 Hz policy a 10 ms setpoint lag
    # leaves 14 of 32 worlds running and a 20 ms lag leaves none, while first-order hold
    # and a faster control clock - the two that add no delay - keep all 32 (roughness,
    # jerk and the per-substep setpoint step of 155 mrad against 36 are in
    # outputs/bilateral/grf_spike_2026_10_14/smooth_force/REPORT.md). Index 0 is the
    # control at the same seed, data and checkpoint. Every arm carries the ramped
    # leg-collision law, as the shoe-adaptation runs do, so the arms share the world the
    # world-model teacher is trained in. Score them with the same actuation they trained
    # with: it rides in the policy JSON, and ``smooth_force/roll.py`` rebuilds it.
    "spatial_actuation": (
        Run("zoh", SPATIAL_STATIC_FIXED, _SPATIAL_ACTUATION, smoke_args=_SPATIAL_SMOKE),
        Run("foh", SPATIAL_STATIC_FIXED, (*_SPATIAL_ACTUATION, "--actuation", "foh"), smoke_args=_SPATIAL_SMOKE),
        Run("lag10", SPATIAL_STATIC_FIXED, (*_SPATIAL_ACTUATION, "--actuation", "lag10"), smoke_args=_SPATIAL_SMOKE),
        Run("lag20", SPATIAL_STATIC_FIXED, (*_SPATIAL_ACTUATION, "--actuation", "lag20"), smoke_args=_SPATIAL_SMOKE),
        Run(
            "hz200",
            SPATIAL_STATIC_FIXED,
            (*_SPATIAL_ACTUATION, "--substeps", "20", "--gamma", "0.995"),
            smoke_args=_SPATIAL_SMOKE,
        ),
    ),
    # The same two teacher arms with the inter-leg penetration penalty in the reward
    # (weight 0.2, saturating at 30 mm of crossing, so a fully crossed step costs about a
    # third of the total reward on the steps that matter). The frozen tracker crosses its
    # legs on about a fifth of control steps and every tracking term is blind to it, so
    # this is the teacher the contact study can also endorse; the pair to compare against
    # is shoe_adapt index 1 and 3, identical but for this weight.
    "shoe_adapt_penalty": tuple(
        Run(
            f"teacher_{reward}_pen" + ("_hip4" if hip4 else ""),
            SPATIAL_STATIC_FIXED,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "1000", "--save-every", "100", "--seed", "2"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
                *("--leg-contact", "capsule_ramp", "--leg-contact-penalty", "0.2", "--actuation", "foh"),
                *("--resume", _FT100),
                *("--shoe-dr", "wide", "--contact-obs", "--theta-obs"),
                *(
                    (
                        "--stance-pose-relief",
                        "0.5",
                        "--pose-group-weights",
                        "1.0",
                        "0.5",
                        "0.5",
                        "--cadence-weight",
                        "0.05",
                    )
                    if reward == "asym"
                    else ()
                ),
                *(("--pose-dof-weights", "1", "4", "1", "1", "1", "1", "4", "1", "1", "1") if hip4 else ()),
            ),
            smoke_args=_SPATIAL_SMOKE,
        )
        for reward in ("asym", "sym")
        for hip4 in (False, True)
    ),
    "shoe_student": tuple(
        Run(name, SPATIAL_STATIC_FIXED, student_args(TEACHER, name), smoke_args=_SPATIAL_SMOKE)
        for name in ("student_asym", "student_sym", "student_asym_frozen")
    ),
    # The students again from the same teacher, with the two things job 16701841 lacked. Its
    # first update ran at --learning-rate because the cap only bounded the controller's
    # increases, and its critic started from scratch because a frozen policy file does not carry
    # one; together those took the arm roughly 40 per cent below its teacher's pose term inside
    # thirty iterations and PPO at the capped rate never brought it back. Here the cap applies
    # from iteration 1 and --teacher is the ladder arm's training checkpoint, so the critic
    # crosses over with the actor. Same seeds, shoes, plant and iteration count.
    "shoe_student_v2": tuple(
        Run(name, SPATIAL_STATIC_FIXED, student_args(TEACHER_CKPT, name), smoke_args=_SPATIAL_SMOKE)
        for name in ("student_asym", "student_sym", "student_asym_frozen")
    ),
    # Every arm of the ladder, run for a further 1000 iterations from its own checkpoint.
    # Submitted as insurance rather than as a verdict: the theta columns enter the actor at
    # zero weight, so a teacher can sit behind its own control at 1000 iterations purely
    # because it is still learning to read them. Whichever way the held-out evaluation comes
    # out, this is the same comparison at matched compute, which the 1000-iteration table on
    # its own cannot make. Flags mirror each arm exactly, observation set included, so the
    # checkpoint loads without interface surgery and Adam keeps its moments.
    "shoe_adapt_long": tuple(
        Run(
            f"{information}_{reward}_long",
            SPATIAL_STATIC_FIXED,
            (
                "--kind",
                "3d",
                "--worlds",
                "4096",
                "--iterations",
                "1000",
                "--save-every",
                "100",
                "--seed",
                "2",
                "--dt-s",
                "2.5e-4",
                "--substeps",
                "40",
                "--solver",
                "mujoco",
                "--leg-contact",
                "capsule_ramp",
                "--actuation",
                "foh",
                *(("--resume", f"{LADDER}/{index:02d}_{information}_{reward}/checkpoint.pt")),
                "--shoe-dr",
                "wide",
                *(("--contact-obs", "--theta-obs") if information == "teacher" else ()),
                *(
                    (
                        "--stance-pose-relief",
                        "0.5",
                        "--pose-group-weights",
                        "1.0",
                        "0.5",
                        "0.5",
                        "--cadence-weight",
                        "0.05",
                    )
                    if reward == "asym"
                    else ()
                ),
            ),
            smoke_args=_SPATIAL_SMOKE,
        )
        for index, (information, reward) in enumerate(
            (info, rew) for info in ("robust", "teacher") for rew in ("asym", "sym")
        )
    ),
    # The held-out shoe evaluation of the ladder, one array index per arm so the five run in
    # parallel on one GPU each. Every arm is scored in the environment stored beside its own
    # checkpoint, on 32 start states per shoe, against the measured shoe plus the held-out
    # points inside and outside the trained prior, the archetypes, and a mid-run swap. Index
    # 0 is the published tracker, which has never seen a shoe that was not the measured one;
    # the deltas it reports are what the whole adaptation effort is worth.
    "shoe_eval": tuple(
        Run(
            name,
            SPATIAL_STATIC_FIXED,
            (
                *("--arm", f"{name}={policy}"),
                *_HELD_OUT,
            ),
            entry="projects.impedance_instron.bilateral.eval_shoes",
            tensorboard=False,
        )
        for name, policy in (
            ("frozen_headline", FROZEN_POLICY),
            ("robust_asym", "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746/00_robust_asym/policy.pt"),
            ("robust_sym", "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746/01_robust_sym/policy.pt"),
            ("teacher_asym", "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746/02_teacher_asym/policy.pt"),
            ("teacher_sym", "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746/03_teacher_sym/policy.pt"),
        )
    ),
    # Held-out evaluation of the penalty teachers: the same shoes, start states and seed as
    # the ladder evaluation, with the ladder's headline arm scored in the same job so the
    # paired deltas are read off one table rather than across two submissions.
    "shoe_eval_penalty": tuple(
        Run(
            name,
            SPATIAL_STATIC_FIXED,
            (
                *("--arm", f"{name}={policy}"),
                *_HELD_OUT,
            ),
            entry="projects.impedance_instron.bilateral.eval_shoes",
            tensorboard=False,
        )
        for name, policy in (
            (
                "teacher_asym",
                "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt/16701746/02_teacher_asym/policy.pt",
            ),
            (
                "penalty_asym",
                "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt_penalty/16701800/00_teacher_asym_pen/policy.pt",
            ),
            (
                "penalty_hip4_asym",
                "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt_penalty/16701800/01_teacher_asym_pen_hip4/policy.pt",
            ),
            (
                "penalty_sym",
                "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt_penalty/16701800/02_teacher_sym_pen/policy.pt",
            ),
            (
                "penalty_hip4_sym",
                "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt_penalty/16701800/03_teacher_sym_pen_hip4/policy.pt",
            ),
        )
    ),
    # The like-for-like no-adaptation baseline: the plant, reward and starting checkpoint of
    # the ladder's arms - the finished MuJoCo fine-tune the frozen evaluation arm is scored
    # with, so the observation list is the plain one and --resume needs no surgery - with the
    # sole fixed at the measured shoe
    # out of the inputs - no shoe parameters and no contact channels, so the arm never sees a
    # privileged channel of any kind. Read against shoe_adapt/00_robust_asym, which is the
    # same arm with the prior switched on, the gap is the price of randomizing the sole at
    # fixed plant, fixed checkpoint and fixed iteration count; read against the ladder's
    # teacher it is the whole adaptation effect plus the prior.
    "nominal_sameplant": (
        Run(
            "nominal_sameplant",
            SPATIAL_STATIC_FIXED,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "1000", "--save-every", "100", "--seed", "2"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
                *("--leg-contact", "capsule_ramp", "--actuation", "foh"),
                *(
                    "--resume",
                    _FT100,
                    "--shoe-dr",
                    "off",
                ),
                *(
                    "--stance-pose-relief",
                    "0.5",
                    "--pose-group-weights",
                    "1.0",
                    "0.5",
                    "0.5",
                    "--cadence-weight",
                    "0.05",
                ),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
    ),
    # The evaluation the report is built on, in two array indices of four arms each: four GPUs
    # would be needed for one arm per index, and the client caps a job there, so arms sharing
    # an index are scored one after another on the same GPU. It carries the matched-compute
    # ladder (job 16701815, 1000 further iterations from each arm's own checkpoint) alongside
    # the students and the no-adaptation baseline, because the report's ladder rows should be
    # the best-trained version; the 16701746 rows stay as the shorter-training reference.
    # Same shoes, same 32 start states, same seed 3 and the same raised compression screen as
    # 16701820, so worlds pair one to one across the two jobs.
    # The evaluation the report is built on: one arm per array index, as for 16701820, so an
    # index is one GPU for about half an hour; eight indices go in two submissions of four,
    # which stays inside the client's four-GPU cap. It carries the matched-compute ladder
    # (job 16701815, 1000 further iterations from each arm's own checkpoint) next to the
    # students and the no-adaptation baseline, because the report's ladder rows should be the
    # best-trained version; the 16701746 rows stay as the shorter-training reference. Same
    # shoes, same 32 start states, same seed 3 and the same raised compression screen as
    # 16701820, so the worlds pair one to one across all of these jobs.
    "shoe_eval_wm": tuple(
        Run(
            name,
            SPATIAL_STATIC_FIXED,
            (*arms((name, policy)), *(estimators((name, encoder)) if encoder else ()), *_HELD_OUT),
            entry="projects.impedance_instron.bilateral.eval_shoes",
            tensorboard=False,
        )
        for name, policy, encoder in (
            ("robust_asym_long", f"{LONG}/{LONG_IDX['robust_asym_long']}/policy.pt", None),
            ("robust_sym_long", f"{LONG}/{LONG_IDX['robust_sym_long']}/policy.pt", None),
            ("teacher_asym_long", f"{LONG}/{LONG_IDX['teacher_asym_long']}/policy.pt", None),
            ("teacher_sym_long", f"{LONG}/{LONG_IDX['teacher_sym_long']}/policy.pt", None),
            (
                "student_asym",
                f"{STUDENTS}/{STUDENT_IDX['student_asym']}/policy.pt",
                f"{STUDENTS}/00_student_asym/encoder.pt",
            ),
            (
                "student_sym",
                f"{STUDENTS}/{STUDENT_IDX['student_sym']}/policy.pt",
                f"{STUDENTS}/01_student_sym/encoder.pt",
            ),
            # A-RMA's third phase: the policy adapts to a frozen estimator, so it is scored
            # against the same encoder file it trained against.
            (
                "student_asym_frozen",
                f"{STUDENTS}/{STUDENT_IDX['student_asym_frozen']}/policy.pt",
                f"{STUDENTS}/02_student_asym_frozen/encoder.pt",
            ),
            ("nominal_sameplant", NOMINAL, None),
        )
    ),
    # The frozen tracker rescored on the plant the ladder arms trained on, for one number.
    # Crossing is measured through the inter-leg contact law and this arm's checkpoint has that
    # law off, so until this run lands the held-out table cannot place the frozen tracker's
    # crossing beside the arms trained with it; every other column it reports is the same
    # worlds and the same shoes as its row in 16701820.
    "shoe_eval_frozen_ramp": (
        Run(
            "frozen_headline_ramp",
            SPATIAL_STATIC_FIXED,
            (
                *arms(
                    (
                        "frozen_headline",
                        "/home/gridsan/JO31399/newton_grid/runs/spatial_mujoco/16685657/00_ft100_mujoco/policy.pt",
                    )
                ),
                *_HELD_OUT,
                "--leg-contact",
                "capsule_ramp",
            ),
            entry="projects.impedance_instron.bilateral.eval_shoes",
            tensorboard=False,
        ),
    ),
    # The frozen tracker on the ladder's plant: first-order hold on the impedance setpoints and
    # the ramped inter-leg contact law, the two things the ladder arms trained under and this
    # checkpoint did not. With the same shoes, start states and seed as 16701820 it is the row
    # the adaptive arms' deltas should be read against, because it holds the plant fixed and
    # leaves only what the training changed. Its crossing is measured by the law itself, since
    # the law is on here.
    "shoe_eval_frozen_ladder_plant": (
        Run(
            "frozen_ladder_plant",
            SPATIAL_STATIC_FIXED,
            (
                *arms(("frozen_headline", FROZEN_POLICY)),
                *_HELD_OUT,
                "--leg-contact",
                "capsule_ramp",
                "--actuation",
                "foh",
            ),
            entry="projects.impedance_instron.bilateral.eval_shoes",
            tensorboard=False,
        ),
    ),
    # The no-shoe-randomization arm on the shoes it never saw. It trained on the ladder's plant,
    # so the two flags state that rather than change it: they pin the scored plant to first-order
    # hold plus the ramped inter-leg law whatever the checkpoint records, which is the discipline
    # the frozen cross-check row needed.
    "shoe_eval_nominal": (
        Run(
            "nominal_held_out",
            SPATIAL_STATIC_FIXED,
            (
                *arms(("nominal_sameplant", NOMINAL)),
                *_HELD_OUT,
                "--leg-contact",
                "capsule_ramp",
                "--actuation",
                "foh",
            ),
            entry="projects.impedance_instron.bilateral.eval_shoes",
            tensorboard=False,
        ),
    ),
    # The legs as MuJoCo collision shapes from the first step: the penalty kernel's capsules, right
    # leg against left only, and MuJoCo's own contact constraint instead of a capped force, so a
    # policy never meets passable legs. Every checkpoint so far trained with passable legs and
    # crosses them (the 300 N ramp still on 11 % of steps; under hard contact it keeps 3 of 64
    # worlds running), so the main arms start from scratch on the scratch MuJoCo recipe
    # (spatial_mujoco index 1: 3000 iterations, default reward) moved onto the side-fixed dataset
    # and first-order hold. Index 2 is the same run with contact off, the control for what the
    # contact itself changes; index 3 fine-tunes the published tracker under the same contact,
    # beside leg_contact index 3 (same start, the 300 N ramp). Scored with the force-free probe in
    # outputs/bilateral/grf_spike_2026_10_14/leg_collision_mujoco/REPORT.md.
    "leg_contact_mujoco": (
        *(
            Run(
                name,
                SPATIAL_STATIC_FIXED,
                (
                    *("--kind", "3d", "--worlds", "4096", "--iterations", "3000", "--save-every", "100"),
                    *("--seed", seed, "--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
                    *("--actuation", "foh", "--leg-contact", contact),
                ),
                smoke_args=_SPATIAL_SMOKE,
            )
            for name, seed, contact in (
                ("scratch_mujoco", "0", "mujoco"),
                ("scratch_mujoco_s1", "1", "mujoco"),
                ("scratch_off", "0", "off"),
            )
        ),
        Run(
            "ft100_mujoco_contact",
            SPATIAL_STATIC_FIXED,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "1500", "--save-every", "100", "--seed", "2"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
                *("--actuation", "foh", "--leg-contact", "mujoco", "--resume", _FT100),
            ),
            smoke_args=_SPATIAL_SMOKE,
        ),
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
    # The shoe-adaptive teachers with the ground-force tracking term taken out of the reward.
    # That term is a squared error against one shoe's measured per-side force, so every
    # force change a different sole would cause is penalized, and a held-out shoe can only
    # show up in the gait as a tracking error. Here force is an outcome of the shoe rather
    # than a target. The other five weights are divided by 0.75 so they keep their ratios
    # and still sum to one: the per-step reward stays on the scale the resumed critic was
    # fitted to, and the cadence term and the crossing law keep their weight relative to
    # the tracking sum, so the force term is the only thing that changes. Everything else -
    # plant, shoe prior, seed, starting checkpoint, 1000 iterations - is shoe_adapt index 2
    # and 3, the pair to read these against.
    "shoe_adapt_noforce": tuple(
        Run(
            f"teacher_{reward}_noforce",
            SPATIAL_STATIC_FIXED,
            (
                *("--kind", "3d", "--worlds", "4096", "--iterations", "1000", "--save-every", "100", "--seed", "2"),
                *("--dt-s", "2.5e-4", "--substeps", "40", "--solver", "mujoco"),
                *("--leg-contact", "capsule_ramp", "--actuation", "foh", "--resume", _FT100),
                *("--shoe-dr", "wide", "--contact-obs", "--theta-obs"),
                *("--reward-weights", "0.4666667", "0.1333333", "0.2666667", "0", "0.1333333", "0.1333333"),
                *(
                    (
                        "--stance-pose-relief",
                        "0.5",
                        "--pose-group-weights",
                        "1.0",
                        "0.5",
                        "0.5",
                        "--cadence-weight",
                        "0.05",
                    )
                    if reward == "asym"
                    else ()
                ),
            ),
            smoke_args=_SPATIAL_SMOKE,
        )
        for reward in ("asym", "sym")
    ),
    # Held-out evaluation of the no-force teachers, with the two shoe_adapt teachers they are
    # read against rescored in the same job: same snapshot, shoes, 32 start states and seed 3
    # as shoe_eval, so the worlds pair one to one across all four arms and the comparison
    # carries no code drift between evaluation snapshots.
    "shoe_eval_noforce": tuple(
        Run(
            name,
            SPATIAL_STATIC_FIXED,
            (*arms((name, policy)), *_HELD_OUT),
            entry="projects.impedance_instron.bilateral.eval_shoes",
            tensorboard=False,
        )
        for name, policy in (
            ("teacher_asym", f"{LADDER}/02_teacher_asym/policy.pt"),
            ("teacher_sym", f"{LADDER}/03_teacher_sym/policy.pt"),
            (
                "teacher_asym_noforce",
                "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt_noforce/16702795/00_teacher_asym_noforce/policy.pt",
            ),
            (
                "teacher_sym_noforce",
                "/home/gridsan/JO31399/newton_grid/runs/shoe_adapt_noforce/16702795/01_teacher_sym_noforce/policy.pt",
            ),
        )
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
