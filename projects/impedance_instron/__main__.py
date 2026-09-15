# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the simple two-stiffness impedance workflow."""

# Command-local imports keep --help independent of Warp and Torch.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INPUTS = Path("outputs/impedance_instron/inputs")
OUTPUTS = Path("outputs/impedance_instron/simple")


def create_parser() -> argparse.ArgumentParser:
    """Create one small CLI for preparation, baseline, training and frozen evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build and freeze the inverse-dynamics equilibrium reference.")
    prepare.add_argument("--profile", type=Path, default=INPUTS / "reference_profile.json")
    prepare.add_argument("--variability", type=Path, default=INPUTS / "stride_variability.json")
    prepare.add_argument("--artifact", type=Path, default=INPUTS / "digital_shoe.json")
    prepare.add_argument("--output", type=Path, default=OUTPUTS / "reference.json")
    for name, help_text in (
        ("run", "Run the nominal two-stiffness baseline."),
        ("train", "Train two stiffness outputs on the fixed reference."),
    ):
        p = commands.add_parser(name, help=help_text)
        p.add_argument("--reference", type=Path, default=OUTPUTS / "reference.json")
        p.add_argument("--artifact", type=Path, default=INPUTS / "digital_shoe.json")
        p.add_argument("--device", default=None)
        p.add_argument("--worlds", type=int, default=1 if name == "run" else 32)
        p.add_argument("--output", type=Path, default=OUTPUTS / ("baseline" if name == "run" else "training"))
        if name == "train":
            p.add_argument("--iterations", type=int, default=100)
            p.add_argument("--seed", type=int, default=0)
    response = commands.add_parser("response", help="Compare movement intent and impedance under paired disturbances.")
    response.add_argument("--reference", type=Path, default=OUTPUTS / "reference.json")
    response.add_argument("--artifact", type=Path, default=INPUTS / "digital_shoe.json")
    response.add_argument("--device", default=None)
    response.add_argument("--output", type=Path, default=OUTPUTS / "response")
    response.add_argument("--stiffness-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    response.add_argument("--push-force-x-n", type=float, default=150.0, help="Peak raised-cosine upper-body push [N].")
    response.add_argument("--push-duration-s", type=float, default=0.04)
    response.add_argument(
        "--ground-offset-m", type=float, default=0.005, help="Test static planes at +/- this height [m]."
    )
    sensitivity = commands.add_parser(
        "sensitivity", help="Test independent impedance gains and shoe-material perturbations."
    )
    sensitivity.add_argument("--reference", type=Path, default=OUTPUTS / "reference.json")
    sensitivity.add_argument("--artifact", type=Path, default=INPUTS / "digital_shoe.json")
    sensitivity.add_argument("--output", type=Path, default=OUTPUTS / "sensitivity")
    sensitivity.add_argument("--device", default=None)
    sensitivity.add_argument("--modes", nargs="+", choices=("intent", "equilibrium"), default=["intent"])
    sensitivity.add_argument("--gain-multipliers", type=float, nargs="*", default=[0.5, 2.0])
    sensitivity.add_argument("--modulus-multipliers", type=float, nargs="*", default=[0.75, 1.25])
    sensitivity.add_argument("--relaxation-multipliers", type=float, nargs="*", default=[0.5, 2.0])
    sensitivity.add_argument(
        "--material",
        type=Path,
        action="append",
        default=[],
        help="Additional same-geometry material artifact; repeat to compare several.",
    )
    sensitivity.add_argument(
        "--push-force-n", type=float, default=150.0, help="Peak force for each signed directional pulse [N]."
    )
    sensitivity.add_argument("--push-duration-s", type=float, default=0.04)
    sensitivity.add_argument(
        "--push-phase", type=float, default=0.25, help="Pulse start as a fraction of recorded contact duration."
    )
    sensitivity.add_argument(
        "--full-factorial",
        action="store_true",
        help="Test directional pushes for every controller/material combination.",
    )
    sensitivity.add_argument("--minimum-recovery-window-s", type=float, default=0.15)
    sensitivity.add_argument("--recovery-dwell-s", type=float, default=0.05)
    sensitivity.add_argument("--position-tolerance-m", type=float, default=0.001)
    sensitivity.add_argument("--angle-tolerance-rad", type=float, default=0.001)
    sensitivity.add_argument("--velocity-tolerance-m-s", type=float, default=0.01)
    sensitivity.add_argument("--angular-velocity-tolerance-rad-s", type=float, default=0.01)
    report = commands.add_parser("report", help="Redraw figures from saved results without rerunning physics.")
    report.add_argument("directory", type=Path, help="Saved response or sensitivity suite directory.")
    report.add_argument(
        "--overview-only",
        action="store_true",
        help="Redraw only the sensitivity overview, retaining existing case pages.",
    )
    evaluation = commands.add_parser("evaluate", help="Restore a frozen experiment; optionally swap only material.")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument(
        "--material", type=Path, default=None, help="Explicit material-only replacement; geometry must match."
    )
    evaluation.add_argument("--device", default=None)
    evaluation.add_argument(
        "--allow-physics-update",
        action="store_true",
        help="Re-evaluate unchanged policy weights with updated source; old scores no longer apply.",
    )
    evaluation.add_argument("--output", type=Path, default=OUTPUTS / "evaluation")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute the requested workflow without importing the deprecated experiment."""
    args = create_parser().parse_args(argv)
    if args.command == "prepare":
        from .simple.reference import prepare_reference

        reference = prepare_reference(args.profile, args.variability, args.artifact)
        reference.save(args.output)
        print(f"Frozen reference: {args.output}\nIdentity: {reference.identity}")
        print(json.dumps(reference.provenance, indent=2, default=str))
        return
    if args.command == "train":
        from .simple.policy import train
        from .simple.reference import Reference

        result = train(
            Reference.load(args.reference),
            args.artifact,
            args.output,
            iterations=args.iterations,
            num_worlds=args.worlds,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(result, indent=2, default=str))
        return
    if args.command == "response":
        from .simple.reference import Reference
        from .simple.response import run_response_suite

        result = run_response_suite(
            Reference.load(args.reference),
            args.artifact,
            args.output,
            device=args.device,
            stiffness_multipliers=args.stiffness_multipliers,
            push_force_x_n=args.push_force_x_n,
            push_duration_s=args.push_duration_s,
            ground_offset_m=args.ground_offset_m,
            command=list(sys.argv) if argv is None else ["projects.impedance_instron", *argv],
        )
        print(f"Report: {result}")
        return
    if args.command == "sensitivity":
        from .simple.recovery import RecoveryConfig
        from .simple.sensitivity import run_sensitivity_suite

        recovery_config = RecoveryConfig(
            minimum_window_s=args.minimum_recovery_window_s,
            dwell_s=args.recovery_dwell_s,
            position_tolerance_m=args.position_tolerance_m,
            angle_tolerance_rad=args.angle_tolerance_rad,
            velocity_tolerance_m_s=args.velocity_tolerance_m_s,
            angular_velocity_tolerance_rad_s=args.angular_velocity_tolerance_rad_s,
        )
        result = run_sensitivity_suite(
            args.reference,
            args.artifact,
            args.output,
            device=args.device,
            modes=args.modes,
            gain_multipliers=args.gain_multipliers,
            modulus_multipliers=args.modulus_multipliers,
            relaxation_multipliers=args.relaxation_multipliers,
            material_paths=args.material,
            push_force_n=args.push_force_n,
            push_duration_s=args.push_duration_s,
            push_phase=args.push_phase,
            recovery_config=recovery_config,
            full_factorial=args.full_factorial,
            command=list(sys.argv) if argv is None else ["projects.impedance_instron", *argv],
        )
        print(f"Report: {result}")
        return
    if args.command == "report":
        from .simple.render_report import render_saved_report

        result = render_saved_report(args.directory, overview_only=args.overview_only)
        print(f"Figures redrawn from saved results: {result}")
        return
    from .simple.report import write_report

    if args.command == "run":
        from .simple.policy import evaluate_policy
        from .simple.reference import Reference
        from .simple.rig import Rig

        rig = Rig(Reference.load(args.reference), args.artifact, num_worlds=args.worlds, device=args.device)
        summary = evaluate_policy(rig)
        summary["physics"] = rig.metadata
    else:
        from .simple.policy import evaluate

        rig, summary = evaluate(
            args.checkpoint,
            artifact_path=args.material,
            device=args.device,
            allow_physics_update=args.allow_physics_update,
        )
    path = write_report(rig, args.output, summary)
    print(json.dumps(summary, indent=2, default=str))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
