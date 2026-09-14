# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the simple two-stiffness impedance workflow."""

# Command-local imports keep --help independent of Warp and Torch.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import json
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
