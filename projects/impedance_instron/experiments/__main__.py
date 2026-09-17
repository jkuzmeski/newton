# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Create and run frozen-controller shoe variants and paired hysteresis benches."""

from __future__ import annotations

import argparse
from pathlib import Path

from .campaign import DEFAULT_BUNDLE, bench, create, qualify, reports, run


def main(argv: list[str] | None = None) -> None:
    """Expose separate sealed creation, qualification, rollout, bench, and report stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("create", help="Seal controller and build the selected shoe suite; no rollout")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    build.add_argument(
        "--suite",
        choices=("sensitivity", "paper_compression"),
        default="sensitivity",
        help="Default 31-case sensitivity sweep or baseline plus two paper-compression surrogates",
    )
    build.add_argument("--fixtures", nargs="+", default=["rearfoot_punch", "fullfoot_last"])
    qualification = commands.add_parser("qualify", help="Run unchanged baseline numerical qualification; no fit")
    qualification.add_argument("campaign", type=Path)
    rollout = commands.add_parser("run", help="Run native/refined frozen conditions; no optimizer")
    rollout.add_argument("campaign", type=Path)
    rollout.add_argument(
        "--case", action="append", dest="selected", help="Repeat to select conditions; includes baseline"
    )
    rollout.add_argument("--device", default="cuda:0")
    rollout.add_argument("--primary-only", action="store_true")
    rollout.add_argument(
        "--diagnostic", action="store_true", help="Mark unqualified smoke evidence; do not claim acceptance"
    )
    cycle = commands.add_parser("hysteresis", help="Run independent rearfoot/fullfoot loading cycles")
    cycle.add_argument("campaign", type=Path)
    cycle.add_argument("--case", action="append", dest="selected")
    cycle.add_argument("--device", default="cuda:0")
    report = commands.add_parser("report", help="Build offline comparison and every condition's hysteresis plots")
    report.add_argument("campaign", type=Path)
    report.add_argument(
        "--individual", action="store_true", help="Also audit spring replays and write individual stance reports"
    )
    args = parser.parse_args(argv)
    if args.command == "create":
        plan = create(args.output, bundle=args.bundle, fixtures=tuple(args.fixtures), suite=args.suite)
        print(f"Sealed {len(plan['conditions'])} conditions: {args.output / 'plan.json'}")
    elif args.command == "qualify":
        qualify(args.campaign)
    elif args.command == "run":
        run(
            args.campaign,
            selected=args.selected,
            device=args.device,
            clearance_matched=not args.primary_only,
            diagnostic=args.diagnostic,
        )
    elif args.command == "hysteresis":
        bench(args.campaign, selected=args.selected, device=args.device)
    elif args.command == "report":
        print(reports(args.campaign, individual=args.individual))


if __name__ == "__main__":
    main()
