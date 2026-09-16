# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render saved Cartesian motion and verified spring histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .report import write_report


def main(argv: list[str] | None = None) -> None:
    """Write an offline HTML report without refitting the controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report",))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mesh-only", action="store_true")
    args = parser.parse_args(argv)
    directory = args.directory
    with np.load(directory / "reference.npz", allow_pickle=False) as archive:
        reference = dict(archive)
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    summary = json.loads((directory / "summary.json").read_text())
    profile = json.loads((directory / "profile.json").read_text())
    write_report(directory, reference, trace, summary, profile=profile, include_springs=not args.mesh_only)
    print(directory / "report.html")


if __name__ == "__main__":
    main()
