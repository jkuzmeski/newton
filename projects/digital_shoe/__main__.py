# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""View, report, and record portable Digital Shoe artifacts."""

from projects._cli import dispatch

_COMMANDS = {
    "view": ("projects.digital_shoe.showcase", "View artifact-only Instron, drop, or rocker scenes."),
    "report": ("projects.digital_shoe.report", "Rebuild the self-contained HTML report from an artifact."),
    "record": ("projects.digital_shoe.record_gifs", "Record all three scenes and rebuild their report."),
    "check-acquisition": ("projects.digital_shoe.acquisition", "Validate a new-data acquisition manifest."),
}


def main(argv: list[str] | None = None) -> None:
    """Select a task while retaining its existing module command and flags."""
    dispatch("projects.digital_shoe", __doc__, _COMMANDS, argv)


if __name__ == "__main__":
    main()
