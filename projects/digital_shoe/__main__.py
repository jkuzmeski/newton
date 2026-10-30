# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate, view, record, or report a portable Digital Shoe artifact."""

from projects._cli import dispatch

_COMMANDS = {
    "view": ("projects.digital_shoe.showcase", "View artifact-only Instron, drop, or rocker scenes."),
    "report": ("projects.digital_shoe.report", "Rebuild the self-contained HTML validation report."),
    "record": ("projects.digital_shoe.record_gifs", "Record the three scenes and rebuild their report."),
}


def main(argv: list[str] | None = None) -> None:
    """Run one Digital Shoe task."""
    dispatch("projects.digital_shoe", __doc__, _COMMANDS, argv)


if __name__ == "__main__":
    main()
