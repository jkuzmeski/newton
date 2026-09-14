# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit and validate Digital Instron data, then export a portable shoe."""

from projects._cli import dispatch

_COMMANDS = {
    "fit": (
        "projects.digital_instron_v2.workflow",
        "Fit averaged cycles; this command does not run held-out validation.",
    ),
    "validate": ("projects.digital_instron_v2.phase1", "Fit training cycles and evaluate adjacent held-out cycles."),
    "replay": ("projects.digital_instron_v2.phase2", "Fit and run dynamic bench validation."),
    "export": ("projects.digital_instron_v2.export_digital_shoe", "Fit, validate, and export an artifact and report."),
    "view": ("projects.digital_instron_v2.example", "View source-backed Instron, settle, stride, or attached scenes."),
    "profile": (
        "projects.digital_instron_v2.profile_calibration",
        "Benchmark forward fitting against a saved baseline.",
    ),
}


def main(argv: list[str] | None = None) -> None:
    """Select a task while retaining its existing module command and flags."""
    dispatch("projects.digital_instron_v2", __doc__, _COMMANDS, argv)


if __name__ == "__main__":
    main()
