# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit Instron sweeps and export a portable Digital Shoe artifact."""

from projects._cli import dispatch

_COMMANDS = {
    "fit": ("projects.digital_instron_v2.sweep_fit", "Fit one shoe's material to its Instron foam sweep."),
    "export": (
        "projects.digital_instron_v2.export_digital_shoe",
        "Export a sweep_fit material as a Digital Shoe artifact and report.",
    ),
}


def main(argv: list[str] | None = None) -> None:
    """Select a task while retaining its existing module command and flags."""
    dispatch("projects.digital_instron_v2", __doc__, _COMMANDS, argv)


if __name__ == "__main__":
    main()
