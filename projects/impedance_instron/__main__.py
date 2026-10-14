# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare, fit, report, and generate with the Hogan variable-impedance runner."""

import sys

from projects._cli import dispatch

_IDENTIFY_COMMANDS = ("inspect", "fit", "evaluate")
_COMMANDS = {
    "prepare": ("projects.impedance_instron.cartesian.prepare_dataset", "Prepare peak-to-peak stance observations."),
    "inspect": ("projects.impedance_instron.hogan.identify", "Check physical input compatibility."),
    "fit": ("projects.impedance_instron.hogan.identify", "Fit the shared runner with LM and write its HTML report."),
    "evaluate": ("projects.impedance_instron.hogan.identify", "Evaluate a frozen runner without refitting."),
    "generate": (
        "projects.impedance_instron.hogan.generate",
        "Predict from a frozen model and initial-state scenario.",
    ),
    "report": ("projects.impedance_instron.hogan.fit_report", "Rebuild a saved LM fit's HTML report."),
    "visual3d": ("projects.impedance_instron.cartesian.visual3d", "Inspect or import processed measurements."),
    "prepare-visual3d": ("projects.impedance_instron.cartesian.prepare_visual3d", "Prepare a Visual3D reference."),
    "prepare-bilateral": (
        "projects.impedance_instron.bilateral.data",
        "Prepare continuous two-leg multi-stride segments.",
    ),
    "mirror-shoe": ("projects.impedance_instron.bilateral.shoes", "Mirror a shoe artifact for the other foot."),
    "fit-bilateral": (
        "projects.impedance_instron.bilateral.linear",
        "Fit the shared two-leg linear Hogan baseline with LM.",
    ),
    "train-policy": ("projects.impedance_instron.bilateral.ppo", "Train the two-leg neural impedance policy (PPO)."),
    "evaluate-bilateral": (
        "projects.impedance_instron.bilateral.evaluate",
        "Score a frozen two-leg policy or linear runner.",
    ),
    "prepare-bilateral-3d": (
        "projects.impedance_instron.bilateral.data3d",
        "Prepare 3D two-leg segments on a planar dataset's windows.",
    ),
    "evaluate-bilateral-3d": (
        "projects.impedance_instron.bilateral.evaluate3d",
        "Score a frozen 3D two-leg policy.",
    ),
}


def main(argv: list[str] | None = None) -> None:
    """Forward commands without duplicating the tools' parsers."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in _IDENTIFY_COMMANDS:
        arguments.insert(1, arguments[0])
    dispatch("projects.impedance_instron", __doc__, _COMMANDS, arguments)


if __name__ == "__main__":
    main()
