# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""View, report, and record portable Digital Shoe artifacts."""

from projects._cli import dispatch

_COMMANDS = {
    "friction": ("projects.digital_shoe.friction_example", "Compare tangential laws with prescribed normal loads."),
    "friction-controller-refit": (
        "projects.digital_shoe.friction_controller_refit",
        "Refit equilibrium commands with fixed friction, gains and normal mechanics.",
    ),
    "friction-controller-check": (
        "projects.digital_shoe.friction_controller_validate",
        "Independently check refitted commands with fixed contact and gains.",
    ),
    "friction-continuity": (
        "projects.digital_shoe.friction_continuity",
        "Compare raw contact-force continuity with internal shear relaxation.",
    ),
    "friction-leg": ("projects.digital_shoe.friction_leg", "Replay frozen leg motion to compare friction forces."),
    "friction-metrics": ("projects.digital_shoe.friction_metrics", "Score braking and propulsive leg forces."),
    "friction-onset": (
        "projects.digital_shoe.friction_onset",
        "Audit initial foot motion and replay friction with frozen normal loads.",
    ),
    "friction-cache": (
        "projects.digital_shoe.friction_history",
        "Freeze normal and kinematic inputs for friction studies.",
    ),
    "friction-sweep": ("projects.digital_shoe.friction_sweep", "Sweep friction against fixed leg motion."),
    "friction-fit": (
        "projects.digital_shoe.friction_dynamic_search",
        "Fit friction in free leg dynamics with a frozen controller.",
    ),
    "friction-check": ("projects.digital_shoe.friction_dynamic", "Independently check a friction candidate on CPU."),
    "friction-report": ("projects.digital_shoe.friction_comparison", "Compare saved free-leg friction runs offline."),
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
