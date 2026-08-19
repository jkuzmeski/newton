# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Project-local C3D-to-OpenSim gait reconstruction pipeline."""


def main(argv: list[str] | None = None) -> int:
    """Run :mod:`projects.gait_c3d.pipeline` without eager solver imports."""
    from .pipeline import create_parser, run_pipeline  # noqa: PLC0415

    run_pipeline(create_parser().parse_args(argv))
    return 0


__all__ = ["main"]
