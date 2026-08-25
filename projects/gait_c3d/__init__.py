# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""C3D gait project; ``main`` launches the reference-only source-analysis pipeline."""

ARCHITECTURE_ROLE = "compatibility_entrypoint"


def main(argv: list[str] | None = None) -> int:
    """Run :mod:`projects.gait_c3d.compatibility.pipeline` without eager solver imports."""
    from .compatibility.pipeline import main as reference_main  # noqa: PLC0415

    return reference_main(argv)


__all__ = ["main"]
