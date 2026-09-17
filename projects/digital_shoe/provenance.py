# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source identities for the shared shoe mechanics, independent of checkout paths."""

import hashlib
from pathlib import Path


def physics_source_identity() -> str:
    """Hash the runtime and every local shared-law implementation it calls."""
    base = Path(__file__).parent
    digest = hashlib.sha256()
    for name in (
        "runtime.py",
        "material.py",
        "contact.py",
        "friction_law.py",
        "friction_deflection.py",
        "friction_stribeck.py",
        "friction_pressure.py",
        "friction_slip_history.py",
        "friction_maxwell.py",
        "friction_parameter_adapter.py",
        "friction_solver.py",
        "friction_adapter.py",
    ):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((base / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
