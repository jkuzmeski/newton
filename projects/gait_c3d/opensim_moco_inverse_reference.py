# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deprecated redirect to :mod:`projects.gait_c3d.oracles.opensim_moco_inverse_reference`."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

ARCHITECTURE_ROLE = "deprecated_redirect"
_REPLACEMENT = "projects.gait_c3d.oracles.opensim_moco_inverse_reference"

warnings.warn(
    f"{__name__} moved to {_REPLACEMENT}; update imports and module commands",
    DeprecationWarning,
    stacklevel=2,
)
_target = importlib.import_module(_REPLACEMENT)
__all__: list[str] = []
__all__.extend(getattr(_target, "__all__", [name for name in dir(_target) if not name.startswith("_")]))


def __getattr__(name: str) -> Any:
    """Forward attributes during the deprecation period."""
    return getattr(_target, name)


def __dir__() -> list[str]:
    """List replacement-module attributes during the deprecation period."""
    return sorted(set(globals()) | set(dir(_target)))


if __name__ == "__main__":
    raise SystemExit(_target.main())
