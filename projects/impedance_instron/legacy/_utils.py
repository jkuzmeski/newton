# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared host serialization, formatting and trace reductions for retired tools."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


def json_default(value):
    """Serialize NumPy scalars, arrays and paths for a JSON default callback."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def integral(time: np.ndarray, values: np.ndarray) -> float:
    """Integrate sampled values with the existing trapezoidal reduction order."""
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))


def escape(value) -> str:
    """Escape a string-coerced value for HTML text and quoted attributes."""
    return html.escape(str(value), quote=True)


def momentum_checkpoints(
    times: np.ndarray, velocity: np.ndarray, loaded: np.ndarray, checkpoints: np.ndarray
) -> list[float]:
    """Sample contact-relative velocity changes at explicit checkpoint fractions.

    Args:
        times: Sample times [s], shape [sample_count].
        velocity: Velocity component [m/s], shape [sample_count].
        loaded: Loaded-sample mask, shape [sample_count].
        checkpoints: Contact phase fractions, shape [checkpoint_count].
    """
    if loaded.sum() < 2:
        return [0.0] * len(checkpoints)
    span, values = times[loaded], velocity[loaded]
    phase = (span - span[0]) / max(span[-1] - span[0], 1.0e-9)
    return np.interp(checkpoints, phase, values - values[0]).tolist()
