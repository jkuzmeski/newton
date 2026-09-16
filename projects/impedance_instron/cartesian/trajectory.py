# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Four-channel C2 hip-position and knee/ankle equilibrium splines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spline import _derivative_control_polygons, basis


def _vector(value, name: str) -> np.ndarray:
    """Validate a finite vector in hip-x, hip-z, knee, ankle order."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite four-element vector")
    return array


@dataclass(frozen=True)
class Spline:
    """Clamped cubic equilibrium with position [m, m, rad, rad] channels.

    Simple interior knots give continuous position, rate, and acceleration.
    The first two channels are Cartesian hip positions, not hip joint angles.
    """

    duration_s: float
    coefficients: np.ndarray

    def __post_init__(self):
        """Copy finite coefficients and validate the spline dimensions."""
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] != 4 or len(coefficients) < 4:
            raise ValueError("coefficients must have shape (control_count >= 4, 4)")
        if not np.isfinite(coefficients).all():
            raise ValueError("coefficients must be finite")
        duration = float(self.duration_s)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be finite and positive")
        coefficients = coefficients.copy()
        coefficients.setflags(write=False)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "coefficients", coefficients)

    def sample(self, time_s):
        """Return equilibrium, rate, and acceleration in [m, m, rad, rad] per time order."""
        return tuple(
            basis(time_s, self.duration_s, len(self.coefficients), derivative=order) @ self.coefficients
            for order in range(3)
        )

    def bounds(self, lower, upper, rate, acceleration) -> bool:
        """Check sufficient global mixed-unit position and derivative limits.

        The convex hulls of the control polygons bound the entire curve.
        These bounds are conservative, not sampled approximations.

        Args:
            lower: Lower equilibrium bounds [m, m, rad, rad].
            upper: Upper equilibrium bounds [m, m, rad, rad].
            rate: Absolute rate limits [m/s, m/s, rad/s, rad/s].
            acceleration: Absolute acceleration limits [m/s^2, m/s^2, rad/s^2, rad/s^2].
        """
        lower, upper = _vector(lower, "lower"), _vector(upper, "upper")
        rate, acceleration = _vector(rate, "rate"), _vector(acceleration, "acceleration")
        if np.any(lower > upper) or np.any(rate < 0) or np.any(acceleration < 0):
            raise ValueError("Bounds must be ordered and derivative limits nonnegative")
        first, second = _derivative_control_polygons(self.coefficients)
        return bool(
            np.all(self.coefficients >= lower)
            and np.all(self.coefficients <= upper)
            and np.all(np.abs(first) <= rate * self.duration_s)
            and np.all(np.abs(second) <= acceleration * self.duration_s**2)
        )
