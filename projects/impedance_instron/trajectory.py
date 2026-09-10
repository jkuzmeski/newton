# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construct a smooth pitch command without reconstructing marker translations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryCubic:
    """Natural C2 cubic through optionally smoothed, uniformly timed angle knots."""

    time_s: np.ndarray
    angle_rad: np.ndarray
    acceleration_rad_s2: np.ndarray
    smoothing: dict

    @classmethod
    def fit(cls, time_s, angle_rad, *, cutoff_hz: float = 12.0):
        """Smooth optical knots and retain analytic, continuous acceleration.

        A second-difference penalty has the requested approximate -3 dB cutoff
        on an infinite uniform grid. This is offline reference processing, not
        a causal robot filter or a claim of recovered optical bandwidth.
        """
        time = np.asarray(time_s, dtype=np.float64)
        raw = np.asarray(angle_rad, dtype=np.float64)
        if time.ndim != 1 or len(time) < 4 or raw.shape != time.shape:
            raise ValueError("Pitch reconstruction requires at least four scalar optical knots")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(raw)) or np.any(np.diff(time) <= 0.0):
            raise ValueError("Pitch knots must be finite and strictly time-ordered")
        dt = float(np.median(np.diff(time)))
        if not np.allclose(np.diff(time), dt, rtol=1.0e-6, atol=1.0e-9):
            raise ValueError("Pitch smoothing requires the original uniform optical clock")
        if not np.isfinite(cutoff_hz) or cutoff_hz < 0.0 or cutoff_hz >= 0.5 / dt:
            raise ValueError("Pitch cutoff must be zero (disabled) or below the optical Nyquist rate")
        raw = np.unwrap(raw)
        penalty = 0.0
        angle = raw.copy()
        if cutoff_hz > 0.0:
            penalty = (np.sqrt(2.0) - 1.0) / (2.0 * np.sin(np.pi * cutoff_hz * dt)) ** 4
            difference = np.diff(np.eye(len(time)), n=2, axis=0)
            angle = np.linalg.solve(np.eye(len(time)) + penalty * difference.T @ difference, raw)
        interval = np.diff(time)
        matrix = np.zeros((len(time) - 2, len(time) - 2))
        np.fill_diagonal(matrix, 2.0 * (interval[:-1] + interval[1:]))
        index = np.arange(len(time) - 3)
        matrix[index, index + 1] = interval[1:-1]
        matrix[index + 1, index] = interval[1:-1]
        rhs = 6.0 * np.diff(np.diff(angle) / interval)
        acceleration = np.r_[0.0, np.linalg.solve(matrix, rhs), 0.0]
        return cls(
            time,
            angle,
            acceleration,
            {
                "method": "second-difference smoothing + natural C2 cubic",
                "cutoff_hz": float(cutoff_hz),
                "source_rate_hz": 1.0 / dt,
                "penalty": float(penalty),
                "boundary": "natural cubic outside stance, retained optical context",
                "max_knot_change_deg": float(np.degrees(np.max(np.abs(angle - raw)))),
                "causal": False,
            },
        )

    def evaluate(self, time_s):
        """Return angle [rad], velocity [rad/s], and acceleration [rad/s^2]."""
        time = np.asarray(time_s, dtype=np.float64)
        if not np.all(np.isfinite(time)) or np.any(time < self.time_s[0]) or np.any(time > self.time_s[-1]):
            raise ValueError("Do not extrapolate pitch beyond the recorded optical context")
        index = np.clip(np.searchsorted(self.time_s, time, side="right") - 1, 0, len(self.time_s) - 2)
        h = self.time_s[index + 1] - self.time_s[index]
        left, right = self.time_s[index + 1] - time, time - self.time_s[index]
        a0, a1 = self.acceleration_rad_s2[index], self.acceleration_rad_s2[index + 1]
        y0, y1 = self.angle_rad[index], self.angle_rad[index + 1]
        angle = (
            a0 * left**3 / (6 * h)
            + a1 * right**3 / (6 * h)
            + (y0 - a0 * h * h / 6) * left / h
            + (y1 - a1 * h * h / 6) * right / h
        )
        velocity = -a0 * left**2 / (2 * h) + a1 * right**2 / (2 * h) + (y1 - y0) / h - (a1 - a0) * h / 6
        acceleration = (a0 * left + a1 * right) / h
        return angle, velocity, acceleration
