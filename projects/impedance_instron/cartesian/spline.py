# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Clamped cubic spline basis and derivative control polygons."""

from __future__ import annotations

import numpy as np

_DEGREE = 3


def _as_time_array(time_s) -> np.ndarray:
    """Convert times to floating point without changing their shape."""
    try:
        time = np.asarray(time_s, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_s must contain real numbers") from exc
    if not np.all(np.isfinite(time)):
        raise ValueError("time_s must be finite")
    return time


def _validate_duration(duration_s) -> float:
    """Validate and return a positive duration in seconds."""
    try:
        duration = float(duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_s must be a positive finite scalar") from exc
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be a positive finite scalar")
    return duration


def _validate_control_count(control_count: int) -> int:
    """Validate and return the number of cubic control points."""
    if isinstance(control_count, bool):
        raise ValueError("control_count must be an integer of at least four")
    try:
        count = int(control_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("control_count must be an integer of at least four") from exc
    if count != control_count or count < _DEGREE + 1:
        raise ValueError("control_count must be an integer of at least four")
    return count


def _uniform_knots(control_count: int, degree: int) -> np.ndarray:
    """Build a clamped uniform knot vector."""
    internal = np.arange(1, control_count - degree, dtype=np.float64) / (control_count - degree)
    return np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))


def _basis_from_knots(normalized_time: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate non-derivative B-spline basis functions for a knot vector."""
    # The temporary degree-zero basis includes the extra knot spans that are
    # eliminated by the Cox-de Boor recursion.
    control_count = len(knots) - degree - 1
    flat_time = normalized_time.reshape(-1)
    zero = np.zeros((flat_time.size, len(knots) - 1), dtype=np.float64)
    zero[(flat_time[:, None] >= knots[:-1]) & (flat_time[:, None] < knots[1:])] = 1.0
    zero[flat_time == 1.0, :] = 0.0
    zero[flat_time == 1.0, -1] = 1.0

    values = zero
    for order in range(1, degree + 1):
        left_denominator = knots[order:-1] - knots[: -order - 1]
        right_denominator = knots[order + 1 :] - knots[1:-order]
        left = np.zeros((flat_time.size, len(left_denominator)), dtype=np.float64)
        right = np.zeros_like(left)
        np.divide(
            (flat_time[:, None] - knots[: -order - 1]) * values[:, :-1],
            left_denominator,
            out=left,
            where=left_denominator != 0.0,
        )
        np.divide(
            (knots[order + 1 :] - flat_time[:, None]) * values[:, 1:],
            right_denominator,
            out=right,
            where=right_denominator != 0.0,
        )
        values = left + right
        values[flat_time == 1.0, :] = 0.0
        values[flat_time == 1.0, -1] = 1.0
    return values.reshape(*normalized_time.shape, control_count)


def basis(time_s, duration_s: float, control_count: int, derivative: int = 0) -> np.ndarray:
    """Evaluate clamped uniform cubic B-spline basis functions.

    Args:
        time_s: Evaluation time or times [s], restricted to ``[0, duration_s]``.
        duration_s: Spline duration [s].
        control_count: Number of control points, at least four.
        derivative: Time derivative order, zero, one, or two.

    Returns:
        Basis weights with shape ``time_s.shape + (control_count,)``. Derivative
        orders one and two have units of [1/s] and [1/s^2], respectively.
    """
    duration = _validate_duration(duration_s)
    count = _validate_control_count(control_count)
    if isinstance(derivative, bool) or derivative not in (0, 1, 2):
        raise ValueError("derivative must be zero, one, or two")
    time = _as_time_array(time_s)
    if np.any(time < 0.0) or np.any(time > duration):
        raise ValueError("time_s must not extrapolate beyond the spline duration")
    normalized = time / duration
    knots = _uniform_knots(count, _DEGREE)

    if derivative == 0:
        return _basis_from_knots(normalized, knots, _DEGREE)

    if derivative == 1:
        lower = _basis_from_knots(normalized, knots[1:-1], _DEGREE - 1)
        scale = _DEGREE / (knots[_DEGREE + 1 : count + _DEGREE] - knots[1:count])
        mapping = np.zeros((count - 1, count), dtype=np.float64)
        np.fill_diagonal(mapping, -scale)
        np.fill_diagonal(mapping[:, 1:], scale)
        return (lower @ mapping) / duration

    first_scale = _DEGREE / (knots[_DEGREE + 1 : count + _DEGREE] - knots[1:count])
    derivative_knots = knots[1:-1]
    second_scale = (_DEGREE - 1) / (derivative_knots[_DEGREE : count + 1] - derivative_knots[1 : count - 1])
    first_mapping = np.zeros((count - 1, count), dtype=np.float64)
    np.fill_diagonal(first_mapping, -first_scale)
    np.fill_diagonal(first_mapping[:, 1:], first_scale)
    second_mapping = np.zeros((count - 2, count - 1), dtype=np.float64)
    np.fill_diagonal(second_mapping, -second_scale)
    np.fill_diagonal(second_mapping[:, 1:], second_scale)
    lower = _basis_from_knots(normalized, knots[2:-2], _DEGREE - 2)
    return (lower @ second_mapping @ first_mapping) / duration**2


def _derivative_control_polygons(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return first and second normalized-time derivative control polygons."""
    count = len(coefficients)
    knots = _uniform_knots(count, _DEGREE)
    first_scale = _DEGREE / (knots[_DEGREE + 1 : count + _DEGREE] - knots[1:count])
    first = first_scale[:, None] * np.diff(coefficients, axis=0)
    derivative_knots = knots[1:-1]
    second_scale = (_DEGREE - 1) / (derivative_knots[_DEGREE : count + 1] - derivative_knots[1 : count - 1])
    second = second_scale[:, None] * np.diff(first, axis=0)
    return first, second
