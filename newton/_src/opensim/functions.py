# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OpenSim coordinate ``Function`` evaluation for joint kinematics.

A ``CustomJoint`` drives each of its six :class:`OsimTransformAxis` by a scalar
function of one (occasionally several) generalized coordinates. This module
ports the OpenSim functions needed to reproduce joint kinematics exactly:

- :class:`Constant`
- :class:`LinearFunction`
- :class:`PiecewiseLinearFunction`
- :class:`SimmSpline` (a faithful port of ``OpenSim::SimmSpline``, including its
  SIMM natural-cubic end conditions and out-of-range linear extrapolation)
- :class:`MultiplierFunction`

Functions are pure NumPy so kinematics and inverse kinematics run without Warp
or a GPU. :func:`build_function` constructs the right evaluator from the
serialized parameters produced by :func:`~newton.opensim.parse_osim`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TINY = 1.0e-8


class CoordinateFunction:
    """Base class for a scalar function of a single coordinate value."""

    def value(self, x: float) -> float:
        """Return the function value at ``x``."""
        raise NotImplementedError

    def __call__(self, x: float) -> float:
        return self.value(x)


@dataclass
class Constant(CoordinateFunction):
    """A constant function ``f(x) = c``."""

    c: float = 0.0

    def value(self, x: float) -> float:
        return self.c


@dataclass
class LinearFunction(CoordinateFunction):
    """An affine function ``f(x) = slope * x + intercept``."""

    slope: float = 1.0
    intercept: float = 0.0

    def value(self, x: float) -> float:
        return self.slope * x + self.intercept


@dataclass
class PiecewiseLinearFunction(CoordinateFunction):
    """Piecewise-linear interpolation through ``(x, y)`` knots."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def value(self, x: float) -> float:
        return float(np.interp(x, self.x, self.y))


@dataclass
class MultiplierFunction(CoordinateFunction):
    """Scales an inner function: ``f(x) = scale * inner(x)``."""

    inner: CoordinateFunction = field(default_factory=Constant)
    scale: float = 1.0

    def value(self, x: float) -> float:
        return self.scale * self.inner.value(x)


class SimmSpline(CoordinateFunction):
    """Faithful port of ``OpenSim::SimmSpline`` (SIMM natural cubic spline).

    Reproduces OpenSim's coefficient computation (``calcCoefficients``) and
    evaluation (``calcValue``) exactly, including the SIMM end conditions (third
    derivatives from divided differences) and linear extrapolation outside the
    knot range. Used for coupled ``CustomJoint`` coordinates such as the
    tibiofemoral translations of a gait model knee.
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._calc_coefficients()

    def _calc_coefficients(self) -> None:
        x, y = self.x, self.y
        n = len(x)
        b = np.zeros(n)
        c = np.zeros(n)
        d = np.zeros(n)
        if n < 2:
            self._b, self._c, self._d = b, c, d
            return
        if n == 2:
            t = max(_TINY, x[1] - x[0])
            b[0] = b[1] = (y[1] - y[0]) / t
            self._b, self._c, self._d = b, c, d
            return
        nm1, nm2 = n - 1, n - 2
        d[0] = max(_TINY, x[1] - x[0])
        c[1] = (y[1] - y[0]) / d[0]
        for i in range(1, nm1):
            d[i] = max(_TINY, x[i + 1] - x[i])
            b[i] = 2.0 * (d[i - 1] + d[i])
            c[i + 1] = (y[i + 1] - y[i]) / d[i]
            c[i] = c[i + 1] - c[i]
        # End conditions: third derivatives at endpoints from divided differences.
        b[0] = -d[0]
        b[nm1] = -d[nm2]
        c[0] = 0.0
        c[nm1] = 0.0
        if n > 3:
            d31 = max(_TINY, x[3] - x[1])
            d20 = max(_TINY, x[2] - x[0])
            d1 = max(_TINY, x[nm1] - x[n - 3])
            d2 = max(_TINY, x[nm2] - x[n - 4])
            d30 = max(_TINY, x[3] - x[0])
            d3 = max(_TINY, x[nm1] - x[n - 4])
            c[0] = c[2] / d31 - c[1] / d20
            c[nm1] = c[nm2] / d1 - c[n - 3] / d2
            c[0] = c[0] * d[0] * d[0] / d30
            c[nm1] = -c[nm1] * d[nm2] * d[nm2] / d3
        # Forward elimination.
        for i in range(1, n):
            t = d[i - 1] / b[i - 1]
            b[i] -= t * d[i - 1]
            c[i] -= t * c[i - 1]
        # Back substitution.
        c[nm1] /= b[nm1]
        for j in range(nm1):
            i = nm2 - j
            c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
        # Polynomial coefficients.
        b[nm1] = (y[nm1] - y[nm2]) / d[nm2] + d[nm2] * (c[nm2] + 2.0 * c[nm1])
        for i in range(nm1):
            b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
            d[i] = (c[i + 1] - c[i]) / d[i]
            c[i] *= 3.0
        c[nm1] *= 3.0
        d[nm1] = d[nm2]
        self._b, self._c, self._d = b, c, d

    def value(self, x: float) -> float:
        xs, ys = self.x, self.y
        n = len(xs)
        b, c, d = self._b, self._c, self._d
        if n == 0:
            return float("nan")
        # Out-of-range: extrapolate with the end slope (matches SIMM).
        if x < xs[0]:
            return float(ys[0] + (x - xs[0]) * b[0])
        if x > xs[n - 1]:
            return float(ys[n - 1] + (x - xs[n - 1]) * b[n - 1])
        if n < 3:
            k = 0
        else:
            i, j = 0, n
            while True:
                k = (i + j) // 2
                if x < xs[k]:
                    j = k
                elif x > xs[k + 1]:
                    i = k
                else:
                    break
        dx = x - xs[k]
        return float(ys[k] + dx * (b[k] + dx * (c[k] + dx * d[k])))


def build_function(function_type: str | None, params: dict) -> CoordinateFunction:
    """Construct a :class:`CoordinateFunction` from serialized parameters.

    Args:
        function_type: OpenSim function class name (``"LinearFunction"``,
            ``"SimmSpline"``, ``"Constant"``, ``"MultiplierFunction"``, ...).
        params: Serialized parameters as produced by the parser. Nested
            functions (``MultiplierFunction``) carry an ``"inner"`` dict with its
            own ``"type"`` and parameters.

    Returns:
        A callable coordinate function. Unknown types fall back to a zero
        :class:`Constant`.
    """
    t = function_type or params.get("type")
    if t in ("LinearFunction",):
        coeffs = params.get("coefficients")
        if coeffs is not None:
            coeffs = list(coeffs)
            return LinearFunction(coeffs[0] if coeffs else 1.0, coeffs[1] if len(coeffs) > 1 else 0.0)
        return LinearFunction(params.get("slope", 1.0), params.get("intercept", 0.0))
    if t in ("Constant",):
        return Constant(params.get("value", 0.0))
    if t in ("SimmSpline", "NaturalCubicSpline", "GCVSpline"):
        return SimmSpline(params.get("x", []), params.get("y", []))
    if t in ("PiecewiseLinearFunction",):
        return PiecewiseLinearFunction(np.asarray(params.get("x", []), float), np.asarray(params.get("y", []), float))
    if t in ("MultiplierFunction",):
        inner = params.get("inner", {})
        inner_fn = build_function(inner.get("type"), inner)
        return MultiplierFunction(inner_fn, params.get("scale", 1.0))
    return Constant(0.0)
