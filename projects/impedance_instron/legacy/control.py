# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""Equilibrium-point parameterization of the virtual leg and ankle of the impedance rig.

The virtual leg applies an impedance law in the sense of Hogan (1985):

.. math::

    f(t) = k(t)\,(L_0(t) - L(t)) + b(t)\,(\dot{L}_0(t) - \dot{L}(t))

``L0`` is the equilibrium (virtual) leg length, i.e. the motor command, while
``k`` and ``b`` set the mechanical impedance around it. The ankle uses the same
law about a commanded equilibrium angle:

.. math::

    \tau(t) = k_\theta(t)\,(\theta_0(t) - \theta(t))
              + b_\theta(t)\,(\dot{\theta}_0(t) - \dot{\theta}(t))

This module owns only the parameterization: it maps a flat parameter vector, the
decision variable of an outer optimizer, to smooth time profiles of ``L0``,
``L0dot``, ``k`` and ``b``, or of ``theta0``, ``theta0dot``, ``k_theta`` and
``b_theta``. It contains no simulation state and no rig-specific geometry.

Design choices that matter to the optimizer:

* All profiles are clamped cubic B-splines, so the commanded length is C2 and
  its rate is analytic rather than a finite difference of a sampled signal.
* Stiffness is parameterized in log space, so no point of the bound box can
  produce a non-positive stiffness.
* Damping is derived from a damping ratio, ``b = 2 zeta sqrt(k m)``, so the
  ratio stays invariant when the optimizer changes stiffness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["AnkleCommand", "LegCommand"]

_LENGTH_MIN_M = 0.5
_LENGTH_MAX_M = 1.6
_STIFFNESS_MIN_N_M = 500.0
_STIFFNESS_MAX_N_M = 120000.0
_ZETA_MIN = 0.05
_ZETA_MAX = 3.0
_MAX_DEGREE = 3

# The measured pitch spline of the running stance spans -0.452 rad to 1.183 rad.
# A 0.35 rad (20 deg) margin at each end, rounded outward, leaves the optimizer
# room to command an equilibrium outside the measured swing without letting it
# fold the fixture through the ground.
_ANGLE_MIN_RAD = -0.8
_ANGLE_MAX_RAD = 1.6
# Peak ankle torque recorded on this rig is 130-220 N*m. At 100 N*m/rad that peak
# deflects the ankle by more than a radian, i.e. fully compliant; at 20000 N*m/rad
# it deflects it by about 0.01 rad, i.e. as rigid as the prescribed-pitch replay it
# replaces. Stiffer commands are outside the plausible mechanical band and are also
# outside what the explicit substep can integrate.
_ANKLE_STIFFNESS_MIN_N_M_PER_RAD = 100.0
_ANKLE_STIFFNESS_MAX_N_M_PER_RAD = 20000.0


def _clamped_knots(control_points: int, degree: int) -> np.ndarray:
    """Return a clamped, uniformly spaced knot vector on ``[0, 1]``.

    Args:
        control_points: Number of B-spline coefficients.
        degree: Polynomial degree of the spline.
    """
    interior = control_points - degree - 1
    if interior < 0:
        raise ValueError(f"cubic basis needs at least {degree + 1} control points, got {control_points}")
    inner = np.linspace(0.0, 1.0, interior + 2)[1:-1]
    return np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])


def _basis(sites: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Return the Cox-de Boor basis matrix, shape ``[len(sites), control_points]``.

    Args:
        sites: Normalized evaluation sites in ``[0, 1]``.
        knots: Clamped knot vector.
        degree: Polynomial degree of the spline.
    """
    u = np.clip(np.asarray(sites, dtype=float), knots[0], knots[-1])
    spans = len(knots) - 1
    basis = np.zeros((u.size, spans))
    for i in range(spans):
        basis[:, i] = np.logical_and(u >= knots[i], u < knots[i + 1])
    # The clamped right end lies outside every half-open span, so it is folded
    # into the last non-empty span to keep the partition of unity intact.
    last = int(np.max(np.nonzero(knots[:-1] < knots[-1])[0]))
    at_end = u >= knots[-1]
    basis[at_end, :] = 0.0
    basis[at_end, last] = 1.0
    for p in range(1, degree + 1):
        columns = len(knots) - p - 1
        raised = np.zeros((u.size, columns))
        for i in range(columns):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0.0:
                raised[:, i] += (u - knots[i]) / left * basis[:, i]
            if right > 0.0:
                raised[:, i] += (knots[i + p + 1] - u) / right * basis[:, i + 1]
        basis = raised
    return basis


def _basis_derivative(sites: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Return the analytic derivative basis matrix with respect to the site.

    Args:
        sites: Normalized evaluation sites in ``[0, 1]``.
        knots: Clamped knot vector.
        degree: Polynomial degree of the spline.
    """
    control_points = len(knots) - degree - 1
    if degree == 0:
        return np.zeros((np.asarray(sites).size, control_points))
    lower = _basis(sites, knots, degree - 1)
    derivative = np.zeros((lower.shape[0], control_points))
    for i in range(control_points):
        left = knots[i + degree] - knots[i]
        right = knots[i + degree + 1] - knots[i + 1]
        if left > 0.0:
            derivative[:, i] += degree * lower[:, i] / left
        if right > 0.0:
            derivative[:, i] -= degree * lower[:, i + 1] / right
    return derivative


def _profile_basis(sites: np.ndarray, control_points: int, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the value basis and the time-derivative basis for one profile.

    Args:
        sites: Normalized evaluation sites in ``[0, 1]``.
        control_points: Number of B-spline coefficients.
        duration_s: Span of the evaluation time grid [s].
    """
    degree = min(_MAX_DEGREE, control_points - 1)
    knots = _clamped_knots(control_points, degree)
    value = _basis(sites, knots, degree)
    # Chain rule: sites are time normalized by the grid duration.
    rate = _basis_derivative(sites, knots, degree) / duration_s
    return value, rate


class LegCommand:
    """Flat optimizer parameters mapped to smooth virtual-leg profiles.

    The parameter vector is laid out as a single stable, documented block::

        [ length knots (length_knots) |
          log-stiffness knots (stiffness_knots) |
          damping-ratio knots (damping_knots) ]

    Length knots are in metres, log-stiffness knots are ``log(k)`` with ``k`` in
    N/m, and damping-ratio knots are dimensionless. Because a clamped B-spline
    is a convex combination of its coefficients, every profile stays inside the
    convex hull of its knots, so the bounds in :meth:`bounds` are also bounds on
    the evaluated profiles.

    Splines use degree ``min(3, knots - 1)``; a cubic basis needs at least four
    coefficients, and with fewer the spline degenerates to a single polynomial
    segment, which is smoother still.
    """

    @dataclass
    class Profile:
        """Virtual-leg profiles sampled on the time grid of the owning command."""

        length_m: np.ndarray
        """Equilibrium leg length L0 [m], shape [len(times)]."""

        length_rate_m_s: np.ndarray
        """Analytic equilibrium leg-length rate L0dot [m/s], shape [len(times)]."""

        stiffness_n_m: np.ndarray
        """Leg stiffness k [N/m], shape [len(times)]."""

        damping_n_s_m: np.ndarray
        """Leg damping b [N·s/m], shape [len(times)]."""

    def __init__(
        self,
        times: np.ndarray,
        length_knots: int = 6,
        stiffness_knots: int = 6,
        damping_knots: int = 3,
        mass_kg: float = 79.93,
    ):
        """Build the spline bases for a fixed evaluation time grid.

        Args:
            times: Strictly increasing evaluation times [s], shape [sample_count].
            length_knots: Number of equilibrium-length coefficients.
            stiffness_knots: Number of log-stiffness coefficients.
            damping_knots: Number of damping-ratio coefficients.
            mass_kg: Effective mass [kg] used in ``b = 2 zeta sqrt(k m)``.
        """
        grid = np.asarray(times, dtype=float).reshape(-1)
        if grid.size < 2:
            raise ValueError("times must contain at least two samples")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError("times must be strictly increasing")
        for name, count in (
            ("length_knots", length_knots),
            ("stiffness_knots", stiffness_knots),
            ("damping_knots", damping_knots),
        ):
            if int(count) < 2:
                raise ValueError(f"{name} must be at least 2, got {count}")
        if not mass_kg > 0.0:
            raise ValueError(f"mass_kg must be positive, got {mass_kg}")

        self.times = grid
        self.length_knots = int(length_knots)
        self.stiffness_knots = int(stiffness_knots)
        self.damping_knots = int(damping_knots)
        self.mass_kg = float(mass_kg)

        self.duration_s = float(grid[-1] - grid[0])
        sites = (grid - grid[0]) / self.duration_s

        self._length_basis, self._length_rate_basis = self._make_basis(sites, self.length_knots)
        self._stiffness_basis, _ = self._make_basis(sites, self.stiffness_knots)
        self._damping_basis, _ = self._make_basis(sites, self.damping_knots)

    def _make_basis(self, sites: np.ndarray, control_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the value basis and the time-derivative basis for one profile.

        Args:
            sites: Normalized evaluation sites in ``[0, 1]``.
            control_points: Number of B-spline coefficients.
        """
        return _profile_basis(sites, control_points, self.duration_s)

    @property
    def size(self) -> int:
        """Number of free scalar parameters."""
        return self.length_knots + self.stiffness_knots + self.damping_knots

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper parameter bounds, each shape [size].

        Log-stiffness bounds are the logarithms of the 500 N/m and 120000 N/m
        stiffness limits, so the box cannot express a non-positive stiffness.
        """
        lower = np.concatenate(
            [
                np.full(self.length_knots, _LENGTH_MIN_M),
                np.full(self.stiffness_knots, math.log(_STIFFNESS_MIN_N_M)),
                np.full(self.damping_knots, _ZETA_MIN),
            ]
        )
        upper = np.concatenate(
            [
                np.full(self.length_knots, _LENGTH_MAX_M),
                np.full(self.stiffness_knots, math.log(_STIFFNESS_MAX_N_M)),
                np.full(self.damping_knots, _ZETA_MAX),
            ]
        )
        return lower, upper

    def unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a parameter vector into length, log-stiffness and zeta knots.

        Args:
            params: Flat parameter vector, shape [size].
        """
        flat = np.asarray(params, dtype=float).reshape(-1)
        if flat.size != self.size:
            raise ValueError(f"expected {self.size} parameters, got {flat.size}")
        first = self.length_knots
        second = first + self.stiffness_knots
        return flat[:first], flat[first:second], flat[second:]

    def pack(
        self,
        length_knots: np.ndarray,
        log_stiffness_knots: np.ndarray,
        damping_ratio_knots: np.ndarray,
    ) -> np.ndarray:
        """Concatenate per-profile knots into one parameter vector, shape [size].

        Args:
            length_knots: Equilibrium-length coefficients [m], shape [length_knots].
            log_stiffness_knots: Log-stiffness coefficients, shape [stiffness_knots].
            damping_ratio_knots: Damping-ratio coefficients, shape [damping_knots].
        """
        blocks = (
            (np.asarray(length_knots, dtype=float).reshape(-1), self.length_knots),
            (np.asarray(log_stiffness_knots, dtype=float).reshape(-1), self.stiffness_knots),
            (np.asarray(damping_ratio_knots, dtype=float).reshape(-1), self.damping_knots),
        )
        for block, expected in blocks:
            if block.size != expected:
                raise ValueError(f"expected {expected} knots, got {block.size}")
        return np.concatenate([block for block, _ in blocks])

    def initial(self, length_m: np.ndarray, stiffness_n_m: float, damping_ratio: float) -> np.ndarray:
        """Return a seed vector reproducing a reference length at constant impedance.

        The length knots are the least-squares fit of the spline basis to
        ``length_m``, then clipped into the length bounds, so the seed tracks an
        existing hand-tuned reference instead of starting from a flat guess.

        Args:
            length_m: Reference equilibrium length [m], shape [len(times)].
            stiffness_n_m: Constant seed stiffness [N/m].
            damping_ratio: Constant seed damping ratio.
        """
        reference = np.asarray(length_m, dtype=float).reshape(-1)
        if reference.size != self.times.size:
            raise ValueError(f"length_m must have {self.times.size} samples, got {reference.size}")
        if not stiffness_n_m > 0.0:
            raise ValueError(f"stiffness_n_m must be positive, got {stiffness_n_m}")
        if not damping_ratio > 0.0:
            raise ValueError(f"damping_ratio must be positive, got {damping_ratio}")

        fit, *_ = np.linalg.lstsq(self._length_basis, reference, rcond=None)
        fit = np.clip(fit, _LENGTH_MIN_M, _LENGTH_MAX_M)
        log_k = math.log(float(np.clip(stiffness_n_m, _STIFFNESS_MIN_N_M, _STIFFNESS_MAX_N_M)))
        zeta = float(np.clip(damping_ratio, _ZETA_MIN, _ZETA_MAX))
        return self.pack(fit, np.full(self.stiffness_knots, log_k), np.full(self.damping_knots, zeta))

    def evaluate(self, params: np.ndarray) -> LegCommand.Profile:
        """Evaluate the profiles on the constructor time grid.

        Args:
            params: Flat parameter vector, shape [size].
        """
        length, log_stiffness, zeta = self.unpack(params)
        stiffness = np.exp(self._stiffness_basis @ log_stiffness)
        damping_ratio = self._damping_basis @ zeta
        return LegCommand.Profile(
            length_m=self._length_basis @ length,
            length_rate_m_s=self._length_rate_basis @ length,
            stiffness_n_m=stiffness,
            damping_n_s_m=2.0 * damping_ratio * np.sqrt(stiffness * self.mass_kg),
        )


class AnkleCommand:
    """Flat optimizer parameters mapped to smooth virtual-ankle profiles.

    The parameter vector is laid out as a single stable, documented block::

        [ angle knots (angle_knots) |
          log-stiffness knots (stiffness_knots) |
          damping-ratio knots (damping_knots) ]

    Angle knots are in radians, log-stiffness knots are ``log(k_theta)`` with
    ``k_theta`` in N·m/rad, and damping-ratio knots are dimensionless. This is
    :class:`LegCommand` with a rotational metric: the same clamped cubic
    B-splines, the same log-space stiffness, and the same commanded damping
    ratio, with ``b_theta = 2 zeta sqrt(k_theta I)`` about the pitch inertia
    instead of the leg mass.

    Splines use degree ``min(3, knots - 1)``; a cubic basis needs at least four
    coefficients, and with fewer the spline degenerates to a single polynomial
    segment, which is smoother still.
    """

    @dataclass
    class Profile:
        """Virtual-ankle profiles sampled on the time grid of the owning command."""

        angle_rad: np.ndarray
        """Equilibrium pitch angle theta0 [rad], shape [len(times)]."""

        angle_rate_rad_s: np.ndarray
        """Analytic equilibrium pitch rate theta0dot [rad/s], shape [len(times)]."""

        stiffness_nm_per_rad: np.ndarray
        """Ankle stiffness k_theta [N·m/rad], shape [len(times)]."""

        damping_nms_per_rad: np.ndarray
        """Ankle damping b_theta [N·m·s/rad], shape [len(times)]."""

    def __init__(
        self,
        times: np.ndarray,
        angle_knots: int = 6,
        stiffness_knots: int = 6,
        damping_knots: int = 3,
        inertia_kg_m2: float = 0.025,
    ):
        """Build the spline bases for a fixed evaluation time grid.

        Args:
            times: Strictly increasing evaluation times [s], shape [sample_count].
            angle_knots: Number of equilibrium-angle coefficients.
            stiffness_knots: Number of log-stiffness coefficients.
            damping_knots: Number of damping-ratio coefficients.
            inertia_kg_m2: Pitch inertia [kg·m²] used in ``b = 2 zeta sqrt(k I)``.
        """
        grid = np.asarray(times, dtype=float).reshape(-1)
        if grid.size < 2:
            raise ValueError("times must contain at least two samples")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError("times must be strictly increasing")
        for name, count in (
            ("angle_knots", angle_knots),
            ("stiffness_knots", stiffness_knots),
            ("damping_knots", damping_knots),
        ):
            if int(count) < 2:
                raise ValueError(f"{name} must be at least 2, got {count}")
        if not inertia_kg_m2 > 0.0:
            raise ValueError(f"inertia_kg_m2 must be positive, got {inertia_kg_m2}")

        self.times = grid
        self.angle_knots = int(angle_knots)
        self.stiffness_knots = int(stiffness_knots)
        self.damping_knots = int(damping_knots)
        self.inertia_kg_m2 = float(inertia_kg_m2)

        self.duration_s = float(grid[-1] - grid[0])
        sites = (grid - grid[0]) / self.duration_s

        self._angle_basis, self._angle_rate_basis = _profile_basis(sites, self.angle_knots, self.duration_s)
        self._stiffness_basis, _ = _profile_basis(sites, self.stiffness_knots, self.duration_s)
        self._damping_basis, _ = _profile_basis(sites, self.damping_knots, self.duration_s)

    @property
    def size(self) -> int:
        """Number of free scalar parameters."""
        return self.angle_knots + self.stiffness_knots + self.damping_knots

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper parameter bounds, each shape [size].

        Angle bounds are the measured pitch range widened by 0.35 rad at each
        end. Log-stiffness bounds are the logarithms of the 100 N·m/rad and
        20000 N·m/rad limits, so the box cannot express a non-positive ankle
        stiffness; the span covers everything from an ankle that yields by more
        than a radian under the measured peak torque to one that tracks its
        commanded equilibrium within about 0.01 rad.
        """
        lower = np.concatenate(
            [
                np.full(self.angle_knots, _ANGLE_MIN_RAD),
                np.full(self.stiffness_knots, math.log(_ANKLE_STIFFNESS_MIN_N_M_PER_RAD)),
                np.full(self.damping_knots, _ZETA_MIN),
            ]
        )
        upper = np.concatenate(
            [
                np.full(self.angle_knots, _ANGLE_MAX_RAD),
                np.full(self.stiffness_knots, math.log(_ANKLE_STIFFNESS_MAX_N_M_PER_RAD)),
                np.full(self.damping_knots, _ZETA_MAX),
            ]
        )
        return lower, upper

    def unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a parameter vector into angle, log-stiffness and zeta knots.

        Args:
            params: Flat parameter vector, shape [size].
        """
        flat = np.asarray(params, dtype=float).reshape(-1)
        if flat.size != self.size:
            raise ValueError(f"expected {self.size} parameters, got {flat.size}")
        first = self.angle_knots
        second = first + self.stiffness_knots
        return flat[:first], flat[first:second], flat[second:]

    def pack(
        self,
        angle_knots: np.ndarray,
        log_stiffness_knots: np.ndarray,
        damping_ratio_knots: np.ndarray,
    ) -> np.ndarray:
        """Concatenate per-profile knots into one parameter vector, shape [size].

        Args:
            angle_knots: Equilibrium-angle coefficients [rad], shape [angle_knots].
            log_stiffness_knots: Log-stiffness coefficients, shape [stiffness_knots].
            damping_ratio_knots: Damping-ratio coefficients, shape [damping_knots].
        """
        blocks = (
            (np.asarray(angle_knots, dtype=float).reshape(-1), self.angle_knots),
            (np.asarray(log_stiffness_knots, dtype=float).reshape(-1), self.stiffness_knots),
            (np.asarray(damping_ratio_knots, dtype=float).reshape(-1), self.damping_knots),
        )
        for block, expected in blocks:
            if block.size != expected:
                raise ValueError(f"expected {expected} knots, got {block.size}")
        return np.concatenate([block for block, _ in blocks])

    def initial(self, angle_rad: np.ndarray, stiffness_n_m_per_rad: float, damping_ratio: float) -> np.ndarray:
        """Return a seed vector reproducing a reference pitch at constant impedance.

        The angle knots are the least-squares fit of the spline basis to
        ``angle_rad``, then clipped into the angle bounds, so the seed tracks the
        measured pitch trajectory instead of starting from a flat guess.

        Args:
            angle_rad: Reference equilibrium pitch [rad], shape [len(times)].
            stiffness_n_m_per_rad: Constant seed ankle stiffness [N·m/rad].
            damping_ratio: Constant seed damping ratio.
        """
        reference = np.asarray(angle_rad, dtype=float).reshape(-1)
        if reference.size != self.times.size:
            raise ValueError(f"angle_rad must have {self.times.size} samples, got {reference.size}")
        if not stiffness_n_m_per_rad > 0.0:
            raise ValueError(f"stiffness_n_m_per_rad must be positive, got {stiffness_n_m_per_rad}")
        if not damping_ratio > 0.0:
            raise ValueError(f"damping_ratio must be positive, got {damping_ratio}")

        fit, *_ = np.linalg.lstsq(self._angle_basis, reference, rcond=None)
        fit = np.clip(fit, _ANGLE_MIN_RAD, _ANGLE_MAX_RAD)
        stiffness = np.clip(stiffness_n_m_per_rad, _ANKLE_STIFFNESS_MIN_N_M_PER_RAD, _ANKLE_STIFFNESS_MAX_N_M_PER_RAD)
        log_k = math.log(float(stiffness))
        zeta = float(np.clip(damping_ratio, _ZETA_MIN, _ZETA_MAX))
        return self.pack(fit, np.full(self.stiffness_knots, log_k), np.full(self.damping_knots, zeta))

    def evaluate(self, params: np.ndarray) -> AnkleCommand.Profile:
        """Evaluate the profiles on the constructor time grid.

        Args:
            params: Flat parameter vector, shape [size].
        """
        angle, log_stiffness, zeta = self.unpack(params)
        stiffness = np.exp(self._stiffness_basis @ log_stiffness)
        damping_ratio = self._damping_basis @ zeta
        return AnkleCommand.Profile(
            angle_rad=self._angle_basis @ angle,
            angle_rate_rad_s=self._angle_rate_basis @ angle,
            stiffness_nm_per_rad=stiffness,
            damping_nms_per_rad=2.0 * damping_ratio * np.sqrt(stiffness * self.inertia_kg_m2),
        )
