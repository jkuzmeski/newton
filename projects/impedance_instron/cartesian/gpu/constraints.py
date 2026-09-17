# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linear control-polygon constraints and ray-feasibility kernels for 12x4 splines.

The 264 linear inequalities A @ theta <= b represent the algebraic control-polygon
bounds from Spline.bounds and canonical_knot_scales:
  - 96 position constraints (12 control points x 4 channels x 2 bounds)
  - 88 rate constraints (11 derivative intervals x 4 channels x 2 bounds)
  - 80 acceleration constraints (10 second-derivative intervals x 4 channels x 2 bounds)

Coordinates can be mapped between physical equilibrium coefficients theta and
dimensionless optimizer coordinates z:
  theta = theta_start + S * z
  A_z = A * S
  b_z = b - A @ theta_start
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ..trajectory import Spline
from .resident import _check_bounds, canonical_knot_scales

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

_CONTROL_COUNT = 12
_CHANNELS = 4
_VARIABLE_COUNT = _CONTROL_COUNT * _CHANNELS  # 48
_POSITION_CONSTRAINTS = _CONTROL_COUNT * _CHANNELS * 2  # 96
_RATE_CONSTRAINTS = (_CONTROL_COUNT - 1) * _CHANNELS * 2  # 88
_ACC_CONSTRAINTS = (_CONTROL_COUNT - 2) * _CHANNELS * 2  # 80
_TOTAL_CONSTRAINTS = _POSITION_CONSTRAINTS + _RATE_CONSTRAINTS + _ACC_CONSTRAINTS  # 264


@wp.kernel
def _physical_points(
    points: wp.array2d[wp.float64],
    anchor: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    physical: wp.array3d[wp.float64],
):
    """Map resident scaled proposals into the canonical coefficient layout."""
    w, j = wp.tid()
    physical[w, j // 4, j % 4] = anchor[j] + scale[j] * points[w, j]


@wp.kernel
def _canonical_feasibility(
    coefficients: wp.array3d[wp.float64],
    duration: wp.float64,
    lower: wp.vec4d,
    upper: wp.vec4d,
    rates: wp.vec4d,
    accelerations: wp.vec4d,
    first: wp.array[wp.float64],
    second: wp.array[wp.float64],
    feasible: wp.array[int],
):
    """Apply the unchanged hard-bound expressions with no polytope tolerance."""
    w = wp.tid()
    feasible[w] = _check_bounds(coefficients, w, duration, lower, upper, rates, accelerations, first, second, 12)


@wp.kernel
def evaluate_feasibility_kernel(
    poly_a: wp.array2d[wp.float64],
    poly_b: wp.array[wp.float64],
    candidates: wp.array2d[wp.float64],
    feasible: wp.array[int],
    tolerance: wp.float64,
    constraint_count: int,
    dim: int,
):
    """Check linear polytope inequality feasibility A @ x <= b + tol for candidate points."""
    candidate_idx = wp.tid()
    is_feas = int(1)
    for c in range(constraint_count):
        row_val = wp.float64(0.0)
        for j in range(dim):
            row_val += poly_a[c, j] * candidates[candidate_idx, j]
        if not wp.isfinite(row_val) or row_val > poly_b[c] + tolerance:
            is_feas = int(0)
    feasible[candidate_idx] = is_feas


@wp.kernel
def compute_max_ray_step_kernel(
    poly_a: wp.array2d[wp.float64],
    poly_b: wp.array[wp.float64],
    origins: wp.array2d[wp.float64],
    directions: wp.array2d[wp.float64],
    max_steps: wp.array[wp.float64],
    tolerance: wp.float64,
    constraint_count: int,
    dim: int,
):
    """Compute maximum step alpha >= 0 such that A @ (x + alpha * d) <= b + tol."""
    candidate_idx = wp.tid()
    limit = wp.float64(wp.inf)
    for c in range(constraint_count):
        row_x = wp.float64(0.0)
        row_d = wp.float64(0.0)
        for j in range(dim):
            row_x += poly_a[c, j] * origins[candidate_idx, j]
            row_d += poly_a[c, j] * directions[candidate_idx, j]
        if not wp.isfinite(row_x) or not wp.isfinite(row_d) or row_x > poly_b[c] + tolerance:
            limit = wp.float64(0.0)
            break
        if row_d > wp.float64(0.0):
            slack = poly_b[c] - row_x
            step = (slack + tolerance) / row_d
            if step < wp.float64(0.0):
                step = wp.float64(0.0)
            if step < limit:
                limit = step
    max_steps[candidate_idx] = limit


def build_polytope_system(
    duration_s: float,
    lower: np.ndarray,
    upper: np.ndarray,
    rate_limit: np.ndarray,
    acc_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the 264 linear control-polygon inequalities A @ theta <= b.

    Args:
        duration_s: Positive spline duration [s].
        lower: Lower equilibrium bounds [m, m, rad, rad], shape (4,).
        upper: Upper equilibrium bounds [m, m, rad, rad], shape (4,).
        rate_limit: Absolute rate limits [m/s, m/s, rad/s, rad/s], shape (4,).
        acc_limit: Absolute acceleration limits [m/s^2, m/s^2, rad/s^2, rad/s^2], shape (4,).

    Returns:
        Matrix A of shape (264, 48) and vector b of shape (264,).
    """
    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    rate_limit = np.asarray(rate_limit, dtype=np.float64)
    acc_limit = np.asarray(acc_limit, dtype=np.float64)
    if (
        lower.shape != (4,)
        or upper.shape != (4,)
        or rate_limit.shape != (4,)
        or acc_limit.shape != (4,)
        or not (np.isfinite(lower).all() and np.isfinite(upper).all())
        or not (np.isfinite(rate_limit).all() and np.isfinite(acc_limit).all())
        or np.any(lower > upper)
        or np.any(rate_limit < 0.0)
        or np.any(acc_limit < 0.0)
    ):
        raise ValueError("Bounds must be finite, ordered, and derivative limits nonnegative")

    first_scale, second_scale = canonical_knot_scales()
    a_rows: list[np.ndarray] = []
    b_vals: list[float] = []

    # 1. Position bounds: 12 control points * 4 channels * 2 bounds = 96
    for i in range(_CONTROL_COUNT):
        for c in range(_CHANNELS):
            idx = i * _CHANNELS + c
            # theta[idx] <= upper[c]
            row_u = np.zeros(_VARIABLE_COUNT, dtype=np.float64)
            row_u[idx] = 1.0
            a_rows.append(row_u)
            b_vals.append(float(upper[c]))

            # -theta[idx] <= -lower[c]
            row_l = np.zeros(_VARIABLE_COUNT, dtype=np.float64)
            row_l[idx] = -1.0
            a_rows.append(row_l)
            b_vals.append(float(-lower[c]))

    # 2. Rate bounds: 11 control intervals * 4 channels * 2 bounds = 88
    for r in range(_CONTROL_COUNT - 1):
        for c in range(_CHANNELS):
            idx_r = r * _CHANNELS + c
            idx_next = (r + 1) * _CHANNELS + c
            s1 = float(first_scale[r])
            # first = s1 * (theta[r+1, c] - theta[r, c]) <= rate_limit[c] * duration
            row_u = np.zeros(_VARIABLE_COUNT, dtype=np.float64)
            row_u[idx_next] = s1
            row_u[idx_r] = -s1
            a_rows.append(row_u)
            b_vals.append(float(rate_limit[c] * duration))

            # -first <= rate_limit[c] * duration
            row_l = np.zeros(_VARIABLE_COUNT, dtype=np.float64)
            row_l[idx_next] = -s1
            row_l[idx_r] = s1
            a_rows.append(row_l)
            b_vals.append(float(rate_limit[c] * duration))

    # 3. Acceleration bounds: 10 control intervals * 4 channels * 2 bounds = 80
    for r in range(_CONTROL_COUNT - 2):
        for c in range(_CHANNELS):
            idx_r = r * _CHANNELS + c
            idx_r1 = (r + 1) * _CHANNELS + c
            idx_r2 = (r + 2) * _CHANNELS + c
            s2 = float(second_scale[r])
            s1_curr = float(first_scale[r])
            s1_next = float(first_scale[r + 1])
            # second = s2 * (s1_next * (theta[r+2, c] - theta[r+1, c]) - s1_curr * (theta[r+1, c] - theta[r, c]))
            #        = s2 * s1_next * theta[r+2] - s2 * (s1_next + s1_curr) * theta[r+1] + s2 * s1_curr * theta[r]
            w_r2 = s2 * s1_next
            w_r1 = -s2 * (s1_next + s1_curr)
            w_r = s2 * s1_curr

            row_u = np.zeros(_VARIABLE_COUNT, dtype=np.float64)
            row_u[idx_r2] = w_r2
            row_u[idx_r1] = w_r1
            row_u[idx_r] = w_r
            a_rows.append(row_u)
            b_vals.append(float(acc_limit[c] * (duration**2)))

            row_l = np.zeros(_VARIABLE_COUNT, dtype=np.float64)
            row_l[idx_r2] = -w_r2
            row_l[idx_r1] = -w_r1
            row_l[idx_r] = -w_r
            a_rows.append(row_l)
            b_vals.append(float(acc_limit[c] * (duration**2)))

    return np.array(a_rows, dtype=np.float64), np.array(b_vals, dtype=np.float64)


class SplineConstraints:
    """Manages the 264 linear control-polygon inequalities for 12x4 splines.

    Supports both original unscaled coefficients theta in R^48 (A @ theta <= b)
    and scaled dimensionless coordinates z in R^48 (A_z @ z <= b_z),
    where theta = theta_start + S @ z. Matrix checks and ray caps are algebraic
    tools; use the canonical hard-bound checker before admitting any candidate.
    """

    def __init__(
        self,
        duration_s: float,
        lower: Any,
        upper: Any,
        rate_limit: Any,
        acc_limit: Any,
        parameter_scale: Any | None = None,
        theta_start: Any | None = None,
        device: Any = None,
    ):
        """Construct the polytope system and optional coordinate scaling.

        Args:
            duration_s: Spline duration [s].
            lower: Equilibrium lower bounds [m, m, rad, rad].
            upper: Equilibrium upper bounds [m, m, rad, rad].
            rate_limit: Rate limits [m/s, m/s, rad/s, rad/s].
            acc_limit: Acceleration limits [m/s^2, m/s^2, rad/s^2, rad/s^2].
            parameter_scale: Four-channel parameter scale [m, m, rad, rad].
            theta_start: Initial anchor coefficients, shape (12, 4) or (48,).
            device: Warp device for array allocation.
        """
        self.duration_s = float(duration_s)
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)
        self.rate_limit = np.array(rate_limit, dtype=np.float64, copy=True)
        self.acc_limit = np.array(acc_limit, dtype=np.float64, copy=True)

        self.a_np, self.b_np = build_polytope_system(
            self.duration_s, self.lower, self.upper, self.rate_limit, self.acc_limit
        )

        if parameter_scale is not None:
            p_scale = np.asarray(parameter_scale, dtype=np.float64)
            if p_scale.shape != (4,) or not np.isfinite(p_scale).all() or np.any(p_scale <= 0.0):
                raise ValueError("parameter_scale must contain four finite positive scales")
            self.scale_diag = np.tile(p_scale, _CONTROL_COUNT)
        else:
            self.scale_diag = np.ones(_VARIABLE_COUNT, dtype=np.float64)

        if theta_start is not None:
            t_start = np.asarray(theta_start, dtype=np.float64)
            if t_start.size != _VARIABLE_COUNT or not np.isfinite(t_start).all():
                raise ValueError("theta_start must contain 48 finite values")
            self.theta_start = t_start.reshape(_VARIABLE_COUNT).copy()
        else:
            self.theta_start = np.zeros(_VARIABLE_COUNT, dtype=np.float64)

        self.a_z_np = self.a_np * self.scale_diag[None, :]
        self.b_z_np = self.b_np - self.a_np @ self.theta_start

        self.device = device
        self._a_wp = None
        self._b_wp = None
        self._a_z_wp = None
        self._b_z_wp = None

        if device is not None:
            self._init_device_arrays(device)

    def _init_device_arrays(self, device: Any):
        """Allocate device Warp arrays without pinning an inferred input device."""
        device = wp.get_device(device)
        self._a_wp = wp.array(self.a_np, dtype=wp.float64, device=device)
        self._b_wp = wp.array(self.b_np, dtype=wp.float64, device=device)
        self._a_z_wp = wp.array(self.a_z_np, dtype=wp.float64, device=device)
        self._b_z_wp = wp.array(self.b_z_np, dtype=wp.float64, device=device)
        first, second = canonical_knot_scales()
        self._first_wp = wp.array(first, dtype=wp.float64, device=device)
        self._second_wp = wp.array(second, dtype=wp.float64, device=device)
        self._anchor_wp = wp.array(self.theta_start, dtype=wp.float64, device=device)
        self._scale_wp = wp.array(self.scale_diag, dtype=wp.float64, device=device)
        self._physical_wp = None

    @staticmethod
    def _tolerance(value: float) -> float:
        """Validate an algebraic diagnostic tolerance, not a physical acceptance allowance."""
        if isinstance(value, bool) or not np.isfinite(value) or value < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        return float(value)

    @staticmethod
    def _device_points(values: wp.array[Any], device):
        """Validate layouts before creating a safe, owner-retaining array view."""
        valid_shape = (values.ndim == 2 and values.shape[1] == 48) or (values.ndim == 3 and values.shape[1:] == (12, 4))
        if not valid_shape or values.dtype != wp.float64 or not values.is_contiguous or values.device != device:
            raise ValueError(
                "Points must be contiguous float64 (batch,48) or (batch,12,4) arrays on the execution device"
            )
        return values.reshape((values.shape[0], 48))

    def check_control_bounds_host(self, candidate: np.ndarray, *, scaled: bool = False) -> bool:
        """Recheck the exact canonical hard bounds before admitting an optimizer candidate.

        Expanded matrix dot products can round differently at an active boundary.
        A polytope tolerance or ray length never replaces this strict check.
        """
        values = self.to_physical(candidate) if scaled else np.asarray(candidate, dtype=np.float64)
        if not np.isfinite(values).all() or values.size == 0:
            return False
        return all(
            Spline(self.duration_s, row).bounds(self.lower, self.upper, self.rate_limit, self.acc_limit)
            for row in values.reshape(-1, 12, 4)
        )

    def to_scaled(self, theta: np.ndarray) -> np.ndarray:
        """Map physical coefficients theta to dimensionless coordinates z = S^-1 (theta - theta_start)."""
        theta_arr = np.asarray(theta, dtype=np.float64)
        flat = theta_arr.reshape(-1, _VARIABLE_COUNT)
        z = (flat - self.theta_start) / self.scale_diag
        return z.reshape(theta_arr.shape)

    def to_physical(self, z: np.ndarray) -> np.ndarray:
        """Map dimensionless coordinates z to physical coefficients theta = theta_start + S * z."""
        z_arr = np.asarray(z, dtype=np.float64)
        flat = z_arr.reshape(-1, _VARIABLE_COUNT)
        theta = self.theta_start + flat * self.scale_diag
        return theta.reshape(z_arr.shape)

    def check_feasibility_host(
        self,
        candidate: np.ndarray,
        *,
        scaled: bool = False,
        tolerance: float = 0.0,
    ) -> bool:
        """Check feasibility on host using the linear polytope system.

        Args:
            candidate: Point in R^48 (or shape (12, 4)), or batch of points.
            scaled: Whether candidate is in scaled coordinates z.
            tolerance: Nonnegative tolerance added to RHS slacks.

        Returns:
            True if all inequalities are satisfied within tolerance.
        """
        tolerance = self._tolerance(tolerance)
        cand = np.asarray(candidate, dtype=np.float64)
        flat = cand.reshape(-1, _VARIABLE_COUNT)
        if not len(flat) or not np.isfinite(flat).all():
            return False
        a_mat = self.a_z_np if scaled else self.a_np
        b_vec = self.b_z_np if scaled else self.b_np
        slacks = a_mat @ flat.T - b_vec[:, None]
        return bool(np.all(slacks <= tolerance))

    def max_ray_step_host(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        *,
        scaled: bool = False,
        tolerance: float = 0.0,
    ) -> float:
        """Compute the maximum feasible step alpha >= 0 along a ray origin + alpha * direction.

        Args:
            origin: Feasible starting point (48,).
            direction: Search direction vector (48,).
            scaled: Whether origin and direction are in scaled coordinates z.
            tolerance: Slack tolerance for numerical boundary stability.

        Returns:
            Maximum feasible step length alpha >= 0.
        """
        tolerance = self._tolerance(tolerance)
        x = np.asarray(origin, dtype=np.float64).reshape(_VARIABLE_COUNT)
        d = np.asarray(direction, dtype=np.float64).reshape(_VARIABLE_COUNT)
        a_mat = self.a_z_np if scaled else self.a_np
        b_vec = self.b_z_np if scaled else self.b_np

        if not np.isfinite(x).all() or not np.isfinite(d).all():
            return 0.0
        ad = a_mat @ d
        slacks = b_vec - a_mat @ x + tolerance
        if np.any(slacks < 0.0):
            return 0.0
        active = ad > 0.0
        if not np.any(active):
            return float("inf")

        steps = slacks[active] / ad[active]
        min_step = float(np.min(steps))
        return max(0.0, min_step)

    def evaluate_feasibility_device(
        self,
        candidates: wp.array2d[wp.float64] | wp.array3d[wp.float64],
        feasible_out: wp.array[int],
        *,
        scaled: bool = False,
        tolerance: float = 0.0,
        device: Any = None,
    ):
        """Evaluate candidate feasibility on device.

        Args:
            candidates: 2D Warp array of shape (batch, 48) or 3D of shape (batch, 12, 4).
            feasible_out: 1D int32 Warp array of shape (batch,).
            scaled: Whether candidates are in scaled coordinates z.
            tolerance: Tolerance added to RHS.
            device: Warp execution device.
        """
        tolerance = self._tolerance(tolerance)
        dev = wp.get_device(device or self.device or candidates.device)
        cand_2d = self._device_points(candidates, dev)
        if feasible_out.shape != (cand_2d.shape[0],) or feasible_out.dtype != wp.int32 or feasible_out.device != dev:
            raise ValueError("feasible_out must be an int32 batch vector on the execution device")
        if self._a_wp is None or self._a_wp.device != dev:
            self._init_device_arrays(dev)

        a_wp = self._a_z_wp if scaled else self._a_wp
        b_wp = self._b_z_wp if scaled else self._b_wp

        wp.launch(
            evaluate_feasibility_kernel,
            dim=cand_2d.shape[0],
            inputs=[
                a_wp,
                b_wp,
                cand_2d,
                feasible_out,
                float(tolerance),
                _TOTAL_CONSTRAINTS,
                _VARIABLE_COUNT,
            ],
            device=dev,
            record_tape=False,
        )

    def evaluate_control_bounds_device(
        self,
        candidates: wp.array2d[wp.float64] | wp.array3d[wp.float64],
        feasible_out: wp.array[int],
        *,
        scaled: bool = False,
        device: Any = None,
    ):
        """Recheck exact physical control bounds on device before admitting a proposal.

        Warm the batch shape before capture. No tolerance expands the physical
        limits, even when the algebraic polytope checks use a diagnostic tolerance.
        """
        dev = wp.get_device(device or self.device or candidates.device)
        points = self._device_points(candidates, dev)
        count = points.shape[0]
        if feasible_out.shape != (count,) or feasible_out.dtype != wp.int32 or feasible_out.device != dev:
            raise ValueError("feasible_out must be an int32 batch vector on the execution device")
        if self._a_wp is None or self._a_wp.device != dev:
            self._init_device_arrays(dev)
        if scaled:
            if self._physical_wp is None or self._physical_wp.shape != (count, 12, 4):
                self._physical_wp = wp.empty((count, 12, 4), dtype=wp.float64, device=dev)
            wp.launch(
                _physical_points,
                dim=(count, 48),
                inputs=[points, self._anchor_wp, self._scale_wp, self._physical_wp],
                device=dev,
                record_tape=False,
            )
            physical = self._physical_wp
        else:
            physical = points.reshape((count, 12, 4))
        wp.launch(
            _canonical_feasibility,
            dim=count,
            inputs=[
                physical,
                self.duration_s,
                wp.vec4d(*self.lower),
                wp.vec4d(*self.upper),
                wp.vec4d(*self.rate_limit),
                wp.vec4d(*self.acc_limit),
                self._first_wp,
                self._second_wp,
                feasible_out,
            ],
            device=dev,
            record_tape=False,
        )

    def compute_max_ray_step_device(
        self,
        origins: wp.array2d[wp.float64] | wp.array3d[wp.float64],
        directions: wp.array2d[wp.float64] | wp.array3d[wp.float64],
        max_steps_out: wp.array[wp.float64],
        *,
        scaled: bool = False,
        tolerance: float = 0.0,
        device: Any = None,
    ):
        """Compute maximum feasible ray step on device.

        Args:
            origins: 2D Warp array of shape (batch, 48).
            directions: 2D Warp array of shape (batch, 48).
            max_steps_out: 1D float64 Warp array of shape (batch,).
            scaled: Whether inputs are in scaled coordinates z.
            tolerance: Tolerance for slacks.
            device: Warp execution device.
        """
        tolerance = self._tolerance(tolerance)
        dev = wp.get_device(device or self.device or origins.device)
        orig_2d = self._device_points(origins, dev)
        dir_2d = self._device_points(directions, dev)
        if orig_2d.shape != dir_2d.shape:
            raise ValueError("Origins and directions must have the same batch size")
        if (
            max_steps_out.shape != (orig_2d.shape[0],)
            or max_steps_out.dtype != wp.float64
            or max_steps_out.device != dev
        ):
            raise ValueError("max_steps_out must be a float64 batch vector on the execution device")
        if self._a_wp is None or self._a_wp.device != dev:
            self._init_device_arrays(dev)

        a_wp = self._a_z_wp if scaled else self._a_wp
        b_wp = self._b_z_wp if scaled else self._b_wp

        wp.launch(
            compute_max_ray_step_kernel,
            dim=orig_2d.shape[0],
            inputs=[
                a_wp,
                b_wp,
                orig_2d,
                dir_2d,
                max_steps_out,
                float(tolerance),
                _TOTAL_CONSTRAINTS,
                _VARIABLE_COUNT,
            ],
            device=dev,
            record_tape=False,
        )
