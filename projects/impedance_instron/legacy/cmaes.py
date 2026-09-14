# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-contained CMA-ES for expensive, mildly noisy black-box objectives.

The optimizer implements standard CMA-ES (Hansen): weighted intermediate
recombination, rank-one and rank-mu covariance updates, and cumulative
step-size adaptation. It depends on NumPy and the standard library only.

The interface is ask/tell, so the caller owns the evaluation loop and can log,
cache, or parallelize a whole generation of expensive rollouts. :meth:`CMAES.state`
returns a JSON-serializable checkpoint and :meth:`CMAES.restore` rebuilds an
optimizer that produces the identical next generation.

Bound handling: **reflection** into the box. A sampled coordinate that leaves
``[lower, upper]`` is folded back at the violated face, repeatedly if needed
(implemented in closed form with a modulo of the doubled box width). Reflection
is used instead of clipping because clipping maps a whole half-space onto a
single face and piles probability mass on the boundary, which biases the mean
and collapses the covariance in the violated direction. Reflection is a
piecewise-isometric bijection of the box, so the folded density stays bounded
and no face accumulates mass. The reflected (feasible) point is the point that
is evaluated *and* the point that is fed back into the distribution update, so
the sample used for the covariance update is always the sample that produced the
objective value. Coordinates with a non-finite or degenerate bound are clipped
instead, since folding is undefined there.

Non-finite objective values (``inf``/``nan``) are ranked worst instead of being
used numerically, so a failed rollout cannot poison the mean or the covariance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

# Smallest eigenvalue kept in the covariance factorization, relative to the
# largest one; bounds the condition number and keeps sigma updates finite.
_MIN_EIGENVALUE_RATIO = 1e-14
_MAX_CONDITION = 1e14
_MIN_SIGMA = 1e-300
_MAX_SIGMA = 1e150


@dataclass
class Result:
    """Outcome of a :func:`minimize` run."""

    x: np.ndarray
    fun: float
    evaluations: int
    generations: int
    history: list[float] = field(default_factory=list)


def _as_vector(values: Sequence[float] | np.ndarray, dim: int | None, name: str) -> np.ndarray:
    """Return ``values`` as a contiguous 1-D float64 array of length ``dim``."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if dim is not None and vector.size != dim:
        raise ValueError(f"{name} must have {dim} entries, got {vector.size}")
    return np.ascontiguousarray(vector)


def _reflect(x: np.ndarray, lower: np.ndarray, upper: np.ndarray, foldable: np.ndarray) -> np.ndarray:
    """Fold ``x`` into ``[lower, upper]`` along the foldable coordinates.

    Args:
        x: Points to repair, shape ``[..., dim]``.
        lower: Lower bounds, shape ``[dim]``.
        upper: Upper bounds, shape ``[dim]``.
        foldable: Mask of coordinates with a finite, non-degenerate width.
    """
    width = np.where(foldable, upper - lower, 1.0)
    # Unbounded coordinates are excluded from the modulo, which is undefined for them.
    shifted = np.where(foldable, x - np.where(foldable, lower, 0.0), 0.0)
    offset = np.mod(shifted, 2.0 * width)
    folded = np.where(foldable, lower, 0.0) + np.where(offset > width, 2.0 * width - offset, offset)
    repaired = np.where(foldable, folded, np.clip(x, lower, upper))
    # Guard against ties at a face produced by rounding in the modulo above.
    return np.clip(repaired, lower, upper)


def _rank_order(values: np.ndarray) -> np.ndarray:
    """Return indices sorting ``values`` ascending with non-finite entries last."""
    finite = np.isfinite(values)
    keys = np.where(finite, values, 0.0)
    return np.lexsort((keys, ~finite))


class CMAES:
    """Ask/tell CMA-ES with box bounds."""

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        population: int | None = None,
        seed: int = 0,
    ):
        """Initialize the search distribution.

        Args:
            x0: Initial distribution mean, shape ``[dim]``.
            sigma0: Initial global step size, in the units of ``x0``.
            bounds: Lower and upper box bounds, each shape ``[dim]``, or ``None``.
            population: Generation size; defaults to ``4 + floor(3 * ln(dim))``.
            seed: Seed of the internal random generator.
        """
        mean = _as_vector(x0, None, "x0")
        dim = mean.size
        if dim == 0:
            raise ValueError("x0 must not be empty")
        if not np.all(np.isfinite(mean)):
            raise ValueError("x0 must be finite")
        if not (sigma0 > 0.0) or not math.isfinite(sigma0):
            raise ValueError("sigma0 must be a positive finite number")

        if bounds is None:
            lower = np.full(dim, -np.inf)
            upper = np.full(dim, np.inf)
        else:
            lower = _as_vector(bounds[0], dim, "lower bound")
            upper = _as_vector(bounds[1], dim, "upper bound")
            if np.any(lower > upper):
                raise ValueError("every lower bound must not exceed its upper bound")

        self.dim = dim
        self.lower = lower
        self.upper = upper
        self._foldable = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        self.mean = _reflect(mean, lower, upper, self._foldable)
        self.sigma = float(sigma0)

        self.population = int(population) if population is not None else 4 + int(math.floor(3.0 * math.log(dim)))
        if self.population < 2:
            raise ValueError("population must be at least 2")
        self.mu = self.population // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = float(1.0 / np.sum(self.weights**2))

        n = float(dim)
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
        )
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.chin = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.covariance = np.eye(dim)
        self.eigenvectors = np.eye(dim)
        self.eigenvalues = np.ones(dim)

        self.generations = 0
        self.evaluations = 0
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._best_x = self.mean.copy()
        self._best_f = math.inf

    # -- sampling ----------------------------------------------------------

    def ask(self) -> np.ndarray:
        """Return the next generation, shape [population, dim], already inside bounds."""
        z = self.rng.standard_normal((self.population, self.dim))
        y = (z * self.eigenvalues) @ self.eigenvectors.T
        candidates = self.mean + self.sigma * y
        return _reflect(candidates, self.lower, self.upper, self._foldable)

    def tell(self, candidates: np.ndarray, values: np.ndarray) -> None:
        """Update the search distribution from evaluated candidates.

        Args:
            candidates: Evaluated points, shape ``[population, dim]``.
            values: Objective values, shape ``[population]``; non-finite entries rank worst.
        """
        points = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
        if points.shape != (scores.size, self.dim):
            raise ValueError(f"candidates must have shape [{scores.size}, {self.dim}], got {points.shape}")
        if scores.size < self.mu:
            raise ValueError(f"tell needs at least {self.mu} values, got {scores.size}")

        order = _rank_order(scores)
        selected = points[order[: self.mu]]

        self.evaluations += scores.size
        self.generations += 1
        top = order[0]
        if np.isfinite(scores[top]) and scores[top] < self._best_f:
            self._best_f = float(scores[top])
            self._best_x = points[top].copy()

        mean_old = self.mean
        self.mean = self.weights @ selected
        y_w = (self.mean - mean_old) / self.sigma

        # C^{-1/2} y_w in the current eigenbasis.
        inv_sqrt_y = self.eigenvectors @ ((self.eigenvectors.T @ y_w) / self.eigenvalues)
        self.p_sigma = (1.0 - self.cs) * self.p_sigma + math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * inv_sqrt_y

        p_sigma_norm = float(np.linalg.norm(self.p_sigma))
        decay = 1.0 - (1.0 - self.cs) ** (2 * self.generations)
        hsig = p_sigma_norm / math.sqrt(max(decay, 1e-16)) / self.chin < 1.4 + 2.0 / (self.dim + 1.0)
        self.p_c = (1.0 - self.cc) * self.p_c + (
            math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_w if hsig else 0.0
        )

        y_selected = (selected - mean_old) / self.sigma
        rank_mu = (y_selected * self.weights[:, None]).T @ y_selected
        correction = (1.0 - float(hsig)) * self.cc * (2.0 - self.cc)
        self.covariance = (
            (1.0 - self.c1 - self.cmu) * self.covariance
            + self.c1 * (np.outer(self.p_c, self.p_c) + correction * self.covariance)
            + self.cmu * rank_mu
        )

        self.sigma *= math.exp(min(1.0, (self.cs / self.damps) * (p_sigma_norm / self.chin - 1.0)))
        self.sigma = float(min(max(self.sigma, _MIN_SIGMA), _MAX_SIGMA))

        self._decompose()

    def _decompose(self) -> None:
        """Refresh the eigenbasis of the covariance, repairing degenerate matrices."""
        covariance = 0.5 * (self.covariance + self.covariance.T)
        if not np.all(np.isfinite(covariance)):
            covariance = np.eye(self.dim)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            covariance = np.eye(self.dim)
            eigenvalues, eigenvectors = np.ones(self.dim), np.eye(self.dim)
        if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
            covariance = np.eye(self.dim)
            eigenvalues, eigenvectors = np.ones(self.dim), np.eye(self.dim)

        largest = float(np.max(eigenvalues))
        if largest <= 0.0:
            covariance = np.eye(self.dim)
            eigenvalues, eigenvectors = np.ones(self.dim), np.eye(self.dim)
            largest = 1.0
        # Condition-number cap: flat directions would otherwise make C^{-1/2} explode.
        floor = max(largest * _MIN_EIGENVALUE_RATIO, largest / _MAX_CONDITION)
        eigenvalues = np.maximum(eigenvalues, floor)

        self.covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.eigenvectors = eigenvectors
        self.eigenvalues = np.sqrt(eigenvalues)

    # -- results and checkpointing ----------------------------------------

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Best point seen so far and its objective value."""
        return self._best_x.copy(), self._best_f

    def state(self) -> dict:
        """JSON-serializable state for checkpointing."""
        bit_state = self.rng.bit_generator.state
        return {
            "version": 1,
            "dim": self.dim,
            "population": self.population,
            "seed": self.seed,
            "sigma": self.sigma,
            "mean": self.mean.tolist(),
            "lower": [_encode_float(v) for v in self.lower],
            "upper": [_encode_float(v) for v in self.upper],
            "p_sigma": self.p_sigma.tolist(),
            "p_c": self.p_c.tolist(),
            "covariance": self.covariance.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "generations": self.generations,
            "evaluations": self.evaluations,
            "best_x": self._best_x.tolist(),
            "best_f": _encode_float(self._best_f),
            "rng": _encode_rng(bit_state),
        }

    @classmethod
    def restore(cls, state: dict) -> CMAES:
        """Rebuild an optimizer from :meth:`state`.

        Args:
            state: Mapping produced by :meth:`state`, possibly after a JSON round trip.
        """
        lower = np.array([_decode_float(v) for v in state["lower"]], dtype=np.float64)
        upper = np.array([_decode_float(v) for v in state["upper"]], dtype=np.float64)
        optimizer = cls(
            x0=np.asarray(state["mean"], dtype=np.float64),
            sigma0=float(state["sigma"]),
            bounds=(lower, upper),
            population=int(state["population"]),
            seed=int(state["seed"]),
        )
        optimizer.mean = np.asarray(state["mean"], dtype=np.float64)
        optimizer.sigma = float(state["sigma"])
        optimizer.p_sigma = np.asarray(state["p_sigma"], dtype=np.float64)
        optimizer.p_c = np.asarray(state["p_c"], dtype=np.float64)
        optimizer.covariance = np.asarray(state["covariance"], dtype=np.float64)
        optimizer.eigenvectors = np.asarray(state["eigenvectors"], dtype=np.float64)
        optimizer.eigenvalues = np.asarray(state["eigenvalues"], dtype=np.float64)
        optimizer.generations = int(state["generations"])
        optimizer.evaluations = int(state["evaluations"])
        optimizer._best_x = np.asarray(state["best_x"], dtype=np.float64)
        optimizer._best_f = _decode_float(state["best_f"])
        optimizer.rng.bit_generator.state = _decode_rng(state["rng"])
        return optimizer


def _encode_float(value: float) -> float | str:
    """Return a JSON-safe representation of a possibly infinite float."""
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def _decode_float(value: float | str) -> float:
    """Return the float encoded by :func:`_encode_float`."""
    return float(value)


def _encode_rng(bit_state: dict) -> dict:
    """Return the bit generator state with big integers encoded as strings."""
    encoded = dict(bit_state)
    encoded["state"] = {key: str(val) for key, val in bit_state["state"].items()}
    return encoded


def _decode_rng(encoded: dict) -> dict:
    """Return the bit generator state rebuilt by inverting :func:`_encode_rng`."""
    decoded = dict(encoded)
    decoded["state"] = {key: int(val) for key, val in encoded["state"].items()}
    return decoded


def minimize(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    sigma0: float,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    population: int | None = None,
    max_evaluations: int = 1000,
    seed: int = 0,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
    tolerance: float = 1e-9,
) -> Result:
    """Minimize a scalar objective with CMA-ES.

    Args:
        objective: Function mapping a point of shape ``[dim]`` to a scalar; may return
            ``inf`` or ``nan`` for a failed evaluation.
        x0: Initial distribution mean, shape ``[dim]``.
        sigma0: Initial global step size, in the units of ``x0``.
        bounds: Lower and upper box bounds, each shape ``[dim]``, or ``None``.
        population: Generation size; defaults to ``4 + floor(3 * ln(dim))``.
        max_evaluations: Budget of objective evaluations.
        seed: Seed of the internal random generator.
        callback: Called as ``callback(generation, best_x, best_f)`` after each generation.
        tolerance: Stop once the spread of finite values in a generation falls below this.
    """
    optimizer = CMAES(x0=x0, sigma0=sigma0, bounds=bounds, population=population, seed=seed)
    history: list[float] = []

    while optimizer.evaluations < max_evaluations:
        candidates = optimizer.ask()
        values = np.array([float(objective(candidate)) for candidate in candidates], dtype=np.float64)
        optimizer.tell(candidates, values)

        best_x, best_f = optimizer.best
        history.append(best_f)
        if callback is not None:
            callback(optimizer.generations, best_x, best_f)

        finite = values[np.isfinite(values)]
        spread = float(np.max(finite) - np.min(finite)) if finite.size > 1 else math.inf
        if spread < tolerance or optimizer.sigma <= _MIN_SIGMA:
            break

    best_x, best_f = optimizer.best
    return Result(
        x=best_x,
        fun=best_f,
        evaluations=optimizer.evaluations,
        generations=optimizer.generations,
        history=history,
    )
