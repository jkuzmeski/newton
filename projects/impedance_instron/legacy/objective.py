# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Score one simulated stance as three ordered tiers instead of one weighted sum.

The previous cost added nine incommensurable terms. Four of them (penetration, compression,
saturation, residual) were identically zero at every solved command, so they were feasibility
filters wearing the clothes of objectives: a large enough gain elsewhere could in principle have
bought a constraint violation. The stated objective, actuator effort, varied only 1.24x across five
solved runs while momentum varied 3.8x and dissipation 6.8x, so effort was a near-constant offset
rather than a discriminator. Worst of all, task accuracy was tradeable against preference, so runs
with different weights landed on different points of a Pareto frontier and their totals were not
comparable.

This module answers all three with a lexicographic verdict:

Tier 1, feasibility, is a binary gate. The rollout must complete, touch the ground, keep the rigid
last out of the floor, stay off the actuator force limit, release the ground at the end, and keep
peak foam compression inside its bounds. A failure returns a large base value plus the summed
normalized violations, so the search still sees a descent direction toward feasibility and can
never buy its way out with a good objective.

Tier 2, task tolerances, is a deadband, not a weight. Stance duration, vertical impulse, and the
momentum history each cost exactly zero inside a physically stated tolerance and grow outside it.
Any feasible-but-off-task rollout scores strictly worse than any on-task rollout.

Tier 3, the objective, is the single quantity minimized once the rollout is feasible and on task:
a muscle-efficiency work proxy on the leg actuator's source power.

Assumptions, stated because they are choices and not measurements:

* The work proxy is ``W_plus / positive_efficiency + abs(W_minus) / negative_efficiency`` with
  default efficiencies 0.25 and 1.20, the conventional positive and negative muscle work
  efficiencies. It is applied to a single lumped virtual leg actuator, not to muscles, so it reads
  as "how hard the leg must work" and is emphatically **not** a metabolic measurement. Both
  efficiencies are constructor arguments so a different convention can be substituted.
* The tolerances (5 ms of stance duration, 1 % of vertical impulse, 0.05 m/s momentum RMS) are
  engineering choices sized to measurement resolution, not derived quantities. All are overridable.
* The feasibility limits (1 mm of allowed last clearance loss, 2 % of body weight of residual load,
  and 1 to 45 mm of peak compression) are rig constraints of the impedance Instron, also overridable.
* Violations and excursions are reported in multiples of their own limit or tolerance, which is what
  makes summing them across different physical units defensible.

The module deliberately depends on nothing but NumPy and the standard library: no optimizer, no
Warp, no Newton. The same :class:`Objective` is meant to serve as the reinforcement-learning reward,
where the negated :attr:`Verdict.value` is the return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Body weight of the captured subject, 81.9 kg at 9.81 m/s^2. Only used to normalize force
# violations when the caller does not supply the value the rollout was produced with.
DEFAULT_BODY_WEIGHT_N = 803.47


@runtime_checkable
class TargetLike(Protocol):
    """The measured task, structurally compatible with ``projects.impedance_instron.optimize.Target``."""

    delta_vx_m_s: float
    delta_vz_m_s: float
    duration_s: float
    momentum_vx_m_s: list[float]
    momentum_vz_m_s: list[float]


@runtime_checkable
class RolloutLike(Protocol):
    """One simulated stance, structurally compatible with ``projects.impedance_instron.optimize.Rollout``."""

    completed: bool
    contact: bool
    contact_duration_s: float
    delta_vx_m_s: float
    delta_vz_m_s: float
    peak_compression_m: float
    min_last_height_m: float
    residual_load_n: float
    saturation_excess_n: float
    saturated: bool
    momentum_vx_m_s: list[float]
    momentum_vz_m_s: list[float]
    positive_work_j: float
    negative_work_j: float
    # Optional: absent on rollouts recorded before the ankle actuator was charged, read with getattr
    # and defaulted to zero so those records still score rather than becoming NaN.
    ankle_positive_work_j: float
    ankle_negative_work_j: float


@dataclass
class Tolerances:
    """Deadband half-widths of the tier 2 task test.

    The defaults are derived from the subject's OWN step-to-step variability in the same classified
    90-95 s window, not chosen for convenience. Demanding better repeatability than the measurement
    itself shows would not be defensible. From the profile provenance:

    * ``window_stance_duration_range_s`` spans 12.5 ms, so the half-spread is 6.25 ms.
    * ``window_flight_range_s`` spans 18 ms. Reading take-off speed as ``g * flight / 2`` gives
      0.250-0.338 m/s, a half-spread of 0.044 m/s.
    * Propagating both through ``impulse = m * (dv + g * T)`` gives 17.3 N s on 282.6 N s, or 3.1 %.

    Caveat on ``momentum_m_s``: the take-off speed spread bounds an ENDPOINT velocity, while the test
    applies it to the RMS of a velocity history. It is an order-of-magnitude basis, not a rigorous
    bound on the history. Tighten it if a per-checkpoint spread is ever measured across strides.
    """

    duration_s: float = 0.00625
    impulse_fraction: float = 0.031
    momentum_m_s: float = 0.044


@dataclass
class Verdict:
    """Outcome of scoring one rollout, with the reason for the score kept alongside it."""

    feasible: bool
    on_task: bool
    value: float
    objective_j: float
    violations: dict[str, float] = field(default_factory=dict)
    excursions: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-line, human-readable account of the verdict and what decided it."""
        objective = "objective n/a" if not math.isfinite(self.objective_j) else f"objective {self.objective_j:.1f} J"
        if not self.feasible:
            reasons = ", ".join(f"{name} {amount:.3g}" for name, amount in sorted(self.violations.items()))
            return f"infeasible [{reasons}] value {self.value:.4g} ({objective})"
        if not self.on_task:
            reasons = ", ".join(f"{name} {amount:.3g}x tol" for name, amount in sorted(self.excursions.items()))
            return f"off-task [{reasons}] value {self.value:.4g} ({objective})"
        return f"on-task {objective} value {self.value:.4g}"


def _bounded(magnitude: float, ceiling: float, span: float) -> float:
    """Map a non-negative magnitude into ``[0, span)``, leaving it untouched below ``ceiling``.

    A tier can only stay below the next tier's base if its own contribution is bounded, but
    compressing the whole range would destroy the property that an on-task value *is* the work
    proxy in joules. So the map is the identity up to ``ceiling`` and then approaches ``span``
    asymptotically, continuously and strictly monotonically.
    """
    if magnitude <= ceiling:
        return magnitude
    return ceiling + (span - ceiling) * (1.0 - ceiling / magnitude)


def _finite(*values: float) -> bool:
    """Return whether every supplied value is a finite real number."""
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)


class Objective:
    """Score rollouts lexicographically: feasibility, then task tolerances, then leg work.

    Args:
        target: Measured task the command must reproduce.
        tolerances: Tier 2 deadbands; defaults to :class:`Tolerances`.
        positive_efficiency: Efficiency of positive muscle work used by the tier 3 proxy.
        negative_efficiency: Efficiency of negative muscle work used by the tier 3 proxy.
        body_weight_n: Body weight [N] used to normalize force violations.
        penetration_allowance_m: Rigid-last clearance [m] that may be lost before penetration counts.
        compression_bounds_m: Allowed peak foam compression range [m] as ``(minimum, maximum)``.
        charge_ankle: Charge the ankle pitch actuator's work alongside the leg's. Set False only to
            reproduce values recorded before the ankle was charged.
        residual_fraction: Fraction of body weight of end-of-rollout load that still counts as released.
        off_task_base: Value floor [-] of tier 2; every on-task value is strictly below it.
        infeasible_base: Value floor [-] of tier 1; every feasible value is strictly below it.
        tier_ceiling: Magnitude above which a tier's own contribution starts saturating.
    """

    def __init__(
        self,
        target: TargetLike,
        tolerances: Tolerances | None = None,
        positive_efficiency: float = 0.25,
        negative_efficiency: float = 1.20,
        charge_ankle: bool = True,
        body_weight_n: float = DEFAULT_BODY_WEIGHT_N,
        penetration_allowance_m: float = 0.001,
        compression_bounds_m: tuple[float, float] = (0.001, 0.045),
        residual_fraction: float = 0.02,
        off_task_base: float = 1.0e6,
        infeasible_base: float = 1.0e9,
        tier_ceiling: float = 1.0e5,
    ):
        if positive_efficiency <= 0.0 or negative_efficiency <= 0.0:
            raise ValueError("Muscle work efficiencies must be positive")
        self.charge_ankle = bool(charge_ankle)
        if not off_task_base < infeasible_base:
            raise ValueError("The infeasible base must sit above the off-task base")
        if tier_ceiling >= off_task_base:
            raise ValueError("The tier ceiling must leave room below the off-task base")
        self.target = target
        self.tolerances = tolerances or Tolerances()
        self.positive_efficiency = float(positive_efficiency)
        self.negative_efficiency = float(negative_efficiency)
        self.body_weight_n = float(body_weight_n)
        self.penetration_allowance_m = float(penetration_allowance_m)
        self.compression_bounds_m = (float(compression_bounds_m[0]), float(compression_bounds_m[1]))
        self.residual_fraction = float(residual_fraction)
        self.off_task_base = float(off_task_base)
        self.infeasible_base = float(infeasible_base)
        self.tier_ceiling = float(tier_ceiling)

    def _charge(self, positive: float, negative: float) -> float:
        """Charge one actuator's positive and negative work at their separate efficiencies."""
        if not _finite(positive, negative):
            return float("nan")
        return abs(float(positive)) / self.positive_efficiency + abs(float(negative)) / self.negative_efficiency

    def leg_work_proxy_j(self, rollout: RolloutLike) -> float:
        """Return the work proxy [J] of the axial leg actuator alone."""
        return self._charge(rollout.positive_work_j, rollout.negative_work_j)

    def ankle_work_proxy_j(self, rollout: RolloutLike) -> float:
        """Return the work proxy [J] of the ankle pitch actuator alone, zero when it is not reported."""
        positive = getattr(rollout, "ankle_positive_work_j", 0.0)
        negative = getattr(rollout, "ankle_negative_work_j", 0.0)
        if positive is None or negative is None:
            return 0.0
        return self._charge(positive, negative)

    def work_proxy_j(self, rollout: RolloutLike) -> float:
        """Return the tier 3 muscle-efficiency work proxy [J] of BOTH rig actuators.

        Positive and negative actuator work are charged at different efficiencies because a leg pays
        much less for absorbing energy than for producing it. The result is a work demand on lumped
        virtual actuators; it is not a metabolic rate and must not be reported as one.

        The ankle pitch actuator is charged alongside the leg. Leaving it free is not a neutral
        simplification: an uncharged actuator is a resource the search will spend without limit, the
        same failure that let virtual damper dissipation grow 4.8x above the analytic seed while the
        cost still fell. It applies to prescribed pitch too, where the replay motor does whatever
        work the trajectory demands and was historically never charged.

        Consequence for comparisons: work proxies recorded before the ankle was charged are LEG ONLY
        and are not comparable with values from this method. :meth:`leg_work_proxy_j` reproduces the
        old quantity when continuity with those numbers is needed.
        """
        leg = self.leg_work_proxy_j(rollout)
        if not self.charge_ankle:
            return leg
        ankle = self.ankle_work_proxy_j(rollout)
        if not _finite(leg, ankle):
            return float("nan")
        return leg + ankle

    def violations(self, rollout: RolloutLike) -> dict[str, float]:
        """Return the tier 1 violations, each normalized by its own limit, empty when feasible."""
        found: dict[str, float] = {}
        scalars = (
            rollout.contact_duration_s,
            rollout.delta_vx_m_s,
            rollout.delta_vz_m_s,
            rollout.peak_compression_m,
            rollout.min_last_height_m,
            rollout.residual_load_n,
            rollout.saturation_excess_n,
        )
        histories = list(rollout.momentum_vx_m_s) + list(rollout.momentum_vz_m_s)
        if not _finite(*scalars, *histories):
            found["finite"] = 1.0
        if not rollout.completed:
            found["completed"] = 1.0
        if not rollout.contact:
            found["contact"] = 1.0
        if "finite" in found:
            # Comparing a non-finite depth or load would only produce a non-finite violation, and
            # the gate has already been failed by the rollout being unusable.
            return found
        depth = -float(rollout.min_last_height_m) - self.penetration_allowance_m
        if depth > 0.0:
            found["penetration"] = depth / self.penetration_allowance_m
        excess = max(0.0, float(rollout.saturation_excess_n))
        if rollout.saturated or excess > 0.0:
            found["saturation"] = excess / self.body_weight_n
        residual = float(rollout.residual_load_n) - self.residual_fraction * self.body_weight_n
        if residual > 0.0:
            found["residual"] = residual / self.body_weight_n
        low, high = self.compression_bounds_m
        compression = float(rollout.peak_compression_m)
        if compression > high:
            found["compression"] = (compression - high) / high
        elif compression < low:
            found["compression"] = (low - compression) / high
        return found

    def excursions(self, rollout: RolloutLike) -> dict[str, float]:
        """Return the tier 2 excursions in multiples of their tolerance, empty when on task.

        Vertical impulse is compared through the stance velocity change it produces, which is the
        impulse divided by the constant body mass, so a fractional tolerance is the same test.
        The momentum excursion is the RMS over both axes of all checkpoints, because fore-aft
        momentum is part of the task even though only vertical impulse has its own tolerance.
        """
        found: dict[str, float] = {}
        duration = abs(float(rollout.contact_duration_s) - float(self.target.duration_s))
        if duration > self.tolerances.duration_s:
            found["duration"] = (duration - self.tolerances.duration_s) / self.tolerances.duration_s
        reference = abs(float(self.target.delta_vz_m_s))
        impulse = abs(float(rollout.delta_vz_m_s) - float(self.target.delta_vz_m_s)) / max(reference, 1.0e-9)
        if impulse > self.tolerances.impulse_fraction:
            found["impulse"] = (impulse - self.tolerances.impulse_fraction) / self.tolerances.impulse_fraction
        error = np.concatenate(
            [
                np.asarray(rollout.momentum_vx_m_s, dtype=float) - np.asarray(self.target.momentum_vx_m_s, dtype=float),
                np.asarray(rollout.momentum_vz_m_s, dtype=float) - np.asarray(self.target.momentum_vz_m_s, dtype=float),
            ]
        )
        momentum = float(np.sqrt(np.mean(np.square(error))))
        if momentum > self.tolerances.momentum_m_s:
            found["momentum"] = (momentum - self.tolerances.momentum_m_s) / self.tolerances.momentum_m_s
        return found

    def evaluate(self, rollout: RolloutLike) -> Verdict:
        """Score one rollout and return the tier that decided it.

        The three tiers occupy disjoint value ranges, so no amount of tier 3 excellence can offset a
        tier 2 miss and no amount of tier 2 accuracy can offset a tier 1 violation.
        """
        objective = self.work_proxy_j(rollout)
        violations = self.violations(rollout)
        if violations:
            return Verdict(False, False, self.infeasible_base + sum(violations.values()), objective, violations, {})
        excursions = self.excursions(rollout)
        if excursions:
            span = self.infeasible_base - self.off_task_base
            value = self.off_task_base + _bounded(sum(excursions.values()), self.tier_ceiling, span)
            return Verdict(True, False, value, objective, {}, excursions)
        if not math.isfinite(objective):
            # Reachable only if the work fields are absent while every graded field is finite.
            return Verdict(False, False, self.infeasible_base + 1.0, objective, {"finite": 1.0}, {})
        return Verdict(True, True, _bounded(objective, self.tier_ceiling, self.off_task_base), objective, {}, {})

    def describe(self) -> dict[str, float | dict[str, float]]:
        """Return the tolerances, efficiencies, and limits so a solved command can record them."""
        return {
            "tolerance_duration_s": self.tolerances.duration_s,
            "tolerance_impulse_fraction": self.tolerances.impulse_fraction,
            "tolerance_momentum_m_s": self.tolerances.momentum_m_s,
            "positive_efficiency": self.positive_efficiency,
            "negative_efficiency": self.negative_efficiency,
            "body_weight_n": self.body_weight_n,
            "penetration_allowance_m": self.penetration_allowance_m,
            "compression_bounds_m": list(self.compression_bounds_m),
            "residual_fraction": self.residual_fraction,
            "off_task_base": self.off_task_base,
            "infeasible_base": self.infeasible_base,
        }
