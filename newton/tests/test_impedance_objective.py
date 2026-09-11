# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the three-tier stance objective in projects/impedance_instron.

The objective replaces a nine-term weighted sum, so the properties worth testing are structural:
the tiers must not interleave, the deadbands must cost exactly nothing inside their tolerance, and
no violation may ever be paid for with a better objective. Everything here is pure Python: rollouts
are built directly, so the suite needs neither Warp nor a GPU.
"""

import ast
import math
import unittest
from dataclasses import dataclass, field, fields, replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from projects.impedance_instron.objective import (
    DEFAULT_BODY_WEIGHT_N,
    Objective,
    RolloutLike,
    TargetLike,
    Tolerances,
    Verdict,
)

CHECKPOINTS = 9


@dataclass
class FakeTarget:
    """Stand-in for the measured task with the same fields the objective reads."""

    delta_vx_m_s: float = 0.0496
    delta_vz_m_s: float = 0.5561
    duration_s: float = 0.295
    momentum_vx_m_s: list[float] = field(default_factory=lambda: [0.01 * i for i in range(CHECKPOINTS)])
    momentum_vz_m_s: list[float] = field(default_factory=lambda: [0.1 * i for i in range(CHECKPOINTS)])


@dataclass
class FakeRollout:
    """Stand-in for one simulated stance, feasible and on task unless a field is overridden."""

    completed: bool = True
    contact: bool = True
    contact_duration_s: float = 0.295
    delta_vx_m_s: float = 0.0496
    delta_vz_m_s: float = 0.5561
    peak_compression_m: float = 0.022
    min_last_height_m: float = 0.003
    residual_load_n: float = 0.0
    saturation_excess_n: float = 0.0
    saturated: bool = False
    momentum_vx_m_s: list[float] = field(default_factory=lambda: [0.01 * i for i in range(CHECKPOINTS)])
    momentum_vz_m_s: list[float] = field(default_factory=lambda: [0.1 * i for i in range(CHECKPOINTS)])
    positive_work_j: float = 100.0
    negative_work_j: float = -60.0


#: Tolerances pinned for the behaviour tests. They are deliberately NOT the shipped defaults: these
#: tests check the deadband arithmetic, which must stay valid when the shipped policy changes.
TEST_TOLERANCES = Tolerances(duration_s=0.005, impulse_fraction=0.01, momentum_m_s=0.05)


def _objective(**kwargs) -> Objective:
    """Return an objective against the default fake target, with test-pinned tolerances."""
    kwargs.setdefault("tolerances", TEST_TOLERANCES)
    return Objective(FakeTarget(), **kwargs)


def _shift(values: list[float], offset: float) -> list[float]:
    """Return a momentum history displaced by a constant offset."""
    return [value + offset for value in values]


class TestTierSeparation(unittest.TestCase):
    """Lexicographic ordering of the three tiers."""

    def test_infeasible_always_worse_than_feasible(self):
        """Verify every infeasible rollout scores above every feasible one across a random sweep.

        The feasible samples are given extreme leg work and large task errors, and the infeasible
        samples are given a perfect task and no work at all, so the sweep would catch any leakage of
        tier 2 or tier 3 magnitude into the feasibility gate.
        """
        score = _objective()
        rng = np.random.default_rng(0)
        feasible, infeasible = [], []
        for _ in range(300):
            verdict = score.evaluate(
                replace(
                    FakeRollout(),
                    contact_duration_s=0.295 + float(rng.uniform(-0.2, 0.2)),
                    delta_vz_m_s=0.5561 * float(rng.uniform(0.2, 3.0)),
                    momentum_vz_m_s=_shift(FakeTarget().momentum_vz_m_s, float(rng.uniform(-2.0, 2.0))),
                    positive_work_j=float(rng.uniform(0.0, 1.0e7)),
                    negative_work_j=-float(rng.uniform(0.0, 1.0e7)),
                )
            )
            self.assertTrue(verdict.feasible)
            feasible.append(verdict.value)
            broken = replace(
                FakeRollout(),
                completed=bool(rng.integers(0, 2)),
                contact=bool(rng.integers(0, 2)),
                min_last_height_m=float(rng.uniform(-0.05, -0.002)),
                residual_load_n=float(rng.uniform(20.0, 500.0)),
                saturation_excess_n=float(rng.uniform(1.0, 500.0)),
                saturated=True,
                peak_compression_m=float(rng.uniform(0.046, 0.2)),
                positive_work_j=0.0,
                negative_work_j=0.0,
            )
            verdict = score.evaluate(broken)
            self.assertFalse(verdict.feasible)
            infeasible.append(verdict.value)
        self.assertLess(max(feasible), min(infeasible))

    def test_off_task_always_worse_than_on_task(self):
        """Verify every feasible-but-off-task rollout scores above every on-task one."""
        score = _objective()
        rng = np.random.default_rng(1)
        on_task, off_task = [], []
        for _ in range(300):
            verdict = score.evaluate(
                replace(
                    FakeRollout(),
                    positive_work_j=float(rng.uniform(0.0, 1.0e8)),
                    negative_work_j=-float(rng.uniform(0.0, 1.0e8)),
                )
            )
            self.assertTrue(verdict.on_task)
            on_task.append(verdict.value)
            # The smallest possible miss: just outside one tolerance with no leg work at all.
            verdict = score.evaluate(
                replace(
                    FakeRollout(),
                    contact_duration_s=0.295 + 0.005 + float(rng.uniform(1.0e-9, 1.0e-6)),
                    positive_work_j=0.0,
                    negative_work_j=0.0,
                )
            )
            self.assertTrue(verdict.feasible)
            self.assertFalse(verdict.on_task)
            off_task.append(verdict.value)
        self.assertLess(max(on_task), min(off_task))

    def test_bases_bound_each_tier(self):
        """Verify each tier stays inside its own value band."""
        score = _objective()
        self.assertLess(score.evaluate(FakeRollout()).value, score.off_task_base)
        off_task = score.evaluate(replace(FakeRollout(), contact_duration_s=1.0)).value
        self.assertTrue(score.off_task_base <= off_task < score.infeasible_base)
        self.assertGreaterEqual(score.evaluate(replace(FakeRollout(), contact=False)).value, score.infeasible_base)


class TestDeadband(unittest.TestCase):
    """Tier 2 behaviour inside and outside its tolerances."""

    def test_inside_tolerance_costs_nothing(self):
        """Verify a rollout inside every tolerance reports no excursion and scores the objective."""
        score = _objective()
        target = FakeTarget()
        rollout = replace(
            FakeRollout(),
            contact_duration_s=target.duration_s + 0.0049,
            delta_vz_m_s=target.delta_vz_m_s * 1.009,
            momentum_vz_m_s=_shift(target.momentum_vz_m_s, 0.04),
        )
        verdict = score.evaluate(rollout)
        self.assertEqual(verdict.excursions, {})
        self.assertTrue(verdict.on_task)
        self.assertEqual(verdict.value, verdict.objective_j)
        self.assertEqual(verdict.value, score.evaluate(FakeRollout()).value)

    def test_tolerances_are_overridable(self):
        """Verify a tightened tolerance turns a previously on-task rollout off task."""
        rollout = replace(FakeRollout(), contact_duration_s=0.295 + 0.004)
        self.assertTrue(_objective().evaluate(rollout).on_task)
        tight = _objective(tolerances=Tolerances(duration_s=0.001))
        verdict = tight.evaluate(rollout)
        self.assertFalse(verdict.on_task)
        self.assertAlmostEqual(verdict.excursions["duration"], 3.0, places=9)

    def test_excursion_keys_and_magnitudes(self):
        """Verify excursions name the missed tolerances and report multiples of each tolerance."""
        score = _objective()
        target = FakeTarget()
        rollout = replace(
            FakeRollout(),
            contact_duration_s=target.duration_s + 0.010,
            delta_vz_m_s=target.delta_vz_m_s * 1.03,
            momentum_vz_m_s=_shift(target.momentum_vz_m_s, 0.15),
            momentum_vx_m_s=_shift(target.momentum_vx_m_s, 0.15),
        )
        verdict = score.evaluate(rollout)
        self.assertEqual(set(verdict.excursions), {"duration", "impulse", "momentum"})
        self.assertAlmostEqual(verdict.excursions["duration"], 1.0, places=6)
        self.assertAlmostEqual(verdict.excursions["impulse"], 2.0, places=6)
        self.assertAlmostEqual(verdict.excursions["momentum"], 2.0, places=6)
        self.assertAlmostEqual(verdict.value - score.off_task_base, 5.0, places=6)

    def test_momentum_excursion_covers_both_axes(self):
        """Verify a fore-aft-only momentum error is still an excursion."""
        target = FakeTarget()
        rollout = replace(FakeRollout(), momentum_vx_m_s=_shift(target.momentum_vx_m_s, 0.4))
        verdict = _objective().evaluate(rollout)
        self.assertEqual(set(verdict.excursions), {"momentum"})
        # Half of the eighteen checkpoints are displaced, so the RMS is 0.4 / sqrt(2).
        self.assertAlmostEqual(verdict.excursions["momentum"], (0.4 / math.sqrt(2.0) - 0.05) / 0.05, places=6)


class TestShippedTolerances(unittest.TestCase):
    """The shipped defaults must stay tied to the measured step-to-step variability."""

    def test_defaults_match_the_subject_step_variability(self):
        """Pin each shipped tolerance to the measurement it is derived from.

        The defaults are not free parameters. They come from the subject's own spread in the same
        classified window: stance duration ranges over 12.5 ms, flight time over 18 ms. Reading
        take-off speed as g*flight/2 gives a 0.044 m/s half-spread, and propagating both through
        impulse = m*(dv + g*T) on 282.6 N s gives 3.1 %. A tolerance tighter than this would demand
        better repeatability from the controller than the measurement itself demonstrates.
        """
        gravity, mass, impulse_n_s = 9.80665, 81.93121179999996, 282.5724029282729
        duration_spread_s, flight_low_s, flight_high_s = 0.0125, 0.051, 0.069
        velocity_half = 0.5 * (gravity * flight_high_s / 2.0 - gravity * flight_low_s / 2.0)
        impulse_half = 0.5 * (mass * 2.0 * velocity_half + mass * gravity * duration_spread_s)
        shipped = Tolerances()
        self.assertAlmostEqual(shipped.duration_s, duration_spread_s / 2.0, places=6)
        self.assertAlmostEqual(shipped.momentum_m_s, velocity_half, places=3)
        self.assertAlmostEqual(shipped.impulse_fraction, impulse_half / impulse_n_s, places=3)


class TestFeasibility(unittest.TestCase):
    """Tier 1 gate contents."""

    def test_violation_keys_and_magnitudes(self):
        """Verify violations name the broken constraints and report them normalized by their limit."""
        score = _objective(body_weight_n=DEFAULT_BODY_WEIGHT_N)
        rollout = replace(
            FakeRollout(),
            min_last_height_m=-0.003,
            residual_load_n=0.05 * DEFAULT_BODY_WEIGHT_N,
            saturation_excess_n=0.1 * DEFAULT_BODY_WEIGHT_N,
            saturated=True,
            peak_compression_m=0.045 + 0.0045,
        )
        verdict = score.evaluate(rollout)
        self.assertFalse(verdict.feasible)
        self.assertEqual(verdict.excursions, {})
        self.assertEqual(set(verdict.violations), {"penetration", "residual", "saturation", "compression"})
        self.assertAlmostEqual(verdict.violations["penetration"], 2.0, places=6)
        self.assertAlmostEqual(verdict.violations["residual"], 0.03, places=6)
        self.assertAlmostEqual(verdict.violations["saturation"], 0.1, places=6)
        self.assertAlmostEqual(verdict.violations["compression"], 0.1, places=6)
        self.assertAlmostEqual(verdict.value - score.infeasible_base, 2.23, places=6)

    def test_incomplete_and_contactless_rollouts_are_infeasible(self):
        """Verify a rollout that did not finish or never touched the ground fails the gate."""
        score = _objective()
        self.assertEqual(score.evaluate(replace(FakeRollout(), completed=False)).violations, {"completed": 1.0})
        self.assertEqual(score.evaluate(replace(FakeRollout(), contact=False)).violations, {"contact": 1.0})

    def test_compression_below_bound_is_infeasible(self):
        """Verify a stance that barely compresses the foam fails the compression bound."""
        verdict = _objective().evaluate(replace(FakeRollout(), peak_compression_m=0.0))
        self.assertEqual(set(verdict.violations), {"compression"})
        self.assertAlmostEqual(verdict.violations["compression"], 0.001 / 0.045, places=9)

    def test_feasibility_ignores_the_objective(self):
        """Verify an enormous objective never removes and never adds a violation."""
        score = _objective()
        broken = replace(FakeRollout(), min_last_height_m=-0.01)
        cheap = score.evaluate(replace(broken, positive_work_j=0.0, negative_work_j=0.0))
        costly = score.evaluate(replace(broken, positive_work_j=1.0e9, negative_work_j=-1.0e9))
        self.assertEqual(cheap.value, costly.value)
        self.assertEqual(cheap.violations, costly.violations)

    def test_non_finite_rollout_is_infeasible_and_finite_valued(self):
        """Verify any NaN or infinite rollout field yields an infeasible, finite-valued verdict."""
        score = _objective()
        for name in (
            "contact_duration_s",
            "delta_vx_m_s",
            "delta_vz_m_s",
            "peak_compression_m",
            "min_last_height_m",
            "residual_load_n",
            "saturation_excess_n",
        ):
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=name, value=bad):
                    verdict = score.evaluate(replace(FakeRollout(), **{name: bad}))
                    self.assertFalse(verdict.feasible)
                    self.assertIn("finite", verdict.violations)
                    self.assertTrue(math.isfinite(verdict.value))
        broken = replace(FakeRollout(), momentum_vz_m_s=[float("nan")] * CHECKPOINTS)
        self.assertFalse(score.evaluate(broken).feasible)
        self.assertTrue(math.isfinite(score.evaluate(broken).value))

    def test_missing_work_is_infeasible_rather_than_nan(self):
        """Verify a rollout without recorded actuator work cannot be scored as on task."""
        score = _objective()
        verdict = score.evaluate(replace(FakeRollout(), positive_work_j=float("nan")))
        self.assertFalse(verdict.feasible)
        self.assertTrue(math.isnan(verdict.objective_j))
        self.assertTrue(math.isfinite(verdict.value))


class TestWorkProxy(unittest.TestCase):
    """Tier 3 muscle-efficiency work proxy."""

    def test_proxy_matches_hand_computed_value(self):
        """Verify 100 J positive and 60 J negative work cost 100/0.25 + 60/1.20 joules."""
        score = _objective()
        rollout = replace(FakeRollout(), positive_work_j=100.0, negative_work_j=-60.0)
        self.assertAlmostEqual(score.work_proxy_j(rollout), 450.0, places=9)
        self.assertAlmostEqual(score.evaluate(rollout).objective_j, 450.0, places=9)
        self.assertAlmostEqual(score.evaluate(rollout).value, 450.0, places=9)

    def test_efficiencies_are_respected(self):
        """Verify changed efficiencies rescale the two halves of the work independently."""
        rollout = replace(FakeRollout(), positive_work_j=100.0, negative_work_j=-60.0)
        self.assertAlmostEqual(
            _objective(positive_efficiency=0.5, negative_efficiency=2.0).work_proxy_j(rollout), 230.0, places=9
        )
        self.assertAlmostEqual(
            _objective(positive_efficiency=0.25, negative_efficiency=0.25).work_proxy_j(rollout), 640.0, places=9
        )

    def test_negative_work_is_cheaper_than_positive(self):
        """Verify absorbing a joule costs less than producing one at the default efficiencies."""
        score = _objective()
        producing = score.work_proxy_j(replace(FakeRollout(), positive_work_j=100.0, negative_work_j=0.0))
        absorbing = score.work_proxy_j(replace(FakeRollout(), positive_work_j=0.0, negative_work_j=-100.0))
        self.assertLess(absorbing, producing)

    def test_non_positive_efficiency_is_rejected(self):
        """Verify a zero or negative efficiency is refused at construction."""
        with self.assertRaises(ValueError):
            _objective(positive_efficiency=0.0)
        with self.assertRaises(ValueError):
            _objective(negative_efficiency=-1.0)

    def test_proxy_grows_with_work(self):
        """Verify more actuator work never scores better."""
        score = _objective()
        values = [
            score.evaluate(replace(FakeRollout(), positive_work_j=work, negative_work_j=-0.5 * work)).value
            for work in (0.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e9)
        ]
        self.assertTrue(all(a < b for a, b in pairwise(values)))


class TestMonotonicity(unittest.TestCase):
    """Worsening any single term must never improve the score."""

    def _sweep(self, name, values):
        """Return the scores of one field swept through the supplied values."""
        score = _objective()
        return [score.evaluate(replace(FakeRollout(), **{name: value})).value for value in values]

    def test_worsening_one_term_never_lowers_the_value(self):
        """Verify each violation and excursion knob is non-decreasing as it worsens."""
        sweeps = {
            "min_last_height_m": [0.01, 0.005, 0.0, -0.001, -0.005, -0.02, -0.1],
            "residual_load_n": [0.0, 5.0, 16.0, 20.0, 100.0, 800.0],
            "saturation_excess_n": [0.0, 1.0, 10.0, 100.0, 800.0],
            "peak_compression_m": [0.02, 0.045, 0.05, 0.1, 0.5],
            "contact_duration_s": [0.295, 0.297, 0.30, 0.32, 0.4, 1.0],
        }
        for name, values in sweeps.items():
            with self.subTest(term=name):
                scores = self._sweep(name, values)
                self.assertTrue(all(a <= b for a, b in pairwise(scores)), scores)

    def test_momentum_error_is_monotone(self):
        """Verify a growing momentum error never lowers the score."""
        score = _objective()
        target = FakeTarget()
        scores = [
            score.evaluate(replace(FakeRollout(), momentum_vz_m_s=_shift(target.momentum_vz_m_s, offset))).value
            for offset in (0.0, 0.02, 0.05, 0.1, 0.5, 2.0, 10.0)
        ]
        self.assertTrue(all(a <= b for a, b in pairwise(scores)), scores)

    def test_impulse_error_is_monotone_on_both_sides(self):
        """Verify overshooting and undershooting the impulse both cost, growing with the error."""
        score = _objective()
        target = FakeTarget()
        for direction in (1.0, -1.0):
            with self.subTest(direction=direction):
                scores = [
                    score.evaluate(
                        replace(FakeRollout(), delta_vz_m_s=target.delta_vz_m_s * (1.0 + direction * f))
                    ).value
                    for f in (0.0, 0.005, 0.01, 0.05, 0.2, 0.9)
                ]
                self.assertTrue(all(a <= b for a, b in pairwise(scores)), scores)


class TestVerdictReporting(unittest.TestCase):
    """The verdict must explain itself."""

    def test_summary_names_the_deciding_tier(self):
        """Verify the summary reports feasibility, task, and objective in plain text."""
        score = _objective()
        self.assertIn("on-task", score.evaluate(FakeRollout()).summary())
        self.assertIn("450", score.evaluate(FakeRollout()).summary())
        off_task = score.evaluate(replace(FakeRollout(), contact_duration_s=0.4)).summary()
        self.assertIn("off-task", off_task)
        self.assertIn("duration", off_task)
        broken = score.evaluate(replace(FakeRollout(), min_last_height_m=-0.01)).summary()
        self.assertIn("infeasible", broken)
        self.assertIn("penetration", broken)

    def test_verdict_defaults_are_empty_dictionaries(self):
        """Verify a verdict can be built without supplying the two report dictionaries."""
        verdict = Verdict(True, True, 1.0, 1.0)
        self.assertEqual((verdict.violations, verdict.excursions), ({}, {}))

    def test_describe_records_the_choices(self):
        """Verify the objective reports its tolerances and efficiencies for the command file."""
        recorded = _objective(tolerances=Tolerances(0.004, 0.02, 0.06), positive_efficiency=0.3).describe()
        self.assertEqual(recorded["tolerance_duration_s"], 0.004)
        self.assertEqual(recorded["tolerance_impulse_fraction"], 0.02)
        self.assertEqual(recorded["tolerance_momentum_m_s"], 0.06)
        self.assertEqual(recorded["positive_efficiency"], 0.3)
        self.assertEqual(recorded["negative_efficiency"], 1.2)


class TestOptimizerCompatibility(unittest.TestCase):
    """The optimizer dataclasses must keep satisfying the objective's protocols."""

    def _dataclass_fields(self, name):
        """Return the field names of one dataclass in optimize.py, parsed without importing Warp."""
        source = Path(__file__).resolve().parents[2] / "projects" / "impedance_instron" / "optimize.py"
        tree = ast.parse(source.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
        raise AssertionError(f"optimize.py does not define {name}")

    def test_rollout_supplies_every_field_the_objective_reads(self):
        """Verify Rollout still carries every attribute the objective needs, including the work split."""
        available = self._dataclass_fields("Rollout")
        required = set(RolloutLike.__annotations__)
        self.assertLessEqual(required, available)
        self.assertIn("positive_work_j", available)
        self.assertIn("negative_work_j", available)

    def test_target_supplies_every_field_the_objective_reads(self):
        """Verify Target still carries every attribute the objective needs."""
        self.assertLessEqual(set(TargetLike.__annotations__), self._dataclass_fields("Target"))

    def test_fake_rollout_matches_the_protocol(self):
        """Verify the test stand-ins cover exactly the protocol attributes."""
        self.assertEqual({f.name for f in fields(FakeRollout)}, set(RolloutLike.__annotations__))
        self.assertEqual({f.name for f in fields(FakeTarget)}, set(TargetLike.__annotations__))


if __name__ == "__main__":
    unittest.main()
