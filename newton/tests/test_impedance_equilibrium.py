# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the equilibrium-point leg kernel of the impedance rig against its own physics claims."""

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.impedance_instron.example import (
    LEG_DIAGNOSTIC_COUNT,
    Example,
    _apply_equilibrium_leg,
    create_parser,
)

# Reference columns the equilibrium controller reads; see ``Example._make_equilibrium_command``.
_EQUILIBRIUM, _EQUILIBRIUM_RATE = 13, 14
_MEASURED_FZ, _MEASURED_FX = 11, 22
_OTHER_FZ, _OTHER_FX = 12, 23
_STIFFNESS, _DAMPING, _STIFFNESS_RATE = 25, 26, 27
_COLUMNS = 28

# One binary-exact configuration: length 1.0 m along +Z, rate -0.5 m/s, equilibrium error
# -0.03125 m, slip -0.375 m/s, so the unclamped law asks for exactly 567 N of push.
_NOMINAL = {_EQUILIBRIUM: 1.03125, _EQUILIBRIUM_RATE: -0.125, _STIFFNESS: 12000.0, _DAMPING: 512.0}
_P0, _P1 = (0.125, 0.0, 0.25), (0.125, 0.0, 1.25)
_V0, _V1 = (0.0, 0.0, 0.25), (0.0, 0.0, -0.25)

_PROFILE = Path("outputs/impedance_instron/stance_planar_context.json")
_ARTIFACT = Path("DigitalInstron/digital_shoe_showcase/digital_shoe.json")


def _launch(
    columns: dict[int, float],
    p0: tuple[float, float, float] = _P0,
    p1: tuple[float, float, float] = _P1,
    v0: tuple[float, float, float] = _V0,
    v1: tuple[float, float, float] = _V1,
    force_limit: float = 1.0e5,
    unilateral: int = 0,
    seed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Launch one equilibrium leg step on the CPU and return the body wrenches and diagnostics.

    Args:
        columns: Reference-column values written into the single sample row.
        p0: World position of body 0, the ankle fixture [m].
        p1: World position of body 1, the upper mass [m].
        v0: Linear velocity stored in the top half of ``body_qd[0]`` [m/s].
        v1: Linear velocity stored in the top half of ``body_qd[1]`` [m/s].
        force_limit: Signed axial actuator limit [N].
        unilateral: Nonzero restricts the leg to pushing.
        seed: Optional pre-existing ``body_f`` content [N, N·m], shape [2, 6].
    """
    reference = np.zeros((1, _COLUMNS), dtype=np.float32)
    for column, value in columns.items():
        reference[0, column] = value
    pose = np.zeros((2, 7), dtype=np.float32)
    pose[0, :3], pose[1, :3], pose[:, 6] = p0, p1, 1.0
    velocity = np.zeros((2, 6), dtype=np.float32)
    velocity[0, :3], velocity[1, :3] = v0, v1
    body_f = np.zeros((2, 6), dtype=np.float32) if seed is None else np.asarray(seed, dtype=np.float32).copy()
    force = wp.array(body_f, dtype=wp.spatial_vector, device="cpu")
    diagnostics = wp.zeros(LEG_DIAGNOSTIC_COUNT, dtype=wp.float32, device="cpu")
    wp.launch(
        _apply_equilibrium_leg,
        dim=1,
        inputs=[
            wp.zeros(1, dtype=wp.int32, device="cpu"),
            wp.array(reference, dtype=wp.float32, device="cpu"),
            force_limit,
            unilateral,
            wp.array(pose, dtype=wp.transform, device="cpu"),
            wp.array(velocity, dtype=wp.spatial_vector, device="cpu"),
            force,
            diagnostics,
        ],
        device="cpu",
    )
    return force.numpy(), diagnostics.numpy()


def _ledger(columns: dict[int, float], diagnostics) -> float:
    """Rebuild source power from first principles as P_body + dE/dt + D.

    This is deliberately independent of the kernel's own algebra: it uses only the recorded
    kinematics and the commanded impedance, so it catches an error in the kernel's closed form
    instead of restating it.
    """
    k = columns.get(_STIFFNESS, 0.0)
    b = columns.get(_DAMPING, 0.0)
    stiffness_rate = columns.get(_STIFFNESS_RATE, 0.0)
    equilibrium_rate = columns.get(_EQUILIBRIUM_RATE, 0.0)
    force, length, rate = float(diagnostics[0]), float(diagnostics[5]), float(diagnostics[6])
    error = length - columns.get(_EQUILIBRIUM, 0.0)
    slip = rate - equilibrium_rate
    stored_rate = 0.5 * stiffness_rate * error * error + k * error * slip
    return force * rate + stored_rate + b * slip * slip


class TestEquilibriumLegKernel(unittest.TestCase):
    """Exercise ``_apply_equilibrium_leg`` directly with hand-built two-body states.

    The nominal configuration is exact in binary32, so the expected values below are
    the arithmetic of the impedance law itself rather than a reimplementation of it.
    """

    def test_force_follows_the_impedance_law(self):
        """Reproduce -k*(L - L0) - b*(Ldot - L0dot) for a hand-computed configuration."""
        wrench, diagnostics = _launch(_NOMINAL)
        error, slip = -0.03125, -0.375
        expected = -12000.0 * error - 512.0 * slip
        self.assertAlmostEqual(float(diagnostics[0]), expected, delta=1.0e-3)
        self.assertAlmostEqual(float(diagnostics[5]), 1.0, delta=1.0e-6)
        self.assertAlmostEqual(float(diagnostics[6]), -0.5, delta=1.0e-6)
        self.assertAlmostEqual(float(diagnostics[2]), -512.0 * slip * slip, delta=1.0e-3)
        self.assertAlmostEqual(float(diagnostics[3]), 0.5 * 12000.0 * error * error, delta=1.0e-3)
        self.assertEqual(float(diagnostics[4]), 0.0)
        np.testing.assert_allclose(wrench[1, :3], [0.0, 0.0, expected], atol=1.0e-3)

    def test_leg_force_is_equal_and_opposite(self):
        """Split the leg force into exact negatives and add the opposite foot only to body 1."""
        columns = _NOMINAL | {_OTHER_FZ: 137.0, _OTHER_FX: -41.0}
        seed = np.arange(12, dtype=np.float32).reshape(2, 6)
        wrench, diagnostics = _launch(columns, seed=seed)
        applied = wrench - seed
        other = np.array([-41.0, 0.0, 137.0], np.float32)
        np.testing.assert_array_equal(applied[0, :3], -(applied[1, :3] - other))
        np.testing.assert_array_equal(applied[0, :3], [0.0, 0.0, -diagnostics[0]])
        np.testing.assert_array_equal(applied[1, :3], [-41.0, 0.0, diagnostics[0] + 137.0])
        np.testing.assert_array_equal(applied[:, 3:], np.zeros((2, 3), np.float32))

    def test_unilateral_leg_releases_instead_of_pulling(self):
        """Zero the force when a shorter equilibrium asks the leg to pull the shoe back down.

        This unilateral release replaces the old hand-tuned unload and engage schedule:
        stance now ends when the commanded equilibrium stops loading the leg rather than
        on a clock, so the pulling branch must vanish exactly instead of fading out.

        Release and force-limit saturation are reported on SEPARATE channels. A search that
        penalizes saturation must not be penalized for letting go, so the saturation flag stays
        clear here and a dedicated release flag is raised instead. Virtual spring energy is still
        reported while no force acts; that is controller-internal bookkeeping, not shoe energy.
        """
        columns = _NOMINAL | {_EQUILIBRIUM: 0.96875, _EQUILIBRIUM_RATE: -0.5}
        held, held_diagnostics = _launch(columns, unilateral=1)
        pulled, pulled_diagnostics = _launch(columns, unilateral=0)
        self.assertEqual(float(held_diagnostics[0]), 0.0)
        np.testing.assert_array_equal(held[:, :3], np.zeros((2, 3), np.float32))
        self.assertAlmostEqual(float(pulled_diagnostics[0]), -12000.0 * 0.03125, delta=1.0e-3)
        self.assertLess(float(pulled_diagnostics[0]), 0.0)
        np.testing.assert_allclose(pulled[1, :3], [0.0, 0.0, pulled_diagnostics[0]], atol=1.0e-3)
        self.assertEqual(float(held_diagnostics[4]), 0.0)
        self.assertEqual(float(held_diagnostics[10]), 1.0)
        self.assertEqual(float(pulled_diagnostics[10]), 0.0)
        self.assertAlmostEqual(float(held_diagnostics[3]), 0.5 * 12000.0 * 0.03125**2, delta=1.0e-3)
        # The released leg applies no force, so it can supply no power either.
        self.assertAlmostEqual(float(held_diagnostics[1]), _ledger(columns, held_diagnostics), delta=1.0e-3)
        self.assertAlmostEqual(float(held_diagnostics[1]), 0.0, delta=1.0e-3)

    def test_measured_force_never_feeds_forward(self):
        """Ignore the measured foot force the legacy planar leg consumed, but keep the opposite foot.

        Columns 11 and 22 replay the capture-trial force of the measured shoe. Feeding
        them forward would cancel the simulated shoe's own response, so the leg force
        must stay bit-identical when they change, while the opposite-foot columns 12 and
        23 remain an external input that still loads body 1.
        """
        quiet, quiet_diagnostics = _launch(_NOMINAL)
        loud, loud_diagnostics = _launch(_NOMINAL | {_MEASURED_FZ: 1750.0, _MEASURED_FX: -420.0})
        np.testing.assert_array_equal(quiet, loud)
        np.testing.assert_array_equal(quiet_diagnostics, loud_diagnostics)
        external, external_diagnostics = _launch(_NOMINAL | {_OTHER_FZ: 1750.0, _OTHER_FX: -420.0})
        np.testing.assert_array_equal(external_diagnostics, quiet_diagnostics)
        np.testing.assert_array_equal(external[0], quiet[0])
        np.testing.assert_array_equal(external[1, :3] - quiet[1, :3], [-420.0, 0.0, 1750.0])

    def test_source_power_is_equilibrium_and_stiffness_work(self):
        """Close the source-power ledger in the unsaturated and force-limited branches alike.

        The equilibrium-work term must carry the PRE-clamp force. Substituting the applied force
        misreports every limited sample, so both branches are checked against an independently
        computed P_body + dE/dt + D.
        """
        error, rate = -0.03125, -0.5
        columns = _NOMINAL | {_STIFFNESS_RATE: -8192.0}
        _, free = _launch(columns)
        stiffness_work = 0.5 * -8192.0 * error * error
        self.assertEqual(float(free[4]), 0.0)
        self.assertAlmostEqual(float(free[1]), float(free[0]) * -0.125 + stiffness_work, delta=1.0e-3)
        self.assertAlmostEqual(float(free[1]), _ledger(columns, free), delta=1.0e-3)
        _, limited = _launch(columns, force_limit=300.0)
        raw = -12000.0 * error - 512.0 * (rate + 0.125)
        self.assertEqual(float(limited[0]), 300.0)
        self.assertEqual(float(limited[4]), 1.0)
        self.assertEqual(float(limited[10]), 0.0)
        self.assertAlmostEqual(float(limited[9]), abs(300.0 - raw), delta=2.0e-3)
        clamped = raw * -0.125 + stiffness_work + (300.0 - raw) * rate
        self.assertAlmostEqual(float(limited[1]), clamped, delta=2.0e-3)
        self.assertAlmostEqual(float(limited[1]), _ledger(columns, limited), delta=2.0e-3)
        # Using the applied force instead of the pre-clamp force would report this wrong value.
        self.assertGreater(abs(clamped - (300.0 * -0.125 + stiffness_work + (300.0 - raw) * rate)), 10.0)

    def test_source_power_holds_over_swept_commands(self):
        """Hold the equilibrium-point power identity across a sweep of unsaturated commands."""
        rng = np.random.default_rng(0)
        for _ in range(32):
            height = float(rng.uniform(0.7, 1.3))
            equilibrium, equilibrium_rate = float(rng.uniform(0.7, 1.3)), float(rng.uniform(-1.5, 1.5))
            stiffness, damping = float(rng.uniform(2000.0, 30000.0)), float(rng.uniform(50.0, 900.0))
            stiffness_rate = float(rng.uniform(-4.0e4, 4.0e4))
            speed = float(rng.uniform(-1.0, 1.0))
            columns = {
                _EQUILIBRIUM: equilibrium,
                _EQUILIBRIUM_RATE: equilibrium_rate,
                _STIFFNESS: stiffness,
                _DAMPING: damping,
                _STIFFNESS_RATE: stiffness_rate,
            }
            _, d = _launch(columns, (0.0, 0.0, 0.0), (0.0, 0.0, height), (0.0, 0.0, 0.0), (0.0, 0.0, speed))
            self.assertEqual(float(d[4]), 0.0)
            error = float(d[5]) - np.float32(equilibrium)
            expected = float(d[0]) * equilibrium_rate + 0.5 * stiffness_rate * error * error
            self.assertAlmostEqual(float(d[1]), expected, delta=1.0e-3 * max(1.0, abs(expected)))

    def test_source_power_closes_the_unsaturated_energy_ledger(self):
        """Balance source power against leg output, spring storage, and damper loss.

        The stored energy is differentiated numerically along the commanded trajectory
        instead of reusing the kernel's own algebra, so the check is independent of the
        expression under test. It covers the unsaturated branch only: clipping (force
        limit or unilateral release) reports ``force * L0dot`` where closure needs the
        pre-clamp ``raw * L0dot``, which is a separate, unverified claim.
        """
        length, rate = 1.0, -0.5
        equilibrium, equilibrium_rate = 1.03125, -0.125
        stiffness, stiffness_rate = 12000.0, -8192.0
        columns = _NOMINAL | {_STIFFNESS_RATE: stiffness_rate}
        _, d = _launch(columns)
        step = 1.0e-6
        stored = [
            0.5 * (stiffness + stiffness_rate * t) * (length + rate * t - equilibrium - equilibrium_rate * t) ** 2
            for t in (-step, step)
        ]
        storage_rate = (stored[1] - stored[0]) / (2.0 * step)
        output = float(d[0]) * rate
        dissipated = -float(d[2])
        self.assertAlmostEqual(float(d[1]), output + storage_rate + dissipated, delta=1.0e-3)

    def test_force_acts_along_the_line_between_the_bodies(self):
        """Direct the off-axis leg force along the unit vector between the two bodies."""
        p0, p1 = (0.5, 0.0, 0.25), (1.25, 0.0, 1.25)
        columns = _NOMINAL | {_EQUILIBRIUM: 1.28125, _EQUILIBRIUM_RATE: 0.125}
        wrench, diagnostics = _launch(columns, p0, p1, (0.0, 0.0, 0.0), (0.3, 0.0, 0.4), unilateral=1)
        axis = np.array([0.6, 0.0, 0.8])
        self.assertAlmostEqual(float(diagnostics[5]), 1.25, delta=1.0e-6)
        self.assertAlmostEqual(float(diagnostics[6]), 0.5, delta=1.0e-6)
        expected = -12000.0 * (1.25 - 1.28125) - 512.0 * (0.5 - 0.125)
        self.assertAlmostEqual(float(diagnostics[0]), expected, delta=1.0e-3)
        np.testing.assert_allclose(wrench[1, :3], float(diagnostics[0]) * axis, atol=1.0e-3)
        np.testing.assert_allclose(wrench[0, :3], -float(diagnostics[0]) * axis, atol=1.0e-3)
        self.assertAlmostEqual(float(diagnostics[7]), float(wrench[1, 0]), delta=1.0e-4)
        self.assertAlmostEqual(float(diagnostics[8]), float(wrench[1, 2]), delta=1.0e-4)

    def test_degenerate_impedance_stays_finite(self):
        """Produce a finite zero force for zero gains and for coincident body positions."""
        wrench, diagnostics = _launch(_NOMINAL | {_STIFFNESS: 0.0, _DAMPING: 0.0, _STIFFNESS_RATE: 0.0})
        np.testing.assert_array_equal(wrench, np.zeros((2, 6), np.float32))
        np.testing.assert_array_equal(diagnostics[:4], np.zeros(4, np.float32))
        self.assertTrue(np.all(np.isfinite(diagnostics)))
        collapsed, collapsed_diagnostics = _launch(_NOMINAL, _P0, _P0)
        self.assertTrue(np.all(np.isfinite(collapsed)))
        self.assertTrue(np.all(np.isfinite(collapsed_diagnostics)))
        np.testing.assert_array_equal(collapsed[:, :3], np.zeros((2, 3), np.float32))


@unittest.skipUnless(_PROFILE.is_file() and _ARTIFACT.is_file(), "Export the planar stance profile and shoe first")
class TestEquilibriumRegistration(unittest.TestCase):
    """Check what the equilibrium controller removes from the recorded rig registration."""

    def test_equilibrium_control_drops_the_release_schedule(self):
        """Report a commanded equilibrium controller with no engage, unload, or feedforward columns."""
        args = create_parser().parse_args(
            ["--viewer", "null", "--control", "equilibrium", "--substeps", "2", "--num-frames", "2"]
        )
        with wp.ScopedDevice("cpu"):
            example = Example(newton.viewer.ViewerNull(num_frames=2), args)
        for retired in ("unload_duration_s", "engage_duration_s", "unload_policy"):
            self.assertNotIn(retired, example.registration)
        self.assertIn("controller", example.registration)
        self.assertIn("release_policy", example.registration)
        self.assertEqual(example.reference.shape[1], _COLUMNS)
        np.testing.assert_array_equal(example.reference[:, 19], 1.0)
        np.testing.assert_array_equal(example.reference[:, [20, 21, 24]], 0.0)


if __name__ == "__main__":
    unittest.main()
