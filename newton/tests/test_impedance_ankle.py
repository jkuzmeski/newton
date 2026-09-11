# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the equilibrium-point ankle of the impedance rig against its own physics claims.

The ankle replaces a prescribed pitch replay with the rotational form of the leg law,

.. math::

    \\tau(t) = k_\\theta(t)\\,(\\theta_0(t) - \\theta(t)) + b_\\theta(t)\\,(\\dot{\\theta}_0(t) - \\dot{\\theta}(t))

so the tests here cover the torque law, the energy ledger in both the clamped and the
unclamped branch, the torque-limit diagnostic, and the stiff limit in which the whole rig
reproduces the prescribed rollout it replaces.
"""

import hashlib
import math
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import warp as wp

from projects.impedance_instron.control import AnkleCommand
from projects.impedance_instron.example import (
    ANKLE_ANGLE,
    ANKLE_ANGLE_RATE,
    ANKLE_COLUMN_COUNT,
    ANKLE_DAMPING,
    ANKLE_DIAGNOSTIC_COUNT,
    ANKLE_STIFFNESS,
    ANKLE_STIFFNESS_RATE,
    Example,
    _apply_ankle_impedance,
    create_parser,
)

_PROFILE = Path("outputs/impedance_instron/stance_planar_context.json")
_ARTIFACT = Path("DigitalInstron/digital_shoe_showcase/digital_shoe.json")

# One configuration whose commanded quantities are exact in binary32: angle 0.25 rad,
# equilibrium 0.3125 rad, rate -0.5 rad/s, equilibrium rate -0.125 rad/s. The error is
# -0.0625 rad and the slip -0.375 rad/s, so the unclamped law asks for exactly 268 N*m.
_ANGLE, _RATE = 0.25, -0.5
_NOMINAL = {
    ANKLE_ANGLE: 0.3125,
    ANKLE_ANGLE_RATE: -0.125,
    ANKLE_STIFFNESS: 4096.0,
    ANKLE_DAMPING: 32.0,
}
_ERROR, _SLIP = -0.0625, -0.375

# The pitch angle survives a float32 quaternion round trip to about 1e-7 rad, so a hand
# computed torque is reproduced to roughly stiffness * 1e-7 rather than exactly.
_TORQUE_TOLERANCE_N_M = 1.0e-2

# Regression pin of the prescribed rollout. Recorded on the CPU, where the rollout is
# bit-reproducible, from the unmodified rig before the ankle actuator existed. The pin
# covers the whole rig, so it also pins the shared foam runtime; the test that uses it
# skips rather than fails when that file is a different build.
_PRESCRIBED_SAMPLES = 600
_PRESCRIBED_COLUMNS = 41
_PRESCRIBED_SHA256 = "5df197af6eca9800293c8145bbe06e4f09e5465adabd4e6f7032a9ba26971bfe"
_FOUNDATION_SOURCE = Path("projects/digital_shoe/runtime.py")
_FOUNDATION_SHA256 = "87f5b28a43e6dceef1c15457bd1570e533c7fd587d99ff1e8579f9d6f2c60be7"


def _foundation_is_pinned_build() -> bool:
    """Return True when the shared foam runtime is the build the trace pin was recorded on."""
    if not _FOUNDATION_SOURCE.is_file():
        return False
    return hashlib.sha256(_FOUNDATION_SOURCE.read_bytes()).hexdigest() == _FOUNDATION_SHA256


def _launch(
    columns: dict[int, float],
    angle_rad: float = _ANGLE,
    rate_rad_s: float = _RATE,
    torque_limit: float = 1.0e5,
    seed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Launch one ankle impedance step on the CPU and return the body wrenches and diagnostics.

    Args:
        columns: Reference-column values written into the single sample row.
        angle_rad: Fixture pitch stored as a Y-axis rotation of body 0 [rad].
        rate_rad_s: Fixture pitch rate stored in the bottom half of ``body_qd[0]`` [rad/s].
        torque_limit: Signed ankle actuator limit [N·m].
        seed: Optional pre-existing ``body_f`` content [N, N·m], shape [2, 6].
    """
    reference = np.zeros((1, ANKLE_COLUMN_COUNT), dtype=np.float32)
    for column, value in columns.items():
        reference[0, column] = value
    pose = np.zeros((2, 7), dtype=np.float32)
    pose[0, 3:7] = (0.0, math.sin(0.5 * angle_rad), 0.0, math.cos(0.5 * angle_rad))
    pose[1, 6] = 1.0
    velocity = np.zeros((2, 6), dtype=np.float32)
    velocity[0, 4] = rate_rad_s
    body_f = np.zeros((2, 6), dtype=np.float32) if seed is None else np.asarray(seed, dtype=np.float32).copy()
    force = wp.array(body_f, dtype=wp.spatial_vector, device="cpu")
    diagnostics = wp.zeros(ANKLE_DIAGNOSTIC_COUNT, dtype=wp.float32, device="cpu")
    wp.launch(
        _apply_ankle_impedance,
        dim=1,
        inputs=[
            wp.zeros(1, dtype=wp.int32, device="cpu"),
            wp.array(reference, dtype=wp.float32, device="cpu"),
            torque_limit,
            wp.array(pose, dtype=wp.transform, device="cpu"),
            wp.array(velocity, dtype=wp.spatial_vector, device="cpu"),
            force,
            diagnostics,
        ],
        device="cpu",
    )
    return force.numpy(), diagnostics.numpy()


def _ledger(columns: dict[int, float], diagnostics) -> float:
    """Rebuild ankle source power from first principles as P_body + dE/dt + D.

    This is deliberately independent of the kernel's own algebra: it uses only the recorded
    pitch state and the commanded impedance, so it catches an error in the kernel's closed
    form instead of restating it.

    Args:
        columns: Reference-column values the kernel was launched with.
        diagnostics: Diagnostic slots the kernel wrote.
    """
    k = columns.get(ANKLE_STIFFNESS, 0.0)
    b = columns.get(ANKLE_DAMPING, 0.0)
    stiffness_rate = columns.get(ANKLE_STIFFNESS_RATE, 0.0)
    equilibrium_rate = columns.get(ANKLE_ANGLE_RATE, 0.0)
    torque, angle, rate = float(diagnostics[0]), float(diagnostics[5]), float(diagnostics[6])
    error = angle - columns.get(ANKLE_ANGLE, 0.0)
    slip = rate - equilibrium_rate
    stored_rate = 0.5 * stiffness_rate * error * error + k * error * slip
    return torque * rate + stored_rate + b * slip * slip


class TestAnkleImpedanceKernel(unittest.TestCase):
    """Exercise ``_apply_ankle_impedance`` directly with hand-built fixture states."""

    def test_torque_follows_the_impedance_law(self):
        """Reproduce k*(theta0 - theta) - b*(thetadot - theta0dot) for a hand-computed state."""
        wrench, diagnostics = _launch(_NOMINAL)
        expected = -4096.0 * _ERROR - 32.0 * _SLIP
        self.assertAlmostEqual(expected, 268.0, delta=1.0e-9)
        self.assertAlmostEqual(float(diagnostics[0]), expected, delta=_TORQUE_TOLERANCE_N_M)
        self.assertAlmostEqual(float(diagnostics[5]), _ANGLE, delta=1.0e-6)
        self.assertAlmostEqual(float(diagnostics[6]), _RATE, delta=1.0e-6)
        self.assertAlmostEqual(float(diagnostics[2]), -32.0 * _SLIP * _SLIP, delta=_TORQUE_TOLERANCE_N_M)
        self.assertAlmostEqual(float(diagnostics[3]), 0.5 * 4096.0 * _ERROR * _ERROR, delta=_TORQUE_TOLERANCE_N_M)
        self.assertEqual(float(diagnostics[4]), 0.0)
        self.assertEqual(float(diagnostics[7]), 0.0)
        np.testing.assert_allclose(wrench[0, 3:], [0.0, expected, 0.0], atol=_TORQUE_TOLERANCE_N_M)

    def test_torque_acts_only_about_the_pitch_axis(self):
        """Add the ankle torque to the fixture pitch axis alone and leave body 1 untouched."""
        seed = np.arange(12, dtype=np.float32).reshape(2, 6)
        wrench, diagnostics = _launch(_NOMINAL, seed=seed)
        applied = wrench - seed
        np.testing.assert_array_equal(applied[1], np.zeros(6, np.float32))
        np.testing.assert_array_equal(applied[0, :3], np.zeros(3, np.float32))
        np.testing.assert_array_equal(applied[0, 3:], [0.0, diagnostics[0], 0.0])

    def test_pitch_state_is_read_from_the_fixture_pose(self):
        """Recover the commanded Y-axis rotation and pitch rate from the body state."""
        for angle in (-0.75, -0.125, 0.0, 0.5, 1.25):
            _, diagnostics = _launch({ANKLE_STIFFNESS: 1.0}, angle_rad=angle, rate_rad_s=0.25 * angle)
            self.assertAlmostEqual(float(diagnostics[5]), angle, delta=1.0e-6)
            self.assertAlmostEqual(float(diagnostics[6]), 0.25 * angle, delta=1.0e-6)

    def test_source_power_closes_the_unclamped_energy_ledger(self):
        """Close P_body + dE/dt + D with the commanded equilibrium and stiffness work.

        The stored energy is differentiated along the commanded trajectory instead of
        reusing the kernel's own expression, so the check is independent of the closed
        form under test.
        """
        stiffness_rate = -2048.0
        columns = _NOMINAL | {ANKLE_STIFFNESS_RATE: stiffness_rate}
        _, diagnostics = _launch(columns)
        stiffness_work = 0.5 * stiffness_rate * _ERROR * _ERROR
        self.assertEqual(float(diagnostics[4]), 0.0)
        self.assertAlmostEqual(
            float(diagnostics[1]), float(diagnostics[0]) * -0.125 + stiffness_work, delta=_TORQUE_TOLERANCE_N_M
        )
        self.assertAlmostEqual(float(diagnostics[1]), _ledger(columns, diagnostics), delta=_TORQUE_TOLERANCE_N_M)

        step = 1.0e-6
        stored = [
            0.5 * (4096.0 + stiffness_rate * t) * (_ANGLE + _RATE * t - 0.3125 - (-0.125) * t) ** 2
            for t in (-step, step)
        ]
        storage_rate = (stored[1] - stored[0]) / (2.0 * step)
        output = float(diagnostics[0]) * _RATE
        dissipated = -float(diagnostics[2])
        self.assertAlmostEqual(float(diagnostics[1]), output + storage_rate + dissipated, delta=1.0e-2)

    def test_source_power_closes_the_clamped_energy_ledger(self):
        """Carry the pre-clamp torque in the equilibrium term of a torque-limited sample.

        Substituting the applied torque would misreport every limited sample, which is
        exactly the defect the leg kernel once had, so the wrong form is also shown to
        differ by far more than the tolerance.
        """
        stiffness_rate = -2048.0
        columns = _NOMINAL | {ANKLE_STIFFNESS_RATE: stiffness_rate}
        _, diagnostics = _launch(columns, torque_limit=120.0)
        raw = -4096.0 * _ERROR - 32.0 * _SLIP
        stiffness_work = 0.5 * stiffness_rate * _ERROR * _ERROR
        self.assertAlmostEqual(float(diagnostics[0]), 120.0, delta=1.0e-6)
        self.assertEqual(float(diagnostics[4]), 1.0)
        self.assertAlmostEqual(float(diagnostics[7]), abs(120.0 - raw), delta=_TORQUE_TOLERANCE_N_M)
        expected = raw * -0.125 + stiffness_work + (120.0 - raw) * _RATE
        self.assertAlmostEqual(float(diagnostics[1]), expected, delta=1.0e-2)
        self.assertAlmostEqual(float(diagnostics[1]), _ledger(columns, diagnostics), delta=1.0e-2)
        wrong = 120.0 * -0.125 + stiffness_work + (120.0 - raw) * _RATE
        self.assertGreater(abs(expected - wrong), 10.0)

    def test_source_power_holds_over_swept_commands(self):
        """Hold the ankle power identity across random clamped and unclamped commands."""
        rng = np.random.default_rng(0)
        clamped = 0
        for _ in range(64):
            columns = {
                ANKLE_ANGLE: float(rng.uniform(-0.6, 1.2)),
                ANKLE_ANGLE_RATE: float(rng.uniform(-8.0, 8.0)),
                ANKLE_STIFFNESS: float(rng.uniform(100.0, 20000.0)),
                ANKLE_DAMPING: float(rng.uniform(1.0, 60.0)),
                ANKLE_STIFFNESS_RATE: float(rng.uniform(-4.0e4, 4.0e4)),
            }
            angle, rate = float(rng.uniform(-0.6, 1.2)), float(rng.uniform(-8.0, 8.0))
            _, diagnostics = _launch(columns, angle_rad=angle, rate_rad_s=rate, torque_limit=200.0)
            clamped += int(diagnostics[4] != 0.0)
            expected = _ledger(columns, diagnostics)
            self.assertAlmostEqual(float(diagnostics[1]), expected, delta=1.0e-3 * max(1.0, abs(expected)))
        self.assertGreater(clamped, 0)
        self.assertLess(clamped, 64)

    def test_torque_limit_clamps_symmetrically(self):
        """Clip both torque signs at the same magnitude and report the same clipped amount."""
        push = _NOMINAL | {ANKLE_ANGLE: 0.3125}
        pull = _NOMINAL | {ANKLE_ANGLE: 0.1875}
        raw_push = -4096.0 * _ERROR - 32.0 * _SLIP
        raw_pull = -4096.0 * 0.0625 - 32.0 * _SLIP
        self.assertGreater(raw_push, 0.0)
        self.assertLess(raw_pull, 0.0)
        _, high = _launch(push, torque_limit=120.0)
        _, low = _launch(pull, torque_limit=120.0)
        self.assertAlmostEqual(float(high[0]), 120.0, delta=1.0e-6)
        self.assertAlmostEqual(float(low[0]), -120.0, delta=1.0e-6)
        self.assertEqual(float(high[4]), 1.0)
        self.assertEqual(float(low[4]), 1.0)
        self.assertAlmostEqual(float(high[7]), abs(raw_push) - 120.0, delta=_TORQUE_TOLERANCE_N_M)
        self.assertAlmostEqual(float(low[7]), abs(raw_pull) - 120.0, delta=_TORQUE_TOLERANCE_N_M)

    def test_torque_limit_diagnostic_reports_only_saturation(self):
        """Keep the saturation channel clear for every event that is not the torque limit.

        A search that penalizes the torque limit must not be penalized for a pulling
        ankle, a collapsing stiffness or a fast equilibrium, so those cases leave both
        the flag and the clipped magnitude at zero.
        """
        cases = (
            _NOMINAL,
            _NOMINAL | {ANKLE_ANGLE: 0.0},
            _NOMINAL | {ANKLE_ANGLE_RATE: 6.0},
            _NOMINAL | {ANKLE_STIFFNESS: 0.0},
            _NOMINAL | {ANKLE_DAMPING: 0.0},
            _NOMINAL | {ANKLE_STIFFNESS_RATE: 5.0e4},
        )
        for columns in cases:
            _, diagnostics = _launch(columns, torque_limit=1.0e5)
            self.assertEqual(float(diagnostics[4]), 0.0)
            self.assertEqual(float(diagnostics[7]), 0.0)
        _, saturated = _launch(_NOMINAL, torque_limit=10.0)
        self.assertEqual(float(saturated[4]), 1.0)
        self.assertGreater(float(saturated[7]), 0.0)

    def test_degenerate_impedance_stays_finite(self):
        """Produce a finite zero torque for zero gains and for an unrotated fixture."""
        wrench, diagnostics = _launch(_NOMINAL | {ANKLE_STIFFNESS: 0.0, ANKLE_DAMPING: 0.0, ANKLE_STIFFNESS_RATE: 0.0})
        np.testing.assert_array_equal(wrench, np.zeros((2, 6), np.float32))
        np.testing.assert_array_equal(diagnostics[:4], np.zeros(4, np.float32))
        self.assertTrue(np.all(np.isfinite(diagnostics)))
        flat, flat_diagnostics = _launch({}, angle_rad=0.0, rate_rad_s=0.0)
        np.testing.assert_array_equal(flat, np.zeros((2, 6), np.float32))
        self.assertTrue(np.all(np.isfinite(flat_diagnostics)))


class TestAnkleCommandParameterization(unittest.TestCase):
    """Check the rotational command the ankle kernel reads its columns from."""

    def test_stiffness_positive_over_the_bound_box(self):
        """Keep k_theta positive and inside [100, 20000] N·m/rad over the whole bound box."""
        times = np.linspace(0.0, 0.375, 121)
        command = AnkleCommand(times)
        lower, upper = command.bounds()
        rng = np.random.default_rng(5)

        samples = [lower, upper, 0.5 * (lower + upper)]
        for _ in range(64):
            samples.append(np.where(rng.random(command.size) < 0.5, lower, upper))
        for _ in range(64):
            samples.append(lower + rng.random(command.size) * (upper - lower))

        for params in samples:
            profile = command.evaluate(params)
            self.assertTrue(np.all(profile.stiffness_nm_per_rad > 0.0))
            self.assertTrue(np.all(profile.stiffness_nm_per_rad >= 100.0 - 1.0e-6))
            self.assertTrue(np.all(profile.stiffness_nm_per_rad <= 20000.0 + 1.0e-6))
            self.assertTrue(np.all(profile.damping_nms_per_rad > 0.0))
            self.assertTrue(np.all(np.isfinite(profile.damping_nms_per_rad)))

    def test_initial_round_trip_reproduces_a_supplied_angle(self):
        """Reproduce a smooth reference pitch and match the analytic rate to a fine difference."""
        times = np.linspace(0.0, 0.375, 401)
        reference = 0.3 * np.sin(np.pi * times / times[-1]) - 0.2
        command = AnkleCommand(times)
        params = command.initial(reference, 4000.0, 0.5)
        profile = command.evaluate(params)
        self.assertLess(float(np.max(np.abs(profile.angle_rad - reference))), 2.0e-3)
        np.testing.assert_allclose(profile.stiffness_nm_per_rad, 4000.0, rtol=1.0e-9)
        implied = profile.damping_nms_per_rad / (2.0 * np.sqrt(profile.stiffness_nm_per_rad * command.inertia_kg_m2))
        np.testing.assert_allclose(implied, 0.5, rtol=1.0e-9)

        # The basis depends only on normalized time, so a denser grid over the same span
        # samples the same curve at higher resolution.
        fine_times = np.linspace(times[0], times[-1], 20001)
        fine = AnkleCommand(fine_times).evaluate(params)
        step = fine_times[1] - fine_times[0]
        central = (fine.angle_rad[2:] - fine.angle_rad[:-2]) / (2.0 * step)
        np.testing.assert_allclose(central, fine.angle_rate_rad_s[1:-1], atol=1.0e-8)


def _cuda_device() -> str | None:
    """Return the first CUDA device, or None when this machine has none."""
    devices = [device for device in wp.get_devices() if device.is_cuda]
    return str(devices[0]) if devices else None


def _ankle_vector(stiffness_n_m_per_rad: float, damping_ratio: float = 0.05) -> np.ndarray:
    """Return a constant-impedance ankle parameter vector, shape [15].

    The vector is packed instead of seeded through :meth:`AnkleCommand.initial` so the
    stiff-limit sweep can leave the optimizer bound box, which stops at 20000 N·m/rad.
    The angle knots are unused because the sweep commands the measured equilibrium.

    Args:
        stiffness_n_m_per_rad: Constant ankle stiffness [N·m/rad].
        damping_ratio: Constant commanded damping ratio.
    """
    return np.concatenate([np.zeros(6), np.full(6, math.log(stiffness_n_m_per_rad)), np.full(3, damping_ratio)])


def _rollout(
    device: str,
    samples: int | None = None,
    graph: bool = False,
    substeps: int = 64,
    stiffness_n_m_per_rad: float | None = None,
    torque_limit: float | None = None,
) -> Example:
    """Run one impedance rollout and return the finished example.

    Args:
        device: Warp device that carries the whole rollout.
        samples: Substep samples to stop after, or None for the whole stance.
        graph: Value of the ``--graph`` switch under test.
        substeps: Solver substeps per display frame.
        stiffness_n_m_per_rad: Constant ankle stiffness, or None for prescribed pitch.
        torque_limit: Ankle actuator torque limit [N·m].
    """
    args = create_parser().parse_args(["--viewer", "null", "--substeps", str(substeps)])
    args.graph = graph
    if stiffness_n_m_per_rad is not None:
        args.ankle_control = "impedance"
        args.ankle_equilibrium = "measured"
        args.ankle_vector = _ankle_vector(stiffness_n_m_per_rad)
        if torque_limit is not None:
            args.ankle_torque_limit = torque_limit
    with wp.ScopedDevice(device):
        example = Example(MagicMock(), args)
        if samples is not None:
            example.sample_count = min(samples, example.sample_count)
        while example.index < example.sample_count:
            example.step()
    return example


def _difference(reference: np.ndarray, achieved: np.ndarray, column: int) -> tuple[float, float]:
    """Return the maximum absolute and relative difference of one trace column.

    Args:
        reference: Prescribed-pitch trace.
        achieved: Ankle-impedance trace.
        column: Trace column to compare.
    """
    count = min(len(reference), len(achieved))
    left = reference[:count, column].astype(np.float64)
    right = achieved[:count, column].astype(np.float64)
    largest = float(np.max(np.abs(left)))
    absolute = float(np.max(np.abs(left - right)))
    return absolute, absolute / largest


@unittest.skipUnless(_PROFILE.is_file() and _ARTIFACT.is_file(), "Export the planar stance profile and shoe first")
class TestPrescribedPitchRegression(unittest.TestCase):
    """Check that adding the ankle actuator leaves the prescribed default path alone."""

    @unittest.skipUnless(_foundation_is_pinned_build(), "The shared foam runtime is not the pinned build")
    def test_prescribed_trace_is_byte_for_byte_unchanged(self):
        """Reproduce the exact prescribed trace recorded before the ankle actuator existed.

        The pin is a sha256 of the first 41 trace columns of a 600 substep CPU rollout of
        the default rig, which covers touchdown and the first 865 N of loading. It is an
        exact bit comparison over the whole rig, so a change to the foam runtime, the
        solver or the float stack moves it as well; the test therefore skips unless the
        shared foam runtime is still the build the pin was recorded against.
        """
        example = _rollout("cpu", samples=_PRESCRIBED_SAMPLES)
        trace = example.trace_device.numpy()[:_PRESCRIBED_SAMPLES, :_PRESCRIBED_COLUMNS]
        self.assertGreater(float(trace[:, 0].max()), 100.0)
        self.assertEqual(hashlib.sha256(trace.tobytes()).hexdigest(), _PRESCRIBED_SHA256)

    def test_prescribed_path_ignores_every_ankle_setting(self):
        """Reproduce the default trace exactly with hostile ankle settings still parsed.

        This claim survives an unrelated change to the shared foam runtime, which the
        sha256 pin above cannot: both rollouts run against whatever the rest of the rig
        currently is, and only the ankle settings differ.
        """
        default = _rollout("cpu", samples=200)
        args = create_parser().parse_args(["--viewer", "null"])
        args.graph = False
        args.ankle_torque_limit = 1.0
        args.ankle_stiffness = 1.0e9
        args.ankle_damping_ratio = 3.0
        args.ankle_vector = _ankle_vector(1.0e9)
        with wp.ScopedDevice("cpu"):
            hostile = Example(MagicMock(), args)
            hostile.sample_count = 200
            while hostile.index < hostile.sample_count:
                hostile.step()
        np.testing.assert_array_equal(hostile.trace_device.numpy()[:200], default.trace_device.numpy()[:200])

    def test_prescribed_rollout_never_drives_the_ankle_actuator(self):
        """Leave every ankle diagnostic and ankle trace column at zero in prescribed mode."""
        example = _rollout("cpu", samples=64)
        trace = example.trace_device.numpy()[:64]
        np.testing.assert_array_equal(example.ankle_diagnostics.numpy(), np.zeros(ANKLE_DIAGNOSTIC_COUNT, np.float32))
        np.testing.assert_array_equal(trace[:, 43:], np.zeros((64, 5), np.float32))
        # The achieved pitch is still recorded, and under replay it is the reference itself.
        np.testing.assert_array_almost_equal(trace[:, 41], example.reference[:64, 2], decimal=6)
        np.testing.assert_array_equal(trace[:, 42], example.reference[:64, 7])
        self.assertEqual(example.reference.shape[1], 25)
        self.assertEqual(example.metadata["ankle_control"], "prescribed")


@unittest.skipUnless(_PROFILE.is_file() and _ARTIFACT.is_file(), "Export the planar stance profile and shoe first")
class TestAnkleImpedanceRollout(unittest.TestCase):
    """Check what the ankle impedance controller changes in the assembled rig."""

    def test_impedance_mode_appends_the_ankle_reference_block(self):
        """Append theta0, theta0dot, k_theta, b_theta and k_theta_dot as the ankle columns."""
        example = _rollout("cpu", samples=2, stiffness_n_m_per_rad=4000.0)
        self.assertEqual(example.reference.shape[1], ANKLE_COLUMN_COUNT)
        np.testing.assert_array_equal(example.reference[:, ANKLE_ANGLE], example.reference[:, 2])
        np.testing.assert_array_equal(example.reference[:, ANKLE_ANGLE_RATE], example.reference[:, 7])
        np.testing.assert_allclose(example.reference[:, ANKLE_STIFFNESS], 4000.0, rtol=1.0e-5)
        expected = 2.0 * 0.05 * math.sqrt(4000.0 * example.pitch_inertia)
        np.testing.assert_allclose(example.reference[:, ANKLE_DAMPING], expected, rtol=1.0e-5)
        np.testing.assert_allclose(example.reference[:, ANKLE_STIFFNESS_RATE], 0.0, atol=1.0e-2)
        self.assertEqual(example.metadata["ankle_control"], "impedance")
        self.assertIn("ankle_controller", example.registration)
        self.assertIn("ankle_torque_limit_n_m", example.metadata)

    def test_entry_pitch_state_is_declared_not_replayed(self):
        """Start the free pitch state at the measured entry angle and angular rate."""
        example = _rollout("cpu", samples=1, stiffness_n_m_per_rad=4000.0)
        rotation = example.state_0.body_q.numpy()[0, 3:7]
        angle = 2.0 * math.atan2(float(rotation[1]), float(rotation[3]))
        self.assertAlmostEqual(angle, float(example.reference[0, 2]), delta=1.0e-6)
        self.assertAlmostEqual(
            float(example.state_0.body_qd.numpy()[0, 4]), float(example.reference[0, 7]), delta=1.0e-6
        )

    def test_contact_rotates_the_free_ankle_away_from_the_replay(self):
        """Let a compliant ankle rotate under contact load instead of following the replay.

        This is the whole point of the change: with a soft ankle the achieved pitch must
        leave the measured trajectory once the shoe is loaded, and the actuator must report
        a torque, a source power and a stored energy while it happens.
        """
        prescribed = _rollout("cpu", samples=_PRESCRIBED_SAMPLES)
        free = _rollout("cpu", samples=_PRESCRIBED_SAMPLES, stiffness_n_m_per_rad=1000.0)
        replay = prescribed.trace_device.numpy()[:_PRESCRIBED_SAMPLES]
        achieved = free.trace_device.numpy()[:_PRESCRIBED_SAMPLES]
        self.assertTrue(np.all(np.isfinite(achieved)))
        self.assertGreater(float(np.max(np.abs(achieved[:, 41] - replay[:, 41]))), 1.0e-3)
        self.assertGreater(float(np.max(np.abs(achieved[:, 18]))), 1.0)
        self.assertGreater(float(np.max(np.abs(achieved[:, 44]))), 0.0)
        self.assertGreater(float(np.max(achieved[:, 46])), 0.0)
        self.assertEqual(float(achieved[:, 47].max()), 0.0)
        rows = free.rows()
        self.assertAlmostEqual(rows[-1]["ankle_angle_rad"], float(achieved[-1, 41]), delta=1.0e-9)
        self.assertAlmostEqual(rows[-1]["pitch_rad"], float(achieved[-1, 41]), delta=1.0e-9)
        self.assertAlmostEqual(rows[-1]["ankle_source_power_w"], rows[-1]["pitch_power_w"], delta=1.0e-9)


@unittest.skipUnless(_PROFILE.is_file() and _ARTIFACT.is_file(), "Export the planar stance profile and shoe first")
@unittest.skipUnless(_cuda_device() is not None, "The stiff-limit sweep needs a CUDA device")
class TestAnkleStiffLimit(unittest.TestCase):
    """Check that prescribed pitch is the stiff limit of the ankle impedance.

    The new formulation has to contain the old one. With the equilibrium angle seeded from
    the measured pitch spline, the free-pitch rollout must approach the prescribed rollout
    as ``k_theta`` grows, and the approach must be monotone rather than accidental.
    """

    def test_free_pitch_converges_to_the_prescribed_rollout(self):
        """Shrink the pitch, shoe force and leg length differences monotonically with k_theta.

        Each decade of stiffness has to remove about a decade of difference, which is the
        signature of a steady-state deflection ``tau / k_theta`` rather than of an
        unrelated error that happens to be small.
        """
        device = _cuda_device()
        replay = _rollout(device, graph=True).trace_device.numpy()
        errors = {}
        for stiffness in (1.0e2, 1.0e4, 1.0e6):
            free = _rollout(device, graph=True, stiffness_n_m_per_rad=stiffness, torque_limit=1.0e9)
            achieved = free.trace_device.numpy()
            self.assertTrue(np.all(np.isfinite(achieved)))
            errors[stiffness] = {
                name: _difference(replay, achieved, column)[1]
                for name, column in (("pitch", 41), ("force", 0), ("length", 25))
            }
        for name in ("pitch", "force", "length"):
            decades = [errors[stiffness][name] for stiffness in (1.0e2, 1.0e4, 1.0e6)]
            self.assertTrue(all(later < earlier for earlier, later in pairwise(decades)), (name, decades))
            # Between 1e4 and 1e6 the ankle is the only limiting compliance left, so two
            # decades of stiffness have to remove close to two decades of difference.
            self.assertGreater(decades[1] / decades[2], 20.0, (name, decades))
            self.assertLess(decades[-1], 1.0e-2 * decades[0], (name, decades))
            self.assertLess(decades[-1], 1.0e-3, (name, decades))

    def test_captured_frames_reproduce_the_ankle_rollout(self):
        """Replay the ankle rollout from one captured frame with the same trace and state.

        The comparison stops before touchdown, where the whole substep sequence is exactly
        reproducible, so any mismatch is the capture itself rather than contact.
        """
        device = _cuda_device()
        plain = _rollout(device, samples=120, graph=False, stiffness_n_m_per_rad=4000.0)
        captured = _rollout(device, samples=120, graph=True, stiffness_n_m_per_rad=4000.0)
        self.assertFalse(plain.use_graph)
        self.assertTrue(captured.use_graph, captured.graph_status)
        trace = plain.trace_device.numpy()[:120]
        self.assertEqual(float(np.abs(trace[:, [0, 9, 11]]).max()), 0.0)
        np.testing.assert_array_equal(captured.trace_device.numpy()[:120], trace)
        np.testing.assert_array_equal(captured.state_0.body_q.numpy(), plain.state_0.body_q.numpy())
        np.testing.assert_array_equal(captured.state_0.body_qd.numpy(), plain.state_0.body_qd.numpy())


if __name__ == "__main__":
    unittest.main()
