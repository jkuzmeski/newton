# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the vectorized impedance stance environment against the open-loop rig it replaces.

The acceptance test of this suite is the zero-action episode: an environment driven with no
residual must reproduce the solved open-loop command of
``outputs/impedance_instron/command_j.json`` as recorded in ``outputs/impedance_instron/eval_j``.
Agreement with the stored reference is asserted to a tolerance taken from the rig's own
reproducibility, not chosen to pass. That reference was recorded when the foundation still summed
each world's ~910 column forces with float atomics, whose order the scheduler chose, so replaying
the stored command against it differed by a few milli-newtons once contact started: 5.2e-3 N on a
1792 N peak, about 3e-6 of the peak, and 1.8e-7 m of leg length. The tolerances below sit roughly
thirty times above that band and about four orders of magnitude below any change a real residual
makes. The foundation now reduces in a fixed order
(:func:`projects.digital_shoe.runtime.foundation_partial`), so the band no longer widens run to
run, but it still separates the current build from a reference recorded before that change.

What *is* exact now is the batch itself: worlds carrying the same material and the same action
produce bit-identical traces, and a perturbed world leaves its neighbours untouched to the last
bit. Those two checks assert equality, not closeness.
"""

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.digital_shoe.runtime import ShoeMaterial
from projects.impedance_instron import env as env_module
from projects.impedance_instron.control import AnkleCommand, LegCommand
from projects.impedance_instron.env import (
    ACTION_RATE_LIMIT,
    ACTION_SCALE,
    ANKLE_OBSERVATION_LAYOUT,
    CONTACT_FORCE_FRACTION,
    MOMENTUM_REWARD_SCALE_M_S,
    OBSERVATION_LAYOUT,
    TRACE_ANKLE_SOURCE_POWER,
    TRACE_ANKLE_TORQUE,
    TRACE_ANKLE_VX,
    TRACE_ANKLE_VZ,
    TRACE_ANKLE_Z,
    TRACE_COLUMNS,
    TRACE_COM_Z,
    TRACE_LEG_LENGTH,
    TRACE_PITCH,
    TRACE_PITCH_RATE,
    TRACE_SHOE_FX,
    TRACE_SHOE_FZ,
    TRACE_UPPER_VX,
    TRACE_UPPER_VZ,
    TRACE_UPPER_Z,
    WORK_REWARD_SCALE_J,
    ImpedanceEnv,
    ankle_seed,
)
from projects.impedance_instron.example import create_parser
from projects.impedance_instron.optimize import CHECKPOINTS

_PROFILE = Path("outputs/impedance_instron/stance_planar_context.json")
_ARTIFACT = Path("DigitalInstron/digital_shoe_showcase/digital_shoe.json")
_COMMAND = Path("outputs/impedance_instron/command_j.json")
_REFERENCE_TRACE = Path("outputs/impedance_instron/eval_j/trace.csv")
_INPUTS = _PROFILE.is_file() and _ARTIFACT.is_file() and _COMMAND.is_file() and _REFERENCE_TRACE.is_file()

# Fractions of the trajectory's own peak magnitude; see the module docstring for their provenance.
_FORCE_TOLERANCE = 1.0e-4
_LENGTH_TOLERANCE = 1.0e-5

# The observation the policy is allowed to see. A material constant appearing here would let the
# policy read the answer instead of inferring it, so the names are pinned by this test.
# Positional readers index the trace directly, so the meaning of every column is pinned here.
# New columns may only be appended.
_EXPECTED_TRACE_COLUMNS = (
    ("TRACE_SHOE_FZ", 0),
    ("TRACE_SHOE_FX", 1),
    ("TRACE_ANKLE_Z", 2),
    ("TRACE_ANKLE_VZ", 3),
    ("TRACE_ANKLE_VX", 4),
    ("TRACE_UPPER_VZ", 5),
    ("TRACE_UPPER_VX", 6),
    ("TRACE_LEG_FORCE", 7),
    ("TRACE_SOURCE_POWER", 8),
    ("TRACE_DAMPER_POWER", 9),
    ("TRACE_LEG_LENGTH", 10),
    ("TRACE_LEG_RATE", 11),
    ("TRACE_COMPRESSION", 12),
    ("TRACE_SATURATED", 13),
    ("TRACE_SATURATION_EXCESS", 14),
    ("TRACE_UPPER_Z", 15),
    ("TRACE_COM_Z", 16),
    ("TRACE_ANKLE_SOURCE_POWER", 17),
    ("TRACE_ANKLE_DAMPER_POWER", 18),
    ("TRACE_ANKLE_TORQUE", 19),
    ("TRACE_ANKLE_SATURATION_EXCESS", 20),
    ("TRACE_PITCH", 21),
    ("TRACE_PITCH_RATE", 22),
)

# Semi-implicit Euler advances the height with the post-step velocity, while the trapezoid rule
# uses the mean of two samples, so the two disagree by about 0.5 dt times the velocity change.
# Measured peak over one episode is 5.1e-5 m; the bar below leaves an order of magnitude.
_COM_INTEGRATION_TOLERANCE_M = 5.0e-4

_EXPECTED_OBSERVATION = (
    "leg_length_offset",
    "leg_length_rate",
    "shoe_fz_bw",
    "shoe_fx_bw",
    "foot_pitch",
    "foot_pitch_rate",
    "ankle_height",
    "ankle_vz",
    "com_vz",
    "episode_phase",
    "stance_phase",
    "in_contact",
    "previous_d_length",
    "previous_d_log_stiffness",
    "previous_d_damping_ratio",
)


def _nominal() -> np.ndarray:
    """Return the solved 15-parameter command the residual policy acts around."""
    return np.asarray(json.loads(_COMMAND.read_text())["parameters"], dtype=float)


def _args():
    """Return a fresh parsed namespace on the default rig settings."""
    return create_parser().parse_args(["--viewer", "null"])


def _build(num_worlds: int, args=None, **kwargs) -> ImpedanceEnv:
    """Build an environment on the default rig settings.

    Args:
        num_worlds: Number of independent stance episodes in the batch.
        args: Parsed namespace to build against; defaults to the rig defaults.
        **kwargs: Forwarded to :class:`ImpedanceEnv`.
    """
    return ImpedanceEnv(num_worlds, args if args is not None else _args(), _nominal(), **kwargs)


def _build_ankle(num_worlds: int, stiffness_n_m_per_rad: float, times, pitch_rad) -> ImpedanceEnv:
    """Build an ankle-impedance environment seeded to hold the measured pitch.

    ``ankle_equilibrium`` is forced to ``measured`` so the commanded equilibrium is the measured
    pitch itself. That isolates the finite ankle stiffness as the only reason the achieved pitch
    can differ from the prescribed rollout; leaving it on the commanded 6-knot angle spline would
    add the spline's own 0.0998 rad fit error and hide the convergence being tested.

    Args:
        num_worlds: Number of independent stance episodes in the batch.
        stiffness_n_m_per_rad: Seed ankle stiffness [N·m/rad].
        times: Evaluation grid of the rig [s].
        pitch_rad: Measured fixture pitch [rad].
    """
    args = _args()
    args.ankle_stiffness = stiffness_n_m_per_rad
    args.ankle_equilibrium = "measured"
    return ImpedanceEnv(num_worlds, args, _nominal(), ankle=ankle_seed(args, times, pitch_rad))


def _run_episode(env: ImpedanceEnv, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """Run one whole episode with a fixed action.

    Args:
        env: Environment to drive.
        actions: Raw action applied on every frame, shape [num_worlds, 3].

    Returns:
        The per-frame total reward, the per-frame dense work reward, the observations including
        the one returned by :meth:`ImpedanceEnv.reset`, and the final info dict.
    """
    observations = [env.reset()]
    rewards = np.zeros((env.episode_frames, env.num_worlds))
    work = np.zeros((env.episode_frames, env.num_worlds))
    info: dict = {}
    for frame in range(env.episode_frames):
        observation, reward, _done, info = env.step(actions)
        observations.append(observation)
        rewards[frame] = reward
        work[frame] = info["work_reward"]
    return rewards, work, observations, info


def _reference(columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read named columns out of the stored ``eval_j`` trace.

    Args:
        columns: Column names of ``outputs/impedance_instron/eval_j/trace.csv``.
    """
    with _REFERENCE_TRACE.open() as handle:
        header = handle.readline().strip().split(",")
    indices = [header.index(name) for name in columns]
    values = np.loadtxt(_REFERENCE_TRACE, delimiter=",", skiprows=1, usecols=indices)
    return dict(zip(columns, values.T, strict=True))


@unittest.skipUnless(_INPUTS, "Export the planar stance profile, the shoe, and the solved command first")
class TestZeroActionReproducesTheSolvedCommand(unittest.TestCase):
    """Drive one world with no residual and hold it against the stored open-loop result."""

    @classmethod
    def setUpClass(cls):
        """Run one zero-action episode and keep its trace, rewards and verdict."""
        cls.env = _build(1)
        cls.rewards, cls.work, cls.observations, cls.info = _run_episode(cls.env, np.zeros((1, 3)))
        cls.trace = cls.env.trace(0)
        cls.frames = np.arange(0, cls.env.sample_count, cls.env.substeps)
        cls.reference = _reference(("shoe_fz_n", "leg_length_m", "upper_slider_z_m", "com_z_m"))

    def test_shoe_vertical_force_matches_the_reference_trace(self):
        """Reproduce the stored shoe vertical force at every frame boundary."""
        stored = self.reference["shoe_fz_n"][self.frames]
        produced = self.trace[self.frames, TRACE_SHOE_FZ]
        error = float(np.max(np.abs(produced - stored)))
        self.assertLess(error, _FORCE_TOLERANCE * float(np.max(np.abs(stored))))

    def test_leg_length_matches_the_reference_trace(self):
        """Reproduce the stored geometric leg length at every frame boundary."""
        stored = self.reference["leg_length_m"][self.frames]
        produced = self.trace[self.frames, TRACE_LEG_LENGTH]
        error = float(np.max(np.abs(produced - stored)))
        self.assertLess(error, _LENGTH_TOLERANCE * float(np.max(np.abs(stored))))

    def test_the_whole_substep_trace_matches_the_reference(self):
        """Reproduce the stored force and length at every substep, not only at frame boundaries.

        Frame boundaries alone would hide a controller that only agrees where it is sampled, so
        the same comparison is repeated over all 2881 recorded substeps.
        """
        for name, column, tolerance in (
            ("shoe_fz_n", TRACE_SHOE_FZ, _FORCE_TOLERANCE),
            ("leg_length_m", TRACE_LEG_LENGTH, _LENGTH_TOLERANCE),
        ):
            with self.subTest(signal=name):
                stored = self.reference[name]
                error = float(np.max(np.abs(self.trace[:, column] - stored)))
                self.assertLess(error, tolerance * float(np.max(np.abs(stored))))

    def test_work_reward_sums_to_the_tier_three_proxy(self):
        """Sum the dense work reward over the episode and recover the tier 3 work proxy.

        The dense term is the negated increment of ``W+ / 0.25 + abs(W-) / 1.20`` over one frame.
        The trapezoid rule is additive over subintervals sharing their endpoints, so the sum must
        equal the proxy the objective computes from the complete trace up to float64 regrouping.
        """
        rollout = self.info["rollouts"][0]
        proxy = self.env.objective.work_proxy_j(rollout)
        summed = -float(np.sum(self.work[:, 0])) * WORK_REWARD_SCALE_J
        self.assertGreater(proxy, 100.0)
        self.assertAlmostEqual(summed / proxy, 1.0, delta=1.0e-6)
        # With a prescribed pitch motor there is no ankle actuator to charge, so the proxy is the
        # leg's alone and the two methods must agree exactly.
        self.assertEqual(rollout.ankle_positive_work_j, 0.0)
        self.assertEqual(rollout.ankle_negative_work_j, 0.0)
        self.assertEqual(proxy, self.env.objective.leg_work_proxy_j(rollout))

    def test_reset_returns_a_finite_observation_of_the_documented_shape(self):
        """Return a float32 [num_worlds, observation_dim] observation with no NaN or infinity."""
        observation = self.env.reset()
        self.assertEqual(observation.shape, (1, self.env.observation_dim))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(observation)))

    def test_repeated_reset_is_deterministic(self):
        """Return the same first observation from two consecutive resets of the same seed."""
        first = self.env.reset()
        second = self.env.reset()
        np.testing.assert_allclose(second, first, rtol=0.0, atol=0.0)

    def test_observation_is_mechanical_only(self):
        """Pin the observation width and layout so no shoe material constant can be added to it."""
        self.assertEqual(self.env.observation_dim, len(_EXPECTED_OBSERVATION))
        self.assertEqual(tuple(entry[0] for entry in OBSERVATION_LAYOUT), _EXPECTED_OBSERVATION)
        self.assertEqual(self.env.action_dim, 3)
        self.assertEqual(self.env.episode_frames, (self.env.sample_count - 1) // self.env.substeps)

    def test_reference_is_public_and_cannot_corrupt_the_device_upload(self):
        """Expose the measured reference rows read-only and leave the device copy untouched.

        Two separate claims are checked: an ordinary caller cannot write into the returned array
        at all, and a caller that deliberately forces the host view writable still cannot reach
        the device, because the upload is its own allocation made once at construction.
        """
        reference = self.env.reference
        self.assertEqual(reference.shape, (self.env.sample_count, 28))
        with self.assertRaises(ValueError):
            reference[0, 11] = 1.0
        np.testing.assert_array_equal(self.env.reference_device.numpy(), reference)
        forced = self.env.reference
        forced.setflags(write=True)
        original = float(forced[0, 11])
        try:
            forced[0, 11] = original + 1000.0
            self.assertEqual(float(self.env.reference_device.numpy()[0, 11]), original)
        finally:
            forced[0, 11] = original
        np.testing.assert_array_equal(self.env.reference_device.numpy(), self.env.reference)
        # The columns the trainer reads must carry the measured task, not zeros.
        for column in (1, 4, 9, 11, 22):
            with self.subTest(column=column):
                self.assertGreater(float(np.max(np.abs(reference[:, column]))), 0.0)

    def test_trace_column_indices_are_stable(self):
        """Pin every trace column index so positional readers keep working.

        New columns are appended; changing an existing index would silently re-interpret every
        stored trace and every analysis script that slices one.
        """
        for name, index in _EXPECTED_TRACE_COLUMNS:
            with self.subTest(column=name):
                self.assertEqual(getattr(env_module, name), index)
        self.assertEqual(TRACE_COLUMNS, len(_EXPECTED_TRACE_COLUMNS))
        self.assertEqual(self.trace.shape[1], TRACE_COLUMNS)

    def test_traced_com_height_is_the_mass_weighted_mix(self):
        """Recover the traced COM height from the traced ankle and upper heights.

        The kernel applies ``share = foot_mass / mass`` in float32; recomputing the same mix on
        the host in float64 must agree to the rounding of that single cast.
        """
        share = self.env.foot_mass / self.env.mass
        mixed = share * self.trace[:, TRACE_ANKLE_Z] + (1.0 - share) * self.trace[:, TRACE_UPPER_Z]
        np.testing.assert_allclose(self.trace[:, TRACE_COM_Z], mixed, rtol=0.0, atol=1.0e-6)

    def test_traced_com_height_agrees_with_integrated_com_velocity(self):
        """Integrate the traced COM velocity and land on the traced COM height.

        This is the mutual-consistency check between the new height column and the velocity
        columns that were already there: a height recorded from the wrong body, or a velocity
        recorded from the wrong one, would separate immediately.
        """
        share = self.env.foot_mass / self.env.mass
        velocity = share * self.trace[:, TRACE_ANKLE_VZ] + (1.0 - share) * self.trace[:, TRACE_UPPER_VZ]
        step = np.diff(self.env.times)
        integrated = self.trace[0, TRACE_COM_Z] + np.concatenate(
            [[0.0], np.cumsum(0.5 * (velocity[1:] + velocity[:-1]) * step)]
        )
        error = float(np.max(np.abs(integrated - self.trace[:, TRACE_COM_Z])))
        self.assertLess(error, _COM_INTEGRATION_TOLERANCE_M)
        self.assertGreater(float(np.ptp(self.trace[:, TRACE_COM_Z])), 0.05)

    def test_traced_heights_match_the_reference_trace(self):
        """Reproduce the stored upper-slider and COM heights of the open-loop run.

        The stored ``com_z_m`` column is assembled independently by ``Example.rows`` from its own
        trace, so this checks the new columns against a value that was never derived from them.
        """
        for name, column in (("upper_slider_z_m", TRACE_UPPER_Z), ("com_z_m", TRACE_COM_Z)):
            with self.subTest(signal=name):
                stored = self.reference[name]
                error = float(np.max(np.abs(self.trace[:, column] - stored)))
                self.assertLess(error, _LENGTH_TOLERANCE * float(np.max(np.abs(stored))))

    def test_realised_command_reproduces_the_nominal_profiles(self):
        """Return the nominal equilibrium, stiffness and damping when the residual was zero."""
        realised = self.env.realised_command(0)
        command = LegCommand(
            self.env.times,
            length_knots=self.env.args.length_knots,
            stiffness_knots=self.env.args.stiffness_knots,
            damping_knots=self.env.args.damping_knots,
            mass_kg=self.env.com_mass,
        )
        profile = command.evaluate(_nominal())
        samples = len(realised["length_m"])
        np.testing.assert_allclose(realised["length_m"], profile.length_m[:samples], rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(realised["stiffness_n_m"], profile.stiffness_n_m[:samples], rtol=1.0e-6, atol=1.0e-3)


@unittest.skipUnless(_INPUTS, "Export the planar stance profile, the shoe, and the solved command first")
class TestBatchedWorlds(unittest.TestCase):
    """Check that the batch is a batch: worlds share a launch but never share a trajectory."""

    @classmethod
    def setUpClass(cls):
        """Run one four-world zero-action episode and keep its complete trace."""
        cls.env = _build(4)
        _run_episode(cls.env, np.zeros((4, 3)))
        cls.baseline = cls.env.trace_device.numpy().copy()

    def test_identical_worlds_follow_the_same_trajectory(self):
        """Give four worlds of one material and one action bit-identical traces.

        Equality holds for every traced column, not just the contact ones, because each
        world's totals are summed over its own columns in a fixed order by
        :func:`projects.digital_shoe.runtime.foundation_partial`. While the foundation
        accumulated with ``wp.atomic_add`` this could only be checked to a few
        milli-newtons, since the scheduler chose the summation order; a tolerance here now
        would hide a real coupling between worlds.
        """
        np.testing.assert_array_equal(self.baseline, np.repeat(self.baseline[:, :1], self.baseline.shape[1], axis=1))
        # The episode has to have loaded the shoe, or identical zeros would pass this.
        self.assertGreater(float(np.max(np.abs(self.baseline[:, :, TRACE_SHOE_FZ]))), 100.0)

    def test_a_perturbed_world_leaves_its_neighbours_untouched(self):
        """Change only world 2's action and confirm worlds 0, 1 and 3 repeat their baseline.

        The check has teeth only if the perturbation actually did something, so the same
        assertion demands that world 2 moved by orders of magnitude more than the untouched ones.
        """
        actions = np.zeros((4, 3))
        actions[2, 0] = 1.0
        _run_episode(self.env, actions)
        perturbed = self.env.trace_device.numpy()
        untouched = [0, 1, 3]
        force = float(np.max(np.abs(self.baseline[:, :, TRACE_SHOE_FZ])))
        change = float(np.max(np.abs(perturbed[:, 2, TRACE_SHOE_FZ] - self.baseline[:, 2, TRACE_SHOE_FZ])))
        # Exact, not close: the untouched worlds must not move by a single bit.
        np.testing.assert_array_equal(perturbed[:, untouched], self.baseline[:, untouched])
        self.assertGreater(change, 1.0e-2 * force)

    def test_extreme_actions_stay_inside_the_leg_command_box(self):
        """Saturate the action in both directions and keep the resolved command inside its bounds.

        The bounds are re-derived here from :meth:`LegCommand.bounds` instead of read back from
        the environment, so a change to either side of the clip is caught.
        """
        command = LegCommand(
            self.env.times,
            length_knots=self.env.args.length_knots,
            stiffness_knots=self.env.args.stiffness_knots,
            damping_knots=self.env.args.damping_knots,
            mass_kg=self.env.com_mass,
        )
        lower, upper = command.bounds()
        first = command.length_knots
        second = first + command.stiffness_knots
        actions = np.array([[-40.0, -40.0, -40.0], [40.0, 40.0, 40.0], [-40.0, 40.0, -40.0], [40.0, -40.0, 40.0]])
        self.env.reset()
        # Ten frames, because the residual is rate limited and needs five to reach a saturated
        # target from zero; four would assert against a residual still on its way there.
        for _ in range(10):
            self.env.step(actions)
        for world in range(4):
            with self.subTest(world=world):
                realised = self.env.realised_command(world)
                self.assertTrue(np.all(realised["stiffness_n_m"] > 0.0))
                self.assertTrue(np.all(realised["length_m"] >= lower[0] - 1.0e-6))
                self.assertTrue(np.all(realised["length_m"] <= upper[0] + 1.0e-6))
                self.assertTrue(np.all(realised["stiffness_n_m"] >= np.exp(lower[first]) - 1.0e-3))
                self.assertTrue(np.all(realised["stiffness_n_m"] <= np.exp(upper[first]) + 1.0e-3))
                self.assertTrue(np.all(realised["damping_ratio"] >= lower[second] - 1.0e-6))
                self.assertTrue(np.all(realised["damping_ratio"] <= upper[second] + 1.0e-6))
                # The residual is rate limited, so the first five frames are still travelling
                # toward the saturated target; from there it sits exactly on ACTION_SCALE.
                settled = realised["residual"][5:]
                expected = np.tile(ACTION_SCALE[: self.env.action_dim], (len(settled), 1))
                np.testing.assert_allclose(np.abs(settled), expected, rtol=1.0e-9)
                self.assertTrue(np.all(np.abs(realised["residual"]) <= expected[0] + 1.0e-12))

    def test_the_shoe_material_is_invisible_until_the_shoe_is_loaded(self):
        """Give two worlds different foam and show the observation only learns it through contact.

        If a material constant were in the observation, the two worlds would already differ
        before the outsole touches the floor. They do not: the observation is identical while the
        shoe carries no load, and separates once it does.
        """
        softer = replace(
            self.env.material,
            instantaneous_shear_modulus_pa=0.4 * self.env.material.instantaneous_shear_modulus_pa,
            instantaneous_shear_modulus_2_pa=0.4 * self.env.material.instantaneous_shear_modulus_2_pa,
            pasternak_n_per_m=0.4 * self.env.material.pasternak_n_per_m,
        )
        self.assertIsInstance(softer, ShoeMaterial)
        self.env.set_world_materials([self.env.material, softer, self.env.material, softer])
        observation = self.env.reset()
        unloaded = observation.copy()
        for _ in range(4):
            observation, _reward, _done, _info = self.env.step(np.zeros((4, 3)))
            unloaded = observation.copy()
        trace = self.env.trace_device.numpy()[: 4 * self.env.substeps + 1]
        self.assertEqual(float(np.max(np.abs(trace[:, :, TRACE_SHOE_FZ]))), 0.0)
        self.assertEqual(float(np.max(np.abs(trace[:, :, TRACE_SHOE_FX]))), 0.0)
        np.testing.assert_allclose(unloaded[1], unloaded[0], rtol=0.0, atol=0.0)
        for _ in range(self.env.episode_frames - 4):
            observation, _reward, _done, _info = self.env.step(np.zeros((4, 3)))
        self.assertGreater(float(np.max(np.abs(observation[1] - observation[0]))), 1.0e-2)


@unittest.skipUnless(_INPUTS, "Export the planar stance profile, the shoe, and the solved command first")
class TestAnkleImpedance(unittest.TestCase):
    """Drive foot pitch with a commanded rotational impedance instead of an ideal motor.

    The prescribed pitch motor is the stiff limit of this controller, so the acceptance check is
    a convergence check rather than an equality: raising ``k_theta`` must shrink the difference
    between the achieved pitch and the prescribed schedule. It cannot be driven to float32 noise
    from inside :meth:`AnkleCommand.bounds`, whose ceiling is 2e4 N·m/rad.
    """

    @classmethod
    def setUpClass(cls):
        """Run the prescribed baseline and two ankle stiffnesses a decade apart."""
        baseline = _build(1)
        _run_episode(baseline, np.zeros((1, 3)))
        cls.prescribed = baseline.trace(0).copy()
        cls.times = baseline.times.copy()
        cls.measured_pitch = np.asarray(baseline.reference[:, 2], dtype=float).copy()
        cls.prescribed_span = float(np.ptp(cls.prescribed[:, TRACE_PITCH]))
        del baseline
        cls.traces = {}
        for stiffness in (2.0e3, 2.0e4):
            env = _build_ankle(1, stiffness, cls.times, cls.measured_pitch)
            _run_episode(env, np.zeros((1, env.action_dim)))
            cls.traces[stiffness] = env.trace(0).copy()
            if stiffness == 2.0e4:
                cls.env = env
            else:
                del env

    def _pitch_error(self, stiffness: float) -> float:
        """Return the peak achieved-pitch difference from the prescribed rollout, relative to its span.

        Args:
            stiffness: Seed ankle stiffness [N·m/rad] of the stored run.
        """
        difference = self.traces[stiffness][:, TRACE_PITCH] - self.prescribed[:, TRACE_PITCH]
        return float(np.max(np.abs(difference))) / self.prescribed_span

    def test_the_stiff_limit_approaches_the_prescribed_pitch(self):
        """Land within a justified bound of the prescribed pitch at the top of the stiffness box.

        Measured here: 4.80e-03 of the 1.635 rad prescribed pitch span at the 2e4 N·m/rad
        ceiling, against 4.34e-02 a decade below. The ankle agent reports about 1.3e-02 for the
        same limit on their own seeding, so the bar is set at 2.0e-02: above both measurements,
        and forty times below the 2.1e-01 a soft 200 N·m/rad ankle produces. Asserting equality
        is not available, because the box ceiling is a finite stiffness.
        """
        self.assertLess(self._pitch_error(2.0e4), 2.0e-2)

    def test_one_decade_of_stiffness_removes_the_pitch_difference(self):
        """Shrink the pitch difference by about a decade for each decade of ankle stiffness.

        This is the claim that the prescribed motor is the stiff limit of the ankle actuator,
        rather than a different controller that happens to look similar. It also proves pitch is
        genuinely free: at 2e3 N·m/rad the fixture departs from the prescribed schedule by more
        than four percent of its own range, which a written-back pitch column could not do.
        """
        soft, stiff = self._pitch_error(2.0e3), self._pitch_error(2.0e4)
        self.assertGreater(soft, 1.0e-2)
        self.assertGreater(soft / stiff, 5.0)

    def test_the_ankle_source_power_is_traced_and_reproducible_from_the_command(self):
        """Recompute the traced ankle torque and source power from the realised command.

        The objective charges the ankle actuator from this column, so it has to carry the
        equilibrium-point ledger and not a stand-in. Recomputing it on the host from the realised
        impedance and the achieved pitch is independent of the kernel's own algebra.
        """
        trace = self.traces[2.0e4]
        realised = self.env.realised_command(0)
        error = trace[:, TRACE_PITCH] - realised["angle_rad"]
        slip = trace[:, TRACE_PITCH_RATE] - realised["angle_rate_rad_s"]
        raw = -realised["ankle_stiffness_n_m_per_rad"] * error - realised["ankle_damping_n_m_s_per_rad"] * slip
        torque = np.clip(raw, -self.env.torque_limit_n_m, self.env.torque_limit_n_m)
        power = (
            raw * realised["angle_rate_rad_s"]
            + 0.5 * realised["ankle_stiffness_rate_n_m_per_rad_s"] * error * error
            + (torque - raw) * trace[:, TRACE_PITCH_RATE]
        )
        scale = float(np.max(np.abs(power)))
        self.assertGreater(scale, 1.0)
        self.assertGreater(float(np.max(np.abs(trace[:, TRACE_ANKLE_TORQUE]))), 1.0)
        np.testing.assert_allclose(trace[:, TRACE_ANKLE_TORQUE], torque, rtol=0.0, atol=1.0e-2)
        np.testing.assert_allclose(trace[:, TRACE_ANKLE_SOURCE_POWER], power, rtol=0.0, atol=1.0e-3 * scale)

    def test_the_ankle_work_reaches_the_rollout_and_the_dense_reward(self):
        """Charge the ankle actuator in both the verdict and the per-frame reward.

        An actuator the reward does not charge is free, and a policy spends a free actuator on
        everything. The check has three parts: the rollout carries the ankle work with its
        negative half still NEGATIVE, because
        :meth:`~projects.impedance_instron.objective.Objective.ankle_work_proxy_j` takes the
        absolute value itself and charges the halves at different efficiencies; the verdict's
        proxy is strictly larger than the leg's alone; and the summed dense reward reproduces the
        combined proxy, so tier 3 is still decomposed exactly rather than approximated.
        """
        _rewards, work, _observations, info = _run_episode(self.env, np.zeros((1, 6)))
        rollout = info["rollouts"][0]
        objective = self.env.objective
        self.assertTrue(objective.charge_ankle)
        self.assertGreater(rollout.ankle_positive_work_j, 1.0)
        self.assertLess(rollout.ankle_negative_work_j, 0.0)
        np.testing.assert_allclose(rollout.ankle_positive_work_j, info["ankle_positive_work_j"][0], rtol=1.0e-12)
        np.testing.assert_allclose(rollout.ankle_negative_work_j, info["ankle_negative_work_j"][0], rtol=1.0e-12)
        leg = objective.leg_work_proxy_j(rollout)
        proxy = objective.work_proxy_j(rollout)
        self.assertGreater(proxy, leg + 100.0)
        summed = -float(np.sum(work[:, 0])) * WORK_REWARD_SCALE_J
        self.assertAlmostEqual(summed / proxy, 1.0, delta=1.0e-6)

    def test_the_prescribed_path_leaves_the_ankle_columns_empty(self):
        """Record no ankle torque, power or saturation while the pitch motor is prescribed.

        A nonzero ankle column in the leg-only path would mean the objective was about to charge
        a rollout for an actuator that never ran.
        """
        for column in (TRACE_ANKLE_SOURCE_POWER, TRACE_ANKLE_TORQUE):
            with self.subTest(column=column):
                self.assertEqual(float(np.max(np.abs(self.prescribed[:, column]))), 0.0)
        # The prescribed motor writes the schedule straight into the pose, so the achieved pitch
        # is the reference pitch up to the float32 round trip through the quaternion.
        np.testing.assert_allclose(self.prescribed[:, TRACE_PITCH], self.measured_pitch, rtol=0.0, atol=1.0e-6)

    def test_the_action_and_observation_widen_with_the_ankle(self):
        """Report six actions and sixteen observations, with the leg-only layout untouched."""
        self.assertEqual(self.env.action_dim, 6)
        self.assertEqual(self.env.observation_dim, len(OBSERVATION_LAYOUT) + len(ANKLE_OBSERVATION_LAYOUT))
        self.assertEqual(
            tuple(entry[0] for entry in self.env.observation_layout)[: len(_EXPECTED_OBSERVATION)],
            _EXPECTED_OBSERVATION,
        )
        self.assertEqual(
            tuple(entry[0] for entry in ANKLE_OBSERVATION_LAYOUT),
            ("previous_d_angle", "previous_d_log_ankle_stiffness", "previous_d_ankle_damping_ratio"),
        )
        observation = self.env.reset()
        self.assertEqual(observation.shape, (1, len(OBSERVATION_LAYOUT) + len(ANKLE_OBSERVATION_LAYOUT)))
        self.assertTrue(np.all(np.isfinite(observation)))
        with self.assertRaises(ValueError):
            self.env.step(np.zeros((1, 3)))

    def test_saturated_actions_stay_inside_both_command_boxes(self):
        """Keep the resolved leg and ankle commands inside their own bounds at full action.

        Both boxes are re-derived here from the command classes rather than read back from the
        environment, so a change to either clip is caught.
        """
        leg = LegCommand(
            self.env.times,
            length_knots=self.env.args.length_knots,
            stiffness_knots=self.env.args.stiffness_knots,
            damping_knots=self.env.args.damping_knots,
            mass_kg=self.env.com_mass,
        )
        ankle = AnkleCommand(
            self.env.times,
            angle_knots=self.env.args.ankle_angle_knots,
            stiffness_knots=self.env.args.ankle_stiffness_knots,
            damping_knots=self.env.args.ankle_damping_knots,
            inertia_kg_m2=self.env.pitch_inertia,
        )
        for sign in (-1.0, 1.0):
            with self.subTest(sign=sign):
                self.env.reset()
                for _ in range(10):
                    self.env.step(np.full((1, 6), 40.0 * sign))
                realised = self.env.realised_command(0)
                for command, keys in (
                    (leg, ("length_m", "stiffness_n_m", "damping_ratio")),
                    (ankle, ("angle_rad", "ankle_stiffness_n_m_per_rad", "ankle_damping_ratio")),
                ):
                    lower, upper = command.bounds()
                    first = len(lower) - command.stiffness_knots - command.damping_knots
                    second = first + command.stiffness_knots
                    windows = (
                        (lower[0], upper[0], 1.0e-6),
                        (np.exp(lower[first]), np.exp(upper[first]), 1.0e-3),
                        (lower[second], upper[second], 1.0e-6),
                    )
                    for key, (low, high, slack) in zip(keys, windows, strict=True):
                        self.assertTrue(np.all(realised[key] >= low - slack), key)
                        self.assertTrue(np.all(realised[key] <= high + slack), key)
                self.assertTrue(np.all(realised["stiffness_n_m"] > 0.0))
                self.assertTrue(np.all(realised["ankle_stiffness_n_m_per_rad"] > 0.0))

    def test_the_ankle_path_stays_bit_deterministic(self):
        """Repeat one ankle episode exactly, and keep identical worlds identical.

        The foundation reduces each world's columns in a fixed order, so two runs of the same
        command must agree bit for bit. A tolerance here would hide a nondeterminism the ankle
        actuator reintroduced.
        """
        actions = np.zeros((1, 6))
        _run_episode(self.env, actions)
        first = self.env.trace_device.numpy().copy()
        _run_episode(self.env, actions)
        np.testing.assert_array_equal(self.env.trace_device.numpy(), first)
        batch = _build_ankle(4, 2.0e4, self.times, self.measured_pitch)
        _run_episode(batch, np.zeros((4, 6)))
        traces = batch.trace_device.numpy()
        np.testing.assert_array_equal(traces, np.repeat(traces[:, :1], traces.shape[1], axis=1))
        self.assertGreater(float(np.max(np.abs(traces[:, :, TRACE_ANKLE_TORQUE]))), 1.0)

    def test_graph_capture_survives_the_batched_ankle_path(self):
        """Keep replaying one captured frame graph at 1, 16 and 64 worlds with the ankle enabled.

        The ankle branch is a launch argument, not a new launch, so the captured frame is still
        one graph; this checks that claim at the batch sizes training uses.
        """
        for worlds in (1, 16, 64):
            with self.subTest(worlds=worlds):
                env = _build_ankle(worlds, 2.0e4, self.times, self.measured_pitch)
                env.reset()
                for _ in range(2):
                    env.step(np.zeros((worlds, 6)))
                self.assertEqual(env.graph_status, "enabled")
                self.assertTrue(np.all(np.isfinite(env.trace_device.numpy())))
                del env


@unittest.skipUnless(_INPUTS, "Export the planar stance profile, the shoe, and the solved command first")
class TestMomentumRewardTracksTheCriterion(unittest.TestCase):
    """Keep the dense momentum reward ranking episodes the way the tier 2 excursion ranks them.

    This is a regression class for a real failure. The dense term used to anchor its stance clock
    and its velocity datum on the COMMANDED touchdown, a fixed sample index, while
    :func:`projects.impedance_instron.optimize.simulate` measures the tier 2 momentum excursion on
    the run's OWN contact interval. With a prescribed pitch motor the actual touchdown barely
    moved and the two agreed, so the inconsistency was invisible. Once the ankle actuator made
    pitch a free state the foot began landing up to 20 ms early or late, the two measures came
    apart, and a 950-iteration policy improved the reward monotonically for 800 iterations while
    the excursion it was supposed to predict degraded from 1.079 to 1.951.

    The two episodes below land 37 ms apart, which is what makes the check meaningful; the test
    asserts that separation before it asserts anything about the ranking.
    """

    @classmethod
    def setUpClass(cls):
        """Run a zero-residual episode and a lengthened-leg one that lands much earlier."""
        probe = _build(1)
        times = probe.times.copy()
        pitch = np.asarray(probe.reference[:, 2], dtype=float).copy()
        del probe
        cls.env = _build_ankle(1, 2.0e4, times, pitch)
        cls.cases = {}
        for label, action in (("late", np.zeros(6)), ("early", np.array([40.0, 0.0, 0.0, 0.0, 0.0, 0.0]))):
            cls.env.reset()
            dense = 0.0
            for _ in range(cls.env.episode_frames):
                _observation, _reward, _done, info = cls.env.step(action[None, :])
                dense += float(info["momentum_reward"][0])
            verdict = info["verdicts"][0]
            cls.cases[label] = {
                "dense": dense,
                "excursion": float(verdict.excursions.get("momentum", 0.0)),
                "feasible": bool(verdict.feasible),
                "contact": int(cls.env._contact_sample[0]),
                "trace": cls.env.trace(0).copy(),
            }
        cls.times = cls.env.times.copy()

    def _commanded_anchor_dense(self, trace: np.ndarray) -> float:
        """Return the dense momentum total the retired commanded-touchdown anchor would give.

        Recomputed on the SAME trace, so the two anchors are compared on identical physics. This
        is the defect itself, kept executable so the regression test cannot pass vacuously.

        Args:
            trace: One world's substep trace of a finished episode.
        """
        env = self.env
        share = env.foot_mass / env.mass
        vx = share * trace[:, TRACE_ANKLE_VX] + (1.0 - share) * trace[:, TRACE_UPPER_VX]
        vz = share * trace[:, TRACE_ANKLE_VZ] + (1.0 - share) * trace[:, TRACE_UPPER_VZ]
        commanded = int(np.argmin(np.abs(self.times - env.touchdown_time_s)))
        frames = np.arange(0, self.times.size, env.substeps)[1:]
        phase = np.clip((self.times[frames] - self.times[commanded]) / env.target.duration_s, 0.0, 1.0)
        reference_vx = np.interp(phase, CHECKPOINTS, env.target.momentum_vx_m_s)
        reference_vz = np.interp(phase, CHECKPOINTS, env.target.momentum_vz_m_s)
        error = np.abs(vx[frames] - vx[commanded] - reference_vx) + np.abs(vz[frames] - vz[commanded] - reference_vz)
        started = self.times[frames] >= self.times[commanded]
        return -float(np.sum(np.where(started, error, 0.0))) / MOMENTUM_REWARD_SCALE_M_S

    def test_the_two_episodes_land_at_measurably_different_instants(self):
        """Establish the premise: the perturbed episode touches down well before the other one.

        Without a real shift in contact timing the commanded and the detected anchors coincide
        and the rest of this class would pass no matter which one the reward used.
        """
        late, early = self.cases["late"], self.cases["early"]
        separation = float(self.times[late["contact"]] - self.times[early["contact"]])
        self.assertGreater(separation, 0.010)
        self.assertTrue(late["feasible"] and early["feasible"])

    def test_touchdown_is_detected_at_the_gate_the_criterion_scores_with(self):
        """Detect touchdown at the same substep the scored stance starts at.

        The reward and the verdict must agree on when stance began, or they are again measuring
        different intervals. Both use :data:`CONTACT_FORCE_FRACTION` of body weight.
        """
        for label, case in self.cases.items():
            with self.subTest(case=label):
                loaded = case["trace"][:, TRACE_SHOE_FZ] > CONTACT_FORCE_FRACTION * self.env.body_weight_n
                self.assertEqual(case["contact"], int(np.nonzero(loaded)[0][0]))

    def test_the_dense_momentum_reward_ranks_like_the_tier_two_excursion(self):
        """Pay the episode with the worse momentum excursion the more negative dense reward.

        They need not be numerically equal: one is a per-frame absolute-time comparison and the
        other nine checkpoints on relative phase. They must agree in rank, or improving the
        reward can degrade the criterion, which is exactly what happened.
        """
        late, early = self.cases["late"], self.cases["early"]
        self.assertGreater(early["excursion"], late["excursion"])
        self.assertLess(early["dense"], late["dense"])

    def test_the_retired_commanded_anchor_would_have_inverted_the_ranking(self):
        """Show the defect is real by scoring the same two traces with the old anchor.

        If this ever stops inverting, the two episodes no longer separate the anchors and
        :meth:`test_the_dense_momentum_reward_ranks_like_the_tier_two_excursion` has lost its
        teeth; the pair must be re-chosen rather than the assertion relaxed.
        """
        late, early = self.cases["late"], self.cases["early"]
        self.assertGreater(early["excursion"], late["excursion"])
        self.assertGreater(self._commanded_anchor_dense(early["trace"]), self._commanded_anchor_dense(late["trace"]))

    def test_the_stance_phase_observation_follows_the_detected_touchdown(self):
        """Hold stance phase and the contact flag at zero until this world's shoe is loaded.

        A memoryless policy can only act on its own stance phase if the observation carries it,
        and reporting a phase before contact would restate the commanded-clock defect inside the
        observation.
        """
        names = [entry[0] for entry in self.env.observation_layout]
        phase_index, contact_index = names.index("stance_phase"), names.index("in_contact")
        observation = self.env.reset()
        self.assertEqual(float(observation[0, phase_index]), 0.0)
        self.assertEqual(float(observation[0, contact_index]), 0.0)
        phases, flags = [], []
        for _ in range(self.env.episode_frames):
            observation, _reward, _done, _info = self.env.step(np.zeros((1, 6)))
            phases.append(float(observation[0, phase_index]))
            flags.append(float(observation[0, contact_index]))
        flags = np.asarray(flags)
        phases = np.asarray(phases)
        self.assertEqual(sorted(set(flags.tolist())), [0.0, 1.0])
        self.assertEqual(int(np.sum(np.abs(np.diff(flags)))), 1)
        self.assertTrue(np.all(phases[flags == 0.0] == 0.0))
        # Inside stance the clock only advances, and it saturates at the measured duration.
        inside = phases[flags == 1.0]
        self.assertTrue(np.all(np.diff(inside) >= 0.0))
        self.assertLessEqual(float(inside.max()), 1.0)
        self.assertGreater(float(inside.max()), 0.5)


@unittest.skipUnless(_INPUTS, "Export the planar stance profile, the shoe, and the solved command first")
class TestCommandedProfileIsContinuous(unittest.TestCase):
    """Keep the commanded impedance continuous across the once-per-frame action boundaries.

    A residual applied as a STEP made the command piecewise constant with one discontinuity per
    frame. Measured on a trained policy's archive it moved the commanded stiffness by 23154 N/m
    across a single 130 us substep, 528 times the largest change inside a frame, and made ``kdot``
    a spike 523 times its own median whose size was set by the substep rather than by any physical
    rate. That term feeds ``0.5 kdot e^2`` in the source power and therefore the tier 3 objective,
    so it was charging an artefact of the decision rate.

    The action here VARIES from frame to frame. A constant action would leave every boundary after
    the first with nothing to step across and the class would pass on any implementation.
    """

    @classmethod
    def setUpClass(cls):
        """Run one episode driven by a seeded action walk and keep the command it produced."""
        probe = _build(1)
        times = probe.times.copy()
        pitch = np.asarray(probe.reference[:, 2], dtype=float).copy()
        del probe
        cls.env = _build_ankle(1, 4.0e3, times, pitch)
        rng = np.random.default_rng(0)
        cls.actions = np.clip(np.cumsum(0.6 * rng.normal(size=(cls.env.episode_frames, 6)), axis=0), -3.0, 3.0)
        cls.env.reset()
        for frame in range(cls.env.episode_frames):
            cls.env.step(cls.actions[frame][None, :])
        cls.command = cls.env.realised_command(0)
        cls.boundary = (np.arange(cls.env.times.size - 1) % cls.env.substeps) == 0

    def _steps(self, signal: np.ndarray) -> tuple[float, float]:
        """Return the largest sample-to-sample change at an action boundary and inside a frame.

        Args:
            signal: One commanded profile sampled at every substep.
        """
        change = np.abs(np.diff(np.asarray(signal, dtype=float)))
        return float(change[self.boundary].max()), float(change[~self.boundary].max())

    def test_the_command_does_not_step_at_the_action_boundaries(self):
        """Keep the boundary change within a small factor of the change inside a frame.

        Equality is not expected: the nominal spline and the ramp both move the command, and the
        boundary is where the ramp's slope changes. A factor of two is the bar; the defect this
        replaces was 528.
        """
        for name in ("stiffness_n_m", "length_m", "damping_ratio", "ankle_stiffness_n_m_per_rad", "angle_rad"):
            with self.subTest(signal=name):
                boundary, inside = self._steps(self.command[name])
                self.assertGreater(inside, 0.0)
                self.assertLess(boundary, 2.0 * inside)

    def test_the_stepped_residual_would_have_failed_that_check(self):
        """Rebuild the retired stepped profile from the SAME residuals and show it violates it.

        Without this the continuity check could pass on an episode whose residual barely moved.
        """
        residual = np.asarray(self.command["residual"], dtype=float)
        nominal = np.asarray(self.env.reference[:, 25], dtype=float)
        stepped = nominal.copy()
        for frame in range(residual.shape[0]):
            rows = slice(frame * self.env.substeps + 1, (frame + 1) * self.env.substeps + 1)
            stepped[rows] = np.clip(nominal[rows] * np.exp(residual[frame, 1]), *self.env.stiffness_bounds_n_m)
        change = np.abs(np.diff(stepped))
        self.assertGreater(change[self.boundary].max(), 10.0 * change[~self.boundary].max())

    def test_the_commanded_stiffness_rate_stays_near_its_own_median(self):
        """Bound the true derivative of the commanded stiffness, which the energy ledger charges.

        ``kdot`` is now the derivative of the command actually written, by the product rule, so a
        substep refinement cannot change it. Measured here: max over median falls from 14.9 with a
        stepped residual to 1.8. The bar is 10, well below the defect and well above the ramp.
        """
        rate = np.abs(np.gradient(np.asarray(self.command["stiffness_n_m"], dtype=float), self.env.times))
        reported = np.abs(np.asarray(self.command["stiffness_rate_n_m_s"], dtype=float))
        self.assertLess(rate.max(), 10.0 * np.median(rate))
        # The reported rate is the true one, not a smooth stand-in for it.
        scale = max(float(rate.max()), 1.0)
        self.assertLess(float(np.max(np.abs(reported - rate))), 0.2 * scale)

    def test_the_residual_respects_its_own_rate_limit(self):
        """Move each residual channel by no more than :data:`ACTION_RATE_LIMIT` per frame."""
        residual = np.asarray(self.command["residual"], dtype=float)
        change = np.abs(np.diff(np.vstack([np.zeros(self.env.action_dim), residual]), axis=0))
        limit = np.asarray(ACTION_RATE_LIMIT[: self.env.action_dim])
        self.assertTrue(np.all(change <= limit + 1.0e-12))
        # The walk has to actually reach the limit, or the assertion above is vacuous.
        self.assertGreater(float(change.max(axis=0).min()), 0.0)
        self.assertTrue(np.any(change >= limit - 1.0e-12))

    def test_a_zero_action_episode_is_exactly_the_nominal_command(self):
        """Leave the nominal spline untouched when every residual is zero.

        The ramp and the rate limit are both identities at zero, so this must be bit equality, not
        a tolerance. If it ever needs one, the ramp is wrong.
        """
        self.env.reset()
        for _ in range(self.env.episode_frames):
            self.env.step(np.zeros((1, self.env.action_dim)))
        command = self.env.realised_command(0)
        for key, column in (
            ("length_m", 0),
            ("length_rate_m_s", 1),
            ("stiffness_n_m", 2),
            ("damping_n_s_m", 3),
            ("stiffness_rate_n_m_s", 4),
        ):
            with self.subTest(signal=key):
                nominal = self.env._nominal[:, column].astype(np.float32)
                np.testing.assert_array_equal(np.asarray(command[key], dtype=np.float32), nominal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
