# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the vectorized impedance stance environment against the open-loop rig it replaces.

The acceptance test of this suite is the zero-action episode: an environment driven with no
residual must reproduce the solved open-loop command of
``outputs/impedance_instron/command_j.json`` as recorded in ``outputs/impedance_instron/eval_j``.
Agreement is asserted to a tolerance taken from the rig's own reproducibility, not chosen to pass:
the elastic foundation reduces about a thousand column forces per world with float atomics, so two
runs of the same command on the same commit differ by a few milli-newtons once contact starts.
Re-running the stored command through HEAD measures that band at 5.2e-3 N on a 1792 N peak, i.e.
about 3e-6 of the peak, and 1.8e-7 m of leg length. The tolerances below sit roughly thirty times
above the measured band and about four orders of magnitude below any change a real residual makes.
"""

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.digital_shoe.runtime import ShoeMaterial
from projects.impedance_instron import env as env_module
from projects.impedance_instron.control import LegCommand
from projects.impedance_instron.env import (
    ACTION_SCALE,
    OBSERVATION_LAYOUT,
    TRACE_ANKLE_VZ,
    TRACE_ANKLE_Z,
    TRACE_COLUMNS,
    TRACE_COM_Z,
    TRACE_LEG_LENGTH,
    TRACE_SHOE_FX,
    TRACE_SHOE_FZ,
    TRACE_UPPER_VZ,
    TRACE_UPPER_Z,
    WORK_REWARD_SCALE_J,
    ImpedanceEnv,
)
from projects.impedance_instron.example import create_parser

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
    "contact_phase",
    "previous_d_length",
    "previous_d_log_stiffness",
    "previous_d_damping_ratio",
)


def _nominal() -> np.ndarray:
    """Return the solved 15-parameter command the residual policy acts around."""
    return np.asarray(json.loads(_COMMAND.read_text())["parameters"], dtype=float)


def _build(num_worlds: int, **kwargs) -> ImpedanceEnv:
    """Build an environment on the default rig settings.

    Args:
        num_worlds: Number of independent stance episodes in the batch.
        **kwargs: Forwarded to :class:`ImpedanceEnv`.
    """
    args = create_parser().parse_args(["--viewer", "null"])
    return ImpedanceEnv(num_worlds, args, _nominal(), **kwargs)


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
        """Keep four worlds of one material and one action inside the atomic-reduction band.

        Bit equality is not available: the foundation reduces each world's column forces with
        float atomics, whose order depends on the scheduler, so two worlds of the same material
        differ by the same few milli-newtons two runs of one world differ by.
        """
        for column, tolerance in ((TRACE_SHOE_FZ, _FORCE_TOLERANCE), (TRACE_LEG_LENGTH, _LENGTH_TOLERANCE)):
            with self.subTest(column=column):
                values = self.baseline[:, :, column]
                spread = float(np.max(np.abs(values - values[:, :1])))
                self.assertLess(spread, tolerance * float(np.max(np.abs(values))))

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
        drift = float(np.max(np.abs(perturbed[:, untouched] - self.baseline[:, untouched])))
        change = float(np.max(np.abs(perturbed[:, 2, TRACE_SHOE_FZ] - self.baseline[:, 2, TRACE_SHOE_FZ])))
        self.assertLess(drift, _FORCE_TOLERANCE * force)
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
        for _ in range(4):
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
                np.testing.assert_allclose(
                    np.abs(realised["residual"]), np.tile(ACTION_SCALE, (len(realised["residual"]), 1)), rtol=1.0e-9
                )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
