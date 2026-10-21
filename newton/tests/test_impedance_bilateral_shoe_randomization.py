# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the per-world shoe randomization: its prior, its arithmetic, and its write path.

The load-bearing test is :meth:`ShoeRandomizationWriteTest.test_matches_set_world_materials`,
which drives the same shoe through two mechanisms - this module\'s packed-table kernel and
the shared runtime\'s :meth:`MidsoleFoundation.set_world_materials` - and requires the
simulated ground force to agree bit for bit. Everything else is the prior: the ranges a
policy is trained on, the held-out points that must not overlap them, and the encoding the
critic and the estimator readout share.
"""

import unittest
from dataclasses import replace

import numpy as np

from projects.digital_shoe.runtime import FoundationParams, ShoeMaterial, set_material_block
from projects.impedance_instron.bilateral import env3d
from projects.impedance_instron.bilateral import shoe_randomization as sr
from projects.impedance_instron.bilateral.endurance3d import reset_once

DATASET = "outputs/bilateral/dataset_3d_static_sidefix"
THETA = {
    "g1_scale": 1.4,
    "alpha1": 0.3,
    "feq": 0.6,
    "tau_scale": 2.0,
    "g2_scale": 3.0,
    "normal_damping_n_s_m": 1.5,
    "mu": 0.9,
    "stack_dz_m": 0.0,
    "poisson": 0.15,
}


class ShoePriorTest(unittest.TestCase):
    """The prior, the archetypes, and the held-out split are pure numpy."""

    def setUp(self):
        self.sr = sr
        self.reference = {
            "instantaneous_shear_modulus_pa": 75474.90657,
            "instantaneous_shear_modulus_2_pa": 16.627388,
            "hyperfoam_exponent": 0.24493010,
            "hyperfoam_exponent_2": 1.78834402,
            "equilibrium_fraction": 0.69529549,
            "maxwell_relaxation_time_s": 0.00500041,
            "effective_poisson_ratio": 0.0,
            "mu": 0.8,
        }

    def test_nominal_row_reproduces_the_artifact(self):
        ref = self.reference
        packed = sr.pack(sr.nominal_row(ref), ref)[0]
        columns = dict(zip(sr.PACKED_FIELDS, range(len(sr.PACKED_FIELDS)), strict=True))
        self.assertAlmostEqual(packed[columns["alpha"]], ref["hyperfoam_exponent"], places=6)
        self.assertAlmostEqual(
            packed[columns["g_eq"]] + packed[columns["g_eq2"]],
            (ref["instantaneous_shear_modulus_pa"] + ref["instantaneous_shear_modulus_2_pa"])
            * ref["equilibrium_fraction"],
            places=1,
        )
        self.assertAlmostEqual(packed[columns["tau_s"]], ref["maxwell_relaxation_time_s"], places=8)
        self.assertAlmostEqual(packed[columns["normal_damping"]], 0.0)
        self.assertAlmostEqual(packed[columns["mu"]], ref["mu"])

    def test_matches_the_runtimes_own_material_block(self):
        """The packed table must equal what ``set_material_block`` writes for the same shoe."""
        ref = self.reference
        row = sr.as_row(sr.ShoeTheta(**THETA), ref)
        packed = sr.pack(row[None, :], ref)[0]
        columns = dict(zip(sr.PACKED_FIELDS, range(len(sr.PACKED_FIELDS)), strict=True))
        material = ShoeMaterial(
            instantaneous_shear_modulus_pa=ref["instantaneous_shear_modulus_pa"] * THETA["g1_scale"],
            hyperfoam_exponent=THETA["alpha1"],
            equilibrium_fraction=THETA["feq"],
            pasternak_n_per_m=0.0,
            effective_poisson_ratio=THETA["poisson"],
            maxwell_relaxation_time_s=ref["maxwell_relaxation_time_s"] * THETA["tau_scale"],
            instantaneous_shear_modulus_2_pa=ref["instantaneous_shear_modulus_2_pa"] * THETA["g2_scale"],
            hyperfoam_exponent_2=ref["hyperfoam_exponent_2"],
        )
        block = FoundationParams()
        set_material_block(block, material)
        for name in ("g_eq", "alpha", "g_eq2", "beta", "one_minus_two_poisson", "tau_s", "overstress"):
            self.assertAlmostEqual(
                float(packed[columns[name]]),
                float(getattr(block, name)),
                delta=1e-4 * max(1.0, abs(float(getattr(block, name)))),
            )

    def test_draws_stay_in_the_prior_and_vary(self):
        spec = sr.PRESETS["wide"]

        class Fake:
            worlds = 512

        rng = np.random.default_rng(0)
        del Fake
        draws = {name: interval.draw(rng, 512) for name, interval in spec.ranges.items()}
        for name, interval in spec.ranges.items():
            self.assertGreaterEqual(draws[name].min(), interval.low - 1e-9, name)
            self.assertLessEqual(draws[name].max(), interval.high + 1e-9, name)
            self.assertGreater(np.ptp(draws[name]), 0.2 * interval.width, name)

    def test_heldout_shoes_are_outside_the_trained_prior(self):
        spec = sr.PRESETS["wide"]
        outside = 0
        for name, (theta, axis) in sr.heldout().items():
            values = {key: getattr(theta, key) for key in sr.THETA_NAMES}
            moved = 0
            for component, value in values.items():
                interval = spec.ranges.get(component)
                if interval is None:
                    continue
                nominal = 1.0 if component in sr.SCALED else {"feq": 0.69529549, "mu": 0.8}.get(component, 0.0)
                if np.isclose(value, nominal):
                    continue
                moved += 1
                if axis == "extrapolation" and not interval.low <= value <= interval.high:
                    outside += 1
            self.assertGreater(moved, 0, name)
        self.assertGreaterEqual(outside, 4, "extrapolation rows must actually leave the trained box")

    def test_normalize_is_zero_at_nominal_and_monotone(self):
        import torch

        ref = self.reference
        spec = sr.PRESETS["wide"]
        base = sr.nominal_row(ref)
        rows = np.stack((base, base, base))
        rows[1, sr.THETA_NAMES.index("g1_scale")] = 2.0
        rows[2, sr.THETA_NAMES.index("g1_scale")] = 0.5
        out = sr.normalize(torch.tensor(rows, dtype=torch.float32), spec, ref)
        self.assertTrue(torch.all(out[0] == 0.0))
        self.assertTrue(torch.allclose(out[1], -out[2], atol=1e-6))
        self.assertGreater(float(out[1, 0]), 0.0)


class ShoeRandomizationWriteTest(unittest.TestCase):
    """The kernel path must be indistinguishable from the runtime\'s own material write."""

    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():  # pragma: no cover
            raise unittest.SkipTest("the 3D runner needs a CUDA device")

    def _library(self):
        return env3d.ReferenceLibrary3D(DATASET, ("train",), 1.0e-2, "cuda:0")

    def _environment(self, config, seed=3):
        env = env3d.BilateralEnv3D(self._library(), 8, config, seed=seed)
        reset_once(env)
        return env

    def test_matches_set_world_materials(self):
        import torch

        control = env3d.Env3DConfig(dt_s=2.5e-4, substeps=40, solver="mujoco")
        randomized = replace(control, shoe_dr="wide", shoe_swap_probability=0.0, shoe_dr_seed=7)

        env_a = self._environment(randomized)
        reference = sr.reference_of(env_a)
        row = sr.as_row(sr.ShoeTheta(**THETA), reference)
        env_a.shoe_randomizer.host[:] = row
        env_a.shoe_randomizer.theta.copy_(torch.tensor(row, device=env_a.torch_device))
        env_a.shoe_randomizer.apply()

        env_b = self._environment(control, seed=3)
        material = ShoeMaterial(
            instantaneous_shear_modulus_pa=reference["instantaneous_shear_modulus_pa"] * THETA["g1_scale"],
            hyperfoam_exponent=THETA["alpha1"],
            equilibrium_fraction=THETA["feq"],
            pasternak_n_per_m=0.0,
            effective_poisson_ratio=THETA["poisson"],
            maxwell_relaxation_time_s=reference["maxwell_relaxation_time_s"] * THETA["tau_scale"],
            instantaneous_shear_modulus_2_pa=reference["instantaneous_shear_modulus_2_pa"] * THETA["g2_scale"],
            hyperfoam_exponent_2=reference["hyperfoam_exponent_2"],
        )
        for side in ("right", "left"):
            foundation = env_b.shoes[side].foundation
            foundation.set_world_materials([material] * env_b.worlds)
            for block in foundation.world_blocks:
                block.normal_damping = THETA["normal_damping_n_s_m"]
            foundation.world_params.assign(foundation.world_blocks)
            settings = foundation.friction_solver.settings
            values = settings.numpy()
            values[:, 1] = THETA["mu"]
            values[:, 11] = reference["maxwell_relaxation_time_s"] * THETA["tau_scale"]
            settings.assign(values)
            foundation._materials_dirty = True

        self.assertTrue(torch.equal(env_a.joint_q, env_b.joint_q), "the two routes started apart")
        action = torch.zeros(env_a.worlds, env_a.action_size, device=env_a.torch_device)
        peak = 0.0
        for _ in range(120):
            env_a.advance(action)
            env_b.advance(action)
            force_a, force_b = env_a._interval_force(), env_b._interval_force()
            self.assertTrue(
                torch.equal(force_a, force_b),
                f"routes diverged: max |dF| {float((force_a - force_b).abs().max()):.3e} N",
            )
            peak = max(peak, float(force_b.abs().max()))
        # A flight initial state has to reach the ground for the comparison to mean
        # anything: two worlds that never touch the floor agree trivially.
        self.assertGreater(peak, 100.0, "the feet never loaded, so nothing was compared")

    def test_randomized_worlds_run_and_differ(self):
        """A wide draw has to change the ground force, and the runner has to survive it."""
        import torch

        config = env3d.Env3DConfig(
            dt_s=2.5e-4,
            substeps=40,
            solver="mujoco",
            shoe_dr="wide",
            shoe_swap_probability=0.0,
            contact_observation=True,
            theta_observation=True,
        )
        env = self._environment(config)
        self.assertEqual(env.observation_size, len(env.observation_names))
        action = torch.zeros(env.worlds, env.action_size, device=env.torch_device)
        for _ in range(40):
            env.advance(action)
        grf = env._interval_force().view(env.worlds, 2, 3)
        scale = env.shoe_randomizer.host[:, env.shoe_randomizer.index("g1_scale")]
        self.assertGreater(float(np.ptp(scale)), 0.1, "every world drew the same shoe")
        self.assertTrue(bool(torch.isfinite(grf).all()))
        self.assertGreater(float(grf[:, 0, 2].max()), 100.0)
        channels = env._contact_channels()
        self.assertEqual(tuple(channels.shape), (env.worlds, 12))
        self.assertTrue(bool(torch.isfinite(channels).all()))
        self.assertGreater(float(channels[:, 0:2].max()), 0.0, "the foam travel channels are dead")
        self.assertEqual(tuple(env._theta_channels().shape), (env.worlds, 9))


if __name__ == "__main__":
    unittest.main()
