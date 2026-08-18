# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Bhargava (2004) muscle metabolic cost estimation."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton.opensim as osim

_PIN_MUSCLE_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="pin_muscle">
    <gravity>0 -9.80665 0</gravity>
    <BodySet name="bodyset"><objects>
      <Body name="seg">
        <mass>0.5</mass>
        <mass_center>0 -0.15 0</mass_center>
        <inertia>0.01 0.01 0.01 0 0 0</inertia>
      </Body>
    </objects></BodySet>
    <JointSet name="jointset"><objects>
      <PinJoint name="pin">
        <socket_parent_frame>g_off</socket_parent_frame>
        <socket_child_frame>c_off</socket_child_frame>
        <coordinates><objects>
          <Coordinate name="pin_angle"><default_value>0.0</default_value><range>-2 2</range></Coordinate>
        </objects></coordinates>
        <frames>
          <PhysicalOffsetFrame name="g_off">
            <socket_parent>/ground</socket_parent>
            <translation>0 0.5 0</translation><orientation>0 0 0</orientation>
          </PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="c_off">
            <socket_parent>/bodyset/seg</socket_parent>
            <translation>0 0 0</translation><orientation>0 0 0</orientation>
          </PhysicalOffsetFrame>
        </frames>
      </PinJoint>
    </objects></JointSet>
    <ForceSet name="forceset"><objects></objects></ForceSet>
  </Model>
</OpenSimDocument>"""


def _pin_model(muscle_type: str = "DeGrooteFregly2016Muscle") -> osim.OsimModel:
    """Build a pin model with two muscles that have distinct paths."""
    model = osim.parse_osim(_PIN_MUSCLE_OSIM)
    parameters = {
        "max_isometric_force": 200.0,
        "optimal_fiber_length": 0.15,
        "tendon_slack_length": 0.05,
        "pennation_angle_at_optimal": 0.0,
        "max_contraction_velocity": 10.0,
    }
    model.muscles = [
        osim.OsimMuscle(
            name="flex",
            type=muscle_type,
            path_points=[
                osim.OsimPathPoint(name="a", body="ground", location=(0.08, 0.5, 0.0)),
                osim.OsimPathPoint(name="b", body="seg", location=(0.03, -0.1, 0.0)),
            ],
            params=parameters.copy(),
        ),
        osim.OsimMuscle(
            name="flex2",
            type=muscle_type,
            path_points=[
                osim.OsimPathPoint(name="c", body="ground", location=(0.12, 0.5, 0.0)),
                osim.OsimPathPoint(name="d", body="seg", location=(0.05, -0.1, 0.0)),
            ],
            params=parameters.copy(),
        ),
    ]
    return model


def _piecewise_reference(estimator, activations, excitations, coordinates, speeds):
    """Evaluate the original OpenSim equations independently in NumPy."""
    activations = np.asarray(activations, dtype=float)
    excitations = np.asarray(excitations, dtype=float)
    _, _, state = estimator.muscles._analysis_quantities(activations, coordinates, speeds)
    normalized_length = state["normalized_fiber_length"]
    fiber_velocity = (
        state["normalized_fiber_velocity"] * estimator.muscles._l_opt[None, :] * estimator.muscles._vmax[None, :]
    )
    active_force = state["active_fiber_force"]
    total_force = state["fiber_force"]

    slow_ratio = np.full(estimator.num_muscles, 0.5)
    slow_excitation = slow_ratio[None, :] * np.sin(0.5 * np.pi * excitations)
    fast_excitation = (1.0 - slow_ratio[None, :]) * (1.0 - np.cos(0.5 * np.pi * excitations))
    activation_rate = estimator.muscle_mass[None, :] * (40.0 * slow_excitation + 133.0 * fast_excitation)
    length_factor = np.interp(normalized_length, [0.0, 0.5, 1.0, 1.5, 10.0], [0.5, 0.5, 1.0, 0.0, 0.0])
    maintenance_rate = (
        estimator.muscle_mass[None, :] * length_factor * (74.0 * slow_excitation + 111.0 * fast_excitation)
    )
    alpha = np.where(fiber_velocity <= 0.0, 0.25 * total_force, 0.0)
    shortening_rate = -alpha * fiber_velocity
    mechanical_work_rate = -active_force * fiber_velocity
    preclamp = activation_rate + maintenance_rate + shortening_rate + mechanical_work_rate
    shortening_rate -= np.minimum(preclamp, 0.0)
    heat_rate = np.maximum(activation_rate + maintenance_rate + shortening_rate, estimator.muscle_mass[None, :])
    muscle_rate = heat_rate + mechanical_work_rate
    return activation_rate, maintenance_rate, shortening_rate, mechanical_work_rate, muscle_rate


class TestMuscleMetabolicsBhargava2004(unittest.TestCase):
    """Verify OpenSim-compatible muscle metabolic rates and trajectory costs."""

    def test_default_mass_and_basal_rate_match_opensim(self):
        """Use OpenSim's default muscle-mass approximation and basal coefficient."""
        metabolics = osim.MuscleMetabolicsBhargava2004(_pin_model())
        expected_mass = 200.0 / 0.25e6 * 1059.7 * 0.15
        np.testing.assert_allclose(metabolics.muscle_mass, expected_mass, rtol=1.0e-12)
        self.assertAlmostEqual(metabolics.body_mass, 0.5)
        self.assertAlmostEqual(metabolics.basal_rate, 0.6)

    def test_piecewise_rates_match_independent_equations(self):
        """Match every component of OpenSim's original piecewise implementation."""
        metabolics = osim.MuscleMetabolicsBhargava2004(_pin_model())
        activations = np.array([[0.2, 0.3], [0.4, 0.5], [0.7, 0.6]])
        excitations = np.array([[0.25, 0.35], [0.5, 0.45], [0.8, 0.7]])
        coordinates = np.array([[0.1], [0.2], [-0.1]])
        speeds = np.array([[-1.0], [1.0], [2.0]])
        result = metabolics.compute(activations, excitations, coordinates, speeds)
        expected = _piecewise_reference(metabolics, activations, excitations, coordinates, speeds)
        actual = (
            result.activation_rate,
            result.maintenance_rate,
            result.shortening_rate,
            result.mechanical_work_rate,
            result.muscle_rate,
        )
        for actual_component, expected_component in zip(actual, expected, strict=True):
            np.testing.assert_allclose(actual_component, expected_component, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(result.total_rate, np.sum(expected[-1], axis=1) + 0.6, rtol=2.0e-5)

    def test_force_dependent_shortening_and_positive_work_options(self):
        """Match OpenSim's optional force-dependent alpha and exclude eccentric work."""
        metabolics = osim.MuscleMetabolicsBhargava2004(
            _pin_model(),
            use_force_dependent_shortening_constant=True,
            include_negative_mechanical_work=False,
            forbid_negative_total_power=False,
            enforce_minimum_heat_rate=False,
        )
        activations = np.array([[0.3, 0.4], [0.6, 0.7]])
        excitations = np.array([[0.4, 0.5], [0.7, 0.8]])
        coordinates = np.array([[0.1], [-0.1]])
        speeds = np.array([[-1.0], [2.0]])
        result = metabolics.compute(activations, excitations, coordinates, speeds)
        _, _, state = metabolics.muscles._analysis_quantities(activations, coordinates, speeds)
        normalized_velocity = state["normalized_fiber_velocity"]
        fiber_velocity = normalized_velocity * metabolics.muscles._l_opt * metabolics.muscles._vmax
        force_velocity = -0.318 * np.arcsinh(-8.149 * normalized_velocity - 0.374) + 0.886
        isometric_active_force = state["active_fiber_force"] / force_velocity
        total_fiber_force = state["fiber_force"]
        alpha = np.where(
            fiber_velocity <= 0.0,
            0.16 * isometric_active_force + 0.18 * total_fiber_force,
            0.157 * total_fiber_force,
        )
        expected_shortening = -alpha * fiber_velocity
        expected_work = np.where(fiber_velocity <= 0.0, -state["active_fiber_force"] * fiber_velocity, 0.0)
        np.testing.assert_allclose(result.shortening_rate, expected_shortening, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(result.mechanical_work_rate, expected_work, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(
            result.muscle_rate,
            result.activation_rate + result.maintenance_rate + expected_shortening + expected_work,
            rtol=2.0e-5,
            atol=2.0e-6,
        )

    def test_custom_parameters_control_isometric_heat(self):
        """Apply provided muscle mass, fiber ratio, and heat constants per muscle."""
        custom = osim.MuscleMetabolicsBhargava2004Parameters(
            slow_twitch_ratio=1.0,
            muscle_mass=2.0,
            activation_constant_slow_twitch=10.0,
            activation_constant_fast_twitch=0.0,
            maintenance_constant_slow_twitch=20.0,
            maintenance_constant_fast_twitch=0.0,
        )
        metabolics = osim.MuscleMetabolicsBhargava2004(_pin_model(), muscle_parameters={"flex": custom})
        result = metabolics.compute([0.5, 0.5], [1.0, 0.0], [0.0])
        self.assertAlmostEqual(result.muscle_mass[0], 2.0)
        self.assertAlmostEqual(result.activation_rate[0, 0], 20.0, places=5)
        normalized_length = metabolics.muscles.fiber_kinematics([[0.0]])["normalized_fiber_length"][0, 0]
        length_factor = np.interp(normalized_length, [0.0, 0.5, 1.0, 1.5, 10.0], [0.5, 0.5, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(result.maintenance_rate[0, 0], 40.0 * length_factor, places=4)
        self.assertAlmostEqual(result.activation_rate[0, 1], 0.0)

    def test_energy_and_cost_of_transport_integrate_total_power(self):
        """Integrate trajectory power and normalize cost by mass and distance."""
        metabolics = osim.MuscleMetabolicsBhargava2004(_pin_model())
        times = np.array([0.0, 0.25, 1.0])
        result = metabolics.compute(
            np.full((3, 2), 0.3),
            np.full((3, 2), 0.35),
            np.zeros((3, 1)),
        )
        expected_energy = np.sum(0.5 * np.diff(times) * (result.total_rate[:-1] + result.total_rate[1:]))
        self.assertAlmostEqual(result.total_energy(times), expected_energy)
        self.assertAlmostEqual(result.cost_of_transport(times, 2.0), expected_energy / (0.5 * 2.0))
        expected_muscle_energy = np.sum(
            0.5 * np.diff(times)[:, None] * (result.muscle_rate[:-1] + result.muscle_rate[1:]), axis=0
        )
        np.testing.assert_allclose(result.muscle_energy(times), expected_muscle_energy)

    def test_compute_keeps_geometry_on_device_until_readback(self):
        """Avoid host muscle-path and analysis helpers in the metabolic pipeline."""
        metabolics = osim.MuscleMetabolicsBhargava2004(_pin_model())
        with (
            mock.patch.object(metabolics.muscles.paths, "lengths", side_effect=AssertionError("host lengths")),
            mock.patch.object(metabolics.muscles.paths, "velocities", side_effect=AssertionError("host velocities")),
            mock.patch.object(
                metabolics.muscles, "_analysis_quantities", side_effect=AssertionError("host muscle analysis")
            ),
        ):
            result = metabolics.compute([[0.2, 0.3], [0.4, 0.5]], [0.3, 0.4], [[0.0], [0.1]], [[0.0], [1.0]])
        self.assertEqual(result.muscle_rate.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(result.total_rate)))

    def test_tanh_smoothing_matches_every_opensim_conditional(self):
        """Match exact tanh velocity, work, power-clamp, and heat-floor branches."""
        metabolics = osim.MuscleMetabolicsBhargava2004(
            _pin_model(),
            use_smoothing=True,
            include_negative_mechanical_work=False,
            use_force_dependent_shortening_constant=True,
        )
        activations = np.array([[0.8, 0.7], [0.0, 0.0], [0.5, 0.6]])
        excitations = np.array([[0.0, 0.0], [0.0, 0.0], [0.6, 0.7]])
        coordinates = np.array([[0.1], [0.0], [-0.1]])
        speeds = np.array([[-5.0], [0.0], [5.0]])
        result = metabolics.compute(activations, excitations, coordinates, speeds)
        activation_rate, maintenance_rate, *_ = _piecewise_reference(
            metabolics, activations, excitations, coordinates, speeds
        )
        _, _, state = metabolics.muscles._analysis_quantities(activations, coordinates, speeds)
        normalized_velocity = state["normalized_fiber_velocity"]
        fiber_velocity = normalized_velocity * metabolics.muscles._l_opt * metabolics.muscles._vmax
        force_velocity = -0.318 * np.arcsinh(-8.149 * normalized_velocity - 0.374) + 0.886
        isometric_active_force = np.divide(
            state["active_fiber_force"],
            force_velocity,
            out=np.zeros_like(force_velocity),
            where=force_velocity != 0.0,
        )

        def conditional(condition, left, right, smoothing=10.0):
            """Evaluate OpenSim's tanh conditional in NumPy."""
            return left + (right - left) * (0.5 + 0.5 * np.tanh(smoothing * condition))

        condition = fiber_velocity + 1.0e-16
        alpha = conditional(
            condition,
            0.16 * isometric_active_force + 0.18 * state["fiber_force"],
            0.157 * state["fiber_force"],
        )
        shortening_rate = -alpha * condition
        mechanical_work_rate = conditional(condition, -state["active_fiber_force"] * fiber_velocity, 0.0)
        power_before_clamp = activation_rate + maintenance_rate + shortening_rate + mechanical_work_rate
        shortening_rate -= conditional(-power_before_clamp, 0.0, power_before_clamp)
        total_heat_rate = activation_rate + maintenance_rate + shortening_rate
        total_heat_rate = conditional(
            -total_heat_rate + metabolics.muscle_mass,
            total_heat_rate,
            metabolics.muscle_mass,
        )
        muscle_rate = total_heat_rate + mechanical_work_rate
        for actual, expected in zip(
            (
                result.activation_rate,
                result.maintenance_rate,
                result.shortening_rate,
                result.mechanical_work_rate,
                result.muscle_rate,
            ),
            (activation_rate, maintenance_rate, shortening_rate, mechanical_work_rate, muscle_rate),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-6)

    def test_unsupported_muscle_models_require_explicit_approximation(self):
        """Reject non-DGF mechanics unless the caller explicitly opts into approximation."""
        model = _pin_model("Thelen2003Muscle")
        with self.assertRaisesRegex(ValueError, "allow_approximate_muscle_models"):
            osim.MuscleMetabolicsBhargava2004(model)
        metabolics = osim.MuscleMetabolicsBhargava2004(model, allow_approximate_muscle_models=True)
        result = metabolics.compute([0.2, 0.3], [0.3, 0.4], [0.0])
        self.assertTrue(np.all(np.isfinite(result.total_rate)))

    def test_input_domains_and_required_model_parameters(self):
        """Reject empty batches, out-of-range muscle states, and invalid mechanics."""
        metabolics = osim.MuscleMetabolicsBhargava2004(_pin_model())
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            metabolics.compute(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 1)))
        with self.assertRaisesRegex(ValueError, r"activations must be in \[0, 1\]"):
            metabolics.compute([1.01, 0.5], [0.5, 0.5], [0.0])
        with self.assertRaisesRegex(ValueError, r"excitations must be in \[0, 1\]"):
            metabolics.compute([0.5, 0.5], [-0.01, 0.5], [0.0])
        invalid_model = _pin_model()
        invalid_model.muscles[0].params["optimal_fiber_length"] = 0.0
        with self.assertRaisesRegex(ValueError, "optimal_fiber_length"):
            osim.MuscleMetabolicsBhargava2004(invalid_model)

    def test_input_validation_reports_shape_and_parameter_errors(self):
        """Reject invalid muscle mappings, trajectory shapes, times, and distance."""
        with self.assertRaisesRegex(ValueError, "unknown muscles"):
            osim.MuscleMetabolicsBhargava2004(
                _pin_model(), muscle_parameters={"missing": osim.MuscleMetabolicsBhargava2004Parameters()}
            )
        with self.assertRaisesRegex(ValueError, "slow_twitch_ratio"):
            osim.MuscleMetabolicsBhargava2004(
                _pin_model(),
                muscle_parameters={"flex": osim.MuscleMetabolicsBhargava2004Parameters(slow_twitch_ratio=1.1)},
            )
        with self.assertRaisesRegex(ValueError, "muscle_mass"):
            osim.MuscleMetabolicsBhargava2004(
                _pin_model(), muscle_parameters={"flex": osim.MuscleMetabolicsBhargava2004Parameters(muscle_mass=0.0)}
            )
        metabolics = osim.MuscleMetabolicsBhargava2004(_pin_model())
        with self.assertRaisesRegex(ValueError, "activations"):
            metabolics.compute([0.5], [0.5, 0.5], [0.0])
        result = metabolics.compute(np.full((2, 2), 0.5), np.full((2, 2), 0.5), [[0.0], [0.0]])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            result.total_energy([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "distance"):
            result.cost_of_transport([0.0, 1.0], 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is not available")
    def test_cuda_matches_cpu(self):
        """Match CPU and CUDA metabolic-rate estimates."""
        model = _pin_model()
        activations = np.array([[0.2, 0.3], [0.7, 0.6]])
        excitations = np.array([[0.25, 0.35], [0.8, 0.7]])
        coordinates = np.array([[0.1], [-0.1]])
        speeds = np.array([[-1.0], [2.0]])
        cpu = osim.MuscleMetabolicsBhargava2004(model, device="cpu").compute(
            activations, excitations, coordinates, speeds
        )
        cuda = osim.MuscleMetabolicsBhargava2004(model, device="cuda:0").compute(
            activations, excitations, coordinates, speeds
        )
        np.testing.assert_allclose(cuda.muscle_rate, cpu.muscle_rate, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(cuda.total_rate, cpu.total_rate, rtol=2.0e-5, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
