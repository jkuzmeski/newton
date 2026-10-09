# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that material values and derivatives use one source on every backend."""

import unittest
from dataclasses import replace

import numpy as np
import warp as wp

from newton.tests.unittest_utils import get_test_devices
from projects.digital_instron_v2 import core
from projects.digital_shoe import material, runtime


@wp.kernel
def _evaluate_material(params: wp.array[wp.float32], out: wp.array[wp.float32]):
    """Evaluate equilibrium pressure and one Maxwell step with material gradients."""
    pressure = material.hyperfoam_pressure(
        params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    )
    decay, ramp = material.maxwell_coefficients(params[8], params[9])
    out[0] = pressure
    out[1] = material.maxwell_step(params[10], pressure, params[11], params[12], decay, ramp)


@wp.kernel
def _shared_pressure(strain: wp.array[wp.float32], p: runtime.FoundationParams, out: wp.array[wp.float32]):
    """Evaluate the differentiable shared law from a runtime parameter block."""
    i = wp.tid()
    out[i] = material.hyperfoam_pressure(
        strain[i], p.g_eq, p.alpha, p.g_eq2, p.alpha2, p.beta, p.one_minus_two_poisson, p.stretch_floor
    )


@wp.kernel
def _forward_pressure(strain: wp.array[wp.float32], p: runtime.FoundationParams, out: wp.array[wp.float32]):
    """Evaluate the runtime's forward pressure entry point."""
    i = wp.tid()
    out[i] = runtime._hyperfoam_pressure(strain[i], p)


def _host_values(params):
    """Evaluate both observables with the shared float64 NumPy backend."""
    pressure = material.hyperfoam_pressure_numpy(*params[:8])
    decay, ramp = material.maxwell_coefficients_numpy(params[8], params[9])
    return np.array([pressure, material.maxwell_step_numpy(params[10], pressure, params[11], params[12], decay, ramp)])


class TestDigitalShoeMaterial(unittest.TestCase):
    """Verify source sharing, double-precision fitting, and Tape derivatives."""

    def test_backends_share_the_same_function_code(self):
        """Require one code object, not separate formulas that merely agree today."""
        for name in (
            "ogden_hill_term",
            "hyperfoam_pressure",
            "hyperfoam_pressure_from_stretch",
            "maxwell_coefficients",
            "maxwell_increment_step",
            "maxwell_step",
        ):
            with self.subTest(function=name):
                self.assertIs(getattr(material, name).func.__code__, getattr(material, name + "_numpy").__code__)

    def test_ogden_hill_preserves_legacy_term_fixtures(self):
        """Keep fixed outputs of the pre-consolidation Ogden-Hill term."""
        # Frozen from the old host formula, not evaluated through another
        # backend of the new implementation. Include both zero-alpha branches.
        fixtures = (
            ((0.8, 0.8, 86400.0, 1.4, 0.0), 41396.69575279714),
            ((0.25, 0.25, 12000.0, -1.5, 0.0), 448000.0),
            ((0.4, 0.5770799623628855, 12000.0, 0.0, 1.0 / 3.0), 65972.93269493915),
            ((0.4, 0.5770799623628855, 12000.0, 0.0005, 1.0 / 3.0), 65972.93269493915),
            ((0.6, 0.48911586576355365, 180000.0, 5.1, -1.0 / 7.0), 61178.08593556015),
            ((0.8, 0.8, 0.0, 0.0, 0.0), 0.0),
        )
        for arguments, expected in fixtures:
            with self.subTest(arguments=arguments):
                np.testing.assert_allclose(
                    material.ogden_hill_term_numpy(*arguments), expected, rtol=1.0e-13, atol=1.0e-10
                )

    def test_host_adapter_keeps_its_unilateral_boundary_and_stretch_floor(self):
        """Preserve the fitting adapter's compression-only boundary and 1e-3 floor."""
        foam = core.Material(100000.0, 2.0, 0.7, 0.01, 12000.0, -0.58)
        strain = np.array([-0.2, 0.0, 0.6, 0.9995, 1.1])
        pressure = core._hyperfoam_pressure(strain, foam)
        np.testing.assert_array_equal(pressure[:2], np.zeros(2))
        self.assertGreater(pressure[2], 0.0)
        self.assertGreater(pressure[3], pressure[2])
        self.assertEqual(pressure[3], pressure[4])
        self.assertTrue(np.all(np.isfinite(pressure)))

    def test_numpy_preserves_float64_vectorization(self):
        """Retain tiny host strains that a float32 device adapter would round away."""
        strain = np.array([0.0, 1.0e-10, 2.0e-10, 0.2, 0.8], dtype=np.float64)
        pressure = material.hyperfoam_pressure_numpy(strain, 1.0e5, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0e-3)
        self.assertEqual(pressure.dtype, np.float64)
        self.assertGreater(pressure[1], 0.0)
        self.assertGreater(pressure[2], pressure[1])
        expected = 1.0e5 * (2.0 * strain - strain * strain) / (1.0 - strain)
        np.testing.assert_allclose(pressure, expected, rtol=2.0e-7, atol=1.0e-10)

    def test_forward_pressure_matches_the_shared_law_bitwise(self):
        """Skip only volumetric powers that are exactly one, on every backend and branch.

        Zero-Poisson materials take the forward shortcut; the small-exponent limit, a
        disabled second term, and a nonzero Poisson ratio keep the shared expression.
        """
        strain = np.linspace(-0.2, 1.2, 4001, dtype=np.float32)
        for device in get_test_devices():
            for g_eq2, alpha2, poisson in (
                (12836.0, -0.586, 0.0),
                (12836.0, 0.0005, 0.0),
                (0.0, 4.0, 0.0),
                (9000.0, -2.0, 0.2),
            ):
                with self.subTest(device=str(device), alpha2=alpha2, g_eq2=g_eq2, poisson=poisson):
                    p = runtime.FoundationParams()
                    p.g_eq, p.alpha, p.g_eq2, p.alpha2 = 176270.67, 18.077, g_eq2, alpha2
                    p.beta = poisson / (1.0 - 2.0 * poisson)
                    p.one_minus_two_poisson = 1.0 - 2.0 * poisson
                    p.stretch_floor = 1.0e-3
                    values = wp.array(strain, dtype=wp.float32, device=device)
                    outputs = [wp.zeros_like(values) for _ in range(2)]
                    for kernel, out in zip((_shared_pressure, _forward_pressure), outputs, strict=True):
                        wp.launch(kernel, dim=len(strain), inputs=[values, p, out], device=device)
                    shared, forward = (out.numpy() for out in outputs)
                    self.assertTrue(np.all(np.isfinite(shared)))
                    np.testing.assert_array_equal(forward, shared)

    def test_legacy_laplacian_warns_and_preserves_force(self):
        """Deprecate the old subset shear input without changing its force history."""
        foam = core.Material(100000.0, 2.0, 1.0)
        slack = np.array([0.02, 0.03])
        lengths = np.array([[0.02, 0.03], [0.019, 0.028], [0.018, 0.027]])
        areas = np.array([1.0e-4, 2.0e-4])
        laplacian = np.array([[0.0, 0.0], [10.0, -10.0], [20.0, -20.0]])
        trial = core.Trial(
            "legacy",
            slack,
            areas,
            lengths,
            np.full(3, 0.01),
            np.zeros(3),
            np.array([0.0, 0.001, 0.002]),
            laplacian,
        )
        pressure = core._hyperfoam_pressure(np.maximum(slack[None, :] - lengths, 0.0) / slack[None, :], foam)
        expected = np.sum(areas * (np.maximum(pressure, 0.0) - foam.coupling_n_per_m(slack) * laplacian), axis=1)
        with self.assertWarnsRegex(DeprecationWarning, "Trial.surround"):
            actual = core.predict(trial, foam)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-12)
        uncoupled = replace(trial, compression_laplacian_m_inv=None)
        np.testing.assert_allclose(core.predict(uncoupled, foam), np.sum(areas * np.maximum(pressure, 0.0), axis=1))

    def test_warp_values_and_material_gradients(self):
        """Compare CPU and CUDA Tape with float64 finite differences of the same law."""
        for device in get_test_devices():
            for alpha2, modulus2, strain, poisson in (
                (-0.58, 12000.0, 0.6, 0.0),
                (-2.0, 12000.0, 0.8, -0.2),
                (0.0, 12000.0, 0.6, 0.3),
                (0.0005, 12000.0, 0.6, 0.1),
                (4.0, 0.0, 0.6, 0.0),
                (-0.58, 12000.0, 1.1, 0.0),
            ):
                with self.subTest(device=str(device), alpha2=alpha2, modulus2=modulus2, strain=strain, poisson=poisson):
                    values = np.array(
                        [
                            strain,
                            180000.0,
                            5.1,
                            modulus2,
                            alpha2,
                            poisson / (1.0 - 2.0 * poisson),
                            1.0 - 2.0 * poisson,
                            0.05,
                            0.001,
                            0.008,
                            300.0,
                            9000.0,
                            0.4,
                        ],
                        dtype=np.float32,
                    )
                    params = wp.array(values, device=device, requires_grad=True)
                    out = wp.zeros(2, device=device, requires_grad=True)
                    with wp.Tape() as tape:
                        wp.launch(_evaluate_material, 1, [params, out], device=device)
                    tape.backward(grads={out: wp.ones_like(out)})
                    host_values = values.astype(np.float64)
                    np.testing.assert_allclose(out.numpy(), _host_values(host_values), rtol=3.0e-6, atol=1.0e-2)
                    finite_difference = np.empty_like(host_values)
                    for index, value in enumerate(host_values):
                        step = max(abs(value) * 1.0e-4, 1.0e-5)
                        if index in (8, 9):
                            step = abs(value) * 1.0e-4
                        plus, minus = host_values.copy(), host_values.copy()
                        plus[index] += step
                        minus[index] -= step
                        finite_difference[index] = (_host_values(plus).sum() - _host_values(minus).sum()) / (2.0 * step)
                    np.testing.assert_allclose(params.grad.numpy(), finite_difference, rtol=8.0e-4, atol=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
