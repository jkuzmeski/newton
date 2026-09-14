# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the McCulloch foam mapping against the paper and against the runtime."""

import unittest

import numpy as np
import warp as wp

from newton.tests.unittest_utils import add_function_test, get_test_devices
from projects.digital_shoe.runtime import FoundationParams, cycle_force, cycle_overstress, set_material_block
from projects.impedance_instron.mcclough import (
    PAPER_COMPRESSION_KPA,
    PAPER_COMPRESSION_RAMP_S,
    PAPER_MAX_COMPRESSIVE_STRAIN,
    PAPER_PROPERTIES,
    RANDOMIZATION_BAND,
    REFERENCE_SHOE,
    STANCE_RAMP_S,
    compressive_stiffness_pa,
    energy_return,
    equilibrium_pressure_pa,
    materials,
)


def _runtime_energy_return(device, material, max_strain: float, ramp_s: float, steps: int = 400) -> float:
    """Relative energy return of one load-unload ramp, stepped through the runtime kernels.

    Drives a single unit column with a triangular compression history and reads the
    force the runtime's own :func:`cycle_overstress` and :func:`cycle_force` kernels
    produce, so the closed form in :mod:`projects.impedance_instron.mcclough` is checked
    against the law the solver actually integrates rather than against itself.
    """
    load = np.linspace(0.0, max_strain, steps + 1)
    history = np.concatenate([load, load[-2::-1]]).astype(np.float32)
    frames = history.size
    params = FoundationParams()
    set_material_block(params, material)
    params.stretch_floor = 0.05

    with wp.ScopedDevice(device):
        compression = wp.array(history.reshape(frames, 1), dtype=wp.float32)
        slack = wp.array([1.0], dtype=wp.float32)
        dt_s = wp.full(frames, ramp_s / steps, dtype=wp.float32)
        overstress = wp.zeros((frames, 1), dtype=wp.float32)
        force = wp.zeros(frames, dtype=wp.float32)
        wp.launch(
            cycle_overstress,
            dim=1,
            inputs=[
                compression,
                slack,
                dt_s,
                params,
                (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction,
                material.maxwell_relaxation_time_s,
            ],
            outputs=[overstress],
        )
        wp.launch(cycle_force, dim=(frames, 1), inputs=[compression, overstress, slack, params, 1.0], outputs=[force])
        stress = force.numpy()

    loading = float(np.trapezoid(stress[: steps + 1], history[: steps + 1]))
    unloading = -float(np.trapezoid(stress[steps:], history[steps:]))
    return unloading / loading


class TestMcCloughMapping(unittest.TestCase):
    """Check the published foam properties that the mapping claims to reproduce."""

    def test_materials_cover_both_reported_foams(self):
        """Verify materials() exposes exactly the two foams the paper characterizes."""
        self.assertEqual(sorted(materials()), sorted(PAPER_PROPERTIES))
        self.assertEqual(sorted(materials()), ["FF_LEAP", "FF_TURBO_PLUS"])

    def test_effective_poisson_ratio_is_zero(self):
        """Verify both foams carry the paper's measured zero effective Poisson ratio."""
        for name, material in materials().items():
            with self.subTest(foam=name):
                self.assertEqual(material.effective_poisson_ratio, 0.0)

    def test_compressive_stiffness_matches_the_paper(self):
        """Verify the paper's own 0-10% estimator on each fitted curve returns its E_com."""
        for name, material in materials().items():
            with self.subTest(foam=name):
                reported = PAPER_PROPERTIES[name]["compressive_stiffness_pa"][0]
                self.assertAlmostEqual(compressive_stiffness_pa(material) / reported, 1.0, places=3)

    def test_compression_curve_matches_the_published_tables(self):
        """Verify each fit tracks its Table 1 or Table 2 compression points.

        The tolerances are the residuals the module docstring claims, so a refit that
        silently degrades the large-strain backbone fails here.
        """
        strain = 1.0 - PAPER_COMPRESSION_KPA["stretch"]
        limits = {"FF_LEAP": (2.1, 4.6), "FF_TURBO_PLUS": (5.4, 10.9)}
        for name, material in materials().items():
            with self.subTest(foam=name):
                model = equilibrium_pressure_pa(material, strain) / 1.0e3
                residual = model - PAPER_COMPRESSION_KPA[name]
                rms_limit, max_limit = limits[name]
                self.assertLess(float(np.sqrt(np.mean(residual**2))), rms_limit)
                self.assertLess(float(np.max(np.abs(residual))), max_limit)

    def test_energy_return_matches_at_stance_rate_only(self):
        """Verify the ratio-anchored equilibrium fractions hit eta_com at stance rate.

        The same materials must return close to 100% at the paper's own 0.25/s bench
        rate; that gap is the documented limitation of a single Maxwell branch and is
        asserted so it cannot be removed unnoticed.
        """
        for name, material in materials().items():
            with self.subTest(foam=name):
                reported = PAPER_PROPERTIES[name]["energy_return_compression"][0]
                self.assertAlmostEqual(energy_return(material, ramp_s=STANCE_RAMP_S[name]), reported, places=2)
                self.assertGreater(energy_return(material, ramp_s=PAPER_COMPRESSION_RAMP_S), 0.99)

    def test_shear_stiffness_is_not_reproduced(self):
        """Verify the forced G = E/2 shear modulus inverts the paper's shear ordering.

        The Pasternak coefficient is derived from the equilibrium Ogden-Hill modulus, so
        shear is not a free parameter. This records the consequence rather than hiding it.
        """
        foams = materials()
        paper_ratio = (
            PAPER_PROPERTIES["FF_TURBO_PLUS"]["shear_stiffness_pa"][0]
            / PAPER_PROPERTIES["FF_LEAP"]["shear_stiffness_pa"][0]
        )
        model_ratio = (
            foams["FF_TURBO_PLUS"].equilibrium_shear_modulus_pa / foams["FF_LEAP"].equilibrium_shear_modulus_pa
        )
        self.assertGreater(paper_ratio, 1.0)
        self.assertLess(model_ratio, 1.0)

    def test_randomization_band_brackets_both_foams(self):
        """Verify every fractional half-width in RANDOMIZATION_BAND covers both foams."""
        for field, band in RANDOMIZATION_BAND.items():
            if not isinstance(band, float):
                continue
            reference = getattr(REFERENCE_SHOE, field)
            for name, material in materials().items():
                with self.subTest(field=field, foam=name):
                    self.assertLessEqual(abs(getattr(material, field) / reference - 1.0), band + 1.0e-9)

    def test_randomization_band_response_ratio_is_tight(self):
        """Verify the quoted response-space band is the smallest one that still holds."""
        low, high = RANDOMIZATION_BAND["equilibrium_pressure_ratio"]
        strain = np.linspace(0.02, PAPER_MAX_COMPRESSIVE_STRAIN, 60)
        reference = equilibrium_pressure_pa(REFERENCE_SHOE, strain)
        ratios = np.concatenate([equilibrium_pressure_pa(m, strain) / reference for m in materials().values()])
        self.assertGreaterEqual(float(ratios.min()), low)
        self.assertLessEqual(float(ratios.max()), high)
        self.assertLess(float(ratios.min()) - low, 0.02)
        self.assertLess(high - float(ratios.max()), 0.02)


devices = get_test_devices()


class TestMcCloughRuntime(unittest.TestCase):
    """Check the host-side closed forms against the runtime kernels they describe."""


def test_pressure_matches_runtime(test: unittest.TestCase, device):
    """Verify the closed-form energy return agrees with the runtime's own overstress."""
    for name, material in materials().items():
        with test.subTest(foam=name):
            runtime = _runtime_energy_return(device, material, PAPER_MAX_COMPRESSIVE_STRAIN, STANCE_RAMP_S[name])
            closed = energy_return(material, ramp_s=STANCE_RAMP_S[name])
            test.assertAlmostEqual(runtime, closed, places=3)


add_function_test(TestMcCloughRuntime, "test_pressure_matches_runtime", test_pressure_matches_runtime, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
