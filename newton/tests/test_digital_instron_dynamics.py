# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the column-bed midsole example."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.viewer
from projects.digital_instron_v2 import core, dynamics
from projects.digital_instron_v2.example import FOAM_SWEEP, Example, ShoeInstron
from projects.digital_instron_v2.geometry import build_column_grid, load_mesh
from projects.digital_shoe import runtime

MANIFEST = "DigitalInstron/manifest_v2.json"


def _available_devices() -> list:
    """Return every Warp device the probe kernels should be checked on."""
    devices = [wp.get_device("cpu")]
    if wp.is_cuda_available():
        devices.append(wp.get_device("cuda:0"))
    return devices


@wp.kernel
def _hyperfoam_probe(
    strain: wp.array[wp.float32],
    params: runtime.FoundationParams,
    pressure_out: wp.array[wp.float32],
):
    """Evaluate the runtime Hyperfoam law once per sample so a host test can read it."""
    i = wp.tid()
    pressure_out[i] = runtime._hyperfoam_pressure(strain[i], params)


def _probe_params(material: core.Material) -> runtime.FoundationParams:
    """Build device-side constants for the probe kernels from a fitted material."""
    params = runtime.FoundationParams()
    poisson = core.EFFECTIVE_POISSON_RATIO
    # Both Ogden-Hill terms, through the one host helper that fills them, so the
    # probe cannot silently collapse a two-term material onto its first term.
    runtime.set_hyperfoam_series(params, material)
    params.beta = poisson / (1.0 - 2.0 * poisson)
    params.one_minus_two_poisson = 1.0 - 2.0 * poisson
    params.stretch_floor = 1.0e-3
    return params


def _run_mode(mode: str, num_frames: int) -> Example:
    """Instantiate and step the midsole example headlessly for ``num_frames``."""

    class _Args:
        pass

    args = _Args()
    args.mode = mode
    args.manifest = MANIFEST
    args.fit = None
    args.eager = False
    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    example = Example(viewer, args)
    for _ in range(num_frames):
        example.step()
    return example


class TestFoundationGeometry(unittest.TestCase):
    def test_column_bed(self):
        """Sample the whole midsole with valid clearances, thicknesses, neighbours, and a driven subset.

        The bed spans every column of the manifest grid, so the only negative
        neighbour code left is the natural outer boundary; an interior gap code
        would mean the bed had dropped material the shear layer needs.
        """
        geo = dynamics.build_foundation_geometry(MANIFEST)
        grid = build_column_grid(load_mesh(geo.midsole_mesh_path, 0.001), geo.spacing_m)
        column_count = len(geo.slack_m)
        self.assertEqual(column_count, len(grid.slack_m))
        self.assertTrue(np.all(geo.slack_m > 0.0))
        self.assertTrue(np.all(geo.gap0_m >= 0.0))
        self.assertTrue(np.all(geo.z_free_m > geo.z_bottom_m))
        self.assertEqual(geo.neighbors.shape, (column_count, 4))
        self.assertTrue(np.all(geo.neighbors < column_count))
        self.assertTrue(np.all(geo.neighbors >= -1))
        # The indenter drives a strict, nonempty subset; the rest is free foam.
        self.assertEqual(geo.driven.shape, (column_count,))
        self.assertGreater(int(geo.driven.sum()), 0)
        self.assertLess(int(geo.driven.sum()), column_count)
        # Columns the indenter misses are anchored on their own foam top.
        self.assertTrue(np.all(geo.gap0_m[~geo.driven] == 0.0))

    def test_zero_poisson_hyperfoam_matches_the_host_law_on_every_device(self):
        """Evaluate the Hyperfoam law at zero Poisson ratio on CPU and GPU without a degenerate power.

        At ``nu = 0`` the volumetric exponent is one and ``beta`` is zero, so the
        volumetric factor is ``pow(x, 0)``. The stretch floor keeps ``x`` strictly
        positive, so that is exactly one and no ``pow(0, 0)`` or division by zero
        can appear, including past full densification.
        """
        material = core.CALIBRATED_MATERIAL
        params = _probe_params(material)
        strain = np.ascontiguousarray(np.linspace(0.0, 1.2, 25), np.float32)
        expected = core._hyperfoam_pressure(strain.astype(np.float64), material)

        self.assertEqual(params.beta, 0.0)
        self.assertEqual(params.one_minus_two_poisson, 1.0)
        for device in _available_devices():
            with self.subTest(device=str(device)):
                pressure = wp.zeros(len(strain), dtype=wp.float32, device=device)
                wp.launch(
                    _hyperfoam_probe,
                    dim=len(strain),
                    inputs=[wp.array(strain, dtype=wp.float32, device=device), params],
                    outputs=[pressure],
                    device=device,
                )
                values = pressure.numpy().astype(np.float64)
                self.assertTrue(np.all(np.isfinite(values)))
                np.testing.assert_allclose(values, expected, rtol=1.0e-5)


class TestMidsoleExample(unittest.TestCase):
    def test_instron_hysteresis(self):
        """Run the digital Instron mode and audit its dissipative hysteresis loop."""
        example = _run_mode("instron", 190)
        example.test_final()

    def test_settle_supports_mass_with_friction(self):
        """Run the massive-midsole mode and audit stable weight support and lateral grip."""
        example = _run_mode("settle", 150)
        example.test_final()

    def test_stride_ground_reaction(self):
        """Run the synthetic-stride mode and audit its ground-reaction force and center-of-pressure roll."""
        example = _run_mode("stride", 40)
        example.test_final()

    def test_attached_dynamic_stride(self):
        """Run the attached, foot-mounted dynamic shoe and audit its stance/flight ground reaction.

        The dynamic shoe carries mass and inertia and stays coupled to the foot the whole
        stride, so the run must remain finite (no flight blow-up), develop a real stance
        ground reaction, and unload to near zero in flight without the shoe separating.
        """
        example = _run_mode("attached", 95)
        example.test_final()

    @unittest.skipUnless(
        wp.is_cuda_available() and (FOAM_SWEEP / "fit" / "puma-fast-r-nitro-elite-3").is_dir(),
        "needs CUDA and the fitted FR3 shoe",
    )
    def test_fitted_shoe_scene(self):
        """Press a fitted shoe through one measured validate cycle and audit its finite, tracking load."""

        class _Args:
            shoe = ("DigitalInstron/puma-fast-r-nitro-elite-3",)
            stance_ms = 200
            target_n = 2800
            slow_motion = 1.0

        example = ShoeInstron(newton.viewer.ViewerNull(num_frames=30), _Args())
        for _ in range(30):
            example.step()
            example.render()
        example.test_final()


if __name__ == "__main__":
    unittest.main()
