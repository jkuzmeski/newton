# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare cooperative GPU kernels with the unchanged scalar shoe law."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, ShoeMaterial, SurroundConfig
from projects.impedance_instron.cartesian.gpu.engine import _compression_partial
from projects.impedance_instron.cartesian.gpu.foundation import FoundationFused

_DT = 0.0000625
_FIELDS = (
    "surround_compression",
    "surround_scratch",
    "surround_previous",
    "surround_rate",
    "z_free",
    "q_state",
    "peq_prev",
    "compression",
    "column_force",
    "ground_force",
    "tangent_anchor",
)


def _pair(columns, device, sweeps):
    """Build three independent synthetic worlds with cross-tile neighbors."""
    rng = np.random.default_rng(columns)
    anchors = np.zeros((columns, 3))
    anchors[:, 0] = np.arange(columns) * 0.0001
    rest = rng.uniform(0.019, 0.023, columns)
    neighbors = np.full((columns, 4), -1, dtype=np.int32)
    neighbors[1:, 0] = np.arange(columns - 1)
    neighbors[:-1, 1] = np.arange(1, columns)
    driven = np.zeros(columns, dtype=bool)
    driven[::7] = True
    if columns > 511:
        driven[511] = True
    material = ShoeMaterial(
        instantaneous_shear_modulus_pa=74000.0,
        hyperfoam_exponent=0.22,
        equilibrium_fraction=0.7,
        pasternak_n_per_m=1500.0,
        instantaneous_shear_modulus_2_pa=18000.0,
        hyperfoam_exponent_2=3.1,
    )
    results = []
    for cls in (MidsoleFoundation, FoundationFused):
        state = SimpleNamespace(
            body_q=wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, -0.004 * (w + 1)), wp.quat_identity()) for w in range(3)],
                dtype=wp.transform,
                device=device,
            ),
            body_qd=wp.zeros(3, dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(3, dtype=wp.spatial_vector, device=device),
        )
        foundation = cls(
            anchors,
            np.zeros(columns),
            rest,
            np.full(columns, 1.0e-5),
            neighbors,
            0.005,
            material,
            np.arange(3),
            wp.zeros(3, dtype=wp.vec3, device=device),
            FoundationConfig(ground_height_m=0.0),
            device,
            SurroundConfig(driven=driven, sweeps=sweeps, max_strain=0.9, carrier_bond=True),
            world_count=3,
        )
        foundation.set_world_materials([replace(material, equilibrium_fraction=0.4 + w * 0.2) for w in range(3)])
        results.append((foundation, state))
    for name in ("surround_compression", "surround_scratch", "surround_previous", "q_state", "peq_prev"):
        values = rng.uniform(0.0, 0.005 if "surround" in name else 100.0, 3 * columns).astype(np.float32)
        for foundation, _ in results:
            getattr(foundation, name).assign(values)
    return results


class TestFoundationGpuFusion(unittest.TestCase):
    """Preserve material histories, world isolation, and graph replay semantics."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp once for CPU and optional CUDA comparisons."""
        wp.init()

    def _assert_pair(self, pair):
        """Compare every retained history and the carrier wrench exactly."""
        (original, state_a), (fused, state_b) = pair
        for name in _FIELDS:
            np.testing.assert_array_equal(getattr(fused, name).numpy(), getattr(original, name).numpy(), err_msg=name)
        np.testing.assert_array_equal(state_b.body_f.numpy(), state_a.body_f.numpy())

    def test_cpu_fallback(self):
        """Retain the original CPU relaxation implementation."""
        pair = _pair(31, wp.get_device("cpu"), 3)
        for foundation, state in pair:
            foundation.apply(state, _DT, clear_body_force=True)
        self._assert_pair(pair)

    def test_sweeps_and_world_histories(self):
        """Match odd and even sweeps across padding, tile boundaries, and large-bed fallback."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA is required")
        for columns in (31, 910, 1024, 1025):
            for sweeps in (1, 2, 3, 8):
                with self.subTest(columns=columns, sweeps=sweeps):
                    pair = _pair(columns, wp.get_device("cuda:0"), sweeps)
                    for _ in range(3):
                        for foundation, state in pair:
                            foundation.apply(state, _DT, clear_body_force=True)
                        self._assert_pair(pair)
                    if columns > 512:
                        compression = pair[1][0].surround_compression.numpy().reshape(3, columns)
                        self.assertTrue(np.all(compression[:, 512] > 0.0))

    def test_graph_reset_and_launch_count(self):
        """Replay all eight sweeps as one kernel without stale ping-pong histories."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA is required")
        device = wp.get_device("cuda:0")
        for sweeps in (3, 8):
            pair = _pair(910, device, sweeps)
            for foundation, state in pair:
                foundation.apply(state, _DT, clear_body_force=True)
            graphs = []
            for foundation, state in pair:
                with wp.ScopedCapture(device=device) as capture:
                    foundation.reset()
                    for _ in range(3):
                        foundation.apply(state, _DT, clear_body_force=True)
                graphs.append(capture.graph)
            for _ in range(2):
                for graph in graphs:
                    wp.capture_launch(graph)
                self._assert_pair(pair)
            fused, state = pair[1]
            with patch.object(wp, "launch_tiled", wraps=wp.launch_tiled) as tiled:
                fused.relax_surround(state, _DT)
            self.assertEqual(tiled.call_count, 1)


class TestCompressionPartialKernel(unittest.TestCase):
    """Keep all finite extrema, cap counts, and nonfinite failure flags."""

    def test_compression_partial_vs_numpy_float64(self):
        """Reduce every column exactly, including long beds and nonfinite boundary cases."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA is required")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(123)
        for columns in (1, 31, 910, 1024, 1025, 2101):
            for groups in (1, min(columns, 32)):
                with self.subTest(columns=columns, groups=groups):
                    rest = rng.uniform(0.015, 0.025, columns)
                    driven = rng.integers(0, 2, columns, dtype=np.int32)
                    compression = rng.uniform(-0.01, 0.03, (3, columns)).astype(np.float32)
                    if columns > 3:
                        compression[0, 0] = np.nan
                        compression[1, 1] = np.inf
                        compression[2, 2] = -np.inf
                        driven[3] = 0
                        rest[3] = 1.0
                        boundary = np.float32(0.85 - 1.0e-6)
                        compression[:, 3] = [
                            np.nextafter(boundary, np.float32(-np.inf)),
                            boundary,
                            np.nextafter(boundary, np.float32(np.inf)),
                        ]
                    fraction = compression.astype(np.float64) / rest
                    finite = np.isfinite(fraction)
                    expected_max = np.stack(
                        [
                            np.max(np.where(finite & (driven != 0), np.maximum(fraction, 0.0), 0.0), axis=1),
                            np.max(np.where(finite & (driven == 0), np.maximum(fraction, 0.0), 0.0), axis=1),
                        ],
                        axis=1,
                    )
                    expected_caps = np.sum(finite & (driven == 0) & (fraction >= 0.85 - 1.0e-6), axis=1)
                    maxima = wp.zeros((3, groups), dtype=wp.vec2d, device=device)
                    caps = wp.zeros((3, groups), dtype=int, device=device)
                    invalid = wp.zeros_like(caps)
                    wp.launch(
                        _compression_partial,
                        dim=(3, groups, 32),
                        block_dim=32,
                        inputs=[
                            groups,
                            columns,
                            wp.float64(0.85),
                            wp.array(compression.ravel(), dtype=wp.float32, device=device),
                            wp.array(rest, dtype=wp.float64, device=device),
                            wp.array(driven, dtype=int, device=device),
                            maxima,
                            caps,
                            invalid,
                        ],
                        device=device,
                    )
                    np.testing.assert_array_equal(maxima.numpy().max(axis=1), expected_max)
                    np.testing.assert_array_equal(caps.numpy().sum(axis=1), expected_caps)
                    np.testing.assert_array_equal(invalid.numpy().max(axis=1), np.any(~finite, axis=1))


if __name__ == "__main__":
    unittest.main()
