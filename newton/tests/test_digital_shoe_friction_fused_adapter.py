# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for FoundationFused compatibility with attached friction solvers."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.friction_adapter import FrictionAdapter
from projects.digital_shoe.friction_parameter_adapter import FrictionParameterAdapter
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, ShoeMaterial, SurroundConfig
from projects.impedance_instron.cartesian.gpu.foundation import FoundationFused


class TestDigitalShoeFrictionFusedAdapter(unittest.TestCase):
    """Verify FoundationFused correctly delegates to attached friction adapters."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp once for testing."""
        wp.init()

    def _build_foundations(self, device):
        """Construct matching MidsoleFoundation and FoundationFused instances under load."""
        columns = 4
        anchors = np.zeros((columns, 3))
        anchors[:, 0] = np.arange(columns) * 0.0001
        anchors[:, 2] = -0.02
        rest = np.full(columns, 0.02)
        neighbors = np.full((columns, 4), -1, dtype=np.int32)
        material = ShoeMaterial(
            instantaneous_shear_modulus_pa=74000.0,
            hyperfoam_exponent=0.22,
            equilibrium_fraction=0.7,
            pasternak_n_per_m=1500.0,
        )
        driven = np.ones(columns, dtype=bool)

        config = FoundationConfig(
            ground_height_m=0.0,
            friction_stiffness=10000.0,
            friction=10.0,
            mu=0.8,
        )
        surround = SurroundConfig(driven=driven, sweeps=1, carrier_bond=True)

        midsole = MidsoleFoundation(
            anchors,
            np.zeros(columns),
            rest,
            np.full(columns, 1.0e-5),
            neighbors,
            0.005,
            material,
            np.arange(1),
            wp.zeros(1, dtype=wp.vec3, device=device),
            config,
            device,
            surround,
            world_count=1,
        )
        fused = FoundationFused(
            anchors,
            np.zeros(columns),
            rest,
            np.full(columns, 1.0e-5),
            neighbors,
            0.005,
            material,
            np.arange(1),
            wp.zeros(1, dtype=wp.vec3, device=device),
            config,
            device,
            surround,
            world_count=1,
        )
        return midsole, fused

    def _build_state(self, device):
        """Build a carrier state penetrating ground with tangential horizontal velocity."""
        return SimpleNamespace(
            body_q=wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity())], dtype=wp.transform, device=device
            ),
            body_qd=wp.array([wp.spatial_vector(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        )

    def test_friction_adapter_deflection_mode(self):
        """Verify FrictionAdapter in deflection mode modifies tangential force and matches MidsoleFoundation."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        device = wp.get_device("cuda:0")

        midsole, fused = self._build_foundations(device)
        mobility = wp.array(
            [np.diag([0.5, 0.5, 0.5, 5.0, 4.0, 8.0]).astype(np.float32)], dtype=wp.spatial_matrix, device=device
        )

        adapter_midsole = FrictionAdapter(midsole, mobility, mode="deflection")
        adapter_fused = FrictionAdapter(fused, mobility, mode="deflection")

        self.assertFalse(fused.fused_diagnostics)

        state_midsole = self._build_state(device)
        state_fused = self._build_state(device)

        dt = 0.001
        midsole.apply(state_midsole, dt, clear_body_force=True)
        fused.apply(state_fused, dt, clear_body_force=True)

        force_fused = state_fused.body_f.numpy()[0]
        force_midsole = state_midsole.body_f.numpy()[0]

        # Verify force actually changes and tangential force is active
        self.assertLess(force_fused[0], -1.0)

        # Fused foundation with adapter must match MidsoleFoundation exactly
        np.testing.assert_allclose(force_fused, force_midsole, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fused.ground_force.numpy(), midsole.ground_force.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            adapter_fused.deflection.numpy(), adapter_midsole.deflection.numpy(), rtol=1e-5, atol=1e-5
        )

        # Normal fields (Z force, compression, etc.) remain unchanged and positive
        self.assertGreater(force_fused[2], 5.0)
        np.testing.assert_allclose(fused.compression.numpy(), midsole.compression.numpy(), rtol=1e-5, atol=1e-5)

    def test_friction_parameter_adapter_maxwell_mode(self):
        """Verify FrictionParameterAdapter in Maxwell mode modifies tangential force and matches MidsoleFoundation."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        device = wp.get_device("cuda:0")

        midsole, fused = self._build_foundations(device)

        adapter_midsole = FrictionParameterAdapter(midsole, world_count=1)
        adapter_fused = FrictionParameterAdapter(fused, world_count=1)

        # Method 7: Maxwell tangential bristle (viscous_ratio must be 0.0)
        cand_maxwell = [7.0, 0.8, 1.0, 1.0, 0.0, 0.0005, 0.0, 0.8, 0.1, 1e12, 0.001, 0.005]
        params = np.array([cand_maxwell], dtype=np.float32)
        adapter_midsole.set_parameters(params)
        adapter_fused.set_parameters(params)

        self.assertFalse(fused.fused_diagnostics)

        state_midsole = self._build_state(device)
        state_fused = self._build_state(device)

        dt = 0.001
        midsole.apply(state_midsole, dt, clear_body_force=True)
        fused.apply(state_fused, dt, clear_body_force=True)

        force_fused = state_fused.body_f.numpy()[0]
        force_midsole = state_midsole.body_f.numpy()[0]

        self.assertLess(force_fused[0], -1.0)

        np.testing.assert_allclose(force_fused, force_midsole, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fused.ground_force.numpy(), midsole.ground_force.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            adapter_fused.maxwell_force.numpy(), adapter_midsole.maxwell_force.numpy(), rtol=1e-5, atol=1e-5
        )

        # Normal fields remain unchanged
        self.assertGreater(force_fused[2], 5.0)
        np.testing.assert_allclose(fused.compression.numpy(), midsole.compression.numpy(), rtol=1e-5, atol=1e-5)

    def test_default_fused_behavior_preserved_without_adapter(self):
        """Preserve default fused behavior and diagnostics when no adapter is attached."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        device = wp.get_device("cuda:0")

        _midsole, fused = self._build_foundations(device)
        self.assertIsNone(fused.friction_solver)

        fused.diagnostics = (
            wp.zeros(fused.column_count, dtype=wp.float64, device=device),
            0.85,
            wp.zeros((1, 1), dtype=wp.vec2d, device=device),
            wp.zeros((1, 1), dtype=wp.int32, device=device),
            wp.zeros((1, 1), dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
        )
        self.assertTrue(fused.fused_diagnostics)

        state = self._build_state(device)
        fused.apply(state, 0.001, clear_body_force=True)
        self.assertGreater(state.body_f.numpy()[0, 2], 5.0)


if __name__ == "__main__":
    unittest.main()
