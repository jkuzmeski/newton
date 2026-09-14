# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for attached-shoe drawing and external ground-wrench accounting.

Use a synthetic two-column bed, not fitted assets or a controller rollout. The
passive column is attached laterally; these tests do not assert a literal rigid
last bond or validate the reduced foam model as a continuum.
"""

import unittest

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, ShoeMaterial, SurroundConfig
from projects.impedance_instron.simple.example import _columns

DT_S = 1.0e-4


def _pair(
    *,
    friction=False,
    viscoelastic=False,
    ground_height=0.0,
    device="cpu",
    reference_height=None,
    driven=(1, 0),
    carrier_bond=True,
):
    """Build the audit's driven/passive pair without external artifacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    carrier = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
    model = builder.finalize(device=device)
    config = FoundationConfig(
        friction_stiffness=100.0 if friction else 0.0,
        mu=0.6 if friction else 0.0,
        ground_height_m=ground_height,
    )
    foundation = MidsoleFoundation(
        anchor_local=np.array([[0.0, 0.0, -0.002], [0.005, 0.0, -0.002]], dtype=np.float32),
        z_free=np.full(2, ground_height if reference_height is None else reference_height),
        rest_len=np.full(2, 0.02),
        area=np.full(2, 0.005**2),
        neighbors=np.array([[1, -1, -1, -1], [0, -1, -1, -1]], dtype=np.int32),
        spacing_m=0.005,
        material=ShoeMaterial(
            instantaneous_shear_modulus_pa=10000.0,
            hyperfoam_exponent=2.0,
            equilibrium_fraction=0.6 if viscoelastic else 1.0,
            pasternak_n_per_m=200.0,
        ),
        carrier_body=carrier,
        body_com=model.body_com,
        config=config,
        device=device,
        surround=SurroundConfig(driven=np.array(driven), sweeps=32, relaxation_time_s=0.0, carrier_bond=carrier_bond),
    )
    return model, model.state(), foundation


def _pose(state, *, x=0.0, z=0.0, pitch=0.0):
    """Set a stationary carrier pose without updating contact history."""
    state.body_q.assign(np.array([[x, 0.0, z, 0.0, np.sin(pitch / 2.0), 0.0, np.cos(pitch / 2.0)]], np.float32))
    state.body_qd.zero_()


def _nominal(state, foundation):
    """Return the undeformed carrier-attached endpoints in world space."""
    q = state.body_q.numpy()[0]
    pitch = 2.0 * np.arctan2(q[4], q[6])
    c, s = np.cos(pitch), np.sin(pitch)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    anchors = foundation.anchor_local.numpy()
    upper = anchors.copy()
    upper[:, 2] += foundation.rest_len.numpy()
    return anchors @ rotation.T + q[:3], upper @ rotation.T + q[:3]


def _draw(state, foundation):
    """Call the same production drawing kernel as the runnable example."""
    bottoms = wp.zeros(foundation.column_count, dtype=wp.vec3, device=foundation.device)
    tops = wp.zeros_like(bottoms)
    wp.launch(
        _columns,
        dim=foundation.column_count,
        inputs=[
            state.body_q,
            foundation.anchor_local,
            foundation.rest_len,
            foundation.compression,
            foundation.driven,
            bottoms,
            tops,
        ],
        device=foundation.device,
    )
    return bottoms.numpy(), tops.numpy()


def _settle(state, foundation):
    """Relax the fixed-pose pair before testing its external wrench."""
    for _ in range(20):
        foundation.apply(state, DT_S, clear_body_force=True)


class TestImpedancePassiveAttachment(unittest.TestCase):
    def assert_flight(self, state, foundation):
        """Require zero contact force and nominal airborne endpoints."""
        np.testing.assert_allclose(foundation.compression.numpy(), 0.0, atol=1.0e-8)
        np.testing.assert_allclose(foundation.column_force.numpy(), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(foundation.ground_force.numpy(), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(foundation.resultant_force.numpy(), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(foundation.resultant_moment_origin.numpy(), 0.0, atol=1.0e-8)
        np.testing.assert_allclose(state.body_f.numpy(), 0.0, atol=1.0e-7)
        for actual, expected in zip(_draw(state, foundation), _nominal(state, foundation), strict=True):
            np.testing.assert_allclose(actual, expected, atol=2.0e-7, rtol=1.0e-6)
            self.assertGreater(float(actual[:, 2].min()), 0.0)

    def test_stationary_flight_survives_contact_evaluation(self):
        """Keep passive endpoints attached after an airborne force evaluation."""
        _, state, foundation = _pair()
        _pose(state, z=0.03, pitch=-0.4)
        before = _draw(state, foundation)
        foundation.apply(state, DT_S, clear_body_force=True)
        self.assert_flight(state, foundation)
        for actual, expected in zip(_draw(state, foundation), before, strict=True):
            np.testing.assert_allclose(actual, expected, atol=2.0e-7, rtol=1.0e-6)

    def test_cold_flight_follows_lift_and_pitch(self):
        """Preserve world clearance after lifting and pitching a cold bed."""
        _, state, foundation = _pair(friction=True)
        _pose(state, x=0.012, z=0.1, pitch=0.55)
        foundation.apply(state, DT_S, clear_body_force=True)
        self.assert_flight(state, foundation)

    def test_warm_flight_releases_contact_and_follows_carrier(self):
        """Release a compressed viscoelastic bed without pinning its drawing."""
        _, state, foundation = _pair(friction=True, viscoelastic=True)
        _settle(state, foundation)
        self.assertTrue(np.all(foundation.compression.numpy() > 0.0))
        self.assertGreater(float(np.abs(foundation.q_state.numpy()).max()), 0.0)
        _pose(state, x=0.001)
        foundation.apply(state, DT_S, clear_body_force=True)
        self.assertGreater(float(np.abs(foundation.column_force.numpy()[:, :2]).max()), 0.0)
        _pose(state, x=0.015, z=0.1, pitch=-0.55)
        foundation.apply(state, DT_S, clear_body_force=True)
        self.assert_flight(state, foundation)
        # Rendering sees post-integration poses with previous contact arrays.
        _pose(state, x=0.018, z=0.12, pitch=0.3)
        self.assert_flight(state, foundation)

    def test_render_does_not_advance_histories(self):
        """Leave Maxwell, compression, bristle, and body state unchanged when drawing."""
        _, state, foundation = _pair(friction=True, viscoelastic=True)
        _settle(state, foundation)
        names = (
            "compression",
            "base_pressure",
            "z_free",
            "q_state",
            "peq_prev",
            "surround_compression",
            "surround_rate",
            "tangent_anchor",
            "tangent_stuck",
            "tangent_dwell",
            "column_force",
        )
        before = {name: getattr(foundation, name).numpy().copy() for name in names}
        body_before = {name: getattr(state, name).numpy().copy() for name in ("body_q", "body_qd", "body_f")}
        _draw(state, foundation)
        _draw(state, foundation)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(foundation, name).numpy(), values, err_msg=name)
        for name, values in body_before.items():
            np.testing.assert_array_equal(getattr(state, name).numpy(), values, err_msg=name)

    def test_pitched_bristle_segments_do_not_exceed_rest_length(self):
        """Bound Euclidean segment length for pitched and offset bristle endpoints."""
        _, state, foundation = _pair(friction=True)
        for pitch in (-0.6, 0.6):
            for offset in (-0.002, 0.0, 0.002):
                with self.subTest(pitch=pitch, offset=offset):
                    _pose(state, z=0.0014, pitch=pitch)
                    _settle(state, foundation)
                    nominal, _ = _nominal(state, foundation)
                    foundation.tangent_anchor.assign((nominal[:, :2] + [offset, 0.0]).astype(np.float32))
                    foundation.tangent_stuck.assign(np.ones(2, dtype=np.int32))
                    foundation.column_force.assign(np.array([[0.0, 0.0, 1.0]] * 2, dtype=np.float32))
                    bottom, top = _draw(state, foundation)
                    self.assertTrue(np.isfinite(bottom).all() and np.isfinite(top).all())
                    self.assertTrue(np.all(bottom[:, 2] >= -1.0e-7))
                    length = np.linalg.norm(top - bottom, axis=1)
                    self.assertTrue(np.all(length <= foundation.rest_len.numpy() + 2.0e-7), (length, pitch, offset))

    def test_partial_contact_keeps_passive_compression_unilateral(self):
        """Keep bonded passive compression inside its current rigid-penetration bound."""
        _, state, foundation = _pair()
        for pitch in (-0.6, 0.6):
            with self.subTest(pitch=pitch):
                _pose(state, z=0.0014, pitch=pitch)
                _settle(state, foundation)
                nominal, _ = _nominal(state, foundation)
                compression = foundation.compression.numpy()
                self.assertTrue(np.all(compression >= 0.0))
                self.assertLessEqual(float(compression[1]), max(-float(nominal[1, 2]), 0.0) + 1.0e-7)
                ground = foundation.ground_force.numpy()
                transfer = foundation.column_force.numpy()
                self.assertTrue(np.all(ground[:, 2] >= 0.0))
                np.testing.assert_allclose(ground[nominal[:, 2] > 0.0], 0.0, atol=1.0e-7)
                # Internal shear can pull a local top without creating floor suction.
                np.testing.assert_allclose(transfer.sum(axis=0), ground.sum(axis=0), atol=1.0e-7)
                if pitch < 0.0:
                    self.assertLess(float(transfer[1, 2]), 0.0)

    def test_two_column_ground_cop_and_normal_wrench(self):
        """Reduce ground pressure rather than redistributed top loads into COP and moment."""
        _, state, foundation = _pair()
        _settle(state, foundation)
        reaction = np.maximum(foundation.base_pressure.numpy(), 0.0) * foundation.area.numpy()
        self.assertTrue(np.all(reaction > 0.01))
        expected_cop = float(np.dot(reaction, [0.0, 0.005]) / reaction.sum())
        expected_moment = np.array([0.0, -reaction[1] * 0.005, 0.0])
        self.assertAlmostEqual(float(foundation.normal_force.numpy()[0]), float(reaction.sum()), delta=1.0e-7)
        self.assertAlmostEqual(
            float(foundation.cop_moment.numpy()[0, 0] / foundation.pressed_force.numpy()[0]),
            expected_cop,
            delta=1.0e-7,
        )
        np.testing.assert_allclose(foundation.resultant_moment_origin.numpy()[0], expected_moment, atol=1.0e-8)
        np.testing.assert_allclose(state.body_f.numpy()[0, 3:], expected_moment, atol=1.0e-8)

    def test_two_column_friction_uses_local_ground_pressure(self):
        """Give both ground-loaded columns their own pressure-based Coulomb capacity."""
        _, state, foundation = _pair(friction=True)
        _settle(state, foundation)
        _pose(state, x=0.002)
        foundation.apply(state, DT_S, clear_body_force=True)
        reaction = np.maximum(foundation.base_pressure.numpy(), 0.0) * foundation.area.numpy()
        force = foundation.ground_force.numpy()
        expected = -0.6 * reaction
        np.testing.assert_allclose(force[:, 0], expected, atol=1.0e-7, rtol=1.0e-5)
        np.testing.assert_allclose(force[:, 2], reaction, atol=1.0e-7, rtol=1.0e-5)
        self.assertTrue(np.all(np.linalg.norm(force[:, :2], axis=1) <= 0.6 * reaction + 1.0e-7))

    def test_friction_moment_uses_ground_plane_height(self):
        """Apply the full external friction wrench at the actual ground plane."""
        _, state, foundation = _pair(friction=True)
        _settle(state, foundation)
        _pose(state, x=0.002)
        foundation.apply(state, DT_S, clear_body_force=True)
        reaction = np.maximum(foundation.base_pressure.numpy(), 0.0) * foundation.area.numpy()
        force = np.column_stack([-0.6 * reaction, np.zeros(2), reaction])
        point = np.array([[0.002, 0.0, 0.0], [0.007, 0.0, 0.0]])
        expected_force = force.sum(axis=0)
        expected_moment = np.cross(point, force).sum(axis=0)
        np.testing.assert_allclose(foundation.resultant_force.numpy()[0], expected_force, atol=1.0e-7)
        np.testing.assert_allclose(foundation.resultant_moment_origin.numpy()[0], expected_moment, atol=1.0e-8)
        com = state.body_q.numpy()[0, :3]
        expected_torque = expected_moment - np.cross(com, expected_force)
        np.testing.assert_allclose(state.body_f.numpy()[0, :3], expected_force, atol=1.0e-7)
        np.testing.assert_allclose(state.body_f.numpy()[0, 3:], expected_torque, atol=1.0e-8)

    def test_elevated_plane_preserves_contact_torque_and_power(self):
        """Use the declared plane height for contact points, carrier torque, and power."""
        height = 0.07
        _, state, foundation = _pair(friction=True, ground_height=height)
        _pose(state, z=height)
        _settle(state, foundation)
        _pose(state, x=0.002, z=height)
        velocity = np.array([[0.1, 0.0, 0.02, 0.0, 2.0, 0.0]], dtype=np.float32)
        state.body_qd.assign(velocity)
        foundation.apply(state, DT_S, clear_body_force=True)
        reaction = np.maximum(foundation.base_pressure.numpy(), 0.0) * foundation.area.numpy()
        force = np.column_stack([-0.6 * reaction, np.zeros(2), reaction])
        point = np.array([[0.002, 0.0, height], [0.007, 0.0, height]])
        expected_force = force.sum(axis=0)
        expected_moment = np.cross(point, force).sum(axis=0)
        com = state.body_q.numpy()[0, :3]
        expected_torque = np.cross(point - com, force).sum(axis=0)
        expected_power = np.dot(expected_force, velocity[0, :3]) + np.dot(expected_torque, velocity[0, 3:])
        np.testing.assert_allclose(foundation.contact_point.numpy(), point, atol=1.0e-8)
        np.testing.assert_allclose(foundation.resultant_force.numpy()[0], expected_force, atol=1.0e-7)
        np.testing.assert_allclose(foundation.resultant_moment_origin.numpy()[0], expected_moment, atol=1.0e-8)
        np.testing.assert_allclose(state.body_f.numpy()[0, 3:], expected_torque, atol=1.0e-8)
        self.assertAlmostEqual(float(foundation.contact_power.numpy()[0]), float(expected_power), delta=1.0e-8)

    def test_airborne_maxwell_memory_cannot_create_ground_traction(self):
        """Gate external traction by clearance even when Maxwell pressure remains positive."""
        _, state, foundation = _pair(friction=True, viscoelastic=True)
        _pose(state, z=0.03, pitch=-0.4)
        foundation.q_state.assign(np.full(2, 1000.0, dtype=np.float32))
        foundation.apply(state, DT_S, clear_body_force=True)
        self.assertTrue(np.all(foundation.base_pressure.numpy() > 0.0))
        self.assert_flight(state, foundation)

    def test_plane_rejects_mismatched_pressure_reference(self):
        """Reject pressure-reference heights that do not represent the declared plane."""
        for ground_height, reference_height in ((0.0, 0.001), (0.07, 0.0), (0.0, [0.0, 0.001])):
            with self.subTest(ground_height=ground_height, reference_height=reference_height):
                with self.assertRaisesRegex(ValueError, "initial z_free must equal ground_height_m"):
                    _pair(ground_height=ground_height, reference_height=reference_height)

    def test_plane_rejects_unbonded_passive_surround(self):
        """Reject a free bench surround in the external ground-contact mode."""
        with self.assertRaisesRegex(ValueError, "passive columns require surround.carrier_bond=True"):
            _pair(carrier_bond=False)

    def test_plane_accepts_all_driven_unbonded_bed(self):
        """Allow all-driven columns without a passive carrier-bound constraint."""
        for height in (0.0, 0.07):
            with self.subTest(height=height):
                # Distinct float64 inputs may describe the same float32 plane.
                reference = np.nextafter(height, np.inf)
                _, state, foundation = _pair(
                    ground_height=height, reference_height=reference, driven=(1, 1), carrier_bond=False
                )
                _pose(state, z=height)
                foundation.apply(state, DT_S, clear_body_force=True)
                reaction = np.maximum(foundation.base_pressure.numpy(), 0.0) * foundation.area.numpy()
                self.assertTrue(np.all(reaction > 0.0))
                np.testing.assert_allclose(foundation.ground_force.numpy()[:, 2], reaction, atol=1.0e-7)
                np.testing.assert_allclose(
                    foundation.column_force.numpy(), foundation.ground_force.numpy(), atol=1.0e-7
                )

    def test_plane_height_requires_finite_float32(self):
        """Reject plane heights that cannot remain finite in device-side arithmetic."""
        for height in (np.nan, np.inf, -np.inf, 1.0e40, -1.0e40):
            with self.subTest(height=height):
                with self.assertRaisesRegex(ValueError, "ground_height_m must be finite in float32"):
                    FoundationConfig(ground_height_m=height)

    def test_cpu_cuda_and_graph_preserve_contact_history(self):
        """Match plane-contact histories through loading, slip, partial contact, and flight."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = devices[0]
        runs = [
            _pair(friction=True, viscoelastic=True),
            _pair(friction=True, viscoelastic=True, device=device),
            _pair(friction=True, viscoelastic=True, device=device),
        ]
        # Upload material/substep constants before capture, as required by the runtime.
        for _, state, foundation in runs:
            foundation.apply(state, DT_S, clear_body_force=True)
        _, captured_state, captured_foundation = runs[2]
        with wp.ScopedCapture(device) as capture:
            captured_foundation.apply(captured_state, DT_S, clear_body_force=True)
        fields = (
            "compression",
            "z_free",
            "q_state",
            "peq_prev",
            "surround_compression",
            "surround_rate",
            "ground_force",
            "contact_point",
            "column_force",
            "tangent_anchor",
            "tangent_stuck",
            "tangent_dwell",
            "normal_force",
            "cop_moment",
            "resultant_force",
            "resultant_moment_origin",
            "contact_power",
        )
        poses = (
            {"z": 0.0},
            {"x": 0.002},
            {"x": 0.002, "z": 0.0014, "pitch": 0.6},
            {"x": 0.015, "z": 0.1, "pitch": -0.4},
        )
        for pose in poses:
            with self.subTest(pose=pose):
                for _, state, _ in runs:
                    _pose(state, **pose)
                for _ in range(4):
                    for _, state, foundation in runs[:2]:
                        foundation.apply(state, DT_S, clear_body_force=True)
                    wp.capture_launch(capture.graph)
                snapshots = []
                for _, state, foundation in runs:
                    values = {name: getattr(foundation, name).numpy() for name in fields}
                    values["body_f"] = state.body_f.numpy()
                    values["bottoms"], values["tops"] = _draw(state, foundation)
                    snapshots.append(values)
                cpu, eager, graph = snapshots
                for name in cpu:
                    np.testing.assert_array_equal(graph[name], eager[name], err_msg=f"CUDA graph: {name}")
                    atol = 1.0e-4 if name in ("q_state", "peq_prev") else 1.0e-6
                    np.testing.assert_allclose(eager[name], cpu[name], rtol=2.0e-5, atol=atol, err_msg=name)
        for _, state, foundation in runs:
            self.assert_flight(state, foundation)


if __name__ == "__main__":
    unittest.main()
