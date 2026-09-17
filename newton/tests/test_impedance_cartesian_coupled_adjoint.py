# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard replay correctness in the shared taped controller step."""

import unittest

import numpy as np
import warp as wp

from projects.impedance_instron.cartesian.gpu.adjoint import _advance_taped, _prepare_taped, _terminal_seed
from projects.impedance_instron.cartesian.gpu.engine import _Settings
from projects.impedance_instron.cartesian.gpu.mechanics import Params, Vec5

wp.set_module_options({"enable_backward": True, "fuse_fp": False})


@wp.kernel
def _carrier_seed(q: wp.array[wp.transform], v: wp.array[wp.spatial_vector], loss: wp.array[wp.float64]):
    """Seed carrier translation and velocity to expose a missing contact-state gradient path."""
    p = wp.transform_get_translation(q[0])
    loss[0] = (
        wp.float64(0.75) * wp.float64(p[0])
        + wp.float64(0.2) * wp.float64(p[2])
        + wp.float64(0.25) * wp.float64(v[0][0])
    )


def _fixture(device):
    """Create a valid leg whose zero-equilibrium replay would fail the force screen."""
    p = Params()
    p.lengths = wp.vec2d(0.4, 0.43)
    p.masses = wp.vec3d(7.0, 3.0, 0.7)
    p.com0 = wp.vec2d(0.21, 0.02)
    p.com1 = wp.vec2d(0.16, -0.01)
    p.com2 = wp.vec2d(0.08, 0.02)
    p.inertias = wp.vec3d(0.09, 0.05, 0.003)
    cfg = _Settings()
    cfg.stiffness = wp.vec4d(10000.0, 10000.0, 80.0, 60.0)
    cfg.damping = wp.vec4d(20.0, 20.0, 2.0, 1.0)
    cfg.lower = wp.vec2d(-4.0, -4.0)
    cfg.upper = wp.vec2d(4.0, 4.0)
    cfg.dt, cfg.gravity, cfg.pitch = 6.25e-5, 9.81, 0.0
    cfg.hip_floor, cfg.max_speed, cfg.max_force = 0.2, 100.0, 6000.0
    cfg.compression_limit, cfg.passive_cap = 0.9, 0.9
    cfg.joint_diagnostic, cfg.steps, cfg.controls = 1, 4, 12
    values = np.array([[[0.7, 0.93, -1.25, -0.73, 0.13]]])
    q = wp.array(values, dtype=Vec5, device=device, requires_grad=True)
    v = wp.zeros_like(q, requires_grad=True)
    coefficients = wp.array(
        np.tile(values[0, 0, [0, 1, 3, 4]], (1, 12, 1)), dtype=wp.float64, device=device, requires_grad=True
    )
    basis = np.zeros((5, 12))
    basis[:, 0] = 1.0
    return p, cfg, q, v, coefficients, wp.array(basis, dtype=wp.float64, device=device)


class TestCoupledAdjointReplay(unittest.TestCase):
    """Keep nonzero carrier derivatives and immutable primal state during reverse replay."""

    @classmethod
    def setUpClass(cls):
        """Initialize CPU and available CUDA devices."""
        wp.init()
        cls.devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            cls.devices.append(wp.get_device("cuda:0"))

    def test_prepare_reconstructs_equilibrium_before_screen(self):
        """Preserve carrier gradients when an omitted spline replay would falsely trigger a failure screen."""
        for device in self.devices:
            with self.subTest(device=device):
                p, cfg, q, v, coefficients, basis = _fixture(device)
                eq = wp.zeros((1, 1), dtype=wp.vec4d, device=device, requires_grad=True)
                act = wp.zeros_like(eq, requires_grad=True)
                pose = wp.zeros(1, dtype=wp.transform, device=device, requires_grad=True)
                twist = wp.zeros(1, dtype=wp.spatial_vector, device=device, requires_grad=True)
                failure = wp.zeros(1, dtype=int, device=device)
                failure_step = wp.full(1, -1, dtype=int, device=device)
                range_step = wp.full(1, -1, dtype=int, device=device)
                range_mask = wp.zeros(1, dtype=int, device=device)
                loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    wp.launch(
                        _prepare_taped,
                        dim=1,
                        inputs=[
                            0,
                            p,
                            cfg,
                            basis,
                            coefficients,
                            q,
                            v,
                            eq,
                            act,
                            pose,
                            twist,
                            failure,
                            failure_step,
                            range_step,
                            range_mask,
                        ],
                        device=device,
                        block_dim=1,
                    )
                    wp.launch(_carrier_seed, dim=1, inputs=[pose, twist, loss], device=device)
                before = pose.numpy().copy()
                tape.backward(loss)
                np.testing.assert_array_equal(pose.numpy(), before)
                np.testing.assert_array_equal(failure.numpy(), 0)
                self.assertAlmostEqual(q.grad.numpy()[0, 0, 0], 0.75, places=12)
                self.assertAlmostEqual(q.grad.numpy()[0, 0, 1], 0.2, places=7)
                self.assertAlmostEqual(v.grad.numpy()[0, 0, 0], 0.25, places=12)

    def test_reverse_does_not_increment_forward_counters(self):
        """Retain completed-step counters and primal states while differentiating the shared integration function."""
        for device in self.devices:
            with self.subTest(device=device):
                p, cfg, q, v, _, _ = _fixture(device)
                next_q = wp.zeros_like(q, requires_grad=True)
                next_v = wp.zeros_like(v, requires_grad=True)
                act = wp.zeros((1, 1), dtype=wp.vec4d, device=device, requires_grad=True)
                wrench = wp.zeros(1, dtype=wp.spatial_vector, device=device, requires_grad=True)
                maxima = wp.zeros((1, 1), dtype=wp.vec2d, device=device)
                caps = wp.zeros((1, 1), dtype=int, device=device)
                invalid = wp.zeros_like(caps)
                force = wp.zeros((1, 1), dtype=wp.vec2d, device=device)
                moment = wp.zeros((1, 1), dtype=wp.float64, device=device)
                fraction = wp.zeros((1, 1), dtype=wp.vec3d, device=device)
                cap_out = wp.zeros_like(caps)
                integrated = wp.zeros(1, dtype=int, device=device)
                recorded = wp.zeros_like(integrated)
                failure = wp.zeros_like(integrated)
                failure_step = wp.full(1, -1, dtype=int, device=device)
                loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    wp.launch(
                        _advance_taped,
                        dim=1,
                        inputs=[
                            0,
                            p,
                            cfg,
                            q,
                            v,
                            next_q,
                            next_v,
                            act,
                            wrench,
                            1,
                            maxima,
                            caps,
                            invalid,
                            force,
                            moment,
                            fraction,
                            cap_out,
                            integrated,
                            recorded,
                            failure,
                            failure_step,
                        ],
                        device=device,
                        block_dim=1,
                    )
                    wp.launch(_terminal_seed, dim=1, inputs=[next_q, next_v, loss], device=device)
                before = next_q.numpy().copy()
                tape.backward(loss)
                np.testing.assert_array_equal(next_q.numpy(), before)
                np.testing.assert_array_equal(integrated.numpy(), 1)
                np.testing.assert_array_equal(recorded.numpy(), 1)
                np.testing.assert_array_equal(failure.numpy(), 0)
                self.assertTrue(np.isfinite(q.grad.numpy()).all())


if __name__ == "__main__":
    unittest.main()
