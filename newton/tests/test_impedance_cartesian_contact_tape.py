# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check out-of-place carried contact against the shared forward foundation."""

import unittest

import numpy as np
import warp as wp

from newton.tests.test_impedance_cartesian_gpu_fusion import _pair
from projects.impedance_instron.cartesian.gpu.adjoint_contact import ContactTape

wp.set_module_options({"enable_backward": True, "fuse_fp": True})


@wp.kernel
def _pose(p: wp.array[wp.vec3], q: wp.array[wp.transform]):
    """Map translation and planar angle into a unit carrier transform."""
    w = wp.tid()
    v = p[w]
    angle = v[2] * 0.5
    q[w] = wp.transform(wp.vec3(v[0], 0.0, v[1]), wp.quat(0.0, -wp.sin(angle), 0.0, wp.cos(angle)))


@wp.kernel
def _wrench_loss(wrench: wp.array[wp.spatial_vector], loss: wp.array[wp.float64]):
    """Seed every wrench component with a diagnostic scalar."""
    w = wp.tid()
    value = wrench[w]
    total = wp.float64(0.0)
    for j in range(6):
        total += wp.float64(value[j]) * wp.float64(j + 1) * wp.float64(0.001)
    wp.atomic_add(loss, 0, total)


class TestContactTape(unittest.TestCase):
    """Preserve contact histories and obtain finite branch-stable derivatives."""

    @classmethod
    def setUpClass(cls):
        """Initialize CPU and available CUDA test devices."""
        wp.init()
        cls.devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            cls.devices.append(wp.get_device("cuda:0"))

    def test_contact_history_and_wrench_parity(self):
        """Match the original fixed-order wrench and all recurrent state over multiple steps."""
        for device in self.devices:
            for columns, sweeps in ((31, 3), (910, 8)):
                with self.subTest(device=device, columns=columns, sweeps=sweeps):
                    (source, carrier), _ = _pair(columns, device, sweeps)
                    tape = ContactTape(source, 3, 6.25e-5)
                    tape.states[0].copy_from(source)
                    for t in range(3):
                        source.apply(carrier, tape.dt, clear_body_force=True)
                        out = tape.apply(t, carrier.body_q, carrier.body_qd)
                        for name in ContactTape.State.FIELDS:
                            np.testing.assert_array_equal(
                                getattr(tape.states[t + 1], name).numpy(), getattr(source, name).numpy(), err_msg=name
                            )
                        for name in ("compression", "base_pressure", "column_force", "ground_force", "contact_point"):
                            np.testing.assert_array_equal(
                                getattr(out, name).numpy(), getattr(source, name).numpy(), err_msg=name
                            )
                        np.testing.assert_array_equal(out.body_f.numpy(), carrier.body_f.numpy(), err_msg="wrench")

    def test_pose_gradient_through_contact_history(self):
        """Match pose directional differences through three complete contact updates."""
        for device in self.devices:
            with self.subTest(device=device):
                (source, carrier), _ = _pair(31, device, 3)
                parameters = np.array(
                    [[0.0, -0.004, 0.02], [0.0, -0.006, -0.01], [0.0, -0.008, 0.03]], dtype=np.float32
                )
                p = wp.array(parameters, dtype=wp.vec3, device=device, requires_grad=True)
                pose = wp.zeros(3, dtype=wp.transform, device=device, requires_grad=True)
                velocities = np.tile([0.01, 0.0, -0.002, 0.0, 0.02, 0.0], (3, 1)).astype(np.float32)
                velocity = wp.array(velocities, dtype=wp.spatial_vector, device=device, requires_grad=True)
                wp.launch(_pose, dim=3, inputs=[p, carrier.body_q], device=device)
                carrier.body_qd.assign(velocities)
                for _ in range(16):
                    source.apply(carrier, 6.25e-5, clear_body_force=True)
                contact = ContactTape(source, 3, 6.25e-5)
                contact.states[0].copy_from(source)
                loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)

                def forward(loss=loss, p=p, pose=pose, device=device, contact=contact, velocity=velocity):
                    loss.zero_()
                    wp.launch(_pose, dim=3, inputs=[p, pose], device=device)
                    for t in range(3):
                        out = contact.apply(t, pose, velocity)
                        wp.launch(_wrench_loss, dim=3, inputs=[out.body_f, loss], device=device)

                with wp.Tape() as tape:
                    forward()
                tape.backward(loss)
                gradient = p.grad.numpy()
                self.assertTrue(np.isfinite(gradient).all(), gradient)
                self.assertTrue(np.isfinite(velocity.grad.numpy()).all())
                for name in ContactTape.State.FIELDS:
                    if name != "tangent_stuck":
                        self.assertTrue(np.isfinite(getattr(contact.states[0], name).grad.numpy()).all(), name)
                direction = np.array([[0.03, 0.7, 0.2], [-0.02, -0.4, 0.1], [0.01, 0.3, -0.15]], dtype=np.float32)
                direction /= np.linalg.norm(direction)
                predicted = float(np.sum(gradient.astype(np.float64) * direction))
                for eps in (1.0e-4, 3.0e-5):
                    p.assign(parameters + eps * direction)
                    forward()
                    plus = float(loss.numpy()[0])
                    p.assign(parameters - eps * direction)
                    forward()
                    minus = float(loss.numpy()[0])
                    actual = (plus - minus) / (2.0 * eps)
                    self.assertAlmostEqual(predicted, actual, delta=1e-3 + 5e-3 * abs(predicted))


if __name__ == "__main__":
    unittest.main()
