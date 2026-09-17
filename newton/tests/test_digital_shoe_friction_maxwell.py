# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify tangential mechanical relaxation and Coulomb return without output filtering."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.friction_maxwell import bristle_maxwell_step


@wp.kernel
def _eval(
    v: wp.array[wp.vec2],
    p: wp.array[float],
    h: float,
    n: float,
    z: wp.vec2,
    q: wp.vec2,
    outputs: wp.array[wp.vec2],
    jac: wp.array[wp.mat22],
    loss: wp.array[float],
):
    f, J, zn, qn, _s, _d = bristle_maxwell_step(v[0], h, n, p[0], p[1], p[2], p[3], 0.0, z, q, 1, 0.0)
    outputs[0] = f
    outputs[1] = zn
    outputs[2] = qn
    jac[0] = J
    loss[0] = f[0] + 0.2 * f[1]


class TestFrictionMaxwell(unittest.TestCase):
    """Test bounded stress, passive storage and force continuity."""

    def _step(self, device, v, dt=0.001, n=100.0, z=(0.0, 0.0), q=(0.0, 0.0), p=(1000.0, 10.0, 0.005, 0.8), grad=False):
        va = wp.array([v], dtype=wp.vec2, device=device, requires_grad=grad)
        pa = wp.array(p, dtype=float, device=device, requires_grad=grad)
        out = wp.zeros(3, dtype=wp.vec2, device=device, requires_grad=grad)
        jac = wp.zeros(1, dtype=wp.mat22, device=device)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=grad)
        if grad:
            tape = wp.Tape()
            with tape:
                wp.launch(_eval, dim=1, inputs=[va, pa, dt, n, wp.vec2(*z), wp.vec2(*q), out, jac, loss], device=device)
            tape.backward(loss=loss)
            return out.numpy(), jac.numpy()[0], float(loss.numpy()[0]), pa.grad.numpy(), va.grad.numpy()[0]
        wp.launch(_eval, dim=1, inputs=[va, pa, dt, n, wp.vec2(*z), wp.vec2(*q), out, jac, loss], device=device)
        return out.numpy(), jac.numpy()[0], float(loss.numpy()[0])

    def test_velocity_step_has_no_finite_force_jump(self):
        """Make the force response to a velocity step vanish with timestep."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            magnitudes = []
            for h in [0.001, 0.0005, 0.00025, 0.000125]:
                result = self._step(device, [0.1, 0.0], dt=h)[0]
                magnitudes.append(abs(result[0, 0]))
            self.assertTrue(np.all(np.diff(magnitudes) < 0))
            self.assertLess(magnitudes[-1], 0.2 * magnitudes[0])
            # A direct 10 N s/m parallel dashpot would jump by 1 N, independent of h.
            self.assertLess(magnitudes[-1], 0.04)

    def test_relaxation_has_a_physical_time_scale(self):
        """Converge to Maxwell relaxation in seconds rather than per-step creep."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            errors = []
            for steps in [10, 20, 40]:
                z = np.array([0.001, 0.0])
                q = np.array([0.5, 0.0])
                h = 0.005 / steps
                for _ in range(steps):
                    result = self._step(device, [0.0, 0.0], dt=h, z=z, q=q)[0]
                    z = result[1]
                    q = result[2]
                errors.append(abs(q[0] - 0.5 * np.exp(-1)))
            self.assertGreater(errors[0], errors[1])
            self.assertGreater(errors[1], errors[2])
            np.testing.assert_allclose(z, [0.001, 0.0], atol=1e-8)

    def test_passivity_with_changing_normal_and_reversal(self):
        """Bound traction and dissipate both elastic and Maxwell stored energy."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            z = np.array([0.0001, -0.0002])
            q = np.array([0.3, 0.1])
            p = (1000.0, 10.0, 0.005, 0.8)
            for n, v in [
                (10.0, [0.2, 0.1]),
                (0.2, [2.0, -0.5]),
                (1.0, [-0.1, 0.3]),
                (0.0, [0.5, 0.0]),
                (10.0, [0.0, 0.0]),
            ]:
                old = 0.5 * p[0] * np.dot(z, z) + 0.5 * p[2] * np.dot(q, q) / p[1]
                result = self._step(device, v, n=n, z=z, q=q, p=p)[0]
                f, z, q = result
                energy = 0.5 * p[0] * np.dot(z, z) + 0.5 * p[2] * np.dot(q, q) / p[1]
                work = np.dot(f, v) * 0.001
                self.assertLessEqual(energy - old + work, 1e-6 * (old + energy + abs(work) + 1e-5))
                self.assertLessEqual(np.linalg.norm(f), p[3] * n + 1e-6)

    def test_tangent_and_parameter_gradients(self):
        """Match force Jacobians and parameter Tape gradients to finite differences."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            for n in [100.0, 0.05]:
                p = np.array([1000.0, 10.0, 0.005, 0.8])
                v = np.array([0.2, 0.1])
                result = self._step(device, v, n=n, p=p, grad=True)
                for axis in range(2):
                    vh = v.copy()
                    vl = v.copy()
                    vh[axis] += 1e-4
                    vl[axis] -= 1e-4
                    high = self._step(device, vh, n=n, p=p)
                    low = self._step(device, vl, n=n, p=p)
                    fd = (high[0][0] - low[0][0]) / 0.0002
                    np.testing.assert_allclose(result[1][:, axis], fd, rtol=0.005, atol=1e-4)
                for axis, h in enumerate([0.1, 0.001, 1e-6, 1e-4]):
                    high = p.copy()
                    low = p.copy()
                    high[axis] += h
                    low[axis] -= h
                    fd = (self._step(device, v, n=n, p=high)[2] - self._step(device, v, n=n, p=low)[2]) / (2 * h)
                    np.testing.assert_allclose(result[3][axis], fd, rtol=0.01, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
