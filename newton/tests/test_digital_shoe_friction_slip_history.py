# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check provisional loaded-slip weakening without output smoothing."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.friction_slip_history import bristle_slip_history_step


@wp.kernel
def _step(
    v: wp.array[wp.vec2],
    p: wp.array[float],
    z: wp.vec2,
    n: float,
    stuck: int,
    dwell: float,
    distance: float,
    force: wp.array[wp.vec2],
    out_z: wp.array[wp.vec2],
    metadata: wp.array[wp.vec3],
    loss: wp.array[float],
):
    f, next_z, active, elapsed, s = bristle_slip_history_step(
        v[0], 0.001, n, 10000.0, 0.0, p[0], p[1], p[2], 0.0, 0.0005, z, stuck, dwell, distance
    )
    force[0] = f
    out_z[0] = next_z
    metadata[0] = wp.vec3(float(active), elapsed, s)
    loss[0] = f[0] + 0.31 * f[1]


class TestFrictionSlipHistory(unittest.TestCase):
    """Validate passive state advance, release, and differentiability."""

    def _evaluate(
        self,
        device,
        vel,
        parameters=(1.2, 0.3, 0.001),
        z=(0.0, 0.0),
        normal=0.5,
        stuck=1,
        dwell=0.0,
        distance=0.0,
        grad=False,
    ):
        v = wp.array([vel], dtype=wp.vec2, device=device, requires_grad=grad)
        p = wp.array(parameters, dtype=float, device=device, requires_grad=grad)
        f = wp.zeros(1, dtype=wp.vec2, device=device, requires_grad=grad)
        oz = wp.zeros(1, dtype=wp.vec2, device=device, requires_grad=grad)
        m = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=grad)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=grad)
        if grad:
            tape = wp.Tape()
            with tape:
                wp.launch(
                    _step,
                    dim=1,
                    inputs=[v, p, wp.vec2(*z), normal, stuck, dwell, distance, f, oz, m, loss],
                    device=device,
                )
            tape.backward(loss=loss)
            return f.numpy()[0], oz.numpy()[0], m.numpy()[0], float(loss.numpy()[0]), p.grad.numpy(), v.grad.numpy()[0]
        wp.launch(
            _step, dim=1, inputs=[v, p, wp.vec2(*z), normal, stuck, dwell, distance, f, oz, m, loss], device=device
        )
        return f.numpy()[0], oz.numpy()[0], m.numpy()[0], float(loss.numpy()[0])

    def test_no_creep_or_unloaded_exposure(self):
        """Hold admissible static force and reset exposure on unloaded release."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            f, z, m, _ = self._evaluate(device, [0.0, 0.0], z=(1e-5, 0), distance=0.002)
            np.testing.assert_allclose(f, [-0.1, 0], atol=1e-7)
            self.assertAlmostEqual(float(m[2]), 0.002, places=8)
            f2, z2, _m2, _ = self._evaluate(device, [0.0, 0.0], z=z, distance=float(m[2]))
            np.testing.assert_array_equal(f2, f)
            np.testing.assert_array_equal(z2, z)
            f, z, m, _ = self._evaluate(device, [2.0, 0.0], normal=0.0, distance=0.002)
            np.testing.assert_array_equal(f, [0.0, 0.0])
            self.assertEqual(float(m[2]), 0.0)

    def test_passivity_and_cold_cone(self):
        """Respect cold friction bound and dissipate energy under changing loads."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            z = np.zeros(2)
            s = 0.0
            stuck = 0
            dwell = 0.0
            for normal, vel in [
                (1.0, [0.4, 0.1]),
                (4.0, [-0.3, 0.2]),
                (0.1, [0.1, -0.2]),
                (0.0, [2.0, 0.0]),
                (10.0, [0.0, 0.0]),
                (0.3, [0.2, 0.0]),
            ]:
                old = 0.5 * 10000 * np.dot(z, z)
                f, znew, m, _ = self._evaluate(device, vel, z=z, normal=normal, stuck=stuck, dwell=dwell, distance=s)
                energy = 0.5 * 10000 * np.dot(znew, znew)
                work = np.dot(f, vel) * 0.001
                self.assertLessEqual(energy - old + work, 1e-6 * (energy + old + abs(work) + 1e-5))
                self.assertLessEqual(np.linalg.norm(f), 1.2 * normal + 1e-6)
                z = znew
                s = float(m[2])
                stuck = int(m[0])
                dwell = float(m[1])

    def test_parameter_and_velocity_gradients(self):
        """Match Tape to finite differences away from slip and force branches."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            p = np.array([1.2, 0.3, 0.001])
            v = np.array([0.3, 0.1])
            result = self._evaluate(device, v, p, distance=0.0005, grad=True)
            for i, h in enumerate([1e-3, 1e-3, 1e-6]):
                high = p.copy()
                low = p.copy()
                high[i] += h
                low[i] -= h
                fd = (
                    self._evaluate(device, v, high, distance=0.0005)[3]
                    - self._evaluate(device, v, low, distance=0.0005)[3]
                ) / (2 * h)
                np.testing.assert_allclose(result[4][i], fd, rtol=0.015, atol=1e-4)
            for i in range(2):
                high = v.copy()
                low = v.copy()
                high[i] += 1e-4
                low[i] -= 1e-4
                fd = (
                    self._evaluate(device, high, p, distance=0.0005)[3]
                    - self._evaluate(device, low, p, distance=0.0005)[3]
                ) / 0.0002
                np.testing.assert_allclose(result[5][i], fd, rtol=0.015, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
