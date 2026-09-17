# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check resident friction sweeps against independent force and metric calculations."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.contact import bristle_step
from projects.digital_shoe.friction_metrics import score_friction_trace
from projects.digital_shoe.friction_sweep import FrictionSweep


@wp.kernel
def _reference(
    steps: int,
    dt: float,
    pos: wp.array2d[wp.vec2],
    vel: wp.array2d[wp.vec2],
    normal: wp.array2d[float],
    kt: wp.array[float],
    kv: wp.array[float],
    output: wp.array2d[wp.vec2],
):
    c = wp.tid()
    anchor = wp.vec2(0.0)
    stuck = int(0)
    dwell = float(0.0)
    for t in range(steps):
        force, anchor, stuck, dwell = bristle_step(
            pos[t, c], vel[t, c], dt, normal[t, c], kt[c], kv[c], 0.8, 0.2, 0.0, anchor, stuck, dwell
        )
        output[t, c] = force


class TestFrictionSweep(unittest.TestCase):
    """Verify no physics or state leaks between candidates in the cached sweep."""

    @staticmethod
    def history():
        """Create a small coherent slip history with reversal and a contact dropout."""
        n, c, dt = 24, 3, 0.002
        t = np.arange(n) * dt
        v = np.zeros((n, c, 2), np.float32)
        v[:, :, 0] = np.cos(2 * np.pi * np.arange(n) / n)[:, None] * 0.1
        pos = np.zeros_like(v)
        pos[1:] = np.cumsum(v[:-1] * dt, axis=0)
        pos += np.array([[-0.02, 0], [0, 0], [0.02, 0]], np.float32)[None, :, :]
        normal = np.full((n, c), 40.0, np.float32)
        normal[8:10] = 0.0
        measured = np.column_stack((20 * np.sin(2 * np.pi * np.arange(n) / n), np.full(n, 120.0)))
        return {
            "time_s": t,
            "position_xy": pos,
            "velocity_xy": v,
            "nominal_velocity_xy": v.copy(),
            "normal_n": normal,
            "baseline_kt_n_m": np.array([1000, 1200, 1400], np.float32),
            "baseline_kv_ns_m": np.array([1, 2, 3], np.float32),
            "measured_time_s": t.copy(),
            "measured_force_n": measured,
        }

    def test_original_force_and_metrics(self):
        """Match the canonical bristle recurrence and official phase metrics on CPU and CUDA."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            h = self.history()
            sweep = FrictionSweep(h, 2, device=device)
            settings = np.array([[0, 0.8, 1.0, 1.0, 0.2, 0.0, 0.0], [1, 0.8, 1.0, 1.0, 0.2, 0.0, 0.0]])
            before = {k: v.copy() for k, v in h.items()}
            score, curves = sweep.evaluate(settings, curves=True)
            output = wp.zeros((24, 3), dtype=wp.vec2, device=device)
            wp.launch(
                _reference,
                dim=3,
                inputs=[24, 0.002, sweep.position, sweep.velocity, sweep.normal, sweep.kt, sweep.kv, output],
                device=device,
            )
            expected = output.numpy().sum(axis=1)
            np.testing.assert_allclose(curves[0], expected, atol=2e-5, rtol=2e-6)
            np.testing.assert_allclose(curves[1], expected, atol=2e-4, rtol=2e-5)
            reference = {"grf_time_s": h["measured_time_s"], "grf_target_n": h["measured_force_n"]}
            trace = {"time_s": h["time_s"], "grf_n": np.column_stack((curves[0, :, 0], h["normal_n"].sum(axis=1)))}
            metrics = score_friction_trace(reference, trace, 1)
            self.assertTrue(metrics["complete"])
            self.assertAlmostEqual(
                float(score[0, 1]), metrics["comparison_metrics"]["full_horizontal_force_rmse_n"], delta=2e-5
            )
            self.assertAlmostEqual(float(score[0, 2]), metrics["trace_metrics"]["braking_impulse_ns"], delta=2e-6)
            self.assertAlmostEqual(float(score[0, 3]), metrics["trace_metrics"]["propulsive_impulse_ns"], delta=2e-6)
            self.assertLess(float(score[1, 6]), 1e-5)
            self.assertLess(float(score[:, 8].max()), 1e-4)
            for key in h:
                np.testing.assert_array_equal(h[key], before[key])

    def test_batch_reset_and_graph_parity(self):
        """Keep world histories isolated and reset every candidate evaluation."""
        h = self.history()
        rows = np.array([[0, 0.4, 0.5, 0.0, 0.2, 0.0005, 0.0], [1, 0.8, 2.0, 0.5, 0.2, 0.001, 0.0]])
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            batch = FrictionSweep(h, 2, device=device)
            expected, curve = batch.evaluate(rows, curves=True)
            swapped = batch.evaluate(rows[::-1])
            np.testing.assert_array_equal(swapped, expected[::-1])
            np.testing.assert_array_equal(batch.evaluate(rows), expected)
            single = FrictionSweep(h, 1, device=device, use_graph=False)
            for i in range(2):
                isolated, isolated_curve = single.evaluate(rows[i : i + 1], curves=True)
                np.testing.assert_allclose(isolated[0], expected[i], rtol=1e-6, atol=1e-6)
                np.testing.assert_array_equal(isolated_curve[0], curve[i])
            if device.is_cuda:
                self.assertEqual(batch.graph_kind, "device_while")

    def test_full_rate_score_detects_between_sample_chatter(self):
        """Reject force oscillation hidden by sampling only every fourth simulation step."""
        from projects.digital_shoe.friction_sweep import _score  # noqa: PLC0415

        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            force = np.zeros((1, 17, 2), np.float32)
            force[0, 1::4, 0] = 20.0
            force[0, 3::4, 0] = -20.0
            curves = wp.array(force, dtype=wp.vec2, device=device)
            totals = wp.zeros(1, dtype=wp.vec4, device=device)
            scores = []
            for indices in (np.arange(0, 17, 4), np.arange(17)):
                n = len(indices)
                target = wp.zeros(n, dtype=float, device=device)
                times = wp.array(indices * 0.0001, dtype=float, device=device)
                active = wp.ones(n, dtype=int, device=device)
                index = wp.array(indices, dtype=int, device=device)
                fraction = wp.zeros(n, dtype=float, device=device)
                out = wp.zeros((1, 10), dtype=float, device=device)
                wp.launch(
                    _score,
                    dim=1,
                    inputs=[n, target, times, active, index, index, fraction, wp.vec4(0.0), 1.0, curves, totals, out],
                    device=device,
                )
                scores.append(out.numpy().copy())
            self.assertEqual(float(scores[0][0, 1]), 0.0)
            self.assertGreater(float(scores[1][0, 1]), 10.0)

    def test_invalid_candidates_and_clock(self):
        """Reject malformed settings and clocks before evaluating a candidate."""
        h = self.history()
        sweep = FrictionSweep(h, 2, device="cpu")
        for row in ([[1, 0.8, 0.0, 1.0, 0.2, 0, 0]], [[1, 0.8, 1, 1, 0.2, 0, 0.5]], [[1.5, 0.8, 1, 1, 0.2, 0, 0]]):
            with self.assertRaises(ValueError):
                sweep.evaluate(np.asarray(row))
        h["time_s"][2] = h["time_s"][1]
        with self.assertRaises(ValueError):
            FrictionSweep(h, 2, device="cpu")


if __name__ == "__main__":
    unittest.main()
