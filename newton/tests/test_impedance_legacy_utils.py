# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared legacy helpers without changing numeric or callback contracts."""

from __future__ import annotations

import importlib
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.impedance_instron.legacy import dashboard, env, example, explain, optimize, report, summary

# Capture these binary64 outputs and signatures before extracting shared helpers.
_GOLDEN = {
    "integrals": {
        "empty": {"time": [], "values": [], "hex": "0x0.0p+0"},
        "singleton": {"time": [0.0], "values": [7.0], "hex": "0x0.0p+0"},
        "irregular": {
            "time": [0.0, 0.01, 0.3, 0.30000003, 1.0],
            "values": [1000000000000000.0, 0.2, -30000.0, 1e-11, -9.0],
            "hex": "0x1.2309ce4fbfb85p+42",
        },
        "descending": {"time": [3.0, 2.0, 0.0], "values": [-2.0, 0.0, 4.0], "hex": "-0x1.8000000000000p+1"},
    },
    "momentum": {
        "none": {
            "times": [0.0, 0.01, 0.04, 0.09, 0.25, 0.9, 1.5],
            "velocity": [3.0, 4.0, -2.0, 5.0, 6.0, -4.0, 7.0],
            "loaded": [False, False, False, False, False, False, False],
            "hex": [
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
            ],
        },
        "one": {
            "times": [0.0, 0.01, 0.04, 0.09, 0.25, 0.9, 1.5],
            "velocity": [3.0, 4.0, -2.0, 5.0, 6.0, -4.0, 7.0],
            "loaded": [False, True, False, False, False, False, False],
            "hex": [
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
            ],
        },
        "all": {
            "times": [0.0, 0.01, 0.04, 0.09, 0.25, 0.9, 1.5],
            "velocity": [3.0, 4.0, -2.0, 5.0, 6.0, -4.0, 7.0],
            "loaded": [True, True, True, True, True, True, True],
            "hex": [
                "0x1.3000000000000p+1",
                "0x1.1d89d89d89d89p+1",
                "-0x1.3b13b13b13b60p-4",
                "-0x1.313b13b13b13cp+1",
                "-0x1.2c4ec4ec4ec50p+2",
                "-0x1.c000000000000p+2",
                "-0x1.0fffffffffffdp+2",
                "-0x1.7fffffffffff8p+0",
                "0x1.4000000000008p+0",
            ],
        },
        "gaps": {
            "times": [0.0, 0.01, 0.04, 0.09, 0.25, 0.9, 1.5],
            "velocity": [3.0, 4.0, -2.0, 5.0, 6.0, -4.0, 7.0],
            "loaded": [False, True, True, False, True, True, False],
            "hex": [
                "-0x1.e04e04e04e04dp+1",
                "-0x1.7297297297290p-2",
                "0x1.95a95a95a95a6p+0",
                "0x1.b91b91b91b910p-3",
                "-0x1.2762762762762p+0",
                "-0x1.42f42f42f42f4p+1",
                "-0x1.f237237237238p+1",
                "-0x1.50bd0bd0bd0bdp+2",
                "-0x1.a85e85e85e85ep+2",
            ],
        },
    },
    "signatures": {
        "env._momentum_checkpoints": "(times: 'np.ndarray', velocity: 'np.ndarray', loaded: "
        "'np.ndarray') -> 'list[float]'",
        "optimize._momentum": "(times: 'np.ndarray', velocity: 'np.ndarray', loaded: 'np.ndarray') -> 'list[float]'",
        "report._integral": "(time: numpy.ndarray, values: numpy.ndarray) -> float",
        "explain._integral": "(time: 'np.ndarray', values: 'np.ndarray') -> 'float'",
        "report._json_default": "(value)",
        "explain._json_default": "(value)",
    },
    "checkpoints": [0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6, 0.7000000000000001, 0.8, 0.9],
}


@wp.kernel
def _evaluate_pitch(rotation: wp.array[wp.quat], result: wp.array2d[float]):
    """Evaluate both old helper names on the same quaternion samples."""
    i = wp.tid()
    result[i, 0] = env._pitch_of(rotation[i])
    result[i, 1] = example._pitch_of(rotation[i])


class TestLegacySharedUtilities(unittest.TestCase):
    """Retain exact host outputs, old names, and module-local callback bindings."""

    def test_shared_helper_dispatch(self):
        """Delegate all duplicate computations to the one shared helper owner."""
        utils = importlib.import_module("projects.impedance_instron.legacy._utils")
        self.assertIs(report._json_default, utils.json_default)
        self.assertIs(explain._json_default, utils.json_default)
        for name, modules, method in (
            ("integral", (report, explain), "_integral"),
            ("escape", (report, explain, dashboard, summary), "_escape"),
        ):
            with self.subTest(helper=name):
                arguments = (np.array([0.0, 1.0]), np.array([2.0, 3.0])) if name == "integral" else (None,)
                with patch.object(utils, name, return_value="shared-result") as shared:
                    for module in modules:
                        self.assertEqual(getattr(module, method)(*arguments), "shared-result")
                    self.assertEqual(shared.call_count, len(modules))
        with patch.object(utils, "momentum_checkpoints", return_value=[123.0]) as shared:
            args = (np.array([0.0, 1.0]), np.array([1.0, 3.0]), np.array([True, True]))
            self.assertEqual(env._momentum_checkpoints(*args), [123.0])
            self.assertIs(shared.call_args.args[3], env.CHECKPOINTS)
            self.assertEqual(optimize._momentum(*args), [123.0])
            self.assertIs(shared.call_args.args[3], optimize.CHECKPOINTS)

    def test_original_signatures(self):
        """Keep argument names and evaluated or deferred annotations unchanged."""
        modules = {"env": env, "optimize": optimize, "report": report, "explain": explain}
        for name, expected in _GOLDEN["signatures"].items():
            module, symbol = name.split(".")
            with self.subTest(function=name):
                self.assertEqual(str(inspect.signature(getattr(modules[module], symbol))), expected)

    def test_integral_binary64_goldens(self):
        """Retain exact reduction results for empty, irregular and descending clocks."""
        for name, case in _GOLDEN["integrals"].items():
            for module in (report, explain):
                with self.subTest(case=name, module=module.__name__):
                    value = module._integral(np.array(case["time"]), np.array(case["values"]))
                    self.assertIs(type(value), float)
                    self.assertEqual(value.hex(), case["hex"])

    def test_integral_errors_and_nonfinite_values(self):
        """Keep existing NumPy shape errors and non-finite propagation visible."""
        for module in (report, explain):
            with self.subTest(module=module.__name__):
                with self.assertRaises(ValueError):
                    module._integral(np.arange(3.0), np.arange(5.0))
                self.assertTrue(np.isnan(module._integral(np.array([0.0, 1.0]), np.array([np.nan, 1.0]))))

    def test_momentum_binary64_goldens(self):
        """Retain exact contact checkpoints with missing or disjoint loaded samples."""
        for name, case in _GOLDEN["momentum"].items():
            args = (np.array(case["times"]), np.array(case["velocity"]), np.array(case["loaded"]))
            for function in (env._momentum_checkpoints, optimize._momentum):
                with self.subTest(case=name, function=function.__name__):
                    value = function(*args)
                    self.assertIs(type(value), list)
                    self.assertEqual([float(item).hex() for item in value], case["hex"])

    def test_checkpoint_patches_stay_module_local(self):
        """Use each old module's current checkpoint array rather than captured globals."""
        args = (np.array([0.0, 1.0, 2.0]), np.array([3.0, 5.0, 7.0]), np.ones(3, dtype=bool))
        for name, local, other in (
            ("env", env._momentum_checkpoints, optimize._momentum),
            ("optimize", optimize._momentum, env._momentum_checkpoints),
        ):
            unchanged = other(*args)
            with (
                self.subTest(module=name),
                patch(f"projects.impedance_instron.{name}.CHECKPOINTS", np.array([0.0, 1.0])),
            ):
                self.assertEqual(local(*args), [0.0, 4.0])
                self.assertEqual(other(*args), unchanged)
                self.assertEqual(local(args[0], args[1], np.zeros(3, dtype=bool)), [0.0, 0.0])
        with patch("projects.impedance_instron.env.CHECKPOINTS", np.array([])):
            self.assertEqual(env._momentum_checkpoints(*args), [])

    def test_momentum_degenerate_clock(self):
        """Keep the existing phase-floor and interpolation behavior for duplicate times."""
        args = (np.array([1.0, 1.0]), np.array([3.0, 7.0]), np.ones(2, dtype=bool))
        for function in (env._momentum_checkpoints, optimize._momentum):
            self.assertEqual(function(*args), [4.0] * len(_GOLDEN["checkpoints"]))
            with self.assertRaises(IndexError):
                function(args[0], args[1], np.ones(3, dtype=bool))

    def test_json_callback_goldens(self):
        """Keep NumPy, path, unsupported-type and callback serialization behavior."""
        value = {
            "path": Path("shoe.json"),
            "array": np.array([1, 2]),
            "scalar": np.float32(1.25),
            "bool": np.bool_(True),
        }
        expected = '{"array": [1, 2], "bool": true, "path": "shoe.json", "scalar": 1.25}'
        for module in (report, explain):
            self.assertEqual(json.dumps(value, default=module._json_default, sort_keys=True), expected)
            self.assertEqual(module._json_default(np.array(3.0)), 3.0)
            with self.assertRaisesRegex(TypeError, "^Cannot serialize object to JSON$"):
                module._json_default(object())
            with self.assertRaisesRegex(TypeError, "^Cannot serialize complex to JSON$"):
                json.dumps(np.complex128(1.0 + 2.0j), default=module._json_default)

    def test_html_escape_goldens(self):
        """Preserve string coercion and quote escaping across all historical reports."""
        values = [(None, "None"), (2, "2"), ("<&\"'", "&lt;&amp;&quot;&#x27;"), ("é", "é")]
        for module in (report, explain, dashboard, summary):
            for value, expected in values:
                self.assertEqual(module._escape(value), expected)

    def test_old_namespace_callback_patches(self):
        """Keep callers bound to patchable helper names in their original modules."""
        time, power = np.array([0.0, 1.0]), np.array([2.0, 2.0])
        with patch("projects.impedance_instron.report._integral", return_value=8.0) as callback:
            self.assertEqual(report._work(time, power), (2.0, 6.0, 8.0))
            callback.assert_called_once()
        with patch("projects.impedance_instron.explain._integral", return_value=9.0) as callback:
            self.assertEqual(explain._rms_error(time, power, power), 3.0)
            callback.assert_called_once()
        with (
            patch("projects.impedance_instron.report._columns", return_value={}),
            patch("projects.impedance_instron.report._json_default", side_effect=RuntimeError("callback reached")),
            self.assertRaisesRegex(RuntimeError, "callback reached"),
        ):
            report.write_report(Path("unused"), [], {"array": np.array([1.0])})


class TestLegacyWarpHelperReuse(unittest.TestCase):
    """Reuse the same Warp objects without changing counters or planar pitch."""

    def test_same_warp_objects(self):
        """Expose canonical example helpers under the historical environment names."""
        self.assertIs(env._pitch_of, example._pitch_of)
        self.assertIs(env._advance_sample_index, example._advance_index)

    def _check_eager(self, device):
        """Compare both helper names on one device using synthetic arrays."""
        rotations = np.array(
            [[0, 0, 0, 1], [0.2, 0.3, 0.4, 0.5], [0.2, -0.3, -0.4, -0.5], [0, 1, 0, 0]], dtype=np.float32
        )
        q = wp.array(rotations, dtype=wp.quat, device=device)
        result = wp.empty((len(rotations), 2), dtype=float, device=device)
        wp.launch(_evaluate_pitch, dim=len(rotations), inputs=[q, result], device=device)
        values = result.numpy()
        np.testing.assert_array_equal(values[:, 0], values[:, 1])
        expected = 2.0 * np.arctan2(rotations[:, 1], rotations[:, 3])
        np.testing.assert_allclose(values[:, 0], expected, rtol=1e-6, atol=1e-6)
        counters = [wp.array([-2], dtype=wp.int32, device=device) for _ in range(2)]
        for _ in range(5):
            wp.launch(env._advance_sample_index, dim=1, inputs=[counters[0]], device=device)
            wp.launch(example._advance_index, dim=1, inputs=[counters[1]], device=device)
        for counter in counters:
            np.testing.assert_array_equal(counter.numpy(), [3])

    def test_cpu_pitch_and_counter(self):
        """Match planar pitch and repeated counter increments on the CPU."""
        self._check_eager("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_cuda_pitch_and_counter(self):
        """Match planar pitch and repeated counter increments on CUDA."""
        self._check_eager("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_cuda_graph_counter_replay_and_reset(self):
        """Replay and reset captured increments through the old environment alias."""
        device = "cuda:0"
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(env._advance_sample_index, dim=1, inputs=[counter], device=device)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(env._advance_sample_index, dim=1, inputs=[counter], device=device)
        for initial in (-2, 11):
            counter.assign(np.array([initial], dtype=np.int32))
            for _ in range(3):
                wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(counter.numpy(), [initial + 3])


if __name__ == "__main__":
    unittest.main()
