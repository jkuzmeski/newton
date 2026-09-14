# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check resident predictor cache isolation and invalidation through core.predict."""

import gc
import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_instron_v2 import core


def _trial(name="same_name"):
    """Build a mutable-array trial with a frozen container and passive neighbor."""
    surround = core.Surround(
        driven=np.array([True, False]),
        neighbors=np.array([[1, -1, -1, -1], [0, -1, -1, -1]], np.int32),
        slack_m=np.array([0.02, 0.03]),
        area_m2=0.0001,
        spacing_m=0.005,
        sweeps=5,
    )
    return core.Trial(
        name,
        np.array([0.02]),
        0.0001,
        np.array([[0.02], [0.018], [0.02]]),
        np.full(3, 0.01),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.002, 0.0]),
        surround=surround,
    )


class _Workspace:
    """Expose the resident interface while isolating cache behavior from the kernels."""

    created: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.compression = object()
        self.stats = {"pass_change_m": [], "max_compression_m": 0.0}
        self.calls = []
        self.frames = len(args[0])
        self.created.append(self)

    def solve(self, params, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(numpy=lambda: np.full(self.frames, params.g_eq, dtype=np.float32))


class TestResidentCalibrationCache(unittest.TestCase):
    """Keep prepared GPU state private to a trial and its current input content."""

    def setUp(self):
        """Replace only workspace execution; retain the actual public predictor routing."""
        core._SURROUND_WORKSPACES.clear()
        core._SURROUND_WARM_START.clear()
        _Workspace.created.clear()
        self.material = core.Material(1e5, 2.0, 0.6, 0.01, 1e4, -0.5)
        self.scope = wp.ScopedDevice("cpu")
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        patcher = patch("projects.digital_shoe.calibration.CalibrationWorkspace", _Workspace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(core._SURROUND_WORKSPACES.clear)
        self.addCleanup(core._SURROUND_WARM_START.clear)

    def test_material_change_reuses_static_workspace(self):
        """Update only material data between finite-difference evaluations."""
        trial = _trial()
        first = core.predict(trial, self.material)
        other = core.Material(2e5, 2.0, 0.6, 0.01, 1e4, -0.5)
        second = core.predict(trial, other)
        self.assertEqual(len(_Workspace.created), 1)
        self.assertIsNone(_Workspace.created[0].calls[0]["initial"])
        self.assertIs(_Workspace.created[0].calls[1]["initial"], _Workspace.created[0].compression)
        np.testing.assert_array_equal(second, first * 2.0)
        self.assertEqual(second.dtype, np.float64)

    def test_same_name_trials_do_not_share_warm_state(self):
        """Isolate separate trials even when their human-readable labels coincide."""
        first, second = _trial(), _trial()
        core.predict(first, self.material)
        core.predict(second, self.material)
        self.assertEqual(len(_Workspace.created), 2)
        self.assertTrue(all(w.calls[0]["initial"] is None for w in _Workspace.created))

    def test_threads_do_not_share_mutable_workspace(self):
        """Isolate cached fields if a caller evaluates the same trial on another thread."""
        trial = _trial()
        with patch("projects.digital_instron_v2.core.threading.get_ident", return_value=101):
            core.predict(trial, self.material)
        with patch("projects.digital_instron_v2.core.threading.get_ident", return_value=102):
            core.predict(trial, self.material)
        self.assertEqual(len(_Workspace.created), 2)
        self.assertTrue(all(w.calls[0]["initial"] is None for w in _Workspace.created))

    def test_cuda_streams_do_not_share_mutable_workspace(self):
        """Retain separate captured buffers for independent CUDA streams."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA streams require a CUDA device")
        trial = _trial()
        first = wp.Stream(devices[0])
        second = wp.Stream(devices[0])
        with wp.ScopedStream(first):
            core.predict(trial, self.material)
        with wp.ScopedStream(second):
            core.predict(trial, self.material)
        self.assertEqual(len(_Workspace.created), 2)
        self.assertTrue(all(w.calls[0]["initial"] is None for w in _Workspace.created))

    def test_changed_input_content_rebuilds_workspace(self):
        """Invalidate cached buffers after in-place edits of a trial array."""
        trial = _trial()
        core.predict(trial, self.material)
        trial.lengths_m[1, 0] -= 0.001
        core.predict(trial, self.material)
        self.assertEqual(len(_Workspace.created), 2)
        self.assertIsNone(_Workspace.created[-1].calls[0]["initial"])

    def test_trial_release_removes_cache_entry(self):
        """Release resident storage when its trial is no longer referenced."""
        trial = _trial()
        core.predict(trial, self.material)
        self.assertEqual(len(core._SURROUND_WORKSPACES), 1)
        del trial
        gc.collect()
        self.assertFalse(core._SURROUND_WORKSPACES)
        self.assertFalse(core._SURROUND_WARM_START)

    def test_clear_warm_state_restarts_same_workspace_cold(self):
        """Allow controlled cold/warm benchmarks without rebuilding geometry."""
        trial = _trial()
        core.predict(trial, self.material)
        core._SURROUND_WARM_START.clear()
        core.predict(trial, self.material)
        self.assertEqual(len(_Workspace.created), 1)
        self.assertIsNone(_Workspace.created[0].calls[-1]["initial"])


if __name__ == "__main__":
    unittest.main()
