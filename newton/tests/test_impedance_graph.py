# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that replaying the captured impedance frame reproduces the uncaptured rollout."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import warp as wp

from projects.impedance_instron.example import Example, create_parser

_PROFILE = Path("outputs/impedance_instron/stance_planar_context.json")
_ARTIFACT = Path("DigitalInstron/digital_shoe_showcase/digital_shoe.json")

# Trace columns carrying shoe contact: normal force, contact power, and peak compression.
_CONTACT_COLUMNS = [0, 9, 11]

# Substep samples to run. No budget is a multiple of its substep count, so every comparison
# ends with an uncaptured partial frame, and every budget now runs well past its own touchdown:
# sample 342 for the legacy schedule, 147 for the equilibrium controller, which drops the shoe
# earlier, and 37 for the seven substep frame, whose samples are an order of magnitude coarser.
_SAMPLES, _EQUILIBRIUM_SAMPLES, _ODD_SAMPLES = 600, 400, 120


def _cuda_device() -> str | None:
    """Return the first CUDA device, or None when this machine has none."""
    devices = [device for device in wp.get_devices() if device.is_cuda]
    return str(devices[0]) if devices else None


def _rollout(device: str, graph: bool, samples: int, extra: tuple[str, ...] = ()) -> Example:
    """Run one impedance rollout and return the finished example.

    Args:
        device: Warp device that carries the whole rollout.
        graph: Value of the ``--graph`` switch under test.
        samples: Substep samples to stop after.
        extra: Additional command line arguments.
    """
    args = create_parser().parse_args(["--viewer", "null", *extra])
    args.graph = graph
    with wp.ScopedDevice(device):
        example = Example(MagicMock(), args)
        example.sample_count = samples
        while example.index < example.sample_count:
            example.step()
    return example


@unittest.skipUnless(_PROFILE.is_file() and _ARTIFACT.is_file(), "Export the planar stance profile and shoe first")
@unittest.skipUnless(_cuda_device() is not None, "CUDA graph capture requires a CUDA device")
class TestImpedanceGraphCapture(unittest.TestCase):
    """Compare a captured rollout against the plain per-substep launches it replaces.

    Every comparison now runs through touchdown and into loading, which is where a capture
    bug would actually hide: a stale sample index, a swapped state binding or a missing
    launch shows up once contact makes consecutive substeps differ.

    Running there took two fixed-order reductions. The foundation sums its per-world totals
    in a fixed order (:func:`projects.digital_shoe.runtime.foundation_partial`), and this
    example's own contact and passive-region diagnostics do the same through
    ``_contact_motion_partial`` and ``_free_column_partial`` in
    :mod:`projects.impedance_instron.example`. While those two still used ``wp.atomic_add``
    over ~910 columns, captured and uncaptured traces differed by 1.2e-3 N on an 865 N peak
    (1.4e-6 relative) even though the final ``body_q`` was already identical, so the window
    had to stop short of contact.
    """

    def assert_same_rollout(self, samples: int, extra: tuple[str, ...] = ()):
        """Assert captured and uncaptured rollouts agree bit for bit through contact.

        Args:
            samples: Substep samples both rollouts run.
            extra: Additional command line arguments shared by both rollouts.
        """
        device = _cuda_device()
        plain = _rollout(device, False, samples, extra)
        captured = _rollout(device, True, samples, extra)
        self.assertFalse(plain.use_graph)
        self.assertTrue(captured.use_graph, captured.graph_status)
        self.assertIsNotNone(captured.graph)
        trace = plain.trace_device.numpy()[:samples]
        # The window is worthless if it never loads the shoe, so require real contact rather
        # than the absence of it: a peak above a quarter of body weight and measurable
        # compression mean the comparison is running where capture bugs live.
        self.assertGreater(float(trace[:, 0].max()), 0.25 * plain.mass * plain.gravity)
        self.assertGreater(float(trace[:, 11].max()), 1.0e-3)
        self.assertGreater(float(np.abs(trace[:, 9]).max()), 0.0)
        np.testing.assert_array_equal(captured.trace_device.numpy()[:samples], trace)
        np.testing.assert_array_equal(captured.state_0.body_q.numpy(), plain.state_0.body_q.numpy())
        np.testing.assert_array_equal(captured.state_0.body_qd.numpy(), plain.state_0.body_qd.numpy())
        for example in (plain, captured):
            self.assertEqual(int(example.index_device.numpy()[0]), example.index - 1)

    def test_captured_frames_reproduce_the_legacy_controller(self):
        """Replay the scheduled-impedance rollout with the same trace and final state."""
        self.assert_same_rollout(_SAMPLES)

    def test_captured_frames_reproduce_the_equilibrium_controller(self):
        """Replay the equilibrium-point rollout, whose leg kernel reads other reference columns."""
        self.assert_same_rollout(_EQUILIBRIUM_SAMPLES, ("--control", "equilibrium"))

    def test_captured_frames_reproduce_the_ankle_impedance_controller(self):
        """Replay the free-pitch rollout, whose fixture rotation is an integrated state."""
        self.assert_same_rollout(_SAMPLES, ("--ankle-control", "impedance"))

    def test_repeated_rollouts_are_bit_identical_through_contact(self):
        """Reproduce the whole trace of an uncaptured rollout by running it again.

        Capture can only be compared against a reference that is itself reproducible. While
        the diagnostics used atomics, two identical uncaptured rollouts already differed by
        1.2e-3 N once the shoe was loaded, so any capture comparison through contact was
        measuring that instead. This asserts the reference is exact first.
        """
        device = _cuda_device()
        first = _rollout(device, False, _SAMPLES)
        second = _rollout(device, False, _SAMPLES)
        trace = first.trace_device.numpy()[:_SAMPLES]
        self.assertGreater(float(trace[:, 0].max()), 0.25 * first.mass * first.gravity)
        np.testing.assert_array_equal(second.trace_device.numpy()[:_SAMPLES], trace)

    def test_odd_substep_count_replays_without_swapping_bindings(self):
        """Reproduce the rollout when a frame holds an odd number of substeps.

        An odd count leaves the state ping-pong exchanged at the end of a frame, so the
        captured graph would replay from the buffer it wrote last. The final substep of a
        captured frame copies instead of swapping; this checks that copy against the plain
        run rather than trusting the frame to end in its recorded arrangement.
        """
        self.assert_same_rollout(_ODD_SAMPLES, ("--substeps", "7"))


@unittest.skipUnless(_PROFILE.is_file() and _ARTIFACT.is_file(), "Export the planar stance profile and shoe first")
class TestImpedanceGraphFallback(unittest.TestCase):
    """Check the uncaptured path stays available where CUDA graphs are not."""

    def test_cpu_rollout_runs_without_capture(self):
        """Step a CPU rollout with graph capture requested and keep both counters in step."""
        example = _rollout("cpu", True, 5, ("--substeps", "2"))
        self.assertFalse(example.use_graph)
        self.assertIsNone(example.graph)
        self.assertEqual(example.index, 5)
        self.assertEqual(int(example.index_device.numpy()[0]), 4)


if __name__ == "__main__":
    unittest.main()
