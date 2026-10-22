# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

try:
    import torch

    from projects.supermimic.body import STATE_DIM, BodyParams
    from projects.supermimic.buffer import ReplayBuffer
    from projects.supermimic.collect import (
        AUX_DIM,
        CENTER_SHOE,
        loaded_moment,
        make_sim,
        moment_stiffness_floor,
        read_episode,
        reference_action,
        sim_aux,
        summarize,
    )
    from projects.supermimic.context import GEOMETRY_NAMES, decode
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "PyTorch with CUDA is required")
class TestSupermimicCollect(unittest.TestCase):
    def _buffer(self) -> ReplayBuffer:
        """Two worlds, six rows; world 0 resets at row 3 after a fall, right foot lands at rows 1 and 4."""
        buffer = ReplayBuffer(2, 6, state_shape=STATE_DIM, action_shape=(10, 3), xi_shape=9, aux_shape=AUX_DIM)
        right_z = [0.0, 300.0, 600.0, 0.0, 200.0, 0.0]
        for t in range(6):
            state = torch.arange(STATE_DIM, device="cuda").float().repeat(2, 1) + 100.0 * t
            aux = torch.arange(AUX_DIM, device="cuda").float().repeat(2, 1)
            aux[:, 2] = right_z[t]
            aux[:, 5] = 0.0
            action = torch.arange(30, device="cuda").float().reshape(10, 3).repeat(2, 1, 1)
            reset = torch.tensor([t in (0, 3), t == 0], device="cuda")
            done = torch.tensor([t == 2, False], device="cuda")
            buffer.push(
                state,
                action,
                torch.zeros((2, 9), device="cuda"),
                aux,
                torch.full((2,), 800 + t, dtype=torch.int64, device="cuda"),
                reset,
                done,
            )
        return buffer

    def test_summary_counts_episodes_falls_and_touchdowns(self):
        """Count touchdowns within episodes, not across a reset, and failure rates per world-second."""
        meta = {
            "dt_s": 0.01,
            "failures": [1, 0],
            "xi": [[-0.9] * 9, [0.9] * 9],
            "shoe": ["a", "b"],
            "xi_coordinates": ["c"],
            "config": {"touchdown_force_n": 50.0},
            "stiffness_short_steps": {"knee": 3},
        }
        summary = summarize(self._buffer(), meta)
        self.assertEqual(summary["rows"], 12)
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["falls"], 1)
        self.assertEqual(summary["touchdowns"], 4)
        self.assertEqual(summary["completed_episode_steps_mean"], 3.0)
        self.assertAlmostEqual(summary["sim_failures_per_world_s"], 1.0 / 0.12)
        self.assertAlmostEqual(summary["sim_failures_per_world_s_by_shoe"]["a"], 1.0 / 0.06)
        self.assertEqual(summary["sim_failures_per_world_s_by_xi_quartile"]["c"][3], 0.0)
        self.assertEqual(summary["stiffness_short_fraction"]["knee"], 0.25)

    def test_make_sim_runs_the_belt_at_the_reference_speed(self):
        """Run the treadmill at the belt speed passed from the reference, on the center shoe's geometry."""
        sim, xi = make_sim(BodyParams(), 2, "train", np.random.default_rng(0), 3.1)

        self.assertEqual(sim.config.belt_speed_m_s, 3.1)
        self.assertEqual(xi.shape[0], 2)
        np.testing.assert_array_equal(decode(xi)[2], GEOMETRY_NAMES.index(CENTER_SHOE))

    def test_loaded_moment_drops_stance_moment_of_unloaded_feet(self):
        """A reference-stance leg keeps its moment only with a loaded sim foot; swing legs keep theirs."""
        moment = torch.arange(1.0, 11.0, device="cuda").repeat(3, 1)
        stance = torch.tensor([[True, False], [True, True], [False, True]], device="cuda")
        loaded = torch.tensor([[False, True], [True, False], [False, False]], device="cuda")
        out = loaded_moment(moment, stance, loaded)
        zero = torch.zeros(5, device="cuda")
        torch.testing.assert_close(out[0], torch.cat((zero, moment[0, 5:])))
        torch.testing.assert_close(out[1], torch.cat((moment[1, :5], zero)))
        torch.testing.assert_close(out[2], torch.cat((moment[2, :5], zero)))

    def test_moment_stiffness_floor_bounds_posture_lead(self):
        """Use the room to the limit in the moment's direction, capped at the largest lead."""
        q_limits = torch.tensor([[-1.0, 1.0]] * 4, device="cuda")
        stiffness_limits = torch.tensor([[1.0, 1000.0]] * 4, device="cuda")
        q_ref = torch.tensor([0.9, 0.9, 0.0, 0.0], device="cuda")
        moment = torch.tensor([10.0, -10.0, 70.0, 0.1], device="cuda")
        floor = moment_stiffness_floor(q_ref, moment, q_limits, stiffness_limits, 0.35)
        torch.testing.assert_close(floor, torch.tensor([100.0, 10.0 / 0.35, 200.0, 1.0], device="cuda"))

    def test_reference_action_damps_the_velocity_error(self):
        """Carry the moment at the reference state, so the damping only acts on the velocity error."""
        q_limits = torch.tensor([[-1.0, 1.0]] * 3, device="cuda")
        stiffness_limits = torch.tensor([[10.0, 1000.0]] * 3, device="cuda")
        q_ref = torch.zeros((2, 3), device="cuda")
        qd_ref = torch.tensor([[8.0, -4.0, 0.0]] * 2, device="cuda")
        moment = torch.tensor([[0.0, 20.0, -30.0]] * 2, device="cuda")
        damping = torch.full((2, 3), 5.0, device="cuda")
        quantile = torch.tensor([[0.0] * 3, [0.5] * 3], device="cuda")
        action, short = reference_action(
            q_ref, qd_ref, moment, damping, quantile, torch.zeros_like(q_ref), q_limits, stiffness_limits, 0.35
        )
        q_eq, stiffness = action[..., 0], action[..., 1]
        torch.testing.assert_close(stiffness * (q_eq - q_ref) - damping * qd_ref, moment)
        torch.testing.assert_close(stiffness[0], torch.tensor([40.0 / 0.35, 10.0, 30.0 / 0.35], device="cuda"))
        torch.testing.assert_close(action[..., 2], damping)
        self.assertFalse(bool(short.any()))

    def test_sim_aux_makes_the_center_of_pressure_root_relative(self):
        """Subtract the start root x/y from a loaded foot's center of pressure and keep zero without contact."""
        grf = np.zeros((1, 2, 3), dtype=np.float32)
        grf[0, 0, 2] = 500.0
        cop = np.zeros((1, 2, 3), dtype=np.float32)
        cop[0, 0] = (0.5, -0.25, 0.0)
        joints = np.zeros((1, 10), dtype=np.float32)
        sim = SimpleNamespace(
            world_count=1,
            grf=wp.array(grf, device="cuda"),
            cop=wp.array(cop, device="cuda"),
            joint_torque=wp.array(joints, device="cuda"),
            joint_work_pos=wp.array(joints, device="cuda"),
            joint_work_neg=wp.array(joints, device="cuda"),
        )
        state = torch.zeros((1, STATE_DIM), device="cuda")
        state[0, :2] = torch.tensor((0.125, 0.25))
        aux = sim_aux(sim, state)
        torch.testing.assert_close(aux[0, 6:12].cpu(), torch.tensor((0.375, -0.5, 0.0, 0.0, 0.0, 0.0)))
        self.assertEqual(float(wp.to_torch(sim.cop)[0, 0, 0]), 0.5)

    def test_read_episode_restores_sim_layouts(self):
        """Split a world into episodes and rebuild joint_q, joint_qd and aux arrays."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._buffer().save(folder / "buffer.npz")
            (folder / "meta.json").write_text(
                json.dumps({"dt_s": 0.01, "shoe": ["a", "b"], "reference_frames_per_window": 400})
            )
            episode = read_episode(folder, 0, 1)
        state = np.arange(STATE_DIM, dtype=np.float32)
        self.assertEqual(episode["joint_q"].shape, (3, 17))
        np.testing.assert_array_equal(episode["joint_q"][0], np.concatenate((state[:7], state[13:23])) + 300.0)
        np.testing.assert_array_equal(episode["joint_qd"][0], np.concatenate((state[7:13], state[23:])) + 300.0)
        self.assertEqual(episode["grf"][1, 0, 2], 200.0)
        np.testing.assert_array_equal(episode["cop"][0].ravel(), np.arange(6, 12))
        np.testing.assert_array_equal(episode["cop"][1, 0], [6.0 + 400.0, 7.0 + 401.0, 8.0])
        np.testing.assert_allclose(episode["joint_power"][0], (np.arange(22, 32) + np.arange(32, 42)) / 0.01)
        np.testing.assert_array_equal(episode["action"][0, :, 1], np.arange(1, 30, 3))
        self.assertEqual((episode["dt"], episode["shoe"]), (0.01, "a"))
        self.assertEqual((episode["reference_window"], episode["reference_frame"]), (2, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
