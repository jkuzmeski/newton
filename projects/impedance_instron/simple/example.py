# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""View the two-stiffness rig without prescribed pelvis motion or foot pitch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from projects.digital_shoe.rendering import carried_column_segment, column_colors

from .policy import restore
from .reference import Reference
from .report import write_report
from .rig import Rig


@wp.kernel
def _columns(
    body: wp.array[wp.transform],
    anchors: wp.array[wp.vec3],
    rest: wp.array[float],
    compression: wp.array[float],
    driven: wp.array[int],
    bottoms: wp.array[wp.vec3],
    tops: wp.array[wp.vec3],
):
    """Draw shoe-relative deformation, never the pressure proxy or bristle memory."""
    i = wp.tid()
    bottom, top = carried_column_segment(body[0], anchors[i], rest[i], compression[i], driven[i], 0.0)
    bottoms[i] = bottom
    tops[i] = top


class Example:
    """Run one stance and display the measured pelvis target separately from state."""

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.policy = None
        material = getattr(args, "material", None)
        if material is not None and not args.checkpoint:
            raise ValueError("--material requires --checkpoint; use --artifact for a baseline shoe")
        if args.checkpoint:
            self.rig, self.policy = restore(
                args.checkpoint,
                artifact_path=material,
                device=args.device,
                allow_physics_update=args.allow_physics_update,
            )
        else:
            self.rig = Rig(Reference.load(args.reference), args.artifact, device=args.device)
        self.model = self.rig.model
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.45, -2.0, 0.9), pitch=-8.0, yaw=90.0)
        self.observation = self.rig.reset()
        self.done, self.frame, self.sim_time = False, 0, 0.0
        self.info = {}
        self.returns = np.zeros(1)
        self._saved = False
        self._screenshot_saved = False
        foundation = self.rig.foundation
        device = self.rig.device
        self.bottoms = wp.zeros(foundation.column_count, dtype=wp.vec3, device=device)
        self.tops = wp.zeros_like(self.bottoms)
        self.colors = wp.zeros_like(self.bottoms)
        self.foot_point = wp.zeros(1, dtype=wp.vec3, device=device)
        self.pelvis_point = wp.zeros_like(self.foot_point)
        self.target_point = wp.zeros_like(self.foot_point)
        self.pelvis_color = wp.array([[0.2, 0.6, 0.9]], dtype=wp.vec3, device=device)
        self.target_color = wp.array([[0.9, 0.45, 0.1]], dtype=wp.vec3, device=device)
        mesh = self.rig.shoe.visual_mesh("fullfoot_last")
        self.vertices = np.asarray(mesh.vertices_m) - self.rig.ankle_mount
        triangles = np.asarray(mesh.triangles).reshape(-1, 3)
        self.edges = np.unique(
            np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0
        )
        self.edge_a = wp.zeros(len(self.edges), dtype=wp.vec3, device=device)
        self.edge_b = wp.zeros_like(self.edge_a)

    def step(self):
        """Advance only the stiffness-controlled physics, then hold the finished stance."""
        if self.done:
            return
        if self.policy is None:
            action = np.zeros((1, 2), dtype=np.float32)
        else:
            action = self.policy.act(self.observation)
        self.observation, reward, self.done, self.info = self.rig.step(action)
        self.returns += reward
        self.frame += 1
        self.sim_time = self.frame * self.rig.frame_dt
        if self.done and not self._saved:
            write_report(
                self.rig,
                self.args.output,
                {
                    "return": self.returns,
                    "safety": self.info,
                    "policy": self.policy.checkpoint_metadata if self.policy is not None else None,
                    "physics": self.rig.metadata,
                },
            )
            self._saved = True

    def render(self):
        """Draw the actual shoe, upper mass, and distinct optical height target."""
        rig, foundation = self.rig, self.rig.foundation
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(rig.state_0)
        wp.launch(
            _columns,
            dim=foundation.column_count,
            inputs=[
                rig.state_0.body_q,
                foundation.anchor_local,
                foundation.rest_len,
                foundation.compression,
                foundation.driven,
                self.bottoms,
                self.tops,
            ],
            device=rig.device,
        )
        wp.launch(
            column_colors,
            dim=foundation.column_count,
            inputs=[foundation.compression, 0.03, self.colors],
            device=rig.device,
        )
        self.viewer.log_lines("shoe/columns", self.bottoms, self.tops, self.colors, width=0.002)
        poses = rig.state_0.body_q.numpy()
        self.foot_point.assign(poses[0:1, :3])
        self.pelvis_point.assign(poses[1:2, :3])
        target = poses[1:2, :3].copy()
        target[0, 2] = rig.reference.sample(self.sim_time)["pelvis_z_m"]
        target[0, 0] += 0.12
        self.target_point.assign(target)
        self.viewer.log_lines("leg", self.foot_point, self.pelvis_point, (0.2, 0.6, 0.9), width=0.012)
        self.viewer.log_points("pelvis_simulated", self.pelvis_point, radii=0.055, colors=self.pelvis_color)
        self.viewer.log_points("pelvis_reference", self.target_point, radii=0.03, colors=self.target_color)
        angle = 2 * np.arctan2(poses[0, 4], poses[0, 6])
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        vertices = self.vertices @ rotation.T + poses[0, :3]
        self.edge_a.assign(vertices[self.edges[:, 0]].astype(np.float32))
        self.edge_b.assign(vertices[self.edges[:, 1]].astype(np.float32))
        self.viewer.log_lines("last", self.edge_a, self.edge_b, (0.65, 0.68, 0.72), width=0.0007)
        self.viewer.end_frame()
        if self.args.screenshot and not self._screenshot_saved and self.sim_time >= 0.45 * rig.reference.duration_s:
            if not hasattr(self.viewer, "get_frame"):
                raise ValueError("Screenshots require the OpenGL viewer")
            from PIL import Image, ImageOps

            path = Path(self.args.screenshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            ImageOps.fit(Image.fromarray(self.viewer.get_frame().numpy()), (320, 320)).convert("RGB").save(path)
            self._screenshot_saved = True

    def test_final(self):
        """Require a complete finite rollout without claiming target reachability."""
        if not self.done:
            raise AssertionError("Run at least one full stance")
        if not np.isfinite(self.rig.state_0.body_q.numpy()).all():
            raise AssertionError("Nonfinite rig state")
        if not np.isfinite(self.returns).all():
            raise AssertionError("Nonfinite tracking return")
        if not np.all(self.info.get("safety_ok", False)):
            raise AssertionError(f"Physical safety checks failed: {self.info}")


def create_parser():
    """Expose reference, material, checkpoint and output—not extra control modes."""
    parser = newton.examples.create_parser()
    parser.add_argument("--reference", type=Path, default=Path("outputs/impedance_instron/simple/reference.json"))
    parser.add_argument("--artifact", type=Path, default=Path("outputs/impedance_instron/inputs/digital_shoe.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--material",
        type=Path,
        default=None,
        help="Explicit same-geometry material or relocated artifact for a saved checkpoint.",
    )
    parser.add_argument(
        "--allow-physics-update",
        action="store_true",
        help="Re-evaluate old checkpoint weights with corrected physics; old scores no longer apply.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/impedance_instron/simple/viewer"))
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser


def main():
    """Run through Newton's normal example interface."""
    viewer, args = newton.examples.init(create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
