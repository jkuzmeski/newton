# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Attach the shared shoe foundation to a planar foot without a pitch servo."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.artifact import load_artifact
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, SurroundConfig


def _hull(points: np.ndarray) -> np.ndarray:
    ordered = sorted(set(map(tuple, points)))
    lower, upper = [], []
    for chain, sequence in ((lower, ordered), (upper, reversed(ordered))):
        for point in sequence:
            while len(chain) >= 2:
                a, b = np.subtract(chain[-1], chain[-2]), np.subtract(point, chain[-1])
                if a[0] * b[1] - a[1] * b[0] > 0:
                    break
                chain.pop()
            chain.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


class Shoe:
    """Evaluate contact at an ankle-centered foot pose supplied by the limb solver.

    The fullfoot last and the bed's rigid fixture-footprint backing share one
    carrier, with a passive outer region. Fixed attachment offsets retain the
    calibrated assembly; no second mesh-collision force or upper sliding contact
    is introduced. The
    planar angle increases from +X toward +Z (rotation about Newton's -Y axis).

    Args:
        artifact_path: Identified portable shoe artifact.
        mount_m: Ankle location in the intrinsic shoe frame [m], shape [3].
        static_pitch_rad: Measured static foot-axis angle [rad]. The intrinsic
            shoe is level in the static calibration, not aligned to skin markers.
        device: Warp device. CPU avoids device round trips in this reference solver.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        mount_m,
        static_pitch_rad: float,
        device: str = "cpu",
        *,
        friction_model: str = "elastic_coulomb",
    ):
        self.artifact_path = Path(artifact_path).resolve()
        self.shoe = load_artifact(self.artifact_path)
        self.mount_m = np.asarray(mount_m, dtype=float)
        self.static_pitch_rad = float(static_pitch_rad)
        if self.mount_m.shape != (3,) or not np.isfinite(self.mount_m).all():
            raise ValueError("Shoe mount must be a finite three-vector in metres")
        if not np.isfinite(self.static_pitch_rad):
            raise ValueError("Static foot pitch must be finite")
        coordinate = self.shoe.raw["coordinate_system"]
        if coordinate.get("up_axis") != "+Z" or coordinate.get("length_unit") != "m":
            raise ValueError("The shoe must use the +Z-up metre convention")
        self.device = wp.get_device(device)
        bed = self.shoe.column_bed
        fixture = self.shoe.instron_fixture("fullfoot_last")
        lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(bed.anchor_bottom_m[:, :2])}
        fixture_keys = [tuple(np.round(point, 8)) for point in fixture.carrier_anchor_m[:, :2]]
        if len(lookup) != len(bed.rest_length_m) or len(set(fixture_keys)) != len(fixture_keys):
            raise ValueError("Shoe bed and fixture footprint must have unique planar column coordinates")
        driven = np.zeros(len(bed.rest_length_m), dtype=np.int32)
        for key in fixture_keys:
            if key not in lookup:
                raise ValueError("Fixture footprint does not match the intrinsic shoe bed")
            driven[lookup[key]] = 1
        supported = np.asarray([lookup[key] for key in fixture_keys])
        self.anchor_local_m = bed.anchor_bottom_m - self.mount_m
        self.attachment_local_m = self.anchor_local_m.copy()
        self.attachment_local_m[:, 2] += bed.rest_length_m
        sites = fixture.carrier_anchor_m.copy()
        sites[:, 2] += bed.anchor_bottom_m[supported, 2] - fixture.foam_bottom_m
        self.attachment_local_m[supported] = sites - self.mount_m
        mesh = self.shoe.visual_mesh("fullfoot_last")
        self.last_vertices_local_m = mesh.vertices_m - self.mount_m
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)), label="fullfoot_last_carrier")
        # Only the column foundation supplies contact; reports use artifact geometry.
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.foundation = MidsoleFoundation(
            self.anchor_local_m,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            self.shoe.material,
            0,
            self.model.body_com,
            FoundationConfig(
                ground_height_m=0.0,
                normal_damping=0.0,
                friction_stiffness=10000.0
                if friction_model == "legacy"
                else (1000.0 if friction_model == "maxwell" else 0.0),
                friction=10.0 if friction_model in ("legacy", "maxwell") else 0.0,
                mu=0.8,
                friction_model=friction_model,
            ),
            self.device,
            SurroundConfig(driven=driven, carrier_bond=True),
        )
        self.foundation.reset()
        self.metadata = {
            "path": str(self.artifact_path),
            "sha256": hashlib.sha256(self.artifact_path.read_bytes()).hexdigest(),
            "shoe_id": self.shoe.shoe_id,
            "mount_m": self.mount_m.tolist(),
            "static_pitch_rad": self.static_pitch_rad,
            "friction_model": friction_model,
            "column_count": len(driven),
            "driven_columns": int(driven.sum()),
            "passive_columns": int(len(driven) - driven.sum()),
        }
        points = bed.anchor_bottom_m[:, (0, 2)] - self.mount_m[[0, 2]]
        # A sagittal outline is an undeformed registration diagnostic, not a
        # claim that rigid columns remain undeformed during contact.
        self.outline_local = _hull(points)

    def outline(self, ankle_m, pitch_rad: float) -> np.ndarray:
        """Return the undeformed registered sagittal outline [m], shape [N, 2]."""
        angle = pitch_rad - self.static_pitch_rad
        c, s = np.cos(angle), np.sin(angle)
        return self.outline_local @ np.array([[c, s], [-s, c]]) + ankle_m

    def apply(self, ankle_m, velocity_m_s, pitch_rad: float, angular_velocity_rad_s: float, dt: float):
        """Advance contact once and return ankle wrench [N, N, N m] and compression [m]."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Contact timestep must be finite and positive")
        angle = pitch_rad - self.static_pitch_rad
        position = np.array([[ankle_m[0], 0.0, ankle_m[1], 0.0, -np.sin(angle / 2), 0.0, np.cos(angle / 2)]])
        velocity = np.array([[velocity_m_s[0], 0.0, velocity_m_s[1], 0.0, -angular_velocity_rad_s, 0.0]])
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("Foot pose and velocity must be finite")
        self.state.body_q.assign(position.astype(np.float32))
        self.state.body_qd.assign(velocity.astype(np.float32))
        self.foundation.apply(self.state, dt, clear_body_force=True)
        wrench = self.state.body_f.numpy()[0].astype(float)
        # Newton +Y torque is opposite to the mathematical X/Z angle.
        return np.array([wrench[0], wrench[2], -wrench[4]]), float(self.foundation.max_compression.numpy()[0])
