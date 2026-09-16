# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build view-only last and midsole meshes in the saved ankle-centered frame."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def load_geometry(shoe: dict) -> dict:
    """Load the saved rigid foot geometry without changing placement or contact.

    Returned vertices are intrinsic shoe coordinates minus the saved ankle
    mount [m]. Rotate their X/Z projection by foot angle minus static pitch,
    then translate by the simulated ankle. The connector is only a drawing
    of the existing rigid offset, not another body or a contact force path.
    """
    from projects.digital_shoe.artifact import load_artifact  # noqa: PLC0415

    if not shoe.get("path") or not Path(shoe["path"]).is_file():
        return {"available": False, "reason": "Saved shoe artifact is unavailable; no substitute foot point is drawn."}
    path = Path(shoe["path"])
    if shoe.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != shoe["sha256"]:
        raise ValueError("Saved shoe artifact changed; refusing to draw different last geometry")
    artifact = load_artifact(path)
    mount = np.asarray(shoe["mount_m"], dtype=float)
    result = {
        "available": True,
        "static_pitch_rad": float(shoe["static_pitch_rad"]),
        "mount_m": mount.tolist(),
        "scope": "rigid last and undeformed midsole meshes; view-only, unchanged physics",
    }
    for key, name in (("last", "fullfoot_last"), ("midsole", "midsole")):
        mesh = artifact.visual_meshes[name]
        vertices = np.asarray(mesh.vertices_m) - mount
        triangles = np.asarray(mesh.triangles, dtype=int)
        # A fixed sagittal view can reuse a side-depth ordering under planar pitch.
        triangles = triangles[np.argsort(vertices[triangles, 1].mean(axis=1), kind="stable")]
        faces = vertices[triangles]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-15)
        light = np.array([0.25, 0.8, 0.55])
        light /= np.linalg.norm(light)
        # Both windings are drawn; shading must not imply repaired source topology.
        shade = 0.72 + 0.28 * np.abs(normals @ light)
        result[key] = {
            "vertices": vertices,
            "triangles": triangles,
            "shade": shade,
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
        }
    last = result["last"]["vertices"]
    nearest = last[np.argmin(np.linalg.norm(last, axis=1))]
    result["mount_connector_local_m"] = nearest
    result["mount_connector_length_m"] = float(np.linalg.norm(nearest))
    return result
