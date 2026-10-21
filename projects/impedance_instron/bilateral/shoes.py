# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mirror a digital-shoe artifact for the opposite foot.

The F01 shoe artifact describes the right shoe. The left foot uses its mirror
image about the shoe's sagittal (x-z) plane: every y coordinate changes sign,
the -y and +y neighbor slots swap, and triangle winding reverses so outward
normals stay outward. The constitutive model, identification, and validation
records are unchanged; the mirror is a geometric assumption, not a measured
right shoe.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from projects.digital_shoe.artifact import validate_artifact

MIRROR_OPERATION = "reflect y -> -y; swap -y/+y neighbor slots; reverse triangle winding"


def _neighbor_permutation(points: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Return the slot permutation that restores the slot directions after reflecting y."""
    directions = np.zeros((4, 3))
    for slot in range(4):
        valid = neighbors[:, slot] >= 0
        if valid.any():
            directions[slot] = np.mean(points[neighbors[valid, slot]] - points[valid], axis=0)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Every neighbor slot needs at least one neighbor to infer its direction")
    directions /= norms
    mirrored = directions * np.array([1.0, -1.0, 1.0])
    permutation = np.argmax(mirrored @ directions.T, axis=1)
    if sorted(permutation.tolist()) != [0, 1, 2, 3] or np.any(np.max(mirrored @ directions.T, axis=1) < 0.9):
        raise ValueError("Neighbor slots are not axis-aligned grid directions")
    return permutation


def _mirror_grid(points: list, neighbors: list) -> tuple[list, list]:
    points = np.asarray(points, dtype=float)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    permutation = _neighbor_permutation(points, neighbors)
    reordered = np.empty_like(neighbors)
    reordered[:, permutation] = neighbors
    return (points * np.array([1.0, -1.0, 1.0])).tolist(), reordered.tolist()


def mirror_artifact(data: dict[str, Any], *, source_sha256: str | None = None) -> dict[str, Any]:
    """Return the mirror image of a ``digital_shoe_1`` artifact about its x-z plane.

    Args:
        data: Decoded artifact JSON.
        source_sha256: Hash of the source file, recorded in the provenance.
    """
    validate_artifact(data)
    result = copy.deepcopy(data)
    bed = result["column_bed"]
    bed["anchor_bottom_m"], bed["neighbors"] = _mirror_grid(bed["anchor_bottom_m"], bed["neighbors"])
    for fixture in result.get("instron_fixtures", {}).values():
        fixture["carrier_anchor_m"], fixture["neighbors"] = _mirror_grid(
            fixture["carrier_anchor_m"], fixture["neighbors"]
        )
    for mesh in result.get("visual_meshes", {}).values():
        mesh["vertices_m"] = (np.asarray(mesh["vertices_m"], dtype=float) * np.array([1.0, -1.0, 1.0])).tolist()
        mesh["triangles"] = np.asarray(mesh["triangles"], dtype=np.int64)[:, [0, 2, 1]].tolist()
    source_id = data["shoe"]["id"]
    result["shoe"]["id"] = f"{source_id}_mirrored"
    result["shoe"]["mirrored_from"] = source_id
    result["provenance"] = dict(result["provenance"])
    result["provenance"]["mirror"] = {
        "source_shoe_id": source_id,
        "source_sha256": source_sha256,
        "operation": MIRROR_OPERATION,
        "claim": "Geometric mirror of the source shoe; not a measured opposite-foot shoe.",
    }
    validate_artifact(result)
    return result


def write_mirrored(source: str | Path, destination: str | Path) -> Path:
    """Write the mirrored artifact without overwriting an existing file."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    raw = source.read_bytes()
    mirrored = mirror_artifact(json.loads(raw), source_sha256=hashlib.sha256(raw).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(mirrored, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mirror a digital-shoe artifact for the opposite foot.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    print(write_mirrored(args.source, args.destination))


if __name__ == "__main__":
    main()
