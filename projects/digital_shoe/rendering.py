# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side helpers for compression-coloured Digital Shoe rendering."""

import numpy as np
import warp as wp


@wp.kernel
def column_endpoints(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    rest_len: wp.array[wp.float32],
    top: wp.array[wp.float32],
    bottom: wp.array[wp.float32],
    bottoms: wp.array[wp.vec3],
    tops: wp.array[wp.vec3],
):
    """Draw two-surface columns from their top and bottom displacements [m].

    The carried rest column stands on ``anchor_local``; both displacements are
    positive down (:class:`~projects.digital_shoe.column_bed.ColumnBed`).
    """
    i = wp.tid()
    point = wp.transform_point(body_q[carrier], anchor_local[i])
    bottoms[i] = point - wp.vec3(0.0, 0.0, bottom[i])
    tops[i] = point + wp.vec3(0.0, 0.0, rest_len[i] - top[i])


@wp.func
def carried_column_segment(
    transform: wp.transform,
    anchor: wp.vec3,
    rest: float,
    compression: float,
    driven: int,
    ground_height: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Reconstruct non-extending carrier-relative endpoints [m].

    Only free surround retreats relative to the carrier. The pressure proxy and
    friction reference anchor are not geometric displacements. Using current
    penetration bounds also handles a render pose one integration step ahead of
    the force history without advancing that history.
    """
    bottom = wp.transform_point(transform, anchor)
    top = wp.transform_point(transform, anchor + wp.vec3(0.0, 0.0, rest))
    retreat = float(0.0)
    if driven == 0:
        penetration = wp.max(ground_height - bottom[2], 0.0)
        retreat = penetration - wp.clamp(compression, 0.0, penetration)
    return (
        wp.vec3(bottom[0], bottom[1], wp.max(bottom[2] + retreat, ground_height)),
        wp.vec3(top[0], top[1], wp.max(top[2] + retreat, ground_height)),
    )


@wp.kernel
def column_colors(compression: wp.array[wp.float32], reference_m: wp.float32, colors: wp.array[wp.vec3]):
    """Map compression blue to cyan to yellow to red on one fixed scale."""
    i = wp.tid()
    value = wp.clamp(compression[i] / reference_m, 0.0, 1.0)
    if value < 1.0 / 3.0:
        blend = 3.0 * value
        colors[i] = wp.vec3(0.0, blend, 1.0)
    elif value < 2.0 / 3.0:
        blend = 3.0 * value - 1.0
        colors[i] = wp.vec3(blend, 1.0, 1.0 - blend)
    else:
        blend = 3.0 * value - 2.0
        colors[i] = wp.vec3(1.0, 1.0 - blend, 0.0)


def camera_look_at(eye, target) -> tuple[wp.vec3, float, float]:
    """Return camera position, pitch and yaw for a Z-up view.

    Args:
        eye: World camera position [m], three coordinates.
        target: World look-at position [m], three coordinates.

    Returns:
        Camera position [m] and pitch/yaw [degrees].
    """
    direction = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    pitch = np.degrees(np.arcsin(direction[2]))
    yaw = np.degrees(np.arctan2(direction[1], direction[0]))
    return wp.vec3(*[float(value) for value in eye]), float(pitch), float(yaw)
