# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side helpers for compression-coloured Digital Shoe rendering."""

import warp as wp


@wp.kernel
def column_world_positions(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
):
    """Transform shoe-local column anchors to world positions."""
    i = wp.tid()
    points[i] = wp.transform_point(body_q[carrier], anchor_local[i])


@wp.kernel
def attached_column_tops(
    bottom: wp.array[wp.vec3],
    rest_length: wp.array[wp.float32],
    top: wp.array[wp.vec3],
):
    """Create visual foam tops above the current outsole points."""
    i = wp.tid()
    p = bottom[i]
    top[i] = wp.vec3(p[0], p[1], p[2] + rest_length[i])


@wp.kernel
def column_colors(compression: wp.array[wp.float32], reference_m: wp.float32, colors: wp.array[wp.vec3]):
    """Map unloaded blue through green to highly compressed red."""
    i = wp.tid()
    value = wp.clamp(compression[i] / reference_m, 0.0, 1.0)
    colors[i] = wp.vec3(value, 1.0 - wp.abs(2.0 * value - 1.0), 1.0 - value)
