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
def attached_column_endpoints(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    anchor_bottom: wp.array[wp.vec3],
    rest_length: wp.array[wp.float32],
    bottom_out: wp.array[wp.vec3],
    top_out: wp.array[wp.vec3],
):
    """Transform attached columns and clamp their outsole ends to the ground."""
    i = wp.tid()
    transform = body_q[carrier]
    bottom = wp.transform_point(transform, anchor_bottom[i])
    top = wp.transform_point(transform, anchor_bottom[i] + wp.vec3(0.0, 0.0, rest_length[i]))
    bottom_z = wp.max(bottom[2], 0.0)
    top_z = wp.max(top[2], 0.0)
    if top_z < bottom_z:
        bottom_z = top_z
    bottom_out[i] = wp.vec3(bottom[0], bottom[1], bottom_z)
    top_out[i] = wp.vec3(top[0], top[1], top_z)


@wp.kernel
def deform_attached_mesh(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    source_vertices: wp.array[wp.vec3],
    output_vertices: wp.array[wp.vec3],
):
    """Transform the calibrated midsole and flatten ground-penetrating vertices."""
    i = wp.tid()
    point = wp.transform_point(body_q[carrier], source_vertices[i])
    output_vertices[i] = wp.vec3(point[0], point[1], wp.max(point[2], 0.0))


@wp.kernel
def deform_instron_mesh(
    source_vertices: wp.array[wp.vec3],
    column_index: wp.array[wp.int32],
    height_fraction: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    output_vertices: wp.array[wp.vec3],
):
    """Compress the midsole top toward its fixed base using the nearest fixture column."""
    i = wp.tid()
    point = source_vertices[i]
    column = column_index[i]
    displacement = 0.0
    if column >= 0:
        displacement = height_fraction[i] * compression[column]
    output_vertices[i] = wp.vec3(point[0], point[1], point[2] - displacement)


@wp.kernel
def column_colors(compression: wp.array[wp.float32], reference_m: wp.float32, colors: wp.array[wp.vec3]):
    """Map unloaded blue through green to highly compressed red."""
    i = wp.tid()
    value = wp.clamp(compression[i] / reference_m, 0.0, 1.0)
    colors[i] = wp.vec3(value, 1.0 - wp.abs(2.0 * value - 1.0), 1.0 - value)
