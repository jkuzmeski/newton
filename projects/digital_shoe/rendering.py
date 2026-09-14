# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side helpers for compression-coloured Digital Shoe rendering."""

import warp as wp


@wp.func
def bench_column_segment(bottom: wp.vec3, rest: float, compression: float) -> tuple[wp.vec3, wp.vec3]:
    """Return bench endpoints from fixed vertical support and solved shortening [m]."""
    length = wp.max(rest - wp.max(compression, 0.0), 0.0)
    return bottom, bottom + wp.vec3(0.0, 0.0, length)


@wp.kernel
def bench_column_endpoints_at_sites(
    carrier: int,
    body_q: wp.array[wp.transform],
    anchors: wp.array[wp.vec3],
    fixed_bottom: wp.array[wp.vec3],
    rest: wp.array[float],
    compression: wp.array[float],
    bottoms: wp.array[wp.vec3],
    tops: wp.array[wp.vec3],
):
    """Draw bench columns at their nominal top-site XY and fixed base elevations."""
    i = wp.tid()
    world = wp.transform_point(body_q[carrier], anchors[i])
    base = wp.vec3(world[0], world[1], fixed_bottom[i][2])
    bottom, top = bench_column_segment(base, rest[i], compression[i])
    bottoms[i] = bottom
    tops[i] = top


@wp.kernel
def bench_column_endpoints(
    fixed_bottom: wp.array[wp.vec3],
    rest: wp.array[float],
    compression: wp.array[float],
    bottoms: wp.array[wp.vec3],
    tops: wp.array[wp.vec3],
):
    """Draw the solved full bench bed, including unloaded and passive columns [m]."""
    i = wp.tid()
    bottom, top = bench_column_segment(fixed_bottom[i], rest[i], compression[i])
    bottoms[i] = bottom
    tops[i] = top


@wp.kernel
def bench_column_top_points(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    out_points: wp.array[wp.vec3],
):
    """Draw bench foam tops beneath the imposed indenter, not a carried outsole."""
    i = wp.tid()
    world = wp.transform_point(body_q[carrier], anchor_local[i])
    top = z_free[i]
    if world[2] < top:
        top = world[2]
    out_points[i] = wp.vec3(world[0], world[1], top)


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
def carried_column_endpoints(
    carrier: int,
    body_q: wp.array[wp.transform],
    anchors: wp.array[wp.vec3],
    rest: wp.array[float],
    compression: wp.array[float],
    driven: wp.array[int],
    ground_height: float,
    bottoms: wp.array[wp.vec3],
    tops: wp.array[wp.vec3],
):
    """Draw one carried shoe with its passive relative deformation and ground plane."""
    i = wp.tid()
    bottom, top = carried_column_segment(body_q[carrier], anchors[i], rest[i], compression[i], driven[i], ground_height)
    bottoms[i] = bottom
    tops[i] = top


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
    bottom, top = carried_column_segment(body_q[carrier], anchor_bottom[i], rest_length[i], 0.0, 1, 0.0)
    bottom_out[i] = bottom
    top_out[i] = top


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
