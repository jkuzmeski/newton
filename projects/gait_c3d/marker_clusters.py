# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tracking-marker cluster definitions shared by the gait pipeline."""

from __future__ import annotations

# Each cluster is represented by the arithmetic mean of its available valid
# marker positions. The source labels are the names used in the gait2354 C3D files.
TRACKING_CLUSTER_MARKERS = {
    "L.Thigh.Centroid": ("L.Thigh.Upper", "L.Thigh.Front", "L.Thigh.Rear"),
    "R.Thigh.Centroid": ("R.Thigh.Upper", "R.Thigh.Front", "R.Thigh.Rear"),
    "L.Shank.Centroid": ("L.Shank.Upper", "L.Shank.Front", "L.Shank.Rear"),
    "R.Shank.Centroid": ("R.Shank.Upper", "R.Shank.Front", "R.Shank.Rear"),
}

TRACKING_CLUSTER_C3D_SOURCES = {
    "L.Thigh.Centroid": ("LTH2", "LTH3", "LTH4"),
    "R.Thigh.Centroid": ("RTH2", "RTH3", "RTH4"),
    "L.Shank.Centroid": ("LTIB2", "LTIB3", "LTIB4"),
    "R.Shank.Centroid": ("RTIB2", "RTIB3", "RTIB4"),
}
