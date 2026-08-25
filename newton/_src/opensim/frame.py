# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coordinate-frame conversion at the OpenSim-to-Newton world boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core import Axis, quat_between_axes


@dataclass(frozen=True, slots=True)
class OpenSimFrameConverter:
    """Map OpenSim's Y-up world into a configured Newton world.

    OpenSim model-local data and generalized coordinates remain unchanged. The
    converter left-composes absolute world poses and rotates world vectors. With
    Newton's default Z-up target, ``(x, y, z)`` maps to ``(x, -z, y)``.

    Args:
        target_up_axis: Target Newton up axis.
    """

    target_up_axis: Axis = Axis.Z

    @property
    def world_xform(self) -> wp.transform:
        """Transform from the OpenSim Y-up world to the target Newton world."""
        return wp.transform(wp.vec3(0.0), quat_between_axes(Axis.Y, self.target_up_axis))

    @property
    def matrix(self) -> np.ndarray:
        """Rotation matrix from OpenSim world vectors to target-world vectors."""
        if self.target_up_axis == Axis.Y:
            return np.eye(3, dtype=np.float64)
        if self.target_up_axis == Axis.Z:
            return np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
        return np.asarray(wp.quat_to_matrix(self.world_xform.q), dtype=np.float64).reshape(3, 3)

    def transform_vectors(self, values: np.ndarray) -> np.ndarray:
        """Rotate row-vector world quantities into the target Newton frame."""
        array = np.asarray(values)
        if array.shape[-1] != 3:
            raise ValueError("values must have a trailing dimension of 3")
        return array @ self.matrix.T

    def inverse_vectors(self, values: np.ndarray) -> np.ndarray:
        """Rotate row-vector world quantities back into OpenSim's Y-up frame."""
        array = np.asarray(values)
        if array.shape[-1] != 3:
            raise ValueError("values must have a trailing dimension of 3")
        return array @ self.matrix
