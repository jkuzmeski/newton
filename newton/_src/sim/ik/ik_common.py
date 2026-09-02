# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common enums and utility kernels shared across IK components."""

from __future__ import annotations

from enum import Enum

import warp as wp

from ..articulation import eval_single_articulation_fk
from ..enums import BodyFlags
from ..model import Model


class IKJacobianType(Enum):
    """
    Specifies the backend used for Jacobian computation in inverse kinematics.
    """

    AUTODIFF = "autodiff"
    """Use Warp's reverse-mode autodiff for every objective."""

    ANALYTIC = "analytic"
    """Use analytic Jacobians for objectives that support them."""

    MIXED = "mixed"
    """Use analytic Jacobians where available, otherwise use autodiff."""


@wp.kernel
def _eval_fk_articulation_batched(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_articulation: wp.array[int],
    joint_q: wp.array2d[wp.float32],
    joint_qd: wp.array2d[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    body_q: wp.array2d[wp.transform],
    body_qd: wp.array2d[wp.spatial_vector],
):
    problem_idx, articulation_idx = wp.tid()

    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_end[articulation_idx]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_articulation,
        joint_q[problem_idx],
        joint_qd[problem_idx],
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        body_flags,
        int(BodyFlags.ALL),
        body_q[problem_idx],
        body_qd[problem_idx],
    )


def eval_fk_batched(
    model: Model,
    joint_q: wp.array2d[wp.float32],
    joint_qd: wp.array2d[wp.float32],
    body_q: wp.array2d[wp.transform],
    body_qd: wp.array2d[wp.spatial_vector],
) -> None:
    """Evaluate forward kinematics for independent batched configurations.

    This evaluates every articulation and writes all body transforms and
    velocities. Unlike :func:`eval_fk`, it does not support articulation masks
    or body-flag filtering.

    Args:
        model: Model whose articulations are evaluated.
        joint_q: Batched generalized coordinates [m or rad], shape
            [batch, joint_coord_count].
        joint_qd: Batched generalized velocities [m/s or rad/s], shape
            [batch, joint_dof_count].
        body_q: Output batched body transforms, shape [batch, body_count].
        body_qd: Output batched body twists [m/s, rad/s], shape
            [batch, body_count].

    Raises:
        ValueError: If an array has an incompatible shape, dtype, or device.
    """
    if joint_q.dtype != wp.float32 or joint_qd.dtype != wp.float32:
        raise ValueError("joint_q and joint_qd must have dtype wp.float32")
    if body_q.dtype != wp.transform or body_qd.dtype != wp.spatial_vector:
        raise ValueError("body_q and body_qd have incompatible dtypes")
    if joint_q.ndim != 2 or joint_q.shape[1] != model.joint_coord_count:
        raise ValueError("joint_q has incompatible shape")
    if joint_qd.ndim != 2 or joint_qd.shape != (joint_q.shape[0], model.joint_dof_count):
        raise ValueError("joint_qd has incompatible shape")
    if body_q.ndim != 2 or body_q.shape != (joint_q.shape[0], model.body_count):
        raise ValueError("body_q has incompatible shape")
    if body_qd.ndim != 2 or body_qd.shape != (joint_q.shape[0], model.body_count):
        raise ValueError("body_qd has incompatible shape")
    arrays = (joint_q, joint_qd, body_q, body_qd)
    if any(array.device != model.device for array in arrays):
        raise ValueError("all batched FK arrays must be on the model device")

    n_problems = joint_q.shape[0]
    if n_problems == 0:
        return
    wp.launch(
        kernel=_eval_fk_articulation_batched,
        dim=[n_problems, model.articulation_count],
        inputs=[
            model.articulation_start,
            model.articulation_end,
            model.joint_articulation,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
            model.body_flags,
        ],
        outputs=[body_q, body_qd],
        device=model.device,
    )


@wp.kernel
def fk_accum(
    joint_ancestor: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_local: wp.array2d[wp.transform],
    body_q: wp.array2d[wp.transform],
):
    problem_idx, joint_idx = wp.tid()
    # joint_ancestor is indexed by joints; joint_parent is body-indexed.
    child = joint_child[joint_idx]
    if child < 0:
        return
    Xw = X_local[problem_idx, joint_idx]
    ancestor = joint_ancestor[joint_idx]
    while ancestor >= 0:
        Xw = X_local[problem_idx, ancestor] * Xw
        ancestor = joint_ancestor[ancestor]
    body_q[problem_idx, child] = Xw


@wp.kernel
def compute_costs(
    residuals: wp.array2d[wp.float32],
    num_residuals: int,
    costs: wp.array[wp.float32],
):
    problem_idx = wp.tid()
    cost = float(0.0)
    for i in range(num_residuals):
        r = residuals[problem_idx, i]
        cost += r * r
    costs[problem_idx] = cost
