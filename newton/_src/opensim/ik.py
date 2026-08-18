# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""Marker-based inverse kinematics for the OpenSim port, in Warp kernels.

Given a model and experimental marker trajectories (from 3D motion capture),
recover the generalized coordinate trajectory that best fits the markers, in the
weighted least-squares sense OpenSim uses:

.. math::

    \min_q \sum_m w_m \, \lVert p_m^{model}(q) - p_m^{exp} \rVert^2

By default every frame is solved together in one device-resident
Levenberg-Marquardt loop: the batched forward kinematics
(``kinematics``) evaluates each frame's base pose and its
finite-difference perturbations, and the residual, normal equations
(:math:`J^\top J`, :math:`J^\top r`), the damped ``num_coordinates`` solves over
the damping ladder (an in-kernel Cholesky), the candidate costs and the
accept/reject bookkeeping all run in Warp kernels. CUDA runs fixed-size LM
iteration chunks in a captured graph and reads one scalar active-frame count per
chunk for early exit, so the per-frame and per-iteration host round-trips of a
sequential solve disappear. Because the frames
are independent, a persistent LM damping schedule reaches the same least-squares
minimum a warm-started sequential solve would, to machine precision.

The per-frame :meth:`InverseKinematics.solve_frame` entry point (and
``batched=False``) still runs the original one-frame-at-a-time solve for
streaming use. Rotational coordinates are solved in radians and reported in
degrees to match OpenSim ``.mot`` output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kinematics import ForwardKinematics
from .mocap import MarkerData, Storage, read_trc, write_storage
from .parser import parse_osim
from .types import OsimModel

wp.set_module_options({"enable_backward": False})

_f64 = wp.float64
_vec3d = wp.vec3d

# Levenberg-Marquardt candidate damping ladder (multipliers on the current lambda).
_LAM_LADDER = np.array([0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0])
_EPS = 1.0e-7


@wp.kernel
def cost_batch_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    obs: wp.array[_vec3d],
    sqrt_w: wp.array[_f64],
    nused: int,
    costs: wp.array[_f64],
):
    """Weighted squared marker error for every pose in a batch."""
    s = wp.tid()
    acc = _f64(0.0)
    for u in range(nused):
        mi = used_idx[u]
        w = sqrt_w[u]
        d = (pos[s, mi] - obs[u]) * w
        acc += d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    costs[s] = acc


@wp.kernel
def jtj_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    sqrt_w: wp.array[_f64],
    eps: _f64,
    nused: int,
    jtj: wp.array2d[_f64],
):
    """Assemble ``J^T J`` from the finite-difference marker Jacobian.

    Row ``0`` of ``pos`` is the base pose; row ``i + 1`` is the pose perturbed in
    coordinate ``i`` by ``eps``.
    """
    i, j = wp.tid()
    acc = _f64(0.0)
    for u in range(nused):
        mi = used_idx[u]
        g = sqrt_w[u] / eps
        base = pos[0, mi]
        ci = (pos[i + 1, mi] - base) * g
        cj = (pos[j + 1, mi] - base) * g
        acc += ci[0] * cj[0] + ci[1] * cj[1] + ci[2] * cj[2]
    jtj[i, j] = acc


@wp.kernel
def jtr_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    obs: wp.array[_vec3d],
    sqrt_w: wp.array[_f64],
    eps: _f64,
    nused: int,
    jtr: wp.array[_f64],
):
    """Assemble ``J^T r`` from the finite-difference marker Jacobian and residual."""
    i = wp.tid()
    acc = _f64(0.0)
    for u in range(nused):
        mi = used_idx[u]
        w = sqrt_w[u]
        base = pos[0, mi]
        r = (base - obs[u]) * w
        c = (pos[i + 1, mi] - base) * (w / eps)
        acc += c[0] * r[0] + c[1] * r[1] + c[2] * r[2]
    jtr[i] = acc


@wp.kernel
def marker_dist_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    obs: wp.array[_vec3d],
    nused: int,
    dist: wp.array[_f64],
):
    """Unweighted Euclidean marker error [m] at the base pose (row ``0``)."""
    u = wp.tid()
    mi = used_idx[u]
    d = pos[0, mi] - obs[u]
    dist[u] = wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


# --------------------------------------------------------------------------- #
# Batched-over-frames kernels: solve every frame in one device-resident LM loop.
# --------------------------------------------------------------------------- #
@wp.kernel
def expand_perturb_kernel(q: wp.array2d[_f64], eps: _f64, nc: int, out: wp.array2d[_f64]):
    """Expand ``q[F, nc]`` into ``out[F*(nc+1), nc]``: base pose + one ``+eps`` column per coordinate."""
    f, r = wp.tid()
    row = f * (nc + 1) + r
    for c in range(nc):
        out[row, c] = q[f, c]
    if r > 0:
        out[row, r - 1] = out[row, r - 1] + eps


@wp.kernel
def jtj_batch_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    sqrt_w: wp.array[_f64],
    eps: _f64,
    nc: int,
    nused: int,
    jtj: wp.array3d[_f64],
):
    """Assemble ``J^T J`` per frame from the finite-difference marker Jacobian."""
    f, i, j = wp.tid()
    base_row = f * (nc + 1)
    acc = _f64(0.0)
    for u in range(nused):
        mi = used_idx[u]
        g = sqrt_w[u] / eps
        base = pos[base_row, mi]
        ci = (pos[base_row + i + 1, mi] - base) * g
        cj = (pos[base_row + j + 1, mi] - base) * g
        acc += ci[0] * cj[0] + ci[1] * cj[1] + ci[2] * cj[2]
    jtj[f, i, j] = acc


@wp.kernel
def jtr_cost_batch_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    obs: wp.array2d[_vec3d],
    sqrt_w: wp.array[_f64],
    eps: _f64,
    nc: int,
    nused: int,
    jtr: wp.array2d[_f64],
    cost0: wp.array[_f64],
):
    """Assemble ``J^T r`` per frame and the base-pose cost (thread ``i == 0`` writes the cost)."""
    f, i = wp.tid()
    base_row = f * (nc + 1)
    acc = _f64(0.0)
    csum = _f64(0.0)
    for u in range(nused):
        mi = used_idx[u]
        w = sqrt_w[u]
        base = pos[base_row, mi]
        r = (base - obs[f, u]) * w
        c = (pos[base_row + i + 1, mi] - base) * (w / eps)
        acc += c[0] * r[0] + c[1] * r[1] + c[2] * r[2]
        if i == 0:
            csum += r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
    jtr[f, i] = acc
    if i == 0:
        cost0[f] = csum


@wp.kernel
def build_ladder_systems_kernel(
    jtj: wp.array3d[_f64],
    jtr: wp.array2d[_f64],
    lam: wp.array[_f64],
    ladder: wp.array[_f64],
    active: wp.array[wp.int32],
    nc: int,
    nlad: int,
    A: wp.array3d[_f64],
    b: wp.array2d[_f64],
):
    """For each ``(frame, ladder)`` build ``A = J^T J + lam*mult*diag_floor`` and ``b = -J^T r``.

    A locally unobserved coordinate keeps a floored diagonal so the damped normal
    equations stay positive definite (matching the host solve's ``diag_floor``).
    """
    f, l = wp.tid()
    s = f * nlad + l
    if active[f] == 0:
        for i in range(nc):
            for j in range(nc):
                A[s, i, j] = wp.where(i == j, _f64(1.0), _f64(0.0))
            b[s, i] = _f64(0.0)
        return
    dmax = _f64(0.0)
    for i in range(nc):
        d = jtj[f, i, i]
        if d > dmax:
            dmax = d
    floor = _f64(1.0e-6) * dmax + _f64(1.0e-12)
    mult = lam[f] * ladder[l]
    for i in range(nc):
        for j in range(nc):
            A[s, i, j] = jtj[f, i, j]
        di = jtj[f, i, i]
        A[s, i, i] = A[s, i, i] + mult * wp.max(di, floor)
        b[s, i] = -jtr[f, i]


@wp.kernel
def chol_solve_kernel(A: wp.array3d[_f64], b: wp.array2d[_f64], nc: int, x: wp.array2d[_f64]):
    """In-place Cholesky (lower) of each SPD system ``A[s]`` and solve ``A[s] x = b[s]``."""
    s = wp.tid()
    for j in range(nc):
        d = A[s, j, j]
        for k in range(j):
            d -= A[s, j, k] * A[s, j, k]
        if d < _f64(1.0e-300):
            d = _f64(1.0e-300)
        d = wp.sqrt(d)
        A[s, j, j] = d
        for i in range(j + 1, nc):
            v = A[s, i, j]
            for k in range(j):
                v -= A[s, i, k] * A[s, j, k]
            A[s, i, j] = v / d
    for i in range(nc):
        v = b[s, i]
        for k in range(i):
            v -= A[s, i, k] * x[s, k]
        x[s, i] = v / A[s, i, i]
    for ii in range(nc):
        i = nc - 1 - ii
        v = x[s, i]
        for k in range(i + 1, nc):
            v -= A[s, k, i] * x[s, k]
        x[s, i] = v / A[s, i, i]


@wp.kernel
def make_candidates_kernel(q: wp.array2d[_f64], dq: wp.array2d[_f64], nc: int, nlad: int, cand: wp.array2d[_f64]):
    """``cand[f*nlad+l] = q[f] + dq[f*nlad+l]`` for the whole ladder."""
    f, l = wp.tid()
    s = f * nlad + l
    for c in range(nc):
        cand[s, c] = q[f, c] + dq[s, c]


@wp.kernel
def cand_cost_batch_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    obs: wp.array2d[_vec3d],
    sqrt_w: wp.array[_f64],
    nlad: int,
    nused: int,
    costs: wp.array2d[_f64],
):
    """Weighted squared marker error for every ladder candidate pose."""
    f, l = wp.tid()
    s = f * nlad + l
    acc = _f64(0.0)
    for u in range(nused):
        mi = used_idx[u]
        w = sqrt_w[u]
        d = (pos[s, mi] - obs[f, u]) * w
        acc += d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    costs[f, l] = acc


@wp.kernel
def accept_kernel(
    cand: wp.array2d[_f64],
    costs: wp.array2d[_f64],
    cost0: wp.array[_f64],
    dq: wp.array2d[_f64],
    ladder: wp.array[_f64],
    accuracy: _f64,
    nc: int,
    nlad: int,
    q: wp.array2d[_f64],
    lam: wp.array[_f64],
    active: wp.array[wp.int32],
):
    """Per frame: pick the best ladder candidate, accept/reject, update ``lam`` and convergence.

    Persistent Levenberg-Marquardt: a rejected step raises the damping and retries
    instead of terminating, so a cold batched start still reaches the same minimum
    the warm-started per-frame solve does. A frame deactivates on a converged step
    or a diverged damping.
    """
    f = wp.tid()
    if active[f] == 0:
        return
    best = int(0)
    bestc = costs[f, 0]
    for l in range(1, nlad):
        if costs[f, l] < bestc:
            bestc = costs[f, l]
            best = l
    if bestc < cost0[f]:
        s = f * nlad + best
        step_inf = _f64(0.0)
        for c in range(nc):
            q[f, c] = cand[s, c]
            a = wp.abs(dq[s, c])
            if a > step_inf:
                step_inf = a
        lam[f] = wp.max(lam[f] * ladder[best] * _f64(0.5), _f64(1.0e-12))
        if step_inf < accuracy:
            active[f] = 0
    else:
        lam[f] = lam[f] * _f64(10.0)
        if lam[f] > _f64(1.0e12):
            active[f] = 0


@wp.kernel
def count_active_kernel(active: wp.array[wp.int32], out: wp.array[wp.int32]):
    """Atomic sum of the per-frame active flags into ``out[0]`` for the early-exit test."""
    f = wp.tid()
    if active[f] != 0:
        wp.atomic_add(out, 0, 1)


@wp.kernel
def marker_dist_batch_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    obs: wp.array2d[_vec3d],
    nused: int,
    dist: wp.array2d[_f64],
):
    """Unweighted Euclidean marker error [m] per ``(frame, used marker)``."""
    f, u = wp.tid()
    mi = used_idx[u]
    d = pos[f, mi] - obs[f, u]
    dist[f, u] = wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


@wp.kernel
def marker_error_reduce_kernel(
    pos: wp.array2d[_vec3d],
    used_idx: wp.array[wp.int32],
    obs: wp.array2d[_vec3d],
    nused: int,
    rms: wp.array[_f64],
    maximum: wp.array[_f64],
):
    """Reduce final marker distances to per-frame RMS and maximum errors."""
    f = wp.tid()
    sum_squared = _f64(0.0)
    max_distance = _f64(0.0)
    for u in range(nused):
        d = pos[f, used_idx[u]] - obs[f, u]
        squared = wp.dot(d, d)
        sum_squared += squared
        max_distance = wp.max(max_distance, wp.sqrt(squared))
    rms[f] = wp.sqrt(sum_squared / _f64(nused))
    maximum[f] = max_distance


@dataclass
class IKResult:
    """Result of a marker-based inverse-kinematics solve.

    Attributes:
        times: Frame times [s], shape ``[num_frames]``.
        coordinate_names: Coordinate names in column order.
        values: Coordinate values in native units (radians/meters),
            shape ``[num_frames, num_coordinates]``.
        motion_types: Motion type per coordinate (``"rotational"`` etc.).
        marker_rms: Per-frame root-mean-square marker error [m].
        marker_max: Per-frame maximum marker error [m].
        marker_names: Markers used in the solve.
    """

    times: np.ndarray
    coordinate_names: list[str]
    values: np.ndarray
    motion_types: list[str]
    marker_rms: np.ndarray
    marker_max: np.ndarray
    marker_names: list[str]

    def values_in_degrees(self) -> np.ndarray:
        """Return coordinate values with rotational columns converted to degrees."""
        out = self.values.copy()
        for i, mt in enumerate(self.motion_types):
            if mt == "rotational":
                out[:, i] = np.rad2deg(out[:, i])
        return out

    def to_storage(self) -> Storage:
        """Return the result as a :class:`~newton.opensim.Storage` (degrees)."""
        return Storage(
            times=self.times,
            labels=list(self.coordinate_names),
            data=self.values_in_degrees(),
            in_degrees=True,
            name="inverse kinematics",
        )

    def write_mot(self, path: str | os.PathLike) -> None:
        """Write the coordinate trajectory to an OpenSim ``.mot`` file (degrees)."""
        write_storage(
            path, self.times, list(self.coordinate_names), self.values_in_degrees(), name="inverse kinematics"
        )


class InverseKinematics:
    """Solve marker-based inverse kinematics frame by frame with Warp kernels.

    Args:
        model: Parsed model IR.
        marker_weights: Optional map of marker name to weight. Markers with a
            positive weight that exist in both the model and the data are used;
            ``None`` weights all common markers equally.
        accuracy: Convergence tolerance on the coordinate step [rad or m].
        max_iters: Maximum Levenberg-Marquardt iterations per frame.
        device: Warp device for the kernels (``"cpu"``, ``"cuda"``, a
            :class:`warp.context.Device`, or ``None`` for the CPU).
        batched: Solve all frames together in one device-resident LM loop
            (default). Set ``False`` to fall back to the sequential,
            warm-started, one-frame-at-a-time solve.
    """

    def __init__(
        self,
        model: OsimModel,
        marker_weights: dict[str, float] | None = None,
        accuracy: float = 1.0e-8,
        max_iters: int = 60,
        device=None,
        batched: bool = True,
    ):
        self.model = model
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.accuracy = accuracy
        self.max_iters = max_iters
        self.batched = batched
        self.marker_weights = marker_weights
        self._model_marker_index = {name: i for i, name in enumerate(self.fk.marker_names)}
        self._default_q = np.array([c.default_value for j in model.joints for c in j.coordinates], dtype=float)
        self.coordinate_names = list(self.fk.coordinate_names)
        self.motion_types = [self.fk.coordinate_motion[c] for c in self.coordinate_names]
        self._ladder = wp.array(_LAM_LADDER, dtype=_f64, device=self.device)

    def _select_markers(self, data_marker_names: list[str]) -> tuple[list[str], np.ndarray]:
        used: list[str] = []
        weights: list[float] = []
        for name in data_marker_names:
            if name not in self._model_marker_index:
                continue
            w = 1.0 if self.marker_weights is None else float(self.marker_weights.get(name, 0.0))
            if w <= 0.0:
                continue
            used.append(name)
            weights.append(w)
        return used, np.asarray(weights, dtype=float)

    def _lm_frame(
        self,
        obs_wp: wp.array,
        used_idx_wp: wp.array,
        sqrt_w_wp: wp.array,
        nused: int,
        q0: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Run the Warp Levenberg-Marquardt iteration for a single frame."""
        n = self.fk.ncoord
        dev = self.device
        eps = _EPS
        q = np.asarray(q0, dtype=float).copy()
        lam = 1.0e-3

        jtj = wp.zeros((n, n), dtype=_f64, device=dev)
        jtr = wp.zeros(n, dtype=_f64, device=dev)
        base_costs = wp.zeros(n + 1, dtype=_f64, device=dev)
        cand_costs = wp.zeros(len(_LAM_LADDER), dtype=_f64, device=dev)

        for _ in range(self.max_iters):
            # Base pose plus one +eps perturbation per coordinate.
            batch = np.repeat(q[None, :], n + 1, axis=0)
            batch[1:] += eps * np.eye(n)
            q_wp = wp.array(batch, dtype=_f64, device=dev)
            pos = self.fk._launch_markers(q_wp)

            wp.launch(
                cost_batch_kernel,
                dim=n + 1,
                inputs=[pos, used_idx_wp, obs_wp, sqrt_w_wp, nused, base_costs],
                device=dev,
            )
            wp.launch(jtj_kernel, dim=(n, n), inputs=[pos, used_idx_wp, sqrt_w_wp, _f64(eps), nused, jtj], device=dev)
            wp.launch(
                jtr_kernel, dim=n, inputs=[pos, used_idx_wp, obs_wp, sqrt_w_wp, _f64(eps), nused, jtr], device=dev
            )

            cost0 = float(base_costs.numpy()[0])
            a = jtj.numpy()
            g = jtr.numpy()
            # Floor the LM damping so a locally unobserved coordinate (e.g. a toe
            # angle with no distinguishing marker) stays put instead of making
            # J^T J + lam*diag singular.
            diag_a = np.diag(a)
            diag_floor = np.maximum(diag_a, 1.0e-6 * float(np.max(diag_a)) + 1.0e-12)

            ladder = lam * _LAM_LADDER
            steps = np.zeros((len(ladder), n))
            cand = np.repeat(q[None, :], len(ladder), axis=0)
            for li, lam_l in enumerate(ladder):
                try:
                    dq = np.linalg.solve(a + lam_l * np.diag(diag_floor), -g)
                except np.linalg.LinAlgError:
                    dq = np.zeros(n)
                steps[li] = dq
                cand[li] = q + dq
            cand_wp = wp.array(cand, dtype=_f64, device=dev)
            cand_pos = self.fk._launch_markers(cand_wp)
            wp.launch(
                cost_batch_kernel,
                dim=len(ladder),
                inputs=[cand_pos, used_idx_wp, obs_wp, sqrt_w_wp, nused, cand_costs],
                device=dev,
            )
            costs = cand_costs.numpy()

            best = int(np.argmin(costs))
            if costs[best] < cost0:
                q = q + steps[best]
                lam = max(ladder[best] * 0.5, 1.0e-12)
                if float(np.max(np.abs(steps[best]))) < self.accuracy:
                    break
            else:
                break

        q_wp = wp.array(q[None, :], dtype=_f64, device=dev)
        pos = self.fk._launch_markers(q_wp)
        dist = wp.zeros(nused, dtype=_f64, device=dev)
        wp.launch(marker_dist_kernel, dim=nused, inputs=[pos, used_idx_wp, obs_wp, nused, dist], device=dev)
        d = dist.numpy()
        rms = float(np.sqrt(np.mean(d**2))) if nused else 0.0
        return q, rms, (float(np.max(d)) if nused else 0.0)

    def solve_frame(
        self, obs_markers: dict[str, np.ndarray], q0: np.ndarray | None = None
    ) -> tuple[np.ndarray, float, float]:
        """Solve one frame.

        Args:
            obs_markers: Observed marker positions [m] as a name->position dict.
            q0: Initial coordinate guess (native units); defaults to the model's
                default pose.

        Returns:
            Tuple of ``(q, rms, max)`` with the solved coordinates (native
            units) and the unweighted RMS and maximum marker errors [m].
        """
        used, weights = self._select_markers(list(obs_markers.keys()))
        q_init = self._default_q.copy() if q0 is None else np.asarray(q0, dtype=float).copy()
        if not used:
            return q_init, 0.0, 0.0
        used_idx = np.array([self._model_marker_index[name] for name in used], np.int32)
        obs = np.array([obs_markers[name] for name in used], np.float64).reshape(-1, 3)
        obs_wp = wp.array(obs, dtype=_vec3d, device=self.device)
        used_idx_wp = wp.array(used_idx, dtype=wp.int32, device=self.device)
        sqrt_w_wp = wp.array(np.sqrt(weights), dtype=_f64, device=self.device)
        return self._lm_frame(obs_wp, used_idx_wp, sqrt_w_wp, len(used), q_init)

    def _initial_pose(self, initial_coordinates: dict[str, float] | None) -> np.ndarray:
        q = self._default_q.copy()
        if initial_coordinates:
            for name, val in initial_coordinates.items():
                if name in self.fk._index:
                    q[self.fk._index[name]] = val
        return q

    def solve(self, markers: MarkerData, initial_coordinates: dict[str, float] | None = None) -> IKResult:
        """Solve inverse kinematics over all frames of ``markers``.

        Uses the batched device-resident solve by default (see
        :class:`InverseKinematics`); pass ``batched=False`` to the constructor for
        the sequential, warm-started fallback.

        Args:
            markers: Experimental marker trajectories.
            initial_coordinates: Optional starting pose overrides (native units).

        Returns:
            The solved coordinate trajectory and per-frame marker errors.
        """
        if self.batched:
            return self._solve_batched(markers, initial_coordinates)
        return self._solve_sequential(markers, initial_coordinates)

    def _solve_batched(self, markers: MarkerData, initial_coordinates: dict[str, float] | None = None) -> IKResult:
        """Solve every frame together in one device-resident Levenberg-Marquardt loop."""
        dev = self.device
        nc = self.fk.ncoord
        eps = _EPS
        n_frames = len(markers.times)
        used, weights = self._select_markers(markers.marker_names)
        q0 = self._initial_pose(initial_coordinates)

        if not used or n_frames == 0:
            return IKResult(
                times=np.asarray(markers.times),
                coordinate_names=self.coordinate_names,
                values=np.repeat(q0[None, :], n_frames, axis=0) if n_frames else np.zeros((0, nc)),
                motion_types=self.motion_types,
                marker_rms=np.zeros(n_frames),
                marker_max=np.zeros(n_frames),
                marker_names=used,
            )

        nused = len(used)
        used_idx = np.array([self._model_marker_index[name] for name in used], np.int32)
        col = {name: i for i, name in enumerate(markers.marker_names)}
        cols = [col[name] for name in used]
        obs = np.ascontiguousarray(markers.data[:, cols, :], dtype=np.float64)

        F = n_frames
        initial_q = np.repeat(q0[None, :], F, axis=0)
        q = wp.array(initial_q, dtype=_f64, device=dev)
        lam = wp.full(F, 1.0e-3, dtype=_f64, device=dev)
        active = wp.ones(F, dtype=wp.int32, device=dev)
        obs_wp = wp.array(obs, dtype=_vec3d, device=dev)
        sqrt_w = wp.array(np.sqrt(weights), dtype=_f64, device=dev)
        used_wp = wp.array(used_idx, dtype=wp.int32, device=dev)
        ladder = self._ladder
        nlad = len(_LAM_LADDER)

        qbatch = wp.empty((F * (nc + 1), nc), dtype=_f64, device=dev)
        jtj = wp.empty((F, nc, nc), dtype=_f64, device=dev)
        jtr = wp.empty((F, nc), dtype=_f64, device=dev)
        cost0 = wp.empty(F, dtype=_f64, device=dev)
        A = wp.empty((F * nlad, nc, nc), dtype=_f64, device=dev)
        b = wp.empty((F * nlad, nc), dtype=_f64, device=dev)
        dq = wp.empty((F * nlad, nc), dtype=_f64, device=dev)
        cand = wp.empty((F * nlad, nc), dtype=_f64, device=dev)
        costs = wp.empty((F, nlad), dtype=_f64, device=dev)

        def iteration():
            wp.launch(expand_perturb_kernel, dim=(F, nc + 1), inputs=[q, _f64(eps), nc, qbatch], device=dev)
            pos = self.fk._launch_markers(qbatch)
            wp.launch(
                jtj_batch_kernel,
                dim=(F, nc, nc),
                inputs=[pos, used_wp, sqrt_w, _f64(eps), nc, nused, jtj],
                device=dev,
            )
            wp.launch(
                jtr_cost_batch_kernel,
                dim=(F, nc),
                inputs=[pos, used_wp, obs_wp, sqrt_w, _f64(eps), nc, nused, jtr, cost0],
                device=dev,
            )
            wp.launch(
                build_ladder_systems_kernel,
                dim=(F, nlad),
                inputs=[jtj, jtr, lam, ladder, active, nc, nlad, A, b],
                device=dev,
            )
            wp.launch(chol_solve_kernel, dim=F * nlad, inputs=[A, b, nc, dq], device=dev)
            wp.launch(make_candidates_kernel, dim=(F, nlad), inputs=[q, dq, nc, nlad, cand], device=dev)
            cpos = self.fk._launch_markers(cand)
            wp.launch(
                cand_cost_batch_kernel,
                dim=(F, nlad),
                inputs=[cpos, used_wp, obs_wp, sqrt_w, nlad, nused, costs],
                device=dev,
            )
            wp.launch(
                accept_kernel,
                dim=F,
                inputs=[cand, costs, cost0, dq, ladder, _f64(self.accuracy), nc, nlad, q, lam, active],
                device=dev,
            )

        active_count = wp.zeros(1, dtype=wp.int32, device=dev)

        def any_active():
            active_count.zero_()
            wp.launch(count_active_kernel, dim=F, inputs=[active, active_count], device=dev)
            return int(active_count.numpy()[0]) != 0

        if dev.is_cuda and self.max_iters > 0:
            chunk_size = min(8, self.max_iters)
            iteration()
            q.assign(initial_q)
            lam.assign(np.full(F, 1.0e-3, dtype=np.float64))
            active.assign(np.ones(F, dtype=np.int32))
            with wp.ScopedCapture(device=dev) as capture:
                for _ in range(chunk_size):
                    iteration()
            converged = False
            for _ in range(self.max_iters // chunk_size):
                wp.capture_launch(capture.graph)
                if not any_active():
                    converged = True
                    break
            if not converged:
                for _ in range(self.max_iters % chunk_size):
                    iteration()
                    if not any_active():
                        break
        else:
            for _ in range(self.max_iters):
                iteration()
                if not any_active():
                    break

        values = q.numpy()
        posf = self.fk._launch_markers(q)
        rms_wp = wp.empty(F, dtype=_f64, device=dev)
        max_wp = wp.empty(F, dtype=_f64, device=dev)
        wp.launch(
            marker_error_reduce_kernel,
            dim=F,
            inputs=[posf, used_wp, obs_wp, nused, rms_wp, max_wp],
            device=dev,
        )
        rms = rms_wp.numpy()
        mx = max_wp.numpy()

        return IKResult(
            times=np.asarray(markers.times),
            coordinate_names=self.coordinate_names,
            values=values,
            motion_types=self.motion_types,
            marker_rms=rms,
            marker_max=mx,
            marker_names=used,
        )

    def _solve_sequential(self, markers: MarkerData, initial_coordinates: dict[str, float] | None = None) -> IKResult:
        """Sequential fallback: solve frames one at a time, warm-starting each from the last."""
        used, weights = self._select_markers(markers.marker_names)
        n_frames = len(markers.times)
        n_coords = len(self.coordinate_names)
        values = np.zeros((n_frames, n_coords))
        rms = np.zeros(n_frames)
        mx = np.zeros(n_frames)

        q = self._initial_pose(initial_coordinates)

        if not used:
            return IKResult(
                times=np.asarray(markers.times),
                coordinate_names=self.coordinate_names,
                values=np.repeat(q[None, :], n_frames, axis=0),
                motion_types=self.motion_types,
                marker_rms=rms,
                marker_max=mx,
                marker_names=used,
            )

        used_idx = np.array([self._model_marker_index[name] for name in used], np.int32)
        used_idx_wp = wp.array(used_idx, dtype=wp.int32, device=self.device)
        sqrt_w_wp = wp.array(np.sqrt(weights), dtype=_f64, device=self.device)
        name_to_frame_col = {name: i for i, name in enumerate(markers.marker_names)}
        cols = [name_to_frame_col[name] for name in used]

        for fi in range(n_frames):
            obs = np.ascontiguousarray(markers.data[fi][cols], dtype=np.float64)
            obs_wp = wp.array(obs, dtype=_vec3d, device=self.device)
            q, rms[fi], mx[fi] = self._lm_frame(obs_wp, used_idx_wp, sqrt_w_wp, len(used), q)
            values[fi] = q

        return IKResult(
            times=np.asarray(markers.times),
            coordinate_names=self.coordinate_names,
            values=values,
            motion_types=self.motion_types,
            marker_rms=rms,
            marker_max=mx,
            marker_names=used,
        )


def solve_marker_ik(
    model: OsimModel | str | os.PathLike,
    markers: MarkerData | str | os.PathLike,
    marker_weights: dict[str, float] | None = None,
    initial_coordinates: dict[str, float] | None = None,
    accuracy: float = 1.0e-8,
    device=None,
) -> IKResult:
    """Run marker-based inverse kinematics end to end.

    Args:
        model: A parsed :class:`OsimModel`, or a path/XML string to parse.
        markers: A :class:`~newton.opensim.MarkerData`, or a path to a
            ``.trc`` file.
        marker_weights: Optional marker name to weight map (default: uniform).
        initial_coordinates: Optional first-frame pose overrides (native units).
        accuracy: Convergence tolerance on the coordinate step.
        device: Warp device for the kernels (``None`` for the CPU).

    Returns:
        The solved coordinate trajectory and per-frame marker errors.
    """
    if not isinstance(model, OsimModel):
        model = parse_osim(model)
    if not isinstance(markers, MarkerData):
        markers = read_trc(markers)
    ik = InverseKinematics(model, marker_weights=marker_weights, accuracy=accuracy, device=device)
    return ik.solve(markers, initial_coordinates=initial_coordinates)
