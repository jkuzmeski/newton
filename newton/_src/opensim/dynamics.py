# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""Inverse and forward dynamics for the OpenSim port, in Warp kernels.

Given a model and a generalized-coordinate trajectory, recover the joint
(generalized) forces that produce it, reproducing OpenSim's
``InverseDynamicsTool`` 1-for-1. The tool's pipeline is:

1. pad the coordinate storage by reflection (``Storage::pad``),
2. low-pass filter it with a zero-lag Butterworth IIR (``Signal::LowpassIIR``,
   6 Hz by default),
3. fit a quintic GCV spline and evaluate its
   0th/1st/2nd derivatives to get :math:`q,\dot q,\ddot q`, and
4. solve the Newton-Euler inverse dynamics per frame, excluding muscle forces.

The dynamics core runs in Warp kernels. It reuses the batched forward kinematics
(``kinematics``): for each frame it evaluates the pose,
a velocity/acceleration finite-difference stencil, and one
:math:`\pm\varepsilon` perturbation per coordinate in a single launch, then
:func:`bodyforce_kernel` forms each body's spatial force (inertial minus gravity
minus external loads) and :func:`tau_kernel` projects them onto the generalized
coordinates with the transpose of the geometric Jacobian,
:math:`\tau_i=\sum_b J_{\omega,i}^b\!\cdot N_b + J_{v,i}^b\!\cdot F_b`. Only the
signal preprocessing and spline fitting run on the host, as OpenSim bakes them
before the solve.

:class:`ForwardDynamics` inverts this: it recovers the accelerations produced by
applied generalized forces (and optional external loads) and integrates the
equations of motion, reproducing OpenSim's ``ForwardTool``. Because the
Newton-Euler inverse dynamics is affine in the accelerations, the mass matrix and
bias forces are read straight out of the same Warp kernels (the composite
rigid-body method), so only the small dense solve and the time stepping run on
the host.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from . import gcvspl
from .kinematics import ForwardKinematics, fk_kernel
from .mocap import Storage, read_storage, write_storage
from .parser import parse_osim
from .types import OsimModel

wp.set_module_options({"enable_backward": False})

_f64 = wp.float64
_vec3d = wp.vec3d
_mat33d = wp.mat33d
_mat44d = wp.mat44d


@wp.func
def _rot_of(X: _mat44d) -> _mat33d:
    return _mat33d(X[0, 0], X[0, 1], X[0, 2], X[1, 0], X[1, 1], X[1, 2], X[2, 0], X[2, 1], X[2, 2])


@wp.func
def _pos_of(X: _mat44d) -> _vec3d:
    return _vec3d(X[0, 3], X[1, 3], X[2, 3])


@wp.func
def _vee(M: _mat33d) -> _vec3d:
    return _vec3d(_f64(0.5) * (M[2, 1] - M[1, 2]), _f64(0.5) * (M[0, 2] - M[2, 0]), _f64(0.5) * (M[1, 0] - M[0, 1]))


@wp.kernel
def bodyforce_kernel(
    poses: wp.array2d[_mat44d],
    mass: wp.array[_f64],
    rcom: wp.array[_vec3d],
    inertia: wp.array[_mat33d],
    gravity: _vec3d,
    stride: int,
    num_ext: int,
    ext_body: wp.array[wp.int32],
    wrench: wp.array3d[_f64],
    h: _f64,
    torque_out: wp.array2d[_vec3d],
    force_out: wp.array2d[_vec3d],
):
    """Form each body's required spatial force (about its mass center, in ground).

    One thread handles one (frame, body). The base pose and the two
    velocity/acceleration stencil poses live at the start of each frame's pose
    block; external loads are transferred from their ground application point to
    the body mass center.
    """
    f, b = wp.tid()
    m = mass[b]
    base = f * stride
    x0 = poses[base + 0, b]
    xp = poses[base + 1, b]
    xm = poses[base + 2, b]
    r0 = _rot_of(x0)
    rp = _rot_of(xp)
    rm = _rot_of(xm)
    rc = rcom[b]
    pc0 = _pos_of(x0) + r0 * rc
    pcp = _pos_of(xp) + rp * rc
    pcm = _pos_of(xm) + rm * rc
    accel = (pcp - _f64(2.0) * pc0 + pcm) / (h * h)
    r0t = wp.transpose(r0)
    rdot = (rp - rm) / (_f64(2.0) * h)
    rddot = (rp - _f64(2.0) * r0 + rm) / (h * h)
    omega = _vee(rdot * r0t)
    alpha = _vee(rddot * r0t)
    iw = r0 * inertia[b] * r0t
    torque = iw * alpha + wp.cross(omega, iw * omega)
    force = m * accel - m * gravity
    for e in range(num_ext):
        if ext_body[e] == b:
            fe = _vec3d(wrench[f, e, 0], wrench[f, e, 1], wrench[f, e, 2])
            pe = _vec3d(wrench[f, e, 3], wrench[f, e, 4], wrench[f, e, 5])
            te = _vec3d(wrench[f, e, 6], wrench[f, e, 7], wrench[f, e, 8])
            force = force - fe
            torque = torque - (wp.cross(pe - pc0, fe) + te)
    torque_out[f, b] = torque
    force_out[f, b] = force


@wp.kernel
def tau_kernel(
    poses: wp.array2d[_mat44d],
    rcom: wp.array[_vec3d],
    torque_in: wp.array2d[_vec3d],
    force_in: wp.array2d[_vec3d],
    stride: int,
    nbody: int,
    eps: _f64,
    tau_out: wp.array2d[_f64],
):
    """Project body spatial forces onto coordinate ``i`` with the Jacobian transpose.

    One thread handles one (frame, coordinate). The geometric Jacobian columns
    come from the central-difference poses stored for coordinate ``i``.
    """
    f, i = wp.tid()
    base = f * stride
    val = _f64(0.0)
    for b in range(nbody):
        rc = rcom[b]
        xp = poses[base + 3 + 2 * i, b]
        xm = poses[base + 4 + 2 * i, b]
        r0 = _rot_of(poses[base + 0, b])
        pcp = _pos_of(xp) + _rot_of(xp) * rc
        pcm = _pos_of(xm) + _rot_of(xm) * rc
        jv = (pcp - pcm) / (_f64(2.0) * eps)
        dr = (_rot_of(xp) - _rot_of(xm)) / (_f64(2.0) * eps)
        jw = _vee(dr * wp.transpose(r0))
        val += wp.dot(jw, torque_in[f, b]) + wp.dot(jv, force_in[f, b])
    tau_out[f, i] = val


@wp.kernel
def id_stencil_kernel(
    q: wp.array2d[_f64],
    qd: wp.array2d[_f64],
    qdd: wp.array2d[_f64],
    h: _f64,
    eps: _f64,
    nc: int,
    poses_in: wp.array2d[_f64],
):
    r"""Write the inverse-dynamics finite-difference coordinate stencil on device.

    One thread fills one ``(frame, stencil-row, coordinate)`` entry of the
    ``[num_frames * (3 + 2 * nc), nc]`` coordinate batch consumed by the batched
    forward kinematics: row 0 is the base pose, rows 1/2 the second-order
    velocity/acceleration stencil, and rows ``3 + 2 i`` / ``4 + 2 i`` the
    :math:`\pm\varepsilon` Jacobian perturbations of coordinate ``i``.
    """
    f, r, c = wp.tid()
    stride = 3 + 2 * nc
    base = f * stride
    qv = q[f, c]
    if r == 0:
        poses_in[base + 0, c] = qv
    elif r == 1:
        poses_in[base + 1, c] = qv + h * qd[f, c] + _f64(0.5) * h * h * qdd[f, c]
    elif r == 2:
        poses_in[base + 2, c] = qv - h * qd[f, c] + _f64(0.5) * h * h * qdd[f, c]
    else:
        i = (r - 3) // 2
        val = qv
        if c == i:
            if (r - 3) % 2 == 0:
                val = qv + eps
            else:
                val = qv - eps
        poses_in[base + r, c] = val


# --------------------------------------------------------------------------- #
# Multi-trajectory forward-dynamics kernels: integrate B trajectories in lockstep
# so the per-step Cholesky solve runs over B systems and fills the GPU that a
# single sequential trajectory leaves idle. The whole RK4 step is device-resident
# and captured once with a CUDA graph, then replayed per timestep.
# --------------------------------------------------------------------------- #
@wp.kernel
def build_id_batch_kernel(
    q: wp.array2d[_f64],
    v: wp.array2d[_f64],
    h: _f64,
    eps: _f64,
    nc: int,
    stride: int,
    out: wp.array2d[_f64],
):
    """Expand ``(q[B, nc], v[B, nc])`` into the inverse-dynamics finite-difference pose batch.

    For trajectory ``b`` and sub-problem ``j`` (0 gives the bias, ``i+1`` the unit
    acceleration of coordinate ``i``) the ``stride`` rows are the base pose, the
    +/- velocity-acceleration stencil, and the +/- ``eps`` Jacobian columns.
    """
    b, j, r = wp.tid()
    frame = b * (nc + 1) + j
    row = frame * stride + r
    for c in range(nc):
        out[row, c] = q[b, c]
    if r == 1 or r == 2:
        sgn = wp.where(r == 1, _f64(1.0), _f64(-1.0))
        for c in range(nc):
            out[row, c] = q[b, c] + sgn * h * v[b, c]
        if j >= 1:
            out[row, j - 1] = out[row, j - 1] + _f64(0.5) * h * h
    elif r >= 3:
        i = (r - 3) // 2
        if (r - 3) % 2 == 0:
            out[row, i] = out[row, i] + eps
        else:
            out[row, i] = out[row, i] - eps


@wp.kernel
def expand_external_wrench_kernel(wrench: wp.array3d[_f64], columns_per_frame: int, out: wp.array3d[_f64]):
    """Repeat each frame's external wrenches across its inverse-dynamics columns."""
    column, external, component = wp.tid()
    out[column, external, component] = wrench[column // columns_per_frame, external, component]


@wp.kernel
def mass_bias_kernel(
    tau_id: wp.array2d[_f64],
    tau_app: wp.array2d[_f64],
    nc: int,
    mass: wp.array3d[_f64],
    rhs: wp.array2d[_f64],
):
    """Per trajectory build the symmetric mass matrix and ``rhs = tau_app - bias``.

    ``bias`` is the ID column for zero acceleration and ``mass[:, i]`` is the ID
    response to a unit acceleration of coordinate ``i`` minus the bias (composite
    rigid-body method), symmetrized.
    """
    b, i, k = wp.tid()
    f0 = b * (nc + 1)
    bi = tau_id[f0, i]
    bk = tau_id[f0, k]
    mik = tau_id[f0 + k + 1, i] - bi
    mki = tau_id[f0 + i + 1, k] - bk
    mass[b, i, k] = _f64(0.5) * (mik + mki)
    if k == 0:
        rhs[b, i] = tau_app[b, i] - bi


@wp.kernel
def constrain_fixed_coordinates_kernel(
    mass: wp.array3d[_f64],
    rhs: wp.array2d[_f64],
    fixed: wp.array[wp.int32],
):
    """Impose zero acceleration on locked and zero-width clamped coordinates."""
    b, i, k = wp.tid()
    if fixed[i] != 0 or fixed[k] != 0:
        mass[b, i, k] = _f64(1.0) if i == k and fixed[i] != 0 else _f64(0.0)
    if k == 0 and fixed[i] != 0:
        rhs[b, i] = _f64(0.0)


@wp.kernel
def spd_solve_kernel(A: wp.array3d[_f64], b: wp.array2d[_f64], nc: int, x: wp.array2d[_f64]):
    """In-place Cholesky of each SPD system ``A[s]`` and solve ``A[s] x = b[s]``."""
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
def rk4_stage_kernel(
    q: wp.array2d[_f64],
    v: wp.array2d[_f64],
    a: wp.array2d[_f64],
    cq: _f64,
    ca: _f64,
    qs: wp.array2d[_f64],
    vs: wp.array2d[_f64],
):
    """``qs = q + cq*v`` ; ``vs = v + ca*a`` (Runge-Kutta stages 1 and 2)."""
    b, c = wp.tid()
    qs[b, c] = q[b, c] + cq * v[b, c]
    vs[b, c] = v[b, c] + ca * a[b, c]


@wp.kernel
def rk4_stage3_kernel(
    q: wp.array2d[_f64],
    v: wp.array2d[_f64],
    a1: wp.array2d[_f64],
    a2: wp.array2d[_f64],
    cq: _f64,
    cav: _f64,
    ca: _f64,
    qs: wp.array2d[_f64],
    vs: wp.array2d[_f64],
):
    """``qs = q + cq*(v + cav*a1)`` ; ``vs = v + ca*a2`` (Runge-Kutta stages 3 and 4)."""
    b, c = wp.tid()
    qs[b, c] = q[b, c] + cq * (v[b, c] + cav * a1[b, c])
    vs[b, c] = v[b, c] + ca * a2[b, c]


@wp.kernel
def rk4_update_kernel(
    q: wp.array2d[_f64],
    v: wp.array2d[_f64],
    a1: wp.array2d[_f64],
    a2: wp.array2d[_f64],
    a3: wp.array2d[_f64],
    a4: wp.array2d[_f64],
    dt: _f64,
):
    """``q += dt*(v + dt/6*(a1+a2+a3))`` ; ``v += dt/6*(a1+2a2+2a3+a4)``."""
    b, c = wp.tid()
    s = dt / _f64(6.0)
    q[b, c] = q[b, c] + dt * (v[b, c] + s * (a1[b, c] + a2[b, c] + a3[b, c]))
    v[b, c] = v[b, c] + s * (a1[b, c] + _f64(2.0) * a2[b, c] + _f64(2.0) * a3[b, c] + a4[b, c])


@wp.kernel
def semi_implicit_update_kernel(q: wp.array2d[_f64], v: wp.array2d[_f64], a: wp.array2d[_f64], dt: _f64):
    """Symplectic Euler update ``v += dt*a`` ; ``q += dt*v``."""
    b, c = wp.tid()
    v[b, c] = v[b, c] + dt * a[b, c]
    q[b, c] = q[b, c] + dt * v[b, c]


@wp.kernel
def record_state_kernel(q: wp.array2d[_f64], v: wp.array2d[_f64], k: int, out: wp.array3d[_f64], ncoord: int):
    """Copy the current state into trajectory sample ``k`` (on device, no host sync)."""
    b, c = wp.tid()
    out[k, b, c] = q[b, c]
    out[k, b, ncoord + c] = v[b, c]


@wp.kernel
def record_state_dynamic_kernel(
    q: wp.array2d[_f64],
    v: wp.array2d[_f64],
    sample: wp.array[wp.int32],
    out: wp.array3d[_f64],
    ncoord: int,
):
    """Record the current state at a device-resident sample index."""
    b, c = wp.tid()
    index = sample[0]
    out[index, b, c] = q[b, c]
    out[index, b, ncoord + c] = v[b, c]


@wp.kernel
def increment_state_sample_kernel(sample: wp.array[wp.int32]):
    """Advance a device trajectory sample index after recording."""
    sample[0] = sample[0] + 1


# --------------------------------------------------------------------------- #
# Host-side signal preprocessing (OpenSim Storage::pad / Signal::LowpassIIR).
# --------------------------------------------------------------------------- #
def pad_signal(signal: np.ndarray, num_pad: int) -> np.ndarray:
    """Reflect-and-negate padding through the endpoints (OpenSim ``Signal::Pad``).

    Prepends and appends ``num_pad`` points that mirror the signal about each
    endpoint, preserving value and slope, matching ``Storage::pad``.
    """
    n = len(signal)
    if num_pad <= 0:
        return np.asarray(signal, float).copy()
    out = np.empty(n + 2 * num_pad)
    for i in range(num_pad):
        out[i] = 2.0 * signal[0] - signal[num_pad - i]
    out[num_pad : num_pad + n] = signal
    j = n - 2
    for i in range(num_pad):
        out[num_pad + n + i] = 2.0 * signal[n - 1] - signal[j]
        j -= 1
    return out


def lowpass_iir(signal: np.ndarray, dt: float, cutoff: float) -> np.ndarray:
    """Zero-lag 3rd-order Butterworth low-pass (OpenSim ``Signal::LowpassIIR``).

    Bilinear-transform IIR applied forward then reversed to cancel phase lag.
    """
    n = len(signal)
    if n < 4:
        return np.asarray(signal, float).copy()
    fs = 1.0 / dt
    if cutoff >= 0.5 * fs:
        cutoff = 0.49 * fs
    wa = np.tan(np.pi * cutoff * dt)
    wa2, wa3 = wa * wa, wa * wa * wa
    denom = (wa + 1.0) * (wa2 + wa + 1.0)
    a = (wa3 / denom, 3.0 * wa3 / denom, 3.0 * wa3 / denom, wa3 / denom)
    b = (
        1.0,
        (3.0 * wa3 + 2.0 * wa2 - 2.0 * wa - 3.0) / denom,
        (3.0 * wa3 - 2.0 * wa2 - 2.0 * wa + 3.0) / denom,
        (wa - 1.0) * (wa2 - wa + 1.0) / denom,
    )

    def _forward(x):
        y = np.array(x, float)
        for i in range(3, len(x)):
            y[i] = (
                a[0] * x[i]
                + a[1] * x[i - 1]
                + a[2] * x[i - 2]
                + a[3] * x[i - 3]
                - b[1] * y[i - 1]
                - b[2] * y[i - 2]
                - b[3] * y[i - 3]
            )
        return y

    forward = _forward(signal)
    return _forward(forward[::-1])[::-1].copy()


def differentiate_coordinates(
    times: np.ndarray,
    values: np.ndarray,
    is_rotational: list[bool],
    output_times: np.ndarray | None = None,
    cutoff: float = 6.0,
    in_degrees: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter and differentiate coordinate signals as OpenSim's ID tool does.

    Pads and low-pass filters each column, converts rotational columns to
    radians, fits a quintic GCVSpline, and evaluates value/velocity/acceleration
    at ``output_times`` (default: the input ``times``).

    Args:
        times: Sample times [s], shape ``[num_samples]``.
        values: Coordinate samples, shape ``[num_samples, num_coordinates]``.
        is_rotational: Whether each column is a rotational coordinate.
        output_times: Times to evaluate at; defaults to ``times``.
        cutoff: Butterworth cutoff [Hz]; ``<= 0`` disables filtering.
        in_degrees: Whether rotational input columns are in degrees.

    Returns:
        ``(q, qd, qdd)`` each shape ``[len(output_times), num_coordinates]`` in
        native units (radians for rotational coordinates).
    """
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if output_times is None:
        output_times = times
    output_times = np.asarray(output_times, float)
    num_pad = len(times) // 2
    dt = float(np.min(np.diff(times)))
    tpad = pad_signal(times, num_pad)
    ncol = values.shape[1]
    # Fit each column on the host (Woltring banded solve), then evaluate the
    # fitted splines (value, velocity, acceleration) on-device in one launch.
    coeffs = np.zeros((ncol, len(tpad)))
    for c in range(ncol):
        col = pad_signal(values[:, c], num_pad)
        if cutoff and cutoff > 0.0:
            col = lowpass_iir(col, dt, cutoff)
        if is_rotational[c] and in_degrees:
            col = np.deg2rad(col)
        coeffs[c] = gcvspl.fit_gcvspline(tpad, col, half_order=3)
    out = gcvspl.eval_gcvspline_batch(tpad, coeffs, output_times)
    return out[:, :, 0], out[:, :, 1], out[:, :, 2]


# --------------------------------------------------------------------------- #
# External loads (ground reaction forces).
# --------------------------------------------------------------------------- #
@dataclass
class ExternalForce:
    """A single applied force/point/torque record from an ``ExternalLoads`` set.

    Attributes:
        applied_to_body: Body the wrench acts on.
        force_identifier: Column-name prefix locating the force [N] in the data.
        point_identifier: Column-name prefix locating the point [m] in the data.
        torque_identifier: Column-name prefix locating the free torque [N*m].
        applies_force: Whether the force is active.
    """

    applied_to_body: str
    force_identifier: str
    point_identifier: str
    torque_identifier: str
    applies_force: bool = True


@dataclass
class ExternalLoads:
    """A set of applied external forces sampled from a data storage (GRF ``.mot``).

    Attributes:
        forces: The individual :class:`ExternalForce` records.
        data: The storage supplying force/point/torque columns over time.
    """

    forces: list[ExternalForce] = field(default_factory=list)
    data: Storage | None = None
    _splines: dict = field(default_factory=dict, repr=False)

    def _columns(self, identifier: str) -> list[int]:
        return [i for i, lab in enumerate(self.data.labels) if lab.startswith(identifier)][:3]

    def _spline(self, col: int):
        if col not in self._splines:
            self._splines[col] = gcvspl.fit_gcvspline(self.data.times, self.data.data[:, col], half_order=2)
        return self._splines[col]

    def sample(self, output_times: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Return applied bodies and per-frame wrenches at ``output_times``.

        Each active force is evaluated with a cubic GCVSpline (matching OpenSim's
        ``ExternalForce``) into a ground-frame ``[Fx Fy Fz Px Py Pz Tx Ty Tz]``.

        Returns:
            ``(bodies, wrenches)`` with ``wrenches`` shape
            ``[len(output_times), num_active_forces, 9]``.
        """
        active = [ef for ef in self.forces if ef.applies_force]
        bodies = [ef.applied_to_body for ef in active]
        out = np.zeros((len(output_times), len(active), 9))
        for e, ef in enumerate(active):
            cols = (
                self._columns(ef.force_identifier)
                + self._columns(ef.point_identifier)
                + self._columns(ef.torque_identifier)
            )
            for k, col in enumerate(cols):
                coeffs = self._spline(col)
                out[:, e, k] = [gcvspl.eval_gcvspline(self.data.times, coeffs, t, 0) for t in output_times]
        return bodies, out


def read_external_loads(setup_path: str | os.PathLike, data_path: str | os.PathLike | None = None) -> ExternalLoads:
    """Parse an OpenSim ``ExternalLoads`` setup XML and its data storage.

    Args:
        setup_path: Path to the ``ExternalLoads`` (``*_grf.xml``) file.
        data_path: Path to the GRF ``.mot``; defaults to the ``<datafile>``
            referenced by the setup, resolved next to it.

    Returns:
        The parsed :class:`ExternalLoads`.
    """
    root = ET.parse(os.fspath(setup_path)).getroot()
    loads = root.find(".//ExternalLoads")
    if loads is None:
        loads = root

    def _text(element, tag, default=""):
        el = element.find(tag)
        return el.text.strip() if el is not None and el.text else default

    forces = []
    for ef in loads.findall(".//ExternalForce"):
        forces.append(
            ExternalForce(
                applied_to_body=_text(ef, "applied_to_body"),
                force_identifier=_text(ef, "force_identifier"),
                point_identifier=_text(ef, "point_identifier"),
                torque_identifier=_text(ef, "torque_identifier"),
                applies_force=_text(ef, "appliesForce", "true").lower() != "false",
            )
        )
    if data_path is None:
        datafile_el = loads.find("datafile")
        datafile = datafile_el.text.strip() if datafile_el is not None and datafile_el.text else ""
        data_path = os.path.join(os.path.dirname(os.fspath(setup_path)), datafile)
    return ExternalLoads(forces=forces, data=read_storage(data_path))


# --------------------------------------------------------------------------- #
# Result + solver.
# --------------------------------------------------------------------------- #
@dataclass
class IDResult:
    """Result of an inverse-dynamics solve.

    Attributes:
        times: Frame times [s], shape ``[num_frames]``.
        coordinate_names: Coordinate names in column order.
        generalized_forces: Joint moments [N*m] (rotational) or forces [N]
            (translational), shape ``[num_frames, num_coordinates]``.
        motion_types: Motion type per coordinate (``"rotational"`` etc.).
        coordinates: Filtered coordinate values [m or rad], shape
            ``[num_frames, num_coordinates]``. Available for results produced by
            :meth:`InverseDynamics.solve_from_motion`.
        speeds: Filtered coordinate speeds [m/s or rad/s], same shape as
            ``coordinates``. Available for results produced by
            :meth:`InverseDynamics.solve_from_motion`.
        accelerations: Filtered coordinate accelerations [m/s^2 or rad/s^2],
            same shape as ``coordinates``. Available for results produced by
            :meth:`InverseDynamics.solve_from_motion`.
    """

    times: np.ndarray
    coordinate_names: list[str]
    generalized_forces: np.ndarray
    motion_types: list[str]
    coordinates: np.ndarray | None = None
    speeds: np.ndarray | None = None
    accelerations: np.ndarray | None = None

    @property
    def column_labels(self) -> list[str]:
        """Return OpenSim-style labels (``<coord>_moment`` / ``<coord>_force``)."""
        return [
            f"{n}_moment" if mt == "rotational" else f"{n}_force"
            for n, mt in zip(self.coordinate_names, self.motion_types, strict=True)
        ]

    def to_storage(self) -> Storage:
        """Return the generalized forces as a :class:`Storage`."""
        return Storage(
            times=self.times,
            labels=self.column_labels,
            data=self.generalized_forces,
            in_degrees=False,
            name="Inverse Dynamics Generalized Forces",
        )

    def write_sto(self, path: str | os.PathLike) -> None:
        """Write the generalized forces to an OpenSim ``.sto`` file."""
        write_storage(
            path,
            self.times,
            self.column_labels,
            self.generalized_forces,
            name="Inverse Dynamics Generalized Forces",
            in_degrees=False,
        )


class InverseDynamics:
    """Solve inverse dynamics frame by frame with Warp kernels.

    Args:
        model: Parsed model IR.
        device: Warp device for the kernels (``"cpu"``, ``"cuda"``, a
            :class:`warp.context.Device`, or ``None`` for the CPU).
    """

    def __init__(self, model: OsimModel, device=None):
        self.model = model
        self.fk = ForwardKinematics(model, device=device)
        self.device = self.fk.device
        self.coordinate_names = list(self.fk.coordinate_names)
        self.motion_types = [self.fk.coordinate_motion[c] for c in self.coordinate_names]
        self.ncoord = self.fk.ncoord
        nb = self.fk.nbody
        body_index = {name: i for i, name in enumerate(self.fk.body_names)}
        self._body_index = body_index
        mass = np.zeros(nb)
        rcom = np.zeros((nb, 3))
        inertia = np.zeros((nb, 3, 3))
        for body in model.bodies:
            i = body_index[body.name]
            mass[i] = body.mass
            rcom[i] = body.mass_center
            t = body.inertia
            inertia[i] = [[t[0], t[3], t[4]], [t[3], t[1], t[5]], [t[4], t[5], t[2]]]
        self.gravity = np.asarray(model.gravity, dtype=float)
        self._mass = wp.array(mass, dtype=_f64, device=self.device)
        self._rcom = wp.array(rcom, dtype=_vec3d, device=self.device)
        self._inertia = wp.array(inertia.reshape(nb, 9), dtype=_mat33d, device=self.device)

    def solve(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        external_bodies: list[str] | None = None,
        external_wrenches: np.ndarray | None = None,
        h: float = 1.0e-4,
        eps: float = 1.0e-6,
    ) -> np.ndarray:
        """Return generalized forces for a coordinate trajectory.

        Args:
            q: Coordinate values in native units, shape ``[num_frames, num_coordinates]``.
            qd: Coordinate velocities, same shape.
            qdd: Coordinate accelerations, same shape.
            external_bodies: Bodies each external wrench acts on (length ``num_ext``).
            external_wrenches: Ground-frame ``[Fx Fy Fz Px Py Pz Tx Ty Tz]`` per
                frame and force, shape ``[num_frames, num_ext, 9]``.
            h: Finite-difference step for velocities/accelerations [s].
            eps: Finite-difference step for the geometric Jacobian [rad or m].

        Returns:
            Generalized forces, shape ``[num_frames, num_coordinates]``.
        """
        q = np.ascontiguousarray(q, dtype=float)
        qd = np.ascontiguousarray(qd, dtype=float)
        qdd = np.ascontiguousarray(qdd, dtype=float)
        n_frames, nc = q.shape
        nb = self.fk.nbody
        stride = 3 + 2 * nc
        d_q = wp.array(q, dtype=_f64, device=self.device)
        d_qd = wp.array(qd, dtype=_f64, device=self.device)
        d_qdd = wp.array(qdd, dtype=_f64, device=self.device)
        q_wp = wp.empty((n_frames * stride, nc), dtype=_f64, device=self.device)
        wp.launch(
            id_stencil_kernel,
            dim=(n_frames, stride, nc),
            inputs=[d_q, d_qd, d_qdd, _f64(h), _f64(eps), nc],
            outputs=[q_wp],
            device=self.device,
        )
        poses = self.fk._launch_body_transforms(q_wp)

        if external_bodies:
            ext_idx = np.array([self._body_index[name] for name in external_bodies], dtype=np.int32)
            wrench = np.ascontiguousarray(external_wrenches, dtype=float)
        else:
            ext_idx = np.zeros(0, dtype=np.int32)
            wrench = np.zeros((n_frames, 0, 9))
        d_ext = wp.array(ext_idx, dtype=wp.int32, device=self.device)
        d_wrench = wp.array(wrench, dtype=_f64, device=self.device)

        torque = wp.empty((n_frames, nb), dtype=_vec3d, device=self.device)
        force = wp.empty((n_frames, nb), dtype=_vec3d, device=self.device)
        tau = wp.empty((n_frames, nc), dtype=_f64, device=self.device)
        wp.launch(
            bodyforce_kernel,
            dim=(n_frames, nb),
            inputs=[
                poses,
                self._mass,
                self._rcom,
                self._inertia,
                _vec3d(*self.gravity),
                stride,
                len(ext_idx),
                d_ext,
                d_wrench,
                _f64(h),
            ],
            outputs=[torque, force],
            device=self.device,
        )
        wp.launch(
            tau_kernel,
            dim=(n_frames, nc),
            inputs=[poses, self._rcom, torque, force, stride, nb, _f64(eps)],
            outputs=[tau],
            device=self.device,
        )
        return tau.numpy()

    def solve_from_motion(
        self,
        coordinates: Storage | str | os.PathLike,
        external_loads: ExternalLoads | None = None,
        cutoff: float = 6.0,
        time_range: tuple[float, float] | None = None,
        output_times: np.ndarray | None = None,
    ) -> IDResult:
        """Run the full OpenSim inverse-dynamics pipeline from a coordinate motion.

        Args:
            coordinates: Coordinate trajectory (``.mot`` path or :class:`Storage`).
            external_loads: Optional applied external loads (e.g. ground reactions).
            cutoff: Butterworth low-pass cutoff [Hz]; ``<= 0`` disables filtering.
            time_range: Optional ``(start, end)`` [s] limiting the output frames.
            output_times: Explicit output times [s]; overrides ``time_range``.

        Returns:
            The generalized forces over the output frames.
        """
        if not isinstance(coordinates, Storage):
            coordinates = read_storage(coordinates)
        times = np.asarray(coordinates.times, float)
        col_index = {lab: i for i, lab in enumerate(coordinates.labels)}
        values = np.zeros((len(times), self.ncoord))
        defaults = {c.name: c.default_value for j in self.model.joints for c in j.coordinates}
        is_rot = [mt == "rotational" for mt in self.motion_types]
        for i, name in enumerate(self.coordinate_names):
            if name in col_index:
                values[:, i] = coordinates.data[:, col_index[name]]
            else:
                default = defaults.get(name, 0.0)
                values[:, i] = np.rad2deg(default) if (is_rot[i] and coordinates.in_degrees) else default

        if output_times is None:
            output_times = times if time_range is None else times[(times >= time_range[0]) & (times <= time_range[1])]
        output_times = np.asarray(output_times, float)

        q, qd, qdd = differentiate_coordinates(
            times, values, is_rot, output_times=output_times, cutoff=cutoff, in_degrees=coordinates.in_degrees
        )
        bodies, wrenches = (None, None)
        if external_loads is not None:
            bodies, wrenches = external_loads.sample(output_times)
        tau = self.solve(q, qd, qdd, external_bodies=bodies, external_wrenches=wrenches)
        return IDResult(
            times=output_times,
            coordinate_names=self.coordinate_names,
            generalized_forces=tau,
            motion_types=self.motion_types,
            coordinates=q,
            speeds=qd,
            accelerations=qdd,
        )


def solve_inverse_dynamics(
    model: OsimModel | str | os.PathLike,
    coordinates: Storage | str | os.PathLike,
    external_loads: ExternalLoads | str | os.PathLike | None = None,
    cutoff: float = 6.0,
    time_range: tuple[float, float] | None = None,
    device=None,
) -> IDResult:
    """Run inverse dynamics end to end, matching OpenSim's ``InverseDynamicsTool``.

    Args:
        model: A parsed :class:`OsimModel`, or a path/XML string to parse.
        coordinates: Coordinate trajectory (``.mot`` path or :class:`Storage`).
        external_loads: Optional :class:`ExternalLoads`, or a path to an
            ``ExternalLoads`` setup XML.
        cutoff: Butterworth low-pass cutoff [Hz]; ``<= 0`` disables filtering.
        time_range: Optional ``(start, end)`` [s] limiting the output frames.
        device: Warp device for the kernels (``None`` for the CPU).

    Returns:
        The generalized forces over the output frames.
    """
    if not isinstance(model, OsimModel):
        model = parse_osim(model)
    if external_loads is not None and not isinstance(external_loads, ExternalLoads):
        external_loads = read_external_loads(external_loads)
    solver = InverseDynamics(model, device=device)
    return solver.solve_from_motion(coordinates, external_loads=external_loads, cutoff=cutoff, time_range=time_range)


@dataclass
class FDResult:
    """Result of a forward-dynamics simulation.

    Attributes:
        times: State times [s], shape ``[num_frames]``.
        coordinate_names: Coordinate names in column order.
        coordinates: Coordinate values [m or rad], shape ``[num_frames, num_coordinates]``.
        speeds: Coordinate speeds [m/s or rad/s], same shape.
        motion_types: Motion type per coordinate (``"rotational"`` etc.).
    """

    times: np.ndarray
    coordinate_names: list[str]
    coordinates: np.ndarray
    speeds: np.ndarray
    motion_types: list[str]

    @property
    def column_labels(self) -> list[str]:
        """Return OpenSim-style state labels (``<coord>`` value, ``<coord>_u`` speed)."""
        labels = []
        for name in self.coordinate_names:
            labels.append(name)
            labels.append(f"{name}_u")
        return labels

    def _state_data(self) -> np.ndarray:
        n_frames, nc = self.coordinates.shape
        data = np.empty((n_frames, 2 * nc))
        data[:, 0::2] = self.coordinates
        data[:, 1::2] = self.speeds
        return data

    def to_storage(self) -> Storage:
        """Return the states (value/speed pairs) as a :class:`Storage`."""
        return Storage(
            times=self.times,
            labels=self.column_labels,
            data=self._state_data(),
            in_degrees=False,
            name="Forward Dynamics States",
        )

    def write_sto(self, path: str | os.PathLike) -> None:
        """Write the states to an OpenSim ``.sto`` file."""
        write_storage(
            path,
            self.times,
            self.column_labels,
            self._state_data(),
            name="Forward Dynamics States",
            in_degrees=False,
        )


@dataclass
class FDBatchResult:
    """Result of a batched forward-dynamics simulation over several trajectories.

    Attributes:
        times: Sample times [s], shape ``[num_samples]``.
        coordinate_names: Coordinate names in column order.
        coordinates: Coordinate values [m or rad], shape ``[num_samples, num_trajectories, num_coordinates]``.
        speeds: Coordinate speeds [m/s or rad/s], same shape.
        motion_types: Motion type per coordinate (``"rotational"`` etc.).
    """

    times: np.ndarray
    coordinate_names: list[str]
    coordinates: np.ndarray
    speeds: np.ndarray
    motion_types: list[str]

    def trajectory(self, index: int) -> FDResult:
        """Return trajectory ``index`` as a single-trajectory :class:`FDResult`."""
        return FDResult(
            times=self.times,
            coordinate_names=list(self.coordinate_names),
            coordinates=self.coordinates[:, index, :],
            speeds=self.speeds[:, index, :],
            motion_types=list(self.motion_types),
        )


@dataclass
class _ForwardDynamicsWorkspace:
    """Fixed-shape device buffers for repeated forward-acceleration evaluations."""

    batch: int
    stride: int
    num_external: int
    qbatch: wp.array[_f64]
    poses: wp.array[_mat44d]
    torque: wp.array[_vec3d]
    force: wp.array[_vec3d]
    tau_id: wp.array[_f64]
    mass_matrix: wp.array[_f64]
    rhs: wp.array[_f64]
    external_bodies: wp.array[wp.int32]
    external_wrenches: wp.array[_f64]


class ForwardDynamics:
    r"""Solve forward dynamics with Warp kernels, reusing the inverse-dynamics core.

    The equations of motion are :math:`M(q)\,\ddot q + b(q,\dot q) = \tau`, where
    :math:`b` collects gravity, Coriolis/centrifugal, and (negated) external-load
    generalized forces. Both :math:`M` and :math:`b` are obtained from the
    Newton-Euler inverse dynamics, which is affine in the accelerations
    (:math:`\tau_\mathrm{ID}(q,\dot q,\ddot q)=M(q)\,\ddot q+b(q,\dot q)`):
    the bias is :math:`b=\tau_\mathrm{ID}(q,\dot q,0)` and each mass-matrix
    column is :math:`M_{:,i}=\tau_\mathrm{ID}(q,\dot q,e_i)-b` (the composite
    rigid-body method of Walker and Orin, 1982). All of these inverse-dynamics
    evaluations and the batched Cholesky solve
    :math:`M(q)\ddot q = \tau-b` run in Warp kernels. Repeated simulations
    reuse fixed-shape device workspaces and capture supported CUDA steps.
    Coordinates marked ``locked`` and clamped coordinates with a zero-width
    range are imposed as zero-acceleration constraints. Forward simulation also
    zeros their initial speeds and sets zero-width coordinates to their bound.

    Args:
        model: Parsed model IR.
        device: Warp device for the kernels (``"cpu"``, ``"cuda"``, a
            :class:`warp.context.Device`, or ``None`` for the CPU).
    """

    def __init__(self, model: OsimModel, device=None):
        self.idyn = InverseDynamics(model, device=device)
        self.model = model
        self.fk = self.idyn.fk
        self.device = self.idyn.device
        self.coordinate_names = self.idyn.coordinate_names
        self.motion_types = self.idyn.motion_types
        self.ncoord = self.idyn.ncoord
        coordinates = {coordinate.name: coordinate for joint in model.joints for coordinate in joint.coordinates}

        def is_zero_width_clamp(name: str) -> bool:
            coordinate = coordinates[name]
            return coordinate.clamped and coordinate.range is not None and coordinate.range[0] == coordinate.range[1]

        fixed = np.asarray(
            [coordinates[name].locked or is_zero_width_clamp(name) for name in self.coordinate_names],
            dtype=np.int32,
        )
        fixed_values = np.asarray(
            [coordinates[name].range[0] if is_zero_width_clamp(name) else np.nan for name in self.coordinate_names],
            dtype=float,
        )
        self._fixed_coordinates = fixed
        self._fixed_values = fixed_values
        self._fixed_coordinates_device = wp.array(fixed, dtype=wp.int32, device=self.device)

    def _create_device_workspace(
        self, batch: int, external_bodies: list[str] | None = None
    ) -> _ForwardDynamicsWorkspace:
        """Allocate reusable device buffers for a fixed acceleration batch."""
        nc = self.ncoord
        nb = self.fk.nbody
        stride = 3 + 2 * nc
        columns = batch * (nc + 1)
        external_indices = [] if not external_bodies else [self.idyn._body_index[name] for name in external_bodies]
        num_external = len(external_indices)
        return _ForwardDynamicsWorkspace(
            batch=batch,
            stride=stride,
            num_external=num_external,
            qbatch=wp.empty((columns * stride, nc), dtype=_f64, device=self.device),
            poses=wp.empty((columns * stride, nb), dtype=_mat44d, device=self.device),
            torque=wp.empty((columns, nb), dtype=_vec3d, device=self.device),
            force=wp.empty((columns, nb), dtype=_vec3d, device=self.device),
            tau_id=wp.empty((columns, nc), dtype=_f64, device=self.device),
            mass_matrix=wp.empty((batch, nc, nc), dtype=_f64, device=self.device),
            rhs=wp.empty((batch, nc), dtype=_f64, device=self.device),
            external_bodies=wp.array(external_indices, dtype=wp.int32, device=self.device),
            external_wrenches=wp.empty((columns, num_external, 9), dtype=_f64, device=self.device),
        )

    def _mass_bias_device(
        self,
        coords: wp.array[_f64],
        speeds: wp.array[_f64],
        applied_forces: wp.array[_f64],
        workspace: _ForwardDynamicsWorkspace,
        h: float = 1.0e-4,
        eps: float = 1.0e-6,
        external_wrenches: wp.array[_f64] | None = None,
    ) -> None:
        """Build the mass matrix and applied-minus-bias right-hand side on device."""
        nc = self.ncoord
        nb = self.fk.nbody
        batch = workspace.batch
        stride = workspace.stride
        columns = batch * (nc + 1)
        if workspace.num_external:
            if external_wrenches is None:
                raise ValueError("external_wrenches are required for the configured external bodies")
            wp.launch(
                expand_external_wrench_kernel,
                dim=(columns, workspace.num_external, 9),
                inputs=[external_wrenches, nc + 1, workspace.external_wrenches],
                device=self.device,
            )
        wp.launch(
            build_id_batch_kernel,
            dim=(batch, nc + 1, stride),
            inputs=[coords, speeds, _f64(h), _f64(eps), nc, stride, workspace.qbatch],
            device=self.device,
        )
        wp.launch(
            fk_kernel,
            dim=columns * stride,
            inputs=[
                workspace.qbatch,
                self.fk.njoint,
                nb,
                self.fk.d_joint_parent,
                self.fk.d_joint_child,
                self.fk.d_xpf,
                self.fk.d_xbm_inv,
                self.fk.d_axis_dir,
                self.fk.d_axis_type,
                self.fk.d_axis_coord,
                self.fk.d_axis_p0,
                self.fk.d_axis_p1,
                self.fk.d_koff,
                self.fk.d_kcnt,
                self.fk.d_kx,
                self.fk.d_ky,
                self.fk.d_kb,
                self.fk.d_kc,
                self.fk.d_kd,
                workspace.poses,
            ],
            device=self.device,
        )
        wp.launch(
            bodyforce_kernel,
            dim=(columns, nb),
            inputs=[
                workspace.poses,
                self.idyn._mass,
                self.idyn._rcom,
                self.idyn._inertia,
                _vec3d(*self.idyn.gravity),
                stride,
                workspace.num_external,
                workspace.external_bodies,
                workspace.external_wrenches,
                _f64(h),
            ],
            outputs=[workspace.torque, workspace.force],
            device=self.device,
        )
        wp.launch(
            tau_kernel,
            dim=(columns, nc),
            inputs=[workspace.poses, self.idyn._rcom, workspace.torque, workspace.force, stride, nb, _f64(eps)],
            outputs=[workspace.tau_id],
            device=self.device,
        )
        wp.launch(
            mass_bias_kernel,
            dim=(batch, nc, nc),
            inputs=[workspace.tau_id, applied_forces, nc, workspace.mass_matrix, workspace.rhs],
            device=self.device,
        )

    def _accelerations_device(
        self,
        coords: wp.array[_f64],
        speeds: wp.array[_f64],
        applied_forces: wp.array[_f64],
        out: wp.array[_f64],
        workspace: _ForwardDynamicsWorkspace,
        h: float = 1.0e-4,
        eps: float = 1.0e-6,
        external_wrenches: wp.array[_f64] | None = None,
    ) -> None:
        """Evaluate forward accelerations with device-resident inputs and workspace."""
        self._mass_bias_device(coords, speeds, applied_forces, workspace, h, eps, external_wrenches)
        if np.any(self._fixed_coordinates):
            wp.launch(
                constrain_fixed_coordinates_kernel,
                dim=(workspace.batch, self.ncoord, self.ncoord),
                inputs=[workspace.mass_matrix, workspace.rhs, self._fixed_coordinates_device],
                device=self.device,
            )
        wp.launch(
            spd_solve_kernel,
            dim=workspace.batch,
            inputs=[workspace.mass_matrix, workspace.rhs, self.ncoord, out],
            device=self.device,
        )

    def _id_columns(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        external_bodies: list[str] | None,
        external_wrenches: np.ndarray | None,
        h: float,
        eps: float,
    ) -> np.ndarray:
        """Return ``tau_ID`` for accelerations ``{0, e_0, ..., e_{nc-1}}`` per frame.

        The result has shape ``[num_frames, num_coordinates + 1, num_coordinates]``:
        row 0 is the bias and row ``i+1`` is the response to a unit acceleration of
        coordinate ``i``.
        """
        n_frames, nc = q.shape
        accel_set = np.vstack([np.zeros(nc), np.eye(nc)])  # (nc + 1, nc)
        rows = nc + 1
        big_q = np.repeat(q, rows, axis=0)
        big_qd = np.repeat(qd, rows, axis=0)
        big_qdd = np.tile(accel_set, (n_frames, 1))
        big_w = None
        if external_bodies:
            big_w = np.repeat(external_wrenches, rows, axis=0)
        tau_id = self.idyn.solve(
            big_q, big_qd, big_qdd, external_bodies=external_bodies, external_wrenches=big_w, h=h, eps=eps
        )
        return tau_id.reshape(n_frames, rows, nc)

    def mass_matrix(self, q: np.ndarray, h: float = 1.0e-4, eps: float = 1.0e-6) -> np.ndarray:
        """Return the joint-space mass matrix ``M(q)``.

        Args:
            q: Coordinate values, shape ``[num_frames, num_coordinates]`` or ``[num_coordinates]``.
            h: Finite-difference step for the inverse-dynamics stencil [s].
            eps: Finite-difference step for the geometric Jacobian [rad or m].

        Returns:
            Mass matrix, shape ``[num_frames, num_coordinates, num_coordinates]``
            (a single ``[num_coordinates, num_coordinates]`` matrix if ``q`` is 1-D).
        """
        single = np.asarray(q).ndim == 1
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=float)
        q_wp = wp.array(q, dtype=_f64, device=self.device)
        zeros = wp.zeros(q.shape, dtype=_f64, device=self.device)
        workspace = self._create_device_workspace(q.shape[0])
        self._mass_bias_device(q_wp, zeros, zeros, workspace, h, eps)
        mass = workspace.mass_matrix.numpy()
        return mass[0] if single else mass

    def accelerations(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        external_bodies: list[str] | None = None,
        external_wrenches: np.ndarray | None = None,
        h: float = 1.0e-4,
        eps: float = 1.0e-6,
    ) -> np.ndarray:
        r"""Return coordinate accelerations for applied generalized forces.

        Solves :math:`M(q)\,\ddot q = \tau - b(q,\dot q)` frame by frame.

        Args:
            q: Coordinate values, shape ``[num_frames, num_coordinates]``.
            qd: Coordinate speeds, same shape.
            tau: Applied generalized forces (joint moments/forces), same shape.
            external_bodies: Bodies each external wrench acts on (length ``num_ext``).
            external_wrenches: Ground-frame ``[Fx Fy Fz Px Py Pz Tx Ty Tz]`` per
                frame and force, shape ``[num_frames, num_ext, 9]``.
            h: Finite-difference step for the inverse-dynamics stencil [s].
            eps: Finite-difference step for the geometric Jacobian [rad or m].

        Returns:
            Coordinate accelerations, shape ``[num_frames, num_coordinates]``.
        """
        q = np.ascontiguousarray(q, dtype=float).copy()
        qd = np.ascontiguousarray(qd, dtype=float).copy()
        tau = np.ascontiguousarray(tau, dtype=float)
        fixed_values = np.isfinite(self._fixed_values)
        q[:, fixed_values] = self._fixed_values[fixed_values]
        qd[:, self._fixed_coordinates != 0] = 0.0
        q_wp = wp.array(q, dtype=_f64, device=self.device)
        qd_wp = wp.array(qd, dtype=_f64, device=self.device)
        tau_wp = wp.array(tau, dtype=_f64, device=self.device)
        wrench_wp = None
        if external_bodies:
            if external_wrenches is None:
                raise ValueError("external_wrenches are required when external_bodies are provided")
            wrench_wp = wp.array(np.ascontiguousarray(external_wrenches, dtype=float), dtype=_f64, device=self.device)
        out = wp.empty(q.shape, dtype=_f64, device=self.device)
        workspace = self._create_device_workspace(q.shape[0], external_bodies)
        self._accelerations_device(q_wp, qd_wp, tau_wp, out, workspace, h, eps, wrench_wp)
        return out.numpy()

    def simulate(
        self,
        initial_coordinates: np.ndarray,
        initial_speeds: np.ndarray,
        duration: float,
        dt: float,
        start_time: float = 0.0,
        controls=None,
        external_loads: ExternalLoads | None = None,
        contact_forces=None,
        integrator: str = "rk4",
        h: float = 1.0e-4,
        contact_h: float = 1.0e-6,
        eps: float = 1.0e-6,
        use_graph: bool = True,
    ) -> FDResult:
        """Integrate the equations of motion forward in time.

        Args:
            initial_coordinates: Initial coordinate values, shape ``[num_coordinates]``.
            initial_speeds: Initial coordinate speeds, shape ``[num_coordinates]``.
            duration: Length of the simulation [s].
            dt: Fixed integration step [s].
            start_time: Time of the initial state [s].
            controls: Optional ``controls(t, q, qd) -> generalized_forces`` callable
                returning applied joint moments/forces (length ``num_coordinates``);
                ``None`` leaves the model passive.
            external_loads: Optional measured/prescribed :class:`ExternalLoads`
                sampled at each step.
            contact_forces: Optional :class:`newton.opensim.OpenSimContact`-like
                evaluator. Its OpenSim-frame body wrenches are recomputed from the
                current state at every integration stage.
            integrator: ``"rk4"`` (fixed-step Runge-Kutta 4) or ``"semi_implicit"``
                (symplectic Euler).
            h: Finite-difference step for the inverse-dynamics stencil [s].
            contact_h: Coordinate perturbation used by contact point-velocity
                evaluation [rad or m].
            eps: Finite-difference step for the geometric Jacobian [rad or m].
            use_graph: Take the device-resident, CUDA-graph-captured stepper when
                the trajectory is passive (no ``controls``, ``external_loads``,
                or ``contact_forces``) on a CUDA device. Falls back to the host
                loop otherwise.

        Returns:
            The coordinate and speed trajectory over ``[start_time, start_time + duration]``.
        """
        if contact_forces is not None and (not np.isfinite(contact_h) or contact_h <= 0.0):
            raise ValueError("contact_h must be finite and positive")
        if use_graph and controls is None and external_loads is None and contact_forces is None and self.device.is_cuda:
            batch = self.simulate_batch(
                np.asarray(initial_coordinates, dtype=float)[None, :],
                np.asarray(initial_speeds, dtype=float)[None, :],
                duration,
                dt,
                start_time=start_time,
                integrator=integrator,
                h=h,
                eps=eps,
                record_every=1,
                use_graph=True,
            )
            return batch.trajectory(0)

        q = np.asarray(initial_coordinates, dtype=float).copy()
        v = np.asarray(initial_speeds, dtype=float).copy()
        fixed_values = np.isfinite(self._fixed_values)
        q[fixed_values] = self._fixed_values[fixed_values]
        v[self._fixed_coordinates != 0] = 0.0
        nc = self.ncoord
        n_steps = int(round(duration / dt))

        def applied(t, qc, vc):
            return np.zeros(nc) if controls is None else np.asarray(controls(t, qc, vc), dtype=float)

        def accel(t, qc, vc):
            body_groups: list[list[str]] = []
            wrench_groups: list[np.ndarray] = []
            if external_loads is not None:
                ext_bodies, ext_wrenches = external_loads.sample(np.array([t]))
                body_groups.append(list(ext_bodies))
                wrench_groups.append(np.asarray(ext_wrenches, dtype=float))
            if contact_forces is not None:
                contact_bodies, contact_wrenches = contact_forces.body_wrenches(
                    qc[None], vc[None], h=contact_h, frame="opensim"
                )
                body_groups.append(list(contact_bodies))
                wrench_groups.append(np.asarray(contact_wrenches, dtype=float))
            ext_bodies = [body for group in body_groups for body in group] or None
            wrench = np.concatenate(wrench_groups, axis=1) if wrench_groups else None
            return self.accelerations(
                qc[None],
                vc[None],
                applied(t, qc, vc)[None],
                external_bodies=ext_bodies,
                external_wrenches=wrench,
                h=h,
                eps=eps,
            )[0]

        times = np.empty(n_steps + 1)
        coords = np.empty((n_steps + 1, nc))
        speeds = np.empty((n_steps + 1, nc))
        times[0] = start_time
        coords[0] = q
        speeds[0] = v
        for k in range(n_steps):
            t = start_time + k * dt
            if integrator == "semi_implicit":
                a = accel(t, q, v)
                v = v + dt * a
                q = q + dt * v
            elif integrator == "rk4":
                a1 = accel(t, q, v)
                a2 = accel(t + 0.5 * dt, q + 0.5 * dt * v, v + 0.5 * dt * a1)
                a3 = accel(t + 0.5 * dt, q + 0.5 * dt * (v + 0.5 * dt * a1), v + 0.5 * dt * a2)
                a4 = accel(t + dt, q + dt * (v + 0.5 * dt * a2), v + dt * a3)
                q = q + dt * (v + dt / 6.0 * (a1 + a2 + a3))
                v = v + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            else:
                raise ValueError(f"unknown integrator {integrator!r}")
            times[k + 1] = start_time + (k + 1) * dt
            coords[k + 1] = q
            speeds[k + 1] = v
        return FDResult(
            times=times,
            coordinate_names=list(self.coordinate_names),
            coordinates=coords,
            speeds=speeds,
            motion_types=list(self.motion_types),
        )

    def simulate_batch(
        self,
        initial_coordinates: np.ndarray,
        initial_speeds: np.ndarray,
        duration: float,
        dt: float,
        start_time: float = 0.0,
        tau_applied: np.ndarray | None = None,
        integrator: str = "rk4",
        h: float = 1.0e-4,
        eps: float = 1.0e-6,
        record_every: int = 1,
        use_graph: bool = True,
    ) -> FDBatchResult:
        r"""Integrate several trajectories forward in time, in lockstep on the device.

        Every device kernel carries a leading trajectory axis, so the per-step
        Cholesky solve of :math:`M(q)\,\ddot q = 	au - b` runs over all
        trajectories at once and fills the GPU width that a single sequential
        trajectory leaves idle. On a CUDA device the whole step is captured with a
        graph and replayed, so there are no per-step host round-trips.

        Args:
            initial_coordinates: Initial coordinates, shape ``[num_trajectories, num_coordinates]``
                (a single ``[num_coordinates]`` vector is treated as one trajectory).
            initial_speeds: Initial speeds, same shape.
            duration: Length of the simulation [s].
            dt: Fixed integration step [s].
            start_time: Time of the initial state [s].
            tau_applied: Optional constant applied generalized forces, shape
                ``[num_coordinates]`` (shared) or ``[num_trajectories, num_coordinates]``;
                ``None`` leaves the trajectories passive.
            integrator: ``"rk4"`` (fixed-step Runge-Kutta 4) or ``"semi_implicit"``
                (symplectic Euler).
            h: Finite-difference step for the inverse-dynamics stencil [s].
            eps: Finite-difference step for the geometric Jacobian [rad or m].
            record_every: Store the state every ``record_every`` steps (plus the
                initial state); ``0`` keeps only the final state.
            use_graph: Capture and replay the step with a CUDA graph on a CUDA device.

        Returns:
            The sampled coordinate and speed trajectories for every trajectory.
        """
        dev = self.device
        nc = self.ncoord
        q0 = np.ascontiguousarray(np.atleast_2d(np.asarray(initial_coordinates, dtype=float))).copy()
        v0 = np.ascontiguousarray(np.atleast_2d(np.asarray(initial_speeds, dtype=float))).copy()
        fixed_values = np.isfinite(self._fixed_values)
        q0[:, fixed_values] = self._fixed_values[fixed_values]
        v0[:, self._fixed_coordinates != 0] = 0.0
        n_traj = q0.shape[0]
        n_steps = int(round(duration / dt))
        if integrator not in ("rk4", "semi_implicit"):
            raise ValueError(f"unknown integrator {integrator!r}")

        q = wp.array(q0, dtype=_f64, device=dev)
        v = wp.array(v0, dtype=_f64, device=dev)
        qs = wp.empty((n_traj, nc), dtype=_f64, device=dev)
        vs = wp.empty((n_traj, nc), dtype=_f64, device=dev)
        a = [wp.empty((n_traj, nc), dtype=_f64, device=dev) for _ in range(4)]
        a0 = wp.zeros((n_traj, nc), dtype=_f64, device=dev)
        workspace = self._create_device_workspace(n_traj)
        tau_app_np = (
            np.zeros((n_traj, nc))
            if tau_applied is None
            else np.broadcast_to(np.asarray(tau_applied, dtype=float), (n_traj, nc))
        )
        tau_app = wp.array(np.ascontiguousarray(tau_app_np), dtype=_f64, device=dev)

        def accel(qc, vc, a_out):
            self._accelerations_device(qc, vc, tau_app, a_out, workspace, h, eps)

        def step():
            if integrator == "semi_implicit":
                accel(q, v, a[0])
                wp.launch(semi_implicit_update_kernel, dim=(n_traj, nc), inputs=[q, v, a[0], _f64(dt)], device=dev)
                return
            wp.launch(rk4_stage_kernel, dim=(n_traj, nc), inputs=[q, v, a0, _f64(0.0), _f64(0.0), qs, vs], device=dev)
            accel(qs, vs, a[0])
            wp.launch(
                rk4_stage_kernel,
                dim=(n_traj, nc),
                inputs=[q, v, a[0], _f64(0.5 * dt), _f64(0.5 * dt), qs, vs],
                device=dev,
            )
            accel(qs, vs, a[1])
            wp.launch(
                rk4_stage3_kernel,
                dim=(n_traj, nc),
                inputs=[q, v, a[0], a[1], _f64(0.5 * dt), _f64(0.5 * dt), _f64(0.5 * dt), qs, vs],
                device=dev,
            )
            accel(qs, vs, a[2])
            wp.launch(
                rk4_stage3_kernel,
                dim=(n_traj, nc),
                inputs=[q, v, a[1], a[2], _f64(dt), _f64(0.5 * dt), _f64(dt), qs, vs],
                device=dev,
            )
            accel(qs, vs, a[3])
            wp.launch(rk4_update_kernel, dim=(n_traj, nc), inputs=[q, v, a[0], a[1], a[2], a[3], _f64(dt)], device=dev)

        rec = record_every and record_every > 0
        dynamic_record = bool(rec and record_every == 1 and use_graph and dev.is_cuda)
        sample_counter = None
        if rec:
            nrec = 1 + n_steps // record_every
            out_state = wp.empty((nrec, n_traj, 2 * nc), dtype=_f64, device=dev)
            wp.launch(record_state_kernel, dim=(n_traj, nc), inputs=[q, v, 0, out_state, nc], device=dev)
            sample = 1
            if dynamic_record:
                sample_counter = wp.array([1], dtype=wp.int32, device=dev)

        def captured_step():
            step()
            if dynamic_record:
                wp.launch(
                    record_state_dynamic_kernel,
                    dim=(n_traj, nc),
                    inputs=[q, v, sample_counter, out_state, nc],
                    device=dev,
                )
                wp.launch(increment_state_sample_kernel, dim=1, inputs=[sample_counter], device=dev)

        graph = None
        if use_graph and dev.is_cuda:
            captured_step()
            q.assign(q0)
            v.assign(v0)
            if dynamic_record:
                sample_counter.assign(np.array([1], dtype=np.int32))
            with wp.ScopedCapture(device=dev) as cap:
                captured_step()
            graph = cap.graph

        for k in range(n_steps):
            if graph is not None:
                wp.capture_launch(graph)
            else:
                captured_step()
            if rec and not dynamic_record and (k + 1) % record_every == 0:
                wp.launch(record_state_kernel, dim=(n_traj, nc), inputs=[q, v, sample, out_state, nc], device=dev)
                sample += 1

        if rec:
            times = start_time + np.arange(nrec) * (record_every * dt)
            state = out_state.numpy()
            coords = state[:, :, :nc]
            speeds = state[:, :, nc:]
        else:
            times = np.array([start_time + n_steps * dt])
            out_state = wp.empty((1, n_traj, 2 * nc), dtype=_f64, device=dev)
            wp.launch(record_state_kernel, dim=(n_traj, nc), inputs=[q, v, 0, out_state, nc], device=dev)
            state = out_state.numpy()
            coords = state[:, :, :nc]
            speeds = state[:, :, nc:]
        return FDBatchResult(
            times=times,
            coordinate_names=list(self.coordinate_names),
            coordinates=coords,
            speeds=speeds,
            motion_types=list(self.motion_types),
        )


def solve_forward_dynamics(
    model: OsimModel | str | os.PathLike,
    initial_coordinates: np.ndarray,
    initial_speeds: np.ndarray,
    duration: float,
    dt: float,
    start_time: float = 0.0,
    controls=None,
    external_loads: ExternalLoads | str | os.PathLike | None = None,
    integrator: str = "rk4",
    device=None,
) -> FDResult:
    """Integrate forward dynamics end to end.

    Args:
        model: A parsed :class:`OsimModel`, or a path/XML string to parse.
        initial_coordinates: Initial coordinate values, shape ``[num_coordinates]``.
        initial_speeds: Initial coordinate speeds, shape ``[num_coordinates]``.
        duration: Length of the simulation [s].
        dt: Fixed integration step [s].
        start_time: Time of the initial state [s].
        controls: Optional ``controls(t, q, qd) -> generalized_forces`` callable.
        external_loads: Optional :class:`ExternalLoads`, or a path to a setup XML.
        integrator: ``"rk4"`` or ``"semi_implicit"``.
        device: Warp device for the kernels (``None`` for the CPU).

    Returns:
        The coordinate and speed trajectory.
    """
    if not isinstance(model, OsimModel):
        model = parse_osim(model)
    if external_loads is not None and not isinstance(external_loads, ExternalLoads):
        external_loads = read_external_loads(external_loads)
    solver = ForwardDynamics(model, device=device)
    return solver.simulate(
        initial_coordinates,
        initial_speeds,
        duration,
        dt,
        start_time=start_time,
        controls=controls,
        external_loads=external_loads,
        integrator=integrator,
    )
