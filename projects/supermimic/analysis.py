# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inverse dynamics, joint quasi-stiffness and stride metrics of the SuperMimic runner.

Joint moments are generalized forces on the ten actuated coordinates of
:data:`~projects.supermimic.body.JOINT_NAMES`: the moment the parent body
applies to the child body, projected on the joint axis. They have the sign of
the joint coordinate, as the simulated joint torque, so measured and
simulated runs go through the same :func:`joint_summary` and
:func:`stride_metrics`. Pass simulated tensors as NumPy arrays of one world.

Joint quasi-stiffness follows Hirschman, Montgomery and Grabowski (2024), The
contribution of lower-limb joint quasi-stiffness to theoretical leg stiffness
during level, uphill and downhill running at different speeds, R. Soc. Open
Sci. 11:231133: the change in joint moment over the change in joint angle in
two stance phases. The loading phase runs from the instant the moment first
reaches 20 % of its stance peak to the maximum joint displacement against
that peak moment (peak knee flexion, peak dorsiflexion); the unloading phase
runs from there to the last instant the moment is at 20 % of its peak.
Hirschman et al. report the knee and ankle; the same phases are computed for
every joint and are empty where the joint does not yield, such as the hip in
flexion, which extends through stance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .body import JOINT_NAMES, BodyParams, mirror_axis
from .reference import (
    CONTACT_THRESHOLD_N,
    Subject,
    body_params,
    forward_kinematics,
    load_shoe,
    load_subject,
    trial_reference,
)
from .sim import Shoe, shod_foot

__all__ = [
    "GRAVITY",
    "inverse_dynamics",
    "joint_summary",
    "main",
    "quasi_stiffness",
    "stance_intervals",
    "stride_metrics",
    "subject_summary",
]

GRAVITY = 9.81
"""Gravitational acceleration [m/s^2]."""

QUASI_STIFFNESS_MOMENT_FRACTION = 0.2
"""Fraction of the stance peak moment that bounds the quasi-stiffness phases (Hirschman et al. 2024)."""

LOADING_RATE_FRACTIONS = (0.2, 0.8)
"""Fractions of the impact peak between which the vertical loading rate is averaged (Milner et al. 2006)."""


def _rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] about a unit ``axis`` by ``angle`` [rad]."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = np.asarray(angle)[..., None, None]
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _angular_velocity(rotation: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocities [frames, 3] of rotations [frames, 3, 3] by central differences."""
    w = np.gradient(rotation, dt, axis=0) @ np.swapaxes(rotation, -1, -2)
    return 0.5 * np.stack((w[:, 2, 1] - w[:, 1, 2], w[:, 0, 2] - w[:, 2, 0], w[:, 1, 0] - w[:, 0, 1]), -1)


def inverse_dynamics(
    joint_q: np.ndarray,
    dt: float,
    params: BodyParams,
    grf: np.ndarray,
    cop: np.ndarray,
    free_moment: np.ndarray,
    shoe: Shoe | None = None,
) -> np.ndarray:
    """Net joint moments by Newton-Euler recursion from each foot to the pelvis.

    Segment velocities and accelerations are central differences of the
    coordinates, so ``joint_q`` should be smooth (low-pass filtered).

    Args:
        joint_q: Generalized coordinates [frames, 17]: root position [m], root quaternion (xyzw), joint angles [rad].
        dt: Frame interval [s].
        params: Body geometry and inertia.
        grf: Ground reaction force on each foot (right, left) [frames, 2, 3] [N], zero off the ground.
        cop: Point of application of ``grf`` [frames, 2, 3] [m], finite in every frame.
        free_moment: Moment on each foot about ``cop`` [frames, 2, 3] [N m].
        shoe: Shoe pair on the feet, its mass added to them as in :func:`~projects.supermimic.sim.shod_foot`.

    Returns:
        Joint moments [frames, 10] [N m] in :data:`~projects.supermimic.body.JOINT_NAMES` order.
    """
    frames = forward_kinematics(joint_q, params)
    angles = np.asarray(joint_q)[:, 7:]
    gravity = np.array((0.0, 0.0, -GRAVITY))
    moments = np.empty((len(angles), len(JOINT_NAMES)))
    for i, side in enumerate(("r", "l")):
        mirror = np.array((1.0, -1.0, 1.0) if side == "l" else (1.0, 1.0, 1.0))
        axes = [np.asarray(a, dtype=float) for a in (*params.hip_axes, params.knee_axis, params.ankle_axis)]
        if side == "l":
            axes = [mirror_axis(a) for a in axes]
        force, point, moment = grf[:, i], cop[:, i], free_moment[:, i]
        proximal = {}
        segments = [
            (s.mass, mirror * np.asarray(s.com), np.diag(s.inertia)) for s in (params.foot, params.shank, params.thigh)
        ]
        if shoe is not None:
            segments[0] = shod_foot(params, shoe, ("right", "left")[i])
        for name, (mass, local_com, local_inertia) in zip(("foot", "shank", "thigh"), segments, strict=True):
            rotation, origin = frames[f"{name}_{side}"]
            com = origin + rotation @ local_com
            omega = _angular_velocity(rotation, dt)
            inertia = rotation @ local_inertia @ np.swapaxes(rotation, -1, -2)
            momentum_rate = np.einsum("fij,fj->fi", inertia, np.gradient(omega, dt, axis=0)) + np.cross(
                omega, np.einsum("fij,fj->fi", inertia, omega)
            )
            # Wrench applied by the parent at the segment origin (its proximal joint center).
            joint_force = mass * (np.gradient(np.gradient(com, dt, axis=0), dt, axis=0) - gravity) - force
            joint_moment = momentum_rate - np.cross(point - com, force) - moment - np.cross(origin - com, joint_force)
            proximal[name] = (rotation, joint_moment)
            force, point, moment = -joint_force, origin, -joint_moment
        pelvis = frames["pelvis"][0]
        hip_axes = (
            pelvis @ axes[0],
            pelvis @ _rotation(axes[0], angles[:, 5 * i]) @ axes[1],
            proximal["thigh"][0] @ axes[2],
        )
        for k, axis in enumerate(hip_axes):
            moments[:, 5 * i + k] = np.einsum("fi,fi->f", proximal["thigh"][1], axis)
        moments[:, 5 * i + 3] = np.einsum("fi,fi->f", proximal["shank"][1], proximal["shank"][0] @ axes[3])
        moments[:, 5 * i + 4] = np.einsum("fi,fi->f", proximal["foot"][1], proximal["foot"][0] @ axes[4])
    return moments


def stance_intervals(grf_z: np.ndarray) -> np.ndarray:
    """Complete stance phases of one foot, where its vertical force exceeds the contact threshold.

    Args:
        grf_z: Vertical ground reaction force [frames] [N].

    Returns:
        Touchdown (inclusive) and toe-off (exclusive) frames [stances, 2]; stances cut by the record ends are dropped.
    """
    edges = np.flatnonzero(np.diff(np.concatenate(([0], (grf_z > CONTACT_THRESHOLD_N).astype(np.int8), [0]))))
    stances = edges.reshape(-1, 2)
    return stances[(stances[:, 0] > 0) & (stances[:, 1] < len(grf_z))]


def quasi_stiffness(angle: np.ndarray, moment: np.ndarray, stances: np.ndarray) -> np.ndarray:
    """Loading and unloading quasi-stiffness of one joint in each stance (Hirschman et al. 2024).

    The value is ``-dM / dq`` over the phase, positive for a spring that
    resists displacement, so it compares with a joint stiffness ``K``.

    Args:
        angle: Joint angle [frames] [rad].
        moment: Joint moment [frames] [N m].
        stances: Stance frames from :func:`stance_intervals` [stances, 2].

    Returns:
        Quasi-stiffness [stances, 2] [N m/rad], loading then unloading; NaN where a phase is empty.
    """
    result = np.full((len(stances), 2), np.nan)
    for n, (a, b) in enumerate(stances):
        m, q = moment[a:b], angle[a:b]
        sign = np.sign(m[np.argmax(np.abs(m))])
        loaded = np.flatnonzero(sign * m >= QUASI_STIFFNESS_MOMENT_FRACTION * np.max(np.abs(m)))
        start, end = loaded[0], loaded[-1]
        peak = start + np.argmax(-sign * q[start : end + 1])
        for phase, (i, j) in enumerate(((start, peak), (peak, end))):
            if j > i:
                result[n, phase] = -(m[j] - m[i]) / (q[j] - q[i])
    return result


def _crossing(values: np.ndarray, level: float) -> float:
    """Fractional index where rising ``values`` first reach ``level``."""
    j = np.argmax(values >= level)
    return j - 1 + (level - values[j - 1]) / (values[j] - values[j - 1])


def stride_metrics(grf_z: np.ndarray, dt: float, body_weight: float) -> dict:
    """Contact time, flight time, cadence, peak vertical force and vertical loading rate.

    The loading rate is the average slope of the vertical force between 20 %
    and 80 % of the impact peak (Milner et al. 2006, Med. Sci. Sports Exerc.
    38:323-328), the first local maximum of the vertical force in stance; when
    there is no impact transient this is the active peak.

    Args:
        grf_z: Vertical ground reaction force per foot (right, left) [frames, 2] [N].
        dt: Frame interval [s].
        body_weight: Body weight [N].

    Returns:
        Per-stance means over both feet: ``contact_time_s``, ``flight_time_s``,
        ``cadence_steps_per_min``, ``peak_grf_bw`` and ``loading_rate_bw_s``, and ``stances``.
    """
    stances = sorted((a, b, i) for i in range(2) for a, b in stance_intervals(grf_z[:, i]))
    peaks, rates = [], []
    for a, b, i in stances:
        force = grf_z[a - 1 : b + 1, i] / body_weight
        local = np.flatnonzero((force[1:-1] >= force[:-2]) & (force[1:-1] > force[2:])) + 1
        impact = force[local[0]]
        low, high = (_crossing(force[: local[0] + 1], f * impact) for f in LOADING_RATE_FRACTIONS)
        rates.append((LOADING_RATE_FRACTIONS[1] - LOADING_RATE_FRACTIONS[0]) * impact / ((high - low) * dt))
        peaks.append(force.max())
    starts = np.array([a for a, _, _ in stances])
    ends = np.array([b for _, b, _ in stances])
    return {
        "stances": len(stances),
        "contact_time_s": float(np.mean(ends - starts) * dt),
        "flight_time_s": float(np.mean(starts[1:] - ends[:-1]) * dt),
        "cadence_steps_per_min": float(60.0 / (np.mean(np.diff(starts)) * dt)),
        "peak_grf_bw": float(np.mean(peaks)),
        "loading_rate_bw_s": float(np.mean(rates)),
    }


def joint_summary(angles: np.ndarray, moments: np.ndarray, stances: tuple[np.ndarray, np.ndarray], mass: float) -> dict:
    """Stance moments and quasi-stiffness of every joint, measured or simulated.

    Args:
        angles: Joint angles [frames, 10] [rad].
        moments: Joint moments [frames, 10] [N m], from :func:`inverse_dynamics` or the simulated joint torque.
        stances: Stance frames of the right and left foot, as from :func:`stance_intervals`.
        mass: Body mass [kg].

    Returns:
        Per joint, the ``peak_moment_nm_kg`` (most negative and most positive
        stance moment) and the ``quasi_stiffness_nm_rad_kg`` of the loading and
        unloading phases, each as mean and standard deviation over stances.
    """
    joints = {}
    for side in range(2):
        for k in range(5 * side, 5 * side + 5):
            peaks = np.array([(moments[a:b, k].min(), moments[a:b, k].max()) for a, b in stances[side]]) / mass
            stiffness = quasi_stiffness(angles[:, k], moments[:, k], stances[side]) / mass
            defined = np.isfinite(stiffness).any(0)
            joints[JOINT_NAMES[k]] = {
                "peak_moment_nm_kg": {"mean": peaks.mean(0).tolist(), "std": peaks.std(0).tolist()},
                "quasi_stiffness_nm_rad_kg": {
                    "mean": [float(np.nanmean(s)) if d else None for s, d in zip(stiffness.T, defined, strict=True)],
                    "std": [float(np.nanstd(s)) if d else None for s, d in zip(stiffness.T, defined, strict=True)],
                    "stances": np.isfinite(stiffness).sum(0).tolist(),
                },
            }
    return joints


def subject_summary(subject: Subject) -> dict:
    """Inverse dynamics, joint summary and stride metrics of the trials of ``subject``.

    Joint moments come from the marker-rate reference and plate signals. The
    exported free moment is not used: it has a trial-dependent offset of the
    same sign under both feet (median 13-24 N m in stance) and 23-31 Hz
    oscillation, unrelated to the plate forces, so it is not a physiological
    free moment. Stances with a rejected marker fit are dropped. Stride metrics
    use the analog-rate plate force. The feet wear the subject's shoe, and
    body weight includes it.

    Args:
        subject: The runner.

    Returns:
        Per trial, ``stride`` (:func:`stride_metrics`) and ``joints`` (:func:`joint_summary`).
    """
    shoe = load_shoe(subject.shoe)
    params = body_params(subject)
    mass = params.mass + shoe.pair_mass_kg
    summary = {}
    for trial in subject.trials:
        data = trial_reference(subject, trial, params)
        dt = float(np.mean(np.diff(data["time_s"])))
        grf = data["grf"]
        moments = inverse_dynamics(
            data["joint_q"], dt, params, grf, np.nan_to_num(data["cop"]), np.zeros_like(grf), shoe
        )
        stances = tuple(
            np.array([(a, b) for a, b in stance_intervals(grf[:, side, 2]) if data["valid"][a:b].all()])
            for side in range(2)
        )
        analog_dt = float(np.mean(np.diff(data["analog_time_s"])))
        summary[trial] = {
            "stride": stride_metrics(data["analog_grf"][:, :, 2], analog_dt, mass * GRAVITY),
            "joints": joint_summary(data["joint_q"][:, 7:], moments, stances, mass),
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize a subject's joint moments, quasi-stiffness and strides.")
    parser.add_argument(
        "--subject", type=Path, required=True, help="subject spec JSON, e.g. projects/supermimic/subjects/F01.json"
    )
    parser.add_argument("--output", type=Path, help="JSON file to write")
    args = parser.parse_args(argv)
    text = json.dumps(subject_summary(load_subject(args.subject)), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
