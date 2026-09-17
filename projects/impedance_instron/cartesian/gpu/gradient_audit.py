# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit local controller/leg derivatives, not a full differentiable shoe rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from ..trajectory import basis
from .mechanics import Params, Vec5, ankle, dynamics, make_params, solve
from .profile_search import _load_bundle
from .provenance import source_snapshot

wp.set_module_options({"enable_backward": True, "fuse_fp": False})


@wp.kernel
def _step(
    p: Params,
    dt: wp.float64,
    gravity: wp.float64,
    stiffness: wp.vec4d,
    damping: wp.vec4d,
    q_in: wp.array[Vec5],
    v_in: wp.array[Vec5],
    coeff: wp.array2d[wp.float64],
    basis: wp.array[wp.float64],
    wrench_in: wp.array[wp.vec3d],
    q_out: wp.array[Vec5],
    v_out: wp.array[Vec5],
):
    """Probe the shared leg mechanics with an independent, fixed contact wrench."""
    q = q_in[0]
    v = v_in[0]
    eq = wp.vec4d(wp.float64(0.0))
    for row in range(12):
        for c in range(4):
            eq[c] += basis[row] * coeff[row, c]
    control = wp.vec4d(
        stiffness[0] * (eq[0] - q[0]) - damping[0] * v[0],
        stiffness[1] * (eq[1] - q[1]) - damping[1] * v[1],
        stiffness[2] * (eq[2] - q[3]) - damping[2] * v[3],
        stiffness[3] * (eq[3] - q[4]) - damping[3] * v[4],
    )
    wrench = wrench_in[0]
    _position, jx, jz = ankle(q, p)
    load = Vec5(control[0], control[1], wp.float64(0.0), control[2], control[3])
    for j in range(5):
        external = jx[j] * wrench[0] + jz[j] * wrench[1]
        if j >= 2:
            external = external + wrench[2]
        load[j] += external
    mass, bias = dynamics(q, v, p, gravity)
    acceleration = solve(mass, load - bias)
    nv = v + dt * acceleration
    q_out[0] = q + dt * nv
    v_out[0] = nv


@wp.kernel
def _objective(q: wp.array[Vec5], v: wp.array[Vec5], loss: wp.array[wp.float64]):
    """Seed both position and velocity outputs with a diagnostic linear scalar."""
    loss[0] = wp.dot(
        q[0], Vec5(wp.float64(0.3), wp.float64(-0.2), wp.float64(0.1), wp.float64(-0.4), wp.float64(0.2))
    ) + wp.dot(v[0], Vec5(wp.float64(0.2), wp.float64(-0.3), wp.float64(0.4), wp.float64(0.1), wp.float64(-0.2)))


def audit_step(params, dt, gravity, stiffness, damping, basis_values, values, *, device, seed=31):
    """Compare a leg-step VJP with central differences in four input spaces.

    The scalar is a diagnostic weighted sum of next position and velocity, not
    the measured fitting objective. Contact wrench is an independent input;
    this check does not differentiate shoe histories or an entire stance.
    """
    device = wp.get_device(device)
    arrays = {
        name: wp.array(
            value,
            dtype=Vec5 if name in ("q", "v") else wp.vec3d if name == "wrench" else wp.float64,
            device=device,
            requires_grad=True,
        )
        for name, value in values.items()
    }
    basis_wp = wp.array(basis_values, dtype=wp.float64, device=device)
    q_out = wp.zeros(1, dtype=Vec5, device=device, requires_grad=True)
    v_out = wp.zeros_like(q_out, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)

    def forward():
        wp.launch(
            _step,
            dim=1,
            inputs=[
                params,
                dt,
                gravity,
                wp.vec4d(*stiffness),
                wp.vec4d(*damping),
                arrays["q"],
                arrays["v"],
                arrays["coeff"],
                basis_wp,
                arrays["wrench"],
                q_out,
                v_out,
            ],
            device=device,
        )
        wp.launch(_objective, dim=1, inputs=[q_out, v_out, loss], device=device)

    with wp.Tape() as tape:
        forward()
    tape.backward(loss)
    gradients = {name: array.grad.numpy() for name, array in arrays.items()}
    rng = np.random.default_rng(seed)
    checks = {}
    passed = True
    for name, array in arrays.items():
        direction = rng.normal(size=values[name].shape)
        direction /= np.linalg.norm(direction)
        predicted = float(np.sum(gradients[name] * direction))
        passed = passed and bool(np.isfinite(gradients[name]).all())
        differences = []
        for epsilon in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
            array.assign(values[name] + epsilon * direction)
            forward()
            plus = float(loss.numpy()[0])
            array.assign(values[name] - epsilon * direction)
            forward()
            minus = float(loss.numpy()[0])
            fd = (plus - minus) / (2.0 * epsilon)
            absolute_error = abs(fd - predicted)
            relative_error = absolute_error / max(abs(fd), abs(predicted), 1.0e-10)
            differences.append(
                {
                    "epsilon": epsilon,
                    "finite_difference": fd,
                    "autodiff": predicted,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )
        array.assign(values[name])
        checks[name] = differences
        # Require agreement at two adjacent scales, not a single hand-picked epsilon.
        passed = passed and all(
            np.isfinite(item["absolute_error"]) and item["absolute_error"] <= 1.0e-8 + 1.0e-5 * abs(predicted)
            for item in differences[1:3]
        )
    return {"passed": bool(passed), "checks": checks, "device": str(device)}


def main(argv: list[str] | None = None) -> None:
    """Write an explicitly local gradient audit from a saved baseline pose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/impedance_instron/baseline12"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    reference, profile, initial, summary, manifest = _load_bundle(args.baseline)
    if "trace.npz" not in manifest["files_sha256"]:
        raise ValueError("The baseline manifest must cover the sampled trace")
    with np.load(args.baseline / "trace.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    sample = len(trace["time_s"]) // 2
    time_s = float(trace["time_s"][sample])
    values = {
        "q": trace["state"][sample : sample + 1],
        "v": trace["velocity"][sample : sample + 1],
        "coeff": initial.coefficients,
        "wrench": np.array([[*trace["grf_n"][sample], trace["ankle_contact_moment_nm"][sample]]]),
    }
    report = audit_step(
        make_params(reference, profile),
        summary["simulation_config"]["dt_s"],
        summary["simulation_config"]["gravity_m_s2"],
        [*profile["hip_stiffness_n_m"], *profile["joint_stiffness_nm_rad"]],
        [*profile["hip_damping_ns_m"], *profile["joint_damping_nms_rad"]],
        basis(np.array([time_s]), initial.duration_s, 12)[0],
        values,
        device=args.device,
    )
    report.update(
        schema="cartesian_local_gradient_audit_1",
        scope="One fixed-wrench leg step only; not full-contact BPTT, fitting, or numerical qualification.",
        sampled_time_s=time_s,
        warp_version=wp.__version__,
        component_overwrites=wp.config.enable_vector_component_overwrites,
        source_sha256=source_snapshot(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve failed diagnostics without emitting nonstandard JSON NaNs.
    from .benchmark import _plain  # noqa: PLC0415

    args.output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output), "scope": report["scope"]}))
    if not report["passed"]:
        raise RuntimeError("The local gradient audit failed")


if __name__ == "__main__":
    main()
