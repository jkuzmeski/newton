# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve the virtual leg's equilibrium trajectory and impedance profile.

The legacy controller reads its rest length from twice-integrated capture-trial force and fades
hand-tuned gains on a clock. Both choices bake the capture shoe into the command. Here the
equilibrium trajectory and the impedance profile are decision variables, and the score is stated at
the task level by :mod:`projects.impedance_instron.objective`: feasibility first, then the measured
stance task inside stated tolerances, then the leg work proxy. Measured ground reaction force never
enters the score, so a shoe change is answered by a different command instead of being cancelled by
replayed force.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import newton

from .cmaes import CMAES
from .control import LegCommand
from .example import Example, create_parser
from .objective import Objective, Tolerances

TRACE_SHOE_FZ = 0
TRACE_ANKLE_Z = 2
TRACE_UPPER_VZ = 4
TRACE_LEG_FORCE = 5
TRACE_SOURCE_POWER = 6
TRACE_DAMPER_POWER = 7
TRACE_COMPRESSION = 11
TRACE_SATURATED = 12
TRACE_ANKLE_VZ = 15
TRACE_SATURATION_EXCESS = 16
TRACE_ANKLE_VX = 21
TRACE_UPPER_VX = 22


@dataclass
class Rollout:
    """Task-level outcome of one forward simulation."""

    completed: bool
    contact: bool
    contact_start_s: float
    contact_end_s: float
    contact_duration_s: float
    delta_vx_m_s: float
    delta_vz_m_s: float
    effort: float
    actuator_work_j: float
    peak_leg_force_n: float
    peak_shoe_force_n: float
    peak_compression_m: float
    min_last_height_m: float
    residual_load_n: float
    saturation_excess_n: float
    saturated: bool
    momentum_vx_m_s: list[float]
    momentum_vz_m_s: list[float]
    damper_dissipation_j: float
    # Net actuator work hides how the leg spent it, but the two halves are charged at different
    # muscle efficiencies. Defaulted so rollouts stored before the split still load.
    positive_work_j: float = float("nan")
    negative_work_j: float = float("nan")


# Fractions of contact at which the momentum history is compared. Endpoint-only matching leaves
# the force free to arrive at any time inside stance, which is exactly how the first solved command
# came to peak about 28 points of stance late.
CHECKPOINTS = np.arange(0.1, 0.91, 0.1)


@dataclass
class Target:
    """Measured task the command must reproduce, taken from stance impulse only."""

    delta_vx_m_s: float
    delta_vz_m_s: float
    duration_s: float
    momentum_vx_m_s: list[float]
    momentum_vz_m_s: list[float]


def measured_target(profile: dict, mass: float, gravity: float) -> Target:
    """Derive the stance velocity change directly from the measured impulse.

    Absolute COM velocity depends on an assumed initial condition, but the change over stance is
    the measured impulse divided by mass. Only that change is used as the task target.
    """
    time_s = np.asarray(profile["time_s"], dtype=float)
    start, end = profile["provenance"]["running"]["selected_stance_source_s"]
    origin = float(profile["source_time_s"][0])
    window = (time_s >= start - origin) & (time_s <= end - origin)
    if window.sum() < 2:
        raise ValueError("The profile does not contain the selected stance window")
    span, duration = time_s[window], float(time_s[window][-1] - time_s[window][0])
    fx = np.asarray(profile["total_measured_fx_n"], dtype=float)[window]
    fz = np.asarray(profile["total_measured_fz_n"], dtype=float)[window]
    phase = (span - span[0]) / duration
    running_vx = np.concatenate([[0.0], np.cumsum(0.5 * (fx[1:] + fx[:-1]) * np.diff(span))]) / mass
    running_vz = np.concatenate([[0.0], np.cumsum(0.5 * (fz[1:] + fz[:-1]) * np.diff(span))]) / mass
    running_vz -= gravity * (span - span[0])
    return Target(
        delta_vx_m_s=float(np.trapezoid(fx, span) / mass),
        delta_vz_m_s=float(np.trapezoid(fz, span) / mass - gravity * duration),
        duration_s=duration,
        momentum_vx_m_s=np.interp(CHECKPOINTS, phase, running_vx).tolist(),
        momentum_vz_m_s=np.interp(CHECKPOINTS, phase, running_vz).tolist(),
    )


def simulate(args, parameters: np.ndarray | None = None) -> tuple[Rollout, Example]:
    """Run one headless stance and reduce it to task-level outcomes."""
    args.control_vector = None if parameters is None else np.asarray(parameters, dtype=float)
    viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
    example = Example(viewer, args)
    while example.index < example.sample_count:
        example.step()
    trace = example.trace_device.numpy()[: example.index]
    times = example.times[: example.index]
    share = example.foot_mass / example.mass
    com_vx = share * trace[:, TRACE_ANKLE_VX] + (1.0 - share) * trace[:, TRACE_UPPER_VX]
    com_vz = share * trace[:, TRACE_ANKLE_VZ] + (1.0 - share) * trace[:, TRACE_UPPER_VZ]
    weight = example.mass * example.gravity
    loaded = trace[:, TRACE_SHOE_FZ] > 0.02 * weight
    finite = bool(np.all(np.isfinite(trace)))
    rollout = Rollout(
        completed=example.index == example.sample_count and finite,
        contact=bool(loaded.any()),
        contact_start_s=float(times[loaded][0]) if loaded.any() else float("nan"),
        contact_end_s=float(times[loaded][-1]) if loaded.any() else float("nan"),
        contact_duration_s=float(times[loaded][-1] - times[loaded][0]) if loaded.sum() > 1 else 0.0,
        delta_vx_m_s=float(com_vx[loaded][-1] - com_vx[loaded][0]) if loaded.sum() > 1 else 0.0,
        delta_vz_m_s=float(com_vz[loaded][-1] - com_vz[loaded][0]) if loaded.sum() > 1 else 0.0,
        effort=float(np.mean((trace[:, TRACE_LEG_FORCE] / weight) ** 2)) if finite else float("inf"),
        actuator_work_j=float(np.trapezoid(trace[:, TRACE_SOURCE_POWER], times)) if finite else float("nan"),
        peak_leg_force_n=float(np.abs(trace[:, TRACE_LEG_FORCE]).max()) if finite else float("nan"),
        peak_shoe_force_n=float(trace[:, TRACE_SHOE_FZ].max()) if finite else float("nan"),
        peak_compression_m=float(trace[:, TRACE_COMPRESSION].max()) if finite else float("nan"),
        min_last_height_m=float(np.min(trace[:, TRACE_ANKLE_Z] + example.minimum_last_offsets[: example.index]))
        if finite
        else float("nan"),
        residual_load_n=float(trace[-1, TRACE_SHOE_FZ]) if finite else float("nan"),
        saturation_excess_n=float(np.mean(trace[:, TRACE_SATURATION_EXCESS])) if finite else float("inf"),
        saturated=bool(np.any(trace[:, TRACE_SATURATED] != 0.0)) if finite else True,
        momentum_vx_m_s=_momentum(times, com_vx, loaded),
        momentum_vz_m_s=_momentum(times, com_vz, loaded),
        damper_dissipation_j=-float(np.trapezoid(trace[:, TRACE_DAMPER_POWER], times)) if finite else float("inf"),
        positive_work_j=float(np.trapezoid(np.clip(trace[:, TRACE_SOURCE_POWER], 0.0, None), times))
        if finite
        else float("nan"),
        negative_work_j=float(np.trapezoid(np.clip(trace[:, TRACE_SOURCE_POWER], None, 0.0), times))
        if finite
        else float("nan"),
    )
    return rollout, example


def _momentum(times: np.ndarray, velocity: np.ndarray, loaded: np.ndarray) -> list[float]:
    """Sample the velocity change through contact at fixed fractions of the contact interval."""
    if loaded.sum() < 2:
        return [0.0] * len(CHECKPOINTS)
    span, values = times[loaded], velocity[loaded]
    phase = (span - span[0]) / max(span[-1] - span[0], 1.0e-9)
    return np.interp(CHECKPOINTS, phase, values - values[0]).tolist()


def create_optimizer_parser():
    """Extend the example parser with search settings."""
    parser = create_parser()
    parser.set_defaults(control="equilibrium", viewer="null", num_frames=120)
    parser.add_argument("--max-evaluations", type=int, default=600, help="Rollout budget for the outer search.")
    parser.add_argument("--population", type=int, default=None, help="CMA-ES population; default follows dimension.")
    parser.add_argument("--sigma", type=float, default=0.2, help="Initial CMA-ES step in normalized command units.")
    parser.add_argument("--seed", type=int, default=0, help="Search seed.")
    parser.add_argument(
        "--tolerance-duration-ms",
        type=float,
        default=1.0e3 * Tolerances.duration_s,
        help="Stance duration deadband half-width [ms]; inside it the duration costs nothing.",
    )
    parser.add_argument(
        "--tolerance-impulse-pct",
        type=float,
        default=1.0e2 * Tolerances.impulse_fraction,
        help="Vertical impulse deadband half-width [%%] of the measured impulse.",
    )
    parser.add_argument(
        "--tolerance-momentum",
        type=float,
        default=Tolerances.momentum_m_s,
        help="Momentum-history deadband [m/s RMS] across the stance checkpoints.",
    )
    parser.add_argument(
        "--positive-efficiency",
        type=float,
        default=0.25,
        help="Efficiency charged for positive leg work in the objective proxy.",
    )
    parser.add_argument(
        "--negative-efficiency",
        type=float,
        default=1.20,
        help="Efficiency charged for negative leg work in the objective proxy.",
    )
    parser.add_argument(
        "--command-output",
        type=Path,
        default=Path("outputs/impedance_instron/command.json"),
        help="Where to write the solved command vector.",
    )
    return parser


def main():
    """Search the equilibrium trajectory and impedance profile against the measured task."""
    args = create_optimizer_parser().parse_args()
    args.control = "equilibrium"
    tolerances = Tolerances(
        duration_s=1.0e-3 * args.tolerance_duration_ms,
        impulse_fraction=1.0e-2 * args.tolerance_impulse_pct,
        momentum_m_s=args.tolerance_momentum,
    )

    seed_rollout, seed_example = simulate(args, None)
    target = measured_target(seed_example.profile, seed_example.mass, seed_example.gravity)
    command = LegCommand(
        seed_example.times,
        length_knots=args.length_knots,
        stiffness_knots=args.stiffness_knots,
        damping_knots=args.damping_knots,
        mass_kg=seed_example.com_mass,
    )
    start = np.asarray(seed_example.command_parameters, dtype=float)
    lower, upper = command.bounds()
    score = Objective(
        target,
        tolerances,
        positive_efficiency=args.positive_efficiency,
        negative_efficiency=args.negative_efficiency,
        body_weight_n=seed_example.mass * seed_example.gravity,
    )
    seed_verdict = score.evaluate(seed_rollout)
    print(f"target dvx={target.delta_vx_m_s:+.3f} dvz={target.delta_vz_m_s:+.3f} m/s over {target.duration_s:.4f} s")
    print(f"seed {seed_verdict.summary()}")

    history: list[dict] = []
    best = {"value": seed_verdict.value, "x": start.copy(), "rollout": asdict(seed_rollout), "verdict": seed_verdict}
    began = time.perf_counter()

    span = upper - lower

    def denormalize(z: np.ndarray) -> np.ndarray:
        """Map a unit-box search point onto the real command vector."""
        return lower + np.asarray(z, dtype=float) * span

    def objective(z: np.ndarray) -> float:
        x = denormalize(z)
        rollout, _ = simulate(args, x)
        verdict = score.evaluate(rollout)
        history.append({"value": verdict.value, "verdict": asdict(verdict), "rollout": asdict(rollout)})
        if verdict.value < best["value"]:
            best.update({"value": verdict.value, "x": x.copy(), "rollout": asdict(rollout), "verdict": verdict})
        return verdict.value

    def report(generation: int, _x: np.ndarray, value: float) -> None:
        rate = (time.perf_counter() - began) / max(len(history), 1)
        print(f"gen {generation:3d}  best={value:.5f}  evals={len(history)}  {rate:.1f} s/rollout", flush=True)

    # The knots mix metres, log-stiffness, and a dimensionless ratio, so a single CMA-ES step
    # size is only meaningful after normalizing every coordinate onto the unit box.
    search = CMAES(
        (start - lower) / span,
        args.sigma,
        (np.zeros_like(lower), np.ones_like(upper)),
        args.population,
        args.seed,
    )
    checkpoint = args.command_output.with_suffix(".checkpoint.json")
    while len(history) < args.max_evaluations:
        candidates = search.ask()
        values = np.array([objective(c) for c in candidates])
        search.tell(candidates, values)
        report(search.generations, denormalize(search.best[0]), search.best[1])
        # Long searches must survive interruption; the optimizer state round trips exactly.
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"search": search.state(), "best": best["x"].tolist()}))

    document = {
        "parameters": best["x"].tolist(),
        "cost": best["value"],
        "seed_cost": seed_verdict.value,
        "verdict": asdict(best["verdict"]),
        "seed_verdict": asdict(seed_verdict),
        "target": asdict(target),
        "objective": score.describe(),
        "rollout": best["rollout"],
        "evaluations": len(history),
        "knots": {
            "length": args.length_knots,
            "stiffness": args.stiffness_knots,
            "damping": args.damping_knots,
        },
        "damping_effective_mass_kg": seed_example.com_mass,
        "cost_definition": (
            "lexicographic: rig feasibility, then the measured stance task inside stated tolerances, "
            "then the muscle-efficiency leg work proxy; measured force histories are never tracked"
        ),
        "identified": False,
    }
    args.command_output.parent.mkdir(parents=True, exist_ok=True)
    args.command_output.write_text(json.dumps(document, indent=2))
    print(f"best {best['verdict'].summary()}; wrote {args.command_output}")


if __name__ == "__main__":
    main()
