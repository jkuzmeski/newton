# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference integration of the bilateral runner with CPU shoes.

Mirrors one world of :mod:`.gpu` in neural-impedance mode, step for step: torque
response and slew, both shoe carriers, the bilateral dynamics, semi-implicit
Euler, foot-load filters, and the shared oscillator. It is the numerical
reference for the CUDA environment, not a training path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..cartesian.shoe import Shoe
from ..hogan.runner import Bounds
from .env import EnvConfig
from .mechanics import FOOT_BODY, SIDES, BilateralChain


@dataclass
class ReferenceState:
    q: np.ndarray
    v: np.ndarray
    phase_rad: float = 0.0
    load_bw: np.ndarray = field(default_factory=lambda: np.zeros(2))
    torque_nm: np.ndarray = field(default_factory=lambda: np.zeros(6))


def step(
    chain: BilateralChain,
    shoes: dict[str, Shoe],
    state: ReferenceState,
    impedance: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: EnvConfig,
    bounds: Bounds,
    speed_m_s: float,
) -> dict:
    """Advance one substep in place and return the foot wrenches [N, N, N m] per side."""
    dt = config.dt_s
    equilibrium, stiffness, damping = impedance
    cap = np.tile(bounds.torque_max_nm, 2)
    rate = np.tile(bounds.torque_rate_max_nm_s, 2) * dt
    q, v = state.q, state.v
    spring = stiffness * (equilibrium - q[3:])
    command = spring if config.immediate_damping else spring - damping * v[3:]
    immediate = damping if config.immediate_damping else np.zeros(6)
    response = -math.expm1(-dt / config.response_time_s)
    change = np.clip(response * (np.clip(command, -cap, cap) - state.torque_nm), -rate, rate)
    state.torque_nm = np.clip(state.torque_nm + change, -cap, cap)
    load = np.clip(state.torque_nm - immediate * v[3:], -cap, cap)
    force = np.zeros(9)
    force[3:] = load
    wrenches = {}
    for side in SIDES:
        body = FOOT_BODY[side]
        ankle, jacobian = chain.ankle(q, side)
        angular = chain.angular_jacobian(body)
        wrench, _ = shoes[side].apply(ankle, jacobian @ v, chain.angle(q, body), float(angular @ v), dt)
        wrench = np.asarray(wrench, dtype=float)
        wrenches[side] = wrench
        force += jacobian.T @ wrench[:2] + angular * wrench[2]
    mass, bias = chain.dynamics(q, v, config.gravity_m_s2)
    velocity = v + dt * np.linalg.solve(mass, force - bias)
    state.q = q + dt * velocity
    state.v = velocity
    weight = chain.total_mass_kg * config.gravity_m_s2
    filter_rate = -math.expm1(-dt / config.load_time_s)
    for s, side in enumerate(SIDES):
        state.load_bw[s] += filter_rate * (max(wrenches[side][1], 0.0) / weight - state.load_bw[s])
    drive = (min(state.load_bw[0], 1.0) - min(state.load_bw[1], 1.0)) * math.sin(state.phase_rad)
    frequency = config.frequency_hz * math.exp(0.0 * (speed_m_s - config.reference_speed_m_s))
    phase = state.phase_rad + 2 * math.pi * frequency * (1.0 - config.phase_feedback * drive) * dt
    state.phase_rad = phase - 2 * math.pi * math.floor(phase / (2 * math.pi))
    return wrenches
