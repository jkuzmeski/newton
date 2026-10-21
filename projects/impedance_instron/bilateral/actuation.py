# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smooth substep actuation of the impedance setpoints.

The 3D runner applies one impedance command per control step as a zero-order hold:
:meth:`.env3d.BilateralEnv3D.set_impedance` writes equilibrium, stiffness, and
damping once, and :func:`.env3d._actuate` runs that same command at every substep
of the step (40 substeps of 0.25 ms in the headline policy). Every commanded
change therefore lands as a step in the stiffness of the leg and in the
equilibrium pull, once every 10 ms, and the shoe's strain-stiffening law turns a
setpoint step into a force step. The force the environment reports is the mean of
the substep forces over the step, which hides part of what that does.

The options, all opt-in and all off by default:

``hold``
    ``"zoh"`` reproduces the environment exactly. ``"foh"`` interpolates the
    setpoint linearly across the substeps, so the command reaches its new value
    at the end of the step instead of at its start. The commanded change is
    unchanged; its rate is divided by the substep count, and the loop sees about
    half a control step of extra delay.
``tau_s``
    First-order filter on the setpoints, applied at substep resolution. 10-40 ms
    is the physiological activation time constant of muscle, and the order of the
    series-elastic time constants the runner does not model. This is not
    :attr:`.env3d.Env3DConfig.response_time_s`, which lags the joint *torque*
    after the impedance law; this filter lags the *commanded* impedance, so it
    shapes stiffness and damping as well as the equilibrium.
``slew_per_s``
    Hard bound on the rate of the commanded fraction, in fractions of the joint
    range per second, applied at the control boundary. Zero removes it.
``mode``
    ``"rate"`` reads the action as a setpoint rate: the fraction moves by
    ``tanh(action) * rate_scale_per_s`` per second, which cannot jump. It changes
    what the action means, so it needs training. The other options only change how
    a given action is realized, so a frozen policy stays usable.

:class:`Actuator` is the environment hook: :meth:`apply` replaces
:meth:`.env3d.BilateralEnv3D.set_impedance` once per control step, and
:meth:`begin_substep` runs ahead of every :func:`.env3d._actuate` launch. The
substep write is a Warp kernel, because the environment records the substeps in a
Warp graph that does not capture work launched from Python on the torch stream.
Where that hook is not merged yet, :func:`patch` installs the same two calls on one
instance; :func:`wrap` needs no environment change at all and runs the filter at
control resolution, which is exact for a slew bound and a fair approximation for
a time constant well above the control step.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import torch
import warp as wp

from . import model3d

if TYPE_CHECKING:  # pragma: no cover
    from .env3d import BilateralEnv3D

wp.set_module_options({"enable_backward": False})

_J = wp.constant(model3d.ACTUATED)
_GROUPS = wp.constant(3)
_A = wp.constant(_J * _GROUPS)
VecJ = wp.types.vector(model3d.ACTUATED, wp.float32)

HOLDS = ("zoh", "foh")
MODES = ("absolute", "rate")

PROFILES: dict[str, ActuationConfig] = {}
"""Named actuator settings, filled in at the bottom of the module. A run stores one of
these names in :attr:`.env3d.Env3DConfig.actuation`, so its policy JSON reads
``"lag20_foh"`` rather than an untyped mapping; :meth:`ActuationConfig.parse` also
takes a JSON string, which is what a sweep uses."""


@dataclass(frozen=True)
class ActuationConfig:
    """Smooth actuation options; every default reproduces the environment as trained."""

    hold: str = "zoh"
    tau_s: float = 0.0
    slew_per_s: float = 0.0
    mode: str = "absolute"
    rate_scale_per_s: float = 1.0
    """Fraction of the joint range per second reached by an action of unit magnitude."""

    def __post_init__(self):
        if self.hold not in HOLDS:
            raise ValueError(f"hold must be one of {HOLDS}, not {self.hold!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        if self.tau_s < 0 or self.slew_per_s < 0 or self.rate_scale_per_s <= 0:
            raise ValueError("tau_s and slew_per_s are nonnegative and rate_scale_per_s is positive")

    @property
    def active(self) -> bool:
        """Whether any option changes the zero-order hold."""
        return self.hold != "zoh" or self.tau_s > 0.0 or self.slew_per_s > 0.0 or self.mode != "absolute"

    @classmethod
    def parse(cls, spec: str | dict | ActuationConfig | None) -> ActuationConfig:
        """Build from a :data:`PROFILES` name, a JSON string, a mapping, an instance, or ``None``."""
        if isinstance(spec, ActuationConfig):
            return spec
        if spec is None:
            return cls()
        if isinstance(spec, str):
            if spec in PROFILES:
                return PROFILES[spec]
            spec = json.loads(spec)
        known = {f.name for f in fields(cls)}
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"unknown actuation options {sorted(unknown)}; expected a subset of {sorted(known)}")
        return cls(**spec)


def fraction(logits: torch.Tensor) -> torch.Tensor:
    """Impedance fractions in [0, 1] of the environment's bounded ranges."""
    return torch.sigmoid(logits.float().clamp(-40.0, 40.0))


def logits(fraction: torch.Tensor) -> torch.Tensor:
    """Logits that :meth:`.env3d.BilateralEnv3D.set_impedance` maps back to ``fraction``."""
    clipped = fraction.float().clamp(1.0e-17, 1.0 - 1.0e-17)
    return (torch.log(clipped) - torch.log1p(-clipped)).clamp(-40.0, 40.0)


def _alpha(dt_s: float, tau_s: float) -> float:
    """Exact first-order step fraction for time constant ``tau_s`` over interval ``dt_s``."""
    return 1.0 if tau_s <= 0.0 else -math.expm1(-dt_s / tau_s)


@wp.struct
class ActuatorParams:
    ramp: int
    alpha: float
    substeps: int
    lower: VecJ
    span: VecJ
    stiffness_max: VecJ
    damping_max: VecJ


@wp.struct
class ActuatorBuffers:
    start: wp.array2d[float]
    target: wp.array2d[float]
    held: wp.array2d[float]
    slot: wp.array[int]


@wp.kernel
def _substep(
    buffers: ActuatorBuffers,
    p: ActuatorParams,
    equilibrium: wp.array[VecJ],
    stiffness: wp.array[VecJ],
    damping: wp.array[VecJ],
):
    """One substep: interpolate the command, filter it, and write the impedance setpoints.

    The step's substep index comes from device memory, not from Python: the
    environment records the substeps in a graph whose slot order at replay is not
    the slot order at capture.
    """
    w = wp.tid()
    # One counter per world: every world sees the same number of launches, so the
    # phase stays aligned, and no two threads touch the same element.
    slot = buffers.slot[w] + 1
    buffers.slot[w] = slot
    weight = wp.float32(slot) / wp.float32(p.substeps)
    eq, k, d = VecJ(), VecJ(), VecJ()
    for group in range(_GROUPS):
        for j in range(_J):
            index = group * _J + j
            value = buffers.target[w, index]
            if p.ramp != 0:
                value = buffers.start[w, index] + (value - buffers.start[w, index]) * weight
            held = buffers.held[w, index] + p.alpha * (value - buffers.held[w, index])
            buffers.held[w, index] = held
            if group == 0:
                eq[j] = p.lower[j] + p.span[j] * held
            elif group == 1:
                k[j] = p.stiffness_max[j] * held
            else:
                d[j] = p.damping_max[j] * held
    equilibrium[w] = eq
    stiffness[w] = k
    damping[w] = d


class Actuator:
    """Realizes each control action inside its step, writing the substep's setpoints.

    Call :meth:`apply` once per control step where the environment would call
    :meth:`.env3d.BilateralEnv3D.set_impedance`, and :meth:`begin_substep` at the
    start of every substep. Both write the environment's own ``equilibrium``,
    ``stiffness``, and ``damping`` arrays, so :func:`.env3d._actuate`, the
    observations, and the reward see the actuated command. Everything runs on
    buffers allocated here: :meth:`begin_substep` is a single Warp launch, and
    :meth:`apply` is torch work on the same memory, so both are recorded by the
    environment's CUDA graphs.
    """

    def __init__(self, env: BilateralEnv3D, config: ActuationConfig | str | dict | None = None):
        self.env = env
        self.config = config = ActuationConfig.parse(config)
        cfg, bounds = env.config, env.bounds
        self.worlds = int(env.worlds)
        self.joints = int(env.action_size // _GROUPS)
        self.substeps = int(cfg.substeps)
        self.control_dt_s = float(cfg.control_dt_s)
        self.substep_dt_s = float(cfg.dt_s)
        self.substep_alpha = _alpha(self.substep_dt_s, config.tau_s)
        self.slew_bound = config.slew_per_s * self.control_dt_s
        self.rate_bound = config.rate_scale_per_s * self.control_dt_s
        self.ramp = config.hold == "foh" or config.mode == "rate"
        device = env.device
        shape = (self.worlds, int(_A))
        self.buffers = buffers = ActuatorBuffers()
        for name in ("start", "target", "held"):
            setattr(buffers, name, wp.zeros(shape, dtype=float, device=device))
        buffers.slot = wp.zeros(self.worlds, dtype=int, device=device)
        self.slot = wp.to_torch(self.buffers.slot)
        self.start = wp.to_torch(self.buffers.start)
        """Fraction held when the current step began, a view of :attr:`buffers`."""
        self.target = wp.to_torch(self.buffers.target)
        """Fraction reached at the end of the current step."""
        self.held = wp.to_torch(self.buffers.held)
        """Fraction applied at the last substep: the state the next step starts from."""
        both = lambda values: VecJ(*values, *values)  # noqa: E731  one leg's bounds, both sides, as in env3d
        self.params = ActuatorParams()
        self.params.ramp = int(self.ramp)
        self.params.alpha = self.substep_alpha
        self.params.substeps = self.substeps
        self.params.lower = both(bounds.equilibrium_lower_rad)
        self.params.span = both(
            [
                upper - lower
                for upper, lower in zip(bounds.equilibrium_upper_rad, bounds.equilibrium_lower_rad, strict=False)
            ]
        )
        self.params.stiffness_max = both(bounds.stiffness_max_nm_rad)
        self.params.damping_max = both(bounds.damping_max_nms_rad)
        self.command = torch.zeros(shape, device=env.torch_device)
        self.difference = torch.zeros(shape, device=env.torch_device)
        self.product = torch.zeros((self.worlds, self.joints), device=env.torch_device)
        self._fresh = True

    def apply(self, action: torch.Tensor) -> None:
        """Accept one control action; the step's first substep becomes substep 0.

        The rates are resolved here, at the control boundary, where one action
        stands for one step. Every write targets a buffer allocated in
        :meth:`__init__`, because the environment records this call in a CUDA graph.
        """
        torch.clamp(action.float(), -40.0, 40.0, out=self.command)
        torch.sigmoid(self.command, out=self.command)
        if self._fresh:
            # The first command is where the actuator already is: an untriggered
            # muscle starts relaxed, and a captured warm-up must not leak into a run.
            self.held.copy_(self.command)
            self._fresh = False
        self.start.copy_(self.held)
        self.slot.zero_()
        if self.config.mode == "rate":
            torch.tanh(action.float(), out=self.difference)
            torch.mul(self.difference, self.rate_bound, out=self.command)
            torch.add(self.start, self.command, out=self.command).clamp_(0.0, 1.0)
        if self.slew_bound > 0.0:
            torch.sub(self.command, self.start, out=self.difference)
            self.difference.clamp_(-self.slew_bound, self.slew_bound)
            torch.add(self.start, self.difference, out=self.command)
        self.target.copy_(self.command)
        # Between steps the environment reads the commanded value, so its ``jump``
        # term scores the change the actor asked for; :meth:`begin_substep` replaces
        # it with the realized value as soon as the step runs.
        self.write(self.target)

    def begin_substep(self) -> None:
        """Interpolate, filter, and write the setpoints of one substep."""
        data = self.env.data
        wp.launch(
            _substep,
            dim=self.worlds,
            inputs=[self.buffers, self.params, data.equilibrium, data.stiffness, data.damping],
            device=self.env.device,
        )

    def reset(self) -> None:
        """Snap every world's actuator state onto its current command, as if it had held it."""
        self.held.copy_(self.target)
        self.start.copy_(self.target)
        self.slot.zero_()
        self._fresh = False

    def snap(self, mask: torch.Tensor) -> None:
        """Snap the actuator state of the worlds an episode reset onto their current command.

        :attr:`held` is the actuator's state, and it otherwise carries over the
        impedance of the episode that ended: under a slew bound a reset world spends
        seconds climbing from where the last episode left it, under a filter time
        constant a fraction of a second. Call from the environment's reset with the
        same mask it resets with; unmasked worlds keep their state.

        The substep counter is shared by all worlds and is left alone: the next
        :meth:`apply` re-zeroes it for every world, and zeroing it here would restart
        the ramp of a step already in progress for the worlds that did not reset.
        """
        where = mask.reshape(-1, 1)
        torch.where(where, self.target, self.held, out=self.held)
        torch.where(where, self.target, self.start, out=self.start)

    def write(self, held: torch.Tensor) -> None:
        """Map fractions onto the environment's bounded impedance arrays."""
        env, joints = self.env, self.joints
        for group, (destination, (lower, span)) in enumerate(
            zip((env.t.equilibrium, env.t.stiffness, env.t.damping), self._bounds(), strict=True)
        ):
            torch.mul(held[:, group * joints : (group + 1) * joints], span, out=self.product)
            if lower is None:
                destination.copy_(self.product)
            else:
                torch.add(self.product, lower, out=destination)

    def _bounds(self):
        env = self.env
        joints = self.joints
        return (
            (env.lower.view(1, -1)[:, :joints], (env.upper - env.lower).view(1, -1)[:, :joints]),
            (None, env.stiffness_max.view(1, -1)[:, :joints]),
            (None, env.damping_max.view(1, -1)[:, :joints]),
        )

    def describe(self) -> dict:
        """Configuration of this actuator, for a run manifest. Reads the device, so not during a step."""
        return {
            **asdict(self.config),
            "substeps": self.substeps,
            "control_dt_s": self.control_dt_s,
            "substep_filter_alpha": self.substep_alpha,
        }


def attach(env: BilateralEnv3D) -> Actuator | None:
    """Build the actuator an environment's own configuration asks for, or ``None``.

    Reads ``Env3DConfig.actuation``: when it is set, the environment calls
    :meth:`Actuator.apply` in place of :meth:`.env3d.BilateralEnv3D.set_impedance`
    and :meth:`Actuator.begin_substep` before every substep.
    """
    config = ActuationConfig.parse(getattr(env.config, "actuation", None))
    return Actuator(env, config) if config.active else None


def patch(env: BilateralEnv3D, config: ActuationConfig | str | dict | None) -> Actuator | None:
    """Install an actuator on one instance without an :mod:`.env3d` change.

    Scaffolding for prototypes and this study. It replaces the two places the
    environment hook would touch: the instance's ``set_impedance``, which ``_pre``
    already calls, and a per-substep call ahead of ``_substep``'s launch of
    :func:`.env3d._actuate`. Calling it again installs a fresh actuator; passing an
    inactive or ``None`` configuration restores the environment's own method.
    """
    parsed = ActuationConfig.parse(config)
    if not parsed.active:
        for name in ("set_impedance", "_substep"):
            if name in vars(env):  # only an earlier patch puts these on the instance
                delattr(env, name)
        env.actuator = None
        return None
    actuator = Actuator(env, parsed)
    substep = type(env)._substep
    env.set_impedance = actuator.apply
    env._substep = lambda *states: (actuator.begin_substep(), substep(env, *states))[1]
    env.actuator = actuator
    return actuator


class SmoothedActor:
    """Prefilter of a policy's action at control resolution; needs no environment change.

    Wraps any callable from observations to impedance logits and returns the logits
    whose fractions the :class:`Actuator` would have applied at the end of the step,
    so the environment's own zero-order hold reproduces the filtered command one
    control step late. ``hold`` has no effect here: intra-step interpolation needs
    the substep hook.
    """

    def __init__(
        self,
        actor,
        config: ActuationConfig | str | dict | None = None,
        *,
        control_dt_s: float,
        worlds: int,
        action_size: int,
        device: str = "cuda:0",
    ):
        self.actor = actor
        self.config = config = ActuationConfig.parse(config)
        self.control_dt_s = float(control_dt_s)
        self.alpha = _alpha(self.control_dt_s, config.tau_s)
        self.slew = config.slew_per_s * self.control_dt_s
        self.rate = config.rate_scale_per_s * self.control_dt_s
        self.state: torch.Tensor | None = None
        self.worlds, self.action_size, self.device = int(worlds), int(action_size), device

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.actor(observation)
        command = fraction(action)
        if self.state is None:
            self.state = command.clone()
            return logits(self.state)
        start = self.state
        if self.config.mode == "rate":
            command = (start + self.rate * torch.tanh(action.float())).clamp_(0.0, 1.0)
        if self.slew > 0.0:
            command = start + (command - start).clamp(-self.slew, self.slew)
        if self.alpha < 1.0:
            command = torch.lerp(start, command, self.alpha)
        self.state = command
        return logits(command)


def wrap(actor, config: ActuationConfig | str | dict | None, env: BilateralEnv3D):
    """:class:`SmoothedActor` bound to an environment, or ``actor`` when nothing is enabled."""
    parsed = ActuationConfig.parse(config)
    if not parsed.active:
        return actor
    return SmoothedActor(
        actor, parsed, control_dt_s=env.config.control_dt_s, worlds=env.worlds, action_size=env.action_size
    )


PROFILES.update(
    {
        "zoh": ActuationConfig(),
        "foh": ActuationConfig(hold="foh"),
        "lag10": ActuationConfig(tau_s=0.010),
        "lag20": ActuationConfig(tau_s=0.020),
        "lag40": ActuationConfig(tau_s=0.040),
        "lag20_foh": ActuationConfig(tau_s=0.020, hold="foh"),
        "rate_limit2": ActuationConfig(slew_per_s=2.0),
        "rate_limit20": ActuationConfig(slew_per_s=20.0),
        "rate_action": ActuationConfig(mode="rate", rate_scale_per_s=2.0),
    }
)
