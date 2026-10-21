# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-world shoe parameters (theta), their prior, and the write path into the shoe kernels.

The measured shoe is one identified specimen, so a policy that has to predict how the
runner reacts to a *novel* shoe needs many shoes under its feet at once. The runtime
already supports one foam per world (:attr:`MidsoleFoundation.world_params` holds one
:class:`FoundationParams` block per world), so randomizing the foam costs no extra
device launches. This module owns

* the theta vector and its prior (:class:`ShoeSpec`, :data:`PRESETS`), the named
  archetypes sampled with it (:func:`archetypes`), and the held-out shoes that must
  never appear in training (:func:`heldout`);
* :class:`ShoeRandomizer`, which draws theta per world, re-draws a fraction of the
  worlds every control step (RMA resamples its environment parameters inside an episode
  at 0.01 per step), and installs the result in both foot foundations.

The write path is a kernel rather than
:meth:`~projects.digital_shoe.runtime.MidsoleFoundation.set_world_materials` for two
measured reasons. The environment records the control step\'s torch work and the whole
substep sequence into CUDA graphs, and the foundation refreshes its per-world Maxwell
constants with a Python loop over worlds whenever ``_materials_dirty`` is set; both are
fine once per rollout and unacceptable every control step at 4,096 worlds.
:meth:`ShoeRandomizer.apply` instead writes the constitutive fields of ``world_params``
in place from a packed float table, refreshes the cached zero-strain pressure with the
foundation\'s own kernel, updates the two per-world friction settings (the Coulomb
coefficient and the shear relaxation time) in the adapter\'s own settings array, and
recomputes the surround Maxwell constants with the runtime\'s own scalar function.
Per-column constants - geometry, tributary areas, mount, bristle base stiffness - stay
shared, exactly as in the runtime, so a randomized batch is a study of the foam and of
the friction and never of the sole shape.

Two parameters from the design document are deliberately absent, because the runtime
cannot express them per world today: the static foot-pitch offset and anything that
changes the column bed (its anchors and rest lengths are shared constants). The
stack-height offset is carried instead on the per-world mount shift the environment
already keeps, which moves the whole bed in the foot frame - see
:attr:`ShoeTheta.stack_dz_m`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from projects.digital_shoe.runtime import FoundationParams, maxwell_coefficients_numpy

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for the type checker only
    from .env3d import BilateralEnv3D

wp.set_module_options({"enable_backward": False})

SIDES = ("right", "left")
"""Foot sides, in the order the environment stacks its per-side arrays."""


@dataclass(frozen=True)
class ShoeTheta:
    """One shoe, expressed as the parameters this environment can move.

    ``*_scale`` entries multiply the identified value of the measured specimen, so the
    fitted hyperfoam *shape* stays measured and the prior is a statement about scale.
    The absolute entries replace the identified value, which is zero for
    :attr:`normal_damping_n_s_m` and :attr:`poisson` (the artifact declares both
    unfitted) and 0.8 for :attr:`mu`.

    Attributes:
        g1_scale: First Ogden-Hill shear modulus multiplier on
            ``instantaneous_shear_modulus_pa`` [-].
        alpha1: First Hyperfoam exponent [-].
        feq: Equilibrium fraction; ``1 - feq`` is the viscoelastic share [-].
        tau_scale: Maxwell relaxation time multiplier [-].
        g2_scale: Second Ogden-Hill shear modulus multiplier [-]; zero disables the term.
        normal_damping_n_s_m: Per-column Kelvin-Voigt normal damping [N s/m].
        mu: Elastic-Coulomb coefficient [-].
        stack_dz_m: Stack-height offset [m]; positive is a thicker sole, i.e. the bed sits
            further from the ground for the same foot pose.
        poisson: Effective Poisson ratio [-].
    """

    g1_scale: float = 1.0
    alpha1: float = 0.0
    feq: float = 0.0
    tau_scale: float = 1.0
    g2_scale: float = 1.0
    normal_damping_n_s_m: float = 0.0
    mu: float = 0.0
    stack_dz_m: float = 0.0
    poisson: float = 0.0


THETA_NAMES = (
    "g1_scale",
    "alpha1",
    "feq",
    "tau_scale",
    "g2_scale",
    "normal_damping_n_s_m",
    "mu",
    "stack_dz_m",
    "poisson",
)
"""Theta in the order of the packed device table, the privileged vector and the readout head."""

N_THETA = len(THETA_NAMES)
"""Width of one theta vector."""

SCALED = tuple(name for name in THETA_NAMES if name.endswith("_scale"))
"""Components that multiply an identified value and are therefore sampled in log."""

CONTACT_FORCE_N = 50.0
"""Normal force above which a foot counts as loaded, the same threshold the gait scorer
and the measured-contact penalty of the planar environment use."""

STANCE_CLOCK_STEPS = 40
"""Steps the stance clock saturates at: 0.4 s at 100 Hz, past the longest stance of the
measured gait (232 ms), so a foot that has been down for longer than any real stance is
simply reported as ``1``."""

CONTACT_OBSERVATION_NAMES = (
    "foam_travel_r",
    "foam_travel_l",
    "load_normal_r_bw",
    "load_normal_l_bw",
    "load_tangential_r_bw",
    "load_tangential_l_bw",
    "traction_r",
    "traction_l",
    "stance_r",
    "stance_l",
    "stance_clock_r",
    "stance_clock_l",
)
"""Per-foot channels of the adaptation module: driven-column travel as a fraction of
foam thickness, interval-mean ground force resolved into normal and tangential parts, the
tangential-to-normal ratio the friction cone allows, a touchdown flag, and milliseconds
since touchdown. These are the only observation channels that carry information about the
foam; pelvis state and joint state do not, because the impedance law hides the
difference between a soft foam and a soft actuator."""

THETA_OBSERVATION_NAMES = tuple(f"theta_{name}" for name in THETA_NAMES)
"""The privileged teacher input: theta in the normalization of :func:`normalize`."""


@dataclass(frozen=True)
class Range:
    """One prior interval; ``log=True`` samples uniformly in log, so a scale prior is
    symmetric in perception rather than in absolute value."""

    low: float
    high: float
    log: bool = False

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.log:
            return np.exp(rng.uniform(math.log(self.low), math.log(self.high), size))
        return rng.uniform(self.low, self.high, size)

    def clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.low, self.high)

    @property
    def width(self) -> float:
        """Prior width in the sampled space, which is the unit of the readout target."""
        return math.log(self.high / self.low) if self.log else self.high - self.low


@dataclass(frozen=True)
class ShoeSpec:
    """A named theta prior: which components move, and how far."""

    name: str
    ranges: Mapping[str, Range] = field(default_factory=dict)
    manifold_fraction: float = 0.0
    """Probability of drawing a jittered named archetype instead of an independent draw."""
    manifold_sigma: float = 0.08
    """Log-normal jitter applied to the covered components of a drawn archetype [-]."""
    swap_probability: float = 0.01
    """Mid-episode resampling probability per world per control step."""


_REGISTRATION = {
    "mu": Range(0.5, 1.1),
    "stack_dz_m": Range(-0.008, 0.008),
}
_BAND = {
    "g1_scale": Range(0.8, 1.25, log=True),
    "alpha1": Range(0.196, 0.294),
    "feq": Range(0.556, 0.834),
    "tau_scale": Range(0.8, 1.25, log=True),
    "g2_scale": Range(0.8, 1.25, log=True),
    **_REGISTRATION,
}
_WIDE = {
    "g1_scale": Range(0.5, 2.0, log=True),
    "alpha1": Range(0.12, 0.62),
    "feq": Range(0.45, 1.0),
    "tau_scale": Range(0.2, 6.0, log=True),
    "g2_scale": Range(0.2, 20.0, log=True),
    "normal_damping_n_s_m": Range(0.0, 4.0),
    "mu": Range(0.5, 1.1),
    "stack_dz_m": Range(-0.008, 0.008),
}

PRESETS: dict[str, ShoeSpec] = {
    "off": ShoeSpec("off"),
    # S1 of the design document: registration and traction only. The foam is the
    # measured one in every world, so a policy that passes this stage is not yet
    # shoe-aware, it is only mount- and grip-robust.
    "registration": ShoeSpec("registration", _REGISTRATION),
    # S2: the identification error band of the artifact itself, about +-20 % on the
    # fitted terms, plus registration.
    "band": ShoeSpec("band", _BAND, manifold_fraction=0.25),
    # S3: the full prior of design document section 5, archetypes included.
    "wide": ShoeSpec("wide", _WIDE, manifold_fraction=0.5),
}
"""Named priors, keyed by the value of ``Env3DConfig.shoe_dr``."""

PACKED_FIELDS = (
    "g_eq",
    "alpha",
    "g_eq2",
    "beta",
    "one_minus_two_poisson",
    "tau_s",
    "overstress",
    "normal_damping",
    "mu",
    "friction_relaxation_time_s",
)
"""The :class:`FoundationParams` fields the per-world table writes, in row order."""

MU_COLUMN = PACKED_FIELDS.index("mu")
TAU_COLUMN = PACKED_FIELDS.index("tau_s")
OVERSTRESS_COLUMN = PACKED_FIELDS.index("overstress")
NORMAL_DAMPING_COLUMN = PACKED_FIELDS.index("normal_damping")
RELAXATION_COLUMN = PACKED_FIELDS.index("friction_relaxation_time_s")
_SETTINGS_MU = 1
_SETTINGS_TAU = 11


@wp.kernel
def _write_world_friction(
    values: wp.array2d[wp.float64],
    settings: wp.array2d[wp.float32],
):
    """Give every world its Coulomb coefficient and its shear relaxation time.

    The tangent law reads both out of the friction adapter's settings table rather than
    out of the constitutive block: :func:`.gpu_shoe._ground_contact` reads column 1 and
    :mod:`.friction_parameter_adapter` reads column 11. The runtime writes the same two
    columns for a whole batch in :meth:`.MidsoleFoundation.set_world_materials`, which
    pushes the relaxation time through :meth:`FrictionParameterAdapter.update_taus`; this
    kernel is that write without the host round trip. The constitutive block itself is
    written by :func:`set_block_row` and one array copy.
    """
    w = wp.tid()
    settings[w, _SETTINGS_MU] = wp.float32(values[w, MU_COLUMN])
    settings[w, _SETTINGS_TAU] = wp.float32(values[w, RELAXATION_COLUMN])


def set_block_row(block: FoundationParams, row: Sequence[float]) -> None:
    """Write one packed theta row into a host constitutive block.

    This is :func:`~projects.digital_shoe.runtime.set_material_block` for a row of the
    table instead of a material object, and it is not optional: the midsole foundation
    recomputes the surround Maxwell constants lazily from the **host** blocks
    (``_refresh_surround_constants``), so a shoe written only into the device array is
    later overwritten by constants of the previous foam. Both copies are written, as
    :meth:`.MidsoleFoundation.set_world_materials` does.

    Fields the table does not carry - the second-term exponent, the constitutive block's
    Coulomb coefficient (which :func:`~projects.digital_shoe.runtime.set_material_block`
    also leaves alone; :func:`_write_world_friction` is where a per-world shoe puts mu),
    and everything a :class:`~projects.digital_shoe.runtime.FoundationConfig` owns,
    including a densification branch - are kept as found, so this cannot switch off a law
    another module owns.
    """
    block.g_eq = float(row[0])
    block.alpha = float(row[1])
    block.g_eq2 = float(row[2])
    block.beta = float(row[3])
    block.one_minus_two_poisson = float(row[4])
    block.tau_s = float(row[TAU_COLUMN])
    block.overstress = float(row[OVERSTRESS_COLUMN])
    block.normal_damping = float(row[NORMAL_DAMPING_COLUMN])
    block.friction_relaxation_time_s = float(row[RELAXATION_COLUMN])


def archetypes(nominal: Mapping[str, float]) -> dict[str, ShoeTheta]:
    """Named shoes with provenance, as theta relative to the measured specimen.

    A real shoe trades several parameters at once, so independent draws cover the box
    without ever landing on a believable shoe. The names are stated as modifications of
    the identified foam (design document section 5): PEBA racing foams are softer and
    more elastic than the nominal, EVA-like foams the other way, and a worn midsole
    loses modulus and gains hysteresis (+0.32 W/kg over 450 km, 10.1111/sms.14526).
    None of these is a measured shoe; they are syntheses inside one identified family
    and are reported as such.
    """
    return {
        "fr3_nominal": ShoeTheta(),
        "peba_like": ShoeTheta(g1_scale=0.7, feq=0.92),
        "eva_like": ShoeTheta(g1_scale=1.6, feq=0.5, tau_scale=3.0),
        "worn": ShoeTheta(g1_scale=0.85, feq=0.6, normal_damping_n_s_m=1.5),
        "elastic_corner": ShoeTheta(feq=1.0),
        "viscous_corner": ShoeTheta(feq=0.45, tau_scale=6.0),
    }


def heldout() -> dict[str, tuple[ShoeTheta, str]]:
    """Shoes for the evaluation protocol, with the axis each one probes.

    ``interp`` rows sit strictly inside the trained box but are fixed points a random
    draw almost never returns; ``extrap`` rows leave it, either in the soft direction
    (where the foam law has no post-densification branch and the environment aborts the
    world), in traction, in dissipation, or in stack height.
    """
    return {
        "interp_soft_elastic": (ShoeTheta(g1_scale=0.72, feq=0.86, tau_scale=1.7), "interpolation"),
        "interp_stiff_damped": (
            ShoeTheta(g1_scale=1.42, alpha1=0.34, feq=0.58, normal_damping_n_s_m=1.1),
            "interpolation",
        ),
        "extrap_stiff_3x": (ShoeTheta(g1_scale=3.0), "extrapolation"),
        "extrap_soft_0p4": (ShoeTheta(g1_scale=0.4), "extrapolation"),
        "extrap_viscous": (ShoeTheta(feq=0.42, tau_scale=12.0), "extrapolation"),
        "extrap_slick": (ShoeTheta(mu=0.42), "extrapolation"),
        "extrap_thick_15mm": (ShoeTheta(stack_dz_m=-0.015), "extrapolation"),
        "extrap_heavy_damping": (ShoeTheta(normal_damping_n_s_m=8.0), "extrapolation"),
    }


def nominal_row(reference: Mapping[str, float]) -> np.ndarray:
    """The theta row that reproduces the measured artifact exactly."""
    return np.array(
        [
            1.0,
            reference["hyperfoam_exponent"],
            reference["equilibrium_fraction"],
            1.0,
            1.0,
            0.0,
            reference["mu"],
            0.0,
            reference["effective_poisson_ratio"],
        ],
        # float64: this row is what the constitutive table is computed from, and the
        # runtime derives its constants from float64 material fields. Storing theta at
        # float32 here would round a fitted constant twice where the runtime rounds once,
        # which is a difference the contact law can see.
        dtype=np.float64,
    )


def as_row(theta: ShoeTheta, reference: Mapping[str, float]) -> np.ndarray:
    """One :class:`ShoeTheta` as a theta vector against ``reference``."""
    row = nominal_row(reference)
    for name in THETA_NAMES:
        value = getattr(theta, name)
        if name in SCALED or value != 0.0:
            row[THETA_NAMES.index(name)] = value
    return row


def pack(theta: np.ndarray, reference: Mapping[str, float]) -> np.ndarray:
    """Turn theta rows ``[worlds, N_THETA]`` into the packed constitutive table.

    The expressions are the runtime\'s own (``runtime.set_material_block``): each
    equilibrium term carries the equilibrium fraction, ``overstress = (1 - feq) / feq``,
    ``beta = nu / (1 - 2 nu)``. The test suite checks the two agree.
    """
    rows = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    # float64, not float32: the constitutive block is float32, and casting the row
    # before the write would round a fitted constant twice where the runtime rounds it once.
    out = np.zeros((len(rows), len(PACKED_FIELDS)), dtype=np.float64)
    take = lambda name: rows[:, THETA_NAMES.index(name)]  # noqa: E731
    g1 = take("g1_scale") * reference["instantaneous_shear_modulus_pa"]
    g2 = take("g2_scale") * reference["instantaneous_shear_modulus_2_pa"]
    feq = np.clip(take("feq"), 1.0e-3, 1.0)
    tau = np.clip(take("tau_scale") * reference["maxwell_relaxation_time_s"], 1.0e-6, None)
    poisson = np.clip(take("poisson"), -0.9, 0.45)
    out[:, PACKED_FIELDS.index("g_eq")] = g1 * feq
    out[:, PACKED_FIELDS.index("alpha")] = take("alpha1")
    out[:, PACKED_FIELDS.index("g_eq2")] = g2 * feq
    out[:, PACKED_FIELDS.index("beta")] = poisson / (1.0 - 2.0 * poisson)
    out[:, PACKED_FIELDS.index("one_minus_two_poisson")] = 1.0 - 2.0 * poisson
    out[:, TAU_COLUMN] = tau
    out[:, OVERSTRESS_COLUMN] = (1.0 - feq) / feq
    out[:, NORMAL_DAMPING_COLUMN] = take("normal_damping_n_s_m")
    out[:, MU_COLUMN] = take("mu")
    out[:, RELAXATION_COLUMN] = tau
    return out


def normalize(theta: torch.Tensor, spec: ShoeSpec, reference: Mapping[str, float]) -> torch.Tensor:
    """Put theta on a common scale: zero at the nominal shoe, one at the prior edge.

    Scaled components enter in log, so a factor two is the same distance whichever way
    it goes. The critic and the estimator readout see the same encoding, which is what
    makes the readout error quotable as a fraction of the prior width.
    """
    out = torch.empty_like(theta)
    base = nominal_row(reference)
    for index, name in enumerate(THETA_NAMES):
        column = theta[:, index]
        if name in SCALED:
            center = torch.log(column.clamp(min=1.0e-6)) - math.log(float(base[index]))
        else:
            center = column - float(base[index])
        width = 0.5 * spec.ranges[name].width if name in spec.ranges else 1.0
        out[:, index] = center / max(float(width), 1.0e-9)
    return out.clamp(-6.0, 6.0)


class ShoeRandomizer:
    """Draw, resample, and install per-world shoe parameters.

    Args:
        env: The 3D environment whose two foot foundations get randomized.
        spec: The theta prior.
        seed: Stream seed. Two randomizers with the same seed and world count draw the
            same shoes, which is what makes a paired-world evaluation possible.
        swap_probability: Probability per control step that a world gets a new shoe.
            RMA uses 0.01 per step. ``None`` takes the prior\'s own value; zero keeps
            one shoe per world per episode.
    """

    def __init__(
        self,
        env: BilateralEnv3D,
        spec: ShoeSpec,
        *,
        seed: int = 0,
        swap_probability: float | None = None,
    ) -> None:
        self.env = env
        self.spec = spec
        self.worlds = int(env.worlds)
        self.device = env.device
        self.torch_device = env.torch_device
        self.swap_probability = spec.swap_probability if swap_probability is None else float(swap_probability)
        self.rng = np.random.default_rng(seed)
        self.reference = reference_of(env)
        self.archetypes = archetypes(self.reference)
        self.base = nominal_row(self.reference)
        self.host = np.tile(self.base, (self.worlds, 1))
        self.theta = torch.as_tensor(self.host, dtype=torch.float32, device=self.torch_device)
        self._packed = {}
        self._settings = {}
        self.friction = 0
        for side in SIDES:
            foundation = env.shoes[side].foundation
            adapter = foundation.friction_solver
            self.friction = int(adapter is not None)
            self._packed[side] = wp.zeros((self.worlds, len(PACKED_FIELDS)), dtype=wp.float64, device=self.device)
            self._settings[side] = adapter.settings if adapter is not None else self._zeros()
        self.dt_s = float(env.config.dt_s)
        self._packed_host = np.zeros((self.worlds, len(PACKED_FIELDS)), dtype=np.float64)
        self._decay: dict[str, np.ndarray] = {}
        self._gain: dict[str, np.ndarray] = {}
        self._shift = {side: wp.to_torch(getattr(env.data, f"shift_{side[0]}")).clone() for side in SIDES}
        self.swaps = 0
        self.swap_steps: list[int] = []
        # Every world starts on its own shoe: domain randomization is per world from the
        # first step, not something the swap timer has to grow. ``swap_probability`` then
        # only decides whether a world also changes shoe mid-episode.
        self.draw()
        self.apply()

    def _zeros(self):
        return wp.zeros((1, 12), dtype=wp.float64, device=self.device)

    # ----- drawing -----

    def draw(self, rows: Sequence[int] | np.ndarray | None = None) -> None:
        """Draw fresh theta for the given worlds (default: all of them)."""
        rows = np.arange(self.worlds) if rows is None else np.asarray(rows, dtype=np.int64)
        if not len(rows):
            return
        draw = np.tile(self.base, (len(rows), 1))
        named = np.zeros(len(rows), dtype=bool)
        if self.spec.manifold_fraction > 0.0:
            names = list(self.archetypes)
            named = self.rng.random(len(rows)) < self.spec.manifold_fraction
            for index in np.flatnonzero(named):
                theta = self.archetypes[names[int(self.rng.integers(len(names)))]]
                draw[index] = as_row(theta, self.reference)
        independent = ~named
        for name, interval in self.spec.ranges.items():
            column = THETA_NAMES.index(name)
            draw[independent, column] = interval.draw(self.rng, int(independent.sum()))
            if named.any():
                # An archetype is a fixed composition; jitter only the components the
                # prior covers, so the named trade-off between them survives.
                sigma = self.spec.manifold_sigma if name in SCALED else 0.5 * interval.width * self.spec.manifold_sigma
                jitter = np.exp(self.rng.normal(0.0, sigma, int(named.sum())))
                draw[named, column] = interval.clip(draw[named, column] * jitter)
        self.host[rows] = draw
        self.theta[rows] = torch.as_tensor(self.host[rows], dtype=torch.float32, device=self.torch_device)

    def pin(self, points: Sequence[ShoeTheta], rows: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """Install fixed theta points, one per world, and stop mid-episode swapping.

        The held-out evaluation protocol needs shoes that stay put and that a random draw
        would essentially never produce; :meth:`draw` cannot give that because it samples
        the prior and jitters the archetypes. A single point is broadcast to every row, so
        the baseline condition (the measured shoe everywhere) is one call.

        Returns:
            The installed theta rows ``[len(rows), N_THETA]`` in float64, which is the
            ground truth an estimator is scored against.
        """
        rows = np.arange(len(points)) if rows is None else np.asarray(rows, dtype=np.int64)
        if len(points) == 1 and len(rows) > 1:
            points = tuple(points) * len(rows)
        if len(points) != len(rows):
            raise ValueError(f"{len(points)} shoe points for {len(rows)} worlds")
        for row, theta in zip(rows, points, strict=True):
            self.host[int(row)] = as_row(theta, self.reference)
        self.theta[rows] = torch.as_tensor(self.host[rows], dtype=torch.float32, device=self.torch_device)
        self.apply(rows)
        self.swap_probability = 0.0
        return self.host[rows].copy()

    def advance(self, step: int) -> int:
        """Resample the worlds that swap this control step, then install every shoe.

        The environment calls this between control steps, never inside a captured graph.
        The draw stays on the host: one comparison per world is cheaper than the host
        round trip that would be needed to read a device decision back, and it keeps the
        swap schedule independent of the CUDA-graph philox state.

        Returns the number of worlds that got a new shoe.
        """
        if self.swap_probability <= 0.0:
            return 0
        rows = np.flatnonzero(self.rng.random(self.worlds) < self.swap_probability)
        if len(rows):
            self.draw(rows)
            self.swaps += int(len(rows))
            self.swap_steps.append(int(step))
        self.apply(rows)
        return int(len(rows))

    # ----- installing -----

    def apply(self, rows: slice | np.ndarray = slice(None)) -> None:
        """Install theta into both foot foundations, in place, for ``rows`` or for all worlds.

        Only the packed rows that changed are recomputed; the table copy, the
        constitutive kernel, and the surround constants are per world anyway, so a swap
        of forty worlds out of four thousand costs the same handful of launches as a
        full redraw.
        """
        packed = pack(self.host[rows], self.reference)
        if isinstance(rows, slice):
            self._packed_host = np.ascontiguousarray(packed)
            self._decay = {}
            self._gain = {}
            dirty = slice(None)
        else:
            self._packed_host[rows] = packed
            dirty = rows
        touched: Sequence[int] = range(self.worlds) if isinstance(rows, slice) else rows
        for side in SIDES:
            foundation = self.env.shoes[side].foundation
            for index in touched:
                set_block_row(foundation.world_blocks[int(index)], self._packed_host[int(index)])
            foundation.world_params.assign(foundation.world_blocks)
            self._packed[side].assign(self._packed_host)
            if self.friction:
                wp.launch(
                    _write_world_friction,
                    dim=self.worlds,
                    inputs=[self._packed[side], self._settings[side]],
                    device=self.device,
                )
            foundation._refresh_zero_pressure()
            decay, gain = self._decay.get(side), self._gain.get(side)
            if decay is None:
                decay = np.empty(self.worlds, np.float32)
                gain = np.empty(self.worlds, np.float32)
                self._fill_surround(decay, gain, foundation, slice(None))
            else:
                self._fill_surround(decay, gain, foundation, dirty)
            self._decay[side], self._gain[side] = decay, gain
            wp.to_torch(foundation.surround_decay).copy_(torch.as_tensor(decay, device=self.torch_device))
            wp.to_torch(foundation.surround_gain).copy_(torch.as_tensor(gain, device=self.torch_device))
        self._install_shift()

    def _fill_surround(self, decay: np.ndarray, gain: np.ndarray, foundation, rows) -> None:
        """Recompute the per-world Maxwell constants the passive-column sweep relaxes against.

        The shared runtime does this for every world whenever its dirty flag is set,
        which is right once per rollout and too slow every control step; the runtime's
        own scalar function is still the only place the numbers come from, it is just
        called only for the worlds whose relaxation time moved.
        """
        indices = range(self.worlds) if isinstance(rows, slice) else rows
        for index in indices:
            block = foundation.world_blocks[int(index)]
            world_decay, ramp = maxwell_coefficients_numpy(self.dt_s, float(block.tau_s))
            decay[index] = world_decay
            gain[index] = float(block.overstress * ramp)

    def _install_shift(self) -> None:
        """Apply the stack-height draw: a thicker sole moves the whole bed away from the ground."""
        dz = self.theta[:, THETA_NAMES.index("stack_dz_m")]
        for side in SIDES:
            array = getattr(self.env.data, f"shift_{side[0]}")
            shifted = self._shift[side].clone()
            shifted[:, 2] -= dz
            wp.to_torch(array).copy_(shifted)

    # ----- readings -----

    def privileged(self) -> torch.Tensor:
        """Theta in the shared normalization, for the critic, the actor, and the readout target."""
        return normalize(self.theta, self.spec, self.reference)

    @staticmethod
    def index(component: str) -> int:
        """Column of one theta component in :attr:`ShoeRandomizer.host` and :attr:`theta`."""
        return THETA_NAMES.index(component)

    def shoe(self, world: int) -> dict:
        """One world\'s shoe as named values, for logs and for the evaluation record."""
        row = self.host[int(world)]
        packed = pack(row[None, :], self.reference)[0]
        return {
            "theta": {name: round(float(row[index]), 6) for index, name in enumerate(THETA_NAMES)},
            "constants": {name: round(float(packed[index]), 6) for index, name in enumerate(PACKED_FIELDS)},
            "equilibrium_shear_modulus_pa": round(float(packed[0] + packed[2]), 3),
        }

    def describe(self) -> dict:
        """What this randomizer can cover, for the run provenance."""
        return {
            "preset": self.spec.name,
            "swap_probability_per_step": self.swap_probability,
            "theta_names": list(THETA_NAMES),
            "components": {
                name: (
                    None
                    if name not in self.spec.ranges
                    else {
                        "low": self.spec.ranges[name].low,
                        "high": self.spec.ranges[name].high,
                        "space": "log" if self.spec.ranges[name].log else "linear",
                    }
                )
                for name in THETA_NAMES
            },
            "manifold_fraction": self.spec.manifold_fraction,
            "nominal": {key: float(value) for key, value in self.reference.items()},
            "archetypes": {
                name: {
                    key: round(float(value), 6)
                    for key, value in zip(THETA_NAMES, as_row(theta, self.reference), strict=True)
                }
                for name, theta in self.archetypes.items()
            },
            "not_randomizable": [
                "static foot pitch: the bed anchors are shared per-column constants",
                "column geometry, tributary areas, and the driven (plate footprint) mask",
            ],
        }


def reference_of(env: BilateralEnv3D) -> dict[str, float]:
    """The identified constitutive values the environment was built with, both feet.

    Read from the artifact files recorded in ``env.shoe_metadata`` rather than from the
    device block, so the prior is centred on the exact fitted numbers instead of on
    their float32 round trip. The mirrored right shoe is the same material, and this
    checks rather than assumes it.
    """
    from projects.digital_shoe.artifact import load_artifact  # noqa: PLC0415

    materials = {}
    for side in SIDES:
        metadata = env.shoe_metadata[side]
        materials[side] = load_artifact(metadata["path"]).material
    first, other = materials["right"], materials["left"]
    for name in (
        "instantaneous_shear_modulus_pa",
        "instantaneous_shear_modulus_2_pa",
        "hyperfoam_exponent",
        "hyperfoam_exponent_2",
        "equilibrium_fraction",
        "maxwell_relaxation_time_s",
        "effective_poisson_ratio",
    ):
        if float(getattr(first, name)) != float(getattr(other, name)):
            raise ValueError(f"the two feet do not share the nominal material ({name} differs on the left)")
    adapter = env.shoes["right"].foundation.friction_solver
    return {
        "instantaneous_shear_modulus_pa": float(first.instantaneous_shear_modulus_pa),
        "instantaneous_shear_modulus_2_pa": float(first.instantaneous_shear_modulus_2_pa),
        "hyperfoam_exponent": float(first.hyperfoam_exponent),
        "hyperfoam_exponent_2": float(first.hyperfoam_exponent_2),
        "equilibrium_fraction": float(first.equilibrium_fraction),
        "maxwell_relaxation_time_s": float(first.maxwell_relaxation_time_s),
        "effective_poisson_ratio": float(first.effective_poisson_ratio),
        "mu": 0.8 if adapter is None else float(adapter.settings.numpy()[0, _SETTINGS_MU]),
    }
