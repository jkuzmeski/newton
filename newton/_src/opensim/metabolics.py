# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

r"""Warp-native muscle metabolic cost estimation.

This module implements the Bhargava et al. (2004) phenomenological muscle
energy model using the conventions and defaults of OpenSim's
``Bhargava2004SmoothedMuscleMetabolics`` component. The model decomposes each
muscle's metabolic power into activation, maintenance, shortening, and
contractile-element mechanical-work rates. It reproduces OpenSim's original
piecewise rate equations or their hyperbolic-tangent approximation.

The public :meth:`MuscleMetabolicsBhargava2004.compute` method is a batched
analysis/post-processing API: kernels execute on the selected Warp device and
results return as NumPy arrays. It is not a differentiable optimal-control cost
interface.

The equations and defaults were ported from OpenSim's implementation:
https://github.com/opensim-org/opensim-core/blob/1f6723555065755a25c9973ae60b4c653e18e215/OpenSim/Simulation/Model/Bhargava2004SmoothedMuscleMetabolics.cpp

The underlying model is described by Bhargava, Pandy, and Anderson (2004),
"A phenomenological model for estimating metabolic energy consumption in
muscle contraction," Journal of Biomechanics 37(1), 81--88,
https://doi.org/10.1016/S0021-9290(03)00239-2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .muscle import dgf_active_force_length, dgf_force_velocity, dgf_passive_force_length, pennated_fiber_length
from .muscle_force import MuscleForces
from .types import OsimModel

_f32 = wp.float32
_f64 = wp.float64


@dataclass(frozen=True)
class MuscleMetabolicsBhargava2004Parameters:
    """Bhargava (2004) parameters for one muscle.

    ``muscle_mass=None`` uses OpenSim's physiological cross-sectional-area
    approximation ``F_max / specific_tension * density * optimal_fiber_length``.

    Attributes:
        slow_twitch_ratio: Fraction of slow-twitch fibers, in [0, 1].
        specific_tension: Muscle specific tension [N/m^2].
        density: Muscle density [kg/m^3].
        muscle_mass: Optional provided muscle mass [kg].
        activation_constant_slow_twitch: Slow-twitch activation heat constant [W/kg].
        activation_constant_fast_twitch: Fast-twitch activation heat constant [W/kg].
        maintenance_constant_slow_twitch: Slow-twitch maintenance heat constant [W/kg].
        maintenance_constant_fast_twitch: Fast-twitch maintenance heat constant [W/kg].
    """

    slow_twitch_ratio: float = 0.5
    specific_tension: float = 0.25e6
    density: float = 1059.7
    muscle_mass: float | None = None
    activation_constant_slow_twitch: float = 40.0
    activation_constant_fast_twitch: float = 133.0
    maintenance_constant_slow_twitch: float = 74.0
    maintenance_constant_fast_twitch: float = 111.0


@dataclass(frozen=True)
class MuscleMetabolicsBhargava2004Result:
    """Metabolic power estimates over a trajectory.

    The five per-muscle arrays have shape ``[num_frames, num_muscles]``. OpenSim
    applies the minimum-heat and non-negative-power constraints to
    ``muscle_rate`` after calculating the reported components, so the component
    arrays need not sum exactly to ``muscle_rate`` when a constraint is active.

    Attributes:
        muscle_names: Muscle names in column order.
        muscle_mass: Muscle masses [kg], shape ``[num_muscles]``.
        activation_rate: Activation heat rates [W].
        maintenance_rate: Maintenance heat rates [W].
        shortening_rate: Shortening heat rates [W].
        mechanical_work_rate: Contractile-element mechanical-work rates [W].
        muscle_rate: Total per-muscle metabolic rates [W].
        basal_rate: Whole-body basal metabolic rate [W].
        total_rate: Whole-body total metabolic rate [W], shape ``[num_frames]``.
        body_mass: Whole-body mass used for basal power and normalization [kg].
    """

    muscle_names: list[str]
    muscle_mass: np.ndarray
    activation_rate: np.ndarray
    maintenance_rate: np.ndarray
    shortening_rate: np.ndarray
    mechanical_work_rate: np.ndarray
    muscle_rate: np.ndarray
    basal_rate: float
    total_rate: np.ndarray
    body_mass: float

    def total_energy(self, times: np.ndarray) -> float:
        """Integrate total metabolic power over ``times`` and return energy [J].

        Args:
            times: Strictly increasing sample times [s], shape ``[num_frames]``.
        """
        times = self._validate_times(times)
        intervals = np.diff(times)
        return float(np.sum(0.5 * intervals * (self.total_rate[:-1] + self.total_rate[1:])))

    def muscle_energy(self, times: np.ndarray) -> np.ndarray:
        """Integrate each muscle's metabolic power and return energy [J].

        Args:
            times: Strictly increasing sample times [s], shape ``[num_frames]``.

        Returns:
            Per-muscle metabolic energy [J], shape ``[num_muscles]``.
        """
        times = self._validate_times(times)
        intervals = np.diff(times)[:, None]
        return np.sum(0.5 * intervals * (self.muscle_rate[:-1] + self.muscle_rate[1:]), axis=0)

    def cost_of_transport(self, times: np.ndarray, distance: float) -> float:
        """Return gross metabolic cost of transport [J/(kg·m)].

        This matches OpenSim Moco's metabolic output-goal normalization when
        ``distance`` is the norm of the whole-body center-of-mass displacement
        between the initial and final states. The gross cost includes basal
        metabolic power.

        Args:
            times: Strictly increasing sample times [s], shape ``[num_frames]``.
            distance: Positive normalization displacement [m].
        """
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("distance must be finite and positive")
        if not np.isfinite(self.body_mass) or self.body_mass <= 0.0:
            raise ValueError("body_mass must be finite and positive")
        return self.total_energy(times) / (self.body_mass * float(distance))

    def _validate_times(self, times: np.ndarray) -> np.ndarray:
        """Validate integration sample times."""
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or times.shape[0] != self.total_rate.shape[0]:
            raise ValueError(f"times must have shape [{self.total_rate.shape[0]}]")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be finite and strictly increasing")
        return times


@wp.func
def _tanh_conditional(cond: _f32, left: _f32, right: _f32, smoothing: _f32) -> _f32:
    """Return OpenSim's tanh-smoothed piecewise conditional."""
    weight = _f32(0.5) + _f32(0.5) * wp.tanh(smoothing * cond)
    return left + (right - left) * weight


@wp.func
def _maintenance_length_factor(normalized_fiber_length: _f32) -> _f32:
    """Evaluate OpenSim's default Bhargava maintenance length curve."""
    if normalized_fiber_length <= _f32(0.5):
        return _f32(0.5)
    if normalized_fiber_length <= _f32(1.0):
        return normalized_fiber_length
    if normalized_fiber_length <= _f32(1.5):
        return _f32(3.0) - _f32(2.0) * normalized_fiber_length
    return _f32(0.0)


@wp.kernel
def metabolic_rate_kernel(
    activation: wp.array2d[_f32],
    excitation: wp.array2d[_f32],
    lmt: wp.array2d[_f64],
    vmt: wp.array2d[_f64],
    fmax: wp.array[_f32],
    l_opt: wp.array[_f32],
    lt_slack: wp.array[_f32],
    vmax: wp.array[_f32],
    cos_penn: wp.array[_f32],
    muscle_mass: wp.array[_f32],
    slow_twitch_ratio: wp.array[_f32],
    activation_constant_slow: wp.array[_f32],
    activation_constant_fast: wp.array[_f32],
    maintenance_constant_slow: wp.array[_f32],
    maintenance_constant_fast: wp.array[_f32],
    effort_scaling: _f32,
    use_force_dependent_shortening_constant: bool,
    include_negative_mechanical_work: bool,
    forbid_negative_total_power: bool,
    enforce_minimum_heat_rate: bool,
    use_smoothing: bool,
    velocity_smoothing: _f32,
    power_smoothing: _f32,
    heat_rate_smoothing: _f32,
    out: wp.array3d[_f32],
):
    """Evaluate Bhargava metabolic power for one frame and muscle per thread."""
    b, m = wp.tid()
    act = effort_scaling * activation[b, m]
    exc = effort_scaling * excitation[b, m]
    slow_ratio = slow_twitch_ratio[m]
    slow_excitation = slow_ratio * wp.sin(_f32(0.5) * wp.pi * exc)
    fast_excitation = (_f32(1.0) - slow_ratio) * (_f32(1.0) - wp.cos(_f32(0.5) * wp.pi * exc))

    length = _f32(lmt[b, m])
    path_velocity = _f32(vmt[b, m])
    optimal_length = l_opt[m]
    normalized_length = pennated_fiber_length(length, lt_slack[m], optimal_length, cos_penn[m])
    fiber_length = normalized_length * optimal_length
    current_cos_penn = wp.max(
        (length - lt_slack[m]) / wp.max(fiber_length, _f32(1.0e-9)),
        _f32(0.0),
    )
    normalized_velocity = (path_velocity * current_cos_penn) / wp.max(optimal_length * vmax[m], _f32(1.0e-9))
    fiber_velocity = normalized_velocity * optimal_length * vmax[m]

    active_force_length = dgf_active_force_length(normalized_length)
    active_fiber_force = fmax[m] * act * active_force_length * dgf_force_velocity(normalized_velocity)
    passive_fiber_force = fmax[m] * dgf_passive_force_length(normalized_length)
    total_fiber_force = active_fiber_force + passive_fiber_force
    isometric_active_force = act * active_force_length * fmax[m]

    mass = muscle_mass[m]
    activation_rate = mass * (
        activation_constant_slow[m] * slow_excitation + activation_constant_fast[m] * fast_excitation
    )
    maintenance_rate = (
        mass
        * _maintenance_length_factor(normalized_length)
        * (maintenance_constant_slow[m] * slow_excitation + maintenance_constant_fast[m] * fast_excitation)
    )

    eps = _f32(1.0e-16)
    velocity_condition = fiber_velocity + eps
    alpha = _f32(0.0)
    if use_force_dependent_shortening_constant:
        concentric_alpha = _f32(0.16) * isometric_active_force + _f32(0.18) * total_fiber_force
        eccentric_alpha = _f32(0.157) * total_fiber_force
        if use_smoothing:
            alpha = _tanh_conditional(velocity_condition, concentric_alpha, eccentric_alpha, velocity_smoothing)
        elif velocity_condition <= _f32(0.0):
            alpha = concentric_alpha
        else:
            alpha = eccentric_alpha
    else:
        concentric_alpha = _f32(0.25) * total_fiber_force
        if use_smoothing:
            alpha = _tanh_conditional(velocity_condition, concentric_alpha, _f32(0.0), velocity_smoothing)
        elif velocity_condition <= _f32(0.0):
            alpha = concentric_alpha

    shortening_rate = -alpha * velocity_condition
    mechanical_work_rate = -active_fiber_force * fiber_velocity
    if not include_negative_mechanical_work:
        if use_smoothing:
            mechanical_work_rate = _tanh_conditional(
                velocity_condition, mechanical_work_rate, _f32(0.0), velocity_smoothing
            )
        elif velocity_condition > _f32(0.0):
            mechanical_work_rate = _f32(0.0)

    power_before_clamp = activation_rate + maintenance_rate + shortening_rate + mechanical_work_rate
    if forbid_negative_total_power:
        if use_smoothing:
            adjustment = _tanh_conditional(-power_before_clamp, _f32(0.0), power_before_clamp, power_smoothing)
            shortening_rate -= adjustment
        elif power_before_clamp < _f32(0.0):
            shortening_rate -= power_before_clamp

    total_heat_rate = activation_rate + maintenance_rate + shortening_rate
    minimum_heat_rate = mass
    if enforce_minimum_heat_rate:
        if use_smoothing:
            total_heat_rate = _tanh_conditional(
                -total_heat_rate + minimum_heat_rate,
                total_heat_rate,
                minimum_heat_rate,
                heat_rate_smoothing,
            )
        elif total_heat_rate < minimum_heat_rate:
            total_heat_rate = minimum_heat_rate

    out[b, m, 0] = activation_rate
    out[b, m, 1] = maintenance_rate
    out[b, m, 2] = shortening_rate
    out[b, m, 3] = mechanical_work_rate
    out[b, m, 4] = total_heat_rate + mechanical_work_rate


class MuscleMetabolicsBhargava2004:
    """Estimate Bhargava (2004) muscle metabolic cost.

    All model muscles are included. Per-muscle physiological parameters can be
    overridden with ``muscle_parameters``; omitted muscles use OpenSim's
    defaults. Rigid-tendon De Groote-Fregly fiber kinematics and forces are
    shared with :class:`MuscleForces`. The metabolic-rate equations match
    OpenSim; end-to-end values also depend on using equivalent muscle and
    tendon-compliance settings in OpenSim and Newton. Newton evaluates a
    rigid-tendon De Groote-Fregly muscle state, so exact comparison requires
    ``DeGrooteFregly2016Muscle`` and ignored tendon compliance in OpenSim.
    Tanh smoothing applies to OpenSim's conditional branches, but the
    maintenance length curve remains piecewise linear.

    Args:
        model: Parsed OpenSim model.
        device: Warp device (defaults to CPU, matching the rest of the port).
        muscle_parameters: Optional parameters keyed by muscle name.
        allow_approximate_muscle_models: Include non-De Groote-Fregly muscles
            by evaluating them with Newton's rigid-tendon De Groote-Fregly
            mechanics. The default rejects these model types to prevent an
            accidental claim of end-to-end OpenSim parity.
        body_mass: Whole-body mass [kg]. ``None`` sums all model body masses.
        basal_coefficient: Basal metabolic coefficient
            [W/kg^``basal_exponent``]. OpenSim's default 1.2 with exponent 1
            gives a standing basal rate of 1.2 W/kg.
        basal_exponent: Dimensionless exponent applied to body mass for basal power.
        effort_scaling_factor: Dimensionless non-negative scale applied to
            activations, excitations, and active fiber force. Scaled values are
            not clamped back to [0, 1], matching OpenSim.
        use_force_dependent_shortening_constant: Use Bhargava's force-dependent
            shortening proportional constant instead of OpenSim's default
            Anderson (1999) constant.
        include_negative_mechanical_work: Include negative work during fiber
            lengthening.
        forbid_negative_total_power: Increase shortening heat as needed to
            clamp each muscle's total metabolic power non-negative. With tanh
            smoothing this is an asymptotic approximation to the clamp.
        enforce_minimum_heat_rate: Apply the 1 W/kg muscle heat-rate floor.
            With tanh smoothing this is an asymptotic approximation to the floor.
        use_smoothing: Use OpenSim's tanh approximation at conditional transitions.
        velocity_smoothing: Tanh steepness for velocity conditionals [s/m].
        power_smoothing: Tanh steepness for the non-negative-power constraint [1/W].
        heat_rate_smoothing: Tanh steepness for the minimum-heat constraint [1/W].
    """

    def __init__(
        self,
        model: OsimModel,
        device=None,
        muscle_parameters: dict[str, MuscleMetabolicsBhargava2004Parameters] | None = None,
        allow_approximate_muscle_models: bool = False,
        body_mass: float | None = None,
        basal_coefficient: float = 1.2,
        basal_exponent: float = 1.0,
        effort_scaling_factor: float = 1.0,
        use_force_dependent_shortening_constant: bool = False,
        include_negative_mechanical_work: bool = True,
        forbid_negative_total_power: bool = True,
        enforce_minimum_heat_rate: bool = True,
        use_smoothing: bool = False,
        velocity_smoothing: float = 10.0,
        power_smoothing: float = 10.0,
        heat_rate_smoothing: float = 10.0,
    ):
        self.model = model
        self.muscles = MuscleForces(model, device=device)
        self.device = self.muscles.device
        self.muscle_names = list(self.muscles.muscle_names)
        if not self.muscle_names:
            raise ValueError("model must contain at least one muscle")
        self.allow_approximate_muscle_models = bool(allow_approximate_muscle_models)
        unsupported = [
            muscle.name
            for muscle in model.muscles
            if muscle.type != "DeGrooteFregly2016Muscle" or bool(muscle.params.get("ignore_passive_fiber_force", False))
        ]
        if unsupported and not self.allow_approximate_muscle_models:
            raise ValueError(
                "MuscleMetabolicsBhargava2004 requires DeGrooteFregly2016Muscle "
                "with passive fiber force enabled; pass "
                "allow_approximate_muscle_models=True to use Newton's rigid-tendon "
                f"De Groote-Fregly approximation for: {unsupported}"
            )
        self._validate_model_parameters()

        provided = {} if muscle_parameters is None else dict(muscle_parameters)
        unknown = set(provided).difference(self.muscle_names)
        if unknown:
            raise ValueError(f"unknown muscles in muscle_parameters: {sorted(unknown)}")
        if any(not isinstance(value, MuscleMetabolicsBhargava2004Parameters) for value in provided.values()):
            raise TypeError("muscle_parameters values must be MuscleMetabolicsBhargava2004Parameters")

        if body_mass is None:
            body_mass = sum(float(body.mass) for body in model.bodies)
        self.body_mass = self._positive_finite("body_mass", body_mass)
        self.basal_coefficient = self._nonnegative_finite("basal_coefficient", basal_coefficient)
        self.basal_exponent = self._finite("basal_exponent", basal_exponent)
        self.effort_scaling_factor = self._nonnegative_finite("effort_scaling_factor", effort_scaling_factor)
        self.use_force_dependent_shortening_constant = bool(use_force_dependent_shortening_constant)
        self.include_negative_mechanical_work = bool(include_negative_mechanical_work)
        self.forbid_negative_total_power = bool(forbid_negative_total_power)
        self.enforce_minimum_heat_rate = bool(enforce_minimum_heat_rate)
        self.use_smoothing = bool(use_smoothing)
        self.velocity_smoothing = self._positive_finite("velocity_smoothing", velocity_smoothing)
        self.power_smoothing = self._positive_finite("power_smoothing", power_smoothing)
        self.heat_rate_smoothing = self._positive_finite("heat_rate_smoothing", heat_rate_smoothing)

        masses = []
        slow_ratios = []
        activation_slow = []
        activation_fast = []
        maintenance_slow = []
        maintenance_fast = []
        self.muscle_parameters: dict[str, MuscleMetabolicsBhargava2004Parameters] = {}
        for index, name in enumerate(self.muscle_names):
            params = provided.get(name, MuscleMetabolicsBhargava2004Parameters())
            self._validate_muscle_parameters(name, params)
            mass = params.muscle_mass
            if mass is None:
                mass = self.muscles._fmax[index] / params.specific_tension * params.density * self.muscles._l_opt[index]
            masses.append(self._positive_finite(f"muscle mass for {name!r}", mass))
            slow_ratios.append(params.slow_twitch_ratio)
            activation_slow.append(params.activation_constant_slow_twitch)
            activation_fast.append(params.activation_constant_fast_twitch)
            maintenance_slow.append(params.maintenance_constant_slow_twitch)
            maintenance_fast.append(params.maintenance_constant_fast_twitch)
            self.muscle_parameters[name] = params

        self.muscle_mass = np.asarray(masses, dtype=np.float64)
        self.d_muscle_mass = wp.array(self.muscle_mass, dtype=_f32, device=self.device)
        self.d_slow_twitch_ratio = wp.array(slow_ratios, dtype=_f32, device=self.device)
        self.d_activation_constant_slow = wp.array(activation_slow, dtype=_f32, device=self.device)
        self.d_activation_constant_fast = wp.array(activation_fast, dtype=_f32, device=self.device)
        self.d_maintenance_constant_slow = wp.array(maintenance_slow, dtype=_f32, device=self.device)
        self.d_maintenance_constant_fast = wp.array(maintenance_fast, dtype=_f32, device=self.device)

    @property
    def num_muscles(self) -> int:
        """Number of muscles included in the estimate."""
        return len(self.muscle_names)

    @property
    def basal_rate(self) -> float:
        """Whole-body basal metabolic rate [W]."""
        return self.basal_coefficient * self.body_mass**self.basal_exponent

    def compute(
        self,
        activations: np.ndarray,
        excitations: np.ndarray,
        coordinates: np.ndarray,
        speeds: np.ndarray | None = None,
    ) -> MuscleMetabolicsBhargava2004Result:
        """Compute metabolic power for one state or a batch of states.

        Args:
            activations: Muscle activations in [0, 1], shape
                ``[num_frames, num_muscles]`` or ``[num_muscles]``.
            excitations: Neural excitations in [0, 1], with the same accepted
                shapes as ``activations``. A single frame broadcasts over the batch.
            coordinates: Coordinate values [m or rad], shape
                ``[num_frames, num_coordinates]`` or ``[num_coordinates]``.
            speeds: Coordinate speeds [m/s or rad/s], matching ``coordinates``.
                ``None`` treats the states as isometric.

        Returns:
            Metabolic power components and totals for every input frame.
        """
        d_activation, d_excitation, d_coordinates, d_speeds = self._device_inputs(
            activations, excitations, coordinates, speeds
        )
        output = self._compute_device(d_activation, d_excitation, d_coordinates, d_speeds)
        values = output.numpy().astype(np.float64)
        muscle_rate = values[:, :, 4]
        total_rate = np.sum(muscle_rate, axis=1) + self.basal_rate
        return MuscleMetabolicsBhargava2004Result(
            muscle_names=list(self.muscle_names),
            muscle_mass=self.muscle_mass.copy(),
            activation_rate=values[:, :, 0],
            maintenance_rate=values[:, :, 1],
            shortening_rate=values[:, :, 2],
            mechanical_work_rate=values[:, :, 3],
            muscle_rate=muscle_rate,
            basal_rate=self.basal_rate,
            total_rate=total_rate,
            body_mass=self.body_mass,
        )

    def _compute_device(
        self,
        activations: wp.array2d[_f32],
        excitations: wp.array2d[_f32],
        coordinates: wp.array2d[_f64],
        speeds: wp.array2d[_f64] | None,
    ) -> wp.array3d[_f32]:
        """Compute packed metabolic rates without leaving the device."""
        batch = coordinates.shape[0]
        lengths = self.muscles.paths._lengths_qwp(coordinates)
        if speeds is None:
            velocities = wp.zeros((batch, self.num_muscles), dtype=_f64, device=self.device)
        else:
            moment_arms = self.muscles.paths._moment_arms_device(coordinates, 1.0e-5)
            velocities = self.muscles.paths._velocities_qwp(coordinates, speeds, 1.0e-5, moment_arms)
        output = wp.empty((batch, self.num_muscles, 5), dtype=_f32, device=self.device)
        wp.launch(
            metabolic_rate_kernel,
            dim=(batch, self.num_muscles),
            inputs=[
                activations,
                excitations,
                lengths,
                velocities,
                self.muscles.d_fmax,
                self.muscles.d_l_opt,
                self.muscles.d_lt_slack,
                self.muscles.d_vmax,
                self.muscles.d_cos_penn,
                self.d_muscle_mass,
                self.d_slow_twitch_ratio,
                self.d_activation_constant_slow,
                self.d_activation_constant_fast,
                self.d_maintenance_constant_slow,
                self.d_maintenance_constant_fast,
                _f32(self.effort_scaling_factor),
                self.use_force_dependent_shortening_constant,
                self.include_negative_mechanical_work,
                self.forbid_negative_total_power,
                self.enforce_minimum_heat_rate,
                self.use_smoothing,
                _f32(self.velocity_smoothing),
                _f32(self.power_smoothing),
                _f32(self.heat_rate_smoothing),
                output,
            ],
            device=self.device,
        )
        return output

    def _device_inputs(self, activations, excitations, coordinates, speeds):
        """Validate and upload one trajectory batch."""
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim == 1:
            coordinates = coordinates[None, :]
        expected_coordinates = len(self.muscles.coordinate_names)
        if coordinates.ndim != 2 or coordinates.shape[1] != expected_coordinates:
            raise ValueError(f"coordinates must have shape [num_frames, {expected_coordinates}]")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be finite")
        coordinates = np.ascontiguousarray(coordinates)
        batch = coordinates.shape[0]
        if batch == 0:
            raise ValueError("coordinates must contain at least one frame")

        activation_values = self._muscle_values("activations", activations, batch)
        excitation_values = self._muscle_values("excitations", excitations, batch)
        speed_values = None
        if speeds is not None:
            speed_values = np.asarray(speeds, dtype=np.float64)
            if speed_values.ndim == 1:
                speed_values = speed_values[None, :]
            if speed_values.shape == (1, expected_coordinates) and batch > 1:
                speed_values = np.repeat(speed_values, batch, axis=0)
            if speed_values.shape != coordinates.shape:
                raise ValueError(f"speeds must have shape {coordinates.shape}")
            if not np.all(np.isfinite(speed_values)):
                raise ValueError("speeds must be finite")
            speed_values = np.ascontiguousarray(speed_values)

        return (
            wp.array(activation_values, dtype=_f32, device=self.device),
            wp.array(excitation_values, dtype=_f32, device=self.device),
            wp.array(coordinates, dtype=_f64, device=self.device),
            None if speed_values is None else wp.array(speed_values, dtype=_f64, device=self.device),
        )

    def _muscle_values(self, name: str, values, batch: int) -> np.ndarray:
        """Normalize and validate one activation-like input."""
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape == (1, self.num_muscles) and batch > 1:
            values = np.repeat(values, batch, axis=0)
        if values.shape != (batch, self.num_muscles):
            raise ValueError(f"{name} must have shape [{batch}, {self.num_muscles}]")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{name} must be in [0, 1]")
        return np.ascontiguousarray(values)

    def _validate_model_parameters(self) -> None:
        """Validate model quantities required by rigid-tendon muscle mechanics."""
        required = {
            "max_isometric_force": self.muscles._fmax,
            "optimal_fiber_length": self.muscles._l_opt,
            "tendon_slack_length": self.muscles._lt_slack,
            "max_contraction_velocity": self.muscles._vmax,
            "cosine of pennation angle": self.muscles._cos_penn,
        }
        for name, values in required.items():
            invalid = ~np.isfinite(values) | (values <= 0.0)
            if np.any(invalid):
                muscles = [self.muscle_names[index] for index in np.flatnonzero(invalid)]
                raise ValueError(f"{name} must be finite and positive for muscles: {muscles}")

    @classmethod
    def _validate_muscle_parameters(cls, name: str, params: MuscleMetabolicsBhargava2004Parameters) -> None:
        """Validate one muscle's physiological parameters."""
        if not np.isfinite(params.slow_twitch_ratio) or not 0.0 <= params.slow_twitch_ratio <= 1.0:
            raise ValueError(f"slow_twitch_ratio for {name!r} must be in [0, 1]")
        cls._positive_finite(f"specific_tension for {name!r}", params.specific_tension)
        cls._positive_finite(f"density for {name!r}", params.density)
        if params.muscle_mass is not None:
            cls._positive_finite(f"muscle_mass for {name!r}", params.muscle_mass)
        cls._nonnegative_finite(f"activation_constant_slow_twitch for {name!r}", params.activation_constant_slow_twitch)
        cls._nonnegative_finite(f"activation_constant_fast_twitch for {name!r}", params.activation_constant_fast_twitch)
        cls._nonnegative_finite(
            f"maintenance_constant_slow_twitch for {name!r}", params.maintenance_constant_slow_twitch
        )
        cls._nonnegative_finite(
            f"maintenance_constant_fast_twitch for {name!r}", params.maintenance_constant_fast_twitch
        )

    @staticmethod
    def _finite(name: str, value: float) -> float:
        """Return a finite float or raise ``ValueError``."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive_finite(cls, name: str, value: float) -> float:
        """Return a positive finite float or raise ``ValueError``."""
        value = cls._finite(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _nonnegative_finite(cls, name: str, value: float) -> float:
        """Return a non-negative finite float or raise ``ValueError``."""
        value = cls._finite(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value
