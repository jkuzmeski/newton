# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenSim controllers: :class:`ControlSet` and :class:`PrescribedController`.

A :class:`ControlSet` stores named, time-sampled control signals (OpenSim's
``ControlSet`` / ``ControlLinear``) and evaluates them by linear interpolation. A
:class:`PrescribedController` applies a control set (or per-actuator callables) to
a fixed ordered list of actuators, producing the control vector consumed by a
forward simulation (e.g. muscle excitations for
:func:`~newton.opensim.simulate_muscle_driven`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .mocap import Storage, read_storage, write_storage


@dataclass
class ControlSet:
    """A set of named, time-sampled control signals evaluated by interpolation.

    Attributes:
        labels: Control (actuator) names in column order.
        times: Sample times [s], shape ``[num_samples]``.
        data: Control values, shape ``[num_samples, num_controls]``.
    """

    labels: list[str]
    times: np.ndarray
    data: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float).ravel()
        self.data = np.atleast_2d(np.asarray(self.data, dtype=float))
        if self.data.shape[0] != self.times.shape[0] and self.data.shape[1] == self.times.shape[0]:
            self.data = self.data.T
        self._index = {name: i for i, name in enumerate(self.labels)}

    @classmethod
    def from_storage(cls, storage: Storage | str | os.PathLike) -> ControlSet:
        """Build a control set from a controls :class:`~newton.opensim.Storage` or ``.sto`` path."""
        if not isinstance(storage, Storage):
            storage = read_storage(storage)
        return cls(
            labels=list(storage.labels), times=np.asarray(storage.times, float), data=np.asarray(storage.data, float)
        )

    @classmethod
    def from_constant(cls, labels: list[str], values, time_range: tuple[float, float] = (0.0, 1.0)) -> ControlSet:
        """Build a control set holding constant ``values`` over ``time_range``."""
        values = np.asarray(values, dtype=float).ravel()
        times = np.asarray(time_range, dtype=float)
        data = np.tile(values[None, :], (len(times), 1))
        return cls(labels=list(labels), times=times, data=data)

    def value(self, name: str, t) -> np.ndarray:
        """Interpolate control ``name`` at time(s) ``t`` (0 outside the sample span)."""
        if name not in self._index:
            return np.zeros_like(np.atleast_1d(np.asarray(t, float)))
        col = self.data[:, self._index[name]]
        return np.interp(np.atleast_1d(np.asarray(t, float)), self.times, col)

    def sample(self, t) -> np.ndarray:
        """Interpolate every control at time(s) ``t``.

        Returns an array of shape ``[num_controls]`` for scalar ``t`` or
        ``[len(t), num_controls]`` for array ``t``.
        """
        scalar = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, float))
        out = np.empty((tt.shape[0], len(self.labels)))
        for j in range(len(self.labels)):
            out[:, j] = np.interp(tt, self.times, self.data[:, j])
        return out[0] if scalar else out

    def to_storage(self, name: str = "ControlSet") -> Storage:
        """Return the control set as a :class:`~newton.opensim.Storage`."""
        return Storage(
            times=self.times.copy(), labels=list(self.labels), data=self.data.copy(), in_degrees=False, name=name
        )

    def write_sto(self, path: str | os.PathLike, name: str = "ControlSet") -> None:
        """Write the control set to an OpenSim ``.sto`` file."""
        write_storage(path, self.times, list(self.labels), self.data, name=name, in_degrees=False)


class PrescribedController:
    """Apply prescribed control signals to an ordered list of actuators.

    Args:
        actuator_names: Actuators to emit controls for, in the desired output order.
        controls: A :class:`ControlSet`, a controls :class:`Storage`/``.sto`` path,
            a mapping ``name -> value`` (constant) or ``name -> callable(t)``, or a
            single ``callable(t) -> vector`` aligned to ``actuator_names``.
        default: Control value for actuators not present in ``controls``.
    """

    def __init__(
        self,
        actuator_names: list[str],
        controls: ControlSet | Storage | str | os.PathLike | dict | Callable,
        default: float = 0.0,
    ):
        self.actuator_names = list(actuator_names)
        self.default = float(default)
        self._callable = None
        self._funcs: dict[str, Callable] | None = None
        self._set: ControlSet | None = None
        if isinstance(controls, ControlSet):
            self._set = controls
        elif isinstance(controls, Storage) or isinstance(controls, (str, os.PathLike)):
            self._set = ControlSet.from_storage(controls)
        elif isinstance(controls, dict):
            self._funcs = {}
            for name, val in controls.items():
                self._funcs[name] = val if callable(val) else (lambda t, v=float(val): v)
        elif callable(controls):
            self._callable = controls
        else:
            raise TypeError("controls must be a ControlSet, Storage, path, dict, or callable")

    def controls_at(self, t: float) -> np.ndarray:
        """Return the control vector at time ``t`` aligned to ``actuator_names``."""
        if self._callable is not None:
            vec = np.asarray(self._callable(t), dtype=float).ravel()
            if vec.shape[0] != len(self.actuator_names):
                raise ValueError(f"callable returned {vec.shape[0]} controls, expected {len(self.actuator_names)}")
            return vec
        out = np.full(len(self.actuator_names), self.default, dtype=float)
        for i, name in enumerate(self.actuator_names):
            if self._set is not None:
                if name in self._set._index:
                    out[i] = float(self._set.value(name, t)[0])
            elif self._funcs is not None and name in self._funcs:
                out[i] = float(self._funcs[name](t))
        return out

    def __call__(self, t: float) -> np.ndarray:
        return self.controls_at(t)
