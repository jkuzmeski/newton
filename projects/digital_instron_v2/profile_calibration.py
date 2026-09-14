# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile resident forward calibration against an explicitly saved baseline.

Run from the repository root; see --help. The benchmark reads existing training
traces and never exports or promotes a fitted material. Optional short optimizer
runs are diagnostic only and retain the same law, objective and SciPy bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_shoe.artifact import load_artifact

from . import core
from .geometry import build_column_grid, load_mesh
from .workflow import prepare_trials


def _load_baseline(path: Path):
    """Load the named pre-change module without modifying the live source tree."""
    name = "projects.digital_instron_v2._profile_calibration_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Transfers:
    """Count explicit launch calls and completed device-to-host array payload."""

    def __init__(self):
        self.calls = Counter()
        self.shapes = Counter()
        self.bytes = 0
        self.original_numpy = wp.array.numpy
        self.original_launch = wp.launch
        self.original_capture = wp.capture_launch

    def numpy(self, array, *args, **kwargs):
        value = self.original_numpy(array, *args, **kwargs)
        self.calls["numpy"] += 1
        self.shapes[str(tuple(array.shape))] += 1
        if array.device.is_cuda:
            self.bytes += value.nbytes
        return value

    def launch(self, *args, **kwargs):
        self.calls["launch"] += 1
        return self.original_launch(*args, **kwargs)

    def capture(self, *args, **kwargs):
        self.calls["graph_launch"] += 1
        return self.original_capture(*args, **kwargs)

    def run(self, function):
        """Measure one completed call, including its final force readback."""

        def read(array, *args, **kwargs):
            return self.numpy(array, *args, **kwargs)

        with (
            patch.object(wp.array, "numpy", read),
            patch.object(wp, "launch", self.launch),
            patch.object(wp, "capture_launch", self.capture),
        ):
            start = time.perf_counter()
            result = function()
            elapsed = time.perf_counter() - start
        return result, {
            "wall_s": elapsed,
            "calls": dict(self.calls),
            "numpy_shapes": dict(self.shapes),
            "device_to_host_bytes": self.bytes,
        }


def _clear(module):
    """Start a sequence from the same physical cold state in either implementation."""
    module._SURROUND_WARM_START.clear()
    module.SURROUND_CONVERGENCE.clear()


def _material(module, values):
    """Retain six physical parameter meanings across independently loaded modules."""
    return module.Material(**values)


def main():
    """Compare force parity, warm timing, transfers and optional optimizer work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("DigitalInstron/manifest_v2.json"))
    parser.add_argument("--artifact", type=Path, default=Path("outputs/impedance_instron/inputs/digital_shoe.json"))
    parser.add_argument("--baseline-core", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/impedance_instron/gpu_forward_calibration/profile")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fit-evaluations", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1 or args.fit_evaluations < 0:
        parser.error("repeats must be positive and fit-evaluations nonnegative")
    wp.init()
    wp.set_device(args.device)
    baseline = _load_baseline(args.baseline_core.resolve())
    manifest = args.manifest.resolve()
    config = json.loads(manifest.read_text())
    base, windows = manifest.parent, config["cycle_windows"]
    paths = {
        source["name"]: base / windows["output_dir"] / f"{source['split_prefix']}_{windows['train']['suffix']}.csv"
        for source in config["trials"]
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepare the fixed training split before profiling: {missing}")
    mesh = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(mesh, config["grid"]["coarse_spacing_m"])
    trials, _, _ = prepare_trials(base, config, grid, mesh, trace_paths=paths)
    material = load_artifact(args.artifact).material
    values = {field.name: getattr(material, field.name) for field in fields(core.Material)}
    candidates = [values.copy()]
    for name, factor in (
        ("instantaneous_shear_modulus_pa", 1.001),
        ("instantaneous_shear_modulus_2_pa", 0.999),
        ("equilibrium_fraction", 1.0005),
        ("maxwell_relaxation_time_s", 1.001),
    ):
        candidate = values.copy()
        candidate[name] *= factor
        candidates.append(candidate)
    args.output.mkdir(parents=True, exist_ok=True)
    records = {
        "device": str(wp.get_device()),
        "material": values,
        "trials": {},
        "fits": {},
        "baseline_source_sha256": hashlib.sha256(args.baseline_core.read_bytes()).hexdigest(),
        "candidate_count": len(candidates),
        "repeats": args.repeats,
        "scope": "execution-only benchmark; short fit candidates are not promoted",
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")

    for trial in trials:
        entry = {"frames": len(trial.dt_s), "columns": len(trial.surround.slack_m), "paths": {}}
        curves = {}
        for label, module in (("baseline", baseline), ("resident", core)):
            _clear(module)
            start = time.perf_counter()
            module.predict(trial, _material(module, values))
            setup_s = time.perf_counter() - start
            sequences = []
            all_curves = []
            for _repeat in range(args.repeats):
                _clear(module)
                # Match physical warm state; both methods start this query sequence at the baseline material.
                module.predict(trial, _material(module, values))
                sequence = []
                for candidate in candidates:
                    meter = _Transfers()
                    force, measured = meter.run(
                        lambda m=candidate, mod=module, tr=trial: mod.predict(tr, _material(mod, m))
                    )
                    measured["solver"] = module.SURROUND_CONVERGENCE[trial.name]
                    sequence.append(measured)
                    all_curves.append(force.copy())
                sequences.append(sequence)
            curves[label] = np.stack(all_curves)
            totals = [sum(item["wall_s"] for item in sequence) for sequence in sequences]
            entry["paths"][label] = {
                "first_call_s": setup_s,
                "sequence_s": totals,
                "median_sequence_s": float(np.median(totals)),
                "measurements": sequences,
            }
        scale = max(float(np.max(np.abs(curves["baseline"]))), 1.0)
        error = np.abs(curves["resident"] - curves["baseline"])
        entry["max_force_error_n"] = float(np.max(error))
        entry["max_force_error_fraction"] = float(np.max(error) / scale)
        entry["rms_force_error_fraction"] = float(np.sqrt(np.mean(error**2)) / scale)
        entry["speedup"] = (
            entry["paths"]["baseline"]["median_sequence_s"] / entry["paths"]["resident"]["median_sequence_s"]
        )
        records["trials"][trial.name] = entry
        np.savez_compressed(args.output / f"{trial.name}_curves.npz", **curves)
        save()
        print(
            json.dumps({"trial": trial.name, "speedup": entry["speedup"], "error": entry["max_force_error_fraction"]}),
            flush=True,
        )
    if args.fit_evaluations:
        for label, module in (("baseline", baseline), ("resident", core)):
            _clear(module)
            count = 0
            original = module._surround_force

            def counted(*positional, original=original, **keywords):
                nonlocal count
                count += 1
                return original(*positional, **keywords)

            history = []
            with patch.object(module, "_surround_force", counted):
                start = time.perf_counter()
                fitted = module.fit_material(trials, _material(module, values), args.fit_evaluations, history=history)
                seconds = time.perf_counter() - start
            # Qualification uses the original six-parameter residual definition, not history[-1].
            residual = np.concatenate([module._trial_residual(t, fitted) for t in trials])
            records["fits"][label] = {
                "wall_s": seconds,
                "forward_trial_evaluations": count,
                "material": fitted.__dict__,
                "final_mean_square_residual": float(np.mean(residual**2)),
                "requested_max_nfev": args.fit_evaluations,
                "history_entries": len(history),
            }
            save()
            print(json.dumps({"fit": label, **records["fits"][label]}), flush=True)
    save()


if __name__ == "__main__":
    main()
