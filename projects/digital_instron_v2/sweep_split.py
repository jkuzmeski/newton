# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split and validate the October 2026 Digital Instron foam sweeps."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cycle_windows import CycleWindowTrace, _interp_cycle, write_cycle_window_trace
from .frame_qc import read_numeric_csv

STANCE_STEPS = {160: 3, 200: 6, 240: 9, 280: 12, 320: 15}
LOOP_COLUMNS = {
    160: "160ms Loop(1)",
    200: "200ms Loop(2)",
    240: "240ms Loop(3)",
    280: "280ms Loop(4)",
    320: "320ms Loop(5)",
}
TARGETS_N = tuple(range(800, 2801, 400))
TRAIN_CYCLES = tuple(range(90, 99))
VALIDATION_CYCLES = (99, 100)
PHASE_COUNT = 501


def _condition_rows(columns: dict[str, np.ndarray], stance_ms: int, target_n: int) -> np.ndarray:
    """Select logged cycles for one condition using both counters."""
    loop_index = (target_n - 800) // 400 + 1
    loop_column = LOOP_COLUMNS[stance_ms]
    expected_step = STANCE_STEPS[stance_ms]
    required = ("Step", "Elapsed Cycles", "Total Cycles", loop_column)
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"tracking CSV is missing counter columns: {', '.join(missing)}")
    active = np.isclose(columns[loop_column], loop_index)
    logged = (columns["Elapsed Cycles"] >= 90) & (columns["Elapsed Cycles"] <= 100)
    if np.any(active & logged & ~np.isclose(columns["Step"], expected_step)):
        raise ValueError(
            f"Counter mismatch for {stance_ms} ms / {target_n} N: "
            f"{loop_column}={loop_index} is active outside Step {expected_step} in cycles 90-100"
        )
    cycle_window = np.isclose(columns["Elapsed Cycles"], 1) | logged
    mask = np.isclose(columns["Step"], expected_step) & active & cycle_window
    if not np.any(mask):
        raise ValueError(f"No logged cycles for {stance_ms} ms / {target_n} N")
    return mask


def _cycle_data(
    columns: dict[str, np.ndarray],
    rows: np.ndarray,
    cycles: tuple[int, ...],
    *,
    position_sign: float,
    force_sign: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Phase-average requested elapsed cycles into the shared trace schema."""
    time = columns["Cycle Elapsed Time (s)"]
    elapsed = columns["Elapsed Cycles"]
    position = columns["Position (mm)"]
    force = columns["Force (N)"]
    phase_grid = np.linspace(0.0, 1.0, PHASE_COUNT)
    position_rows, force_rows, raw_force_rows, durations = [], [], [], []
    for cycle in cycles:
        mask = rows & np.isclose(elapsed, cycle)
        if np.count_nonzero(mask) < 3:
            raise ValueError(f"Elapsed cycle {cycle} has fewer than three samples")
        local_time = time[mask]
        duration = float(np.max(local_time) - np.min(local_time))
        if not np.isfinite(duration) or not np.isclose(duration, 0.5, atol=0.01):
            raise ValueError(f"Elapsed cycle {cycle} period is {duration:.6g} s, expected 0.5 s")
        phase = (local_time - np.min(local_time)) / duration
        position_rows.append(_interp_cycle(phase_grid, phase, position[mask]))
        force_rows.append(_interp_cycle(phase_grid, phase, force_sign * force[mask]))
        raw_force_rows.append(_interp_cycle(phase_grid, phase, force[mask]))
        durations.append(duration)
    mean_position = np.mean(position_rows, axis=0)
    mean_force = np.mean(force_rows, axis=0)
    mean_raw_force = np.mean(raw_force_rows, axis=0)
    signed_position = position_sign * mean_position
    displacement_m = (signed_position - np.min(signed_position)) * 0.001
    time_grid = phase_grid * float(np.mean(durations))
    data = {
        "phase": phase_grid,
        "time_s": time_grid,
        "displacement_m": displacement_m,
        "displacement_mm": displacement_m * 1000.0,
        "force_n": mean_force,
        "position_mm_raw": mean_position,
        "force_n_raw": mean_raw_force,
        "velocity_m_s": np.gradient(displacement_m, time_grid),
        "cycle_energy_j": np.zeros_like(phase_grid),
    }
    return data, {
        "cycles": list(cycles),
        "phase_count": PHASE_COUNT,
        "displacement_zero_policy": "top_of_stroke",
        "force_zero_policy": "sign_flip_only",
    }


def _measure_condition(
    columns: dict[str, np.ndarray], rows: np.ndarray, stance_ms: int, target_n: int
) -> dict[str, float]:
    """Measure signal-level force, period, preload, stroke, and loaded duration."""
    peaks, loaded, cycle_periods, tops, preloads = [], [], [], [], []
    for cycle in range(90, 101):
        mask = rows & np.isclose(columns["Elapsed Cycles"], cycle)
        if np.count_nonzero(mask) < 3:
            raise ValueError(f"Missing cycle {cycle} for {stance_ms} ms / {target_n} N")
        t = columns["Cycle Elapsed Time (s)"][mask]
        raw_force = columns["Force (N)"][mask]
        comp_force = -raw_force
        peak = float(np.max(comp_force))
        if peak <= 0:
            raise ValueError(f"Non-compressive force for {stance_ms} ms / {target_n} N")
        peaks.append(peak)
        above = np.flatnonzero(comp_force >= 0.1 * peak)
        loaded.append(float(t[above[-1]] - t[above[0]]) if len(above) else 0.0)
        cycle_periods.append(float(np.max(t) - np.min(t)))
        pos = columns["Position (mm)"][mask]
        tops.append(float(np.max(pos)))
        preloads.append(float(np.median(comp_force[pos >= np.quantile(pos, 0.95)])))
    measured_peak = float(np.median(peaks))
    error = abs(measured_peak - target_n) / target_n
    if error > 0.03:
        raise ValueError(
            f"Signal/counter mismatch for {stance_ms} ms / {target_n} N: "
            f"median compressive peak is {measured_peak:.1f} N ({error:.1%} error)"
        )
    median_period = float(np.median(cycle_periods))
    if not np.isclose(median_period, 0.5, atol=0.01):
        raise ValueError(f"{stance_ms} ms / {target_n} N period is {median_period:.4f} s, expected 0.5 s")
    first_mask = rows & np.isclose(columns["Elapsed Cycles"], 1)
    if np.count_nonzero(first_mask) < 3:
        raise ValueError(f"Missing cycle 1 for {stance_ms} ms / {target_n} N")
    first_peak = float(np.max(-columns["Force (N)"][first_mask]))
    first_top = float(np.max(columns["Position (mm)"][first_mask]))
    return {
        "measured_peak_n": measured_peak,
        "preload_n": float(np.median(preloads)),
        "top_of_stroke_position_mm": float(np.median(tops)),
        "loaded_duration_s": float(np.median(loaded)),
        "cycle_peak_scatter_n": float(np.std(peaks, ddof=1)),
        "cycle_peak_scatter_fraction": float(np.std(peaks, ddof=1) / np.mean(peaks)),
        "period_s": median_period,
        "cycle_peak_min_n": float(np.min(peaks)),
        "cycle_peak_max_n": float(np.max(peaks)),
        "cycle_top_position_min_mm": float(np.min(tops)),
        "cycle_top_position_max_mm": float(np.max(tops)),
        "cycle1_peak_n": first_peak,
        "cycle1_to_late_peak_difference_n": measured_peak - first_peak,
        "cycle1_top_position_mm": first_top,
        "cycle1_to_late_top_position_difference_mm": float(np.median(tops) - first_top),
    }


def _source_shoe(path: Path) -> str:
    name = path.parent.name.lower()
    if "dev3elite" in name:
        return "Dev3Elite"
    if "_fr3_" in name:
        return "FR3"
    raise ValueError(f"Cannot infer shoe label from {path}")


def _waveform(
    columns: dict[str, np.ndarray], rows: np.ndarray, *, position_sign: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train-cycle mean phase, compression displacement, and force."""
    data, _ = _cycle_data(columns, rows, TRAIN_CYCLES, position_sign=position_sign, force_sign=-1.0)
    return data["phase"], data["displacement_m"], data["force_n"]


def _repeat_metrics(
    original: tuple[np.ndarray, np.ndarray, np.ndarray],
    repeat: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_n: int,
) -> dict[str, float]:
    """Compare repeat waveform to its original, rejecting non-matching profiles."""
    phase, displacement, force = original
    repeat_phase, repeat_displacement, repeat_force = repeat
    if not np.array_equal(phase, repeat_phase):
        raise ValueError("Repeat waveform phase grids differ")
    peak_error = abs(float(np.max(repeat_force)) - float(np.max(force)))
    max_point_error = float(np.max(np.abs(repeat_force - force)))
    if max_point_error > 0.03 * float(np.max(force)):
        raise ValueError(
            f"320 ms programmed profile does not match the 200 ms condition at {target_n} N: "
            f"pointwise force difference {max_point_error:.1f} N exceeds 3% of target"
        )

    def energy(x: np.ndarray, y: np.ndarray) -> float:
        return float(abs(np.trapezoid(y, x)))

    original_energy = energy(displacement, force)
    repeat_energy = energy(repeat_displacement, repeat_force)
    return {
        "repeat_peak_difference_n": peak_error,
        "repeat_force_rmse_n": float(np.sqrt(np.mean((repeat_force - force) ** 2))),
        "repeat_force_max_abs_difference_n": max_point_error,
        "original_loop_energy_j": original_energy,
        "repeat_loop_energy_j": repeat_energy,
        "repeat_loop_energy_difference_j": repeat_energy - original_energy,
    }


def process_sweep(path: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    """Validate one raw sweep and write train/validation traces plus an index."""
    path = Path(path)
    columns = read_numeric_csv(path)
    for name in ("Total Time (s)", "Cycle Elapsed Time (s)", "Elapsed Cycles", "Step", "Position (mm)", "Force (N)"):
        if name not in columns:
            raise ValueError(f"{path} is missing required column {name!r}")
    shoe = _source_shoe(path)
    out_dir = Path(output_root) if output_root else path.parents[1] / "processed" / shoe
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compression makes force positive and displacement increase as raw position falls.
    force_sign = -1.0
    corr = float(np.corrcoef(columns["Position (mm)"], -columns["Force (N)"])[0, 1])
    position_sign = -1.0 if corr < 0.0 else 1.0
    conditions: list[dict[str, Any]] = []
    original_waveforms: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for stance_ms in (160, 200, 240, 280):
        for target_n in TARGETS_N:
            rows = _condition_rows(columns, stance_ms, target_n)
            metrics = _measure_condition(columns, rows, stance_ms, target_n)
            for split, cycles in (("train", TRAIN_CYCLES), ("validate", VALIDATION_CYCLES)):
                data, provenance = _cycle_data(
                    columns, rows, cycles, position_sign=position_sign, force_sign=force_sign
                )
                stem = f"stance{stance_ms}ms_{target_n}N_{split}"
                write_cycle_window_trace(
                    out_dir / f"{stem}.csv",
                    CycleWindowTrace(
                        data,
                        {
                            "source_csv": str(path),
                            "cycle_column": "Elapsed Cycles",
                            **provenance,
                            "condition": {"stance_ms": stance_ms, "target_n": target_n, "role": "condition"},
                            "frame_config": {
                                "time_column": "Cycle Elapsed Time (s)",
                                "position_column": "Position (mm)",
                                "force_column": "Force (N)",
                                "position_sign": position_sign,
                                "force_sign": force_sign,
                            },
                        },
                    ),
                )
            conditions.append(
                {
                    "stance_ms": stance_ms,
                    "programmed_label_ms": stance_ms,
                    "target_n": target_n,
                    "role": "condition",
                    **metrics,
                    "train_path": f"stance{stance_ms}ms_{target_n}N_train.csv",
                    "validate_path": f"stance{stance_ms}ms_{target_n}N_validate.csv",
                }
            )
            if stance_ms == 200:
                original_waveforms[target_n] = _waveform(columns, rows, position_sign=position_sign)
    # Step 15 is counted as programmed 320 ms, but must be identified from signal.
    for target_n in TARGETS_N:
        rows = _condition_rows(columns, 320, target_n)
        repeat_metrics = _measure_condition(columns, rows, 320, target_n)
        repeat_waveform = _waveform(columns, rows, position_sign=position_sign)
        comparison = _repeat_metrics(original_waveforms[target_n], repeat_waveform, target_n)
        data, provenance = _cycle_data(columns, rows, TRAIN_CYCLES, position_sign=position_sign, force_sign=force_sign)
        repeat_path = f"stance200ms_repeat_{target_n}N.csv"
        write_cycle_window_trace(
            out_dir / repeat_path,
            CycleWindowTrace(
                data,
                {
                    "source_csv": str(path),
                    "cycle_column": "Elapsed Cycles",
                    **provenance,
                    "condition": {"stance_ms": 200, "programmed_label_ms": 320, "target_n": target_n, "role": "repeat"},
                },
            ),
        )
        conditions.append(
            {
                "stance_ms": 200,
                "programmed_label_ms": 320,
                "target_n": target_n,
                "role": "repeat",
                **repeat_metrics,
                **comparison,
                "repeat_path": repeat_path,
            }
        )
    # The four distinct signal stances must increase in loaded duration.
    for target_n in TARGETS_N:
        durations = [
            next(
                c["loaded_duration_s"]
                for c in conditions
                if c["role"] == "condition" and c["stance_ms"] == stance and c["target_n"] == target_n
            )
            for stance in (160, 200, 240, 280)
        ]
        if any(right <= left for left, right in itertools.pairwise(durations)):
            raise ValueError(
                f"Loaded duration is not monotonically increasing for {target_n} N: "
                + ", ".join(
                    f"{stance} ms={duration:.3f} s"
                    for stance, duration in zip((160, 200, 240, 280), durations, strict=True)
                )
            )
    index = {"shoe": shoe, "source_csv": str(path), "conditions": conditions}
    (out_dir / "conditions.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main() -> None:
    """Regenerate processed traces for both FoamSweep tests."""
    data_root = Path("DigitalInstron/FoamSweep")
    paths = sorted(data_root.glob("*/*.steps.tracking.csv"))
    if len(paths) != 2:
        raise ValueError(f"Expected two FoamSweep tracking CSVs under {data_root}, found {len(paths)}")
    for path in paths:
        result = process_sweep(path)
        print(f"{result['shoe']}: wrote {len(result['conditions'])} validated conditions")


if __name__ == "__main__":
    main()
