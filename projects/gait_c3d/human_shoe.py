# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare and optionally run exact human-shoe replay from mapped C3D gait."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import newton.opensim as opensim
from projects.human_shoe.contact_sidecar import inject_contact_sidecar, load_contact_sidecar
from projects.human_shoe.replay import (
    PrescribedReplayConfig,
    find_contact_windows,
    replay_prescribed_shoe_load,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_EXPERIMENT = _REPOSITORY_ROOT / "experiments/human_shoe/baseline_gait2354.json"
_CONTACT_SIDECAR = _REPOSITORY_ROOT / "experiments/human_shoe/gait2354_subject01_contacts.json"


def prepare_experiment(data_dir: str | Path) -> tuple[Path, tuple]:
    """Create an exact-replay experiment using the mapped scaled model and IK motion."""
    data_dir = Path(data_dir).resolve()
    model_path = data_dir / "S001_scaled.osim"
    motion_path = data_dir / "trial_ik_human_shoe_context.mot"
    manifest_path = data_dir / "manifest.json"
    for path in (model_path, motion_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = opensim.parse_osim(model_path)
    sidecar = load_contact_sidecar(_CONTACT_SIDECAR)
    injected = inject_contact_sidecar(model, sidecar, replace_existing=True)
    injected.version = 40000
    contact_model_path = data_dir / "S001_scaled_with_shoe_contacts.osim"
    opensim.write_osim(injected, contact_model_path)
    reparsed = opensim.parse_osim(contact_model_path)
    if len(reparsed.contact_geometry) < len(sidecar.contacts):
        raise ValueError("contact-augmented scaled model is missing shoe anchors")

    baseline = json.loads(_BASELINE_EXPERIMENT.read_text(encoding="utf-8"))
    experiment = {
        "schema_version": baseline["schema_version"],
        "human_model_path": str(contact_model_path),
        "motion_path": str(motion_path),
        "initial_motion_frame": 0,
        "contact_sidecar_path": str(_CONTACT_SIDECAR),
        "shoe_manifest_path": str((_REPOSITORY_ROOT / baseline["shoe_manifest_path"]).resolve()),
        "controller_id": "c3d_overground_exact_replay_v1",
        "attachment": baseline["attachment"],
        "time_step_s": baseline["time_step_s"],
        "random_seed": baseline["random_seed"],
    }
    experiment_path = data_dir / "human_shoe_overground_experiment.json"
    experiment_path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    windows = find_contact_windows(experiment_path, device="cpu")
    window_data = [
        {
            "stance_index": window.stance_index,
            "start_time_s": window.start_time_s,
            "end_time_s": window.end_time_s,
            "contact_start_time_s": window.contact_start_time_s,
            "contact_end_time_s": window.contact_end_time_s,
        }
        for window in windows
    ]
    (data_dir / "human_shoe_overground_windows.json").write_text(
        json.dumps(
            {
                "frame": "virtual-overground stationary ground; OpenSim exact FK",
                "source_manifest": str(manifest_path),
                "experiment": str(experiment_path),
                "windows": window_data,
                "warning": "Shoe loads are prescribed exact-kinematics replay with no feedback into human dynamics.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return experiment_path, windows


def create_parser() -> argparse.ArgumentParser:
    """Create the mapped human-shoe replay parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="/home/jo31399/newton-data/gait/processed/trial_101/latest",
        help="Completed gait-pipeline artifact directory.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the experiment without replaying loads.")
    parser.add_argument("--replay-dt", type=float, default=0.001, help="Exact replay sample interval [s].")
    parser.add_argument("--stance-index", type=int, default=0, help="Complete right-shoe contact window to replay.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare and optionally execute the exact mapped human-shoe replay."""
    args = create_parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    experiment_path, windows = prepare_experiment(data_dir)
    print(f"[gait_c3d] human-shoe experiment: {experiment_path}")
    print(f"[gait_c3d] complete right-shoe windows: {len(windows)}")
    if args.prepare_only:
        return 0
    if not windows:
        raise ValueError("mapped stride contains no complete right-shoe contact window")
    result = replay_prescribed_shoe_load(
        experiment_path,
        PrescribedReplayConfig(dt_s=args.replay_dt, stance_index=args.stance_index),
        device="cpu",
    )
    csv_path, metadata_path = result.write_csv(data_dir / "human_shoe_overground_replay.csv")
    with np.load(data_dir / "trial_cache.npz", allow_pickle=False) as cache:
        cache_times = np.asarray(cache["times"], float)
        measured_fz = np.asarray(cache["grf"], float)[:, 1, 1]
    measured_mask = (cache_times >= result.window.contact_start_time_s) & (
        cache_times <= result.window.contact_end_time_s
    )
    measured_peak = float(np.max(measured_fz[measured_mask]))
    measured_impulse = float(np.trapezoid(measured_fz[measured_mask], cache_times[measured_mask]))
    peak_error = abs(result.peak_vertical_force_n - measured_peak) / measured_peak
    impulse_error = abs(result.final_vertical_impulse_ns - measured_impulse) / measured_impulse
    gates = {
        "peak_vertical_force": {
            "passed": peak_error <= 0.10,
            "value_relative_error": peak_error,
            "threshold_relative_error": 0.10,
        },
        "vertical_impulse": {
            "passed": impulse_error <= 0.05,
            "value_relative_error": impulse_error,
            "threshold_relative_error": 0.05,
        },
    }
    integration_qc = {
        "status": "passed" if all(gate["passed"] for gate in gates.values()) else "failed",
        "gates": gates,
        "frame": "virtual-overground stationary ground",
        "experiment": str(experiment_path),
        "measured_force_source": "right C3D force platform, OpenSim ground",
        "measured_peak_vertical_force_N": measured_peak,
        "measured_vertical_impulse_Ns": measured_impulse,
        "replay_peak_vertical_force_N": result.peak_vertical_force_n,
        "replay_vertical_impulse_Ns": result.final_vertical_impulse_ns,
        "peak_relative_error": peak_error,
        "impulse_relative_error": impulse_error,
        "warnings": [
            "Prescribed exact-kinematics shoe replay; shoe loads do not feed back into the human motion.",
            "Baseline gait2354 shoe anchors are reused on the scaled S001 model without independent subject/shoe registration.",
        ],
    }
    (data_dir / "human_shoe_integration_qc.json").write_text(
        json.dumps(integration_qc, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "metadata": str(metadata_path),
                "integration_qc": str(data_dir / "human_shoe_integration_qc.json"),
                "integration_status": integration_qc["status"],
                "peak_vertical_force_N": result.peak_vertical_force_n,
                "vertical_impulse_Ns": result.final_vertical_impulse_ns,
                "contact_work_J": result.final_contact_work_j,
                "frame": "virtual-overground stationary ground",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
