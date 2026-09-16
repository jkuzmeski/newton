# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export verified spring views by replaying saved contact history, not dynamics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

import numpy as np

_SCHEMA = "cartesian_saved_contact_springs_1"
_ARRAY_KEYS = (
    "time_s",
    "trace_index",
    "bottom_m",
    "top_m",
    "compression_m",
    "rest_length_m",
    "anchor_local_m",
    "driven",
)
_TOLERANCES = {"force_n": 1e-5, "moment_nm": 1e-6, "compression_fraction": 1e-6, "passive_cap_count": 0}
_ALIGNMENT = (
    "PREINTEGRATION saved state and velocity; snapshot immediately after the one contact update "
    "for that same trace row. time_s is the saved row time, not time_s + dt. "
    "No terminal contact update and no endpoint interpolation."
)


def _plain(value):
    """Convert metadata to JSON without changing numeric values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _content_sha256(value) -> str:
    """Hash actual input values, including array shape, dtype and complete contents."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Object arrays cannot identify a saved contact replay")
            digest.update(b"array:")
            digest.update(json.dumps([item.dtype.str, item.shape]).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"mapping:")
            for key in sorted(item):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}:".encode())
            for entry in item:
                update(entry)
        else:
            digest.update(b"scalar:")
            digest.update(json.dumps(_plain(item), sort_keys=True, allow_nan=False).encode())
        digest.update(b";")

    update(value)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Identify a source or artifact by its bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_trace(summary: dict) -> bool:
    """Select CUDA replay only for the resident GPU experiment's saved traces."""
    run = summary.get("run", summary)
    return run.get("mechanics_backend") == "Warp float64 CUDA; resident batched limb/contact/objective"


def _identity(reference, trace, summary, profile, artifact: Path) -> dict:
    """Identify the replay inputs, contact implementation, renderer and producer."""
    root = Path(__file__).resolve().parents[3]
    sources = [
        "projects/impedance_instron/cartesian/springs.py",
        "projects/impedance_instron/cartesian/mechanics.py",
        "projects/impedance_instron/cartesian/run.py",
        "projects/impedance_instron/cartesian/shoe.py",
        "projects/digital_shoe/artifact.py",
        "projects/digital_shoe/runtime.py",
        "projects/digital_shoe/contact.py",
        "projects/digital_shoe/material.py",
        "projects/digital_shoe/rendering.py",
    ]
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "warp": importlib.metadata.version("warp-lang"),
        "device": "cpu",
    }
    if _gpu_trace(summary):
        from .gpu import springs as gpu_springs  # noqa: PLC0415
        from .gpu.benchmark import execution_identity  # noqa: PLC0415

        sources.extend(
            "projects/impedance_instron/cartesian/gpu/" + name
            for name in ("springs.py", "contact_replay.py", "engine.py", "mechanics.py", "benchmark.py", "objective.py")
        )
        runtime.update(
            device=summary["run"]["shoe_device"],
            execution_identity=execution_identity(),
            pose_module_options=gpu_springs.module_options(),
        )
    return {
        "schema": _SCHEMA,
        "input_content_sha256": {
            name: _content_sha256(value)
            for name, value in (("reference", reference), ("trace", trace), ("summary", summary), ("profile", profile))
        },
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": _file_sha256(artifact),
        "source_sha256": {name: _file_sha256(root / name) for name in sources},
        "runtime": runtime,
    }


def _frame_indices(times: np.ndarray) -> np.ndarray:
    """Keep both ends and actual rows nearest the three figure times."""
    uniform = np.rint(np.linspace(0, len(times) - 1, min(len(times), 181))).astype(np.int64)
    figures = [int(np.argmin(np.abs(times - value))) for value in (0.06, 0.18, 0.30)]
    return np.unique(np.concatenate((uniform, figures)))


def _check_trace(reference, trace, summary, profile) -> tuple[np.ndarray, float]:
    """Require a finite full-rate trace starting from the saved zero-history state."""
    run = summary.get("run", summary)
    dt = float(run["actual_dt_s"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Saved actual_dt_s must be finite and positive")
    if run.get("trace_sampling") != "preintegration, one contact update per row; terminal state/velocity in summary":
        raise ValueError("Saved trace does not declare the required preintegration contact sampling")
    if run.get("initial_contact_state") != "zero material/friction histories; not a settled or periodic contact state":
        raise ValueError("Saved trace does not declare zero initial contact histories")
    times = np.asarray(trace["time_s"], dtype=float)
    count = len(times)
    if times.shape != (count,) or not np.isfinite(times).all():
        raise ValueError("Saved trace times must be a finite vector")
    if not np.allclose(times, np.arange(count) * dt, rtol=0, atol=1e-12):
        raise ValueError("Contact replay requires every row from time zero at actual_dt_s; no downsampled trace")
    shapes = {
        "state": (count, 5),
        "velocity": (count, 5),
        "grf_n": (count, 2),
        "ankle_contact_moment_nm": (count,),
        "compression_fraction": (count,),
        "driven_compression_fraction": (count,),
        "passive_compression_fraction": (count,),
        "passive_cap_column_count": (count,),
    }
    for name, shape in shapes.items():
        value = np.asarray(trace[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Saved {name} must be finite with shape {shape}")
    caps = np.asarray(trace["passive_cap_column_count"])
    if np.any(caps < 0) or np.any(caps != np.floor(caps)):
        raise ValueError("Saved passive cap counts must be nonnegative integers")
    for name in ("state", "velocity"):
        if not np.array_equal(trace[name][0], reference[name][0]):
            raise ValueError(f"Saved trace initial {name} differs from the supplied reference")
    frozen_profile = summary.get("identity", {}).get("profile_sha256")
    profile_hash = hashlib.sha256(json.dumps(_plain(profile), sort_keys=True).encode()).hexdigest()
    if frozen_profile and frozen_profile != profile_hash:
        raise ValueError("Supplied profile differs from the frozen profile identity")
    return times, dt


def _write_audit(path: Path, audit: dict) -> None:
    """Write only the derived spring audit, leaving original evidence unchanged."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".spring_view-", suffix=".json.tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(_plain(audit), indent=2, allow_nan=False) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_springs(directory, reference, trace, summary, profile, *, refresh=False) -> dict:
    """Load or export contact-history-only spring frames for an existing trace.

    Args:
        directory: Existing run directory. Only spring_view.npz and spring_view.json are written.
        reference: Actual reference arrays. Only fixed geometry and initial-state identity are used.
        trace: Full-rate saved simulated state, velocity, wrench and compression diagnostics.
        summary: Saved run summary with shoe identity and actual contact timestep [s].
        profile: Saved three-body profile. Inertias initialize the kinematics helper, not a dynamics solve.
        refresh: Recompute contact history even when a verified cache matches.

    Returns:
        Availability and audit, plus sampled time_s [s], trace_index, bottom_m/top_m
        [m, frames x columns x 3], compression_m [m, frames x columns], rest_length_m
        [m, columns], anchor_local_m [m, columns x 3], driven [columns], spacing_m
        [m] and fixed color_max_mm [mm]. Endpoints and compression are float32.
        Failed agreement returns available=False and never exports visualization arrays.
    """
    directory = Path(directory)
    if not len(trace.get("time_s", [])):
        return {"available": False, "reason": "Saved trace is empty; no spring contact frames are available."}
    archive_path, audit_path = directory / "spring_view.npz", directory / "spring_view.json"
    identity = None
    try:
        if not _gpu_trace(summary):
            raise ValueError("Spring export requires a saved GPU controller trace")
        metadata = summary["shoe"]
        artifact = Path(metadata["path"])
        identity = _identity(reference, trace, summary, profile, artifact)
        if identity["artifact_sha256"] != metadata["sha256"]:
            raise ValueError("Saved shoe artifact changed; refusing spring export")
        if not refresh and audit_path.is_file() and archive_path.is_file():
            try:
                cached = json.loads(audit_path.read_text())
                if (
                    cached.get("identity") == identity
                    and cached.get("available")
                    and cached.get("validation", {}).get("passed")
                    and cached.get("archive_sha256") == _file_sha256(archive_path)
                ):
                    with np.load(archive_path, allow_pickle=False) as archive:
                        arrays = {name: archive[name].copy() for name in _ARRAY_KEYS}
                    return {**cached, **arrays, "cache_hit": True}
            except (OSError, ValueError, KeyError):
                pass  # A damaged or outdated derived cache is rebuilt, never trusted.
        from .gpu.springs import replay  # noqa: PLC0415

        arrays, details = replay(reference, trace, summary, profile, identity)
        audit = {"available": details["validation"]["passed"], "identity": identity, **details}
        if not audit["available"]:
            audit["reason"] = "Saved-contact replay failed numeric agreement; spring visualization export refused."
        else:
            # Unique temporary files keep independently started report writers from sharing a partial archive.
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".spring_view-", suffix=".npz.tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                try:
                    np.savez_compressed(stream, **arrays)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                audit["archive_sha256"] = _file_sha256(temporary)
                temporary.replace(archive_path)
            finally:
                temporary.unlink(missing_ok=True)
        _write_audit(audit_path, audit)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        audit = {
            "available": False,
            "reason": str(error),
            "identity": identity,
            "validation": {"passed": False, "reason": str(error)},
            "time_alignment": _ALIGNMENT,
        }
        _write_audit(audit_path, audit)
    if not audit["available"]:
        archive_path.unlink(missing_ok=True)
        return {**audit, "cache_hit": False}
    return {**audit, **arrays, "cache_hit": False}


def main(argv=None) -> None:
    """Export spring frames from native saved NPZ and JSON files without a rollout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    with np.load(args.directory / "reference.npz", allow_pickle=False) as archive:
        reference = dict(archive)
    with np.load(args.directory / "trace.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    summary = json.loads((args.directory / "summary.json").read_text())
    profile = json.loads((args.directory / "profile.json").read_text())
    result = load_springs(args.directory, reference, trace, summary, profile, refresh=args.refresh)
    print(
        json.dumps(
            {
                "available": result["available"],
                "reason": result.get("reason"),
                "cache_hit": result.get("cache_hit", False),
                "archive": str(args.directory / "spring_view.npz") if result["available"] else None,
                "audit": str(args.directory / "spring_view.json"),
                "validation": result.get("validation"),
            },
            indent=2,
            allow_nan=False,
        )
    )
    if not result["available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
