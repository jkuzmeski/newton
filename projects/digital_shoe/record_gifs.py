# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record all Digital Shoe experiment loops and embed them in the HTML report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .report import write_report

EXPERIMENTS = {
    "instron": 180,
    "drop": 60,
    "rocker": 72,
}


def record_experiment_gifs(
    artifact_path: str | Path,
    output_dir: str | Path,
    *,
    clear_kernel_cache: bool = False,
) -> tuple[list[Path], Path]:
    """Run the three audited scenes headlessly and rebuild the report with embedded GIFs."""
    artifact = Path(artifact_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if clear_kernel_cache:
        import warp as wp

        wp.clear_kernel_cache()

    gifs = []
    for mode, frame_count in EXPERIMENTS.items():
        gif_path = output / f"{mode}.gif"
        command = [
            sys.executable,
            "-m",
            "projects.digital_shoe.showcase",
            "--artifact",
            str(artifact),
            "--mode",
            mode,
            "--viewer",
            "gl",
            "--headless",
            "--num-frames",
            str(frame_count),
            "--test",
            "--record-gif",
            str(gif_path),
        ]
        subprocess.run(command, check=True)
        gifs.append(gif_path)

    report = write_report(artifact, output / "validation_report.html", media_dir=output)
    return gifs, report


def main() -> None:
    """Record the showcase loops from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="GIF/report directory; defaults beside the artifact.")
    parser.add_argument(
        "--clear-kernel-cache",
        action="store_true",
        help="Clear Warp's compiled-kernel cache before recording. Use after an incomplete PTX error.",
    )
    args = parser.parse_args()
    output = args.output or args.artifact.resolve().parent
    gifs, report = record_experiment_gifs(
        args.artifact,
        output,
        clear_kernel_cache=args.clear_kernel_cache,
    )
    for gif in gifs:
        print(f"GIF: {gif}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
