#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Create motion YAML configuration from motion files.

This script generates a YAML configuration file listing all .motion files
in a directory, ready for packaging into a MotionLib.

Usage:
    python create_motion_yaml.py motion_dir output.yaml --fps 30

Author: BioMotions Team
"""

from pathlib import Path
from typing import List, Optional
import yaml
import typer

app = typer.Typer(pretty_exceptions_enable=False)


def create_yaml_from_motion_files(
    motion_files: List[Path],
    output_file: Path,
    fps: int = 30,
    weight: float = 1.0,
):
    """
    Create a motion YAML configuration from a list of motion files.
    
    Args:
        motion_files: List of .motion file paths
        output_file: Output YAML file path
        fps: Frame rate of the motions
        weight: Sampling weight for motions
    """
    motions = []
    
    for motion_file in motion_files:
        motion_entry = {
            "file": str(motion_file.name),
            "fps": fps,
            "weight": weight,
        }
        motions.append(motion_entry)
    
    yaml_data = {"motions": motions}
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created YAML with {len(motions)} motions: {output_file}")


@app.command()
def main(
    motion_dir: Path = typer.Argument(
        ..., exists=True, help="Directory containing .motion files"
    ),
    output_file: Path = typer.Argument(..., help="Output YAML file path"),
    fps: int = typer.Option(30, "--fps", "-f", help="Frame rate of motions"),
    weight: float = typer.Option(1.0, "--weight", "-w", help="Sampling weight"),
    pattern: str = typer.Option("*.motion", "--pattern", help="File pattern"),
):
    """
    Create a motion YAML configuration from .motion files in a directory.
    """
    motion_files = sorted(list(motion_dir.glob(pattern)))
    
    if not motion_files:
        print(f"No files matching '{pattern}' found in {motion_dir}")
        raise typer.Exit(1)
    
    print(f"Found {len(motion_files)} motion files")
    
    create_yaml_from_motion_files(
        motion_files=motion_files,
        output_file=output_file,
        fps=fps,
        weight=weight,
    )
    
    print("✅ YAML creation complete!")


if __name__ == "__main__":
    app()
