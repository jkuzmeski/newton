#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Package motion files into a MotionLib .pt file.

This script packages individual .motion files (or a YAML config pointing to them)
into a single .pt file that can be used for training with ProtoMotions.

Usage:
    python package_motions.py motion.yaml output.pt --model-xml model.xml

Author: BioMotions Team
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import typer

app = typer.Typer(pretty_exceptions_enable=False)

# Add protomotions to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protomotions.components.motion_lib import MotionLib, MotionLibConfig


def package_motions(
    yaml_file: Path,
    output_file: Path,
    motion_dir: Path,
    model_xml: Path,
    device: str = "cpu",
):
    """
    Package motions from a YAML config into a MotionLib .pt file.
    
    Args:
        yaml_file: Path to YAML config listing motions
        output_file: Output .pt file path
        motion_dir: Directory containing .motion files
        model_xml: Path to MJCF model file
        device: Device for loading motions
    """
    print(f"Packaging motions from: {yaml_file}")
    print(f"Motion directory: {motion_dir}")
    print(f"Model XML: {model_xml}")
    
    # Create MotionLib config
    config = MotionLibConfig(
        motion_file=str(yaml_file),
    )
    
    # Load motion library
    motion_lib = MotionLib(config, device=device)
    
    # Package and save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    motion_lib.save(str(output_file))
    
    print(f"✅ Saved MotionLib to: {output_file}")
    print(f"   - {motion_lib.num_motions} motions")
    print(f"   - Total duration: {motion_lib.motion_lengths.sum():.1f}s")


@app.command()
def main(
    yaml_file: Path = typer.Argument(
        ..., exists=True, help="YAML file listing motions"
    ),
    output_file: Path = typer.Argument(..., help="Output .pt file path"),
    motion_dir: Path = typer.Option(
        None, "--motion-dir", "-d",
        help="Directory containing .motion files (default: yaml_file parent)"
    ),
    model_xml: Path = typer.Option(
        ..., "--model-xml", "-m", exists=True,
        help="Path to MJCF model file"
    ),
    device: str = typer.Option("cpu", "--device", help="Device for loading"),
):
    """
    Package motions from a YAML config into a MotionLib .pt file.
    
    Example:
        python package_motions.py motions.yaml output.pt \\
            --model-xml ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml
    """
    if motion_dir is None:
        motion_dir = yaml_file.parent
    
    package_motions(
        yaml_file=yaml_file,
        output_file=output_file,
        motion_dir=motion_dir,
        model_xml=model_xml,
        device=device,
    )


if __name__ == "__main__":
    app()
