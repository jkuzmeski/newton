#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Check pipeline data for validity and ground intersection.

This script provides tools to check intermediate data files in the biomechanics pipeline.
It can check keypoints (.npy) and retargeted motions (.npz) for issues like
ground intersection or NaN values.

Usage:
    python check_pipeline_data.py keypoints path/to/keypoints.npy
    python check_pipeline_data.py retargeted path/to/retargeted.npz --model-xml path/to/model.xml
"""

import sys
from pathlib import Path
import numpy as np
import torch
import typer

# Add protomotions to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from protomotions.components.pose_lib import (
        extract_kinematic_info,
        extract_transforms_from_qpos,
        fk_from_transforms_with_velocities,
    )
except ImportError:
    print("Warning: Could not import protomotions. Retargeted check will fail.")

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def keypoints(
    file_path: Path = typer.Argument(..., exists=True, help="Path to keypoints .npy file"),
    threshold: float = typer.Option(0.0, help="Ground height threshold"),
):
    """Check keypoints for ground intersection and validity."""
    print(f"Checking keypoints: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    
    # Handle 0-d array containing dict (common with np.save of dict)
    if data.ndim == 0:
        data = data.item()

    if isinstance(data, np.ndarray):
        positions = data
    elif isinstance(data, dict) or hasattr(data, "files"):
        if "positions" in data:
            positions = data["positions"]
        elif "keypoints" in data:
            positions = data["keypoints"]
        else:
            positions = data[data.files[0]]
    else:
        print("❌ Unknown data format")
        return

    if len(positions.shape) != 3 or positions.shape[2] != 3:
        print(f"❌ Invalid shape: {positions.shape}. Expected (T, N, 3)")
        return

    # Check for NaNs
    if np.isnan(positions).any():
        print("❌ Data contains NaNs!")
    else:
        print("✅ No NaNs found.")

    # Check ground intersection (Z < threshold)
    min_z = np.min(positions[..., 2])
    print(f"Minimum Z value: {min_z:.6f}")
    
    if min_z < threshold:
        print(f"⚠️ WARNING: Keypoints intersect ground! Min Z: {min_z:.6f}")
        num_under = np.sum(positions[..., 2] < threshold)
        total_points = positions.size / 3
        print(f"   {num_under} points ({num_under/total_points*100:.2f}%) are below {threshold}")
    else:
        print("✅ Keypoints are above ground.")


@app.command()
def retargeted(
    file_path: Path = typer.Argument(..., exists=True, help="Path to retargeted .npz file"),
    model_xml: Path = typer.Option(..., "--model-xml", "-m", exists=True, help="Path to MJCF model file"),
    threshold: float = typer.Option(0.0, help="Ground height threshold"),
):
    """Check retargeted motion for ground intersection using FK."""
    print(f"Checking retargeted motion: {file_path}")
    
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    root_pos = torch.from_numpy(data["base_frame_pos"]).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(data["base_frame_wxyz"]).to(device, dtype)
    joint_angles = torch.from_numpy(data["joint_angles"]).to(device, dtype)
    
    # Compute FK
    print(f"Computing FK using model: {model_xml}")
    kinematic_info = extract_kinematic_info(str(model_xml))
    
    qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)
    root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos_from_qpos,
        joint_rot_mats=joint_rot_mats,
        fps=30, # Dummy FPS
        compute_velocities=False,
    )
    
    # Check rigid body positions
    rb_pos = motion.rigid_body_pos
    min_z = rb_pos[..., 2].min().item()
    
    print(f"Minimum Z value (rigid bodies): {min_z:.6f}")
    
    # Identify which body is lowest
    min_vals = rb_pos[..., 2].min(dim=0).values
    lowest_body_idx = torch.argmin(min_vals).item()
    lowest_body_name = kinematic_info.body_names[lowest_body_idx]
    print(f"   Lowest body: {lowest_body_name} (Min Z: {min_vals[lowest_body_idx]:.6f})")

    if min_z < threshold:
        print(f"⚠️ WARNING: Retargeted motion intersects ground! Min Z: {min_z:.6f}")
    else:
        print("✅ Retargeted motion is above ground.")

if __name__ == "__main__":
    app()
