#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Convert retargeted motion data to ProtoMotions .motion format.

This script converts the output from PyRoki retargeting (NPZ files) to the
.motion format used by ProtoMotions for training and inference.

Input format (from PyRoki):
    - base_frame_pos: (T, 3) - root position XYZ
    - base_frame_wxyz: (T, 4) - root orientation quaternion WXYZ
    - joint_angles: (T, num_dofs) - joint angles in radians

Output format (.motion file):
    - Dictionary containing full motion state saved with torch.save()

Usage:
    python convert_retargeted_to_motion.py \\
        input.npz output.motion \\
        --model-xml ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml \\
        --input-fps 200 --output-fps 30 --height-offset 0.09

Author: BioMotions Team
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

# --- Environment Setup ---
# Add project root to path to allow importing protomotions
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add data/scripts to path for motion_filter
sys.path.append(str(PROJECT_ROOT / "data" / "scripts"))

# --- Imports ---
try:
    from protomotions.components.pose_lib import (
        extract_kinematic_info,
        fk_from_transforms_with_velocities,
        compute_cartesian_velocity,
        extract_transforms_from_qpos,
        extract_qpos_from_transforms,
    )
except ImportError as e:
    print(f"Error importing ProtoMotions components: {e}")
    print("Ensure you are running this from the correct directory or have installed the package.")
    sys.exit(1)

try:
    from motion_filter import passes_exclude_motion_filter
except ImportError:
    # print("Warning: Could not import motion_filter. Motion filtering will be disabled.")
    passes_exclude_motion_filter = None

app = typer.Typer(pretty_exceptions_enable=False)


def get_resample_indices(num_frames: int, input_fps: int, output_fps: int):
    """
    Calculate indices for resampling motion data using nearest neighbor.
    Handles cases where input_fps is not divisible by output_fps.
    """
    duration = num_frames / input_fps
    num_output_frames = int(duration * output_fps)
    
    # Calculate indices corresponding to the target timestamps
    # This effectively does nearest-neighbor sampling
    indices = np.linspace(0, num_frames - 1, num_output_frames).round().astype(int)
    
    # Clip to ensure valid range
    indices = np.clip(indices, 0, num_frames - 1)
    
    return indices


def load_npz_file(
    npz_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    input_fps: int,
    output_fps: int,
    target_joint_names: Optional[list] = None,
):
    """Load retargeted motion from NPZ file and resample."""
    data = np.load(npz_path, allow_pickle=True)

    base_pos = data["base_frame_pos"]
    num_frames = base_pos.shape[0]
    
    # Get resampling indices
    indices = get_resample_indices(num_frames, input_fps, output_fps)
    
    # Extract and resample
    root_pos = torch.from_numpy(data["base_frame_pos"][indices]).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(data["base_frame_wxyz"][indices]).to(device, dtype)
    
    raw_joint_angles = torch.from_numpy(data["joint_angles"][indices]).to(device, dtype)
    
    # Reorder joints if names are provided
    if "joint_names" in data and target_joint_names is not None:
        source_names = data["joint_names"].tolist()
        
        if len(source_names) == 0:
            print("Warning: Source motion has empty joint names. Skipping reordering.")
            joint_angles = raw_joint_angles
        else:
            # Handle bytes vs string
            if isinstance(source_names[0], bytes):
                source_names = [n.decode("utf-8") for n in source_names]
                
            print(f"Reordering joints from {len(source_names)} source to {len(target_joint_names)} target...")
            
            # Create mapping
            reorder_indices = []
            for target_name in target_joint_names:
                if target_name in source_names:
                    reorder_indices.append(source_names.index(target_name))
                else:
                    print(f"Warning: Joint {target_name} not found in source motion! Filling with zeros.")
                    reorder_indices.append(-1) # Marker for missing
            
            # Apply reordering
            new_joint_angles = torch.zeros((raw_joint_angles.shape[0], len(target_joint_names)), device=device, dtype=dtype)
            for i, src_idx in enumerate(reorder_indices):
                if src_idx != -1:
                    new_joint_angles[:, i] = raw_joint_angles[:, src_idx]
            
            joint_angles = new_joint_angles
    else:
        joint_angles = raw_joint_angles
    
    return root_pos, root_rot_wxyz, joint_angles


def load_contact_labels(
    contact_file: Path,
    motion_length: int,
    left_foot_idx: int,
    right_foot_idx: int,
    num_bodies: int,
    device: torch.device,
    input_fps: int,
    output_fps: int,
):
    """Load and format contact labels from NPZ file."""
    contact_data = np.load(contact_file, allow_pickle=True)
    foot_contacts = contact_data["foot_contacts"]  # [T, 2] - left, right

    # Resample contacts
    contact_frames = foot_contacts.shape[0]
    indices = get_resample_indices(contact_frames, input_fps, output_fps)
    foot_contacts = foot_contacts[indices]
    
    # Ensure same length (just in case of slight mismatch due to different duration calcs)
    contact_length = foot_contacts.shape[0]
    if contact_length != motion_length:
        print(f"Warning: Contact length ({contact_length}) != motion length ({motion_length}) after resampling.")
        if contact_length > motion_length:
            foot_contacts = foot_contacts[:motion_length]
        else:
            padding = np.repeat(foot_contacts[-1:], motion_length - contact_length, axis=0)
            foot_contacts = np.concatenate([foot_contacts, padding], axis=0)
    
    # Create rigid body contacts tensor
    rigid_body_contacts = np.zeros((motion_length, num_bodies), dtype=bool)
    rigid_body_contacts[:, left_foot_idx] = foot_contacts[:, 0] > 0.5
    rigid_body_contacts[:, right_foot_idx] = foot_contacts[:, 1] > 0.5
    
    return torch.from_numpy(rigid_body_contacts).to(device)


def convert_npz_to_motion(
    npz_file: Path,
    output_file: Path,
    model_xml: Path,
    input_fps: int = 30,
    output_fps: int = 30,
    contact_file: Optional[Path] = None,
    ignore_first_n_frames: int = 0,
    height_offset: float = 0.0,  # No offset - let motion determine ground height
    apply_motion_filter: bool = False,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.1,
    duration_height_seconds: float = 0.6,
) -> bool:
    """
    Convert a PyRoki retargeted NPZ file to ProtoMotions .motion format.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Extract kinematic info from model
    kinematic_info = extract_kinematic_info(str(model_xml))
    
    # Get DOF names from kinematic info (excluding root)
    # Note: kinematic_info.dof_names ALREADY excludes root DOFs
    # in current pose_lib implementation
    target_dof_names = kinematic_info.dof_names

    print(f"Loading motion from: {npz_file}")
    root_pos, root_rot_wxyz, joint_angles = load_npz_file(
        npz_file, device, dtype, input_fps, output_fps, target_joint_names=target_dof_names
    )
    
    print(f"Loaded motion: {root_pos.shape[0]} frames (resampled from {input_fps} -> {output_fps} fps)")
    
    # Skip initial frames if requested
    if ignore_first_n_frames > 0:
        if ignore_first_n_frames >= root_pos.shape[0]:
            print(f"Error: ignore_first_n_frames ({ignore_first_n_frames}) >= motion length")
            return False
        root_pos = root_pos[ignore_first_n_frames:]
        root_rot_wxyz = root_rot_wxyz[ignore_first_n_frames:]
        joint_angles = joint_angles[ignore_first_n_frames:]
    
    # Extract kinematic info from model
    # kinematic_info = extract_kinematic_info(str(model_xml))
    
    # Build qpos [root_pos, root_rot_wxyz, joint_angles]
    qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)
    
    # Extract transforms from qpos
    root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    
    # Compute forward kinematics with velocities
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos_from_qpos,
        joint_rot_mats=joint_rot_mats,
        fps=output_fps,
        compute_velocities=True,
    )
    
    # Use the original joint angles directly from PyRoki
    # PyRoki outputs Euler XYZ angles for each joint, which is exactly what we need.
    # Re-extracting from transforms can cause angle wrapping issues.
    motion.dof_pos = joint_angles

    # Compute DOF velocities using finite differences
    dof_vel = compute_cartesian_velocity(
        batched_robot_pos=joint_angles.unsqueeze(1),
        fps=output_fps,
    )
    motion.dof_vel = dof_vel.squeeze(1)
    
    # --- FIX HEIGHT ---
    # User requested to move motion based on minimum value and zero off of that.
    # This implies a global shift (fix_height) rather than per-frame adjustment.
    
    # We skip fix_height_per_frame to preserve flight phases and vertical dynamics.
    # motion.fix_height_per_frame(height_offset=0.02, min_clamp=-10.0)
    
    # Apply global fix
    # Use the provided height_offset directly.
    motion.fix_height(height_offset=height_offset)

    # Handle contact labels
    motion_length = motion.rigid_body_pos.shape[0]
    num_bodies = motion.rigid_body_pos.shape[1]
    body_names = kinematic_info.body_names
    
    # Attempt to automatically find foot indices
    try:
        left_foot_idx = body_names.index("L_Ankle")
        right_foot_idx = body_names.index("R_Ankle")
    except ValueError:
        try:
            left_foot_idx = body_names.index("L_Toe")
            right_foot_idx = body_names.index("R_Toe")
        except ValueError:
            # Last resort: use last two bodies
            left_foot_idx = len(body_names) - 2
            right_foot_idx = len(body_names) - 1
            print(f"Warning: Could not find foot names. Defaulting to indices {left_foot_idx}, {right_foot_idx}")
    
    if contact_file is not None and contact_file.exists():
        print(f"Loading contact labels from: {contact_file}")
        motion.rigid_body_contacts = load_contact_labels(
            contact_file=contact_file,
            motion_length=motion_length,
            left_foot_idx=left_foot_idx,
            right_foot_idx=right_foot_idx,
            num_bodies=num_bodies,
            device=device,
            input_fps=input_fps,
            output_fps=output_fps
        )
    else:
        # Default: zero contacts (can be recomputed later)
        motion.rigid_body_contacts = torch.zeros(
            motion_length, num_bodies, device=device, dtype=torch.bool
        )
    
    # HACK: prevent motion_lib from interpolating using stored rotations (can cause issues)
    motion.local_rigid_body_rot = None
    
    # Apply motion filter if enabled
    if apply_motion_filter and passes_exclude_motion_filter is not None:
        if not passes_exclude_motion_filter(
            motion,
            min_height_threshold=min_height_threshold,
            max_velocity_threshold=max_velocity_threshold,
            max_dof_vel_threshold=max_dof_vel_threshold,
            duration_height_filter=duration_height_filter,
            duration_height_seconds=duration_height_seconds,
        ):
            print(f"Skipping {npz_file.name} because it does not pass motion filter")
            return False

    # Save motion
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving motion to: {output_file}")
    torch.save(motion.to_dict(), str(output_file))
    return True


@app.command()
def main(
    npz_file: Path = typer.Argument(..., exists=True, help="Input NPZ file from PyRoki"),
    output_file: Path = typer.Argument(..., help="Output .motion file path"),
    model_xml: Path = typer.Option(
        ..., "--model-xml", "-m", exists=True,
        help="Path to MJCF model file"
    ),
    input_fps: int = typer.Option(30, "--input-fps", help="Input frame rate of the NPZ motion"),
    output_fps: int = typer.Option(30, "--output-fps", help="Target output frame rate"),
    contact_file: Optional[Path] = typer.Option(
        None, "--contact-file", "-c",
        help="Path to contact labels NPZ file"
    ),
    ignore_first_n_frames: int = typer.Option(
        0, "--ignore-first-n", help="Number of frames to skip at start"
    ),
    height_offset: float = typer.Option(
        0.0, "--height-offset", help="Height offset for ground contact (Default: 0.0m)"
    ),
):
    """
    Convert a single PyRoki retargeted NPZ file to ProtoMotions .motion format.
    """
    with torch.no_grad():
        convert_npz_to_motion(
            npz_file=npz_file,
            output_file=output_file,
            model_xml=model_xml,
            input_fps=input_fps,
            output_fps=output_fps,
            contact_file=contact_file,
            ignore_first_n_frames=ignore_first_n_frames,
            height_offset=height_offset,
        )
    
    print("✅ Conversion complete!")


# --- Batch Processing Utilities ---

def batch_convert(
    retargeted_dir: Path,
    output_dir: Path,
    model_xml: Path,
    contacts_dir: Optional[Path] = None,
    input_fps: int = 200,
    output_fps: int = 30,
    ignore_first_n_frames: int = 0,
    height_offset: float = 0.0,  # No offset - use motion ground height
    force_remake: bool = False,
):
    """Batch convert all NPZ files in a directory."""
    import glob
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = sorted(glob.glob(str(retargeted_dir / "*.npz")))
    print(f"Found {len(npz_files)} NPZ files to convert")
    
    for npz_path in tqdm(npz_files, desc="Converting motions"):
        npz_file = Path(npz_path)
        base_name = npz_file.stem.replace("_retargeted", "")
        output_file = output_dir / f"{base_name}.motion"
        
        if output_file.exists() and not force_remake:
            continue
        
        # Find contact file if contacts_dir provided
        contact_file = None
        if contacts_dir is not None:
            contact_file = contacts_dir / f"{base_name}_contacts.npz"
            if not contact_file.exists():
                contact_file = None
        
        try:
            with torch.no_grad():
                convert_npz_to_motion(
                    npz_file=npz_file,
                    output_file=output_file,
                    model_xml=model_xml,
                    input_fps=input_fps,
                    output_fps=output_fps,
                    contact_file=contact_file,
                    ignore_first_n_frames=ignore_first_n_frames,
                    height_offset=height_offset,
                )
        except Exception as e:
            print(f"Error converting {npz_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Batch conversion complete! Output: {output_dir}")


@app.command("batch")
def batch_command(
    retargeted_dir: Path = typer.Argument(..., exists=True, help="Directory with NPZ files"),
    output_dir: Path = typer.Argument(..., help="Output directory for .motion files"),
    model_xml: Path = typer.Option(
        ..., "--model-xml", "-m", exists=True,
        help="Path to MJCF model file"
    ),
    contacts_dir: Optional[Path] = typer.Option(
        None, "--contacts-dir", "-c",
        help="Directory with contact labels"
    ),
    input_fps: int = typer.Option(200, "--input-fps", help="Input frame rate of the motion"),
    output_fps: int = typer.Option(30, "--output-fps", help="Target output frame rate"),
    ignore_first_n_frames: int = typer.Option(
        0, "--ignore-first-n", help="Number of frames to skip at start"
    ),
    height_offset: float = typer.Option(
        0.09, "--height-offset", help="Height offset for ground contact"
    ),
    force_remake: bool = typer.Option(False, "--force", help="Force remake existing files"),
):
    """
    Batch convert all NPZ files in a directory to .motion format.
    """
    batch_convert(
        retargeted_dir=retargeted_dir,
        output_dir=output_dir,
        model_xml=model_xml,
        contacts_dir=contacts_dir,
        input_fps=input_fps,
        output_fps=output_fps,
        ignore_first_n_frames=ignore_first_n_frames,
        height_offset=height_offset,
        force_remake=force_remake,
    )


if __name__ == "__main__":
    app()