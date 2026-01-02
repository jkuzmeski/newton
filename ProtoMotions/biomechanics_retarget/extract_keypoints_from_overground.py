#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Extract keypoints from overground motion data for PyRoki retargeting.

This script converts the joint position data from treadmill2overground.py
output into the keypoint format expected by PyRoki's batch retargeting scripts.

The lower-body SMPL humanoid has 9 joints:
    - Pelvis (root)
    - L_Hip, L_Knee, L_Ankle, L_Toe
    - R_Hip, R_Knee, R_Ankle, R_Toe

Output format (compatible with PyRoki):
    - positions: (T, N_KEYPOINTS, 3) - XYZ coordinates
    - orientations: (T, N_KEYPOINTS, 3, 3) - rotation matrices (estimated)
    - left_foot_contacts: (T, 2) - contact labels for left ankle and toebase
    - right_foot_contacts: (T, 2) - contact labels for right ankle and toebase

Usage:
    python extract_keypoints_from_overground.py input.npy output.npy --fps 200

Author: BioMotions Team
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
from scipy.ndimage import binary_closing, binary_opening
from scipy.spatial.transform import Rotation as R

app = typer.Typer(pretty_exceptions_enable=False)

# Joint ordering from treadmill2overground.py
TREADMILL_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
]

# Mapping to PyRoki keypoint names (9 keypoints for lower body)
PYROKI_KEYPOINT_NAMES = [
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
]


def calculate_kinematics(
    positions: np.ndarray, fps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate velocities and accelerations from positions."""
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    time_delta = 1.0 / fps
    velocities = np.gradient(positions, time_delta, axis=0)
    accelerations = np.gradient(velocities, time_delta, axis=0)
    return velocities, accelerations


def detect_foot_contacts(
    foot_positions: np.ndarray,
    foot_velocities: np.ndarray,
    foot_accelerations: np.ndarray,
    height_threshold: float = 0.05,
    vertical_velocity_threshold: float = 0.15,
    horizontal_acceleration_threshold: float = 0.5,
) -> np.ndarray:
    """
    Detect stance phases (foot contacts) using kinematic thresholds.
    
    Returns:
        stance_mask: Boolean array of shape (T,) indicating stance phase
    """
    # Height condition: foot is close to ground
    height_condition = foot_positions[:, 2] < height_threshold
    
    # Vertical velocity condition: foot is not moving vertically
    vert_vel_cond = np.abs(foot_velocities[:, 2]) < vertical_velocity_threshold
    
    # Horizontal acceleration condition: foot is not accelerating horizontally
    horiz_accel = np.linalg.norm(foot_accelerations[:, :2], axis=1)
    horiz_accel_cond = horiz_accel < horizontal_acceleration_threshold
    
    # Combine conditions
    stance_mask = height_condition & vert_vel_cond & horiz_accel_cond
    
    # Apply morphological operations to clean up noise
    stance_mask = binary_closing(stance_mask, structure=np.ones(5))
    stance_mask = binary_opening(stance_mask, structure=np.ones(3))
    
    return stance_mask


def estimate_orientations(
    positions: np.ndarray,
) -> np.ndarray:
    """
    Estimate joint orientations from position data.
    
    For lower-body retargeting, we primarily need:
    - Pelvis orientation (from hip positions)
    - Foot orientations (estimated from ankle-toe vectors)
    
    Returns:
        orientations: (T, N_joints, 3, 3) rotation matrices
    """
    n_frames, n_joints, _ = positions.shape
    orientations = np.zeros((n_frames, n_joints, 3, 3))
    
    # Initialize all as identity
    for t in range(n_frames):
        for j in range(n_joints):
            orientations[t, j] = np.eye(3)
    
    # Estimate pelvis orientation from hip positions
    # Pelvis x-axis: from right hip to left hip
    # Pelvis y-axis: forward direction (perpendicular to x, in ground plane)
    # Pelvis z-axis: up
    l_hip_idx = TREADMILL_JOINT_NAMES.index("L_Hip")
    r_hip_idx = TREADMILL_JOINT_NAMES.index("R_Hip")
    pelvis_idx = TREADMILL_JOINT_NAMES.index("Pelvis")
    
    for t in range(n_frames):
        l_hip = positions[t, l_hip_idx]
        r_hip = positions[t, r_hip_idx]
        
        # X-axis: right to left hip (lateral)
        x_axis = l_hip - r_hip
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
        
        # Z-axis: up
        z_axis = np.array([0, 0, 1])
        
        # Y-axis: forward (perpendicular to both)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
        
        # Recompute z to ensure orthogonality
        z_axis = np.cross(x_axis, y_axis)
        
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        orientations[t, pelvis_idx] = rot_matrix
    
    # Estimate foot orientations from ankle-toe vectors
    l_ankle_idx = TREADMILL_JOINT_NAMES.index("L_Ankle")
    l_toe_idx = TREADMILL_JOINT_NAMES.index("L_Toe")
    r_ankle_idx = TREADMILL_JOINT_NAMES.index("R_Ankle")
    r_toe_idx = TREADMILL_JOINT_NAMES.index("R_Toe")
    
    for t in range(n_frames):
        # Left foot
        l_ankle = positions[t, l_ankle_idx]
        l_toe = positions[t, l_toe_idx]
        l_forward = l_toe - l_ankle
        l_forward[2] = 0  # Project to ground plane
        l_forward = l_forward / (np.linalg.norm(l_forward) + 1e-6)
        
        l_z = np.array([0, 0, 1])
        l_x = np.cross(l_forward, l_z)
        l_x = l_x / (np.linalg.norm(l_x) + 1e-6)
        l_y = l_forward
        
        orientations[t, l_ankle_idx] = np.column_stack([l_x, l_y, l_z])
        orientations[t, l_toe_idx] = np.column_stack([l_x, l_y, l_z])
        
        # Right foot
        r_ankle = positions[t, r_ankle_idx]
        r_toe = positions[t, r_toe_idx]
        r_forward = r_toe - r_ankle
        r_forward[2] = 0
        r_forward = r_forward / (np.linalg.norm(r_forward) + 1e-6)
        
        r_z = np.array([0, 0, 1])
        r_x = np.cross(r_forward, r_z)
        r_x = r_x / (np.linalg.norm(r_x) + 1e-6)
        r_y = r_forward
        
        orientations[t, r_ankle_idx] = np.column_stack([r_x, r_y, r_z])
        orientations[t, r_toe_idx] = np.column_stack([r_x, r_y, r_z])
    
    return orientations


def extract_foot_contacts(
    positions: np.ndarray,
    fps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract foot contact labels from position data.
    
    Returns:
        left_foot_contacts: (T, 2) - contacts for ankle and toebase
        right_foot_contacts: (T, 2) - contacts for ankle and toebase
    """
    n_frames = positions.shape[0]
    
    # Get foot positions
    l_ankle_idx = TREADMILL_JOINT_NAMES.index("L_Ankle")
    l_toe_idx = TREADMILL_JOINT_NAMES.index("L_Toe")
    r_ankle_idx = TREADMILL_JOINT_NAMES.index("R_Ankle")
    r_toe_idx = TREADMILL_JOINT_NAMES.index("R_Toe")
    
    l_ankle_pos = positions[:, l_ankle_idx, :]
    l_toe_pos = positions[:, l_toe_idx, :]
    r_ankle_pos = positions[:, r_ankle_idx, :]
    r_toe_pos = positions[:, r_toe_idx, :]
    
    # Calculate kinematics
    l_ankle_vel, l_ankle_acc = calculate_kinematics(l_ankle_pos, fps)
    l_toe_vel, l_toe_acc = calculate_kinematics(l_toe_pos, fps)
    r_ankle_vel, r_ankle_acc = calculate_kinematics(r_ankle_pos, fps)
    r_toe_vel, r_toe_acc = calculate_kinematics(r_toe_pos, fps)
    
    # Detect contacts
    l_ankle_contact = detect_foot_contacts(l_ankle_pos, l_ankle_vel, l_ankle_acc)
    l_toe_contact = detect_foot_contacts(l_toe_pos, l_toe_vel, l_toe_acc)
    r_ankle_contact = detect_foot_contacts(r_ankle_pos, r_ankle_vel, r_ankle_acc)
    r_toe_contact = detect_foot_contacts(r_toe_pos, r_toe_vel, r_toe_acc)
    
    # Stack into output format
    left_foot_contacts = np.stack([l_ankle_contact, l_toe_contact], axis=1).astype(float)
    right_foot_contacts = np.stack([r_ankle_contact, r_toe_contact], axis=1).astype(float)
    
    return left_foot_contacts, right_foot_contacts


def extract_keypoints_for_pyroki(
    input_file: Path,
    output_file: Path,
    fps: int = 200,
    output_fps: int = 30,
) -> None:
    """
    Extract keypoints from overground motion data for PyRoki retargeting.
    
    Args:
        input_file: Path to input .npy file from treadmill2overground.py
        output_file: Path to output .npy file for PyRoki
        fps: Input motion capture frame rate
        output_fps: Output frame rate for retargeting
    """
    print(f"Loading motion from: {input_file}")
    positions = np.load(input_file)
    
    n_frames_orig, n_joints, _ = positions.shape
    print(f"Loaded {n_frames_orig} frames, {n_joints} joints")
    
    # Downsample if needed
    if fps != output_fps:
        factor = fps // output_fps
        positions = positions[::factor]
        print(f"Downsampled by factor {factor}: {positions.shape[0]} frames")
    
    n_frames = positions.shape[0]
    
    # Estimate orientations
    print("Estimating joint orientations...")
    orientations = estimate_orientations(positions)
    
    # Extract foot contacts at original FPS, then downsample
    print("Extracting foot contacts...")
    positions_orig = np.load(input_file)  # Reload at original FPS
    left_contacts, right_contacts = extract_foot_contacts(positions_orig, fps)
    
    # Downsample contacts
    if fps != output_fps:
        factor = fps // output_fps
        left_contacts = left_contacts[::factor]
        right_contacts = right_contacts[::factor]
    
    # Ensure same length
    min_len = min(n_frames, len(left_contacts), len(right_contacts))
    positions = positions[:min_len]
    orientations = orientations[:min_len]
    left_contacts = left_contacts[:min_len]
    right_contacts = right_contacts[:min_len]
    
    # Save in PyRoki-compatible format
    output_data = {
        "positions": positions,
        "orientations": orientations,
        "left_foot_contacts": left_contacts,
        "right_foot_contacts": right_contacts,
    }
    
    print(f"Saving keypoints to: {output_file}")
    print(f"  - positions: {positions.shape}")
    print(f"  - orientations: {orientations.shape}")
    print(f"  - left_foot_contacts: {left_contacts.shape}")
    print(f"  - right_foot_contacts: {right_contacts.shape}")
    
    np.save(output_file, output_data)


@app.command()
def main(
    input_file: Path = typer.Argument(
        ..., exists=True, help="Input .npy file from treadmill2overground.py"
    ),
    output_file: Path = typer.Argument(
        ..., help="Output .npy file for PyRoki retargeting"
    ),
    fps: int = typer.Option(200, "--fps", "-f", help="Input frame rate (Hz)"),
    output_fps: int = typer.Option(30, "--output-fps", help="Output frame rate (Hz)"),
):
    """
    Extract keypoints from overground motion data for PyRoki retargeting.
    
    Converts joint position data from treadmill2overground.py into the format
    expected by PyRoki's batch retargeting scripts.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    extract_keypoints_for_pyroki(
        input_file=input_file,
        output_file=output_file,
        fps=fps,
        output_fps=output_fps,
    )
    
    print("✅ Keypoint extraction complete!")


if __name__ == "__main__":
    app()
