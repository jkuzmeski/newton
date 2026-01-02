#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Visualize keypoints from .npy files.

This script visualizes the keypoints extracted from overground motion data.
It helps in debugging if the keypoints are correct and not intersecting the ground
before retargeting.

Usage:
    python visualize_keypoints.py path/to/keypoints.npy
"""

import sys
import time
from pathlib import Path
import numpy as np
import typer

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    sys.exit(1)

app = typer.Typer(pretty_exceptions_enable=False)

# Connectivity for lower body (indices based on extract_keypoints_from_overground.py)
# 0: Pelvis
# 1: L_Hip, 2: L_Knee, 3: L_Ankle, 4: L_Toe
# 5: R_Hip, 6: R_Knee, 7: R_Ankle, 8: R_Toe
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Left leg
    (0, 5), (5, 6), (6, 7), (7, 8),  # Right leg
]

@app.command()
def main(
    file_path: Path = typer.Argument(..., exists=True, help="Path to keypoints .npy file"),
    fps: int = typer.Option(30, "--fps", help="Playback FPS"),
    save_video: bool = typer.Option(False, "--save", help="Save animation to video"),
):
    """Visualize 3D keypoints from a .npy file."""
    print(f"Loading {file_path}...")
    data = np.load(file_path, allow_pickle=True)
    
    # Handle 0-d array containing dict
    if data.shape == () and data.dtype == 'O':
        data = data.item()
    
    # Handle different data structures
    if isinstance(data, np.ndarray):
        positions = data
    elif isinstance(data, dict) or hasattr(data, "files"):
        if "positions" in data:
            positions = data["positions"]
        elif "keypoints" in data:
            positions = data["keypoints"]
        else:
            # Try the first array
            positions = data[data.files[0]]
    else:
        print("Unknown data format")
        return

    print(f"Data shape: {positions.shape}")
    if len(positions.shape) != 3 or positions.shape[2] != 3:
        print("Expected shape (T, N, 3)")
        return

    num_frames = positions.shape[0]
    
    # Calculate bounds
    min_vals = np.min(positions, axis=(0, 1))
    max_vals = np.max(positions, axis=(0, 1))
    mid_vals = (min_vals + max_vals) / 2
    range_vals = max_vals - min_vals
    max_range = np.max(range_vals)
    
    print(f"Bounds: X[{min_vals[0]:.2f}, {max_vals[0]:.2f}], "
          f"Y[{min_vals[1]:.2f}, {max_vals[1]:.2f}], "
          f"Z[{min_vals[2]:.2f}, {max_vals[2]:.2f}]")
    
    # Check for ground intersection (assuming Z is up and ground is at 0)
    min_z = np.min(positions[..., 2])
    print(f"Minimum Z value: {min_z:.4f}")
    if min_z < 0:
        print("⚠️ WARNING: Keypoints intersect the ground (Z < 0)!")

    # Setup plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Keypoints Visualization: {file_path.name}")
    
    # Set consistent axis limits
    ax.set_xlim(mid_vals[0] - max_range/2, mid_vals[0] + max_range/2)
    ax.set_ylim(mid_vals[1] - max_range/2, mid_vals[1] + max_range/2)
    ax.set_zlim(0, max_vals[2] + 0.1)  # Assume ground at 0
    
    # Plot ground plane
    xx, yy = np.meshgrid(np.linspace(min_vals[0]-1, max_vals[0]+1, 10),
                         np.linspace(min_vals[1]-1, max_vals[1]+1, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # Scatter plot for joints
    scat = ax.scatter([], [], [], c='blue', s=20)
    
    # Lines for bones
    lines = [ax.plot([], [], [], 'r-')[0] for _ in SKELETON_CONNECTIONS]
    
    text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def update(frame):
        current_pos = positions[frame]
        
        # Update scatter
        scat._offsets3d = (current_pos[:, 0], current_pos[:, 1], current_pos[:, 2])
        
        # Update lines
        for line, (i, j) in zip(lines, SKELETON_CONNECTIONS):
            if i < current_pos.shape[0] and j < current_pos.shape[0]:
                line.set_data([current_pos[i, 0], current_pos[j, 0]],
                              [current_pos[i, 1], current_pos[j, 1]])
                line.set_3d_properties([current_pos[i, 2], current_pos[j, 2]])
        
        text.set_text(f"Frame: {frame}/{num_frames}")
        return [scat] + lines + [text]

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)
    
    if save_video:
        output_video = file_path.with_suffix('.mp4')
        print(f"Saving video to {output_video}...")
        ani.save(output_video, writer='ffmpeg', fps=fps)
    else:
        plt.show()

if __name__ == "__main__":
    app()
