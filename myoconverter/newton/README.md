# Newton Converter for Gait2354

This directory contains tools to convert the Gait2354 OpenSim model to a Newton-compatible MuJoCo XML, and utilities for motion retargeting.

## Workflow

1.  **Extract Kinematics**: Parse the original OpenSim/MuJoCo XML to a CSV format.
    ```bash
    python extract_joints.py
    ```
    Output: `gait2354_kinematics.csv`

2.  **Generate MJCF**: specific MuJoCo XML from the CSV.
    ```bash
    python generate_mjcf.py
    ```
    Output: `gait2354_newton.xml`

3.  **Validate**: Load the generated model into Newton to verify kinematic chain.
    ```bash
    python newton_loader.py
    ```

4.  **Retarget**: Use Newton's IK solver to retarget motion to this model.
    ```bash
    python retarget_motion.py
    ```
    (Modify `retarget_motion.py` main block to load your specific motion data).

## CSV Schema

The `gait2354_kinematics.csv` file defines the kinematic tree:

*   `body_name`: Name of the body link.
*   `parent_body`: Name of the parent body.
*   `joint_name`: Name of the joint connecting to parent.
*   `joint_type`: `hinge`, `slide`, or `fixed`.
*   `joint_axis`: Axis vector (e.g., `0 0 1`).
*   `range_min`, `range_max`: Joint limits.
*   `pos_x`, `pos_y`, `pos_z`: Relative position of the body/joint.
*   `is_path_point`: Boolean, true if the body represents a muscle path point.

## ProtoMotions Integration

A robot config has been created at `ProtoMotions/protomotions/robot_configs/gait2354.py` which points to the generated `gait2354_newton.xml`.
