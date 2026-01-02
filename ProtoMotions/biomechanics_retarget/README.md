# Biomechanics Motion Processing Pipeline

This directory contains a complete pipeline for processing treadmill motion capture data and preparing it for simulation training with ProtoMotions. The pipeline transforms treadmill motions to overground, retargets them to SMPL lower-body humanoid models using PyRoki, and packages them for use in simulation environments.

## Overview

The pipeline follows this workflow:

```
Treadmill Motion Data (.txt)
        │
        ▼ (treadmill2overground.py)
Overground Motion Data (.npy)
        │
        ▼ (extract_keypoints_from_overground.py)
PyRoki Keypoints (.npy)
        │
        ▼ (batch_retarget_to_smpl_lower_body.py) [PyRoki environment]
Retargeted Motion (.npz)
        │
        ▼ (convert_retargeted_to_motion.py)
ProtoMotions Format (.motion)
        │
        ▼ (package_motions.py)
MotionLib (.pt)
```

## Files Overview

### Main Pipeline Scripts

| Script | Description |
|--------|-------------|
| `pipeline.py` | Main pipeline orchestrator |
| `quick_rescale.py` | Quick rescaling to new subject heights |
| `treadmill2overground.py` | Transform treadmill motions to overground |
| `extract_keypoints_from_overground.py` | Extract PyRoki-compatible keypoints |
| `convert_retargeted_to_motion.py` | Convert PyRoki output to .motion format |
| `create_motion_yaml.py` | Create motion list YAML files |
| `package_motions.py` | Package motions into MotionLib .pt files |

### Legacy Scripts (Mink-based, for reference)

| Script | Description |
|--------|-------------|
| `process_treadmill_motions.py` | Old pipeline using Mink IK |
| `retarget_treadmill_motion.py` | Old Mink-based retargeting |
| `create_motion_fps_yaml.py` | Old YAML creation |
| `package_motion_lib.py` | Old motion packaging |

## Quick Start

### Installation

1. Install ProtoMotions dependencies:
```bash
pip install -r requirements_genesis.txt  # or your preferred simulator
```

2. Install PyRoki in a **separate** Python environment:
```bash
conda create -n pyroki python=3.10
conda activate pyroki
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

### Basic Usage

#### Option 1: Height-Based Pipeline (Recommended)

Process motions using subject height for automatic model selection:

```bash
# Process data for a 156cm subject - auto-creates/selects model
python biomechanics_retarget/pipeline.py \
    ./treadmill_data/S02 \
    ./processed_data/S02 \
    --height 156 \
    --fps 200

# If you will train with the smpl_lower_body_170cm_contact_pads robot,
# pass --contact-pads so the pipeline uses the *_contact_pads MJCF when converting/packaging.
# (You should also pass --pyroki-urdf-path to retarget using the contact-pad URDF.)

# Process for a 180cm subject with torque control variant
python biomechanics_retarget/pipeline.py \
    ./treadmill_data/S03 \
    ./processed_data/S03 \
    --height 180 \
    --variant adjusted_torque
```

#### Option 2: Manual Model Specification

Specify the model file explicitly:

```bash
python biomechanics_retarget/pipeline.py \
    ./treadmill_data/S02 \
    ./processed_data/S02 \
    --model ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml \
    --fps 200 \
    --output-fps 30
```

#### Option 2: Step-by-Step Processing

1. **Transform treadmill to overground:**
```bash
python biomechanics_retarget/treadmill2overground.py \
    ./treadmill_data/S02 \
    ./processed_data/S02/overground_data \
    --fps 200 \
    --transform y_to_x_forward
```

2. **Extract keypoints for PyRoki:**
```bash
python biomechanics_retarget/extract_keypoints_from_overground.py \
    ./processed_data/S02/overground_data/S02_20ms.npy \
    ./processed_data/S02/keypoints/S02_20ms.npy \
    --fps 200 --output-fps 30
```

3. **Retarget with PyRoki** (in PyRoki environment):
```bash
conda activate pyroki

# Full retargeting
python pyroki/batch_retarget_to_smpl_lower_body.py \
    --keypoints-folder-path ./processed_data/S02/keypoints \
    --output-dir ./processed_data/S02/retargeted_motions \
    --urdf-path ./protomotions/data/assets/urdf/smpl_lower_body.urdf \
    --source-type treadmill \
    --no-visualize

# If you are training the *_contact_pads robot, retarget using the contact-pad URDF:
python pyroki/batch_retarget_to_smpl_lower_body.py \
    --keypoints-folder-path ./processed_data/S02/keypoints \
    --output-dir ./processed_data/S02/retargeted_motions \
    --urdf-path ./protomotions/data/assets/urdf/for_retargeting/smpl_lower_body_contact_pads.urdf \
    --source-type treadmill \
    --no-visualize

# Extract contact labels
python pyroki/batch_retarget_to_smpl_lower_body.py \
    --keypoints-folder-path ./processed_data/S02/keypoints \
    --contacts-dir ./processed_data/S02/contacts \
    --source-type treadmill \
    --save-contacts-only

# Contact labels with contact-pad URDF:
python pyroki/batch_retarget_to_smpl_lower_body.py \
    --keypoints-folder-path ./processed_data/S02/keypoints \
    --contacts-dir ./processed_data/S02/contacts \
    --urdf-path ./protomotions/data/assets/urdf/for_retargeting/smpl_lower_body_contact_pads.urdf \
    --source-type treadmill \
    --save-contacts-only
```

4. **Convert to ProtoMotions format:**
```bash
python biomechanics_retarget/convert_retargeted_to_motion.py batch \
    ./processed_data/S02/retargeted_motions \
    ./processed_data/S02/motion_files \
    --model-xml ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml \
    --contacts-dir ./processed_data/S02/contacts \
    --fps 30
```

5. **Create YAML and package:**
```bash
# Create YAML
python biomechanics_retarget/create_motion_yaml.py \
    ./processed_data/S02/motion_files \
    ./processed_data/S02/yaml_data/motions.yaml \
    --fps 30

# Package MotionLib
python biomechanics_retarget/package_motions.py \
    ./processed_data/S02/yaml_data/motions.yaml \
    ./processed_data/S02/packaged_data/S02.pt \
    --model-xml ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml
```

## Pipeline Steps

The pipeline performs these steps automatically:

1. **Treadmill to Overground Transformation**
   - Loads joint position data from .txt files
   - Applies coordinate system transformations
   - Adjusts motion vertically to ground plane
   - Transforms treadmill motion to overground motion

2. **Keypoint Extraction**
   - Converts positions to PyRoki-compatible format
   - Estimates joint orientations from positions
   - Detects foot contacts using kinematic thresholds
   - Downsamples from input FPS to output FPS

3. **PyRoki Retargeting** (requires separate environment)
   - Performs trajectory-level kinematic optimization
   - Applies foot contact constraints to prevent skating
   - Produces smooth, temporally consistent motion
   - Outputs joint angles and root transforms

4. **ProtoMotions Conversion**
   - Converts PyRoki output to .motion format
   - Applies forward kinematics to compute full body state
   - Incorporates contact labels
   - Fixes height for ground contact

5. **YAML and Packaging**
   - Creates motion list YAML with FPS and weights
   - Packages all motions into a single .pt MotionLib

## Output Structure

The pipeline creates an organized output directory:

```text
processed_data/S02/
├── overground_data/     # Overground motion files (.npy)
│   └── metadata/        # CSV, JSON, TXT metadata files
├── keypoints/           # PyRoki-compatible keypoints (.npy)
├── contacts/            # Foot contact labels (.npz)
├── retargeted_motions/  # Retargeted motion files (.npz)
├── motion_files/        # ProtoMotions format (.motion)
├── yaml_data/           # Motion list YAML file
└── packaged_data/       # Final packaged MotionLib (.pt)
```

## Input File Format

The pipeline expects treadmill motion data in tab-separated `.txt` files with:

- Each row representing a time frame
- The first column containing the frame number
- Remaining columns containing joint positions (X, Y, Z for each joint)
- Joints ordered as Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe

## Subject Height Scaling

### Quick Rescaling for New Heights

The pipeline supports automatic model creation for different subject heights:

```bash
# Create a 156cm model
python biomechanics_retarget/quick_rescale.py --height 156

# Create a 180cm model with torque control
python biomechanics_retarget/quick_rescale.py --height 180 --variant adjusted_torque

# List available height models
python biomechanics_retarget/quick_rescale.py --list

# Preview without creating files
python biomechanics_retarget/quick_rescale.py --height 165 --dry-run
```

### Using Height-Scaled Models in Code

Height-scaled models are automatically registered with ProtoMotions:

```python
from protomotions.robot_configs.factory import robot_config

# Get config for specific height
config_156cm = robot_config("smpl_lower_body_156cm")
config_180cm = robot_config("smpl_lower_body_180cm")

# Torque control variant
config_torque = robot_config("smpl_lower_body_180cm_torque")

# Base model (170cm)
config_base = robot_config("smpl_lower_body")
```

### Complete Workflow for New Subject

```bash
# 1. Rescale model to subject height (if not already done)
python biomechanics_retarget/quick_rescale.py --height 156

# 2. Process their motion data
python biomechanics_retarget/pipeline.py \
    ./treadmill_data/subject_156cm \
    ./processed_data/subject_156cm \
    --height 156

# 3. Use in training
python protomotions/train_agent.py \
    robot=smpl_lower_body_156cm \
    ...
```

## Model Requirements

The target MJCF model should:

- Be compatible with the ProtoMotions framework
- Have the required joint structure for lower body motions
- Include appropriate scaling for the source motion data

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and ProtoMotions is properly configured.
1. **Model Loading Errors**: Check that the MJCF model path is correct and the file is valid.
1. **Motion Processing Failures**: Verify input motion files are in the correct format.
1. **Scale Factor Issues**: Try manual scaling if auto-scaling produces poor results.
1. **Height Model Not Found**: Run `quick_rescale.py --height XXX` to create it.

### Debug Output

The pipeline provides detailed console output for each step. Look for:

- ✅ Success indicators
- ⚠️ Warning messages
- ❌ Error messages

### Performance Tips

- Use `--clean` to save disk space by removing intermediate files.
- Process smaller batches if memory is limited.
- Use manual scaling (`--scale`) if you know the correct factor.

## Integration

The final packaged motion library can be used in:

- IsaacLab simulation environments
- ProtoMotions training pipelines
- Custom simulation frameworks

Load the packaged motions in your simulation code:

```python
import torch

motion_lib = torch.load("processed_model/packaged/motion_lib_model.pkl")
```

### Training with Custom Robot

```python
# In your training script
from protomotions.robot_configs.factory import robot_config

# Automatically uses height-scaled model
config = robot_config("smpl_lower_body_156cm")

# Access robot properties
print(f"Root height: {config.default_root_height}m")
print(f"Asset file: {config.asset.asset_file_name}")
```