# PyRoki Integration Guide

## Overview

The pipeline now supports **automatic PyRoki environment switching**. You don't need to manually activate the PyRoki environment before running the pipeline.

## Quick Start

### Option 1: Auto-detection (Recommended)

If you have PyRoki installed in a conda environment named `pyroki`:

```bash
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 --height 160
```

The pipeline will automatically find and use the `pyroki` environment.

### Option 2: Explicit PyRoki Path

If your PyRoki environment is in a different location:

```bash
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 \
    --pyroki-python "C:\path\to\pyroki\env\Scripts\python.exe"
```

## How It Works

When you run the `retarget` step, the pipeline:

1. **Auto-finds** the PyRoki Python interpreter by searching conda environments for one named `pyroki`
2. **Spawns a subprocess** using that interpreter to run `batch_retarget_to_smpl_lower_body.py`
3. **Extracts contact labels** in a second subprocess call
4. **Returns to the main environment** automatically

No manual environment switching needed!

## Setting Up PyRoki

If you don't have PyRoki installed yet:

```bash
# Create a new conda environment for PyRoki
conda create -n pyroki python=3.10 -y

# Activate it
conda activate pyroki

# Clone and install PyRoki
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .

# Verify installation
python -c "import pyroki; print(pyroki.__version__)"
```

## Usage Examples

### Run only retargeting (step 3)
```bash
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 \
    --step retarget
```

### Run full pipeline (all steps) with explicit PyRoki path
```bash
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 \
    --pyroki-python "C:\Users\username\miniconda3\envs\pyroki\Scripts\python.exe"
```

### Use different model variant
```bash
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 \
    --variant adjusted_torque \
    --step retarget
```

## Troubleshooting

### PyRoki environment not found

**Error message:**
```
⚠️ PyRoki environment not found. Provide path with --pyroki-python.
```

**Solution:** Either:
1. Create a conda environment named `pyroki` with PyRoki installed
2. Provide explicit path using `--pyroki-python`

```bash
# Find your PyRoki Python path
conda run -n pyroki python -c "import sys; print(sys.executable)"

# Use that path
python pipeline.py ./input ./output --height 160 \
    --pyroki-python "C:\Users\...\pyroki\Scripts\python.exe"
```

### Import errors when running retarget step

The pipeline runs PyRoki in a subprocess, so ensure your PyRoki environment has all dependencies:

```bash
conda activate pyroki
pip install jax jax-dataclasses jaxlie jaxls yourdfpy numpy scipy
```

### Contact labels not extracted

If you see warnings about contact extraction, the main retargeting still completes. You can:

1. Check that the PyRoki environment has all dependencies installed
2. Run extraction manually:
   ```bash
   conda activate pyroki
   python pyroki/batch_retarget_to_smpl_lower_body.py \
       --keypoints-folder-path ./processed_data/S02/keypoints \
       --contacts-dir ./processed_data/S02/contacts \
       --source-type treadmill \
       --save-contacts-only
   ```

## Full Pipeline with PyRoki

The complete workflow:

```bash
# 1. Extract keypoints (uses main environment)
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 --step keypoints

# 2. Retarget with PyRoki (auto-switches environment)
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 --step retarget

# 3. Convert to ProtoMotions format (uses main environment)
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 --step convert

# 4. Package into MotionLib (uses main environment)
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160 --step package

# Or run all steps at once
python pipeline.py ./treadmill_data/S02 ./processed_data/S02 \
    --height 160  # Auto-runs all steps
```

## Advanced: Manual PyRoki Execution

If you need to run PyRoki manually (not through the pipeline):

```bash
# Activate PyRoki environment
conda activate pyroki

# Run retargeting
python pyroki/batch_retarget_to_smpl_lower_body.py \
    --keypoints-folder-path ./processed_data/S02/keypoints \
    --output-dir ./processed_data/S02/retargeted_motions \
    --source-type treadmill \
    --no-visualize \
    --skip-existing

# Extract contacts
python pyroki/batch_retarget_to_smpl_lower_body.py \
    --keypoints-folder-path ./processed_data/S02/keypoints \
    --contacts-dir ./processed_data/S02/contacts \
    --source-type treadmill \
    --save-contacts-only \
    --skip-existing

# Switch back to main environment
conda deactivate
```

## See Also

- PyRoki GitHub: https://github.com/chungmin99/pyroki
- Pipeline README: `README.md` in this directory
- Keypoint extraction: `extract_keypoints_from_overground.py`
- Retargeting script: `pyroki/batch_retarget_to_smpl_lower_body.py`
