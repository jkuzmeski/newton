#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Biomechanics Motion Processing Pipeline

This module provides a complete pipeline for processing treadmill motion capture data
into ProtoMotions-compatible MotionLib format. The pipeline supports lower-body 
SMPL humanoid models and integrates with PyRoki for trajectory optimization-based retargeting.

Pipeline Steps:
    1. Treadmill to Overground: Convert treadmill motions to overground locomotion
    2. Extract Keypoints: Convert positions to PyRoki-compatible keypoint format
    3. Retarget with PyRoki: Trajectory-level kinematic optimization to target robot
    4. Convert to ProtoMotions: Generate .motion files from retargeted data
    5. Package MotionLib: Create .pt file for training

Usage:
    # Run the complete pipeline
    python biomechanics_retarget/pipeline.py \\
        --input-dir ./treadmill_data/S02 \\
        --output-dir ./processed_data/S02 \\
        --model-xml ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml \\
        --fps 200

    # Run individual steps
    python biomechanics_retarget/pipeline.py \\
        --input-dir ./treadmill_data/S02 \\
        --output-dir ./processed_data/S02 \\
        --model-xml ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml \\
        --step overground  # or: keypoints, retarget, convert, package

Author: BioMotions Team
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))
# Ensure data/scripts can be imported (for motion_filter)
sys.path.append(str(Path(__file__).parent.parent / "data" / "scripts"))

try:
    import typer
    from rich.console import Console
    from rich.progress import Progress, TaskID
    from rich.panel import Panel
except ImportError as e:
    print("❌ Missing required dependencies.")
    print("\nPlease install with:")
    print("  pip install typer rich numpy scipy pandas")
    print("\nOr if using conda:")
    print("  conda install typer rich numpy scipy pandas -c conda-forge")
    print(f"\nMissing module: {e}")
    sys.exit(1)

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)


class PipelineStep(str, Enum):
    """Pipeline execution steps."""
    OVERGROUND = "overground"
    KEYPOINTS = "keypoints"
    RETARGET = "retarget"
    CONVERT = "convert"
    FILTER = "filter"
    PACKAGE = "package"
    CHECK = "check"
    ALL = "all"


@dataclass
class PipelineConfig:
    """Configuration for the biomechanics pipeline."""
    input_dir: Path
    output_dir: Path
    model_xml: Path
    fps: int = 200
    output_fps: int = 30
    coordinate_transform: str = "y_to_x_forward"
    speed_override: Optional[float] = None
    auto_scale: bool = True
    scale_override: Optional[float] = None
    force_remake: bool = False
    clean_intermediate: bool = False
    subject_height_cm: Optional[int] = None  # Auto model selection
    model_variant: str = "adjusted_pd"  # adjusted_pd or adjusted_torque
    pyroki_python: Optional[Path] = None  # PyRoki Python interpreter
    pyroki_urdf_path: Optional[Path] = None  # Optional URDF override for PyRoki
    jax_platform: str = "cuda"  # JAX backend: cuda, rocm, cpu, etc
    apply_motion_filter: bool = True  # Apply motion quality filter
    filter_config: Optional[Path] = None  # Motion quality filter config
    
    # Derived paths
    @property
    def overground_dir(self) -> Path:
        return self.output_dir / "overground_data"
    
    @property
    def keypoints_dir(self) -> Path:
        return self.output_dir / "keypoints"
    
    @property
    def contacts_dir(self) -> Path:
        return self.output_dir / "contacts"
    
    @property
    def retargeted_dir(self) -> Path:
        return self.output_dir / "retargeted_motions"
    
    @property
    def motion_dir(self) -> Path:
        return self.output_dir / "motion_files"
    
    @property
    def yaml_dir(self) -> Path:
        return self.output_dir / "yaml_data"
    
    @property
    def packaged_dir(self) -> Path:
        return self.output_dir / "packaged_data"
    
    @property
    def model_name(self) -> str:
        return self.model_xml.stem.replace("_", "-")
    
    def create_directories(self):
        """Create all output directories."""
        for dir_path in [
            self.overground_dir,
            self.keypoints_dir,
            self.contacts_dir,
            self.retargeted_dir,
            self.motion_dir,
            self.yaml_dir,
            self.packaged_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


class BiomechanicsPipeline:
    """Main pipeline class for processing treadmill motion data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.create_directories()
        
    def find_input_files(self) -> List[Path]:
        """Find all .txt motion files in the input directory."""
        txt_files = list(self.config.input_dir.glob("**/*.txt"))
        console.print(f"Found {len(txt_files)} .txt files in {self.config.input_dir}")
        return sorted(txt_files)
    
    def step_overground(self, progress: Optional[Progress] = None, task_id: Optional[TaskID] = None) -> List[Path]:
        """Step 1: Transform treadmill motions to overground."""
        from treadmill2overground import process_motion_file
        
        console.print("\n[bold blue]Step 1: Transforming treadmill motions to overground[/bold blue]")
        
        input_files = self.find_input_files()
        successful_files = []
        
        for i, motion_file in enumerate(input_files):
            if progress and task_id:
                progress.update(task_id, completed=i)
            
            # Check if output already exists
            output_file = self.config.overground_dir / f"{motion_file.stem}_positions_lowerbody.npy"
            if output_file.exists() and not self.config.force_remake:
                console.print(f"   ⏭️ Skipping {motion_file.name} (already processed)")
                successful_files.append(output_file)
                continue
            
            console.print(f"   Processing {motion_file.name}...")
            
            try:
                success = process_motion_file(
                    motion_file=motion_file,
                    output_dir=self.config.overground_dir,
                    fps=self.config.fps,
                    coordinate_transform=self.config.coordinate_transform,
                    speed_override=self.config.speed_override,
                )
                
                if success:
                    # Find the generated .npy file
                    expected_npy = self.config.overground_dir / f"{motion_file.stem}.npy"
                    if expected_npy.exists():
                        successful_files.append(expected_npy)
                        console.print(f"   ✅ Successfully processed {motion_file.name}")
                    else:
                        console.print(f"   ⚠️ Expected output file not found: {expected_npy}")
                else:
                    console.print(f"   ❌ Failed to process {motion_file.name}")
                    
            except Exception as e:
                console.print(f"   ❌ Error processing {motion_file.name}: {e}")
                continue
        
        if progress and task_id:
            progress.update(task_id, completed=len(input_files))
        
        console.print(f"\n✅ Overground transformation completed. {len(successful_files)} files successful.")
        return successful_files
    
    def step_keypoints(self, overground_files: Optional[List[Path]] = None) -> List[Path]:
        """Step 2: Convert overground positions to PyRoki-compatible keypoints."""
        from extract_keypoints_from_overground import extract_keypoints_for_pyroki
        
        console.print("\n[bold blue]Step 2: Extracting keypoints for retargeting[/bold blue]")
        
        if overground_files is None:
            overground_files = list(self.config.overground_dir.glob("*.npy"))
        
        successful_files = []
        
        for motion_file in overground_files:
            output_file = self.config.keypoints_dir / f"{motion_file.stem}.npy"
            
            if output_file.exists() and not self.config.force_remake:
                console.print(f"   ⏭️ Skipping {motion_file.name} (already processed)")
                successful_files.append(output_file)
                continue
            
            console.print(f"   Extracting keypoints from {motion_file.name}...")
            
            try:
                extract_keypoints_for_pyroki(
                    input_file=motion_file,
                    output_file=output_file,
                    fps=self.config.fps,
                    output_fps=self.config.output_fps,
                )
                successful_files.append(output_file)
                console.print(f"   ✅ Extracted keypoints to {output_file.name}")
            except Exception as e:
                console.print(f"   ❌ Error extracting keypoints from {motion_file.name}: {e}")
                continue
        
        console.print(f"\n✅ Keypoint extraction completed. {len(successful_files)} files successful.")
        return successful_files
    
    def step_retarget(self, keypoint_files: Optional[List[Path]] = None) -> List[Path]:
        """Step 3: Retarget keypoints to target robot using PyRoki."""
        import subprocess
        import sys
        from pathlib import Path
        
        console.print("\n[bold blue]Step 3: Retargeting motions with PyRoki[/bold blue]")
        
        # Check for existing retargeted files first
        retargeted_files = list(self.config.retargeted_dir.glob("*_retargeted.npz"))
        
        # If retargeted files exist and not forcing remake, skip retargeting
        if retargeted_files and not self.config.force_remake:
            console.print(f"✅ Found {len(retargeted_files)} retargeted motion files (skipping).")
            return retargeted_files
        
        # Try to find PyRoki environment
        pyroki_python = self._find_pyroki_python()
        
        if pyroki_python is None:
            console.print("[yellow]⚠️ PyRoki environment not found. Provide path with --pyroki-python.[/yellow]")
            console.print(f"Run manually: python pyroki/batch_retarget_to_smpl_lower_body.py \\")
            console.print(f"    --keypoints-folder-path {self.config.keypoints_dir} \\")
            console.print(f"    --output-dir {self.config.retargeted_dir} \\")
            if self.config.pyroki_urdf_path is not None:
                console.print(f"    --urdf-path {self.config.pyroki_urdf_path} \\")
            console.print(f"    --source-type treadmill \\")
            console.print(f"    --no-visualize")
            return []
        
        # Run retargeting in PyRoki environment
        console.print(f"🔄 Running PyRoki with: {pyroki_python}")
        
        try:
            # Step 1: Run retargeting
            # Script is in root pyroki/ folder, not biomechanics_retarget/pyroki/
            project_root = Path(__file__).parent.parent
            retarget_script = (project_root / "pyroki" / 
                               "batch_retarget_to_smpl_lower_body.py")
            
            if not retarget_script.exists():
                console.print(f"[red]❌ Retarget script not found: "
                              f"{retarget_script}[/red]")
                return []
            
            # Create environment with JAX settings
            env = os.environ.copy()
            # Set JAX platform with fallback to CPU on failure
            jax_platform = self.config.jax_platform
            env['JAX_PLATFORMS'] = jax_platform
            console.print(f"⚙️ JAX platform: {jax_platform or 'auto-detect'}")
            # Enable full traceback for JAX errors
            env['JAX_TRACEBACK_FILTERING'] = 'off'
            env['JAX_DEFAULT_PREC'] = 'float32'
            
            # Add NVIDIA paths for Windows
            self._add_nvidia_paths(env, pyroki_python)
            
            cmd = [
                str(pyroki_python),
                str(retarget_script),
                "--keypoints-folder-path", str(self.config.keypoints_dir),
                "--output-dir", str(self.config.retargeted_dir),
                "--source-type", "treadmill",
                "--target-raw-frames", "-1",
                "--no-visualize",
                "--skip-existing",
            ]

            if self.config.pyroki_urdf_path is not None:
                cmd.extend(["--urdf-path", str(self.config.pyroki_urdf_path)])
            
            console.print("   Running retargeting...")
            console.print(f"   Script: {retarget_script}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env)
            
            # If CUDA fails, retry with CPU
            if result.returncode != 0 and \
               jax_platform == 'cuda' and \
               'AssertionError' in result.stderr:
                console.print("[yellow]⚠️ CUDA initialization failed, "
                              "retrying with CPU...[/yellow]")
                console.print(f"[red]Error details:[/red]\n{result.stderr}")
                env['JAX_PLATFORMS'] = 'cpu'
                result = subprocess.run(
                    cmd, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                console.print("[red]❌ Retargeting failed:[/red]")
                console.print(result.stderr)
                return []
            
            # Step 2: Extract contact labels
            console.print(f"   Extracting contact labels...")
            cmd = [
                str(pyroki_python),
                str(retarget_script),
                "--keypoints-folder-path", str(self.config.keypoints_dir),
                "--contacts-dir", str(self.config.contacts_dir),
                "--source-type", "treadmill",
                "--target-raw-frames", "-1",
                "--save-contacts-only",
                "--skip-existing",
            ]

            if self.config.pyroki_urdf_path is not None:
                cmd.extend(["--urdf-path", str(self.config.pyroki_urdf_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                console.print(f"[yellow]⚠️ Contact extraction had issues (continuing):[/yellow]")
                console.print(result.stderr)
            
            # Check results
            retargeted_files = list(self.config.retargeted_dir.glob("*_retargeted.npz"))
            if retargeted_files:
                console.print(f"✅ Retargeting complete: {len(retargeted_files)} files")
            else:
                console.print(f"[red]❌ No retargeted files created[/red]")
            
            return retargeted_files
            
        except Exception as e:
            console.print(f"[red]❌ Error running PyRoki: {e}[/red]")
            return []
    
    def _find_pyroki_python(self) -> Optional[Path]:
        """Find PyRoki Python interpreter, checking conda environments."""
        import subprocess
        import json
        
        # Check if pyroki_python was provided via CLI
        if hasattr(self.config, 'pyroki_python') and self.config.pyroki_python:
            path = Path(self.config.pyroki_python)
            if path.exists():
                return path
        
        # Try to find pyroki conda environment
        try:
            result = subprocess.run(
                ["conda", "info", "--json"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                conda_info = json.loads(result.stdout)
                envs = conda_info.get("envs", [])
                
                for env_path in envs:
                    python_exe = Path(env_path) / "Scripts" / "python.exe"
                    if not python_exe.exists():
                        python_exe = Path(env_path) / "bin" / "python"
                    
                    if "pyroki" in env_path.lower() and python_exe.exists():
                        return python_exe
        except:
            pass
        
        return None

    def _add_nvidia_paths(self, env: dict, python_exe: Path) -> None:
        """Add NVIDIA library paths to environment PATH for Windows JAX support."""
        if sys.platform != "win32":
            return
            
        # Assuming python_exe is in Scripts/ or bin/
        # site-packages is in Lib/site-packages relative to env root
        env_root = python_exe.parent.parent
        site_packages = env_root / "Lib" / "site-packages"
        
        if not site_packages.exists():
            return
            
        nvidia_dir = site_packages / "nvidia"
        if not nvidia_dir.exists():
            return
            
        # List of nvidia packages that might have bin dirs
        pkgs = [
            "cudnn", "cublas", "cuda_cupti", "cuda_nvcc", 
            "cuda_runtime", "cufft", "cusolver", "cusparse", "nvjitlink"
        ]
        
        new_paths = []
        for pkg in pkgs:
            bin_dir = nvidia_dir / pkg / "bin"
            if bin_dir.exists():
                new_paths.append(str(bin_dir))
        
        if new_paths:
            console.print(f"   Adding {len(new_paths)} NVIDIA paths to PATH for JAX")
            env["PATH"] = os.pathsep.join(new_paths) + os.pathsep + env.get("PATH", "")
    
    def step_convert(self, retargeted_files: Optional[List[Path]] = None) -> List[Path]:
        """Step 4: Convert retargeted motions to ProtoMotions .motion format."""
        from convert_retargeted_to_motion import convert_npz_to_motion
        
        console.print("\n[bold blue]Step 4: Converting to ProtoMotions format[/bold blue]")
        
        if retargeted_files is None:
            retargeted_files = list(self.config.retargeted_dir.glob("*_retargeted.npz"))
        
        successful_files = []
        
        for motion_file in retargeted_files:
            base_name = motion_file.stem.replace("_retargeted", "")
            output_file = self.config.motion_dir / f"{base_name}.motion"
            
            if output_file.exists() and not self.config.force_remake:
                console.print(f"   ⏭️ Skipping {motion_file.name} (already converted)")
                successful_files.append(output_file)
                continue
            
            # Find corresponding contact file
            contact_file = self.config.contacts_dir / f"{base_name}_contacts.npz"
            
            console.print(f"   Converting {motion_file.name}...")
            
            try:
                # Updated call signature to match convert_retargeted_to_motion.py
                # Note: Retargeted motions are at output_fps (keypoints are downsampled)
                success = convert_npz_to_motion(
                    npz_file=motion_file,
                    output_file=output_file,
                    model_xml=self.config.model_xml,
                    input_fps=self.config.output_fps,  # Retargeted data is at output_fps
                    output_fps=self.config.output_fps,
                    contact_file=contact_file if contact_file.exists() else None,
                    apply_motion_filter=self.config.apply_motion_filter,
                    height_offset=0.0,  # Ground the robot (feet on floor)
                )
                if success:
                    successful_files.append(output_file)
                    console.print(f"   ✅ Converted to {output_file.name}")
                else:
                    console.print(f"   ⚠️ Skipped {motion_file.name} (filtered out)")
            except Exception as e:
                console.print(f"   ❌ Error converting {motion_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        console.print(f"\n✅ Conversion completed. {len(successful_files)} files successful.")
        return successful_files
    
    def step_check(self) -> None:
        """Run validation checks on pipeline outputs."""
        import subprocess
        
        console.print("\n[bold blue]Step: Validation Checks[/bold blue]")
        
        # Check keypoints
        keypoint_files = list(self.config.keypoints_dir.glob("*.npy"))
        if keypoint_files:
            console.print(f"   Checking {len(keypoint_files)} keypoint files...")
            for kf in keypoint_files:
                cmd = [sys.executable, str(Path(__file__).parent / "check_pipeline_data.py"), "keypoints", str(kf)]
                subprocess.run(cmd)
        else:
            console.print("   ⚠️ No keypoint files found to check.")

        # Check retargeted
        retargeted_files = list(self.config.retargeted_dir.glob("*_retargeted.npz"))
        if retargeted_files:
            console.print(f"   Checking {len(retargeted_files)} retargeted files...")
            for rf in retargeted_files:
                cmd = [sys.executable, str(Path(__file__).parent / "check_pipeline_data.py"), "retargeted", str(rf), "--model-xml", str(self.config.model_xml)]
                subprocess.run(cmd)
        else:
            console.print("   ⚠️ No retargeted files found to check.")

    def step_filter(self, motion_files: List[Path]) -> List[Path]:
        """Step: Filter motions based on a configuration file."""
        console.print("\n[bold blue]Step: Filtering Motions[/bold blue]")

        if not self.config.filter_config:
            console.print("   ⏭️ Skipping motion filtering (no --filter-config provided).")
            return motion_files
        
        if not self.config.filter_config.exists():
            console.print(f"   [red]❌ Filter config not found: {self.config.filter_config}[/red]")
            return []

        try:
            from motion_filter import filter_motions
        except ImportError:
            console.print("[red]❌ Could not import 'motion_filter'. Make sure it is in the python path.[/red]")
            console.print(f"   Attempted to import from: {Path(__file__).parent.parent / 'data' / 'scripts'}")
            return []

        console.print(f"   Using filter configuration: {self.config.filter_config}")
        
        # Convert Path objects to strings for the filter function
        motion_file_paths = [str(p) for p in motion_files]
        
        # This function should return a list of file paths that passed the filter
        kept_files_str = filter_motions(
            motion_files=motion_file_paths,
            filter_config_path=str(self.config.filter_config)
        )
        
        kept_files = {Path(p) for p in kept_files_str}
        all_files = set(motion_files)
        discarded_files = all_files - kept_files

        console.print(f"   [green]Kept {len(kept_files)} files.[/green]")
        for file in sorted(list(discarded_files)):
            console.print(f"      [yellow]Discarded: {file.name}[/yellow]")
            
        console.print(f"\n✅ Motion filtering completed.")
        return sorted(list(kept_files))

    def step_package(self, motion_files: Optional[List[Path]] = None) -> Path:
        """Step 5: Package motion files into a MotionLib .pt file."""
        import yaml
        import torch
        
        console.print("\n[bold blue]Step 5: Packaging MotionLib[/bold blue]")
        
        if motion_files is None:
            motion_files = list(self.config.motion_dir.glob("*.motion"))
        
        if not motion_files:
            console.print("❌ No motion files found to package")
            raise ValueError("No motion files found")
        
        # Create YAML configuration
        yaml_file = self.config.yaml_dir / f"motions_fps_{self.config.model_name}.yaml"
        self._create_motion_yaml(motion_files, yaml_file)
        console.print(f"   ✅ Created motion YAML: {yaml_file}")
        
        # Package into MotionLib
        output_file = self.config.packaged_dir / f"{self.config.input_dir.name}.pt"
        self._package_motion_lib(yaml_file, output_file)
        console.print(f"   ✅ Packaged MotionLib: {output_file}")
        
        return output_file
    
    def _create_motion_yaml(self, motion_files: List[Path], output_file: Path) -> None:
        """Create a YAML file listing all motion files with their FPS."""
        import yaml
        import torch
        
        motions_list = []
        
        for idx, motion_path in enumerate(sorted(motion_files)):
            # Load motion to get duration
            try:
                motion_data = torch.load(motion_path, map_location="cpu", weights_only=False)
                if isinstance(motion_data, dict) and "rigid_body_pos" in motion_data:
                    num_frames = motion_data["rigid_body_pos"].shape[0]
                else:
                    num_frames = 100  # Default fallback
            except Exception:
                num_frames = 100
            
            duration = num_frames / self.config.output_fps
            
            motion_entry = {
                "file": str(motion_path.resolve().as_posix()),
                "fps": self.config.output_fps,
                "idx": idx,
                "sub_motions": [{
                    "idx": idx,
                    "timings": {
                        "start": 0.0,
                        "end": duration
                    },
                    "weight": 1.0
                }]
            }
            motions_list.append(motion_entry)
        
        yaml_data = {"motions": motions_list}
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
    
    def _package_motion_lib(self, yaml_file: Path, output_file: Path) -> None:
        """Package motion files into a MotionLib .pt file."""
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig
        
        # Create motion lib config
        config = MotionLibConfig(
            motion_file=str(yaml_file.resolve()),
            world_size=1,
        )
        
        # Create motion library - it will load all motions from the YAML
        console.print("   Loading motions into MotionLib...")
        mlib = MotionLib(config=config, device="cpu")
        
        # Save the motion library state
        output_file.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"   Saving {mlib.num_motions()} motions to {output_file}")
        mlib.save_to_file(str(output_file))
    
    def run(self, step: PipelineStep = PipelineStep.ALL) -> Optional[Path]:
        """Run the pipeline or specific step."""
        console.print(Panel.fit(
            "[bold green]🏃 Biomechanics Motion Processing Pipeline 🏃[/bold green]",
            title="BioMotions"
        ))
        console.print(f"📁 Input directory: {self.config.input_dir}")
        console.print(f"📂 Output directory: {self.config.output_dir}")
        console.print(f"🤖 Target model: {self.config.model_xml}")
        console.print(f"📊 Input FPS: {self.config.fps}, Output FPS: {self.config.output_fps}")
        
        if step == PipelineStep.OVERGROUND:
            self.step_overground()
        elif step == PipelineStep.KEYPOINTS:
            self.step_keypoints()
        elif step == PipelineStep.RETARGET:
            self.step_retarget()
        elif step == PipelineStep.CONVERT:
            self.step_convert()
        elif step == PipelineStep.CHECK:
            self.step_check()
        elif step == PipelineStep.PACKAGE:
            return self.step_package()
        elif step == PipelineStep.ALL:
            with Progress() as progress:
                # Step 1: Overground transformation
                task1 = progress.add_task("Transforming to overground...", total=100)
                overground_files = self.step_overground(progress, task1)
                
                if not overground_files:
                    console.print("❌ No files successfully transformed to overground")
                    return None
                
                # Step 2: Extract keypoints
                keypoint_files = self.step_keypoints(overground_files)
                
                if not keypoint_files:
                    console.print("❌ No keypoints successfully extracted")
                    return None
                
                # Step 3: Retarget (requires separate PyRoki environment)
                retargeted_files = self.step_retarget(keypoint_files)
                
                if not retargeted_files:
                    console.print("\n⚠️ Pipeline paused. Please run PyRoki retargeting manually.")
                    console.print("After retargeting, run: python pipeline.py ... --step convert")
                    return None
                
                # Step 4: Convert to ProtoMotions format
                motion_files = self.step_convert(retargeted_files)
                
                if not motion_files:
                    console.print("❌ No motion files successfully converted")
                    return None
                
                # Step 5: Package MotionLib
                output_file = self.step_package(motion_files)
                
                console.print("\n" + "=" * 60)
                console.print("[bold green]🎉 Pipeline completed successfully! 🎉[/bold green]")
                console.print(f"📦 Packaged MotionLib: {output_file}")
                
                return output_file
        
        return None


def get_model_for_height(
    height_cm: int,
    variant: str = "adjusted_pd",
    rescale_dir: Optional[Path] = None,
    contact_pads: bool = False,
) -> Tuple[Path, str]:
    """
    Get or create the model file for a given subject height.
    
    Args:
        height_cm: Subject height in centimeters
        variant: Model variant ('adjusted_pd' or 'adjusted_torque')
        rescale_dir: Directory containing model files
        
    Returns:
        Tuple of (model_xml_path, robot_name)
    """
    if rescale_dir is None:
        rescale_dir = Path(__file__).parent.parent / "rescale"
    
    # Construct expected filename
    base_name = f"smpl_humanoid_lower_body_{variant}"
    height_name = f"{base_name}_height_{height_cm}cm"
    model_stem = f"{height_name}{'_contact_pads' if contact_pads else ''}"
    model_path = rescale_dir / f"{model_stem}.xml"
    
    # Robot name for factory
    robot_name = f"smpl_lower_body_{height_cm}cm"
    if contact_pads:
        robot_name += "_contact_pads"
    if variant == "adjusted_torque":
        robot_name += "_torque"

    # Prefer the already-generated assets under protomotions/data/assets/mjcf
    assets_mjcf_dir = (
        Path(__file__).parent.parent
        / "protomotions"
        / "data"
        / "assets"
        / "mjcf"
    )
    assets_model_path = assets_mjcf_dir / f"{model_stem}.xml"
    if assets_model_path.exists():
        console.print(
            f"✅ Found model for {height_cm}cm in assets: {assets_model_path.name}"
        )
        return assets_model_path, robot_name
    
    # Check if model exists in rescale directory
    if model_path.exists():
        console.print(f"✅ Found model for {height_cm}cm: {model_path.name}")
        return model_path, robot_name
    
    # Try to create it
    console.print(f"⚠️ Model for {height_cm}cm not found, attempting to create...")
    
    try:
        # Import from the same directory as this file
        import sys
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        from quick_rescale import QuickRescaler
        
        rescaler = QuickRescaler(
            height_cm=height_cm,
            variant=variant,
            rescale_dir=rescale_dir,
        )
        
        rescaler.run(force_overwrite=False)

        # After attempting creation (or skipping because files exist), re-check.
        if model_path.exists():
            console.print(f"✅ Using model for {height_cm}cm: {model_path.name}")
            _create_robot_config_for_height(
                height_cm, variant, robot_name, contact_pads=contact_pads
            )
            return model_path, robot_name

        if assets_model_path.exists():
            console.print(
                f"✅ Using model for {height_cm}cm in assets: {assets_model_path.name}"
            )
            _create_robot_config_for_height(
                height_cm, variant, robot_name, contact_pads=contact_pads
            )
            return assets_model_path, robot_name
    except ImportError as e:
        console.print(f"   Could not import rescaling module: {e}")
    except Exception as e:
        console.print(f"   Rescaling failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fall back to base model
    base_model = rescale_dir / f"{base_name}.xml"
    console.print(f"⚠️ Using base model: {base_model.name}")
    return base_model, "smpl_lower_body"


def _create_robot_config_for_height(
    height_cm: int,
    variant: str,
    robot_name: str,
    contact_pads: bool = False,
) -> None:
    """
    Create a robot config for a given height using the factory.
    
    Args:
        height_cm: Subject height in centimeters
        variant: Model variant ('adjusted_pd' or 'adjusted_torque')
        robot_name: Robot configuration name
    """
    try:
        from protomotions.robot_configs.smpl_lower_body import SmplLowerBodyConfigFactory
        
        # Get the absolute path to the assets directory
        project_root = Path(__file__).parent.parent
        asset_root = str(project_root / "protomotions" / "data" / "assets")
        
        # Create config using the factory
        config = SmplLowerBodyConfigFactory.create(
            height_cm=height_cm,
            variant=variant,
            asset_root=asset_root,
            contact_pads=contact_pads,
        )
        
        console.print(f"✅ Created robot config for {height_cm}cm")
        console.print(f"   Root height: {config.default_root_height:.3f}m")
        
    except Exception as e:
        console.print(f"⚠️ Could not create robot config: {e}")
        console.print(f"   You may need to add {height_cm}cm to smpl_lower_body.py")


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ..., exists=True, help="Directory with treadmill motion .txt files"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    model_xml: Optional[Path] = typer.Option(
        None, "--model", "-m", exists=True, help="Path to MJCF model file"
    ),
    subject_height: Optional[int] = typer.Option(
        None, "--height", "-h",
        help="Subject height in cm (auto-selects/creates model)"
    ),
    model_variant: str = typer.Option(
        "adjusted_pd", "--variant", "-v",
        help="Model variant: adjusted_pd or adjusted_torque"
    ),
    fps: int = typer.Option(200, "--fps", "-f", help="Motion capture FPS"),
    output_fps: int = typer.Option(30, "--output-fps", help="Output FPS"),
    speed_override: Optional[float] = typer.Option(
        None, "--speed", "-s", help="Override treadmill speed (m/s)"
    ),
    coordinate_transform: str = typer.Option(
        "y_to_x_forward", "--transform", "-t",
        help="Coordinate transform: 'none' or 'y_to_x_forward'"
    ),
    auto_scale: bool = typer.Option(
        True, "--auto-scale/--no-auto-scale", help="Auto-scale motions"
    ),
    scale_override: Optional[float] = typer.Option(
        None, "--scale", help="Manual scale factor"
    ),
    force_remake: bool = typer.Option(
        False, "--force", help="Force reprocessing"
    ),
    step: PipelineStep = typer.Option(
        PipelineStep.ALL, "--step", help="Pipeline step to run"
    ),
    clean_intermediate: bool = typer.Option(
        False, "--clean", help="Remove intermediate files"
    ),
    pyroki_python: Optional[Path] = typer.Option(
        None, "--pyroki-python", help="Path to PyRoki Python interpreter"
    ),
    pyroki_urdf_path: Optional[Path] = typer.Option(
        None,
        "--pyroki-urdf-path",
        exists=True,
        help=(
            "Optional URDF to use for PyRoki retargeting (e.g. the contact-pad URDF). "
            "If omitted, PyRoki's script default is used."
        ),
    ),
    jax_platform: str = typer.Option(
        "cuda", "--jax-platform",
        help="JAX backend: 'cuda', 'rocm', 'cpu', or '' (auto-detect)"
    ),
    contact_pads: bool = typer.Option(
        False,
        "--contact-pads/--no-contact-pads",
        help=(
            "Use the *_contact_pads MJCF (when available) so conversion/packaging "
            "matches the smpl_lower_body_170cm_contact_pads robot."
        ),
    ),
) -> None:
    """
    Process treadmill motions to ProtoMotions MotionLib format.
    
    Supports automatic model selection by subject height:
    
        python pipeline.py ./treadmill_data/S02 ./processed_S02 --height 156
    
    Or manual model specification:
    
        python pipeline.py ./treadmill_data/S02 ./processed_S02 \\
            --model ./rescale/smpl_humanoid_lower_body_adjusted_pd.xml
    """
    # Determine model to use
    if subject_height is not None:
        # Auto-select or create model based on height
        model_xml, robot_name = get_model_for_height(
            height_cm=subject_height,
            variant=model_variant,
            contact_pads=contact_pads,
        )
        console.print(f"\n📏 Subject height: {subject_height}cm")
        console.print(f"🤖 Robot config: {robot_name}")
    elif model_xml is None:
        # No model specified - error
        console.print("❌ Must specify either --model or --height")
        console.print("\nExamples:")
        console.print("  --height 156              (auto-select 156cm model)")
        console.print("  --model path/to/model.xml (manual model path)")
        raise typer.Exit(1)
    
    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model_xml=model_xml,
        fps=fps,
        output_fps=output_fps,
        coordinate_transform=coordinate_transform,
        speed_override=speed_override,
        auto_scale=auto_scale,
        scale_override=scale_override,
        force_remake=force_remake,
        clean_intermediate=clean_intermediate,
        subject_height_cm=subject_height,
        model_variant=model_variant,
        pyroki_python=pyroki_python,
        pyroki_urdf_path=pyroki_urdf_path,
        jax_platform=jax_platform,
    )
    
    pipeline = BiomechanicsPipeline(config)
    pipeline.run(step)


if __name__ == "__main__":
    app()