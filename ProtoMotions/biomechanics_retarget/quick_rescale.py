#!/usr/bin/env python3
"""
Quick Robot Rescaling Pipeline for SMPL Lower Body Model.

This script provides a streamlined way to:
1. Rescale the SMPL lower body XML/USDA to a new height
2. Copy assets to protomotions data/assets folders
3. Automatically register the configuration with ProtoMotions

Usage:
    # Rescale to 156cm and deploy to assets folder
    python quick_rescale.py --height 156
    
    # Rescale with torque control variant
    python quick_rescale.py --height 180 --variant adjusted_torque
    
    # Check what would be generated (dry run)
    python quick_rescale.py --height 165 --dry-run

Author: BioMotions Team
Date: 2025
"""

import sys
import argparse
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Add rescale module to path
SCRIPT_DIR = Path(__file__).parent.resolve()
RESCALE_DIR = SCRIPT_DIR.parent / "rescale"
if str(RESCALE_DIR) not in sys.path:
    sys.path.insert(0, str(RESCALE_DIR))

# Import rescaling utilities
try:
    from scaling_xml import copy_and_scale_xml
    from scaling_usda import extract_body_data_from_xml, update_usda_file
except ImportError as e:
    print(f"Warning: Could not import rescaling modules: {e}")
    print("XML/USDA generation will be skipped.")
    copy_and_scale_xml = None
    extract_body_data_from_xml = None
    update_usda_file = None


# Base height of the SMPL lower body model
BASE_HEIGHT_M = 1.70

# Default paths
DEFAULT_RESCALE_DIR = SCRIPT_DIR.parent / "rescale"
DEFAULT_ASSETS_DIR = SCRIPT_DIR.parent / "protomotions" / "data" / "assets"
DEFAULT_MJCF_DIR = DEFAULT_ASSETS_DIR / "mjcf"
DEFAULT_USD_DIR = DEFAULT_ASSETS_DIR / "usd"


class QuickRescaler:
    """Quick rescaling utility for SMPL lower body model."""
    
    def __init__(
        self,
        height_cm: int,
        variant: str = "adjusted_pd",
        rescale_dir: Optional[Path] = None,
        assets_dir: Optional[Path] = None,
        dry_run: bool = False,
        keep_in_rescale: bool = False,
    ):
        """
        Initialize the quick rescaler.
        
        Args:
            height_cm: Target height in centimeters
            variant: Model variant ('adjusted_pd' or 'adjusted_torque')
            rescale_dir: Directory containing base model files
            assets_dir: ProtoMotions assets directory (contains mjcf/ and usd/)
            dry_run: If True, only show what would be done
            keep_in_rescale: If True, also keep copies in rescale folder
        """
        self.height_cm = height_cm
        self.height_m = height_cm / 100.0
        self.variant = variant
        self.rescale_dir = Path(rescale_dir or DEFAULT_RESCALE_DIR)
        self.assets_dir = Path(assets_dir or DEFAULT_ASSETS_DIR)
        self.mjcf_dir = self.assets_dir / "mjcf"
        self.usd_dir = self.assets_dir / "usd"
        self.dry_run = dry_run
        self.keep_in_rescale = keep_in_rescale
        
        # Compute scale factor
        self.scale_factor = self.height_m / BASE_HEIGHT_M
        
        # Set up file names
        self.base_name = f"smpl_humanoid_lower_body_{variant}"
        self.height_suffix = f"_height_{height_cm}cm"
        self.output_name = f"{self.base_name}{self.height_suffix}"
        
        # Generated files tracking
        self.generated_files: Dict[str, Path] = {}
        
    def _get_input_xml(self) -> Path:
        """Get path to input XML file."""
        return self.rescale_dir / f"{self.base_name}.xml"
    
    def _get_input_usda(self) -> Path:
        """Get path to input USDA file."""
        return self.rescale_dir / f"{self.base_name}.usda"
    
    def _get_output_xml(self) -> Path:
        """Get path to output XML file in mjcf folder."""
        return self.mjcf_dir / f"{self.output_name}.xml"
    
    def _get_output_usda(self) -> Path:
        """Get path to output USDA file in usd folder."""
        return self.usd_dir / f"{self.output_name}.usda"
    
    def _get_rescale_xml(self) -> Path:
        """Get path to XML in rescale folder (for keeping copies)."""
        return self.rescale_dir / f"{self.output_name}.xml"
    
    def _get_rescale_usda(self) -> Path:
        """Get path to USDA in rescale folder (for keeping copies)."""
        return self.rescale_dir / f"{self.output_name}.usda"
    
    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """
        Check if all prerequisites are met.
        
        Returns:
            Tuple of (success, list of error messages)
        """
        errors = []
        
        # Check input files exist
        input_xml = self._get_input_xml()
        if not input_xml.exists():
            errors.append(f"Input XML not found: {input_xml}")
        
        input_usda = self._get_input_usda()
        if not input_usda.exists():
            errors.append(f"Input USDA not found: {input_usda}")
        
        # Check/create output directories
        for dir_path in [self.mjcf_dir, self.usd_dir]:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {dir_path}: {e}")
        
        return len(errors) == 0, errors
    
    def check_existing_files(self) -> Dict[str, Tuple[bool, Path]]:
        """
        Check which output files already exist.
        
        Returns:
            Dictionary mapping file type to (exists, path) tuples
        """
        return {
            "xml": (self._get_output_xml().exists(), self._get_output_xml()),
            "usda": (self._get_output_usda().exists(), self._get_output_usda()),
        }
    
    def rescale_xml(self) -> bool:
        """
        Rescale the XML file to the target height.
        
        Returns:
            True if successful
        """
        input_xml = self._get_input_xml()
        output_xml = self._get_output_xml()
        
        print(f"\n📐 Rescaling XML to {self.height_cm}cm...")
        print(f"   Input:  {input_xml.name}")
        print(f"   Output: {output_xml.name}")
        print(f"   Scale:  {self.scale_factor:.4f}x")
        
        if self.dry_run:
            print("   [DRY RUN] Would create XML file")
            self.generated_files["xml"] = output_xml
            return True
        
        if copy_and_scale_xml is None:
            print("   ⚠️  Scaling module not available, skipping XML generation")
            return False
        
        try:
            # Use the scaling module - generate to mjcf folder
            result = copy_and_scale_xml(
                str(input_xml),
                self.height_m,
                self.height_suffix,
                output_dir=str(self.mjcf_dir),
            )
            
            if result:
                # Rename to expected output name if needed
                result_path = Path(result)
                if result_path != output_xml:
                    if output_xml.exists():
                        output_xml.unlink()
                    shutil.move(str(result_path), str(output_xml))
                
                self.generated_files["xml"] = output_xml
                print(f"   ✅ Created: {output_xml.name}")
                return True
            else:
                print("   ❌ Failed to create XML file")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def generate_usda(self) -> bool:
        """
        Generate USDA file from the scaled XML.
        
        Returns:
            True if successful
        """
        xml_path = self.generated_files.get("xml", self._get_output_xml())
        ref_usda = self._get_input_usda()
        output_usda = self._get_output_usda()
        
        print(f"\n🔧 Generating USDA from scaled XML...")
        print(f"   Reference: {ref_usda.name}")
        print(f"   Output:    {output_usda.name}")
        
        if self.dry_run:
            print("   [DRY RUN] Would create USDA file")
            self.generated_files["usda"] = output_usda
            return True
        
        if extract_body_data_from_xml is None or update_usda_file is None:
            print("   ⚠️  USDA module not available, skipping USDA generation")
            return False
        
        if not xml_path.exists():
            print(f"   ❌ Scaled XML not found: {xml_path}")
            return False
        
        try:
            # Extract body data from scaled XML
            body_data, hierarchy = extract_body_data_from_xml(str(xml_path))
            
            if not body_data or not hierarchy:
                print("   ❌ Failed to extract body data from XML")
                return False
            
            # Update USDA file
            update_usda_file(str(ref_usda), body_data, hierarchy, str(output_usda))
            
            self.generated_files["usda"] = output_usda
            print(f"   ✅ Created: {output_usda.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def print_config_registration_info(self) -> None:
        """Print information about how to use the new configuration."""
        robot_name = f"smpl_lower_body_{self.height_cm}cm"
        if self.variant == "adjusted_torque":
            robot_name += "_torque"
        
        root_height = 0.95 * self.scale_factor
        
        print(f"\n📝 Robot Configuration Info:")
        print(f"   Robot name: {robot_name}")
        print(f"   Root height: {root_height:.3f}m")
        print()
        print("   Usage in code:")
        print(f'   >>> from protomotions.robot_configs.factory import robot_config')
        print(f'   >>> config = robot_config("{robot_name}")')
        print()
        print("   Usage in training:")
        print(f"   --robot {robot_name}")
        print()
        print("   The factory will automatically create the configuration")
        print("   with the correct asset paths and scaled parameters.")
    
    def run(self, force_overwrite: bool = False) -> bool:
        """
        Run the complete rescaling pipeline.
        
        Args:
            force_overwrite: If True, overwrite existing files
            
        Returns:
            True if successful
        """
        print(f"=" * 60)
        print(f"Quick Rescale: SMPL Lower Body to {self.height_cm}cm")
        print(f"=" * 60)
        print(f"Variant: {self.variant}")
        print(f"Scale factor: {self.scale_factor:.4f}")
        if self.dry_run:
            print("[DRY RUN MODE - No files will be modified]")
        
        # Check prerequisites
        success, errors = self.check_prerequisites()
        if not success:
            print("\n❌ Prerequisites check failed:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        # Check existing files
        existing = self.check_existing_files()
        any_exist = any(exists for exists, _ in existing.values())
        
        if any_exist and not force_overwrite:
            print("\n⚠️  Some files already exist:")
            for file_type, (exists, path) in existing.items():
                if exists:
                    print(f"   - {path.name}")
            
            if not self.dry_run:
                response = input("\nOverwrite? (y/N): ").strip().lower()
                if response != "y":
                    print("Aborted.")
                    return False
        
        # Step 1: Rescale XML
        if not self.rescale_xml():
            print("\n❌ XML rescaling failed")
            return False
        
        # Step 2: Generate USDA
        if not self.generate_usda():
            print("\n⚠️  USDA generation failed (non-critical)")
        
        # Print registration info
        self.print_config_registration_info()
        
        print(f"\n{'=' * 60}")
        print("✅ Rescaling complete!")
        print(f"{'=' * 60}")
        
        return True


def list_available_heights() -> List[int]:
    """
    List all available height-scaled models in the rescale directory.
    
    Returns:
        List of heights in centimeters
    """
    heights = []
    pattern = re.compile(r"smpl_humanoid_lower_body_.*_height_(\d+)cm\.xml")
    
    rescale_dir = DEFAULT_RESCALE_DIR
    if rescale_dir.exists():
        for f in rescale_dir.iterdir():
            match = pattern.match(f.name)
            if match:
                heights.append(int(match.group(1)))
    
    return sorted(set(heights))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick rescaling pipeline for SMPL lower body model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Rescale to 156cm
    python quick_rescale.py --height 156
    
    # Rescale with torque control variant  
    python quick_rescale.py --height 180 --variant adjusted_torque
    
    # Preview what would be done
    python quick_rescale.py --height 165 --dry-run
    
    # List existing height variants
    python quick_rescale.py --list
    
    # Force overwrite existing files
    python quick_rescale.py --height 156 --force
        """
    )
    
    parser.add_argument(
        "--height",
        type=int,
        help="Target height in centimeters (e.g., 156, 170, 180)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["adjusted_pd", "adjusted_torque"],
        default="adjusted_pd",
        help="Model variant (default: adjusted_pd)"
    )
    parser.add_argument(
        "--rescale-dir",
        type=str,
        default=None,
        help="Directory containing base model files"
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=None,
        help="ProtoMotions data/assets directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available height variants"
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        heights = list_available_heights()
        print("Available height variants in rescale directory:")
        if heights:
            for h in heights:
                print(f"  - {h}cm")
        else:
            print("  (none found)")
        
        print("\nTo use a height variant:")
        print('  robot_config("smpl_lower_body_XXXcm")')
        return 0
    
    # Rescale mode requires height
    if args.height is None:
        parser.error("--height is required (or use --list to see available heights)")
    
    # Validate height
    if args.height < 100 or args.height > 250:
        print(f"Error: Height {args.height}cm seems unrealistic.")
        print("Please use a value between 100 and 250 cm.")
        return 1
    
    # Run rescaling
    rescaler = QuickRescaler(
        height_cm=args.height,
        variant=args.variant,
        rescale_dir=args.rescale_dir,
        assets_dir=args.assets_dir,
        dry_run=args.dry_run,
    )
    
    success = rescaler.run(force_overwrite=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
