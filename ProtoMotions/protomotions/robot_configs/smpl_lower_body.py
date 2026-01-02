# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
SMPL Humanoid Lower Body Robot Configuration.

This module provides configurations for the SMPL lower body humanoid model,
supporting height-scaled variants for biomechanics research.

The lower body model includes:
- Pelvis (root body with freejoint)
- L_Hip, L_Knee, L_Ankle, L_Toe (left leg chain)
- R_Hip, R_Knee, R_Ankle, R_Toe (right leg chain)

Each joint has 3 DOF (x, y, z axes) for a total of 24 DOF.
"""

from protomotions.robot_configs.base import (
    RobotAssetConfig,
    RobotConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimParams,
    IsaacLabPhysXParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# Base dimensions for the SMPL lower body model.
BASE_MODEL_HEIGHT_M = 1.70
BASE_HEIGHT_CM = 170
BASE_ROOT_HEIGHT_M = 0.95

# Default PD gains for lower body joints
# These values are tuned for stable locomotion
HIP_STIFFNESS = 250.0
HIP_DAMPING = 25.0
KNEE_STIFFNESS = 200.0
KNEE_DAMPING = 20.0
ANKLE_STIFFNESS = 150.0
ANKLE_DAMPING = 15.0
TOE_STIFFNESS = 75.0
TOE_DAMPING = 8.0


def compute_root_height(height_cm: int) -> float:
    """
    Compute the default root height for a given body height.
    
    The root height is approximately the pelvis height when standing,
    which scales proportionally with body height.
    
    Args:
        height_cm: Body height in centimeters
        
    Returns:
        Default root height in meters
    """
    if height_cm <= 0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")

    scale_factor = height_cm / float(BASE_HEIGHT_CM)
    return BASE_ROOT_HEIGHT_M * scale_factor


def _asset_base_name(variant: str) -> str:
    if variant == "base":
        return "smpl_humanoid_lower_body"
    if variant == "adjusted_torque":
        return "smpl_humanoid_lower_body_adjusted_torque"
    if variant == "adjusted_pd":
        return "smpl_humanoid_lower_body_adjusted_pd"

    raise ValueError(
        (
            f"Invalid variant: {variant}. Supported variants: "
            "adjusted_pd, adjusted_torque, base"
        )
    )


def get_asset_paths(
    height_cm: Optional[int] = None,
    variant: str = "adjusted_pd",
) -> Dict[str, str]:
    """
    Get asset file paths for a given height variant.
    
    Args:
        height_cm: Height in centimeters (None for base model)
        variant: Model variant ('adjusted_pd', 'adjusted_torque', or 'base')
        
    Returns:
        Dictionary with 'mjcf' and 'usd' keys
    """
    base_name = _asset_base_name(variant)
    
    if height_cm is not None:
        if height_cm <= 0:
            raise ValueError(f"height_cm must be positive, got {height_cm}")
        suffix = f"_height_{height_cm}cm"
    else:
        suffix = ""
    
    return {
        "mjcf": f"mjcf/{base_name}{suffix}.xml",
        "usd": f"usd/{base_name}{suffix}.usda",
    }


@dataclass
class SmplLowerBodyConfig(RobotConfig):
    """
    Configuration for SMPL humanoid lower body model.
    
    This is the base configuration for the 170cm model. For height-scaled
    variants, use SmplLowerBodyConfigFactory.create() instead.
    """
    
    # Lower body specific body mappings
    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Pelvis",
            "L_Ankle",
            "R_Ankle",
            "L_Toe",
            "R_Toe",
        ]
    )
    
    non_termination_contact_bodies: List[str] = field(
        default_factory=lambda: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
    )

    contact_bodies: List[str] = field(
        default_factory=lambda: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
    )

    imu_body: Optional[str] = "Pelvis"

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["L_Ankle", "L_Toe"],
            "all_right_foot_bodies": ["R_Ankle", "R_Toe"],
            "all_left_hand_bodies": [],  # No hands in lower body model
            "all_right_hand_bodies": [],
            "head_body_name": [],  # No head in lower body model
            "torso_body_name": ["Pelvis"],  # Pelvis acts as torso
        }
    )
    
    # Default root height for 170cm model
    default_root_height: float = 0.95

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="protomotions/data/assets",
            # NOTE: The 170cm lower-body assets in this repo use the explicit
            # "_height_170cm" suffix.
            asset_file_name=(
                "mjcf/smpl_humanoid_lower_body_adjusted_pd_height_170cm.xml"
            ),
            usd_asset_file_name=(
                "usd/smpl_humanoid_lower_body_adjusted_pd_height_170cm.usda"
            ),
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                # Hip joints - highest stiffness for stability
                ".*_Hip_.*": ControlInfo(
                    stiffness=HIP_STIFFNESS,
                    damping=HIP_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                # Knee joints
                ".*_Knee_.*": ControlInfo(
                    stiffness=KNEE_STIFFNESS,
                    damping=KNEE_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                # Ankle joints
                ".*_Ankle_.*": ControlInfo(
                    stiffness=ANKLE_STIFFNESS,
                    damping=ANKLE_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                # Toe joints - lowest stiffness
                ".*_Toe_.*": ControlInfo(
                    stiffness=TOE_STIFFNESS,
                    damping=TOE_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=8,
                    max_depenetration_velocity=2,
                ),
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=4,
            ),
        )
    )


class SmplLowerBodyConfigFactory:
    """
    Factory for creating height-scaled SMPL lower body configurations.
    
    This factory enables quick creation of robot configurations for
    different subject heights without manually defining each variant.
    
    Usage:
        # Create config for 156cm subject
        config = SmplLowerBodyConfigFactory.create(height_cm=156)
        
        # Create config with custom asset root
        config = SmplLowerBodyConfigFactory.create(
            height_cm=180,
            asset_root="protomotions/data/assets"
        )
    """
    
    # Registry of known height configurations
    _registry: Dict[int, type] = {}
    
    @classmethod
    def create(
        cls,
        height_cm: int,
        variant: str = "adjusted_pd",
        asset_root: str = "protomotions/data/assets",
        contact_pads: bool = False,
    ) -> SmplLowerBodyConfig:
        """
        Create a height-scaled SMPL lower body configuration.
        
        Args:
            height_cm: Subject height in centimeters
            variant: Model variant ('adjusted_pd', 'adjusted_torque')
            asset_root: Root directory for robot assets
            
        Returns:
            Configured SmplLowerBodyConfig instance
        """
        if height_cm <= 0:
            raise ValueError(f"height_cm must be positive, got {height_cm}")
        if variant == "torque":
            variant = "adjusted_torque"

        root_height = compute_root_height(height_cm)

        # Get asset paths
        base_name = _asset_base_name(variant)

        if contact_pads:
            if height_cm != 170 or variant != "adjusted_pd":
                raise ValueError(
                    "Contact pads are currently only available for the "
                    "170cm adjusted_pd SMPL lower-body asset. "
                    f"Requested height_cm={height_cm}, variant={variant}."
                )
            mjcf_file = (
                f"mjcf/{base_name}_height_{height_cm}cm_contact_pads.xml"
            )
            usd_file = (
                f"usd/{base_name}_height_{height_cm}cm_contact_pads.usda"
            )
        else:
            mjcf_file = f"mjcf/{base_name}_height_{height_cm}cm.xml"
            usd_file = f"usd/{base_name}_height_{height_cm}cm.usda"
        
        # Create the asset config first
        asset_config = RobotAssetConfig(
            asset_root=asset_root,
            asset_file_name=mjcf_file,
            usd_asset_file_name=usd_file,
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            self_collisions=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
        
        # Create configuration directly with the correct asset
        config_cls = (
            SmplLowerBody170cmContactPadsConfig
            if contact_pads
            else SmplLowerBodyConfig
        )
        config = config_cls(
            default_root_height=root_height,
            asset=asset_config,
        )
        
        return config
    
    @classmethod
    def register(cls, height_cm: int, config_class: type) -> None:
        """Register a custom configuration class for a specific height."""
        cls._registry[height_cm] = config_class
    
    @classmethod
    def get_registered_heights(cls) -> List[int]:
        """Get list of registered height variants."""
        return list(cls._registry.keys())


# Pre-defined height configurations for common subject heights
@dataclass
class SmplLowerBody156cmConfig(SmplLowerBodyConfig):
    """Configuration for 156cm subject."""
    
    default_root_height: float = field(
        default_factory=lambda: compute_root_height(156)
    )
    
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="protomotions/data/assets",
            asset_file_name=(
                "mjcf/smpl_humanoid_lower_body_adjusted_pd_height_156cm.xml"
            ),
            usd_asset_file_name=(
                "usd/smpl_humanoid_lower_body_adjusted_pd_height_156cm.usda"
            ),
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            self_collisions=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )


@dataclass
class SmplLowerBody170cmConfig(SmplLowerBodyConfig):
    """Configuration for 170cm subject (base height)."""
    pass  # Uses default values


@dataclass
class SmplLowerBody170cmContactPadsConfig(SmplLowerBodyConfig):
    """Configuration for 170cm subject with foot contact pads for COP/GRF."""

    non_termination_contact_bodies: List[str] = field(
        default_factory=lambda: [
            "L_Heel",
            "L_MetMedial",
            "L_MetLateral",
            "L_ToeTip",
            "R_Heel",
            "R_MetMedial",
            "R_MetLateral",
            "R_ToeTip",
        ]
    )

    contact_bodies: List[str] = field(
        default_factory=lambda: [
            "L_Heel",
            "L_MetMedial",
            "L_MetLateral",
            "L_ToeTip",
            "R_Heel",
            "R_MetMedial",
            "R_MetLateral",
            "R_ToeTip",
        ]
    )

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": [
                "L_Heel",
                "L_MetMedial",
                "L_MetLateral",
                "L_ToeTip",
            ],
            "all_right_foot_bodies": [
                "R_Heel",
                "R_MetMedial",
                "R_MetLateral",
                "R_ToeTip",
            ],
            "all_left_hand_bodies": [],
            "all_right_hand_bodies": [],
            "head_body_name": [],
            "torso_body_name": ["Pelvis"],
        }
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="protomotions/data/assets",
            asset_file_name=(
                "mjcf/smpl_humanoid_lower_body_adjusted_pd_"
                "height_170cm_contact_pads.xml"
            ),
            usd_asset_file_name=(
                "usd/smpl_humanoid_lower_body_adjusted_pd_"
                "height_170cm_contact_pads.usda"
            ),
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            self_collisions=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )


@dataclass
class SmplLowerBody180cmConfig(SmplLowerBodyConfig):
    """Configuration for 180cm subject."""
    
    default_root_height: float = field(
        default_factory=lambda: compute_root_height(180)
    )
    
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="protomotions/data/assets",
            asset_file_name=(
                "mjcf/smpl_humanoid_lower_body_adjusted_pd_height_180cm.xml"
            ),
            usd_asset_file_name=(
                "usd/smpl_humanoid_lower_body_adjusted_pd_height_180cm.usda"
            ),
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            self_collisions=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )


@dataclass
class SmplLowerBody195cmConfig(SmplLowerBodyConfig):
    """Configuration for 195cm subject."""
    
    default_root_height: float = field(
        default_factory=lambda: compute_root_height(195)
    )
    
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="protomotions/data/assets",
            asset_file_name=(
                "mjcf/smpl_humanoid_lower_body_adjusted_pd_height_195cm.xml"
            ),
            usd_asset_file_name=(
                "usd/smpl_humanoid_lower_body_adjusted_pd_height_195cm.usda"
            ),
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            self_collisions=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )


# Register pre-defined configurations
SmplLowerBodyConfigFactory.register(170, SmplLowerBody170cmConfig)

