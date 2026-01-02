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
from protomotions.robot_configs.base import RobotConfig
import re


def _parse_smpl_lower_body_height(robot_name: str) -> tuple:
    """
    Parse SMPL lower body robot name to extract height and variant.
    
    Supported formats:
        - smpl_lower_body          -> (None, "adjusted_pd")
        - smpl_lower_body_156cm    -> (156, "adjusted_pd")
        - smpl_lower_body_180cm_torque -> (180, "adjusted_torque")
        
    Returns:
        Tuple of (height_cm: int or None, variant: str)
    """
    # Base case: smpl_lower_body
    if robot_name == "smpl_lower_body":
        return (None, "adjusted_pd")
    
    # Pattern: smpl_lower_body_XXXcm or smpl_lower_body_XXXcm_variant
    match = re.match(r"smpl_lower_body_(\d+)cm(?:_(\w+))?", robot_name)
    if match:
        height_cm = int(match.group(1))
        variant = match.group(2) if match.group(2) else "adjusted_pd"
        if variant == "torque":
            variant = "adjusted_torque"
        return (height_cm, variant)
    
    return (None, None)  # Invalid format


def robot_config(robot_name: str, **updates) -> RobotConfig:
    """Factory function to create robot configuration based on robot type.

    Args:
        robot_name: Name of the robot type. Supported types:
            - smpl, smplx, amp, g1, h1_2, rigv1
            - smpl_lower_body (base 170cm model)
            - smpl_lower_body_XXXcm (height-scaled, e.g., smpl_lower_body_156cm)
            - smpl_lower_body_XXXcm_torque (torque control variant)
        **updates: Optional field updates to apply to the robot config

    Returns:
        RobotConfig: Robot configuration object

    Raises:
        ValueError: If robot_name is not recognized
    """
    # Handle SMPL lower body variants (with optional height suffix)
    if robot_name.startswith("smpl_lower_body"):
        contact_pads = False
        if robot_name.endswith("_contact_pads"):
            contact_pads = True
            robot_name = robot_name[: -len("_contact_pads")]

        height_cm, variant = _parse_smpl_lower_body_height(robot_name)
        
        if variant is None:
            raise ValueError(
                f"Invalid smpl_lower_body format: {robot_name}. "
                "Use 'smpl_lower_body' or 'smpl_lower_body_XXXcm'"
            )
        
        from protomotions.robot_configs.smpl_lower_body import (
            SmplLowerBodyConfig,
            SmplLowerBodyConfigFactory,
        )
        
        if height_cm is None:
            # Base model (170cm)
            if contact_pads:
                config = SmplLowerBodyConfigFactory.create(
                    height_cm=170,
                    variant=variant,
                    contact_pads=True,
                )
            else:
                config = SmplLowerBodyConfig()
        else:
            # Height-scaled model
            config = SmplLowerBodyConfigFactory.create(
                height_cm=height_cm,
                variant=variant,
                contact_pads=contact_pads,
            )
    elif robot_name == "smpl":
        from protomotions.robot_configs.smpl import SmplRobotConfig

        config = SmplRobotConfig()
    elif robot_name == "smplx":
        from protomotions.robot_configs.smplx import SMPLXRobotConfig

        config = SMPLXRobotConfig()
    elif robot_name == "amp":
        from protomotions.robot_configs.amp import AMPRobotConfig

        config = AMPRobotConfig()
    elif robot_name == "g1":
        from protomotions.robot_configs.g1 import G1RobotConfig

        config = G1RobotConfig()
    elif robot_name == "h1_2":
        from protomotions.robot_configs.h1_2 import H1_2RobotConfig

        config = H1_2RobotConfig()
    elif robot_name == "rigv1":
        from protomotions.robot_configs.rigv1 import Rigv1RobotConfig

        config = Rigv1RobotConfig()
    else:
        raise ValueError(f"Invalid robot name: {robot_name}")

    # Apply any updates
    if updates:
        config.update_fields(**updates)

    return config
