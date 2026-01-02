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
from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    ControlInfo,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Gait2354RobotConfig(RobotConfig):
    """Configuration for the Gait2354 biomechanical humanoid model."""
    
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_right_foot_bodies": ["calcn_r", "toes_r"],
            "all_left_foot_bodies": ["calcn_l", "toes_l"],
            "all_left_hand_bodies": [],
            "all_right_hand_bodies": [],
            "head_body_name": ["torso"],
            "torso_body_name": ["torso"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "pelvis",
            "torso",
            "femur_r",
            "tibia_r",
            "calcn_r",
            "toes_r",
            "femur_l",
            "tibia_l",
            "calcn_l",
            "toes_l",
        ]
    )

    default_root_height: float = 0.95

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/gait2354_simplified.xml",
            usd_asset_file_name="usd/gait2354_simplified.usda",
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
                # Hip joints
                "hip_.*": ControlInfo(
                    stiffness=800,
                    damping=80,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                # Knee joints
                "knee_.*": ControlInfo(
                    stiffness=800,
                    damping=80,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                # Ankle joints
                "ankle_.*": ControlInfo(
                    stiffness=800,
                    damping=80,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                # Toe joints (metatarsophalangeal)
                "mtp_.*": ControlInfo(
                    stiffness=500,
                    damping=50,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                # Lumbar (torso) joints
                "lumbar_.*": ControlInfo(
                    stiffness=1000,
                    damping=100,
                    effort_limit=500,
                    velocity_limit=100,
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
