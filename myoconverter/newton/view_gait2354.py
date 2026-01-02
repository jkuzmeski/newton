# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

###########################################################################
# View Gait2354 Newton Model
#
# Shows how to set up a simulation of the converted Gait2354 model.
#
# Command: python myoconverter/newton/view_gait2354.py --num-worlds 1
#
###########################################################################

import os
import warp as wp

import newton
import newton.examples
from newton.geometry import SDFHydroelasticConfig


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        # Load the Gait2354 model
        gait = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(gait)
        
        # Default joint and shape configs
        gait.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        
        # Enable hydroelastic contacts on shapes
        gait.default_shape_cfg.ke = 1.0e5  # Standard contact stiffness
        gait.default_shape_cfg.kd = 1.0e3  # Increased damping
        gait.default_shape_cfg.kf = 1.0e3
        gait.default_shape_cfg.mu = 0.75
        gait.default_shape_cfg.is_hydroelastic = True  # Enable hydroelastic contacts
        gait.default_shape_cfg.k_hydro = 1.0e11  # Hydroelastic contact stiffness
        gait.default_shape_cfg.sdf_max_resolution = 64  # SDF grid resolution

        # Path to the generated MJCF
        mjcf_path = os.path.join(os.path.dirname(__file__), "gait2354_newton.xml")
        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(f"MJCF not found at {mjcf_path}. Please run generate_mjcf.py first.")

        # Add the MJCF to the builder
        gait.add_mjcf(
            mjcf_path,
            ignore_names=["ground"],
            xform=wp.transform(wp.vec3(0, 0, 1.05)),  # Higher to ensure feet clear ground
            floating=True # Pelvis has 6-DOF joint usually
        )

        # Disable PD control - let the model flop naturally
        for i in range(len(gait.joint_target_ke)):
            gait.joint_target_ke[i] = 1000.0  # No stiffness
            gait.joint_target_kd[i] = 500  # Some damping to prevent explosion

        # Build the final world
        builder = newton.ModelBuilder()
        builder.replicate(gait, self.num_worlds)
        builder.add_ground_plane()

        self.model = builder.finalize()
        use_mujoco_contacts = args.use_mujoco_contacts if args is not None else False
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=1000,
            nconmax=250,
            use_mujoco_contacts=use_mujoco_contacts,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create hydroelastic collision pipeline
        sdf_hydroelastic_config = SDFHydroelasticConfig(
            output_contact_surface=True,  # Enable contact surface visualization
            reduce_contacts=True,  # Use fast discrete approximation
        )
        
        # Use CollisionPipelineUnified with hydroelastic support
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            rigid_contact_max_per_pair=100,
            broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Skip contact logging for now due to CUDA memory issues
        # You can still see contacts by checking the collision shapes
        # self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds, args)

    newton.examples.run(example, args)
