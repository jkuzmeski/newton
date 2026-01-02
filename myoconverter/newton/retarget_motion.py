import warp as wp
import newton
import newton.ik as ik
import numpy as np
import os
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

class NewtonRetargeter:
    def __init__(self, model_path, target_bodies):
        """
        model_path: path to MJCF
        target_bodies: list of body names to track
        """
        print(f"Initializing NewtonRetargeter with model: {model_path}")
        builder = newton.ModelBuilder()
        builder.add_mjcf(model_path)
        self.model = builder.finalize()
        
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        
        self.target_bodies = target_bodies
        self.body_indices = []
        self.body_name_to_idx = {}
        
        # Build name map
        for i in range(self.model.body_count):
            key = self.model.body_key[i]
            self.body_name_to_idx[key] = i
            
        # Find indices
        for name in target_bodies:
            idx = self.body_name_to_idx.get(name)
            if idx is None:
                print(f"Warning: Body '{name}' not found in model.")
                self.body_indices.append(-1)
            else:
                self.body_indices.append(idx)
        
        # Setup IK
        self._setup_ik()

    def _setup_ik(self):
        self.pos_objs = []
        # Create objectives for each target
        for idx in self.body_indices:
            if idx == -1:
                # Placeholder for missing body
                continue
                
            # Initialize with current pos
            # We need to use wp.transform_get_translation
            # body_q is array of transforms
            # We can't easily read individual transforms during init for objective setup if we want to update them later
            # But IKPositionObjective takes target_positions as array.
            
            # We create objective with dummy target, will update in loop
            obj = ik.IKPositionObjective(
                link_index=idx,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([wp.vec3(0,0,0)], dtype=wp.vec3, device=self.model.device),
                weight=1.0
            )
            self.pos_objs.append(obj)
            
        # Joint limits
        self.limit_obj = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0
        )
        
        # Solver
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[*self.pos_objs, self.limit_obj],
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
            lambda_initial=0.1
        )
        
        dofs = self.model.joint_coord_count
        self.q_out = wp.zeros((1, dofs), dtype=float, device=self.model.device)
        self.current_q_buffer = wp.zeros((1, dofs), dtype=float, device=self.model.device)

    def retarget(self, target_positions_sequence):
        """
        target_positions_sequence: dict of body_name -> list/array of vec3 (T, 3)
        Returns: joint_q trajectory (T, joint_dim)
        """
        # Assume all sequences have same length
        first_key = list(target_positions_sequence.keys())[0]
        n_frames = len(target_positions_sequence[first_key])
        
        results = []
        
        print(f"Retargeting {n_frames} frames...")
        
        # Initial guess from current pose
        q_init = self.model.joint_q.numpy()
        self.current_q_buffer = wp.array(q_init.reshape(1, -1), dtype=float, device=self.model.device)
        
        for t in range(n_frames):
            # Update targets
            valid_obj_idx = 0
            for name, idx in zip(self.target_bodies, self.body_indices):
                if idx == -1: 
                    continue
                
                if name in target_positions_sequence:
                    pos = target_positions_sequence[name][t]
                    # Update objective
                    # set_target_position expects index in the batch (0 for us)
                    self.pos_objs[valid_obj_idx].set_target_position(0, wp.vec3(*pos))
                
                valid_obj_idx += 1
            
            # Solve
            self.solver.step(self.current_q_buffer, self.q_out, iterations=20)
            
            # Store result
            q_np = self.q_out.numpy()[0] # batch 0
            results.append(q_np.copy())
            
            # Use result as next guess
            wp.copy(self.current_q_buffer, self.q_out)
            
        results = np.array(results)
        
        # Smooth
        if savgol_filter is not None and n_frames > 5:
            print("Applying Savitzky-Golay smoothing...")
            # Apply per channel
            for i in range(results.shape[1]):
                try:
                    results[:, i] = savgol_filter(results[:, i], window_length=min(11, n_frames), polyorder=2)
                except ValueError:
                    pass
                    
        return results

if __name__ == "__main__":
    # Test
    model_file = os.path.join(os.path.dirname(__file__), 'gait2354_newton.xml')
    if not os.path.exists(model_file):
        print("Please generate model first.")
        exit(1)
        
    retargeter = NewtonRetargeter(model_file, ["pelvis", "toes_r", "toes_l"])
    
    # Generate dummy motion
    # Pelvis moves up/down sine wave
    # Feet stay roughly in place
    T = 50
    targets = {
        "pelvis": [],
        "toes_r": [],
        "toes_l": []
    }
    
    # Get initial positions
    state = retargeter.model.state()
    newton.eval_fk(retargeter.model, retargeter.model.joint_q, retargeter.model.joint_qd, state)
    body_q = state.body_q.numpy()
    
    p_idx = retargeter.body_name_to_idx["pelvis"]
    tr_idx = retargeter.body_name_to_idx["toes_r"]
    tl_idx = retargeter.body_name_to_idx["toes_l"]
    
    p0 = body_q[p_idx][:3]
    tr0 = body_q[tr_idx][:3]
    tl0 = body_q[tl_idx][:3]
    
    for t in range(T):
        offset = 0.1 * np.sin(t / 10.0)
        targets["pelvis"].append(p0 + np.array([0, 0, offset]))
        targets["toes_r"].append(tr0)
        targets["toes_l"].append(tl0)
        
    qs = retargeter.retarget(targets)
    print(f"Retargeted shape: {qs.shape}")
    print("Done.")
