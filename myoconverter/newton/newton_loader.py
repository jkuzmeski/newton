import newton
import warp as wp
import os
import argparse

def load_and_validate(mjcf_path):
    builder = newton.ModelBuilder()
    
    # Load MJCF
    print(f"Loading MJCF from: {mjcf_path}")
    try:
        builder.add_mjcf(mjcf_path)
    except Exception as e:
        print(f"Failed to load MJCF: {e}")
        return

    model = builder.finalize()
    print("Model finalized.")
    
    # State
    state = model.state()
    
    # Forward Kinematics
    # joint_q and joint_qd are usually initialized to 0 or defaults
    # eval_fk(model, q, qd, state)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    
    print("Forward kinematics evaluated successfully.")
    
    print(f"Number of bodies: {model.body_count}")
    print(f"Number of joints: {model.joint_count}")
    print(f"Number of DOFs: {model.joint_dof_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_path = os.path.join(os.path.dirname(__file__), 'gait2354_newton.xml')
    parser.add_argument('--path', default=default_path, help='Path to MJCF file')
    args = parser.parse_args()
    
    if os.path.exists(args.path):
        load_and_validate(args.path)
    else:
        print(f"File not found: {args.path}. Please generate it first.")
