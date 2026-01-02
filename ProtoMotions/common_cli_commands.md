python scripts/analyze_biomechanics.py D:\Isaac\BioMotions\results\test_training\biomechanics\test_imu\biomechanics_data.npz

python protomotions/inference_biomechanics.py --simulator isaaclab --checkpoint results/mimic_base/last.ckpt --num-envs 1 --experiment-name test_imu --overrides env.max_episode_length=200

python protomotions/inference_agent.py --simulator newton --checkpoint results/test_training/epoch_30000.ckpt

python pipeline.py ./treadmill_data/S02_long  ./processed_data/S02_170_long_200fps --height 170 --contact-pads --fps 200 --output-fps 200 --pyroki-python "D:\Isaac\pyroki_env\Scripts\python.exe" --pyroki-urdf-path ../protomotions/data/assets/urdf/for_retargeting/smpl_lower_body.urdf


python protomotions/train_agent.py --robot-name smpl_lower_body_170cm  --simulator isaaclab  --experiment-path examples/experiments/mimic/mlp.py  --experiment-name test_training  --motion-file biomechanics_retarget/processed_data/S02_170/packaged_data/S02.pt  --num-envs 64  --batch-size 256 ^ --use-wandb      

python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/add/mlp.py --experiment-name add_200fps_torque --motion-file biomechanics_retarget\processed_data\S02_long_170_200fps\packaged_data\S02_long.pt --num-envs 4096 --batch-size 16384 --use-wandb

python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/mimic/transformer.py --experiment-name mimic_base_mesh_v2 --motion-file biomechanics_retarget\processed_data\S02_170_long\packaged_data\S02_long.pt --num-envs 1024 --batch-size 4096  --overrides robot.asset.use_mesh_collisions=true robot.asset.mesh_collision_bodies="['L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe']" --max-epochs 20000 --use-wandb

python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/masked_mimic/transformer.py --experiment-name mimic_synth_test --motion-file biomechanics_retarget\processed_data\S02_170_long\packaged_data\S02_long.pt --checkpoint results\mimic_base\score_based.ckpt --num-envs 512 --batch-size 2048

# Step 1: Train Expert (Standard Mimic) on Synthetic Data
# This trains a standard tracker to perfectly follow the synthetic motions.
# It uses the default config: future_steps=20, type=max-coords-future-rel
python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/mimic/transformer.py --experiment-name mimic_synth_expert --motion-file data/motions/synthetic/straight_line_motions.yaml --num-envs 512 --batch-size 2048

# Step 2: Train Masked Mimic on Synthetic Data (using Expert from Step 1)
# Note: Wait for Step 1 to finish and produce 'results/mimic_synth_expert/score_based.ckpt'
# We updated the experiment config to match the expert, so we don't need the complex overrides anymore.
python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/masked_mimic/transformer.py --experiment-name mimic_synth_pelvis_xy --motion-file data/motions/synthetic/straight_line_motions.yaml --num-envs 512 --batch-size 2048 --overrides agent.expert_model_path=results/mimic_synth_expert/score_based.ckpt env.masked_mimic_obs.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning="[{'body_name': 'Pelvis', 'constraint_state': 3}]"

# Alternative: Test Z-Masking on Real Data (using existing mimic_base expert)
# This uses your existing expert trained on S02_long.pt
python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/masked_mimic/transformer.py --experiment-name mimic_real_pelvis_xy --motion-file biomechanics_retarget\processed_data\S02_170_long\packaged_data\S02_long.pt --num-envs 1024 --batch-size 4096 --overrides agent.expert_model_path=results/mimic_base/score_based.ckpt env.masked_mimic_obs.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning="[{'body_name': 'Pelvis', 'constraint_state': 3}]" --use-wandb

# Inference/Evaluation for Masked Mimic (Pelvis XY conditioning)
python protomotions/inference_agent.py --simulator isaaclab --checkpoint results\mimic_real_pelvis_xy\last.ckpt --motion-file data\motions\synthetic\straight_d10.0m_v3.0ms_h0deg.motion --num-envs 1 --overrides env.masked_mimic_obs.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning="[{'body_name': 'Pelvis', 'constraint_state': 3}]"