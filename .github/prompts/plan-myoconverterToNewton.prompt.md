## Plan: Convert Myoconverter to Newton (Joint-Focused, CSV Input)

Create a simplified converter that takes CSV joint center definitions, produces Newton-compatible MuJoCo XML for the Gait2354 model (preserving moving path points), and enables motion retargeting for ProtoMotions training.

### Steps

1. **Define CSV joint definition schema** in [myoconverter/newton/](myoconverter/newton/) - Create a CSV format with columns: `body_name, parent_body, joint_type, joint_axis, range_min, range_max, pos_x, pos_y, pos_z, is_path_point` to capture Gait2354's kinematic tree including moving path point bodies.

2. **Extract Gait2354 joint structure to CSV** - Parse [gait2354_cvt1.xml](myoconverter/models/mjc/Gait2354Simbody/gait2354_cvt1.xml) to generate the reference CSV, preserving all 54+ bodies including `grac_r_grac_r-P2` style moving path point bodies for future muscle work.

3. **Build Newton model loader/validator** - Create `newton_loader.py` using Newton's `ModelBuilder.add_mjcf()` to load generated XMLs and validate joint ranges via `eval_fk()`, following patterns in [newton/examples/](newton/examples/).

4. **Implement CSV-to-MJCF generator** - Adapt [converter.py](myoconverter/myoconverter/xml/converter.py) to generate MuJoCo XML from CSV input, stripping OpenSim dependencies while preserving the body hierarchy and moving path point structure.

5. **Create ProtoMotions robot config** - Add a `gait2354.py` config to [protomotions/robot_configs/](ProtoMotions/protomotions/robot_configs/) defining keypoint mappings, DOF limits, and contact bodies for the biomechanical model.

6. **Build Newton IK-based retargeting module** - Implement trajectory retargeting using Newton's `IKSolver` with `IKPositionObjective` for anatomical landmarks, adding temporal smoothing as post-processing to replace PyRoki dependency.

### Further Considerations

1. **PyRoki vs Newton IK for retargeting** — See analysis below. Recommend **hybrid approach**: Newton IK for simple cases, PyRoki for complex trajectory optimization until Newton IK gains temporal smoothing.

2. **Keypoint mapping for Gait2354** — Should we use anatomical landmarks (ASIS, greater trochanter, medial/lateral epicondyles) or simplified joint centers? Anatomical landmarks match motion capture marker sets better.

3. **Validation strategy** — Compare Newton FK joint angles against OpenSim inverse kinematics on reference motions to ensure biomechanical fidelity?

---

## PyRoki vs Alternatives Analysis

### PyRoki Strengths
- **Trajectory-level optimization** — Solves entire motion at once (800 iterations), ensuring temporal consistency
- **Contact-aware** — Penalizes foot movement during ground contact (weight=30.0)
- **Proven pipeline** — Already integrated with ProtoMotions biomechanics workflow

### PyRoki Weaknesses
- **Separate environment** — Requires dedicated conda env with JAX, complicates deployment
- **Fixed keypoint mappings** — Hard-coded for G1/H1_2/SMPL, needs new script per robot
- **Manual scaling heuristics** — Segment-specific scale factors (`[0.9, 0.9, 0.85]`) need tuning per model
- **No muscle/biomechanics awareness** — Pure kinematic, ignores physiological constraints

### Newton IK Alternative
Newton's [IKSolver](newton/ik.py) offers:
- **Native integration** — Runs in your existing `.venv`, GPU-accelerated via Warp
- **Flexible objectives** — `IKPositionObjective`, `IKRotationObjective`, `IKJointLimitObjective`
- **Batch solving** — Multiple problems in parallel
- **Missing**: Built-in temporal smoothing and contact constraints (would need post-processing)

### Recommendation: Hybrid Approach

For the goal of **high-fidelity skeletons → Newton → ProtoMotions**:

| Phase | Tool | Rationale |
|-------|------|-----------|
| **Initial retargeting** | Newton IK + Savitzky-Golay smoothing | Stays in your UV `.venv`, simpler setup |
| **Complex motions** | PyRoki (fallback) | When temporal coherence matters (fast locomotion, transitions) |
| **Runtime tracking** | Newton IK | During simulation for pose refinement |

**Long-term**: Build temporal smoothing into Newton IK (chain frames, add velocity regularization objectives). This would fully replace PyRoki for your use case.
