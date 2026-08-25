# Newton-native simple gait model

This project-local model is the first articulated Newton runtime scaffold for
gait solver and contact experiments. It deliberately favors a small, stable
joint set over one-to-one OpenSim mechanics.

## Topology

- free pelvis root;
- torso fixed to the pelvis;
- three independent rotational hip axes per side;
- one revolute knee hinge per side;
- one revolute ankle hinge per side; and
- four sphere contacts per foot on a stationary Z-up ground plane.

The default rounded dimensions and masses are derived offline from the sealed
S001 reference model, but the runtime does not parse or import `.osim` files.
The model has 8 bodies, 8 joints, 17 generalized coordinates, and 16 velocity
DOFs. The six free-pelvis controls start and remain uncommanded.

This is an engineering approximation. It is not OpenSim parity, predictive gait,
or an FD-1 result. The next milestone adds bounded non-root torque control and
25/50/100 ms restart tests before attempting a stride.

Run its focused tests from the repository root:

```bash
uv run --extra dev -m newton.tests -k test_gait_simple_joints
```
