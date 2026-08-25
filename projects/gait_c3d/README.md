# Newton-native simple gait model

This project-local model is the first articulated Newton runtime scaffold for
gait solver and contact experiments. It deliberately favors a small, stable
joint set over one-to-one OpenSim mechanics.

## Topology

- free pelvis root;
- torso fixed to the pelvis;
- three independent rotational hip axes per side;
- one revolute knee hinge per side;
- one revolute ankle hinge per side;
- box fallback geometry for the pelvis and torso;
- capsule fallback geometry for each thigh and shank; and
- four contact spheres per foot on a stationary Z-up ground plane.

`SimpleGaitConfig.for_subject()` scales all segment lengths from standing height
and all segment masses from body mass. An optional measured hip width overrides
the uniformly scaled hip spacing. The default rounded dimensions and masses are
derived offline from the sealed S001 reference model, but the runtime does not
parse or import `.osim` files.

The primitive body shapes are non-colliding visual fallbacks. Foot spheres are
the only active body contacts. A later source adapter will convert scaled OpenSim
VTP display meshes into a sealed neutral vertex/index bundle. The same native
builder will attach those meshes as non-colliding visuals without parsing VTP or
`.osim` files at runtime.
The model has 8 bodies, 8 joints, 17 generalized coordinates, and 16 velocity
DOFs. The six free-pelvis controls start and remain uncommanded.

This is an engineering approximation. It is not OpenSim parity, predictive gait,
or an FD-1 result. The next milestone adds bounded non-root torque control and
25/50/100 ms restart tests before attempting a stride.

Run its focused tests from the repository root:

```bash
uv run --extra dev -m newton.tests -k test_gait_simple_joints
```

## Saved subject model

`write_subject_mjcf()` exports a scaled `SimpleGaitConfig` as a self-contained
MJCF XML model. The XML includes the full simple-joint topology, inertial
properties, primitive visuals, foot contacts, a neutral keyframe, and bounded
position/velocity controls for all ten non-root DOFs. It deliberately creates
no pelvis/root actuator.

The saved XML is loadable in one Newton builder call and is also accepted by
MuJoCo:

```python
from projects.gait_c3d.subject_mjcf import write_subject_mjcf

path = write_subject_mjcf(config, "subject.xml")
builder = newton.ModelBuilder()
builder.add_mjcf(str(path))
```

The next adapter stage bakes the scaled VTP display geometry into subject-local
mesh assets referenced by this MJCF. A separate manifest seals source C3D,
scaled model, geometry, conversion policy, and output hashes.

## Scaled VTP visual adapter

`compile_scaled_vtp_visuals()` reads the accepted legacy gait2354 display
geometry references, resolves the pinned VTP assets, applies the already-scaled
per-mesh factors and body-local transforms, recenters each mesh at the scaled
body COM, and rotates OpenSim Y-up coordinates into Newton Z-up coordinates. It
writes deterministic OBJ assets and a manifest containing source/output hashes.

Only pelvis, torso, femur, tibia, and fibula visuals are compiled. The simple
model merges the source foot bodies, so feet deliberately retain sphere-only
geometry until their relative transforms are sealed. Meshes are visual-only;
they never replace the primitive collision/contact policy.

```python
from projects.gait_c3d.subject_mjcf import write_subject_mjcf
from projects.gait_c3d.vtp_adapter import compile_scaled_vtp_visuals

visuals = compile_scaled_vtp_visuals(scaled_osim, geometry_dir, bundle_dir, config)
subject_xml = write_subject_mjcf(
    config,
    visuals.root / "subject.xml",
    visual_meshes=visuals.meshes,
    include_fallback_geometry=False,
)
builder = newton.ModelBuilder()
builder.add_mjcf(str(subject_xml))
```

The canonical S001 conversion resolves 13 VTP assets and loads as 8 bodies, 16
velocity DOFs, 13 non-colliding mesh visuals, 8 foot spheres, and one ground
plane through one `ModelBuilder.add_mjcf()` call.

## Direct C3D marker artifacts

The production preprocessing path does not require TRC or MOT intermediates.
`c3d_to_marker_artifact()` decodes C3D point data directly into finite,
Newton-frame SI arrays and atomically publishes `markers.npz` plus a hashed
`manifest.json`. Missing observations are stored as zero with a separate boolean
validity mask, so device arrays never receive marker NaNs.

```python
from projects.gait_c3d.c3d_adapter import c3d_to_marker_artifact, load_marker_artifact

root = c3d_to_marker_artifact(
    "Cal 101.v3d.c3d",
    "subjects/S001/static_markers",
    up_axis="+Z",
    forward_axis="-Y",
)
markers = load_marker_artifact(root)
device_markers = markers.to_warp("cuda:0")
```

C3D decoding is an offline boundary and uses `ezc3d` when available:

```bash
uv run --with ezc3d python -m your_subject_compiler
```

The saved NPZ/Warp runtime path uses only NumPy and Warp. TRC, MOT, and STO
remain optional reference exports for OpenSim interoperability.
