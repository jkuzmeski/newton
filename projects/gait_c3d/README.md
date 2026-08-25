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

## Reusable progress example

Run the current native subject path with generated fallback geometry:

```bash
uv run --extra dev -m newton.examples opensim_subject
```

Run the complete canonical S001 proof with direct C3D decoding and scaled VTP
visuals:

```bash
uv run --extra dev --with ezc3d --with opensim==4.6 -m newton.examples opensim_subject \
  --subject-dir /tmp/newton-opensim-s001-proof \
  --overwrite-subject-dir \
  --c3d "/home/jo31399/newton-data/gait/incoming/Cal 101.v3d.c3d" \
  --template-osim /home/jo31399/newton-worktrees/c3d-predictive-forward-dynamics/newton/examples/assets/gait2354_subject01.osim \
  --body-mass 81.9312118 \
  --body-height 1.695898298375747 \
  --official-marker-placement \
  --geometry-dir ~/.cache/newton/opensim-models_Geometry_fa3fb094_d9b05d47/Geometry
```

The example always writes a reusable MJCF model and runs it through
`ModelBuilder.add_mjcf()`, `CollisionPipeline`, and `SolverFeatherstone`. When
source arguments are supplied it also proves C3D-to-NPZ/Warp conversion and
scaled VTP-to-OBJ attachment. Its `test_final()` checks finite body state, exact
root-force exclusion, artifact publication, and uploaded marker arrays.

## OpenSim-referenced C3D ModelScaler

When `--template-osim` is supplied, the progress example now starts from the
static C3D rather than a pre-scaled model. The offline adapter uses the pinned
OpenSim gait2354 default-marker fixture and a declared Trial 101 measurement
policy derived from ModelScaler semantics. The pelvis and bilateral limb
measurements follow the OpenSim marker-pair method; the torso substitutes
shoulder width and sternum-to-ASIS distances because `Top.Head` is a synthesized
head-cluster centroid rather than the cranial vertex assumed by the original
setup. Patella-only and official manual overrides are not represented in the
sealed simple model. The adapter then applies corrected OpenSim-inspired XML
scaling rules for body geometry,
mass/COM/inertia, joint frames and CustomJoint translations, markers, wraps,
and muscle/path points.

For the canonical S001 static window (0.5–1.0 s), all five recovered segment
factors are within 0.026% of the accepted reference artifact. The output
`scaling/manifest.json` records the C3D/template hashes, measurement ratios,
body factors, subject mass, time window, method reference, and scaled-model
hash. Marker placement remains a separate oracle stage. With
`--official-marker-placement`, the example runs the pinned OpenSim 4.6
MarkerPlacer task weights and coordinate locks against the scaled model, saves
its placed model/MOT/marker set/setup/log as reference artifacts, and reports
RMS/max marker error. The placed OpenSim model is not used as the Newton runtime
model. The current engineering publication gate is RMS <= 0.10 m and maximum
<= 0.25 m; the canonical run reports 0.0680 m and 0.1714 m. These deliberately
broad gates only reject broken placement and do not establish high-fidelity
marker validation.
