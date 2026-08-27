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

The canonical official S001 conversion resolves 19 VTP assets and loads as 8
bodies, 16 free-root velocity DOFs, 19 non-colliding mesh visuals, 8 foot
contact spheres, 8 translucent sphere overlays, and collision/visual ground
planes through one `ModelBuilder.add_mjcf()` call. Official default-pose body
transforms bake talus, calcaneus, and toe meshes into each merged Newton foot
frame. The compiler then raises the complete neutral target hierarchy by one
audited root-height offset so visuals, joint centers, COMs, inertias, and
contacts remain in one frame while the lowest visual sole meets the ground. It
derives contact radius and heel/forefoot/medial/lateral centers from each
converted foot's body-local bounds and makes every sphere surface meet that same
sole plane.

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
  --geometry-dir ~/.cache/newton/opensim-models_Geometry_fa3fb094_d9b05d47/Geometry
```

The example always writes a reusable MJCF model and runs it through
`ModelBuilder.add_mjcf()`, `CollisionPipeline`, and `SolverFeatherstone`. Its
default standing-inspection mode fixes the pelvis because a free-root balance
controller is not implemented yet; pass `--free-root` to run the explicitly
unassisted falling model. The saved MJCF itself retains the free joint. When
source arguments are supplied the example also proves C3D-to-NPZ/Warp
conversion and scaled VTP-to-OBJ attachment. Its `test_final()` checks finite
body state, standing pelvis height in inspection mode, exact zero root effort in
free mode, eight visible foot spheres, one visible ground plane, artifact
publication, and uploaded marker arrays.

The exact OpenSim COM/inertia and eight stiff sphere contacts require a smaller
Featherstone step than the earlier approximate model. The example defaults to
50 solver/contact substeps per 60 Hz display frame (`dt = 1/3000 s`). Ten
substeps caused nonfinite leg state during the second display frame. Override
with `--subject-substeps` only when running an explicit convergence study.

## Official OpenSim subject building

When `--template-osim` is supplied, the progress example starts from static C3D
and uses official OpenSim 4.6 `ScaleTool` as the default backend. The adapter
constructs a Trial 101-specific `MeasurementSet` through OpenSim APIs, selects
`scaling_order = measurements`, preserves mass distribution, and deliberately
adds no inherited manual subject scales. The pelvis and bilateral limb
measurements follow the OpenSim marker-pair method. The torso substitutes
shoulder width and sternum-to-ASIS distances because `Top.Head` is synthesized
from a head cluster rather than measured at the cranial vertex.

One sandboxed ScaleTool run performs both official `ModelScaler` and official
`MarkerPlacer`. It publishes modern OpenSim 4.6 scaled/placed models, scale XML,
marker set, static motion, setup, logs, version, source hashes, parsed body
factors, and marker QC. The canonical S001 run reports 0.0680 m RMS and 0.1714 m
maximum marker error against broad broken-placement gates of 0.10 m and 0.25 m.
These gates are not high-fidelity marker acceptance.

The official scaled model is then translated into the declared simple Newton
model: its modern `attached_geometry/Mesh` and per-joint
`PhysicalOffsetFrame` data drive VTP registration, segment masses and lengths,
and the controlled MJCF. The placed OpenSim model stays an oracle artifact and
is not the Newton runtime model.

Use `--scaling-backend parity` to run the project ModelScaler-derived XML path
instead. That backend is retained for comparison and fallback only; it is not
the accepted subject builder.

## Neutral-pose visual clearance and inertial preservation

The official foot geometry drives the ground/contact layout. Visual-only
`tibia` and `fibula` mesh geoms receive a subject-scaled distal offset (15 mm
for canonical S001); body transforms, joints, contact spheres, COMs, and
inertias do not move. On S001 this removes the 8.60 mm femur/tibia penetration,
leaves 0.176 mm knee surface clearance, and reduces the tibia/fibula-to-foot gap
from 3.20 mm to approximately 1.93 mm on both sides. This gate covers the
neutral pose only; range-of-motion mesh intersection remains separate work.

The official ScaleTool neutral body transforms also map OpenSim COM and full
inertia tensors into each simple Newton body frame. Talus, calcaneus, and toes
are combined with the mass-weighted COM and parallel-axis theorem. The saved
MJCF now preserves all eight official-derived masses, nonzero COM offsets, full
inertia products, and real left/right foot asymmetry. The canonical one-call
Newton import agrees with the offline mapping within 1.3e-6 kg, 3.5e-9 m, and
2.1e-8 kg·m².

## Official joint-center mapping

The official ScaleTool subprocess exports every source body transform at the
OpenSim default state. The converter uses the femur, tibia, and talus origins as
the neutral hip, knee, and ankle centers, rotates them from OpenSim ground into
Newton ground, applies the same audited root-height registration as the visual
skeleton, and expresses each center in its simple target child-body frame. The
MJCF hip, knee, and ankle `joint pos` values come from those mapped centers,
not from half-length approximations.

On canonical S001, the previous approximate centers were displaced by roughly
64–70 mm anteriorly and 103 mm vertically from the displayed OpenSim joints.
After conversion, all six loaded Newton centers agree with the corresponding
official centers within 0.06 micrometers. The 15 mm tibia/fibula mesh clearance
remains visual-only, so it does not move the official knee center or any physics
frame.
