# Newton-native simple gait model

This project-local model is the first articulated Newton runtime scaffold for
gait solver and contact experiments. It deliberately favors a small, stable
joint set over one-to-one OpenSim mechanics. The staged motion-fitting work and
its example-based completion gates are defined in
[`NATIVE_MOTION_RETARGETING_PLAN.md`](NATIVE_MOTION_RETARGETING_PLAN.md).

## Topology

- free pelvis root;
- torso fixed to the pelvis;
- three independent rotational hip axes per side;
- one revolute knee hinge per side;
- one revolute ankle hinge per side;
- box fallback geometry for the pelvis and torso;
- capsule fallback geometry for each thigh and shank;
- invisible box/capsule self-collision proxies for the six non-foot segments; and
- four contact spheres per foot on a stationary Z-up ground plane.

`SimpleGaitConfig.for_subject()` scales all segment lengths from standing height
and all segment masses from body mass. An optional measured hip width overrides
the uniformly scaled hip spacing. The default rounded dimensions and masses are
derived offline from the sealed S001 reference model, but the runtime does not
parse or import `.osim` files.

The primitive body shapes are non-colliding visual fallbacks. Invisible box and
capsule proxies on the pelvis, torso, femurs, and tibias provide the first
self-collision layer. Limb visual and collision proxy ends stop short of each
joint by the configured clearance so adjacent segments do not overlap at their
rounded ends. Adjacent
parent-child links remain filtered so joint attachments do not fight their own
collision response. Foot spheres are the
active ground and foot self-contact proxies. A later source adapter will convert
scaled OpenSim VTP display meshes into a sealed neutral vertex/index bundle. The
same native builder will attach those meshes as non-colliding visuals without
parsing VTP or `.osim` files at runtime.
The baseline simple model has 8 bodies, 8 joints, 17 generalized coordinates,
and 16 velocity DOFs. The six free-pelvis controls start and remain uncommanded.
Static-calibrated subject bundles add three bounded rotational torso axes while
keeping the same eight-body chain.

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
properties, primitive visuals, invisible segment self-collision proxies, foot
contacts, adjacent-link collision exclusions, a neutral keyframe, and bounded
position/velocity controls for all ten non-root DOFs. It deliberately creates
no pelvis/root actuator.

The saved XML is loadable in one Newton builder call and is also accepted by
MuJoCo:

```python
from projects.gait_c3d.subject_mjcf import write_subject_mjcf

path = write_subject_mjcf(config, "subject.xml")
builder = newton.ModelBuilder()
builder.add_mjcf(str(path), enable_self_collisions=True)
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
bodies, 16 free-root velocity DOFs, 19 non-colliding mesh visuals, 6 invisible
segment self-collision proxies, 8 foot contact spheres, 8 translucent sphere
overlays, and collision/visual ground planes through one
`ModelBuilder.add_mjcf()` call. Official default-pose body
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

## Neutral marker layout

The offline subject compiler converts the accepted marker placement into
`model/marker_layout.json`. The sealed layout records each marker's source body,
target native body, body-local position, MJCF site name, source hashes, frame
rotation, and the same vertical registration used for visuals, joint centers,
inertias, and contacts. OpenSim supplies the offline reference placement; the
saved runtime model does not import or execute OpenSim code.

The subject MJCF contains one hidden, non-colliding `<site>` for every converted
marker. Newton imports these sites with `ModelBuilder.add_mjcf(parse_sites=True)`,
so later motion fitting can construct public `newton.ik.IKObjectivePosition`
objectives without a custom forward-kinematics or marker kernel. The canonical
S001 bundle contains 27 native marker sites on the pelvis, torso, bilateral femurs,
tibias, and merged feet.

Run the tracked compact Phase 1 demonstration from a clean checkout. It builds
a persistent project-local demo subject, imports its ten marker sites, and
shows them as green points:

```bash
uv run --extra dev -m newton.examples opensim_subject \
  --subject projects/gait_c3d/subjects/marker-demo \
  --overwrite \
  --marker-demo \
  --show-markers \
  --paused
```

The compact fixture represents the output shape of offline OpenSim marker
placement but does not execute OpenSim. Build or rebuild S001 with the canonical
compiler command below to inspect its 27 native centroid-collapsed sites with the same
`--show-markers` option. The points come from imported MJCF sites and move with
their native bodies. `marker_layout.json` remains the sealed provenance and
name-mapping artifact used by later motion-retargeting phases.

## S001 base marker placement and geometry

`assets/s001_base` is the tracked canonical **S001** base subject. It contains
27 native marker sites. The four three-marker thigh/shank tracking clusters
are collapsed to centroids. The bundle also contains the final neutral MJCF and
19 actual S001 bone meshes.
This is the Phase 1 visual gate: the green marker points are imported MJCF
sites and the gray meshes are the corresponding non-colliding bone visuals.
No OpenSim runtime is needed to open the bundle.

Inspect the exact S001 placement on its actual bone geometry:

```bash
uv run --extra dev -m newton.examples opensim_subject \
  --subject projects/gait_c3d/assets/s001_base \
  --show-markers \
  --paused
```

Create a new native subject by scaling the complete S001 base. The scaler applies
the same length factor to bone meshes, marker sites, body frames, contacts, and
inertial geometry, and applies the mass factor to inertias:

```bash
uv run --extra dev -m newton.examples opensim_subject \
  --base-subject projects/gait_c3d/assets/s001_base \
  --subject /tmp/s001_scaled_subject \
  --height 1.80 \
  --mass 90.0 \
  --overwrite \
  --show-markers
```

The target hip width defaults to the scaled S001 width. Pass `--hip-width` to
apply an explicit width; the MJCF femur frames and marker-layout frames are
updated together.

## Visual marker-set mapping

Use the interactive mapper when a lab uses different C3D labels for the same
S001 anatomical marker protocol. This project example runs from a Newton source
checkout because it uses the `projects/gait_c3d` adapters. It overlays the C3D
points on the neutral MJCF, highlights the selected target and source in orange,
and draws green lines for
completed assignments:

```bash
uv run --extra dev --with ezc3d -m newton.examples marker_mapper \
  --subject projects/gait_c3d/assets/s001_calibrated \
  --c3d "/path/to/static.c3d" \
  --marker-map "/path/to/my_lab_marker_map.json" \
  --paused
```

Choose an MJCF target role and its exact C3D source label in **Example
Options**, then select **Save marker map**. A button can apply unique normalized
name suggestions for review; suggestions are never saved without the user's
explicit action. The editor also includes `C7`, `CLAV`, and `T10` torso
calibration roles even though they are not MJCF sites. The JSON stores only
label aliases, and omitted labels use the canonical S001 name. Final matching
is exact and case-sensitive so the tool never guesses left/right or
medial/lateral anatomy. The visual registration is display-only and is not
written into the map; **Fit and lock display from current map** changes it only
when explicitly selected.

Pass the saved map once when building a subject:

```bash
uv run --extra dev --with ezc3d -m newton.examples opensim_subject \
  --static-cal "/path/to/static.c3d" \
  --marker-map "/path/to/my_lab_marker_map.json" \
  --subject /tmp/my_subject \
  --mass 72 \
  --overwrite
```

The builder copies the validated map into the subject bundle. Later dynamic
motion fitting uses that bundled map automatically:

```bash
uv run --extra dev --with ezc3d -m newton.examples native_motion_fit \
  --subject /tmp/my_subject \
  --c3d "/path/to/walking.c3d"
```

Version 1 changes labels only. It covers the 39 sources used by the 27 MJCF
marker targets plus the `C7`, `CLAV`, and `T10` torso calibration roles;
unrelated C3D labels pass through unchanged. It keeps the current anatomical
landmarks and fixed recipes for the sacrum, head, and three-marker thigh/shank
centroids. A protocol with missing medial landmarks, direct centroid markers,
or different physical placements needs a new calibration profile rather than
an alias map.
Use `--keep-c3d-prefix` when labels such as `Person01:LASI` must remain distinct.

## Static per-segment calibration

Use a static calibration C3D to build a personalized model with Visual3D-style
segment definitions. The compiler averages valid samples in the calibration
window, builds a CODA pelvis from both raw ASIS and PSIS marker pairs,
applies the Bell--Brand hip regression, finds knee and ankle centers from
medial/lateral marker pairs, and builds a flat forward/left/up foot frame. The
PSIS midpoint remains available as the derived `VSAC` compatibility marker, but
the bilateral PSIS slope now contributes to the pelvis frame. The merged torso
uses that posterior midpoint and the superior head as its endpoints, with C7,
T10, sternum, clavicle, and acromion markers defining its frame. It saves the
calibration next to the MJCF so the
model can be reopened without the C3D file. The definitions follow the
[Visual3D CODA pelvis](https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:coda_pelvis),
[hip landmark](https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:hip_joint_landmarks),
and [segment geometry](https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:segment_geometry)
references.

```bash
uv run --extra dev --with ezc3d -m newton.examples opensim_subject \
  --static-cal "/path/to/static_calibration.c3d" \
  --subject /tmp/s001_calibrated_subject \
  --overwrite \
  --show-markers \
  --show-calibration
```

The saved bundle contains `calibration/segment_calibration.json`, marker data,
actual S001-derived meshes, and `model/subject.xml`. Leg visuals are scaled
from the source MJCF joint spans and remain anchored to the source hip and knee
centers. The torso visual spans from the posterior pelvis origin to the saved
`Top.Head` height, and its collision box and inertia use the same map. Collision
capsules reserve clearance at their outer surfaces, and transformed box proxies
follow the same geometry map. In
this marker set,
`LHLX`/`RHLX` are the hallux (big-toe) markers used for the left/right toe
endpoints; `LTOE`/`RTOE` are not used as the toe-tip definition. Its four foot contact
spheres are generated from the calibrated foot mesh frame and their surfaces
are placed on `z=0`.

## Synthetic native marker IK

The Phase 2 synthetic gate uses imported marker sites and only the public
`newton.ik` API. It generates a known free-root native motion, solves it with
analytic Levenberg--Marquardt, warm-starts each frame, and overlays target and
predicted markers.

```bash
uv run --extra dev -m newton.examples native_motion_fit \
  --synthetic \
  --noise-mm 1.0 \
  --occlude-every 4
```

The example reports unweighted marker RMS/max error, public solver cost, and
checks free-root quaternion normalization and native joint limits.

## Real C3D native motion fit

Fit a dynamic trial by name-joining its markers to the saved native sites. The
three-marker thigh and shank tracking clusters are averaged into one
`*.Centroid` target per segment before IK. A centroid is valid only when all
three source markers are valid. The raw C3D marker artifact remains unchanged.
The saved calibrated subject ground offset is applied as an explicit
registration.
The default safety limit fits 300 frames; use `--max-frames 0` for the full
trial, or use `--stride` while validating a long capture:

```bash
uv run --extra dev --with ezc3d -m newton.examples native_motion_fit \
  --c3d "/path/to/Trial 101.v3d.c3d" \
  --subject projects/gait_c3d/assets/s001_calibrated \
  --max-frames 300 \
  --motion-output /tmp/trial_101_native_motion
```

Set `--max-frames 0` for a full solve, and add `--overwrite` only to replace a
verified artifact. Replay the saved output without decoding or solving again:

```bash
uv run --extra dev -m newton.examples native_motion_fit \
  --motion /tmp/trial_101_native_motion \
  --subject projects/gait_c3d/assets/s001_calibrated
```

By default, the sealed motion is stored under the subject bundle at
`<subject>/motions/<trial>_native_motion/`. Pass `--motion-output` to override
that subject-local location. The output directory contains a sealed
`motion.npz` artifact and manifest with
native coordinates, finite-difference velocities, target/predicted markers,
validity, per-marker/per-body residuals, solver costs, limit diagnostics,
registration metadata, and the cluster source mapping.

## Reusable progress example

Run the tracked S001 per-segment calibrated subject by omitting `--subject`.
The default bundle is `projects/gait_c3d/assets/s001_calibrated` and includes
its saved calibration, actual bone meshes, calibrated torso endpoints, and flat
foot contacts.

```bash
uv run --extra dev -m newton.examples opensim_subject
```

Show the invisible pelvis, torso, femur, and tibia self-collision proxies:

```bash
uv run --extra dev -m newton.examples opensim_subject --show-collision
```

Each compiled subject folder is self-contained. `subject.json` stores the
subject mass, height, hip width, and artifact locations; the model folder stores
MJCF, inertials, collision proxies, compiled VTP visuals, and the sealed neutral
marker layout; marker trajectories and offline OpenSim reference artifacts stay
under the same subject root. Reopen the subject without
any source paths:

```bash
uv run --extra dev -m newton.examples opensim_subject \
  --subject projects/gait_c3d/subjects/S001
```

Run the complete canonical S001 proof with direct C3D decoding and scaled VTP
visuals:

```bash
uv run --extra dev --with ezc3d --with opensim==4.6 -m newton.examples opensim_subject \
  --subject projects/gait_c3d/subjects/S001 \
  --overwrite \
  --c3d "/home/jo31399/newton-data/gait/incoming/Cal 101.v3d.c3d" \
  --template /home/jo31399/newton-data/gait/reference/gait2354_subject01.osim \
  --mass 81.9312118 \
  --height 1.695898298375747 \
  --geometry ~/.cache/newton/opensim-models_Geometry_fa3fb094_d9b05d47/Geometry
```

The example always writes a reusable MJCF model and runs it through
`ModelBuilder.add_mjcf()` with nonadjacent self-collision enabled,
`CollisionPipeline`, and `SolverFeatherstone`. Collision proxies are hidden by
default; `--show-collision` reveals them for inspection. Its
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
with `--substeps` only when running an explicit convergence study.

## Official OpenSim subject building

When `--template` is supplied, the progress example starts from static C3D
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
inertia products, and real left/right foot asymmetry. The same mass and inertia
values also drive the segment proxies: pelvis and torso boxes use the
inertia-box COM, principal axes, and extents; limb capsules use the longest
principal axis and the smaller transverse extent before applying joint
clearance. The canonical one-call Newton import agrees with the offline mapping
within 1.3e-6 kg, 3.5e-9 m, and 2.1e-8 kg·m².

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
