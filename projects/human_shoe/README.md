# Human Shoe

This package holds the project-local boundary between OpenSim-based human
motion and Digital Instron shoe models. It is intentionally not a public
``newton.*`` API while the interface is still evolving.

Current scope:

- validated Digital Instron material/geometry manifests
- validated human-shoe experiment definitions
- a project-local adapter that resolves OpenSim foot contact geometry and
  places shoe sole geometry into the foot frame
- an explicit contact-reference-to-shoe offset, expressed along the OpenSim
  foot-body axes
- pinned controller identifier, solver time step, and random seed
- exact prescribed-motion shoe load replay with per-substep wrench and energy export
- explicit pose-fidelity reports for the approximate dynamic OpenSim import

The first template is
`experiments/human_shoe/baseline_gait2354.json`. It couples the Gait2354
example's ``calcn_r`` body to the existing calibrated shoe manifest. Its
`gait2354_subject01_contacts.json` sidecar adds three alignment spheres without
modifying the source model, and generates
`generated/gait2354_subject01_with_shoe_contacts.osim` for the experiment.
The checked-in shoe mesh remains at its physical scale because it already
spans 292 x 113 mm, which exceeds the 230 x 107 mm Gait2354 marker span; the
contacts are calibrated shoe-top anchors, not anatomical marker proxies.

Generate or refresh the derived model with:

```bash
uv run -m projects.human_shoe.contact_sidecar \
  --sidecar experiments/human_shoe/gait2354_subject01_contacts.json
```

The checked-in Gait2354 model has an empty ``ContactGeometrySet``. The
sidecar now stores direct sphere centers at the shoe-top support points
measured in the ``calcn_r`` frame:
`[0.03745469, -0.01977471, 0.00982458]` with radius `0.025`,
`[0.19745469, -0.01333833, 0.02482458]` with radius `0.020`, and
`[0.19745469, -0.01414376, -0.00517542]` with radius `0.020`. The sphere
centers are the support points plus their radii along `+Y`, and no marker
fields remain in the sidecar. The baseline `translation_m` compensates for the
new support-point centroid, keeping the resolved shoe-frame origin fixed at
`[0.13274929, -0.01622024, 0.010912963333333333]` in `calcn_r` while the sphere
bottoms move onto the selected spring tops.

## Required conventions

- Lengths are metres, rotations are degrees, forces are newtons, and torques
  are newton-metres.
- ``foot_body_name`` must match the body name in the selected OpenSim model.
- ``translation_m`` and ``rotation_deg`` offset the shoe from the centroid of
  the foot-top contact support points, along/about the OpenSim foot-body axes.
- ``controller_id`` identifies the versioned controller configuration. Store
  the actual configuration beside a future experiment manifest.
- ``contact_sidecar_path`` identifies how the derived OpenSim model was built;
  generated models should be refreshed from that sidecar rather than hand-edited.

Adapter semantics:

- ``resolve_attachment()`` validates the named foot and shoe-carrier bodies
  after OpenSim import, gathers the foot's contact geometry, and resolves the
  shoe-to-foot transform from the digital-sole basis plus the contract pose.
- ``attach_sole_geometry()`` shifts sole vertices by the top-interface centroid,
  applies the resolved shoe-to-foot transform, and can optionally remap the
  output into a Z-up jump fixture basis.
- The digital sole basis is X-forward, Y-lateral, Z-up. OpenSim body-local
  coordinates remain X-forward, Y-up, Z-lateral, while Newton-facing world
  transforms rotate OpenSim +Y onto Newton +Z.
- Foot contact objects are calibrated shoe-top anchors. When the Digital
  Instron foundation supplies ground reaction, do not also apply the original
  OpenSim foot-contact force law.
- Contact sidecars contain geometry only. They intentionally cannot add OpenSim
  contact-force elements because the Digital Instron sole is the force model.

Viewer:

```bash
uv run --extra examples -m projects.human_shoe.viewer --viewer gl
```

The viewer loads `experiments/human_shoe/baseline_gait2354.json`, resolves the
checked-in derived OpenSim model via `load_experiment()`, plays back
`newton/examples/assets/gait2354_subject01_walk.mot`, and attaches the Digital
Instron midsole mesh as a visual-only `digital_instron_midsole` shape on
`calcn_r`. Pass `--show-columns` or `--show-column-lines` to inspect the
imported foundation sample points from the calibrated Digital Instron geometry.
During the prescribed gait playback, penetrating column bottoms are clamped to
the Newton Z-up ground plane. Ground penetration drives the cool-to-hot compression
colors while the rendered segments show the resulting shortening as the foot
rolls. This is a deformation visualization only; it does not apply foundation
forces to the kinematic human motion.

## Exact prescribed-motion load replay

Use exact OpenSim ``CustomJoint``/``SimmSpline`` kinematics to replay shoe loads
without the approximate Newton D6 articulation or a posture controller:

```bash
uv run -m projects.human_shoe.replay \
  --experiment experiments/human_shoe/baseline_gait2354.json \
  --stance-index 0 \
  --output reports/human_shoe/prescribed_stance.csv
```

The replay interpolates the source coordinate motion, evaluates exact OpenSim
body poses and velocities, prescribes the ``calcn_r`` carrier, and advances the
same Digital Instron foundation used by the dynamic examples. It never integrates
or modifies the human state. The CSV and JSON sidecar contain per-substep 3-D
GRF, world-origin moment, COP, compression, active columns, contact power/work,
and impulse with units and frame metadata.

``find_contact_windows()`` identifies complete stance windows with unloaded
brackets. The checked-in motion contains three complete right-shoe windows.

## Approximate dynamics example

```bash
uv run --extra importers -m projects.human_shoe.landing --viewer gl
```

The landing example imports the same derived Gait2354 model into
`SolverFeatherstone`, maps the manifest-selected gait frame into the imported
D6 coordinates, keeps its measured vertical pelvis speed, leaves that coordinate
unactuated, and lightly stabilizes the remaining pose. It starts the lowest
outsole column 12 mm above the Newton Z-up ground plane, then applies the calibrated
shared Hyperfoam-Maxwell-Pasternak response, normal damping, and stick-slip
friction directly to `state.body_f[calcn_r]`. The shoe mesh, contact-sphere
anchors, compression colors, and deformed columns all use that same dynamic
`calcn_r` transform.

This is a controlled attachment/contact experiment, not a physiological jump
or a validated human landing. The OpenSim `CustomJoint` transformations are an
approximate D6 dynamics import and the harness supplies posture stabilization.
``PoseFidelityReport`` compares that import with exact OpenSim FK; the checked-in
Gait2354 pose currently exceeds the 5 mm / 2 degree acceptance limits, so
scientific shoe-load analysis should use the exact prescribed replay.
The shoe contact law itself is shared with the calibrated Digital Instron jump
scenario.
The shoe is not a separate rigid body: the sole bed is the direct ground-force
law on `calcn_r`, while the contact spheres remain calibrated visual anchors.
