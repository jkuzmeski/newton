# Materials testing with a frozen controller

Evaluate material substitutions with the accepted runner, identical shoe geometry,
trial initial states, body parameters, task, and friction settings:

```console
uv run -m projects.impedance_instron.hogan.materials --dataset PATH_TO_DATASET --output outputs/impedance_instron/materials_paper --run
```

Use a fresh output directory. Omitting `--run` writes a plan without evaluation;
that directory remains a planning artifact and is not overwritten by later runs.
The default controller is the preserved full-score baseline. Every condition
starts from the original trial state; the controller is never fitted or changed.
All 98 training and nine held-out windows are saved and scored separately.

The material ranking depends on the frozen controller, so check it with more
than one. On the column-bed shoe, the score and sample controllers fitted to
that shoe agree in sign on 8 of the 9 material-by-geometry conditions. With
the measured foam, the longer springs lower the weighted-work proxy by 3.0-4.5% and the 40 mm block
raises it by 5.2-7.9% on every held-out stance. The foam swap alone changes it
by 0.7-3.7%, with a size that depends on the controller. The 2026-10-10
controller, fitted under an earlier shoe law, reverses almost every sign.

The [foam paper](https://arxiv.org/html/2602.12694v2) supplies compression means
and standard deviations in Tables 1 and 2, plus independent shear stiffness.
The experiment uses a deterministic two-term Ogden-Hill pressure approximation
and an independent equilibrium shear override for the shear layer and tangential
contact stiffness. This is a reduced foundation surrogate, not the paper's full
three-dimensional constitutive model or an ASICS whole-shoe simulation. Original
Puma geometry, rigid last, mounting, and contact settings are retained.

Four conditions separate the original calibrated viscoelastic material, its
elastic equilibrium response, and elastic FF LEAP and FF TURBO PLUS surrogates.
The paper's measured energy-return values do not uniquely identify a rate law;
these foam surrogates therefore disable the normal Maxwell branch and do not
claim to reproduce measured hysteresis. Report compression-fit error, the
small-strain tangent mismatch, and any strain beyond the measured 0-60% range.

Positive and signed negative joint work integrate actuator torque times angular
velocity at each native pre-step state, using each saved interval duration.
Report hip, knee, and ankle contributions, total positive and absorbed work,
peak power, and GRF/contact metrics. Compare common completed held-out windows;
keep failed windows visible and training results secondary. These are finite
window mechanical-work measurements, not metabolic energy or running economy.

The run saves source and controller hashes, source snapshots, all native traces
and scenarios, material fits, and a half-timestep diagnostic for the same
predeclared held-out sample in every condition. Before publishing, capture
source-checked saved-pose contact replays with per-column compression and
endpoints. The October 10 run includes its standalone `capture_replay.py`;
run it once against each condition directory with `uv run python`. It verifies
source, material, initial state, forces, moments, and compression against the
native CUDA evaluation before publishing a viewer.

Build the comparison report and Materials testing catalog entry after capture:

```console
uv run -m projects.impedance_instron.hogan.materials_report --run PATH_TO_MATERIALS_RUN --outputs outputs
```

Report rendering uses the local `instron-reports` skill installation. Keep the
physical run source frozen until all contact captures finish. The report module
is excluded from the physical-source snapshot because presentation is generated
after execution. Check the artifact inventory and inspect the resulting report
on phone and desktop before marking the run ready.

## Spring geometry comparison

Cross the current material, FF LEAP, and FF TURBO PLUS with the original
geometry, 25% longer springs, and a flat 40 mm block at the existing spring XY
sites. All nine combinations receive fresh rollouts with the accepted controller:

```console
uv run -m projects.impedance_instron.hogan.materials_geometry --dataset PATH_TO_DATASET --output outputs/impedance_instron/materials_geometry --run
uv run -m projects.impedance_instron.hogan.materials_geometry_report --run PATH_TO_GEOMETRY_RUN --outputs outputs
```

The longer-spring condition holds the original free-top surface fixed and moves
the spring bottoms downward. The block has nominal bottom Z=0, top Z=0.040 m,
and every rest length exactly 0.040 m. Both retain the original XY coordinates,
areas, neighbor topology, fullfoot driven footprint, ankle mounting, rigid last,
and material constants. The block changes the foam-to-carrier gap; it does not
replace the rigid last with a planar foot. The existing controller body model
and massless foundation do not acquire additional shoe mass from added volume.

Match each trial's original minimum nominal-bottom ground clearance by a recorded
vertical translation of the initial body. Preserve angles, velocities, phase,
actuator memory, and task. Translate positional comparison references by the
same amount for diagnostic scoring, while keeping measured forces unchanged.
This prevents extra thickness from silently introducing an initially penetrated
shoe. It is a matched-clearance protocol, not identical absolute initial height.

Capture every condition's first held-out native contact replay before rendering,
using the run's source-checked exporter. Geometry replays show the actual free
foam envelope and use a common 0-60 mm compression scale. The report compares
geometry effects within each material and common completed windows across all
nine combinations. It preserves failure counts, out-of-support compression,
representative half-step sensitivity, and original-geometry reproduction checks.
The Materials testing landing page retains both the material and geometry studies.

## Saved spring rendering

The report updater draws each condition from its own saved native poses and
shoe artifact. Run `uv run --no-project python -m
projects.impedance_instron.hogan.spring_viewer_report CONDITION_DIRECTORY`
after qualified capture. It retains the original viewer, writes
`replay_evidence/spring-geometry.npz` and `.json`, and adds a synchronized
oblique spring close-up with the condition's actual rest lengths.

Early CPU capture scripts appended NumPy views of reused Warp buffers, so the
original deformation NPZ endpoint and compression arrays repeated the last
frame. The original viewer retained separately serialized per-frame compression.
The updater uses those captured values (rounded to 0.001 mm), the saved poses,
and the artifact geometry to reconstruct display endpoints without integrating
contact. Original files remain available; native joint-work metrics are unchanged.
Always copy CPU arrays when collecting capture histories, including compression
and column endpoints, before the next kernel overwrites the buffers.
