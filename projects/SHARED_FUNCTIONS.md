# Shared-function review

The organization pass included a source-level duplicate check and a separate
semantic review. It did not merely compare function names or move files.

## Consolidated implementations

| Computation | Shared owner | Consumers |
|---|---|---|
| Camera look-at calculation | `digital_shoe.rendering.camera_look_at` | Portable showcase and source-backed Instron example |
| Quaternion multiplication/conjugation | `digital_instron_v2.scenario_common` | Forward and differentiable scenario trajectories |
| Scenario surround construction/defaults | `digital_instron_v2.scenario_common` | Forward and differentiable examples; local overrides still pass explicitly |
| Attachment PD force/moment expression | `digital_instron_v2.scenario_common.attachment_pd_wrench` | Counter-based forward and tape-safe attachment adapters |
| XYZ rotation and ray-origin setup | `digital_instron_v2.geometry` | Mesh loading/transform and both ray-query paths |
| Legacy JSON conversion, trapezoid integration, HTML escaping and momentum sampling | `impedance_instron.legacy._utils` | Historical reports, explanation, dashboard, environment and optimizer |
| Legacy planar pitch and sample-index increment | Existing definitions in `impedance_instron.legacy.example` | Environment imports the same Warp objects |

The legacy extraction alone replaces 14 computation bodies with 6 shared
implementations, removing 8 redundant bodies. Thin wrappers remain where they
preserve function signatures, callbacks or per-module configuration. The scenario
and geometry extractions remove additional algorithm copies; different buffer
layouts and target indexing remain in their adapters.

The unused private `scenarios_diff._ground_reaction_force` recomputation was
removed. The active scenarios already read the applied force history. Two unused
legacy reference-column labels were removed; the underlying columns and traces
were not. No obviously uncalled private function leaf was found in the retained
legacy controller stack, so no such deletion is claimed.

## Special check

```bash
uv run --no-sync python scripts/check_shoe_project_duplicates.py --check \
  --output outputs/impedance_instron/organization_review/duplicates.json
```

The structural scan fell from 10 exact/renamed clone groups to 4. It also reports
near matches for review. These counts use a minimum AST size of 20 nodes and are
not a claim that every conceivable semantic duplication can be detected.

The four remaining exact groups are listed in `shoe_duplicate_review.json`:
canonical JSON bytes, duplicate-key rejection, canonical JSON hashing, and
chunked file hashing. They are small but real duplicates. They remain because
several containing files participate in frozen reference/rig source identities,
and the profile exporter can execute as a standalone file in another worktree.
A future extraction must version the provenance dependency closure and preserve
standalone execution. The cleanup does not bypass checks or reseal old artifacts.
The checker fails if a new exact/renamed algorithm clone appears outside these
reviewed groups; a regression test enforces that rule.

## Similar code that must not be merged blindly

- Strict canonical JSON, nonfinite-to-null display JSON, and round-trip optimizer
  state encoding have different contracts.
- Hysteresis area functions can have opposite signs or select different branches.
- Supplied-slope Hermite interpolation, estimated-slope interpolation, and a
  natural C2 trajectory differ in construction and endpoint behavior.
- Work integration rules, formatting precision, downsampling and missing-value
  handling are not interchangeable.
- Paired-thickness ray acceptance and near/far surface selection share ray setup,
  not their acceptance rules.
- Mutable forward state, tape-safe history and resident graph buffers call the
  same material/contact functions but still need separate storage/launch adapters.

The shared two-term material/contact source files and active two-stiffness
implementation remain byte-identical. No material parameters, controller
objectives, fitting defaults, source inputs or checkpoint contents changed.
