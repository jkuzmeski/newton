# C3D predictive branch native-architecture audit

Baseline: `6f66a393`. Audited `52` changed or newly created files in this worktree.

## Enforced boundary

- Before conversion: source `.osim`/RRA assets and exact source kinematics may use the optional `newton.opensim` adapter.
- After `newton_contact_input_v1` publication: production mechanics use only neutral Newton/Warp APIs.
- Official OpenSim and `newton.opensim` mechanics are offline references and never FD-1/FD-2 runtime dependencies.

## File-by-file inventory

| File | Role | Required action |
|---|---|---|
| `changelog/+c3d-measured-load-diagnostics-31a7c9e2.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+c3d-newton-native-contact-boundary-8f4c2a71.changed.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+c3d-normal-contact-fit-91e6b3c4.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+c3d-predictive-contact-7f2d4a91.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+c3d-residual-sensitivity-54b8e0d3.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+moco-track-toe-bounds-6a42f17c.fixed.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+official-moco-contact-reference-0d7b315e.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+official-moco-inverse-reference-5b741d8e.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+official-moco-track-contact-5f6c8a1e.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+official-opensim-rra-reference-6c31e7a4.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+opensim-contact-cuda-a617e4c2.changed.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+opensim-contact-parity-b8d402f1.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+opensim-forward-contact-3e7a61b9.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+opensim-modern-motion-types-5de8b9c1.fixed.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+opensim-modern-transform-functions-2a9d71c4.fixed.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+rra-adjusted-contact-input-1cf458d2.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `changelog/+s001-moco-contact-calibration-figures-31e8d6a4.added.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `newton/_src/opensim/contact.py` | `compatibility_library/test` | Keep as optional newton.opensim compatibility/parity support; not core-native production. |
| `newton/_src/opensim/dynamics.py` | `compatibility_library/test` | Keep as optional newton.opensim compatibility/parity support; not core-native production. |
| `newton/_src/opensim/kinematics.py` | `compatibility_library/test` | Keep as optional newton.opensim compatibility/parity support; not core-native production. |
| `newton/_src/opensim/parser.py` | `compatibility_library/test` | Keep as optional newton.opensim compatibility/parity support; not core-native production. |
| `newton/tests/test_gait_c3d_architecture.py` | `native_architecture_test` | Retain; enforces neutral runtime and dependency boundary. |
| `newton/tests/test_gait_c3d_contact_calibration.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_measured_load_diagnostics.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_moco_contact_calibration.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_newton_native_contact.py` | `native_architecture_test` | Retain; enforces neutral runtime and dependency boundary. |
| `newton/tests/test_gait_c3d_opensim_contact_parity.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_opensim_moco_contact_reference.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_opensim_moco_inverse_reference.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_opensim_moco_track_reference.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_opensim_rra_reference.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_predictive_contact.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_residual_sensitivity.py` | `reference_or_adapter_test` | Retain only for adapter/oracle behavior; does not validate production mechanics. |
| `newton/tests/test_gait_c3d_rra_adjusted_contact_input.py` | `adapter_test` | Retain for sealed conversion/file-boundary behavior. |
| `newton/tests/test_opensim.py` | `compatibility_library/test` | Keep as optional newton.opensim compatibility/parity support; not core-native production. |
| `projects/gait_c3d/ARCHITECTURE_BOUNDARIES.json` | `architecture_policy` | Canonical machine-readable role inventory. |
| `projects/gait_c3d/FORWARD_DYNAMICS_ROADMAP.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `projects/gait_c3d/NATIVE_ARCHITECTURE_AUDIT.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `projects/gait_c3d/README.md` | `documentation/changelog` | Ensure language reserves production claims for neutral Newton runtime. |
| `projects/gait_c3d/contact_calibration.py` | `compatibility_reference` | Quarantined with --reference-only; exclude from FD runtime, migrate pure QC then delete/rename. |
| `projects/gait_c3d/measured_load_diagnostics.py` | `compatibility_reference` | Quarantined with --reference-only; exclude from FD runtime, migrate pure QC then delete/rename. |
| `projects/gait_c3d/moco_contact_calibration.py` | `compatibility_reference` | Quarantined with --reference-only; exclude from FD runtime, migrate pure QC then delete/rename. |
| `projects/gait_c3d/newton_contact_calibration.py` | `native_runtime` | Production path; retain and require neutral API guard. |
| `projects/gait_c3d/opensim_contact_parity.py` | `cross_runtime_oracle` | Offline cross-runtime parity fixture only; never import from native runtime. |
| `projects/gait_c3d/opensim_moco_contact_reference.py` | `cross_runtime_oracle` | Offline cross-runtime parity fixture only; never import from native runtime. |
| `projects/gait_c3d/opensim_moco_inverse_reference.py` | `official_oracle` | Offline official oracle only; retain outside production dependency closure. |
| `projects/gait_c3d/opensim_moco_track_reference.py` | `official_oracle` | Offline official oracle only; retain outside production dependency closure. |
| `projects/gait_c3d/opensim_rra_reference.py` | `official_oracle` | Offline official oracle only; retain outside production dependency closure. |
| `projects/gait_c3d/predictive_contact.py` | `compatibility_reference` | Quarantined with --reference-only; exclude from FD runtime, migrate pure QC then delete/rename. |
| `projects/gait_c3d/prepare_newton_contact_input.py` | `source_adapter` | Pre-boundary only; retain, hash-seal output, forbid simulation after publish. |
| `projects/gait_c3d/residual_sensitivity.py` | `compatibility_reference` | Quarantined with --reference-only; exclude from FD runtime, migrate pure QC then delete/rename. |
| `projects/gait_c3d/rra_adjusted_contact_input.py` | `source_adapter` | Pre-boundary only; retain, hash-seal output, forbid simulation after publish. |

## Production dependency closure

```text
official RRA/source assets
  -> rra_adjusted_contact_input.py          [file adapter]
  -> prepare_newton_contact_input.py        [source conversion boundary]
  -> motion_and_targets.npz + topology.json [sealed neutral artifact]
  -> newton_contact_calibration.py           [Newton-native runtime]
```

The production runtime uses `newton.ModelBuilder`, `newton.Model`, `newton.State`, `newton.CollisionPipeline`, `newton.Contacts`, and `SolverSemiImplicit`. It does not directly import `opensim`, `newton.opensim`, an oracle module, or a compatibility-reference module. Importing the public `newton` package currently initializes its exported `newton.opensim` namespace, but the production source neither references nor calls it.

## Quarantined compatibility executables

`measured_load_diagnostics.py`, `predictive_contact.py`, `contact_calibration.py`, `residual_sensitivity.py`, and `moco_contact_calibration.py` execute OpenSim-shaped compatibility mechanics. They declare `ARCHITECTURE_ROLE="compatibility_reference"`, require `--reference-only`, publish `production_eligible=false`, and are excluded from the production dependency closure.

## Native blockers found

1. `add_osim()` produces 27 independent D6 DOFs from 23 source coordinates. Each one-DOF knee incorrectly becomes independent rotation plus two translation DOFs; it is not predictive-ready.
2. A native single-coordinate coupled knee transform, Jacobian, and bias acceleration is required before articulated forward dynamics.
3. MTP must explicitly use either a weld or a ±30 degree native DOF with passive `-25*q-2*qdot` mechanics.
4. Native RRA/CMC are not implemented. Official RRATool/CMC remain offline oracles.

## Automated enforcement

`newton/tests/test_gait_c3d_architecture.py` parses direct and dynamic imports and real call sites, checks transitive production dependencies, discovers committed and untracked modules independently of the allowlist, requires explicit reference acknowledgement, and imports the native runtime in a subprocess that blocks official OpenSim and all adapter/oracle project modules.
