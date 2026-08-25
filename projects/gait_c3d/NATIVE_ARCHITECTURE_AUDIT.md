# C3D project native-architecture audit

Audited the complete current `projects/gait_c3d` tree plus every branch-changed support file: `61` files.

## Directory contract

- `projects/gait_c3d/newton_*.py`: production Newton-native mechanics.
- `projects/gait_c3d/adapters/`: source conversion before the sealed neutral boundary.
- `projects/gait_c3d/oracles/`: official or cross-runtime offline reference tools.
- `projects/gait_c3d/compatibility/`: historical `newton.opensim` analysis/reference executables; every CLI requires `--reference-only`.

No production module imports any of the three boundary namespaces.

## File-by-file inventory

| File | Role | Contract |
|---|---|---|
| `changelog/+c3d-measured-load-diagnostics-31a7c9e2.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+c3d-newton-native-contact-boundary-8f4c2a71.changed.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+c3d-normal-contact-fit-91e6b3c4.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+c3d-predictive-contact-7f2d4a91.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+c3d-residual-sensitivity-54b8e0d3.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+moco-track-toe-bounds-6a42f17c.fixed.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+official-moco-contact-reference-0d7b315e.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+official-moco-inverse-reference-5b741d8e.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+official-moco-track-contact-5f6c8a1e.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+official-opensim-rra-reference-6c31e7a4.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+opensim-contact-cuda-a617e4c2.changed.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+opensim-contact-parity-b8d402f1.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+opensim-forward-contact-3e7a61b9.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+opensim-modern-motion-types-5de8b9c1.fixed.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+opensim-modern-transform-functions-2a9d71c4.fixed.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+rra-adjusted-contact-input-1cf458d2.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `changelog/+s001-moco-contact-calibration-figures-31e8d6a4.added.md` | `documentation/changelog` | Documents scope and provenance. |
| `newton/_src/opensim/contact.py` | `compatibility_library/test` | Optional newton.opensim library or its tests; not production. |
| `newton/_src/opensim/dynamics.py` | `compatibility_library/test` | Optional newton.opensim library or its tests; not production. |
| `newton/_src/opensim/kinematics.py` | `compatibility_library/test` | Optional newton.opensim library or its tests; not production. |
| `newton/_src/opensim/parser.py` | `compatibility_library/test` | Optional newton.opensim library or its tests; not production. |
| `newton/tests/test_gait_c3d_architecture.py` | `native_architecture_test` | Enforces native dependency closure. |
| `newton/tests/test_gait_c3d_contact_calibration.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_measured_load_diagnostics.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_moco_contact_calibration.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_newton_native_contact.py` | `native_architecture_test` | Enforces native dependency closure. |
| `newton/tests/test_gait_c3d_opensim_contact_parity.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_opensim_moco_contact_reference.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_opensim_moco_inverse_reference.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_opensim_moco_track_reference.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_opensim_rra_reference.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_pipeline.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_predictive_contact.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_residual_sensitivity.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_gait_c3d_rra_adjusted_contact_input.py` | `reference_or_adapter_test` | Reference/adapter behavior test only. |
| `newton/tests/test_opensim.py` | `compatibility_library/test` | Optional newton.opensim library or its tests; not production. |
| `projects/gait_c3d/ARCHITECTURE_BOUNDARIES.json` | `architecture_policy` | Machine-readable architecture policy. |
| `projects/gait_c3d/FORWARD_DYNAMICS_ROADMAP.md` | `documentation/changelog` | Documents scope and provenance. |
| `projects/gait_c3d/NATIVE_ARCHITECTURE_AUDIT.md` | `documentation/changelog` | Documents scope and provenance. |
| `projects/gait_c3d/README.md` | `documentation/changelog` | Documents scope and provenance. |
| `projects/gait_c3d/__init__.py` | `compatibility_entrypoint` | Lazy launcher for explicit reference-only pipeline. |
| `projects/gait_c3d/adapters/__init__.py` | `adapter_namespace` | Physically isolated source-adapter namespace. |
| `projects/gait_c3d/adapters/prepare_newton_contact_input.py` | `source_adapter` | Pre-boundary conversion only; hash-seal neutral output. |
| `projects/gait_c3d/adapters/rra_adjusted_contact_input.py` | `source_adapter` | Pre-boundary conversion only; hash-seal neutral output. |
| `projects/gait_c3d/compatibility/__init__.py` | `compatibility_namespace` | Physically isolated compatibility namespace. |
| `projects/gait_c3d/compatibility/contact_calibration.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/human_shoe.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/measured_load_diagnostics.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/moco_contact_calibration.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/pipeline.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/predictive_contact.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/residual_sensitivity.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/torque_reconstruction.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/compatibility/viewer.py` | `compatibility_reference` | Reference-only CLI; never production reachable. |
| `projects/gait_c3d/newton_contact_calibration.py` | `native_runtime` | Production mechanics; neutral Newton APIs only. |
| `projects/gait_c3d/oracles/__init__.py` | `oracle_namespace` | Physically isolated oracle namespace. |
| `projects/gait_c3d/oracles/opensim_contact_parity.py` | `cross_runtime_oracle` | Offline cross-runtime parity oracle. |
| `projects/gait_c3d/oracles/opensim_moco_contact_reference.py` | `cross_runtime_oracle` | Offline cross-runtime parity oracle. |
| `projects/gait_c3d/oracles/opensim_moco_inverse_reference.py` | `official_oracle` | Offline official OpenSim oracle. |
| `projects/gait_c3d/oracles/opensim_moco_track_reference.py` | `official_oracle` | Offline official OpenSim oracle. |
| `projects/gait_c3d/oracles/opensim_rra_reference.py` | `official_oracle` | Offline official OpenSim oracle. |

## Production chain

```text
source/RRA files
  -> adapters/prepare_newton_contact_input.py
  -> sealed newton_contact_input_v1
  -> newton_contact_calibration.py
  -> ModelBuilder / Model / State / CollisionPipeline / Contacts / SolverSemiImplicit
```

## Beyond contact

The historical scaling, IK, inverse dynamics, torque reconstruction, Static Optimization, shoe replay, and visualization modules are now physically isolated under `compatibility/`. They are not yet rebuilt as neutral Newton production mechanics.

The next native model milestone is a neutral articulated gait compiler. It must implement the one-coordinate coupled knee (transform, Jacobian, and bias acceleration), explicit MTP mechanics, native actuators/passive forces, and native inverse-dynamics parity before any free forward rollout. Current `add_osim()` output is rejected for predictive use because it creates 27 independent D6 DOFs from 23 source coordinates.

## Enforcement

`ARCHITECTURE_BOUNDARIES.json` classifies every current Python module. `test_gait_c3d_architecture.py` checks every module, transitive production imports, real native calls, explicit compatibility acknowledgement, and a subprocess import with adapters/oracles blocked.
