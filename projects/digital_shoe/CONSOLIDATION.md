# Shared shoe mechanics

Status: implemented. Shared-source, forward/autodiff and full-bed replay checks pass.

## One home for each law

- `projects.digital_shoe.material`: one source for Ogden-Hill pressure and the
  Maxwell recurrence. The same function bodies run with NumPy elementary
  operations for host fitting and compile with Warp for forward and autodiff.
- `projects.digital_shoe.contact`: unilateral reaction, material-pinned pair
  coupling, passive balance, bristle contact, contact points and wrench mapping.
- `projects.digital_shoe.runtime`: batched mutable-state stepping and periodic
  cycle adapters. These adapters do not own another constitutive/contact law.
- `projects.digital_shoe.rendering`: shared bench and carried-shoe geometry.
- `digital_instron_v2.dynamics_diff` and `inverse_id`: tape-safe/periodic state
  layouts and optimization, calling the same material and contact functions.

A carried shoe and an Instron fixture have different boundary coordinates. A
carried shoe uses nominal outsole anchors and a declared ground plane. A bench
uses imposed indenter/top positions relative to its rest foam-top datum. These
are geometry and loading adapters, not alternate foam laws. Their reported
external ground wrench and transferred indenter load are named separately.

## Guarantees and limits

- Preserve the checked material, source geometry and measured input files.
- Keep prior public import names as aliases or deprecated compatibility wrappers.
- No new required dependencies, per-step allocations or host reads.
- Preserve batched-world isolation and CUDA-graph capture.
- Check forward and differentiable state values as well as finite-difference
  gradients. Sharing only a test while retaining copied equations is insufficient.
- Retain the actual upper-last seating limitation from the passive-column audit;
  consolidation does not create a calibrated upper/insole interface.
- Fixed-order forward cycle-force reduction avoids atomic scheduling noise in
  the objective. This is not a claim that every parallel reverse-mode gradient
  or optimizer is bitwise reproducible across GPU/software versions.

Validation results and runnable examples will be recorded in
`outputs/impedance_instron/consolidation/`.
