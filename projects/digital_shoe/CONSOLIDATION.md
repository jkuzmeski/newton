# Shared shoe mechanics

Status: implemented.

## One home for each law

- `projects.digital_shoe.material`: one source for Ogden-Hill pressure and the
  Maxwell recurrence. The same function bodies run with NumPy elementary
  operations for host fitting and compile with Warp for simulation.
- `projects.digital_shoe.contact`: material-pinned pair coupling, bristle laws
  and the two-surface column balance.
- `projects.digital_shoe.column_bed`: the batched GPU column bed. Each column has
  a bottom that lands on the ground plane and a top that is bonded to, pressed
  by, or free of the carrier surface.
- `projects.digital_shoe.runtime`: the material block and `FoundationConfig`.
- `projects.digital_shoe.rendering`: column drawing from solved endpoints.

A carried shoe and an Instron fixture use the same column bed with different
top modes. A carried shoe bonds its tops to the carrier and lands its bottoms on
the ground plane. A bench presses the tops with the indenter and fixes the
bottoms to the platen. Their reported ground reaction and carried indenter load
are named separately.

## Guarantees and limits

- Preserve the checked material, source geometry and measured input files.
- No new required dependencies, per-step allocations or host reads.
- Preserve batched-world isolation.
- Retain the actual upper-last seating limitation from the passive-column audit;
  the column bed does not create a calibrated upper/insole interface.
