# Area-scaled elastic column friction

Elastic Coulomb friction sets each column's tangential stiffness to
`G_eq A / L`, where `G_eq` is the shoe material's equilibrium shear modulus,
`A` is the column area, and `L` is its rest length. The elastic bristle uses a
Coulomb cap and has no dashpot or relaxation state. This makes the unsaturated
force response scale with the modeled foam geometry and preserves the summed
stiffness when a footprint is refined into smaller columns.

The column bed (`column_bed.ColumnBed`) always uses this law; there is no other
friction model.

This is a shear estimate derived from the fitted normal material, not an
independent outsole shear calibration. The linear shear-layer approximation
can predict large displacements during sliding. Large release/ringing
transients remain, and the model does not establish an independent calibration
of the braking-force transient.
