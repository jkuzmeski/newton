# Column friction

The column bed (`column_bed.ColumnBed`) gives each landed column bottom an
elastic-Coulomb bristle (a reset-integrator or brush model). The bottom's
tangential compliance is a spring of constant stiffness

```text
k_t = bristle_shear_fraction * G_eq * A / L
```

where `G_eq` is the shoe material's rest shear modulus, `A` the column area and
`L` its rest thickness. The spring stores the bottom's tangential deflection
`z`. Each step it takes the deflection the bottom's slip adds, `z + dt v`, and a
radial return holds its force to `mu R` for the column's unilateral ground
reaction `R`:

```text
f_e = -k_t (z + dt v), clipped to |f_e| <= mu R
z  <- -f_e / k_t
f   = f_e + s (-k_t tau v)
```

The damper `k_t tau` with `tau = bristle_damping_time_s` acts only inside the
Coulomb disk: `s` is the largest fraction of it that keeps `|f| <= mu R`, so a
sliding bottom carries exactly `mu R`. A bottom with `R = 0` carries no friction
and loses its deflection, so every landing starts from rest. The column's
`stuck` flag is set while the spring is inside the disk.

The friction is explicit, like the normal force: the carrier gets it at its
start-of-step twist. With the defaults (`bristle_shear_fraction = 1`,
`tau = 1 ms`) the bristle is far softer than the explicit limit of a 1 kg
foot at 1e-4 s, and the force is continuous when a sliding bottom is caught.
Neither default is fitted; shear-compliance data for the shoe would pin them
down. Keep `tau` at least 4 substeps: the explicit damper does net work on the
carrier at toe-off when `tau` is only 2 substeps (measured +0.83 J at
`dt = 5e-4`, `tau = 1 ms`), while 4 and 10 substeps and `tau = 0` stay passive.

The shear layer between columns is separate from friction. It is a linear
Pasternak layer with `G = 0.1 G_eq`. The coupling between two columns joins their
layers in series, so the thinner column bounds the pair.

These are shear estimates derived from the fitted normal material, not an
independent outsole shear calibration. The model does not establish an
independent calibration of the braking-force transient.
