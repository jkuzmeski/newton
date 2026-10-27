# Column friction

The column bed (`column_bed.ColumnBed`) uses plain Coulomb friction. A landed
column bottom sticks to the ground plane or slides at `mu R`, where `R` is the
column's unilateral normal reaction. The law is stateless: below the cap the
force is a stiff damper `R v / v_stick` with `v_stick = 1e-4 m/s`, so a sticking
bottom creeps at up to `mu v_stick`; above the cap the force is clipped to
`mu R` along the slip direction. There is no bristle spring, stored deflection,
or release dwell. With `carrier_mobility`, the carrier takes the friction at its
end-of-step twist in the same implicit solve as the normal force; without it,
the friction is applied explicitly.

With `carrier_mobility` the cap `mu R` uses the column reaction `R` from the
column balance before the carrier solve. After the solve the normal force is
rescaled to the lift the carrier actually gets, which differs from the balance
on impact and on steps where the compression stop acts. The cap is therefore
lagged against the applied normal force, and the Coulomb cone holds exactly only
when the lift equals the carried load, as in steady stance or sliding. The step
count of the excess is not recorded.

The shear layer between columns is separate from friction. It is a linear
Pasternak layer with `G = 0.1 G_eq`, where `G_eq` is the shoe material's rest
shear modulus. The coupling between two columns joins their layers in series, so
the thinner column bounds the pair.

This is a shear estimate derived from the fitted normal material, not an
independent outsole shear calibration. The model does not establish an
independent calibration of the braking-force transient.
