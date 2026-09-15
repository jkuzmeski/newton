<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Footwear simulation: contact and material laws

## Digital Shoe and Digital Instron — two-term model

**Technical report · current implementation audit · internal research use**

This report describes the **two-term Ogden–Hill / Maxwell / Pasternak foundation** used by Digital Shoe, Digital Instron, and the current impedance Instron experiment. All material values, numerical figures, saved validation results, and example commands refer to one explicitly selected two-term artifact. No alternative material generation is used.

**Artifact:** `outputs/impedance_instron/inputs/digital_shoe.json`  
**SHA-256:** `54d756cea78e22a6d253b578cf4c5db279ece63aed3ce359efffb6988d6c765f`  
**Source snapshot:** `00c2804c30f7c56d8a6ea61875cfdafda44708cc`

### Main conclusions

1. The shoe is a **geometry-dependent bed of nonlinear viscoelastic columns**, not a deforming three-dimensional finite-element mesh. Neighbor coupling lets a loaded region recruit adjacent foam.
2. **Two hyperelastic terms** describe the equilibrium compression curve. **One Maxwell memory variable per column** adds rate dependence. “Two-term” does not mean two Maxwell relaxation times or two separate shoe layers.
3. Ground contact is **compression-only normal support plus anchored Coulomb bristles**. Material response, spatial load transfer, friction, and numerical stabilization are distinct mechanisms.
4. Digital Instron identifies **six shared material parameters** from two fixture tests. Friction, tangential stiffness, normal stabilization damping, and the passive-surround lag are not identified material parameters.
5. The selected artifact passes all six stored adjacent-cycle validation gates. This is narrow bench evidence, **not validation of gait, impact, outsole friction, or last-to-shoe seating**.
6. The carried-shoe interface to the rigid last is an **idealized backing over a projected footprint**. The displayed last mesh does not solve a second contact problem against the insole.

**How to read the figures.** Geometry and validation figures come from the selected artifact. Constitutive and bristle figures are fresh, controlled calculations of the implemented laws. They explain mechanisms; they are not additional measurements. Full details are in `figures/metadata.json`.

## 1. What each project does

| Component | Role | What it does not establish |
|---|---|---|
| Digital Instron | Prepares bench cycles and geometry; fits, checks, and exports the material | It does not independently identify friction from normal compression |
| Digital Shoe | Owns the artifact, shared material/contact functions, runtime, and mechanical demonstrations | Its visual meshes are not automatically deformable or collision surfaces |
| Impedance Instron, current `simple/` workflow | Applies the same shoe law beneath a reduced dynamic foot/upper-mass rig | Tracking a movement target is not a new material identification experiment |

Data flow is **measurements → identification → portable artifact → simulation**. Code reuse points in the other direction: both Instron workflows call the shared `digital_shoe.material`, `contact`, and `runtime` functions. The fitting, forward, and active differentiable adapters do not maintain different active constitutive equations. [S1–S3, S6, S11–S13]

## 2. Geometry: what is actually discretized

![Two-term artifact geometry and fixture footprints](../../../outputs/footwear_contact_material_report/figures/bed_geometry.svg)

**Figure 1.** The sampled bed and driven fixture footprints. These are model columns and prescribed support regions, not a measured pressure map.

The bed samples the supplied midsole geometry on a **5 mm grid**. Each valid vertical ray defines a nominal outsole location, a rest thickness, a tributary area, and up to four neighbors. The selected bed has:

| Quantity | Value |
|---|---:|
| Number of columns | 910 |
| Tributary area per column | 25 mm² = 0.000025 m² |
| Total sampled plan area | 227.5 cm² |
| Rest thickness range | 5.033–43.768 mm |
| Full-foot driven footprint | 611 columns; 152.75 cm² |
| Rearfoot driven footprint | 62 columns; 15.5 cm² |

The complete 910-column bed remains in both bench models. Columns outside the fixture footprint are **passive**, not deleted. The force applied through a small punch can therefore spread into the surrounding bed. A punch footprint is discretized; its sampled area is not an exact circular-contact integration.

The artifact uses meters and newtons, a right-handed frame, **+Z upward**, and an X direction recorded as heel-to-toe for this asset. The coordinate origin is the footprint center in XY and the lowest outsole height in Z. Consumer-side orientation and left/right mirroring must remain explicit.

For column `i`, write rest thickness `t_i`, area `A_i`, compression `c_i`, compressive engineering strain `ε_i`, and remaining stretch `λ_i`:

```text
c_i = max(z_free,i − z_anchor,i, 0)
ε_i = c_i / t_i
λ_i = max(1 − ε_i, λ_min)
```

The meaning of `z_free` and the anchor depends on the boundary adapter. On the bench, the anchor is a top/indenter reference. For a carried shoe, the anchor denotes the nominal outsole and the declared plane supplies the contact datum; passive columns use a solved pressure reference. **Do not interpret `z_free` or a friction anchor as a directly rendered material endpoint.**

This is a reduced vertical-compression model. It has no full three-dimensional strain tensor, no lateral Poisson displacement solve, no independently deforming upper, and no resolved rubber tread. Curvature and local thickness enter through geometry, while spatial coupling approximates lateral transfer. [S4, S7, S14]

## 3. Equilibrium material law: two-term Ogden–Hill

![Equilibrium pressure and two-term contributions](../../../outputs/footwear_contact_material_report/figures/material_equilibrium.svg)

**Figure 2.** Each term contributes to the same equilibrium network. The second term becomes increasingly important at large compression. The rapid-loading curve is the ideal instantaneous limit of the same material, not a separate material fit.

The effective Poisson ratio is fixed at **ν = 0**. The general volumetric coefficient therefore vanishes, and the implemented compression law simplifies to:

```text
G_eq,1 = f_eq G_inst,1                 G_eq,2 = f_eq G_inst,2

p_eq(λ) = p_1(λ) + p_2(λ)
p_n(λ)  = [2 G_eq,n / (α_n λ)] [1 − λ^(α_n)]      n = 1, 2
```

Pressure is positive in compression and is measured in Pa. Both terms use the same strain and equilibrium fraction. The first modulus and exponent do not describe one physical layer while the second describes another. They are two mathematical contributions to one effective shoe-level compression curve.

The fitted positive exponent gives a comparatively broad, softer compression response. The negative exponent produces a stronger increase as the remaining thickness becomes small. Neither individual exponent is a universal material constant independent of the identification protocol.

### 3.1 Exact fitted values and derived moduli

| Parameter | Symbol | Selected two-term value | Meaning |
|---|---|---:|---|
| Instantaneous shear modulus, term 1 | G_inst,1 | 267.615 kPa | Amplitude of term 1 in the rapid limit |
| Exponent, term 1 | α₁ | 18.07698 | Shape of term 1 |
| Instantaneous shear modulus, term 2 | G_inst,2 | 19.4877 kPa | Amplitude of term 2 in the rapid limit |
| Exponent, term 2 | α₂ | −0.5855868 | Shape of term 2 |
| Equilibrium fraction | f_eq | 0.6586721 | Long-term fraction of the instantaneous modulus |
| Maxwell time | τ | 0.00515011 s | Decay time of the single overstress branch |

Derived quantities:

```text
G_eq,1 = 176.271 kPa                 G_eq,2 = 12.8360 kPa
G_eq   = G_eq,1 + G_eq,2 = 189.107 kPa
E_eq   = 2 G_eq = 378.213 kPa         [small-strain compression, ν = 0]
E_inst = 2 (G_inst,1 + G_inst,2) = 574.206 kPa
```

The exponents change the finite-strain curve, but each term contributes `2 G_eq,n` to the small-strain compressive tangent. Small-strain tangent modulus, secant stiffness over a finite range, and whole-shoe stiffness in N/m are different quantities. Whole-shoe stiffness also depends on thickness, loaded area, fixture geometry, and neighbor recruitment.

### 3.2 What ν = 0 means here

Zero effective Poisson ratio is a **model assumption**, supported by literature on a related foam class, not a measured property of this particular intact Puma shoe. McCulloch, Delp, and Kuhl report similar confined and unconfined responses for the tested Asics foams. Their specimens and shoe materials are not the fitted intact shoe. The primary paper is used as context, not as validation or as a source for the six fitted numbers. [E1]

The choice makes an independent thickness-compression response plausible without a lateral dilation solve. It does not make the foundation a general isotropic finite-element material. “Hyperfoam” names the constitutive family; the runtime evaluates its **reduced pressure law**, not a full continuum stress update. [S1]

### 3.3 Numerical safeguards

The live runtime default stretch floor is `λ_min = 0.05`. The fitting forward path explicitly uses `λ_min = 0.001`; shared material expressions therefore do not imply identical clipping settings in every adapter. It keeps the material expression finite as a column approaches collapse. Above 95% compression the pressure is evaluated at the floor; this is **not an infinite-stiffness hard stop** and does not, by itself, prevent further driven displacement.

The passive solve separately caps passive strain, normally at 90%, while driven columns do not use that passive cap. These are different safeguards. Clipped samples must be reported; they cannot be interpreted as validated densification behavior.

For `|α| < 0.001`, the source uses the removable-limit approximation `−2 G_eq ln(λ)/λ` for that term. This is a numerical branch with a zero alpha derivative inside the interval. The selected exponents are outside it. [S1–S3]

## 4. Time dependence: one Maxwell branch

![Illustrative two-term material cycles and relaxation](../../../outputs/footwear_contact_material_report/figures/material_rate_dependence.svg)

**Figure 3.** Controlled material-point histories calculated with the shared two-term pressure and Maxwell recurrence. Rate changes and hold relaxation expose the memory mechanism. They are not new experimental validation curves and omit spatial coupling and rigid-body feedback.

Each column stores one overstress `q_i` [Pa] and its previous equilibrium pressure. Define:

```text
r = (1 − f_eq) / f_eq
q_dot + q/τ = r p_eq_dot
p_raw = p_eq + q
```

The state advances during flight as well as contact; lift-off does not reset all Maxwell memory. A loading ramp generally gives positive overstress. During unloading, overstress can be negative and reduces the resultant pressure. The equilibrium curve alone therefore does not describe a full loading/unloading cycle.

The shared discrete update is:

```text
a = exp(−Δt/τ)
b = τ(1 − a)/Δt
q_new = a q_old + r b (p_eq,new − p_eq,old)
```

This is the exact branch solution **if equilibrium pressure varies linearly over the step**. It is not an exact solution of the entire coupled contact and rigid-body problem.

At fixed compression, the overstress decays exponentially. For an ideal instantaneous step from a fully relaxed unloaded state:

```text
p(0+) = p_eq / f_eq
p(t)  = p_eq [1 + r exp(−t/τ)]
p(∞)  = p_eq
```

The current `τ ≈ 5.15 ms` is much shorter than the approximately 0.5 s bench cycle, although loading/unloading transitions can still excite it. Starting every measured cycle from `q = 0` would introduce a different history assumption. The calibration accounts for periodic history and solves the passive field consistently with the Maxwell state.

This is a nonlinear-equilibrium, scalar-overstress model. It contains no explicit fatigue damage, permanent compression set, temperature shift, aging, or a spectrum of independent relaxation times. Passing adjacent-cycle gates cannot establish those behaviors. [S1, S3, S6, S11]

## 5. Spatial load transfer: the Pasternak layer

The material law gives each column a local normal response. The Pasternak layer lets neighboring compression values exchange vertical load:

```text
K_ij = G_eq (t_i + t_j)/2                [N/m]
Φ_i  = Σ over neighbors j: K_ij (c_j − c_i)    [N]
T_i  = R_i − Φ_i                         [N]
```

`R_i` is the unilateral local normal reaction, defined in Section 6. `T_i` is the **signed transferred vertical load** associated with the carrier/indenter boundary.

Because `K_ij = K_ji`, every pair contributes equal and opposite transfer. With a symmetric neighbor table:

```text
Σ over complete bed: Φ_i = 0
Σ T_i = Σ R_i
```

The equality is a whole-bed identity, not a statement that each column's local contact force equals its top transfer. Floating-point summation can leave a small residual. If only a driven subset is summed, the transfer across the subset boundary is physically relevant.

For a square grid with `A = h²`, integrating a pressure-level Laplacian over a cell cancels its `1/h²` factor. The implemented pair coefficient therefore has units N/m and directly multiplies a compression difference. Do not add an extra grid-spacing factor to the source equation.

A useful elastic interpretation is an edge energy `½ K_ij(c_i − c_j)²`. This explains why the layer resists differences in adjacent compression. It is not outsole friction and is not a separate lateral-displacement continuum solve.

The column quantity `G_eq t_i` spans **951.84–8276.74 N/m**, with a bed mean of **5632.83 N/m**. The artifact's `pasternak_n_per_m` records that derived mean. Runtime pair forces use the actual thicknesses and the sum of both equilibrium term moduli, **not a fitted uniform spring equal to the mean**.

Pasternak coupling is equilibrium-elastic in the implementation. It has no independent Maxwell state. Tying it to `G_eq` removes an independent fit parameter but adds a structural modeling assumption. The coupling scale in the current workflow is one. A non-default scale affects the passive solve but does not uniformly rescale the final transfer diagnostic, so it must not be interpreted as a simple global material multiplier. [S2, S3, S6]

## 6. Normal contact: support is not signed transfer

Let `v_z,i` be the velocity of the force application point, positive upward. Let `d_n` be optional per-column normal damping in N·s/m. The shared normal reaction follows this order:

```text
R_trial,i = A_i max(p_eq,i + q_i, 0) − 1[c_i > 0] d_n v_z,i
R_i       = max(R_trial,i, 0)

For an explicit ground plane:
    if nominal outsole gap g_i > 0, set R_i = 0.
```

The two clamps have different roles. The pressure clamp suppresses tensile material support. The final clamp prevents the damping term from creating adhesion during separation. For downward motion, `v_z < 0`, damping adds positive support; during upward motion it reduces support. Ground contact is also suppressed when the nominal outsole anchor clears the plane.

In the current impedance rig, `d_n = 0`. The Maxwell branch remains active; zero added normal damping does **not** mean an elastic, lossless shoe.

### 6.1 Bench boundary

A bench fixture imposes compression on the driven top boundary. The model reads transferred indenter reaction. Passive columns relax outside the driven footprint. This is not a mesh-on-mesh contact search against the displayed punch or last. The fixture mask supplies a prescribed loading region.

The bench uses the signed transfer `T_i = R_i − Φ_i` for load accounting. Positive transfer can be used for a top-boundary support diagnostic. A top-transfer center is not automatically an external outsole center of pressure.

### 6.2 Carried-shoe ground boundary

The carried-shoe path declares a horizontal plane at height `z_g`. A nominal outsole point is transformed with the carrier, then projected to that plane for force and velocity evaluation:

```text
g_i = z_nominal,i − z_g
x_contact,i = (x_nominal,i, y_nominal,i, z_g)
```

For this path, **ground friction capacity and ground wrench use `R_i`, not `max(T_i, 0)`**. The signed `T_i` remains a separate internal-transfer diagnostic. Clamping every negative internal transfer before summation would create spurious net support under neighboring foam.

The ground model is a compliant foundation. For driven columns, nominal outsole penetration is the compression coordinate. Passive compression is separately relaxed and bounded above by penetration; it is not generally equal to it. This is not the same concept as interpenetration between two undeformed rigid meshes. There is no generic hard-contact complementarity solve replacing the shoe's force law. The active rig disables shape collision on its displayed meshes. [S2, S3, S5, S13]

## 7. Passive foam outside the driven footprint

An undriven column is not fixed to a hidden rigid support. Its compression is iterated toward a local balance:

```text
R_i(c_i, history) + k_attach (c_i − c_ref,i)
    − Σ_j K_ij (c_j − c_i) = 0
```

The default `k_attach = 0`. Nonzero attachment would introduce another support contribution and would require explicit force and energy accounting; it is not part of the reported two-term identification.

In this balance, `R_i` is the material-only pressure support; the passive solver does not include extra normal damping. That distinction vanishes in the current impedance rig because its extra normal damping is zero.

The shared solver takes constrained, relaxed Newton/Jacobi steps. It estimates a local material tangent using a small forward compression increment (`0.001 t_i`), includes attachment and neighbor diagonal stiffness, and clips the update into the admissible compression interval. The runtime includes the upcoming Maxwell increment in the local balance.

With `carrier_bond=True`, the free compression is also bounded above by nonnegative nominal carrier penetration. This prevents passive support from remaining planted while the carried shoe lifts into flight. It is a **one-sided retention bound**, not a bilateral bond of every passive top to a rigid foot.

The bench replay uses a quasi-static passive solve with 32 sweeps per substep. The current impedance rig instead uses four sweeps and a declared **2 ms outer-relaxation lag**. This extra lag is distinct from the **5.15 ms Maxwell material time**. The shared material functions are the same, but these boundary and numerical settings need not produce identical histories under the same nominal movement. Sweep, time-step, and lag sensitivity remain relevant. [S2, S3, S5, S13]

## 8. Tangential contact: anchored Coulomb bristles

![Shared bristle force-displacement response](../../../outputs/footwear_contact_material_report/figures/contact_bristle.svg)

**Figure 4.** A controlled tangential cycle through the actual shared bristle update at declared constant normal load. Elastic sticking, return to the Coulomb bound, and unloading depend on a persistent contact anchor. This is a mechanism demonstration, not a measured friction test.

At a contacting column, the tangent plane is world XY. The state consists of an anchor `a`, a grip flag, and an unloaded dwell timer. Tangential stiffness `k_t` is in N/m and damping `k_v` in N·s/m. The friction coefficient is `μ`.

### 8.1 Elastic trial and sliding return

For tangent position `x` and velocity `v`:

```text
F_max = μ R_i                            [carried-shoe ground contact]
x_trial = x + Δt v
f_el,trial = −k_t (x_trial − a)
```

A newly contacting bristle sets its anchor to the current position. If the trial force exceeds the Coulomb disk, the source returns it radially:

```text
if ||f_el,trial|| > F_max:
    f_el = F_max f_el,trial / ||f_el,trial||
    a_new = x_trial + f_el/k_t
else:
    f_el = f_el,trial
```

The `x + Δt v` trial uses current velocity; it is not a fully implicit dynamics solve. The rigid-body solver subsequently advances with updated velocity. This bristle trial alone proves neither unconditional stability nor global passivity.

The anchor update represents accumulated slip. The code's grip flag indicates an attached bristle history; it is not a separate measured static-versus-kinetic friction state and remains active during sliding.

### 8.2 Viscous regularization without breaking the cone

The viscous trial is opposite the slip velocity and is capped before addition:

```text
f_vis = −v/||v|| min(k_v ||v||, γ F_max)
f_tan = f_el + η f_vis
```

`η` is the largest fraction in `[0,1]` that keeps the total force inside the disk. The implementation solves the quadratic inequality `||f_el + η f_vis||² ≤ F_max²`. It does not simply add a damper outside the friction limit.

The retained viscous force opposes velocity. The elastic bristle can store and return energy, so instantaneous positive tangential power during unloading is not automatically an error. Sliding return and viscous work must be distinguished from elastic storage before making a passivity claim.

### 8.3 Loss of contact and area scaling

When normal load is absent, tangential force is zero. Setting `k_t = 0` also disables the complete bristle branch; a positive `k_v` alone does not create viscous-only friction. The anchor can survive a short unloading dropout. If unloading exceeds the configured dwell, or stiffness is disabled, the bristle resets. The default dwell is **0.5 ms**.

Stiffness and damping are normalized by tributary area internally:

```text
k_t,i = k'' A_i          k'' in N/m³
k_v,i = c'' A_i          c'' in N·s/m³
```

This preserves tangent-layer stiffness for a fixed contact patch when sampling is refined with consistent per-area settings. It does not guarantee full grid convergence of geometry, normal coupling, or all other parameters.

The current 25 mm² column settings imply `k'' = 4.0×10⁸ N/m³` and `c'' = 4.0×10⁵ N·s/m³`. At an illustrative 10 N normal force and `μ = 0.8`, a 10,000 N/m elastic bristle reaches its 8 N cap at 0.8 mm elastic displacement. Real columns have varying normal load, so their slip thresholds vary.

There is no fitted velocity-dependent friction curve, Stribeck law, rolling-resistance law, separate static/kinetic coefficient, wet-surface law, or tread contact resolution here. [S2, S3, S13]

## 9. Parameters: identified, derived, assumed, or numerical?

| Mechanism / setting | Current value | Status |
|---|---:|---|
| Two instantaneous moduli, two exponents, equilibrium fraction, Maxwell time | Section 3 table | Six shared fitted parameters |
| Effective Poisson ratio | 0 | Fixed modeling assumption |
| Pasternak faces | G_eq × mean neighboring thickness | Derived from the fitted material and geometry |
| Outer attachment | 0 N/m | Declared boundary assumption |
| Ground friction μ | 0.8 | Assumed in current impedance rig; not bench-fitted |
| Tangential stiffness | 10,000 N/m per 25 mm² column | Assumed tangent layer |
| Tangential damping | 10 N·s/m per 25 mm² column | Assumed regularization |
| Viscous fraction γ | 0.2 | Assumed cap on tangent damping |
| Anchor release dwell | 0.0005 s | Contact-history regularization |
| Added normal damping | 0 N·s/m in current impedance rig | Scenario setting; Maxwell losses remain |
| Live stretch floor | 0.05 | Numerical/material-extrapolation safeguard |
| Calibration stretch floor | 0.001 | Explicit fitting-adapter setting |
| Passive maximum strain | 0.90 | Passive-solve constraint |
| Impedance passive lag / sweeps | 0.002 s / 4 | Scenario and numerical settings |
| Bench replay passive lag / sweeps | 0 s / 32 | Quasi-static numerical settings |
| Impedance frame rate / substeps | 120 Hz / 64 | Time integration settings |

These are defaults in the audited current paths. A saved reference or checkpoint can freeze an explicit configuration; inspect that configuration rather than assuming the table overrides it. Digital Shoe's bare `FoundationConfig` disables friction unless the consuming example enables it. A bench compression fit and a dynamic carried shoe therefore share a material without necessarily sharing every scenario setting.

An 80 kg drop is an optional mechanical demonstration, not an identified boundary condition. Its extra normal damping must be disclosed separately. Neither controller gains nor controller damping should be relabeled as foam material parameters. [S3, S5, S13]

## 10. Force, moment, center of pressure, and solver integration

For the carried-shoe path:

```text
f_i = (f_tan,x,i, f_tan,y,i, R_i)
F = Σ_i f_i
M_COM = Σ_i (x_contact,i − x_COM) × f_i
M_origin = Σ_i x_contact,i × f_i
COP_xy = Σ_i R_i x_contact,xy,i / Σ_i R_i
```

COP is meaningful only when total normal force is positive; a diagnostic threshold should be declared near flight. A full ground wrench includes horizontal force and its lever arm. Reporting only `F_z` and a normal-load centroid discards information relevant to carrier pitch.

The source also computes rigid-body contact power using the velocity at the same force point:

```text
v_contact,i = v_COM + ω × (x_contact,i − x_COM)
P_contact = Σ_i f_i · v_contact,i
```

This is power delivered to the rigid carrier under the massless-shoe approximation. It is not, by itself, a complete balance of material memory, passive relaxation, friction storage, numerical clipping, and actuator work.

The runtime accumulates its custom foundation wrench into `newton.State.body_f`. The current impedance rig then advances `newton.solvers.SolverSemiImplicit` with no generic contact object, and applies a planar guide. The guide restricts the modeled degrees of freedom; it is not an anatomical foot/shank constraint.

One substep conceptually performs:

1. Read the current carrier pose and body velocity, and clear accumulated body forces.
2. Iterate the passive surround, using current material history.
3. Compute compression and equilibrium pressure; update Maxwell overstress.
4. Evaluate unilateral normal support, neighbor transfer, and bristle force.
5. Reduce the contact wrench at the correct force points.
6. Add actuator forces and record the current force-evaluation sample.
7. Integrate the rigid bodies and apply the declared planar restriction.

Force traces refer to the **pre-integration force-evaluation time**, not the later display pose. The impedance frame count is rounded to cover the reference duration; its actual substep is `duration / ceil(duration × 120) / 64`, approximately 0.130 ms, rather than unconditionally exactly `1/7680 s`.

CUDA graphs reduce launch overhead without defining a new law. Shared-law tests compare CPU/CUDA values and derivatives. Active differentiable adapters retain the same piecewise bristle law but use separate history buffers. Derivatives at contact activation, radial return, reset, and clipping boundaries remain branch-dependent. Agreement away from these events does not imply global differentiability or bitwise reproducibility. [S2, S3, S12, S13, S15]

## 11. The upper last is not a solved contact interface

This distinction is essential for interpreting a realistic shoe rendering:

- The rigid-last mesh in the current impedance rig has **shape collision disabled**.
- Its projected footprint determines which columns receive rigidly imposed backing motion.
- The runtime does not solve gap-aware insole seating against that mesh.
- There is no separately calibrated upper-last normal penalty law, upper friction law, or lacing/upper compliance model in this contact path.
- Passive `carrier_bond` only bounds compression; it does not establish a full anatomical attachment.

A mesh-clearance diagnostic can reveal inconsistencies, but it is not a contact solve. Retuning the six material parameters to hide a registration or seating error would confound geometry with material. A coherent upper-interface model would need consistent surface registration, gap kinematics, normal contact, tangential retention, and force/energy accounting.

The defensible statement is: **the current simulation applies an effective two-term shoe foundation beneath an idealized rigid backing and resolves its external plane-contact wrench**. It does not resolve every footwear interface. [S13, S14]

## 12. How the material was identified

The same six-parameter vector is fitted to rearfoot-punch and full-foot-last compression cycles. There are no fixture-specific material parameters. The selected artifact records **training cycles 90–98** and **held-out cycles 99–100**.

The bounded SciPy least-squares workflow scores force samples plus loop-area and peak residuals. With `s = max(max measured force, 1 N)`:

```text
r_force,k = (F_pred,k − F_meas,k) / s
r_loop = w_loop (W_pred − W_meas) / max(|W_meas|, tiny)
r_peak = w_peak (max F_pred − max F_meas) / s
```

The implemented weights are `w_loop = 5` and `w_peak = 6`. Residual vectors are concatenated across trials. The whole-cycle loop integral in this objective is not the same calculation as the official active-branch held-out hysteresis metric. Training loss and validation gates must not be used interchangeably.

The periodic forward solve uses an interval-ratio stopping estimate that is **not a certified remaining-error bound**. Replay and unit-test agreement do not certify that estimator.

The current fit bounds place `α₁` between 0.1 and 20, `α₂` between −20 and 20, `f_eq` between 0.01 and 1, and `τ` between 5 ms and 2 s. The selected `τ = 5.15 ms` is close to its lower bound, and `α₁ = 18.08` is near its upper bound. That warrants bound-sensitivity and identifiability checks rather than treating the printed precision as physical certainty.

The artifact stores a selected training mean-squared residual of approximately **0.00181057** among its recorded starts. It does not supply a complete parameter uncertainty distribution. The present report does not rerun identification or convert stored fit results into a claim of unique parameters.

Same-rate periodic compression constrains combinations of equilibrium fraction and relaxation time. Multi-rate loading and explicit relaxation holds are needed to separate how much stress relaxes from how quickly it relaxes. Shared fixture success also does not prove that the material/geometry decomposition is unique. [S6, S8, S10, S11]

## 13. Two-term held-out validation: what the data show

![Rearfoot measured and predicted held-out response](../../../outputs/footwear_contact_material_report/figures/validation_rearfoot.svg)

**Figure 5.** Stored adjacent-cycle rearfoot data and two-term model prediction. The time trace tests force timing and shape; the force–compression loop tests loading/unloading behavior.

![Full-foot measured and predicted held-out response](../../../outputs/footwear_contact_material_report/figures/validation_fullfoot.svg)

**Figure 6.** Stored adjacent-cycle full-foot data and two-term model prediction. The same six material parameters are used for both fixtures.

The official metrics use baseline-corrected force. An active sample has measured force at least 5% of the robust measured peak. The robust peak is the mean of the five highest active samples; it is not the single largest displayed sample.

```text
Peak error = |robust_peak_pred − robust_peak_meas| / robust_peak_meas
Force error = RMS(F_pred − F_meas over measured-active samples) / robust_peak_meas
Loop error = |W_pred − W_meas| / |W_meas|
```

Hysteresis work splits active loading and unloading at the measured displacement peak and subtracts recovered work from loading work. Each relative error must be **strictly below 10%**. The table below reports the selected artifact's stored metrics. A fresh recomputation from its saved curves also passes all six gates; relative errors differ from stored scores by at most `2.23×10⁻⁷`. Both versions are retained in `stored_curve_metric_check.json`. The small discrepancy is reported rather than silently treating the records as bitwise identical. [S9]

| Fixture | Peak error | Active-force RMSE | Hysteresis error | Stored gates |
|---|---:|---:|---:|---|
| Rearfoot punch | 2.769% | 5.633% | 7.518% | 3/3 pass |
| Full-foot last | 5.853% | 6.433% | 8.831% | 3/3 pass |

| Fixture | Measured robust peak | Predicted robust peak | Measured loop work | Predicted loop work |
|---|---:|---:|---:|---:|
| Rearfoot punch | 1174.54 N | 1142.01 N | 1.12919 J | 1.04430 J |
| Full-foot last | 1977.48 N | 2093.23 N | 1.54734 J | 1.68399 J |

**Supported conclusion:** the shared two-term foundation reproduces nearby held-out cycles from these bench runs within the declared three-metric thresholds.

**Unsupported extension:** this does not validate another shoe, a different temperature, arbitrary rates, large impacts, gait loading, sliding friction, local pressure maps, or rigid-last seating. The held-out cycles are adjacent within the same approximately 0.5 s runs, not independent tests spanning these conditions. These cycles have also been inspected during model development; treat the split as internal/local repeatability evidence, not a blind final model-selection test.

For high-strain context, the saved maximum fixture displacements are 22.760 mm rearfoot and 17.926 mm full-foot. Using the exported driven-column clearances and thicknesses gives maximum raw driven strains of about **89.6% and 93.8%**, respectively. These are geometry-derived model strains, not measured local foam strains. They should not be replaced by the lower strain of a typical column, and the passive 90% cap does not apply to every driven column.

The illustrated material-rate sweeps in Section 4 are predictions from the fitted model. They must not be counted as additional held-out evidence.

## 14. Fresh implementation checks for this report

| Fresh check | Result | Evidence |
|---|---|---|
| Strict artifact load and recomputed saved-curve metrics | **Passed; all six gates pass** | `stored_curve_metric_check.json` |
| Artifact input-file hashes | **All seven recorded inputs match local bytes** | `artifact_input_hash_check.json` |
| Shared material, contact, and consumer tests | **15 tests passed** | `shared_law_tests.log` |
| Digital Instron core tests, separate process | **10 tests passed** | `core_tests.log` |
| Full-foot two-term live bench replay, 180 frames | **Passed**; runtime-to-artifact waveform NRMSE **0.019%** | `fullfoot_replay.log` |
| Rearfoot two-term live bench replay, 180 frames, separate retry | **Passed**; runtime-to-artifact waveform NRMSE **0.211%** | `rearfoot_replay_retry.log` |

The replay test compares the warmed live force waveform with the interpolated artifact prediction at the force-evaluation time. Its threshold is 3% waveform NRMSE. This is **adapter consistency**, not another comparison with new measured data. Printed maximum instantaneous forces and the artifact's top-five robust peaks are not identical statistics.

The first rearfoot run failed with **Warp CUDA error 700: illegal memory access** during a device-to-host diagnostic copy. The later separate-process retry passed. The cause is unresolved; this report does not attribute it to a particular kernel or claim it is fixed. The failed log is retained as `rearfoot_replay.log`. A concurrently running broader core/dynamics/differentiable test invocation was stopped before completion and is **not counted as a passed full suite** (`extended_physics_tests.log`). No physics source was changed to obtain the retry.

These are the report-preparation checks at the source snapshot stated above, not tests rerun every time the report is rendered. The local log files are retained with the generated report package and are not bundled with a source-only checkout. These checks ran in the existing Newton environment with Warp `1.17.0.dev20260807` and NVIDIA RTX A6000. This report does not claim complete cross-hardware regression coverage or a fresh material refit.

A unit test, saved experimental gate, and live replay check answer different questions. Tests check code behavior; stored gates compare the fitted model with measurements; replay checks compare the live adapter with the selected artifact's prediction. Passing one category does not substitute for the others.

## 15. Recommended validation and development priorities

| Priority | Next test or improvement | Why it matters |
|---|---|---|
| 1 | Resolve and measure upper-last seating/registration | Avoid attributing a boundary-geometry error to foam properties |
| 2 | Multi-rate compression and long relaxation holds | Separate equilibrium fraction from relaxation time and test rate extrapolation |
| 3 | Independent shear/friction tests over normal load and slip rate | Identify μ, tangent stiffness, and damping rather than assuming them |
| 4 | Time-step, passive-sweep, lag, and grid refinement | Quantify numerical sensitivity separately from material uncertainty |
| 5 | Spatial pressure measurements under both fixtures | Test load spreading and geometry, not only resultant force |
| 6 | Held-out independent sessions, shoes, and temperatures | Establish generalization and specimen variability |
| 7 | Full work audit including memory, bristles, coupling, clipping, and actuators | Separate recoverable energy from physical and numerical losses |

The existing acquisition protocol provides a starting point for rate, relaxation, metadata, and specimen tracking. A synthetic modulus or time-constant perturbation is useful for sensitivity analysis, but it is not a newly measured or validated material. [S16]

## 16. Reproduce and extend this report

Run from the repository root with its existing environment. The plotting command uses the already available optional `matplotlib` stack; HTML rendering uses `markdown-it-py`. No new required dependency is introduced. The artifact must be present locally. None of these commands refits or overwrites it.

```bash
# Regenerate mechanism figures from the explicit two-term artifact.
uv run --no-sync -m projects.digital_shoe.mechanics_report.figures \
  --artifact outputs/impedance_instron/inputs/digital_shoe.json

# Rebuild the offline HTML from the editable Markdown and local figures.
uv run --no-sync -m projects.digital_shoe.mechanics_report

# View the two-term full-foot bench model.
uv run --no-sync -m projects.digital_shoe view \
  --artifact outputs/impedance_instron/inputs/digital_shoe.json \
  --mode instron --fixture fullfoot_last --viewer gl

# Headless full-foot replay of the same model.
uv run --no-sync -m projects.digital_shoe view \
  --artifact outputs/impedance_instron/inputs/digital_shoe.json \
  --mode instron --fixture fullfoot_last \
  --viewer null --num-frames 180 --test

# Rearfoot replay; see the current run status in Section 14.
uv run --no-sync -m projects.digital_shoe view \
  --artifact outputs/impedance_instron/inputs/digital_shoe.json \
  --mode instron --fixture rearfoot_punch \
  --viewer null --num-frames 180 --test

# Shared material and contact regression checks.
uv run --no-sync -m unittest \
  newton.tests.test_digital_shoe_material \
  newton.tests.test_digital_shoe_shared_contact \
  newton.tests.test_digital_shoe_consumers
```

The HTML is self-contained, including its figures, and requires no server or CDN. Use browser Print → Save as PDF for a print copy. `projects/digital_shoe/mechanics_report/REPORT.md` is the versioned editable narrative. PNGs can be inserted into slides; SVGs preserve vector quality. `projects/digital_shoe/mechanics_report/sources.json` records the exact source-file hashes and artifact identity, while `figures/metadata.json` records the figure inputs and calculation settings.

**Data rights:** this report includes derivatives of local supplied footwear measurements and geometry. It is for internal research use. The repository does not record upstream redistribution approval for those assets. A hash identifies bytes; it does not grant sharing rights. See `ASSET_PROVENANCE.md` before external publication.

## Appendix A. Symbols and dimensions

| Symbol | Meaning | Unit |
|---|---|---|
| t_i, c_i, λ_min | Rest thickness; compression; minimum stretch | m; m; dimensionless |
| ε_i, λ_i | Engineering compressive strain; thickness stretch | dimensionless |
| A_i | Tributary area | m² |
| G_inst,n, G_eq,n | Instantaneous and equilibrium term moduli | Pa |
| α_n, f_eq | Term exponent; equilibrium fraction | dimensionless |
| p_eq, q | Equilibrium pressure; Maxwell overstress | Pa |
| τ, Δt | Relaxation time; substep duration | s |
| K_ij | Neighbor transfer coefficient | N/m |
| Φ_i, T_i, R_i | Neighbor transfer; signed top transfer; unilateral reaction | N |
| k_t, k_v, d_n | Bristle stiffness; tangent damping; normal damping | N/m; N·s/m; N·s/m |
| μ, γ, η | Friction coefficient; viscous cap ratio; retained viscous fraction | dimensionless |
| F, M, P | Resultant force; moment; rigid-body power | N; N·m; W |

## Appendix B. Source and reference ledger

The references below were checked against the local source snapshot named at the start of this report. Paths are links relative to this checkout. Line ranges are audit locations, not a promise of stability in another revision. Referencing solver internals here does not require importing them from an example.

| ID | Claim / implementation | Source and audit lines |
|---|---|---|
| S1 | Two-term pressure and Maxwell recurrence | [`projects/digital_shoe/material.py`](../../../projects/digital_shoe/material.py), 24–143 |
| S2 | Unilateral contact, Pasternak transfer, bristles, passive balance and wrench | [`projects/digital_shoe/contact.py`](../../../projects/digital_shoe/contact.py), 21–234 |
| S3 | Material records, state update, force reductions and configuration | [`projects/digital_shoe/runtime.py`](../../../projects/digital_shoe/runtime.py), 35–92; 204–322; 1022–1266; 1309–1429; 1726–1948 |
| S4 | Portable artifact and foundation/fixture loading | [`projects/digital_shoe/artifact.py`](../../../projects/digital_shoe/artifact.py), whole module |
| S5 | Artifact-only bench and carried-shoe demonstrations | [`projects/digital_shoe/showcase.py`](../../../projects/digital_shoe/showcase.py), whole module; bench configuration and test_final |
| S6 | Fitting problem, material vector, residuals, bounds and optimizer | [`projects/digital_instron_v2/core.py`](../../../projects/digital_instron_v2/core.py), 73–155; 429–579 |
| S7 | Grid geometry and fixture construction | [`projects/digital_instron_v2/geometry.py`](../../../projects/digital_instron_v2/geometry.py), whole module |
| S8 | Training and held-out cycle preparation | [`projects/digital_instron_v2/phase1.py`](../../../projects/digital_instron_v2/phase1.py), whole module |
| S9 | Official held-out metrics | [`projects/digital_instron_v2/validation.py`](../../../projects/digital_instron_v2/validation.py), 47–163 |
| S10 | Two-term artifact export and metadata | [`projects/digital_instron_v2/export_digital_shoe.py`](../../../projects/digital_instron_v2/export_digital_shoe.py), whole module |
| S11 | GPU-resident periodic forward fitting workspace | [`projects/digital_shoe/calibration.py`](../../../projects/digital_shoe/calibration.py), whole module |
| S12 | Active differentiable foundation adapter | [`projects/digital_instron_v2/dynamics_diff.py`](../../../projects/digital_instron_v2/dynamics_diff.py), whole module; active bristle call and history storage |
| S13 | Current impedance rig settings, backing geometry and integration | [`projects/impedance_instron/simple/rig.py`](../../../projects/impedance_instron/simple/rig.py), 40–92; 743–749; 851–950; 1074–1109 |
| S14 | Rendering geometry versus pressure and bristle references | [`projects/digital_shoe/rendering.py`](../../../projects/digital_shoe/rendering.py), whole module |
| S15 | Semi-implicit solver step and rigid-body integration calls | [`newton/_src/solvers/semi_implicit/solver_semi_implicit.py`](../../../newton/_src/solvers/semi_implicit/solver_semi_implicit.py), 124–217 |
| S16 | Proposed acquisition and specimen metadata protocol | [`projects/digital_shoe/ACQUISITION_PROTOCOL.md`](../../../projects/digital_shoe/ACQUISITION_PROTOCOL.md), whole document |


**External primary-source access record**

- **E1.** McCulloch, Delp, and Kuhl, arXiv:2602.12694v1. [Primary full text](https://arxiv.org/html/2602.12694v1). Used only for constitutive context and the related-foam Poisson-ratio assumption. Their compression tables extend to stretch 0.4, or 60% compression, and their compression specimens are cylindrical. These are not measurements on the intact shoe fitted here.
- **E2. Attempted but unavailable.** Abaqus 2024, [Hyperelastic behavior of elastomeric foams](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMATRefMap/simamat-c-hyperfoam.htm). Access returned HTTP 403 Forbidden. No claim in this report relies on unseen manual content. The equations and initial moduli were verified against the Newton project source, shared-law calculations, and tests; this report does not claim equivalence to a full Abaqus finite-element analysis.

The accessible arXiv text and the failed manual-access response are retained in the local research folder for audit. Numerical values and figures in this report come only from the selected two-term artifact and shared Newton project functions, not from another shoe's published material table.
