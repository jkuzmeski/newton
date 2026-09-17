<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Digital Shoe friction identification and dynamic qualification

> Historical study tools use the archived initial controller manifest by default.
> The fork's current accepted controller baseline is preserved. Use
> `NEWTON_BASELINE12_MANIFEST` to select a different pinned manifest and supply
> the matching baseline directory. Local measured traces and fitted coefficient
> artifacts are not distributed by this change.

> **Current priority:** Raw physical-force continuity, not peak fitting. `friction-report`
> and `friction-fit` now default to `raw`. Filtered `matched` output is only an
> optional observation diagnostic. See [FRICTION_CONTINUITY.md](FRICTION_CONTINUITY.md).

> **Observation-policy update:** The archived 22,808-evaluation study below used
> unmatched force comparisons. New `friction-fit` and `friction-report` runs default
> to raw physical-force metrics. Signed, bandwidth-matched observations remain an
> explicitly selected secondary diagnostic. See [FRICTION_ONSET.md](FRICTION_ONSET.md).
> Historical metrics and gates remain available and are not rewritten. Do not
> compare losses across policies as if they were the same objective.

This document details the parameter search methodology, schemas, scoring
protocols, and qualification findings for tangential shoe contact.

**Core qualification outcome:**
- 22,808 candidate parameter evaluations were attempted across fixed-history sweeps
  and free-leg dynamic searches.
- **No fitted parameter set passed all six limb dynamics criteria.**
- **No calibrated parameters are promoted or installed as defaults.**
- MidsoleFoundation defaults to legacy bristle mechanics unless an adapter is explicitly attached.

## Scope and physical invariants

This investigation is strictly confined to tangential friction mechanics:

$$\mathbf{f}_{t, i} = f(\mathbf{v}_{t, i}, F_{n, i}, \dots), \quad \|\mathbf{f}_{t, i}\| \le \mu F_{n, i}$$

The following subsystems are frozen. Normal-law AST and bitwise regression checks cover
contact mechanics; source and sealed-input hashes record the controller and material scope:
- **Normal contact & compression:** Unilateral ground reaction, depth kinematics, and force limits.
- **Midsole constitutive model:** Two-term Ogden-Hill hyperfoam with Maxwell relaxation series (`material.py`).
- **Midsole foundation:** Pasternak inter-column shear coupling, passive surround balance, and anchor layout.
- **Cartesian leg model & controller:** 3-link articulated leg kinematics, mass/inertia properties,
  Cartesian hip spring/damper gains, joint impedance gains, and frozen equilibrium spline trajectory.

Under prescribed identical kinematic inputs, normal ground force $F_z$ matches the baseline
bit-for-bit across candidate worlds (`test_candidate_normal_force_unmodified`). In free-leg
dynamics, normal force histories can vary solely because tangential friction alters limb motion.

## Multi-tool CLI workflow

The identification and qualification suite consists of eight interoperable commands
registered under `projects.digital_shoe`:

```text
projects.digital_shoe
├── friction           (friction_example.py)         Prescribed-load synthetic edge-case verification
├── friction-leg       (friction_leg.py)             Frozen baseline leg kinematics replay
├── friction-metrics   (friction_metrics.py)         Pure NumPy GRF impulse, peak, and RMSE scoring
├── friction-cache     (friction_history.py)         Record frozen normal/kinematic history to NPZ
├── friction-sweep     (friction_sweep.py)           High-throughput GPU parameter sweep on frozen history
├── friction-fit       (friction_dynamic_search.py)  Free-leg bounded elite-centered random search
├── friction-check     (friction_dynamic.py)         Independent single-candidate qualification on CPU
└── friction-report    (friction_comparison.py)      Offline multi-run comparative HTML report generator
```

### Step-by-step reproducible workflow

These are runnable workflow examples, not the exact settings of every archived search.
Replace the baseline path with the sealed input directory and choose unused outputs;
all tools reject overwrites. Raw data and generated derivatives stay local.

#### 1. Build immutable normal/kinematic cache

Execute a baseline forward simulation replay to capture preintegration plane kinematics,
normal loads, and baseline friction forces into a GPU graph-captured archive:

```bash
uv run --no-sync -m projects.digital_shoe friction-cache   --baseline outputs/impedance_instron/baseline12   --output outputs/friction_identification/history_exact.npz   --device cuda:0 --chunk-steps 32
```

#### 2. High-throughput frozen-history parameter sweep

Evaluate candidate parameter sets on GPU against the immutable normal history using
resident `device_while` graph capture (over 5,000 candidates in minutes):

```bash
uv run --no-sync -m projects.digital_shoe friction-sweep   --cache outputs/friction_identification/history_exact.npz   --output outputs/friction_identification/study_seed17   --candidates-per-method 1024 --batch-size 128 --generations 6 --seed 17   --methods legacy deflection --device cuda:0
```

#### 3. Free-leg dynamic parameter search (GPU-batched)

Perform evolutionary search in forward-simulated single leg dynamics under frozen
Cartesian impedance control:

```bash
# Search using consistent deflection (method 1, 7 parameters)
uv run --no-sync -m projects.digital_shoe friction-fit   --baseline outputs/impedance_instron/baseline12   --output outputs/friction_identification/dynamic_constant_seed29   --method 1 --candidates 1024 --worlds 128 --generations 8 --seed 29 --device cuda:0

# Search using dynamic Stribeck cap (method 4, 9 parameters)
uv run --no-sync -m projects.digital_shoe friction-fit   --baseline outputs/impedance_instron/baseline12   --output outputs/friction_identification/dynamic_stribeck_raw_seed89   --method 4 --candidates 1024 --worlds 128 --generations 8 --seed 89 --device cuda:0
```

#### 4. Independent single-candidate verification (CPU)

Independently qualify a candidate JSON file using the original NumPy leg dynamics and Warp CPU contact path to eliminate GPU-specific optimizations or batching artifacts:

```bash
uv run --no-sync -m projects.digital_shoe friction-check   --baseline outputs/impedance_instron/baseline12   --candidate outputs/friction_identification/dynamic_stribeck_raw_seed89/candidate.json   --output outputs/friction_identification/raw_fit_cpu_check   --device cpu
```

#### 5. Assemble offline comparative report

Assemble an offline comparison across multiple completed runs against the measured reference:

```bash
uv run --no-sync -m projects.digital_shoe friction-report   --baseline outputs/impedance_instron/baseline12    --run best_rawfit=outputs/friction_identification/raw_fit_cpu_check   --output outputs/friction_identification/comparison_final
```

## Parameter schemas and mode namespaces

Parameter arrays and serialized JSON schemas use strict, validated namespaces.
Method integer codes differ between the low-level solver and parameter-search adapters:

### Method namespace mapping

| Solver mode name (`FrictionSolver`) | Solver int | Adapter method code (`FrictionParameterAdapter`) | Sweep method (`FrictionSweep`) | Dynamic qualification (`friction_dynamic`) |
|---|---|---|---|---|
| `bristle` | 0 | 0 (`legacy`) | 0 (`legacy`) | 0 (`legacy`) |
| `implicit_bristle` | 1 | N/A (not used by this dynamic adapter) | N/A | N/A (not used by this dynamic adapter) |
| `regularized` | 2 | N/A | 2 (`regularized`, diagnostic) | Rejected (diagnostic) |
| `deflection` | 3 | 1 (`deflection`) | 1 (`deflection`) | 1 (`deflection`) |
| `implicit_deflection` | 4 | N/A (not used by this dynamic adapter) | N/A | N/A (not used by this dynamic adapter) |
| *Stribeck deflection* | N/A | 4 (`stribeck`) | N/A | 4 (`stribeck`) |
| *Anchor nominal* | N/A | N/A | 3 (`anchor_nominal`, diagnostic) | Rejected (diagnostic) |

### Seven-parameter row schema (standard models)

Used for `legacy` (0) and `deflection` (1):

```python
PARAMETER_NAMES = (
    "method",           # int: 0 for legacy, 1 for consistent deflection
    "mu",               # float: effective Coulomb friction coefficient [-]
    "kt_scale",         # float: multiplier on baseline tangential stiffness [-] (> 0)
    "kv_scale",         # float: multiplier on baseline tangential damping [-] (>= 0)
    "viscous_ratio",    # float: damping force cap as fraction of Coulomb limit [-] (>= 0)
    "release_dwell_s",  # float: dwell duration before releasing stuck contact [s] (>= 0)
    "yield_width",      # float: yield shoulder fraction; MUST BE EXACTLY 0.0
)
```

### Nine-parameter row schema (dynamic Stribeck model)

Used for `stribeck` (method 4):

```python
PARAMETER_NAMES = (
    "method",            # int: 4
    "mu",                # float: static Coulomb friction coefficient mu_static [-]
    "kt_scale",          # float: stiffness multiplier [-] (> 0)
    "kv_scale",          # float: damping multiplier [-] (>= 0)
    "viscous_ratio",     # float: viscous limit fraction [-] (>= 0)
    "release_dwell_s",   # float: release dwell duration [s] (>= 0)
    "yield_width",       # float: yield shoulder fraction; MUST BE EXACTLY 0.0
    "mu_dynamic",        # float: dynamic friction coefficient mu_dynamic (0 <= mu_d <= mu_s)
    "transition_speed",  # float: velocity scale for exponential decay [m/s] (> 0)
)
```

## Full-rate vs native-rate scoring

The experimental reference ground reaction force is recorded at 2,000 Hz (721 native samples
spanning $t \in [0, 0.36]$ s, with the 721st sample outside preintegration support and excluded).
The simulation steps at $\Delta t = 62.5\,\mu\text{s}$ (16,000 Hz, 5,760 steps).

Two scoring protocols are implemented:

1. **Native-rate scoring (`friction_metrics.py`, `friction_scores`):**
   Downsamples simulated force curves by linear interpolation to the 720 covered native
   reference sample times. Stance intervals are defined by measured upward force $F_z \ge 50$ N.
   Impulses are integrated using piecewise-linear zero-crossing trapezoids.

2. **Full-rate scoring (`fit_scores` in `FrictionDynamicGPUWorkspace`):**
   Interpolates the measured reference force up to the simulation clock grid ($T = 5,760$).
   Computes RMSE, impulse, and peak metrics across all simulation substeps.

### Detection of sub-sample chatter and aliasing

Evaluating fits solely at native sample times allowed the optimizer to discover solutions
that scored low RMSE at measurement points by exhibiting high-frequency chatter between samples:

- **Rejected candidate (seed 43):**
  - Native-rate RMSE: 92.96 N.
  - Full-rate simulation RMSE: 103.09 N.
  - Timestep sensitivity check (halving $\Delta t$ to $31.25\,\mu\text{s}$): maximum $F_x$
    difference of **91.94 N** (severe numerical instability).
  - This candidate was rejected and excluded from promotion.

- **Stable candidate (seed 89):**
  - Evaluated with full-rate `fit_scores` during optimization.
  - Native-rate RMSE: 98.57 N; Full-rate RMSE: 98.57 N.
  - Timestep sensitivity check (halving $\Delta t$): maximum $F_x$ difference of **1.698 N**,
    maximum $F_z$ difference of **4.177 N**, maximum state displacement of $8.1 \times 10^{-5}$ m.
  - Passes the checked force-difference limit for this step halving; this is not proof
    of general convergence or experimental validity.

## Evaluation summary across 22,808 candidates

| Study / Sweep batch | Method | Attempted | Completed | Selected best loss / candidate status |
|---|---|---:|---:|---|
| `study_seed17` (frozen history) | Legacy, Deflection, Nominal | 5,379 | 5,379 | Completed; non-conjugate work identified |
| `study_seed41` (frozen history) | Legacy, Deflection | 6,146 | 6,146 | Completed; verified replay accuracy |
| `dynamic_constant_seed29` (free leg) | Deflection (7-param) | 2,050 | 1,572 | Failed rollouts marked NaN/inf |
| `dynamic_stribeck_seed43` (free leg) | Stribeck (native scoring) | 3,333 | 1,945 | Rejected: sub-sample aliasing (91.9 N halfstep jump) |
| `dynamic_stribeck_seed67` (free leg) | Stribeck (speed-bounded) | 2,566 | 1,874 | Completed; exploratory search |
| `dynamic_stribeck_raw_seed89` (free leg) | Stribeck (full-rate scoring) | 3,334 | 1,963 | **Selected diagnostic fit** (collapsed to constant) |
| **Total** | | **22,808** | **18,879** | **0 candidates passed all 6 gates** |

### Selected diagnostic parameters (`dynamic_stribeck_raw_seed89`)

```json
{
  "method": 1,
  "mu": 0.53604674,
  "kt_scale": 0.05679486,
  "kv_scale": 0.29514942,
  "viscous_ratio": 0.15907651,
  "release_dwell_s": 0.00225869,
  "yield_width": 0.0
}
```

*Note: In seed 89, $\mu_{\text{dynamic}}$ converged exactly to $\mu_{\text{static}} = 0.5360$,
so the candidate simplifies identically to constant-coefficient consistent deflection (`method=1`).*

### Comparison against baseline and experiment

Peaks, impulses and curve RMSE below use the covered native reference clock.
This differs from the full-rate fitting score.

| Quantity | Experimental measured | Original baseline (legacy bristle) | Repaired law (original params) | Selected diagnostic fit |
|---|---:|---:|---:|---:|
| Braking peak force [N] | 181.59 | 396.41 | 408.80 | **345.97** |
| Propulsive peak force [N] | 187.28 | 207.06 | 211.36 | **198.05** |
| Braking impulse [N·s] | 14.694 | 22.253 | 21.999 | 23.303 |
| Propulsive impulse [N·s] | 16.692 | 16.801 | 17.564 | 15.623 |
| Full horizontal RMSE [N] | — | 97.76 | 100.56 | 98.57 |
| Stance horizontal RMSE [N] | — | 107.63 | 110.72 | 108.46 |

### Six-channel dynamic qualification criteria

The original six-channel RMSE thresholds test experimental tracking, not just
degradation relative to baseline. The original baseline also fails some thresholds:

| Channel | Quantity | Gate threshold | Baseline value | Selected candidate value | Gate status |
|---|---|---:|---:|---:|---|
| 1 | Hip horizontal position $X$ | $< 20$ mm | 8.99 mm | 8.72 mm | **PASS** |
| 2 | Hip vertical position $Z$ | $< 20$ mm | 20.63 mm | 20.36 mm | **FAIL** (exceeds 20 mm) |
| 3 | Knee joint angle | $< 50$ mrad | 20.54 mrad | 24.20 mrad | **PASS** |
| 4 | Ankle joint angle | $< 50$ mrad | 47.36 mrad | 52.70 mrad | **FAIL** (exceeds 50 mrad) |
| 5 | Horizontal ground force $F_x$ | $< 100$ N | 97.83 N | 98.63 N | **PASS** |
| 6 | Vertical ground force $F_z$ | $< 100$ N | 104.04 N | 102.56 N | **FAIL** (exceeds 100 N) |

Because three of the six channels fail the thresholds, this parameter set cannot be
certified or promoted.

## Honest limitations and negative results

1. **Deflection guard boundary:** The 20 mm deflection screen is an engineering sanity
   bound, not a measured sole-thickness or material limit. The selected
   fit reaches 19.98 mm at source step and 20.005 mm under step halving, pressing the screen limit.
2. **Impulse vs peak trade-off in tested searches:** While the selected candidate reduces
   the excessive braking peak force from 396.4 N to 345.9 N, its braking impulse increases
   (23.30 N·s vs measured 14.69 N·s and baseline 22.25 N·s). Across the 22,808 candidate
   evaluations in the tested bounded parameter spaces, tangential tuning did not simultaneously
   reconcile peak force and impulse.
3. **Controller/kinematics consistency diagnostic:** Momentum diagnostics (`check_horizontal_balance.py`)
   confirm the simulated leg achieves discrete horizontal momentum balance to within 0.0045 N·s.
   A separate calculation using perfect reference kinematics and the frozen controller requests
   a different net ground impulse from the measured 1.586 N·s. This is an exploratory consistency
   diagnostic sensitive to endpoint velocities, kinematic interpolation, and permissible tracking
   errors; it is not proof that every friction-only formulation must fail, and controller/normal
   retuning remains outside the investigation's frozen scope.
4. **No parameter promotion:** The Digital Shoe runtime, examples, and downstream consumers
   remain on their existing defaults. The consistent-deflection law is fully tested and
   accessible via `FrictionAdapter(mode='deflection')` or `mode='implicit_deflection'`, but
   no universal default has been modified.
