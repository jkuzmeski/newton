# Cartesian controller implementation

Use the [twelve-point baseline pipeline](../README.md) from the repository root:

```bash
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12
```

This directory contains only the current model and its CPU qualification and
replay support. The CPU rollout is a numerical reference, not an optimizer.
The measured loss and frozen refinement criteria live in `fit.py`.
`shoe.py` attaches the shared Digital Shoe without duplicating its contact law.
`spline.py` owns only cubic basis and derivative-control-polygon algebra.

Rebuild a saved result's HTML report without refitting:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian report outputs/impedance_instron/run12/fit
```

Spring export replays contact at saved simulated states. It verifies force,
moment, compression and cap histories. It does not integrate the leg or use
recorded target motion as simulated motion. Use `--mesh-only` only to explicitly
skip this verification and disable the spring views.
