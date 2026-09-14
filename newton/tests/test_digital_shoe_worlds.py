# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the world-batched Digital Shoe midsole foundation.

:class:`projects.digital_shoe.runtime.MidsoleFoundation` evaluates ``world_count``
independent copies of the same column bed in one launch. These tests pin the two
properties that make that useful: a single world reproduces the pre-batching
runtime bit for bit, and the worlds never touch each other.
"""

import hashlib
import unittest

import numpy as np
import warp as wp

import newton
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, ShoeMaterial, SurroundConfig

MATERIAL = ShoeMaterial(
    instantaneous_shear_modulus_pa=19000.0,
    hyperfoam_exponent=5.1,
    equilibrium_fraction=0.11,
    pasternak_n_per_m=900.0,
    effective_poisson_ratio=0.0,
    maxwell_relaxation_time_s=0.08,
    instantaneous_shear_modulus_2_pa=4000.0,
    hyperfoam_exponent_2=-2.0,
)
# The fitted shoe the RL work randomizes around. ``pasternak_n_per_m`` is carried
# for the artifact schema only: the runtime derives each column's Pasternak
# coefficient from the equilibrium Ogden-Hill modulus and the column thickness, so
# randomizing that field alone cannot change a single result.
FITTED_MATERIAL = ShoeMaterial(
    instantaneous_shear_modulus_pa=74671.42,
    hyperfoam_exponent=0.21550,
    equilibrium_fraction=0.69542,
    pasternak_n_per_m=1546.75,
    effective_poisson_ratio=0.0,
    maxwell_relaxation_time_s=0.00500015,
)
NX, NY = 6, 5
COLUMN_COUNT = NX * NY
SPACING_M = 0.005
STEPS = 16
DT_S = 5.0e-4

# Tiled per-column state: world ``w`` owns ``[w * COLUMN_COUNT : (w + 1) * COLUMN_COUNT]``.
COLUMN_FIELDS = (
    "compression",
    "base_pressure",
    "column_force",
    "z_free",
    "tangent_anchor",
    "tangent_stuck",
    "tangent_dwell",
    "q_state",
    "peq_prev",
    "surround_compression",
    "surround_rate",
)
# Per-world reductions recorded once per substep, shape ``[STEPS, world_count]``.
HISTORY_FIELDS = ("normal_force", "pressed_force", "contact_power", "max_compression", "active")
# Per-world reductions read once at the end, shape ``[world_count, ...]``.
FINAL_WORLD_FIELDS = ("cop_moment", "resultant_force", "resultant_moment_origin", "body_f")
# Reductions that stay exact on every device: an integer count and an atomic max,
# neither of which depends on the order the columns are accumulated in.
EXACT_REDUCTIONS = ("active", "max_compression")
# Margin above the measured CUDA float-atomic ordering noise. See
# :meth:`TestBatchedMidsoleFoundation.assert_reduction_close`.
REDUCTION_RTOL = 1.0e-5

# Digests of every PER-COLUMN array the scenario below produces, plus the two order
# independent reductions, captured on the CPU from the single-carrier runtime that
# predates world batching. They are exact float32 bytes, so any change of the
# single-world arithmetic breaks them; they are not portable across Warp or LLVM
# releases, which is why only the CPU device is pinned. The summed reductions live in
# :data:`PRE_BATCHING_REDUCTIONS` because their summation order legitimately changed.
PRE_BATCHING_DIGESTS = {
    "active": "b03b2ae287be8886341fa43f38e35274",
    "base_pressure": "93dbc9800bbd810b3d85fb9b617d3c65",
    "column_force": "d96e7e797c7da2e448a5d78c7e2820b1",
    "compression": "dfd66c04193cbef160a81185991453ed",
    "max_compression": "57080465daa5488a0fcf40498649746c",
    "peq_prev": "5b0df2ab28a3667cef807407473d328b",
    "q_state": "8849ccb5f8da491b131e882491a4b432",
    "surround_compression": "a01c4781d4348dda882dd71c285d11af",
    "surround_rate": "8a8b629bdd3d12521f69d4c793eab935",
    "tangent_anchor": "d71c63f3300ea4a7806e0f25038be4ec",
    "tangent_dwell": "d76575dff90cc473ecbb66d815626f66",
    "tangent_stuck": "1a57f94c38227bb86154c78d90e5b219",
    "z_free": "6377089e9aa69b416553555006affa39",
}


PRE_BATCHING_REDUCTIONS = {
    "normal_force": (
        0.0,
        1.8461979627609253,
        4.733758926391602,
        7.084722995758057,
        9.412877082824707,
        11.709059715270996,
        13.964193344116211,
        16.16934585571289,
        18.31578254699707,
        20.39500617980957,
        22.398767471313477,
        24.319129943847656,
        26.14849090576172,
        27.879592895507812,
        29.50558090209961,
        31.020008087158203,
    ),
    "pressed_force": (
        0.0,
        1.8467590808868408,
        4.733758926391602,
        7.084722995758057,
        9.412877082824707,
        11.709059715270996,
        13.964193344116211,
        16.16934585571289,
        18.31578254699707,
        20.39500617980957,
        22.398767471313477,
        24.319129943847656,
        26.14849090576172,
        27.879592895507812,
        29.50558090209961,
        31.020008087158203,
    ),
    "contact_power": (
        0.0,
        -0.145333930850029,
        -0.4295562207698822,
        -0.7441718578338623,
        -1.1249979734420776,
        -1.5669269561767578,
        -2.0638582706451416,
        -2.6087894439697266,
        -3.193925619125366,
        -3.810790777206421,
        -4.450364589691162,
        -5.103215217590332,
        -5.759650230407715,
        -6.409860134124756,
        -7.044078350067139,
        -7.65272855758667,
    ),
    "cop_moment": (0.03148314356803894, -6.427057087421417e-06, 0.0),
    "resultant_force": (-18.61200714111328, 0.0, 31.020008087158203),
    "resultant_moment_origin": (-6.427057087421417e-06, -0.016082853078842163, -3.8584694266319275e-06),
    "body_f": (
        -18.61200714111328,
        0.0,
        31.020008087158203,
        -6.427057087421417e-06,
        -0.011281265877187252,
        -3.8584694266319275e-06,
    ),
}


def _randomized_material(world: int) -> ShoeMaterial:
    """Return a deterministic domain-randomization draw around :data:`FITTED_MATERIAL`.

    Every randomized field moves, and the draws are far apart, so a world that
    silently read another world's material would be obvious rather than marginal.
    """
    scale = 1.0 + 0.35 * world
    fraction = float(np.clip(FITTED_MATERIAL.equilibrium_fraction * (1.0 - 0.12 * world), 0.05, 0.95))
    return ShoeMaterial(
        instantaneous_shear_modulus_pa=FITTED_MATERIAL.instantaneous_shear_modulus_pa * scale,
        hyperfoam_exponent=FITTED_MATERIAL.hyperfoam_exponent * (1.0 + 0.5 * world),
        equilibrium_fraction=fraction,
        pasternak_n_per_m=FITTED_MATERIAL.pasternak_n_per_m * scale,
        effective_poisson_ratio=0.0,
        maxwell_relaxation_time_s=FITTED_MATERIAL.maxwell_relaxation_time_s * (1.0 + 2.0 * world),
    )


def _available_devices() -> list:
    """Return every Warp device the batched foundation should be checked on."""
    devices = [wp.get_device("cpu")]
    if wp.is_cuda_available():
        devices.append(wp.get_device("cuda:0"))
    return devices


def _digest(values: np.ndarray) -> str:
    """Return a shortened SHA-256 of an array's exact bytes, dtype, and shape."""
    array = np.ascontiguousarray(values)
    return hashlib.sha256(f"{array.dtype.str}{array.shape}".encode() + array.tobytes()).hexdigest()[:32]


def _build_bed() -> dict:
    """Return a small synthetic column bed with a driven core and a free perimeter."""
    xs = (np.arange(NX) - 0.5 * (NX - 1)) * SPACING_M
    ys = (np.arange(NY) - 0.5 * (NY - 1)) * SPACING_M
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    index = np.arange(COLUMN_COUNT).reshape(NX, NY)
    rest = 0.02 + 0.002 * np.cos(3.0 * gx / SPACING_M) * np.sin(2.0 * gy / SPACING_M)
    neighbors = np.full((COLUMN_COUNT, 4), -1, np.int32)
    for i in range(NX):
        for j in range(NY):
            k = index[i, j]
            if i > 0:
                neighbors[k, 0] = index[i - 1, j]
            if i < NX - 1:
                neighbors[k, 1] = index[i + 1, j]
            if j > 0:
                neighbors[k, 2] = index[i, j - 1]
            if j < NY - 1:
                neighbors[k, 3] = index[i, j + 1]
    return {
        "anchor": np.column_stack([gx.ravel(), gy.ravel(), np.zeros(COLUMN_COUNT)]),
        "z_free": np.zeros(COLUMN_COUNT),
        "rest": rest.ravel(),
        "area": np.full(COLUMN_COUNT, SPACING_M * SPACING_M),
        "neighbors": neighbors,
        "driven": (np.abs(gx.ravel()) <= SPACING_M * 1.01) & (np.abs(gy.ravel()) <= SPACING_M * 1.01),
    }


def _carrier_motion(world: int, step: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the scripted pose and spatial velocity of one world's carrier body.

    A pressing oscillation, a lateral drift that loads the friction bristles, and a
    small pitch, so the wrench, the centre of pressure and the point velocities all
    carry a rotational part. Every world gets its own frequencies.
    """
    t = step * dt
    amplitude = 0.004 + 0.001 * world
    press = 20.0 + 5.0 * world
    z = -amplitude * (1.0 - np.cos(2.0 * np.pi * press * t)) * 0.5
    vz = -amplitude * np.pi * press * np.sin(2.0 * np.pi * press * t)
    lateral = 15.0 + 3.0 * world
    x = 0.001 * np.sin(2.0 * np.pi * lateral * t)
    vx = 0.001 * 2.0 * np.pi * lateral * np.cos(2.0 * np.pi * lateral * t)
    tilt = 9.0 + world
    pitch = 0.02 * np.sin(2.0 * np.pi * tilt * t)
    rate = 0.02 * 2.0 * np.pi * tilt * np.cos(2.0 * np.pi * tilt * t)
    pose = np.array([x, 0.0, z, 0.0, np.sin(0.5 * pitch), 0.0, np.cos(0.5 * pitch)], np.float64)
    return pose, np.array([vx, 0.0, vz, 0.0, rate, 0.0], np.float64)


def _run(
    world_count: int = 1,
    *,
    device=None,
    worlds: list[int] | None = None,
    steps: int = STEPS,
    dt: float = DT_S,
    lift_m: dict[int, float] | None = None,
    poison_slot: int | None = None,
    surround: bool = True,
    materials: list[ShoeMaterial] | None = None,
) -> dict[str, np.ndarray]:
    """Step the batched foundation through the scripted motion and return every public array.

    Args:
        world_count: Number of worlds the foundation carries.
        device: Warp device.
        worlds: Motion identity of each slot, defaulting to ``range(world_count)``,
            so one world can be simulated alone in the same way it is simulated
            inside a batch.
        steps: Substeps to run.
        dt: Substep duration [s].
        lift_m: Vertical offset added to a slot's carrier height [m], used to
            perturb one world without touching the others.
        poison_slot: Slot whose tiled state is overwritten with NaN before every
            substep, so any read across a world boundary is fatal.
        surround: Let the columns the carrier does not drive relax every substep.
            Switching it off freezes ``z_free``, which is what lets a NaN survive
            long enough to poison the compression the shear layer reads.
        materials: One material per world, applied through
            :meth:`~projects.digital_shoe.runtime.MidsoleFoundation.set_world_materials`
            after construction. ``None`` leaves every world on the construction
            material and never calls the setter.
    """
    bed = _build_bed()
    worlds = list(range(world_count)) if worlds is None else list(worlds)
    builder = newton.ModelBuilder()
    bodies = [
        builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3))) for _ in range(world_count)
    ]
    model = builder.finalize(device=device)
    state = model.state()
    foundation = MidsoleFoundation(
        bed["anchor"],
        bed["z_free"],
        bed["rest"],
        bed["area"],
        bed["neighbors"],
        SPACING_M,
        MATERIAL,
        bodies if world_count > 1 else bodies[0],
        model.body_com,
        FoundationConfig(
            stretch_floor=0.05,
            normal_damping=5.0,
            friction_stiffness=1.0e4,
            friction=20.0,
            mu=0.6,
        ),
        device,
        SurroundConfig(driven=bed["driven"], max_strain=0.9, sweeps=3, carrier_bond=True) if surround else None,
        world_count=world_count,
    )
    if materials is not None:
        foundation.set_world_materials(materials)
    history: dict[str, list[np.ndarray]] = {name: [] for name in HISTORY_FIELDS}
    for step in range(steps):
        pose = np.zeros((world_count, 7), np.float32)
        velocity = np.zeros((world_count, 6), np.float32)
        for slot, world in enumerate(worlds):
            q, qd = _carrier_motion(world, step, dt)
            if lift_m is not None and slot in lift_m:
                q[2] += lift_m[slot]
            pose[slot] = q
            velocity[slot] = qd
        state.body_q.assign(pose)
        state.body_qd.assign(velocity)
        state.clear_forces()
        if poison_slot is not None:
            _poison(foundation, poison_slot, surround=surround)
        foundation.apply(state, dt)
        for name in HISTORY_FIELDS:
            history[name].append(getattr(foundation, name).numpy().copy())
    result = {name: np.asarray(values) for name, values in history.items()}
    for name in COLUMN_FIELDS:
        if surround or not name.startswith("surround"):
            result[name] = getattr(foundation, name).numpy().copy()
    for name in ("cop_moment", "resultant_force", "resultant_moment_origin"):
        result[name] = getattr(foundation, name).numpy().copy()
    result["body_f"] = state.body_f.numpy().copy()
    return result


def _poison(foundation: MidsoleFoundation, slot: int, surround: bool = True) -> None:
    """Fill one world's tile of every per-column state array with NaN."""
    lo, hi = slot * COLUMN_COUNT, (slot + 1) * COLUMN_COUNT
    fields = ["z_free", "compression", "base_pressure", "q_state", "peq_prev"]
    if surround:
        fields.append("surround_compression")
    for name in fields:
        values = getattr(foundation, name).numpy()
        values[lo:hi] = np.nan
        getattr(foundation, name).assign(values)


class TestBatchedMidsoleFoundation(unittest.TestCase):
    def assert_reduction_close(self, got, want, name: str, slot: int, device) -> None:
        """Compare one atomically reduced quantity within the CUDA atomic ordering noise.

        ``wp.atomic_add`` on floats sums in whatever order the scheduler runs the
        blocks in, so a reduction is only reproducible up to that ordering. Two
        identical reruns of this very scenario already disagree by ~1.3e-7
        relative on ``normal_force``, ``pressed_force`` and ``contact_power``, and
        the full rig was independently measured at ~5e-6 relative; the original
        single-world runtime has the same property, so this is not a batching
        artifact and cannot be tested away. :data:`REDUCTION_RTOL` is the agreed
        margin above that floor.

        The absolute floor is taken from the largest component of the expected
        value, not from each component: a moment sums terms of the same size with
        opposite signs, so a component that cancels to near zero still carries the
        absolute noise of the big terms that built it. Integer counts and
        ``max_compression`` (an ``atomic_max``, which is order independent) stay
        exact everywhere.
        """
        want = np.asarray(want)
        scale = float(np.max(np.abs(want))) if want.size else 0.0
        np.testing.assert_allclose(
            got,
            want,
            rtol=REDUCTION_RTOL,
            atol=REDUCTION_RTOL * max(scale, 1.0e-9),
            err_msg=f"{name} differs in world {slot} on {device}",
        )

    def assert_world_matches(self, expected: dict, actual: dict, slot: int, device, reference_slot: int = 0) -> None:
        """Assert one world of a batched run reproduces a reference run of the same motion.

        Everything is compared bit for bit on every device, reductions included.
        That is only possible because the totals are no longer accumulated with
        ``wp.atomic_add`` from every column: :func:`~projects.digital_shoe.runtime.foundation_partial`
        gives each group a fixed set of columns and
        :func:`~projects.digital_shoe.runtime.foundation_finalize` folds the groups
        in index order, so a world's sum depends on nothing but its own columns and
        is independent of how the launch was scheduled or of how many worlds shared
        it. The atomic version could only be checked to a tolerance.
        """
        exact_reductions = True
        for name in [field for field in COLUMN_FIELDS if field in actual]:
            np.testing.assert_array_equal(
                actual[name][slot * COLUMN_COUNT : (slot + 1) * COLUMN_COUNT],
                expected[name][reference_slot * COLUMN_COUNT : (reference_slot + 1) * COLUMN_COUNT],
                err_msg=f"{name} differs in world {slot} on {device}",
            )
        for name in HISTORY_FIELDS:
            got, want = actual[name][:, slot], expected[name][:, reference_slot]
            if exact_reductions or name in EXACT_REDUCTIONS:
                np.testing.assert_array_equal(got, want, err_msg=f"{name} differs in world {slot} on {device}")
            else:
                self.assert_reduction_close(got, want, name, slot, device)
        for name in FINAL_WORLD_FIELDS:
            got, want = actual[name][slot], expected[name][reference_slot]
            if exact_reductions:
                np.testing.assert_array_equal(got, want, err_msg=f"{name} differs in world {slot} on {device}")
            else:
                self.assert_reduction_close(got, want, name, slot, device)

    def assert_matches_pre_batching(self, result: dict) -> None:
        """Compare a single-world run against the runtime that predates world batching.

        Per-column state must be bit-identical: none of it passes through a sum, so
        neither batching, nor per-world materials, nor the segmented reduction may
        move a single bit. The summed totals are compared within the ordering band
        instead, because the reduction deliberately replaced ~910 float atomics per
        world with a fixed-order two-pass sum; see
        :meth:`assert_reduction_close`.
        """
        digests = {name: _digest(values) for name, values in result.items() if name in PRE_BATCHING_DIGESTS}
        self.assertEqual(digests, PRE_BATCHING_DIGESTS)
        for name, expected in PRE_BATCHING_REDUCTIONS.items():
            want = np.asarray(expected, np.float32)
            got = result[name][:, 0] if name in HISTORY_FIELDS else result[name][0]
            self.assert_reduction_close(got, want, name, 0, wp.get_device("cpu"))

    def test_single_world_reproduces_the_pre_batching_runtime(self):
        """Reproduce the pre-batching single-carrier runtime with one world.

        The reference comes from the runtime as it stood before the column bed grew a
        world dimension, captured from the same scenario on the same device. Batching
        must be free for existing callers, so exact float32 bytes are the acceptance
        criterion everywhere the arithmetic is unchanged.
        """
        result = _run(world_count=1, device=wp.get_device("cpu"))
        self.assert_matches_pre_batching(result)
        # The scenario has to be loaded, sliding and viscoelastic, or the digests
        # above would pin an idle bed.
        self.assertGreater(float(result["normal_force"].max()), 30.0)
        self.assertEqual(int(result["tangent_stuck"].sum()), COLUMN_COUNT)
        self.assertGreater(float(np.abs(result["column_force"][:, 0]).max()), 0.0)

    def test_public_arrays_keep_their_single_world_lengths(self):
        """Keep every public array at its old length when the default world count is used."""
        foundation = _build_foundation(world_count=1, device=wp.get_device("cpu"))
        for name in COLUMN_FIELDS:
            self.assertEqual(len(getattr(foundation, name)), COLUMN_COUNT, name)
        for name in ("normal_force", "pressed_force", "contact_power", "max_compression", "active"):
            self.assertEqual(len(getattr(foundation, name)), 1, name)

    def test_each_world_reproduces_its_solo_simulation(self):
        """Reproduce four solo simulations with four differently driven worlds in one launch."""
        for device in _available_devices():
            with self.subTest(device=str(device)):
                batched = _run(world_count=4, device=device)
                for slot in range(4):
                    solo = _run(world_count=1, device=device, worlds=[slot])
                    self.assert_world_matches(solo, batched, slot, device)

    def test_perturbing_one_world_leaves_the_others_bit_identical(self):
        """Leave the untouched worlds bit-identical when world 2's carrier is perturbed."""
        for device in _available_devices():
            with self.subTest(device=str(device)):
                reference = _run(world_count=4, device=device)
                perturbed = _run(world_count=4, device=device, lift_m={2: 0.001})
                for slot in (0, 1, 3):
                    self.assert_world_matches(reference, perturbed, slot, device, reference_slot=slot)
                # The perturbation has to actually change the world it targets, by far
                # more than the atomic ordering noise the comparison above tolerates.
                self.assertGreater(
                    float(np.abs(perturbed["normal_force"][:, 2] - reference["normal_force"][:, 2]).max()),
                    1.0,
                )

    def test_neighbour_coupling_never_reaches_across_a_world(self):
        """Keep the Pasternak neighbour table inside one world, proved with a poisoned neighbour world.

        ``neighbors`` is shared between the worlds and holds plain column indices,
        so every kernel has to add its own tile base before it reads a neighbour's
        state. Filling world 1 with NaN before every substep makes any missing or
        wrong offset fatal: a single cross-world read would carry the NaN into the
        shear flux of worlds 0 and 2 and into their reductions.
        """
        for device in _available_devices():
            with self.subTest(device=str(device)):
                foundation = _build_foundation(world_count=3, device=device)
                self.assertEqual(foundation.neighbors.shape, (COLUMN_COUNT, 4))
                table = foundation.neighbors.numpy()
                self.assertTrue(np.all(table < COLUMN_COUNT))
                self.assertTrue(np.all(table >= -1))

                for surround in (True, False):
                    clean = _run(world_count=3, device=device, surround=surround)
                    poisoned = _run(world_count=3, device=device, poison_slot=1, surround=surround)
                    # The NaN has to be alive inside the launch, or the test proves nothing.
                    # Without the surround sweep ``z_free`` is frozen, so the poison reaches
                    # the compression the shear layer reads; with it, the clamped relaxation
                    # rewrites the compression and the Maxwell state carries the NaN instead.
                    poisoned_tile = slice(COLUMN_COUNT, 2 * COLUMN_COUNT)
                    self.assertTrue(np.any(np.isnan(poisoned["q_state"][poisoned_tile])))
                    self.assertTrue(np.any(np.isnan(poisoned["base_pressure"][poisoned_tile])))
                    if not surround:
                        self.assertTrue(np.all(np.isnan(poisoned["compression"][poisoned_tile])))
                    for slot in (0, 2):
                        lo, hi = slot * COLUMN_COUNT, (slot + 1) * COLUMN_COUNT
                        for name in [field for field in COLUMN_FIELDS if field in poisoned]:
                            self.assertFalse(np.any(np.isnan(poisoned[name][lo:hi])), f"NaN leaked into {name}")
                        for name in HISTORY_FIELDS:
                            self.assertFalse(np.any(np.isnan(poisoned[name][:, slot])), f"NaN leaked into {name}")
                        self.assert_world_matches(clean, poisoned, slot, device, reference_slot=slot)

    def test_reductions_are_per_world_and_do_not_mix(self):
        """Reduce each world's columns into its own accumulator slot only."""
        world_count = 3
        for device in _available_devices():
            with self.subTest(device=str(device)):
                foundation = _build_foundation(world_count=world_count, device=device)
                for name in ("normal_force", "pressed_force", "contact_power", "max_compression", "active"):
                    self.assertEqual(len(getattr(foundation, name)), world_count, name)
                for name in ("cop_moment", "resultant_force", "resultant_moment_origin"):
                    self.assertEqual(len(getattr(foundation, name)), world_count, name)
                for name in COLUMN_FIELDS:
                    self.assertEqual(len(getattr(foundation, name)), world_count * COLUMN_COUNT, name)
                for name in ("anchor_local", "area", "rest_len", "friction_kt", "friction_kv", "driven"):
                    self.assertEqual(len(getattr(foundation, name)), COLUMN_COUNT, name)

                result = _run(world_count=world_count, device=device)
                for slot in range(world_count):
                    lo, hi = slot * COLUMN_COUNT, (slot + 1) * COLUMN_COUNT
                    columns = result["column_force"][lo:hi]
                    compression = result["compression"][lo:hi]
                    # The atomic sum and this host sum add the same terms in different
                    # orders, so they agree only to the accumulated rounding of the terms.
                    floor = REDUCTION_RTOL * float(np.abs(columns).sum())
                    np.testing.assert_allclose(
                        result["normal_force"][-1, slot], columns[:, 2].sum(), rtol=REDUCTION_RTOL, atol=floor
                    )
                    np.testing.assert_allclose(
                        result["resultant_force"][slot], columns.sum(axis=0), rtol=REDUCTION_RTOL, atol=floor
                    )
                    self.assertEqual(int(result["active"][-1, slot]), int((compression > 0.0).sum()))
                    self.assertEqual(float(result["max_compression"][-1, slot]), float(compression.max()))
                # Distinct motions must give distinct loads, so an accidental shared
                # accumulator would be visible rather than hidden behind equal worlds.
                peaks = result["normal_force"].max(axis=0)
                self.assertEqual(len(set(peaks.tolist())), world_count)

    def test_setting_the_same_material_everywhere_changes_nothing(self):
        """Reproduce the shared-material path bit for bit when every world is set to that material.

        Per-world materials moved the constitutive constants from a by-value
        struct into an array the kernels index by world. Re-setting the very same
        material through the new API must therefore be a no-op down to the last
        bit, on the digests captured from the runtime that predates both changes.
        """
        result = _run(world_count=1, device=wp.get_device("cpu"), materials=[MATERIAL])
        self.assert_matches_pre_batching(result)
        for device in _available_devices():
            with self.subTest(device=str(device)):
                untouched = _run(world_count=4, device=device)
                uniform = _run(world_count=4, device=device, materials=[MATERIAL] * 4)
                for slot in range(4):
                    self.assert_world_matches(untouched, uniform, slot, device, reference_slot=slot)

    def test_randomized_materials_stay_inside_their_own_world(self):
        """Give every world its own foam and keep each one equal to its solo simulation.

        This is the domain-randomization case: one geometry, one contact law, one
        launch, four different materials. Each world is checked against the same
        world simulated alone under the same material, and world 2's material is
        then re-drawn to show the others do not move. A world reading a neighbour
        world's constitutive block would break both halves.
        """
        materials = [_randomized_material(world) for world in range(4)]
        for device in _available_devices():
            with self.subTest(device=str(device)):
                batched = _run(world_count=4, device=device, materials=materials)
                for slot in range(4):
                    solo = _run(world_count=1, device=device, worlds=[slot], materials=[materials[slot]])
                    self.assert_world_matches(solo, batched, slot, device)
                # The materials have to be far enough apart to be distinguishable.
                peaks = batched["normal_force"].max(axis=0)
                self.assertEqual(len(set(peaks.tolist())), 4)

                redrawn = list(materials)
                redrawn[2] = _randomized_material(7)
                perturbed = _run(world_count=4, device=device, materials=redrawn)
                for slot in (0, 1, 3):
                    self.assert_world_matches(batched, perturbed, slot, device, reference_slot=slot)
                self.assertGreater(
                    float(np.abs(perturbed["normal_force"][:, 2] - batched["normal_force"][:, 2]).max()),
                    1.0,
                )

    def test_reductions_are_reproducible_run_to_run(self):
        """Produce identical totals from two identical runs, which float atomics could not.

        The previous implementation accumulated every world's totals with
        ``wp.atomic_add`` over ~910 columns, so the summation order was whatever the
        scheduler chose and two identical CUDA runs disagreed by ~1e-7 relative on
        ``normal_force``, ``pressed_force`` and ``contact_power``. The segmented
        reduction fixes the order, so repeated runs must now agree exactly, and this
        test is the guard against a future change quietly reintroducing an atomic.
        """
        for device in _available_devices():
            with self.subTest(device=str(device)):
                first = _run(world_count=4, device=device)
                second = _run(world_count=4, device=device)
                for name in first:
                    np.testing.assert_array_equal(
                        first[name], second[name], err_msg=f"{name} is not reproducible on {device}"
                    )

    def test_batched_substeps_survive_a_cuda_graph_capture(self):
        """Capture a batch of substeps in a CUDA graph and replay it against the eager result.

        The RL loop captures one policy step of many substeps, so
        :meth:`~projects.digital_shoe.runtime.MidsoleFoundation.apply` must contain
        no device read, no host branch on device data and no allocation. Anything
        that breaks those rules fails the capture outright; a stale ping-pong
        buffer in the surround sweep would instead pass the capture and then
        diverge on replay, which is what the comparison below catches.
        """
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture needs a CUDA device")
        device = wp.get_device("cuda:0")
        captured = _CaptureHarness(world_count=4, device=device, substeps=6)
        self.assertIsNotNone(captured.graph)
        loose = captured.step_eagerly()
        replayed = captured.step_with_graph()
        for slot in range(4):
            self.assert_world_matches(loose, replayed, slot, device, reference_slot=slot)

    def test_rejects_a_carrier_set_that_cannot_be_one_body_per_world(self):
        """Reject a world count below one, a mismatched carrier list, and a shared carrier body."""
        bed = _build_bed()
        builder = newton.ModelBuilder()
        bodies = [builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3))) for _ in range(2)]
        model = builder.finalize(device=wp.get_device("cpu"))
        common = (
            bed["anchor"],
            bed["z_free"],
            bed["rest"],
            bed["area"],
            bed["neighbors"],
            SPACING_M,
            MATERIAL,
        )
        with self.assertRaisesRegex(ValueError, "world_count"):
            MidsoleFoundation(*common, bodies, model.body_com, world_count=0)
        with self.assertRaisesRegex(ValueError, "one body index per world"):
            MidsoleFoundation(*common, bodies, model.body_com, world_count=3)
        with self.assertRaisesRegex(ValueError, "own carrier body"):
            MidsoleFoundation(*common, [bodies[0], bodies[0]], model.body_com, world_count=2)
        foundation = MidsoleFoundation(*common, bodies, model.body_com, world_count=2)
        with self.assertRaises(IndexError):
            foundation.set_world_material(2, MATERIAL)
        with self.assertRaisesRegex(ValueError, "one material per world"):
            foundation.set_world_materials([MATERIAL])


def _collect(foundation: MidsoleFoundation, state, surround: bool = True) -> dict[str, np.ndarray]:
    """Read every public array of a foundation into the layout the comparisons use."""
    result = {name: getattr(foundation, name).numpy().copy()[None, :] for name in HISTORY_FIELDS}
    for name in COLUMN_FIELDS:
        if surround or not name.startswith("surround"):
            result[name] = getattr(foundation, name).numpy().copy()
    for name in ("cop_moment", "resultant_force", "resultant_moment_origin"):
        result[name] = getattr(foundation, name).numpy().copy()
    result["body_f"] = state.body_f.numpy().copy()
    return result


class _CaptureHarness:
    """Two identical batched foundations, one stepped eagerly and one through a CUDA graph.

    Both are warmed up with one eager substep before the capture, which is the
    contract the class documents: the per-world surround constants are uploaded on
    the first call for a given substep, and a capture must not contain that copy.
    """

    def __init__(self, world_count: int, device, substeps: int, dt: float = DT_S) -> None:
        self.substeps = int(substeps)
        self.dt = dt
        materials = [_randomized_material(world) for world in range(world_count)]
        self.loose = _make_world(world_count, device, materials)
        self.graphed = _make_world(world_count, device, materials)
        pose = np.zeros((world_count, 7), np.float32)
        velocity = np.zeros((world_count, 6), np.float32)
        for world in range(world_count):
            q, qd = _carrier_motion(world, 6, dt)
            pose[world] = q
            velocity[world] = qd
        for foundation, state in (self.loose, self.graphed):
            state.body_q.assign(pose)
            state.body_qd.assign(velocity)
            state.clear_forces()
            foundation.apply(state, self.dt)
        foundation, state = self.graphed
        with wp.ScopedCapture(device) as capture:
            for _ in range(self.substeps):
                state.clear_forces()
                foundation.apply(state, self.dt)
        self.graph = capture.graph

    def step_eagerly(self) -> dict[str, np.ndarray]:
        """Run the uncaptured foundation for the same substeps and read it back."""
        foundation, state = self.loose
        for _ in range(self.substeps):
            state.clear_forces()
            foundation.apply(state, self.dt)
        return _collect(foundation, state)

    def step_with_graph(self) -> dict[str, np.ndarray]:
        """Replay the captured graph once and read the foundation back."""
        wp.capture_launch(self.graph)
        return _collect(*self.graphed)


def _make_world(world_count: int, device, materials: list[ShoeMaterial] | None = None):
    """Return a foundation and the state carrying its carrier bodies."""
    bed = _build_bed()
    builder = newton.ModelBuilder()
    bodies = [
        builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3))) for _ in range(world_count)
    ]
    model = builder.finalize(device=device)
    foundation = MidsoleFoundation(
        bed["anchor"],
        bed["z_free"],
        bed["rest"],
        bed["area"],
        bed["neighbors"],
        SPACING_M,
        MATERIAL,
        bodies if world_count > 1 else bodies[0],
        model.body_com,
        FoundationConfig(stretch_floor=0.05, normal_damping=5.0, friction_stiffness=1.0e4, friction=20.0, mu=0.6),
        device,
        SurroundConfig(driven=bed["driven"], max_strain=0.9, sweeps=3, carrier_bond=True),
        world_count=world_count,
    )
    if materials is not None:
        foundation.set_world_materials(materials)
    return foundation, model.state()


def _build_foundation(world_count: int, device) -> MidsoleFoundation:
    """Build the test foundation without stepping it."""
    bed = _build_bed()
    builder = newton.ModelBuilder()
    bodies = [
        builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3))) for _ in range(world_count)
    ]
    model = builder.finalize(device=device)
    return MidsoleFoundation(
        bed["anchor"],
        bed["z_free"],
        bed["rest"],
        bed["area"],
        bed["neighbors"],
        SPACING_M,
        MATERIAL,
        bodies if world_count > 1 else bodies[0],
        model.body_com,
        FoundationConfig(stretch_floor=0.05, normal_damping=5.0, friction_stiffness=1.0e4, friction=20.0, mu=0.6),
        device,
        SurroundConfig(driven=bed["driven"], max_strain=0.9, sweeps=3, carrier_bond=True),
        world_count=world_count,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
