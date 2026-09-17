# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gradient tests for the differentiable elastic-foundation midsole."""

import json
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.digital_instron_v2 import core, dynamics, dynamics_diff, inverse_id, scenarios_diff, workflow
from projects.digital_instron_v2.dynamics import FoundationConfig
from projects.digital_instron_v2.dynamics_diff import DifferentiableMidsoleFoundation
from projects.digital_instron_v2.geometry import build_column_grid, load_mesh
from projects.digital_shoe import runtime

MANIFEST = "DigitalInstron/manifest_v2.json"


@wp.kernel
def _final_height(body_q: wp.array[wp.transform], out: wp.array[wp.float32]):
    out[0] = wp.transform_get_translation(body_q[0])[2]


@wp.kernel
def _tangential_x(body_f: wp.array[wp.spatial_vector], out: wp.array[wp.float32]):
    out[0] = wp.spatial_top(body_f[0])[0]


class _Drop:
    """Differentiable drop of a unit carrier body onto the calibrated foam bed.

    Wraps the whole ``num_substeps`` substep loop in a single :class:`warp.Tape`
    so an objective on the final state can be differentiated w.r.t. the initial
    pose and the foam ``material_params``. The body starts already compressed
    (``z0 < 0``) so the rollout stays on the continuous branch of the contact,
    where the analytic gradient matches a finite difference; a drop from above
    the foam would cross the touchdown make/break event where the (correct)
    gradient is a subgradient that a finite difference will not match.
    """

    def __init__(self, device, num_substeps=60, config=None):
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        self.device = device
        self.nsub = num_substeps
        self.dt = 1.0 / 60.0 / 32.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        self.model = builder.finalize(requires_grad=True, device=device)
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.states = [self.model.state() for _ in range(num_substeps + 1)]

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.slack_m])
        self.foundation = DifferentiableMidsoleFoundation(
            anchor_local=anchor,
            z_free=geo.slack_m,
            rest_len=geo.slack_m,
            area=np.full(len(geo.slack_m), geo.area_m2),
            neighbors=geo.neighbors,
            spacing_m=geo.spacing_m,
            material=material,
            carrier_body=self.carrier,
            body_com=self.model.body_com,
            num_substeps=num_substeps,
            config=config or FoundationConfig(normal_damping=5.0),
            device=device,
        )
        self.loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    def set_initial(self, z0, vx=0.0):
        q = np.zeros((1, 7), np.float32)
        q[0, 2] = z0
        q[0, 6] = 1.0
        self.states[0].body_q.assign(q)
        qd = np.zeros((1, 6), np.float32)
        qd[0, 0] = vx  # spatial_top (linear) x-velocity
        self.states[0].body_qd.assign(qd)

    def forward(self):
        for t in range(self.nsub):
            self.states[t].body_f.zero_()
            self.foundation.apply(self.states[t], t, self.dt)
            self.solver.step(self.states[t], self.states[t + 1], None, None, self.dt)
        wp.launch(_final_height, dim=1, inputs=[self.states[self.nsub].body_q, self.loss], device=self.device)
        return self.loss


class TestDifferentiableFoundationGradients(unittest.TestCase):
    def test_initial_height_gradient_matches_finite_difference(self):
        """Differentiate the final drop height w.r.t. the initial height and match a converging central difference.

        The difference quotient is evaluated at two steps and must both approach
        the analytic gradient and improve as the step shrinks. A single fixed step
        is not enough here: per-column contact events (the damping branch and the
        unilateral clamp switch as the start height moves) make the rollout
        piecewise smooth, so the central difference converges only first order in
        the step.

        The steps are 200 um and 20 um, both well above the float32 noise floor of
        the rollout. Measured on the corrected foam that noise is about 2e-9 m of
        final height, so a quotient at 2 um -- the step this test used before the
        stiffer material-pinned shear layer -- is dominated by it (2e-4 relative,
        an order of magnitude *worse* than the 20 um quotient) and the two steps no
        longer order reliably. At 200 um the truncation error is 6e-4 relative and
        at 20 um it is 1e-5, so the ordering is real mechanics.
        """
        device = wp.get_preferred_device()
        drop = _Drop(device, num_substeps=60)
        z0 = -0.003

        drop.set_initial(z0)
        tape = wp.Tape()
        with tape:
            loss = drop.forward()
        tape.backward(loss)
        analytic = float(drop.states[0].body_q.grad.numpy()[0][2])

        tape.zero()
        drop.foundation.zero_grad()

        def difference_quotient(eps):
            drop.set_initial(z0 + eps)
            plus = float(drop.forward().numpy()[0])
            drop.set_initial(z0 - eps)
            minus = float(drop.forward().numpy()[0])
            return (plus - minus) / (2.0 * eps)

        coarse = difference_quotient(2.0e-4)
        numeric = difference_quotient(2.0e-5)
        self.assertLess(abs(analytic - numeric), abs(analytic - coarse))  # converging on the analytic gradient
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-9), 1.0e-2)

    def test_material_parameter_gradients_match_finite_difference(self):
        """Differentiate the final drop height w.r.t. each foam material parameter and match central differences.

        Covers the full differentiable constitutive vector -- equilibrium shear
        modulus, Hyperfoam exponent, and Maxwell overstress ratio -- with the
        viscoelastic overstress branch active. The shear layer is pinned to the
        material as ``mu_eq * t``, so its sensitivity rides on ``g_eq``.
        """
        device = wp.get_preferred_device()
        drop = _Drop(device, num_substeps=60)
        z0 = -0.003

        drop.set_initial(z0)
        tape = wp.Tape()
        with tape:
            loss = drop.forward()
        tape.backward(loss)
        analytic = drop.foundation.material_params.grad.numpy().copy()

        tape.zero()
        drop.foundation.zero_grad()
        base = drop.foundation.material_params.numpy().copy()
        numeric = np.empty_like(base)
        for k in range(len(base)):
            h = max(abs(base[k]) * 1.0e-3, 1.0e-6)
            perturbed = base.copy()
            perturbed[k] += h
            drop.foundation.material_params.assign(perturbed)
            drop.set_initial(z0)
            plus = float(drop.forward().numpy()[0])
            perturbed = base.copy()
            perturbed[k] -= h
            drop.foundation.material_params.assign(perturbed)
            drop.set_initial(z0)
            minus = float(drop.forward().numpy()[0])
            numeric[k] = (plus - minus) / (2.0 * h)
            drop.foundation.material_params.assign(base)

        # Vector-norm relative error: robust when one component's gradient is much
        # smaller than the others (the Hyperfoam exponent here), whose per-component
        # finite difference sits near the deterministic floor of the short rollout.
        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-30)
        self.assertLess(rel, 1.0e-2)

    def test_deprecated_raw_kernel_preserves_smooth_friction(self):
        """Keep the old raw-kernel behavior without routing the active class through it."""
        device = wp.get_preferred_device()
        drop = _Drop(device, num_substeps=1, config=FoundationConfig(mu=0.6))
        drop.set_initial(-0.004, vx=0.03)
        state = drop.states[0]
        foundation = drop.foundation
        foundation.apply(state, 0, drop.dt)
        self.assertEqual(float(state.body_f.numpy()[0, 0]), 0.0)
        pressed = np.maximum(foundation.column_force[0].numpy()[:, 2], 0.0).sum()
        state.body_f.zero_()
        wp.launch(
            dynamics_diff.foundation_apply_diff,
            dim=foundation.column_count,
            inputs=[
                foundation.carrier,
                state.body_q,
                state.body_qd,
                foundation.body_com,
                foundation.anchor_local,
                foundation.area,
                foundation.rest_len,
                foundation.neighbors,
                foundation.compression[0],
                foundation.base_pressure[0],
                foundation.params,
                foundation.material_params,
                foundation.friction_params,
                foundation.friction_smoothing,
                state.body_f,
                foundation.normal_force,
                foundation.cop_moment,
                foundation.pressed_force,
                foundation.active,
            ],
            device=device,
        )
        expected = -0.6 * pressed * 0.03 / np.sqrt(0.03**2 + 0.05**2)
        self.assertAlmostEqual(float(state.body_f.numpy()[0, 0]) / expected, 1.0, places=5)

    def test_bristle_friction_respects_cone_and_is_differentiable(self):
        """Verify the shared bristle law respects the cone and matches finite differences.

        A single substep with a planted, laterally sliding contact patch must
        produce a tangential force that opposes motion, is bounded by ``mu * fn``,
        and whose sensitivity to the slip velocity matches a central difference.
        """
        device = wp.get_preferred_device()
        config = FoundationConfig(normal_damping=5.0, friction_stiffness=1.0e4, mu=0.6)
        drop = _Drop(device, num_substeps=1, config=config)

        out = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        vx0 = 0.03

        def tangential(vx):
            drop.set_initial(-0.004, vx=vx)
            drop.states[0].body_f.zero_()
            drop.foundation.apply(drop.states[0], 0, drop.dt)
            wp.launch(_tangential_x, dim=1, inputs=[drop.states[0].body_f, out], device=device)
            return float(out.numpy()[0])

        drop.set_initial(-0.004, vx=vx0)
        drop.states[0].body_f.zero_()
        tape = wp.Tape()
        with tape:
            drop.foundation.apply(drop.states[0], 0, drop.dt)
            wp.launch(_tangential_x, dim=1, inputs=[drop.states[0].body_f, out], device=device)
        tape.backward(out)
        analytic = float(drop.states[0].body_qd.grad.numpy()[0][0])

        ft_x = tangential(vx0)
        fn = drop.foundation.diagnostics()["normal_force_n"]
        self.assertGreater(fn, 0.0)
        self.assertLess(ft_x, 0.0)  # friction opposes +x motion
        self.assertLessEqual(abs(ft_x), config.mu * fn)  # inside the cone

        eps = 1.0e-3
        numeric = (tangential(vx0 + eps) - tangential(vx0 - eps)) / (2.0 * eps)
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-9), 5.0e-2)

    def test_inverse_identification_recovers_stiffness(self):
        """Recover a perturbed foam stiffness from a target drop height using the analytic gradient.

        Gauss-Newton on the residual final_height(s) - target, stepping with the
        exact tape gradient d(final_height)/ds, must recover the true stiffness
        scale from a 30% error in a few iterations -- the differentiable-contact
        inverse-identification payoff.
        """
        device = wp.get_preferred_device()
        drop = _Drop(device, num_substeps=60)
        z0 = -0.003
        base = drop.foundation.material_params.numpy().copy()
        g_eq_ref = float(base[0])

        drop.set_initial(z0)
        target = float(drop.forward().numpy()[0])

        base[0] = 0.7 * g_eq_ref  # 30% stiffness error
        drop.foundation.material_params.assign(base)
        for _ in range(6):
            drop.set_initial(z0)
            tape = wp.Tape()
            with tape:
                z = drop.forward()
            tape.backward(z)
            drds = float(drop.foundation.material_params.grad.numpy()[0])
            residual = float(z.numpy()[0]) - target
            tape.zero()
            drop.foundation.zero_grad()
            if abs(residual) < 1.0e-7 or abs(drds) < 1.0e-15:
                break
            cur = drop.foundation.material_params.numpy().copy()
            cur[0] -= residual / drds
            drop.foundation.material_params.assign(cur)

        recovered = float(drop.foundation.material_params.numpy()[0])
        self.assertLess(abs(recovered - g_eq_ref) / g_eq_ref, 1.0e-2)


@wp.kernel
def _sum_force(force: wp.array[wp.float32], out: wp.array[wp.float32]):
    wp.atomic_add(out, 0, force[wp.tid()])


class TestForceMatchingInverseIdentification(unittest.TestCase):
    def _replay(self, device, samples=48):
        geometry = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        displacement = inverse_id._triangular_cycle(peak_m=0.006, samples=samples)
        replay = inverse_id.InstronReplay(displacement, 1.0 / 200.0, material, geometry, device=device)
        return replay, displacement, material, geometry

    def test_instron_force_gradient_matches_finite_difference(self):
        """Differentiate the total Instron reaction impulse w.r.t. the foam material and match central differences.

        Exercises the prescribed-displacement force readout
        (:func:`~projects.digital_instron_v2.inverse_id._accumulate_normal_force`)
        with the Maxwell recurrence active.
        """
        device = wp.get_preferred_device()
        replay, _, _, _ = self._replay(device)
        out = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        replay.zero_grad()
        out.zero_()
        tape = wp.Tape()
        with tape:
            force = replay.forward()
            wp.launch(_sum_force, dim=replay.nsteps, inputs=[force, out], device=device)
        tape.backward(out)
        analytic = replay.material_params.grad.numpy().copy()

        tape.zero()
        replay.zero_grad()
        base = replay.material_params.numpy().copy()
        numeric = np.empty_like(base)
        for k in range(len(base)):
            h = max(abs(base[k]) * 1.0e-3, 1.0e-6)
            perturbed = base.copy()
            perturbed[k] += h
            replay.set_material(perturbed)
            plus = float(replay.forward().numpy().sum())
            perturbed = base.copy()
            perturbed[k] -= h
            replay.set_material(perturbed)
            minus = float(replay.forward().numpy().sum())
            numeric[k] = (plus - minus) / (2.0 * h)
            replay.set_material(base)

        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-30)
        self.assertLess(rel, 1.0e-2)

    def test_force_matching_recovers_elastic_material(self):
        """Recover the perturbed elastic foam parameters a uniform platen can identify, and pin the one it cannot.

        A single-rate load/unload cycle constrains stiffness and Maxwell overstress
        along a strongly correlated direction, so the overstress ratio (obtained
        separately from a relaxation test) is held fixed while the equilibrium
        modulus and the Hyperfoam exponent are fit to the curve.

        The shear layer is no longer a fitted parameter at all: its coefficient is
        the material-pinned ``mu_eq * t`` of every face. A uniform platen presses
        every column equally, so each face difference ``c_j - c_i`` vanishes and
        the layer carries exactly nothing here -- use a shaped indenter
        (:class:`~projects.digital_instron_v2.inverse_id.DifferentiableTrial`) to
        exercise it.
        """
        device = wp.get_preferred_device()
        replay, displacement, material, geometry = self._replay(device)
        target = replay.forward().numpy().copy()

        result = inverse_id.fit_material_to_force_curve(
            target,
            displacement,
            1.0 / 200.0,
            material,
            geometry,
            # Both Ogden-Hill terms are in the vector now; perturb and fit the
            # first-term pair and hold the overstress and the second term fixed.
            scale0=np.array([1.3, 0.85, 1.0, 1.0, 1.0], np.float32),
            fit_mask=np.array([1.0, 1.0, 0.0, 0.0, 0.0], np.float32),
            iterations=400,
            device=device,
        )
        self.assertLess(abs(result.scale[0] - 1.0), 2.0e-2)  # g_eq
        self.assertLess(abs(result.scale[1] - 1.0), 5.0e-2)  # hyperfoam exponent
        residual = float(np.sqrt(np.mean((result.force - target) ** 2)))
        self.assertLess(residual / target.max(), 1.0e-3)


def _synthetic_trial(frames: int = 80, columns: int = 24, seed: int = 0):
    """Build a core.Trial with a shaped (per-column) triangular compression cycle."""
    rng = np.random.default_rng(seed)
    slack = np.full(columns, 0.03, np.float32)  # 30 mm foam columns
    peak_compression = (0.004 + 0.006 * rng.random(columns)).astype(np.float32)  # 4-10 mm, varies per column
    half = frames // 2
    ramp = np.concatenate([np.linspace(0.0, 1.0, half, endpoint=False), np.linspace(1.0, 0.0, frames - half)]).astype(
        np.float32
    )
    compression = ramp[:, None] * peak_compression[None, :]
    lengths = slack[None, :] - compression
    dt = np.full(frames, 1.0 / 200.0, np.float64)
    # A smooth, both-signed Pasternak Laplacian field so the coupling and pressure floor are exercised.
    laplacian = (200.0 * (compression - compression.mean(axis=1, keepdims=True))).astype(np.float32)
    displacement = (ramp * float(peak_compression.max())).astype(np.float32)
    force = np.zeros(frames, np.float32)
    return core.Trial(
        "synthetic", slack, float(np.pi * 0.011**2 / columns), lengths, dt, force, displacement, laplacian
    )


def _bed_neighbors(nx: int, ny: int) -> np.ndarray:
    """Return the four in-plane neighbour indices of a regular nx-by-ny grid (-1 outside)."""
    neighbors = np.full((nx * ny, 4), -1, np.int32)
    for iy in range(ny):
        for ix in range(nx):
            index = iy * nx + ix
            for side, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny:
                    neighbors[index, side] = jy * nx + jx
    return neighbors


def _synthetic_surround_trial(nx: int = 7, ny: int = 6, frames: int = 24, spacing: float = 0.005):
    """Build a core.Trial whose dome indenter drives a patch of a whole relaxing foam bed.

    The bed is small enough that the shipped ``surround.sweeps`` relaxation is
    fully converged (the predicted force is unchanged by 16x more sweeps), so the
    forward model really is the quasi-static fixed point whose implicit-function
    adjoint ``DifferentiableTrial`` differentiates.
    """
    slack = np.full(nx * ny, 0.03)  # 30 mm foam columns
    xs, ys = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing, indexing="xy")
    uv = np.column_stack([xs.ravel(), ys.ravel()])
    center = uv.mean(axis=0)
    driven = np.linalg.norm(uv - center, axis=1) <= 1.6 * spacing
    half = frames // 2
    ramp = np.concatenate([np.linspace(0.0, 1.0, half, endpoint=False), np.linspace(1.0, 0.0, frames - half)])
    profile = 0.006 - 0.15 * np.linalg.norm(uv[driven] - center, axis=1)  # dome punch, 4-6 mm deep
    lengths = slack[driven][None, :] - ramp[:, None] * profile[None, :]
    surround = core.Surround(
        driven=driven,
        neighbors=_bed_neighbors(nx, ny),
        slack_m=slack,
        area_m2=spacing**2,
        spacing_m=spacing,
    )
    return core.Trial(
        "synthetic_surround",
        slack[driven],
        spacing**2,
        lengths,
        np.full(frames, 1.0 / 200.0),
        np.zeros(frames),
        ramp * float(profile.max()),
        None,
        surround,
    )


class TestMeasuredTrialForceMatching(unittest.TestCase):
    @staticmethod
    def _load_measured_trials():
        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        midsole = load_mesh(str(base / config["midsole_mesh"]), 0.001)
        grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
        trials, _, _ = workflow.prepare_trials(base, config, grid, midsole)
        material = dynamics.load_fitted_material(MANIFEST)
        return trials, material

    def test_predictor_reproduces_core_predict(self):
        """Match the quasi-static core.predict force curve on a shaped-indenter trial to machine precision.

        The differentiable predictor evaluates the periodic generalized-Maxwell
        overstress fixed point with an explicit transient-then-cycle pass, so it
        must reproduce the closed-form periodic branch of core.predict.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        trial = _synthetic_trial()
        predicted = inverse_id.DifferentiableTrial(trial, material, device=device).forward().numpy()
        reference = core.predict(trial, material)
        peak = max(float(np.max(np.abs(reference))), 1.0e-9)
        self.assertLess(float(np.max(np.abs(predicted - reference))) / peak, 1.0e-3)

    def test_material_gradient_matches_finite_difference(self):
        """Differentiate a shaped-indenter reaction impulse w.r.t. the foam material and match central differences.

        Covers the full ``[g_eq, alpha, overstress, g_eq2, alpha2]`` vector through the two-pass
        periodic recurrence with the viscoelastic branch active.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        trial = _synthetic_trial()
        predictor = inverse_id.DifferentiableTrial(trial, material, device=device)
        target = wp.zeros(predictor.frame_count, dtype=wp.float32, device=device)  # loss = sum(force^2)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        predictor.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            force = predictor.forward()
            wp.launch(
                inverse_id._weighted_force_mse,
                dim=predictor.frame_count,
                inputs=[force, target, 1.0, loss],
                device=device,
            )
        tape.backward(loss)
        analytic = predictor.material_params.grad.numpy().copy()

        tape.zero()
        predictor.zero_grad()
        base = predictor.material_params.numpy().copy()
        numeric = np.empty_like(base)
        for k in range(len(base)):
            h = max(abs(base[k]) * 1.0e-3, 1.0e-6)
            perturbed = base.copy()
            perturbed[k] += h
            predictor.set_material(perturbed)
            plus = float(np.sum(predictor.forward().numpy().astype(np.float64) ** 2))
            perturbed = base.copy()
            perturbed[k] -= h
            predictor.set_material(perturbed)
            minus = float(np.sum(predictor.forward().numpy().astype(np.float64) ** 2))
            numeric[k] = (plus - minus) / (2.0 * h)
            predictor.set_material(base)

        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-30)
        self.assertLess(rel, 1.0e-2)

    def test_surround_predictor_reproduces_core_predict(self):
        """Match core.predict on a trial whose indenter drives a patch of a whole relaxing bed.

        Exercises the surround branch of core.predict -- the passive whole-midsole
        relaxation with a self-consistent Maxwell overstress -- which the
        differentiable predictor runs with the shared runtime kernels.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        trial = _synthetic_surround_trial()
        predicted = inverse_id.DifferentiableTrial(trial, material, device=device).forward().numpy()
        reference = core.predict(trial, material)
        peak = max(float(np.max(np.abs(reference))), 1.0e-9)
        self.assertLess(float(np.max(np.abs(predicted - reference))) / peak, 1.0e-3)

    def test_surround_material_gradient_matches_finite_difference(self):
        """Differentiate the relaxing-surround reaction w.r.t. the material and match central differences.

        The surround is an implicit quasi-static solve, so the gradient comes from
        the adjoint fixed point of the converged bed rather than from unrolling the
        relaxation. This checks that adjoint against a central finite difference of
        the forward force on a bed small enough to be fully relaxed, and asserts
        that the adjoint iteration reported convergence instead of hitting its cap.
        The material-pinned shear layer only reaches the summed load through the
        relaxation (the pairwise flux cancels over the bed), so the part of the
        ``g_eq`` gradient it carries exists only if the adjoint is right.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        trial = _synthetic_surround_trial()
        predictor = inverse_id.DifferentiableTrial(trial, material, device=device)
        force = predictor.forward().numpy()
        weight = 1.0 / max(float(np.max(force)), 1.0e-9) ** 2
        target = wp.zeros(predictor.frame_count, dtype=wp.float32, device=device)  # loss = weight * sum(force^2)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        predictor.zero_grad()
        predictor.accumulate_gradient(target, weight, loss)
        analytic = predictor.material_params.grad.numpy().copy()
        self.assertTrue(predictor.adjoint_converged)

        base = predictor.material_params.numpy().copy()
        numeric = np.empty_like(base)
        for k in range(len(base)):
            h = max(abs(base[k]) * 1.0e-3, 1.0e-6)
            values = []
            for sign in (1.0, -1.0):
                perturbed = base.copy()
                perturbed[k] += sign * h
                predictor.set_material(perturbed)
                values.append(weight * float(np.sum(predictor.forward().numpy().astype(np.float64) ** 2)))
            numeric[k] = (values[0] - values[1]) / (2.0 * h)
        predictor.set_material(base)

        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-30)
        self.assertLess(rel, 1.0e-2)

    @unittest.skipUnless(Path(MANIFEST).exists(), "requires the DigitalInstron dataset")
    def test_reproduces_core_predict_on_measured_trials(self):
        """Reproduce core.predict on both measured fixtures (spherical punch and shoe last)."""
        device = wp.get_preferred_device()
        trials, material = self._load_measured_trials()
        for trial in trials:
            predicted = inverse_id.DifferentiableTrial(trial, material, device=device).forward().numpy()
            reference = core.predict(trial, material)
            peak = max(float(np.max(np.abs(reference))), 1.0e-9)
            self.assertLess(float(np.max(np.abs(predicted - reference))) / peak, 1.0e-3)

    @unittest.skipUnless(Path(MANIFEST).exists(), "requires the DigitalInstron dataset")
    def test_joint_fit_reproduces_shipped_calibration(self):
        """Hold the shipped calibration in place under joint gradient fitting to the measured trials.

        Starting from the shipped material (scale = 1), the differentiable joint
        fit must not improve materially on it and must keep reproducing the
        production forward path. That cross-validates the exact gradients -- the
        implicit whole-bed surround adjoint included -- against the scipy fit.

        The fit scores the *identification's* residual, hence
        ``shape_residuals=True``: core._trial_residual weights the force residual
        together with a hysteresis-loop-area and a peak-force residual, and the
        pure force MSE is nearly flat along the documented stiffness/overstress
        valley.

        **No parameter is pinned to a box, and that is a property of the
        objective, not a weakened bound.** Measured here on the two-term material
        over 30 Adam iterations at lr 0.01: the loss moves from 1.82258 to
        1.80626 (-0.9%) while the fitted scale reaches
        ``[1.0330, 1.0859, 1.0056, 1.0268, 0.9615]``, so the first-term exponent
        drifts 8.6% for under 1% of loss. A tight box on an exponent would
        therefore measure the conditioning of the valley, not the calibration. So
        the test pins the calibration through the objective and through the
        predictions:

        * the loss cannot be improved by more than 5%,
        * the loss at the *fitted* scale is within 1% of the loss at scale 1, so
          the optimizer only moved along directions the objective cannot resolve,
        * the equilibrium modulus and the overstress ratio, which the objective
          *does* resolve, stay inside 5% (measured 3.3% and 0.6%),
        * and the per-trial force RMS the differentiable path reports equals
          core.metrics evaluated with core.predict **at the same fitted
          material** to 1e-4 (measured 6.3e-7 and 2.9e-7). Comparing it against
          the shipped material instead would measure how far the Adam fit walked,
          not whether the two paths agree.

        The two fixtures still pull in nearly opposite directions. Per-fixture
        adjoints of this objective in scale space at the shipped material:

        * rearfoot_140ms ``[-0.855, 0.350, -5.006, -2.641, -1.167]``
        * fullfoot_185ms ``[0.828, -0.731, 4.970, 2.653, 1.203]``

        Their sum ``[-0.026, -0.381, -0.036, 0.012, 0.036]`` is 6.6% of either
        fixture's gradient on a loss of 1.8225. One shared material is still in
        genuine tension between the fixtures; the difference from the single-term
        law is that the compromise now satisfies both fixtures' gates rather than
        neither. This test pins where that tension settles rather than pretending
        it is a sharp minimum.
        """
        device = wp.get_preferred_device()
        trials, material = self._load_measured_trials()
        result = inverse_id.fit_material_to_trials(
            trials, material, iterations=30, learning_rate=0.01, shape_residuals=True, device=device
        )
        start_loss = float(result.loss_history[0])
        self.assertLess(float(result.loss_history[-1]), 1.05 * start_loss)

        # Loss at the fitted scale, evaluated with the identification's own
        # objective: one iteration reports the loss of the scale it starts from.
        settled = inverse_id.fit_material_to_trials(
            trials,
            material,
            scale0=result.scale,
            iterations=1,
            learning_rate=0.0,
            shape_residuals=True,
            device=device,
        )
        self.assertLess(abs(float(settled.loss_history[0]) - start_loss) / start_loss, 0.01)

        resolved = result.scale[[inverse_id.MAT_G_EQ, inverse_id.MAT_OVERSTRESS]]
        self.assertTrue(np.all(np.abs(resolved - 1.0) < 0.05))
        for trial in trials:
            expected = core.metrics(trial.force_n, core.predict(trial, result.material), trial.displacement_m)[
                "force_rmse_relative"
            ]
            self.assertAlmostEqual(result.rms_relative[trial.name], expected, delta=1.0e-4)


class TestDifferentiableGaitScenarios(unittest.TestCase):
    @staticmethod
    def _press_driver(device, nsteps=160, config=None):
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        center = geo.uv_m.mean(axis=0)
        press_z = float(geo.z_free_m.mean()) - 0.02  # penetrate the ground plane for continuous contact
        pose = np.array([center[0], center[1], press_z, 0.0, 0.0, 0.0, 1.0], np.float32)
        targets = np.tile(pose, (nsteps, 1))
        velocities = np.zeros((nsteps, 6), np.float32)
        dt = (1.0 / 60.0) / 128.0
        return scenarios_diff.DifferentiableAttached(
            geo, material, targets, velocities, dt, config=config, device=device
        )

    def test_stride_reproduces_shipped_forward_model(self):
        """Reproduce the shipped MidsoleFoundation GRF over a kinematic heel-to-toe stride to float32 noise.

        Both models now relax the same passive surround and add the shear-layer
        flux unclamped, so the per-column terms are kilonewton-sized and cancel
        against each other in the summed reaction. Two float32 atomic sums of that
        cancelling set differ by their summation order alone, which is a relative
        error of about 1e-5 -- hence a relative bound instead of the previous
        absolute milli-newton one, which measured rounding rather than mechanics.
        """
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        stride = scenarios_diff.DifferentiableStride(geo, material, device=device)
        self.assertGreater(stride.foundation.free_column_count, 0)  # the surround is really there
        diff = stride.forward().numpy()
        reference = stride.reference_grf()
        peak = float(np.max(reference))
        self.assertGreater(peak, 100.0)  # the stride actually loads the bed
        self.assertLess(float(np.max(np.abs(diff - reference))) / peak, 1.0e-4)

    def test_stride_impulse_gradient_matches_finite_difference(self):
        """Differentiate the kinematic stride GRF impulse w.r.t. the equilibrium modulus and match a central difference."""
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        stride = scenarios_diff.DifferentiableStride(geo, material, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        stride.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            grf = stride.forward()
            wp.launch(scenarios_diff._reduce_sum, dim=stride.substep_count, inputs=[grf, loss], device=device)
        tape.backward(loss)
        analytic = float(stride.material_params.grad.numpy()[0])

        tape.zero()
        stride.zero_grad()
        base = stride.material_params.numpy().copy()
        h = base[0] * 1.0e-3

        def impulse(g_eq):
            perturbed = base.copy()
            perturbed[0] = g_eq
            stride.material_params.assign(perturbed.astype(np.float32))
            return float(stride.forward().numpy().astype(np.float64).sum())

        numeric = (impulse(base[0] + h) - impulse(base[0] - h)) / (2.0 * h)
        stride.material_params.assign(base.astype(np.float32))
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-30), 1.0e-2)

    def test_attached_press_gradient_matches_finite_difference(self):
        """Differentiate the fully dynamic press GRF impulse w.r.t. the equilibrium modulus under continuous contact.

        Drives the shoe with a constant target pose so the contact patch stays
        engaged for the whole rollout; the gradient through the PD upper, the
        semi-implicit solver, and shared bristle foundation matches a central
        difference away from contact and bristle branch transitions.
        """
        device = wp.get_preferred_device()
        driver = self._press_driver(device, nsteps=160)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        driver.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            grf = driver.forward()
            wp.launch(scenarios_diff._reduce_sum, dim=driver.substep_count, inputs=[grf, loss], device=device)
        tape.backward(loss)
        analytic = float(driver.material_params.grad.numpy()[0])

        tape.zero()
        driver.zero_grad()
        base = driver.material_params.numpy().copy()
        h = base[0] * 1.0e-3

        def impulse(g_eq):
            perturbed = base.copy()
            perturbed[0] = g_eq
            driver.material_params.assign(perturbed.astype(np.float32))
            return float(driver.forward().numpy().astype(np.float64).sum())

        numeric = (impulse(base[0] + h) - impulse(base[0] - h)) / (2.0 * h)
        driver.material_params.assign(base.astype(np.float32))
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-30), 2.0e-2)

    def test_attached_uses_external_ground_with_supplied_config(self):
        """Use the fixed external plane without discarding supplied contact settings."""
        config = FoundationConfig(normal_damping=3.25, friction_stiffness=6000.0, friction=0.8, mu=0.4)
        driver = self._press_driver(wp.get_preferred_device(), nsteps=32, config=config)
        self.assertIsNone(config.ground_height_m)
        self.assertEqual(driver.config.ground_height_m, 0.0)
        self.assertEqual(driver.config.normal_damping, config.normal_damping)
        self.assertEqual(driver.config.friction_stiffness, config.friction_stiffness)
        self.assertEqual(driver.config.mu, config.mu)
        self.assertIs(driver.foundation.applied_force, driver.foundation.ground_force)
        force = driver.forward().numpy()
        reference = driver.reference_grf()
        np.testing.assert_allclose(force, reference, rtol=2.0e-5, atol=1.0e-3)
        with self.assertRaisesRegex(ValueError, "ground plane is fixed at zero"):
            self._press_driver(wp.get_preferred_device(), nsteps=1, config=FoundationConfig(ground_height_m=0.1))

    def test_attached_forward_produces_valid_grf(self):
        """A fully dynamic attached press yields a finite, non-negative GRF with active contact."""
        device = wp.get_preferred_device()
        driver = self._press_driver(device, nsteps=96)
        self.assertEqual(driver.foundation.ground_height_m, 0.0)
        self.assertIs(driver.foundation.applied_force, driver.foundation.ground_force)
        grf = driver.forward().numpy()
        self.assertTrue(np.all(np.isfinite(grf)))
        self.assertGreaterEqual(float(np.min(grf)), 0.0)
        self.assertGreater(float(np.max(grf)), 100.0)


class TestDifferentiableFriction(unittest.TestCase):
    def test_slide_friction_gradient_matches_finite_difference(self):
        """Differentiate the lateral drag impulse of a constant slide w.r.t. the friction coefficient.

        Drives the foam bed at a fixed penetration and constant lateral speed so
        the contact patch stays engaged and the tangential velocity never crosses
        zero; the gradient of the accumulated drag with respect to ``mu`` then
        matches a central difference.
        """
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        slide = scenarios_diff.DifferentiableSlide(
            geo, material, depth_m=0.012, slide_speed_m_s=0.25, substeps=32, device=device
        )
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        slide.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            shear = slide.forward()
            wp.launch(scenarios_diff._drag_impulse, dim=slide.substep_count, inputs=[shear, loss], device=device)
        tape.backward(loss)
        analytic = float(slide.friction_params.grad.numpy()[0])

        tape.zero()
        slide.zero_grad()
        base = float(slide.friction_params.numpy()[0])
        h = base * 1.0e-3

        def drag(mu):
            slide.friction_params.assign(np.array([mu], np.float32))
            return -float(slide.forward().numpy()[:, 0].astype(np.float64).sum())

        numeric = (drag(base + h) - drag(base - h)) / (2.0 * h)
        slide.friction_params.assign(np.array([base], np.float32))
        self.assertGreater(analytic, 0.0)  # more friction => more drag
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-30), 5.0e-3)

    def test_slide_recovers_friction_coefficient(self):
        """Recover the Coulomb friction coefficient from a lateral-force target using the analytic gradient.

        The bristle drag is piecewise linear in ``mu`` at fixed kinematics.
        Re-evaluate its gradient as the stick/slip active set changes instead of
        assuming the obsolete smooth surrogate's globally linear response.
        """
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        slide = scenarios_diff.DifferentiableSlide(
            geo, material, depth_m=0.012, slide_speed_m_s=0.25, substeps=32, device=device
        )
        mu_ref = float(slide.friction_params.numpy()[0])
        target = -float(slide.forward().numpy()[:, 0].astype(np.float64).sum())

        slide.friction_params.assign(np.array([mu_ref * 0.4], np.float32))
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        for _ in range(4):
            slide.zero_grad()
            loss.zero_()
            tape = wp.Tape()
            with tape:
                shear = slide.forward()
                wp.launch(scenarios_diff._drag_impulse, dim=slide.substep_count, inputs=[shear, loss], device=device)
            tape.backward(loss)
            sensitivity = float(slide.friction_params.grad.numpy()[0])
            self.assertGreater(sensitivity, 0.0)
            drag_guess = float(loss.numpy()[0])
            recovered = float(slide.friction_params.numpy()[0]) + (target - drag_guess) / sensitivity
            slide.friction_params.assign(np.array([recovered], np.float32))
        self.assertLess(abs(recovered - mu_ref) / mu_ref, 5.0e-3)

    def test_attached_records_lateral_shear(self):
        """The dynamic attached rollout records a finite, non-trivial lateral shear on the tape."""
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        com_z = float(geo.z_free_m.mean())
        poses, velocities, dt = scenarios_diff.stride_trajectory(
            geo,
            com_z,
            period_s=0.15,
            peak_depth_m=0.03,
            pitch_deg=8.0,
            roll_fraction=0.15,
            frame_dt=1.0 / 60.0,
            substeps=64,
            with_velocity=True,
        )
        attached = scenarios_diff.DifferentiableAttached(geo, material, poses, velocities, dt, device=device)
        attached.forward()
        shear = attached.shear.numpy()
        self.assertEqual(shear.shape, (attached.substep_count, 2))
        self.assertTrue(np.all(np.isfinite(shear)))
        self.assertGreater(float(np.max(np.abs(shear))), 1.0)
        self.assertEqual(float(attached.friction_params.numpy()[0]), attached.config.mu)


@wp.kernel
def _column_drag(force: wp.array[wp.vec3], loss: wp.array[float]):
    """Accumulate streamwise drag from tape-safe per-column force history."""
    wp.atomic_add(loss, 0, -force[wp.tid()][0])


class TestSharedBristleParity(unittest.TestCase):
    """Pin the tape-safe adapter to live contact state and piecewise derivatives."""

    def _make(
        self, device, plane, speed=0.5, kt=2.0e4, release_dwell=0.004, height_offset=0.0007, friction_model="maxwell"
    ):
        """Build a nonuniform bed with load, flight, re-entry and nonzero Poisson ratio."""
        n = 10
        material = runtime.ShoeMaterial(
            instantaneous_shear_modulus_pa=8.0e4,
            hyperfoam_exponent=3.0,
            equilibrium_fraction=0.65,
            pasternak_n_per_m=0.0,
            effective_poisson_ratio=0.1,
            maxwell_relaxation_time_s=0.017,
        )
        cfg = FoundationConfig(
            mu=0.55,
            normal_damping=0.2,
            friction_stiffness=kt,
            friction=0.7,
            friction_release_dwell_s=release_dwell,
            ground_height_m=0.0 if plane else None,
            friction_model=friction_model,
        )
        rest = np.array([0.02, 0.025], np.float32)
        anchor = np.array([[-0.02, 0.01, 0], [0.015, -0.01, 0]], np.float32)
        if not plane:
            anchor[:, 2] = rest
        anchor[1, 2] += height_offset
        zfree = np.zeros(2, np.float32) if plane else rest
        com = wp.array([wp.vec3(0, 0, 0.01)], dtype=wp.vec3, device=device)
        args = (
            anchor,
            zfree,
            rest,
            np.array([2.5e-5, 3.5e-5], np.float32),
            np.array([[1, -1, -1, -1], [0, -1, -1, -1]], np.int32),
            0.005,
            material,
            0,
            com,
        )
        live = runtime.MidsoleFoundation(*args, config=cfg, device=device)
        diff = DifferentiableMidsoleFoundation(*args, num_substeps=n, config=cfg, device=device)
        states = []
        for t in range(n):
            z = -0.003
            if t in (4, 5):
                z = 0.001
            q = np.array([[speed * t * 0.001, 0, z, 0, 0, 0, 1]], np.float32)
            vel = np.array([[speed, 0, 0, 0.0, 0.2, 0.0]], np.float32)
            states.append(
                scenarios_diff._State(
                    body_q=wp.array(q, dtype=wp.transform, device=device, requires_grad=True),
                    body_qd=wp.array(vel, dtype=wp.spatial_vector, device=device, requires_grad=True),
                    body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device, requires_grad=True),
                )
            )
        return live, diff, states

    def test_forward_parity(self):
        """Match live wrench, material state and bristle history on both devices."""
        for device in ["cpu", *wp.get_cuda_devices()]:
            for plane in (False, True):
                live, diff, states = self._make(device, plane)
                for t, s in enumerate(states):
                    live.apply(s, 0.001)
                    expected = s.body_f.numpy()
                    s.body_f.zero_()
                    diff.apply(s, t, 0.001)
                    np.testing.assert_allclose(s.body_f.numpy(), expected, rtol=2e-6, atol=1e-6)
                    np.testing.assert_allclose(
                        diff.column_force[t].numpy(), live.column_force.numpy(), rtol=2e-6, atol=1e-6
                    )
                    if plane:
                        self.assertIs(diff.applied_force, diff.ground_force)
                        np.testing.assert_allclose(
                            diff.ground_force[t].numpy(), live.ground_force.numpy(), rtol=2e-6, atol=1e-6
                        )
                        if t == 0:
                            self.assertGreater(
                                float(np.max(np.abs(diff.column_force[t].numpy() - diff.ground_force[t].numpy()))),
                                0.1,
                            )
                    else:
                        self.assertIs(diff.applied_force, diff.column_force)
                    np.testing.assert_allclose(diff.q_state[t].numpy(), live.q_state.numpy(), rtol=2e-6, atol=1e-5)
                    np.testing.assert_allclose(
                        diff.tangent_anchor[t].numpy(), live.tangent_anchor.numpy(), rtol=2e-6, atol=1e-8
                    )
                    np.testing.assert_array_equal(diff.tangent_stuck[t].numpy(), live.tangent_stuck.numpy())
                    np.testing.assert_allclose(
                        diff.tangent_dwell[t].numpy(), live.tangent_dwell.numpy(), rtol=0, atol=1e-9
                    )

    def test_stick_and_release_history(self):
        """Match sticking, delayed release and fresh re-entry with separate tape history."""
        for device in ["cpu", *wp.get_cuda_devices()]:
            for plane in (False, True):
                live, diff, states = self._make(
                    device,
                    plane,
                    speed=0.0001,
                    kt=100.0,
                    release_dwell=0.001,
                    height_offset=0.00001,
                    friction_model="legacy",
                )
                for t, state in enumerate(states):
                    live.apply(state, 0.001)
                    expected = state.body_f.numpy()
                    state.body_f.zero_()
                    diff.apply(state, t, 0.001)
                    np.testing.assert_allclose(state.body_f.numpy(), expected, rtol=2e-6, atol=1e-6)
                    np.testing.assert_array_equal(diff.tangent_stuck[t].numpy(), live.tangent_stuck.numpy())
                    np.testing.assert_allclose(
                        diff.tangent_anchor[t].numpy(), live.tangent_anchor.numpy(), rtol=2e-6, atol=1e-8
                    )
                np.testing.assert_array_equal(diff.tangent_anchor[3].numpy(), diff.tangent_anchor[0].numpy())
                np.testing.assert_array_equal(diff.tangent_stuck[4].numpy(), np.ones(2, np.int32))
                np.testing.assert_array_equal(diff.tangent_stuck[5].numpy(), np.zeros(2, np.int32))
                np.testing.assert_array_equal(diff.tangent_stuck[6].numpy(), np.ones(2, np.int32))

    def test_maxwell_state_parity(self):
        """Retain matching default Maxwell elastic and branch-force histories on Tape."""
        for device in ["cpu", *wp.get_cuda_devices()]:
            live, diff, states = self._make(device, True, release_dwell=0.001)
            self.assertEqual(live.config.friction_model, "maxwell")
            for t, state in enumerate(states):
                live.apply(state, 0.001)
                state.body_f.zero_()
                diff.apply(state, t, 0.001)
                np.testing.assert_allclose(
                    diff.tangent_deflection[t].numpy(), live.tangent_deflection.numpy(), rtol=2e-6, atol=1e-8
                )
                np.testing.assert_allclose(
                    diff.tangent_maxwell_force[t].numpy(), live.tangent_maxwell_force.numpy(), rtol=2e-6, atol=1e-7
                )
            np.testing.assert_array_equal(diff.tangent_maxwell_force[5].numpy(), np.zeros((2, 2)))

    def test_mu_gradient(self):
        """Match multi-step piecewise Coulomb coefficient gradients to central differences."""
        for device in ["cpu", *wp.get_cuda_devices()]:
            _live, diff, states = self._make(device, True)
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

            def forward(loss=loss, states=states, diff=diff, device=device):
                loss.zero_()
                for t, s in enumerate(states):
                    s.body_f.zero_()
                    diff.apply(s, t, 0.001)
                    wp.launch(_column_drag, dim=2, inputs=[diff.applied_force[t], loss], device=device)
                return float(loss.numpy()[0])

            tape = wp.Tape()
            with tape:
                forward()
            tape.backward(loss)
            analytic = float(diff.friction_params.grad.numpy()[0])
            mu = 0.55
            h = mu * 1e-3
            diff.friction_params.assign(np.array([mu + h], np.float32))
            hi = forward()
            diff.friction_params.assign(np.array([mu - h], np.float32))
            lo = forward()
            numeric = (hi - lo) / (2 * h)
            self.assertGreater(analytic, 0)
            self.assertLess(abs(analytic - numeric) / abs(numeric), 0.003)


class TestSharedContactMechanics(unittest.TestCase):
    """The differentiable path must be a transcription of the shipped forward model, not a second model."""

    def test_surround_balance_matches_runtime(self):
        """Pin the differentiable surround balance to the runtime balance over a spread of column states.

        :func:`~projects.digital_instron_v2.dynamics_diff._surround_balance_diff`
        exists only because Warp accumulates adjoints into arrays, so a
        ``requires_grad`` material cannot travel inside the by-value
        ``FoundationParams`` struct the runtime uses. Nothing else about it may
        differ, so both are evaluated on the same free and carrier-bonded column
        states and must agree to float32 precision.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        params = _balance_params(material)
        rng = np.random.default_rng(0)
        count = 256
        thickness = np.full(count, 0.03, np.float32)
        c = (rng.random(count) * 0.02).astype(np.float32)
        rigid = (rng.random(count) * 0.02).astype(np.float32)
        pull = ((rng.random(count) - 0.5) * 40.0).astype(np.float32)
        # Summed face coefficients of zero to four neighbours, on the scale the
        # material-pinned rule mu_eq * t gives a 30 mm column.
        face = float(material.coupling_n_per_m(0.03))
        coupling_sum = (rng.integers(0, 5, count) * face).astype(np.float32)
        base = ((rng.random(count) - 0.2) * 5.0e3).astype(np.float32)
        material_params = wp.array(_material_vector(material), dtype=wp.float32, device=device)
        args = [wp.array(a, dtype=wp.float32, device=device) for a in (c, rigid, pull, coupling_sum, thickness, base)]
        out_runtime = wp.zeros(count, dtype=wp.float32, device=device)
        out_diff = wp.zeros(count, dtype=wp.float32, device=device)

        for gain, bond in ((0.0, 0), (0.38, 1)):  # the fit's frozen overstress, and a shod carrier
            common = [*args, float(gain), params, 2.5e-5, 200.0, 0.9, 1.0, int(bond)]
            wp.launch(_runtime_balance, dim=count, inputs=[*common, out_runtime], device=device)
            wp.launch(_diff_balance, dim=count, inputs=[*common, material_params, out_diff], device=device)
            np.testing.assert_allclose(out_diff.numpy(), out_runtime.numpy(), rtol=1.0e-6, atol=1.0e-9)

    def test_pasternak_flux_cancels_over_a_free_edged_bed(self):
        """Verify the pairwise shear flux sums to machine zero over a free-edged bed.

        The material-pinned face coefficient is symmetric, ``k_ij = k_ji``, so the
        two halves of every face term cancel exactly and the shear layer can only
        move load between columns, never create it. That guarantee is what lets
        the summed unilateral ground reaction be reported as the whole applied
        load. It is checked on an irregular bed -- varying rest thickness, a
        random compression field, and free outer edges -- and on the uniform field
        a flat platen imposes, where every face difference must vanish outright.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        # The shear layer follows the series modulus, the sum over both terms.
        mu_eq = float(material.equilibrium_shear_modulus_pa)
        nx, ny = 9, 7
        count = nx * ny
        rng = np.random.default_rng(3)
        neighbors = wp.array(_bed_neighbors(nx, ny), dtype=wp.int32, device=device)
        rest_len = wp.array((0.02 + 0.02 * rng.random(count)).astype(np.float32), dtype=wp.float32, device=device)
        flux = wp.zeros(count, dtype=wp.float32, device=device)

        compression = wp.array((0.012 * rng.random(count)).astype(np.float32), dtype=wp.float32, device=device)
        wp.launch(_bed_flux, dim=count, inputs=[compression, rest_len, neighbors, mu_eq, flux], device=device)
        values = flux.numpy().astype(np.float64)
        moved = float(np.sum(np.abs(values)))
        self.assertGreater(moved, 1.0)  # the layer really is moving load around
        self.assertLess(abs(float(np.sum(values))) / moved, 1.0e-5)  # but creates none of it

        uniform = wp.array(np.full(count, 0.006, np.float32), dtype=wp.float32, device=device)
        wp.launch(_bed_flux, dim=count, inputs=[uniform, rest_len, neighbors, mu_eq, flux], device=device)
        np.testing.assert_array_equal(flux.numpy(), np.zeros(count, np.float32))

    def test_stride_surround_carries_load_under_untouched_foam(self):
        """Verify the stride's untouched midsole really relaxes into contact instead of riding along rigidly.

        The shoe last drives only its own footprint, so every remaining column is
        held at zero compression by its carrier anchor. Only the passive surround
        can push it into the ground, and the load it then carries is the support
        the identification fits against.
        """
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        stride = scenarios_diff.DifferentiableStride(geo, material, device=device)
        free = ~np.asarray(geo.driven, bool)
        self.assertGreater(int(free.sum()), 0)

        stride.forward()
        relaxed = np.array([c[-1].numpy()[free] for c in stride.foundation.surround_compression])
        self.assertGreater(float(relaxed.max()), 1.0e-4)  # the surround is pressed into the ground
        self.assertGreaterEqual(float(relaxed.min()), 0.0)  # and never pulls itself down

        pressure = np.array([p.numpy()[free] for p in stride.foundation.base_pressure])
        self.assertGreater(float(pressure.max()), 0.0)


@wp.kernel
def _bed_flux(
    compression: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    mu_eq: wp.float32,
    flux: wp.array[wp.float32],
):
    """Write the pairwise Pasternak shear flux of every column of a bed [N]."""
    i = wp.tid()
    flux[i] = dynamics_diff._pasternak_flux(i, compression, rest_len, neighbors, mu_eq)


@wp.kernel
def _runtime_balance(
    c: wp.array[wp.float32],
    rigid: wp.array[wp.float32],
    pull: wp.array[wp.float32],
    coupling_sum: wp.array[wp.float32],
    thickness: wp.array[wp.float32],
    overstress_base: wp.array[wp.float32],
    overstress_gain: wp.float32,
    params: runtime.FoundationParams,
    area: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
    out: wp.array[wp.float32],
):
    """Evaluate the shipped runtime surround balance for one sampled column state."""
    i = wp.tid()
    out[i] = runtime._surround_balance(
        c[i],
        rigid[i],
        pull[i],
        coupling_sum[i],
        thickness[i],
        overstress_base[i],
        overstress_gain,
        params,
        area,
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


@wp.kernel
def _diff_balance(
    c: wp.array[wp.float32],
    rigid: wp.array[wp.float32],
    pull: wp.array[wp.float32],
    coupling_sum: wp.array[wp.float32],
    thickness: wp.array[wp.float32],
    overstress_base: wp.array[wp.float32],
    overstress_gain: wp.float32,
    params: runtime.FoundationParams,
    area: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
    material_params: wp.array[wp.float32],
    out: wp.array[wp.float32],
):
    """Evaluate the differentiable surround balance for one sampled column state."""
    i = wp.tid()
    out[i] = dynamics_diff._surround_balance_diff(
        c[i],
        rigid[i],
        pull[i],
        coupling_sum[i],
        thickness[i],
        overstress_base[i],
        overstress_gain,
        material_params[dynamics_diff.MAT_G_EQ],
        material_params[dynamics_diff.MAT_ALPHA],
        material_params[dynamics_diff.MAT_G_EQ2],
        material_params[dynamics_diff.MAT_ALPHA2],
        params,
        area,
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


def _material_vector(material) -> np.ndarray:
    """Pack a material into the differentiable ``[g_eq, alpha, overstress, g_eq2, alpha2]`` vector."""
    return np.array(
        [
            material.instantaneous_shear_modulus_pa * material.equilibrium_fraction,
            material.hyperfoam_exponent,
            (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction,
            material.instantaneous_shear_modulus_2_pa * material.equilibrium_fraction,
            material.hyperfoam_exponent_2,
        ],
        np.float32,
    )


def _balance_params(material) -> runtime.FoundationParams:
    """Device-side constitutive constants for a surround-balance comparison."""
    poisson = core.EFFECTIVE_POISSON_RATIO
    params = runtime.FoundationParams()
    runtime.set_hyperfoam_series(params, material)
    params.beta = poisson / (1.0 - 2.0 * poisson)
    params.one_minus_two_poisson = 1.0 - 2.0 * poisson
    params.stretch_floor = 1.0e-3
    return params


if __name__ == "__main__":
    unittest.main()
