# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scope freeze regression tests for digital shoe normal kinematics and contact laws.

Validates that normal contact mechanics, Pasternak coupling, surround balance/relaxation,
material hyperfoam laws, and calibration remain byte-for-byte and AST-identical to the
pre-friction baseline. Also verifies that normal runtime output arrays match the
pinned pre-batching baseline values and remain bit-identical across all optional
friction adapter modes (bristle, implicit_bristle, regularized) under prescribed motion,
both with nominal carrier anchors and with an explicit ground plane on CPU and CUDA devices.
"""

import ast
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests.test_digital_shoe_worlds import (
    DT_S,
    MATERIAL,
    PRE_BATCHING_DIGESTS,
    PRE_BATCHING_REDUCTIONS,
    SPACING_M,
    STEPS,
    _build_bed,
    _carrier_motion,
    _digest,
)
from projects.digital_shoe import calibration, contact, material, runtime
from projects.digital_shoe.friction_adapter import FrictionAdapter
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, SurroundConfig

# Whole-file SHA-256 digests from baseline at outputs/friction_review/baseline/hashes.json
MATERIAL_FILE_SHA256 = "b35478d605e800560c041450b278fd63fbdef8f59561f47057caa90c9811b154"
CALIBRATION_FILE_SHA256 = "c449affc0fd5999c0911fd876df677a92cbd105d1bd8a795f11690a7a3485fd8"

# Normal function bodies from publication base 28c4e777 (before friction dispatch).
# Include extracted helper bodies, not only their public launch wrappers.
CONTACT_AST_SHA256 = {
    "_pasternak_coupling": "181574b24c8a0d6084f73bfae707b7491eb90e34980530269cef087d433c47b7",
    "_normal_reaction_function": "9da7681693cc503d29d8221ed4ca92517871fbe6bef1051ab5570a274131b0b2",
    "contact_wrench": "5a2dc2ed70a402a9be73479c478c82fec3a796bfffda418ed53d231e1149259b",
    "surround_balance": "089a1b586c8368e6f8b067262076c0d7132d7233bc0f78b8026181c4e84d6e38",
    "contact_kinematics": "89b9b61a84e1bde4cfb2720eb68e08d70eb2fc3715e9353d18a2692066b8c0f5",
    "pasternak_flux": "22c8435cc3ad4b80622c8785acddd709546531252a3942fc692cb40c5cee622f",
    "_surround_balance_pressures": "873c49320e134fb42da9bece8d1d9dc82743ac2fc9fcbdf1b78169f013490b81",
}

RUNTIME_AST_SHA256 = {
    "foundation_pressure": "d46ed6745528a932cdb4a90d8f491b9cca9a325dad07cc1ad41e371981fe0a3e",
    "_surround_balance": "c801de14ffa20d529896a9040376e0e303c44a8a0c42b310ac8f2382d5223e17",
    "_surround_sweep_cell": "50354222bc1367451113470a6d35188efdc5095c5c2e7a1877001a1a167d6daa",
    "surround_sweep": "9f35801ca9f71b37556fb64fe83f09053131dc0a67f573112cfa4374efd4bce8",
    "surround_relax": "ad1fd719252f4b33c377d32fcf9970018f3e581922d90fd4ae65be98f4defb23",
    "surround_write_free_top": "1db9aa4dcc2f4383986262f23530a6bb44270d5e7c1c5298c5dc0598c74949ee",
    "surround_seed_driven": "43e9874a1a3e1fac1d32e6cf54995c99d4e03d086d0835ae9d0d8ce9162e131c",
    "surround_update_max": "bbbf9bd4a5db1a1171f1832993c4d765ff957f33565167d1235ca7112d84cf50",
    "relax_surround": "f38749e54f5e593248ba2db6abaad5301d7bdf039ee3d93c20c125b9d6535153",
    "_foundation_pressure": "79eb7d1ad36ac7e6783c0201c520ffbd4c3a33a00d1949945499a00527c311f3",
    "_surround_write_free_top": "dd11a7cb30725af5a13baa7813283147e6239d571ff082391d5ce1f14c94bf82",
}

NORMAL_COLUMN_FIELDS = (
    "compression",
    "base_pressure",
    "z_free",
    "q_state",
    "peq_prev",
    "surround_compression",
    "surround_rate",
)


def _ast_sha256(source_text: str, func_name: str) -> str:
    """Return the SHA-256 digest of the AST dump for a specific top-level function."""
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:

            def canonical(value):
                if isinstance(value, ast.AST):
                    return {
                        "node": type(value).__name__,
                        "fields": {
                            name: canonical(child)
                            for name, child in ast.iter_fields(value)
                            if name != "type_params" or child
                        },
                    }
                if isinstance(value, list):
                    return [canonical(child) for child in value]
                return value

            dumped = json.dumps(canonical(node), sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    raise ValueError(f"Function '{func_name}' not found in AST")


def _test_devices() -> list:
    """Return available CPU and CUDA devices for testing."""
    devices = [wp.get_device("cpu")]
    if wp.is_cuda_available():
        devices.append(wp.get_device("cuda:0"))
    return devices


def _run_digital_shoe_simulation(
    adapter_mode: str | None = None,
    mobility_diag: float = 0.0,
    ground_height_m: float | None = None,
    carrier_lift_m: float = 0.0,
    device=None,
):
    """Step the single-carrier foundation under scripted motion with optional friction adapter.

    Args:
        adapter_mode: None for unmodified baseline foundation, or one of
            'bristle', 'implicit_bristle', 'regularized'.
        mobility_diag: Value along the diagonal of the 6x6 spatial mobility matrix.
        ground_height_m: Optional explicit ground plane height [m].
        carrier_lift_m: Vertical offset added to carrier height [m].
        device: Warp device.

    Returns:
        Tuple of (normal_history, pressed_history, max_comp_history, active_history, cop_history, normal_arrays).
    """
    if device is None:
        device = wp.get_device("cpu")
    bed = _build_bed()
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
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
        body,
        model.body_com,
        FoundationConfig(
            stretch_floor=0.05,
            normal_damping=5.0,
            friction_stiffness=1.0e4,
            friction=20.0,
            mu=0.6,
            ground_height_m=ground_height_m,
        ),
        device,
        SurroundConfig(driven=bed["driven"], max_strain=0.9, sweeps=3, carrier_bond=True),
        world_count=1,
    )

    if adapter_mode is not None:
        mob_mat = np.zeros((1, 6, 6), dtype=np.float32)
        if mobility_diag > 0.0:
            np.fill_diagonal(mob_mat[0], mobility_diag)
        mobility = wp.array(mob_mat, dtype=wp.spatial_matrix, device=device)
        FrictionAdapter(foundation, mobility, mode=adapter_mode, iterations=8)

    normal_history = []
    pressed_history = []
    max_comp_history = []
    active_history = []
    cop_history = []

    for step in range(STEPS):
        q, qd = _carrier_motion(0, step, DT_S)
        q[2] += carrier_lift_m
        state.body_q.assign(np.array([q], dtype=np.float32))
        state.body_qd.assign(np.array([qd], dtype=np.float32))
        state.clear_forces()
        foundation.apply(state, DT_S)

        normal_history.append(float(foundation.normal_force.numpy()[0]))
        pressed_history.append(float(foundation.pressed_force.numpy()[0]))
        max_comp_history.append(float(foundation.max_compression.numpy()[0]))
        active_history.append(int(foundation.active.numpy()[0]))
        cop_history.append(foundation.cop_moment.numpy()[0].copy())

    normal_arrays = {
        "compression": foundation.compression.numpy().copy(),
        "base_pressure": foundation.base_pressure.numpy().copy(),
        "z_free": foundation.z_free.numpy().copy(),
        "q_state": foundation.q_state.numpy().copy(),
        "peq_prev": foundation.peq_prev.numpy().copy(),
        "surround_compression": foundation.surround_compression.numpy().copy(),
        "surround_rate": foundation.surround_rate.numpy().copy(),
        "column_force_z": foundation.column_force.numpy()[:, 2].copy(),
        "column_pressed": foundation.column_pressed.numpy().copy(),
        "contact_point": foundation.contact_point.numpy().copy(),
    }
    if ground_height_m is not None:
        normal_arrays["ground_force_z"] = foundation.ground_force.numpy()[:, 2].copy()

    return normal_history, pressed_history, max_comp_history, active_history, cop_history, normal_arrays


class TestDigitalShoeFrictionScope(unittest.TestCase):
    """Freeze regressions for normal contact laws and verify unmodified baseline identity."""

    def test_material_file_matches_baseline_sha256(self):
        """Verify material.py matches the complete baseline SHA-256 digest."""
        path = Path(material.__file__)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, MATERIAL_FILE_SHA256)

    def test_calibration_file_matches_baseline_sha256(self):
        """Verify calibration.py matches the complete baseline SHA-256 digest."""
        path = Path(calibration.__file__)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, CALIBRATION_FILE_SHA256)

    def test_contact_normal_functions_match_baseline_ast(self):
        """Verify contact.py normal and surround functions match embedded baseline AST digests."""
        source = Path(contact.__file__).read_text()
        for func_name, expected_digest in CONTACT_AST_SHA256.items():
            with self.subTest(function=func_name):
                actual_digest = _ast_sha256(source, func_name)
                self.assertEqual(actual_digest, expected_digest)

    def test_runtime_normal_functions_match_baseline_ast(self):
        """Verify runtime.py normal pressure and surround functions match embedded baseline AST digests."""
        source = Path(runtime.__file__).read_text()
        for func_name, expected_digest in RUNTIME_AST_SHA256.items():
            with self.subTest(function=func_name):
                actual_digest = _ast_sha256(source, func_name)
                self.assertEqual(actual_digest, expected_digest)

    def test_runtime_normal_arrays_match_pre_batching_baseline(self):
        """Verify baseline simulation normal arrays and force reductions match pre-batching digests."""
        (
            normal_history,
            pressed_history,
            max_comp_history,
            active_history,
            _cop_history,
            normal_arrays,
        ) = _run_digital_shoe_simulation(adapter_mode=None)

        # Verify normal per-column arrays against pinned baseline digests
        for field in NORMAL_COLUMN_FIELDS:
            with self.subTest(field=field):
                arr = normal_arrays[field]
                self.assertEqual(_digest(arr), PRE_BATCHING_DIGESTS[field])

        # Verify normal order-independent reductions
        active_arr = np.array(active_history, dtype=np.int32).reshape(STEPS, 1)
        max_comp_arr = np.array(max_comp_history, dtype=np.float32).reshape(STEPS, 1)
        self.assertEqual(_digest(active_arr), PRE_BATCHING_DIGESTS["active"])
        self.assertEqual(_digest(max_comp_arr), PRE_BATCHING_DIGESTS["max_compression"])

        # Verify normal force histories match baseline reductions
        np.testing.assert_allclose(
            normal_history,
            PRE_BATCHING_REDUCTIONS["normal_force"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            pressed_history,
            PRE_BATCHING_REDUCTIONS["pressed_force"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_normal_arrays_identical_across_friction_adapter_modes(self):
        """Verify normal fields remain bit-identical across all friction adapter modes and mobilities."""
        cases = [
            ("bristle", 0.0),
            ("implicit_bristle", 0.0),
            ("implicit_bristle", 0.5),
            ("regularized", 0.0),
            ("regularized", 0.5),
        ]

        for device in _test_devices():
            with self.subTest(device=str(device)):
                # Baseline simulation without friction adapter
                (
                    base_norm,
                    base_pres,
                    base_maxc,
                    base_act,
                    base_cop,
                    base_arrs,
                ) = _run_digital_shoe_simulation(adapter_mode=None, device=device)

                for mode, mob in cases:
                    with self.subTest(mode=mode, mobility=mob, device=str(device)):
                        (
                            m_norm,
                            m_pres,
                            m_maxc,
                            m_act,
                            m_cop,
                            m_arrs,
                        ) = _run_digital_shoe_simulation(adapter_mode=mode, mobility_diag=mob, device=device)

                        # Step-by-step normal scalar reductions
                        np.testing.assert_array_equal(m_norm, base_norm)
                        np.testing.assert_array_equal(m_pres, base_pres)
                        np.testing.assert_array_equal(m_maxc, base_maxc)
                        np.testing.assert_array_equal(m_act, base_act)
                        np.testing.assert_array_equal(m_cop, base_cop)

                        # Per-column normal output arrays including contact_point
                        for field_name in base_arrs:
                            np.testing.assert_array_equal(
                                m_arrs[field_name],
                                base_arrs[field_name],
                                err_msg=f"Normal array '{field_name}' differed for mode={mode}, mob={mob} on {device}",
                            )

    def test_normal_arrays_identical_with_ground_plane(self):
        """Verify normal fields and ground reactions remain bit-identical with an explicit ground plane."""
        cases = [
            ("bristle", 0.0),
            ("implicit_bristle", 0.0),
            ("implicit_bristle", 0.5),
            ("regularized", 0.0),
            ("regularized", 0.5),
        ]

        for device in _test_devices():
            with self.subTest(device=str(device)):
                # Baseline simulation with ground plane contact (carrier penetrating plane at z=0.0)
                (
                    base_norm,
                    base_pres,
                    base_maxc,
                    base_act,
                    base_cop,
                    base_arrs,
                ) = _run_digital_shoe_simulation(
                    adapter_mode=None,
                    ground_height_m=0.0,
                    carrier_lift_m=-0.003,
                    device=device,
                )

                # Ensure non-trivial ground contact occurred
                self.assertGreater(np.max(base_arrs["ground_force_z"]), 0.0)
                self.assertGreater(np.sum(base_act), 0)

                for mode, mob in cases:
                    with self.subTest(mode=mode, mobility=mob, device=str(device)):
                        (
                            m_norm,
                            m_pres,
                            m_maxc,
                            m_act,
                            m_cop,
                            m_arrs,
                        ) = _run_digital_shoe_simulation(
                            adapter_mode=mode,
                            mobility_diag=mob,
                            ground_height_m=0.0,
                            carrier_lift_m=-0.003,
                            device=device,
                        )

                        # Step-by-step normal scalar reductions
                        np.testing.assert_array_equal(m_norm, base_norm)
                        np.testing.assert_array_equal(m_pres, base_pres)
                        np.testing.assert_array_equal(m_maxc, base_maxc)
                        np.testing.assert_array_equal(m_act, base_act)
                        np.testing.assert_array_equal(m_cop, base_cop)

                        # Per-column normal output arrays including ground_force_z and contact_point
                        for field_name in base_arrs:
                            np.testing.assert_array_equal(
                                m_arrs[field_name],
                                base_arrs[field_name],
                                err_msg=f"Ground normal array '{field_name}' differed for mode={mode}, mob={mob} on {device}",
                            )


if __name__ == "__main__":
    unittest.main()
