# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bilateral variable-impedance runner: two legs, persistent shoes, and neural policies.

:mod:`.mechanics` describes both legs on a floating pelvis that carries only the
trunk, arms, and head. :mod:`.shoes` mirrors the shoe artifact for the right
foot. :mod:`.data` prepares continuous multi-stride references, :mod:`.env`
integrates batched CUDA worlds, and :mod:`.ppo` trains a neural impedance
policy. Measurements enter only rewards and scores, never policy inputs.
"""
