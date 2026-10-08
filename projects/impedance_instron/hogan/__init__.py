# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generative variable-impedance runner identification and prediction.

Use :mod:`.runner` for causal joint-only dynamics, :mod:`.identify` for shared
offline identification, and :mod:`.generate` for frozen initial-condition runs.
These paths never use a measured reference or inverse-dynamics feedforward at
runtime.
"""
