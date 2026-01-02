# Newton Project Context

## Project Overview
**Newton** is a GPU-accelerated physics simulation engine built upon NVIDIA Warp, targeting robotics and simulation researchers. It integrates MuJoCo Warp as its backend and emphasizes GPU computation, OpenUSD support, and differentiability.

The repository also contains two related projects:
*   **MyoConverter:** A tool for converting OpenSim musculoskeletal models to MuJoCo format.
*   **ProtoMotions 3:** A GPU-accelerated framework for simulated humanoids and reinforcement learning.

## 1. Newton (Main Project)

### 🛠 Setup & Dependencies
The project recommends using **[uv](https://docs.astral.sh/uv/)** for dependency management.

*   **Sync dependencies:**
    ```bash
    uv sync --extra examples --extra dev
    ```
*   **Update lockfile:**
    ```bash
    uv lock -U
    ```
2
### 🚀 Running Examples
Examples are located in `newton/examples/`.

*   **List examples:**
    ```bash
    uv run -m newton.examples
    ```
*   **Run a specific example:**
    ```bash
    uv run -m newton.examples basic_pendulum
    ```
*   **Common arguments:**
    *   `--viewer`: `gl` (default), `usd`, `rerun`, `null`.
    *   `--device`: `cpu`, `cuda:0`.

### 🧪 Testing
Tests are located in `newton/tests/`.

*   **Run all tests:**
    ```bash
    uv run --extra dev -m newton.tests
    ```
*   **Run tests needing PyTorch (RL inference):**
    ```bash
    uv run --extra dev --extra torch-cu12 -m newton.tests
    ```
*   **Run with coverage:**
    ```bash
    uv run --extra dev -m newton.tests --coverage --coverage-html htmlcov
    ```

### 🎨 Linting & Formatting
*   **Tools:** `ruff` (lint/format), `typos` (spellcheck).
*   **Run checks (via pre-commit):**
    ```bash
    uvx pre-commit run -a
    ```

### 📚 Documentation
*   **Build docs:**
    ```bash
    uv run --extra docs --extra sim sphinx-build -W -b html docs docs/_build/html
    ```
*   **Run doctests:**
    ```bash
    uv run --extra docs --extra sim sphinx-build -W -b doctest docs docs/_build/doctest
    ```

### 📊 Benchmarking
*   **Tool:** `asv` (airspeed velocity).
*   **Run benchmarks:**
    ```bash
    asv run --launch-method spawn main^!
    ```

---

## 2. MyoConverter

Located in `myoconverter/`. Converts OpenSim models to MuJoCo.

*   **Key Files:** `myoconverter/O2MPipeline.py` (Main entry point).
*   **Setup:** Recommends `conda` or `mamba` (see `myoconverter/conda_env.yml`) or Docker.
*   **Usage Example (Python):**
    ```python
    from myoconverter.O2MPipeline import O2MPipeline
    O2MPipeline(osim_file, geometry_folder, output_folder, **kwargs)
    ```

---

## 3. ProtoMotions

Located in `ProtoMotions/`. RL framework for simulated humanoids.

*   **Setup:**
    ```bash
    pip install -e .
    ```
    (See `requirements_*.txt` for specific environment needs like IsaacGym or Genesis).
*   **Key Capabilities:** Motion learning from AMASS, Multi-GPU training, Retargeting, Sim2Sim testing.

---

## 📂 Key Directory Structure

*   `newton/` - Core Python package source.
    *   `examples/` - Example scripts (basic, robot, cloth, ik, etc.).
    *   `tests/` - Test suite.
    *   `usd.py`, `ik.py`, `solvers.py` - Core modules.
*   `myoconverter/` - OpenSim to MuJoCo converter tool.
*   `ProtoMotions/` - RL and motion learning framework.
*   `docs/` - Sphinx documentation source.
*   `asv/` - Benchmarks.
*   `pyproject.toml` - Project configuration (dependencies, build, linting).
*   `uv.lock` - Dependency lockfile.
