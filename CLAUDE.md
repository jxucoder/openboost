# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenBoost is a GPU-native, all-Python gradient boosting library (~20K lines). It uses Numba JIT for CPU kernels and CuPy/numba-cuda for GPU acceleration. Designed as a research-friendly alternative to XGBoost/LightGBM with full Python source.

## Commands

```bash
# Environment (always use uv, never pip/conda/poetry)
uv sync                                         # Install/sync dependencies
uv sync --extra cuda                            # With GPU support
uv sync --extra dev                             # With dev tools (test + bench + sklearn + ruff)

# Testing
uv run pytest tests/ -v --tb=short              # All tests (CPU)
uv run pytest tests/test_core.py -v             # Single test file
uv run pytest tests/test_core.py::test_name -v  # Single test
OPENBOOST_BACKEND=cuda uv run pytest tests/     # GPU tests
OPENBOOST_BACKEND=cpu uv run pytest tests/      # Force CPU

# Linting
uv run ruff check src/openboost/               # Lint
uv run ruff check src/openboost/ --fix          # Autofix

# Docs
uv run mkdocs serve                             # Local docs server
uv run mkdocs build                             # Build docs

# Build
uv build                                        # Build wheel/sdist
```

## Architecture

### Layer Overview

```
Models (_models/)  →  Core (_core/)  →  Backends (_backends/)
    ↓                     ↓                    ↓
GradientBoosting     fit_tree()          _cpu.py (Numba JIT)
NaturalBoost         histograms          _cuda.py (CuPy kernels)
OpenBoostGAM         split finding
DART, LinearLeaf     growth strategies
```

### Data Layer (`_array.py`)
`BinnedArray` is the fundamental data structure — quantile-bins continuous features into uint8 (max 255 bins). Missing values encode as `MISSING_BIN = 255`. Native categorical feature support. All tree-building operates on binned data.

### Core (`_core/`)
- **`_tree.py`** — `fit_tree()`, `fit_tree_gpu_native()`, `fit_tree_symmetric()`: the main tree-fitting entry points
- **`_primitives.py`** — Low-level histogram building, split finding, sample partitioning
- **`_growth.py`** — Three growth strategies: `LevelWiseGrowth` (XGBoost-style), `LeafWiseGrowth` (LightGBM-style), `SymmetricGrowth` (CatBoost-style)

### Backend Dispatch (`_backends/`)
`get_backend()` / `set_backend()` switch between CPU and CUDA implementations. Same interface, different kernels. Control via `OPENBOOST_BACKEND` env var or `set_backend('cuda')`.

### Models (`_models/`)
- **`_boosting.py`** — `GradientBoosting`: main model class, owns the training loop
- **`_sklearn.py`** — sklearn-compatible wrappers (`OpenBoostRegressor`, `OpenBoostClassifier`)
- **`_distributional.py`** — `NaturalBoost`: distributional GBDT (natural gradient boosting)
- **`_dart.py`**, **`_linear_leaf.py`**, **`_gam.py`** — Specialized model variants

### Loss Functions (`_loss.py`)
50+ loss implementations. Each returns `(gradient, hessian)`. Custom losses are callables with signature `fn(pred, y) -> (grad, hess)`.

### Distributions (`_distributions.py`)
8 distributional families for NaturalBoost (Normal, LogNormal, Gamma, Poisson, StudentT, Tweedie, NegativeBinomial). Each implements `nll_grad_hess()` for natural gradient computation.

## Key Conventions

- **Python 3.10+** target. Ruff rules: E, F, I, UP, B, SIM (line length 100, E501 ignored).
- **uv only** for package management — never `pip install` or `conda`.
- All Numba-jitted functions use `@njit` or `@cuda.jit`. CPU kernels are in `_backends/_cpu.py`, CUDA in `_backends/_cuda.py`.
- Test environment variable `OPENBOOST_BACKEND=cpu` forces CPU backend in CI.
