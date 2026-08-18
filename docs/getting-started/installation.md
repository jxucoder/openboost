# Installation

OpenBoost 1.0 is a release candidate. Install with `--pre` until 1.0.0.

## Quick install

=== "pip"

    ```bash
    pip install --pre openboost
    ```

=== "uv"

    ```bash
    uv add --prerelease=allow openboost
    ```

`pip install openboost` without `--pre` still resolves the older stable
release.

## GPU support

Numba CUDA kernels for histogram trees. Requires an NVIDIA GPU.

=== "pip"

    ```bash
    pip install --pre "openboost[cuda]"
    ```

=== "uv"

    ```bash
    uv add --prerelease=allow "openboost[cuda]"
    ```

Then:

```python
import openboost as ob
print(ob.get_backend(), ob.is_cuda())   # "cuda" True when a GPU is visible
```

See [GPU setup](gpu-setup.md) for backend pinning, multi-GPU, and
troubleshooting.

## Optional extras

| Extra | What it includes | Install |
|-------|-----------------|---------|
| `cuda` | numba-cuda + CuPy for GPU trees | `pip install --pre "openboost[cuda]"` |
| `sklearn` | scikit-learn wrappers | `pip install --pre "openboost[sklearn]"` |
| `jax` | autodiff for custom distributions / formulas | `pip install --pre "openboost[jax]"` |
| `distributed` | Ray for multi-GPU | `pip install --pre "openboost[distributed]"` |
| `all` | Everything | `pip install --pre "openboost[all]"` |

Finite-difference Jacobians work without JAX. Install `jax` when you want
autodiff on a custom NLL.

## Requirements

- Python 3.10+
- NumPy 1.24+
- Numba 0.60+
- SciPy 1.10+

### For GPU

- NVIDIA GPU, CUDA Compute Capability 3.5+
- CUDA Toolkit 11 or 12
- `numba-cuda>=0.23`, `cupy-cuda12x>=13`

## Verify

```python
import openboost as ob

print(f"OpenBoost {ob.__version__}")
print(f"Backend: {ob.get_backend()}")   # "cuda" or "cpu"
print(f"GPU:     {ob.is_cuda()}")
```

## Development install

```bash
git clone https://github.com/jxucoder/openboost.git
cd openboost
uv sync --extra dev
# GPU kernels:
uv sync --extra cuda --extra dev
```
