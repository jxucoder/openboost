# Installation

## Quick Install

=== "pip"

    ```bash
    pip install openboost
    ```

=== "uv"

    ```bash
    uv add openboost
    ```

=== "conda"

    ```bash
    # Coming soon
    conda install -c conda-forge openboost
    ```

## With GPU Support

For CUDA GPU acceleration:

=== "pip"

    ```bash
    pip install "openboost[cuda]"
    ```

=== "uv"

    ```bash
    uv add "openboost[cuda]"
    ```

## Optional Dependencies

| Extra | What it includes | Install |
|-------|-----------------|---------|
| `cuda` | CuPy for GPU acceleration | `pip install "openboost[cuda]"` |
| `sklearn` | scikit-learn integration | `pip install "openboost[sklearn]"` |
| `distributed` | Ray for multi-GPU training | `pip install "openboost[distributed]"` |
| `all` | Everything | `pip install "openboost[all]"` |

## Requirements

- Python 3.10+
- NumPy 1.24+
- Numba 0.60+

### For GPU Support

- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.0+ or 12.0+

## Verify Installation

```python
import openboost as ob

print(f"OpenBoost version: {ob.__version__}")
print(f"Backend: {ob.get_backend()}")  # "cuda" or "cpu"
print(f"GPU available: {ob.is_cuda()}")
```

## Development Installation

```bash
git clone https://github.com/jxucoder/openboost.git
cd openboost
uv sync --extra dev
```
