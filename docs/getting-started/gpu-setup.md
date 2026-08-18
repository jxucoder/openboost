# GPU Setup

OpenBoost uses CUDA for histogram building and tree construction.
NaturalBoost, FormulaBoost, and WeibullAFT share that tree path. Some
objective math still runs on the host (FormulaBoost GGN today; LogNormal /
digamma families). Trees are the expensive part at scale.

## Verify detection

```python
import openboost as ob

print(ob.get_backend())   # "cuda" or "cpu"
print(ob.is_cuda())       # True if a GPU is active
```

Install the extra first: `pip install --pre "openboost[cuda]"`.

## Pin a backend

```python
import openboost as ob

ob.set_backend("cpu")    # debug / comparison
ob.set_backend("cuda")

# Or:
# export OPENBOOST_BACKEND=cuda
```

`backend_context("cpu")` is a temporary switch that restores the previous
backend on exit.

## What the A100 numbers actually are

NaturalBoost vs NGBoost, heteroscedastic Normal, 80 features, 500 trees,
Modal A100. NGBoost has **no GPU implementation**, so this is GPU OpenBoost
against CPU NGBoost, which is the comparison that exists in the world.

| n_train | OpenBoost (A100) | NGBoost (CPU) | speedup |
|--------:|-----------------:|--------------:|--------:|
| 45K | 3.01s | 1414s | 470× |
| 90K | 2.21s | 2716s | **1229×** |
| 450K | 5.40s | skipped (hours) | n/a |
| 900K | 6.94s | skipped | n/a |

NLL is tied at every size that both ran. Full tables:
[Benchmarks](../benchmarks.md).

!!! tip "When GPU helps"
    Histogram trees win at tens of thousands of rows and up. Below ~5K
    samples, kernel launch overhead often matches CPU. Use `float32`
    features. Missing values and categoricals fall back from the
    GPU-native builder to the hybrid path (warning emitted).

On CPU, NaturalBoost and NGBoost are ~parity (0.8–1.3×). Do not quote the
A100 ratio as a CPU claim.

## FormulaBoost and WeibullAFT on GPU

Both use GPU trees. FormulaBoost's GGN (finite-difference Jacobian +
`K×K` solve) currently runs on the host; at 200K rows full GGN still beat
an XGBoost custom objective (17s vs 20s on A100). WeibullAFT's expected
Fisher step is cheap (`2×2` per row).

## Multi-GPU

```python
import openboost as ob

model = ob.GradientBoosting(n_trees=100, n_gpus=4)
model.fit(X, y)

model = ob.GradientBoosting(n_trees=100, devices=[0, 2])
model.fit(X, y)
```

Requires `pip install --pre "openboost[distributed]"` (Ray).
NaturalBoost / FormulaBoost / WeibullAFT currently train on one GPU.

## Requirements

- NVIDIA GPU, CUDA Compute Capability 3.5+
- CUDA Toolkit 11 or 12
- `numba-cuda>=0.23`

## Troubleshooting

**Training seems slow on GPU.** Features should be `float32`. Tiny datasets
do not amortize kernel launch. Confirm `ob.is_cuda()` is True.

**CUDA not detected.**

1. `nvidia-smi`
2. `python -c "from numba import cuda; print(list(cuda.gpus))"`
3. Reinstall `openboost[cuda]` against the CUDA version on the machine

**Trained on GPU, loading on CPU.** Saved models are backend-agnostic.

```python
model.save("model.joblib")
loaded = ob.NaturalBoostNormal.load("model.joblib")  # CPU or GPU
```
