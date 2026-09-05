# GPU Setup

OpenBoost uses CUDA for histogram building and tree construction.
NaturalBoost, FormulaBoost, and WeibullAFT share a tree-building path.
FormulaBoost's finite-difference Jacobian and GGN solve run on CPU, as does
WeibullAFT's expected-Fisher step. Normal and Poisson objectives have device
kernels for eligible configurations.

## Verify detection

Install with `uv add --prerelease=allow "openboost[cuda]"`. For a repository
checkout use `uv sync --extra cuda --extra dev`.

```python
import openboost as ob
print(ob.get_backend())
print(ob.is_cuda())
```

Detection confirms backend availability, not that every operation runs on GPU.
The native builder excludes missing values, categorical features, L1
regularization, and row/column sampling in the shared trainer. Such inputs
may select another tree path. Check the actual model and configuration before
interpreting timing as a fully device-resident fit.

## Pin a backend

```python
ob.set_backend("cpu")
ob.set_backend("cuda")
with ob.backend_context("cpu"):
    print(ob.get_backend())
```

Alternatively set `OPENBOOST_BACKEND=cpu` or `OPENBOOST_BACKEND=cuda`.
The backend is process-global; do not run mixed-backend fits concurrently in
one process. A context restores the previous backend on exit.

## Measure the relevant workload

GPU benefit depends on dataset shape, tree parameters, distribution, CUDA
stack, transfers, and JIT warm-up. A universal crossover size or speedup is
not established by backend detection.

1. Force the backend for each comparison.
2. Record first-use compilation separately from repeated warm timings.
3. Compare predictions and task metrics before runtime.
4. Measure end-to-end fit and prediction, peak memory, and failures.
5. Save raw results with source SHA, data/split, hardware, dependencies,
   actual execution path, and exact commands.

The ScoringBench integration under `benchmarks/scoringbench/` defines separate
official-quality and scale-extension protocols for probabilistic models.
Historical benchmark summaries do not replace reproducible raw artifacts.

## Experimental multi-GPU

The `distributed` extra installs Ray for experimental multi-GPU work.
Exact correctness and scaling evidence are still required before treating
that path as a supported production capability. NaturalBoost, FormulaBoost,
and WeibullAFT currently use one GPU. A single-GPU result does not validate
distributed training.

## Requirements and troubleshooting

- An NVIDIA GPU supported by the installed CUDA and Numba stack.
- A CUDA 12 runtime compatible with `cupy-cuda12x` from the CUDA extra.
- `numba-cuda>=0.23` and `cupy-cuda12x>=13` as specified by the package.

If CUDA is not detected, inspect `nvidia-smi` and run:

```bash
uv run python -c "from numba import cuda; print(cuda.is_available())"
```

If training is slow, use float32 features, account for compilation and data
transfers, and inspect which objective/tree path actually executes. Report
CPU and GPU resources separately when comparing different libraries.

Saved models store host tree state and support CPU inference. Verify a
prediction round trip for the model and feature types used in your workload:

```python
model.save("model.joblib")
with ob.backend_context("cpu"):
    loaded = ob.NaturalBoostNormal.load("model.joblib")
```
