# Scaling Training

OpenBoost has a supported full-dataset CPU/CUDA path, a supported sampling
option, and several experimental scaling primitives. Keep those categories
separate when choosing a training path or reporting a benchmark.

| Capability | Status | Important boundary |
|------------|--------|--------------------|
| Single-device full-data training | Supported | Dataset and histograms must fit in memory |
| GOSS sampling | Supported | Speed and quality are data-dependent |
| `fit_trees_batch` | Reference implementation | Shares bins; configurations fit sequentially |
| Mini-batch histogram helpers | Low-level primitive | Not integrated with `model.fit` |
| Memory-mapped binned arrays | Storage primitive | Not an out-of-core `model.fit` path |
| Distributed/multi-GPU | Experimental | No published parity or scaling artifact yet |

## GOSS Sampling

Gradient-based One-Side Sampling keeps high-gradient observations and samples
from the remainder on every boosting round:

```python
import openboost as ob

model = ob.GradientBoosting(
    n_trees=100,
    subsample_strategy="goss",
    goss_top_rate=0.2,
    goss_other_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)
```

With these rates, approximately 28% of observations participate in a round.
That arithmetic is not a speed or accuracy guarantee: compare GOSS with the
full-data path on fixed folds and seeds before using it.

## Single-GPU Scaling

Select CUDA explicitly so an unavailable GPU cannot silently turn a benchmark
into a CPU run:

```python
import openboost as ob

ob.set_backend("cuda")
model = ob.GradientBoosting(n_trees=100, random_state=42)
model.fit(X_train, y_train)
```

Report the OpenBoost commit, environment, GPU model, driver/CUDA versions,
warm-up policy, seeds, fit time, prediction time, peak memory, and CPU/CUDA
prediction parity. Use the scale-extension protocol in
`benchmarks/scoringbench/` for probabilistic benchmark work.

## Mini-Batch and Memory-Mapped Primitives

`MiniBatchIterator`, `accumulate_histograms_minibatch`,
`create_memmap_binned`, and `load_memmap_binned` are low-level building blocks.
The memmap layout is feature-major `(n_features, n_samples)`; it is not a raw
sample-major matrix accepted by high-level `model.fit`.

The `batch_size` model parameter is reserved. Passing a non-`None` value raises
`NotImplementedError` rather than pretending to perform mini-batch training.
Do not claim datasets larger than memory are supported end to end until a
training loop integrates these primitives and has correctness tests.

## Train Many Configurations

Bin a dataset once and fit a grid of configurations against the same target:

```python
import openboost as ob

X_binned = ob.array(X_train)
configs = ob.ConfigBatch.from_grid(
    max_depth=[4, 6, 8],
    reg_lambda=[0.1, 1.0],
    learning_rate=[0.05, 0.1],
    n_rounds=100,
)

trees_by_config = ob.fit_trees_batch(
    X_binned,
    configs=configs,
    y=y_train,
    loss="mse",
)
```

The current implementation shares the binned input but fits configurations
sequentially. Treat it as a correctness reference, not fused GPU training.

## Experimental Multi-GPU Path

The Ray multi-GPU path is available for development experiments:

```python
import openboost as ob

model = ob.GradientBoosting(n_trees=100, devices=[0, 1])
model.fit(X_train, y_train)
```

It does not support `sample_weight`, and the repository does not yet contain a
validated two-/four-GPU parity and scaling artifact. Do not use it for release
claims until exact single-device parity, repeated timings, peak memory, and
failure cases are published on real multi-GPU hardware.

## Evidence Gate

A scaling claim is ready only when the checked-in artifact records:

1. a frozen OpenBoost commit and dependency lock;
2. real datasets plus at least one controlled synthetic scaling curve;
3. CPU/CUDA prediction parity and task-quality metrics;
4. repeated fit/predict timings after an explicit warm-up policy;
5. hardware, drivers, thread counts, peak memory, seeds, and failures;
6. comparisons against maintained baselines under the same protocol.
