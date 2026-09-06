# OpenBoost v1

**A programmable boosting foundation for researchers and AI agents.**

The new design combines public algorithm components, ordinary Python recipes,
and explicit data, model state and CPU/CUDA execution. It targets the cost of
making a correct, reproducible algorithm change.

## Implementation status

The old production code has been retired. The current package is an empty
namespace awaiting v1 components; there is no training or prediction API yet.
Sprints 001 and 003–008 delivered independent data, tree, classification, ranking,
quantile, vector-leaf, positive-target, AFT, Normal, Formula, isolated-run and author-task references with 255 passing CPU tests.
Those checks are preparation for implementation, not evidence that v1 is complete.

The repository's `v1-sprints/` directory contains execution plans, verification
results and reflections. `planning/foundation-construction-design.md` defines
what gets built; the task and evaluation documents require every listed use case.

## Historical APIs

Use Git revision `50acfc6` to reproduce the retired implementation and its examples.
No compatibility layer is provided. Historical benchmarks remain available in
the repository and retain their original scope; they do not measure the new design.
