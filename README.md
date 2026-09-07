# OpenBoost

**A programmable boosting foundation for researchers and AI agents.**

OpenBoost v1 is being rebuilt around composable algorithm components, ordinary
Python recipes and explicit CPU/CUDA execution. The goal is to reduce the cost
of making a correct, reproducible algorithm change.

## Current state

This checkout is **under construction**. The retired implementation is not restored.
Initial public CPU components now provide owned numeric inputs, explicit problems,
run identity, immutable proposal/accept/reject state and mapped tree/constant ensemble
artifacts. See [B03 usage and boundaries](docs/v1/cpu-state.md).

[Categorical input and equality splits](docs/v1/categorical.md) support typed
train-only dictionaries and explicit missing/unseen routing.

Public [numeric operations](docs/v1/numeric-ops.md) now add quantile binning,
weighted row fields, histograms, split callbacks, routing and scalar/vector leaves.
Public [depthwise, best-first and symmetric growers](docs/v1/trees.md) compose
these operations and persist validated numeric/categorical trees. The first complete [squared-error recipe](docs/v1/squared.md)
and [Normal recipe](docs/v1/normal.md) support weights, offsets and
fixed/backtracking steps on CPU. Normal exposes ordinary/Fisher directions and
joint mean/log-scale updates. [Formula and sequential runs](docs/v1/formula-runs.md)
add structured full-metric updates and independent heterogeneous jobs. CUDA
training is not implemented yet. [Binary classification](docs/v1/binary.md) now
persists typed class order and exposes probability/label inference.
[Multiclass and vector leaves](docs/v1/multiclass.md) add joint softmax updates
and separate split/leaf statistics with arbitrary output mappings.

Independent references and comparator/data checks remain evaluation preparation.
F0.3 is still open; the user approved overlapping B03–B06 construction without
removing any v1 scope or acceptance requirements. No real quality, GPU performance
or agent/adoption advantage has been established for the new foundation.

- [Execution and reflections](v1-sprints/README.md)
- [Construction design](planning/foundation-construction-design.md)
- [v1 plan](planning/agent-boosting-foundation-plan.md)
- [Required tasks](planning/foundation-tasks.md)
- [Acceptance and evaluation](planning/openboost-v1-evaluation.md)

All R1–R9 / C1–C7 / A1–A13 remain required. Classification, regression, ranking,
quantiles, multi-output, count/positive/aggregate targets, survival, distributional
and formula models, and train-many each need their own implementation and evidence.

## Development

```bash
uv sync --extra test
uv run pytest tests/ -m "not gpu and not benchmark" -n 0 -q
uv run ruff check src/openboost tests/v1 tests/conftest.py
uv build
```

Python 3.10+. Current tests cover CPU implementation, independent references and
evaluation infrastructure. Experimental CUDA storage, named fields and histograms
have real T4 evidence. CUDA training and the full required device gates remain
open. Run GPU-marked tests only on real hardware; publishing remains separate.

## Historical implementation and evidence

Revision `50acfc6` is the last revision containing the old production code plus
Sprint 001 references. Use that revision in a separate checkout to reproduce
old APIs, examples and experiments; no compatibility layer remains here.

Historical tests, examples, documentation and benchmark artifacts are retained
as evidence and sources of mathematical counterexamples. Default test discovery
runs `tests/v1/` only. Old tests are not counted as v1 passes or skips.
The current documentation build uses `docs/v1/`; other documentation describes
the retired implementation. Published packages and historical results do not
establish the new architecture's quality, speed or adoption.

[Query-local ranking](docs/v1/ranking.md) adds pairwise/lambda CPU geometry and
fixed-step recipes with validation NDCG selection. Real A4 evaluation remains open.

[Quantile and penalized leaves](docs/v1/quantile.md) expose routed residuals/original
weights and compose all three CPU growth policies. Real A5 evaluation remains open.

[Poisson counts and exposure](docs/v1/poisson.md) add a CPU count recipe with explicit
rate/count outputs. Real A7 evaluation remains open.

[Gamma positive-target means](docs/v1/gamma.md) add weighted CPU mean regression.
Real A8 quality and distributional calibration remain unverified.

[Tweedie nonnegative means](docs/v1/tweedie.md) support fixed-power CPU fitting and
explicit annualized-loss weight semantics. Real A9 evaluation remains open.

[Frequency–severity composition](docs/v1/frequency-severity.md) binds matched paid-loss aggregates
and persists two-model inference with explicit output units. Real A9 evaluation remains open.

[Log-normal AFT](docs/v1/aft.md) adds event/right-censored CPU training and
persisted scale-aware survival outputs. Real A10 evaluation remains open.

[Current retrospective and plan](v1-sprints/063-retrospective-and-next-plan.md)
separates implemented CPU coverage from remaining authoring, practical execution,
real selection and GPU evidence.

[Multi-output squared regression](docs/v1/multioutput.md) supports independent/shared trees,
projected splits and persisted training-only target scaling. Real A6 evaluation remains open.

[Shared training preparation](docs/v1/preparation.md) reuses fitted CPU binning/codes
across independent jobs, verified at M=1/8/32.
[Independent stopping](docs/v1/stopping.md) adds validation patience to every CPU
recipe while keeping model acceptance and best-model selection independent.


[Experimental CUDA operations](docs/v1/execution.md) provide context-owned buffers,
named fields, once-only weighting and routed histograms, with
[33 passing real T4 checks](benchmarks/v1/evidence/cuda-aggregation-078/README.md).
Training recipes remain CPU-only; device candidates, trees and boosting are next.
