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
add structured full-metric updates and independent heterogeneous jobs. Experimental
resident scalar CUDA training passes all 212 bounded T4 checks, including
weighted/missing parity, transactions and saved CPU inference. The shared scoring
correction resolves the previous 14 failures without changing tolerances. See the
[recorded result and scope](benchmarks/v1/evidence/cuda-score-symmetry-089/README.md).
The first [Normal K=2 and installed D2 T4 run](benchmarks/v1/evidence/cuda-normal-090/README.md)
passes 381/383 checks, including all earlier scalar cases and nineteen saved-model
CPU replays. Two ordered acceptance decisions fail the frozen reference; full
Normal conformance remains open. The [follow-up diagnostic run](benchmarks/v1/evidence/cuda-acceptance-091/README.md)
preserves both failures and identifies rounding-induced false improvement at
near-stationary loss. The subsequent objective-owned comparison correction and
[bounded revalidation](benchmarks/v1/evidence/cuda-recipe-103/README.md) establish
514 earlier passes plus fifteen new recipe passes with identical production.
All 529 revised requirements have passing evidence across two executions;
the original failed verdicts remain preserved. This is not full Normal conformance.
[Binary classification](docs/v1/binary.md) now
persists typed class order and exposes probability/label inference.
[Multiclass and vector leaves](docs/v1/multiclass.md) add joint softmax updates
and separate split/leaf statistics with arbitrary output mappings.

Independent references and comparator/data checks remain evaluation preparation.
F0.3 is still open; the user approved overlapping B03–B06 construction without
removing any v1 scope or acceptance requirements. No real quality, competitive GPU performance
or agent/adoption advantage has been established for the new foundation.

The [early same-host performance checkpoint](benchmarks/v1/evidence/early-performance-104/README.md)
measures squared-error boosting at 10,000 rows in 7.04 s on CPU and 2.94 s on a
warm T4, with comparable quality. The four other timing pairs and the separate
profile are incomplete after deadlines. This synthetic internal result establishes
neither external-library speed parity nor practical performance across all recipes.

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
and bounded resident squared training have real T4 evidence. The full required
device recipe, quality and cost gates remain open. Run GPU-marked tests only on
real hardware; publishing remains separate.

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

[Current execution and reflections](v1-sprints/README.md)
separates implemented CPU coverage from remaining authoring, practical execution,
real selection and GPU evidence.

[Multi-output squared regression](docs/v1/multioutput.md) supports independent/shared trees,
projected splits and persisted training-only target scaling. Real A6 evaluation remains open.

[Shared training preparation](docs/v1/preparation.md) reuses fitted CPU binning/codes
across independent jobs, verified at M=1/8/32.
[Independent stopping](docs/v1/stopping.md) adds validation patience to every CPU
recipe while keeping model acceptance and best-model selection independent.


[Experimental CUDA operations](docs/v1/execution.md) provide context-owned buffers,
named fields, once-only weighting, routed histograms, candidate scores, composable
feasibility masks, routing and scalar leaves, with
[88 passing real T4 checks](benchmarks/v1/evidence/cuda-splits-078/README.md).
Independent cohort constraints change split selection through the public device
operations. Separate experimental resident squared geometry, scalar trees and
accepted/proposal training now pass the separate
[212-case T4 matrix](benchmarks/v1/evidence/cuda-score-symmetry-089/README.md).
Shared mapped transactions, Normal geometry and joint/ordered recipes now have
[bounded passing comparison and recipe evidence](benchmarks/v1/evidence/cuda-recipe-103/README.md),
with the historical numerical failures preserved in the earlier archives.
The installed D2 learner uses the same public field/feasibility/tree operations.
Other required CUDA recipes and full phase acceptance remain open.
