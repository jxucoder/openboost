# OpenBoost

**A programmable boosting foundation for researchers and AI agents.**

Build boosting algorithms in Python by composing objectives, statistics, split
rules, weak learners and update policies. OpenBoost provides a NumPy CPU reference
and experimental CUDA execution with Python-authored kernels.

## Vision

An algorithm change can require more than a new loss function. It may need extra
statistics, a different split constraint, a custom leaf solver, coupled outputs,
or a different acceptance rule. OpenBoost aims to make each of those decisions
accessible through public components and ordinary Python training loops.

The design takes inspiration from PyTorch's composable building blocks and explicit
execution. Here the building blocks are boosting operations: objective geometry,
named row fields, histograms, candidate scoring, feasibility, routing, leaf solving,
and immutable training transactions. A recipe is a working composition that a
researcher or agent can inspect, modify and reuse.

Standard GBDT, NaturalBoost/NGBoost-style distributional methods, FormulaBoost-style
structured models and training many models guide the abstraction boundaries.
Applications matter equally: classification, regression, ranking, quantiles,
multi-output prediction, counts, positive and aggregate targets, survival,
distributional modeling and model selection all remain required v1 scope.

Success means making a correct algorithm change easier, then demonstrating useful
quality and execution cost on real workloads. CPU is the semantic reference and
usable development path; CUDA is the route toward efficient execution. Comparative
agent-authoring studies are currently deferred while foundation construction and
validation continue. Author productivity and adoption benefits remain hypotheses.

## What works today

**Experimental v1, under construction.** This branch curates a later foundation
snapshot for review; the [checkpoint guide](docs/v1/checkpoint.md) records its source,
available evidence and pending candidate validation. It is not a drop-in replacement
for XGBoost, LightGBM or CatBoost, and full v1 acceptance remains open.

Public CPU components include typed numeric/categorical data, fitted preparation,
weights and offsets, named statistics, composable split/routing/leaf operations,
and depthwise, best-first and symmetric growers. Scalar/vector leaves and mapped
outputs share explicit proposal/accept/reject state. All twelve CPU recipes support
independent validation patience; saved models retain their required inference metadata.

The CUDA implementation uses CuPy-owned storage and streams with Python kernels
compiled by `numba-cuda`. Public operations expose storage, fields, histograms,
scores, feasibility masks, routing, scalar leaves, depthwise trees and resident
transactions. Training uses explicit device interfaces; the CPU recipe API does
not automatically dispatch to CUDA. Current CUDA tree growth covers numeric and
missing features, with scalar trees and mapped multi-parameter updates.

| Use case | CPU implementation | Experimental CUDA implementation |
| --- | --- | --- |
| Regression | [Squared error](docs/v1/squared.md) | Resident squared recipe |
| Classification | [Binary](docs/v1/binary.md), [multiclass](docs/v1/multiclass.md) | Binary and independent-tree multiclass recipes |
| Counts and positive/aggregate targets | [Poisson with exposure](docs/v1/poisson.md), [Gamma](docs/v1/gamma.md), [fixed-power Tweedie](docs/v1/tweedie.md), [frequency–severity composition](docs/v1/frequency-severity.md) | Poisson recipe; other cells pending |
| Ranking and quantiles | [Query-local pairwise/lambda ranking](docs/v1/ranking.md), [quantile and penalized leaves](docs/v1/quantile.md) | Pending |
| Survival | [Fixed-scale log-normal AFT with events/right censoring](docs/v1/aft.md) | Fixed-scale event/right-censored AFT recipe |
| Distributional and structured models | [Normal ordinary/Fisher updates](docs/v1/normal.md), [saturation Formula/full-GGN updates](docs/v1/formula-runs.md) | Exact-policy Normal joint/ordered recipes; Formula pending |
| Multi-output regression | [Independent/shared trees, projected splits and target scaling](docs/v1/multioutput.md) | Independent/shared multi-output squared recipes |
| Train-many | [Shared preparation and independent sequential runs](docs/v1/preparation.md), explicit run ownership | Compatible scalar squared scheduling |

CUDA entries describe implemented interfaces. Validation for the curated candidate
remains pending; historical checks do not establish complete feature coverage or a
speed guarantee. Categorical CUDA growth and arbitrary grouped recipes remain open. See the [CPU component guide](docs/v1/numeric-ops.md),
[tree contracts](docs/v1/trees.md), [stopping semantics](docs/v1/stopping.md) and
[explicit CUDA interfaces](docs/v1/execution.md) for supported inputs and limits.

## Evidence and performance

The [checkpoint guide](docs/v1/checkpoint.md) links the compact historical audit of
the real two-round Housing train-many diagnostic and explains the omitted replay
archives. Full-budget feasibility, performance and complete v1 remain open.
The results below are earlier observations retained on public main, tied to their
original source revisions; they are not results for this curated candidate.

- **Earlier CUDA validation:** [run 12](benchmarks/v1/evidence/cuda-glm-108/README.md)
  passes 571/571 real T4 cases: 153 binary/Poisson checks and 418 regressions. All
  77 JSON artifacts are retained. The offline audit verifies 246 numerical
  loss-change comparisons and replays 32 final/best models from saved input bytes.
- **Reliable Normal decisions:** [comparison and revalidation evidence](benchmarks/v1/evidence/cuda-recipe-103/README.md)
  covers all 529 revised requirements across two executions. Earlier failed
  verdicts remain preserved. This is bounded coverage, not full Normal conformance.
- **Measured internal improvement:** [run 11](benchmarks/v1/evidence/parallel-validation-105/README.md)
  passes 474 T4 checks and three cost gates. Parallel field validation reduces
  median warm fit time for synthetic squared boosting at 100,000 rows from
  13.513 to 8.947 seconds, with unchanged model/prediction bytes. That workload
  uses 16 features, depth three and 20 rounds; the reduction is 33.79% against the
  earlier OpenBoost implementation on the same T4.
- **Earlier CPU and packaging:** the public-main checkpoint has 2,292 local CPU tests passing.
  [Hosted CI](https://github.com/jxucoder/openboost/actions/runs/34241939802) passes
  Linux/macOS on Python 3.10/3.12, including offline audits and package builds;
  [strict documentation checks](https://github.com/jxucoder/openboost/actions/runs/34241939812)
  also pass. Historical tests are explicitly separated from current conformance.

These results do not establish competitive speed or predictive quality against
mature boosting libraries. Complete application evaluation and formal end-to-end
quality/cost acceptance remain open for the current foundation. The [earlier incomplete performance checkpoint](benchmarks/v1/evidence/early-performance-104/README.md)
is retained alongside the later complete measurements.

## Try the CPU foundation

Use Python 3.10+ and install from this checkout:

```bash
uv sync --extra test
```

This example supplies a custom learner through the public growth and feasibility
operations, fits a squared-error recipe, and saves its best validation model.

```python
from functools import partial

from openboost import NumericData, Problem, RunContext
from openboost.artifacts import Model
from openboost.ops import feasible
from openboost.recipes import squared
from openboost.tree import depthwise

train_x = NumericData([[0], [1], [2], [3]], [10, 11, 12, 13], ("x",))
valid_x = NumericData([[0.5], [2.5]], [20, 21], ("x",))
train = Problem(train_x, [[-3], [-1], [1], [3]], train_x.row_ids)
valid = Problem(valid_x, [[-2], [2]], valid_x.row_ids)


def learner(binned, fields):
    return depthwise(
        binned, fields, max_depth=1,
        legality=partial(feasible, min_child_h=2),
    )


fit = squared(
    train, valid, context=RunContext("example", seed=7),
    learner=learner, rounds=3, learning_rate=0.5, bins=4,
)
model = fit.state.best_model
prediction = model.predict(valid_x)  # Shape: (2, 1)
model.save("model.json")
restored = Model.load("model.json")
```

For deeper changes, compose [objective/statistics operations](docs/v1/numeric-ops.md)
and [run transactions](docs/v1/cpu-state.md) directly. CUDA users need real NVIDIA
hardware and the optional dependencies (`uv sync --extra cuda`); start with the
[separate device execution guide](docs/v1/execution.md).

## Next milestones

1. Validate the exact curated package, installed consumers, CPU/CUDA behavior,
   persistence and documentation in bounded Modal execution.
2. Reconcile the omitted archive-dependent tests and complete full-budget
   train-many resource and correctness evaluation.
3. Complete real-workload quality and execution-cost evidence for every required
   application, then stabilize the contracts supported by that evidence.

All [R1–R9 / C1–C7 / A1–A13 requirements](planning/openboost-v1-evaluation.md)
remain in scope. Each [application family](planning/foundation-application-contracts.md)
needs its own implementation and evaluation. Multi-GPU, Ray and out-of-core
expansion are outside the active plan.

- [Construction design](planning/foundation-construction-design.md)
- [v1 plan](planning/agent-boosting-foundation-plan.md)
- [Current checkpoint and evidence scope](docs/v1/checkpoint.md)
- [Retained earlier execution history](v1-sprints/README.md)

## Development and history

```bash
uv run pytest tests/ -m "not gpu and not benchmark" -n 0 -q
uv run ruff check src/openboost tests/v1 tests/conftest.py
uv run mkdocs build --strict
uv build
```

Default discovery runs `tests/v1/`; see [test scope](tests/README.md). Run GPU-marked
tests only on real hardware. The current documentation lives in `docs/v1/`.

The retired implementation remains at revision `50acfc6` for reproducing old APIs,
examples and experiments. There is no compatibility layer in v1. Historical
packages and benchmarks describe their recorded revisions; all new claims must
link to reproducible evidence for the current foundation.
