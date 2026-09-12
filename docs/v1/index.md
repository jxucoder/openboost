# OpenBoost v1

OpenBoost is a programmable boosting foundation for researchers and agents.
Public data, state, tree, recipe and model components support building algorithm
changes from explicit operations. CPU recipes and a required subset of CUDA
recipes have implementation and bounded correctness evidence. V1 is still under
construction; implementation does not by itself establish real-data quality,
speed or complete acceptance.

The [foundation checkpoint](checkpoint.md) describes this PR's source scope,
available historical evidence and pending candidate validation. Individual guides
explain the interfaces; they do not establish complete v1 acceptance.

## Start with the components

1. [CPU data, state and models](cpu-state.md): construct a problem, own run state
   and persist an ensemble with explicit output mappings.
2. [Numeric operations](numeric-ops.md) and [tree policies](trees.md): compose
   binning, statistics, candidate selection, routing and leaves. CPU trees support
   depthwise, leafwise and oblivious growth, with [categorical conditions](categorical.md).
3. [Squared-error boosting](squared.md): follow a complete recipe, then use
   [validation stopping](stopping.md) and [shared preparation](preparation.md)
   for independent runs.
4. [Installed extensions](extensions.md): replace split constraints or leaves
   through public interfaces and verify inference after removing training plugins.

## Choose a recipe

| Task | Guide and distinguishing behavior |
| --- | --- |
| Regression | [Squared error](squared.md), fixed or backtracking updates |
| Classification | [Binary](binary.md) class schemas and probabilities; [multiclass](multiclass.md) softmax, vector leaves and output mappings |
| Ranking | [Query-local ranking](ranking.md), pairwise and lambda geometry |
| Quantiles | [Quantile and penalized leaves](quantile.md), routed residuals and original weights |
| Counts and positive targets | [Poisson](poisson.md) exposure and rate/count units; [Gamma](gamma.md) positive means; [Tweedie](tweedie.md) nonnegative means |
| Composed loss | [Frequency–severity](frequency-severity.md), matched aggregates and two-model inference |
| Survival | [Log-normal AFT](aft.md), event/right-censored targets and survival outputs |
| Distributional prediction | [Normal](normal.md), ordinary/Fisher directions and joint/ordered mean and log-scale updates |
| Structured prediction | [Formula and sequential runs](formula-runs.md), structural Jacobians and independent heterogeneous execution |
| Multiple targets | [Multi-output squared](multioutput.md), independent/shared trees, projected splits and training-fitted target scales |

## Execution and current evidence

[Exact Newton ordering](newton-order.md) and exact original-row leaves define
the numerical choices used by Normal. [CUDA execution](execution.md) provides
context-owned storage and resident operations; [device runs](device-runs.md)
describe independent outcomes and scheduling. CUDA coverage is narrower than CPU
coverage: consult the [required-device contract](https://github.com/jxucoder/openboost/blob/main/v1-sprints/080-cuda-required-recipes.md)
and the [checkpoint scope](checkpoint.md) before selecting a recipe or interpreting an older test result.

Compatible scalar squared scheduling, exact Normal policies and installed
extensions have source-specific historical validation. The
[checkpoint report](checkpoint.md#historical-observations-and-available-evidence)
retains a compact audit of a real two-round Housing M=1/8/32 diagnostic. Its
underlying replay archives are omitted from this PR. Full-budget train-many,
complete real-data quality, execution cost and validation of the curated candidate
remain open; no speed claim follows.

V1 APIs are intentionally not backward compatible with the retired production
API. Historical examples require revision `50acfc6`. For current requirements,
see the [application contracts](https://github.com/jxucoder/openboost/blob/main/planning/foundation-application-contracts.md)
and [evaluation protocol](https://github.com/jxucoder/openboost/blob/main/planning/openboost-v1-evaluation.md); for evidence
scope, start at the [foundation checkpoint](checkpoint.md).
