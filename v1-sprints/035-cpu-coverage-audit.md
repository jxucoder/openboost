# Sprint 035: CPU coverage audit before B11

Audited revision: b456bf3. Status: complete as an audit, not a phase exit.

## Plan

1. Compare public implementation and exercised tests with R1–R9/C1–C7/A1–A13.
2. Verify concrete gaps by reading call paths and running bounded probes.
3. Order remaining CPU/evaluation work without changing required scope or gates.

## Verdict

OpenBoost has a substantial tested CPU foundation and eleven built-in recipes.
It does not yet satisfy complete F1/B11 readiness, E0, E1 or E2. Do not infer phase
completion from sprint count or start GPU expansion as though CPU coverage were
complete. Preserve the existing order and all required applications.

The 716-test suite includes references and evaluation infrastructure. The public
production subset is 228 tests, all passing in this audit. Neither number is an
application-quality or agent-effort result.

## Application mapping

All entries below are bounded CPU mechanics, not real-data acceptance.

| Application / family | Public implementation and evidence | Remaining acceptance work |
|---|---|---|
| A1 / R1 regression | recipes.squared; test_public_squared.py | Real five-split workflow, selected baseline comparison and artifacts |
| A2 / R1 binary | recipes.binary, ClassSchema; test_public_binary.py | Full real workflow and output/evaluation metadata |
| A3 / R1 multiclass | recipes.multiclass, vector trees; test_public_vector_multiclass.py | Real quality and complete classifier workflow |
| A4 / R3 ranking | ranking.Ranking, recipes.ranking; test_public_ranking.py | Pair-weight/sampling limitations, real query splits, NDCG comparisons; fixed steps only |
| A5 / R2 quantile | recipes.quantile, routed residual leaves; test_public_quantile.py | Temporal real evaluation, installed author extension, declared quantile reporting |
| A6 / R8 multi-output | vector fields/leaves, projected split/full leaf fields and mappings; test_public_vector_multiclass.py | Missing complete multi-output squared objective/recipe with independent and shared trees, multi-round target metrics and output roundtrip |
| A7 / R4 counts | recipes.poisson, explicit exposure; test_public_poisson.py | Real frequency workflow/deviance, output-unit/protocol metadata |
| A8 / R4 positive | recipes.gamma; test_public_gamma.py | Real eligibility/weights, Gamma deviance and comparison |
| A9 / R4 aggregate | recipes.tweedie, FrequencySeverity and paid_loss_problems; test_public_tweedie.py, test_public_composition.py | Raw join audit, real aggregate quality and joint selection, units/provenance |
| A10 / R5 AFT | recipes.aft, explicit event_right target, AFTModel; test_public_aft.py | Real censored NLL/IPCW evaluation, scale protocol and dataset-license closure |
| A11 / R6 Normal | recipes.normal, ordinary/Fisher joint updates; test_public_normal.py | Ordered parameter mutations, complete output/metric workflow and real proper scores |
| A12 / R7 Formula | recipes.formula, full GGN saturation structure; test_public_formula_runs.py | Installed external formula mutation/dependencies, complete real and recovery/misspecification workflow |
| A13 / R9 train-many | runs.run_many, stable IDs/error isolation; test_public_formula_runs.py | Shared prepared binning, validation-driven independent stopping, M=32 and grouped/permuted equivalence, real selection and full-set cost |

Test paths are under tests/v1; public code is under src/openboost.

## Capability mapping and concrete findings

| Capability | Verified boundary | Gap or limit |
|---|---|---|
| C1 data/targets | Owned inputs, categories, class order, weights, offsets, structure, explicit event/right bounds | Binding does not certify entity/query/time split independence; real workflow mapping remains |
| C2 composable trees | Shared histogram/candidate/routing with three policies and extra independent fields | External D2 mutation still needs installed public-only execution evidence |
| C3 leaves/outputs | Scalar/vector, separate split/leaf fields, residual views, quantile/D3 solver | A6 regression composition missing; ResidualContext is scalar-specific |
| C4 runtime | Immutable proposal/accept/reject, best-model state, scoped RNG and sequential failures | No validation-patience stop state, no shared prepared input path, no CUDA/workspace/transfer semantics |
| C5 artifacts | Raw models, typed class mapping, AFT scale and frequency-severity roles | Normal/Formula and mean-family output/evaluation metadata remain external; no training-resume claim |
| C6 evaluation/authoring | Independent references, comparator/data/quality/selection infrastructure, sealed held-out manifest | No current v1 external author-package verification or scored author cohort; full expected matrix/E-gate aggregation incomplete |
| C7 usable workflow | Seventeen installed-wheel documentation examples | Complete real baseline→mutation→verification workflows are missing, particularly A6/A13 |

### Directly reproduced gaps

- Calling recipes.squared on a two-column target raises:
  "squared recipe requires scalar [N, 1] targets". Shared multiclass trees do
  not establish A6 multi-output regression.
- All eleven recipe signatures lack an early_stopping argument. Each recipe
  loops over a fixed round budget; best_model retention is not early stopping.
- Each recipe calls Binning.fit(...).transform(...) internally. run_many shares
  immutable Problem/data records but does not reuse fitted binning across fits.
- Public run-many tests parameterize M=1/2/8, not M=32. Different round budgets
  exercise configuration independence, not validation-triggered stop isolation.
- Normal and Formula build both channel terms from one snapshot and commit
  jointly. There is no ordered parameter-update recipe; D4 still needs execution.
- examples/extensions/demo.py imports openboost.experimental, which is retired.
  Those packages are historical evidence, not verified v1 extension wheels.
- benchmarks/v1/baseline_worker.py accepts xgboost/lightgbm/catboost/ngboost only.
  Existing baseline smokes do not run current OpenBoost or establish its quality.
- No CUDA implementation exists: RunContext rejects devices other than CPU.

## Ordered execution checklist

1. **A6 / R8 completion.** Add multi-output squared geometry and explicit
   independent-tree/shared-vector recipes. Require two or more rounds with
   original weights/offsets, per-target metrics, projected/full split comparisons,
   persistence and rejection against independent references. Reject censored
   target kinds explicitly even when all bounds are finite.
2. **R9 / C4 prepared inputs and stopping.** Add validated shared preparation,
   independent patience/stop state, and M=1/8/32 same-ID independent/reordered/
   regrouped equivalence with a failed run and different validation stop rounds.
   No fusion or speed claim. Reuse must verify data/transformer/config identity.
3. **R6/R7 ordered mutation and C5 outputs.** Exercise changed parameter order
   and line-search/rejection using public state, with joint versus ordered
   reference checks. Complete runnable output dependencies for Normal/Formula
   and explicit units/metadata for real application artifacts.
4. **C6 / E2 external author packages.** Implement D1–D5 against current installed
   public interfaces, isolated from core edits/private imports. Verify all five
   required mutation types, persistence and failures. Exploratory findings can
   change interfaces; they are not E5 comparisons.
5. **B11/F0.3 integration.** Connect every A1–A13 to current recipes, frozen
   datasets/partitions/config budgets, and independent quality/selection outputs.
   Reconcile existing data/baseline preparation instead of repeating smokes.
   Resolve A4/A10 source/license issues and complete the expected matrix and
   E-gate aggregation. Run only comparisons whose prerequisites are frozen.
6. **Formal author/GPU phases.** Freeze the justified CPU contracts and judges,
   run authorized independent held-out evaluation and preregistered author
   comparisons, then build required B12 CUDA subsets and B13 batching. Preserve
   all real quality, cost and independent-adoption gates.

This list sequences work; it does not waive other task-card requirements.
Independent F0.3 data/protocol work can progress alongside CPU construction.
Do not inspect sealed H1/H2 contents while redesigning the foundation.

## Verification and reflection

- Read implementation, public tests, current docs, main/construction/evaluation
  plans, worker dispatch and historical extension imports.
- Ran bounded A6/signature probes on b456bf3.
- UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest
  tests/v1/test_public*.py -q --tb=short: 228 passed (macOS, Python 3.12.12).
- Prior full regression at the same production revision: 716 passed, Sprint 034.
- This audit makes no new external release/version, benchmark or quality claims.

Observation: required families increasingly share real code, but the remaining
gaps concern orchestration, authorability and application evidence as much as
objectives. Decision: prioritize A6 and train-many/stopping over another
unrequested model family or premature interface freeze. Next slice is item 1.
