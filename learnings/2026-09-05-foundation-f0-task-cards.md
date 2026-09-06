# 2026-09-05: F0.1 executable specifications for every v1 use case

## Context

The user asked to continue after requiring individual implementation and
evaluation for every A1–A13 use case. The next committed slice is F0.1, rather
than another strategy reset or a premature runtime rewrite.

## Decision or Result

Complete task specifications for all applications, recipe/component mappings,
five author modifications and a minimal public-interface sketch. Explicit
mathematics and failure cases distinguish expressivity from predictive value.
F0.2 will implement independent references; F0.3 will freeze actual artifacts,
baseline builds, budgets, held-out tasks and the evaluation runner.

Select concrete datasets and split rules for every use case, including
Concrete Compressive Strength for FormulaBoost. Composition predicts two
parameters of a monotone saturation hypothesis in age; the formula is a proposed
model, not a source-established law. A single-row two-parameter GGN has rank one;
parameter recovery needs separate identifiable synthetic fixtures. Real task
quality and wrong-formula failures remain required.

Other consequential contracts include ranking query/pair weights, quantile
leaf access to residuals, the softmax diagonal curvature approximation, explicit
offset persistence, and separate positive-payment count/severity definitions
for aggregate-loss composition. No application can substitute for another.

## Changes

- [Task cards](../planning/foundation-tasks.md): A1–A13 inputs/outputs, data and
  splits, two-round semantics, math, failure conditions and baselines; R1–R9 /
  C1–C7 mappings, D1–D5 author tasks, device boundaries and minimal interfaces.
- [Active plan](../planning/agent-boosting-foundation-plan.md): mark only F0.1
  complete and hand off F0.2; the entire F0 and E-gates remain incomplete.
- [Application contracts](../planning/foundation-application-contracts.md):
  link concrete task decisions and replace the remaining unselected A10/A12 data.
- Learning index: expose the execution milestone ahead of the planning record.

## Verification

- Before editing, confirmed the task-card file was absent. Read current model,
  objective, batch and experimental entry points together with tests and docs.
- Primary-source audit covers current XGBoost/LightGBM/CatBoost objectives and
  extension boundaries, NGBoost geometry, Py-Boost GPU customization and the
  newly selected datasets; sources are linked in task cards. None of these new
  baseline versions was installed or benchmarked in this slice.
- Numerical sanity using `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run
  --no-sync python`: central finite differences matched the stated Poisson,
  Gamma, Tweedie p=1.5 and right-censored log-normal AFT derivatives. The Formula
  raw Jacobian matched finite differences and its single-row GGN had rank one.
  The D2 fixture selected cut 1 without the cohort constraint and cut 2 with it.
  These small checks do not replace F0.2 executable oracle tests or production parity.
- Documentation checks passed for all five changed Markdown files: 47 local
  links resolve and code fences are balanced. Coverage checks found all 13
  application cards and matrix rows, all nine recipes, seven capabilities and
  five author tasks; each application has inputs/outputs, data, failure cases
  and baselines. Only F0.1 is checked off; F0.2/F0.3 remain pending.
- `git diff --check` passed. No runtime code changed and no GPU job or full
  model regression suite was run.

## Failed Attempts

- Direct CatBoost multi-output documentation fetching timed out; the official
  indexed page supplied the MultiRMSE device support table. Py-Boost's latest
  release URL did not yield a release, so no version/tag was invented.
- Existing Formula code already computes a coupled direction before fitting
  parameter trees. A comparison that forbids incumbent outer loops would
  manufacture a false capability advantage.
- Py-Boost's actual builder sketches G/H for split selection but computes leaf
  values from the original gradients. That boundary is an existing competitor
  capability, not an OpenBoost invention; modifications must also update cached
  train/validation predictions before the next gradient evaluation.
- A CPU-only author comparison would exclude Py-Boost unfairly. It remains an
  eligible direct competitor; if selected, its arm runs on GPU with declared
  device/cost under the same E5 wall/token budget.

## Risks and Follow-ups

- Data bytes, parsed schemas, actual group counts, licenses for downloaded
  artifacts, frozen builds and runtime capability still require F0.3 verification.
- Formula misspecification, category quality and bounded-pair ranking costs may
  fail their gates; retain failures instead of deleting tasks or changing targets.
- H1/H2 are reserved evaluation slots, not completed hidden tasks. The evaluation
  side must freeze content/verifiers without exposing them to F1 interface design.
- Next: F0.2 independent scalar/tree references, then the other task-specific
  geometries and state/run failure fixtures, without production core changes.

## Commits

- `7b57436` — every listed v1 use case made individually required.
- This slice: `docs: specify all v1 foundation tasks and baselines`.
