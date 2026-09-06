# 2026-09-06: First complete squared CPU recipe

## Context

The depthwise scalar learner existed, but accepted state only represented constant
updates. B05 needs complete recipes sharing the same foundation and transactions.

## Decision or Result

Model replaces ConstantModel and the constant-only format, as permitted by the
clean-redesign instruction. Immutable tree terms carry a scalar learner, explicit
[1, K] output map and coefficient. Multiple terms can be proposed atomically.
The squared recipe composes public geometry, statistics, growth and transactions.
Fixed steps and bounded backtracking share this path; validation selects best
state independently of training acceptance.

## Changes

- `artifacts.py`: mapped tree/constant ensemble, strict nested tree persistence,
  conservative finite-output envelope and separate inference offsets.
- `runtime.py`: atomic term tuples and owned direct proposal construction.
- `objectives.py`: scalar squared base, unweighted gradients, once-weighted fields
  and weighted mean half-square loss with offsets applied outside raw caches.
- `recipes.py`: complete fixed/backtracking loop and per-round evidence, with an
  ordinary custom learner callable. Conflicting growth options fail explicitly.
- Existing state tests retain their mathematical checks using the new model name.

## Verification

- Initial focused recipe test failed with ModuleNotFoundError before implementation.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run
  --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  560 passed, including seven new squared/ensemble tests. Python 3.12.12,
  NumPy 2.3.5, macOS; no CUDA execution.
- Three rounds match independent exhaustive reference gradients, losses, raw
  predictions and base with nonuniform weights, zero weight and missing features.
- Offset-shift equivalence, distinct validation caches and earlier best retention.
- Backtracking fits one learner, tries 16/8/4/2 and accepts 2; a six-trial rejection
  preserves terms, version, best model and identical raw state across rounds.
- Mixed mapped-tree/constant terms commit jointly; fresh-process inference matches
  after persistence on unseen numeric and missing inputs with explicit offsets.
- Corrupt nested topology/maps/coefficients/kinds and invalid zero-round options
  are rejected. Direct Proposal copies its term sequence before exposing it.
- Ruff, strict MkDocs and offline build pass. All four public docs examples pass
  under Python -I from an isolated installed wheel outside the checkout.
  Wheel SHA256: f16630722fc448a301e036972cd0723329da85194f3edea7ea91a2183272d560.

## Failed Attempts

Initial lint required sorting imports and replacing a test lambda assignment.
Final review found direct Proposal construction could retain a mutable term list;
normalizing to an owned tuple closed that path and a regression test covers it.
No oracle or acceptance threshold was changed.

## Risks and Follow-ups

This is the squared half of B05, not completion of Normal, F1 or v1 evaluation.
Normal requires raw parameter widths distinct from observed target widths; the
current Problem/state shape contract still couples them. That is the next design
probe, followed by Formula and heterogeneous runs in B06 before stabilization.
Backtracking reuses fitted trees but recomputes ensemble predictions. Traces retain
round arrays; no large-workload memory or performance claim is made. The finite
absolute-value envelope may reject extreme terms that would cancel. Artifacts
support inference only, not training resumption. Categories, vector/structured
leaves, CUDA and full A1–A13 evaluation remain required future work.

## Commits

- This slice: feat: compose squared boosting with mapped tree transactions.
