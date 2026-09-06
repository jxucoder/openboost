# Sprint 006: Normal/NaturalBoost and Formula directions and commits

Starting revision: `c5e9899`. Status: sprint complete; F0.2 ongoing.
B01/F0.2, independent A11/A12/D4 mathematics and candidate-state probes.

## Plan and acceptance

1. Hand/finite-difference checks for Normal ordinary gradients/Fisher/natural directions,
   independent NLL/CRPS, and Formula softplus/Jacobian/GGN/ordinary-diagonal-full directions.
2. Fit negative directions using independent scalar trees; apply fit weight once. Check two-round
   joint/ordered updates, fixed steps and bounded backtracking. Rejection must preserve raw/terms.
3. Repeated Z with different x, single-x nonidentifiability and misspecification counterexamples;
   full regression/lint/reflection/commit.

These are mathematical/minimal update references, not production runtime or full transactions.
Fisher is not a Hessian; GGN does not establish identifiability. Initialization cannot read validation.
Persistence, real quality, agent costs, CUDA and E-gates remain pending; all A1–A13 remain required.

## Results and verification

Added coupled.py, 25 tests and import isolation. **208 passed, no skips**; Ruff passed.
Local macOS CPU/Python 3.12.12/NumPy 2.3.5.

- Normal: hand Fisher/natural direction, weighted training mean/scale and explicit scale floor.
  Independent evaluator does not call the objective; closed-form NLL/CRPS checked separately.
  Two-round natural root leaves match per-round mu/sigma algebra. Nonunit fit weights equal replication.
- Formula: stable softplus/expm1, analytic Jacobian matches finite differences, GGN=JᵀJ. All three
  directions have two-round joint checks. Full uses explicit2×2 inverse and rejects undamped
  rank-deficient GGN. Ordinary direction does not use metric/damping as a preconditioner.
- Each direction-tree leaf matches an independent sum of same-snapshot negative directions.
  Predictions rebuild from tree/channel/coefficient. Z/x are separate, with repeated Z/different x
  and known-a/b generated fixtures.
- Two-round ordered Normal recomputes geometry after each parameter and differs from joint log-scale.
  Explicit fixed finite steps can allow higher loss; backtracking requires strict decrease.
  Fixed steps still reject invalid geometry.
- Backtracking rejects invalid step1000 and finite-but-worse step5, then accepts.1; this equals direct.1.
  Full rejection leaves no terms and raw_before=raw_after. Caller mutation cannot alter snapshots.
- Two a/b pairs at one x give the same output: full GGN is not identifiability. A decreasing target
  for one recipe contradicts a saturating increasing formula. No real Concrete parameter-recovery claim.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

Missing-module collection failed first; the first12 mathematical cases passed. Lint formatting was
fixed; review added explicit rejection of softplus underflow to0 without changing positive support.
No dependencies added. Full persistence/best/early-stop, production CPU, real NLL/CRPS/Formula quality,
CUDA and E-gates remain unverified. This step is a small reference, not a universal trainer/runtime.

## Reflection

Direction, direction fitting and acceptance can be expressed separately; Normal/Formula need not
be forced into scalar Newton interfaces. Both reuse scalar trees, solve Fisher/GGN before fitting,
apply fit weights once and use original loss for acceptance. Preserve direction-adapter/transaction boundaries.

A1–A12 mathematical probes still do not complete F0.2. Missing items include A13 isolation/selection,
row binding, full categorical/vector growth, D3 penalized leaves and full composition/persistence
references. Next prioritize state/run/identity and an acceptance ledger. Do not optimize production
prematurely or endlessly add similar mathematical tests. Freeze F0.3 before F1 public components;
all A1–A13 still need production and real evaluation.

## Commits

- This slice: `test: add Normal and Formula directional update references for v1`.
