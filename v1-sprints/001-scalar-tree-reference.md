# Sprint 001: Independent scalar/tree references

Started: 2026-09-05. Starting revision: `9700845`. Status: **this sprint complete; F0.2 ongoing**.
Plan mapping: first part of B01/F0.2; A1/R1, scalar subset of C2/C3, D2; E1 reference preparation.

## Purpose and scope

Provide independent judging for F1 histogram/split/route/leaf/grow. The new implementation
and oracle must not share old production algorithms and repeat the same mistake.
Tests exhaustively enumerate and reduce original rows only on tiny fixtures.

- Scalar weighted half-squared loss, base, Newton leaves and half gain; apply weight once.
- All fixed numeric-bin candidates, both missing routes, ties, infeasible candidates and actual children.
- D2 independent cohort information mass; total H alone is insufficient.
- Explicit topology and budgets for depthwise, best-first and symmetric growth; two-round squared-error traces.
- Isolated execution without production imports. Root `tests/conftest.py` imports old
  openboost, so reference tests use `--confcutdir=tests/v1` and a clean subprocess import check.

Not completed here: categorical/vector/other objectives, production trainers, persistence,
CUDA, real-data/baseline evaluation, or all F0.2. Other required cases remain required.

## Execution checklist

- [x] Read design, old split/plugin semantics and tests; identify doubled old gain and production imports.
- [x] Write hand-calculated tests and confirm failure before reference modules exist.
- [x] Implement independent scalar and brute-force tree functions in `tests/v1/reference/`.
- [x] Verify two rounds, three growth policies, missing/weight/cohort/ties/invalid inputs and import isolation.
- [x] Run focused tests and changed-file lint; review and commit.
- [x] Record reflection, results and handoff; update the index.

## Acceptance

1. For `[-2,-2,2,2]`, two feature bins and lambda=1, first leaves are ±4/3 and net gain=16/3.
   With eta=.1, second-round residuals must change; hand calculations check both rounds.
2. Integer weights equal replicated rows with fixed bins. Zero weights cannot create splittable
   nodes; invalid denominators and negative/non-finite values are rejected.
3. Both missing directions have optimal counterexamples; exact ties follow feature/candidate/missing order.
4. In D2's six-row fixture, unconstrained cut 1 becomes cut 2 with cohort constraints; no feasible candidate means no split.
5. Distinct fixtures exercise all three policies. Symmetric growth combines layer gains for a
   common candidate before selecting; it cannot combine independently selected node winners.
6. References use only stdlib/NumPy, never OpenBoost objectives, splits, histograms or trainers.
   They prepare E1; production parity still needs future components under test.

## Execution and verification

Deliverables: [scalar](../tests/v1/reference/scalar.py), [tree](../tests/v1/reference/tree.py),
[hand calculations and counterexamples](../tests/v1/test_tree_reference.py),
[isolation](../tests/v1/test_reference_independence.py), [instructions](../tests/v1/reference/README.md).

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1 --confcutdir=tests/v1 -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check tests/v1
```

- Red: two collection errors because `tests.v1.reference` did not exist.
- Green: **55 passed**, no skips; changed-file Ruff passed.
- Environment: local macOS CPU, Python 3.12.12, NumPy 2.3.5, pytest 9.0.2.
- A subprocess blocks all `openboost` imports; all three policies still produce two-round hand results.
- First gain=16/3; second leaves=±56/45; final raw=±58/225. D2 changes cut 1 to 2.
- This verifies reference fixtures, not new production parity or execution cost.

## Reflection

Opening observation: the architecture is concrete, but executable independent oracles are missing.
Evidence: `tests/v1/` does not exist; old tests use old gain/weight conventions and import production.
Decision: build a bounded scalar/tree reference first, without changing APIs or optimizing GPUs.
Next: implement against hand fixtures, then audit coverage and remaining F0.2 work.

Closing observation: all policies share candidate/routing mathematics, but symmetric selection
must combine common candidates first. The left/right nodes prefer features 1/2 individually;
the symmetric oracle chooses common feature2, retaining the valid zero-gain left candidate.
Decision: public candidates must distinguish validity from gain; growth policy decides when
to discard nonpositive gain.

Coverage: only numeric scalar references are complete. Categorical/binning, classification,
ranking, quantile/vector, positive/AFT, Normal/Formula and transaction/run work remain.
Fifty-five internal tests do not complete the foundation. During execution the user requested
retirement of all old production code and a clean v1 rebuild. Record/commit that separately,
preserving references and historical experiments without changing mathematical/quality gates.

## Commits

- `9700845`: prerequisite construction design.
- `e76a2cd`: sprint execution and reflection process.
- Implementation slice: `test: add independent scalar and tree references for v1`.
- `50acfc6`: verified independent references and 55 tests.
