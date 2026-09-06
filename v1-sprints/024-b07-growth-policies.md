# Sprint 024: B07 numeric growth policies

Starting revision: f6ad95f. Status: numeric policy slice complete.

## Plan and acceptance

1. Add best-first heap scheduling and symmetric common-condition layer selection
   using existing histogram/scoring/legality/routing/leaf operations.
2. Compare all three policies with independent exhaustive topology/prediction
   references, including weights, missingness, leaf budgets and custom callbacks.
3. Exercise recipe substitution and persistence; verify regressions, lint/docs,
   record reflection and commit locally. Categorical support remains next.

The first failing check imports best_first and symmetric for oracle comparisons.
No library parity, speed or completed B07/F1 claim is intended.

## Result and reflection

All three numeric growth policies now use shared public operations and the same
inference artifact. All 603 CPU tests pass; 15 independent policy/budget cases
match topology and predictions, with additional callback, symmetric-layer and
recipe/persistence cases. Lint/docs/build and installed-wheel examples pass.
See the [learning record](../learnings/2026-09-06-v1-b07-growth-policies.md).

Observation: changing growth order required no objective or run-state change.
Symmetric selection did require its own aggregate decision rather than reuse of
per-node winners. The existing operation boundary supported that distinction.
Categorical conditions will next test whether the numeric representation is too
restrictive. Preserve explicit conditions and transformer identity rather than
encoding categories as ordered numeric thresholds.

Categorical support remains the next B07 slice. Full B07, F1, F0.3 and evaluation
remain incomplete. No speed or upstream-library parity claim; nothing pushed.
