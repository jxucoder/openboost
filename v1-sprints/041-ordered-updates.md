# Sprint 041: Public ordered parameter updates

Parent: 6b2385f. Status: complete for the D4 development slice. Mapping: Sprint 038 M2, D4/R6/R7/C4.

## Plan and acceptance

1. Add an installed development package composing public objective geometry,
   least-squares fields, trees and propose/preview/resolve transactions.
2. Recompute geometry after each accepted parameter; test both orders for Normal
   ordinary/Fisher and Formula full-GGN directions over multiple rounds.
3. Compare independent reference updates, including coefficients and rejection,
   verify outer-round stopping and fresh inference after plugin removal.
4. Record measured boundaries, run regression/lint/docs and commit locally.

D4 uses at most six rates 0.1*0.5**j and strict finite training-loss descent.
Validation chooses best snapshots after each accepted substep; patience observes
once per outer round. No core change is presumed. This is exploratory development,
not scored E5 or full E2/E6. Existing joint recipes remain explicit alternatives.

## Results and reflection

The ob-ordered-updates wheel supplies an objective-independent ordered sweep/fit
loop with Normal ordinary/Fisher and Formula full-GGN convenience wrappers. It
uses public geometry, fields, trees, preparation, transactions and StopState;
no src/openboost changes or private imports were needed.

Six installed cases cover both parameter orders for Normal natural/ordinary and
Formula, with three rounds each. Every accepted raw update and tried coefficient
matches a separate reference trace (maximum absolute raw difference
2.220446049250313e-16). The reference producer runs before isolated execution;
the installed extension receives traces, not reference code. Eight saved models
(D2/D3 plus these six) preserve exact raw predictions after all plugins are removed.

Sixteen focused tests also cover nonzero offsets/weights, different results for
joint versus ordered and reversed order, full reversal/NaN rejection, rejection
of the first parameter followed by acceptance of the second, recovery at the
second step size, invalid order and outer-round patience. CPU regression: 783
passed. Ruff, strict MkDocs and four-wheel installation/removal verification pass
on macOS/Python 3.12.12/NumPy 2.3.5. Core wheel hash still matches Sprint 039.

Observation: ordered adaptive algorithms can be expressed with public transactions
and a short external loop, without copying a built-in recipe. Evidence: both
structurally different parameter geometries share the same sweep implementation.
Counterexample: run_many rejects OrderedResult because it requires concrete
FitResult; the installed checker retains this failure. Decision: fix the shared
result contract during D5 rather than adding objective-name scheduler branches or
pretending external scheduling already works. This prevents a full E2/F1 exit.

Next: shared result interoperability and remaining D1/D5 installed tasks, then
current real-data worker integration. Normal/Formula complete output/dependency
workflows and formal author/quality/GPU gates remain open. These are internal
development trials, not independent authors or E5 wins.

See [raw evidence](../benchmarks/v1/evidence/ordered-updates-041/README.md) and
[learning record](../learnings/2026-09-06-v1-ordered-updates.md).
