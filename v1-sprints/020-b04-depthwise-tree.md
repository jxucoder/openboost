# Sprint 020: B04 depthwise numeric tree

Starting revision: b77dbc5. Status: complete for this slice. The approved B03–B06 overlap
applies; F0.3 remains open.

## Plan and acceptance

1. Assemble depthwise growth through public histogram, candidate, feasibility,
   scoring, routing and leaf callbacks. Compare topology and predictions against
   the independent exhaustive oracle, including limited leaf budgets.
2. Validate explicit child-index topology and persist the numeric transformer,
   missing routes and scalar leaves. Reject corrupt graphs and round-trip in a
   fresh process with unseen numeric values and missing values.
3. Run focused and CPU regression tests, lint and documentation checks; record
   limitations and commit the verified slice locally.

The first acceptance check is an independent depthwise topology/prediction comparison.
Callback checks must demonstrate a changed algorithm, not merely a callable API.
This slice does not integrate tree terms into accepted run state or provide a
boosting loop, vector leaves, categories or CUDA.

## Result and reflection

The grower and inference artifact are implemented. All 553 CPU tests pass,
including 26 tree cases. Independent exhaustive growth agrees on topology, leaves
and predictions; callbacks demonstrably alter the algorithm. Strict docs, lint,
build and all three installed-wheel examples pass. See the
[learning record](../learnings/2026-09-06-v1-b04-depthwise-tree.md).

Observation: composing numeric operations into a learner required no task-name
branches or legacy trainer. Layer budgets did require an explicit policy and
cached callback gains, which the independent reference made reviewable. This
supports the scalar tree boundary only; it does not yet validate distributional
or structured learners. Next: integrate tree terms and squared/Normal recipes,
then probe Formula and heterogeneous runs before stabilizing the foundation.
F0.3, full F1 and later evaluation gates remain incomplete. Nothing was pushed.
