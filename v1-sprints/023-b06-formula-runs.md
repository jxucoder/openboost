# Sprint 023: B06 Formula and sequential heterogeneous runs

Starting revision: fd3dbd9. Status: initial probe complete. Approved B03–B06 overlap applies;
F0.3 and complete application evaluations remain open.

## Plan and acceptance

1. Bind owned row-aligned structure separately from features; expose saturation
   Formula prediction, Jacobian/GGN geometry and damped full-metric directions.
2. Compose Formula through existing learners/transactions and compare two or more
   rounds with independent exhaustive references, including rank deficiency.
3. Add sequential run_many with independent contexts, heterogeneous raw widths,
   termination lengths and failures. Test M=1/2/8 and reordered scheduling.
4. Verify inference round trips, regression, lint/docs/build, reflect and commit.

The first failing test requires Formula's structural role, not an added feature.
No fused execution, GPU speed, full A12 value or finished v1 claim is intended.

## Result and reflection

Formula and sequential mixed-run probes are implemented. All 581 CPU tests pass;
three Formula rounds agree with independent geometry/growth, and M=1/2/8 jobs
match independent and reordered execution. Lint/docs/build and six installed-wheel
examples pass. See the [learning record](../learnings/2026-09-06-v1-b06-formula-runs.md).

Observation: the full-metric structured case needed new geometry/structure, but
reused scalar statistics, growth, mapped terms and transactions. Heterogeneous
execution required no shared mutable model or target padding. These are useful
abstraction probes toward programmable boosting. They do not establish agent
productivity, real predictive quality or efficient execution. Runtime remains
CPU-oriented and scalar-tree payloads still constrain upcoming leaf families.

This closes the initial B03–B06 construction probe, not F1 or v1 acceptance.
Next is B07 growth policies and categorical preparation, followed by vector and
specialized leaves. Keep ordered updates, stopping policies and every required
application/evaluation case visible. F0.3 remains open; no interfaces are frozen.
Nothing was pushed.
