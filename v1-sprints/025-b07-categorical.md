# Sprint 025: B07 categorical preparation and inference

Starting revision: ea13cce. Status: categorical slice complete.

## Plan and acceptance

1. Add owned mixed feature input and train-only typed category dictionaries.
   Generalize NumericBinning/NumericTree to Binning/Tree without compatibility shims.
2. Use dictionary equality candidates with explicit feature kinds through shared
   histograms, routing and all growth policies; unknown tokens follow missing routes.
3. Compare mixed topology and predictions against independent raw-row references;
   test corrupt dictionaries/schema, fresh-process persistence and recipe integration.
4. Verify regressions/lint/docs/build, reflect and commit locally. No CUDA or
   upstream categorical-feature parity claim is intended.

The first failing test constructs mixed input with a categorical column.

## Result and reflection

Mixed input, train-only dictionaries, equality candidates and inference persistence
are implemented across all three policies. All 615 CPU tests pass; independent
mixed-reference topology/output, missing/unseen routing and fresh-process ensemble
round trips pass. Lint/docs/build and all seven installed-wheel documentation
pages pass. See the [learning record](../learnings/2026-09-06-v1-b07-categorical.md).

Observation: categorical support changed preparation and condition interpretation,
not growth or transactions. Explicit dictionary metadata preserved the distinction
between numeric ordering and categorical equality. Empty dictionaries exposed a
validation difference between histogram storage capacity and valid split values;
validation now uses actual category count. Keep this distinction for upcoming
payload and class schemas rather than treating padded storage as semantic state.

Next: B08 classification/class schemas and vector leaves/mappings. Full F1 and
F0.3 remain incomplete; every required objective/application and CUDA evaluation
still needs evidence. No interfaces are frozen, no upstream parity claimed and
nothing pushed.
