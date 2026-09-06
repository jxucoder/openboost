# Sprint 044: Current OpenBoost evaluation worker

Parent: 03ff34e. Status: complete for first-wave validation integration. Mapping: Sprint 038 M3, A1/A11 first wave.

## Plan and acceptance

1. Add a bounded numeric CPU worker for current squared and Normal recipes.
   Accept frozen train/validation packets, reject test arrays and unknown options.
2. Declare final/best selection and mean/scale output semantics in a JSON model
   bundle. Check fresh-process prediction without training data or recipe imports.
3. Exercise frozen housing packets in bounded subprocesses, preserving source,
   input and output hashes, failures and environment metadata.
4. Run focused negative/schema and direct-recipe parity tests, regression/lint,
   document limits and commit locally. No search/test-score or speed claims.

Smallest failing test: the current worker module is absent. All A1–A13 remain
required; this adapter does not close F0.3, E3 or the remaining D5 author checks.

## Results and reflection

The new current worker accepts A1 squared and A11 joint Normal, with explicit
validation labels and optional weights. Unknown inputs/options, CUDA and other
applications fail. It requires a one-thread process budget. Fixed-budget output
uses the final model; enabled patience selects the strict best validation snapshot.
Training metadata separates outer rounds, accepted commits and model identity.
The JSON evaluation bundle declares scalar mean or Normal mean/standard-deviation
semantics; a separate inference entry point needs neither recipes nor training data.

Fifteen focused tests pass: direct-recipe parity for both tasks with/without
patience, fresh-process exact replay, corrupt semantics and nine unsupported or
contaminated input cases. All 814 CPU tests pass, along with Ruff and strict docs.
All ten real validation cells pass: A1/A11 across five frozen housing folds,
four rounds, depth two and 32 bins. Exact fresh-process validation predictions
match, and all Normal scales are positive. Raw results and provenance are in
[current-worker-044](../benchmarks/v1/evidence/current-worker-044/README.md).

This resolves the missing current-implementation path into the evaluation harness.
It does not close a complete selected train/test workflow: no 16-configuration
search, test scoring, competitor comparison or formal OS label isolation ran.
The 32-bin/four-round settings are explicit plumbing choices, not the frozen
quality grid. No GPU or speed claim. Next extend A6/A13 scale/selection integration
and remaining application adapters alongside D5 and F0.3 judge/protocol closure.
See [learning](../learnings/2026-09-06-v1-current-worker.md).
