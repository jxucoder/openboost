# Shared recipe results

`openboost.results.RecipeResult` is a structural protocol for completed recipes.
The scheduler accepts any object exposing:

- `state`: an AcceptedState belonging to the requested context and train/validation problems.
- `steps`: a tuple with one entry per completed outer round. Each entry's contents
  belong to the recipe; an ordered recipe may store a tuple of parameter substeps.
- `stop`: a completed StopState, with reason `budget` or `patience`.

Authors need not inherit a base class or convert their result into the built-in
FitResult. The scheduler retains the original object and its diagnostic payloads.
`validate_result(result, context=..., train=..., validation=...)` exposes the same
checks for callers outside run_many. A declared protocol alone is insufficient:
runtime validation checks field types, matching identities, completion and trace
length. Invalid results become retained per-run errors; other jobs continue.

Accepted-state version counts model commits, not outer rounds. Two parameter
commits can correspond to one trace entry and one patience observation. An
unfinished stop record cannot represent a successful run. This is an in-memory
contract, not a resume format or process-isolation boundary. External callbacks
must respect immutable input/state ownership; the scheduler does not copy or
interpret arbitrary diagnostic payloads.

The ordered-update development wheel now runs unchanged beside built-in squared
recipes. Installed checks cover M=1/8/32, shared preparation without refitting,
different validation stop rounds, reversed/regrouped execution, retries and an
isolated malformed result. These are sequential correctness checks, not batching,
speed, real model selection or a complete D5/E5 result.
