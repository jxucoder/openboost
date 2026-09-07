# Shared recipe results

`openboost.results.RecipeResult` is a structural protocol for completed recipes.
The scheduler accepts any object exposing:

- `state`: an AcceptedState belonging to the requested context and train/validation problems.
- `steps`: a tuple with one entry per completed outer round. Each entry's contents
  belong to the recipe; an ordered recipe may store a tuple of parameter substeps.
- `stop`: a structural `openboost.stopping.StoppingStatus`, exposing integer
  `rounds`, integer `completed_rounds` and a nonempty terminal `reason` string.
  The default StopState implements this contract with `budget`/`patience` reasons;
  external policies can retain their own reason and diagnostic fields.

Authors need not inherit a base class or convert their result into the built-in
FitResult. The scheduler retains the original object and its diagnostic payloads.
`validate_result(result, context=..., train=..., validation=...)` exposes the same
checks for callers outside run_many. A declared protocol alone is insufficient:
runtime validation checks field types, matching identities, completion and trace
length. Invalid results become retained per-run errors; other jobs continue.

Counts must be Python integers (not booleans), with nonnegative budget and
0 <= completed_rounds <= rounds. The reserved reason `budget` requires every
budgeted round to be completed. None/empty reasons are unfinished/invalid results.
Other nonempty strings are allowed without interpreting policy-specific statistics.
The scheduler preserves the stopping object; it does not convert it into StopState.

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

Installed checks also mix the external loss-threshold policy with built-in and
ordered recipes, retaining true completion and payloads through M=1/8/32 schedules.
Distinct-ID RNG, same-ID recovery, stale feature/row preparation rejection and
valid target/weight reuse are checked without refitting shared binning. A model
from the custom loop replays after the policy source and training plugins are removed.

## Diagnostic retention

All twelve built-in recipes accept `retention="full"` (the default) or
`retention="summary"`. Summary converts each round immediately, keeping a
`TraceSummary` with `kind`, immutable named scalar `values`, and `omitted` field
names. Losses, acceptance, coefficients and failure outcomes present in the full
record remain available. Multi-output MSE vectors become per-output scalar tuples.
Gradients, geometry, directions and per-sample raw snapshots are omitted. Inspect
named values with `dict(step.values)`; array fields remain available in full mode.

Summary keeps final/current raw state and the selected best model unchanged,
and retains one entry per completed outer round. It does not disable validation,
alter stopping, remove model terms or summarize after first retaining a full run.
For fixed output width and trial budget, built-in history is O(rounds) scalar
records plus O(rows * outputs) run arrays and model/encoding storage; full history
can retain O(rounds * rows * outputs) arrays. Logical retained bytes differ from
process peak RSS; temporary training arrays still exist in summary mode.

External diagnostic payloads remain author-owned. The structural result validator
does not rewrite them. Authors may explicitly construct `TraceSummary`; its values
reject arrays, states, mappings and models, allowing scalars and scalar tuples.
The installed ordered-update example opts into this mode itself: each outer entry
contains two scalar substeps, including commit versions, without old state objects.
Outer-round counts remain distinct from accepted commit counts.
