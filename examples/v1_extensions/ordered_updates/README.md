# Ordered Normal and Formula updates

`ob-ordered-updates` is a repository-authored D4 development package using public
OpenBoost CPU operations. `normal(...)` supports ordinary and Fisher directions;
`formula(...)` uses the damped full GGN direction. Both accept `order=(0,1)` or
`(1,0)`, rounds, bins, explicit prepared input and validation patience/min_delta.
The built-in OpenBoost recipes continue to use joint updates.

Each parameter recomputes geometry from the latest accepted state, fits one
depth-two scalar tree (or a caller-supplied learner), and tries at most six rates
`0.1 * 0.5**j`. Only finite strict training-loss descent commits. Numerical trial
failures are recorded without changing accepted models, raw values or RNG. Invalid
learner construction fails outside the numerical search. Later parameters continue
from the previous accepted state even if the preceding parameter fully rejected.

Validation chooses the strict best model after each accepted parameter. Patience
observes once after a full sweep. Thus accepted version, parameter attempts and
completed outer rounds are distinct. Substep records retain before/after immutable
states and trial outcomes for diagnosis; this is not a memory-optimized trainer or
resume checkpoint.

The public `sweep` operation accepts geometry, direction, loss, order and learner
callbacks. The `fit` loop is objective-independent; convenience wrappers bind
Normal/Formula geometry. No private imports or OpenBoost core edits are required.

## Reproduce

Run `uv run --no-sync python examples/v1_extensions/verify.py OUTPUT_DIR` from
the repository. The source-side reference generator creates immutable comparison
traces before the isolated installed package is run. The installed checks receive
these traces, not the reference implementation. The verifier records all source,
reference and wheel hashes. After removing all three extension packages, a new
interpreter verifies exact predictions from all eight saved models.

Source tests additionally cover nonzero offsets, both parameter orders, comparison
with joint updates, full and partial rejection, recovery after a nonfinite trial,
invalid order and outer-round stopping. These are exploratory development checks,
not timed E5, held-out evidence, real quality or adoption.

## Scheduler integration

OrderedResult contains state, nested per-round substeps and stop metadata. Sprint
041 recorded its rejection by run_many. Sprint 042 fixes that boundary with the
structural RecipeResult protocol: no extension modification or conversion is
required. The installed checker verifies independent/scheduled equality for all
six ordered cases. Mixed built-in/ordered M=1/8/32 jobs also verify shared input,
different stopping, failure isolation, reordering, regrouping and retry.
Historical Sprint 041 artifacts retain the original failure; current passing
evidence is recorded separately. Full D5 author evaluation remains open.
