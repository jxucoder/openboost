# Sprint 070: Scratch vector scoring diagnostic

Clean source revision: `21f76b1`. All 32 source hashes and six returned artifact
hashes verify. The installed worker uses the unchanged approved Parkinsons packet,
shared fold-zero configuration and 60-second diagnostic, with stack dumps disabled.
The child exits 124 at the intended soft deadline and retains raw pstats.

## Measured call-path change

Public vector_leaf runs 77 times and newton_leaf 154 times, consistent with actual
leaf construction rather than per-candidate temporary artifacts. The prior layout
profile records 904838 vector_leaf calls in its separate instrumented prefix.
Raw pstats retains the small-call functions omitted from the top-100 JSON listing.

| Call path | Calls | Cumulative seconds |
| --- | ---: | ---: |
| choose | 43 | 49.397 |
| vector_score | 352465 | 32.586 |
| vector_feasible | 371453 | 15.106 |
| candidates | 43 | 9.788 |
| _newton_value | 2114942 | 7.598 |
| _nonnegative | 3191482 | 7.456 |
| _owned | 752376 | 5.622 |
| _vector_indices | 723995 | 2.991 |

Times overlap; do not sum them. This sample scores 352465 candidates versus 301590
in the prior layout profile. Instrumentation, host variation and different prefixes
prevent an end-to-end speed claim from these observations.

## Correctness and next step

Forty-two focused checks include exact score/tie/near-tie behavior, malformed leaf
inputs, and identical model bytes/predictions across all growers. Full CPU suite:
1077 passed, one Linux-only skip; lint/docs pass. Public leaf ownership, callbacks,
feasibility, division and dot/gain ordering remain unchanged.

Next run a bounded paired real fit using the original scoring baseline and current
code in the same environment, with fixed packet/configuration, identical selected
models/predictions/stopping and complete fit/replay/resource accounting. Do not
continue stacking micro-optimizations or expand the full search based on profiles.
No formal quality, authoring, GPU or full-search resource gate is passed here.
