# Sprint 039: Independent validation-driven stopping

Parent: 7a61744. Status: complete for the bounded CPU slice.
Mapping: Sprint 038 M1, R9/C4/A13, F1.4 and D5 prerequisites.

## Plan and acceptance

1. Add an immutable public stop record with finite smaller-is-better validation
   observations, optional positive patience, nonnegative min_delta and round budget.
   Keep it separate from accepted model state and strict best-model selection.
2. Observe once after each logical recipe round, including full rejection, in all
   twelve recipes. Return completed rounds and budget/patience termination reason.
3. Verify hand-calculated sequences and real M=1/8/32 heterogeneous stop isolation,
   including reordering, regrouping, retry, failures and shared preparation.
4. Update public examples, run relevant regression/lint/docs/build, reflect and
   commit locally. No GPU, fusion, speed or formal phase-exit claim.

First failing test imports the missing public StopState. Patience compares each
observation with the last qualifying improvement; strict improvement must exceed
min_delta. Ties consume patience. Initial validation is the baseline and consumes
no round. Budget zero completes immediately. Patience wins the reason if both
limits are reached on the same round. Best-model selection retains its existing
strict minimum independently of the patience threshold.

## Results and reflection

StopState is a public immutable operation independent of AcceptedState. All twelve
recipes accept scalar patience/min_delta and return the final stop record in
FitResult. RunSpec already forwards these scalar options without a scheduler or
objective branch. Validation stopping observes accepted raw values once per outer
round; existing training-loss acceptance and strict validation best snapshots are
unchanged. Current and best models remain explicit in the result.

Thirty new tests verify threshold/tie/budget sequences, invalid and nonfinite
observations, zero-budget behavior in every recipe, improving training with
worsening validation, strict best selection below min_delta, and six rejected
trials per round consuming only one patience observation. M=1/8/32 scalar/Normal
jobs with different real validation stop rounds agree with independently scanned
fixed-budget prefixes, reversed/regrouped runs and retries. Failed configuration
and nonfinite validation runs retain their errors; shared preparation never refits.

CPU regression: 765 passed. Ruff, strict MkDocs, offline wheel/sdist and twenty
installed-wheel documentation examples pass on macOS/Python 3.12.12/NumPy 2.3.5.
Wheel SHA256 is recorded in the linked learning entry.

Observation: stop progress and model commit progress are different clocks.
Evidence: two fully rejected rounds attempt twelve steps while accepted version
remains zero; two worsening-validation accepted rounds retain the initial best
model. Decision: keep a separate public stop operation, which external ordered
loops can call at their declared outer-round boundary.

This verifies CPU semantics, not training resume, real selection, GPU/fused
execution or performance. Metric observation adds an explicit validation-score
evaluation over cached raw predictions; inference/metric cost remains a profiling
follow-up. Next execute Sprint 038 M2 installed D2/D3 extensions and ordered
updates, with M3 current OpenBoost real-data integration. Formal phase gates remain
open. See [learning record](../learnings/2026-09-06-v1-independent-stopping.md).
