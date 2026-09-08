# Sprint 077: Formal E5 and the F2 decision

Status: planned. Mapping: N5 / B11 / F2 / C6–C7 / E5.
Entry: 069 accounting smoke, 070 independent judging/coverage audit, completed
required F0/F1 CPU capabilities/workflows and frozen interfaces/judges. It need
not wait for all E3 results in 072–076. Independent execution must be authorized.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Measure whether researchers/agents can make verified changes with less work using
OpenBoost. First ensure a candidate cannot modify its judge, observe another arm's
patch, exceed its declared budget silently or access sealed tasks ahead of its attempt.
The foundation designer must not inspect H1/H2 contents or author their verifiers.

## Work

- Freeze five D modification types and two sealed H types, the appropriate opponent
  per task, model/settings/tools/docs/prompts, source/wheels and accounting policy.
  Allow normal incumbent hooks, outer loops and source edits. Retain the control.
- Run three independent attempts per task/arm: 15 development and six held-out
  attempts for each arm. Cap each at 30 minutes or 20k generated tokens, whichever
  comes first; rotate arm order and prevent cross-attempt leakage.
- Record install/active/blocked time, first correct completion, assistance, tokens,
  compute, core/private edits and all failures. Failures count full time budget;
  also report successful-only results separately, never as the main cost gate.
- Use the authorized independent evaluator for sealed material within its approved
  scope. Additional participant/agent execution requires authorization; a prepared
  cohort is not a completed experiment. Exposed tasks become development.

## Acceptance and reflection

Apply the unchanged E5 gates: OpenBoost at least 12/15 development correct and
at least 2/3 per type; at least 2/3 on each held-out and 4/6 total; no core/private
dependencies. At least two deep-change types reduce capped median completion time
by at least 30% against the appropriate opponent without fewer correct completions.

Publish full attempt records and independent verification, preserving small-sample
limits. If only controls improve, inspect documentation/opponent choice. If changes
need core edits, revise the boundary in a new development sprint and freeze a new
cohort; do not overwrite failed attempts or add hints and call them clean successes.

Record the formal F2 pass/fail and implications for GPU entry. No external adoption
claim follows from internal agent trials. Independent E7 remains Sprint 084.

## Results

Not run. Existing installed development packages are not formal E5 attempts.
