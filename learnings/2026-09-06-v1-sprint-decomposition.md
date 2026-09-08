# 2026-09-06: Decompose the remaining v1 plan into bounded sprints

## Context

The user requested a sprint breakdown of the retrospective/landscape plan.
Parent `df23796`. Sprint 064 previously bundled stopping and installed isolation,
while N2–N5 and CUDA remained broad milestones. This is planning work only.

## Decision or Result

The [064–084 roadmap](../v1-sprints/roadmap-after-063.md) defines 21 scope cards,
not 21 calendar weeks or a strictly serial schedule. Current work is 064 stopping;
065 installed isolation, 066 profiling, 067 incremental execution and 068 trace
retention each have their own outcome. Runtime edits depend on measured diagnosis.

069–077 separate author measurement, judging and every application's selected
quality. A6/A13 proves the shared real selection path first; other families keep
independent gates and source dependencies. Formal E5 needs F0/F1 prerequisites
and frozen interfaces/judges, not all lower-numbered quality sprints by convention.
078–082 separate scalar CUDA, Normal/public extension, required recipes, batching
and cost. 083 engineering acceptance and 084 external adoption are distinct.

The proposed early CUDA overlap remains unadopted. No GPU job, agent dispatch,
outreach, source retrieval, push or publication is implied. All R/C/A cases and
existing gate thresholds remain mandatory. A diagnosed failed experiment may
close a sprint record but cannot pass its gate or unblock a pass-dependent action.

## Changes

- Split the earlier 064 card, preserving its concrete contract design and installed
  cases in 064/065. Move the planned installed evidence destination to scheduling-065.
- Add the roadmap and 066–084 cards with entry dependencies, first falsifiable
  checks, work boundaries, acceptance, reflection and explicit not-run status.
- Update AGENTS, main-plan handoff, sprint navigation and retrospective links.
  Retain the earlier learning as a historical planning result with a supersession link.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict`
  passed. MkDocs does not render these planning cards; local targets were also
  checked independently with a standard-library script over changed/new Markdown.
- All 21 cards 064–084 are unique, linked from the roadmap, and carry acceptance,
  reflection and explicit not-run status. A1–A13 are individually named; manual
  review reconciled R/C/A and E gates with the canonical plan/protocol.
- Dependency review checked the original P7 raw-evidence README: it is a Normal
  distribution workload. Moved actual reproduction from scalar 078 to Normal 079;
  078 prepares the frozen protocol. Original methods/settings and threshold remain.
- All 186 local Markdown targets in 28 changed/new Markdown files exist. English
  prose, Markdown-only scope and `git diff --check` passed after the final card edits.

No production code changed; no new test, model fit, quality, CUDA or author pass
is claimed by this planning slice.

## Failed Attempts

The initial documentation split command used an unavailable `python` executable;
it made no edits. Re-running the same standard-library script with `python3` worked.
No implementation or evaluation experiment was attempted.

## Risks and Follow-ups

Later cards are conditional scope, not credible duration/compute estimates before
066 and the full-search preflight. New substantial defects should create focused
follow-up cards, not silently expand a sprint or shrink required workloads. Source
and independent-execution dependencies must stay explicit without stalling unrelated
work. Next implement 064 only when execution resumes; keep changes local.

## Commits

- This sprint-decomposition planning slice; parent `df23796`.
