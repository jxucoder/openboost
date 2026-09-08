# 2026-09-07: Foundation progress and the next evidence priorities

## Context

The user requested an explanation and plan before further execution. Baseline
`8a70e0b` has a clean checkout and a pending run-8 freeze. This review changes
planning only; it does not approve or launch that run.

## Decision or Result

Implementation is ahead of product validation. CPU components and twelve recipes
exist; bounded scalar CUDA is verified at its recorded revision. Normal/D2 has
real execution evidence with two known acceptance failures. The current correction
is CPU-verified and awaiting CUDA validation. Independent author cost, full real
quality, GPU cost and adoption remain open.

Close the existing comparison validation cycle, then prioritize the already
planned D1/D2 accounting/isolation pilot and a specifically frozen real-workload
cost question alongside remaining required construction. Keep all applications,
formal thresholds and paused-search constraints. See the
[checkpoint plan](../v1-sprints/093-foundation-progress-and-next-steps.md).

## Changes

- Added a checkpoint mapping implementation to evidence and proposed next steps,
  with per-step exit criteria and links to existing execution cards.
- Recorded an author-material gap: the old packet predates device construction,
  and `docs/v1/index.md` still calls training CPU-only. Refresh and audit the
  author view before independent attempts; this review does not edit frozen sources.

## Verification

- Re-read current code paths, public docs, priority amendments, author readiness
  and canonical evaluation requirements. Source review confirms Python numba-cuda
  kernels and CuPy-owned storage; it does not establish GPU performance.
- Independently checked every artifact hash and JUnit counts in the three
  recorded hardware archives: run 5 is 212/212; run 6 is 381/383; run 7 is 383/385.
  All archive hashes match. Both latter runs retain two failures and no skips/errors.
- Confirmed run 8 has both authorizations pending and no output directory.
  The 1794 CPU passes/one skip are the previous recorded run, not a fresh test run.
- Planning links and diff hygiene are checked; MkDocs builds with its existing
  execution-page evidence-link warning. No new runtime validation is claimed.

## Failed Attempts

- The first documentation lookup used `docs/execution.md`; discovery resolved
  the current page to `docs/v1/execution.md`. No implementation or evidence changed.

## Risks and Follow-ups

Passing the revised comparison gate would still leave real-scale performance,
other required CUDA cells, train-many batching and author/application gates open.
The proposed real-workload study needs a concrete freeze at retrospective and
any required execution allowance. Existing author-view isolation is not an OS
sandbox, and generated-token accounting is still missing.

## Commits

- `8a70e0b` — baseline request and current guidance.
- This entry accompanies the requested planning checkpoint; no sources are refrozen.
