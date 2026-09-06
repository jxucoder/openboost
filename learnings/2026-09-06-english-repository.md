# 2026-09-06: English repository prose

## Context

The user requested that all repository files be in English while A2 evaluation
preparation was in progress. This applies to existing prose and future work.

## Decision or Result

Translate documentation, plans, sprint records, and historical strategy prose.
Preserve identifiers, mathematical notation, literal dataset values, raw evidence,
and historical decision boundaries. Require English in canonical AGENTS guidance.
The active foundation scope and evaluation-first execution order remain unchanged.

## Changes

- [Sprint 015](../v1-sprints/015-english-repository.md) records the plan, checks,
  results, and reflection.
- Active planning/evaluation/application documents and historical GPU plans now
  use English, along with the existing sprint records and strategy learning.
- Three stale links to retired implementation files now point to their locally
  verified historical revision `05cd8bc800595a2f40c4d08f51afb697968b9b3e`.
- Adult data preparation was committed separately as `594519f`.

## Verification

- Repository text inventory: no remaining CJK ideographs; remaining non-ASCII
  letters are mathematical notation. Ignored build caches and binary files are
  outside the repository prose audit.
- Source URL/hash comparison, Markdown local-link/anchor/fence checks, and
  `git diff --check`: pass. Link validation does not assert remote availability.
- Full current suite: 396 passed, no skips. Ruff and strict MkDocs build: pass.
  Exact commands are recorded in Sprint 015. Environment: macOS, Python 3.12.12.
- Benchmark raw artifacts and executable code were not changed by translation.

## Failed Attempts

- An initial simple link regex misread a mathematical expression as a link.
  Replaced that check with Markdown parsing, including fenced-code handling.
- The link audit found three existing targets removed during production retirement;
  verified their contents exist in the historical Git tree and pinned the links.

## Risks and Follow-ups

- Text/structure checks complement semantic translation review; they are not a
  proof that all prose is equivalent. Requirements and quantitative gates remain
  authoritative and must not be relaxed during execution.
- Complete F0.3 evaluation preparation next; no new production or CUDA claim.

## Commits

- This slice: `docs: use English throughout repository prose` (parent `594519f`).
