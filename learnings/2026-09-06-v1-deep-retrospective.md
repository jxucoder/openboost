# 2026-09-06: Make the next foundation milestones resolve product uncertainty

## Context

The user requested a deep retrospective and plan after PR #24 merged at
`47108db5d9fe3157e59621115860aa1dd06e9bf4`. Sprint 062 already listed incomplete
CPU/GPU gates; repeating that list would not explain how execution should change.

## Decision or Result

The [retrospective](../v1-sprints/063-retrospective-and-next-plan.md) preserves the
shared semantic foundation and all required cases. Move from repeated short-fit
coverage toward installed scheduling probes, practical state/trace execution,
comparative authoring measurement and a complete real selection workflow.

A tiny [operation-count diagnostic](../benchmarks/v1/evidence/runtime-audit-063/README.md)
confirms a known correctness-first tradeoff remains active: always-accepted squared
fits replay trees `3*T*(T+1)` times, and joint Normal fits twice that number.
Distinct retained step-array bytes grow with rounds. Full final raw replay agrees
exactly. These are operation counts/logical array sizes, not wall time, peak memory,
quality, real-workload dominance or CUDA measurements.

Recommend bounded B12 feasibility after installed D5 and relevant state/diagnostic
ownership checks, overlapping unfinished author/quality work. This is a proposed
sequencing amendment, not approved execution or a phase pass. Current F2→F3 order
remains active. Formal E5 revisions/cohorts must stay frozen; all E0–E7 thresholds
and R1–R9/C1–C7/A1–A13 requirements are unchanged.

## Changes

- Sprint 063: retrospective, individual application obligations, N1–N5 milestones,
  acceptance/failure decisions and the explicitly proposed CUDA overlap.
- Runtime diagnostic and raw counts with production/diagnostic source hashes.
- Concise current navigation in AGENTS, sprint index, active plan and README;
  retain historical records and remove stale current handoff instructions.
- Rechecked official objective/leaf API documentation and Py-Boost's stated scope;
  no opponent installation, benchmark or frozen-version change.

## Verification

Using `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`:

- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  923 passed on Python 3.12.12 / NumPy 2.3.5 / macOS.
- `uv run --no-sync ruff check src/openboost benchmarks/v1/runtime_cost_audit.py`:
  passed; `uv run --no-sync mkdocs build --strict` and `git diff --check`: passed.
- Checked 160 local Markdown targets in changed documents and all 20 recorded
  source hashes: no missing targets or stale hashes.
- The diagnostic ran eight tiny squared/Normal CPU fits with exact raw replay;
  the evidence README records its exact command and measurement boundary.

No production files changed, real-data training, GPU job, formal author attempt,
held-out inspection, publication or new external contact occurred.

## Failed Attempts

Initial diagnostic lint flagged loop-variable closure binding; bind the instrument's
counter and original method explicitly before execution. All eight fits then passed.
A CatBoost documentation URL returned an error; the canonical documentation path
succeeded. No failed measurement was omitted or threshold relaxed.

## Risks and Follow-ups

The operation counts do not replace the earlier full-input Covertype profile or
establish the dominant practical cost. N2 must profile bounded longer runs before
claiming throughput or memory benefits. Phase-gate source/workflow gaps remain.
Authoring/adoption is still unvalidated; repository-authored packages are not users.
The next implementation is N1 installed D5. Further agents/outreach and an early
CUDA phase overlap are not authorized by this retrospective itself.

## Commits

- This retrospective slice; parent `47108db`.
