# 2026-09-08: Present the foundation's current state and vision

## Context

After PR 25 merges as `167ed53`, the user asks to update the README. Its appended
checkpoint narrative mixes early CPU construction, old CUDA failures and newer
results, obscuring both the programmable-foundation vision and present capability.

## Decision or Result

Present the public component model first, then distinguish implemented CPU scope,
bounded CUDA evidence, real-workload evaluation gaps and next construction. Keep
all required application families visible. Agent productivity is a hypothesis;
comparative authoring studies remain deferred. CPU remains the semantic reference
and usable development path, with explicit Python-authored CUDA execution.

## Changes

- Rewrite `README.md` with the vision, CPU/CUDA recipe matrix, linked run-11/run-12
  evidence, a public learner example, and the required recipe/train-many/cost work.
- Replace the incomplete run-10 timing headline with the complete run-11 internal
  comparison, retaining the earlier failure link and precise limits on speed claims.
- Add a runnable CPU example that replaces growth/feasibility through a learner
  callback and saves/loads the best model. It does not imply automatic GPU dispatch.
- Record the completed PR merge in the 109 sprint, execution index and agent guide.
  The local documentation branch starts from merged main; no new push is requested.

## Verification

- Execute the exact README Python block in a temporary output directory: three
  accepted rounds, three-node stumps, improved training loss, finite `(2, 1)`
  predictions and exact saved-model prediction replay all pass.
- All 34 relative README links resolve. English-prose and whitespace checks pass.
- Ruff passes production and the extracted example.
- `uv run --no-sync mkdocs build --strict` passes.
- `uv build --offline` builds wheel/sdist; the wheel metadata contains the complete
  updated README verbatim.
- Capability and cost statements are checked against public code/tests and the
  committed run-11/run-12 reports. No production, frozen source or raw result changes.

## Failed Attempts

The first temporary validation script used `TreeTerm.tree` to inspect stump size;
the actual public field is `TreeTerm.learner`. Correcting that extra check passes.
The README example itself already executed, trained and replayed successfully.

## Risks and Follow-ups

The README describes the merged experimental checkpoint, not complete v1 acceptance.
No new GPU, external-baseline, quality or author study is run for this update.
Refresh the capability matrix only when the next construction has its own evidence;
keep test counts and internal timings separate from product/adoption claims.

## Commits

- `167ed53`: merged PR 25 and preserved execution history.
- `bbd69ea`: the verified implementation/test checkpoint described in the README.
- This entry accompanies the README documentation commit.
