# 2026-09-07: Defer agent evaluation and return to foundation engineering

## Context

After local active-cancellation construction at `2e446d6`, the user questioned
why OpenBoost needed AI model calls and asked to skip agent friendliness for now.
Those calls supported a future authoring study, not boosting execution. The next
proposed cancellation test had not been approved or run.

## Decision or Result

Defer agent trials and their evaluation infrastructure. Keep composable public
components, correctness and installed-extension checks. F2/E5 remains unpassed
but no longer blocks required CUDA construction, train-many or scoped quality/cost
work. Resume author evaluation only at the user's direction. The next engineering
checkpoint is the existing Normal CUDA comparison correction, whose hardware
allowance is still pending.

## Changes

- [Sprint 101](../v1-sprints/101-defer-author-evaluation.md) records the user-directed
  sequencing change, retained scope and concrete engineering checkpoints.
- Canonical agent guidance, the main plan and sprint navigation identify the
  current priority instead of directing continuation into accounting work.
- The evaluation protocol preserves original thresholds and explicitly marks
  agent benefit as deferred, without claiming that its gate passed.
- Historical packets, evidence, production code, tests and dependencies are
  preserved. No external model call, worker upload or GPU run occurs.

## Verification

- Local Markdown link validation passes across all seven changed documentation
  files. `git diff --check` passes.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build`
  passes with the existing `execution.md` link warning for the 090 evidence page.
- All 85 run-8, 24 worker-smoke-097 and 17 cancellation-smoke-100 source hashes
  match their frozen inputs. Their packet bytes are unchanged from `2e446d6`;
  authorization remains pending, consumed and pending respectively.
- Changed-file inspection confirms that evidence, authoring support, production
  code, tests and dependencies are unchanged. No runtime suite is rerun for this
  documentation-only decision; prior test results are not new execution evidence.

## Failed Attempts

Treating complete author accounting as an immediate prerequisite prolonged
evaluation preparation without resolving the foundation's device correctness
questions. Keep the existing results, but do not let sunk preparation cost dictate
the next engineering priority.

## Risks and Follow-ups

Agent benefit and adoption remain hypotheses. All application families, required
device subsets and mathematical/state/persistence obligations remain in scope.
Run 8 requires its existing concrete allowance; this planning change does not
authorize it. Preserve the retrospective after that run and do not automatically
resume the paused authoring study.

## Commits

- `2e446d6` — last local author-accounting slice; its live packet remains unapproved.
- This commit records the deferral and revised engineering order.
