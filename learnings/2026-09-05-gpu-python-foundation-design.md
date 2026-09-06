# 2026-09-05: GPU Python Foundation Design

## Context

After the impact/adoption/value research, the user asked whether a GPU Python
boosting foundation was promising, then requested branch confirmation and a
design-first handoff for later implementation with a medium model. Modal use
was explicitly authorized. This turn is planning, not implementation.

## Decision or Result

- Created `codex/gpu-python-foundation-design` from clean local `main` at
  `82cf1e25b21a69093e85a270af7eb93c9ae7aa19`; local main was not moved.
- Fetched origin/main, now `6ebe3a8ced0e621b17e3cf63e31721af58471053`.
  At branch creation the histories had 12 local-only and 9 remote-only commits.
  The execution plan starts by merging these histories, preserving both local
  correctness fixes and the remote unified trainer/FormulaBoost/WeibullAFT.
- Proposed an experimental, single-GPU, dense-numeric substrate with objective,
  tree/leaf, and scheduled coefficient extension points. Reuse the existing
  trainer instead of creating another training loop.
- Require CPU mathematical references, real CUDA parity, two separately
  installed extension packages, explicit device/fallback reporting, coefficient
  persistence, and source-linked raw artifacts. External adoption remains a
  separate product gate; self-authored packages do not prove it.
- Keep the repository mission and broad experimental capability limits intact.

Static inspection found that the extensible primitive path downloads full
histograms/sample node IDs, and `compute_leaf_values_gpu` downloads inputs to
delegate to CPU. The native path bypasses the extension strategy. The design
therefore calls for device batch primitives and explicit dispatch semantics.

Remote trainer inspection also found a possible mismatch between weighted
Hessians and `const_hess=1` selected using `unit_hessian`. This is a hypothesis
requiring a weighted CUDA regression, not an independently reproduced GPU bug.
Name-based device dispatch and broad exception fallback also need tests.

## Changes

- [Architecture design](../planning/gpu-python-foundation-design.md): target
  contracts, scope, persistence, evidence thresholds, Modal execution plan,
  and conditions under which to stop expanding the foundation.
- [Medium execution checklist](../planning/gpu-python-foundation-execution.md):
  P0–P7 tasks, initial failing tests, merge boundaries, verification, and handoff.
- [Learning index](README.md): added this design decision.

No production code, package configuration, model behavior, or CI was changed.
No actual merge, GPU job, deployment, external message, or push was performed.

## Verification

Read the local implementation, tests, public docs, canonical AGENTS and audit,
and the remote trainer/objectives/Modal runner using `git show` at the fetched
revision. A read-only `git merge-tree` inspection identified text conflict
markers in CLAUDE.md and the GPU setup/installation documents; actual merge
resolution remains P0 and may require additional semantic reconciliation.

Baseline command, run before writing the design:

```bash
OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/test_extensibility.py -n 0 -q
```

Result: **35 passed, 1 skipped in 0.84s**, Python 3.12.12, Intel macOS.
The skipped test requires GPU. This is only the existing extensibility suite,
not validation of the fetched trainer, new design, or CUDA correctness.

Documentation checks passed for 4 Markdown files, 9 relative links, balanced
fences, 1 Python contract block, and all 7 learning sections. `git diff --check`
passed. Checks use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache` because the
default uv cache is not writable in this sandbox. These files are outside
MkDocs navigation; no production-code regression test was invented for a
reversible design edit.

## Failed Attempts

- The initial ahead-only interpretation used stale origin/main information.
  Fetch showed both histories diverged. The plan now pins both SHAs and starts
  with integration instead of rebuilding functionality already implemented.
- An attempted lookup of remote `tests/test_unified_engine.py` found no such
  file. Actual remote model tests are `test_formula.py`, `test_survival.py`,
  `test_distributional.py`, and the CUDA verification harness. Do not invent
  existing coverage from design-document labels.

## Risks and Follow-ups

- No real GPU validation, Modal credential/quota check, timing, or current
  billing estimate was performed. Run the bounded smoke after harness work.
- Device-resident custom tree building, safe native dispatch, and model
  persistence under nonconstant coefficients are implementation work, not
  completed capabilities.
- The suggested 6–8 week window and numeric quality/performance thresholds are
  proposed experiment budgets, not measured results or delivery guarantees.
- Contacting external developers has not been authorized; prepare runnable
  materials without sending messages. External adoption may remain unverified
  after the technical checklist completes.

## Commits

- `82cf1e2` — local starting commit, impact/adoption/value research.
- `6ebe3a8` — fetched remote integration target, not merged in this design turn.
- This design and learning entry are committed together; locate the design
  commit with `git log -- planning/gpu-python-foundation-design.md`.
