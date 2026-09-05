# 2026-09-05: Make foundation construction an explicit design deliverable

## Context

After the F0.1 task-card handoff, the user asked whether the plan included how
to build the foundation. The main plan contained architecture principles and
F1 stages, but the latest deliverable and summary emphasized tasks/evaluation.
They did not give an executor enough concrete module, data and operation detail.

## Decision or Result

Add an engineering construction design, not another scope reset. Task cards
define required behavior; the construction design defines how to implement it;
independent references and evaluation check the result. All A1–A13 remain required.
The architecture is a proposed implementation choice, not an implemented API.

Choose ordinary Python recipes over a universal trainer, separately replaceable
split and leaf statistics, routed leaf row access, scalar/vector tree payloads,
versioned candidate transactions and explicit run/device ownership. CPU starts
with NumPy and suitable CPU kernels; CUDA uses resident arrays and bulk operations.
State ownership and device residency are designed before attempting fusion.

## Changes

- [Construction design](../planning/foundation-construction-design.md): public
  module dependencies; data/binning/identity and eight small records; explicit
  weighting, histogram layout and memory budgets; nine operation contracts;
  three growth algorithms; topology/payload/mapping and transactional updates;
  all A1–A13 composition paths; CUDA and independent/batched run semantics;
  inference format and B01–B14 implementation slices with failure fixtures.
- [Main plan](../planning/agent-boosting-foundation-plan.md) and
  [task cards](../planning/foundation-tasks.md): link the construction design and
  distinguish independent oracle preparation from F1 product implementation.
- [Agent guide](../AGENTS.md) and learning index: make this design discoverable
  for later execution without claiming new components already exist.

## Verification

- Before editing, confirmed the construction document did not exist; reviewed
  the main architecture/F1 stages, eval gates, existing core primitives/growth,
  experimental contracts, their public documentation and dispatch/boundary tests.
- Documentation validation via `UV_CACHE_DIR=/tmp/openboost-research-uv-cache
  uv run --no-sync python` passed: six Markdown files, 57 resolving local links,
  balanced fences, 13 application composition paths, 14 build slices, eight
  data/state records and nine operation contracts. All canonical entry points
  link the new design; F0.2/F0.3 and all six F1 substeps remain pending.
- `git diff --check` passed. Review corrected the symmetric-growth outline to
  align and aggregate all candidates before selection, rather than combining
  per-node winners; added prediction-time binning identity validation. Retained
  minimum-valued cuts: `[0,0,0,1]` with two bins has a valid cut at zero, which a
  strict-interior-only cut rule would incorrectly remove.
- No production code or GPU job changed. Documentation checks cannot establish
  runtime correctness, quality, speed or ease of algorithm authoring.

## Failed Attempts

- Treating task cards plus a minimal interface sketch as the complete planning
  handoff left the actual construction path implicit. Naming component layers
  without their inputs/outputs, data ownership and connection paths was insufficient.
- Existing experimental contracts apply objective weights and use unhalved gain;
  the new specification uses an explicit statistics adapter and half-gain convention.
  Copying old test numbers unchanged would conceal a semantic mismatch.

## Risks and Follow-ups

- New data layout, candidate transactions, serialization and batching are design
  decisions that must face F0.2 oracles and F1 use cases before F2 interface freeze.
- F0.2 and F0.3 remain next. F1 starts with minimal state and manually specified
  trees, then real scalar grow and complete GBDT/Normal recipes; Formula/run-many
  probe those boundaries early. Every remaining application still needs implementation.
- Performance costs, artifact readers and external extension usability remain
  unverified. No changes to E0–E7 thresholds or backward-compatibility requirements.

## Commits

- `be373e6` — prior task-card and baseline audit slice.
- This slice: `docs: specify how to construct the v1 foundation`.
