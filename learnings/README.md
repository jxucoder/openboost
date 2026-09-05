# OpenBoost Learnings

This directory is the repository's durable engineering memory. It records why a
change was made, what evidence supports it, which attempts failed, and what
remains unknown. It is deliberately separate from release notes and generated
benchmark output.

## When to Write an Entry

Create or update an entry when work includes any of the following:

- a non-obvious correctness fix;
- a benchmark, experiment, or falsified hypothesis;
- an architecture or product-scope decision;
- a dependency, platform, CI, packaging, or release incident;
- a user correction that should change future agent behavior.

Use `YYYY-MM-DD-short-topic.md`. Prefer one entry per coherent investigation;
append follow-up evidence rather than creating many tiny diary files.

## Required Content

Start from `TEMPLATE.md` and include:

- context and the question being answered;
- decision or result;
- files/behavior changed;
- verification and artifact locations;
- failed attempts and why they failed;
- remaining risks and next action.

Keep entries factual and concise. Do not include credentials, secrets, private
URLs, or copied raw logs. Link the relevant commit after it exists.

## Current Entries

- [2026-09-05-foundation-p4-splits-goal-review.md](2026-09-05-foundation-p4-splits-goal-review.md) — goal/value checkpoint, earlier independent CPU package validation, numeric split/routing and real T4 evidence.

- [2026-09-05-foundation-p4-histograms.md](2026-09-05-foundation-p4-histograms.md) — fixed-slot CPU/CUDA histogram contract, independent sample oracle and real T4 validation.

- [2026-09-05-foundation-p3-cpu-contract.md](2026-09-05-foundation-p3-cpu-contract.md) — experimental CPU objective/builder/schedule, plugin-free wheel inference, coefficient persistence and installation limits.
- [2026-09-05-foundation-p2-boundaries-baseline.md](2026-09-05-foundation-p2-boundaries-baseline.md) — real T4 boundary verification and frozen California Housing baseline; P2 complete.
- [2026-09-05-foundation-p2-correctness.md](2026-09-05-foundation-p2-correctness.md) — real T4 weighted regression before/after evidence, scoped RNG and explicit execution boundaries.
- [2026-09-05-foundation-p1-modal.md](2026-09-05-foundation-p1-modal.md) — isolated Modal wheel smoke, provenance checks, and explicit failure propagation.
- [2026-09-05-foundation-p0-integration.md](2026-09-05-foundation-p0-integration.md) — integrate unified trainer, preserve local fixes, and cover generic loading of the new models.
- [2026-09-05-gpu-python-foundation-design.md](2026-09-05-gpu-python-foundation-design.md) — scoped GPU Python foundation design, branch integration, experimental contracts, and medium execution checklist.
- [2026-09-05-impact-adoption-value-strategy.md](2026-09-05-impact-adoption-value-strategy.md) — impact/adoption/value research, current online versus local evidence, and proposed validation gates.
- `2026-08-15-repository-audit.md` — product focus, correctness risks, and
  evidence gaps found in the deep audit.
- `2026-08-15-scoringbench-integration.md` — third-party benchmark integration,
  validation, and Intel macOS runtime limitation.
