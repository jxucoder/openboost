# OpenBoost

**A programmable boosting foundation for researchers and AI agents.**

OpenBoost v1 is being rebuilt around composable algorithm components, ordinary
Python recipes and explicit CPU/CUDA execution. The goal is to reduce the cost
of making a correct, reproducible algorithm change.

## Current state

This checkout is **under construction**. The old production implementation has
been retired; the package currently provides a namespace, not a training API.
Independent data, scalar/tree, classification, ranking, quantile and vector-leaf
references have 122 passing tests. They are
correctness preparation, not proof of a completed foundation or product parity.

- [Execution and reflections](v1-sprints/README.md)
- [Construction design](planning/foundation-construction-design.md)
- [v1 plan](planning/agent-boosting-foundation-plan.md)
- [Required tasks](planning/foundation-tasks.md)
- [Acceptance and evaluation](planning/openboost-v1-evaluation.md)

All R1–R9 / C1–C7 / A1–A13 remain required. Classification, regression, ranking,
quantiles, multi-output, count/positive/aggregate targets, survival, distributional
and formula models, and train-many each need their own implementation and evidence.

## Development

```bash
uv sync --extra test
uv run pytest tests/ -n 0 -q
uv run ruff check src/openboost tests/v1 tests/conftest.py
uv build
```

Python 3.10+. Current tests are CPU-only reference checks. CUDA is a future
required execution subset, not an implemented capability of this reset checkout.
GPU and publishing workflows stay unavailable until their v1 gates are met.

## Historical implementation and evidence

Revision `50acfc6` is the last revision containing the old production code plus
Sprint 001 references. Use that revision in a separate checkout to reproduce
old APIs, examples and experiments; no compatibility layer remains here.

Historical tests, examples, documentation and benchmark artifacts are retained
as evidence and sources of mathematical counterexamples. Default test discovery
runs `tests/v1/` only. Old tests are not counted as v1 passes or skips.
The current documentation build uses `docs/v1/`; other documentation describes
the retired implementation. Published packages and historical results do not
establish the new architecture's quality, speed or adoption.
