# Sprint 011: F0.3 evaluation artifact integrity judge

Starting revision: `bc68ab9`. Status: complete for the F0.3 integrity subset only.

## Plan and acceptance

1. Create versioned `benchmarks/v1/` integrity contracts: expected cells, run identity, cache keys
   and artifact hashes. Smallest failure: a missing required fold fails even if all remaining cases say pass.
2. Offline judge/CLI with strict JSON and counterexamples for missing/duplicate/unknown cases,
   cache pollution, NaN, wrong backend, worker error, timeout, unrun GPU and missing/corrupt predictions.
   Synthetic artifacts are test inputs, not frozen real data.
3. Full regression/lint/docs, scope record and commit. Output integrity_pass only, never an E-gate pass.

No training, real quality evaluator or invented data/library/resource freezes. F0.3 remains incomplete;
next slices must advance real data and baseline preparation.

## Results and acceptance

Added48 tests; 336 total. Cases cover a missing second fold with all A-IDs present, duplicate/unknown
cases, eight changed cache inputs, non-finite metrics/predictions, false passes, nonzero workers,
missing/corrupt files, path/symlink escape, optional unsupported GPU and CLI exit codes.
Every A-ID needs a required CPU cell, but that structural check is not a sufficient experimental design.
Every declared cell needs a record; optional failures remain visible. Only integrity_pass can pass;
gate_results stays empty.

Cache keys bind the complete manifest and cell. Only dirty=false is supported because a boolean
cannot identify uncommitted patches; dirty runs need content digests. No provenance authenticity
claim is made. Rectangular finite prediction JSON is checked, not target alignment/units or recomputed metrics.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

Local macOS CPU/Python 3.12.12; no real training, CUDA or quality result. Manifests are explicitly temporary synthetic inputs.

## Reflection

Mathematical references exist, but trusting producer pass records cannot exclude missing folds,
stale caches or worker failures. Build an independent integrity check with distinct E-gate fields.
Initial collection failed on the absent module; counterexamples passed after implementation.
Review added a missing second A1 fold: deleting the only A13 record alone did not test repeated folds.
Next pursue real sources/licenses/hashes and split adapters; schema growth is not real evaluation progress.
All A1–A13 remain required. Budgets, baseline capabilities, held-out tasks and full judge remain unfrozen.

## Commits

- This slice: `test: add v1 artifact integrity judge`.
