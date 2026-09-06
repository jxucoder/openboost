# 2026-09-06: Public callbacks support installed D2/D3 extensions

## Context

Sprint 038 M2 calls for testing actual public authoring boundaries before adding
more built-ins. Parent 2f6c77a. Historical extension wheels import retired APIs
and cannot establish current v1 support.

## Decision or Result

Two new development wheels implement independent cohort feasibility and penalized
leaves using public callbacks. Neither requires core edits, private imports or
recipe-loop copying. The core wheel is byte-identical to Sprint 039's wheel.
Installed checks validate split choice, leaf mathematics, subsequent updates and
plugin-free inference. These are repository-authored exploratory trials, not an
independent author cohort, full E2/E6 or measured E5 advantage.

## Changes

- examples/v1_extensions/: two packages, independent oracle checks, isolated
  build/install/remove/inference verifier and usage/limitations.
- tests/v1/test_public_extensions.py: source development oracle and import checks,
  explicitly distinguished from installed evidence.
- [Raw artifacts](../benchmarks/v1/evidence/installed-extensions-040/README.md):
  hashes, environment, commands, outputs and persisted models.
- [Sprint 040](../v1-sprints/040-installed-extensions.md), public docs and execution
  pointers record the bounded result and remaining work.

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_public_extensions.py -q -o addopts=''`:
  two passed, including the final routed-leaf strengthening.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  767 passed before that strengthening; affected focused checks passed afterward.
- `uv run --no-sync ruff check src/openboost examples/v1_extensions tests/v1/test_public_extensions.py`:
  passed; `uv run --no-sync mkdocs build --strict`: passed.
- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-extension-evidence-040-routed`:
  three offline wheel builds, fresh environment installation, isolated `-I`
  execution, two plugin uninstalls and exact fresh-process core inference passed.
- Thirty deterministic D3 oracle comparisons have maximum absolute error
  8.881784197001252e-16. D2 constrained cut 1 is selected by all three growers.
- Platform: macOS x86_64, Python 3.12.12, NumPy 2.3.5, one BLAS/OpenMP thread.
  No timing, GPU, broader-platform or real-data claim.

## Failed Attempts

- Initial imports failed because the new packages did not exist yet.
- The checker initially used `TreeTerm.tree`; the public field is `learner`.
  Corrected the checker, with no core compatibility shim or API change.
- Initial D3 integration only checked a root leaf; strengthened it to routed
  multi-leaf trees before recording final installed evidence.

## Risks and Follow-ups

Leaf replacement currently needs a grower adapter and duplicate quantile settings.
The package owns penalty/anchor; recipe defaults must remain unused by the custom
solver. This is documented authoring friction, not measured task-cost advantage.
Next D4 ordered updates, D1/D5 installed probes and real worker integration; preserve
held-out separation and every required application. No push or publication.

## Commits

- This development-extension commit; parent `2f6c77a`.
