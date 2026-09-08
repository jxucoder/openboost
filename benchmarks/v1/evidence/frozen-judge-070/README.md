# Sprint 070 frozen execution-manifest smoke

[smoke.json](smoke.json) records clean revision `63700db`, source hashes,
environment and nine synthetic fault cases. One valid case passes; all eight
injected faults fail anchored integrity. Rehashed fold omission and code change
still pass producer-relative integrity, illustrating why the external freeze is
needed. gate_results stays empty in every case.

Reproduce with `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.frozen_judge_smoke /tmp/openboost-frozen-smoke-070` using a fresh directory.
Source hashes were independently checked against the recorded Git revision.

The manifest hashes and model/prediction bytes are intentionally synthetic fixtures,
not real data, real provenance or trained models. Timeout/worker-failure cases
inject status fields into the judge; they do not run over-budget workers or prove
OS access/resource isolation. CLI file-pin/path checks are covered separately by
the committed 58-test judge suite. Full R/C/A/E coverage and experimental validity
remain separate open obligations.
