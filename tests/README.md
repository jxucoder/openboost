# Test scope during the v1 rebuild

`tests/v1/` is the active suite. Its reference modules run without importing
production algorithms. The root conftest no longer loads the old OpenBoost.

All other test files and directories are historical evidence for the retired
implementation. They are excluded from default discovery, not marked as passing
or skipped. Reproduce them at revision `50acfc6`, which also has the original
conftest. Extract useful mathematical cases into independent v1 checks as each
required capability is implemented; do not copy obsolete API expectations.

The current 55 tests do not establish production parity, full v1 coverage,
GPU execution, predictive quality or adoption. See `v1-sprints/` for progress.
