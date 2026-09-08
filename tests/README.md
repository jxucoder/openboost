# Test scope during the v1 rebuild

`tests/v1/` is the active suite. Its reference modules run without importing
production algorithms. The root conftest no longer loads the old OpenBoost.

All other test files and directories are historical evidence for the retired
implementation. They are excluded from default discovery, not marked as passing
or skipped. Reproduce them at revision `50acfc6`, which also has the original
conftest. Extract useful mathematical cases into independent v1 checks as each
required capability is implemented; do not copy obsolete API expectations.

The ninety full-loss Normal transaction checks in
`v1/test_device_normal_reference.py` are also historical. Their source is retained
byte-for-byte for frozen CUDA source manifests; default collection deselects them,
without counting passes or skips. `--include-historical-normal` explicitly restores
them for historical study. Their rounded full-loss decisions are platform-sensitive
and are not the current Normal recipe contract.

`v1/test_compared_normal_transactions.py` covers all ninety settings using the
public objective comparison and the independent 092 reference, preserving the
gradient, split, leaf, prediction and metric assertions. Consumer tests also check
that native and controlled lower/equal/higher reported scores cannot reverse a
mathematically worsening decision. See Sprint 109 for the hosted counterexample.

Test counts do not establish production parity, full v1 coverage, GPU execution,
predictive quality or adoption. See `v1-sprints/` for progress.
