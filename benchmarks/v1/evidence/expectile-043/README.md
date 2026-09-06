# Installed D1 expectile development evidence

Reproduce from the repository with cached build/NumPy dependencies:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-expectile-043
```

`manifest.json` records parent revision and dirty state, exact commands, source
and wheel SHA256s, reference hashes, environment and artifact hashes. This is a
CPU correctness fixture, not a timing or quality benchmark. No GPU was used.
`expectile-expected.json` is generated separately from independent reference
formulas and exhaustive trees. `expectile-checks.json` records installed two-round
comparison; `expectile-model.json` contains the plugin-independent raw model.
Other JSON files rerun D2/D3/D4 and mixed scheduling checks. All nine models retain
exact raw predictions after uninstalling all four training plugins.

Synthetic fixtures and versioned source define data; no split or quality search
is implied. The developer-authored checks do not close E5 or establish independent
adoption. Prior sprint evidence is retained unchanged. See
[Sprint 043](../../../../v1-sprints/043-expectile-extension.md).
