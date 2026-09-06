# Installed ordered-update development evidence

Parent 6b2385f plus the dirty development sources hashed in manifest.json.
No OpenBoost core source changed. This extends the D2/D3 checks with six D4 cases:
Normal ordinary/Fisher and Formula full GGN, both parameter orders, three rounds.

- ordered-expected.json: independently generated reference trace supplied to the
  installed checker; generator and all reference sources are hashed in the manifest.
- ordered-checks.json: resulting predictions, version counts, maximum reference
  difference and the retained unsupported OrderedResult/run_many counterexample.
- ordered-0.json through ordered-5.json: saved raw inference artifacts.
- checks.json, d2.json and d3.json: rerun D2/D3 cases from the preceding sprint.
- manifest.json: exact commands/cwds, versions, source/wheel/artifact hashes and
  successful isolated checks plus inference after removing all three plugins.

Reproduce from the repository with
`uv run --no-sync python examples/v1_extensions/verify.py OUTPUT_DIR`.
Use cached offline build dependencies, Python 3.12 and NumPy 2.3.5. Temporary paths
in the command log describe the actual run; reproduction creates new paths.
These are deterministic CPU development fixtures, not scored E5, external
adoption, complete E2/E6, performance or real-data quality evidence.
