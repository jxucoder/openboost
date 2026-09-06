# Installed D2/D3 development evidence

Produced at parent 2f6c77a plus the recorded dirty extension/verifier sources.
No src/openboost changes were made; the core wheel matches Sprint 039's hash.
This is repository-authored exploratory CPU evidence, not E5, external adoption,
complete E2/E6, real-data quality or GPU performance.

- manifest.json: environment, versions, exact commands and working directories,
  source/wheel hashes, artifact hashes and installation/inference status.
- checks.json: deterministic fixture, three-round raw outputs, constrained cuts
  and maximum leaf-oracle difference.
- d2.json / d3.json: core inference artifacts created with the two plugins.

The isolated environment was removed after verification. Its absolute paths in
the command log identify the actual run; the reproduction script creates fresh
paths. Run `uv run --no-sync python examples/v1_extensions/verify.py OUTPUT_DIR`
from the repository with offline build dependencies cached. Python 3.12 and
NumPy 2.3.5 match this record. Checks use one BLAS/OpenMP thread and fixed seed
123 for the solver fixtures. No time, memory or quality benchmark is claimed.
