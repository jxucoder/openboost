# P6 CPU independent-wheel conformance

Clean source: `3f8addda5a6c9494164b5e69923538f0bf33b56b`.

**5 passed / 0 skipped**, plus the executed weighted 64-row public demo and
**6 exact CPU prediction roundtrips after uninstalling both extension packages**
in a fresh interpreter. JUnit records the five independent math/composition
cases. This is CPU conformance, not an end-to-end performance benchmark.

Three wheels were built and installed into a disposable fresh venv outside the
repository. Module paths resolve under that venv's site-packages, with neither
editable installation nor PYTHONPATH injection. Source imports of the two method
packages use only NumPy and the public openboost.experimental module. The Normal
objective/schedule occupies 89 source lines and the bounded leaf 22 (including
blank lines/docstrings). No core change was required: the OpenBoost wheel hash
is identical to the P4.4 wheel.

Wheel SHA256:

- OpenBoost: `46dec4c697a5dfa21d8c6bca7e17a26019fd3c3373f0dd2ad539eed3ff819450`
- normal_fisher: `7024337a14fe927b9dfc830d619fa9b9faabc917c3ca0f0ba89adcf134ff9b88`
- bounded_leaves: `3f19f3d600c798e12d2c759ccd480c2e431359399a2136868333253ac4e0a6e6`

Independent finite-difference NLL verifies weighted gradients; analytic Fisher
checks effective curvature. Exhaustive original-row depth-one reference checks
two rounds of tree predictions and channel updates. Separate and combined fits
establish that schedule and clipping change predictions, and clipping changes
the next mean gradient. An opposing-target validation fixture triggers genuine
early stopping and restores both trees and coefficients to the first round.

The demo uses seed 7, 64 rows, nonuniform/zero weights, two rounds and depth two.
Its actual coefficients are mu=[0.2,0.1], log_sigma=[0.1,0.05]. Dataset hash and
checks are recorded in results.json. Saved raw predictions are reproduced
exactly without either extension importable for all four composition variants,
the early-stopped model and the weighted demo.

Environment: Python 3.12.12, Intel macOS x86_64, CPU thread settings=1,
NumPy 2.2.6, Numba 0.61.2, llvmlite 0.44.0, SciPy 1.15.3. Full package pins,
uv version, source/lock hashes and installed paths are recorded in results.json.
The initial attempt with development Numba 0.63.1 / llvmlite 0.46.0 failed in a
source build on this platform. The committed example requirements select an
installable binary stack satisfying OpenBoost's unchanged dependency ranges.
That setup obstacle is part of the result, not a cross-platform compatibility claim.

Reproduce from the clean source revision (development environment provides uv):

```sh
uv run --no-sync python examples/extensions/verify_wheels.py /tmp/openboost-extension-evidence
```

The recorded final run used UV_OFFLINE=1 after the fixed public dependencies had
been cached. Source file hashes were independently checked against git objects;
JUnit has exactly five passing cases and no failures/errors/skips. Temporary
paths and hostname are omitted from committed evidence.

Boundary: both example packages declare CPU only. P5 strict GPU trainer
integration and real-device package conformance remain open. These are
repository-authored examples, not third-party adoption or algorithm novelty.
