# Independent CPU extension wheels

Two small packages demonstrate three extension points without private imports,
core edits or a fork:

- `normal_fisher`: independent weighted Normal NLL/Fisher objective and
  `ChannelDecay` predetermined per-channel coefficients.
- `bounded_leaves`: `BoundedNewton` clips Newton leaf values during training.

These are repository-maintained examples, not evidence of external adoption.
They currently declare CPU support only. GPU package conformance remains after
P5 experimental trainer integration; prior primitive GPU tests do not establish
these packages' GPU support. Numerical fixtures are not quality benchmarks.

## Reproduce the installation boundary

From the repository with the development environment installed:

```sh
uv run --no-sync python examples/extensions/verify_wheels.py /tmp/openboost-extension-evidence
```

The verifier builds OpenBoost and both extension wheels, creates a disposable
venv outside the repository, installs wheels without editable mode or PYTHONPATH,
checks installed module locations, runs independent package tests and the
combined example, then uninstalls both plugins. A new interpreter loads six
saved models and requires exact CPU raw predictions with neither plugin
importable. Only sanitized JSON and JUnit evidence leave the temporary directory.
Use a clean commit for recorded evidence; development runs explicitly record
the dirty source state. No files are uploaded or packages published.

`requirements-cpu.txt` pins the tested Python 3.12 CPU environment. On Intel
macOS, copying the development environment's Numba 0.63.1 / llvmlite 0.46.0
pins triggered an unsuccessful source build. This example instead tests the
compatible binary stack Numba 0.61.2 / llvmlite 0.44.0 / NumPy 2.2.6.
This does not change OpenBoost's project-wide dependency ranges.

For manual installation, build all three wheels with `uv build --wheel` (use
`--out-dir` to collect them), create a new environment with `uv venv`, and use
`uv pip install --python <venv-python> -r examples/extensions/requirements-cpu.txt
<openboost-wheel> <normal-fisher-wheel> <bounded-leaves-wheel>`. Copy `demo.py`
to a directory outside the repository and run it with that environment's Python
and `OPENBOOST_BACKEND=cpu`. It saves `demo.ob` and prints the actual coefficients.
The verifier executes this same file; it is the runnable public example.

## What the checks establish

Finite differences of independent float64 weighted NLL verify both gradients;
analytic expected Fisher verifies effective curvature. Two rounds of depth-one
training are checked against exhaustive original-row splits and scalar Newton
updates, including the actual schedule and bound. Separate objective-only,
scheduled, bounded and combined fits distinguish the effects. The next-round
mean gradient changes after clipping. A real opposing-target validation set
triggers early stopping and checks tree/coefficient restoration.

The objective initializes weighted mean/variance (variance floor 1e-6), weights
both gradients and curvature exactly once, and rejects unsupported extra targets.
No raw clipping is applied. Non-finite states fail. Builder scope is numeric,
nonmissing, L2, full sampling, depth 0–8. Clipping retains the original split
criterion. Loading requires only OpenBoost; the saved model is inference-only.

## Usability observations

The packages require zero private OpenBoost imports and no core changes. The
bounded rule uses public `NewtonLeafRule`; objective math is independently
implemented rather than a built-in alias. Users explicitly select a builder to
attach a leaf rule and call the objective's `constrain` to obtain sigma. Separate
package metadata/builds and dependency selection are real setup costs. Source
hashes, wheel hashes, installed versions, module paths and test outcomes are
recorded so this narrow installation result is reproducible. A clean-room
external author experiment and end-to-end GPU value remain unverified.
