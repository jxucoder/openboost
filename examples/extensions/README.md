> **Historical implementation:** these extension packages use the retired API.
> Reproduce at revision `50acfc6`; they are not current v1 author-package evidence.
> See [CPU coverage audit](../../v1-sprints/035-cpu-coverage-audit.md).

# Independent CPU/CUDA extension wheels

Two small packages demonstrate three extension points without private imports,
core edits or a fork:

- `normal_fisher`: independent weighted Normal NLL/Fisher objective and
  `ChannelDecay` predetermined per-channel coefficients.
- `bounded_leaves`: `BoundedNewton` clips Newton leaf values during training.

These are repository-maintained examples, not evidence of external adoption.
Version 0.2.0 declares NumPy/CuPy support. Real-device installed-wheel conformance
is a separate gate from the primitive and built-in adapter tests. Numerical fixtures are not quality benchmarks.

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
No raw clipping is applied. Non-finite states and zero precision from extreme-scale underflow fail. Builder scope is numeric,
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
external author experiment remains unverified. The subsequent
[P7 resident value matrix](../../benchmarks/results/foundation/20260905T183820Z-3c245f2d/README.md)
passes default quality but fails the GPU performance budget; the independent
example also has worse proper scores on that dataset/configuration.


## Real CUDA installation check

From a clean committed checkout with Modal configured:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite extensions
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_extensions
```

Only the three built wheels, exact test/demo files and a hashed manifest upload
into an isolated Linux T4 container; repository source is not mounted. The two
extension modules are compared byte-for-byte with the installed wheel contents.
GPU finite-difference NLL/Fisher checks precede actual two-round fits at 16 and
4097 rows, separately enabling schedule and clipping and then composing all three
extensions. CPU/CUDA raw/NLL/CRPS agreement, changed subsequent gradients, bounded
leaf values and nonconstant coefficients are checked. The public example runs as
`python extension_demo.py --device cuda` in a new process. Both extension packages
are then uninstalled and another interpreter checks exact CPU predictions for
nine GPU-trained saved models without the plugins importable.

For an independent CUDA environment, install the extension's `cuda` extra or
OpenBoost's CUDA extra as well as both wheels. CUDA requires supported NVIDIA
hardware; CPU imports do not import CuPy. `step` uses the explicit context's
array module and rejects mixed/host arrays in CUDA calls; `constrain` accepts
same-device NumPy/CuPy raw arrays. Initialization remains on CPU. `loss_value`
returns an explicit host scalar; step/rule vector arithmetic stays on device.
The strict trainer's copies/scalar synchronization and unsupported eval/callbacks
remain as documented in the public experimental guide. No profiler trace or
speed/cost/adoption conclusion follows from this conformance suite.


The runnable demo uses a `main` entry guard. A standalone GPU script without
that guard failed when CUDA availability discovery spawned a Python worker and
re-entered top-level training. Keep training out of import-time execution; the
CPU installer explicitly tests importing the demo as `__mp_main__` without
creating a model. The failed real-device run is retained in foundation evidence.


## Independent author trial

[AUTHOR_TASK.md](AUTHOR_TASK.md) provides an unsolved extension task and a record
for elapsed time, assistance, private imports/core changes, GPU results and
willingness to depend on the package. No outside author has completed it yet.
