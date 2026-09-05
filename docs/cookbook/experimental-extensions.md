# Write an experimental extension

Use `openboost.experimental` to prototype distributional objectives, leaf rules
and predetermined schedules in an independent package. The API is experimental;
keep the tested OpenBoost version in your package metadata. Start on CPU, then
verify the same math and final predictions on real CUDA hardware before declaring
CUDA support. The `examples/extensions/` packages and their standalone `demo.py`
are the executable reference.

## Implement the contract

| Extension | Implement | Required behavior |
|---|---|---|
| Objective | `channel_names`, `supported_devices`, `init_raw(y, sample_weight=None, extra=None)`, `step(raw, y, sample_weight=None, extra=None, *, context)`, `loss_value(...)`, `constrain(raw, extra=None)` | Initialize on CPU; return one contiguous float32 G/H pair per channel on `context.device`; weight both exactly once; curvature must be nonnegative. Reject unsupported extra targets. |
| Leaf rule | `supported_devices`, `values(G, H, *, config, context)` | Return finite contiguous float32 leaf values on the same device; zero empty/inactive/zero-G-and-H slots. Attach through `LevelWiseBuilder(leaf_rule=rule)`. |
| Schedule | `coefficients(round_idx, channel_names, base_learning_rate)` | Return all channel names with full finite nonnegative coefficients; the trainer applies them once. Use the supplied zero-based round, never global RNG state. |

Use `context.xp` for vector arithmetic and `context.rng` for randomness. CPU
inputs are read-only; CUDA boundaries isolate and check borrowed arrays. Return
independent output buffers, never mutate inputs or reuse another channel's G/H.
Effective curvature may be Fisher curvature rather than the exact Hessian:
document the distinction and test it against an independent mathematical oracle.
`loss_value` returns a host scalar; `constrain` converts raw scores to parameters.

Compose `Booster(objective=..., tree_builder=..., step_schedule=...,
config=TrainerConfig(...), device="cpu")`. Save with `model.save(...)`; a fresh
interpreter can call `Booster.load(...).predict_raw(X)` without training plugins.
The loaded model is inference-only. Preserve your parameter transformation
separately when applications need constrained parameters rather than raw scores.

## Choose a supported path

| Capability | CPU default builder | LevelWiseBuilder CPU / strict CUDA |
|---|---|---|
| Numeric nonmissing, L2, full sampling | Yes | Yes |
| Missing numeric values, L1, sampling | Yes | Rejected |
| Custom leaf rule through public interface | Select LevelWiseBuilder | Yes, with declared device support |
| Eval sets, callbacks, early stopping | Yes | CPU yes; strict CUDA rejected |
| Objective inputs during rounds | NumPy | NumPy / CuPy |
| Fit inputs and binning | Host | Host, including CUDA fits |
| Prediction after fit/load | CPU | CPU |
| Non-default CUDA stream | Not applicable | Rejected |
| Further training after load | Rejected | Rejected |

Depth is 0–8 and regular bins are 2–254. The process-global backend does not
support concurrent mixed-backend fits. Categorical input is outside the numeric
experimental facade. `fallback="error"` is the default; `fallback="warn"` handles
unsupported capability by selecting the entire CPU fit and reporting the reason.
Runtime errors do not trigger a silent retry. Inspect `fit_report_` for execution.

## Verify the package, then the result

1. Test independent float64 loss/gradient/curvature references, nonunit and zero
   weights, finite outputs, forbidden aliasing and explicit unsupported inputs.
2. Test at least two rounds: leaf/schedule changes must affect later gradients,
   and CPU/CUDA final predictions and task metrics must agree within declared
   tolerances. A skipped device test provides no CUDA evidence.
3. Build with `uv build --wheel`; install the core and plugin wheels into a fresh
   `uv venv` outside the checkout. Use no editable install or private OpenBoost
   imports. Run the standalone example under an `if __name__ == "__main__":`
   guard, then uninstall plugins and verify CPU predictions in a new interpreter.

Reproduce the repository's CPU check with
`uv run --no-sync python examples/extensions/verify_wheels.py /tmp/openboost-extension-evidence`.
The CUDA installation check uses `benchmarks.foundation.prepare --suite extensions`
and the `foundation_extensions` Modal entrypoint, as documented in the examples.

A real failure shaped this workflow: top-level GPU training in the demo re-entered
when CUDA discovery spawned a worker. A main guard fixed the failure; the original
JUnit remains committed. Another checked failure is extreme Normal log scale
causing zero precision: the objective rejects it instead of accepting a zero
curvature. Unsupported eval in strict CUDA is a capability error, not a successful
GPU fit. See the [full API boundary](../user-guide/experimental.md) for details.
