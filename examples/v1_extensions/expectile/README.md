# External expectile objective (D1 development)

`ob-expectile` defines loss `abs(tau - I[y-F<0]) * (y-F)^2`, default tau=0.8.
`Expectile.geometry(problem, raw)` returns mean weighted loss and **unweighted**
analytic gradient/curvature. Public `newton` applies training weights once.
At zero residual the gradient is zero and curvature is `2*tau`; finite-difference
second derivatives across that kink are not the declared curvature convention.
Initialization bisects the weighted derivative of target minus offset. Zero-weight
rows do not determine its bracket. Unsupported problem structures fail explicitly.

```python
from ob_expectile import fit
result = fit(train, validation, context=context, tau=0.8, rounds=2)
result.state.model.save("expectile.json")
```

The recipe composes public binning, Newton statistics, depth-two growth,
transactions and StopState. Its explicit signature accepts preparation, learning
rate, round budget and independent validation stopping; other options fail.
Fixed finite steps commit without line search. The resulting core raw model needs
no training plugin to load or predict. Add row offsets separately when converting
raw values to final expectile predictions. Best-model selection is separate from
the final model used in the development trace.

The objective and small outer loop are authored here; no core code or private
imports are required. This duplicates loop wiring for statistics, transactions
and stopping, not tree construction. It is a deliberate measurement of current
public composition, not justification for adding a generic trainer yet.

Run the parent `verify.py` for an isolated wheel check. Independent references use
stationary-interval base enumeration and exhaustive numeric tree growth; two rounds
cover missing values, zero weights and nonzero offsets. A fresh process loads the
raw model after all training plugins are uninstalled. These internal development
checks do not measure independent author cost, GPU or real-data quality. D1 is an
incumbent-friendly control: built-in expectile or custom-objective hooks remain
valid comparator approaches. Formal E2/E5/E6 gates remain open.
