# Independent author trial: task and record

Status: prepared, **not attempted by an external author**. Repository-maintained
examples are not third-party adoption. Do not fill this in on an author's behalf.

## Task handed to the author

Build a separate installable package providing a Laplace distribution objective
with location and log-scale channels. Choose and document a defensible nonnegative
effective curvature, including how you handle the location loss's nonsmooth point.
Use sample weights. Add a leaf rule limiting absolute updates and a predetermined
per-channel schedule. Run at least two boosting rounds through the public
`openboost.experimental` API without private imports or core changes.

Supply an independent mathematical reference, CPU results, held-out NLL and a
calibration diagnostic on a dataset you can redistribute or identify publicly.
If CUDA is available, verify the same objective and training behavior there;
otherwise record that GPU validation was unavailable. Build and install your
wheel outside the repository. Save the model, uninstall the plugin and confirm
CPU raw prediction after loading in a fresh process. Record failures and requests
for help. No implementation or formulas are supplied with this task.

Start with the experimental extension cookbook and API guide. The author may
consult public examples; record every resource and assistance interaction used.
Agree on a time budget before starting. Stopping without a correct result is a
valid outcome and must remain in the record.

## Record completed by the author

| Field | Observation |
|---|---|
| Date, author role, prior boosting/Python/CUDA experience | |
| Agreed time budget; environment and dependency versions | |
| Dataset/version/hash; split seed and target preprocessing | |
| Core version/SHA; plugin source and wheel hashes | |
| Start time; first installed import; first mathematically correct fit | |
| Total active time; environment/setup time; blocked time | |
| Assistance count; question and response for each interaction | |
| Documentation/examples consulted | |
| Private imports or core changes required (include attempted ones) | |
| Independent oracle, tolerances, failures, CPU outputs | |
| CUDA hardware, actual execution report, parity or reason unavailable | |
| Held-out NLL/calibration and fit/predict timing, including scope | |
| Plugin-free inference result in a fresh interpreter | |
| Would you depend on OpenBoost in an independent package? Why? | |
| Hardest step; missing abstraction; next change you would request | |

Keep raw logs/artifacts with permission to share them. An unsuccessful or
unfinished attempt is evidence too. This form does not authorize contacting an
author, posting results or publishing their information.
