# Sprint 079: CUDA distribution updates and a public extension

Status: local construction underway in [090](090-normal-device-construction.md);
device implementation and acceptance open. Mapping: B12 / F3.2 / R6 / C2–C5 / E1–E2 device conformance.
Depends on: [078](078-cuda-scalar-path.md) correctness/residency acceptance.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first failing checks

Verify that the device boundary supports a structurally different algorithm and
an externally defined tree operation. First run deterministic Normal K=2 fixtures
that force accepted and rejected backtracking trials, and the independently checked
D2 cohort-feasibility fixture that changes a legal split choice.

## Work

- Implement resident Normal ordinary/Fisher directions with fixed and actual
  bounded backtracking; preserve parameter mapping, links and loss normalization.
  Verify joint updates and the ordered state boundary exercised by public loops.
- Port the verified D2 cohort candidate-feasibility component through a public bulk
  device interface, including its extra additive information field. Demonstrate
  that the installed external operation executes and changes feasibility on device.
- Record dynamic metric synchronizations, rejected-trial work, host decisions,
  transfers, compilation, workspace and total fit/prediction cost. Do not replace
  adaptive steps with fixed steps to avoid synchronization.
- Keep CPU oracle and installed extension independent from optimized kernels.
  No training-loop host fallback or hidden task-name special case is allowed.
- Reproduce the original P7 Normal workload and method matrix in a separate
  current-revision record, preserving its data, seeds, settings, paired historical
  baseline and original 1.2 threshold. Use the protocol prepared in 078; neither
  squared error nor the D2 probe substitutes for an original P7 method.

## Acceptance and reflection

Independent E1 checks cover geometry, extra statistics, candidate legality/gains,
leaves, both parameter states, stop/best/rejection, raw predictions and final scores.
The extension's nondefault choice is observed, not merely imported. Fresh CPU
inference works after removing training plugins and CUDA requirements.

Report all costs and failures under a preregistered bounded device workload. Keep
P7 correctness/quality/performance outcomes distinct; a 1.2-gate failure cannot
be relabeled using E4's 2x threshold or removed from the final audit. If
the bulk interface cannot express the external operation without core/private
edits, fix the demonstrated boundary and reverify affected CPU/F2 contracts;
separate cohorts after a semantic change. Next: 080's required device matrix.

## Results

No Normal device run. [090-A](090-normal-device-construction.md) records the local
design and 101 independent-math/public-CPU checks before new kernels, including a
retained zero-weight split ambiguity. 090-B prepares explicit float32-domain
checks and resident operations next. A scalar CUDA pass or these CPU fixtures
alone will not establish this sprint's acceptance.
