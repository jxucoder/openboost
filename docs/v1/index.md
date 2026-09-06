# OpenBoost v1

OpenBoost is a programmable boosting foundation for researchers and agents, under
construction. Initial public CPU ownership, run-state and mapped ensemble artifact
components are available; see [CPU state usage](cpu-state.md).

[Numeric operations](numeric-ops.md) provide binning, histograms, candidate
selection, routing and scalar leaves. [Depthwise trees](trees.md) compose these
operations and support numeric inference/persistence. The first complete
[squared-error recipe](squared.md) provides fixed/backtracking CPU boosting.
[Normal boosting](normal.md) uses the same state and scalar learners for joint
mean/log-scale updates. [Formula and sequential runs](formula-runs.md) add
full-metric structured updates and independent heterogeneous execution. CUDA
execution is not implemented yet.
All R1–R9/C1–C7/A1–A13 remain required. Evaluation preparation continues alongside
the user-approved B03–B06 construction overlap. No complete quality, speed or
agent/adoption result is claimed for the new production foundation.

The old production API was retired. Historical examples require revision
`50acfc6`; current APIs are not backward compatible. See the repository's
`v1-sprints/` for construction records and `planning/` for requirements and gates.
