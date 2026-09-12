# PR 27 candidate validation

All three Modal phases pass for source `312d8213900c0e4b35e7bdcac3d92d74ca5dbb81`.
The [result index](report.json) binds the unchanged manifests and the tracked
source inventory. [The summary](summary.json) records per-phase job counts.

| Phase | Test passes | Expected skips | Worker seconds | Original evidence |
| --- | ---: | ---: | ---: | --- |
| cpu310 | 4631 | 2 | 414.595 | [Receipt](cpu310/manifest.json) |
| cpu312 | 4631 | 2 | 390.468 | [Receipt](cpu312/manifest.json) |
| gpu | 441 | 0 | 508.127 | [Receipt](gpu/manifest.json) |

Both CPU phases pass lint and build the installed wheel and source distribution.
Python 3.12 also passes strict documentation, 48 installed prerequisites and
the exact 441-case CUDA collection. The T4 phase repeats all 48 prerequisites
and passes all 441 selected CUDA cases with no skips, including fresh CPU
serialization and eighteen installed extension cases.

The two CPU skips are the optional authoring SDK control and a negative host
check requiring an unsupported platform. They are not test passes.
Worker intervals include nested child work; overlapping CPU intervals must
not be added as elapsed time or treated as a compute invoice.

Original failed attempts remain in [CPU 3.10 run 1](failed-cpu310-run1/failure-analysis.json),
[CPU 3.12 run 1](failed-cpu312-run1/manifest.json), and
[T4 run 1](failed-gpu-run1/manifest.json). The controller interpreter
and historical collection corrections are described in
[the construction correction](construction-correction.json) and
[the installation correction](installation-correction.json).

This validates the selected foundation checkpoint. Full v1 application and
quality/cost obligations remain open; E5 remains deferred.
