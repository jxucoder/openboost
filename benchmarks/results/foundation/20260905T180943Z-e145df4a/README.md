# Failed P6 GPU public-demo process run

Source: `5d10b77347b8c578dfa0b8fc3f97f3b78fbc1531`.
The suite returned **1 failed / 2 passed / 0 skipped**; it is not passing P6 evidence.

The independent GPU math assertions and all eight CPU/CUDA composition cells
completed before the test reached the public demo subprocess. That subprocess
failed during CUDA availability discovery with a multiprocessing traceback and
`CUDA backend requested but CUDA is not available`. The demo executed fitting
at module top level without a main guard, allowing spawned interpreter re-entry.
Add a normal `if __name__ == '__main__'` entry guard and a no-side-effect import
check; rerun the whole package conformance sequence. No metric tolerance changes.

The final checks dictionary was not written because the test had not completed;
no uninstall/inference check ran. The raw failing JUnit/stdout is retained, with
manifest/wheel/source/environment provenance. No claim of GPU package completion,
quality, speed or adoption follows from this failed run.
