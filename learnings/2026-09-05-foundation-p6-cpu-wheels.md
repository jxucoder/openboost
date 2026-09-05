# 2026-09-05: Independent CPU extension wheel boundary

## Context

The goal review moved the CPU portion of P6 ahead of P5 so we test actual
install/use friction before adding more GPU implementation. P4.4 established
an opt-in level-wise builder; independent method packages were still absent.

## Decision or Result

Implement two separately built example wheels with three extension points.
Normal/Fisher math is independently implemented; bounded leaves use the public
Newton rule. No OpenBoost core change or private import is required. Capabilities
remain CPU-only until real GPU package/trainer tests pass.

## Changes

- `examples/extensions/normal_fisher`: weighted two-parameter Normal objective,
  expected diagonal Fisher and prescribed channel decay. Reject unsupported
  extras/non-finite states; do not silently clip raw state.
- `examples/extensions/bounded_leaves`: bounded Newton values during tree growth.
  Original split criterion is retained.
- Separate package mathematical tests plus exhaustive original-row depth-one
  two-round reference; distinguish objective-only, scheduled, bounded and
  combined fits. Check changed second gradients and real early-stop restoration.
- Fresh outside-repo venv installs three wheels without editable/PYTHONPATH;
  records installed paths, versions, source/wheel hashes and JUnit. Uninstall
  both plugins, restart Python and compare six saved models exactly, including
  the executed 64-row weighted public demo.
- Fixed CPU dependency requirements and source-line counts record setup cost.
  JSON excludes temporary/user paths; JUnit hostname is removed.

## Verification

- Initial package test failed collection with missing normal_fisher, as expected.
- First successful fresh installation: 5 passed, five exact plugin-free model
  roundtrips. Public-demo roundtrip and final clean-source artifact follow below.
- Final development run: 5 installed-package tests passed; executed public demo;
  six exact CPU model roundtrips after uninstall in a new interpreter.
- Existing extension/builder/leaf regression: 76 passed. Production/example lint
  and MkDocs build passed (existing griffe warnings).

## Failed Attempts

- Offline install could not resolve pytest from cache. Retried public dependency
  downloads through uv into a disposable environment; no source upload.
- Copying development Numba 0.63.1 / llvmlite 0.46.0 pins on Intel macOS triggered
  source builds and failed in llvmlite/setuptools with a dry_run argument error.
  Pinned Numba 0.61.2 / llvmlite 0.44.0 / NumPy 2.2.6 for the example verification;
  these satisfy project ranges. Did not copy the development site-packages or
  change the repository's environment/dependency policy to hide installation cost.

## Risks and Follow-ups

- CPU self-authored package conformance is not external adoption. External author
  work, GPU package conformance, held-out quality and measured value remain open.
- P6 CPU portion only; P5 strict GPU dispatch/residency is the next integration
  gate, then run these actual packages on GPU. No GPU capability is declared yet.
- Intel macOS dependency selection is a concrete setup obstacle. The tested pins
  establish one installation, not cross-platform/version compatibility.

## Commits

- `00ca3b2` — P4.4 frozen T4 builder evidence preceding this slice.
