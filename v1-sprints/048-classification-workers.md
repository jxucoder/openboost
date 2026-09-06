# Sprint 048: Classification evaluation workers

Parent: b430df0. Status: complete for classification adapter/Adult integration. Mapping: Sprint 038 M3, A2/A3.

## Plan and acceptance

1. Reproduce unsupported classification jobs and add current binary/multiclass
   adapters with explicit canonical class counts and encoded targets.
2. Persist and validate class order: binary returns P(class=1), multiclass returns
   columns in encoded class order. Reject malformed labels/schema/configuration.
3. Check weighted direct-recipe parity and fresh-process probabilities with/without
   patience for both tasks; run all five frozen Adult folds for A2.
4. Run regression/lint/docs, record evidence and commit locally. A3 full Covertype
   runs remain separate; no quality/search/calibration or GPU acceptance claims.

## Results and reflection

Current A2/A3 workers compose the existing binary/multiclass recipes with explicit
canonical class counts. They return probabilities in declared class order and
persist the schema. Restored predictions reject missing/reordered schemas.
Twelve new tests cover weighted direct parity with/without patience, fresh-process
probabilities, invalid counts/labels/missing classes and unsupported options.
All 844 CPU tests pass; Ruff and strict docs pass.

The first real Adult run failed all five folds: packet source IDs are strings,
while public NumericData requires integer IDs. The adapter now validates and
preserves source IDs at the packet boundary and uses local integer indices for
execution. String-ID fresh-process cases exercise this fix. No core row-identity
contract changed. The rerun passes all five Adult folds with exact probability
replay and external row IDs preserved. Both failed and passing raw evidence are
retained in [classification-048](../benchmarks/v1/evidence/classification-048/README.md).

This is why real packet integration matters beyond small fixtures: it exposed an
actual consumer boundary mismatch. A3 full Covertype runs remain pending; synthetic
A3 checks do not stand in for that required coverage. Next continue real A3 and
remaining application adapters/search integration alongside D5. Four-round probes
are not quality/calibration, search or performance acceptance. See
[learning](../learnings/2026-09-06-v1-classification-workers.md).
