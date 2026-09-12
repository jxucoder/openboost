# Checkpoint merge validation

The user explicitly requests merging PR 27. The checkpoint production files remain
byte-identical to development snapshot `91a519d`; this follow-up changes validation,
historical test selection and documentation only. Engineering v1 remains incomplete.

## Execution plan

1. Commit the candidate and freeze its public Git revision, tracked-file inventory,
   lockfile and build dependencies. Fetch public history inside bounded Modal
   workers; do not upload a local development checkout or private application data.
2. On Python 3.10 and 3.12, build and install the actual wheel, verify all 63 installed
   production modules, then execute the complete current CPU suite. Python 3.12 also
   builds strict documentation and collects the exact installed CUDA population.
3. Only after both CPU phases pass, run the frozen 441-case current CUDA cohort on
   one T4, including original-field mathematical oracles, final/best fresh inference,
   and eighteen installed D2 settings against the preceding ninety Normal cases.
4. Preserve logs, JUnit, model records, package/source identities, build outputs and
   all failures. Commit the source-bound result index and raw phase outputs, verify
   the lightweight GitHub checks, then merge the exact reviewed candidate.

The actual status is in
[the result index](../docs/v1/evidence/pr27-modal-validation/report.json). The initial
metadata report remains a record of the earlier draft inspection, not a runtime
pass. No launch or passing result is implied by this execution plan.

## Commands and evidence

`.github/scripts/modal_validate_checkpoint.py` executes one explicitly frozen phase.
Each protocol pins source revision, inventory, interpreter, packages, test population,
resources, deadlines and zero retries. CPU receipts and the actual installed CUDA
collection are prerequisites for allocating a GPU. The original cumulative $200
ceiling remains in force, with a $5 reserve for this checkpoint phase.

GitHub runs `.github/scripts/check_modal_validation.py` to verify the committed
receipts and raw artifacts against every tracked candidate blob. Only
`docs/v1/evidence/pr27-modal-validation/` is excluded to permit committing results
without changing their tested source. Absent results, failures, changed inputs or
unexpected skips fail the gate. No heavy GitHub or local test/build path remains.

## Reconciled scope

The 47 omitted changed archive/benchmark modules remain explicitly listed in the
selection manifest. They concern original execution archives, omitted benchmark
controllers or historical source-specific comparisons. The earlier 303 derived
Normal/multiclass cases retain their independent mathematics. The additional 18
D2 extension cases retain the original current-policy test and comparison bodies,
with only prior-record lookup changed to consume newly generated records. The two
older 18-case extension trajectories remain available through the existing
historical flag; current nontrajectory guards remain enabled.

All 85 consumed GLM audit sources are reconstructible from public ancestry. The
default CPU gate exercises its existing historical wrapper and original eleven
controls. Full-budget train-many, formal E4, remaining real application evaluations
and deferred E5 remain outside this scoped checkpoint merge and still open for v1.


## First-attempt diagnosis

At candidate `c95dc201`, the Python 3.10 launch failed before worker creation because
serialized controller bytecode used Python 3.12. The correction keeps the Modal
controller on Python 3.12 and runs candidate checks under managed Python 3.10.17.
It rejects incompatible controller/image versions before creating an image.

The Python 3.12 worker built the wheel and source distribution and passed lint.
Its CPU suite returned 4,630 passes, one historical-collection mismatch and two
expected skips. The historical 383-case binding check must explicitly opt into
historical collection; its mapping and original expected cases remain unchanged.
The optional authoring SDK is absent, and the negative unsupported-host check is
inapplicable on supported Linux. The corrected frozen policy allows precisely
these two CPU skips with exact IDs/reasons, without counting them as passes or
activating deferred authoring infrastructure. All CUDA/prerequisite skips still
fail. Original failed phase outputs remain retained; distinct corrected attempts
must validate the new candidate before merge.
