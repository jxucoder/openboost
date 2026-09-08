# Linux author-isolation smoke: failed worker identity

The single approved smoke ran at clean `7985645ce4d1c7346e250a77113525b43f12bb12`.
**14/19 checks pass; the isolation gate fails.** All six public CPU examples run,
but the worker starts as UID/GID 0 and modifies its installed core and materials.
The evaluator remains on the controller, inaccessible at all five tested paths;
the four frozen evaluator files retain their original hashes.

| Observation | Result |
| --- | --- |
| Public examples | 6/6 pass |
| Other successful checks | Workspace write, private read/write denial, child/symlink denial, named credentials absent, loopback usable, outbound connection unavailable |
| Failed checks | `nonroot`, `core_write`, `materials_write`, `regain_root`, `core_unchanged` |
| Worker/provider completion | Exit 1; no provider timeout and no final timeout marker |
| Provisioning/create phase | 63.7336 seconds, including image build |
| Wait phase after create | 2.8082 seconds; not a separately measured full worker runtime |
| Requested resources | One Sandbox, 2 CPUs, 2048 MiB, 90-second lifetime, no GPU |
| Actual runtime | Python 3.12.10, Linux 4.19.0 gVisor x86_64, reported 2 CPUs |
| Packages | OpenBoost 1.0.0.dev0, NumPy 2.3.5, uv 0.12.1; client Modal 1.3.0.post1 |

[Build output](build.log) explicitly reports that the image's `USER` instruction
is unsupported and skipped. [Worker stdout](stdout.jsonl) then records UID 0 and
the failed checks. Only `openboost/__init__.py` changes among the recorded core
files; its resulting hash equals the probe's exact `b"tamper"` bytes. This is a
deliberate mutation inside the disposable worker, not a change to repository code.
[Stderr](stderr.txt) preserves the resulting failure. The future correction must
use process-level privilege reduction; [Modal's image guide](https://modal.com/docs/guide/existing-images#user)
documents this behavior. SDK object construction did not validate it beforehand.

[Manifest](manifest.json) retains exact source/dirty state, approved freeze hash,
worker argv, image/Sandbox identities, all raw outcomes and evaluator hashes.
[Freeze](freeze.json) is the exact approved input; [controller command](controller-command.json)
records the single CLI invocation and its nonzero exit. [Analysis](analysis.json)
is derived from the raw record. [Archive index](archive.json) hashes all seventeen
original artifacts plus the original manifest and two derived/controller records.
The thirteen staged upload files are preserved verbatim. No evaluator bytes were
included in those uploads.

The allowance is consumed with no retry. Timeout enforcement remains **unverified**;
the worker exited before starting the separate-session child. No model call,
independent author attempt or GPU run occurred. No quality, cost, E5 or complete
isolation claim follows these partial results. See the
[retrospective and next local correction](../../../../v1-sprints/096-linux-worker-result.md).
