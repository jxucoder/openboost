# Sprint 070: Deeper real A6 comparator preflight

Clean revision `2521619`. All three frozen fold-zero configuration-05 comparator
jobs pass. They retain learning rate 0.03, zero regularization, 255 bins, a
1000-round maximum and patience 50. XGBoost/CatBoost use depth 6; LightGBM uses
31 leaves. The source-pinned 400-job plan and original approved train/validation
packet are unchanged. No test material was uploaded or scored.

| Comparator | Fit worker seconds | Fresh replay seconds | Recorded rounds | Selected rounds | Peak guest RSS bytes |
| --- | ---: | ---: | --- | --- | ---: |
| XGBoost | 4.321278480 | 3.158498880 | 63 | 13 | 333729792 |
| LightGBM | 4.265050963 | 3.192943237 | 75 / 66 | 25 / 16 | 263995392 |
| CatBoost | 1.940294857 | 1.532248392 | 66 | 16 | 444108800 |

Native early stopping ends each fit well before its requested maximum. This
qualifies these configured resource probes, not sustained 1000-round execution
or other folds. Every saved prediction/row ID replays exactly in a fresh process;
all target scales match the earlier verified OpenBoost training-only fold freeze.

Each fit uses UID/GID 65534, no-new-privileges, an 8-GiB address ceiling, one
thread and 1800-second timeout. Modal requests two CPUs, 8192 MiB and 1900 seconds
per function, with no retries and sequential stop-on-failure dispatch. Replay
has 60 seconds. No failure or retry occurred. Exact commands, dependency versions,
Python/OS, source/image identity and dirty state are retained in the raw records.
Host CPU/physical RAM and cgroup enforcement are unobserved. Guest RSS is distinct
from the address limit and requested container capacity. Fit/replay timings omit
upload, image build, container startup and coordination; no speed ratio is claimed.

All 34 source hashes verify against the clean revision. All 30 outer and fifteen
inner artifact hashes verify; execution records match the manifest. Packet and
plan hashes verify. The original packet is reproducible rather than duplicated;
returned validation feature packets, models, histories, predictions and logs are
retained. The exact CLI is in manifest.json:

```bash
uv run --no-sync python -m benchmarks.v1.a6_resource_preflight /tmp/a6-deeper /tmp/a6-packets --comparators --comparator-config 5
```

Preparation passed 1125 CPU tests with one Linux-only skip, 33 focused checks,
lint and docs. The prior [configuration-00 evidence](../a6-comparators-070/README.md)
remains separate; these observations are not repeated timing samples of one case.

## Reflection and next step

Six of 240 comparator configurations now have bounded real resource/replay
observations across configurations 00 and 05 on fold zero. The other 234 remain
unqualified, and no full search, selected quality, authoring or GPU gate passes.

The next costly check is OpenBoost shared configuration 05 under the unchanged
1800-second limit. Stop at failure and retain the timeout/error rather than
shrinking its budget or silently substituting another case. If it fails, record
the resource gate failure and revisit practical CPU execution before expansion.
The existing comparator checks do not qualify that OpenBoost path. Complete
coverage and independent author accounting remain open alongside resource work.
