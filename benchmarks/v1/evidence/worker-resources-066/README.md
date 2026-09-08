# Numerical worker resource verification

Parent `e3af6c9` plus the dirty probe source hashed in preflight.json. This follows
the preserved [unavailable cgroup inspection](../resource-preflight-066/README.md).
No source data or model fit was used; only the probe script was uploaded.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.resource_preflight /tmp/openboost-resource-worker-066.json --modal --worker
```

Use a fresh output path. All nine worker checks pass. The original allocation,
timeout, partial-log and normal-child checks remain. The numerical process reports
RLIMIT_AS soft/hard limits of 8,589,934,592 bytes and rejects a 9-GiB virtual mapping
with ENOMEM, without touching 9 GiB of physical memory. NumPy 2.3.5 multiplication
passes and loaded OpenBLAS 0.3.30 reports two threads via threadpoolctl 3.6.0.

Guest-reported ru_maxrss increased from 85,733,376 to 152,322,048 bytes while
touching 67,108,864 bytes and remained at the higher value after release. This
passes the preregistered response/retention check for the reported high-water
counter. VmHWM is unavailable and stays null. The result does not certify a
hardware-level peak counter; it establishes usable guest process accounting.

For the diagnostic, every single-process worker will inherit the same 8-GiB
virtual-address ceiling. This is a conservative additional restriction on its
resident memory, not a claim that virtual memory equals RSS or that arbitrary
child-process aggregates are controlled. Numerical threads share that address
space. Requested Modal hard CPU/memory caps remain unchanged; hidden cgroup
values remain null and are not converted into verified host observations.

The revised check targets the actual numerical worker instead of requiring a
particular cgroup filesystem. It does not relax formal search thresholds. Fits
will record the stricter address constraint, guest-reported peak RSS and requested
container limits separately. The profiling image must rerun this preflight before
the frozen matrix. No CPU performance or quality result is established here.
