"""Native weighted early-stopping and selected-iteration replay on CPU."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from benchmarks.v1.worker_smoke import run

if __name__ == "__main__":
    result = dict(
        scope="synthetic CPU native stopping and replay only; not real quality or GPU evidence",
        cells=run(patience=3),
        seed=41,
        threads=2,
        maximum_rounds=24,
        patience=3,
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        python=platform.python_version(),
        os=platform.platform(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        source_hashes={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["early_stopping_smoke.py", "worker_smoke.py", "baseline_worker.py"]
        },
    )
    Path("benchmarks/v1/evidence/early-stopping-cpu.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        [
            (r["library"], r["application"], [s["selected_rounds"] for s in r["stopping"]])
            for r in result["cells"]
        ]
    )
