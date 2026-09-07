"""Unix diagnostic wrapper retaining profiles at a soft worker deadline."""

import argparse
import cProfile
import faulthandler
import json
import math
import pstats
import runpy
import signal
import sys
import time
from pathlib import Path


class ProfileDeadline(TimeoutError):
    """Soft diagnostic interruption; an outer runner must still enforce a hard cap."""


def profile_call(callback, directory, seconds):
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or not 0 < seconds <= 60
    ):
        raise ValueError("profile duration must be finite and in (0,60] seconds")
    root = Path(directory)
    if signal.getitimer(signal.ITIMER_REAL)[0] != 0:
        raise ValueError("profiling cannot replace an active process timer")
    previous = signal.getsignal(signal.SIGALRM)
    profiler = cProfile.Profile()
    start = time.monotonic()
    status = "error"

    def expire(signum, frame):
        raise ProfileDeadline("diagnostic soft deadline reached")

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        profiler.enable()
        result = callback()
        status = "complete"
        return result
    except ProfileDeadline:
        status = "deadline"
        raise
    finally:
        profiler.disable()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        profiler.dump_stats(root / "profile.pstats")
        with (root / "profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(
                60
            )
        stats = pstats.Stats(profiler)
        rows = [
            dict(
                file=key[0],
                line=key[1],
                function=key[2],
                primitive_calls=value[0],
                calls=value[1],
                self_s=value[2],
                cumulative_s=value[3],
            )
            for key, value in stats.stats.items()
        ]
        record = dict(
            status=status,
            soft_limit_s=seconds,
            wall_s=time.monotonic() - start,
            scope="instrumented diagnostic; cumulative times overlap; not a fit pass or speed comparison",
            functions=sorted(rows, key=lambda row: row["cumulative_s"], reverse=True)[:100],
        )
        (root / "profile.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    worker = Path(__file__).with_name("openboost_worker.py")
    sys.argv = [str(worker), str(args.job.resolve())]
    with Path("stacks.txt").open("w") as stream:
        faulthandler.dump_traceback_later(20, repeat=True, file=stream)
        try:
            profile_call(
                lambda: runpy.run_path(str(worker), run_name="__main__"), Path.cwd(), args.seconds
            )
        except ProfileDeadline:
            raise SystemExit(124) from None
        finally:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
