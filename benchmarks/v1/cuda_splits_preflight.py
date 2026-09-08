"""Sprint 087's proposed third T4 run; blocked until a new user allowance is recorded."""

import argparse
from pathlib import Path

from benchmarks.v1.cuda_aggregation_preflight import main

PROTOCOL = "v1-sprints/078-splits-run3.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output.resolve(), protocol_path=PROTOCOL)
