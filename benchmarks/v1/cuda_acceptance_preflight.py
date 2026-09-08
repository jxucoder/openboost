"""091's frozen diagnostic run; requires separate upload and compute allowances."""

import argparse
from pathlib import Path

from benchmarks.v1.cuda_aggregation_preflight import main

PROTOCOL = "v1-sprints/091-acceptance-run7.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output.resolve(), protocol_path=PROTOCOL)
