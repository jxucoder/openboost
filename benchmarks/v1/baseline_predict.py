"""Fresh-process replay of trusted A6 comparator bundles on validation features."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from baseline_worker import predict_saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    saved = pickle.loads(args.model.read_bytes())
    if saved["application"] != "A6":
        raise ValueError("A6 comparator replay required")
    with np.load(args.features, allow_pickle=False) as arrays:
        if set(arrays.files) != {"x", "row_ids"}:
            raise ValueError("features and row IDs only")
        prediction = predict_saved(saved, arrays["x"])
        if prediction.shape[0] != len(arrays["row_ids"]) or not np.isfinite(prediction).all():
            raise ValueError("invalid prediction")
        np.savez(args.output, prediction=prediction, row_ids=arrays["row_ids"])
