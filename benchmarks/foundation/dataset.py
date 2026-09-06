"""Frozen California Housing source and sklearn-compatible transformations."""

import hashlib
import io
import tarfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

URL = "https://ndownloader.figshare.com/files/5976036"
ARCHIVE_SHA256 = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_housing(path):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("California Housing archive hash mismatch")
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        raw = np.loadtxt(archive.extractfile("CaliforniaHousing/cal_housing.data"), delimiter=",")
    # Match sklearn.datasets.fetch_california_housing; no learned preprocessing.
    ordered = raw[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]]
    y, X = ordered[:, 0] / 100000.0, ordered[:, 1:].copy()
    X[:, 2] /= X[:, 5]
    X[:, 3] /= X[:, 5]
    X[:, 5] = X[:, 4] / X[:, 5]
    return np.ascontiguousarray(X, dtype="<f4"), np.ascontiguousarray(y, dtype="<f4")


def split_indices(n_samples, seed):
    order = np.random.default_rng(seed).permutation(n_samples).astype("<i8")
    # 60/20/20, no stratification or target-dependent selection.
    return np.split(order, [int(n_samples * 0.6), int(n_samples * 0.8)])


def describe(path):
    X, y = load_housing(path)
    return {
        "name": "California Housing",
        "source_url": URL,
        "archive_sha256": ARCHIVE_SHA256,
        "member": "CaliforniaHousing/cal_housing.data",
        "features": FEATURES,
        "shape": list(X.shape),
        "target_unit": "100,000 USD",
        "arrays_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "dtype": "little-endian float32",
        "split": "numpy.default_rng(seed), permutation, 60/20/20",
        "split_sha256": {
            str(seed): [
                hashlib.sha256(a.tobytes()).hexdigest() for a in split_indices(len(y), seed)
            ]
            for seed in (0, 1, 2)
        },
    }


def fetch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive hash mismatch")
        path.write_bytes(content)
    return describe(path)
