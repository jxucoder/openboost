"""Normal device configuration/import checks require no CUDA execution."""

import subprocess
import sys

import numpy as np
import pytest

from openboost import device_normal


@pytest.mark.parametrize("floor", [0, -1, True, None, "1", np.nan, np.inf, 1e100, 1e-100])
def test_invalid_initial_scale_floor(floor):
    with pytest.raises(ValueError):
        device_normal.objective(minimum_scale=floor)


def test_import_and_configuration_without_cuda_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys; sys.modules['cupy']=None; sys.modules['numba']=None",
                    "from openboost import device_normal",
                    "operations = device_normal.objective()",
                    "assert callable(operations.loss) and callable(operations.prepare)",
                ]
            ),
        ],
        check=True,
    )
