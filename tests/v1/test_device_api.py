"""CPU import and configuration checks do not imply real-device conformance."""

import subprocess
import sys

import pytest

from openboost.device import DeviceOperations


def test_device_module_import_does_not_require_cuda_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['cupy']=None; sys.modules['numba']=None; from openboost.device import DeviceOperations",
        ],
        check=True,
    )


@pytest.mark.parametrize("context", [None, "cuda:0", object()])
def test_explicit_context_required(context):
    with pytest.raises(ValueError, match="ExecutionContext"):
        DeviceOperations(context)
