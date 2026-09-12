"""Stored leaf fixtures have exact prescribed ratios before device execution."""

import subprocess
import sys
from fractions import Fraction as F

import numpy as np
import pytest

from .reference.binary32_rounding import cases


def stored_case(case):
    ratio, expected = cases()[case]
    if case == 'cancel':
        g, h, regularization = [np.float32(1e16), 1, -np.float32(1e16)], [1, 1, 1], 1
    else:
        n, d = abs(ratio.numerator), ratio.denominator
        shift = min(0, 127-max(n.bit_length()-1, d.bit_length()-1))
        factor = F(2**shift) if shift >= 0 else F(1, 2**(-shift))
        # Every nonzero numerator bit becomes one exactly representable stored
        # binary32 field. Scaling both sides keeps huge denominator cases finite.
        g = [float((1 if ratio < 0 else -1) * 2**i * factor) for i in range(n.bit_length()) if (n >> i) & 1]
        if not g:
            g = [0.]
        h, regularization = [float(d*factor)] + [0.]*(len(g)-1), 0
    return np.column_stack((g, h)).astype(np.float32), regularization, ratio, expected


@pytest.mark.parametrize('case', ['zero', 'cancel', 'one', 'negative_one', 'half_even', 'half_above', 'half_below',
                                  'half_odd', 'negative_half_above', 'minimum', 'half_minimum', 'negative_half_minimum',
                                  'above_half_minimum', 'below_half_minimum', 'odd_subnormal_midpoint', 'smallest_normal',
                                  'normal_boundary', 'below_normal_boundary', 'maximum', 'below_overflow', 'overflow',
                                  'above_overflow', 'negative_below_overflow', 'negative_overflow'])
def test_original_binary32_fields_have_the_prescribed_unrounded_ratio(case):
    stored, regularization, ratio, _ = stored_case(case)
    assert np.isfinite(stored).all() and np.all(stored[:, 1] >= 0)
    gradient = sum((F(float(v)) for v in stored[:, 0]), F(0))
    denominator = sum((F(float(v)) for v in stored[:, 1]), F(regularization))
    assert denominator > 0 and -gradient/denominator == ratio


def test_public_original_row_leaf_operation_exists():
    from openboost.device_newton_leaf import leaf

    assert callable(leaf)


def test_leaf_module_import_does_not_require_cuda():
    code = """
import sys
for name in ('cupy', 'numba', 'openboost.recipes', 'openboost.device_recipes'):
    sys.modules[name] = None
from openboost.device_newton_leaf import leaf
assert callable(leaf)
"""
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True)
