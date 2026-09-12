"""Independent default policy agrees with the original exact fixture oracle."""
import inspect

import pytest

from .reference.normal_split_order import exact_tree
from .reference.tree_numerical_policy import tree
from .test_normal_order_current_reference import prepared


@pytest.mark.parametrize('case', ['captured', 'p24-negative', 'p24-zero', 'p24-positive',
                                  'p54-negative', 'p54-zero', 'p54-positive',
                                  'p100-negative', 'p100-zero', 'p100-positive',
                                  'p149-negative', 'p149-zero', 'p149-positive'])
def test_independent_original_row_policy_matches_fixed_exact_cases(case):
    _, _, _, x, fields = prepared(case)
    assert tree(x, fields) == exact_tree(x, fields)


def test_standard_builder_exposes_explicit_keyword_numerical_policies():
    from openboost.device_tree import depthwise
    parameters = inspect.signature(depthwise).parameters
    for name in ('ordering', 'field_leaf'):
        assert parameters[name].default is None and parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
