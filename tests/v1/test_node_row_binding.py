"""Node validation must reject forged row digests even with valid input binding."""

from dataclasses import replace

import pytest

from openboost.newton_order import rank
from openboost.tree import best_first, depthwise, symmetric

from .test_exact_newton_order import prepared


@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
def test_row_digest_is_checked_even_when_input_binding_is_valid(grow):
    data, fields, _, _ = prepared()

    def forged(d, f, rows):
        records = rank(d, f, rows)
        assert len(records) > 1
        return tuple(
            replace(record, candidate=replace(record.candidate, rows_identity="forged-row-digest"))
            for record in records
        )

    with pytest.raises(ValueError, match="actual node and split fields"):
        grow(data, fields, ordering=forged)
