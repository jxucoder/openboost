"""Independent prescribed winner expectations for the resident reduction."""

import pytest

from .reference.newton_reduction import example, winner


@pytest.mark.parametrize('case', ['single', 'three', 'five', 'seven', 'odd_levels',
                                'block_tail', 'housing_width', 'ties', 'masked_ties',
                                'all_ineligible', 'all_masked', 'one_eligible',
                                'masked_maximum', 'sub_ulp_gap', 'large_cross_product'])
def test_prescribed_rational_winner(case):
    expected = dict(single=0, three=2, five=4, seven=6, odd_levels=29, block_tail=256,
                    housing_width=8159, ties=0, masked_ties=4, all_ineligible=-1,
                    all_masked=-1, one_eligible=6, masked_maximum=7, sub_ulp_gap=5,
                    large_cross_product=8)[case]
    assert winner(*example(case)) == expected


def test_limb_upload_uses_supported_host_dtype_without_losing_bits():
    import numpy as np

    from .reference.newton_reduction import upload_words

    values = [0, 65535, 65536, 2**1599, 2**1600-1]
    words = upload_words(values)
    assert words.dtype == np.dtype('int32') and words.flags.c_contiguous
    assert words.shape == (5, 100)
    unsigned = words.view(np.uint32)
    assert [sum(int(v) << (16*i) for i, v in enumerate(row)) for row in unsigned] == values
