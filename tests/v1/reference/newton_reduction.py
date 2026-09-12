"""Prescribed rational selection inputs, independent of the reduction algorithm."""

from fractions import Fraction

import numpy as np


def upload_words(values):
    """Uploadable signed words with identical unsigned base-2^16 limb bits."""
    return np.asarray([[(value >> (16*i)) & 65535 for i in range(100)] for value in values], np.int32)


def example(case):
    size = dict(single=1, three=3, five=5, seven=7, odd_levels=30,
                block_tail=257, housing_width=8160).get(case, 9)
    pairs = [(i + 1, i + 2) for i in range(size)]
    eligible, extra = [True] * size, [True] * size
    if case == 'ties':
        pairs = [(3 * (i + 1), 7 * (i + 1)) for i in range(size)]
    elif case == 'masked_ties':
        pairs = [(3 * (i + 1), 7 * (i + 1)) for i in range(size)]
        extra[:3] = [False] * 3
        eligible[3] = False
    elif case == 'all_ineligible':
        eligible = [False] * size
    elif case == 'all_masked':
        extra = [False] * size
    elif case == 'one_eligible':
        eligible = [i == 6 for i in range(size)]
    elif case == 'masked_maximum':
        extra[-1] = False
    elif case == 'sub_ulp_gap':
        pairs = [(2**600, 2**600)] * size
        pairs[5] = (2**600 + 1, 2**600)
    elif case == 'large_cross_product':
        pairs = [(2**927 + i, 2**618) for i in range(size)]
    return pairs, eligible, extra


def winner(pairs, eligible, extra):
    legal = [i for i in range(len(pairs)) if eligible[i] and extra[i]]
    return max(legal, key=lambda i: (Fraction(*pairs[i]), -i)) if legal else -1
