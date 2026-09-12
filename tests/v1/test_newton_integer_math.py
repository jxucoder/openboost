"""Finite-width Newton operations against unbounded integers and original-row Fractions."""

import struct
from fractions import Fraction as F

import numpy as np
import pytest

from tests.v1.reference.normal_split_order import candidates, captured, winner


def operations():
    from openboost._newton_integer_math import make_math

    return make_math(lambda function: function, np.uint64)


def array(value, width=100):
    assert 0 <= value < 1 << (16*width)
    return np.array([(value >> (16*i)) & 65535 for i in range(width)], np.uint32)


def integer(value):
    return sum(int(v) << (16*i) for i, v in enumerate(value))


def units(value):
    exact = F(value) * 2**149
    assert exact.denominator == 1
    return int(exact)


@pytest.mark.parametrize('power', [0, 1, 15, 16, 17, 148, 149, 253, 276, 307, 319, 1000, 1599])
def test_carries_borrows_products_and_order_match_unbounded_integers(power):
    zero, compare, add, subtract, multiply, _, _, _, _ = operations()
    a, b = 2**power, max(1, 2**power-1)
    out = array(0)
    assert compare(array(a), array(b)) == (0 if a == b else 1)
    assert subtract(array(a), array(b), out) == 0 and integer(out) == a-b
    assert subtract(array(b), array(a), out) == (0 if a == b else 1)
    assert add(array(a), array(b), out) == int(a+b >= 2**1600)
    assert integer(out) == (a+b) % 2**1600
    assert multiply(array(a), array(b), out) == int(a*b >= 2**1600)
    assert integer(out) == (a*b) % 2**1600
    zero(out)
    assert integer(out) == 0


@pytest.mark.parametrize('case', ['zero', 'negative_zero', 'minimum', 'negative_minimum', 'largest_subnormal',
                                  'smallest_normal', 'one', 'negative_one', 'maximum', 'negative_maximum',
                                  'infinity', 'negative_infinity', 'nan'])
def test_decode_preserves_every_finite_binary32_bit(case):
    raw = dict(zero=0, negative_zero=0x80000000, minimum=1, negative_minimum=0x80000001,
               largest_subnormal=0x7fffff, smallest_normal=0x800000, one=0x3f800000,
               negative_one=0xbf800000, maximum=0x7f7fffff, negative_maximum=0xff7fffff,
               infinity=0x7f800000, negative_infinity=0xff800000, nan=0x7fc12345)[case]
    out = array(99, 20)
    *_, decode, score, positive, order = operations()
    sign, invalid = decode(raw, out)
    if case in ('infinity', 'negative_infinity', 'nan'):
        assert invalid == 1
    else:
        value = struct.unpack('<f', struct.pack('<I', raw))[0]
        assert invalid == 0 and sign * integer(out) == units(value)
        assert sign == (0 if value == 0 else -1 if value < 0 else 1)


@pytest.mark.parametrize('case', ['captured', 'p24-negative', 'p24-zero', 'p24-positive',
                                  'p54-negative', 'p54-zero', 'p54-positive',
                                  'p100-negative', 'p100-zero', 'p100-positive',
                                  'p149-negative', 'p149-zero', 'p149-positive'])
def test_rational_score_polynomials_keep_every_observed_non_tie(case):
    *_, score, positive, order = operations()
    x, fields = captured(stored=True)
    if case != 'captured':
        power, direction = case.split('-')
        mass = F(1, 2**int(power[1:]))
        fields[0] = (dict(negative=-1, zero=0, positive=1)[direction]*mass, mass)
    options = [c for c in candidates(x, fields, range(8)) if c['legal']]
    parts, chosen = {}, None
    work = np.zeros((4, 100), np.uint32)
    for c in options:
        (gl, hl), (gr, hr) = c['sums']
        gp, hp = c['parent']
        p, q = array(0), array(0)
        assert score(array(abs(units(gl))), array(abs(units(gr))), array(units(hl+1)), array(units(hr+1)), p, q, work) == 0
        assert F(integer(p), 2*2**149*integer(q)) == gl*gl/(2*(hl+1)) + gr*gr/(2*(hr+1))
        decision, invalid = positive(p, q, array(abs(units(gp))), array(units(hp+1)), array(0), work)
        assert invalid == 0 and bool(decision) == (c['gain'] > 0)
        parts[c['key']] = p, q
        if decision:
            comparison, invalid = (1, 0) if chosen is None else order(p, q, *parts[chosen['key']], work)
            assert invalid == 0
            if comparison > 0:
                chosen = c
    assert chosen['key'] == winner(options)['key']
    left, right = (0, 0, True), (0, 3, False)
    decision, invalid = order(*parts[left], *parts[right], work)
    gap = next(c['gain'] for c in options if c['key'] == left) - next(c['gain'] for c in options if c['key'] == right)
    assert invalid == 0 and decision == (0 if gap == 0 else 1 if gap > 0 else -1)


@pytest.mark.parametrize('penalty', [0, 1, 2, 3])
def test_strict_positive_gain_preserves_zero_and_negative_results(penalty):
    *_, score, positive, _ = operations()
    # G=(-2,2), H=(1,1), lambda=1 gives gain exactly two.
    p, q, work = array(0), array(0), np.zeros((4, 100), np.uint32)
    assert score(array(units(2)), array(units(2)), array(units(2)), array(units(2)), p, q, work) == 0
    decision, invalid = positive(p, q, array(0), array(units(3)), array(units(penalty)), work)
    assert invalid == 0 and bool(decision) == (penalty < 2)


def test_proved_binary32_int32_capacity_including_penalty_and_cross_products():
    from openboost._newton_integer_math import COMPARE_LIMBS, FIELD_LIMBS

    maximum = units(struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0])
    n = 2**31-1
    g, d = n*maximum, (n+1)*maximum
    p, q = 2*g*g*d, d*d
    right = g*g*q + 2*maximum*q*d
    assert max(g, d).bit_length() <= FIELD_LIMBS*16
    assert max(p*q, p*d, right).bit_length() <= COMPARE_LIMBS*16
    *_, score, positive, order = operations()
    work = np.zeros((4, COMPARE_LIMBS), np.uint32)
    a, b = array(0), array(0)
    assert score(array(g), array(g), array(d), array(d), a, b, work) == 0
    assert (integer(a), integer(b)) == (p, q)
    assert order(a, b, a, b, work) == (0, 0)
    decision, invalid = positive(a, b, array(g), array(d), array(maximum), work)
    assert invalid == 0 and bool(decision) == (p*d > right)


def test_real_unsigned_storage_and_insufficient_capacity_report_overflow():
    _, _, add, subtract, multiply, decode, _, _, _ = operations()
    a, b, out = array(65535, 1), array(2, 1), array(0, 1)
    assert add(a, b, out) == 1 and integer(out) == 1
    assert multiply(a, b, out) == 1 and integer(out) == 65534
    assert subtract(b, a, out) == 1
    assert decode(0x3f800000, out)[1] == 1


@pytest.mark.parametrize('operation', ['score', 'positive', 'order'])
def test_nonpositive_denominators_are_invalid_before_any_decision(operation):
    *_, score, positive, order = operations()
    one, empty, out, work = array(1), array(0), array(0), np.zeros((4, 100), np.uint32)
    if operation == 'score':
        assert score(one, one, empty, one, out, array(0), work) == 1
    elif operation == 'positive':
        assert positive(one, one, one, empty, empty, work)[1] == 1
    else:
        assert order(one, empty, one, one, work)[1] == 1
