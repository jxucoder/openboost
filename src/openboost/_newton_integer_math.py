"""Private allocation-free dyadic Newton algebra for explicit CPU/CUDA compilation.

Arrays contain unsigned base-2^16 digits in little-endian order. The caller owns
all arrays and supplies a wide integer cast; products of two digits plus carry
fit unsigned 64 bits. Outputs never alias inputs for multiplication/polynomial
operations. Every capacity failure returns a flag; callers must not publish a
truncated result. This arithmetic is not yet a resident ordering API.
"""

FIELD_LIMBS = 20
COMPARE_LIMBS = 100


def make_math(compile_function, wide):
    """Compile identical integer operations explicitly; no runtime host fallback."""

    @compile_function
    def zero(out):
        for i in range(len(out)):
            out[i] = 0

    @compile_function
    def length(a):
        n = len(a)
        while n > 0 and a[n-1] == 0:
            n -= 1
        return n

    @compile_function
    def compare(a, b):
        na, nb = length(a), length(b)
        if na != nb:
            return 1 if na > nb else -1
        for j in range(na):
            i = na-1-j
            if a[i] != b[i]:
                return 1 if a[i] > b[i] else -1
        return 0

    @compile_function
    def add(a, b, out):
        carry, invalid = wide(0), 0
        n = max(len(a), len(b), len(out))
        for i in range(n):
            value = carry
            if i < len(a):
                value += wide(a[i])
            if i < len(b):
                value += wide(b[i])
            digit, carry = value & wide(65535), value >> wide(16)
            if i < len(out):
                out[i] = digit
            elif digit != 0:
                invalid = 1
        return 1 if carry != 0 else invalid

    @compile_function
    def subtract(a, b, out):
        if compare(a, b) < 0:
            zero(out)
            return 1
        borrow, invalid = wide(0), 0
        n = max(len(a), len(b), len(out))
        for i in range(n):
            av = wide(a[i]) if i < len(a) else wide(0)
            bv = (wide(b[i]) if i < len(b) else wide(0)) + borrow
            if av < bv:
                digit, borrow = av + wide(65536) - bv, wide(1)
            else:
                digit, borrow = av-bv, wide(0)
            if i < len(out):
                out[i] = digit
            elif digit != 0:
                invalid = 1
        return 1 if borrow != 0 else invalid

    @compile_function
    def multiply(a, b, out):
        zero(out)
        invalid = 0
        na, nb = length(a), length(b)
        for i in range(na):
            carry = wide(0)
            for j in range(nb):
                k = i+j
                value = wide(a[i])*wide(b[j]) + carry
                if k < len(out):
                    value += wide(out[k])
                    out[k] = value & wide(65535)
                elif value != 0:
                    invalid = 1
                carry = value >> wide(16)
            k = i+nb
            while carry != 0:
                if k >= len(out):
                    invalid = 1
                    break
                value = wide(out[k])+carry
                out[k], carry = value & wide(65535), value >> wide(16)
                k += 1
        return invalid

    @compile_function
    def decode(raw_bits, out):
        zero(out)
        exponent = (raw_bits >> 23) & 255
        mantissa = raw_bits & 8388607
        if exponent == 255:
            return 0, 1
        shift = 0
        if exponent != 0:
            mantissa |= 8388608
            shift = exponent-1
        if mantissa == 0:
            return 0, 0
        invalid = 0
        for i in range(24):
            if (mantissa >> i) & 1:
                bit = shift+i
                if bit//16 >= len(out):
                    invalid = 1
                else:
                    out[bit//16] |= 1 << (bit % 16)
        return (-1 if raw_bits & 2147483648 else 1), invalid

    @compile_function
    def score_parts(gl, gr, dl, dr, p, q, work):
        invalid = multiply(gl, gl, work[0])
        invalid |= multiply(work[0], dr, work[1])
        invalid |= multiply(gr, gr, work[0])
        invalid |= multiply(work[0], dl, work[2])
        invalid |= add(work[1], work[2], p)
        invalid |= multiply(dl, dr, q)
        if length(dl) == 0 or length(dr) == 0:
            invalid = 1
        return invalid

    @compile_function
    def positive(p, q, gp, dp, penalty, work):
        invalid = multiply(p, dp, work[0])
        invalid |= multiply(gp, gp, work[1])
        invalid |= multiply(work[1], q, work[2])
        invalid |= multiply(q, dp, work[1])
        invalid |= multiply(work[1], penalty, work[3])
        invalid |= add(work[3], work[3], work[1])
        invalid |= add(work[2], work[1], work[3])
        if length(q) == 0 or length(dp) == 0:
            invalid = 1
        return compare(work[0], work[3]) > 0, invalid

    @compile_function
    def order(ap, aq, bp, bq, work):
        invalid = multiply(ap, bq, work[0])
        invalid |= multiply(bp, aq, work[1])
        if length(aq) == 0 or length(bq) == 0:
            invalid = 1
        return compare(work[0], work[1]), invalid

    return zero, compare, add, subtract, multiply, decode, score_parts, positive, order
