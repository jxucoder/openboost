"""Direct rational original-row scalar tree; no production ordering or leaf imports."""
from fractions import Fraction as F


def tree(x, fields, leaf_fields=None, *, depth=2, regularization=1, minimum=0, penalty=0, information_minimum=0):
    fields = [[F(v) for v in r] for r in fields]
    leaves = fields if leaf_fields is None else [[F(v) for v in r] for r in leaf_fields]
    regularization, minimum, penalty, information_minimum = map(F, (regularization, minimum, penalty, information_minimum))
    def total(rows, source, col):
        return sum((source[r][col] for r in rows), F(0))
    def score(rows):
        return total(rows, fields, 0)**2/(2*(total(rows, fields, 1)+regularization))
    def node(rows, level):
        return dict(rows=tuple(rows), level=level, key=None, left=-1, right=-1,
                    value=-total(rows, leaves, 0)/(total(rows, leaves, 1)+regularization))
    result = [node(range(len(x)), 0)]
    for n in result:
        options = []
        if n['level'] < depth:
            for feature in range(len(x[0])):
                for threshold in sorted({r[feature] for r in x if r[feature] is not None}):
                    for missing in (False, True):
                        left, right = [], []
                        for r in n['rows']:
                            value = x[r][feature]
                            (left if (missing if value is None else value <= threshold) else right).append(r)
                        if not left or not right:
                            continue
                        if any(total(rows, fields, 1) <= 0 or total(rows, fields, 1) < minimum for rows in (left, right)):
                            continue
                        if information_minimum and any(total(rows, fields, 2) < information_minimum for rows in (left, right)):
                            continue
                        gain = score(left)+score(right)-score(n['rows'])-penalty
                        if gain > 0:
                            options.append((-gain, (feature, threshold, missing), left, right))
        if options:
            _, n['key'], left, right = min(options)
            n['left'], n['right'] = len(result), len(result)+1
            result.extend((node(left, n['level']+1), node(right, n['level']+1)))
    return result
