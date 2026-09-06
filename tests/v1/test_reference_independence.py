"""Prove that importing and running reference code does not load production."""

import subprocess
import sys
from pathlib import Path


def test_reference_runs_with_production_imports_blocked():
    script = """
import importlib.abc
import sys

class BlockProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openboost" or fullname.startswith("openboost."):
            raise AssertionError("reference tried to import production: " + fullname)

sys.meta_path.insert(0, BlockProduction())
sys.path.insert(0, sys.argv[1])
from tests.v1.reference.data import NumericBinning, CategoryMap
from tests.v1.reference.classification import ClassMap, binary, softmax
assert NumericBinning.fit([0, 0, 0, 1], bins=2).cuts == (0.,)
assert CategoryMap.fit(['a', 'b']).transform(['new'])[1] == (True,)
assert ClassMap.fit(['b', 'a']).encode(['b']) == (1,)
assert binary([0], [1])[1][0] == -.5
assert abs(softmax([[0, 0, 0]], [0])[1][0, 0] - 1/3) < 1e-15
from tests.v1.reference.ranking import pairwise
from tests.v1.reference.quantile import weighted_quantile
from tests.v1.reference.vector import fit_vector_stump
assert pairwise([0, 0], [1, 0], [0, 0]).gradient[0] == -.5
assert weighted_quantile([0, 2, 10], .5, [1, 3, 1]) == 2
assert fit_vector_stump([[0], [1]], [[0], [0]], [[-1], [1]]).predict([[0]])[0, 0] == -.5
from tests.v1.reference.positive import poisson, gamma, tweedie
from tests.v1.reference.survival import aft, normal_tail
assert poisson([0], [1], [1])[1][0] == 0
assert gamma([0], [1])[1][0] == 0
assert tweedie([0], [0])[1][0] == 1
assert aft([0], [1], [1])[2][0] == 1
assert normal_tail(40)[2] > .999
from tests.v1.reference.coupled import normal, formula, directions
_, g, fisher = normal([[0, 0]], [1])
assert directions(g, fisher)[0, 0] == 1
assert formula([[0, 0]], [1], [1])[0] > 0
from tests.v1.reference.runs import RunSpec, derive_seed, run_many
assert derive_seed(7, 'run-α', 2, 'tree', 'rows') == 6175955064790668999
record = run_many([RunSpec('r', 0, 1, .1, 2, 2)], [[0], [1]], [[0], [1]],
                  {'r': ([[-1], [1]], [[-1], [1]])})['r']
assert record.status == 'completed' and record.best_round == 1
from tests.v1.reference.author import expectile_base, penalized_quantile, ordered_normal
assert abs(expectile_base([0, 2]) - 1.6) < 1e-12
assert penalized_quantile([0, 2, 10], .5, [1, 3, 1], penalty=1, anchor=0) == 1.5
assert ordered_normal([[0], [0]], [[0, 0], [0, 0]], [1, 3])[0].accepted
from tests.v1.reference.mixed import Transformer, grow
tr = Transformer.fit([['a'], ['b']], names=('c',), kinds=('categorical',))
assert grow([['a'], ['b']], [[1], [-1]], [[1], [1]], tr).predict([['a']])[0, 0] == -.5
from tests.v1.reference.integration import fit_positive
model, trace = fit_positive([[0], [1]], [0, 2], kind='poisson', exposure=[1, 1])
assert len(model.terms) == len(trace) == 2
from tests.v1.reference.tree import boost_squared
for policy in ("depthwise", "best_first", "symmetric"):
    result = boost_squared([[0], [0], [1], [1]], [-2, -2, 2, 2], policy=policy)
    assert abs(result.predict([[0]])[0] + 58/225) < 1e-12
assert not any(n == "openboost" or n.startswith("openboost.") for n in sys.modules)
print("independent-reference-ok")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(__file__).resolve().parents[2])],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == "independent-reference-ok"
