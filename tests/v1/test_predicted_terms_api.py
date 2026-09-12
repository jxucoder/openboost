"""Pure host bindings; these checks do not emulate CUDA execution."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

from .test_device_group_tree_api import prediction_jobs


def test_public_grouped_prediction_consumer_exists():
    from openboost.device_runtime import DeviceRun, PredictedTerm

    assert callable(PredictedTerm) and callable(DeviceRun.propose_predicted)


def pair(width=2):
    from openboost.device_group_tree import DevicePrediction
    from openboost.device_runtime import DeviceTerm
    from openboost.execution import DeviceBuffer

    job = prediction_jobs(1, width)[0]
    value = DeviceBuffer((job.data.n_rows, width), '<f4', job.data.n_rows*width*4)
    train = DevicePrediction(job.tree, job.data, value)
    validation = DevicePrediction(job.tree, replace(job.data), replace(value))
    return DeviceTerm(job.tree, np.eye(width)), train, validation


@pytest.mark.parametrize('width', [1, 2, 3])
def test_immutable_explicit_term_and_borrowed_prediction_bindings(width):
    from openboost.device_runtime import PredictedTerm

    term, train, validation = pair(width)
    item = PredictedTerm(term, train, validation)
    assert item.term is term and item.train is train and item.validation is validation
    with pytest.raises(FrozenInstanceError):
        item.train = validation
    assert not item.term.mapping.flags.writeable


@pytest.mark.parametrize('fault', ['term', 'train', 'validation', 'train_tree', 'validation_tree'])
def test_explicit_prediction_types_and_exact_tree_identity_required(fault):
    from openboost.device_runtime import PredictedTerm

    term, train, validation = pair()
    if fault == 'term':
        term = object()
    elif fault == 'train':
        train = object()
    elif fault == 'validation':
        validation = object()
    elif fault == 'train_tree':
        train = replace(train, tree=replace(train.tree))
    else:
        validation = replace(validation, tree=replace(validation.tree))
    with pytest.raises(ValueError):
        PredictedTerm(term, train, validation)


def test_stored_term_order_and_signed_coefficients_have_independent_controls():
    from .reference import multi_squared as ref

    raw = np.array([[2**24]], np.float32)
    minus, plus = np.array([[-2**24]], np.float32), np.ones((1, 1), np.float32)
    mapping = np.ones((1, 1), np.float32)
    first = ref.mapped(ref.mapped(raw, minus, mapping, np.float32(1)), plus, mapping, np.float32(1))
    reverse = ref.mapped(ref.mapped(raw, plus, mapping, np.float32(1)), minus, mapping, np.float32(1))
    assert first.item() == 1 and reverse.item() == 0
    assert ref.mapped(np.array([[4]], np.float32), np.array([[2]], np.float32), mapping, np.float32(-.5)).item() == 3
    assert ref.change(np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.ones(1)) == Fraction(1, 2)


@pytest.mark.parametrize('width', [1, 2])
def test_fresh_prediction_artifact_schema_without_device_imports(width, tmp_path):
    import json
    import subprocess

    from openboost.artifacts import Model

    from .multi_squared_artifacts import fresh_command, input_snapshot
    from .test_multi_squared_reference import prepared
    from .test_predicted_proposals_cuda import FRESH

    train, validation, _ = prepared(width)
    model = Model(train.data.feature_names, np.arange(width), ())
    report = dict(runs=[dict(model=model.record(), best_model=model.record(),
                            inputs={k: input_snapshot(p) for k, p in zip(('train', 'validation'), (train, validation), strict=True)},
                            predictions={key: {k: model.predict(p.data, offset=p.offset).tolist()
                                               for k, p in zip(('train', 'validation'), (train, validation), strict=True)}
                                         for key in ('model', 'best_model')})])
    path = tmp_path/'predicted.json'
    path.write_text(json.dumps(report))
    result = subprocess.run(fresh_command(FRESH, str(path)), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == dict(models=2, arrays=4, training_imports_denied=True)
