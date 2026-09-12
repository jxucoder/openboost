"""Current installed D2 against this gate's preceding exact Normal trajectories.

Derived from tests/v1/test_normal_default_cuda.py at source checkpoint
91a519dd5227344266a3eb1c86ce21b45acf32c6 (SHA256 d5ec28bf9e660c065bfa5d88646a6f8cfc154268e244b23c1ecbb28cc5875c13).
The eighteen-case test and comparison/retention helpers preserve original bodies.
Only prior-record lookup changes: this file consumes freshly generated current
Normal records instead of archived run-34 JSON. Run test_current_normal_cuda.py
first in the same frozen installed gate and require all ninety cases to pass.
The collector must start empty; no archived or fallback prior records are used.
"""

import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from openboost import device_recipes
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .test_current_normal_cpu_reference import configuration, inputs
from .test_current_normal_cuda import plain, raw

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]


def retain(tmp_path,name,record):
    root=Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS',tmp_path))/'normal-default'
    root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(plain(record),indent=2,allow_nan=False)+'\n')


def snapshot(result,context):
    run,state=result.run,result.state
    return dict(model=run.export(state).record(),best_model=run.export(state,best=True).record(),
                raw=raw(run,state).tolist(),validation=raw(run,state,validation=True).tolist(),best=raw(run,state,validation=True,best=True).tolist(),
                steps=plain([asdict(s) for s in result.steps]),stop=dict(asdict(result.stop),reason=result.stop.reason),
                state=dict(version=state.version,n_terms=state.n_terms,best_n_terms=state.best_n_terms),metrics=dict(context.metrics))


def previous(name):
    """Bind the completed same-gate explicit trajectory; never use old archives."""
    root = Path(os.environ["OPENBOOST_NORMAL_ARTIFACTS"])
    path = root / "current-normal" / (name + ".json")
    raw_record = path.read_bytes()
    record = json.loads(raw_record)
    assert record["stage"] == "complete" and record["name"] == name
    assert record["closed_live_bytes"] == record["closed_records"] == 0
    return record, hashlib.sha256(raw_record).hexdigest()


def compare_prior(actual,prior):
    # Identical binary32 numerical policies and inputs must retain exact saved
    # model terms/values, arrays, trials and best prefixes, not only tolerances.
    for key in ('model','best_model','raw','validation','best','steps','stop'):
        assert actual[key]==prior[key]
    for key in ('version','n_terms','best_n_terms'):
        assert actual['state'][key]==prior['state'][key]


@pytest.mark.parametrize('geometry',['ordinary','natural','damped'])
@pytest.mark.parametrize('update',['joint','forward','reverse'])
@pytest.mark.parametrize('step',['fixed','backtracking'])
def test_installed_exact_d2_matches_validated_models_and_repeats(geometry,update,step,tmp_path):
    import ob_cohort_splits.device as extension
    path=Path(extension.__file__).resolve()
    source=ROOT/'examples/v1_extensions/cohort_splits/src/ob_cohort_splits/device.py'
    assert 'site-packages' in path.parts and not path.is_relative_to(ROOT)
    assert path.read_bytes()==source.read_bytes() and importlib.metadata.version('ob-cohort-splits')=='0.1.0'
    name='-'.join(('d2_constrained',geometry,update,step))
    config=configuration('d2_constrained',geometry,update,step)
    train,valid,binned,information,_=inputs(config)
    prior,sha=previous(name)
    record=dict(kind='installed',name=name,stage='inputs',configuration=config,prior_sha256=sha,
                extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),repeats=[])
    retain(tmp_path,name,record)
    for repeat in range(2):
        with ExecutionContext() as context:
            ops=DeviceOperations(context)
            learner=extension.DeviceCohortLearner(ops,train,information,binning=binned.binning)
            cohort_bytes=context.metrics['live_bytes']
            assert cohort_bytes==information.size*4
            observed=[]
            def tracked(o,data,fields,learner=learner,observed=observed):
                before=dict(context.metrics)
                result=learner(o,data,fields)
                after=dict(context.metrics)
                observed.append(dict(topology=result.topology,before=before,after=after))
                return result
            result=device_recipes.normal(ops,train,valid,run_id='146-installed',seed=7,rounds=3,binning=binned.binning,
                                         learner=tracked,mode=config['mode'],damping=config['damping'],update=update,step=step,learning_rate=config['rate'])
            current=snapshot(result,context)
            current.update(repeat=repeat,observed=plain(observed),cohort_bytes=cohort_bytes,stage='fit-complete')
            record['repeats'].append(current)
            retain(tmp_path,name,record)
            result.run.close()
            assert context.metrics['live_bytes']==cohort_bytes
            learner.close()
            learner.close()
            current.update(stage='complete',closed_bytes=context.metrics['live_bytes'],closed_records=len(ops._records))
            retain(tmp_path,name,record)
        assert current['closed_bytes']==current['closed_records']==0
        compare_prior(current,prior)
        # Rank uploads two int32 column indices and two float32 minima per
        # internal node actually considered, including terminal unsplit choices.
        # This is explicit policy metadata; independent per-row information was
        # uploaded once before the recipe and is not recopied by the learner.
        for event in current['observed']:
            levels=[0]*len(event['topology'])
            for i,(_,_,_,left,right) in enumerate(event['topology']):
                if left>=0:
                    levels[left]=levels[right]=levels[i]+1
            attempts=sum(level<2 for level in levels)
            assert event['after']['upload_bytes']-event['before']['upload_bytes']==20*len(levels)+16*attempts
    record['stage']='complete'
    retain(tmp_path,name,record)
    for key in ('model','best_model','raw','validation','best','steps','stop'):
        assert record['repeats'][0][key]==record['repeats'][1][key]
