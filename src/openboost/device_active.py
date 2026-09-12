"""Explicit independently owned squared recipe phases for external scheduling."""

from dataclasses import dataclass

from .device import _parameter, _workspace
from .device_group_growth import TreeJob
from .device_group_tree import DevicePrediction
from .device_recipes import DeviceStep
from .device_runs import RunResult
from .device_runtime import DeviceRun, DeviceTerm, PredictedTerm
from .stopping import StopState


@dataclass(frozen=True)
class SquaredConfiguration:
    """Immutable policy matching the existing reported-comparison squared recipe.

    Tree parameters define the default request. A caller may replace the emitted
    TreeJob to supply its own public learner policies. Acceptance/stop parameters
    stay with this phase; changing the returned job cannot change them.
    """

    rounds: int = 2
    learning_rate: float = 0.1
    max_depth: int = 2
    reg_lambda: float = 1.0
    min_child_h: float = 0.0
    split_penalty: float = 0.0
    step: str = 'fixed'
    max_trials: int = 6
    patience: int | None = None
    min_delta: float = 0.0

    def __post_init__(self):
        policy = StopState.start(0, rounds=self.rounds, patience=self.patience, min_delta=self.min_delta)
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError('nonnegative integer max_depth required')
        if not isinstance(self.step, str) or self.step not in ('fixed', 'backtracking') or type(self.max_trials) is not int or not 1 <= self.max_trials <= 6:
            raise ValueError('fixed/backtracking step and 1..6 trials required')
        for name in ('learning_rate', 'reg_lambda', 'min_child_h', 'split_penalty'):
            object.__setattr__(self, name, float(_parameter(getattr(self, name))))
        object.__setattr__(self, 'min_delta', policy.min_delta)


class SquaredPhase:
    """Own one active recipe; expose tree requests and prediction consumption.

    No workspace remains open between calls. The phase owns its DeviceRun,
    accepted state and history. Returned TreeJob fields are independent owned
    records: the caller releases them with ops.release. Tree/prediction inputs to
    advance remain borrowed, reusable and caller-owned. The execution context and
    optional prepared feature pair remain borrowed for this phase's lifetime.

    This component does not schedule other runs. Fixed/backtracking trials match
    device_recipes.squared, using reported losses and strict best selection.
    Call close or use a context manager; result exports detached final/best models
    only after terminal stopping. Arbitrary learner callbacks remain caller work.
    """

    def __init__(self, ops, train, validation, *, run_id, seed, configuration=None,
                 binning=None, bins=254, prepared=None):
        configuration = SquaredConfiguration() if configuration is None else configuration
        if not isinstance(configuration, SquaredConfiguration):
            raise ValueError('explicit SquaredConfiguration required')
        self._configuration = configuration
        self._closed = False
        self._run = DeviceRun(ops, train, validation, run_id=run_id, seed=seed,
                              binning=binning, bins=bins, prepared=prepared)
        try:
            self._state = self._run.initialize()
            self._stop = StopState.start(self._state.validation_score, rounds=configuration.rounds,
                                         patience=configuration.patience, min_delta=configuration.min_delta)
            self._steps = []
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self._closed:
            raise RuntimeError('active recipe phase is closed')
        self._run.validate_state(self._state)

    def _require_active(self):
        self._check()
        if self._stop.reason is not None:
            raise ValueError('recipe has reached terminal stopping')

    @property
    def configuration(self):
        return self._configuration

    @property
    def run(self):
        """Borrow the public run for explicit data/RNG/diagnostic operations."""
        self._check()
        return self._run

    @property
    def state(self):
        """Borrow the current accepted state; this phase owns its release."""
        self._check()
        return self._state

    @property
    def stop(self):
        self._check()
        return self._stop

    @property
    def steps(self):
        self._check()
        return tuple(self._steps)

    @property
    def active(self):
        self._check()
        return self._stop.reason is None

    def request_tree(self):
        """Return a default TreeJob with freshly owned fields; do not consume a round.

        Requests may be repeated on a retained accepted state. The caller chooses
        when/how to group them, owns resulting trees/predictions, and must release
        every returned field record. Copying/replacing job metadata does not change
        the phase's accepted state or policy. No whole-fit workspace is suspended.
        """
        self._require_active()
        run, policy = self._run, self._configuration
        with _workspace(run.ops) as retained:
            fields = run.fields(self._state)
            job = TreeJob(run.run_id, run.data, fields, run.binning, max_depth=policy.max_depth,
                          reg_lambda=policy.reg_lambda, min_child_h=policy.min_child_h,
                          split_penalty=policy.split_penalty)
            retained.add(fields)
            return job

    def _predicted(self, train, validation):
        if not isinstance(train, DevicePrediction) or not isinstance(validation, DevicePrediction):
            raise ValueError('registered training and validation predictions required')
        run = self._run
        predicted = PredictedTerm(DeviceTerm(train.tree, [[1]]), train, validation)
        run.validate_terms((predicted.term,))
        for prediction, data in ((train, run.data), (validation, run.validation_data)):
            run.ops._get(prediction, DevicePrediction)
            if prediction.data is not data:
                raise ValueError('prediction requires exact active run data binding')
            run.ops._float(prediction.values, (data.n_rows, 1))
        return predicted

    def advance(self, train_prediction, validation_prediction):
        """Consume one borrowed scalar tree/prediction pair as one outer round.

        Structural/liveness checks precede all trials. Rejected trials retain the
        parent; accepted trials use the existing transaction's best-model policy.
        Numerical trial failures backtrack only when configured; infrastructure
        failures propagate. Completed rejected trials can consume proposal IDs
        before a later failure, but no stopping observation/history is published
        for an unfinished round. Inputs and the phase remain caller-owned.
        """
        self._require_active()
        predicted = self._predicted(train_prediction, validation_prediction)
        run, policy = self._run, self._configuration
        coefficients, failures, accepted = [], [], False
        for trial in range(1 if policy.step == 'fixed' else policy.max_trials):
            coefficient = policy.learning_rate * 0.5**trial
            coefficients.append(coefficient)
            proposal = None
            try:
                proposal = run.propose_predicted(self._state, (predicted,), coefficient=coefficient)
                accepted = policy.step == 'fixed' or proposal.loss < self._state.loss
                updated = run.resolve(self._state, proposal, accept=accepted)
            except (ValueError, FloatingPointError, OverflowError) as error:
                if policy.step == 'fixed':
                    raise
                accepted = False
                failures.append(type(error).__name__ + ': ' + str(error))
                continue
            finally:
                if proposal is not None:
                    run.release(proposal)
            if accepted:
                previous, self._state = self._state, updated
                run.release(previous)
                break
        step = DeviceStep(self._stop.completed_rounds, tuple(coefficients), accepted, tuple(failures),
                          self._state.loss, self._state.validation_score, self._state.best_score)
        observed = self._stop.observe(self._state.validation_score)
        self._steps.append(step)
        self._stop = observed
        return step

    def result(self):
        """Export detached final/best models at terminal stopping; caller still closes."""
        self._check()
        if self._stop.reason is None:
            raise ValueError('terminal stopping required before result export')
        return RunResult(self._run.export(self._state), self._run.export(self._state, best=True),
                         self._state, tuple(self._steps), self._stop)

    def close(self):
        """Release this phase's run; caller-owned fields, trees and predictions remain."""
        if not self._closed:
            self._run.close()
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
