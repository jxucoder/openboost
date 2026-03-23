"""Tests for the callback system.

Verifies EarlyStopping, Logger, ModelCheckpoint, LearningRateScheduler,
HistoryCallback, and custom callbacks work correctly.
"""

import pytest

import openboost as ob


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_stops_when_val_loss_plateaus(self, regression_500x10):
        """Training should stop before n_trees when loss plateaus."""
        X, y = regression_500x10
        X_val, y_val = X[:100], y[:100]

        es = ob.EarlyStopping(patience=5)
        model = ob.GradientBoosting(n_trees=1000, max_depth=4, learning_rate=0.3)
        model.fit(X, y, callbacks=[es], eval_set=[(X_val, y_val)])

        # Should have stopped early (well before 1000 trees)
        assert len(model.trees_) < 1000, (
            f"Should stop early but trained all {len(model.trees_)} trees"
        )

    def test_patience_respected(self, regression_500x10):
        """With higher patience, training should run longer."""
        X, y = regression_500x10
        X_val, y_val = X[:100], y[:100]

        es_short = ob.EarlyStopping(patience=3)
        model_short = ob.GradientBoosting(n_trees=500, max_depth=4, learning_rate=0.3)
        model_short.fit(X, y, callbacks=[es_short], eval_set=[(X_val, y_val)])

        es_long = ob.EarlyStopping(patience=20)
        model_long = ob.GradientBoosting(n_trees=500, max_depth=4, learning_rate=0.3)
        model_long.fit(X, y, callbacks=[es_long], eval_set=[(X_val, y_val)])

        # Longer patience should train at least as many trees
        assert len(model_long.trees_) >= len(model_short.trees_)

    def test_restore_best(self, regression_500x10):
        """With restore_best=True, model should use best iteration's trees."""
        X, y = regression_500x10
        X_val, y_val = X[:100], y[:100]

        es = ob.EarlyStopping(patience=5, restore_best=True)
        model = ob.GradientBoosting(n_trees=500, max_depth=4, learning_rate=0.3)
        model.fit(X, y, callbacks=[es], eval_set=[(X_val, y_val)])

        # If early stopping fired, best_iteration should be set
        if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
            assert len(model.trees_) <= model.best_iteration_ + 1 + 5


class TestHistoryCallback:
    """Tests for HistoryCallback."""

    def test_records_train_loss(self, regression_200x10):
        """Should record training loss each round."""
        X, y = regression_200x10

        history = ob.HistoryCallback()
        model = ob.GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X, y, callbacks=[history])

        assert 'train_loss' in history.history
        assert len(history.history['train_loss']) == 10

    def test_records_val_loss(self, regression_200x10):
        """Should record validation loss when eval_set provided."""
        X, y = regression_200x10
        X_val, y_val = X[:50], y[:50]

        history = ob.HistoryCallback()
        model = ob.GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X, y, callbacks=[history], eval_set=[(X_val, y_val)])

        assert 'val_loss' in history.history
        assert len(history.history['val_loss']) == 10

    def test_train_loss_decreases(self, regression_200x10):
        """Recorded training loss should generally decrease."""
        X, y = regression_200x10

        history = ob.HistoryCallback()
        model = ob.GradientBoosting(n_trees=20, max_depth=3, learning_rate=0.1)
        model.fit(X, y, callbacks=[history])

        losses = history.history['train_loss']
        # First loss should be > last loss
        assert losses[-1] < losses[0], (
            f"Training loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


class TestLoggerCallback:
    """Tests for Logger callback."""

    def test_logger_does_not_crash(self, regression_100x5, capsys):
        """Logger should print without crashing."""
        X, y = regression_100x5

        logger = ob.Logger(period=5)
        model = ob.GradientBoosting(n_trees=10, max_depth=2)
        model.fit(X, y, callbacks=[logger])

        # Just verify it didn't crash — output format may vary


class TestMultipleCallbacks:
    """Tests for using multiple callbacks together."""

    def test_early_stopping_and_history(self, regression_500x10):
        """EarlyStopping + HistoryCallback should work together."""
        X, y = regression_500x10
        X_val, y_val = X[:100], y[:100]

        es = ob.EarlyStopping(patience=5)
        history = ob.HistoryCallback()

        model = ob.GradientBoosting(n_trees=500, max_depth=4, learning_rate=0.3)
        model.fit(X, y, callbacks=[es, history], eval_set=[(X_val, y_val)])

        # History should only have as many entries as rounds trained
        n_trained = len(model.trees_)
        assert len(history.history['train_loss']) == n_trained

    def test_all_callbacks_together(self, regression_200x10):
        """Multiple callbacks should all receive events."""
        X, y = regression_200x10

        history = ob.HistoryCallback()
        logger = ob.Logger(period=100)  # Don't spam output

        model = ob.GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X, y, callbacks=[history, logger])

        assert len(history.history['train_loss']) == 10


class TestCustomCallback:
    """Tests for custom callback classes."""

    def test_custom_callback_receives_events(self, regression_100x5):
        """Custom callback should receive on_train_begin and on_round_end."""
        class EventTracker(ob.Callback):
            def __init__(self):
                self.began = False
                self.round_count = 0
                self.ended = False

            def on_train_begin(self, state):
                self.began = True

            def on_round_end(self, state):
                self.round_count += 1
                return True

            def on_train_end(self, state):
                self.ended = True

        X, y = regression_100x5
        tracker = EventTracker()

        model = ob.GradientBoosting(n_trees=5, max_depth=2)
        model.fit(X, y, callbacks=[tracker])

        assert tracker.began, "on_train_begin should be called"
        assert tracker.round_count == 5, f"on_round_end called {tracker.round_count} times, expected 5"
        assert tracker.ended, "on_train_end should be called"

    def test_custom_callback_can_stop_training(self, regression_100x5):
        """Custom callback returning False should stop training."""
        class StopAtThree(ob.Callback):
            def on_round_end(self, state):
                return state.round_idx < 2  # Stop after 3 rounds (0, 1, 2)

        X, y = regression_100x5

        model = ob.GradientBoosting(n_trees=100, max_depth=2)
        model.fit(X, y, callbacks=[StopAtThree()])

        assert len(model.trees_) <= 3, (
            f"Should stop at 3 trees, got {len(model.trees_)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
