# Callbacks

Control training with callbacks for early stopping, logging, checkpointing, and more.

## Available Callbacks

| Callback | Purpose |
|----------|---------|
| `EarlyStopping` | Stop when validation metric stops improving |
| `Logger` | Print training progress |
| `ModelCheckpoint` | Save best models during training |
| `LearningRateScheduler` | Dynamic learning rate |

## Early Stopping

```python
import openboost as ob
from openboost import EarlyStopping

model = ob.GradientBoosting(n_trees=500)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[EarlyStopping(patience=10, min_delta=0.001)],
)

print(f"Stopped at {len(model.trees_)} trees")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patience` | int | 10 | Rounds without improvement before stopping |
| `min_delta` | float | 0.0 | Minimum change to qualify as improvement |
| `restore_best` | bool | True | Restore weights from best iteration |

## Logger

```python
from openboost import Logger

model.fit(
    X_train, y_train,
    callbacks=[Logger(every=10)],  # Print every 10 trees
)
```

Output:
```
[10] train_loss: 0.1234
[20] train_loss: 0.0987
[30] train_loss: 0.0856
...
```

## Model Checkpoint

```python
from openboost import ModelCheckpoint

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        ModelCheckpoint(
            filepath='best_model.joblib',
            save_best_only=True,
        )
    ],
)
```

## Learning Rate Scheduler

```python
from openboost import LearningRateScheduler

def lr_schedule(round_num, current_lr):
    """Reduce LR by 10% every 100 rounds."""
    if round_num > 0 and round_num % 100 == 0:
        return current_lr * 0.9
    return current_lr

model.fit(
    X_train, y_train,
    callbacks=[LearningRateScheduler(lr_schedule)],
)
```

## Combining Callbacks

```python
from openboost import EarlyStopping, Logger, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=20),
    Logger(every=10),
    ModelCheckpoint('checkpoints/model_{round}.joblib'),
]

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks,
)
```

## Custom Callbacks

```python
from openboost import Callback

class MyCallback(Callback):
    def on_train_begin(self, state):
        print("Training started!")
    
    def on_round_end(self, state):
        if state.round % 50 == 0:
            print(f"Round {state.round}: loss = {state.train_loss:.4f}")
    
    def on_train_end(self, state):
        print(f"Training finished after {state.round} rounds")

model.fit(X_train, y_train, callbacks=[MyCallback()])
```

### Callback Methods

| Method | When Called |
|--------|-------------|
| `on_train_begin(state)` | Before training starts |
| `on_round_begin(state)` | Before each boosting round |
| `on_round_end(state)` | After each boosting round |
| `on_train_end(state)` | After training completes |
