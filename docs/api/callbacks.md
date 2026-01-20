# Callbacks

Training callbacks for control and monitoring.

## Available Callbacks

### EarlyStopping

::: openboost.EarlyStopping
    options:
      show_root_heading: true

### Logger

::: openboost.Logger
    options:
      show_root_heading: true

### ModelCheckpoint

::: openboost.ModelCheckpoint
    options:
      show_root_heading: true

### LearningRateScheduler

::: openboost.LearningRateScheduler
    options:
      show_root_heading: true

### HistoryCallback

::: openboost.HistoryCallback
    options:
      show_root_heading: true

## Base Classes

### Callback

::: openboost.Callback
    options:
      show_root_heading: true
      members:
        - on_train_begin
        - on_train_end
        - on_round_begin
        - on_round_end

### TrainingState

::: openboost.TrainingState
    options:
      show_root_heading: true

### CallbackManager

::: openboost.CallbackManager
    options:
      show_root_heading: true
