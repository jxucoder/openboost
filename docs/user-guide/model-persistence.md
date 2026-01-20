# Model Persistence

Save and load trained OpenBoost models.

## Basic Save/Load

```python
import openboost as ob

# Train
model = ob.GradientBoosting(n_trees=100)
model.fit(X_train, y_train)

# Save
model.save('my_model.joblib')

# Load
loaded_model = ob.GradientBoosting.load('my_model.joblib')
predictions = loaded_model.predict(X_test)
```

## Supported Models

All models support save/load:

- `GradientBoosting`
- `MultiClassGradientBoosting`
- `DART`
- `OpenBoostGAM`
- `NaturalBoostNormal`, `NaturalBoostGamma`, etc.
- `LinearLeafGBDT`

## Using joblib/pickle Directly

```python
import joblib

# Save
joblib.dump(model, 'model.joblib')

# Load
loaded = joblib.load('model.joblib')
```

Or with pickle:

```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded = pickle.load(f)
```

## Cross-Platform Loading

Models trained on GPU can be loaded on CPU machines:

```python
# Train on GPU
model = ob.GradientBoosting(n_trees=100)
model.fit(X_train, y_train)  # Using CUDA
model.save('model.joblib')

# Load on CPU machine
loaded = ob.GradientBoosting.load('model.joblib')
predictions = loaded.predict(X_test)  # Works on CPU
```

## Versioning

Include version information when saving:

```python
import openboost as ob

metadata = {
    'openboost_version': ob.__version__,
    'model_type': 'GradientBoosting',
    'training_date': '2026-01-20',
}

joblib.dump({'model': model, 'metadata': metadata}, 'model_with_meta.joblib')
```

## Model Checkpointing

Save best models during training:

```python
from openboost import ModelCheckpoint

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        ModelCheckpoint(
            filepath='checkpoints/best_model.joblib',
            save_best_only=True,
        )
    ],
)
```

## File Formats

| Format | Extension | Pros | Cons |
|--------|-----------|------|------|
| joblib | `.joblib` | Fast, compressed | Python only |
| pickle | `.pkl` | Standard library | Slower, larger |

Recommended: Use joblib for best performance.
