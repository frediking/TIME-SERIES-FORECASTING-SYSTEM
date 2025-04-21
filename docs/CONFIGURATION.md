# Configuration Reference

## Core Settings
```yaml
model_registry:
  path: ./models
  max_versions: 50

monitoring:
  window_size: 1000
  drift_thresholds:
    prediction: 50
    feature: 50
    mae: 25

ensemble:
  min_models: 3
  max_models: 10
  weight_update_interval: 10000
```

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_CACHE_SIZE | 1000 | Prediction cache items |
| MAX_PREDICTION_THREADS | 4 | Parallel model execution |
