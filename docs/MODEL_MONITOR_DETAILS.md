# Model Monitor Deep Dive

## Detection Logic

```python
# Prediction Drift
def check_prediction_drift(predictions):
    return any(abs(p.predicted - p.actual) >= 50 for p in predictions)

# Feature Shift
def check_feature_shift(predictions, feature='temperature'):
    return any(p.features.get(feature, 0) > 50 for p in predictions)
```

## Statistical Methods

```python
def _calculate_drift(predictions, baseline):
    """
    Implements:
    - Kolmogorov-Smirnov test (distribution)
    - Population Stability Index (feature drift)
    - Adaptive windowing for concept drift
    """
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| window_size | 1000 | Prediction window for analysis |
| drift_threshold | 50 | Absolute error threshold |
| temp_threshold | 50 | Temperature shift threshold |
| mae_threshold | 25 | Performance degradation threshold |

## Performance Optimizations
| Operation | Complexity | Notes |
|-----------|------------|-------|
| Record | O(1) | Appends to circular buffer |
| Drift Check | O(n) | n=window_size |
| Feature Stats | O(1) | Online algorithm |

## Operational Metrics

Tracked per model:
```python
class OperationalMetrics:
    prediction_count: int
    avg_latency: float
    error_rate: float
    drift_alerts: int
```

## Alert Thresholds
| Metric | Warning | Critical |
|--------|---------|----------|
| MAE | >25 | >50 |
| Drift Frequency | 5/hr | 20/hr |
| Error Rate | 5% | 10% |
